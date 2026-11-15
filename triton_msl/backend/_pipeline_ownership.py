"""Loader-owned pipeline acquisitions with non-owning cross-loader discovery.

Metal objects need not support weak references. The dict index owns buckets of
weak Python records; callbacks prune exact empty buckets. A module unload has no
acquisition token, so records remain until its final matching unload.
"""

from itertools import count
from weakref import ref


_serial = count()


class Record:
    __slots__ = ("registry", "primary", "direct", "proof", "bucket", "key", "serial", "active", "__weakref__")

    def __init__(self, registry, primary, direct, proof, bucket, key):
        self.registry = ref(registry)
        self.primary = primary
        self.direct = direct
        self.proof = proof
        self.bucket = bucket
        self.key = key
        self.serial = None
        self.active = False


class Bucket:
    __slots__ = ("records", "binary", "direct", "weakself", "__weakref__")

    def __init__(self):
        self.records = {}
        self.binary = None
        self.direct = None
        self.weakself = ref(self)

    def cache(self, record):
        for kind, applicable in (("binary", record.proof), ("direct", record.direct)):
            cached = getattr(self, kind)
            previous = None if cached is None else cached()
            if applicable is not None and (
                previous is None
                or not previous.active
                or previous.registry() is None
                or previous.serial < record.serial
            ):
                # The publication weakref is already prepared. Allocating a new
                # one here could run a nested bind between comparison and store.
                setattr(self, kind, self.records[record.serial])

    def latest(self, kind, registry=None):
        cached = self.direct if kind == "direct" else self.binary
        record = None if cached is None else cached()
        if registry is None and record is not None and record.active and record.registry() is not None:
            return record
        # Restoration is proportional to live acquisitions, not historical loads.
        for weak in reversed(tuple(self.records.values())):
            record = weak()
            if record is None:
                continue
            owner = record.registry()
            if not record.active or owner is None or (registry is not None and owner is not registry):
                continue
            if (record.direct if kind == "direct" else record.proof) is not None:
                if registry is None:
                    # The tuple snapshot above may run collection/publication.
                    # Never replace a newer live cached publication with it.
                    current = getattr(self, kind)
                    newer = None if current is None else current()
                    if (
                        newer is not None
                        and newer.active
                        and newer.registry() is not None
                        and newer.serial > record.serial
                    ):
                        return newer
                    setattr(self, kind, weak)
                return record
        return None


class Group:
    __slots__ = ("module", "count", "records")

    def __init__(self, module, count, records):
        self.module = module
        self.count = count
        self.records = records


def _empty(index, key, bucket):
    if not bucket.records and dict.get(index, key) is bucket:
        dict.__delitem__(index, key)


def _weak_record(index, key, bucket, record):
    # Preparation may reenter and replace a canonical bucket. Callback state
    # owns neither a record, registry nor native referent.
    state = [bucket.weakself, None]

    def expired(weak):
        actual = state[0]()
        if actual is not None:
            serial = state[1]
            if actual.records.get(serial) is weak:
                del actual.records[serial]
            _empty(index, key, actual)

    return ref(record, expired), state


class Registry:
    def __init__(self):
        self.modules = {}

    def _detach(self, index, records):
        # Keep every native referent alive until all old indexes are detached.
        for record in records:
            record.active = False
            weak = record.bucket.records.get(record.serial)
            if weak is not None and weak() is record:
                del record.bucket.records[record.serial]
        for record in records:
            _empty(index, record.key, record.bucket)

    def _undo(self, index, module_key, record):
        # A nested acquisition may supersede our Group. Remove only this failed
        # acquisition from its current descendant, preserving nested successes.
        while True:
            group = self.modules.get(module_key)
            if group is None or not any(r is record for r in group.records):
                break
            old_records, old_count = group.records, group.count
            remaining = tuple(r for r in old_records if r is not record)
            if (
                self.modules.get(module_key) is not group
                or group.records is not old_records
                or group.count != old_count
            ):
                continue
            if old_count > 1:
                group.records = remaining
                group.count = old_count - 1
                break
            del self.modules[module_key]
            self._detach(index, old_records)
            break
        self._detach(index, (record,))

    def bind(self, index, key, primary, module, proof, direct):
        if not isinstance(index, dict):
            raise TypeError("pipeline discovery requires a dict index")
        # Builtin setdefault with integer identity keys chooses the canonical
        # bucket after construction, which may execute a nested bind/finalizer.
        bucket = dict.get(index, key)
        if bucket is None:
            bucket = dict.setdefault(index, key, Bucket())
        record = None
        try:
            record = Record(self, primary, direct, proof, bucket, key)
            weak, state = _weak_record(index, key, bucket, record)
            module_key = id(module)
            while True:
                prior = self.modules.get(module_key)
                if prior is not None and prior.module is not module:
                    raise RuntimeError("pipeline module identity collision")
                prior_count = 0 if prior is None else prior.count
                prior_records = () if prior is None else prior.records
                group = Group(module, prior_count + 1, (*prior_records, record))
                bucket = dict.setdefault(index, key, bucket)
                record.bucket = bucket
                state[0] = bucket.weakself
                if self.modules.get(module_key) is prior and (
                    prior is None or (prior.count == prior_count and prior.records is prior_records)
                ):
                    break
            # Assign order at publication, after all callback-capable preparation.
            record.serial = next(_serial)
            state[1] = record.serial
            bucket.records[record.serial] = weak
            self.modules[module_key] = group
            record.active = True
            bucket.cache(record)
        except BaseException as error:
            if record is not None:
                record.active = False
                try:
                    self._undo(index, id(module), record)
                except BaseException as cleanup:
                    # BaseException.add_note is 3.11+; on 3.10 skip the note.
                    add_note = getattr(error, "add_note", None)
                    if add_note is not None:
                        add_note("pipeline publication rollback failed: " + repr(cleanup))
            _empty(index, key, bucket)
            raise
        return record

    def unload(self, index, module):
        key = id(module)
        group = self.modules.get(key)
        if group is None or group.module is not module:
            return
        if group.count > 1:
            group.count -= 1
            return
        del self.modules[key]
        self._detach(index, group.records)
