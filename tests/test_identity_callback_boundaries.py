"""Speculative checks must not consume callbacks belonging to full evaluation."""

import copy
import os

import pytest

from identity_framework_fixture import ordinary_framework

pytestmark = pytest.mark.usefixtures("ordinary_framework")

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_fast_path as fast
from triton_msl.backend import _native_generation as ng
from triton_msl.errors import MetalNonRecoverableError
from test_identity_live_owners import _full, _other_policy, _warm_contract


@pytest.fixture(autouse=True)
def clear_record():
    yield
    fast._record = None


@pytest.mark.parametrize("kind", ["property", "getattribute"])
def test_environment_class_lookup_changes_observed_policy(monkeypatch, kind):
    warm = _warm_contract()
    mapping, cls = os.environ, type(os.environ)
    changed = dict(mapping._data)
    changed[mapping.encodekey("TRITON_MSL_FA_FAST")] = mapping.encodevalue(_other_policy())
    hits = fast._stats["hits"]
    with monkeypatch.context() as m:
        if kind == "property":
            m.setattr(cls, "_data", property(lambda self: changed), raising=False)
        else:

            def lookup(self, name):
                return changed if name == "_data" else object.__getattribute__(self, name)

            m.setattr(cls, "__getattribute__", lookup)
        actual = cache.execution_contract()
        assert actual != warm
        assert fast._stats["hits"] == hits
        assert fast._record is None, "nonstandard attribute lookup must not be speculatively observed"
        assert actual == _full(monkeypatch)


def test_environment_descriptor_is_not_speculatively_replayed(monkeypatch):
    mapping, cls = os.environ, type(os.environ)
    data = mapping._data
    counts = []
    stamps = []
    for mode in ("full", "fast"):
        _warm_contract()
        calls = []

        def observed(self):
            calls.append(1)
            return data

        with monkeypatch.context() as m:
            m.setattr(cls, "_data", property(observed), raising=False)
            stamps.append(_full(monkeypatch) if mode == "full" else cache.execution_contract())
        counts.append(len(calls))
    assert counts[0] > 0
    assert counts[0] == counts[1], "the fast miss must not add descriptor calls"
    assert stamps[0] == stamps[1]


def test_scanner_runtime_error_keeps_wrapped_exception(monkeypatch):
    _warm_contract()
    guard = fw._native_guard
    assert guard is not None
    with monkeypatch.context() as m:
        m.setattr(guard, "_generation_reader", None)
        m.setattr(guard, "name", lambda index: None)
        with pytest.raises(MetalNonRecoverableError) as raised:
            cache.execution_contract()
        assert type(raised.value.__cause__) is RuntimeError
        assert str(raised.value.__cause__) == "native image selection changed during inspection"
        assert fast._record is None


def test_instance_count_policy_callback_runs_only_in_full_evaluation(monkeypatch):
    warm = _warm_contract()
    guard = fw._native_guard
    original = guard.count
    alternate = _other_policy()
    calls = []

    def count():
        calls.append(1)
        # One verify has exactly two count calls. A replay would restore warm.
        monkeypatch.setenv("TRITON_MSL_FA_FAST", alternate if len(calls) <= 2 else ("1" if alternate == "0" else "0"))
        return original()

    with monkeypatch.context() as m:
        m.setattr(guard, "_generation_reader", None)
        m.setattr(guard, "count", count)
        actual = cache.execution_contract()
        assert len(calls) == 2, "no speculative scan before the complete evaluator"
        assert actual != warm
        assert fast._record is None, "unavailable generation reader keeps the full path"


def test_generation_miss_declines_before_any_scan_callback(monkeypatch):
    _warm_contract()
    assert "verify" not in fw._native_guard.__dict__, "copy must inherit its own class-bound verifier"
    guard = copy.copy(fw._native_guard)
    assert guard.verify.__self__ is guard, "the copied guard must not invoke the original guard"
    assert guard._generation_reader is not None
    guard._generation_checked = (0, 0, 0)
    calls = []
    original = guard.count

    def count():
        calls.append(1)
        return original()

    guard.count = count
    with pytest.raises(fw._NativeSelectionChanged, match="complete evaluation"):
        guard.verify(allow_scan=False)
    assert calls == [], "declining the speculative branch must precede image scanning"
    guard.verify()
    assert len(calls) == 2, "the ordinary verifier still performs its complete scan"
    assert guard._generation_checked == guard._generation_reader()


def test_generation_abi_error_inside_fast_branch_keeps_exception_contract(monkeypatch):
    _warm_contract()
    guard = fw._native_guard
    original = guard._generation_reader
    assert original is not None
    # Copy only our own process's public dyld struct; never write loader memory.
    info = ng._ImageInfosPrefix.from_buffer_copy(original.info)
    reader = ng._GenerationReader(info, original.address)
    with monkeypatch.context() as m:
        m.setattr(guard, "_generation_reader", reader)
        _warm_contract()
        record = fast._record
        assert record is not None
        info.version = 0
        assert cache._validation_native.identity_probes(record[3]), "mutation is native data, not a replaced owner"
        with pytest.raises(MetalNonRecoverableError) as raised:
            cache.execution_contract()
        assert type(raised.value.__cause__) is RuntimeError
        assert str(raised.value.__cause__) == "native loader generation ABI changed"
        assert fast._record is None


@pytest.mark.parametrize("name", fast._GUARD_FIELDS)
def test_every_native_guard_owned_field_is_live(monkeypatch, name):
    _warm_contract()
    guard = fw._native_guard
    record = fast._record
    assert record is not None and guard is not None
    with monkeypatch.context() as m:
        m.setattr(guard, name, object())
        assert not cache._validation_native.identity_probes(record[3]), name


@pytest.mark.parametrize(
    "owner,field", [("count", "restype"), ("name", "restype"), ("name", "argtypes"), ("name", "errcheck")]
)
def test_ctypes_hooks_are_live(monkeypatch, owner, field):
    _warm_contract()
    guard = fw._native_guard
    function = getattr(guard, owner)
    original = getattr(function, field)
    try:
        if field == "argtypes":
            function.argtypes = []
        elif field == "errcheck":
            function.errcheck = lambda result, function, args: result
        else:
            function.restype = None
        assert not cache._validation_native.identity_probes(fast._record[3])
    finally:
        if field == "errcheck" and original is None:
            del function.errcheck
        else:
            setattr(function, field, original)


def test_reader_info_owner_is_live(monkeypatch):
    _warm_contract()
    reader = fw._native_guard._generation_reader
    with monkeypatch.context() as m:
        m.setattr(reader, "info", object())
        assert not cache._validation_native.identity_probes(fast._record[3])


def test_recording_does_not_observe_custom_scanner_attributes(monkeypatch):
    warm = _warm_contract()
    guard = fw._native_guard
    original = guard.count
    calls = []

    class Count:
        def __call__(self):
            return original()

        def __getattr__(self, name):
            calls.append(name)
            return None

    with monkeypatch.context() as m:
        m.setattr(guard, "count", Count())
        assert cache.execution_contract() == warm
        assert calls == [], "record construction must not introduce attribute observers"
        assert fast._record is None


def test_class_member_probe_never_invokes_metaclass_lookup():
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    calls = []

    class Meta(type):
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

    value = object()

    class Owner(metaclass=Meta):
        field = value

    assert cache._validation_native.identity_probes(((fast.CLASSGET, Owner, "field", None, value),))
    assert not cache._validation_native.identity_probes(((fast.CLASSGET, Owner, "field", None, object()),))
    assert calls == []


@pytest.mark.parametrize("owner,field", [("guard", "_generation_reader"), ("reader", "info")])
def test_native_owner_class_descriptor_is_not_speculatively_replayed(monkeypatch, owner, field):
    stamps, counts = [], []
    for mode in ("full", "fast"):
        _warm_contract()
        obj = fw._native_guard if owner == "guard" else fw._native_guard._generation_reader
        value = getattr(obj, field)
        calls = []

        def observed(self):
            calls.append(1)
            return value

        with monkeypatch.context() as m:
            m.setattr(type(obj), field, property(observed), raising=False)
            stamps.append(_full(monkeypatch) if mode == "full" else cache.execution_contract())
        counts.append(len(calls))
    assert counts[0] > 0 and counts[0] == counts[1]
    assert stamps[0] == stamps[1]


@pytest.mark.parametrize("saved_custom", [False, True])
def test_path_comparison_declines_without_user_equality(saved_custom):
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    calls = []
    live = []

    class Item:
        def __eq__(self, other):
            calls.append(1)
            live.append("new-entry")
            return True

    custom = Item()
    live.append("saved" if saved_custom else custom)
    saved = (custom,) if saved_custom else ("saved",)
    assert not cache._validation_native.identity_probes(((fast.EQLIST, live, saved, None, None),))
    assert calls == [] and len(live) == 1, "custom equality must only run in the complete evaluator"


def test_path_comparison_exact_strings_keeps_value_semantics():
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    native = cache._validation_native
    live = ["".join(["a-long-path", "-component"])]
    saved = ("a-long-path-component",)
    assert live[0] is not saved[0]
    assert native.identity_probes(((fast.EQLIST, live, saved, None, None),))
    assert not native.identity_probes(((fast.EQLIST, live, ("different",), None, None),))
