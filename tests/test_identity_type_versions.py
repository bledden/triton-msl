"""MRO invalidation must precede speculative instance reads, without callbacks."""

import collections.abc as abc
import os
from types import SimpleNamespace

import pytest
from tests.native_helper_policy import require_selected_native

from identity_framework_fixture import ordinary_framework

pytestmark = pytest.mark.usefixtures("ordinary_framework")

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _identity_fast_path as fast
from test_identity_live_owners import _full, _other_policy, _warm_contract


@pytest.fixture(autouse=True)
def clear_record():
    yield
    fast._record = None


@pytest.mark.parametrize("base", [abc.MutableMapping, abc.Mapping, abc.Collection])
@pytest.mark.parametrize("changed_reads", [0, 1, 4, 5, 6])
def test_inherited_descriptor_exact_outcome_and_callback_count(monkeypatch, base, changed_reads):
    mapping = os.environ
    data = mapping.__dict__["_data"]
    changed = dict(data)
    changed[mapping.encodekey("TRITON_MSL_FA_FAST")] = mapping.encodevalue(_other_policy())
    results = []
    for mode in ("full", "fast"):
        _warm_contract()
        calls = []

        def getter(self):
            calls.append(1)
            return changed if len(calls) <= changed_reads else data

        with monkeypatch.context() as m:
            m.setattr(base, "_data", property(getter), raising=False)
            try:
                stamp = _full(monkeypatch) if mode == "full" else cache.execution_contract()
                result = ("stamp", stamp)
            except Exception as exc:
                result = ("exception", type(exc), type(exc.__cause__), str(exc))
            assert fast._record is None, "inherited observer must prevent record construction too"
        results.append((result, len(calls)))
    assert results[0][1] > 0, "positive observer control"
    assert results[0] == results[1], "no speculative getter or changed exception/stamp"
    _warm_contract()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_type_tag_invalidates_on_each_ancestor_and_stays_invalid_after_restore(level):
    require_selected_native(cache._validation_native, "validation")

    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    native = cache._validation_native
    tag = native.type_version(C, ())
    assert tag > 0
    probe = ((fast.TYPEVERSION, C, None, None, tag),)
    assert native.identity_probes(probe)
    cls = (C, B, A)[level]
    cls.marker = object()
    assert not native.identity_probes(probe)
    del cls.marker
    assert not native.identity_probes(probe), "restoration must not resurrect a stale tag"
    assert native.type_version(C, ()) != tag
    assert not native.identity_probes(probe)


def test_changed_bases_invalidates_tag():
    require_selected_native(cache._validation_native, "validation")

    class A:
        pass

    class B:
        pass

    class C(A):
        pass

    native = cache._validation_native
    tag = native.type_version(C, ())
    assert tag
    C.__bases__ = (B,)
    assert not native.identity_probes(((fast.TYPEVERSION, C, None, None, tag),))


def test_uncacheable_custom_mro_and_zero_tag_decline():
    require_selected_native(cache._validation_native, "validation")

    class Meta(type):
        def mro(cls):
            return super().mro()

    class C(metaclass=Meta):
        pass

    native = cache._validation_native
    assert native.type_version(C, ()) == 0
    assert not native.identity_probes(((fast.TYPEVERSION, C, None, None, 0),))


def test_raw_mro_read_never_executes_descriptor_or_metaclass():
    require_selected_native(cache._validation_native, "validation")
    calls = []

    class Meta(type):
        def __getattribute__(cls, name):
            calls.append(name)
            return super().__getattribute__(name)

    class A(metaclass=Meta):
        value = property(lambda self: calls.append("descriptor"))

    class B(A):
        pass

    calls.clear()
    native = cache._validation_native
    assert not native.mro_absent(B, ("value",))
    assert native.mro_absent(B, ("absent",))
    assert native.type_version(B, ()) == 0
    assert calls == []


def test_type_compaction_checks_expected_schema_not_just_current_tag():
    require_selected_native(cache._validation_native, "validation")

    class C:
        field = object()

    native = cache._validation_native
    assert native.type_version(C, ((fast.ATTR, C, "field", None, C.field),)) > 0
    assert native.type_version(C, ((fast.ATTR, C, "field", None, object()),)) == 0
    assert native.type_version(C, ((fast.CLASSGET, C, "field", None, None),)) == 0


@pytest.mark.parametrize("kind", [fast.EQATTRDEF, fast.EQTUPLEATTRDEF])
def test_spec_value_subclass_declines_without_user_equality(kind):
    require_selected_native(cache._validation_native, "validation")
    calls = []

    class Text(str):
        def __eq__(self, other):
            calls.append(other)
            return super().__eq__(other)

    value = Text("same")
    obj = SimpleNamespace(field=value if kind == fast.EQATTRDEF else [value])
    expected = "same" if kind == fast.EQATTRDEF else ("same",)
    assert not cache._validation_native.identity_probes(((kind, obj, "field", None, expected),))
    assert calls == []


def test_spec_locations_decline_without_bool_or_iter():
    require_selected_native(cache._validation_native, "validation")
    calls = []

    class Locations:
        def __bool__(self):
            calls.append("bool")
            return True

        def __iter__(self):
            calls.append("iter")
            return iter(("location",))

    obj = SimpleNamespace(field=Locations())
    assert not cache._validation_native.identity_probes(((fast.EQTUPLEATTRDEF, obj, "field", None, ("location",)),))
    assert calls == []


@pytest.mark.parametrize("mode", ["swap", "override"])
def test_metaclass_observer_matches_complete_evaluator_without_builder_replay(monkeypatch, mode):
    cls = type(os.environ)
    original_meta = type(cls)
    results = []
    for arm in ("full", "fast"):
        _warm_contract()
        calls = []

        def lookup(owner, name):
            calls.append(name)
            return type.__getattribute__(owner, name)

        class Observing(original_meta):
            __getattribute__ = lookup

        try:
            with monkeypatch.context() as m:
                if mode == "swap":
                    cls.__class__ = Observing
                else:
                    # pytest preserves own-dictionary absence for class targets.
                    m.setattr(original_meta, "__getattribute__", lookup)
                hits = fast._stats["hits"]
                stamp = _full(monkeypatch) if arm == "full" else cache.execution_contract()
                assert fast._stats["hits"] == hits, "custom metaclass lookup must take full evaluation"
                assert fast._record is None, "builder must decline before class attribute reads"
        finally:
            if mode == "swap":
                cls.__class__ = original_meta
        results.append((stamp, calls))
    assert results[0][1], "the complete evaluator must exercise the observer"
    assert results[0] == results[1], "no extra lookup from the fast miss or builder"
    _warm_contract()


def test_each_version_pinned_type_has_a_live_metaclass_owner_probe():
    _warm_contract()
    for phase in (fast._record[3], fast._record[4]):
        for kind, cls, _, _, _ in phase:
            if kind == fast.TYPEVERSION:
                assert (fast.TYPEOF, cls, None, None, type(cls)) in phase
