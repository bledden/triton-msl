"""Saved-identity fast path: every recorded field, when changed, must miss and re-derive; reverting must hit again.

The fast path must never answer from a stale record: a recorded stamp is returned only when every object the
full evaluation would observe is the identical object (or, for mutable spec/path fields, equal). Provider
replacement and native-helper absence take the unchanged full evaluation. Exceptions from observation propagate.
"""

import importlib.util
import os
import sys
import types

import pytest

from identity_framework_fixture import ordinary_framework

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as env
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_fast_path as fast
from triton_msl.backend import _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError

pytestmark = [
    pytest.mark.skipif(cache._validation_native is None, reason="native helper absent: no fast path exists"),
    pytest.mark.usefixtures("ordinary_framework"),
]


def _warm():
    stamp = cache.execution_contract()
    assert fast._record is not None
    before = dict(fast._stats)
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before["hits"] + 1, "second evaluation must be a hit"
    return stamp


def _expect_miss_then_hit(apply, revert, *, stamp_changes):
    stamp = _warm()
    apply()
    try:
        before = dict(fast._stats)
        changed = cache.execution_contract()
        assert fast._stats["misses"] == before["misses"] + 1, "changed field must miss the record"
        if stamp_changes:
            assert changed != stamp
            with pytest.raises(MetalNonRecoverableError):
                cache.validate_execution_contract(stamp)
        else:
            assert changed == stamp
        # a forced full evaluation agrees with whatever the path answered
        fast._record = None
        assert cache.execution_contract() == changed
    finally:
        revert()
    restored = cache.execution_contract()
    assert restored == stamp
    before = dict(fast._stats)
    assert cache.execution_contract() == stamp and fast._stats["hits"] == before["hits"] + 1


def test_hit_returns_the_full_evaluation_stamp():
    stamp = _warm()
    fast._record = None
    assert cache.execution_contract() == stamp


def test_policy_env_change_misses_and_refuses_old_stamp(monkeypatch):
    _expect_miss_then_hit(lambda: monkeypatch.setenv("TRITON_MSL_FA_FAST", "0"), monkeypatch.undo, stamp_changes=True)


@pytest.mark.parametrize("reenter", [False, True])
def test_orphan_finalizer_runs_only_before_complete_observation(monkeypatch, reenter):
    native = cache._validation_native
    native.probe_cache_collect()
    monkeypatch.delenv("TRITON_MSL_FA_FAST", raising=False)
    stamp = _warm()
    calls = []
    nested = []

    class ChangePolicyOnDelete:
        def __del__(self):
            calls.append("finalizer")
            monkeypatch.setenv("TRITON_MSL_FA_FAST", "0")
            if reenter:
                nested.append(cache.execution_contract())

    owner = ChangePolicyOnDelete()
    probes = ((0, owner, None, None, owner),)
    assert native.identity_probes(probes)
    del owner, probes
    # The warm launch boundary must not release ownership or run callbacks.
    before = fast._stats["hits"]
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before + 1
    assert calls == []
    fast._record = None
    changed = cache.execution_contract()
    assert calls == ["finalizer"]
    assert changed != stamp
    if reenter:
        assert nested == [changed]
    with pytest.raises(MetalNonRecoverableError):
        cache.validate_execution_contract(stamp)
    fast._record = None
    assert cache.execution_contract() == changed


def test_program_slots_stay_bounded_across_record_rebuilds(monkeypatch):
    native = cache._validation_native
    native.key_cache_reset()
    monkeypatch.delenv("TRITON_MSL_COMPILE_SHADER", raising=False)
    stamp = _warm()
    for _ in range(12):
        monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
        assert cache.execution_contract() == stamp
        monkeypatch.delenv("TRITON_MSL_COMPILE_SHADER")
        assert cache.execution_contract() == stamp
        assert cache.execution_contract() == stamp
        assert native.probe_cache_info()["entries"] <= 8
        hits = native.probe_cache_info()["hits"]
        assert cache.execution_contract() == stamp
        # Four proof programs plus one live native-observer selector check.
        assert native.probe_cache_info()["hits"] - hits == 5
    assert native.probe_cache_info()["released"] > 0


def test_older_helper_without_cold_collection_keeps_complete_evaluation(monkeypatch):
    stamp = _warm()
    monkeypatch.delattr(cache._validation_native, "probe_cache_collect")
    fast._record = None
    assert cache.execution_contract() == stamp


def test_cold_collection_exception_propagates_without_retry(monkeypatch):
    stamp = _warm()
    calls = []
    error = RuntimeError("cold collection failed")

    def fail():
        calls.append("collect")
        raise error

    monkeypatch.setattr(cache._validation_native, "probe_cache_collect", fail)
    assert cache.execution_contract() == stamp
    assert calls == [], "hot hits must not call the collector"
    fast._record = None
    with pytest.raises(RuntimeError) as caught:
        cache.execution_contract()
    assert caught.value is error
    assert calls == ["collect"]


def test_unrelated_env_addition_and_equal_content_replacement_miss(monkeypatch):
    _expect_miss_then_hit(
        lambda: monkeypatch.setenv("TRITON_MSL_FAST_PATH_PROBE", "1"), monkeypatch.undo, stamp_changes=False
    )
    data = os.environ._data
    key = env._codecs[0]("HOME") if "HOME" in os.environ else next(iter(data))
    original = data[key]
    _expect_miss_then_hit(
        lambda: data.__setitem__(key, bytes(bytearray(original))),
        lambda: data.__setitem__(key, original),
        stamp_changes=False,
    )


def test_decoder_code_mutation_on_same_function_misses():
    # A behaviourally equivalent decoder with a DIFFERENT code object on the same function object:
    # the recognized-implementation guard declines the owned snapshot; the fast path must miss too.
    fn = env._codecs[3]
    original = fn.__code__
    replacement = (lambda encoding: (lambda value: value.decode(encoding, "surrogateescape")))("utf-8").__code__
    _expect_miss_then_hit(
        lambda: setattr(fn, "__code__", replacement), lambda: setattr(fn, "__code__", original), stamp_changes=False
    )


def test_instance_getter_override_misses(monkeypatch):
    mapping = os.environ
    _expect_miss_then_hit(
        lambda: mapping.__dict__.__setitem__("get", mapping.get),
        lambda: mapping.__dict__.pop("get", None),
        stamp_changes=False,
    )


@pytest.mark.parametrize("name", ["all", "map"])
def test_replaced_builtin_observer_takes_the_full_evaluation(monkeypatch, name):
    original = getattr(__builtins__, name) if isinstance(__builtins__, types.ModuleType) else __builtins__[name]

    def observer(*args, **kwargs):
        return original(*args, **kwargs)

    _expect_miss_then_hit(
        lambda: monkeypatch.setattr(env, name, observer, raising=False), monkeypatch.undo, stamp_changes=False
    )


def test_module_and_spec_fields_miss(monkeypatch):
    spec = sys.modules["numpy"].__spec__
    origin = spec.origin
    _expect_miss_then_hit(
        lambda: setattr(spec, "origin", origin + "x"), lambda: setattr(spec, "origin", origin), stamp_changes=False
    )
    # A replaced selected module is a changed framework selection: the fast path misses, and the full
    # evaluation refuses (restart required). The only wrong outcome would be a stale hit.
    stamp = _warm()
    saved = sys.modules.get("mlx")
    sys.modules["mlx"] = types.ModuleType("mlx")
    try:
        before = dict(fast._stats)
        with pytest.raises(MetalNonRecoverableError):
            cache.execution_contract()
        assert fast._stats["misses"] == before["misses"] + 1 and fast._stats["hits"] == before["hits"]
        assert fast._record is None
    finally:
        if saved is None:
            sys.modules.pop("mlx", None)
        else:
            sys.modules["mlx"] = saved
    assert cache.execution_contract() == stamp
    before = dict(fast._stats)
    assert cache.execution_contract() == stamp and fast._stats["hits"] == before["hits"] + 1


def test_sys_path_meta_path_hooks_and_find_spec_miss(monkeypatch):
    _expect_miss_then_hit(
        lambda: sys.path.append("/triton-msl-fast-path-probe"),
        lambda: sys.path.remove("/triton-msl-fast-path-probe"),
        stamp_changes=False,
    )
    _expect_miss_then_hit(
        lambda: sys.meta_path.append(sys.meta_path[0]), lambda: sys.meta_path.pop(), stamp_changes=False
    )
    _expect_miss_then_hit(
        lambda: sys.path_hooks.append(sys.path_hooks[0]), lambda: sys.path_hooks.pop(), stamp_changes=False
    )
    original = importlib.util.find_spec
    _expect_miss_then_hit(
        lambda: monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: original(*a, **k)),
        monkeypatch.undo,
        stamp_changes=False,
    )


def test_spec_getter_exception_propagates_from_the_fast_path():
    """Custom modules use full evaluation, preserving getter calls and errors."""
    module = sys.modules["numpy"]
    original_class = module.__class__
    calls = []

    class Armed(types.ModuleType):
        armed = False

        @property
        def __spec__(self):
            calls.append("spec")
            if Armed.armed:
                raise RuntimeError("spec getter failed later")
            return self.__dict__["__spec__"]

    module.__class__ = Armed
    try:
        fast._record = None
        stamp = cache.execution_contract()
        assert fast._record is None, "custom module lookup is not eligible for recording"
        calls.clear()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fast, "build", lambda value: None)
            assert cache.execution_contract() == stamp
        full_calls = list(calls)
        calls.clear()
        assert cache.execution_contract() == stamp
        assert calls == full_calls and calls, "cold recording must not replay the getter"
        assert fast._record is None
        Armed.armed = True
        calls.clear()
        with pytest.raises(MetalNonRecoverableError) as raised:
            cache.execution_contract()
        assert type(raised.value.__cause__) is RuntimeError
        assert str(raised.value.__cause__) == "spec getter failed later"
        assert calls == ["spec"], "the complete evaluator owns the one failed observation"
        Armed.armed = False
        assert cache.execution_contract() == stamp
    finally:
        Armed.armed = False
        module.__class__ = original_class
        fast._record = None


def test_replaced_providers_take_the_full_evaluation(monkeypatch):
    stamp = _warm()
    calls = []

    def framework():
        calls.append("framework")
        return fw.framework_identity()

    monkeypatch.setattr(cache, "framework_identity", framework)
    before = dict(fast._stats)
    assert cache.execution_contract() == stamp
    assert calls == ["framework"] and fast._stats["hits"] == before["hits"]
    monkeypatch.undo()
    fast._record = None
    _warm()
    original = env.environment_snapshot
    monkeypatch.setattr(env, "environment_snapshot", lambda: original())
    before = dict(fast._stats)
    assert cache.execution_contract() == stamp and fast._stats["hits"] == before["hits"]


def test_native_absence_means_no_fast_path(monkeypatch):
    stamp = _warm()
    monkeypatch.setattr(cache, "_validation_native", None)
    before = dict(fast._stats)
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before["hits"]


def test_native_guard_is_still_verified_on_a_hit(monkeypatch):
    stamp = _warm()
    guard = fw._native_guard
    if guard is None:
        pytest.skip("no native guard in this process")
    calls = []
    original = guard.verify
    monkeypatch.setitem(guard.__dict__, "verify", lambda: (calls.append(1), original())[1])
    assert cache.execution_contract() == stamp
    assert calls == [1], "a hit must still verify the native-library guard"

    # A changed native selection is not a hit: the full evaluation handles it (rebuild), never a stale stamp.
    def changed():
        calls.append("changed")
        raise fw._NativeSelectionChanged("probe")

    monkeypatch.setitem(guard.__dict__, "verify", changed)
    before = dict(fast._stats)
    try:
        cache.execution_contract()
    except MetalNonRecoverableError:
        pass  # the full evaluation may refuse when the rebuilt inventory disagrees; a stale hit is the only wrong outcome
    assert fast._stats["hits"] == before["hits"]
    # A replaced verifier is intentionally handled once by the full evaluator,
    # not speculatively invoked and then replayed after a fast-path miss.
    assert fast._stats["misses"] == before["misses"] + 1
    assert calls.count("changed") == 1
