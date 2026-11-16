"""Live owner edges and the original framework-before-policy observation order."""

import os
import sys
import types

import pytest
from tests.native_helper_policy import require_selected_native

from identity_framework_fixture import ordinary_framework

pytestmark = pytest.mark.usefixtures("ordinary_framework")

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as env
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_fast_path as fast
from triton_msl.errors import MetalNonRecoverableError


def _warm_contract():
    require_selected_native(cache._validation_native, "validation")
    assert cache._validation_native is not None, "these controls require the rebuilt native helper"
    fast._record = None
    stamp = cache.execution_contract()
    before = fast._stats["hits"]
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before + 1
    return stamp


@pytest.fixture
def warm():
    # pytest changes PYTEST_CURRENT_TEST between setup and call. Capture inside
    # the test body, so its mutation (not that phase transition) causes the miss.
    yield _warm_contract
    fast._record = None


def _other_policy():
    return "0" if os.environ.get("TRITON_MSL_FA_FAST", "1") != "0" else "1"


def _full(monkeypatch):
    fast._record = None
    with monkeypatch.context() as m:
        m.setattr(fast, "build", lambda stamp: None)
        return cache.execution_contract()


@pytest.mark.parametrize("change_policy", [False, True])
def test_live_environment_dictionary_not_saved_dictionary(warm, monkeypatch, change_policy):
    warm = warm()
    mapping = os.environ
    original = mapping.__dict__
    replacement = dict(original)
    replacement["_data"] = dict(original["_data"])
    if change_policy:
        replacement["_data"][original["encodekey"]("TRITON_MSL_FA_FAST")] = original["encodevalue"](_other_policy())
    hits = fast._stats["hits"]
    try:
        mapping.__dict__ = replacement
        actual = cache.execution_contract()
        assert fast._stats["hits"] == hits, "changed owner must not use a saved child"
        assert (actual != warm) is change_policy
        assert actual == _full(monkeypatch)
        if change_policy:
            with pytest.raises(MetalNonRecoverableError):
                cache.validate_execution_contract(warm)
    finally:
        mapping.__dict__ = original


def test_live_environment_type_not_saved_type(warm, monkeypatch):
    warm = warm()
    mapping = os.environ
    original = type(mapping)
    alternate = _other_policy()

    class Changed(original):
        def get(self, key, default=None):
            return alternate if key == "TRITON_MSL_FA_FAST" else super().get(key, default)

    hits = fast._stats["hits"]
    try:
        mapping.__class__ = Changed
        actual = cache.execution_contract()
        assert actual != warm
        assert fast._stats["hits"] == hits
        assert actual == _full(monkeypatch)
        with pytest.raises(MetalNonRecoverableError):
            cache.validate_execution_contract(warm)
    finally:
        mapping.__class__ = original


def test_cold_custom_environment_dict_descriptor_declines_without_observation(monkeypatch):
    require_selected_native(cache._validation_native, "validation")
    mapping = os.environ
    original = type(mapping)
    captured = mapping.__dict__
    calls = []

    class Changed(original):
        @property
        def __dict__(self):
            calls.append("dict")
            return captured

    try:
        mapping.__class__ = Changed
        assert mapping.__dict__ is captured, "known-positive custom descriptor did not execute"
        assert calls == ["dict"]
        calls.clear()
        with monkeypatch.context() as m:
            # Align the direct-builder fixture's type identities so the saved
            # original getset descriptor is the sole eligibility failure.
            m.setattr(env, "_type", Changed)
            m.setattr(fast, "_ENV_TYPE", Changed)
            assert fast._framework_environment_probes(cache._validation_native) is None
        assert calls == []
    finally:
        mapping.__class__ = original


def test_type_probe_uses_real_type_not_spoofable_class_attribute():
    require_selected_native(cache._validation_native, "validation")

    class Spoof:
        @property
        def __class__(self):
            return env._type

    native = cache._validation_native
    assert native is not None
    assert native.identity_probes(((fast.TYPEOF, os.environ, None, None, env._type),))
    assert not native.identity_probes(((fast.TYPEOF, Spoof(), None, None, env._type),))


def test_live_sys_implementation_owner(warm, monkeypatch):
    warm = warm()
    replacement = types.SimpleNamespace(**vars(sys.implementation))
    replacement.cache_tag += "-changed"
    hits = fast._stats["hits"]
    with monkeypatch.context() as m:
        m.setattr(sys, "implementation", replacement)
        with pytest.raises(MetalNonRecoverableError, match="framework selection changed"):
            cache.execution_contract()
        assert fast._stats["hits"] == hits


@pytest.mark.parametrize("target", ["verify", "reader"])
def test_native_callback_changes_policy_once_before_capture(warm, monkeypatch, target):
    warm = warm()
    guard = fw._native_guard
    assert guard is not None
    alternate = _other_policy()
    attr = "verify" if target == "verify" else "_generation_reader"
    original = getattr(guard, attr)
    assert original is not None
    calls = []

    def callback():
        result = original()
        calls.append(1)
        monkeypatch.setenv("TRITON_MSL_FA_FAST", alternate)
        return result

    with monkeypatch.context() as m:
        m.setitem(guard.__dict__, attr, callback)
        hits = fast._stats["hits"]
        actual = cache.execution_contract()
        assert calls == [1], "nonstandard native callback must not be speculatively replayed"
        assert actual != warm
        assert fast._stats["hits"] == hits
        assert actual == _full(monkeypatch)


def test_native_callback_exception_once_before_environment(warm, monkeypatch):
    warm = warm()
    guard = fw._native_guard
    assert guard is not None
    error = RuntimeError("native verifier veto")
    calls = []

    def veto():
        calls.append(1)
        raise error

    with monkeypatch.context() as m:
        m.setitem(guard.__dict__, "verify", veto)
        with pytest.raises(MetalNonRecoverableError) as raised:
            cache.execution_contract()
        assert raised.value.__cause__ is error
        assert calls == [1]


def test_original_owned_verifier_bound_on_instance_rebuilds_once_then_hits(warm, monkeypatch):
    warm = warm()
    guard = fw._native_guard
    assert guard is not None
    assert "verify" not in guard.__dict__, "earlier tests must restore inherited lookup"
    with monkeypatch.context() as m:
        m.setitem(guard.__dict__, "verify", guard.verify)
        hits = fast._stats["hits"]
        builds = fast._stats["builds"]
        assert cache.execution_contract() == warm
        assert fast._stats["hits"] == hits
        assert fast._stats["builds"] == builds + 1
        assert cache.execution_contract() == warm
        assert fast._stats["hits"] == hits + 1
        cache.validate_execution_contract(warm)
        assert fast._stats["hits"] == hits + 2
    assert "verify" not in guard.__dict__, "restore absence, not a self-bound method"


def test_none_verifier_is_not_an_absent_override(warm, monkeypatch):
    warm()
    guard = fw._native_guard
    assert guard is not None
    with monkeypatch.context() as m:
        m.setitem(guard.__dict__, "verify", None)
        with pytest.raises(MetalNonRecoverableError) as raised:
            cache.execution_contract()
        assert isinstance(raised.value.__cause__, TypeError)
