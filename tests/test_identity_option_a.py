"""CPU controls for the declared callback-omitting whole-stamp boundary."""

import os
import subprocess
import sys

import pytest
from tests.python_child import package_child_command

from identity_framework_fixture import ordinary_framework

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as env
from triton_msl.backend import _identity_fast_path as fast
from triton_msl.errors import MetalNonRecoverableError


def _warm():
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    fast._record = None
    stamp = cache.execution_contract()
    before = fast._stats["hits"]
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before + 1
    return stamp


# These three controls require an admitted warm record. Full collection may
# import MLX's real namespace, whose observable path iterator correctly takes
# the complete evaluator. The namespace fallback control keeps that real state.
@pytest.mark.usefixtures("ordinary_framework")
def test_disable_switch_precedes_warm_hit(monkeypatch):
    stamp = _warm()
    monkeypatch.setenv("TRITON_MSL_IDENTITY_FAST_PATH", "0")
    before = fast._stats["hits"]
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before


@pytest.mark.usefixtures("ordinary_framework")
def test_policy_change_between_calls_is_not_admitted(monkeypatch):
    stamp = _warm()
    name = "TRITON_MSL_FA_FAST"
    monkeypatch.setenv(name, "0" if os.environ.get(name) != "0" else "1")
    with pytest.raises(MetalNonRecoverableError):
        cache.validate_execution_contract(stamp)


@pytest.mark.usefixtures("ordinary_framework")
def test_function_dependency_change_misses(monkeypatch):
    _warm()
    namespace = env._get.__globals__
    before = fast._stats["hits"]
    with monkeypatch.context() as patch:
        patch.setitem(namespace, "KeyError", RuntimeError)
        with pytest.raises(KeyError):
            cache.execution_contract()
    assert fast._stats["hits"] == before


def test_audit_observation_is_omitted_only_when_enabled():
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    code = r"""
import os, sys
from triton_msl.backend import _cache_contract as c
from triton_msl.backend import _environment_snapshot as e
f=c._identity_fast_path
s=c.execution_contract(); assert c.execution_contract()==s
target=e._implementations[0][0]; seen=[]
def audit(name,args):
    if name=='object.__getattr__' and args[0] is target and args[1]=='__code__': seen.append(name)
sys.addaudithook(audit)
before=f._stats['hits']; assert c.execution_contract()==s
assert f._stats['hits']==before+1 and seen==[]
os.environ['TRITON_MSL_IDENTITY_FAST_PATH']='0'
assert c.execution_contract()==s and seen
"""
    completed = subprocess.run(package_child_command(code), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("name", ["encodekey", "decodevalue"])
def test_preexisting_codec_descriptor_declines_before_admission(name):
    mapping, owner = os.environ, type(os.environ)
    original = getattr(mapping, name)
    calls = []
    setattr(owner, name, property(lambda self: calls.append(name) or original))
    try:
        fast._record = None
        before = fast._stats["hits"]
        first = cache.execution_contract()
        assert cache.execution_contract() == first
        assert fast._stats["hits"] == before
        assert calls
    finally:
        delattr(owner, name)
        fast._record = None
