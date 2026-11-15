"""Foreign colliding keys force the complete evaluator without a fast hit."""

import os
import sys
from types import ModuleType

import pytest

from identity_framework_fixture import ordinary_framework
from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_fast_path as fast


pytestmark = pytest.mark.usefixtures("ordinary_framework")
_KEY = "TRITON_MSL_FA_FAST"


class PolicyVeto(ValueError):
    pass


def _veto():
    raise PolicyVeto("colliding key replaced effective_policy")


def _warm():
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    fast._record = None
    stamp = cache.execution_contract()
    before = fast._stats["hits"]
    assert cache.execution_contract() == stamp
    assert fast._stats["hits"] == before + 1
    assert fast._record is not None
    return stamp


def _outcome(scenario, mode, api, name, baseline):
    original_policy = cache.effective_policy
    original_build = fast.build
    original_env = os.environ.get(_KEY)
    original_modules = sys.modules
    before_keys = {id(key) for key in sys.modules}
    calls = []
    holder = {}

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append(str(other))
            if scenario == "raise_valueerror":
                raise ValueError("colliding key equality veto")
            if scenario.startswith("replace_provider"):
                cache.effective_policy = _veto
            if scenario == "mutate_environment_false":
                os.environ[_KEY] = "0" if original_env != "0" else "1"
            if scenario == "replace_provider_remove_self":
                sys.modules.pop(holder["key"], None)
                return str.__eq__(self, other)
            return False if scenario.endswith("_false") else str.__eq__(self, other)

    key = Key(name)
    holder["key"] = key
    sys.modules[key] = ModuleType(name)
    if mode == "full":
        fast._record = None
        fast.build = lambda stamp: None
    hits = fast._stats["hits"]
    try:
        try:
            value = (
                cache.execution_contract()
                if api == "execution_contract"
                else cache.validate_execution_contract(baseline)
            )
            result = ("value", value == baseline, None, None, None)
        except BaseException as exc:
            result = (
                "exception",
                None,
                type(exc),
                str(exc),
                (type(exc.__cause__), str(exc.__cause__)) if exc.__cause__ else None,
            )
        return result, len(calls), fast._stats["hits"] - hits
    finally:
        sys.modules.pop(key, None)
        cache.effective_policy = original_policy
        fast.build = original_build
        fast._record = None
        if original_env is None:
            os.environ.pop(_KEY, None)
        else:
            os.environ[_KEY] = original_env
        assert sys.modules is original_modules
        assert {id(stored) for stored in sys.modules} == before_keys


_CASES = [
    ("count_true", "execution_contract", 3),
    ("count_false", "execution_contract", 1),
    ("replace_provider_true", "execution_contract", 3),
    ("replace_provider_true", "validate_execution_contract", 3),
    ("replace_provider_false", "execution_contract", 1),
    ("replace_provider_false", "validate_execution_contract", 1),
    ("replace_provider_remove_self", "execution_contract", 1),
    ("replace_provider_remove_self", "validate_execution_contract", 1),
    ("mutate_environment_false", "execution_contract", 1),
    ("raise_valueerror", "execution_contract", 1),
]


@pytest.mark.parametrize("scenario,api,expected_callbacks", _CASES)
def test_colliding_key_full_and_fast_are_equivalent(scenario, api, expected_callbacks):
    baseline = _warm()
    absent = [name for name in fw._SELECTION_NAMES if name not in sys.modules]
    assert absent
    full = _outcome(scenario, "full", api, absent[0], baseline)
    assert _warm() == baseline
    candidate = _outcome(scenario, "fast", api, absent[0], baseline)
    assert full[0] == candidate[0]
    assert full[1] == candidate[1] == expected_callbacks
    assert full[2] == candidate[2] == 0, "foreign-key arm must never record a fast hit"
    assert _warm() == baseline
