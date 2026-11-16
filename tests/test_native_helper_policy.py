"""CPU-only controls: native-test skips must never conceal a failed selection."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.native_helper_policy import require_selected_native


def _expect_error(error, helper, match, selected=None):
    # A mutant that restores the old blanket skip must FAIL these controls;
    # allowing pytest's skip exception to escape would hide the old defect.
    try:
        with pytest.raises(error, match=match):
            require_selected_native(selected, helper)
    except pytest.skip.Exception as exc:
        pytest.fail(f"native selection failure was concealed by a skip: {exc}")


@pytest.fixture(params=["validation", "binder"])
def absent(monkeypatch, request):
    helper = request.param
    name = f"triton_msl.backend._{helper}_native"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv(f"TRITON_MSL_TEST_REQUIRE_{helper.upper()}_NATIVE", raising=False)
    monkeypatch.delenv(f"TRITON_MSL_BUILD_{helper.upper()}_NATIVE", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda query: None)
    return helper, name


def test_proved_optional_absence_skips(absent):
    with pytest.raises(pytest.skip.Exception, match="genuinely absent"):
        require_selected_native(None, absent[0])


@pytest.mark.parametrize("flag", ["TEST_REQUIRE", "BUILD"])
def test_required_but_missing_fails(absent, monkeypatch, flag):
    helper, _ = absent
    monkeypatch.setenv(f"TRITON_MSL_{flag}_{helper.upper()}_NATIVE", "1")
    _expect_error(pytest.fail.Exception, helper, "required but is not selected")


@pytest.mark.parametrize("loaded", [None, ModuleType("native")])
def test_loaded_but_unselected_fails_even_with_no_spec(absent, monkeypatch, loaded):
    helper, name = absent
    monkeypatch.setitem(sys.modules, name, loaded)
    _expect_error(pytest.fail.Exception, helper, "present but is not selected")


def test_discoverable_but_unselected_fails(absent, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda query: object())
    _expect_error(pytest.fail.Exception, absent[0], "present but is not selected")


def test_discovery_failure_propagates(absent, monkeypatch):
    def broken(query):
        raise ImportError("broken extension discovery")

    monkeypatch.setattr(importlib.util, "find_spec", broken)
    _expect_error(ImportError, absent[0], "broken extension discovery")


def test_selected_helper_never_reimports_or_rediscovers(absent, monkeypatch):
    def forbidden(query):
        raise AssertionError("selected helper must not be rediscovered")

    monkeypatch.setattr(importlib.util, "find_spec", forbidden)
    require_selected_native(object(), absent[0])


def test_misspelled_lane_requirement_fails(absent, monkeypatch):
    monkeypatch.setenv(f"TRITON_MSL_TEST_REQUIRE_{absent[0].upper()}_NATIVE", "yes")
    _expect_error(pytest.fail.Exception, absent[0], "must be 0 or 1")


def _loader(path, name, namespace=None):
    """Exercise the real loader definition without importing the backend/frameworks."""
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _backend_file(name):
    # Resolve installed product bytes when this test is copied into an installed
    # lane; importing the package marker does not initialize a device/framework.
    import triton_msl

    return Path(triton_msl.__file__).resolve().parent / "backend" / name


def test_present_broken_validation_loader_stays_loud():
    select = _loader(_backend_file("_cache_contract.py"), "_select_validation_native")

    def broken(name):
        raise ImportError("bad native ABI")

    with pytest.raises(ImportError, match="bad native ABI"):
        select(broken, lambda name: object(), {})
    with pytest.raises(ImportError, match="no callable"):
        select(lambda name: SimpleNamespace(), lambda name: object(), {})


@pytest.mark.parametrize("absent", ["binder"], indirect=True)
def test_real_binder_configure_decline_fails_native_test(monkeypatch, absent):
    helper, name = absent
    monkeypatch.delitem(sys.modules, "triton_msl.backend._validation_native", raising=False)
    plan = object()
    bind = object()
    bind_plan = object()
    calls = []

    def configure(*args):
        calls.append(args)
        return False

    binder = SimpleNamespace(configure=configure)
    import importlib

    monkeypatch.setattr(importlib, "import_module", lambda query: binder)
    monkeypatch.setattr(importlib.util, "find_spec", lambda query: object() if query == name else None)
    select = _loader(
        _backend_file("_launch_signature.py"),
        "_select_binder_native",
        dict(
            __package__="triton_msl.backend", bind_arguments=bind, _BindingPlan=plan, bind_arguments_with_plan=bind_plan
        ),
    )
    # Product fallback remains valid; the native-specific test must fail.
    selected = select()
    assert selected is None and calls == [(bind, plan, bind_plan, None)]
    _expect_error(pytest.fail.Exception, helper, "present but is not selected", selected)
