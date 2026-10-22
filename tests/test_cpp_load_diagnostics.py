"""CPU import and modeled stage diagnostics, never GPU execution evidence."""

import builtins
import gc
import sys
from types import SimpleNamespace
import warnings
import weakref

import pytest
from triton.backends.compiler import GPUTarget
from triton_msl.backend import (
    compiler,
    _cache_contract,
    _argument_abi,
    _launch_contract,
)

TARGET = "triton_msl._triton_msl_cpp"


@pytest.fixture
def imports(monkeypatch):
    original = builtins.__import__
    state = {"error": None, "calls": 0}

    def intercept(name, *args, **kwargs):
        if name != TARGET:
            return original(name, *args, **kwargs)
        state["calls"] += 1
        if state["error"] is not None:
            raise state["error"]
        return SimpleNamespace()

    monkeypatch.setattr(builtins, "__import__", intercept)
    return state


@pytest.mark.parametrize(
    "error,fragment",
    [
        (None, None),
        (ModuleNotFoundError("missing", name=TARGET), "module not found"),
        (
            ModuleNotFoundError("dependency", name="missing_dependency"),
            "failed to load",
        ),
        (ModuleNotFoundError("unknown"), "failed to load"),
        (ImportError("Symbol not found"), "failed to load"),
    ],
)
def test_single_attempt_status(imports, error, fragment):
    imports["error"] = error
    status = compiler.MetalBackend._cpp_load_failure()
    assert imports["calls"] == 1
    assert (status is None) if fragment is None else fragment in status


@pytest.mark.parametrize("error", [None, ImportError("broken"), ModuleNotFoundError(name=TARGET)])
def test_boolean_compatibility_single_attempt(imports, error):
    imports["error"] = error
    assert compiler.MetalBackend._has_cpp_passes() is (error is None)
    assert imports["calls"] == 1


@pytest.mark.parametrize("error", [RuntimeError("unchanged"), KeyboardInterrupt(), SystemExit(19)])
@pytest.mark.parametrize("method", ["_has_cpp_passes", "_cpp_load_failure"])
def test_non_import_errors_propagate_same_object(imports, error, method):
    imports["error"] = error
    with pytest.raises(type(error)) as caught:
        getattr(compiler.MetalBackend, method)()
    assert caught.value is error and imports["calls"] == 1


def test_custom_error_is_not_formatted_or_inspected(imports):
    class Hostile(ModuleNotFoundError):
        def __getattribute__(self, name):
            if name == "name":
                raise AssertionError("diagnostic inspected custom exception")
            return super().__getattribute__(name)

        def __str__(self):
            raise AssertionError("diagnostic formatted custom exception")

    imports["error"] = Hostile()
    assert "failed to load" in compiler.MetalBackend._cpp_load_failure()
    assert imports["calls"] == 1


def test_custom_name_is_not_compared(imports):
    class Name:
        def __eq__(self, other):
            raise AssertionError("diagnostic compared custom name")

    imports["error"] = ModuleNotFoundError(name=Name())
    assert "failed to load" in compiler.MetalBackend._cpp_load_failure()


def test_sequential_observations_have_no_stale_reason(imports):
    states = [
        (ModuleNotFoundError(name=TARGET), "module not found"),
        (ImportError(), "failed to load"),
        (None, None),
    ]
    for error, fragment in states:
        imports["error"] = error
        result = compiler.MetalBackend._cpp_load_failure()
        assert result is None if fragment is None else fragment in result
    assert imports["calls"] == 3


def test_exception_is_not_retained(imports):
    class LoadError(ImportError):
        pass

    error = LoadError("temporary")
    reference = weakref.ref(error)
    imports["error"] = error
    assert "failed to load" in compiler.MetalBackend._cpp_load_failure()
    imports["error"] = None
    del error
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "source,fragment",
    [
        (None, "module not found"),
        ("VALUE = 1\n", None),
        ("raise ImportError('synthetic loader failure')\n", "failed to load"),
        ("import _tm_diagnostic_missing_dependency_6bd52\n", "failed to load"),
    ],
)
def test_real_import_machinery(tmp_path, monkeypatch, source, fragment):
    # Real import machinery over disposable Python fixtures, not native-loader credit.
    import importlib
    import triton_msl

    if source is not None:
        (tmp_path / "_triton_msl_cpp.py").write_text(source)
    with monkeypatch.context() as patch:
        patch.setattr(triton_msl, "__path__", [str(tmp_path)])
        patch.delitem(sys.modules, TARGET, raising=False)
        patch.delattr(triton_msl, "_triton_msl_cpp", raising=False)
        importlib.invalidate_caches()
        try:
            status = compiler.MetalBackend._cpp_load_failure()
            assert status is None if fragment is None else fragment in status
        finally:
            sys.modules.pop(TARGET, None)
            if hasattr(triton_msl, "_triton_msl_cpp"):
                delattr(triton_msl, "_triton_msl_cpp")


@pytest.fixture
def stage(monkeypatch):
    # Modeled production stage body; explicit device/identity/ABI doubles.
    monkeypatch.setattr(compiler.MetalBackend, "_initialize_compilation_runtime", lambda self: None)
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: "test-only")
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda stamp: None)
    monkeypatch.setattr(_argument_abi, "finish_record", lambda *args: None)
    monkeypatch.setattr(_launch_contract, "seal_launch_metadata", lambda *args: None)
    monkeypatch.setattr(compiler.MetalBackend, "make_metallib", staticmethod(lambda *args: b"msl"))
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    monkeypatch.delenv("TRITON_MSL_FORCE_PYTHON", raising=False)
    backend = compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32))

    def construct():
        stages = {}
        backend.add_stages(stages, SimpleNamespace(num_warps=4))
        return stages

    return construct


@pytest.mark.parametrize(
    "error,fragment",
    [
        (ModuleNotFoundError(name=TARGET), "module not found"),
        (ImportError("bad image"), "failed to load"),
    ],
)
def test_diagnostic_reaches_stage(stage, imports, error, fragment):
    imports["error"] = error
    stages = stage()
    assert imports["calls"] == 1
    # The warning must describe the original attempt, without probing again.
    imports["error"] = RuntimeError("a second diagnostic import would fail")
    metadata = {"name": "diagnostic"}
    with pytest.warns(UserWarning, match=fragment):
        assert stages["metallib"]("MSL", metadata) == b"msl"
    assert metadata["binary_route"] == "msl"
    assert fragment in metadata["cpp_fallback_reason"] and imports["calls"] == 1


@pytest.mark.parametrize("forced", [False, True])
def test_default_or_forced_python_has_no_probe(stage, imports, monkeypatch, forced):
    if forced:
        monkeypatch.setenv("TRITON_MSL_FORCE_PYTHON", "1")
    else:
        monkeypatch.delenv("TRITON_MSL_USE_CPP")
    imports["error"] = RuntimeError("must not probe")
    stages = stage()
    metadata = {"name": "default"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert stages["metallib"]("MSL", metadata) == b"msl"
    assert not caught and imports["calls"] == 0
    assert metadata["cpp_fallback_reason"] is None
