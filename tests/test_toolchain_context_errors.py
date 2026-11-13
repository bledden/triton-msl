"""Selection refusal is distinguishable and never becomes a probe fallback."""

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton.backends

from triton_msl import errors
from triton_msl.backend import _toolchain_contract as tc, device_detect as dd, driver


@pytest.fixture
def selection(monkeypatch, tmp_path):
    bundle = tmp_path / "Selected.xctoolchain"
    selector = bundle / "usr/bin/metal"
    selector.parent.mkdir(parents=True)
    selector.write_bytes(b"selector")
    resolver = tmp_path / "xcrun"
    resolver.write_bytes(b"resolver")
    unbundled = tmp_path / "metal"
    unbundled.write_bytes(b"not a bundled selector")
    state = SimpleNamespace(result=str(selector), calls=0, mutate=False)
    monkeypatch.setattr(tc, "_compiler_context", None)
    monkeypatch.setattr(tc.shutil, "which", lambda _: str(resolver))

    def query(*args, **kwargs):
        state.calls += 1
        if state.mutate:
            monkeypatch.setenv("SDKROOT", "/changed-during-resolution")
        return state.result

    monkeypatch.setattr(tc.subprocess, "check_output", query)
    state.unbundled = str(unbundled)
    return state


def assert_selection_type(error):
    assert type(error) is getattr(errors, "MetalToolchainSelectionError", None)
    assert isinstance(error, RuntimeError)
    assert not isinstance(error, (OSError, errors.MetalCompilationError))
    assert "kernel pattern" not in str(error)


@pytest.mark.parametrize(
    "kind,message",
    [
        ("empty", "empty Metal compiler selection"),
        ("layout", "unproved metal toolchain layout"),
        ("during", "selection changed during compiler context construction"),
        ("after", "selection changed in this process; restart"),
    ],
)
def test_explicit_selection_refusals_are_typed(selection, monkeypatch, kind, message):
    saved = None
    if kind == "empty":
        selection.result = ""
    elif kind == "layout":
        selection.result = selection.unbundled
    elif kind == "during":
        selection.mutate = True
    else:
        tc.compiler_process_context()
        saved = tc._compiler_context
        monkeypatch.setenv("TOOLCHAINS", "changed-after-resolution")
    with pytest.raises(RuntimeError, match=message) as caught:
        tc.compiler_process_context()
    assert_selection_type(caught.value)
    assert tc._compiler_context is saved
    assert selection.calls == 1


@pytest.mark.parametrize("probe", ["sdk", "air", "metal", "inferred"])
def test_probe_propagates_refusal_without_retry_or_fallback(selection, monkeypatch, tmp_path, probe):
    selection.result = selection.unbundled
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: object()))
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace())

    def forbidden(*args, **kwargs):
        pytest.fail("compiler command executed after context refusal")

    monkeypatch.setattr(tc.subprocess, "run", forbidden)
    calls = {
        "sdk": dd._detect_sdk_version,
        "air": dd._probe_air_version,
        "metal": dd._probe_metal_compiler,
        "inferred": lambda: dd._infer_metal_version("27.0", "M4"),
    }
    with pytest.raises(RuntimeError, match="unproved metal toolchain layout") as caught:
        calls[probe]()
    assert_selection_type(caught.value)
    assert selection.calls == 1
    assert tc._compiler_context is None
    assert not [p for p in tmp_path.iterdir() if p.suffix in (".metal", ".air", ".ll", ".metallib")]


def test_activation_keeps_typed_refusal_inactive(selection, monkeypatch):
    selection.result = selection.unbundled
    monkeypatch.setattr(driver.platform, "system", lambda: "Darwin")

    def forbidden():
        pytest.fail("device accessed after context refusal")

    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=forbidden))
    with pytest.warns(RuntimeWarning, match="cannot select a usable Metal compiler.*unproved metal toolchain layout"):
        assert driver.MetalDriver.is_active() is False
    assert tc._compiler_context is None


def test_missing_resolver_retains_file_not_found(selection, monkeypatch):
    monkeypatch.setattr(tc.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="xcrun is unavailable"):
        tc.compiler_process_context()


@pytest.mark.parametrize("error", [PermissionError("resolver denied"), subprocess.CalledProcessError(1, "resolver")])
def test_resolution_failure_is_not_reclassified(selection, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tc.subprocess, "check_output", fail)
    with pytest.raises(type(error)) as caught:
        tc.compiler_process_context()
    assert caught.value is error
    assert tc._compiler_context is None


def test_identity_boundary_keeps_selection_error_as_cause(selection, monkeypatch):
    selection.result = selection.unbundled
    monkeypatch.setattr(tc, "_snapshot", None)
    monkeypatch.setattr(tc, "_environment_inputs", None)
    with pytest.raises(errors.MetalNonRecoverableError) as caught:
        tc.toolchain_identity()
    assert_selection_type(caught.value.__cause__)
    assert tc._snapshot is None


def test_error_import_has_no_compiler_or_runtime_dependency():
    code = (
        "import sys; from triton_msl.errors import MetalToolchainSelectionError as E; "
        "assert issubclass(E, RuntimeError); assert 'triton' not in sys.modules; "
        "assert 'triton_msl.backend.compiler' not in sys.modules"
    )
    from tests.python_child import package_child_command

    done = subprocess.run(package_child_command(code), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
