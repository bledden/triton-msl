"""CPU-only Metal-standard qualification, including the load/pipeline distinction."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_msl.backend import device_detect as dd
from triton_msl.errors import MetalCompilationError


@pytest.fixture
def probe(monkeypatch, tmp_path):
    from triton_msl.backend import _toolchain_contract as tc

    monkeypatch.setattr(tc, "compiler_process_context", lambda: ("xcrun", str(tmp_path)))
    state = SimpleNamespace(version=None, failures={}, calls=[], paths=[], devices=0)

    def run(cmd, **kwargs):
        assert kwargs == dict(capture_output=True, timeout=10, check=True, cwd=str(tmp_path))
        if "metal" in cmd:
            state.version = next(x.removeprefix("-std=metal") for x in cmd if x.startswith("-std="))
            state.paths.append(Path(cmd[cmd.index("-c") + 1]))
            stage = "compile"
        else:
            stage = "link"
        state.calls.append((state.version, stage))
        if state.failures.get(state.version) == stage:
            raise dd.subprocess.CalledProcessError(1, cmd)
        state.paths.append(Path(cmd[-1]))
        Path(cmd[-1]).touch()

    class Library:
        def newFunctionWithName_(self, name):
            assert name == "_probe"
            state.calls.append((state.version, "function"))
            return None if state.failures.get(state.version) == "function" else object()

    class Device:
        def name(self):
            return "Apple M4 Max"

        def maxThreadsPerThreadgroup(self):
            return SimpleNamespace(width=1024)

        def newLibraryWithURL_error_(self, url, error):
            assert error is None
            state.calls.append((state.version, "load"))
            failure = state.failures.get(state.version)
            return (
                (None, "rejected")
                if failure == "load"
                else (None, None)
                if failure == "nil_library"
                else (Library(), None)
            )

        def newComputePipelineStateWithFunction_error_(self, function, error):
            assert function is not None and error is None
            state.calls.append((state.version, "pipeline"))
            failure = state.failures.get(state.version)
            return (
                (None, "rejected")
                if failure == "pipeline"
                else (None, None)
                if failure == "nil_pipeline"
                else (object(), None)
            )

    def device():
        state.devices += 1
        return Device()

    monkeypatch.setattr(dd.subprocess, "run", run)
    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=device))
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: p)))
    yield state
    assert all(not path.exists() for path in state.paths)


def test_success_requires_pipeline_creation(probe):
    assert dd._probe_metal_compiler() == "4.1"
    assert probe.calls == [("4.1", stage) for stage in ("compile", "link", "load", "function", "pipeline")]
    assert probe.devices == 1


@pytest.mark.parametrize("failure", ["compile", "link", "load", "nil_library", "function", "pipeline", "nil_pipeline"])
def test_rejected_candidate_tries_next_standard(probe, failure):
    probe.failures["4.1"] = failure
    assert dd._probe_metal_compiler() == "4.0"
    assert probe.calls[-1] == ("4.0", "pipeline")


def test_all_rejected_does_not_guess_from_sdk(probe):
    probe.failures = dict.fromkeys(("4.1", "4.0", "3.2", "3.1", "3.0"), "pipeline")
    with pytest.raises(MetalCompilationError, match="No usable Metal"):
        dd._infer_metal_version("27.0", "M4")


@pytest.mark.parametrize("module", ["Metal", "Foundation"])
def test_missing_runtime_is_unavailable_not_qualified(probe, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, None)
    assert dd._probe_metal_compiler() is None
    assert probe.calls == []


def test_no_device_is_unavailable_not_qualified(probe, monkeypatch):
    monkeypatch.setattr(sys.modules["Metal"], "MTLCreateSystemDefaultDevice", lambda: None)
    assert dd._probe_metal_compiler() is None
    assert probe.calls == []


def test_missing_compiler_is_unavailable(probe, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("xcrun")

    monkeypatch.setattr(dd.subprocess, "run", missing)
    assert dd._probe_metal_compiler() is None


def test_pre_m4_qualifies_selected_standard_instead_of_clamping(probe):
    probe.failures["3.2"] = "pipeline"
    assert dd._infer_metal_version("27.0", "M1") == "3.1"
    assert not any(version.startswith("4") for version, _ in probe.calls)
    assert probe.calls[-1] == ("3.1", "pipeline")


def test_m4_can_keep_pipeline_qualified_metal4(probe):
    assert dd._infer_metal_version("27.0", "M4") == "4.1"
    assert probe.calls[-1] == ("4.1", "pipeline")


def test_unavailable_probe_retains_documented_heuristic(monkeypatch):
    monkeypatch.setattr(dd, "_probe_metal_compiler", lambda **kwargs: None)
    assert dd._infer_metal_version("14.0", "M1") == "3.1"


@pytest.mark.parametrize("exception", [dd.subprocess.TimeoutExpired("xcrun", 10), OSError("compiler failed")])
def test_transient_compiler_failure_tries_next_candidate(probe, monkeypatch, exception):
    original = dd.subprocess.run
    calls = 0

    def fail_once(cmd, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Register the source so the fixture also checks failure cleanup.
            probe.paths.append(Path(cmd[cmd.index("-c") + 1]))
            raise exception
        return original(cmd, **kwargs)

    monkeypatch.setattr(dd.subprocess, "run", fail_once)
    assert dd._probe_metal_compiler() == "4.0"
    assert probe.calls[-1] == ("4.0", "pipeline")


def test_unexpected_runtime_exception_is_not_success_or_sdk_fallback(probe, monkeypatch):
    class BrokenDevice:
        def newLibraryWithURL_error_(self, url, error):
            raise RuntimeError("unexpected runtime failure")

    monkeypatch.setattr(sys.modules["Metal"], "MTLCreateSystemDefaultDevice", BrokenDevice)
    with pytest.raises(RuntimeError, match="unexpected runtime failure"):
        dd._infer_metal_version("27.0", "M4")


@pytest.mark.parametrize("stage", ["compile", "link"])
@pytest.mark.parametrize("as_bytes", [True, False])
def test_missing_toolchain_stops_at_first_failure_with_install_hint(probe, monkeypatch, stage, as_bytes):
    diagnostic = "cannot execute tool 'metal' due to missing Metal Toolchain"
    original = dd.subprocess.run

    def missing(cmd, **kwargs):
        current = "compile" if "metal" in cmd else "link"
        if current != stage:
            return original(cmd, **kwargs)
        if current == "compile":
            probe.paths.append(Path(cmd[cmd.index("-c") + 1]))
        probe.calls.append(("4.1", current))
        raise dd.subprocess.CalledProcessError(1, cmd, stderr=diagnostic.encode() if as_bytes else diagnostic)

    monkeypatch.setattr(dd.subprocess, "run", missing)
    with pytest.raises(MetalCompilationError, match="sudo xcodebuild -downloadComponent MetalToolchain") as caught:
        dd._probe_metal_compiler()
    assert caught.value.stderr == diagnostic
    assert "silently-wrong" not in str(caught.value)
    assert probe.calls == [("4.1", step) for step in (("compile",) if stage == "compile" else ("compile", "link"))]


def test_rejected_probe_does_not_cache_device_info(probe, monkeypatch):
    probe.failures = dict.fromkeys(("4.1", "4.0", "3.2", "3.1", "3.0"), "pipeline")
    monkeypatch.setattr(dd, "_cached_info", None)
    monkeypatch.setattr(dd, "_detect_sdk_version", lambda: "27.0")
    with pytest.raises(MetalCompilationError, match="No usable Metal"):
        dd.get_device_info()
    assert dd._cached_info is None


def test_missing_toolchain_diagnostic_does_not_import_compiler():
    """Keep device-only callers independent of Triton backend discovery."""
    code = """
import subprocess, sys
from types import SimpleNamespace
sys.modules['triton'] = None
sys.modules['Metal'] = SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: object())
sys.modules['Foundation'] = SimpleNamespace()
from triton_msl.backend.device_detect import _probe_metal_compiler
from triton_msl.backend import _toolchain_contract as tc
tc.compiler_process_context = lambda: ('xcrun', '.')
from triton_msl.errors import MetalCompilationError
def fail(cmd, **kwargs):
    raise subprocess.CalledProcessError(1, cmd, stderr=b'missing Metal Toolchain')
subprocess.run = fail
try:
    _probe_metal_compiler()
except MetalCompilationError as error:
    assert 'sudo xcodebuild -downloadComponent MetalToolchain' in str(error)
else:
    raise AssertionError('missing toolchain was not reported')
assert 'triton_msl.backend.compiler' not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    from tests.python_child import package_child_command

    result = dd.subprocess.run(package_child_command(code), env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
