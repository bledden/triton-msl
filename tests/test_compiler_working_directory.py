"""Compiler subprocess context, separate from resident-launch validation."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from triton_msl.backend import _toolchain_contract as tc
from test_native_dependency_contract import _framework, _tool, _version


@pytest.fixture
def selected(tmp_path, monkeypatch):
    bundle = tmp_path / "Selected.xctoolchain"
    selector = bundle / "usr/bin/metal"
    selector.parent.mkdir(parents=True)
    selector.write_bytes(b"selector")
    resolver = tmp_path / "xcrun"
    resolver.write_bytes(b"resolver")
    monkeypatch.setattr(tc, "_compiler_context", None)
    monkeypatch.setattr(tc.shutil, "which", lambda name: str(resolver))
    queries = []

    def query(command, **kwargs):
        queries.append(command)
        assert command == [str(resolver), "-sdk", "macosx", "--find", "metal"]
        return str(selector)

    monkeypatch.setattr(tc.subprocess, "check_output", query)
    return bundle, resolver, queries


def test_context_is_pinned_and_application_chdir_cannot_redirect_it(selected, tmp_path, monkeypatch):
    bundle, resolver, queries = selected
    expected = str(resolver), str(bundle)
    assert tc.compiler_process_context() == expected
    hostile = tmp_path / "application"
    (hostile / "@rpath").mkdir(parents=True)
    (hostile / "@rpath/comparison.dylib").write_bytes(b"not an installation input")
    monkeypatch.chdir(hostile)
    assert tc.compiler_process_context() == expected
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tc.subprocess, "run", run)
    tc.run_metal_tool(["xcrun", "metal", "input.metal"], check=True)
    assert calls == [([str(resolver), "metal", "input.metal"], {"cwd": str(bundle), "check": True})]
    assert len(queries) == 1


def test_changed_selection_refuses_before_subprocess(selected, monkeypatch):
    tc.compiler_process_context()
    monkeypatch.setenv("DEVELOPER_DIR", "/different/toolchain")
    monkeypatch.setattr(tc.subprocess, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    with pytest.raises(RuntimeError, match="selection changed.*restart"):
        tc.run_metal_tool(["xcrun", "metal", "input.metal"])


def test_sdk_query_uses_same_resolver_and_directory(selected, monkeypatch):
    from triton_msl.backend import device_detect as dd

    bundle, resolver, _ = selected
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"26.7\n", b"")

    monkeypatch.setattr(tc.subprocess, "run", run)
    assert dd._detect_sdk_version() == "26.7"
    assert calls == [
        (
            [str(resolver), "-sdk", "macosx", "--show-sdk-version"],
            dict(cwd=str(bundle), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, check=True),
        )
    ]


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("xcrun"), subprocess.CalledProcessError(1, "xcrun"), subprocess.TimeoutExpired("xcrun", 5)],
)
def test_unavailable_sdk_query_retains_none_result(selected, monkeypatch, error):
    from triton_msl.backend import device_detect as dd

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(tc.subprocess, "run", unavailable)
    assert dd._detect_sdk_version() is None


def test_sdk_query_does_not_hide_changed_selection(selected, monkeypatch):
    from triton_msl.backend import device_detect as dd

    tc.compiler_process_context()
    monkeypatch.setenv("DEVELOPER_DIR", "/different/toolchain")
    monkeypatch.setattr(tc.subprocess, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    with pytest.raises(RuntimeError, match="selection changed.*restart"):
        dd._detect_sdk_version()


@pytest.mark.parametrize("command,kwargs", [(["foreign", "metal"], {}), (["xcrun", "metal"], {"cwd": "/tmp"})])
def test_callers_cannot_override_resolver_or_cwd(command, kwargs):
    with pytest.raises(ValueError, match="selected resolver"):
        tc.run_metal_tool(command, **kwargs)


def test_inventory_and_launch_use_same_directory_and_reject_escape(tmp_path, monkeypatch):
    from triton_msl.backend import _native_dependencies as native

    bundle = tmp_path / "Selected.xctoolchain"
    bundle.mkdir(exist_ok=True)
    _framework(bundle / "Frameworks", "Example", install_name="@rpath/Example.dylib", version=_version(2))
    tool = _tool(bundle, versioned="@executable_path/../Frameworks")
    roots = {"metal": bundle, "metallib": bundle}
    recorded = []
    real = native.dependency_manifest

    def graph(*args, **kwargs):
        recorded.append(kwargs["working_directory"])
        return real(*args, **kwargs)

    monkeypatch.setattr(native, "dependency_manifest", graph)
    other = tmp_path / "application"
    (other / "@rpath").mkdir(parents=True)
    (other / "@rpath/Example.dylib").write_bytes(b"application shadow ignored")
    monkeypatch.chdir(other)
    manifests = tc._toolchain_native_dependencies(roots, {"tool": tool})
    assert recorded and set(recorded) == {bundle.resolve()}
    assert tc._working_directory_lookups({"native_providers": manifests}) == ("@rpath/Example.dylib",)
    # The same literal lookup in the controlled installation must refuse.
    (bundle / "@rpath").mkdir()
    (bundle / "@rpath/Example.dylib").write_bytes(b"installation shadow")
    with pytest.raises(ValueError, match="working directory shadows"):
        tc._toolchain_native_dependencies(roots, {"tool": tool})
    (bundle / "@rpath/Example.dylib").unlink()
    _framework(bundle / "Frameworks", "Escape", install_name="../untracked.dylib", version=_version(2))
    with pytest.raises(ValueError, match="escapes the inventoried"):
        tc._toolchain_native_dependencies(roots, {"tool": tool})


@pytest.mark.parametrize("route", ["msl", "llir"])
@pytest.mark.parametrize("relative_cache", [False, True])
def test_actual_compiler_call_sites_pass_fixed_cwd(route, relative_cache, tmp_path, monkeypatch):
    import triton  # noqa: F401
    from triton_msl.backend import compiler, device_detect as dd, _cache_contract as cache

    selected = tmp_path / "Toolchain.xctoolchain"
    selected.mkdir()
    cache_dir = tmp_path / "cache"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", "cache" if relative_cache else str(cache_dir))
    monkeypatch.setattr(tc, "compiler_process_context", lambda: ("/selected/xcrun", str(selected)))
    monkeypatch.setattr(cache, "toolchain_identity", lambda: "controlled-toolchain")
    monkeypatch.setattr(cache, "framework_identity", lambda: "controlled-framework")
    # Force cold detection with a visible fake device on every host. A real
    # device-less sandbox or an already warm cache used to hide the SDK query.
    monkeypatch.setattr(dd, "_cached_info", None)
    device = SimpleNamespace(name=lambda: "Apple M4 Max", maxThreadsPerThreadgroup=lambda: SimpleNamespace(width=1024))
    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: device))
    monkeypatch.setattr(dd, "_probe_metal_compiler", lambda **kwargs: None)
    calls = []
    queries = []

    def run(command, **kwargs):
        assert command[0] == "/selected/xcrun"
        assert kwargs["cwd"] == str(selected)
        if "--show-sdk-version" in command:
            assert command == ["/selected/xcrun", "-sdk", "macosx", "--show-sdk-version"]
            queries.append(command)
            return subprocess.CompletedProcess(command, 0, b"15.0\n", b"")
        # LLIR has '-x ir' after '-c'; verify file operands by suffix instead.
        inputs = [Path(p) for p in command if p.endswith((".ll", ".metal", ".air"))]
        assert inputs and inputs[0].is_absolute() and inputs[0].exists()
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"AIR" if "metal" in command else b"LIB")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    fn = compiler.MetalBackend.make_metallib if route == "msl" else compiler.MetalBackend.make_metallib_from_llir
    assert fn("controlled source", {"name": "cwd_pin"}, compiler.MetalOptions()) == b"LIB"
    assert len(calls) == 2
    assert len(queries) == (1 if route == "msl" else 0)
    if route == "msl":
        assert dd._cached_info.metal_version == "3.2"
        assert "-std=metal3.2" in calls[0]
        # Warm reuse must not repeat the query or compiler/linker calls.
        assert fn("controlled source", {"name": "cwd_pin"}, compiler.MetalOptions()) == b"LIB"
        assert len(queries) == 1 and len(calls) == 2
