"""Synthetic toolchain/framework trees for identity-record tests; nothing is executed.

The Mach-O images are data built by the existing contract-test helpers. Each
fixture gives the record a private directory and resets module state the way
a new process starts, so "restart" below means a fresh process's state.
"""

import builtins
import json
import os
from pathlib import Path
import pathlib
from types import SimpleNamespace

import pytest

# Import the cache contract (and with it the identity fast path, which captures
# framework internals at import) before any fixture replaces those internals.
from triton_msl.backend import _cache_contract  # noqa: F401
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_record as rec
from triton_msl.backend import _toolchain_contract as tc
from test_native_dependency_contract import MH_EXECUTE, _command, _image

LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_RPATH, LC_ID_DYLIB = 0xC, 0x80000018, 0x8000001C, 0xD


@pytest.fixture
def records(tmp_path, monkeypatch):
    directory = tmp_path / "records"
    monkeypatch.setenv(rec.DIRECTORY, str(directory))
    monkeypatch.delenv(rec.SWITCH, raising=False)
    monkeypatch.setattr(rec, "_outcomes", {})
    return directory


def outcome(kind):
    value = rec.last_outcome(kind)
    return None if value is None else value[0]


def record_file(directory, kind):
    files = sorted(directory.glob(f"{kind}-*.json"))
    assert len(files) == 1, files
    return files[0]


def restart_toolchain():
    tc._snapshot = tc._compiler_context = tc._environment_inputs = None


def restart_framework():
    fw._snapshot = fw._native_guard = fw._discovery_snapshot = fw._owned_selection = None


def strict_toolchain(monkeypatch):
    """The digest a strict-mode process computes now, leaving a fresh state behind."""
    restart_toolchain()
    monkeypatch.setenv(rec.SWITCH, "0")
    try:
        return tc.toolchain_identity()
    finally:
        monkeypatch.delenv(rec.SWITCH)
        restart_toolchain()


def strict_framework(monkeypatch):
    restart_framework()
    monkeypatch.setenv(rec.SWITCH, "0")
    try:
        return fw.framework_identity()
    finally:
        monkeypatch.delenv(rec.SWITCH)
        restart_framework()


@pytest.fixture
def toolchain(tmp_path, monkeypatch, records):
    """A writable Metal bundle, an SDK inside an Xcode bundle and a modeled xcrun."""
    for key in list(os.environ):
        if key.startswith("DYLD_") or key in tc._EXTERNAL_SEARCH:
            monkeypatch.delenv(key, raising=False)
    for name in ("_snapshot", "_compiler_context", "_environment_inputs"):
        monkeypatch.setattr(tc, name, None)
    root = tmp_path.resolve()
    bundle = root / "Metal.xctoolchain"
    version = bundle / "usr" / "metal" / "32023"
    for path in (bundle / "usr" / "bin", version / "bin", version / "lib"):
        path.mkdir(parents=True)
    (bundle / "usr" / "metal" / "current").symlink_to("32023")
    (bundle / "ToolchainInfo.plist").write_text("<plist>com.apple.dt.toolchain.Metal.32023</plist>\n")
    (version / "ToolchainInfo.json").write_text('{"Identifier": "32023"}\n')
    external = root / "external" / "libprovider.dylib"
    external.parent.mkdir()
    external.write_bytes(_image() + b"external provider v1")
    absent = root / "external" / "libweak.dylib"
    compiler = version / "bin" / "metal"
    compiler.write_bytes(
        _image(_command(LC_LOAD_DYLIB, str(external)), _command(LC_LOAD_WEAK_DYLIB, str(absent)), file_type=MH_EXECUTE)
    )
    for name in ("metal", "metallib"):
        (bundle / "usr" / "bin" / name).write_bytes(_image(file_type=MH_EXECUTE) + name.encode())
    (version / "lib" / "libmetal.dylib").write_bytes(_image() + b"compiler library")
    header = version / "lib" / "header.h"
    header.write_text("float metal_header(float x);\n")
    (version / "lib" / "other.h").write_text("float metal_other(float x);\n")
    (version / "lib" / "alias.h").symlink_to("header.h")
    contents = root / "Xcode.app" / "Contents"
    sdk = contents / "Developer" / "SDKs" / "MacOSX.sdk"
    (sdk / "usr" / "include").mkdir(parents=True)
    (sdk / "usr" / "include" / "stdio.h").write_text("int printf(const char *, ...);\n")
    (sdk / "SDKSettings.json").write_text('{"Version": "27.2"}\n')
    (sdk / "System" / "Library" / "CoreServices").mkdir(parents=True)
    (sdk / "System" / "Library" / "CoreServices" / "SystemVersion.plist").write_text("<plist>26B118</plist>\n")
    (contents / "version.plist").write_text("<plist>27.2 27B52</plist>\n")
    (contents / "_CodeSignature").mkdir()
    (contents / "_CodeSignature" / "CodeResources").write_text("<plist>seal</plist>\n")
    resolver = root / "bin" / "xcrun"
    resolver.parent.mkdir()
    resolver.write_bytes(_image(file_type=MH_EXECUTE) + b"resolver v1")
    state = SimpleNamespace(
        root=root,
        bundle=bundle,
        version=version,
        header=header,
        compiler=compiler,
        external=external,
        absent=absent,
        sdk=sdk,
        contents=contents,
        resolver=resolver,
        os_build="26B118",
        replies={
            ("--find", "metal"): str(bundle / "usr" / "bin" / "metal"),
            ("--find", "metallib"): str(bundle / "usr" / "bin" / "metallib"),
            ("metal", "--version"): f"Apple metal version 32023.922\nInstalledDir: {bundle}/usr/metal/current/bin",
            ("--show-sdk-path",): str(sdk),
            ("--show-sdk-build-version",): "27B52",
        },
    )

    def query(command, **kwargs):
        if command == ["/usr/sbin/sysctl", "-n", "kern.osversion"]:
            return state.os_build + "\n"
        assert command[:3] == [str(resolver), "-sdk", "macosx"]
        return state.replies[tuple(command[3:])]

    monkeypatch.setattr(tc.shutil, "which", lambda name: str(resolver))
    monkeypatch.setattr(tc.subprocess, "check_output", query)
    return state


@pytest.fixture
def readonly_bundle(monkeypatch):
    """Model a read-only mount for the bundle (temporary directories are writable)."""
    mount = [[7, True, 100, 200]]
    monkeypatch.setattr(tc, "_read_only_mount", lambda value: True)
    monkeypatch.setattr(rec, "_mount_identity", lambda path: list(mount[0]))
    return mount


@pytest.fixture
def framework(tmp_path, monkeypatch, records):
    """Package, stdlib and executable roots with a small loaded native graph."""
    for name in ("_snapshot", "_native_guard", "_discovery_snapshot", "_owned_selection"):
        monkeypatch.setattr(fw, name, None)
    root = tmp_path.resolve() / "frameworks"
    torch = root / "torch"
    (torch / "lib").mkdir(parents=True)
    (torch / "__init__.py").write_text("VERSION = 'unchanged'\n")
    shader = torch / "shader.h"
    shader.write_text("float shader_function(float x) { return x; }\n")
    (torch / "other.h").write_text("float other_function(float x) { return -x; }\n")
    (torch / "alias.h").symlink_to("shader.h")
    (torch / "__pycache__").mkdir()
    (torch / "__pycache__" / "module.cpython.pyc").write_bytes(b"bytecode is not identity")
    dependency = torch / "lib" / "libdep.dylib"
    dependency.write_bytes(_image(_command(LC_ID_DYLIB, "@rpath/libdep.dylib")) + b"dependency v1")
    external = tmp_path.resolve() / "outside" / "libexternal.dylib"
    external.parent.mkdir()
    external.write_bytes(_image() + b"external v1")
    absent = external.parent / "libweak.dylib"
    extension = torch / "_C.so"
    extension.write_bytes(
        _image(
            _command(LC_LOAD_DYLIB, "@rpath/libdep.dylib"),
            _command(LC_RPATH, "@loader_path/lib"),
            _command(LC_LOAD_DYLIB, str(external)),
            _command(LC_LOAD_WEAK_DYLIB, str(absent)),
        )
    )
    stdlib = root / "stdlib"
    (stdlib / "site-packages").mkdir(parents=True)
    (stdlib / "os.py").write_text("base implementation\n")
    (stdlib / "site-packages" / "unrelated.py").write_text("selected separately\n")
    platstdlib = root / "platstdlib"
    platstdlib.mkdir()
    executable = root / "bin" / "Python"
    executable.parent.mkdir()
    executable.write_bytes(_image(file_type=MH_EXECUTE) + b"interpreter")
    roots = {
        "torch": (torch,),
        "mlx": (),
        "python-stdlib": (stdlib,),
        "python-platstdlib": (platstdlib,),
        "python-executable": (executable,),
    }
    metadata = {"python_abi": "fixture", "python_version": "3.x", "byteorder": "little", "runtime_environment": []}
    names = [str(executable).encode(), str(extension).encode()]
    original_guard = fw._LoadedNativeGuard
    guards = []

    def guard(providers):
        result = object.__new__(original_guard)
        result.count = lambda: len(names)
        result.name = lambda index: names[index]
        result.providers = providers
        result.approved_names = set()
        result.checked_count = None
        result.checked_names = None
        guards.append(result)
        return result

    monkeypatch.setattr(fw, "_LoadedNativeGuard", guard)
    monkeypatch.setattr(fw, "_loaded_native_images", lambda: tuple(names))
    monkeypatch.setattr(fw, "_discover_selection", lambda: (roots, metadata))
    return SimpleNamespace(
        root=root,
        torch=torch,
        shader=shader,
        header=shader,
        dependency=dependency,
        external=external,
        absent=absent,
        extension=extension,
        executable=executable,
        roots=roots,
        metadata=metadata,
        names=names,
        guards=guards,
    )


def track_opens(monkeypatch):
    """Every path opened through os.open, open() or Path.open from now on."""
    opened = []
    real_os_open, real_open, real_path_open = os.open, builtins.open, pathlib.Path.open

    def os_open(path, *args, **kwargs):
        opened.append(("os", kwargs.get("dir_fd"), str(path)))
        return real_os_open(path, *args, **kwargs)

    def plain_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            opened.append(("open", None, os.fsdecode(file)))
        return real_open(file, *args, **kwargs)

    def path_open(self, *args, **kwargs):
        opened.append(("path", None, str(self)))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(builtins, "open", plain_open)
    monkeypatch.setattr(pathlib.Path, "open", path_open)
    return opened


def opened_inside(opened, root, allowed=()):
    """Opened paths under ``root`` other than ``allowed``; dir_fd opens are record files."""
    root, allowed = str(root), {str(path) for path in allowed}
    return [
        path
        for _how, dir_fd, path in opened
        if dir_fd is None and (path == root or path.startswith(root + os.sep)) and path not in allowed
    ]


def load_body(path):
    return json.loads(Path(path).read_text())


def write_signed(path, body):
    """A body with a valid corruption checksum, as a same-user writer could produce."""
    body = dict(body)
    body["record_sha256"] = rec._checksum(body)
    data = rec._canonical(body).encode()
    os.chmod(path, 0o600)
    Path(path).write_bytes(data)
    return data
