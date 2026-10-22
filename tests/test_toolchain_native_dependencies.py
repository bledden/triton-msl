"""Toolchain subprocess providers are inputs too; no fixture binary is executed."""

import struct

import pytest

from triton_msl.backend import _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError


def _image(provider=None):
    command = b""
    if provider is not None:
        text = str(provider).encode() + b"\0"
        size = (24 + len(text) + 7) & ~7
        command = (struct.pack("<6I", 0xC, size, 24, 0, 0, 0) + text).ljust(size, b"\0")
    return struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 2, bool(command), len(command), 0, 0) + command


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_compiler_context", None)
    bundle, sdk = tmp_path / "Toolchain.xctoolchain", tmp_path / "SDK.sdk"
    selectors, actual = bundle / "usr/bin", bundle / "usr/metal/current/bin"
    for path in (selectors, actual, sdk):
        path.mkdir(parents=True)
    (sdk / "header.h").write_text("header\n")
    entries = {
        "selector": selectors / "metal",
        "linker": selectors / "metallib",
        "compiler": actual / "metal",
        "resolver": tmp_path / "xcrun",
        "helper": actual / "helper",
    }
    for path in entries.values():
        path.write_bytes(_image())
    external = tmp_path / "outside-bundle.dylib"
    external.write_bytes(_image() + b"first implementation")
    replies = {
        ("--find", "metal"): str(entries["selector"]),
        ("--find", "metallib"): str(entries["linker"]),
        ("metal", "--version"): f"same Metal version\nInstalledDir: {actual}",
        ("--show-sdk-path",): str(sdk),
        ("--show-sdk-build-version",): "same-sdk-build",
    }

    def query(command, **kwargs):
        if command == ["/usr/sbin/sysctl", "-n", "kern.osversion"]:
            return "same-os-build"
        assert command[:3] == [str(entries["resolver"]), "-sdk", "macosx"]
        return replies[tuple(command[3:])]

    monkeypatch.setattr(tc.shutil, "which", lambda name: str(entries["resolver"]))
    monkeypatch.setattr(tc.subprocess, "check_output", query)
    monkeypatch.setattr(tc, "_snapshot", None)
    for name in tc._EXTERNAL_SEARCH:
        monkeypatch.delenv(name, raising=False)
    return entries, external


@pytest.mark.parametrize("role", ["selector", "linker", "compiler", "resolver", "helper"])
def test_same_tool_versions_changed_external_bytes_rekey(selection, monkeypatch, role):
    entries, external = selection
    entries[role].write_bytes(_image(external))
    before = tc.toolchain_identity()
    external.write_bytes(_image() + b"replacement external implementation")
    monkeypatch.setattr(tc, "_snapshot", None)  # installation upgrade / restart
    assert tc.toolchain_identity() != before


@pytest.mark.parametrize("damage", ["missing", "unknown_token", "malformed"])
def test_unproved_tool_provider_refuses_before_cache_identity(selection, damage):
    entries, external = selection
    if damage == "missing":
        provider = external.parent / "missing-provider.dylib"
    elif damage == "unknown_token":
        provider = "@unproved/provider.dylib"
    else:
        provider = external
        external.write_bytes(b"not Mach-O")
    entries["compiler"].write_bytes(_image(provider))
    with pytest.raises(MetalNonRecoverableError, match="native|provider|Mach-O"):
        tc.toolchain_identity()
    assert tc._snapshot is None


@pytest.mark.parametrize("name", ["DYLD_FORCE_FLAT_NAMESPACE", "DYLD_BIND_AT_LAUNCH", "DYLD_FUTURE_PROVIDER_OVERRIDE"])
def test_unproved_dyld_policy_cannot_hide_outside_the_named_environment_list(monkeypatch, name):
    for key in list(tc.os.environ):
        if key.startswith("DYLD_") or key in tc._EXTERNAL_SEARCH:
            monkeypatch.delenv(key, raising=False)
    selected = tuple(tc.os.environ.get(key) for key in tc._SELECTION)
    monkeypatch.setattr(tc, "_snapshot", (selected, "previous-valid-toolchain"))
    monkeypatch.setenv(name, "1")
    with pytest.raises(MetalNonRecoverableError, match="untracked compiler search"):
        tc.toolchain_identity()
