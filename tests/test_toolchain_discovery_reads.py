"""Discovery may omit discarded digests; final identity must still bind all bytes."""

from dataclasses import replace
import hashlib
from pathlib import Path
import struct

import pytest

from triton_msl.backend import _native_dependencies as native, _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError
from test_native_dependency_contract import ARM64, X86_64, _command, _image
from test_toolchain_native_dependencies import selection  # noqa: F401


def test_classifier_does_not_hash_but_selected_graph_does(tmp_path, monkeypatch):
    tool, library = tmp_path / "tool", tmp_path / "unused.dylib"
    tool_bytes = _image(file_type=2) + b"executable payload"
    tool.write_bytes(tool_bytes)
    library.write_bytes(_image() + b"unloaded payload")
    reads = []
    real = native._read_image

    def read(path, cpu_type, *, digest):
        reads.append((Path(path).name, digest))
        return real(path, cpu_type, digest=digest)

    monkeypatch.setattr(native, "_read_image", read)
    result = tc._toolchain_native_dependencies({"metal": tmp_path, "metallib": tmp_path}, {})
    assert sorted(reads) == [("tool", False), ("tool", True), ("unused.dylib", False)]
    assert result == {"metal-helper:tool": ((hashlib.sha256(tool_bytes).hexdigest(), ()),)}


@pytest.mark.parametrize("kind", ["executable", "dylib", "fat"])
def test_header_reader_preserves_all_parser_fields_except_digest(tmp_path, kind):
    arm = _image(
        _command(0xC, "/usr/lib/libSystem.B.dylib"),
        _command(0x8000001C, "@loader_path/lib"),
        file_type=2 if kind == "executable" else 6,
    )
    payload = arm + b"uninterpreted payload"
    if kind == "fat":
        x86 = _image(cpu=X86_64)
        start = 48
        payload = (
            struct.pack(">II", 0xCAFEBABE, 2)
            + struct.pack(">IIIII", X86_64, 0, start, len(x86), 0)
            + struct.pack(">IIIII", ARM64, 0, start + len(x86), len(payload), 0)
            + x86
            + payload
        )
    path = tmp_path / "image"
    path.write_bytes(payload)
    assert native._image_header(path, ARM64) == replace(native.parse_image(path, ARM64), digest=None)


@pytest.mark.parametrize("damage", ["short", "command_bounds", "unsupported_command", "wrong_cpu", "environment"])
def test_header_discovery_retains_parser_refusals(tmp_path, damage):
    payload = _image()
    if damage == "short":
        payload = payload[:12]
    elif damage == "command_bounds":
        payload = payload[:-1]
    elif damage == "unsupported_command":
        payload = _image(struct.pack("<4I", 0x3A, 16, 0, 0))
    elif damage == "wrong_cpu":
        payload = _image(cpu=X86_64)
    elif damage == "environment":
        payload = _image(_command(0x27, "DYLD_LIBRARY_PATH=/untracked"), file_type=2)
    path = tmp_path / "bad.dylib"
    path.write_bytes(payload)
    with pytest.raises(ValueError) as baseline:
        native.parse_image(path, ARM64)
    with pytest.raises(ValueError) as candidate:
        native._image_header(path, ARM64)
    assert str(candidate.value) == str(baseline.value)
    with pytest.raises(ValueError, match="native|Mach-O|load|command"):
        tc._toolchain_native_dependencies({"metal": tmp_path, "metallib": tmp_path}, {})


def test_header_read_still_refuses_detected_mutation(tmp_path, monkeypatch):
    path = tmp_path / "image"
    path.write_bytes(_image())
    real = native._parse

    def changing(data, cpu):
        result = real(data, cpu)
        # Keep size stable while making the timestamp witness deterministic.
        import os

        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(native, "_parse", changing)
    with pytest.raises(ValueError, match="changed while reading"):
        native._image_header(path, ARM64)


@pytest.mark.parametrize("kind", ["unloaded_dylib", "fat_other_slice"])
def test_unselected_payload_is_bound_by_inventory(selection, monkeypatch, kind):
    entries, _ = selection
    library = entries["helper"].parent / "unused.dylib"
    payload = _image() + b"OLD payload"
    if kind == "fat_other_slice":
        arm = _image()
        other = _image(cpu=X86_64) + b"OLD payload"
        start = 48
        payload = (
            struct.pack(">II", 0xCAFEBABE, 2)
            + struct.pack(">IIIII", ARM64, 0, start, len(arm), 0)
            + struct.pack(">IIIII", X86_64, 0, start + len(arm), len(other), 0)
            + arm
            + other
        )
    library.write_bytes(payload)
    first = tc.toolchain_identity()
    before = tc._resolve_toolchain()[1]["native_providers"]
    library.write_bytes(payload.replace(b"OLD payload", b"NEW payload"))
    monkeypatch.setattr(tc, "_snapshot", None)
    assert tc._resolve_toolchain()[1]["native_providers"] == before  # Not a loaded provider.
    assert tc.toolchain_identity() != first  # Still an installation input.


def test_selected_executable_payload_remains_in_graph(selection):
    entries, _ = selection
    tool = entries["helper"]
    before = tc._resolve_toolchain()[1]["native_providers"]
    tool.write_bytes(tool.read_bytes() + b"implementation changed")
    assert tc._resolve_toolchain()[1]["native_providers"] != before


def test_malformed_unselected_native_file_still_refuses_identity(selection):
    entries, _ = selection
    (entries["helper"].parent / "unused.dylib").write_bytes(_image(struct.pack("<4I", 0x3A, 16, 0, 0)))
    with pytest.raises(MetalNonRecoverableError, match="unsupported native load command"):
        tc.toolchain_identity()
    assert tc._snapshot is None
