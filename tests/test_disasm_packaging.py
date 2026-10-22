"""CPU regressions for the installed, package-private AGX decoder."""

import builtins
import importlib
import struct
import sys
from types import ModuleType

import pytest

from triton_msl.profiling import disasm


def archive(tmp_path, payload=b"\x88\x00"):
    # Literal stop instruction; a fat archive containing raw AGX bytes.
    path = tmp_path / "static.archive"
    path.write_bytes(struct.pack(">IIiIIII", 0xCBFEBABE, 1, 0x01000013, 0, 28, len(payload), 4) + payload)
    return str(path)


def test_packaged_decoder_known_stop_without_path_mutation(tmp_path):
    before = sys.path[:]
    result = disasm.disassemble_archive(archive(tmp_path))
    assert result.available
    assert result.instruction_count == result.decoded_count == 1
    assert result.failed_count == 0
    assert result.decode_coverage == 1.0
    assert result.opcode_histogram == {"stop": 1}
    assert sys.path == before


def test_packaged_decoder_ignores_top_level_collisions(tmp_path, monkeypatch):
    for name in ("applegpu", "fma", "srgb"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    # Force fresh private imports even if another test warmed the decoder.
    package = "triton_msl._vendor.applegpu"
    for name in tuple(sys.modules):
        if name == package or name.startswith(package + "."):
            monkeypatch.delitem(sys.modules, name)
    import triton_msl._vendor as vendor

    if hasattr(vendor, "applegpu"):
        monkeypatch.delattr(vendor, "applegpu")
    result = disasm.disassemble_archive(archive(tmp_path))
    assert result.available and result.opcode_histogram == {"stop": 1}
    decoder = importlib.import_module(package + ".applegpu")
    assert decoder.fma.__name__ == package + ".fma"
    assert importlib.import_module(package + ".srgb").SRGB_TABLE is decoder.SRGB_TABLE
    for name in ("applegpu", "fma", "srgb"):
        assert vars(sys.modules[name]).keys() == ModuleType(name).__dict__.keys()


def test_packaged_decoder_import_failure_stays_graceful(tmp_path, monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "_vendor.applegpu":
            raise ImportError("deliberate private decoder refusal")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = disasm.disassemble_archive(archive(tmp_path))
    assert not result.available
    assert "deliberate private decoder refusal" in result.reason


@pytest.mark.parametrize("payload", [b"", b"bad", b"\xcb\xfe\xba\xbe", b"\0" * 32])
def test_packaged_decoder_malformed_archive_stays_unavailable(tmp_path, payload):
    path = tmp_path / "invalid.archive"
    path.write_bytes(payload)
    result = disasm.disassemble_archive(str(path))
    assert not result.available and result.reason
    assert result.instruction_count == 0


def test_packaged_decoder_sample_limit_keeps_counts(tmp_path):
    result = disasm.disassemble_archive(archive(tmp_path), max_sample=0)
    assert result.available and result.sample == []
    assert result.instruction_count == result.decoded_count == 1


def test_packaged_decoder_restores_stdout_on_failed_instruction(tmp_path, monkeypatch):
    from triton_msl._vendor.applegpu import applegpu

    class BadInstruction:
        def matches(self, number):
            print("not a decoded instruction")
            raise ValueError("deliberate decoder failure")

    monkeypatch.setattr(applegpu, "instruction_descriptors", [BadInstruction()])
    stdout = sys.stdout
    result = disasm.disassemble_archive(archive(tmp_path))
    assert sys.stdout is stdout
    assert result.available and result.instruction_count == result.failed_count == 1
    assert result.decoded_count == 0 and result.decode_coverage == 0
