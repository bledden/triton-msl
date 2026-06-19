"""Tests for the static GPU-binary inspection (WS0/C6).

Split into two groups:
  - pure / graceful-degradation tests (no GPU needed) — always run;
  - end-to-end tests that need a real Metal device — gated on requires_metal.
"""
import os
import struct

import pytest

from triton_msl.profiling import disasm

try:
    from tests.conftest import requires_metal
except Exception:  # pragma: no cover - fallback if conftest layout differs
    import Metal  # noqa
    requires_metal = pytest.mark.skipif(
        Metal.MTLCreateSystemDefaultDevice() is None,
        reason="no Metal device")


# ── graceful degradation (no GPU) ───────────────────────────────────────────

def test_disasm_missing_file_is_graceful():
    r = disasm.disassemble_archive("/nonexistent/path.archive")
    assert r.available is False and r.reason
    assert r.instruction_count == 0


def test_extract_agx_slice_rejects_non_fat(tmp_path):
    p = tmp_path / "junk.bin"
    p.write_bytes(b"not a mach-o fat binary at all")
    assert disasm._extract_agx_slice(str(p)) is None


def test_extract_agx_slice_accepts_cbfebabe_magic(tmp_path):
    # Build a minimal 0xCBFEBABE fat header with one applegpu slice whose
    # payload is raw (non-Mach-O) bytes -> returned as-is.
    payload = b"\xde\xad\xbe\xef" * 4
    slice_off = 8 + 20  # header(8) + one fat_arch(20)
    header = struct.pack(">II", 0xCBFEBABE, 1)
    fat_arch = struct.pack(">iIIII", disasm._AGX_CPUTYPE, 0,
                           slice_off, len(payload), 4)
    blob = header + fat_arch + payload
    p = tmp_path / "fat.bin"
    p.write_bytes(blob)
    got = disasm._extract_agx_slice(str(p))
    assert got == payload


def test_extract_agx_slice_no_matching_arch(tmp_path):
    payload = b"\x00" * 8
    header = struct.pack(">II", 0xCBFEBABE, 1)
    fat_arch = struct.pack(">iIIII", 0x01000099, 0, 28, len(payload), 4)
    p = tmp_path / "fat.bin"
    p.write_bytes(header + fat_arch + payload)
    assert disasm._extract_agx_slice(str(p)) is None


def test_disasm_result_serializes():
    r = disasm.DisasmResult(False, reason="x")
    d = r.to_dict()
    assert d["available"] is False and "decode_coverage" in d


# ── end-to-end (needs a Metal device + a cached metallib) ────────────────────

@requires_metal
def test_reflection_and_disasm_end_to_end(tmp_path):
    import glob
    import Metal
    from Foundation import NSURL

    libs = glob.glob(os.path.expanduser("~/.triton/cache/*/*.metallib"))
    if not libs:
        pytest.skip("no cached metallib to inspect")
    dev = Metal.MTLCreateSystemDefaultDevice()
    lib, _ = dev.newLibraryWithURL_error_(NSURL.fileURLWithPath_(libs[0]), None)
    if lib is None or not list(lib.functionNames()):
        pytest.skip("metallib did not load")
    fn = lib.newFunctionWithName_(list(lib.functionNames())[0])
    pso, _ = dev.newComputePipelineStateWithFunction_error_(fn, None)

    refl = disasm.reflect_pipeline(pso)
    assert refl.max_total_threads_per_threadgroup > 0
    assert refl.thread_execution_width == 32  # Apple SIMD width
    assert refl.occupancy_hint

    arch = disasm.serialize_native_archive(
        dev, fn, str(tmp_path / "a.archive"))
    assert arch and os.path.exists(arch)
    r = disasm.disassemble_archive(arch)
    # Best-effort: must succeed structurally and report HONEST coverage in
    # [0, 1]; partial decode on M4 is expected and fine.
    assert r.available is True
    assert 0.0 <= r.decode_coverage <= 1.0
    assert r.instruction_count == r.decoded_count + r.failed_count
