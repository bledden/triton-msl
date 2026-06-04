"""Static GPU-binary inspection for the hardware harness (WS0/C6).

What this can and cannot do — stated honestly, because over-promising here is
the same integrity failure as a silently-wrong kernel:

  RELIABLE (Apple public API):
    - Pipeline reflection: maxTotalThreadsPerThreadgroup, threadExecutionWidth,
      staticThreadgroupMemoryLength -> a threadgroup-occupancy estimate.
    - Serializing the compiled pipeline to a native binary archive
      (MTLBinaryArchive) — a fat Mach-O whose ``applegpu`` slice is the real
      native AGX machine code, available for offline analysis.

  BEST-EFFORT (reverse-engineered, partial on this GPU):
    - Native-AGX disassembly via the vendored ``applegpu`` decoder
      (third_party/applegpu, dougallj — REFERENCES.md [11]). applegpu targets
      the M1-era AGX ISA; the M4 is AGX2, so a fraction of instructions
      decode as ``<disassembly failed>``. We therefore report a *decode
      coverage* and never present the instruction mix as complete.

  NOT AVAILABLE programmatically (Apple limitation):
    - Live GPU counters (ALU utilization, live occupancy, register pressure).
      Apple does not expose per-kernel register usage to client code, and the
      device's only counter set is ``timestamp``. The Metal driver's
      ``n_regs`` is a placeholder for this reason. Full reliable M4 native
      disassembly and live counters require Xcode GPU capture / Instruments —
      see docs/INSTRUMENTS.md (Stage 3).

Everything here degrades gracefully: a failure at any step returns a record
with ``available=False`` and a reason, never an exception that aborts the
harness.
"""
from __future__ import annotations

import contextlib
import io
import os
import struct
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# cputype of the native Apple-GPU slice inside a serialized MTLBinaryArchive
# (the ``applegpu`` arch reported by ``file``).
_AGX_CPUTYPE = 16777235  # 0x01000013
# Apple's serialized GPU binary archives use a fat magic variant 0xCBFEBABE
# (note the 'cb', not the usual 'ca' of 0xCAFEBABE); accept both.
_FAT_MAGICS = (0xCAFEBABE, 0xCBFEBABE)
_MH_MAGIC_64 = 0xFEEDFACF
_MH_CIGAM_64 = 0xCFFAEDFE

# Path to the vendored applegpu decoder.
_APPLEGPU_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party", "applegpu")


@dataclass
class ReflectionMetrics:
    """Reliable static metrics from MTLComputePipelineState reflection."""

    max_total_threads_per_threadgroup: int
    thread_execution_width: int
    static_threadgroup_memory_bytes: int
    # threadgroups can run concurrently per core until the 32 KB TG-memory or
    # the 1024-thread limit binds; this is a coarse occupancy hint, not a
    # live measurement.
    occupancy_hint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DisasmResult:
    available: bool
    reason: str = ""
    archive_path: Optional[str] = None
    instruction_count: int = 0
    decoded_count: int = 0
    failed_count: int = 0
    decode_coverage: float = 0.0          # decoded / total (0..1)
    has_mma: bool = False                  # simdgroup-matrix / matmul op seen
    has_async_load: bool = False           # threadgroup async copy seen
    opcode_histogram: dict = field(default_factory=dict)
    sample: List[str] = field(default_factory=list)  # first decoded lines

    def to_dict(self) -> dict:
        return asdict(self)


def reflect_pipeline(pipeline) -> ReflectionMetrics:
    """Pull the reliable static metrics off a MTLComputePipelineState."""
    max_threads = int(pipeline.maxTotalThreadsPerThreadgroup())
    width = int(pipeline.threadExecutionWidth())
    tg_mem = int(pipeline.staticThreadgroupMemoryLength())
    if tg_mem > 0:
        # 32 KB per-core threadgroup memory roof.
        concurrent = max(1, 32 * 1024 // max(tg_mem, 1))
        hint = (f"TG-memory-bound: ~{concurrent} threadgroups/core "
                f"({tg_mem} B each vs 32 KB)")
    else:
        hint = "not TG-memory-bound (no static threadgroup memory)"
    return ReflectionMetrics(max_threads, width, tg_mem, hint)


def serialize_native_archive(device, function, out_path: str) -> Optional[str]:
    """Serialize a compute pipeline to a native binary archive.

    The result is a fat Mach-O containing the native AGX machine code for this
    GPU. Returns the path on success, None on failure (graceful).
    """
    try:
        import Metal
        from Foundation import NSURL
        desc = Metal.MTLBinaryArchiveDescriptor.alloc().init()
        archive, err = device.newBinaryArchiveWithDescriptor_error_(desc, None)
        if archive is None:
            return None
        cpd = Metal.MTLComputePipelineDescriptor.alloc().init()
        cpd.setComputeFunction_(function)
        ok, err = archive.addComputePipelineFunctionsWithDescriptor_error_(
            cpd, None)
        if not ok:
            return None
        url = NSURL.fileURLWithPath_(out_path)
        ok2, err2 = archive.serializeToURL_error_(url, None)
        return out_path if ok2 and os.path.exists(out_path) else None
    except Exception:
        return None


def _extract_agx_slice(archive_path: str) -> Optional[bytes]:
    """Return the native-AGX text bytes from a serialized binary archive.

    Parses the fat header to find the ``applegpu`` slice, then the thin
    Mach-O's ``__TEXT,__text`` section. Returns None on any parse failure.
    """
    try:
        data = open(archive_path, "rb").read()
        if len(data) < 8:
            return None
        magic = struct.unpack(">I", data[:4])[0]
        if magic not in _FAT_MAGICS:
            return None
        nfat = struct.unpack(">I", data[4:8])[0]
        slice_off = slice_size = None
        for i in range(nfat):
            base = 8 + i * 20  # fat_arch: cputype,cpusubtype,offset,size,align
            if base + 20 > len(data):
                break
            cputype, _sub, off, size, _align = struct.unpack(
                ">iIIII", data[base:base + 20])
            if cputype == _AGX_CPUTYPE:
                slice_off, slice_size = off, size
                break
        if slice_off is None:
            return None
        thin = data[slice_off:slice_off + slice_size]
        # The slice may be a thin Mach-O (extract __text) or raw AGX code.
        if len(thin) >= 4 and struct.unpack("<I", thin[:4])[0] in (
                _MH_MAGIC_64, _MH_CIGAM_64):
            text = _macho_text_section(thin)
            return text if text else thin
        return thin
    except Exception:
        return None


def _macho_text_section(thin: bytes) -> Optional[bytes]:
    """Find __TEXT,__text in a thin 64-bit Mach-O; return its bytes."""
    try:
        if len(thin) < 32:
            return None
        magic = struct.unpack("<I", thin[:4])[0]
        le = magic == _MH_MAGIC_64
        if magic not in (_MH_MAGIC_64, _MH_CIGAM_64):
            return None
        end = "<" if le else ">"
        ncmds = struct.unpack(end + "I", thin[16:20])[0]
        off = 32  # mach_header_64 size
        for _ in range(ncmds):
            if off + 8 > len(thin):
                break
            cmd, cmdsize = struct.unpack(end + "II", thin[off:off + 8])
            if cmd == 0x19:  # LC_SEGMENT_64
                segname = thin[off + 8:off + 24].split(b"\0")[0]
                nsects = struct.unpack(
                    end + "I", thin[off + 64:off + 68])[0]
                soff = off + 72  # first section_64
                for _s in range(nsects):
                    sect = thin[soff:soff + 80]
                    sectname = sect[:16].split(b"\0")[0]
                    if sectname == b"__text":
                        addr_off = struct.unpack(end + "Q", sect[40:48])[0]
                        size = struct.unpack(end + "Q", sect[40:48])[0]
                        s_offset = struct.unpack(end + "I", sect[48:52])[0]
                        s_size = struct.unpack(end + "Q", sect[40:48])[0]
                        return thin[s_offset:s_offset + s_size]
                    soff += 80
            off += cmdsize
        return None
    except Exception:
        return None


def disassemble_archive(archive_path: str, *, max_sample: int = 40) -> DisasmResult:
    """Best-effort native-AGX disassembly with honest coverage reporting.

    Degrades gracefully: import/parse/decode failures return
    ``available=False`` with a reason.
    """
    if not os.path.isdir(_APPLEGPU_DIR):
        return DisasmResult(False, reason=f"applegpu not vendored at {_APPLEGPU_DIR}",
                            archive_path=archive_path)
    agx = _extract_agx_slice(archive_path)
    if not agx:
        return DisasmResult(False, reason="could not extract native AGX slice",
                            archive_path=archive_path)
    if _APPLEGPU_DIR not in sys.path:
        sys.path.insert(0, _APPLEGPU_DIR)
    try:
        import applegpu  # vendored decoder
    except Exception as e:  # pragma: no cover - import guard
        return DisasmResult(False, reason=f"applegpu import failed: {e}",
                            archive_path=archive_path)

    decoded, failed = 0, 0
    histogram: dict = {}
    sample: List[str] = []
    has_mma = has_async = False
    n = len(agx)
    p = 0
    # applegpu prints "TODO: ..." to stdout for instruction forms it doesn't
    # fully model (common on the M4/AGX2). Swallow that noise — we count it as
    # reduced coverage, we don't want it in the harness output.
    # Mirror applegpu's own decode loop (disassemble.py): turn the bytes at p
    # into a number, find the first matching instruction descriptor, and use
    # its decode_size to advance. No match (or a "TODO"/unmodeled form) ->
    # failed, advance by 2 (the AGX minimum instruction granularity).
    #
    # applegpu prints "TODO: ..." to stdout for instruction forms it doesn't
    # fully model (common on the M4/AGX2). Swallow that noise under a stdout
    # redirect with try/finally so stdout is always restored — we count those
    # as reduced coverage, we don't want them in the harness output.
    _saved_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        while p < n - 1:
            try:
                num = applegpu.opcode_to_number(agx[p:])
            except Exception:
                failed += 1
                p += 2
                continue
            length = 2
            asm = None
            mnem = None
            for o in applegpu.instruction_descriptors:
                try:
                    if o.matches(num):
                        mnem = o.decode_mnem(num)
                        length = o.decode_size(num)
                        asm = str(o.disassemble(num, pc=p))
                        break
                except Exception:
                    asm = None
                    continue
            ok = (asm is not None and length >= 2 and length % 2 == 0
                  and "TODO" not in asm and "failed" not in asm)
            if ok:
                decoded += 1
                histogram[mnem] = histogram.get(mnem, 0) + 1
                low = asm.lower()
                if any(k in low for k in ("matrix", "mma", "simd_matrix")):
                    has_mma = True
                if "async" in low:
                    has_async = True
                if len(sample) < max_sample:
                    sample.append(f"{p:#06x}: {asm}")
                if mnem == "stop":
                    break
                p += length
            else:
                failed += 1
                p += 2
    finally:
        sys.stdout = _saved_stdout

    total = decoded + failed
    coverage = (decoded / total) if total else 0.0
    return DisasmResult(
        available=True,
        reason=("partial: applegpu is M1-era; the M4 is AGX2 so some "
                "instructions do not decode" if coverage < 0.95 else ""),
        archive_path=archive_path,
        instruction_count=total,
        decoded_count=decoded,
        failed_count=failed,
        decode_coverage=coverage,
        has_mma=has_mma,
        has_async_load=has_async,
        opcode_histogram=dict(sorted(
            histogram.items(), key=lambda kv: -kv[1])[:20]),
        sample=sample,
    )
