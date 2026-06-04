"""Hardware profiling + disassembly harness (WS0/C6).

Turns "optimal bounds given by the hardware" into a measured, per-kernel
number. For each kernel in the suite it reports:

  * GPU-timestamp timing (median/min/max), via MetalBenchmark.
  * Roofline classification: achieved GB/s & TFLOP/s, % of the M4 Max roofs,
    arithmetic intensity, and whether the kernel is memory- or compute-bound
    (roofline.py). This is the empirical "how close to the bound" number.
  * Static reflection: max threads/threadgroup, exec width, threadgroup
    memory, occupancy hint (disasm.py — reliable, Apple public API).
  * Best-effort native-AGX disassembly via vendored applegpu, reported with
    an explicit decode-coverage % (partial on M4/AGX2 — disasm.py).
  * MLX comparison ratio where an equivalent op exists.

Honesty notes (see disasm.py and docs/INSTRUMENTS.md):
  - Live GPU counters (ALU%, live occupancy, register pressure) are NOT
    available programmatically on Apple Silicon — only `timestamp`. Those
    require Xcode GPU capture / Instruments. The harness reports what IS
    obtainable and points at INSTRUMENTS.md for the rest.
  - Native-AGX disassembly is best-effort (applegpu is M1-era).

Usage:
    python benchmarks/hw_harness.py                 # full suite
    python benchmarks/hw_harness.py vector_add_16M  # one kernel
    python benchmarks/hw_harness.py --no-disasm     # skip disassembly
    python benchmarks/hw_harness.py --no-mlx        # skip MLX comparison

Output: reports/hw_harness/<date>/{kernel}.json + summary.md + baseline.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import Metal
import Foundation

from triton_metal.profiling.metal_bench import MetalBenchmark
from triton_metal.profiling import roofline, disasm
from triton_metal.codegen.msl_emitter import (
    make_vector_add_kernel, make_silu_kernel, make_reduce_kernel,
)

_F32 = 4
_HALF = Metal.MTLResourceStorageModeShared


# ── Metal helpers (mirror benchmarks/bench_all.py) ──────────────────────────

def _compile(device, msl_src, kernel_name):
    """Compile MSL -> (library, function, pipeline). Cached by source hash."""
    cache = os.path.join(tempfile.gettempdir(), "triton_metal_hw_harness")
    os.makedirs(cache, exist_ok=True)
    h = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
    base = os.path.join(cache, f"{kernel_name}_{h}")
    metallib = base + ".metallib"
    if not os.path.exists(metallib):
        with open(base + ".metal", "w") as f:
            f.write(msl_src)
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "-c", base + ".metal",
             "-o", base + ".air", "-std=metal3.2", "-O2"],
            stderr=subprocess.PIPE)
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metallib", base + ".air",
             "-o", metallib], stderr=subprocess.PIPE)
    url = Foundation.NSURL.fileURLWithPath_(metallib)
    library, err = device.newLibraryWithURL_error_(url, None)
    assert err is None, f"load: {err}"
    function = library.newFunctionWithName_(kernel_name)
    assert function is not None, f"no kernel {kernel_name}"
    pipeline, err = device.newComputePipelineStateWithFunction_error_(
        function, None)
    assert err is None, f"pipeline: {err}"
    return library, function, pipeline


def _f32_buf(device, n, pattern="ramp"):
    import struct
    buf = device.newBufferWithLength_options_(n * _F32, _HALF)
    ptr = buf.contents().as_buffer(n * _F32)
    vals = [(i % 97) * 0.01 for i in range(n)] if pattern == "ramp" else [1.0] * n
    ptr[:] = struct.pack(f"{n}f", *vals)
    return buf


def _empty_buf(device, n):
    return device.newBufferWithLength_options_(max(n, 1) * _F32, _HALF)


def _uint_buf(device, value):
    import struct
    buf = device.newBufferWithLength_options_(4, _HALF)
    buf.contents().as_buffer(4)[:] = struct.pack("I", value)
    return buf


# ── Kernel suite ────────────────────────────────────────────────────────────

@dataclass
class KernelSpec:
    name: str
    kernel_name: str                      # MSL function name
    build: Callable[[], str]              # -> MSL source
    setup: Callable                       # (device) -> (buffers, n_elements)
    bytes_model: Callable[[int], int]
    flops_model: Callable[[int], int]
    dtype: str = "fp32"
    mlx: Optional[Callable] = None        # (n) -> callable that runs MLX eq.
    block_size: int = 256


_N16M = 16 * 1024 * 1024
_N8M = 8 * 1024 * 1024


def _mlx_add(n):
    import mlx.core as mx
    a = mx.random.normal((n,)); b = mx.random.normal((n,)); mx.eval(a, b)
    def run():
        c = a + b; mx.eval(c)
    return run


def _mlx_silu(n):
    import mlx.core as mx
    x = mx.random.normal((n,)); mx.eval(x)
    def run():
        y = x * mx.sigmoid(x); mx.eval(y)
    return run


def _mlx_sum(n):
    import mlx.core as mx
    x = mx.random.normal((n,)); mx.eval(x)
    def run():
        y = mx.sum(x); mx.eval(y)
    return run


SUITE: List[KernelSpec] = [
    KernelSpec(
        name="vector_add_16M", kernel_name="vector_add",
        build=lambda: make_vector_add_kernel(block_size=256),
        setup=lambda d: ([_f32_buf(d, _N16M), _f32_buf(d, _N16M),
                          _empty_buf(d, _N16M), _uint_buf(d, _N16M)], _N16M),
        bytes_model=lambda n: 3 * n * _F32,   # 2 read + 1 write
        flops_model=lambda n: n,               # 1 add/elem
        dtype="fp32", mlx=_mlx_add),
    KernelSpec(
        name="silu_16M", kernel_name="silu_kernel",
        build=lambda: make_silu_kernel(block_size=256),
        setup=lambda d: ([_f32_buf(d, _N16M), _empty_buf(d, _N16M),
                          _uint_buf(d, _N16M)], _N16M),
        bytes_model=lambda n: 2 * n * _F32,   # 1 read + 1 write
        flops_model=lambda n: 4 * n,
        dtype="fp32", mlx=_mlx_silu),
    KernelSpec(
        name="reduce_sum_8M", kernel_name="reduce_sum",
        build=lambda: make_reduce_kernel("reduce_sum", "sum", block_size=256),
        setup=lambda d: ([_f32_buf(d, _N8M), _empty_buf(d, 1),
                          _uint_buf(d, _N8M)], _N8M),
        bytes_model=lambda n: n * _F32,        # read-dominated
        flops_model=lambda n: n,
        dtype="fp32", mlx=_mlx_sum),
]

# Suite members from the WS0/C6 spec not yet wired here (need 2-D dispatch /
# tile buffer setup). Logged explicitly so coverage is never silently capped;
# adding them is mechanical via KernelSpec.
PENDING = [
    "softmax_8Kx1K (row reduce; needs 2-D grid)",
    "layernorm_4Kx1K (row reduce; needs 2-D grid)",
    "matmul_512_fp32 / matmul_1K_fp16 (simdgroup tile setup)",
    "attention HEAD_DIM=32 (FA template) / HEAD_DIM=64 (WS1.C target)",
    "chained_reductions (WS1.B target)",
]


# ── MLX timing ──────────────────────────────────────────────────────────────

def _bench_mlx(make_run, warmup=10, rep=50):
    try:
        import mlx.core as mx
        run = make_run()
        for _ in range(warmup):
            run()
        mx.synchronize()
        t = []
        for _ in range(rep):
            s = time.perf_counter(); run(); mx.synchronize()
            t.append((time.perf_counter() - s) * 1e3)
        t.sort()
        return t[len(t) // 2]
    except Exception as e:
        return None


# ── Run one kernel ──────────────────────────────────────────────────────────

def run_one(device, bench, spec: KernelSpec, *, do_disasm=True, do_mlx=True):
    library, function, pipeline = _compile(device, spec.build(), spec.kernel_name)
    buffers, n = spec.setup(device)

    timing = bench.time_kernel(pipeline, buffers, n, block_size=spec.block_size)
    seconds = timing["median_us"] / 1e6
    rl = roofline.classify(spec.bytes_model(n), spec.flops_model(n), seconds,
                           dtype=spec.dtype)
    refl = disasm.reflect_pipeline(pipeline)

    result = {
        "kernel": spec.name,
        "n_elements": n,
        "dtype": spec.dtype,
        "timing_us": {k: timing[k] for k in
                      ("median_us", "min_us", "max_us", "p10_us", "p90_us")},
        "roofline": rl.to_dict(),
        "reflection": refl.to_dict(),
        "roofline_summary": roofline.format_roofline(spec.name, rl),
    }

    if do_disasm:
        out = os.path.join(tempfile.gettempdir(),
                           f"hw_harness_{spec.name}.archive")
        arch = disasm.serialize_native_archive(device, function, out)
        if arch:
            result["disasm"] = disasm.disassemble_archive(arch).to_dict()
        else:
            result["disasm"] = {"available": False,
                                "reason": "archive serialization failed"}

    if do_mlx and spec.mlx is not None:
        mlx_ms = _bench_mlx(lambda: spec.mlx(n))
        if mlx_ms is not None:
            ours_ms = timing["median_us"] / 1e3
            result["mlx"] = {
                "mlx_median_ms": mlx_ms,
                "ours_median_ms": ours_ms,
                "ratio_ours_over_mlx": (ours_ms / mlx_ms) if mlx_ms else None,
            }

    return result


# ── Reporting ───────────────────────────────────────────────────────────────

def _write_reports(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        with open(os.path.join(out_dir, f"{r['kernel']}.json"), "w") as f:
            json.dump(r, f, indent=2)
    # baseline.json: median_us + limiting_pct per kernel, for regression diffs.
    baseline = {r["kernel"]: {
        "median_us": r["timing_us"]["median_us"],
        "bound": r["roofline"]["bound"],
        "limiting_pct": r["roofline"]["limiting_pct"],
    } for r in results}
    with open(os.path.join(out_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    # summary.md
    lines = ["# HW harness summary", "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Device: {Metal.MTLCreateSystemDefaultDevice().name()}", "",
             "| kernel | GPU us | bound | % of roof | GB/s | TFLOP/s | "
             "disasm cov | MLX ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rl = r["roofline"]
        d = r.get("disasm", {})
        cov = (f"{d.get('decode_coverage', 0) * 100:.0f}%"
               if d.get("available") else "n/a")
        mlx = r.get("mlx", {})
        ratio = (f"{mlx['ratio_ours_over_mlx']:.2f}x"
                 if mlx.get("ratio_ours_over_mlx") else "n/a")
        lines.append(
            f"| {r['kernel']} | {r['timing_us']['median_us']:.1f} | "
            f"{rl['bound']} | {rl['limiting_pct'] * 100:.1f}% | "
            f"{rl['achieved_gbps']:.1f} | {rl['achieved_tflops']:.2f} | "
            f"{cov} | {ratio} |")
    lines += ["", "## Pending suite members (not yet wired)", ""]
    lines += [f"- {p}" for p in PENDING]
    lines += ["", "## Honesty notes", "",
              "- Live GPU counters (ALU%, occupancy, registers) are not "
              "programmatically available on Apple Silicon (only `timestamp`);"
              " see docs/INSTRUMENTS.md for the Xcode/Instruments path.",
              "- Native-AGX disassembly is best-effort (applegpu is M1-era; "
              "the M4 is AGX2) — coverage % reflects this.",
              "- Compute roofs are estimates (Apple does not publish GPU "
              "FLOPs); memory roof (546 GB/s) is Apple-published."]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="WS0/C6 hardware harness")
    ap.add_argument("kernels", nargs="*", help="kernel names (default: all)")
    ap.add_argument("--no-disasm", action="store_true")
    ap.add_argument("--no-mlx", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = Metal.MTLCreateSystemDefaultDevice()
    bench = MetalBenchmark()
    specs = ([s for s in SUITE if s.name in args.kernels]
             if args.kernels else SUITE)
    if args.kernels and not specs:
        print(f"no matching kernels; available: {[s.name for s in SUITE]}")
        return 1

    results = []
    for spec in specs:
        print(f"-- {spec.name} ...", flush=True)
        try:
            r = run_one(device, bench, spec,
                        do_disasm=not args.no_disasm, do_mlx=not args.no_mlx)
            results.append(r)
            print("   " + r["roofline_summary"])
            if r.get("disasm", {}).get("available"):
                d = r["disasm"]
                print(f"   disasm: {d['decode_coverage'] * 100:.0f}% decoded "
                      f"({d['decoded_count']}/{d['instruction_count']}), "
                      f"mma={d['has_mma']} async={d['has_async_load']}")
            if r.get("mlx", {}).get("ratio_ours_over_mlx"):
                print(f"   vs MLX: {r['mlx']['ratio_ours_over_mlx']:.2f}x "
                      f"(ours {r['mlx']['ours_median_ms']:.3f}ms / "
                      f"MLX {r['mlx']['mlx_median_ms']:.3f}ms)")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "hw_harness", time.strftime("%Y-%m-%d"))
    _write_reports(results, out_dir)
    print(f"\nreports -> {out_dir}")
    print(f"pending suite members (not silently dropped): {len(PENDING)} "
          f"(see summary.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
