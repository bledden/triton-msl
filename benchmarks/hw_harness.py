"""Hardware profiling + disassembly harness (WS0/C6).

Turns "optimal bounds given by the hardware" into a measured, per-kernel
number. For each kernel in the suite it reports:

  * GPU-timestamp timing (median/min/max), via Metal command-buffer
    GPUStartTime/GPUEndTime.
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
    available programmatically on Apple Silicon (only `timestamp`). Those
    require Xcode GPU capture / Instruments. The harness reports what IS
    obtainable and points at INSTRUMENTS.md for the rest.
  - Native-AGX disassembly is best-effort (applegpu is M1-era).

Usage:
    python benchmarks/hw_harness.py                 # full suite
    python benchmarks/hw_harness.py matmul_512_fp32 # one kernel
    python benchmarks/hw_harness.py --no-disasm     # skip disassembly
    python benchmarks/hw_harness.py --no-mlx        # skip MLX comparison

Output: reports/hw_harness/<date>/{kernel}.json + summary.md + baseline.json.
"""
import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import Metal
import Foundation

from triton_metal.profiling.metal_bench import MetalBenchmark
from triton_metal.profiling import roofline, disasm
from triton_metal.codegen.msl_emitter import (
    make_vector_add_kernel, make_silu_kernel, make_reduce_kernel,
    make_softmax_kernel, make_layer_norm_kernel, make_matmul_kernel,
    make_simdgroup_matmul_kernel, make_simdgroup_matmul_kernel_fast,
)

_SHARED = Metal.MTLResourceStorageModeShared


# ── Metal helpers (mirror benchmarks/bench_all.py) ──────────────────────────

def _compile(device, msl_src, kernel_name=None):
    """Compile MSL -> (function, pipeline). Auto-detects the kernel name from
    the source if not given (avoids name-mismatch bugs). Cached by hash."""
    if kernel_name is None:
        m = re.findall(r"kernel\s+void\s+(\w+)", msl_src)
        assert m, "no `kernel void <name>` found in MSL"
        kernel_name = m[0]
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
    return function, pipeline


def _fbuf(device, n, pattern="ramp"):
    buf = device.newBufferWithLength_options_(n * 4, _SHARED)
    if pattern == "ones":
        vals = [1.0] * n
    elif pattern == "small":
        vals = [((i % 17) - 8) * 0.01 for i in range(n)]
    else:  # ramp
        vals = [(i % 97) * 0.01 for i in range(n)]
    buf.contents().as_buffer(n * 4)[:] = struct.pack(f"{n}f", *vals)
    return buf


def _hbuf(device, n, pattern="small"):
    buf = device.newBufferWithLength_options_(n * 2, _SHARED)
    vals = ([((i % 17) - 8) * 0.01 for i in range(n)] if pattern == "small"
            else [(i % 97) * 0.01 for i in range(n)])
    buf.contents().as_buffer(n * 2)[:] = struct.pack(f"{n}e", *vals)
    return buf


def _empty(device, n, elt=4):
    return device.newBufferWithLength_options_(max(n, 1) * elt, _SHARED)


def _ubuf(device, value):
    buf = device.newBufferWithLength_options_(4, _SHARED)
    buf.contents().as_buffer(4)[:] = struct.pack("I", value)
    return buf


def _time_dispatch(bench, pipeline, buffers, n_groups, threads_per_tg,
                   warmup=10, rep=100):
    """GPU-timestamp timing for an explicit (n_groups x threads_per_tg)
    dispatch — handles both 1-D and tiled 2-D kernels uniformly."""
    grid = Metal.MTLSizeMake(n_groups, 1, 1)
    tg = Metal.MTLSizeMake(threads_per_tg, 1, 1)

    def once():
        cmd = bench.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, b in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(b, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return cmd

    for _ in range(warmup):
        once()
    us = []
    wall_us = []
    for _ in range(rep):
        _s = time.perf_counter()
        cmd = once()
        wall_us.append((time.perf_counter() - _s) * 1e6)
        us.append((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6)
    us.sort()
    wall_us.sort()
    nn = len(us)
    return {"median_us": us[nn // 2], "min_us": us[0], "max_us": us[-1],
            "p10_us": us[int(0.1 * (nn - 1))], "p90_us": us[int(0.9 * (nn - 1))],
            # Wall-clock per dispatch (commit -> waitUntilCompleted), measured the
            # SAME way MLX is (_bench_mlx: perf_counter + synchronize). Use this —
            # not the GPU-only median_us — for the MLX ratio, so both sides
            # include host/submit/sync overhead. Comparing our GPU-only time to
            # MLX wall-clock structurally flattered every ratio (audit #164).
            "wall_median_us": wall_us[nn // 2]}


# ── Kernel suite ────────────────────────────────────────────────────────────

@dataclass
class KernelSpec:
    name: str
    build: Callable[[], str]              # -> MSL source
    # setup(device) -> dict with: buffers, n_groups, threads_per_tg,
    # bytes, flops, dtype, [mlx: callable|None]
    setup: Callable
    dtype: str = "fp32"


_N16M = 16 * 1024 * 1024
_N8M = 8 * 1024 * 1024


def _grid1d(n, block=256):
    return (n + block - 1) // block, block


# MLX equivalents (return a no-arg callable that runs + evals the op).
def _mlx_unary(shape, fn):
    def make():
        import mlx.core as mx
        x = mx.random.normal(shape); mx.eval(x)
        def run():
            mx.eval(fn(mx, x))
        return run
    return make


def _mlx_matmul(M, N, K, dt):
    def make():
        import mlx.core as mx
        t = mx.float16 if dt == "fp16" else mx.float32
        a = (mx.random.normal((M, K)) * 0.01).astype(t)
        b = (mx.random.normal((K, N)) * 0.01).astype(t)
        mx.eval(a, b)
        def run():
            mx.eval(a @ b)
        return run
    return make


SUITE: List[KernelSpec] = [
    # ── memory-bound ──
    KernelSpec("vector_add_16M", lambda: make_vector_add_kernel(block_size=256),
        lambda d: dict(
            buffers=[_fbuf(d, _N16M), _fbuf(d, _N16M), _empty(d, _N16M), _ubuf(d, _N16M)],
            n_groups=_grid1d(_N16M)[0], threads_per_tg=256,
            bytes=3 * _N16M * 4, flops=_N16M,
            mlx=_mlx_add(_N16M))),
    KernelSpec("silu_16M", lambda: make_silu_kernel(block_size=256),
        lambda d: dict(
            buffers=[_fbuf(d, _N16M), _empty(d, _N16M), _ubuf(d, _N16M)],
            n_groups=_grid1d(_N16M)[0], threads_per_tg=256,
            bytes=2 * _N16M * 4, flops=4 * _N16M,
            mlx=_mlx_unary((_N16M,), lambda mx, x: x * mx.sigmoid(x)))),
    KernelSpec("reduce_sum_8M", lambda: make_reduce_kernel("reduce_sum", "sum", block_size=256),
        lambda d: dict(
            buffers=[_fbuf(d, _N8M), _empty(d, 1), _ubuf(d, _N8M)],
            n_groups=_grid1d(_N8M)[0], threads_per_tg=256,
            bytes=_N8M * 4, flops=_N8M,
            mlx=_mlx_unary((_N8M,), lambda mx, x: mx.sum(x)))),
    KernelSpec("softmax_128x4096", lambda: make_softmax_kernel(block_size=256),
        lambda d: (lambda rows=128, cols=4096: dict(
            buffers=[_fbuf(d, rows * cols), _empty(d, rows * cols), _ubuf(d, cols)],
            n_groups=rows, threads_per_tg=256,
            bytes=rows * cols * 4 * 3, flops=rows * cols * 5,
            mlx=_mlx_unary((rows, cols), lambda mx, x: mx.softmax(x, axis=-1))))()),
    KernelSpec("layernorm_128x4096", lambda: make_layer_norm_kernel(block_size=256),
        lambda d: (lambda rows=128, cols=4096: dict(
            buffers=[_fbuf(d, rows * cols), _fbuf(d, cols, "ones"),
                     _fbuf(d, cols, "small"), _empty(d, rows * cols), _ubuf(d, cols)],
            n_groups=rows, threads_per_tg=256,
            bytes=(rows * cols * 2 + cols * 2) * 4, flops=rows * cols * 6,
            mlx=_mlx_unary((rows, cols), lambda mx, x: (x - mx.mean(x, -1, keepdims=True))
                           * mx.rsqrt(mx.var(x, -1, keepdims=True) + 1e-5))))()),
    # ── compute-bound (the matmul gap — the WS1 target) ──
    KernelSpec("matmul_512_fp32_scalar",
        lambda: make_matmul_kernel(block_m=32, block_n=32, block_k=32),
        lambda d: (lambda M=512, N=512, K=512, bm=32, bn=32: dict(
            buffers=[_fbuf(d, M * K, "small"), _fbuf(d, K * N, "small"),
                     _empty(d, M * N), _ubuf(d, M), _ubuf(d, N), _ubuf(d, K)],
            n_groups=((M + bm - 1) // bm) * ((N + bn - 1) // bn),
            threads_per_tg=bm * bn,
            bytes=(M * K + K * N + M * N) * 4, flops=2 * M * N * K,
            mlx=_mlx_matmul(M, N, K, "fp32")))()),
    KernelSpec("matmul_512_fp32_simd",
        lambda: make_simdgroup_matmul_kernel_fast(dtype="fp32"),
        lambda d: (lambda M=512, N=512, K=512: dict(
            buffers=[_fbuf(d, M * K, "small"), _fbuf(d, K * N, "small"),
                     _empty(d, M * N), _ubuf(d, M), _ubuf(d, N), _ubuf(d, K)],
            n_groups=((M + 31) // 32) * ((N + 127) // 128), threads_per_tg=128,
            bytes=(M * K + K * N + M * N) * 4, flops=2 * M * N * K,
            mlx=_mlx_matmul(M, N, K, "fp32")))()),
    KernelSpec("matmul_1024_fp16_simd",
        lambda: make_simdgroup_matmul_kernel_fast(dtype="fp16"),
        lambda d: (lambda M=1024, N=1024, K=1024: dict(
            buffers=[_hbuf(d, M * K), _hbuf(d, K * N), _empty(d, M * N, 4),
                     _ubuf(d, M), _ubuf(d, N), _ubuf(d, K)],
            n_groups=((M + 31) // 32) * ((N + 127) // 128), threads_per_tg=128,
            bytes=(M * K + K * N) * 2 + M * N * 4, flops=2 * M * N * K,
            mlx=_mlx_matmul(M, N, K, "fp16")))(), dtype="fp16"),
    # Large sizes: long enough to run ~ms, so timing is stable and the true
    # compute-bound regime (where MLX's tuned kernels show their edge) is
    # visible. These are the decisive matmul-gap measurements.
    KernelSpec("matmul_2048_fp32_simd",
        lambda: make_simdgroup_matmul_kernel_fast(dtype="fp32"),
        lambda d: (lambda M=2048, N=2048, K=2048: dict(
            buffers=[_fbuf(d, M * K, "small"), _fbuf(d, K * N, "small"),
                     _empty(d, M * N), _ubuf(d, M), _ubuf(d, N), _ubuf(d, K)],
            n_groups=((M + 31) // 32) * ((N + 127) // 128), threads_per_tg=128,
            bytes=(M * K + K * N + M * N) * 4, flops=2 * M * N * K,
            mlx=_mlx_matmul(M, N, K, "fp32")))()),
    KernelSpec("matmul_2048_fp16_simd",
        lambda: make_simdgroup_matmul_kernel_fast(dtype="fp16"),
        lambda d: (lambda M=2048, N=2048, K=2048: dict(
            buffers=[_hbuf(d, M * K), _hbuf(d, K * N), _empty(d, M * N, 4),
                     _ubuf(d, M), _ubuf(d, N), _ubuf(d, K)],
            n_groups=((M + 31) // 32) * ((N + 127) // 128), threads_per_tg=128,
            bytes=(M * K + K * N) * 2 + M * N * 4, flops=2 * M * N * K,
            mlx=_mlx_matmul(M, N, K, "fp16")))(), dtype="fp16"),
    KernelSpec("matmul_4096_fp16_simd",
        lambda: make_simdgroup_matmul_kernel_fast(dtype="fp16"),
        lambda d: (lambda M=4096, N=4096, K=4096: dict(
            buffers=[_hbuf(d, M * K), _hbuf(d, K * N), _empty(d, M * N, 4),
                     _ubuf(d, M), _ubuf(d, N), _ubuf(d, K)],
            n_groups=((M + 31) // 32) * ((N + 127) // 128), threads_per_tg=128,
            bytes=(M * K + K * N) * 2 + M * N * 4, flops=2 * M * N * K,
            mlx=_mlx_matmul(M, N, K, "fp16")))(), dtype="fp16"),
]


def _mlx_add(n):
    def make():
        import mlx.core as mx
        a = mx.random.normal((n,)); b = mx.random.normal((n,)); mx.eval(a, b)
        def run():
            mx.eval(a + b)
        return run
    return make


# Suite members from the WS0/C6 spec not yet wired (need bespoke setup);
# logged so coverage is never silently capped.
PENDING = [
    "attention HEAD_DIM=32 (FA template; Q/K/V buffer setup)",
    "attention HEAD_DIM=64 (WS1 target — currently refused/templated)",
    "chained_reductions (out of MEPT scope; cooperative wrap-loop problem)",
]


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
    except Exception:
        return None


def run_one(device, bench, spec: KernelSpec, *, do_disasm=True, do_mlx=True):
    function, pipeline = _compile(device, spec.build())
    s = spec.setup(device)
    timing = _time_dispatch(bench, pipeline, s["buffers"],
                            s["n_groups"], s["threads_per_tg"])
    seconds = timing["median_us"] / 1e6
    rl = roofline.classify(s["bytes"], s["flops"], seconds, dtype=spec.dtype)
    refl = disasm.reflect_pipeline(pipeline)

    result = {
        "kernel": spec.name, "dtype": spec.dtype,
        "n_groups": s["n_groups"], "threads_per_tg": s["threads_per_tg"],
        "timing_us": timing,
        "roofline": rl.to_dict(),
        "reflection": refl.to_dict(),
        "roofline_summary": roofline.format_roofline(spec.name, rl),
    }

    if do_disasm:
        out = os.path.join(tempfile.gettempdir(), f"hw_harness_{spec.name}.archive")
        arch = disasm.serialize_native_archive(device, function, out)
        result["disasm"] = (disasm.disassemble_archive(arch).to_dict() if arch
                            else {"available": False, "reason": "archive failed"})

    if do_mlx and s.get("mlx") is not None:
        mlx_ms = _bench_mlx(s["mlx"])
        if mlx_ms is not None:
            # Symmetric: both sides wall-clock (audit #164). Fall back to GPU-only
            # only if wall wasn't recorded. Also report the GPU-only ratio so the
            # gap between kernel time and dispatch overhead is visible, not hidden.
            ours_ms = timing.get("wall_median_us", timing["median_us"]) / 1e3
            ours_gpu_ms = timing["median_us"] / 1e3
            result["mlx"] = {"mlx_median_ms": mlx_ms,
                             "ours_median_ms": ours_ms,
                             "ours_gpu_only_ms": ours_gpu_ms,
                             "ratio_ours_over_mlx": (ours_ms / mlx_ms) if mlx_ms else None,
                             "ratio_gpu_only_over_mlx": (ours_gpu_ms / mlx_ms) if mlx_ms else None,
                             "basis": "wall-clock both sides"}
    return result


# ── Reporting ───────────────────────────────────────────────────────────────

def _write_reports(results, out_dir):
    import json
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        with open(os.path.join(out_dir, f"{r['kernel']}.json"), "w") as f:
            json.dump(r, f, indent=2)
    baseline = {r["kernel"]: {
        "median_us": r["timing_us"]["median_us"],
        "bound": r["roofline"]["bound"],
        "limiting_pct": r["roofline"]["limiting_pct"],
        "mlx_ratio": r.get("mlx", {}).get("ratio_ours_over_mlx"),
    } for r in results}
    with open(os.path.join(out_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    lines = ["# HW harness summary", "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Device: {Metal.MTLCreateSystemDefaultDevice().name()}", "",
             "| kernel | GPU us | bound | % of roof | GB/s | TFLOP/s | "
             "disasm cov | MLX ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rl = r["roofline"]; d = r.get("disasm", {}); mlx = r.get("mlx", {})
        cov = f"{d.get('decode_coverage', 0) * 100:.0f}%" if d.get("available") else "n/a"
        ratio = (f"{mlx['ratio_ours_over_mlx']:.2f}x"
                 if mlx.get("ratio_ours_over_mlx") else "n/a")
        bound = ("⚠ SUSPECT" if rl.get("suspect_measurement") else rl["bound"])
        pct = ("—" if rl.get("suspect_measurement")
               else f"{rl['limiting_pct'] * 100:.1f}%")
        lines.append(
            f"| {r['kernel']} | {r['timing_us']['median_us']:.1f} | {bound} | "
            f"{pct} | {rl['achieved_gbps']:.1f} | "
            f"{rl['achieved_tflops']:.2f} | {cov} | {ratio} |")
    lines += ["", "## Pending suite members (not yet wired)", ""]
    lines += [f"- {p}" for p in PENDING]
    lines += ["", "## Honesty notes", "",
              "- Live GPU counters (ALU%, occupancy, registers) are not "
              "programmatically available on Apple Silicon (only `timestamp`);"
              " see docs/INSTRUMENTS.md for the Xcode/Instruments path.",
              "- Native-AGX disassembly is best-effort (applegpu is M1-era; "
              "M4 is AGX2) — coverage % reflects this.",
              "- Compute roofs are estimates (Apple does not publish GPU "
              "FLOPs); the 546 GB/s memory roof is Apple-published. Treat "
              "'% of roof' as indicative, not precise.",
              "- MLX ratio basis (audit #164): both sides are now timed by "
              "WALL-CLOCK (perf_counter around commit->wait, matching MLX's "
              "perf_counter+synchronize). Earlier ratios compared our GPU-only "
              "time to MLX wall-clock, which flattered us by the host/submit "
              "overhead MLX pays — those numbers were optimistic. "
              "ratio_gpu_only_over_mlx is also recorded for reference but is "
              "NOT an apples-to-apples comparison.",
              "- SMALL/FAST kernels (≲50 us: softmax, layernorm, 512 matmul) "
              "have large run-to-run timing variance — a single run can read "
              "above the roof (flagged ⚠ SUSPECT). For trustworthy "
              "compute-bound numbers, use larger problem sizes so each kernel "
              "runs ~ms; treat sub-50us rows as indicative only."]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="WS0/C6 hardware harness")
    ap.add_argument("kernels", nargs="*")
    ap.add_argument("--no-disasm", action="store_true")
    ap.add_argument("--no-mlx", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = Metal.MTLCreateSystemDefaultDevice()
    bench = MetalBenchmark()
    specs = ([s for s in SUITE if s.name in args.kernels]
             if args.kernels else SUITE)
    if args.kernels and not specs:
        print(f"available: {[s.name for s in SUITE]}")
        return 1

    results = []
    for spec in specs:
        print(f"-- {spec.name} ...", flush=True)
        try:
            r = run_one(device, bench, spec, do_disasm=not args.no_disasm,
                        do_mlx=not args.no_mlx)
            results.append(r)
            print("   " + r["roofline_summary"])
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
    print(f"pending (not silently dropped): {len(PENDING)} (see summary.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
