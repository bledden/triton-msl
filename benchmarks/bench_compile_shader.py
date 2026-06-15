"""Benchmark compile_shader zero-copy fast-path ON vs OFF.

Measures GB/s for four memory-bound Triton kernels:
  - vector_add (3 tensors, 16M float32)
  - elementwise fma (x*2+1, 2 tensors, 16M float32)
  - softmax (8K×1K float32)
  - 1-D sum reduction (16M float32)

Runs each measurement in a *separate subprocess* so the in-process JIT and
Metal kernel cache are completely clean for each flag value.

Usage:
    rm -rf ~/.cache/triton_metal ~/.triton/cache
    python benchmarks/bench_compile_shader.py
"""

import json
import os
import subprocess
import sys

# M4 Max memory bandwidth (GB/s)
PEAK_BW = 546.0

# ---------------------------------------------------------------------------
# Inner measurement script — runs in child process with the flag already set
# ---------------------------------------------------------------------------

_INNER = r'''
import os, sys, json, torch, triton, triton.language as tl
from triton.testing import do_bench

FLAG = os.environ.get("TRITON_METAL_COMPILE_SHADER", "1")

# ── kernels ──────────────────────────────────────────────────────────────────

@triton.jit
def _vadd(A, B, OUT, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(OUT + o, tl.load(A + o, mask=m) + tl.load(B + o, mask=m), mask=m)


@triton.jit
def _elem(X, OUT, N, BLOCK: tl.constexpr):
    """elementwise: y = x*2 + 1"""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(OUT + o, tl.load(X + o, mask=m) * 2.0 + 1.0, mask=m)


@triton.jit
def _softmax(X, OUT, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + row * N + o, mask=m, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(OUT + row * N + o, e / s, mask=m)


@triton.jit
def _reduce(X, OUT, N, BLOCK: tl.constexpr):
    """1-D sum reduction pass-1: each program sums its BLOCK-chunk and writes
    to a per-program output slot. A full global reduce requires a second pass,
    but this measures the dominant memory-bandwidth phase."""
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + o, mask=m, other=0.0)
    tl.store(OUT + pid, tl.sum(x, axis=0))


# ── helpers ───────────────────────────────────────────────────────────────────

def gbps(bytes_moved, ms):
    return bytes_moved / ms * 1e-6   # bytes / ms -> GB/s


def warmup_and_bench(fn):
    """Warmup first, then take min of 3 do_bench calls."""
    # warmup
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    times = [do_bench(fn, warmup=25, rep=100, return_mode="min") for _ in range(3)]
    return min(times)


# ── benchmark bodies ──────────────────────────────────────────────────────────

N16M = 16 * 1024 * 1024
BLOCK = 1024

def bench_vadd():
    A = torch.randn(N16M, device="mps", dtype=torch.float32)
    B = torch.randn(N16M, device="mps", dtype=torch.float32)
    OUT = torch.empty(N16M, device="mps", dtype=torch.float32)
    grid = (triton.cdiv(N16M, BLOCK),)
    fn = lambda: _vadd[grid](A, B, OUT, N16M, BLOCK=BLOCK)
    ms = warmup_and_bench(fn)
    # 3 tensors × N16M × 4 bytes: read A+B, write OUT
    return ms, gbps(3 * N16M * 4, ms)


def bench_elem():
    N = N16M
    X = torch.randn(N, device="mps", dtype=torch.float32)
    OUT = torch.empty(N, device="mps", dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK),)
    fn = lambda: _elem[grid](X, OUT, N, BLOCK=BLOCK)
    ms = warmup_and_bench(fn)
    # read X, write OUT
    return ms, gbps(2 * N * 4, ms)


def bench_softmax():
    R, C = 8192, 1024
    SBLOCK = triton.next_power_of_2(C)  # 1024
    X = torch.randn(R, C, device="mps", dtype=torch.float32)
    OUT = torch.empty(R, C, device="mps", dtype=torch.float32)
    grid = (R,)
    fn = lambda: _softmax[grid](X, OUT, C, BLOCK=SBLOCK)
    ms = warmup_and_bench(fn)
    # read X, write OUT  (2 * R*C * 4 bytes)
    return ms, gbps(2 * R * C * 4, ms)


def bench_reduce():
    N = N16M
    RBLOCK = BLOCK
    nblocks = triton.cdiv(N, RBLOCK)
    X = torch.randn(N, device="mps", dtype=torch.float32)
    # Each block writes its partial sum to OUT[pid]
    OUT = torch.empty(nblocks, device="mps", dtype=torch.float32)
    grid = (nblocks,)
    fn = lambda: _reduce[grid](X, OUT, N, BLOCK=RBLOCK)
    ms = warmup_and_bench(fn)
    # Read N floats; write nblocks floats (negligible vs N)
    return ms, gbps(N * 4, ms)


# ── main ─────────────────────────────────────────────────────────────────────

results = {}

ms, g = bench_vadd()
results["vector_add"] = {"ms": ms, "gbps": g}

ms, g = bench_elem()
results["elementwise"] = {"ms": ms, "gbps": g}

ms, g = bench_softmax()
results["softmax"] = {"ms": ms, "gbps": g}

ms, g = bench_reduce()
results["reduction"] = {"ms": ms, "gbps": g}

print(json.dumps(results))
sys.stdout.flush()
'''

# ---------------------------------------------------------------------------
# Orchestrator — spawns two child processes
# ---------------------------------------------------------------------------

def run_one_flag(flag: str) -> dict:
    """Run the inner benchmark in a fresh process with the given flag."""
    # Clear caches before each run
    subprocess.run(["rm", "-rf", os.path.expanduser("~/.cache/triton_metal"),
                    os.path.expanduser("~/.triton/cache")],
                   check=False, capture_output=True)
    env = os.environ.copy()
    env["TRITON_METAL_COMPILE_SHADER"] = flag
    # Remove any inherited value so the child sees only our flag
    result = subprocess.run(
        [sys.executable, "-c", _INNER],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"[bench flag={flag}] STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Inner process exited {result.returncode}")
    # The last line of stdout is the JSON payload
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON found in output:\n{result.stdout}")


def fmt_row(name, on, off):
    peak = PEAK_BW
    pct_on  = on["gbps"]  / peak * 100
    pct_off = off["gbps"] / peak * 100
    speedup = on["gbps"] / off["gbps"] if off["gbps"] > 0 else float("inf")
    return (
        f"  {name:<16} "
        f"ON: {on['ms']:6.3f} ms  {on['gbps']:6.1f} GB/s  ({pct_on:5.1f}% peak)  "
        f"OFF: {off['ms']:6.3f} ms  {off['gbps']:6.1f} GB/s  ({pct_off:5.1f}% peak)  "
        f"speedup: {speedup:.1f}x"
    )


def main():
    print("=" * 100)
    print("compile_shader zero-copy fast-path benchmark")
    print(f"Peak bandwidth: {PEAK_BW} GB/s (M4 Max)")
    print("=" * 100)

    print("\nRunning with TRITON_METAL_COMPILE_SHADER=1 (fast-path ON) ...")
    on = run_one_flag("1")
    print("  done.")

    print("Running with TRITON_METAL_COMPILE_SHADER=0 (fast-path OFF) ...")
    off = run_one_flag("0")
    print("  done.\n")

    kernels = ["vector_add", "elementwise", "softmax", "reduction"]
    print(f"{'Kernel':<16}  {'ON ms':>9}  {'ON GB/s':>9}  {'ON %peak':>9}  "
          f"{'OFF ms':>9}  {'OFF GB/s':>9}  {'OFF %peak':>9}  {'speedup':>8}")
    print("-" * 100)
    results = {}
    for k in kernels:
        o = on[k];  f = off[k]
        pct_on  = o["gbps"] / PEAK_BW * 100
        pct_off = f["gbps"] / PEAK_BW * 100
        speedup = o["gbps"] / f["gbps"] if f["gbps"] > 0 else float("inf")
        print(f"  {k:<16} "
              f"{o['ms']:>9.3f}  {o['gbps']:>9.1f}  {pct_on:>8.1f}%  "
              f"{f['ms']:>9.3f}  {f['gbps']:>9.1f}  {pct_off:>8.1f}%  "
              f"{speedup:>7.1f}x")
        results[k] = {
            "on_ms": o["ms"], "on_gbps": o["gbps"], "on_pct_peak": round(pct_on, 1),
            "off_ms": f["ms"], "off_gbps": f["gbps"], "off_pct_peak": round(pct_off, 1),
            "speedup": round(speedup, 1),
        }
    print("=" * 100)
    print()

    # Emit JSON for downstream use (perf_baseline update)
    print("JSON summary:")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
