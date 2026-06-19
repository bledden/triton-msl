"""Performance regression detection for triton-msl.

Runs key benchmarks and compares against a stored baseline.
Exits with non-zero status if any benchmark regresses >15%.

Usage:
    python benchmarks/bench_regression.py                    # Run and compare
    python benchmarks/bench_regression.py --update-baseline  # Save new baseline
    python benchmarks/bench_regression.py --json             # Print JSON results
"""

import argparse
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(SCRIPT_DIR, "..", "reports", "perf_baseline.json")
REGRESSION_THRESHOLD = 0.15  # 15% regression triggers failure


# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------

@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)


@triton.jit
def _sum_reduce_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(out_ptr, result)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _warmup_and_bench(fn, n_warmup=3, n_iter=10):
    """Run warmup iterations, then benchmark and return (min_ms, avg_ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return min(times), sum(times) / len(times)


def bench_vector_add(n=16 * 1024 * 1024):
    """Vector add throughput (GB/s)."""
    x = torch.randn(n, device="cpu")
    y = torch.randn(n, device="cpu")
    out = torch.empty(n, device="cpu")
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def run():
        _vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)

    min_ms, avg_ms = _warmup_and_bench(run)
    # 3 tensors * n * 4 bytes each
    gbps = (3 * n * 4) / (min_ms / 1000) / 1e9
    return {"name": "vector_add_16M", "min_ms": round(min_ms, 3),
            "avg_ms": round(avg_ms, 3), "throughput_gbps": round(gbps, 1)}


def bench_softmax(rows=8192, cols=1024):
    """Softmax throughput (GB/s)."""
    x = torch.randn(rows, cols, device="cpu")
    out = torch.empty_like(x)
    BLOCK = 1024

    def run():
        _softmax_kernel[(rows,)](x, out, cols, BLOCK_SIZE=BLOCK)

    min_ms, avg_ms = _warmup_and_bench(run)
    # Read + write = 2 * rows * cols * 4 bytes
    gbps = (2 * rows * cols * 4) / (min_ms / 1000) / 1e9
    return {"name": "softmax_8Kx1K", "min_ms": round(min_ms, 3),
            "avg_ms": round(avg_ms, 3), "throughput_gbps": round(gbps, 1)}


def bench_reduction(n=1024):
    """Reduction latency (ms)."""
    x = torch.randn(n, device="cpu")
    out = torch.zeros(1, device="cpu")
    BLOCK = 1024

    def run():
        _sum_reduce_kernel[(1,)](x, out, n, BLOCK_SIZE=BLOCK)

    min_ms, avg_ms = _warmup_and_bench(run)
    return {"name": "sum_reduce_1K", "min_ms": round(min_ms, 3),
            "avg_ms": round(avg_ms, 3)}


def bench_dispatch_overhead(n_dispatches=100):
    """Measure per-kernel dispatch overhead (ms)."""
    n = 1024
    x = torch.randn(n, device="cpu")
    y = torch.randn(n, device="cpu")
    out = torch.empty(n, device="cpu")
    BLOCK = 1024
    grid = (1,)

    # Warmup
    for _ in range(3):
        _vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)

    start = time.perf_counter()
    for _ in range(n_dispatches):
        _vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    total_ms = (time.perf_counter() - start) * 1000
    per_kernel_ms = total_ms / n_dispatches

    return {"name": "dispatch_overhead", "total_ms": round(total_ms, 1),
            "per_kernel_ms": round(per_kernel_ms, 3), "n_dispatches": n_dispatches}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [bench_vector_add, bench_softmax, bench_reduction, bench_dispatch_overhead]


def run_all():
    """Run all benchmarks and return results dict."""
    results = {}
    for bench_fn in ALL_BENCHMARKS:
        result = bench_fn()
        results[result["name"]] = result
        print(f"  {result['name']}: {result.get('min_ms', 'N/A')}ms min", file=sys.stderr)
    return results


def load_baseline():
    """Load baseline results from JSON file."""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)


def save_baseline(results):
    """Save results as new baseline."""
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline saved to {BASELINE_PATH}", file=sys.stderr)


def check_regression(current, baseline):
    """Compare current results against baseline. Returns (passed, details)."""
    regressions = []
    for name, cur in current.items():
        if name not in baseline:
            continue
        base = baseline[name]
        # Compare min_ms (lower is better)
        if "min_ms" in cur and "min_ms" in base and base["min_ms"] > 0:
            ratio = cur["min_ms"] / base["min_ms"]
            if ratio > 1 + REGRESSION_THRESHOLD:
                regressions.append(
                    f"  {name}: {base['min_ms']}ms → {cur['min_ms']}ms "
                    f"({(ratio - 1) * 100:.1f}% slower)"
                )
        # Compare per_kernel_ms for dispatch overhead
        if "per_kernel_ms" in cur and "per_kernel_ms" in base and base["per_kernel_ms"] > 0:
            ratio = cur["per_kernel_ms"] / base["per_kernel_ms"]
            if ratio > 1 + REGRESSION_THRESHOLD:
                regressions.append(
                    f"  {name}: {base['per_kernel_ms']}ms → {cur['per_kernel_ms']}ms "
                    f"({(ratio - 1) * 100:.1f}% slower)"
                )

    if regressions:
        return False, "Performance regressions detected:\n" + "\n".join(regressions)
    return True, "No performance regressions detected."


def main():
    parser = argparse.ArgumentParser(description="triton-msl performance regression detection")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Save current results as new baseline")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON")
    args = parser.parse_args()

    print("Running triton-msl benchmarks...", file=sys.stderr)
    results = run_all()

    if args.json:
        print(json.dumps(results, indent=2))

    if args.update_baseline:
        save_baseline(results)
        return

    baseline = load_baseline()
    if baseline is None:
        print("No baseline found. Run with --update-baseline first.", file=sys.stderr)
        save_baseline(results)
        print("Created initial baseline.", file=sys.stderr)
        return

    passed, details = check_regression(results, baseline)
    print(details, file=sys.stderr)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
