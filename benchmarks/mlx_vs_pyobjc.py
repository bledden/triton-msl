"""Benchmark: MLX metal_kernel path vs PyObjC path vs native MLX.

Run: python benchmarks/mlx_vs_pyobjc.py

Compares dispatch latency and throughput for:
1. triton_call (MLX metal_kernel) — zero-copy MLX dispatch
2. Native MLX operations — baseline
3. Raw MLX metal_kernel — isolates kernel vs wrapper overhead
"""

import time
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    raise SystemExit("MLX not installed. Run: pip install mlx")

import triton
import triton.language as tl
import triton_msl.mlx as tmlx


# ─── Triton Kernels ──────────────────────────────────────────────────────────

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)


# ─── Benchmark Utilities ─────────────────────────────────────────────────────

def sync_mlx():
    """Force MLX to synchronize by evaluating a zero-cost op."""
    mx.synchronize()


def bench(fn, warmup=10, iters=100):
    """Return (median_ms, mean_ms)."""
    for _ in range(warmup):
        fn()
    sync_mlx()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync_mlx()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), np.mean(times)


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Dispatch Latency ──────────────────────────────────────────────────
    print_header("Dispatch Latency (N=256, measures overhead only)")

    N = 256
    a = mx.array(np.ones(N, dtype=np.float32))
    b = mx.array(np.ones(N, dtype=np.float32))
    out = mx.zeros((N,))

    tmlx.triton_call(add_kernel, a, b, out, N, grid=(1,), BLOCK=256)

    t_triton, _ = bench(
        lambda: tmlx.triton_call(add_kernel, a, b, out, N, grid=(1,), BLOCK=256),
        warmup=20, iters=200,
    )
    t_native, _ = bench(lambda: mx.add(a, b), warmup=20, iters=200)

    print(f"  triton_call:  {t_triton:.3f} ms")
    print(f"  native MLX:   {t_native:.3f} ms")
    print(f"  overhead:     {t_triton - t_native:.3f} ms")

    # ── Vector Add Throughput ─────────────────────────────────────────────
    print_header("Vector Add Throughput")
    print(f"  {'N':>10}  {'Triton':>12}  {'Native':>12}  {'Ratio':>8}  {'BW Triton':>12}")

    for N in [1024, 16384, 262144, 1048576, 16777216]:
        x = mx.array(np.random.randn(N).astype(np.float32))
        y = mx.array(np.random.randn(N).astype(np.float32))
        out = mx.zeros((N,))
        BLOCK = 256
        grid = ((N + BLOCK - 1) // BLOCK,)

        tmlx.triton_call(add_kernel, x, y, out, N, grid=grid, BLOCK=BLOCK)

        t_t, _ = bench(lambda: tmlx.triton_call(add_kernel, x, y, out, N, grid=grid, BLOCK=BLOCK))
        t_n, _ = bench(lambda: mx.add(x, y))

        bw = N * 4 * 3 / (t_t / 1000) / 1e9
        print(f"  {N:>10}  {t_t:>9.3f} ms  {t_n:>9.3f} ms  {t_t/t_n:>6.2f}x  {bw:>9.1f} GB/s")

    # ── Softmax Throughput ────────────────────────────────────────────────
    print_header("Softmax Throughput")
    print(f"  {'Shape':>12}  {'Triton':>12}  {'Native':>12}  {'Ratio':>8}")

    for rows, cols in [(64, 128), (256, 256), (1024, 1024), (4096, 1024)]:
        x = mx.array(np.random.randn(rows * cols).astype(np.float32))
        out = mx.zeros((rows * cols,))
        BLOCK = min(cols, 1024)

        tmlx.triton_call(softmax_kernel, x, out, cols, grid=(rows,), BLOCK=BLOCK)

        t_t, _ = bench(lambda: tmlx.triton_call(softmax_kernel, x, out, cols, grid=(rows,), BLOCK=BLOCK))
        x_2d = x.reshape(rows, cols)
        t_n, _ = bench(lambda: mx.softmax(x_2d, axis=1))

        print(f"  {rows}x{cols:>5}  {t_t:>9.3f} ms  {t_n:>9.3f} ms  {t_t/t_n:>6.2f}x")

    print()
    print("Notes:")
    print("  - Triton = triton_call() via mx.fast.metal_kernel (zero-copy)")
    print("  - Native = mx.add / mx.softmax (MLX built-in ops)")
    print("  - Ratio = Triton time / Native time (lower is better)")
    print("  - BW = effective memory bandwidth (reads + writes)")
    print("  - All times are median of 100 iterations after warmup")
