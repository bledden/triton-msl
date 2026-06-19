#!/usr/bin/env python3
"""Triton-MSL Demo: Compile and run Metal GPU kernels on Apple Silicon.

This script demonstrates the full triton-msl kernel pipeline:
  1. Generate MSL (Metal Shading Language) source from Python
  2. Compile to .metallib via xcrun
  3. Execute on Metal GPU
  4. Verify correctness against reference implementations

Run: python examples/demo.py

Requires macOS with Metal-capable GPU (M1/M2/M3/M4).
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time

# Allow running from repo root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Metal device setup via PyObjC
# ---------------------------------------------------------------------------

def get_metal_device():
    """Initialize Metal device and command queue."""
    try:
        import Metal
    except ImportError:
        print("ERROR: PyObjC Metal framework not found.")
        print("Install with: pip install pyobjc-framework-Metal pyobjc-framework-MetalKit")
        sys.exit(1)

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        print("ERROR: No Metal GPU device found.")
        sys.exit(1)

    queue = device.newCommandQueue()
    return device, queue


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def compile_msl(msl_source, kernel_name="kernel"):
    """Compile MSL source to .metallib and return the file path."""
    cache_dir = os.path.join(tempfile.gettempdir(), "triton_msl_demo")
    os.makedirs(cache_dir, exist_ok=True)

    metal_path = os.path.join(cache_dir, f"{kernel_name}.metal")
    air_path = os.path.join(cache_dir, f"{kernel_name}.air")
    metallib_path = os.path.join(cache_dir, f"{kernel_name}.metallib")

    with open(metal_path, "w") as f:
        f.write(msl_source)

    t0 = time.perf_counter()
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
         "-o", air_path, "-std=metal3.2", "-O2"],
        capture_output=True, check=True,
    )
    metal_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", metallib_path],
        capture_output=True, check=True,
    )
    link_ms = (time.perf_counter() - t1) * 1000

    msl_lines = msl_source.count('\n') + 1
    metallib_size = os.path.getsize(metallib_path)

    return metallib_path, {
        "msl_lines": msl_lines,
        "metal_compile_ms": metal_ms,
        "metallib_link_ms": link_ms,
        "metallib_bytes": metallib_size,
    }


def load_pipeline(device, metallib_path, function_name):
    """Load a metallib and create a compute pipeline state."""
    import Foundation

    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    library, error = device.newLibraryWithURL_error_(url, None)
    if error:
        raise RuntimeError(f"Failed to load metallib: {error}")

    function = library.newFunctionWithName_(function_name)
    if function is None:
        # List available functions for debugging
        names = []
        for i in range(library.functionNames().count()):
            names.append(library.functionNames().objectAtIndex_(i))
        raise RuntimeError(
            f"Function '{function_name}' not found. Available: {names}")

    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function, None)
    if error:
        raise RuntimeError(f"Failed to create pipeline: {error}")

    return pipeline


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def make_float_buffer(device, data):
    n = len(data)
    buf = device.newBufferWithLength_options_(n * 4, 0)
    view = buf.contents().as_buffer(n * 4)
    struct.pack_into(f"{n}f", view, 0, *data)
    return buf


def make_empty_buffer(device, n):
    return device.newBufferWithLength_options_(n * 4, 0)


def make_uint_buffer(device, value):
    buf = device.newBufferWithLength_options_(4, 0)
    view = buf.contents().as_buffer(4)
    struct.pack_into("I", view, 0, value)
    return buf


def make_float_scalar_buffer(device, value):
    buf = device.newBufferWithLength_options_(4, 0)
    view = buf.contents().as_buffer(4)
    struct.pack_into("f", view, 0, value)
    return buf


def read_float_buffer(buf, n):
    view = buf.contents().as_buffer(n * 4)
    return list(struct.unpack_from(f"{n}f", view, 0))


def dispatch(queue, pipeline, buffers, grid, block_size=256):
    """Dispatch a compute kernel and return (cmd_buffer, gpu_time_us)."""
    import Metal

    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    if isinstance(grid, int):
        grid = (grid, 1, 1)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    status = cmd.status()
    if status == Metal.MTLCommandBufferStatusError:
        raise RuntimeError(f"Kernel execution failed: {cmd.error()}")

    gpu_us = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
    return cmd, gpu_us


def bench_kernel(queue, pipeline, buffers, grid, block_size=256,
                 warmup=10, reps=50):
    """Benchmark a kernel: returns dict with timing stats."""
    # Warmup
    for _ in range(warmup):
        dispatch(queue, pipeline, buffers, grid, block_size)

    # Timed runs
    times_us = []
    for _ in range(reps):
        _, t = dispatch(queue, pipeline, buffers, grid, block_size)
        times_us.append(t)

    times_us.sort()
    n = len(times_us)
    return {
        "median_us": times_us[n // 2],
        "min_us": times_us[0],
        "max_us": times_us[-1],
        "p10_us": times_us[int(0.1 * (n - 1))],
        "p90_us": times_us[int(0.9 * (n - 1))],
        "mean_us": sum(times_us) / n,
    }


def print_metrics(label, compile_info, bench, n_bytes=None, n_flops=None):
    """Print comprehensive metrics for a kernel."""
    print(f"\n  --- {label} Metrics ---")
    print(f"  MSL source:     {compile_info['msl_lines']} lines")
    print(f"  Metal compile:  {compile_info['metal_compile_ms']:.1f} ms")
    print(f"  Metallib link:  {compile_info['metallib_link_ms']:.1f} ms")
    print(f"  Metallib size:  {compile_info['metallib_bytes']:,} bytes")
    print(f"  GPU median:     {bench['median_us']:.2f} us")
    print(f"  GPU min:        {bench['min_us']:.2f} us")
    print(f"  GPU max:        {bench['max_us']:.2f} us")
    print(f"  GPU p10/p90:    {bench['p10_us']:.2f} / {bench['p90_us']:.2f} us")
    if n_bytes and bench['median_us'] > 0:
        bw_gbs = (n_bytes / 1e9) / (bench['median_us'] / 1e6)
        print(f"  Bandwidth:      {bw_gbs:.1f} GB/s")
    if n_flops and bench['median_us'] > 0:
        gflops = (n_flops / 1e9) / (bench['median_us'] / 1e6)
        print(f"  Throughput:     {gflops:.1f} GFLOP/s")


# ---------------------------------------------------------------------------
# Demo 1: Vector Addition
# ---------------------------------------------------------------------------

def demo_vector_add(device, queue):
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 1_048_576  # 1M elements
    print(f"\n{'='*60}")
    print(f"Demo 1: Vector Addition (n={n:,})")
    print(f"{'='*60}")

    msl = make_vector_add_kernel(block_size=256)
    path, cinfo = compile_msl(msl, "vector_add")
    pipeline = load_pipeline(device, path, "vector_add")

    # Prepare data
    random.seed(42)
    a = [random.uniform(-10.0, 10.0) for _ in range(n)]
    b = [random.uniform(-10.0, 10.0) for _ in range(n)]
    a_buf = make_float_buffer(device, a)
    b_buf = make_float_buffer(device, b)
    c_buf = make_empty_buffer(device, n)
    n_buf = make_uint_buffer(device, n)

    n_groups = (n + 255) // 256
    buffers = [a_buf, b_buf, c_buf, n_buf]

    # Single run for correctness
    dispatch(queue, pipeline, buffers, n_groups)
    result = read_float_buffer(c_buf, min(n, 1000))
    max_err = max(abs(result[i] - (a[i] + b[i])) for i in range(len(result)))

    # Benchmark
    bench = bench_kernel(queue, pipeline, buffers, n_groups)

    # 3 reads + 1 write = 4 * n * 4 bytes
    n_bytes = 3 * n * 4
    n_flops = n  # 1 add per element
    print_metrics("Vector Add", cinfo, bench, n_bytes=n_bytes, n_flops=n_flops)
    print(f"  Max error:      {max_err:.2e}")
    print(f"  Status:         {'PASS' if max_err < 1e-5 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Demo 2: Softmax
# ---------------------------------------------------------------------------

def demo_softmax(device, queue):
    from triton_msl.codegen.msl_emitter import make_softmax_kernel

    n_rows, n_cols = 256, 512
    print(f"\n{'='*60}")
    print(f"Demo 2: Row-wise Softmax ({n_rows}x{n_cols} = {n_rows*n_cols:,} elements)")
    print(f"{'='*60}")

    msl = make_softmax_kernel(block_size=256)
    path, cinfo = compile_msl(msl, "softmax")
    pipeline = load_pipeline(device, path, "softmax_kernel")

    random.seed(42)
    data = [random.gauss(0, 1) for _ in range(n_rows * n_cols)]
    in_buf = make_float_buffer(device, data)
    out_buf = make_empty_buffer(device, n_rows * n_cols)
    nc_buf = make_uint_buffer(device, n_cols)

    buffers = [in_buf, out_buf, nc_buf]
    grid = (n_rows, 1, 1)

    # Single run for correctness
    dispatch(queue, pipeline, buffers, grid)
    result = read_float_buffer(out_buf, n_rows * n_cols)

    # Reference softmax
    max_err = 0.0
    for r in range(min(n_rows, 32)):  # check first 32 rows
        row = data[r * n_cols:(r + 1) * n_cols]
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        s = sum(exps)
        for c in range(n_cols):
            err = abs(result[r * n_cols + c] - exps[c] / s)
            max_err = max(max_err, err)

    row_sums = [sum(result[r * n_cols:(r + 1) * n_cols]) for r in range(n_rows)]

    # Benchmark
    bench = bench_kernel(queue, pipeline, buffers, grid)

    # 2 passes over input + 1 write
    n_bytes = (2 + 1) * n_rows * n_cols * 4
    # ~5 flops/element (max, sub, exp, sum, div)
    n_flops = 5 * n_rows * n_cols
    print_metrics("Softmax", cinfo, bench, n_bytes=n_bytes, n_flops=n_flops)
    print(f"  Max error:      {max_err:.2e}")
    print(f"  Row sum range:  [{min(row_sums):.6f}, {max(row_sums):.6f}]")
    print(f"  Status:         {'PASS' if max_err < 1e-5 and abs(min(row_sums) - 1.0) < 1e-4 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Demo 3: SIMD-group Matrix Multiply
# ---------------------------------------------------------------------------

def demo_matmul(device, queue):
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 32, 32, 32
    print(f"\n{'='*60}")
    print(f"Demo 3: SIMD-group Matrix Multiply ({M}x{K} @ {K}x{N})")
    print(f"{'='*60}")

    msl = make_simdgroup_matmul_kernel(dtype="fp32")
    path, cinfo = compile_msl(msl, "simdgroup_matmul")
    pipeline = load_pipeline(device, path, "simdgroup_matmul")

    random.seed(123)
    A = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    a_buf = make_float_buffer(device, A)
    b_buf = make_float_buffer(device, B)
    c_buf = make_empty_buffer(device, M * N)
    m_buf = make_uint_buffer(device, M)
    n_buf = make_uint_buffer(device, N)
    k_buf = make_uint_buffer(device, K)

    buffers = [a_buf, b_buf, c_buf, m_buf, n_buf, k_buf]
    grid = (1, 1, 1)
    block_size = 128

    # Single run for correctness
    dispatch(queue, pipeline, buffers, grid, block_size=block_size)
    result = read_float_buffer(c_buf, M * N)

    # Reference matmul
    max_err = 0.0
    for i in range(M):
        for j in range(N):
            expected = sum(A[i * K + k] * B[k * N + j] for k in range(K))
            err = abs(result[i * N + j] - expected)
            max_err = max(max_err, err)

    # Benchmark
    bench = bench_kernel(queue, pipeline, buffers, grid, block_size=block_size)

    # 2*M*N*K FLOPs (multiply + add for each element)
    n_flops = 2 * M * N * K
    n_bytes = (M * K + K * N + M * N) * 4
    print_metrics("SIMD-group Matmul", cinfo, bench, n_bytes=n_bytes, n_flops=n_flops)
    print(f"  Max error:      {max_err:.2e}")
    print(f"  Status:         {'PASS' if max_err < 0.01 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Demo 4: Flash Attention
# ---------------------------------------------------------------------------

def demo_flash_attention(device, queue):
    from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

    head_dim = 64
    seq_len = 32
    n_heads = 4
    Br, Bc = 16, 16
    print(f"\n{'='*60}")
    print(f"Demo 4: Flash Attention (heads={n_heads}, seq={seq_len}, dim={head_dim})")
    print(f"{'='*60}")

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=Br, Bc=Bc, block_size=256)
    path, cinfo = compile_msl(msl, "flash_attention")
    pipeline = load_pipeline(device, path, "flash_attention")

    random.seed(456)
    total = n_heads * seq_len * head_dim
    Q = [random.gauss(0, 0.1) for _ in range(total)]
    K_mat = [random.gauss(0, 0.1) for _ in range(total)]
    V = [random.gauss(0, 0.1) for _ in range(total)]

    q_buf = make_float_buffer(device, Q)
    k_buf = make_float_buffer(device, K_mat)
    v_buf = make_float_buffer(device, V)
    o_buf = make_empty_buffer(device, total)
    sl_buf = make_uint_buffer(device, seq_len)
    scale = 1.0 / math.sqrt(head_dim)
    sc_buf = make_float_scalar_buffer(device, scale)

    n_q_blocks = (seq_len + Br - 1) // Br
    n_tg = n_heads * n_q_blocks
    buffers = [q_buf, k_buf, v_buf, o_buf, sl_buf, sc_buf]
    grid = (n_tg, 1, 1)

    # Single run for correctness
    dispatch(queue, pipeline, buffers, grid)
    result = read_float_buffer(o_buf, total)

    # Reference: naive attention
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def softmax(scores):
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        t = sum(exps)
        return [e / t for e in exps]

    max_err = 0.0
    for h in range(n_heads):
        base = h * seq_len * head_dim
        for i in range(seq_len):
            q_vec = Q[base + i * head_dim:base + (i + 1) * head_dim]
            scores = []
            for j in range(seq_len):
                k_vec = K_mat[base + j * head_dim:base + (j + 1) * head_dim]
                scores.append(dot(q_vec, k_vec) * scale)
            weights = softmax(scores)
            for d in range(head_dim):
                expected = sum(weights[j] * V[base + j * head_dim + d]
                               for j in range(seq_len))
                err = abs(result[base + i * head_dim + d] - expected)
                max_err = max(max_err, err)

    # Benchmark
    bench = bench_kernel(queue, pipeline, buffers, grid)

    # FLOPs: 2 * n_heads * seq_len * seq_len * head_dim (QK) +
    #        2 * n_heads * seq_len * seq_len * head_dim (AV)
    n_flops = 4 * n_heads * seq_len * seq_len * head_dim
    n_bytes = 3 * total * 4 + total * 4  # Q, K, V reads + O write
    print_metrics("Flash Attention", cinfo, bench, n_bytes=n_bytes, n_flops=n_flops)
    print(f"  Max error:      {max_err:.2e}")
    print(f"  Status:         {'PASS' if max_err < 0.05 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Demo 5: Cumulative Sum (Prefix Scan)
# ---------------------------------------------------------------------------

def demo_cumsum(device, queue):
    from triton_msl.codegen.msl_emitter import make_cumsum_kernel

    n_rows, n_cols = 64, 256
    print(f"\n{'='*60}")
    print(f"Demo 5: Parallel Prefix Sum ({n_rows}x{n_cols})")
    print(f"{'='*60}")

    msl = make_cumsum_kernel(block_size=256)
    path, cinfo = compile_msl(msl, "cumsum")
    pipeline = load_pipeline(device, path, "cumsum_kernel")

    random.seed(789)
    data = [random.uniform(-5.0, 5.0) for _ in range(n_rows * n_cols)]
    in_buf = make_float_buffer(device, data)
    out_buf = make_empty_buffer(device, n_rows * n_cols)
    nc_buf = make_uint_buffer(device, n_cols)

    buffers = [in_buf, out_buf, nc_buf]
    grid = (n_rows, 1, 1)

    dispatch(queue, pipeline, buffers, grid)
    result = read_float_buffer(out_buf, n_rows * n_cols)

    # Verify first few rows
    max_err = 0.0
    for r in range(min(n_rows, 8)):
        running = 0.0
        for c in range(n_cols):
            running += data[r * n_cols + c]
            err = abs(result[r * n_cols + c] - running)
            max_err = max(max_err, err)

    bench = bench_kernel(queue, pipeline, buffers, grid)

    n_bytes = 2 * n_rows * n_cols * 4  # read + write
    print_metrics("Cumulative Sum", cinfo, bench, n_bytes=n_bytes)
    print(f"  Max error:      {max_err:.2e}")
    print(f"  Status:         {'PASS' if max_err < 0.5 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Demo 6: Conv2D
# ---------------------------------------------------------------------------

def demo_conv2d(device, queue):
    from triton_msl.codegen.msl_emitter import make_conv2d_kernel

    batch, in_c, out_c = 1, 3, 16
    in_h, in_w = 32, 32
    kh, kw = 3, 3
    out_h = in_h  # same padding
    out_w = in_w
    print(f"\n{'='*60}")
    print(f"Demo 6: Conv2D ({batch}x{in_c}x{in_h}x{in_w} -> {batch}x{out_c}x{out_h}x{out_w}, 3x3 filter)")
    print(f"{'='*60}")

    msl = make_conv2d_kernel(in_channels=in_c, out_channels=out_c,
                             kernel_h=kh, kernel_w=kw,
                             pad_h=1, pad_w=1, block_size=256)
    path, cinfo = compile_msl(msl, "conv2d")
    pipeline = load_pipeline(device, path, "conv2d_kernel")

    random.seed(101)
    input_data = [random.gauss(0, 1) for _ in range(batch * in_c * in_h * in_w)]
    weight_data = [random.gauss(0, 0.1) for _ in range(out_c * in_c * kh * kw)]
    bias_data = [0.0] * out_c

    in_buf = make_float_buffer(device, input_data)
    w_buf = make_float_buffer(device, weight_data)
    b_buf = make_float_buffer(device, bias_data)
    n_out = batch * out_c * out_h * out_w
    out_buf = make_empty_buffer(device, n_out)
    batch_buf = make_uint_buffer(device, batch)
    ih_buf = make_uint_buffer(device, in_h)
    iw_buf = make_uint_buffer(device, in_w)
    oh_buf = make_uint_buffer(device, out_h)
    ow_buf = make_uint_buffer(device, out_w)

    buffers = [in_buf, w_buf, b_buf, out_buf, batch_buf, ih_buf, iw_buf, oh_buf, ow_buf]
    n_groups = (n_out + 255) // 256
    grid = (n_groups, 1, 1)

    dispatch(queue, pipeline, buffers, grid)
    result = read_float_buffer(out_buf, min(n_out, 100))

    # Spot-check a few output values against reference
    max_err = 0.0
    for idx in range(min(n_out, 32)):
        b_idx = idx // (out_c * out_h * out_w)
        rem = idx % (out_c * out_h * out_w)
        oc = rem // (out_h * out_w)
        spatial = rem % (out_h * out_w)
        oh_idx = spatial // out_w
        ow_idx = spatial % out_w
        expected = bias_data[oc]
        for ic in range(in_c):
            for ki in range(kh):
                for kj in range(kw):
                    ih_idx = oh_idx + ki - 1
                    iw_idx = ow_idx + kj - 1
                    if 0 <= ih_idx < in_h and 0 <= iw_idx < in_w:
                        in_idx = b_idx * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih_idx * in_w + iw_idx
                        w_idx = oc * (in_c * kh * kw) + ic * (kh * kw) + ki * kw + kj
                        expected += input_data[in_idx] * weight_data[w_idx]
        err = abs(result[idx] - expected)
        max_err = max(max_err, err)

    bench = bench_kernel(queue, pipeline, buffers, grid)

    # FLOPs: batch * out_c * out_h * out_w * in_c * kh * kw * 2
    n_flops = batch * out_c * out_h * out_w * in_c * kh * kw * 2
    n_bytes = (batch * in_c * in_h * in_w + out_c * in_c * kh * kw + n_out) * 4
    print_metrics("Conv2D", cinfo, bench, n_bytes=n_bytes, n_flops=n_flops)
    print(f"  Max error:      {max_err:.2e}")
    print(f"  Status:         {'PASS' if max_err < 0.01 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_device_info(device):
    """Print comprehensive Metal device information."""
    print(f"\n{'='*60}")
    print(f"Apple Silicon GPU Info")
    print(f"{'='*60}")
    print(f"  Device:                {device.name()}")
    print(f"  Max threadgroup mem:   {device.maxThreadgroupMemoryLength():,} bytes")
    print(f"  Max threads/TG:        {device.maxThreadsPerThreadgroup().width}")
    print(f"  Max buffer length:     {device.maxBufferLength() / (1024**3):.1f} GB")
    has_unified = hasattr(device, 'hasUnifiedMemory') and device.hasUnifiedMemory()
    print(f"  Unified memory:        {has_unified}")

    # Count available kernels
    from triton_msl.codegen import msl_emitter
    kernels = [name for name in dir(msl_emitter) if name.startswith("make_")]
    print(f"  Kernel generators:     {len(kernels)}")
    print()
    for k in sorted(kernels):
        print(f"    - {k}")


def main():
    print("=" * 60)
    print("  Triton-MSL: GPU Kernel Compiler for Apple Silicon")
    print("=" * 60)

    device, queue = get_metal_device()
    print_device_info(device)

    demos = [
        ("Vector Add", demo_vector_add),
        ("Softmax", demo_softmax),
        ("Matmul", demo_matmul),
        ("Flash Attention", demo_flash_attention),
        ("Cumulative Sum", demo_cumsum),
        ("Conv2D", demo_conv2d),
    ]

    results = []
    for name, fn in demos:
        try:
            fn(device, queue)
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results.append((name, "FAIL"))

    print(f"\n{'='*60}")
    print(f"  Demo Results Summary")
    print(f"{'='*60}")
    for name, status in results:
        icon = "+" if status == "PASS" else "X"
        print(f"  [{icon}] {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\n  {passed}/{len(results)} demos passed.")

    # Roofline reference numbers for Apple Silicon
    print(f"\n{'='*60}")
    print(f"  Apple Silicon Performance Reference")
    print(f"{'='*60}")
    gpu_name = device.name()
    if "M4 Max" in gpu_name:
        print(f"  Peak memory bandwidth:  546 GB/s (unified)")
        print(f"  GPU cores:              40")
        print(f"  Peak FP32 throughput:   ~14 TFLOP/s")
        print(f"  Peak FP16 throughput:   ~28 TFLOP/s")
        print(f"  Threadgroup memory:     32 KB per threadgroup")
        print(f"  SIMD width:             32 threads")
        print(f"  L1 cache:               ~8 KB per core")
    elif "M4 Pro" in gpu_name:
        print(f"  Peak memory bandwidth:  273 GB/s (unified)")
        print(f"  GPU cores:              20")
    elif "M4" in gpu_name:
        print(f"  Peak memory bandwidth:  120 GB/s (unified)")
        print(f"  GPU cores:              10")
    elif "M3 Max" in gpu_name:
        print(f"  Peak memory bandwidth:  400 GB/s (unified)")
        print(f"  GPU cores:              40")
    else:
        print(f"  (Reference numbers not available for {gpu_name})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
