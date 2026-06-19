"""Benchmark all kernel types on Metal GPU with GPU-precise timing.

Reports throughput in GB/s and GFLOP/s where applicable.
Uses MTLCommandBuffer.GPUStartTime/GPUEndTime for timing.

Usage: python benchmarks/bench_all.py
"""

import hashlib
import os
import struct
import subprocess
import tempfile

# GPU setup
import Metal

from triton_msl.profiling.metal_bench import (
    MetalBenchmark,
    compute_gflops,
    compute_throughput,
    format_benchmark_result,
)
from triton_msl.codegen.msl_emitter import (
    make_vector_add_kernel,
    make_elementwise_kernel,
    make_silu_kernel,
    make_gelu_kernel,
    make_swiglu_kernel,
    make_reduce_kernel,
    make_softmax_kernel,
    make_matmul_kernel,
    make_simdgroup_matmul_kernel,
    make_rms_norm_kernel,
    make_rope_kernel,
    make_layer_norm_kernel,
    make_cross_entropy_kernel,
    make_flash_attention_kernel,
    make_fused_linear_kernel,
    make_top_k_kernel,
    make_kv_cache_attention_kernel,
    make_residual_add_kernel,
    make_embedding_kernel,
    make_scalar_mul_kernel,
    make_rope_attention_kernel,
    make_gqa_attention_kernel,
    make_batched_kv_decode_kernel,
    make_int8_matmul_kernel,
    make_int4_matmul_kernel,
    make_concat_kernel,
    make_split_kernel,
    make_top_p_kernel,
    make_paged_attention_kernel,
    make_multi_head_paged_attention_kernel,
    make_fp16_kv_attention_kernel,
    make_fused_residual_norm_kernel,
    make_speculative_decode_kernel,
    make_beam_search_kernel,
    make_fused_mlp_kernel,
    make_sliding_window_attention_kernel,
    make_repeat_kv_kernel,
    make_matmul_2d_kernel,
    make_matmul_swizzled_kernel,
    make_activation_kernel,
    make_variance_kernel,
    make_batch_norm_kernel,
    make_online_softmax_kernel,
    make_causal_attention_kernel,
    make_group_norm_kernel,
    make_instance_norm_kernel,
    make_fused_dropout_kernel,
    make_gather_kernel,
    make_scatter_kernel,
    make_transpose_kernel,
    make_reduce_scatter_kernel,
    make_all_reduce_kernel,
)


def compile_and_load(device, msl_src, kernel_name):
    """Compile MSL and load pipeline state."""
    import Foundation

    cache_dir = os.path.join(tempfile.gettempdir(), "triton_msl_bench_cache")
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
    base = f"{kernel_name}_{src_hash}"
    metal_path = os.path.join(cache_dir, f"{base}.metal")
    air_path = os.path.join(cache_dir, f"{base}.air")
    metallib_path = os.path.join(cache_dir, f"{base}.metallib")

    if not os.path.exists(metallib_path):
        with open(metal_path, "w") as f:
            f.write(msl_src)
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
             "-o", air_path, "-std=metal3.2", "-O2"],
            stderr=subprocess.PIPE,
        )
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metallib", air_path,
             "-o", metallib_path],
            stderr=subprocess.PIPE,
        )

    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    library, error = device.newLibraryWithURL_error_(url, None)
    assert error is None, f"Load failed: {error}"

    function = library.newFunctionWithName_(kernel_name)
    assert function is not None, f"Kernel '{kernel_name}' not found"

    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    assert error is None, f"Pipeline failed: {error}"
    return pipeline


def make_float_buffer(device, n, pattern="ramp"):
    """Create a float buffer with a fill pattern. Much faster than Python lists."""
    buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(n * 4)
    if pattern == "ramp":
        for i in range(n):
            struct.pack_into("f", view, i * 4, float(i % 1000) * 0.01)
    elif pattern == "ones":
        for i in range(n):
            struct.pack_into("f", view, i * 4, 1.0)
    elif pattern == "small":
        for i in range(n):
            struct.pack_into("f", view, i * 4, float(i % 10) * 0.1)
    return buf


def make_empty_buffer(device, n):
    return device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )


def make_uint_buffer(device, value):
    buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(4)
    struct.pack_into("I", view, 0, value)
    return buf


def make_float_scalar_buffer(device, value):
    buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(4)
    struct.pack_into("f", view, 0, value)
    return buf


def make_half_buffer(device, n, pattern="ramp"):
    """Create a float16 (half) buffer. 2 bytes per element."""
    buf = device.newBufferWithLength_options_(
        n * 2, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(n * 2)
    if pattern == "ramp":
        for i in range(n):
            struct.pack_into("e", view, i * 2, float(i % 100) * 0.01)
    elif pattern == "ones":
        for i in range(n):
            struct.pack_into("e", view, i * 2, 1.0)
    elif pattern == "small":
        for i in range(n):
            struct.pack_into("e", view, i * 2, float(i % 10) * 0.1)
    return buf


def bench_custom_dispatch(bench, pipeline, buffers, n_groups, block_size,
                          n_warmup=10, n_iters=100, dispatch_2d=None):
    """Time a kernel with custom threadgroup dispatch.

    Args:
        dispatch_2d: Optional (x, y) tuple for 2D threadgroup dispatch.
            If provided, n_groups is ignored and dispatch uses 2D grid.
    """
    import Metal as M

    if dispatch_2d:
        grid = M.MTLSizeMake(dispatch_2d[0], dispatch_2d[1], 1)
    else:
        grid = M.MTLSizeMake(n_groups, 1, 1)
    tg_size = M.MTLSizeMake(block_size, 1, 1)

    for _ in range(n_warmup):
        cmd = bench.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg_size)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    gpu_times_us = []
    for _ in range(n_iters):
        cmd = bench.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg_size)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        gpu_times_us.append(
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
        )

    gpu_times_us.sort()
    nn = len(gpu_times_us)
    return {
        "median_us": gpu_times_us[nn // 2],
        "min_us": gpu_times_us[0],
        "max_us": gpu_times_us[-1],
        "p10_us": gpu_times_us[int(0.1 * (nn - 1))],
        "p90_us": gpu_times_us[int(0.9 * (nn - 1))],
        "all_us": gpu_times_us,
    }


def main():
    device = Metal.MTLCreateSystemDefaultDevice()
    print(f"Device: {device.name()}")
    print(f"Max threadgroup memory: {device.maxThreadgroupMemoryLength()} bytes")
    print()

    bench = MetalBenchmark()

    # =========================================================================
    # Elementwise benchmarks
    # =========================================================================
    print("=" * 60)
    print("ELEMENTWISE BENCHMARKS")
    print("=" * 60)

    for n in [1024, 65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        # Vector add: reads 2 inputs + writes 1 output = 3 * n * 4 bytes
        msl = make_vector_add_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "vector_add")

        a_buf = make_float_buffer(device, n, "ramp")
        b_buf = make_float_buffer(device, n, "ones")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [a_buf, b_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4
        print(format_benchmark_result("vector_add", result, n_bytes=n_bytes, n_flops=n))

        # SiLU
        msl = make_silu_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "silu_kernel")

        in_buf = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [in_buf, out_buf, n_buf], n)
        n_bytes = 2 * n * 4  # 1 read + 1 write
        print(format_benchmark_result("silu", result, n_bytes=n_bytes, n_flops=4 * n))

        # GELU
        msl = make_gelu_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "gelu_kernel")

        result = bench.time_kernel(pipeline, [in_buf, out_buf, n_buf], n)
        print(format_benchmark_result("gelu", result, n_bytes=n_bytes, n_flops=8 * n))

    # =========================================================================
    # Reduction benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("REDUCTION BENCHMARKS")
    print("=" * 60)

    for n in [256, 1024, 65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_reduce_kernel("reduce_sum", "sum", block_size=256)
        pipeline = compile_and_load(device, msl, "reduce_sum")

        in_buf = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, 1)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [in_buf, out_buf, n_buf], n)
        n_bytes = n * 4  # read only
        print(format_benchmark_result("reduce_sum", result, n_bytes=n_bytes, n_flops=n))

    # =========================================================================
    # Softmax benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SOFTMAX BENCHMARKS")
    print("=" * 60)

    for n_cols in [64, 256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_softmax_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "softmax_kernel")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = total * 4 * 5
        n_flops = total * 5
        print(format_benchmark_result("softmax", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Layer Norm benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("LAYER NORM BENCHMARKS")
    print("=" * 60)

    for n_cols in [256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_layer_norm_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "layer_norm_kernel")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        gamma_buf = make_float_buffer(device, n_cols, "ones")
        beta_buf = make_float_buffer(device, n_cols, "small")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, gamma_buf, beta_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = (total * 3 + n_cols * 2) * 4
        n_flops = total * 6
        print(format_benchmark_result("layer_norm", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("MATMUL BENCHMARKS")
    print("=" * 60)

    for size in [32, 64, 128, 256]:
        M_dim, N_dim, K_dim = size, size, size
        block_m, block_n, block_k = 32, 32, 32
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} ---")

        msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
        pipeline = compile_and_load(device, msl, "matmul_kernel")

        A_buf = make_float_buffer(device, M_dim * K_dim, "small")
        B_buf = make_float_buffer(device, K_dim * N_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + block_n - 1) // block_n
        n_tile_rows = (M_dim + block_m - 1) // block_m
        n_groups = n_tile_rows * n_tile_cols
        threads_per_tg = block_m * block_n

        result = bench_custom_dispatch(bench, pipeline,
                                        [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
                                        n_groups, threads_per_tg)
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim + K_dim * N_dim + M_dim * N_dim) * 4
        print(format_benchmark_result("matmul", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # simdgroup_matrix matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SIMDGROUP MATRIX MATMUL BENCHMARKS (hardware 8x8 MMA)")
    print("=" * 60)

    for size in [32, 64, 128, 256, 512]:
        M_dim, N_dim, K_dim = size, size, size
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} ---")

        msl = make_simdgroup_matmul_kernel()
        pipeline = compile_and_load(device, msl, "simdgroup_matmul")

        A_buf = make_float_buffer(device, M_dim * K_dim, "small")
        B_buf = make_float_buffer(device, K_dim * N_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + 31) // 32
        n_tile_rows = (M_dim + 31) // 32
        n_groups = n_tile_rows * n_tile_cols

        result = bench_custom_dispatch(bench, pipeline,
                                        [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
                                        n_groups, 128)
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim + K_dim * N_dim + M_dim * N_dim) * 4
        print(format_benchmark_result("simdgroup_matmul", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # simdgroup_matrix FP16 matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SIMDGROUP MATRIX FP16 MATMUL BENCHMARKS (half inputs, float acc)")
    print("=" * 60)

    for size in [64, 128, 256, 512, 1024]:
        M_dim, N_dim, K_dim = size, size, size
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} (FP16) ---")

        msl = make_simdgroup_matmul_kernel(dtype="fp16")
        pipeline = compile_and_load(device, msl, "simdgroup_matmul")

        A_buf = make_half_buffer(device, M_dim * K_dim, "small")
        B_buf = make_half_buffer(device, K_dim * N_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)  # output is float32
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + 31) // 32
        n_tile_rows = (M_dim + 31) // 32
        n_groups = n_tile_rows * n_tile_cols

        result = bench_custom_dispatch(bench, pipeline,
                                        [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
                                        n_groups, 128)
        n_flops = 2 * M_dim * N_dim * K_dim
        # A/B are half (2 bytes), C is float (4 bytes)
        n_bytes = (M_dim * K_dim + K_dim * N_dim) * 2 + M_dim * N_dim * 4
        print(format_benchmark_result("simdgroup_fp16", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Fused Linear (matmul + bias) benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUSED LINEAR BENCHMARKS (simdgroup_matrix + bias)")
    print("=" * 60)

    for size in [32, 64, 128, 256]:
        M_dim, N_dim, K_dim = size, size, size
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} + bias ---")

        msl = make_fused_linear_kernel(has_bias=True)
        pipeline = compile_and_load(device, msl, "fused_linear")

        in_buf = make_float_buffer(device, M_dim * K_dim, "small")
        wt_buf = make_float_buffer(device, N_dim * K_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)
        bias_buf = make_float_buffer(device, N_dim, "small")
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + 31) // 32
        n_tile_rows = (M_dim + 31) // 32
        n_groups = n_tile_rows * n_tile_cols

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, wt_buf, C_buf, bias_buf,
                                         M_buf, N_buf, K_buf],
                                        n_groups, 128)
        n_flops = 2 * M_dim * N_dim * K_dim + M_dim * N_dim
        n_bytes = (M_dim * K_dim + N_dim * K_dim + M_dim * N_dim + N_dim) * 4
        print(format_benchmark_result("fused_linear", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # RMS Norm benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("RMS NORM BENCHMARKS")
    print("=" * 60)

    for n_cols in [64, 256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_rms_norm_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "rms_norm_kernel")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        wt_buf = make_float_buffer(device, n_cols, "ones")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, wt_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = total * 4 * 3 + n_cols * 4
        n_flops = total * 4
        print(format_benchmark_result("rms_norm", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # RoPE benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROPE BENCHMARKS")
    print("=" * 60)

    for dim in [64, 128, 256]:
        seq_len = 512
        print(f"\n--- seq_len={seq_len}, dim={dim} ---")

        msl = make_rope_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "rope_kernel")

        total = seq_len * dim
        in_buf = make_float_buffer(device, total, "ramp")
        freq_buf = make_float_buffer(device, dim // 2, "small")
        out_buf = make_empty_buffer(device, total)
        dim_buf = make_uint_buffer(device, dim)
        pos_buf = make_uint_buffer(device, 0)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, freq_buf, out_buf, dim_buf, pos_buf],
                                        seq_len, 256)
        n_bytes = (total + dim // 2 + total) * 4
        n_flops = (dim // 2) * seq_len * 8
        print(format_benchmark_result("rope", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Cross-Entropy benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("CROSS-ENTROPY LOSS BENCHMARKS")
    print("=" * 60)

    for n_classes in [256, 1024, 32768]:
        n_rows = 64
        print(f"\n--- {n_rows} rows x {n_classes} classes ---")

        msl = make_cross_entropy_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "cross_entropy_kernel")

        total = n_rows * n_classes
        in_buf = make_float_buffer(device, total, "ramp")
        # targets buffer (uint)
        tgt_buf = device.newBufferWithLength_options_(
            n_rows * 4, Metal.MTLResourceStorageModeShared
        )
        tgt_view = tgt_buf.contents().as_buffer(n_rows * 4)
        for i in range(n_rows):
            struct.pack_into("I", tgt_view, i * 4, i % n_classes)
        out_buf = make_empty_buffer(device, n_rows)
        ncls_buf = make_uint_buffer(device, n_classes)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, tgt_buf, out_buf, ncls_buf],
                                        n_rows, 256)
        n_bytes = (total + n_rows + n_rows) * 4
        n_flops = total * 3
        print(format_benchmark_result("cross_entropy", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # SwiGLU benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SWIGLU ACTIVATION BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_swiglu_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "swiglu_kernel")

        x_buf = make_float_buffer(device, n, "ramp")
        gate_buf = make_float_buffer(device, n, "small")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [x_buf, gate_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4
        print(format_benchmark_result("swiglu", result, n_bytes=n_bytes, n_flops=5 * n))

    # =========================================================================
    # Residual Add benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESIDUAL ADD BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_residual_add_kernel(has_bias=False)
        pipeline = compile_and_load(device, msl, "residual_add_kernel")

        x_buf = make_float_buffer(device, n, "ramp")
        r_buf = make_float_buffer(device, n, "small")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [x_buf, r_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4
        print(format_benchmark_result("residual_add", result, n_bytes=n_bytes, n_flops=n))

    # =========================================================================
    # KV-Cache Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("KV-CACHE ATTENTION BENCHMARKS (single query)")
    print("=" * 60)

    import math
    for seq_len in [64, 256, 1024]:
        head_dim = 64
        print(f"\n--- seq_len={seq_len}, head_dim={head_dim} ---")

        msl = make_kv_cache_attention_kernel(head_dim=head_dim)
        pipeline = compile_and_load(device, msl, "kv_cache_attention")

        q_buf = make_float_buffer(device, head_dim, "small")
        k_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        v_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        out_buf = make_empty_buffer(device, head_dim)
        sl_buf = make_uint_buffer(device, seq_len)
        scale_buf = make_float_scalar_buffer(device, 1.0 / math.sqrt(head_dim))

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf,
                                         sl_buf, scale_buf],
                                        1, 256)
        # Q@K^T: seq_len*head_dim*2 flops, P@V: seq_len*head_dim*2 flops
        n_flops = 4 * seq_len * head_dim
        n_bytes = (head_dim + 2 * seq_len * head_dim + head_dim) * 4
        print(format_benchmark_result("kv_cache_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Top-K benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP-K SAMPLING BENCHMARKS")
    print("=" * 60)

    for vocab_size in [1024, 32768, 128000]:
        k = 50
        print(f"\n--- vocab={vocab_size:,}, k={k} ---")

        msl = make_top_k_kernel(k=k, block_size=256)
        pipeline = compile_and_load(device, msl, "top_k")

        logits_buf = make_float_buffer(device, vocab_size, "ramp")
        val_buf = make_empty_buffer(device, k)
        idx_buf = device.newBufferWithLength_options_(
            k * 4, Metal.MTLResourceStorageModeShared
        )
        vsz_buf = make_uint_buffer(device, vocab_size)
        k_buf = make_uint_buffer(device, k)

        result = bench_custom_dispatch(bench, pipeline,
                                        [logits_buf, val_buf, idx_buf,
                                         vsz_buf, k_buf],
                                        1, 256)
        n_bytes = vocab_size * 4 + k * 8
        print(format_benchmark_result("top_k", result, n_bytes=n_bytes))

    # =========================================================================
    # Flash Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FLASH ATTENTION BENCHMARKS")
    print("=" * 60)

    for seq_len in [64, 128, 256]:
        head_dim = 64
        Br, Bc = 16, 16
        print(f"\n--- seq_len={seq_len}, head_dim={head_dim} ---")

        msl = make_flash_attention_kernel(head_dim=head_dim, Br=Br, Bc=Bc)
        pipeline = compile_and_load(device, msl, "flash_attention")

        q_buf = make_float_buffer(device, seq_len * head_dim, "small")
        k_buf = make_float_buffer(device, seq_len * head_dim, "small")
        v_buf = make_float_buffer(device, seq_len * head_dim, "small")
        out_buf = make_empty_buffer(device, seq_len * head_dim)
        sl_buf = make_uint_buffer(device, seq_len)

        n_groups = (seq_len + Br - 1) // Br
        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf, sl_buf],
                                        n_groups, 256)
        n_flops = 4 * seq_len * seq_len * head_dim
        n_bytes = 4 * seq_len * head_dim * 4
        print(format_benchmark_result("flash_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Embedding Lookup benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("EMBEDDING LOOKUP BENCHMARKS")
    print("=" * 60)

    for vocab_size, embed_dim, batch_size in [(32000, 4096, 32), (128000, 4096, 64)]:
        print(f"\n--- vocab={vocab_size:,}, dim={embed_dim}, batch={batch_size} ---")

        msl = make_embedding_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "embedding_kernel")

        table_buf = make_float_buffer(device, vocab_size * embed_dim, "small")
        # indices: int32 array
        idx_buf = device.newBufferWithLength_options_(
            batch_size * 4, Metal.MTLResourceStorageModeShared
        )
        idx_view = idx_buf.contents().as_buffer(batch_size * 4)
        for i in range(batch_size):
            struct.pack_into("i", idx_view, i * 4, i % vocab_size)
        out_buf = make_empty_buffer(device, batch_size * embed_dim)
        dim_buf = make_uint_buffer(device, embed_dim)

        result = bench_custom_dispatch(bench, pipeline,
                                        [table_buf, idx_buf, out_buf, dim_buf],
                                        batch_size, 256)
        n_bytes = batch_size * embed_dim * 4 * 2  # read table row + write output
        print(format_benchmark_result("embedding", result, n_bytes=n_bytes))

    # =========================================================================
    # Scalar Multiply benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCALAR MULTIPLY BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_scalar_mul_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "scalar_mul")

        in_buf = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, n)
        sc_buf = make_float_scalar_buffer(device, 2.5)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [in_buf, out_buf, sc_buf, n_buf], n)
        n_bytes = 2 * n * 4
        print(format_benchmark_result("scalar_mul", result, n_bytes=n_bytes, n_flops=n))

    # =========================================================================
    # Fused RoPE + Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUSED ROPE + ATTENTION BENCHMARKS")
    print("=" * 60)

    for seq_len in [64, 256, 1024]:
        head_dim = 64
        print(f"\n--- seq_len={seq_len}, head_dim={head_dim} ---")

        msl = make_rope_attention_kernel(head_dim=head_dim, block_size=256)
        pipeline = compile_and_load(device, msl, "rope_attention")

        q_buf = make_float_buffer(device, head_dim, "small")
        k_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        v_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        freq_buf = make_float_buffer(device, head_dim // 2, "small")
        out_buf = make_empty_buffer(device, head_dim)
        sl_buf = make_uint_buffer(device, seq_len)
        qpos_buf = make_uint_buffer(device, seq_len - 1)

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, freq_buf, out_buf,
                                         sl_buf, qpos_buf],
                                        1, 256)
        n_flops = seq_len * head_dim * 8  # RoPE + dot + softmax + V mul
        n_bytes = (head_dim + 2 * seq_len * head_dim + head_dim // 2 + head_dim) * 4
        print(format_benchmark_result("rope_attn", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # GQA Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("GROUPED QUERY ATTENTION BENCHMARKS")
    print("=" * 60)

    for n_q_heads, n_q_per_kv, seq_len in [(32, 4, 128), (32, 4, 512)]:
        head_dim = 64
        n_kv_heads = n_q_heads // n_q_per_kv
        print(f"\n--- q_heads={n_q_heads}, kv_heads={n_kv_heads}, seq={seq_len} ---")

        msl = make_gqa_attention_kernel(head_dim=head_dim, n_q_per_kv=n_q_per_kv)
        pipeline = compile_and_load(device, msl, "gqa_attention")

        q_buf = make_float_buffer(device, n_q_heads * head_dim, "small")
        k_buf = make_float_buffer(device, n_kv_heads * seq_len * head_dim, "ramp")
        v_buf = make_float_buffer(device, n_kv_heads * seq_len * head_dim, "ramp")
        out_buf = make_empty_buffer(device, n_q_heads * head_dim)
        sl_buf = make_uint_buffer(device, seq_len)
        scale_buf = make_float_scalar_buffer(device, 1.0 / (head_dim ** 0.5))

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf, sl_buf, scale_buf],
                                        n_q_heads, 256)
        n_flops = n_q_heads * 4 * seq_len * head_dim
        n_bytes = (n_q_heads * head_dim + 2 * n_kv_heads * seq_len * head_dim +
                   n_q_heads * head_dim) * 4
        print(format_benchmark_result("gqa_attn", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Batched KV Decode benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("BATCHED KV DECODE BENCHMARKS")
    print("=" * 60)

    for batch_size, n_heads, max_seq_len in [(4, 8, 256), (8, 8, 512)]:
        head_dim = 64
        print(f"\n--- batch={batch_size}, heads={n_heads}, max_seq={max_seq_len} ---")

        msl = make_batched_kv_decode_kernel(n_heads=n_heads, head_dim=head_dim)
        pipeline = compile_and_load(device, msl, "batched_kv_decode")

        q_buf = make_float_buffer(device, batch_size * n_heads * head_dim, "small")
        k_buf = make_float_buffer(device, batch_size * n_heads * max_seq_len * head_dim, "ramp")
        v_buf = make_float_buffer(device, batch_size * n_heads * max_seq_len * head_dim, "ramp")
        out_buf = make_empty_buffer(device, batch_size * n_heads * head_dim)
        # seq_lens: all set to max_seq_len
        sl_buf = device.newBufferWithLength_options_(
            batch_size * 4, Metal.MTLResourceStorageModeShared
        )
        sl_view = sl_buf.contents().as_buffer(batch_size * 4)
        for i in range(batch_size):
            struct.pack_into("I", sl_view, i * 4, max_seq_len)
        max_sl_buf = make_uint_buffer(device, max_seq_len)
        bs_buf = make_uint_buffer(device, batch_size)

        n_groups = batch_size * n_heads
        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf, sl_buf,
                                         max_sl_buf, bs_buf],
                                        n_groups, 256)
        n_flops = batch_size * n_heads * 4 * max_seq_len * head_dim
        n_bytes = (batch_size * n_heads * (head_dim + 2 * max_seq_len * head_dim + head_dim)) * 4
        print(format_benchmark_result("batched_kv", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # INT8 Quantized Matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("INT8 QUANTIZED MATMUL BENCHMARKS")
    print("=" * 60)

    for M_dim, N_dim, K_dim in [(32, 4096, 4096), (64, 4096, 4096)]:
        print(f"\n--- M={M_dim}, N={N_dim}, K={K_dim} ---")

        msl = make_int8_matmul_kernel()
        pipeline = compile_and_load(device, msl, "int8_matmul")

        in_buf = make_float_buffer(device, M_dim * K_dim, "small")
        # INT8 weights: one byte per element
        w_buf = device.newBufferWithLength_options_(
            N_dim * K_dim, Metal.MTLResourceStorageModeShared
        )
        w_view = w_buf.contents().as_buffer(N_dim * K_dim)
        for i in range(min(N_dim * K_dim, 65536)):  # fill first portion
            struct.pack_into("b", w_view, i, (i % 127) - 63)
        out_buf = make_empty_buffer(device, M_dim * N_dim)
        sc_buf = make_float_buffer(device, N_dim, "ones")
        zp_buf = make_float_buffer(device, N_dim, "small")
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        total = M_dim * N_dim
        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, w_buf, out_buf, sc_buf, zp_buf,
                                         M_buf, N_buf, K_buf],
                                        (total + 255) // 256, 256)
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim * 4 + N_dim * K_dim + M_dim * N_dim * 4 + N_dim * 8)
        print(format_benchmark_result("int8_matmul", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # INT4 Quantized Matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("INT4 QUANTIZED MATMUL BENCHMARKS")
    print("=" * 60)

    for M_dim, N_dim, K_dim in [(32, 4096, 4096), (64, 4096, 4096)]:
        group_size = 128
        print(f"\n--- M={M_dim}, N={N_dim}, K={K_dim}, gs={group_size} ---")

        msl = make_int4_matmul_kernel(group_size=group_size)
        pipeline = compile_and_load(device, msl, "int4_matmul")

        in_buf = make_float_buffer(device, M_dim * K_dim, "small")
        # INT4 packed: 2 values per byte
        packed_size = N_dim * (K_dim // 2)
        w_buf = device.newBufferWithLength_options_(
            packed_size, Metal.MTLResourceStorageModeShared
        )
        w_view = w_buf.contents().as_buffer(packed_size)
        for i in range(min(packed_size, 65536)):
            struct.pack_into("B", w_view, i, (i % 255))
        out_buf = make_empty_buffer(device, M_dim * N_dim)
        n_groups_k = (K_dim + group_size - 1) // group_size
        sc_buf = make_float_buffer(device, N_dim * n_groups_k, "ones")
        zp_buf = make_float_buffer(device, N_dim * n_groups_k, "small")
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        total = M_dim * N_dim
        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, w_buf, out_buf, sc_buf, zp_buf,
                                         M_buf, N_buf, K_buf],
                                        (total + 255) // 256, 256)
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim * 4 + packed_size + M_dim * N_dim * 4 +
                   N_dim * n_groups_k * 8)
        print(format_benchmark_result("int4_matmul", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Concat / Split benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONCAT / SPLIT BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000]:
        print(f"\n--- n = {n:,} per tensor ---")

        # Concat 2 tensors
        msl = make_concat_kernel(n_inputs=2)
        pipeline = compile_and_load(device, msl, "concat_kernel")

        a_buf = make_float_buffer(device, n, "ramp")
        b_buf = make_float_buffer(device, n, "ones")
        out_buf = make_empty_buffer(device, 2 * n)
        n0_buf = make_uint_buffer(device, n)
        n1_buf = make_uint_buffer(device, n)

        total = 2 * n
        result = bench_custom_dispatch(bench, pipeline,
                                        [a_buf, b_buf, out_buf, n0_buf, n1_buf],
                                        (total + 255) // 256, 256)
        n_bytes = total * 4 * 2  # read + write
        print(format_benchmark_result("concat_2", result, n_bytes=n_bytes))

        # Split into 2
        msl = make_split_kernel(n_outputs=2)
        pipeline = compile_and_load(device, msl, "split_kernel")

        in_buf = make_float_buffer(device, total, "ramp")
        o0_buf = make_empty_buffer(device, n)
        o1_buf = make_empty_buffer(device, n)
        cs_buf = make_uint_buffer(device, n)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, o0_buf, o1_buf, cs_buf],
                                        (total + 255) // 256, 256)
        print(format_benchmark_result("split_2", result, n_bytes=n_bytes))

    # =========================================================================
    # Top-P Sampling benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP-P (NUCLEUS) SAMPLING BENCHMARKS")
    print("=" * 60)

    for vocab_size in [1024, 32768, 128000]:
        print(f"\n--- vocab={vocab_size:,}, p=0.9 ---")

        msl = make_top_p_kernel(max_k=256, block_size=256)
        pipeline = compile_and_load(device, msl, "top_p")

        logits_buf = make_float_buffer(device, vocab_size, "ramp")
        val_buf = make_empty_buffer(device, 256)
        idx_buf = device.newBufferWithLength_options_(
            256 * 4, Metal.MTLResourceStorageModeShared
        )
        cnt_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        vsz_buf = make_uint_buffer(device, vocab_size)
        temp_buf = make_float_scalar_buffer(device, 1.0)
        p_buf = make_float_scalar_buffer(device, 0.9)

        result = bench_custom_dispatch(bench, pipeline,
                                        [logits_buf, val_buf, idx_buf, cnt_buf,
                                         vsz_buf, temp_buf, p_buf],
                                        1, 256)
        n_bytes = vocab_size * 4
        print(format_benchmark_result("top_p", result, n_bytes=n_bytes))

    # =========================================================================
    # Fused Residual + LayerNorm benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUSED RESIDUAL + LAYERNORM BENCHMARKS")
    print("=" * 60)

    for n_cols in [256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_fused_residual_norm_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "fused_residual_norm")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        res_buf = make_float_buffer(device, total, "small")
        gamma_buf = make_float_buffer(device, n_cols, "ones")
        beta_buf = make_float_buffer(device, n_cols, "small")
        out_buf = make_empty_buffer(device, total)
        res_out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, res_buf, gamma_buf, beta_buf,
                                         out_buf, res_out_buf, ncols_buf],
                                        n_rows, 256)
        # reads: input + residual + gamma + beta; writes: output + residual_out
        n_bytes = (total * 4 + n_cols * 2) * 4
        n_flops = total * 8  # add + mean + var + norm + scale + shift
        print(format_benchmark_result("fused_res_norm", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Paged Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("PAGED ATTENTION BENCHMARKS")
    print("=" * 60)

    for n_tokens, page_size in [(64, 16), (256, 16), (1024, 16)]:
        head_dim = 64
        n_pages = (n_tokens + page_size - 1) // page_size
        print(f"\n--- tokens={n_tokens}, pages={n_pages}, page_size={page_size} ---")

        msl = make_paged_attention_kernel(head_dim=head_dim, page_size=page_size)
        pipeline = compile_and_load(device, msl, "paged_attention")

        q_buf = make_float_buffer(device, head_dim, "small")
        k_buf = make_float_buffer(device, n_pages * page_size * head_dim, "ramp")
        v_buf = make_float_buffer(device, n_pages * page_size * head_dim, "ramp")
        # page table: identity mapping for benchmark
        pt_buf = device.newBufferWithLength_options_(
            n_pages * 4, Metal.MTLResourceStorageModeShared
        )
        pt_view = pt_buf.contents().as_buffer(n_pages * 4)
        for i in range(n_pages):
            struct.pack_into("I", pt_view, i * 4, i)
        out_buf = make_empty_buffer(device, head_dim)
        sl_buf = make_uint_buffer(device, n_tokens)
        np_buf = make_uint_buffer(device, n_pages)

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, pt_buf, out_buf,
                                         sl_buf, np_buf],
                                        1, 256)
        n_flops = 4 * n_tokens * head_dim
        n_bytes = (head_dim + 2 * n_tokens * head_dim + head_dim) * 4
        print(format_benchmark_result("paged_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Multi-Head Paged Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("MULTI-HEAD PAGED ATTENTION BENCHMARKS")
    print("=" * 60)

    for n_tokens, n_heads in [(256, 8), (512, 8)]:
        head_dim = 64
        page_size = 16
        n_pages = (n_tokens + page_size - 1) // page_size
        print(f"\n--- tokens={n_tokens}, heads={n_heads}, pages={n_pages} ---")

        msl = make_multi_head_paged_attention_kernel(
            n_heads=n_heads, head_dim=head_dim, page_size=page_size)
        pipeline = compile_and_load(device, msl, "multi_head_paged_attention")

        q_buf = make_float_buffer(device, n_heads * head_dim, "small")
        k_buf = make_float_buffer(device, n_pages * page_size * n_heads * head_dim, "ramp")
        v_buf = make_float_buffer(device, n_pages * page_size * n_heads * head_dim, "ramp")
        pt_buf = device.newBufferWithLength_options_(
            n_pages * 4, Metal.MTLResourceStorageModeShared
        )
        pt_view = pt_buf.contents().as_buffer(n_pages * 4)
        for i in range(n_pages):
            struct.pack_into("I", pt_view, i * 4, i)
        out_buf = make_empty_buffer(device, n_heads * head_dim)
        sl_buf = make_uint_buffer(device, n_tokens)
        np_buf = make_uint_buffer(device, n_pages)

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, pt_buf, out_buf,
                                         sl_buf, np_buf],
                                        n_heads, 256)
        n_flops = n_heads * 4 * n_tokens * head_dim
        n_bytes = (n_heads * head_dim + 2 * n_heads * n_tokens * head_dim +
                   n_heads * head_dim) * 4
        print(format_benchmark_result("mh_paged_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # FP16 KV-Cache Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FP16 KV-CACHE ATTENTION BENCHMARKS")
    print("=" * 60)

    for seq_len in [64, 256, 1024]:
        head_dim = 64
        print(f"\n--- seq_len={seq_len}, head_dim={head_dim} (K/V in FP16) ---")

        msl = make_fp16_kv_attention_kernel(head_dim=head_dim)
        pipeline = compile_and_load(device, msl, "fp16_kv_attention")

        q_buf = make_float_buffer(device, head_dim, "small")
        # K/V in half precision (2 bytes per element)
        kv_size = seq_len * head_dim
        k_buf = device.newBufferWithLength_options_(
            kv_size * 2, Metal.MTLResourceStorageModeShared
        )
        k_view = k_buf.contents().as_buffer(kv_size * 2)
        for i in range(kv_size):
            struct.pack_into("e", k_view, i * 2, float(i % 100) * 0.01)
        v_buf = device.newBufferWithLength_options_(
            kv_size * 2, Metal.MTLResourceStorageModeShared
        )
        v_view = v_buf.contents().as_buffer(kv_size * 2)
        for i in range(kv_size):
            struct.pack_into("e", v_view, i * 2, float(i % 100) * 0.01)
        out_buf = make_empty_buffer(device, head_dim)
        sl_buf = make_uint_buffer(device, seq_len)

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf, sl_buf],
                                        1, 256)
        n_flops = 4 * seq_len * head_dim
        # Q: float32, K/V: float16 (2x bandwidth savings)
        n_bytes = head_dim * 4 + 2 * kv_size * 2 + head_dim * 4
        print(format_benchmark_result("fp16_kv_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Fused MLP benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUSED MLP (SwiGLU) BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000, 4_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_fused_mlp_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "fused_mlp")

        gate_buf = make_float_buffer(device, n, "ramp")
        up_buf = make_float_buffer(device, n, "small")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [gate_buf, up_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4  # 2 reads + 1 write
        n_flops = 5 * n  # silu(4 ops) + mul(1 op)
        print(format_benchmark_result("fused_mlp", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Sliding Window Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SLIDING WINDOW ATTENTION BENCHMARKS")
    print("=" * 60)

    for seq_len, window_size in [(256, 128), (1024, 256), (2048, 512)]:
        head_dim = 64
        q_pos = seq_len - 1  # last position
        print(f"\n--- seq_len={seq_len}, window={window_size}, head_dim={head_dim} ---")

        msl = make_sliding_window_attention_kernel(
            head_dim=head_dim, window_size=window_size)
        pipeline = compile_and_load(device, msl, "sliding_window_attention")

        q_buf = make_float_buffer(device, head_dim, "small")
        k_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        v_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        out_buf = make_empty_buffer(device, head_dim)
        qpos_buf = make_uint_buffer(device, q_pos)
        sl_buf = make_uint_buffer(device, seq_len)

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf,
                                         qpos_buf, sl_buf],
                                        1, 256)
        # Only attends within window
        actual_window = min(window_size, seq_len)
        n_flops = 4 * actual_window * head_dim
        n_bytes = (head_dim + 2 * actual_window * head_dim + head_dim) * 4
        print(format_benchmark_result("sliding_win_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Repeat KV benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("REPEAT KV (GQA HEAD EXPANSION) BENCHMARKS")
    print("=" * 60)

    for n_kv_heads, n_rep, seq_len in [(8, 4, 256), (8, 4, 1024), (4, 8, 512)]:
        head_dim = 64
        n_q_heads = n_kv_heads * n_rep
        total_in = n_kv_heads * seq_len * head_dim
        total_out = n_q_heads * seq_len * head_dim
        print(f"\n--- kv_heads={n_kv_heads}, n_rep={n_rep}, seq={seq_len} ---")

        msl = make_repeat_kv_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "repeat_kv")

        in_buf = make_float_buffer(device, total_in, "ramp")
        out_buf = make_empty_buffer(device, total_out)
        nkv_buf = make_uint_buffer(device, n_kv_heads)
        sl_buf = make_uint_buffer(device, seq_len)
        hd_buf = make_uint_buffer(device, head_dim)
        nrep_buf = make_uint_buffer(device, n_rep)

        result = bench.time_kernel(pipeline,
                                    [in_buf, out_buf, nkv_buf, sl_buf, hd_buf, nrep_buf],
                                    total_out)
        n_bytes = (total_in + total_out) * 4
        print(format_benchmark_result("repeat_kv", result, n_bytes=n_bytes))

    # =========================================================================
    # Beam Search benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("BEAM SEARCH BENCHMARKS")
    print("=" * 60)

    for beam_width, vocab_size in [(4, 32768), (4, 128000), (8, 32768)]:
        print(f"\n--- beams={beam_width}, vocab={vocab_size:,} ---")

        msl = make_beam_search_kernel(beam_width=beam_width, block_size=256)
        pipeline = compile_and_load(device, msl, "beam_search")

        scores_buf = make_float_buffer(device, beam_width, "small")
        lp_buf = make_float_buffer(device, beam_width * vocab_size, "ramp")
        out_scores_buf = make_empty_buffer(device, beam_width)
        out_beam_buf = device.newBufferWithLength_options_(
            beam_width * 4, Metal.MTLResourceStorageModeShared
        )
        out_tok_buf = device.newBufferWithLength_options_(
            beam_width * 4, Metal.MTLResourceStorageModeShared
        )
        vsz_buf = make_uint_buffer(device, vocab_size)

        result = bench_custom_dispatch(bench, pipeline,
                                        [scores_buf, lp_buf, out_scores_buf,
                                         out_beam_buf, out_tok_buf, vsz_buf],
                                        1, 256)
        n_bytes = (beam_width + beam_width * vocab_size) * 4
        print(format_benchmark_result("beam_search", result, n_bytes=n_bytes))

    # =========================================================================
    # Speculative Decoding benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SPECULATIVE DECODING BENCHMARKS")
    print("=" * 60)

    for n_tokens, vocab_size in [(4, 32768), (8, 32768), (4, 128000)]:
        print(f"\n--- draft_tokens={n_tokens}, vocab={vocab_size:,} ---")

        msl = make_speculative_decode_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "speculative_decode")

        draft_buf = make_float_buffer(device, n_tokens * vocab_size, "small")
        target_buf = make_float_buffer(device, n_tokens * vocab_size, "small")
        # draft token IDs
        tok_buf = device.newBufferWithLength_options_(
            n_tokens * 4, Metal.MTLResourceStorageModeShared
        )
        tok_view = tok_buf.contents().as_buffer(n_tokens * 4)
        for i in range(n_tokens):
            struct.pack_into("I", tok_view, i * 4, i % vocab_size)
        # random values
        rand_buf = make_float_buffer(device, n_tokens, "small")
        # outputs
        accepted_buf = device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        adj_buf = make_empty_buffer(device, vocab_size)
        nt_buf = make_uint_buffer(device, n_tokens)
        vs_buf = make_uint_buffer(device, vocab_size)

        result = bench_custom_dispatch(bench, pipeline,
                                        [draft_buf, target_buf, tok_buf, rand_buf,
                                         accepted_buf, adj_buf, nt_buf, vs_buf],
                                        1, 256)
        n_bytes = 2 * n_tokens * vocab_size * 4  # read draft + target probs
        print(format_benchmark_result("spec_decode", result, n_bytes=n_bytes))

    # =========================================================================
    # Repeat KV (GQA head expansion)
    # =========================================================================
    print("\n" + "=" * 60)
    print("REPEAT KV BENCHMARKS")
    print("=" * 60)
    msl = make_repeat_kv_kernel(block_size=256)
    pipeline = compile_and_load(device, msl, "repeat_kv")
    for n_kv_heads, n_rep, seq_len, head_dim in [(8, 4, 128, 64), (8, 4, 512, 128)]:
        n_q_heads = n_kv_heads * n_rep
        in_size = n_kv_heads * seq_len * head_dim
        out_size = n_q_heads * seq_len * head_dim
        in_buf = make_float_buffer(device, in_size, "randn")
        out_buf = make_empty_buffer(device, out_size)
        nkv_buf = make_uint_buffer(device, n_kv_heads)
        sl_buf = make_uint_buffer(device, seq_len)
        hd_buf = make_uint_buffer(device, head_dim)
        nrep_buf = make_uint_buffer(device, n_rep)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, nkv_buf, sl_buf, hd_buf, nrep_buf],
                                        (out_size + 255) // 256, 256)
        n_bytes = (in_size + out_size) * 4
        label = f"repeat_kv_h{n_kv_heads}x{n_rep}_s{seq_len}_d{head_dim}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # 2D Dispatch Matmul
    # =========================================================================
    print("\n" + "=" * 60)
    print("2D DISPATCH MATMUL BENCHMARKS")
    print("=" * 60)
    block_m, block_n, block_k = 32, 32, 32
    threads_per_tg = block_m * block_n
    msl = make_matmul_2d_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    pipeline = compile_and_load(device, msl, "matmul_2d")
    for size in [64, 128, 256]:
        M = N = K = size
        a_buf = make_float_buffer(device, M * K, "randn")
        b_buf = make_float_buffer(device, K * N, "randn")
        c_buf = make_empty_buffer(device, M * N)
        m_buf = make_uint_buffer(device, M)
        n_buf = make_uint_buffer(device, N)
        k_buf = make_uint_buffer(device, K)

        n_tile_cols = (N + block_n - 1) // block_n
        n_tile_rows = (M + block_m - 1) // block_m

        result = bench_custom_dispatch(bench, pipeline,
                                        [a_buf, b_buf, c_buf, m_buf, n_buf, k_buf],
                                        n_tile_cols * n_tile_rows, threads_per_tg,
                                        dispatch_2d=(n_tile_cols, n_tile_rows))
        flops = 2 * M * N * K
        label = f"matmul_2d_{size}x{size}"
        print(format_benchmark_result(label, result, n_flops=flops))

    # =========================================================================
    # Swizzled Matmul (L2 cache optimized)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SWIZZLED MATMUL BENCHMARKS")
    print("=" * 60)
    msl = make_matmul_swizzled_kernel(block_m=32, block_n=32, block_k=32, group_size=4)
    pipeline = compile_and_load(device, msl, "matmul_swizzled")
    for size in [64, 128, 256]:
        M = N = K = size
        a_buf = make_float_buffer(device, M * K, "randn")
        b_buf = make_float_buffer(device, K * N, "randn")
        c_buf = make_empty_buffer(device, M * N)
        m_buf = make_uint_buffer(device, M)
        n_buf = make_uint_buffer(device, N)
        k_buf = make_uint_buffer(device, K)

        n_tile_cols = (N + 32 - 1) // 32
        n_tile_rows = (M + 32 - 1) // 32
        n_groups = n_tile_rows * n_tile_cols
        threads_per_tg = 32 * 32

        result = bench_custom_dispatch(bench, pipeline,
                                        [a_buf, b_buf, c_buf, m_buf, n_buf, k_buf],
                                        n_groups, threads_per_tg)
        flops = 2 * M * N * K
        label = f"matmul_swizzled_{size}x{size}"
        print(format_benchmark_result(label, result, n_flops=flops))

    # =========================================================================
    # Activation Functions
    # =========================================================================
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTION BENCHMARKS")
    print("=" * 60)
    n = 1_000_000
    in_buf = make_float_buffer(device, n, "randn")
    out_buf = make_empty_buffer(device, n)
    n_buf = make_uint_buffer(device, n)
    n_bytes = 2 * n * 4  # read + write

    for act_name in ["tanh", "sigmoid", "elu", "leaky_relu", "hardswish"]:
        msl = make_activation_kernel(act_name, block_size=256)
        pipeline = compile_and_load(device, msl, f"{act_name}_kernel")
        result = bench(pipeline, [in_buf, out_buf, n_buf], n)
        print(format_benchmark_result(act_name, result, n_bytes=n_bytes))

    # =========================================================================
    # Variance Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("VARIANCE BENCHMARKS")
    print("=" * 60)
    msl = make_variance_kernel(block_size=256)
    pipeline = compile_and_load(device, msl, "variance_kernel")
    for n_rows, n_cols in [(128, 256), (512, 1024)]:
        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "randn")
        out_buf = make_empty_buffer(device, n_rows)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = total * 4 + n_rows * 4
        label = f"variance_{n_rows}x{n_cols}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Batch Normalization Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("BATCH NORMALIZATION BENCHMARKS")
    print("=" * 60)
    msl = make_batch_norm_kernel(block_size=256)
    pipeline = compile_and_load(device, msl, "batch_norm_kernel")
    for n in [1024, 65536, 262144]:
        in_buf = make_float_buffer(device, n, "randn")
        mean_buf = make_float_buffer(device, n, "randn")
        var_buf = make_float_buffer(device, n, "const", 1.0)
        weight_buf = make_float_buffer(device, n, "const", 1.0)
        bias_buf = make_float_buffer(device, n, "const", 0.0)
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench(pipeline, [in_buf, mean_buf, var_buf, weight_buf, bias_buf, out_buf, n_buf], n)
        n_bytes = n * 4 * 6  # 5 reads + 1 write
        print(format_benchmark_result(f"batch_norm_{n}", result, n_bytes=n_bytes))

    # =========================================================================
    # Online Softmax Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("ONLINE SOFTMAX BENCHMARKS")
    print("=" * 60)
    msl = make_online_softmax_kernel(block_size=256)
    pipeline = compile_and_load(device, msl, "online_softmax_kernel")
    for n_rows, n_cols in [(128, 256), (512, 1024), (1024, 2048)]:
        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "randn")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = total * 4 * 2  # read + write
        label = f"online_softmax_{n_rows}x{n_cols}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Causal Attention Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("CAUSAL ATTENTION BENCHMARKS")
    print("=" * 60)
    for n_heads, head_dim in [(4, 32), (8, 64)]:
        msl = make_causal_attention_kernel(n_heads=n_heads, head_dim=head_dim, block_size=128)
        pipeline = compile_and_load(device, msl, "causal_attention")
        seq_len = 64
        total_q = n_heads * seq_len * head_dim
        total_kv = n_heads * seq_len * head_dim
        q_buf = make_float_buffer(device, total_q, "randn")
        k_buf = make_float_buffer(device, total_kv, "randn")
        v_buf = make_float_buffer(device, total_kv, "randn")
        out_buf = make_empty_buffer(device, total_q)
        seq_buf = make_uint_buffer(device, seq_len)

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf, seq_buf],
                                        n_heads, 128)
        n_bytes = (total_q + total_kv * 2 + total_q) * 4
        label = f"causal_attn_h{n_heads}_d{head_dim}_s{seq_len}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Group Normalization Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("GROUP NORMALIZATION BENCHMARKS")
    print("=" * 60)
    for n_groups, channels, spatial in [(32, 128, 256), (32, 256, 1024)]:
        msl = make_group_norm_kernel(n_groups=n_groups, block_size=256)
        pipeline = compile_and_load(device, msl, "group_norm_kernel")
        total = channels * spatial
        in_buf = make_float_buffer(device, total, "randn")
        weight_buf = make_float_buffer(device, channels, "const", 1.0)
        bias_buf = make_float_buffer(device, channels, "const", 0.0)
        out_buf = make_empty_buffer(device, total)
        nchan_buf = make_uint_buffer(device, channels)
        spatial_buf = make_uint_buffer(device, spatial)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, weight_buf, bias_buf, out_buf,
                                         nchan_buf, spatial_buf],
                                        n_groups, 256)
        n_bytes = total * 4 * 2 + channels * 4 * 2  # input/output + weight/bias
        label = f"group_norm_g{n_groups}_c{channels}_s{spatial}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Instance Normalization Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("INSTANCE NORMALIZATION BENCHMARKS")
    print("=" * 60)
    for channels, spatial in [(64, 256), (128, 1024)]:
        msl = make_instance_norm_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "instance_norm_kernel")
        total = channels * spatial
        in_buf = make_float_buffer(device, total, "randn")
        out_buf = make_empty_buffer(device, total)
        nchan_buf = make_uint_buffer(device, channels)
        spatial_buf = make_uint_buffer(device, spatial)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, nchan_buf, spatial_buf],
                                        channels, 256)
        n_bytes = total * 4 * 2  # input + output
        label = f"instance_norm_c{channels}_s{spatial}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Fused Dropout Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUSED DROPOUT BENCHMARKS")
    print("=" * 60)
    for n in [65536, 262144, 1048576]:
        msl = make_fused_dropout_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "fused_dropout_kernel")
        in_buf = make_float_buffer(device, n, "randn")
        out_buf = make_empty_buffer(device, n)
        seed_buf = make_uint_buffer(device, 42)
        n_buf = make_uint_buffer(device, n)
        n_groups = (n + 255) // 256

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, seed_buf, n_buf],
                                        n_groups, 256)
        n_bytes = n * 4 * 2
        label = f"dropout_n{n}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Gather/Scatter Kernels
    # =========================================================================
    print("\n" + "=" * 60)
    print("GATHER/SCATTER BENCHMARKS")
    print("=" * 60)
    for n in [65536, 262144]:
        # Gather
        msl = make_gather_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "gather_kernel")
        data_buf = make_float_buffer(device, n, "ramp")
        idx_buf = make_uint_buffer(device, 0)  # dummy for now
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)
        n_groups = (n + 255) // 256

        # Use identity indices (gid → gid) for benchmarking
        import struct as st
        idx_buf = device.newBufferWithLength_options_(
            n * 4, Metal.MTLResourceStorageModeShared
        )
        view = idx_buf.contents().as_buffer(n * 4)
        for i in range(n):
            st.pack_into("i", view, i * 4, i)

        result = bench_custom_dispatch(bench, pipeline,
                                        [data_buf, idx_buf, out_buf, n_buf],
                                        n_groups, 256)
        n_bytes = n * 4 * 3  # data + indices + output
        label = f"gather_n{n}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

        # Scatter (same setup)
        msl = make_scatter_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "scatter_kernel")
        result = bench_custom_dispatch(bench, pipeline,
                                        [data_buf, idx_buf, out_buf, n_buf],
                                        n_groups, 256)
        label = f"scatter_n{n}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Transpose Kernel
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRANSPOSE BENCHMARKS")
    print("=" * 60)
    for rows, cols in [(256, 256), (512, 1024)]:
        msl = make_transpose_kernel(block_size=256, tile_size=16)
        pipeline = compile_and_load(device, msl, "transpose_kernel")
        n = rows * cols
        in_buf = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, n)
        rows_buf = make_uint_buffer(device, rows)
        cols_buf = make_uint_buffer(device, cols)
        n_tiles_x = (cols + 15) // 16
        n_tiles_y = (rows + 15) // 16

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, rows_buf, cols_buf],
                                        1, 256,
                                        dispatch_2d=(n_tiles_x, n_tiles_y))
        n_bytes = n * 4 * 2
        label = f"transpose_{rows}x{cols}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Reduce-Scatter / All-Reduce Kernels
    # =========================================================================
    print("\n" + "=" * 60)
    print("REDUCE-SCATTER / ALL-REDUCE BENCHMARKS")
    print("=" * 60)
    for n in [65536, 262144, 1048576]:
        # Reduce-scatter (2 buffers)
        msl = make_reduce_scatter_kernel(n_buffers=2, block_size=256)
        pipeline = compile_and_load(device, msl, "reduce_scatter")
        buf_a = make_float_buffer(device, n, "ramp")
        buf_b = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)
        n_groups = (n + 255) // 256

        result = bench_custom_dispatch(bench, pipeline,
                                        [buf_a, buf_b, out_buf, n_buf],
                                        n_groups, 256)
        n_bytes = n * 4 * 3  # 2 inputs + 1 output
        label = f"reduce_scatter_2buf_n{n}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

        # All-reduce sum
        msl = make_all_reduce_kernel(n_buffers=2, op="sum")
        pipeline = compile_and_load(device, msl, "all_reduce")
        result = bench_custom_dispatch(bench, pipeline,
                                        [buf_a, buf_b, out_buf, n_buf],
                                        n_groups, 256)
        label = f"all_reduce_sum_2buf_n{n}"
        print(format_benchmark_result(label, result, n_bytes=n_bytes))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("M4 Max theoretical peak:")
    print("  Memory bandwidth: 546 GB/s")
    print("  FP32 compute: ~17.2 TFLOP/s (40 cores x 128 ALUs x 2 ops x 1.65 GHz)")
    print("=" * 60)


if __name__ == "__main__":
    main()
