"""Tests for the MSL emitter: generate kernels, compile, run on GPU, verify.

Each test validates:
1. The emitter produces valid MSL
2. The MSL compiles to a metallib
3. The kernel runs on the M4 Max GPU
4. Results match a reference implementation
"""

import math
import platform
import random

import pytest

from tests.conftest import requires_metal

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)

# runner fixture is provided by conftest.py


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add(runner):
    """output = a + b"""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 4096
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) for i in range(n)]
    b_data = [float(i) * 0.5 for i in range(n)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_elementwise_sub(runner):
    """output = a - b"""
    from triton_msl.codegen.msl_emitter import make_elementwise_kernel

    n = 2048
    msl = make_elementwise_kernel("sub_kernel", 2, "sub")
    path = runner.compile(msl, "sub_kernel")
    pipeline = runner.load(path, "sub_kernel")

    a_data = [float(i) * 3.0 for i in range(n)]
    b_data = [float(i) for i in range(n)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] - b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_elementwise_mul(runner):
    """output = a * b"""
    from triton_msl.codegen.msl_emitter import make_elementwise_kernel

    n = 2048
    msl = make_elementwise_kernel("mul_kernel", 2, "mul")
    path = runner.compile(msl, "mul_kernel")
    pipeline = runner.load(path, "mul_kernel")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.02 for i in range(n)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] * b_data[i]
        assert abs(result[i] - expected) < 1e-2, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_scalar_mul(runner):
    """output = input * scalar"""
    from triton_msl.codegen.msl_emitter import make_scalar_mul_kernel

    n = 1024
    scalar = 3.14
    msl = make_scalar_mul_kernel()
    path = runner.compile(msl, "scalar_mul")
    pipeline = runner.load(path, "scalar_mul")

    input_data = [float(i) for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    scalar_buf = runner.make_float_scalar_buffer(scalar)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, scalar_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = input_data[i] * scalar
        assert abs(result[i] - expected) < 1e-2, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_silu(runner):
    """output = x * sigmoid(x) = x / (1 + exp(-x))"""
    from triton_msl.codegen.msl_emitter import make_silu_kernel

    n = 1024
    msl = make_silu_kernel()
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    # Test range [-5, 5] to exercise both sides of sigmoid
    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_gelu(runner):
    """output = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))"""
    from triton_msl.codegen.msl_emitter import make_gelu_kernel

    n = 1024
    msl = make_gelu_kernel()
    path = runner.compile(msl, "gelu_kernel")
    pipeline = runner.load(path, "gelu_kernel")

    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        # GELU tanh approximation
        expected = 0.5 * x * (1.0 + math.tanh(
            0.7978845608028654 * (x + 0.044715 * x ** 3)
        ))
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_exp(runner):
    """output = exp(x)"""
    from triton_msl.codegen.msl_emitter import make_elementwise_kernel

    n = 512
    msl = make_elementwise_kernel("exp_kernel", 1, "exp")
    path = runner.compile(msl, "exp_kernel")
    pipeline = runner.load(path, "exp_kernel")

    # Avoid overflow: test range [-10, 10]
    input_data = [(i - n // 2) * 0.04 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        expected = math.exp(x)
        # Relative tolerance for large values
        tol = max(1e-4, abs(expected) * 1e-5)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_non_power_of_2_size(runner):
    """Test with n not divisible by block_size (tests masking)."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 1000  # Not divisible by 256
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) for i in range(n)]
    b_data = [1.0] * n

    # Allocate buffers slightly larger to detect out-of-bounds writes
    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_large_buffer(runner):
    """Test with a large buffer (1M elements)."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 1_000_000
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    # Use simple patterns to avoid slow Python list creation
    a_buf = runner.make_float_buffer([float(i % 1000) for i in range(n)])
    b_buf = runner.make_float_buffer([1.0] * n)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    # Spot-check a few values
    result = runner.read_float_buffer(out_buf, n)
    for i in [0, 1, 999, 1000, 500_000, 999_999]:
        expected = float(i % 1000) + 1.0
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Reduction tests
# ---------------------------------------------------------------------------

@requires_metal
def test_reduce_sum(runner):
    """output[0] = sum(input)"""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    n = 256  # One threadgroup
    msl = make_reduce_kernel("reduce_sum", "sum", block_size=256)
    path = runner.compile(msl, "reduce_sum")
    pipeline = runner.load(path, "reduce_sum")

    input_data = [float(i) * 0.01 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, 1)
    expected = sum(input_data)
    assert abs(result[0] - expected) < 0.5, (
        f"got {result[0]}, expected {expected}"
    )


@requires_metal
def test_reduce_max(runner):
    """output[0] = max(input)"""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    n = 256
    msl = make_reduce_kernel("reduce_max", "max", block_size=256)
    path = runner.compile(msl, "reduce_max")
    pipeline = runner.load(path, "reduce_max")

    random.seed(42)
    input_data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, 1)
    expected = max(input_data)
    assert abs(result[0] - expected) < 1e-3, (
        f"got {result[0]}, expected {expected}"
    )


@requires_metal
def test_reduce_min(runner):
    """output[0] = min(input)"""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    n = 256
    msl = make_reduce_kernel("reduce_min", "min", block_size=256)
    path = runner.compile(msl, "reduce_min")
    pipeline = runner.load(path, "reduce_min")

    random.seed(42)
    input_data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, 1)
    expected = min(input_data)
    assert abs(result[0] - expected) < 1e-3, (
        f"got {result[0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

@requires_metal
def test_softmax(runner):
    """Row-wise softmax: output[i] = exp(x[i] - max) / sum(exp(x - max))"""
    from triton_msl.codegen.msl_emitter import make_softmax_kernel

    n_rows = 4
    n_cols = 64
    msl = make_softmax_kernel(block_size=256)
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    # Generate input: multiple rows
    random.seed(123)
    input_data = [random.uniform(-3.0, 3.0) for _ in range(n_rows * n_cols)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    # Launch: one threadgroup per row
    runner.run(pipeline, [input_buf, out_buf, ncols_buf], n_cols,
               block_size=256)

    # Override run to use n_rows threadgroups
    import Metal
    n_groups = n_rows
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)

    # Verify each row
    for row in range(n_rows):
        start = row * n_cols
        row_input = input_data[start:start + n_cols]
        row_result = result[start:start + n_cols]

        # Reference softmax
        mx = max(row_input)
        exps = [math.exp(x - mx) for x in row_input]
        s = sum(exps)
        expected = [e / s for e in exps]

        for j in range(n_cols):
            assert abs(row_result[j] - expected[j]) < 1e-4, (
                f"row={row} col={j}: got {row_result[j]}, expected {expected[j]}"
            )

        # Verify probabilities sum to ~1.0
        row_sum = sum(row_result)
        assert abs(row_sum - 1.0) < 1e-4, (
            f"row={row}: sum={row_sum}, expected 1.0"
        )


@requires_metal
def test_softmax_large_row(runner):
    """Softmax with row larger than block_size (tests strided access)."""
    from triton_msl.codegen.msl_emitter import make_softmax_kernel

    n_cols = 512  # Larger than block_size=256
    msl = make_softmax_kernel(block_size=256)
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    random.seed(456)
    input_data = [random.uniform(-5.0, 5.0) for _ in range(n_cols)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    # Reference
    mx = max(input_data)
    exps = [math.exp(x - mx) for x in input_data]
    s = sum(exps)
    expected = [e / s for e in exps]

    for j in range(n_cols):
        assert abs(result[j] - expected[j]) < 1e-4, (
            f"col={j}: got {result[j]}, expected {expected[j]}"
        )

    assert abs(sum(result) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_matmul_small(runner):
    """C = A @ B for small matrices (fits in one tile)."""
    from triton_msl.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 16, 16, 16
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    # Simple test: identity-ish matrices
    random.seed(789)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    # Grid: one threadgroup (16x16 fits in 32x32 tile)
    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_groups = n_tile_rows * n_tile_cols
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    # Reference matmul
    for i in range(M):
        for j in range(N):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_rectangular(runner):
    """C = A @ B for non-square matrices."""
    from triton_msl.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 24, 16, 32
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    random.seed(101)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_groups = n_tile_rows * n_tile_cols
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(M):
        for j in range(N):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_multi_tile(runner):
    """C = A @ B with multiple tiles (matrix larger than tile size)."""
    from triton_msl.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 64, 64, 64
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    random.seed(202)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n  # 2
    n_tile_rows = (M + block_m - 1) // block_m  # 2
    n_groups = n_tile_rows * n_tile_cols  # 4
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    # Spot-check several elements (full check is slow for 64x64)
    for i in range(0, M, 8):
        for j in range(0, N, 8):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# 2D dispatch matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_matmul_2d_small(runner):
    """2D dispatch matmul: 16x16 * 16x16."""
    from triton_msl.codegen.msl_emitter import make_matmul_2d_kernel

    M, N, K = 16, 16, 16
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_2d_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_2d")
    pipeline = runner.load(path, "matmul_2d")

    random.seed(1001)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    # 2D dispatch: (tile_cols, tile_rows, 1)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_tile_cols, n_tile_rows, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(M):
        for j in range(N):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_2d_multi_tile(runner):
    """2D dispatch matmul: 64x64 with 2x2 tile grid."""
    from triton_msl.codegen.msl_emitter import make_matmul_2d_kernel

    M, N, K = 64, 64, 64
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_2d_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_2d")
    pipeline = runner.load(path, "matmul_2d")

    random.seed(1002)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n  # 2
    n_tile_rows = (M + block_m - 1) // block_m  # 2
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_tile_cols, n_tile_rows, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(0, M, 8):
        for j in range(0, N, 8):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_2d_rectangular(runner):
    """2D dispatch matmul: non-square 48x32 * 32x64."""
    from triton_msl.codegen.msl_emitter import make_matmul_2d_kernel

    M, N, K = 48, 64, 32
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_2d_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_2d")
    pipeline = runner.load(path, "matmul_2d")

    random.seed(1003)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n  # 2
    n_tile_rows = (M + block_m - 1) // block_m  # 2
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_tile_cols, n_tile_rows, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(0, M, 6):
        for j in range(0, N, 8):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Swizzled matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_matmul_swizzled_small(runner):
    """Swizzled matmul: 16x16 * 16x16."""
    from triton_msl.codegen.msl_emitter import make_matmul_swizzled_kernel

    M, N, K = 16, 16, 16
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_swizzled_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_swizzled")
    pipeline = runner.load(path, "matmul_swizzled")

    random.seed(2001)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_groups = n_tile_rows * n_tile_cols
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(M):
        for j in range(N):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_swizzled_large(runner):
    """Swizzled matmul: 128x128 with group_size=4."""
    from triton_msl.codegen.msl_emitter import make_matmul_swizzled_kernel

    M, N, K = 128, 128, 64
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_swizzled_kernel(block_m=block_m, block_n=block_n,
                                       block_k=block_k, group_size=4)
    path = runner.compile(msl, "matmul_swizzled")
    pipeline = runner.load(path, "matmul_swizzled")

    random.seed(2002)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_groups = n_tile_rows * n_tile_cols
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(0, M, 16):
        for j in range(0, N, 16):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Activation function tests
# ---------------------------------------------------------------------------

@requires_metal
def test_activation_tanh(runner):
    """Tanh activation kernel."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("tanh")
    path = runner.compile(msl, "tanh_kernel")
    pipeline = runner.load(path, "tanh_kernel")

    in_data = [float(i - 512) * 0.01 for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_float_buffer(out_buf, n)

    for i in [0, 100, 512, 900, 1023]:
        expected = math.tanh(in_data[i])
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_sigmoid(runner):
    """Sigmoid activation kernel."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("sigmoid")
    path = runner.compile(msl, "sigmoid_kernel")
    pipeline = runner.load(path, "sigmoid_kernel")

    in_data = [float(i - 512) * 0.01 for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_float_buffer(out_buf, n)

    for i in [0, 100, 512, 900, 1023]:
        x = in_data[i]
        expected = 1.0 / (1.0 + math.exp(-x))
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_elu(runner):
    """ELU activation kernel."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("elu")
    path = runner.compile(msl, "elu_kernel")
    pipeline = runner.load(path, "elu_kernel")

    in_data = [float(i - 512) * 0.01 for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_float_buffer(out_buf, n)

    for i in [0, 100, 512, 900, 1023]:
        x = in_data[i]
        expected = x if x >= 0 else (math.exp(x) - 1.0)
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_leaky_relu(runner):
    """Leaky ReLU activation kernel."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("leaky_relu")
    path = runner.compile(msl, "leaky_relu_kernel")
    pipeline = runner.load(path, "leaky_relu_kernel")

    in_data = [float(i - 512) * 0.01 for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_float_buffer(out_buf, n)

    for i in [0, 100, 512, 900, 1023]:
        x = in_data[i]
        expected = x if x >= 0 else 0.01 * x
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_hardswish(runner):
    """HardSwish activation kernel."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("hardswish")
    path = runner.compile(msl, "hardswish_kernel")
    pipeline = runner.load(path, "hardswish_kernel")

    in_data = [float(i - 512) * 0.01 for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_float_buffer(out_buf, n)

    for i in [0, 100, 512, 900, 1023]:
        x = in_data[i]
        expected = x * max(0.0, min(1.0, x / 6.0 + 0.5))
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


# ---------------------------------------------------------------------------
# FP16 tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add_fp16(runner):
    """output = a + b in half precision"""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 1024
    msl = make_vector_add_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.005 for i in range(n)]

    a_buf = runner.make_half_buffer(a_data)
    b_buf = runner.make_half_buffer(b_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_half_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        # FP16 has ~3 decimal digits of precision
        tol = max(1e-2, abs(expected) * 1e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_silu_fp16(runner):
    """output = x * sigmoid(x) in half precision"""
    from triton_msl.codegen.msl_emitter import make_silu_kernel

    n = 512
    msl = make_silu_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_half_buffer(input_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_half_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        tol = max(1e-2, abs(expected) * 5e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_elementwise_mul_fp16(runner):
    """output = a * b in half precision"""
    from triton_msl.codegen.msl_emitter import make_elementwise_kernel

    n = 1024
    msl = make_elementwise_kernel("mul_fp16", 2, "mul", dtype="fp16")
    path = runner.compile(msl, "mul_fp16")
    pipeline = runner.load(path, "mul_fp16")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.02 for i in range(n)]

    a_buf = runner.make_half_buffer(a_data)
    b_buf = runner.make_half_buffer(b_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_half_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] * b_data[i]
        tol = max(1e-1, abs(expected) * 5e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# FP16 activation tests
# ---------------------------------------------------------------------------

@requires_metal
def test_activation_tanh_fp16(runner):
    """Tanh activation in FP16."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("tanh", dtype="fp16")
    path = runner.compile(msl, "tanh_kernel")
    pipeline = runner.load(path, "tanh_kernel")

    in_data = [float(i - 512) * 0.005 for i in range(n)]
    in_buf = runner.make_half_buffer(in_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_half_buffer(out_buf, n)

    for i in [0, 256, 512, 768, 1023]:
        expected = math.tanh(in_data[i])
        assert abs(result[i] - expected) < 0.01, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_sigmoid_fp16(runner):
    """Sigmoid activation in FP16."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("sigmoid", dtype="fp16")
    path = runner.compile(msl, "sigmoid_kernel")
    pipeline = runner.load(path, "sigmoid_kernel")

    in_data = [float(i - 512) * 0.005 for i in range(n)]
    in_buf = runner.make_half_buffer(in_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_half_buffer(out_buf, n)

    for i in [0, 256, 512, 768, 1023]:
        x = in_data[i]
        expected = 1.0 / (1.0 + math.exp(-x))
        assert abs(result[i] - expected) < 0.01, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_elu_fp16(runner):
    """ELU activation in FP16."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 1024
    msl = make_activation_kernel("elu", dtype="fp16")
    path = runner.compile(msl, "elu_kernel")
    pipeline = runner.load(path, "elu_kernel")

    in_data = [float(i - 512) * 0.005 for i in range(n)]
    in_buf = runner.make_half_buffer(in_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n)
    result = runner.read_half_buffer(out_buf, n)

    for i in [0, 256, 512, 768, 1023]:
        x = in_data[i]
        expected = x if x >= 0 else (math.exp(x) - 1.0)
        assert abs(result[i] - expected) < 0.01, f"i={i}: {result[i]} != {expected}"


# ---------------------------------------------------------------------------
# simdgroup_matrix matmul tests
# ---------------------------------------------------------------------------

def _dispatch_simdgroup_matmul(runner, pipeline, buffers, M, N):
    """Dispatch a simdgroup matmul kernel with correct grid dimensions."""
    import Metal

    n_tile_cols = (N + 31) // 32
    n_tile_rows = (M + 31) // 32
    n_groups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(128, 1, 1),  # 4 SIMD groups x 32 threads
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"


def _ref_matmul(A, B, M, N, K):
    """Reference matmul for testing."""
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i * K + k] * B[k * N + j]
            C[i * N + j] = s
    return C


@requires_metal
def test_simdgroup_matmul_32x32(runner):
    """simdgroup_matrix matmul: 32x32 @ 32x32 (single tile)."""
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 32, 32, 32
    msl = make_simdgroup_matmul_kernel()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(303)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(M):
        for j in range(N):
            idx = i * N + j
            assert abs(result[idx] - expected[idx]) < 1e-2, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


@requires_metal
def test_simdgroup_matmul_64x64(runner):
    """simdgroup_matrix matmul: 64x64 @ 64x64 (2x2 tiles)."""
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 64, 64, 64
    msl = make_simdgroup_matmul_kernel()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(404)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    # Spot-check every 4th element
    for i in range(0, M, 4):
        for j in range(0, N, 4):
            idx = i * N + j
            assert abs(result[idx] - expected[idx]) < 1e-1, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


@requires_metal
def test_simdgroup_matmul_rectangular(runner):
    """simdgroup_matrix matmul: 64x32 @ 32x64 (rectangular)."""
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 64, 64, 32
    msl = make_simdgroup_matmul_kernel()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(505)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(0, M, 4):
        for j in range(0, N, 4):
            idx = i * N + j
            assert abs(result[idx] - expected[idx]) < 1e-1, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


# ---------------------------------------------------------------------------
# FP16 simdgroup_matrix matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_simdgroup_matmul_fp16_compiles(runner):
    """FP16 simdgroup matmul MSL compiles."""
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    runner.compile(msl, "simdgroup_matmul")


@requires_metal
def test_simdgroup_matmul_fp16_32x32(runner):
    """FP16 simdgroup matmul: half inputs, float accumulation, 32x32."""
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 32, 32, 32
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(707)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    # Half-precision inputs, float output
    A_buf = runner.make_half_buffer(A_data)
    B_buf = runner.make_half_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)  # float output
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(M):
        for j in range(N):
            idx = i * N + j
            # FP16 inputs lose precision → wider tolerance
            tol = max(0.1, abs(expected[idx]) * 0.05)
            assert abs(result[idx] - expected[idx]) < tol, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


@requires_metal
def test_simdgroup_matmul_fp16_64x64(runner):
    """FP16 simdgroup matmul: 64x64 multi-tile."""
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 64, 64, 64
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(808)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_half_buffer(A_data)
    B_buf = runner.make_half_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(0, M, 4):
        for j in range(0, N, 4):
            idx = i * N + j
            tol = max(0.15, abs(expected[idx]) * 0.1)
            assert abs(result[idx] - expected[idx]) < tol, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


# ---------------------------------------------------------------------------
# RMS normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_rms_norm(runner):
    """RMS norm: output = x * rsqrt(mean(x^2) + eps) * weight"""
    from triton_msl.codegen.msl_emitter import make_rms_norm_kernel

    n_cols = 64
    n_rows = 4
    eps = 1e-6
    msl = make_rms_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "rms_norm_kernel")
    pipeline = runner.load(path, "rms_norm_kernel")

    random.seed(321)
    input_data = [random.gauss(0, 1) for _ in range(n_rows * n_cols)]
    weight_data = [random.uniform(0.5, 1.5) for _ in range(n_cols)]

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, weight_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)

    # Reference RMS norm
    for row in range(n_rows):
        row_in = input_data[row * n_cols:(row + 1) * n_cols]
        row_out = result[row * n_cols:(row + 1) * n_cols]

        mean_sq = sum(x * x for x in row_in) / n_cols
        rms = 1.0 / math.sqrt(mean_sq + eps)

        for j in range(n_cols):
            expected = row_in[j] * rms * weight_data[j]
            tol = max(1e-4, abs(expected) * 1e-3)
            assert abs(row_out[j] - expected) < tol, (
                f"Row {row}[{j}] got {row_out[j]}, expected {expected}"
            )


@requires_metal
def test_rms_norm_large_row(runner):
    """RMS norm with row larger than block_size (tests strided access)."""
    from triton_msl.codegen.msl_emitter import make_rms_norm_kernel

    n_cols = 512
    eps = 1e-6
    msl = make_rms_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "rms_norm_kernel")
    pipeline = runner.load(path, "rms_norm_kernel")

    random.seed(654)
    input_data = [random.gauss(0, 2) for _ in range(n_cols)]
    weight_data = [1.0] * n_cols  # unity weight for simplicity

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, weight_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    mean_sq = sum(x * x for x in input_data) / n_cols
    rms = 1.0 / math.sqrt(mean_sq + eps)
    for j in range(n_cols):
        expected = input_data[j] * rms
        tol = max(1e-4, abs(expected) * 1e-3)
        assert abs(result[j] - expected) < tol, (
            f"[{j}] got {result[j]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# RoPE (rotary position embeddings) tests
# ---------------------------------------------------------------------------

@requires_metal
def test_rope(runner):
    """RoPE: apply rotary position embeddings."""
    from triton_msl.codegen.msl_emitter import make_rope_kernel

    dim = 64
    seq_len = 4
    msl = make_rope_kernel(block_size=256)
    path = runner.compile(msl, "rope_kernel")
    pipeline = runner.load(path, "rope_kernel")

    random.seed(987)
    input_data = [random.gauss(0, 1) for _ in range(seq_len * dim)]

    # Pre-compute inverse frequencies: 1 / (10000^(2i/dim))
    freqs = [1.0 / (10000.0 ** (2 * i / dim)) for i in range(dim // 2)]

    input_buf = runner.make_float_buffer(input_data)
    freqs_buf = runner.make_float_buffer(freqs)
    out_buf = runner.make_empty_buffer(seq_len * dim)
    dim_buf = runner.make_uint_buffer(dim)
    pos_buf = runner.make_uint_buffer(0)  # pos_offset = 0

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, freqs_buf, out_buf, dim_buf, pos_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(seq_len, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, seq_len * dim)

    # Reference RoPE
    for pos in range(seq_len):
        for i in range(dim // 2):
            theta = pos * freqs[i]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0 = input_data[pos * dim + 2 * i]
            x1 = input_data[pos * dim + 2 * i + 1]
            exp0 = x0 * cos_t - x1 * sin_t
            exp1 = x0 * sin_t + x1 * cos_t

            got0 = result[pos * dim + 2 * i]
            got1 = result[pos * dim + 2 * i + 1]
            assert abs(got0 - exp0) < 1e-4, (
                f"pos={pos} pair={i}[0]: got {got0}, expected {exp0}"
            )
            assert abs(got1 - exp1) < 1e-4, (
                f"pos={pos} pair={i}[1]: got {got1}, expected {exp1}"
            )


# ---------------------------------------------------------------------------
# Layer normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_layer_norm(runner):
    """Layer norm: output = (x - mean) / sqrt(var + eps) * gamma + beta"""
    from triton_msl.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 64
    n_rows = 4
    eps = 1e-6
    msl = make_layer_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    random.seed(111)
    input_data = [random.gauss(0, 1) for _ in range(n_rows * n_cols)]
    gamma_data = [random.uniform(0.5, 1.5) for _ in range(n_cols)]
    beta_data = [random.uniform(-0.5, 0.5) for _ in range(n_cols)]

    input_buf = runner.make_float_buffer(input_data)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, gamma_buf, beta_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)

    for row in range(n_rows):
        row_in = input_data[row * n_cols:(row + 1) * n_cols]
        row_out = result[row * n_cols:(row + 1) * n_cols]

        mean = sum(row_in) / n_cols
        var = sum((x - mean) ** 2 for x in row_in) / n_cols
        inv_std = 1.0 / math.sqrt(var + eps)

        for j in range(n_cols):
            expected = (row_in[j] - mean) * inv_std * gamma_data[j] + beta_data[j]
            tol = max(1e-4, abs(expected) * 1e-3)
            assert abs(row_out[j] - expected) < tol, (
                f"Row {row}[{j}] got {row_out[j]}, expected {expected}"
            )


@requires_metal
def test_layer_norm_large_row(runner):
    """Layer norm with row larger than block_size."""
    from triton_msl.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 512
    eps = 1e-6
    msl = make_layer_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    random.seed(222)
    input_data = [random.gauss(0, 2) for _ in range(n_cols)]
    gamma_data = [1.0] * n_cols
    beta_data = [0.0] * n_cols

    input_buf = runner.make_float_buffer(input_data)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, gamma_buf, beta_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    mean = sum(input_data) / n_cols
    var = sum((x - mean) ** 2 for x in input_data) / n_cols
    inv_std = 1.0 / math.sqrt(var + eps)

    for j in range(n_cols):
        expected = (input_data[j] - mean) * inv_std
        tol = max(1e-3, abs(expected) * 1e-2)
        assert abs(result[j] - expected) < tol, (
            f"[{j}] got {result[j]}, expected {expected}"
        )

    # Output should have mean ~0 and variance ~1
    out_mean = sum(result) / n_cols
    assert abs(out_mean) < 0.05, f"Output mean = {out_mean}, expected ~0"


# ---------------------------------------------------------------------------
# Cross-entropy loss tests
# ---------------------------------------------------------------------------

@requires_metal
def test_cross_entropy(runner):
    """Cross-entropy loss: loss = log_sum_exp(logits) - logits[target]"""
    from triton_msl.codegen.msl_emitter import make_cross_entropy_kernel

    n_rows = 4
    vocab_size = 32
    msl = make_cross_entropy_kernel(block_size=256)
    path = runner.compile(msl, "cross_entropy_kernel")
    pipeline = runner.load(path, "cross_entropy_kernel")

    random.seed(333)
    logits_data = [random.gauss(0, 2) for _ in range(n_rows * vocab_size)]
    targets_data = [random.randint(0, vocab_size - 1) for _ in range(n_rows)]

    logits_buf = runner.make_float_buffer(logits_data)
    # targets need int32 buffer
    import Metal
    import struct as st
    targets_buf = runner.device.newBufferWithLength_options_(
        n_rows * 4, Metal.MTLResourceStorageModeShared
    )
    view = targets_buf.contents().as_buffer(n_rows * 4)
    for i, t in enumerate(targets_data):
        st.pack_into("i", view, i * 4, t)

    losses_buf = runner.make_empty_buffer(n_rows)
    vocab_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([logits_buf, targets_buf, losses_buf, vocab_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(losses_buf, n_rows)

    for row in range(n_rows):
        row_logits = logits_data[row * vocab_size:(row + 1) * vocab_size]
        target = targets_data[row]

        # Reference: log_sum_exp - logits[target]
        mx = max(row_logits)
        log_sum_exp = mx + math.log(sum(math.exp(x - mx) for x in row_logits))
        expected = log_sum_exp - row_logits[target]

        assert abs(result[row] - expected) < 1e-3, (
            f"Row {row}: got {result[row]}, expected {expected}"
        )
        # Loss should be non-negative
        assert result[row] >= -1e-6, f"Row {row}: negative loss {result[row]}"


@requires_metal
def test_cross_entropy_large_vocab(runner):
    """Cross-entropy with vocab larger than block_size."""
    from triton_msl.codegen.msl_emitter import make_cross_entropy_kernel

    vocab_size = 1024  # > block_size of 256
    msl = make_cross_entropy_kernel(block_size=256)
    path = runner.compile(msl, "cross_entropy_kernel")
    pipeline = runner.load(path, "cross_entropy_kernel")

    random.seed(444)
    logits_data = [random.gauss(0, 1) for _ in range(vocab_size)]
    target = 500  # middle of vocab

    logits_buf = runner.make_float_buffer(logits_data)
    import Metal
    import struct as st
    targets_buf = runner.device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    view = targets_buf.contents().as_buffer(4)
    st.pack_into("i", view, 0, target)

    losses_buf = runner.make_empty_buffer(1)
    vocab_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([logits_buf, targets_buf, losses_buf, vocab_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(losses_buf, 1)

    mx = max(logits_data)
    log_sum_exp = mx + math.log(sum(math.exp(x - mx) for x in logits_data))
    expected = log_sum_exp - logits_data[target]

    assert abs(result[0] - expected) < 1e-2, (
        f"got {result[0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Flash Attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_flash_attention_single_head(runner):
    """Flash Attention: single head, short sequence."""
    from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 16
    head_dim = 64
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    random.seed(555)
    Q_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    K_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * seq_len * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + 15) // 16
    n_groups = n_heads * n_q_blocks

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(O_buf, n_heads * seq_len * head_dim)

    # Reference attention: softmax(Q @ K^T * scale) @ V
    for h in range(n_heads):
        ho = h * seq_len * head_dim
        # Compute S = Q @ K^T
        S = [[0.0] * seq_len for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                dot = 0.0
                for d in range(head_dim):
                    dot += Q_data[ho + i * head_dim + d] * K_data[ho + j * head_dim + d]
                S[i][j] = dot * scale

        # Softmax each row
        P = []
        for i in range(seq_len):
            mx = max(S[i])
            exps = [math.exp(s - mx) for s in S[i]]
            s = sum(exps)
            P.append([e / s for e in exps])

        # O = P @ V
        for i in range(seq_len):
            for d in range(head_dim):
                expected = sum(P[i][j] * V_data[ho + j * head_dim + d] for j in range(seq_len))
                got = result[ho + i * head_dim + d]
                tol = max(1e-3, abs(expected) * 1e-2)
                assert abs(got - expected) < tol, (
                    f"head={h} pos={i} dim={d}: got {got}, expected {expected}"
                )


@requires_metal
def test_flash_attention_multi_block(runner):
    """Flash Attention: sequence longer than one block."""
    from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 48  # 3 blocks of 16
    head_dim = 64
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    random.seed(666)
    Q_data = [random.gauss(0, 0.3) for _ in range(n_heads * seq_len * head_dim)]
    K_data = [random.gauss(0, 0.3) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.3) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * seq_len * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + 15) // 16
    n_groups = n_heads * n_q_blocks

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(O_buf, n_heads * seq_len * head_dim)

    # Reference: full attention
    S = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            dot = sum(Q_data[i * head_dim + d] * K_data[j * head_dim + d] for d in range(head_dim))
            S[i][j] = dot * scale

    P = []
    for i in range(seq_len):
        mx = max(S[i])
        exps = [math.exp(s - mx) for s in S[i]]
        s = sum(exps)
        P.append([e / s for e in exps])

    # Spot-check every 4th position, every 8th dim
    for i in range(0, seq_len, 4):
        for d in range(0, head_dim, 8):
            expected = sum(P[i][j] * V_data[j * head_dim + d] for j in range(seq_len))
            got = result[i * head_dim + d]
            tol = max(1e-2, abs(expected) * 0.05)
            assert abs(got - expected) < tol, (
                f"pos={i} dim={d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# BFloat16 tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add_bf16(runner):
    """output = a + b in bfloat16 precision"""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 1024
    msl = make_vector_add_kernel(block_size=256, dtype="bf16")
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.005 for i in range(n)]

    a_buf = runner.make_bf16_buffer(a_data)
    b_buf = runner.make_bf16_buffer(b_data)
    out_buf = runner.make_empty_bf16_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_bf16_buffer(out_buf, n)
    for i in range(0, n, 16):  # spot-check every 16th
        expected = a_data[i] + b_data[i]
        # BF16 has ~2-3 decimal digits of precision
        tol = max(0.1, abs(expected) * 0.02)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_silu_bf16(runner):
    """output = x * sigmoid(x) in bfloat16 precision"""
    from triton_msl.codegen.msl_emitter import make_silu_kernel

    n = 512
    msl = make_silu_kernel(block_size=256, dtype="bf16")
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_bf16_buffer(input_data)
    out_buf = runner.make_empty_bf16_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_bf16_buffer(out_buf, n)
    for i in range(0, n, 8):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        tol = max(0.05, abs(expected) * 0.1)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Residual + bias add tests
# ---------------------------------------------------------------------------

@requires_metal
def test_residual_add_with_bias(runner):
    """output = input + residual + bias"""
    from triton_msl.codegen.msl_emitter import make_residual_add_kernel

    n = 1024
    msl = make_residual_add_kernel(block_size=256, has_bias=True)
    path = runner.compile(msl, "residual_add_kernel")
    pipeline = runner.load(path, "residual_add_kernel")

    in_data = [float(i) * 0.1 for i in range(n)]
    res_data = [float(i) * 0.05 for i in range(n)]
    bias_data = [0.5] * n

    in_buf = runner.make_float_buffer(in_data)
    res_buf = runner.make_float_buffer(res_data)
    bias_buf = runner.make_float_buffer(bias_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, res_buf, bias_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = in_data[i] + res_data[i] + bias_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_residual_add_no_bias(runner):
    """output = input + residual (no bias)"""
    from triton_msl.codegen.msl_emitter import make_residual_add_kernel

    n = 1024
    msl = make_residual_add_kernel(block_size=256, has_bias=False)
    path = runner.compile(msl, "residual_add_kernel")
    pipeline = runner.load(path, "residual_add_kernel")

    in_data = [float(i) * 0.1 for i in range(n)]
    res_data = [float(i) * 0.05 for i in range(n)]

    in_buf = runner.make_float_buffer(in_data)
    res_buf = runner.make_float_buffer(res_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, res_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = in_data[i] + res_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# KV-cache attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_kv_cache_attention(runner):
    """KV-cache attention: single query token attending to cached KV."""
    from triton_msl.codegen.msl_emitter import make_kv_cache_attention_kernel

    head_dim = 64
    seq_len = 8
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_kv_cache_attention_kernel(head_dim=head_dim, block_size=256)
    path = runner.compile(msl, "kv_cache_attention")
    pipeline = runner.load(path, "kv_cache_attention")

    random.seed(777)
    Q_data = [random.gauss(0, 0.5) for _ in range(n_heads * head_dim)]
    K_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_heads, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(O_buf, n_heads * head_dim)

    # Reference: softmax(Q @ K^T * scale) @ V
    for h in range(n_heads):
        scores = []
        for j in range(seq_len):
            dot = sum(Q_data[h * head_dim + d] * K_data[h * seq_len * head_dim + j * head_dim + d]
                      for d in range(head_dim))
            scores.append(dot * scale)

        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        s = sum(exps)
        attn = [e / s for e in exps]

        for d in range(head_dim):
            expected = sum(attn[j] * V_data[h * seq_len * head_dim + j * head_dim + d]
                          for j in range(seq_len))
            got = result[h * head_dim + d]
            tol = max(1e-3, abs(expected) * 0.01)
            assert abs(got - expected) < tol, (
                f"head={h} dim={d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# SwiGLU tests
# ---------------------------------------------------------------------------

@requires_metal
def test_swiglu(runner):
    """SwiGLU: output = SiLU(gate) * x"""
    from triton_msl.codegen.msl_emitter import make_swiglu_kernel

    n = 1024
    msl = make_swiglu_kernel(block_size=256)
    path = runner.compile(msl, "swiglu_kernel")
    pipeline = runner.load(path, "swiglu_kernel")

    random.seed(888)
    x_data = [random.gauss(0, 1) for _ in range(n)]
    gate_data = [random.gauss(0, 1) for _ in range(n)]

    x_buf = runner.make_float_buffer(x_data)
    gate_buf = runner.make_float_buffer(gate_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [x_buf, gate_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        g = gate_data[i]
        silu_g = g / (1.0 + math.exp(-g))
        expected = silu_g * x_data[i]
        tol = max(1e-4, abs(expected) * 1e-3)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Embedding lookup tests
# ---------------------------------------------------------------------------

@requires_metal
def test_embedding_lookup(runner):
    """Embedding lookup: output[i] = table[indices[i]]"""
    from triton_msl.codegen.msl_emitter import make_embedding_kernel

    vocab_size = 32
    embed_dim = 64
    batch_size = 8

    msl = make_embedding_kernel(block_size=256)
    path = runner.compile(msl, "embedding_kernel")
    pipeline = runner.load(path, "embedding_kernel")

    random.seed(999)
    table_data = [random.gauss(0, 1) for _ in range(vocab_size * embed_dim)]
    indices = [random.randint(0, vocab_size - 1) for _ in range(batch_size)]

    table_buf = runner.make_float_buffer(table_data)

    import Metal
    import struct as st
    indices_buf = runner.device.newBufferWithLength_options_(
        batch_size * 4, Metal.MTLResourceStorageModeShared
    )
    view = indices_buf.contents().as_buffer(batch_size * 4)
    for i, idx in enumerate(indices):
        st.pack_into("i", view, i * 4, idx)

    out_buf = runner.make_empty_buffer(batch_size * embed_dim)
    dim_buf = runner.make_uint_buffer(embed_dim)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([table_buf, indices_buf, out_buf, dim_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(batch_size, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, batch_size * embed_dim)

    for b in range(batch_size):
        token_idx = indices[b]
        for d in range(embed_dim):
            expected = table_data[token_idx * embed_dim + d]
            got = result[b * embed_dim + d]
            assert abs(got - expected) < 1e-5, (
                f"batch={b} dim={d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Causal Flash Attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_flash_attention_causal(runner):
    """Causal Flash Attention: future tokens should be masked."""
    from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 16
    head_dim = 64
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16, causal=True)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    random.seed(1234)
    Q_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    K_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * seq_len * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + 15) // 16
    n_groups = n_heads * n_q_blocks

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(O_buf, n_heads * seq_len * head_dim)

    # Reference: causal attention (lower triangular mask)
    S = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:  # causal: attend only to past + self
                dot = sum(Q_data[i * head_dim + d] * K_data[j * head_dim + d]
                          for d in range(head_dim))
                S[i][j] = dot * scale
            else:
                S[i][j] = float('-inf')

    P = []
    for i in range(seq_len):
        mx = max(S[i])
        exps = [math.exp(s - mx) if s > float('-inf') else 0.0 for s in S[i]]
        s = sum(exps)
        P.append([e / s if s > 0 else 0.0 for e in exps])

    for i in range(seq_len):
        for d in range(0, head_dim, 8):  # spot-check every 8th dim
            expected = sum(P[i][j] * V_data[j * head_dim + d] for j in range(seq_len))
            got = result[i * head_dim + d]
            tol = max(1e-3, abs(expected) * 0.02)
            assert abs(got - expected) < tol, (
                f"pos={i} dim={d}: got {got}, expected {expected}"
            )

    # Verify causal property: first token output should only depend on first token
    # (i.e., O[0] = V[0] since attention is all on position 0)
    for d in range(head_dim):
        got = result[d]
        expected = V_data[d]  # position 0 attends only to itself
        tol = max(1e-3, abs(expected) * 0.02)
        assert abs(got - expected) < tol, (
            f"Causal check: O[0][{d}] = {got}, V[0][{d}] = {expected}"
        )


# ---------------------------------------------------------------------------
# Fused linear kernel tests
# ---------------------------------------------------------------------------

def _dispatch_fused_linear(runner, pipeline, buffers, M, N):
    """Dispatch a fused linear kernel with correct grid dimensions."""
    import Metal

    n_tile_cols = (N + 31) // 32
    n_tile_rows = (M + 31) // 32
    n_groups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(128, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"


@requires_metal
def test_fused_linear_no_bias(runner):
    """Fused linear: output = input @ weight^T (no bias)."""
    from triton_msl.codegen.msl_emitter import make_fused_linear_kernel

    M, N, K = 32, 32, 32
    msl = make_fused_linear_kernel(has_bias=False)
    path = runner.compile(msl, "fused_linear")
    pipeline = runner.load(path, "fused_linear")

    random.seed(1111)
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # weight is [N, K] (row-major)
    weight_data = [random.uniform(-1.0, 1.0) for _ in range(N * K)]

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_fused_linear(runner, pipeline,
                           [input_buf, weight_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)

    # Reference: output = input @ weight^T
    for i in range(M):
        for j in range(N):
            expected = sum(input_data[i * K + k] * weight_data[j * K + k] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_fused_linear_with_bias(runner):
    """Fused linear: output = input @ weight^T + bias."""
    from triton_msl.codegen.msl_emitter import make_fused_linear_kernel

    M, N, K = 32, 32, 32
    msl = make_fused_linear_kernel(has_bias=True)
    path = runner.compile(msl, "fused_linear")
    pipeline = runner.load(path, "fused_linear")

    random.seed(2222)
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    weight_data = [random.uniform(-1.0, 1.0) for _ in range(N * K)]
    bias_data = [random.uniform(-0.5, 0.5) for _ in range(N)]

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    C_buf = runner.make_empty_buffer(M * N)
    bias_buf = runner.make_float_buffer(bias_data)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_fused_linear(runner, pipeline,
                           [input_buf, weight_buf, C_buf, bias_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(M):
        for j in range(N):
            matmul = sum(input_data[i * K + k] * weight_data[j * K + k] for k in range(K))
            expected = matmul + bias_data[j]
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA) tests
# ---------------------------------------------------------------------------


@requires_metal
def test_gqa_attention(runner):
    """GQA: 4 query heads share 1 KV head, verify attention output."""
    from triton_msl.codegen.msl_emitter import make_gqa_attention_kernel

    n_q_heads = 4
    n_kv_heads = 1
    n_q_per_kv = n_q_heads // n_kv_heads
    seq_len = 16
    head_dim = 64

    msl = make_gqa_attention_kernel(n_q_per_kv=n_q_per_kv)
    path = runner.compile(msl, "gqa_attention")
    pipeline = runner.load(path, "gqa_attention")

    random.seed(7777)
    # Q: [n_q_heads, head_dim] — each query head is different
    q_data = [random.uniform(-0.5, 0.5) for _ in range(n_q_heads * head_dim)]
    # K: [n_kv_heads, seq_len, head_dim]
    k_data = [random.uniform(-0.5, 0.5)
              for _ in range(n_kv_heads * seq_len * head_dim)]
    # V: [n_kv_heads, seq_len, head_dim]
    v_data = [random.uniform(-0.5, 0.5)
              for _ in range(n_kv_heads * seq_len * head_dim)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    out_buf = runner.make_empty_buffer(n_q_heads * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_val = 1.0 / math.sqrt(head_dim)
    scale_buf = runner.make_float_scalar_buffer(scale_val)

    import Metal
    n_groups = n_q_heads  # one threadgroup per query head
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf,
                              seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, n_q_heads * head_dim)

    # Reference: softmax(Q @ K^T / sqrt(d)) @ V for each query head
    # All query heads share the same KV (since n_kv_heads=1)
    scale = 1.0 / math.sqrt(head_dim)
    for qh in range(n_q_heads):
        kv_h = qh // n_q_per_kv  # which KV head to use

        # Compute attention scores
        scores = []
        for s in range(seq_len):
            dot = sum(q_data[qh * head_dim + d] *
                      k_data[(kv_h * seq_len + s) * head_dim + d]
                      for d in range(head_dim))
            scores.append(dot * scale)

        # Softmax
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        attn = [e / sum_exp for e in exp_s]

        # Weighted sum of V
        for d in range(head_dim):
            expected = sum(attn[s] *
                          v_data[(kv_h * seq_len + s) * head_dim + d]
                          for s in range(seq_len))
            got = result[qh * head_dim + d]
            assert abs(got - expected) < 0.05, (
                f"head {qh}, dim {d}: got {got}, expected {expected}"
            )

    # Verify different query heads produce different outputs
    head0 = result[:head_dim]
    head1 = result[head_dim:2 * head_dim]
    diff = sum(abs(a - b) for a, b in zip(head0, head1))
    assert diff > 0.01, "Different query heads should produce different outputs"


# ---------------------------------------------------------------------------
# INT8 Quantized Matmul tests
# ---------------------------------------------------------------------------


@requires_metal
def test_int8_matmul(runner):
    """INT8 weight-only quantized matmul with per-row scale/zero_point."""
    from triton_msl.codegen.msl_emitter import make_int8_matmul_kernel
    import struct as struct_mod

    M, N, K = 16, 16, 32
    msl = make_int8_matmul_kernel()
    path = runner.compile(msl, "int8_matmul")
    pipeline = runner.load(path, "int8_matmul")

    random.seed(8888)
    # Input: float [M, K]
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # Quantized weights: int8 [N, K] stored as char
    scale = 0.1
    zero_point = 0
    # Generate int8 weights in range [-10, 10]
    weight_int8 = [random.randint(-10, 10) for _ in range(N * K)]
    # Scale per row: [N]
    scale_data = [scale] * N
    # Zero point per row: [N]
    zp_data = [float(zero_point)] * N

    # Create int8 weight buffer (packed as signed bytes)
    import Metal
    w_buf = runner.device.newBufferWithLength_options_(
        N * K, Metal.MTLResourceStorageModeShared
    )
    w_view = w_buf.contents().as_buffer(N * K)
    for i, val in enumerate(weight_int8):
        struct_mod.pack_into("b", w_view, i, val)

    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(M * N)
    scale_buf = runner.make_float_buffer(scale_data)
    zp_buf = runner.make_float_buffer(zp_data)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    # Dispatch: 1D grid, one thread per output element
    total_elements = M * N
    block_size = 256
    n_groups = (total_elements + block_size - 1) // block_size
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    # Buffer order matches kernel: input, weight, output, scales, zeros, M, N, K
    for i, buf in enumerate([input_buf, w_buf, out_buf, scale_buf, zp_buf,
                              M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, M * N)

    # Reference: dequantize weights then matmul
    for i in range(M):
        for j in range(N):
            expected = sum(
                input_data[i * K + k] *
                (float(weight_int8[j * K + k]) - zero_point) * scale
                for k in range(K)
            )
            got = result[i * N + j]
            assert abs(got - expected) < 0.5, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Concat and Split tests
# ---------------------------------------------------------------------------


@requires_metal
def test_concat_two_tensors(runner):
    """Concatenate two 1D tensors into one."""
    from triton_msl.codegen.msl_emitter import make_concat_kernel

    n0 = 128
    n1 = 256
    total = n0 + n1

    msl = make_concat_kernel(n_inputs=2)
    path = runner.compile(msl, "concat_kernel")
    pipeline = runner.load(path, "concat_kernel")

    a_data = [float(i) for i in range(n0)]
    b_data = [float(i + 1000) for i in range(n1)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(total)
    n0_buf = runner.make_uint_buffer(n0)
    n1_buf = runner.make_uint_buffer(n1)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n0_buf, n1_buf],
               total, block_size=256)

    result = runner.read_float_buffer(out_buf, total)

    # First n0 elements should be a_data
    for i in range(n0):
        assert result[i] == pytest.approx(a_data[i], abs=0.01), (
            f"idx {i}: got {result[i]}, expected {a_data[i]}"
        )
    # Next n1 elements should be b_data
    for i in range(n1):
        assert result[n0 + i] == pytest.approx(b_data[i], abs=0.01), (
            f"idx {n0+i}: got {result[n0+i]}, expected {b_data[i]}"
        )


@requires_metal
def test_split_two_chunks(runner):
    """Split a tensor into two equal chunks."""
    from triton_msl.codegen.msl_emitter import make_split_kernel

    chunk_size = 128
    total = chunk_size * 2

    msl = make_split_kernel(n_outputs=2)
    path = runner.compile(msl, "split_kernel")
    pipeline = runner.load(path, "split_kernel")

    data = [float(i) for i in range(total)]
    in_buf = runner.make_float_buffer(data)
    out0_buf = runner.make_empty_buffer(chunk_size)
    out1_buf = runner.make_empty_buffer(chunk_size)
    cs_buf = runner.make_uint_buffer(chunk_size)

    runner.run(pipeline, [in_buf, out0_buf, out1_buf, cs_buf],
               total, block_size=256)

    r0 = runner.read_float_buffer(out0_buf, chunk_size)
    r1 = runner.read_float_buffer(out1_buf, chunk_size)

    for i in range(chunk_size):
        assert r0[i] == pytest.approx(data[i], abs=0.01)
        assert r1[i] == pytest.approx(data[chunk_size + i], abs=0.01)

# ---------------------------------------------------------------------------
# Top-K Sampling tests
# ---------------------------------------------------------------------------


@requires_metal
def test_top_k_basic(runner):
    """Top-k: find top 5 values from 1024-element vocabulary."""
    from triton_msl.codegen.msl_emitter import make_top_k_kernel
    import struct as struct_mod

    vocab_size = 1024
    k = 5

    msl = make_top_k_kernel(k=k, block_size=256)
    path = runner.compile(msl, "top_k")
    pipeline = runner.load(path, "top_k")

    # Create logits with known top-k values
    random.seed(9999)
    logits = [random.uniform(-2.0, 2.0) for _ in range(vocab_size)]
    # Plant known large values at specific indices
    top_indices = [17, 42, 500, 731, 999]
    top_values = [10.0, 9.5, 9.0, 8.5, 8.0]
    for idx, val in zip(top_indices, top_values):
        logits[idx] = val

    logits_buf = runner.make_float_buffer(logits)
    out_val_buf = runner.make_empty_buffer(k)
    # Create uint output buffer for indices
    import Metal
    out_idx_buf = runner.device.newBufferWithLength_options_(
        k * 4, Metal.MTLResourceStorageModeShared
    )
    vocab_buf = runner.make_uint_buffer(vocab_size)
    k_buf = runner.make_uint_buffer(k)

    runner.run(pipeline, [logits_buf, out_val_buf, out_idx_buf, vocab_buf, k_buf],
               256, block_size=256)

    result_vals = runner.read_float_buffer(out_val_buf, k)
    # Read uint indices
    idx_view = out_idx_buf.contents().as_buffer(k * 4)
    result_idxs = [struct_mod.unpack_from("I", idx_view, i * 4)[0] for i in range(k)]

    # Verify the top-k values are returned in descending order
    assert result_vals[0] == pytest.approx(10.0, abs=0.01)
    assert result_vals[1] == pytest.approx(9.5, abs=0.01)
    assert result_vals[2] == pytest.approx(9.0, abs=0.01)
    assert result_vals[3] == pytest.approx(8.5, abs=0.01)
    assert result_vals[4] == pytest.approx(8.0, abs=0.01)

    # Verify indices match
    assert set(result_idxs) == set(top_indices)
    assert result_idxs[0] == 17   # index of highest value
    assert result_idxs[1] == 42


@requires_metal
def test_top_k_large_vocab(runner):
    """Top-k with a larger vocabulary (32K) and k=10."""
    from triton_msl.codegen.msl_emitter import make_top_k_kernel
    import struct as struct_mod

    vocab_size = 32768
    k = 10

    msl = make_top_k_kernel(k=k, block_size=256)
    path = runner.compile(msl, "top_k")
    pipeline = runner.load(path, "top_k")

    random.seed(1234)
    logits = [random.uniform(-5.0, 5.0) for _ in range(vocab_size)]

    # Reference: sort and take top k
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    expected_indices = [idx for idx, _ in indexed[:k]]
    expected_values = [val for _, val in indexed[:k]]

    logits_buf = runner.make_float_buffer(logits)
    out_val_buf = runner.make_empty_buffer(k)
    import Metal
    out_idx_buf = runner.device.newBufferWithLength_options_(
        k * 4, Metal.MTLResourceStorageModeShared
    )
    vocab_buf = runner.make_uint_buffer(vocab_size)
    k_buf = runner.make_uint_buffer(k)

    runner.run(pipeline, [logits_buf, out_val_buf, out_idx_buf, vocab_buf, k_buf],
               256, block_size=256)

    result_vals = runner.read_float_buffer(out_val_buf, k)
    idx_view = out_idx_buf.contents().as_buffer(k * 4)
    result_idxs = [struct_mod.unpack_from("I", idx_view, i * 4)[0] for i in range(k)]

    # All returned values should match the reference top-k
    for i in range(k):
        assert result_vals[i] == pytest.approx(expected_values[i], abs=0.01), (
            f"top-{i}: got {result_vals[i]}, expected {expected_values[i]}"
        )
        assert result_idxs[i] == expected_indices[i], (
            f"top-{i} index: got {result_idxs[i]}, expected {expected_indices[i]}"
        )


# ---------------------------------------------------------------------------
# Top-P (Nucleus) Sampling tests
# ---------------------------------------------------------------------------


@requires_metal
def test_top_p_sampling(runner):
    """Top-p: nucleus sampling with p=0.9 and temperature=1.0."""
    from triton_msl.codegen.msl_emitter import make_top_p_kernel
    import struct as struct_mod

    vocab_size = 512
    max_k = 256
    temperature = 1.0
    p_threshold = 0.9

    msl = make_top_p_kernel(max_k=max_k, block_size=256)
    path = runner.compile(msl, "top_p")
    pipeline = runner.load(path, "top_p")

    # Create logits with a peaked distribution (few tokens dominate)
    random.seed(5555)
    logits = [random.uniform(-10.0, -5.0) for _ in range(vocab_size)]
    # Make a few tokens very likely (high logits vs very low background)
    logits[10] = 10.0
    logits[20] = 9.5
    logits[30] = 9.0
    logits[40] = 8.5

    logits_buf = runner.make_float_buffer(logits)
    out_val_buf = runner.make_empty_buffer(max_k)
    import Metal
    out_idx_buf = runner.device.newBufferWithLength_options_(
        max_k * 4, Metal.MTLResourceStorageModeShared
    )
    out_count_buf = runner.device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    vocab_buf = runner.make_uint_buffer(vocab_size)
    temp_buf = runner.make_float_scalar_buffer(temperature)
    p_buf = runner.make_float_scalar_buffer(p_threshold)

    runner.run(pipeline,
               [logits_buf, out_val_buf, out_idx_buf, out_count_buf,
                vocab_buf, temp_buf, p_buf],
               256, block_size=256)

    # Read count
    count_view = out_count_buf.contents().as_buffer(4)
    count = struct_mod.unpack_from("I", count_view, 0)[0]

    result_vals = runner.read_float_buffer(out_val_buf, count)
    idx_view = out_idx_buf.contents().as_buffer(count * 4)
    result_idxs = [struct_mod.unpack_from("I", idx_view, i * 4)[0] for i in range(count)]

    # Verify: probabilities should be in descending order
    for i in range(len(result_vals) - 1):
        assert result_vals[i] >= result_vals[i + 1] - 1e-6, (
            f"Not sorted: p[{i}]={result_vals[i]} < p[{i+1}]={result_vals[i+1]}"
        )

    # Verify: cumulative probability should reach p_threshold
    cum = sum(result_vals)
    assert cum >= p_threshold - 0.01, f"Cumulative {cum} < threshold {p_threshold}"

    # Verify: top indices should include our peaked tokens
    assert 10 in result_idxs, "Highest logit token should be in nucleus"
    assert 20 in result_idxs, "Second highest logit token should be in nucleus"

    # The nucleus should be small since the distribution is peaked
    assert count < 50, f"Peaked distribution should have small nucleus, got {count}"


# ---------------------------------------------------------------------------
# Batched KV-Cache Decode tests
# ---------------------------------------------------------------------------


@requires_metal
def test_batched_kv_decode(runner):
    """Batched multi-head KV-cache decode with 2 batch items, 2 heads."""
    from triton_msl.codegen.msl_emitter import make_batched_kv_decode_kernel

    batch_size = 2
    n_heads = 2
    head_dim = 32
    max_seq_len = 16
    # Different sequence lengths per batch item
    actual_seq_lens = [8, 12]

    msl = make_batched_kv_decode_kernel(n_heads=n_heads, head_dim=head_dim,
                                         block_size=256)
    path = runner.compile(msl, "batched_kv_decode")
    pipeline = runner.load(path, "batched_kv_decode")

    random.seed(6666)
    # Q: [batch, n_heads, head_dim]
    q_data = [random.uniform(-0.5, 0.5)
              for _ in range(batch_size * n_heads * head_dim)]
    # K/V: [batch, n_heads, max_seq_len, head_dim]
    kv_size = batch_size * n_heads * max_seq_len * head_dim
    k_data = [random.uniform(-0.5, 0.5) for _ in range(kv_size)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(kv_size)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    out_buf = runner.make_empty_buffer(batch_size * n_heads * head_dim)

    # seq_lens buffer: array of uints
    import Metal
    import struct as struct_mod
    sl_buf = runner.device.newBufferWithLength_options_(
        batch_size * 4, Metal.MTLResourceStorageModeShared
    )
    sl_view = sl_buf.contents().as_buffer(batch_size * 4)
    for i, sl in enumerate(actual_seq_lens):
        struct_mod.pack_into("I", sl_view, i * 4, sl)

    max_sl_buf = runner.make_uint_buffer(max_seq_len)
    bs_buf = runner.make_uint_buffer(batch_size)

    # Dispatch: one threadgroup per (batch, head)
    n_groups = batch_size * n_heads
    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf,
                              max_sl_buf, bs_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, batch_size * n_heads * head_dim)

    # Reference implementation
    scale = 1.0 / math.sqrt(head_dim)
    for b in range(batch_size):
        seq_len = actual_seq_lens[b]
        for h in range(n_heads):
            q_off = (b * n_heads + h) * head_dim
            kv_off = (b * n_heads + h) * max_seq_len * head_dim
            o_off = q_off

            # Compute scores
            scores = []
            for j in range(seq_len):
                dot = sum(q_data[q_off + d] * k_data[kv_off + j * head_dim + d]
                          for d in range(head_dim))
                scores.append(dot * scale)

            # Softmax
            max_s = max(scores)
            exp_s = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exp_s)
            attn = [e / sum_exp for e in exp_s]

            # Weighted V
            for d in range(head_dim):
                expected = sum(attn[j] * v_data[kv_off + j * head_dim + d]
                               for j in range(seq_len))
                got = result[o_off + d]
                assert abs(got - expected) < 0.05, (
                    f"batch {b} head {h} dim {d}: got {got}, expected {expected}"
                )


# ---------------------------------------------------------------------------
# INT4 Quantized Matmul tests
# ---------------------------------------------------------------------------


@requires_metal
def test_int4_matmul(runner):
    """INT4 weight-only quantized matmul with per-group scale/zero_point."""
    from triton_msl.codegen.msl_emitter import make_int4_matmul_kernel
    import struct as struct_mod

    M, N, K = 8, 8, 16  # K must be even for int4 packing
    group_size = 16  # One group covers entire K
    msl = make_int4_matmul_kernel(group_size=group_size)
    path = runner.compile(msl, "int4_matmul")
    pipeline = runner.load(path, "int4_matmul")

    random.seed(7070)
    # Input: float [M, K]
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # INT4 weights: values 0-15, packed 2 per byte
    # Generate random 4-bit values
    weight_int4 = [random.randint(0, 15) for _ in range(N * K)]
    # Pack into bytes: even=low nibble, odd=high nibble
    packed_bytes = []
    for n in range(N):
        for k_pair in range(K // 2):
            low = weight_int4[n * K + k_pair * 2]
            high = weight_int4[n * K + k_pair * 2 + 1]
            packed_bytes.append((high << 4) | low)

    # Scale and zero per group
    n_groups = (K + group_size - 1) // group_size
    scale = 0.1
    zero_point = 8.0  # center of uint4 range
    scale_data = [scale] * (N * n_groups)
    zp_data = [zero_point] * (N * n_groups)

    # Create packed weight buffer
    import Metal
    w_buf = runner.device.newBufferWithLength_options_(
        len(packed_bytes), Metal.MTLResourceStorageModeShared
    )
    w_view = w_buf.contents().as_buffer(len(packed_bytes))
    for i, val in enumerate(packed_bytes):
        struct_mod.pack_into("B", w_view, i, val)

    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(M * N)
    scale_buf = runner.make_float_buffer(scale_data)
    zp_buf = runner.make_float_buffer(zp_data)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    total = M * N
    block_size = 256
    n_threadgroups = (total + block_size - 1) // block_size
    runner.run(pipeline,
               [input_buf, w_buf, out_buf, scale_buf, zp_buf,
                M_buf, N_buf, K_buf],
               total, block_size=block_size)

    result = runner.read_float_buffer(out_buf, M * N)

    # Reference: dequantize and matmul
    for i in range(M):
        for j in range(N):
            expected = sum(
                input_data[i * K + k] *
                (float(weight_int4[j * K + k]) - zero_point) * scale
                for k in range(K)
            )
            got = result[i * N + j]
            assert abs(got - expected) < 0.5, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Fused RoPE + Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_rope_attention(runner):
    """Fused RoPE + single-query attention: applies rotary embeddings to Q and K
    on-the-fly during attention, validating against a Python reference."""
    from triton_msl.codegen.msl_emitter import make_rope_attention_kernel

    head_dim = 64
    seq_len = 16
    q_pos = 10  # Position of the query token

    msl = make_rope_attention_kernel(head_dim=head_dim, block_size=256)
    path = runner.compile(msl, "rope_attention")
    pipeline = runner.load(path, "rope_attention")

    random.seed(8888)
    half_d = head_dim // 2

    # Q: [head_dim] — single query vector
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]
    # K_cache: [seq_len, head_dim] — UN-rotated cached keys
    k_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]
    # V_cache: [seq_len, head_dim]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]
    # freqs: [head_dim/2] — RoPE frequencies: 1/10000^(2i/d)
    freqs = [1.0 / (10000.0 ** (2.0 * i / head_dim)) for i in range(half_d)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    freqs_buf = runner.make_float_buffer(freqs)
    out_buf = runner.make_empty_buffer(head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    qpos_buf = runner.make_uint_buffer(q_pos)

    # Custom dispatch: single threadgroup
    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, freqs_buf, out_buf,
                              seq_buf, qpos_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Python reference: apply RoPE to Q at q_pos
    q_rot = [0.0] * head_dim
    for d in range(half_d):
        theta = q_pos * freqs[d]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        q_r = q_data[2 * d]
        q_i = q_data[2 * d + 1]
        q_rot[2 * d] = q_r * cos_t - q_i * sin_t
        q_rot[2 * d + 1] = q_r * sin_t + q_i * cos_t

    # Compute attention scores with RoPE on each K[j]
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for j in range(seq_len):
        dot = 0.0
        for d in range(half_d):
            theta = j * freqs[d]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            k_r = k_data[j * head_dim + 2 * d]
            k_i = k_data[j * head_dim + 2 * d + 1]
            k_rot_r = k_r * cos_t - k_i * sin_t
            k_rot_i = k_r * sin_t + k_i * cos_t
            dot += q_rot[2 * d] * k_rot_r + q_rot[2 * d + 1] * k_rot_i
        scores.append(dot * scale)

    # Softmax
    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    # Weighted V (V is NOT rotated)
    for d in range(head_dim):
        expected = sum(attn[j] * v_data[j * head_dim + d] for j in range(seq_len))
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Paged Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_paged_attention(runner):
    """Paged attention: KV-cache stored in non-contiguous pages with page table."""
    from triton_msl.codegen.msl_emitter import make_paged_attention_kernel
    import struct as struct_mod

    head_dim = 64
    page_size = 16
    seq_len = 48  # 3 full pages
    n_pages = 3
    # Shuffle physical pages: logical [0,1,2] -> physical [2,0,1]
    page_table = [2, 0, 1]

    msl = make_paged_attention_kernel(head_dim=head_dim, page_size=page_size)
    path = runner.compile(msl, "paged_attention")
    pipeline = runner.load(path, "paged_attention")

    random.seed(7777)

    # Q: [head_dim]
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]

    # K/V pages: [n_physical_pages, page_size, head_dim]
    # We'll create 3 physical pages
    n_physical_pages = 3
    total_page_elems = n_physical_pages * page_size * head_dim
    k_page_data = [random.uniform(-0.5, 0.5) for _ in range(total_page_elems)]
    v_page_data = [random.uniform(-0.5, 0.5) for _ in range(total_page_elems)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_page_data)
    v_buf = runner.make_float_buffer(v_page_data)

    # Page table buffer (uint array)
    import Metal
    pt_buf = runner.device.newBufferWithLength_options_(
        n_pages * 4, Metal.MTLResourceStorageModeShared
    )
    pt_view = pt_buf.contents().as_buffer(n_pages * 4)
    for i, phys in enumerate(page_table):
        struct_mod.pack_into("I", pt_view, i * 4, phys)

    out_buf = runner.make_empty_buffer(head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    np_buf = runner.make_uint_buffer(n_pages)

    # Dispatch: 1 threadgroup
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, pt_buf, out_buf,
                              sl_buf, np_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Python reference: reconstruct contiguous K,V from pages
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for pos in range(seq_len):
        page_idx = pos // page_size
        page_offset = pos % page_size
        phys_page = page_table[page_idx]
        k_base = (phys_page * page_size + page_offset) * head_dim

        dot = sum(q_data[d] * k_page_data[k_base + d] for d in range(head_dim))
        scores.append(dot * scale)

    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    for d in range(head_dim):
        expected = 0.0
        for pos in range(seq_len):
            page_idx = pos // page_size
            page_offset = pos % page_size
            phys_page = page_table[page_idx]
            v_base = (phys_page * page_size + page_offset) * head_dim
            expected += attn[pos] * v_page_data[v_base + d]
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


@requires_metal
def test_paged_attention_partial_page(runner):
    """Paged attention with a partially filled last page (seq_len not page-aligned)."""
    from triton_msl.codegen.msl_emitter import make_paged_attention_kernel
    import struct as struct_mod

    head_dim = 32
    page_size = 16
    seq_len = 25  # 1 full page + 9 tokens in second page
    n_pages = 2
    page_table = [1, 0]  # Reversed physical order

    msl = make_paged_attention_kernel(head_dim=head_dim, page_size=page_size)
    path = runner.compile(msl, "paged_attention_partial")
    pipeline = runner.load(path, "paged_attention")

    random.seed(5555)
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]
    n_physical_pages = 2
    total_elems = n_physical_pages * page_size * head_dim
    k_data = [random.uniform(-0.5, 0.5) for _ in range(total_elems)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(total_elems)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)

    import Metal
    pt_buf = runner.device.newBufferWithLength_options_(
        n_pages * 4, Metal.MTLResourceStorageModeShared
    )
    pt_view = pt_buf.contents().as_buffer(n_pages * 4)
    for i, phys in enumerate(page_table):
        struct_mod.pack_into("I", pt_view, i * 4, phys)

    out_buf = runner.make_empty_buffer(head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    np_buf = runner.make_uint_buffer(n_pages)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, pt_buf, out_buf,
                              sl_buf, np_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Reference
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for pos in range(seq_len):
        pi = pos // page_size
        po = pos % page_size
        pp = page_table[pi]
        kb = (pp * page_size + po) * head_dim
        dot = sum(q_data[d] * k_data[kb + d] for d in range(head_dim))
        scores.append(dot * scale)

    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    for d in range(head_dim):
        expected = 0.0
        for pos in range(seq_len):
            pi = pos // page_size
            po = pos % page_size
            pp = page_table[pi]
            vb = (pp * page_size + po) * head_dim
            expected += attn[pos] * v_data[vb + d]
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Speculative Decoding tests
# ---------------------------------------------------------------------------


@requires_metal
def test_speculative_decode_partial_accept(runner):
    """Speculative decoding: draft 4 tokens, target rejects at position 2."""
    from triton_msl.codegen.msl_emitter import make_speculative_decode_kernel
    import struct as struct_mod

    n_tokens = 4
    vocab_size = 32

    msl = make_speculative_decode_kernel(block_size=256)
    path = runner.compile(msl, "speculative_decode")
    pipeline = runner.load(path, "speculative_decode")

    # Build probability distributions
    # Draft and target agree on tokens 0,1; disagree on token 2
    draft_probs = []
    target_probs = []
    draft_tokens = [5, 10, 20, 15]

    for i in range(n_tokens):
        tok = draft_tokens[i]
        dp = [0.01] * vocab_size  # uniform-ish background
        tp = [0.01] * vocab_size

        if i < 2:
            # Tokens 0,1: target assigns HIGHER prob than draft → always accept
            dp[tok] = 0.5
            tp[tok] = 0.8
        elif i == 2:
            # Token 2: target assigns MUCH LOWER prob → likely reject
            dp[tok] = 0.9
            tp[tok] = 0.01
        else:
            # Token 3: would accept but won't reach here
            dp[tok] = 0.5
            tp[tok] = 0.9

        # Normalize
        total_dp = sum(dp)
        total_tp = sum(tp)
        dp = [x / total_dp for x in dp]
        tp = [x / total_tp for x in tp]
        draft_probs.extend(dp)
        target_probs.extend(tp)

    # Random values: make token 2 always rejected (rand > p/q ≈ 0.01/0.9 ≈ 0.011)
    rand_vals = [0.0, 0.0, 0.5, 0.0]  # 0.0 < 1 (accept), 0.0 < 1 (accept), 0.5 > 0.011 (reject)

    dp_buf = runner.make_float_buffer(draft_probs)
    tp_buf = runner.make_float_buffer(target_probs)

    import Metal
    dt_buf = runner.device.newBufferWithLength_options_(
        n_tokens * 4, Metal.MTLResourceStorageModeShared
    )
    dt_view = dt_buf.contents().as_buffer(n_tokens * 4)
    for i, tok in enumerate(draft_tokens):
        struct_mod.pack_into("I", dt_view, i * 4, tok)

    rv_buf = runner.make_float_buffer(rand_vals)

    na_buf = runner.device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    adj_buf = runner.make_empty_buffer(vocab_size)

    nt_buf = runner.make_uint_buffer(n_tokens)
    vs_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([dp_buf, tp_buf, dt_buf, rv_buf, na_buf, adj_buf,
                              nt_buf, vs_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    # Read n_accepted
    na_view = na_buf.contents().as_buffer(4)
    n_accepted = struct_mod.unpack_from("I", na_view, 0)[0]
    assert n_accepted == 2, f"Expected 2 accepted, got {n_accepted}"

    # Read adjusted probs — should be normalized max(0, target - draft) at position 2
    adj = runner.read_float_buffer(adj_buf, vocab_size)
    adj_sum = sum(adj)
    assert abs(adj_sum - 1.0) < 0.01, f"Adjusted probs sum to {adj_sum}, expected ~1.0"
    # All adjusted values should be >= 0
    for v in range(vocab_size):
        assert adj[v] >= -0.001, f"adj[{v}] = {adj[v]} < 0"


@requires_metal
def test_speculative_decode_all_accepted(runner):
    """Speculative decoding: all 3 draft tokens accepted."""
    from triton_msl.codegen.msl_emitter import make_speculative_decode_kernel
    import struct as struct_mod

    n_tokens = 3
    vocab_size = 16

    msl = make_speculative_decode_kernel(block_size=256)
    path = runner.compile(msl, "speculative_decode_all")
    pipeline = runner.load(path, "speculative_decode")

    # Target always assigns higher probability than draft
    draft_probs = []
    target_probs = []
    draft_tokens = [3, 7, 12]

    for i in range(n_tokens):
        tok = draft_tokens[i]
        dp = [1.0 / vocab_size] * vocab_size
        tp = [0.5 / vocab_size] * vocab_size
        dp[tok] = 0.3
        tp[tok] = 0.8  # target always higher → ratio > 1 → always accept
        total_dp = sum(dp)
        total_tp = sum(tp)
        dp = [x / total_dp for x in dp]
        tp = [x / total_tp for x in tp]
        draft_probs.extend(dp)
        target_probs.extend(tp)

    rand_vals = [0.5, 0.5, 0.5]  # All < 1.0 (min(1, p/q) = 1 since p > q)

    dp_buf = runner.make_float_buffer(draft_probs)
    tp_buf = runner.make_float_buffer(target_probs)

    import Metal
    dt_buf = runner.device.newBufferWithLength_options_(
        n_tokens * 4, Metal.MTLResourceStorageModeShared
    )
    dt_view = dt_buf.contents().as_buffer(n_tokens * 4)
    for i, tok in enumerate(draft_tokens):
        struct_mod.pack_into("I", dt_view, i * 4, tok)

    rv_buf = runner.make_float_buffer(rand_vals)
    na_buf = runner.device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    adj_buf = runner.make_empty_buffer(vocab_size)
    nt_buf = runner.make_uint_buffer(n_tokens)
    vs_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([dp_buf, tp_buf, dt_buf, rv_buf, na_buf, adj_buf,
                              nt_buf, vs_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    na_view = na_buf.contents().as_buffer(4)
    n_accepted = struct_mod.unpack_from("I", na_view, 0)[0]
    assert n_accepted == 3, f"Expected 3 accepted (all), got {n_accepted}"


# ---------------------------------------------------------------------------
# Fused Residual + Layer Norm tests
# ---------------------------------------------------------------------------


@requires_metal
def test_fused_residual_norm(runner):
    """Fused residual add + layer norm: output = LN(input + residual, gamma, beta)."""
    from triton_msl.codegen.msl_emitter import make_fused_residual_norm_kernel

    n_rows = 4
    n_cols = 64

    msl = make_fused_residual_norm_kernel(block_size=256)
    path = runner.compile(msl, "fused_residual_norm")
    pipeline = runner.load(path, "fused_residual_norm")

    random.seed(3333)
    input_data = [random.uniform(-1.0, 1.0) for _ in range(n_rows * n_cols)]
    residual_data = [random.uniform(-1.0, 1.0) for _ in range(n_rows * n_cols)]
    gamma_data = [random.uniform(0.5, 1.5) for _ in range(n_cols)]
    beta_data = [random.uniform(-0.5, 0.5) for _ in range(n_cols)]

    in_buf = runner.make_float_buffer(input_data)
    res_buf = runner.make_float_buffer(residual_data)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    res_out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, res_buf, gamma_buf, beta_buf, out_buf,
                              res_out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)
    res_out = runner.read_float_buffer(res_out_buf, n_rows * n_cols)
    eps = 1e-6

    # Python reference
    for row in range(n_rows):
        # x = input + residual
        x = [input_data[row * n_cols + i] + residual_data[row * n_cols + i]
             for i in range(n_cols)]

        # Check residual_out
        for i in range(n_cols):
            assert abs(res_out[row * n_cols + i] - x[i]) < 0.001, (
                f"res_out row {row} col {i}: got {res_out[row * n_cols + i]}, expected {x[i]}"
            )

        mean = sum(x) / n_cols
        var = sum((xi - mean) ** 2 for xi in x) / n_cols
        inv_std = 1.0 / math.sqrt(var + eps)

        for i in range(n_cols):
            expected = (x[i] - mean) * inv_std * gamma_data[i] + beta_data[i]
            got = result[row * n_cols + i]
            assert abs(got - expected) < 0.01, (
                f"row {row} col {i}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Beam Search tests
# ---------------------------------------------------------------------------


@requires_metal
def test_beam_search(runner):
    """Beam search: selects top beam_width candidates across beams * vocab."""
    from triton_msl.codegen.msl_emitter import make_beam_search_kernel
    import struct as struct_mod

    beam_width = 4
    vocab_size = 32

    msl = make_beam_search_kernel(beam_width=beam_width, block_size=256)
    path = runner.compile(msl, "beam_search")
    pipeline = runner.load(path, "beam_search")

    # Current beam scores (cumulative log-probs)
    beam_scores_data = [-1.0, -2.0, -3.0, -5.0]

    # Next-token log-probs per beam
    random.seed(4242)
    log_probs_data = [random.uniform(-10.0, -1.0)
                      for _ in range(beam_width * vocab_size)]

    # Plant some known high-scoring candidates
    # beam 0, token 5: -1.0 + -0.1 = -1.1 (best)
    log_probs_data[0 * vocab_size + 5] = -0.1
    # beam 1, token 10: -2.0 + -0.2 = -2.2 (second)
    log_probs_data[1 * vocab_size + 10] = -0.2
    # beam 0, token 20: -1.0 + -1.5 = -2.5 (third)
    log_probs_data[0 * vocab_size + 20] = -1.5
    # beam 2, token 3: -3.0 + -0.1 = -3.1 (fourth)
    log_probs_data[2 * vocab_size + 3] = -0.1

    bs_buf = runner.make_float_buffer(beam_scores_data)
    lp_buf = runner.make_float_buffer(log_probs_data)
    out_scores_buf = runner.make_empty_buffer(beam_width)

    import Metal
    out_beams_buf = runner.device.newBufferWithLength_options_(
        beam_width * 4, Metal.MTLResourceStorageModeShared
    )
    out_tokens_buf = runner.device.newBufferWithLength_options_(
        beam_width * 4, Metal.MTLResourceStorageModeShared
    )
    vs_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([bs_buf, lp_buf, out_scores_buf, out_beams_buf,
                              out_tokens_buf, vs_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result_scores = runner.read_float_buffer(out_scores_buf, beam_width)
    beams_view = out_beams_buf.contents().as_buffer(beam_width * 4)
    tokens_view = out_tokens_buf.contents().as_buffer(beam_width * 4)
    result_beams = [struct_mod.unpack_from("I", beams_view, i * 4)[0]
                    for i in range(beam_width)]
    result_tokens = [struct_mod.unpack_from("I", tokens_view, i * 4)[0]
                     for i in range(beam_width)]

    # Verify the top-4 are in descending score order
    assert result_scores[0] > result_scores[1] > result_scores[2] > result_scores[3]

    # Best candidate: beam 0, token 5, score -1.1
    assert result_beams[0] == 0
    assert result_tokens[0] == 5
    assert abs(result_scores[0] - (-1.1)) < 0.01

    # Second: beam 1, token 10, score -2.2
    assert result_beams[1] == 1
    assert result_tokens[1] == 10
    assert abs(result_scores[1] - (-2.2)) < 0.01


# ---------------------------------------------------------------------------
# Multi-Head Paged Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_multi_head_paged_attention(runner):
    """Multi-head paged attention: 2 heads, 3 pages, shuffled physical order."""
    from triton_msl.codegen.msl_emitter import make_multi_head_paged_attention_kernel
    import struct as struct_mod

    n_heads = 2
    head_dim = 32
    page_size = 8
    seq_len = 16  # 2 full pages
    n_pages = 2
    page_table = [1, 0]  # Reversed

    msl = make_multi_head_paged_attention_kernel(
        n_heads=n_heads, head_dim=head_dim, page_size=page_size)
    path = runner.compile(msl, "multi_head_paged_attn")
    pipeline = runner.load(path, "multi_head_paged_attention")

    random.seed(6161)
    n_physical_pages = 2

    # Q: [n_heads, head_dim]
    q_data = [random.uniform(-0.5, 0.5) for _ in range(n_heads * head_dim)]
    # K/V pages: [n_physical_pages, page_size, n_heads, head_dim]
    total_kv = n_physical_pages * page_size * n_heads * head_dim
    k_data = [random.uniform(-0.5, 0.5) for _ in range(total_kv)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(total_kv)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)

    import Metal
    pt_buf = runner.device.newBufferWithLength_options_(
        n_pages * 4, Metal.MTLResourceStorageModeShared
    )
    pt_view = pt_buf.contents().as_buffer(n_pages * 4)
    for i, phys in enumerate(page_table):
        struct_mod.pack_into("I", pt_view, i * 4, phys)

    out_buf = runner.make_empty_buffer(n_heads * head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    np_buf = runner.make_uint_buffer(n_pages)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, pt_buf, out_buf,
                              sl_buf, np_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_heads, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, n_heads * head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    # Python reference per head
    for h in range(n_heads):
        q_off = h * head_dim
        scores = []
        for pos in range(seq_len):
            pi = pos // page_size
            po = pos % page_size
            pp = page_table[pi]
            # K layout: [pp, po, h, d]
            k_base = ((pp * page_size + po) * n_heads + h) * head_dim
            dot = sum(q_data[q_off + d] * k_data[k_base + d] for d in range(head_dim))
            scores.append(dot * scale)

        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        attn = [e / sum_exp for e in exp_s]

        for d in range(head_dim):
            expected = 0.0
            for pos in range(seq_len):
                pi = pos // page_size
                po = pos % page_size
                pp = page_table[pi]
                v_base = ((pp * page_size + po) * n_heads + h) * head_dim
                expected += attn[pos] * v_data[v_base + d]
            got = result[q_off + d]
            assert abs(got - expected) < 0.05, (
                f"head {h} dim {d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# FP16 KV-Cache Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_fp16_kv_attention(runner):
    """FP16 KV-cache attention: Q in float32, K/V in float16, compute in float32."""
    from triton_msl.codegen.msl_emitter import make_fp16_kv_attention_kernel
    import struct as struct_mod

    head_dim = 32
    seq_len = 16

    msl = make_fp16_kv_attention_kernel(head_dim=head_dim, block_size=256)
    path = runner.compile(msl, "fp16_kv_attn")
    pipeline = runner.load(path, "fp16_kv_attention")

    random.seed(5151)
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]
    k_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]

    q_buf = runner.make_float_buffer(q_data)

    # Create half-precision buffers for K and V
    import Metal
    k_buf = runner.device.newBufferWithLength_options_(
        seq_len * head_dim * 2, Metal.MTLResourceStorageModeShared  # 2 bytes per half
    )
    v_buf = runner.device.newBufferWithLength_options_(
        seq_len * head_dim * 2, Metal.MTLResourceStorageModeShared
    )

    # Pack as half-precision (IEEE 754 binary16)
    k_view = k_buf.contents().as_buffer(seq_len * head_dim * 2)
    v_view = v_buf.contents().as_buffer(seq_len * head_dim * 2)
    for i in range(seq_len * head_dim):
        struct_mod.pack_into("e", k_view, i * 2, k_data[i])
        struct_mod.pack_into("e", v_view, i * 2, v_data[i])

    out_buf = runner.make_empty_buffer(head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Reference: use the half-precision-rounded values
    k_half = [struct_mod.unpack_from("e", k_view, i * 2)[0]
              for i in range(seq_len * head_dim)]
    v_half = [struct_mod.unpack_from("e", v_view, i * 2)[0]
              for i in range(seq_len * head_dim)]

    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for j in range(seq_len):
        dot = sum(q_data[d] * k_half[j * head_dim + d] for d in range(head_dim))
        scores.append(dot * scale)

    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    for d in range(head_dim):
        expected = sum(attn[j] * v_half[j * head_dim + d] for j in range(seq_len))
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Fused MLP (SwiGLU) tests
# ---------------------------------------------------------------------------


@requires_metal
def test_fused_mlp(runner):
    """Fused MLP: output = silu(gate) * up."""
    from triton_msl.codegen.msl_emitter import make_fused_mlp_kernel

    n = 512

    msl = make_fused_mlp_kernel(block_size=256)
    path = runner.compile(msl, "fused_mlp")
    pipeline = runner.load(path, "fused_mlp")

    random.seed(1234)
    gate_data = [random.uniform(-2.0, 2.0) for _ in range(n)]
    up_data = [random.uniform(-2.0, 2.0) for _ in range(n)]

    gate_buf = runner.make_float_buffer(gate_data)
    up_buf = runner.make_float_buffer(up_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [gate_buf, up_buf, out_buf, n_buf], n, block_size=256)

    result = runner.read_float_buffer(out_buf, n)

    for i in range(n):
        g = gate_data[i]
        silu_g = g / (1.0 + math.exp(-g))
        expected = silu_g * up_data[i]
        got = result[i]
        assert abs(got - expected) < 0.01, (
            f"idx {i}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Sliding Window Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_sliding_window_attention(runner):
    """Sliding window attention: only attend to last window_size tokens."""
    from triton_msl.codegen.msl_emitter import make_sliding_window_attention_kernel

    head_dim = 32
    window_size = 8
    seq_len = 20
    q_pos = 15  # Will attend to positions 8..15

    msl = make_sliding_window_attention_kernel(
        head_dim=head_dim, window_size=window_size, block_size=256)
    path = runner.compile(msl, "sliding_window_attn")
    pipeline = runner.load(path, "sliding_window_attention")

    random.seed(7890)
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]
    k_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    out_buf = runner.make_empty_buffer(head_dim)
    qpos_buf = runner.make_uint_buffer(q_pos)
    sl_buf = runner.make_uint_buffer(seq_len)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, qpos_buf, sl_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Reference: only attend within window
    win_start = max(0, q_pos + 1 - window_size)
    win_end = min(q_pos + 1, seq_len)
    scale = 1.0 / math.sqrt(head_dim)

    scores = []
    for j in range(win_start, win_end):
        dot = sum(q_data[d] * k_data[j * head_dim + d] for d in range(head_dim))
        scores.append(dot * scale)

    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    for d in range(head_dim):
        expected = sum(attn[w] * v_data[(win_start + w) * head_dim + d]
                       for w in range(len(attn)))
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Repeat KV tests
# ---------------------------------------------------------------------------


@requires_metal
def test_repeat_kv(runner):
    """Repeat KV: expand n_kv_heads to n_q_heads by repeating."""
    from triton_msl.codegen.msl_emitter import make_repeat_kv_kernel
    import struct as struct_mod

    n_kv_heads = 2
    n_rep = 4
    n_q_heads = n_kv_heads * n_rep  # = 8
    seq_len = 4
    head_dim = 8

    msl = make_repeat_kv_kernel(block_size=256)
    path = runner.compile(msl, "repeat_kv")
    pipeline = runner.load(path, "repeat_kv")

    # input: [n_kv_heads, seq_len, head_dim]
    random.seed(2020)
    total_in = n_kv_heads * seq_len * head_dim
    in_data = [random.uniform(-1.0, 1.0) for _ in range(total_in)]

    total_out = n_q_heads * seq_len * head_dim
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(total_out)
    nkvh_buf = runner.make_uint_buffer(n_kv_heads)
    sl_buf = runner.make_uint_buffer(seq_len)
    hd_buf = runner.make_uint_buffer(head_dim)
    nr_buf = runner.make_uint_buffer(n_rep)

    block_size = 256
    n_groups = (total_out + block_size - 1) // block_size

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, nkvh_buf, sl_buf, hd_buf, nr_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, total_out)

    # Reference: output[q_head, s, d] = input[q_head // n_rep, s, d]
    for qh in range(n_q_heads):
        kv_h = qh // n_rep
        for s in range(seq_len):
            for d in range(head_dim):
                out_idx = (qh * seq_len + s) * head_dim + d
                in_idx = (kv_h * seq_len + s) * head_dim + d
                expected = in_data[in_idx]
                got = result[out_idx]
                assert abs(got - expected) < 0.001, (
                    f"q_head={qh} s={s} d={d}: got {got}, expected {expected}"
                )


# ---------------------------------------------------------------------------
# Edge case and error handling tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add_single_element(runner):
    """Vector add with n=1 (single element, partial threadgroup)."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_buf = runner.make_float_buffer([3.14])
    b_buf = runner.make_float_buffer([2.71])
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(1)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], 1, 256)
    result = runner.read_float_buffer(out_buf, 1)
    assert abs(result[0] - 5.85) < 0.001


@requires_metal
def test_vector_add_non_power_of_2(runner):
    """Vector add with n=137 (non-power-of-2, tests masking)."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 137
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) for i in range(n)]
    b_data = [float(i) * 0.5 for i in range(n)]
    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n, 256)
    result = runner.read_float_buffer(out_buf, n)

    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 0.01, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_silu_large_input(runner):
    """SiLU with 100K elements (multi-threadgroup)."""
    from triton_msl.codegen.msl_emitter import make_silu_kernel

    n = 100_000
    msl = make_silu_kernel(block_size=256)
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    # Use values from a range that exercises both positive and negative
    in_data = [float(i % 200 - 100) * 0.05 for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], n, 256)
    result = runner.read_float_buffer(out_buf, n)

    # Check first and last elements plus a sample
    for i in [0, 1, n // 2, n - 2, n - 1]:
        x = in_data[i]
        expected = x / (1.0 + math.exp(-x))
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_reduce_sum_non_power_of_2(runner):
    """Sum reduction with n=1000 (not a power of 2)."""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    n = 1000
    msl = make_reduce_kernel("reduce_sum", "sum", block_size=256)
    path = runner.compile(msl, "reduce_sum")
    pipeline = runner.load(path, "reduce_sum")

    # Sum of 1..1000 = 500500
    in_data = [float(i + 1) for i in range(n)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], 1, 256)
    result = runner.read_float_buffer(out_buf, 1)
    assert abs(result[0] - 500500.0) < 1.0, f"Sum: {result[0]}"


@requires_metal
def test_softmax_single_row(runner):
    """Softmax with a single row (n_rows=1)."""
    from triton_msl.codegen.msl_emitter import make_softmax_kernel

    n_cols = 32
    msl = make_softmax_kernel(block_size=256)
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    in_data = [float(i) * 0.3 for i in range(n_cols)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    runner.run(pipeline, [in_buf, out_buf, ncols_buf], 1, 256)
    result = runner.read_float_buffer(out_buf, n_cols)

    # Should sum to 1
    row_sum = sum(result)
    assert abs(row_sum - 1.0) < 0.001, f"Sum: {row_sum}"
    # All positive
    assert all(v >= 0 for v in result)


@requires_metal
def test_softmax_uniform_input(runner):
    """Softmax with uniform input → all outputs should be equal (1/n)."""
    from triton_msl.codegen.msl_emitter import make_softmax_kernel

    n_cols = 64
    msl = make_softmax_kernel(block_size=256)
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    in_data = [1.0] * n_cols  # all same value
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    runner.run(pipeline, [in_buf, out_buf, ncols_buf], 1, 256)
    result = runner.read_float_buffer(out_buf, n_cols)

    expected = 1.0 / n_cols
    for i, v in enumerate(result):
        assert abs(v - expected) < 0.001, f"i={i}: {v} != {expected}"


@requires_metal
def test_layer_norm_zero_variance(runner):
    """Layer norm with zero variance → output should be beta (all same input)."""
    from triton_msl.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 64
    msl = make_layer_norm_kernel(block_size=256)
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    # All same value → variance = 0, output = (0)*gamma + beta = beta
    in_data = [5.0] * n_cols
    gamma_data = [1.0] * n_cols
    beta_data = [0.5] * n_cols

    in_buf = runner.make_float_buffer(in_data)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    runner.run(pipeline, [in_buf, gamma_buf, beta_buf, out_buf, ncols_buf],
                        1, 256)
    result = runner.read_float_buffer(out_buf, n_cols)

    # With zero variance, normalized x = 0 (since x-mean=0), so output = 0*gamma + beta = beta
    for i, v in enumerate(result):
        assert abs(v - 0.5) < 0.01, f"i={i}: {v} != 0.5"


@requires_metal
def test_variance_kernel(runner):
    """Variance kernel computes row-wise variance."""
    from triton_msl.codegen.msl_emitter import make_variance_kernel

    n_rows, n_cols = 4, 64
    msl = make_variance_kernel(block_size=256)
    path = runner.compile(msl, "variance_kernel")
    pipeline = runner.load(path, "variance_kernel")

    # Row 0: all same → variance = 0
    # Row 1: [0,1,2,...,63] → mean=31.5, var=mean((x-31.5)^2)
    # Row 2: all 1s → variance = 0
    # Row 3: alternating ±1 → mean=0, var=1
    in_data = []
    in_data.extend([5.0] * n_cols)  # row 0
    in_data.extend([float(i) for i in range(n_cols)])  # row 1
    in_data.extend([1.0] * n_cols)  # row 2
    in_data.extend([1.0 if i % 2 == 0 else -1.0 for i in range(n_cols)])  # row 3

    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n_rows)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows)

    # Row 0: all same → var = 0
    assert abs(result[0]) < 0.01, f"Row 0 var: {result[0]}"
    # Row 1: var of [0..63]
    mean_1 = sum(range(n_cols)) / n_cols
    expected_var_1 = sum((x - mean_1) ** 2 for x in range(n_cols)) / n_cols
    assert abs(result[1] - expected_var_1) < 1.0, f"Row 1 var: {result[1]} != {expected_var_1}"
    # Row 2: all same → var = 0
    assert abs(result[2]) < 0.01, f"Row 2 var: {result[2]}"
    # Row 3: ±1 alternating → var = 1.0
    assert abs(result[3] - 1.0) < 0.01, f"Row 3 var: {result[3]}"


@requires_metal
def test_fp64_downcast():
    """FP64 types should be mapped to fp64 (downcast to float in MSL)."""
    from triton_msl.codegen.ttgir_parser import _mlir_type_to_triton_dtype

    result = _mlir_type_to_triton_dtype("f64")
    assert result == "fp64", f"Expected 'fp64', got '{result}'"


@requires_metal
def test_kernel_builder_empty():
    """KernelBuilder with no ops produces valid (empty) kernel."""
    from triton_msl.codegen.msl_emitter import KernelBuilder

    kb = KernelBuilder("empty_test", block_size=256)
    kb.add_ptr_arg("input", dtype="fp32", const=True)
    kb.add_ptr_arg("output", dtype="fp32", const=False)
    kb.add_scalar_arg("n", dtype="u32")

    msl = kb.build()
    assert "empty_test" in msl
    assert "device const float" in msl
    assert "device float" in msl


# ---------------------------------------------------------------------------
# Batch normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_batch_norm_kernel(runner):
    """Batch norm (eval mode) with pre-computed running stats."""
    from triton_msl.codegen.msl_emitter import make_batch_norm_kernel

    C = 4  # channels
    HW = 16  # spatial size
    total = C * HW

    msl = make_batch_norm_kernel(block_size=256)
    path = runner.compile(msl, "batch_norm_kernel")
    pipeline = runner.load(path, "batch_norm_kernel")

    # Input: channels interleaved with spatial
    in_data = [float(i) * 0.1 for i in range(total)]
    gamma_data = [1.0] * C
    beta_data = [0.0] * C
    mean_data = [float(c) * 0.1 * (HW // 2) for c in range(C)]  # approx means
    var_data = [1.0] * C  # unit variance for simplicity

    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(total)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    mean_buf = runner.make_float_buffer(mean_data)
    var_buf = runner.make_float_buffer(var_data)
    nc_buf = runner.make_uint_buffer(C)
    hw_buf = runner.make_uint_buffer(HW)

    runner.run(pipeline,
               [in_buf, out_buf, gamma_buf, beta_buf, mean_buf, var_buf, nc_buf, hw_buf],
               total, 256)
    result = runner.read_float_buffer(out_buf, total)

    # Verify: output = gamma * (input - mean) / sqrt(var + eps) + beta
    eps = 1e-5
    for idx in range(total):
        channel = (idx // HW) % C
        x = in_data[idx]
        expected = gamma_data[channel] * (x - mean_data[channel]) / math.sqrt(var_data[channel] + eps) + beta_data[channel]
        assert abs(result[idx] - expected) < 0.01, f"idx={idx}: {result[idx]} != {expected}"


@requires_metal
def test_batch_norm_with_affine(runner):
    """Batch norm with non-trivial gamma/beta."""
    from triton_msl.codegen.msl_emitter import make_batch_norm_kernel

    C = 2
    HW = 8
    total = C * HW

    msl = make_batch_norm_kernel(block_size=256)
    path = runner.compile(msl, "batch_norm_kernel")
    pipeline = runner.load(path, "batch_norm_kernel")

    in_data = [1.0] * total  # all ones
    gamma_data = [2.0, 0.5]
    beta_data = [1.0, -1.0]
    mean_data = [1.0, 1.0]  # mean matches input → normalized = 0
    var_data = [1.0, 1.0]

    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(total)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    mean_buf = runner.make_float_buffer(mean_data)
    var_buf = runner.make_float_buffer(var_data)
    nc_buf = runner.make_uint_buffer(C)
    hw_buf = runner.make_uint_buffer(HW)

    runner.run(pipeline,
               [in_buf, out_buf, gamma_buf, beta_buf, mean_buf, var_buf, nc_buf, hw_buf],
               total, 256)
    result = runner.read_float_buffer(out_buf, total)

    # input=1, mean=1 → normalized=0 → output = gamma*0 + beta = beta
    for idx in range(total):
        channel = (idx // HW) % C
        expected = beta_data[channel]
        assert abs(result[idx] - expected) < 0.01, f"idx={idx}: {result[idx]} != {expected}"


# ---------------------------------------------------------------------------
# Online softmax tests
# ---------------------------------------------------------------------------

@requires_metal
def test_online_softmax(runner):
    """Online (single-pass) softmax produces correct results."""
    from triton_msl.codegen.msl_emitter import make_online_softmax_kernel

    n_cols = 64
    msl = make_online_softmax_kernel(block_size=256)
    path = runner.compile(msl, "online_softmax_kernel")
    pipeline = runner.load(path, "online_softmax_kernel")

    in_data = [float(i) * 0.2 for i in range(n_cols)]
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    # Verify: sums to 1 and matches reference softmax
    row_sum = sum(result)
    assert abs(row_sum - 1.0) < 0.001, f"Sum: {row_sum}"

    max_val = max(in_data)
    exp_vals = [math.exp(x - max_val) for x in in_data]
    exp_sum = sum(exp_vals)
    for i in [0, 16, 32, 48, 63]:
        expected = exp_vals[i] / exp_sum
        assert abs(result[i] - expected) < 0.001, f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_online_softmax_uniform(runner):
    """Online softmax with uniform input → all equal (1/n)."""
    from triton_msl.codegen.msl_emitter import make_online_softmax_kernel

    n_cols = 128
    msl = make_online_softmax_kernel(block_size=256)
    path = runner.compile(msl, "online_softmax_kernel")
    pipeline = runner.load(path, "online_softmax_kernel")

    in_data = [1.0] * n_cols
    in_buf = runner.make_float_buffer(in_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    expected = 1.0 / n_cols
    for i, v in enumerate(result):
        assert abs(v - expected) < 0.001, f"i={i}: {v} != {expected}"


# ---------------------------------------------------------------------------
# Causal attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_causal_attention_compiles(runner):
    """Causal attention kernel compiles successfully."""
    from triton_msl.codegen.msl_emitter import make_causal_attention_kernel

    msl = make_causal_attention_kernel(n_heads=4, head_dim=32, block_size=128)
    runner.compile(msl, "causal_attention")


# ---------------------------------------------------------------------------
# Group normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_group_norm_kernel(runner):
    """Group norm produces correct results for uniform input."""
    from triton_msl.codegen.msl_emitter import make_group_norm_kernel
    import Metal

    n_groups = 2
    channels = 4
    spatial = 4
    n = channels * spatial  # 16 elements per batch item
    eps = 1e-5

    msl = make_group_norm_kernel(n_groups=n_groups, block_size=256, eps=eps)
    path = runner.compile(msl, "group_norm_kernel")
    pipeline = runner.load(path, "group_norm_kernel")

    # Input: all 1.0 → mean=1.0, var=0.0, normalized=0.0, output = 0*weight+bias = bias
    inp_buf = runner.make_float_buffer([1.0] * n)
    weight_buf = runner.make_float_buffer([2.0] * channels)
    bias_buf = runner.make_float_buffer([0.5] * channels)
    out_buf = runner.make_empty_buffer(n)
    n_channels_buf = runner.make_uint_buffer(channels)
    spatial_buf = runner.make_uint_buffer(spatial)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(n_channels_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(spatial_buf, 0, 5)
    # 1 batch * n_groups = 2 threadgroups
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    # Uniform input → normalized = 0.0, output = 0*weight + bias = 0.5
    for i, v in enumerate(result):
        assert abs(v - 0.5) < 0.01, f"i={i}: {v} != 0.5"


@requires_metal
def test_group_norm_varying_input(runner):
    """Group norm normalizes varying input correctly."""
    from triton_msl.codegen.msl_emitter import make_group_norm_kernel
    import Metal

    n_groups = 1
    channels = 2
    spatial = 2
    n = channels * spatial  # 4 elements
    eps = 1e-5

    msl = make_group_norm_kernel(n_groups=n_groups, block_size=256, eps=eps)
    path = runner.compile(msl, "group_norm_kernel")
    pipeline = runner.load(path, "group_norm_kernel")

    # Input: [1, 2, 3, 4] in a single group
    data = [1.0, 2.0, 3.0, 4.0]
    inp_buf = runner.make_float_buffer(data)
    weight_buf = runner.make_float_buffer([1.0] * channels)
    bias_buf = runner.make_float_buffer([0.0] * channels)
    out_buf = runner.make_empty_buffer(n)
    n_channels_buf = runner.make_uint_buffer(channels)
    spatial_buf = runner.make_uint_buffer(spatial)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(n_channels_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(spatial_buf, 0, 5)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    # mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25+eps)
    import math
    mean = 2.5
    var = sum((x - mean) ** 2 for x in data) / len(data)
    inv_std = 1.0 / math.sqrt(var + eps)
    for i in range(n):
        expected = (data[i] - mean) * inv_std
        assert abs(result[i] - expected) < 0.01, f"i={i}: {result[i]} != {expected}"


# ---------------------------------------------------------------------------
# Instance normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_instance_norm_uniform(runner):
    """Instance norm produces correct results for uniform input."""
    from triton_msl.codegen.msl_emitter import make_instance_norm_kernel
    import Metal

    spatial = 8
    n_channels = 2
    n = n_channels * spatial
    eps = 1e-5

    msl = make_instance_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "instance_norm_kernel")
    pipeline = runner.load(path, "instance_norm_kernel")

    # Uniform input → mean=1.0, var=0.0, output = 0*weight+bias = bias
    inp_buf = runner.make_float_buffer([1.0] * n)
    weight_buf = runner.make_float_buffer([2.0] * n_channels)
    bias_buf = runner.make_float_buffer([0.5] * n_channels)
    out_buf = runner.make_empty_buffer(n)
    spatial_buf = runner.make_uint_buffer(spatial)
    nchan_buf = runner.make_uint_buffer(n_channels)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(spatial_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(nchan_buf, 0, 5)
    # n_channels threadgroups (one per channel instance)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_channels, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    for i, v in enumerate(result):
        assert abs(v - 0.5) < 0.01, f"i={i}: {v} != 0.5"


@requires_metal
def test_instance_norm_varying(runner):
    """Instance norm normalizes varying input per channel."""
    from triton_msl.codegen.msl_emitter import make_instance_norm_kernel
    import Metal
    import math

    spatial = 4
    n_channels = 1
    n = n_channels * spatial
    eps = 1e-5

    msl = make_instance_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "instance_norm_kernel")
    pipeline = runner.load(path, "instance_norm_kernel")

    data = [1.0, 3.0, 5.0, 7.0]
    inp_buf = runner.make_float_buffer(data)
    weight_buf = runner.make_float_buffer([1.0])
    bias_buf = runner.make_float_buffer([0.0])
    out_buf = runner.make_empty_buffer(n)
    spatial_buf = runner.make_uint_buffer(spatial)
    nchan_buf = runner.make_uint_buffer(n_channels)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(spatial_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(nchan_buf, 0, 5)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_channels, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    mean = sum(data) / len(data)
    var = sum((x - mean) ** 2 for x in data) / len(data)
    inv_std = 1.0 / math.sqrt(var + eps)
    for i in range(n):
        expected = (data[i] - mean) * inv_std
        assert abs(result[i] - expected) < 0.01, f"i={i}: {result[i]} != {expected}"


# ---------------------------------------------------------------------------
# Fused dropout tests
# ---------------------------------------------------------------------------

@requires_metal
def test_fused_dropout_compiles(runner):
    """Fused dropout kernel compiles successfully."""
    from triton_msl.codegen.msl_emitter import make_fused_dropout_kernel

    msl = make_fused_dropout_kernel(block_size=256, p=0.5)
    runner.compile(msl, "fused_dropout_kernel")


@requires_metal
def test_fused_dropout_output(runner):
    """Fused dropout zeros ~50% of elements and scales the rest."""
    from triton_msl.codegen.msl_emitter import make_fused_dropout_kernel
    import Metal

    n = 4096
    p = 0.5

    msl = make_fused_dropout_kernel(block_size=256, p=p)
    path = runner.compile(msl, "fused_dropout_kernel")
    pipeline = runner.load(path, "fused_dropout_kernel")

    inp_buf = runner.make_float_buffer([1.0] * n)
    out_buf = runner.make_empty_buffer(n)
    mask_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)
    seed_buf = runner.make_uint_buffer(42)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(mask_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(n_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(seed_buf, 0, 4)
    n_groups = (n + 255) // 256
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    mask = runner.read_float_buffer(mask_buf, n)

    # Check that some elements are zeroed and some are scaled
    n_zeros = sum(1 for v in result if abs(v) < 1e-6)
    n_nonzero = n - n_zeros
    # With p=0.5, expect roughly 50% zeros (allow wide margin for randomness)
    assert 0.2 * n < n_zeros < 0.8 * n, f"Expected ~50% zeros, got {n_zeros}/{n}"
    # Non-zero elements should be scaled by 1/(1-p) = 2.0
    for i in range(n):
        if mask[i] > 0.5:
            assert abs(result[i] - 2.0) < 0.01, f"i={i}: scaled={result[i]} != 2.0"


# ---------------------------------------------------------------------------
# FP16 softmax tests
# ---------------------------------------------------------------------------

@requires_metal
def test_softmax_fp16(runner):
    """FP16 softmax produces valid probability distributions."""
    from triton_msl.codegen.msl_emitter import make_softmax_kernel
    import Metal

    n_rows = 2
    n_cols = 8
    n = n_rows * n_cols

    msl = make_softmax_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    # Simple input: row 0 = [0,1,2,...,7], row 1 = [1,1,1,...,1]
    data = list(range(n_cols)) + [1.0] * n_cols
    inp_buf = runner.make_half_buffer(data)
    out_buf = runner.make_empty_half_buffer(n)
    ncols_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(ncols_buf, 0, 2)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_half_buffer(out_buf, n)

    # Row 0: softmax of [0,1,...,7] — sum should be ~1.0
    row0 = result[:n_cols]
    row0_sum = sum(row0)
    assert abs(row0_sum - 1.0) < 0.05, f"Row 0 sum = {row0_sum}, expected ~1.0"
    assert all(v >= 0 for v in row0), "All softmax values should be non-negative"
    # Last element should be largest
    assert row0[-1] > row0[0], f"row0[-1]={row0[-1]} should be > row0[0]={row0[0]}"

    # Row 1: uniform input → uniform output (each = 1/8 = 0.125)
    row1 = result[n_cols:]
    for i, v in enumerate(row1):
        assert abs(v - 0.125) < 0.02, f"row1[{i}] = {v}, expected ~0.125"


# ---------------------------------------------------------------------------
# FP16 matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_matmul_fp16_identity(runner):
    """FP16 matmul with identity matrix produces correct result."""
    from triton_msl.codegen.msl_emitter import make_matmul_kernel
    import Metal

    M, N, K = 16, 16, 16

    msl = make_matmul_kernel(block_m=16, block_n=16, block_k=16, dtype="fp16")
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    # A = identity matrix, B = [1, 2, ..., 16] repeated as rows
    a_data = []
    for i in range(M):
        for j in range(K):
            a_data.append(1.0 if i == j else 0.0)

    b_data = list(range(1, N + 1)) * K  # Each row is [1..N]

    a_buf = runner.make_half_buffer(a_data)
    b_buf = runner.make_half_buffer(b_data)
    c_buf = runner.make_empty_half_buffer(M * N)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(c_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(m_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(n_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(k_buf, 0, 5)
    n_tiles = ((M + 15) // 16) * ((N + 15) // 16)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_tiles, 1, 1),
        Metal.MTLSizeMake(16 * 16, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_half_buffer(c_buf, M * N)

    # C = I @ B = B → each row should be [1..N]
    for row in range(M):
        for col in range(N):
            expected = float(col + 1)
            actual = result[row * N + col]
            assert abs(actual - expected) < 0.5, \
                f"C[{row},{col}] = {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add_bf16(runner):
    """BF16 vector add produces correct results."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    n = 256
    msl = make_vector_add_kernel(block_size=256, dtype="bf16")
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) for i in range(n)]
    b_data = [float(i * 2) for i in range(n)]
    a_buf = runner.make_bf16_buffer(a_data)
    b_buf = runner.make_bf16_buffer(b_data)
    out_buf = runner.make_empty_bf16_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n, 256)
    result = runner.read_bf16_buffer(out_buf, n)

    for i in range(n):
        expected = a_data[i] + b_data[i]
        # BF16 has ~3 decimal digits precision; tolerance scales with magnitude
        tol = max(2.0, abs(expected) * 0.02)
        assert abs(result[i] - expected) < tol, \
            f"i={i}: {result[i]} != {expected}"


@requires_metal
def test_activation_tanh_bf16(runner):
    """BF16 tanh activation compiles and runs."""
    from triton_msl.codegen.msl_emitter import make_activation_kernel

    n = 256
    msl = make_activation_kernel(activation="tanh", block_size=256, dtype="bf16")
    path = runner.compile(msl, "tanh_kernel")
    pipeline = runner.load(path, "tanh_kernel")

    import math
    data = [0.0, 0.5, 1.0, -1.0] * 64
    inp_buf = runner.make_bf16_buffer(data)
    out_buf = runner.make_empty_bf16_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [inp_buf, out_buf, n_buf], n, 256)
    result = runner.read_bf16_buffer(out_buf, n)

    for i in range(min(4, n)):
        expected = math.tanh(data[i])
        assert abs(result[i] - expected) < 0.05, \
            f"i={i}: {result[i]} != {expected}"


# ---------------------------------------------------------------------------
# Gather kernel tests
# ---------------------------------------------------------------------------

@requires_metal
def test_gather_kernel(runner):
    """Gather kernel reads from indexed positions."""
    from triton_msl.codegen.msl_emitter import make_gather_kernel
    import Metal
    import struct

    n = 4
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    indices = [4, 0, 2, 1]

    msl = make_gather_kernel(block_size=256)
    path = runner.compile(msl, "gather_kernel")
    pipeline = runner.load(path, "gather_kernel")

    inp_buf = runner.make_float_buffer(data)
    # Make int buffer for indices
    idx_buf = runner.device.newBufferWithLength_options_(
        len(indices) * 4, Metal.MTLResourceStorageModeShared
    )
    view = idx_buf.contents().as_buffer(len(indices) * 4)
    for i, idx in enumerate(indices):
        struct.pack_into("i", view, i * 4, idx)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(idx_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(n_buf, 0, 3)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    expected = [data[i] for i in indices]
    for i in range(n):
        assert abs(result[i] - expected[i]) < 0.01, f"i={i}: {result[i]} != {expected[i]}"


# ---------------------------------------------------------------------------
# Scatter kernel tests
# ---------------------------------------------------------------------------

@requires_metal
def test_scatter_kernel(runner):
    """Scatter kernel writes to indexed positions."""
    from triton_msl.codegen.msl_emitter import make_scatter_kernel
    import Metal
    import struct

    n = 4
    data = [100.0, 200.0, 300.0, 400.0]
    indices = [3, 1, 0, 2]

    msl = make_scatter_kernel(block_size=256)
    path = runner.compile(msl, "scatter_kernel")
    pipeline = runner.load(path, "scatter_kernel")

    inp_buf = runner.make_float_buffer(data)
    idx_buf = runner.device.newBufferWithLength_options_(
        len(indices) * 4, Metal.MTLResourceStorageModeShared
    )
    view = idx_buf.contents().as_buffer(len(indices) * 4)
    for i, idx in enumerate(indices):
        struct.pack_into("i", view, i * 4, idx)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(idx_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(n_buf, 0, 3)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)
    # output[indices[i]] = data[i], so output[3]=100, output[1]=200, output[0]=300, output[2]=400
    expected = [0.0] * n
    for i in range(n):
        expected[indices[i]] = data[i]
    for i in range(n):
        assert abs(result[i] - expected[i]) < 0.01, f"i={i}: {result[i]} != {expected[i]}"


# ---------------------------------------------------------------------------
# Transpose kernel tests
# ---------------------------------------------------------------------------

@requires_metal
def test_transpose_kernel(runner):
    """Transpose kernel correctly transposes a small matrix."""
    from triton_msl.codegen.msl_emitter import make_transpose_kernel
    import Metal

    rows, cols = 4, 8
    n = rows * cols
    tile_size = 16

    msl = make_transpose_kernel(tile_size=tile_size)
    path = runner.compile(msl, "transpose_kernel")
    pipeline = runner.load(path, "transpose_kernel")

    # Input: row-major matrix where element = row * cols + col
    data = [float(i) for i in range(n)]
    inp_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n)
    rows_buf = runner.make_uint_buffer(rows)
    cols_buf = runner.make_uint_buffer(cols)

    n_tiles_x = (cols + tile_size - 1) // tile_size
    n_tiles_y = (rows + tile_size - 1) // tile_size

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoderWithDescriptor_(
        Metal.MTLComputePassDescriptor.computePassDescriptor()
    )
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(inp_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(rows_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(cols_buf, 0, 3)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_tiles_x, n_tiles_y, 1),
        Metal.MTLSizeMake(tile_size * tile_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n)

    # Verify: output[col * rows + row] should equal input[row * cols + col]
    for r in range(rows):
        for c in range(cols):
            expected = data[r * cols + c]
            actual = result[c * rows + r]
            assert abs(actual - expected) < 0.01, \
                f"T[{c},{r}] = {actual}, expected {expected} (from [{r},{c}])"


# ---------------------------------------------------------------------------
# Reduce-scatter kernel
# ---------------------------------------------------------------------------

@requires_metal
def test_reduce_scatter_kernel(runner):
    """Reduce-scatter sums 2 input buffers and writes to output."""
    from triton_msl.codegen.msl_emitter import make_reduce_scatter_kernel

    n = 1024
    msl = make_reduce_scatter_kernel(n_buffers=2, block_size=256)
    path = runner.compile(msl, "reduce_scatter")
    pipeline = runner.load(path, "reduce_scatter")

    a = [float(i) for i in range(n)]
    b = [float(i * 2) for i in range(n)]

    buf_a = runner.make_float_buffer(a)
    buf_b = runner.make_float_buffer(b)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_uint_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_out, buf_n], n)
    result = runner.read_float_buffer(buf_out, n)

    for i in range(n):
        expected = a[i] + b[i]
        assert abs(result[i] - expected) < 0.01, \
            f"reduce_scatter[{i}] = {result[i]}, expected {expected}"


@requires_metal
def test_reduce_scatter_3_buffers(runner):
    """Reduce-scatter with 3 input buffers."""
    from triton_msl.codegen.msl_emitter import make_reduce_scatter_kernel

    n = 512
    msl = make_reduce_scatter_kernel(n_buffers=3, block_size=256)
    path = runner.compile(msl, "reduce_scatter")
    pipeline = runner.load(path, "reduce_scatter")

    a = [float(i) for i in range(n)]
    b = [float(i * 2) for i in range(n)]
    c = [float(i * 3) for i in range(n)]

    buf_a = runner.make_float_buffer(a)
    buf_b = runner.make_float_buffer(b)
    buf_c = runner.make_float_buffer(c)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_uint_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_c, buf_out, buf_n], n)
    result = runner.read_float_buffer(buf_out, n)

    for i in range(n):
        expected = a[i] + b[i] + c[i]
        assert abs(result[i] - expected) < 0.01, \
            f"reduce_scatter_3[{i}] = {result[i]}, expected {expected}"


# ---------------------------------------------------------------------------
# All-reduce kernel
# ---------------------------------------------------------------------------

@requires_metal
def test_all_reduce_sum(runner):
    """All-reduce (sum) of 2 input buffers."""
    from triton_msl.codegen.msl_emitter import make_all_reduce_kernel

    n = 1024
    msl = make_all_reduce_kernel(n_buffers=2, op="sum")
    path = runner.compile(msl, "all_reduce")
    pipeline = runner.load(path, "all_reduce")

    a = [float(i) * 0.5 for i in range(n)]
    b = [float(i) * 1.5 for i in range(n)]

    buf_a = runner.make_float_buffer(a)
    buf_b = runner.make_float_buffer(b)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_uint_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_out, buf_n], n)
    result = runner.read_float_buffer(buf_out, n)

    for i in range(n):
        expected = a[i] + b[i]
        assert abs(result[i] - expected) < 0.01, \
            f"all_reduce_sum[{i}] = {result[i]}, expected {expected}"


@requires_metal
def test_all_reduce_max(runner):
    """All-reduce (max) of 2 input buffers."""
    from triton_msl.codegen.msl_emitter import make_all_reduce_kernel

    n = 512
    msl = make_all_reduce_kernel(n_buffers=2, op="max")
    path = runner.compile(msl, "all_reduce")
    pipeline = runner.load(path, "all_reduce")

    random.seed(42)
    a = [random.uniform(-10, 10) for _ in range(n)]
    b = [random.uniform(-10, 10) for _ in range(n)]

    buf_a = runner.make_float_buffer(a)
    buf_b = runner.make_float_buffer(b)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_uint_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_out, buf_n], n)
    result = runner.read_float_buffer(buf_out, n)

    for i in range(n):
        expected = max(a[i], b[i])
        assert abs(result[i] - expected) < 0.01, \
            f"all_reduce_max[{i}] = {result[i]}, expected {expected}"


# ---------------------------------------------------------------------------
# FP16 layer norm
# ---------------------------------------------------------------------------

@requires_metal
def test_layer_norm_fp16(runner):
    """Layer normalization in half precision."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 128
    n_rows = 4
    msl = make_layer_norm_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    data = [float(i % n_cols) * 0.01 - 0.5 for i in range(n_rows * n_cols)]
    gamma = [1.0] * n_cols
    beta = [0.0] * n_cols

    in_buf = runner.make_half_buffer(data)
    gamma_buf = runner.make_half_buffer(gamma)
    beta_buf = runner.make_half_buffer(beta)
    out_buf = runner.make_empty_half_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, gamma_buf, beta_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_half_buffer(out_buf, n_rows * n_cols)

    for row in range(n_rows):
        row_vals = result[row * n_cols : (row + 1) * n_cols]
        row_mean = sum(row_vals) / n_cols
        assert abs(row_mean) < 0.2, f"row {row} mean = {row_mean}, expected ≈ 0"


# ---------------------------------------------------------------------------
# FP16 RMS norm
# ---------------------------------------------------------------------------

@requires_metal
def test_rms_norm_fp16(runner):
    """RMS normalization in half precision."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_rms_norm_kernel

    n_cols = 64
    n_rows = 4
    msl = make_rms_norm_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "rms_norm_kernel")
    pipeline = runner.load(path, "rms_norm_kernel")

    data = [float(i % n_cols) * 0.02 + 0.1 for i in range(n_rows * n_cols)]
    weight = [1.0] * n_cols

    in_buf = runner.make_half_buffer(data)
    w_buf = runner.make_half_buffer(weight)
    out_buf = runner.make_empty_half_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, w_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_half_buffer(out_buf, n_rows * n_cols)

    for row in range(n_rows):
        row_vals = result[row * n_cols : (row + 1) * n_cols]
        rms = math.sqrt(sum(v * v for v in row_vals) / n_cols)
        assert 0.5 < rms < 2.0, f"row {row} RMS = {rms}, expected ≈ 1"


# ---------------------------------------------------------------------------
# FP16 reduction
# ---------------------------------------------------------------------------

@requires_metal
def test_reduce_sum_fp16(runner):
    """Sum reduction in half precision."""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    n = 256
    msl = make_reduce_kernel("reduce_sum", op="sum", block_size=256, dtype="fp16")
    path = runner.compile(msl, "reduce_sum")
    pipeline = runner.load(path, "reduce_sum")

    # Small values to avoid overflow
    data = [float(i) * 0.01 for i in range(n)]
    in_buf = runner.make_half_buffer(data)
    out_buf = runner.make_empty_half_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], 1, block_size=256)
    result = runner.read_half_buffer(out_buf, 1)

    expected = sum(data)
    tol = max(1.0, abs(expected) * 0.05)  # 5% tolerance for FP16 reduction
    assert abs(result[0] - expected) < tol, \
        f"reduce_sum_fp16 = {result[0]}, expected {expected}"


# ---------------------------------------------------------------------------
# FP16 RoPE
# ---------------------------------------------------------------------------

@requires_metal
def test_rope_fp16(runner):
    """RoPE (rotary position embedding) in half precision."""
    from triton_msl.codegen.msl_emitter import make_rope_kernel

    dim = 32
    msl = make_rope_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "rope_kernel")
    pipeline = runner.load(path, "rope_kernel")

    # Simple input: all ones
    data = [1.0] * dim
    # Frequencies: 1/(10000^(2i/dim)) — use simple freqs for testing
    freqs = [1.0 / (10000.0 ** (2.0 * i / dim)) for i in range(dim // 2)]

    in_buf = runner.make_half_buffer(data)
    freq_buf = runner.make_half_buffer(freqs)
    out_buf = runner.make_empty_half_buffer(dim)
    dim_buf = runner.make_uint_buffer(dim)
    pos_buf = runner.make_uint_buffer(0)  # position 0

    runner.run(pipeline, [in_buf, freq_buf, out_buf, dim_buf, pos_buf],
               1, block_size=256)
    result = runner.read_half_buffer(out_buf, dim)

    # At position 0, theta = 0 for all freqs, so cos(0)=1, sin(0)=0
    # out[2i] = x0*1 - x1*0 = x0 = 1.0
    # out[2i+1] = x0*0 + x1*1 = x1 = 1.0
    for i in range(dim):
        assert abs(result[i] - 1.0) < 0.05, \
            f"rope_fp16[{i}] = {result[i]}, expected 1.0 (at position 0)"


# ---------------------------------------------------------------------------
# BF16 reduction
# ---------------------------------------------------------------------------

@requires_metal
def test_reduce_sum_bf16(runner):
    """Sum reduction in BF16 precision."""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    n = 64  # Small to avoid precision issues
    msl = make_reduce_kernel("reduce_sum", op="sum", block_size=256, dtype="bf16")
    path = runner.compile(msl, "reduce_sum")
    pipeline = runner.load(path, "reduce_sum")

    data = [float(i) * 0.5 for i in range(n)]
    in_buf = runner.make_bf16_buffer(data)
    out_buf = runner.make_empty_bf16_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, n_buf], 1, block_size=256)
    result = runner.read_bf16_buffer(out_buf, 1)

    expected = sum(data)
    tol = max(2.0, abs(expected) * 0.05)
    assert abs(result[0] - expected) < tol, \
        f"reduce_sum_bf16 = {result[0]}, expected {expected}"


# ---------------------------------------------------------------------------
# BF16 layer norm
# ---------------------------------------------------------------------------

@requires_metal
def test_layer_norm_bf16(runner):
    """Layer normalization in BF16 precision."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 64
    n_rows = 2
    msl = make_layer_norm_kernel(block_size=256, dtype="bf16")
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    data = [float(i % n_cols) * 0.1 - 3.0 for i in range(n_rows * n_cols)]
    gamma = [1.0] * n_cols
    beta = [0.0] * n_cols

    in_buf = runner.make_bf16_buffer(data)
    gamma_buf = runner.make_bf16_buffer(gamma)
    beta_buf = runner.make_bf16_buffer(beta)
    out_buf = runner.make_empty_bf16_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, gamma_buf, beta_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_bf16_buffer(out_buf, n_rows * n_cols)

    for row in range(n_rows):
        row_vals = result[row * n_cols : (row + 1) * n_cols]
        row_mean = sum(row_vals) / n_cols
        assert abs(row_mean) < 0.5, f"row {row} mean = {row_mean}, expected ≈ 0"


# ---------------------------------------------------------------------------
# Attention kernel GPU tests
# ---------------------------------------------------------------------------


def _ref_softmax(scores):
    """Reference softmax on a list of floats."""
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _ref_dot(a, b):
    """Reference dot product of two lists."""
    return sum(x * y for x, y in zip(a, b))


@requires_metal
def test_causal_attention_gpu(runner):
    """Causal attention kernel: validate softmax(Q@K^T * mask) @ V."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_causal_attention_kernel

    n_heads = 1
    head_dim = 8
    seq_len = 4
    block_size = 256

    random.seed(42)
    total = n_heads * seq_len * head_dim
    Q = [random.gauss(0, 0.5) for _ in range(total)]
    K = [random.gauss(0, 0.5) for _ in range(total)]
    V = [random.gauss(0, 0.5) for _ in range(total)]

    # Reference computation
    scale = 1.0 / math.sqrt(head_dim)
    expected = [0.0] * total
    for h in range(n_heads):
        for qp in range(seq_len):
            # Compute scores with causal mask
            scores = []
            for kp in range(seq_len):
                if kp <= qp:
                    qi = h * seq_len * head_dim + qp * head_dim
                    ki = h * seq_len * head_dim + kp * head_dim
                    dot = _ref_dot(Q[qi:qi+head_dim], K[ki:ki+head_dim])
                    scores.append(dot * scale)
                else:
                    scores.append(float('-inf'))
            weights = _ref_softmax(scores)
            for d in range(head_dim):
                val = 0.0
                for kp in range(seq_len):
                    vi = h * seq_len * head_dim + kp * head_dim + d
                    val += weights[kp] * V[vi]
                oi = h * seq_len * head_dim + qp * head_dim + d
                expected[oi] = val

    msl = make_causal_attention_kernel(n_heads=n_heads, head_dim=head_dim,
                                        block_size=block_size)
    path = runner.compile(msl, "causal_attention")
    pipeline = runner.load(path, "causal_attention")

    q_buf = runner.make_float_buffer(Q)
    k_buf = runner.make_float_buffer(K)
    v_buf = runner.make_float_buffer(V)
    out_buf = runner.make_empty_buffer(total)
    # seq_len is passed as a device buffer (uint*) for this kernel
    sl_buf = runner.make_uint_buffer(seq_len)

    n_threadgroups = n_heads * seq_len
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_threadgroups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, total)
    for i in range(total):
        assert abs(result[i] - expected[i]) < 0.05, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_flash_attention_gpu(runner):
    """Flash attention kernel: validate FlashAttention-2 with online softmax."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

    head_dim = 8
    seq_len = 8
    n_heads = 1
    Br, Bc = 4, 4
    block_size = 256

    random.seed(123)
    total = n_heads * seq_len * head_dim
    Q = [random.gauss(0, 0.3) for _ in range(total)]
    K = [random.gauss(0, 0.3) for _ in range(total)]
    V = [random.gauss(0, 0.3) for _ in range(total)]

    # Reference: standard attention (non-causal)
    scale = 1.0 / math.sqrt(head_dim)
    expected = [0.0] * total
    for h in range(n_heads):
        for qp in range(seq_len):
            scores = []
            for kp in range(seq_len):
                qi = h * seq_len * head_dim + qp * head_dim
                ki = h * seq_len * head_dim + kp * head_dim
                dot = _ref_dot(Q[qi:qi+head_dim], K[ki:ki+head_dim])
                scores.append(dot * scale)
            weights = _ref_softmax(scores)
            for d in range(head_dim):
                val = 0.0
                for kp in range(seq_len):
                    vi = h * seq_len * head_dim + kp * head_dim + d
                    val += weights[kp] * V[vi]
                oi = h * seq_len * head_dim + qp * head_dim + d
                expected[oi] = val

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=Br, Bc=Bc,
                                       block_size=block_size, causal=False)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    q_buf = runner.make_float_buffer(Q)
    k_buf = runner.make_float_buffer(K)
    v_buf = runner.make_float_buffer(V)
    out_buf = runner.make_empty_buffer(total)
    sl_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + Br - 1) // Br
    n_threadgroups = n_heads * n_q_blocks

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_threadgroups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, total)
    for i in range(total):
        assert abs(result[i] - expected[i]) < 0.1, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_kv_cache_attention_gpu(runner):
    """KV-cache attention: single query token attending to cached K,V."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_kv_cache_attention_kernel

    head_dim = 8
    seq_len = 4
    n_heads = 2
    block_size = 256

    random.seed(77)
    Q = [random.gauss(0, 0.5) for _ in range(n_heads * head_dim)]
    K = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    # Reference computation
    scale = 1.0 / math.sqrt(head_dim)
    expected = [0.0] * (n_heads * head_dim)
    for h in range(n_heads):
        scores = []
        for j in range(seq_len):
            qi = h * head_dim
            ki = h * seq_len * head_dim + j * head_dim
            dot = _ref_dot(Q[qi:qi+head_dim], K[ki:ki+head_dim])
            scores.append(dot * scale)
        weights = _ref_softmax(scores)
        for d in range(head_dim):
            val = 0.0
            for j in range(seq_len):
                vi = h * seq_len * head_dim + j * head_dim + d
                val += weights[j] * V[vi]
            expected[h * head_dim + d] = val

    msl = make_kv_cache_attention_kernel(head_dim=head_dim, block_size=block_size)
    path = runner.compile(msl, "kv_cache_attention")
    pipeline = runner.load(path, "kv_cache_attention")

    q_buf = runner.make_float_buffer(Q)
    k_buf = runner.make_float_buffer(K)
    v_buf = runner.make_float_buffer(V)
    out_buf = runner.make_empty_buffer(n_heads * head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_heads, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_heads * head_dim)
    for i in range(n_heads * head_dim):
        assert abs(result[i] - expected[i]) < 0.05, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_gqa_attention_gpu(runner):
    """Grouped query attention: multiple Q heads share fewer KV heads."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_gqa_attention_kernel

    head_dim = 8
    seq_len = 4
    n_q_per_kv = 2
    n_kv_heads = 2
    n_q_heads = n_kv_heads * n_q_per_kv  # = 4
    block_size = 256

    random.seed(99)
    Q = [random.gauss(0, 0.5) for _ in range(n_q_heads * head_dim)]
    K = [random.gauss(0, 0.5) for _ in range(n_kv_heads * seq_len * head_dim)]
    V = [random.gauss(0, 0.5) for _ in range(n_kv_heads * seq_len * head_dim)]

    # Reference computation
    scale = 1.0 / math.sqrt(head_dim)
    expected = [0.0] * (n_q_heads * head_dim)
    for qh in range(n_q_heads):
        kv_head = qh // n_q_per_kv
        scores = []
        for j in range(seq_len):
            qi = qh * head_dim
            ki = kv_head * seq_len * head_dim + j * head_dim
            dot = _ref_dot(Q[qi:qi+head_dim], K[ki:ki+head_dim])
            scores.append(dot * scale)
        weights = _ref_softmax(scores)
        for d in range(head_dim):
            val = 0.0
            for j in range(seq_len):
                vi = kv_head * seq_len * head_dim + j * head_dim + d
                val += weights[j] * V[vi]
            expected[qh * head_dim + d] = val

    msl = make_gqa_attention_kernel(head_dim=head_dim, n_q_per_kv=n_q_per_kv,
                                     block_size=block_size)
    path = runner.compile(msl, "gqa_attention")
    pipeline = runner.load(path, "gqa_attention")

    q_buf = runner.make_float_buffer(Q)
    k_buf = runner.make_float_buffer(K)
    v_buf = runner.make_float_buffer(V)
    out_buf = runner.make_empty_buffer(n_q_heads * head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_q_heads, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_q_heads * head_dim)
    for i in range(n_q_heads * head_dim):
        assert abs(result[i] - expected[i]) < 0.05, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_paged_attention_gpu(runner):
    """Paged attention: KV-cache stored in fixed-size pages with page table."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_paged_attention_kernel

    head_dim = 8
    page_size = 4
    seq_len = 8  # 2 pages
    n_pages = (seq_len + page_size - 1) // page_size  # = 2
    block_size = 256

    random.seed(55)
    Q = [random.gauss(0, 0.5) for _ in range(head_dim)]
    # Pages: [n_physical_pages, page_size, head_dim]
    # Use identity page table (logical == physical)
    K_pages = [random.gauss(0, 0.5) for _ in range(n_pages * page_size * head_dim)]
    V_pages = [random.gauss(0, 0.5) for _ in range(n_pages * page_size * head_dim)]
    page_table = list(range(n_pages))  # identity mapping

    # Reference: flatten pages to contiguous, compute standard attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for j in range(seq_len):
        page_idx = j // page_size
        within_page = j % page_size
        phys_page = page_table[page_idx]
        ki = phys_page * page_size * head_dim + within_page * head_dim
        dot = _ref_dot(Q, K_pages[ki:ki+head_dim])
        scores.append(dot * scale)
    weights = _ref_softmax(scores)

    expected = [0.0] * head_dim
    for d in range(head_dim):
        for j in range(seq_len):
            page_idx = j // page_size
            within_page = j % page_size
            phys_page = page_table[page_idx]
            vi = phys_page * page_size * head_dim + within_page * head_dim + d
            expected[d] += weights[j] * V_pages[vi]

    msl = make_paged_attention_kernel(head_dim=head_dim, page_size=page_size,
                                       block_size=block_size)
    path = runner.compile(msl, "paged_attention")
    pipeline = runner.load(path, "paged_attention")

    q_buf = runner.make_float_buffer(Q)
    kp_buf = runner.make_float_buffer(K_pages)
    vp_buf = runner.make_float_buffer(V_pages)
    # page_table is uint array
    pt_buf = runner.make_float_buffer([0.0] * n_pages)  # placeholder, will pack uints
    import struct as _struct
    pt_view = pt_buf.contents().as_buffer(n_pages * 4)
    for i, p in enumerate(page_table):
        _struct.pack_into("I", pt_view, i * 4, p)
    out_buf = runner.make_empty_buffer(head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    np_buf = runner.make_uint_buffer(n_pages)

    # Single threadgroup dispatch
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, kp_buf, vp_buf, pt_buf, out_buf, sl_buf, np_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, head_dim)
    for d in range(head_dim):
        assert abs(result[d] - expected[d]) < 0.1, \
            f"dim {d}: got {result[d]}, expected {expected[d]}"


@requires_metal
def test_sliding_window_attention_gpu(runner):
    """Sliding window attention: attend only to last window_size tokens."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_sliding_window_attention_kernel

    head_dim = 8
    window_size = 4
    seq_len = 8
    q_pos = 6  # attends to positions 3,4,5,6
    block_size = 256

    random.seed(33)
    Q = [random.gauss(0, 0.5) for _ in range(head_dim)]
    K_cache = [random.gauss(0, 0.5) for _ in range(seq_len * head_dim)]
    V_cache = [random.gauss(0, 0.5) for _ in range(seq_len * head_dim)]

    # Reference
    scale = 1.0 / math.sqrt(head_dim)
    win_start = max(0, q_pos - window_size + 1)
    win_end = min(q_pos + 1, seq_len)
    scores = []
    for j in range(win_start, win_end):
        ki = j * head_dim
        dot = _ref_dot(Q, K_cache[ki:ki+head_dim])
        scores.append(dot * scale)
    weights = _ref_softmax(scores)

    expected = [0.0] * head_dim
    for idx, j in enumerate(range(win_start, win_end)):
        for d in range(head_dim):
            expected[d] += weights[idx] * V_cache[j * head_dim + d]

    msl = make_sliding_window_attention_kernel(head_dim=head_dim,
                                                window_size=window_size,
                                                block_size=block_size)
    path = runner.compile(msl, "sliding_window_attention")
    pipeline = runner.load(path, "sliding_window_attention")

    q_buf = runner.make_float_buffer(Q)
    k_buf = runner.make_float_buffer(K_cache)
    v_buf = runner.make_float_buffer(V_cache)
    out_buf = runner.make_empty_buffer(head_dim)
    qp_buf = runner.make_uint_buffer(q_pos)
    sl_buf = runner.make_uint_buffer(seq_len)

    # Single threadgroup dispatch
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, qp_buf, sl_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, head_dim)
    for d in range(head_dim):
        assert abs(result[d] - expected[d]) < 0.1, \
            f"dim {d}: got {result[d]}, expected {expected[d]}"


# ---------------------------------------------------------------------------
# Matmul and quantized kernel GPU tests
# ---------------------------------------------------------------------------


def _ref_matmul(A, B, M, N, K):
    """Reference matmul: C[M,N] = A[M,K] @ B[K,N]. All flat lists."""
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i * K + k] * B[k * N + j]
            C[i * N + j] = acc
    return C


@requires_metal
def test_matmul_gpu(runner):
    """Tiled matmul kernel: C = A @ B validated against reference."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 8, 8, 8
    block_m, block_n, block_k = 8, 8, 8

    random.seed(42)
    A = [random.gauss(0, 1) for _ in range(M * K)]
    B = [random.gauss(0, 1) for _ in range(K * N)]
    expected = _ref_matmul(A, B, M, N, K)

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    a_buf = runner.make_float_buffer(A)
    b_buf = runner.make_float_buffer(B)
    c_buf = runner.make_empty_buffer(M * N)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    block_size = block_m * block_n
    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_threadgroups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([a_buf, b_buf, c_buf, m_buf, n_buf, k_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_threadgroups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(c_buf, M * N)
    for i in range(M * N):
        assert abs(result[i] - expected[i]) < 0.01, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_matmul_non_square_gpu(runner):
    """Matmul with non-square dimensions: M=4, N=8, K=16."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 4, 8, 16
    block_m, block_n, block_k = 8, 8, 8

    random.seed(101)
    A = [random.gauss(0, 0.5) for _ in range(M * K)]
    B = [random.gauss(0, 0.5) for _ in range(K * N)]
    expected = _ref_matmul(A, B, M, N, K)

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    a_buf = runner.make_float_buffer(A)
    b_buf = runner.make_float_buffer(B)
    c_buf = runner.make_empty_buffer(M * N)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    block_size = block_m * block_n
    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_threadgroups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([a_buf, b_buf, c_buf, m_buf, n_buf, k_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_threadgroups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(c_buf, M * N)
    for i in range(M * N):
        assert abs(result[i] - expected[i]) < 0.05, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_simdgroup_matmul_gpu(runner):
    """simdgroup_matrix matmul: hardware-accelerated 8x8 MMA tiles."""
    import Metal
    from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

    # simdgroup matmul requires M,N multiples of 32
    M, N, K = 32, 32, 32

    random.seed(88)
    A = [random.gauss(0, 0.5) for _ in range(M * K)]
    B = [random.gauss(0, 0.5) for _ in range(K * N)]
    expected = _ref_matmul(A, B, M, N, K)

    msl = make_simdgroup_matmul_kernel(dtype="fp32")
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    a_buf = runner.make_float_buffer(A)
    b_buf = runner.make_float_buffer(B)
    c_buf = runner.make_empty_buffer(M * N)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    # simdgroup_matmul uses block_size=128, 32x32 tiles
    block_size = 128
    n_tile_cols = (N + 31) // 32
    n_tile_rows = (M + 31) // 32
    n_threadgroups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([a_buf, b_buf, c_buf, m_buf, n_buf, k_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_threadgroups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(c_buf, M * N)
    for i in range(M * N):
        assert abs(result[i] - expected[i]) < 0.1, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_int8_matmul_gpu(runner):
    """INT8 quantized matmul: float input @ dequant(int8 weight)."""
    import Metal
    import struct as _struct
    from triton_msl.codegen.msl_emitter import make_int8_matmul_kernel

    M, N, K = 2, 4, 8

    random.seed(66)
    input_data = [random.gauss(0, 1) for _ in range(M * K)]
    # INT8 weights: values in [-128, 127]
    weight_int8 = [random.randint(-5, 5) for _ in range(N * K)]
    # Per-row scale and zero point
    scales = [random.uniform(0.1, 0.5) for _ in range(N)]
    zeros = [random.uniform(-1, 1) for _ in range(N)]

    # Reference: dequantize and matmul
    expected = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                w_float = (float(weight_int8[j * K + k]) - zeros[j]) * scales[j]
                acc += input_data[i * K + k] * w_float
            expected[i * N + j] = acc

    msl = make_int8_matmul_kernel()
    path = runner.compile(msl, "int8_matmul")
    pipeline = runner.load(path, "int8_matmul")

    in_buf = runner.make_float_buffer(input_data)
    # Pack weight as signed chars
    w_buf = runner.device.newBufferWithLength_options_(
        N * K, Metal.MTLResourceStorageModeShared
    )
    w_view = w_buf.contents().as_buffer(N * K)
    for i, v in enumerate(weight_int8):
        _struct.pack_into("b", w_view, i, v)
    out_buf = runner.make_empty_buffer(M * N)
    s_buf = runner.make_float_buffer(scales)
    z_buf = runner.make_float_buffer(zeros)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    n_elements = M * N
    runner.run(pipeline, [in_buf, w_buf, out_buf, s_buf, z_buf,
                          m_buf, n_buf, k_buf], n_elements, block_size=256)

    result = runner.read_float_buffer(out_buf, M * N)
    for i in range(M * N):
        assert abs(result[i] - expected[i]) < 0.01, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_int4_matmul_gpu(runner):
    """INT4 quantized matmul: float input @ dequant(int4 weight)."""
    import Metal
    import struct as _struct
    from triton_msl.codegen.msl_emitter import make_int4_matmul_kernel

    M, N, K = 2, 4, 8
    group_size = 8  # all K elements in one group for simplicity

    random.seed(44)
    input_data = [random.gauss(0, 1) for _ in range(M * K)]
    # INT4 weights: values 0-15 (unsigned 4-bit)
    weight_int4 = [random.randint(0, 15) for _ in range(N * K)]
    n_groups = (K + group_size - 1) // group_size
    scales = [random.uniform(0.1, 0.5) for _ in range(N * n_groups)]
    zeros = [random.uniform(0, 8) for _ in range(N * n_groups)]

    # Reference: dequantize and matmul
    expected = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                g = k // group_size
                s = scales[j * n_groups + g]
                z = zeros[j * n_groups + g]
                w = (float(weight_int4[j * K + k]) - z) * s
                acc += input_data[i * K + k] * w
            expected[i * N + j] = acc

    msl = make_int4_matmul_kernel(group_size=group_size)
    path = runner.compile(msl, "int4_matmul")
    pipeline = runner.load(path, "int4_matmul")

    in_buf = runner.make_float_buffer(input_data)
    # Pack as pairs of int4 into bytes
    n_bytes = N * (K // 2)
    w_buf = runner.device.newBufferWithLength_options_(
        n_bytes, Metal.MTLResourceStorageModeShared
    )
    w_view = w_buf.contents().as_buffer(n_bytes)
    for j in range(N):
        for k in range(0, K, 2):
            lo = weight_int4[j * K + k] & 0x0F
            hi = weight_int4[j * K + k + 1] & 0x0F
            packed = lo | (hi << 4)
            byte_idx = j * (K // 2) + k // 2
            _struct.pack_into("B", w_view, byte_idx, packed)
    out_buf = runner.make_empty_buffer(M * N)
    s_buf = runner.make_float_buffer(scales)
    z_buf = runner.make_float_buffer(zeros)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    n_elements = M * N
    runner.run(pipeline, [in_buf, w_buf, out_buf, s_buf, z_buf,
                          m_buf, n_buf, k_buf], n_elements, block_size=256)

    result = runner.read_float_buffer(out_buf, M * N)
    for i in range(M * N):
        assert abs(result[i] - expected[i]) < 0.1, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


# ---------------------------------------------------------------------------
# 2D axis-aware reduction tests
# ---------------------------------------------------------------------------

@requires_metal
def test_row_reduce_sum_gpu(runner):
    """Row-wise sum: output[row] = sum(input[row, :])"""
    from triton_msl.codegen.msl_emitter import make_row_reduce_kernel
    import Metal

    n_rows, n_cols = 8, 64
    msl = make_row_reduce_kernel("row_sum", "sum", block_size=256)
    path = runner.compile(msl, "row_sum")
    pipeline = runner.load(path, "row_sum")

    random.seed(99)
    data = [random.uniform(-10.0, 10.0) for _ in range(n_rows * n_cols)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n_rows)
    nr_buf = runner.make_uint_buffer(n_rows)
    nc_buf = runner.make_uint_buffer(n_cols)

    # Dispatch one threadgroup per row
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, nr_buf, nc_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    result = runner.read_float_buffer(out_buf, n_rows)
    for r in range(n_rows):
        expected = sum(data[r * n_cols:(r + 1) * n_cols])
        assert abs(result[r] - expected) < 0.5, \
            f"row {r}: got {result[r]}, expected {expected}"


@requires_metal
def test_row_reduce_max_gpu(runner):
    """Row-wise max: output[row] = max(input[row, :])"""
    from triton_msl.codegen.msl_emitter import make_row_reduce_kernel
    import Metal

    n_rows, n_cols = 4, 128
    msl = make_row_reduce_kernel("row_max", "max", block_size=256)
    path = runner.compile(msl, "row_max")
    pipeline = runner.load(path, "row_max")

    random.seed(100)
    data = [random.uniform(-50.0, 50.0) for _ in range(n_rows * n_cols)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n_rows)
    nr_buf = runner.make_uint_buffer(n_rows)
    nc_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, nr_buf, nc_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    result = runner.read_float_buffer(out_buf, n_rows)
    for r in range(n_rows):
        expected = max(data[r * n_cols:(r + 1) * n_cols])
        assert abs(result[r] - expected) < 1e-3, \
            f"row {r}: got {result[r]}, expected {expected}"


@requires_metal
def test_col_reduce_sum_gpu(runner):
    """Column-wise sum: output[col] = sum(input[:, col])"""
    from triton_msl.codegen.msl_emitter import make_col_reduce_kernel
    import Metal

    n_rows, n_cols = 32, 16
    msl = make_col_reduce_kernel("col_sum", "sum", block_size=256)
    path = runner.compile(msl, "col_sum")
    pipeline = runner.load(path, "col_sum")

    random.seed(101)
    data = [random.uniform(-10.0, 10.0) for _ in range(n_rows * n_cols)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n_cols)
    nr_buf = runner.make_uint_buffer(n_rows)
    nc_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, nr_buf, nc_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_cols, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    result = runner.read_float_buffer(out_buf, n_cols)
    for c in range(n_cols):
        expected = sum(data[r * n_cols + c] for r in range(n_rows))
        assert abs(result[c] - expected) < 0.5, \
            f"col {c}: got {result[c]}, expected {expected}"


@requires_metal
def test_col_reduce_min_gpu(runner):
    """Column-wise min: output[col] = min(input[:, col])"""
    from triton_msl.codegen.msl_emitter import make_col_reduce_kernel
    import Metal

    n_rows, n_cols = 16, 8
    msl = make_col_reduce_kernel("col_min", "min", block_size=256)
    path = runner.compile(msl, "col_min")
    pipeline = runner.load(path, "col_min")

    random.seed(102)
    data = [random.uniform(-100.0, 100.0) for _ in range(n_rows * n_cols)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n_cols)
    nr_buf = runner.make_uint_buffer(n_rows)
    nc_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, nr_buf, nc_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_cols, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    result = runner.read_float_buffer(out_buf, n_cols)
    for c in range(n_cols):
        expected = min(data[r * n_cols + c] for r in range(n_rows))
        assert abs(result[c] - expected) < 1e-3, \
            f"col {c}: got {result[c]}, expected {expected}"


# ---------------------------------------------------------------------------
# Cumulative sum (prefix sum) tests
# ---------------------------------------------------------------------------

@requires_metal
def test_cumsum_gpu(runner):
    """Inclusive prefix sum: output[i] = sum(input[0:i+1])"""
    from triton_msl.codegen.msl_emitter import make_cumsum_kernel
    import Metal

    n_rows, n_cols = 4, 64
    msl = make_cumsum_kernel(block_size=256)
    path = runner.compile(msl, "cumsum_kernel")
    pipeline = runner.load(path, "cumsum_kernel")

    random.seed(200)
    data = [random.uniform(-5.0, 5.0) for _ in range(n_rows * n_cols)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    nc_buf = runner.make_uint_buffer(n_cols)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([in_buf, out_buf, nc_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)
    for r in range(n_rows):
        running = 0.0
        for c in range(n_cols):
            running += data[r * n_cols + c]
            got = result[r * n_cols + c]
            assert abs(got - running) < 0.1, \
                f"row {r} col {c}: got {got}, expected {running}"


# ---------------------------------------------------------------------------
# Bitonic sort tests
# ---------------------------------------------------------------------------

@requires_metal
def test_bitonic_sort_gpu(runner):
    """Bitonic sort: verify sorted values and indices"""
    from triton_msl.codegen.msl_emitter import make_bitonic_sort_kernel
    import struct

    n = 256
    msl = make_bitonic_sort_kernel(block_size=256, ascending=True)
    path = runner.compile(msl, "bitonic_sort_kernel")
    pipeline = runner.load(path, "bitonic_sort_kernel")

    random.seed(201)
    data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    in_buf = runner.make_float_buffer(data)
    val_buf = runner.make_empty_buffer(n)
    # Index buffer: uint32 * n
    idx_buf = runner.device.newBufferWithLength_options_(n * 4, 0)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, val_buf, idx_buf, n_buf], n, block_size=256)

    vals = runner.read_float_buffer(val_buf, n)
    # Read uint indices
    idx_view = idx_buf.contents().as_buffer(n * 4)
    indices = list(struct.unpack_from(f"{n}I", idx_view, 0))

    # Verify sorted
    for i in range(n - 1):
        assert vals[i] <= vals[i + 1], \
            f"Not sorted at {i}: {vals[i]} > {vals[i+1]}"

    # Verify indices point back to original values
    for i in range(n):
        assert abs(vals[i] - data[indices[i]]) < 1e-5, \
            f"Index mismatch at {i}: vals[{i}]={vals[i]}, data[{indices[i]}]={data[indices[i]]}"


# ---------------------------------------------------------------------------
# Atomic operations tests
# ---------------------------------------------------------------------------

@requires_metal
def test_atomic_add_gpu(runner):
    """Atomic add: scatter-add input values to output by indices"""
    from triton_msl.codegen.msl_emitter import make_atomic_add_kernel
    import struct

    n = 256
    n_bins = 4
    msl = make_atomic_add_kernel(block_size=256)
    path = runner.compile(msl, "atomic_add_kernel")
    pipeline = runner.load(path, "atomic_add_kernel")

    # All values are 1.0, indices round-robin across 4 bins
    data = [1.0] * n
    indices = [i % n_bins for i in range(n)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_float_buffer([0.0] * n_bins)
    # Pack uint indices
    idx_buf = runner.device.newBufferWithLength_options_(n * 4, 0)
    idx_view = idx_buf.contents().as_buffer(n * 4)
    for i, idx in enumerate(indices):
        struct.pack_into("I", idx_view, i * 4, idx)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, idx_buf, n_buf], n, block_size=256)

    result = runner.read_float_buffer(out_buf, n_bins)
    # Each bin should have n/n_bins = 64 additions of 1.0
    expected_per_bin = n // n_bins
    for i in range(n_bins):
        assert abs(result[i] - expected_per_bin) < 1.0, \
            f"bin {i}: got {result[i]}, expected {expected_per_bin}"


# ---------------------------------------------------------------------------
# Conv2D tests
# ---------------------------------------------------------------------------

@requires_metal
def test_conv2d_gpu(runner):
    """Conv2D: 1x1x4x4 input, 1 output channel, 3x3 filter, pad=1"""
    from triton_msl.codegen.msl_emitter import make_conv2d_kernel

    in_c, out_c = 1, 1
    kh, kw = 3, 3
    in_h, in_w = 4, 4
    out_h, out_w = 4, 4  # same padding
    batch = 1

    msl = make_conv2d_kernel(in_channels=in_c, out_channels=out_c,
                             kernel_h=kh, kernel_w=kw,
                             stride_h=1, stride_w=1,
                             pad_h=1, pad_w=1, block_size=256)
    path = runner.compile(msl, "conv2d_kernel")
    pipeline = runner.load(path, "conv2d_kernel")

    # Simple input: all 1.0
    input_data = [1.0] * (batch * in_c * in_h * in_w)
    # Filter: all 1.0 (sum of 3x3 = 9 for interior, less for edges)
    weight_data = [1.0] * (out_c * in_c * kh * kw)
    bias_data = [0.0] * out_c

    in_buf = runner.make_float_buffer(input_data)
    w_buf = runner.make_float_buffer(weight_data)
    b_buf = runner.make_float_buffer(bias_data)
    out_buf = runner.make_empty_buffer(batch * out_c * out_h * out_w)
    batch_buf = runner.make_uint_buffer(batch)
    ih_buf = runner.make_uint_buffer(in_h)
    iw_buf = runner.make_uint_buffer(in_w)
    oh_buf = runner.make_uint_buffer(out_h)
    ow_buf = runner.make_uint_buffer(out_w)

    n_elements = batch * out_c * out_h * out_w
    runner.run(pipeline, [in_buf, w_buf, b_buf, out_buf,
                          batch_buf, ih_buf, iw_buf, oh_buf, ow_buf],
               n_elements, block_size=256)

    result = runner.read_float_buffer(out_buf, n_elements)

    # Reference: manually compute conv2d with all-1 filter, all-1 input, pad=1
    # Each output pixel = count of valid neighbors in 3x3 window
    expected = []
    for oh_idx in range(out_h):
        for ow_idx in range(out_w):
            val = 0.0
            for kh_idx in range(kh):
                for kw_idx in range(kw):
                    ih_idx = oh_idx + kh_idx - 1  # pad=1
                    iw_idx = ow_idx + kw_idx - 1
                    if 0 <= ih_idx < in_h and 0 <= iw_idx < in_w:
                        val += 1.0
            expected.append(val)

    for i in range(n_elements):
        assert abs(result[i] - expected[i]) < 1e-3, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


# ---------------------------------------------------------------------------
# Pooling tests
# ---------------------------------------------------------------------------

@requires_metal
def test_max_pool2d_gpu(runner):
    """Max pool 2x2 stride 2: 1x1x4x4 → 1x1x2x2"""
    from triton_msl.codegen.msl_emitter import make_max_pool2d_kernel

    batch, channels = 1, 1
    in_h, in_w = 4, 4
    out_h, out_w = 2, 2  # 2x2 pool, stride 2, no pad

    msl = make_max_pool2d_kernel(kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)
    path = runner.compile(msl, "max_pool2d_kernel")
    pipeline = runner.load(path, "max_pool2d_kernel")

    # Input: 0..15
    data = [float(i) for i in range(16)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(batch * channels * out_h * out_w)
    b_buf = runner.make_uint_buffer(batch)
    c_buf = runner.make_uint_buffer(channels)
    ih_buf = runner.make_uint_buffer(in_h)
    iw_buf = runner.make_uint_buffer(in_w)
    oh_buf = runner.make_uint_buffer(out_h)
    ow_buf = runner.make_uint_buffer(out_w)

    n_out = batch * channels * out_h * out_w
    runner.run(pipeline, [in_buf, out_buf, b_buf, c_buf, ih_buf, iw_buf,
                          oh_buf, ow_buf], n_out, block_size=256)

    result = runner.read_float_buffer(out_buf, n_out)
    # Expected: max of each 2x2 block
    # [0,1,2,3; 4,5,6,7; 8,9,10,11; 12,13,14,15] → max(0,1,4,5)=5, max(2,3,6,7)=7,
    # max(8,9,12,13)=13, max(10,11,14,15)=15
    expected = [5.0, 7.0, 13.0, 15.0]
    for i in range(n_out):
        assert abs(result[i] - expected[i]) < 1e-3, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


@requires_metal
def test_avg_pool2d_gpu(runner):
    """Avg pool 2x2 stride 2: 1x1x4x4 → 1x1x2x2"""
    from triton_msl.codegen.msl_emitter import make_avg_pool2d_kernel

    batch, channels = 1, 1
    in_h, in_w = 4, 4
    out_h, out_w = 2, 2

    msl = make_avg_pool2d_kernel(kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)
    path = runner.compile(msl, "avg_pool2d_kernel")
    pipeline = runner.load(path, "avg_pool2d_kernel")

    data = [float(i) for i in range(16)]
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(batch * channels * out_h * out_w)
    b_buf = runner.make_uint_buffer(batch)
    c_buf = runner.make_uint_buffer(channels)
    ih_buf = runner.make_uint_buffer(in_h)
    iw_buf = runner.make_uint_buffer(in_w)
    oh_buf = runner.make_uint_buffer(out_h)
    ow_buf = runner.make_uint_buffer(out_w)

    n_out = batch * channels * out_h * out_w
    runner.run(pipeline, [in_buf, out_buf, b_buf, c_buf, ih_buf, iw_buf,
                          oh_buf, ow_buf], n_out, block_size=256)

    result = runner.read_float_buffer(out_buf, n_out)
    # avg(0,1,4,5)=2.5, avg(2,3,6,7)=4.5, avg(8,9,12,13)=10.5, avg(10,11,14,15)=12.5
    expected = [2.5, 4.5, 10.5, 12.5]
    for i in range(n_out):
        assert abs(result[i] - expected[i]) < 1e-3, \
            f"idx {i}: got {result[i]}, expected {expected[i]}"


# ---------------------------------------------------------------------------
# Index select and where tests
# ---------------------------------------------------------------------------

@requires_metal
def test_index_select_gpu(runner):
    """index_select: gather elements by index"""
    from triton_msl.codegen.msl_emitter import make_index_select_kernel
    import struct

    n = 64
    msl = make_index_select_kernel(block_size=256)
    path = runner.compile(msl, "index_select_kernel")
    pipeline = runner.load(path, "index_select_kernel")

    # Source data: 0..255
    src_data = [float(i) * 0.5 for i in range(256)]
    # Indices: reverse order 63..0
    indices = list(range(n - 1, -1, -1))

    in_buf = runner.make_float_buffer(src_data)
    idx_buf = runner.device.newBufferWithLength_options_(n * 4, 0)
    idx_view = idx_buf.contents().as_buffer(n * 4)
    for i, idx in enumerate(indices):
        struct.pack_into("I", idx_view, i * 4, idx)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, idx_buf, out_buf, n_buf], n, block_size=256)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = src_data[indices[i]]
        assert abs(result[i] - expected) < 1e-5, \
            f"idx {i}: got {result[i]}, expected {expected}"


@requires_metal
def test_where_gpu(runner):
    """where: conditional select between two tensors"""
    from triton_msl.codegen.msl_emitter import make_where_kernel
    import struct

    n = 256
    msl = make_where_kernel(block_size=256)
    path = runner.compile(msl, "where_kernel")
    pipeline = runner.load(path, "where_kernel")

    # Condition: even indices → True, odd → False
    cond = [1 if i % 2 == 0 else 0 for i in range(n)]
    x_data = [1.0] * n
    y_data = [-1.0] * n

    cond_buf = runner.device.newBufferWithLength_options_(n * 4, 0)
    cond_view = cond_buf.contents().as_buffer(n * 4)
    for i, c in enumerate(cond):
        struct.pack_into("I", cond_view, i * 4, c)
    x_buf = runner.make_float_buffer(x_data)
    y_buf = runner.make_float_buffer(y_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [cond_buf, x_buf, y_buf, out_buf, n_buf], n, block_size=256)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = 1.0 if cond[i] else -1.0
        assert abs(result[i] - expected) < 1e-5, \
            f"idx {i}: got {result[i]}, expected {expected}"


# ---------------------------------------------------------------------------
# Clamp and comparison tests
# ---------------------------------------------------------------------------

@requires_metal
def test_clamp_gpu(runner):
    """clamp: clip values to [min, max] range"""
    from triton_msl.codegen.msl_emitter import make_clamp_kernel

    n = 256
    msl = make_clamp_kernel(block_size=256)
    path = runner.compile(msl, "clamp_kernel")
    pipeline = runner.load(path, "clamp_kernel")

    data = [float(i) - 128.0 for i in range(n)]  # -128 to 127
    in_buf = runner.make_float_buffer(data)
    out_buf = runner.make_empty_buffer(n)
    min_buf = runner.make_float_scalar_buffer(-50.0)
    max_buf = runner.make_float_scalar_buffer(50.0)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, out_buf, min_buf, max_buf, n_buf], n, block_size=256)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = max(-50.0, min(50.0, data[i]))
        assert abs(result[i] - expected) < 1e-3, \
            f"idx {i}: got {result[i]}, expected {expected}"


@requires_metal
def test_compare_lt_gpu(runner):
    """compare_lt: element-wise less-than comparison"""
    from triton_msl.codegen.msl_emitter import make_compare_kernel
    import struct

    n = 256
    msl = make_compare_kernel(op="lt", block_size=256)
    path = runner.compile(msl, "compare_lt_kernel")
    pipeline = runner.load(path, "compare_lt_kernel")

    a_data = [float(i) for i in range(n)]
    b_data = [128.0] * n  # a < 128 for first 128 elements
    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.device.newBufferWithLength_options_(n * 4, 0)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n, block_size=256)

    out_view = out_buf.contents().as_buffer(n * 4)
    for i in range(n):
        val = struct.unpack_from("I", out_view, i * 4)[0]
        expected = 1 if a_data[i] < b_data[i] else 0
        assert val == expected, \
            f"idx {i}: got {val}, expected {expected}"
