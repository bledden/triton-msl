"""End-to-end integration tests for the Metal backend.

Tests the full pipeline: TTGIR → MSL → metallib → GPU execution.
These tests don't require Triton itself — they simulate the compilation
pipeline using TTGIR text inputs and validate correctness on GPU.

If Triton is installed, additional tests run @triton.jit kernels through
the Metal backend.
"""

import math
import os
import struct
import subprocess
import tempfile

import pytest

try:
    import Metal
    import Foundation
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

requires_metal = pytest.mark.skipif(
    not HAS_METAL, reason="Metal framework not available"
)
requires_triton = pytest.mark.skipif(
    not HAS_TRITON, reason="Triton not installed"
)


class IntegrationRunner:
    """Full pipeline runner: TTGIR text → MSL → metallib → GPU execute."""

    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()
        self.cache_dir = os.path.join(
            tempfile.gettempdir(), "triton_msl_integration_test"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def compile_ttgir_to_msl(self, ttgir_text):
        """Parse TTGIR and emit MSL source."""
        from triton_msl.codegen.ttgir_parser import parse_ttgir

        class FakeOpts:
            num_warps = 4
            warp_size = 32
            num_stages = 1

        kb = parse_ttgir(ttgir_text, FakeOpts())
        return kb.build()

    def compile_msl_to_metallib(self, msl_src, kernel_name):
        """Compile MSL source to metallib and return pipeline state."""
        import hashlib

        src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
        base = f"{kernel_name}_{src_hash}"
        metal_path = os.path.join(self.cache_dir, f"{base}.metal")
        air_path = os.path.join(self.cache_dir, f"{base}.air")
        metallib_path = os.path.join(self.cache_dir, f"{base}.metallib")

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
        library, error = self.device.newLibraryWithURL_error_(url, None)
        assert error is None, f"Load failed: {error}"
        function = library.newFunctionWithName_(kernel_name)
        assert function is not None, f"Kernel '{kernel_name}' not found"
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        assert error is None, f"Pipeline failed: {error}"
        return pipeline

    def make_buffer(self, data, fmt="f"):
        """Create a Metal buffer from a list of values."""
        size_per = struct.calcsize(fmt)
        n = len(data)
        buf = self.device.newBufferWithLength_options_(
            n * size_per, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(n * size_per)
        for i, v in enumerate(data):
            struct.pack_into(fmt, view, i * size_per, v)
        return buf

    def make_uint_buf(self, value):
        """Create a single-uint buffer."""
        return self.make_buffer([value], "I")

    def make_empty_buffer(self, n, fmt="f"):
        """Create an empty Metal buffer."""
        size_per = struct.calcsize(fmt)
        return self.device.newBufferWithLength_options_(
            n * size_per, Metal.MTLResourceStorageModeShared
        )

    def read_buffer(self, buf, n, fmt="f"):
        """Read values from a Metal buffer."""
        size_per = struct.calcsize(fmt)
        view = buf.contents().as_buffer(n * size_per)
        return [struct.unpack_from(fmt, view, i * size_per)[0] for i in range(n)]

    def dispatch(self, pipeline, buffers, n_groups, block_size):
        """Dispatch a compute kernel."""
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_groups, 1, 1),
            Metal.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


@pytest.fixture
def runner():
    if not HAS_METAL:
        pytest.skip("Metal not available")
    return IntegrationRunner()


# ---------------------------------------------------------------------------
# Full pipeline: TTGIR → MSL → metallib → GPU
# ---------------------------------------------------------------------------

VECADD_TTGIR = """
module {
  tt.func public @vector_add(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = arith.cmpi slt, %4, %arg3 : tensor<256xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %7 = tt.addptr %6, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %8 = tt.load %7, %5 : !tt.ptr<f32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %11 = tt.load %10, %5 : !tt.ptr<f32>
    %12 = arith.addf %8, %11 : tensor<256xf32>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %14 = tt.addptr %13, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %14, %12, %5 : !tt.ptr<f32>
    tt.return
  }
}
"""


@requires_metal
def test_integration_vecadd_pipeline(runner):
    """Full pipeline: TTGIR text → MSL → metallib → GPU → correct results."""
    # Step 1: Parse TTGIR to MSL
    msl = runner.compile_ttgir_to_msl(VECADD_TTGIR)
    assert "vector_add" in msl or "kernel" in msl

    # Step 2: Compile MSL to metallib
    pipeline = runner.compile_msl_to_metallib(msl, "vector_add")

    # Step 3: Create test data
    n = 1024
    a_data = [float(i) * 0.1 for i in range(n)]
    b_data = [float(i) * 0.2 for i in range(n)]

    a_buf = runner.make_buffer(a_data)
    b_buf = runner.make_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buf(n)

    # Step 4: Execute on GPU
    n_groups = (n + 255) // 256
    runner.dispatch(pipeline, [a_buf, b_buf, out_buf, n_buf], n_groups, 256)

    # Step 5: Verify results
    result = runner.read_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"index {i}: got {result[i]}, expected {expected}"
        )


SOFTMAX_TTGIR = """
module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %7 = tt.load %6 : !tt.ptr<f32>

    %row_max = "tt.reduce"(%7) ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxf %a, %b : f32
      "tt.reduce.return"(%mx) : (f32) -> ()
    }) {axis = 1 : i32} : (tensor<256xf32>) -> f32

    %max_splat = tt.splat %row_max : f32 -> tensor<256xf32>
    %shifted = arith.subf %7, %max_splat : tensor<256xf32>
    %exp_val = math.exp %shifted : tensor<256xf32>

    %exp_sum = "tt.reduce"(%exp_val) ({
    ^bb0(%c: f32, %d: f32):
      %sm = arith.addf %c, %d : f32
      "tt.reduce.return"(%sm) : (f32) -> ()
    }) {axis = 1 : i32} : (tensor<256xf32>) -> f32

    %sum_splat = tt.splat %exp_sum : f32 -> tensor<256xf32>
    %result = arith.divf %exp_val, %sum_splat : tensor<256xf32>

    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %9, %result : !tt.ptr<f32>
    tt.return
  }
}
"""


@requires_metal
def test_integration_softmax_pipeline(runner):
    """Full pipeline: TTGIR softmax → MSL → metallib → GPU → correct results."""
    msl = runner.compile_ttgir_to_msl(SOFTMAX_TTGIR)

    # The parser should detect softmax pattern (max + sum reduces)
    assert "softmax" in msl.lower() or "max" in msl.lower()

    pipeline = runner.compile_msl_to_metallib(msl, "softmax_kernel")

    # Test with a single row of 64 elements
    n_cols = 64
    n_rows = 4
    total = n_rows * n_cols

    # Create input data
    input_data = [float(i % 10) * 0.1 for i in range(total)]
    in_buf = runner.make_buffer(input_data)
    out_buf = runner.make_empty_buffer(total)
    ncols_buf = runner.make_uint_buf(n_cols)

    runner.dispatch(pipeline, [in_buf, out_buf, ncols_buf], n_rows, 256)

    result = runner.read_buffer(out_buf, total)

    # Verify each row sums to 1.0 (softmax property)
    for r in range(n_rows):
        row = result[r * n_cols:(r + 1) * n_cols]
        row_sum = sum(row)
        assert abs(row_sum - 1.0) < 1e-4, (
            f"Row {r} sum = {row_sum}, expected ~1.0"
        )
        # All values should be positive
        assert all(v >= 0 for v in row), f"Row {r} has negative values"


@requires_metal
def test_integration_direct_kernel_roundtrip(runner):
    """Direct kernel (not TTGIR): generate MSL → compile → execute → verify."""
    from triton_msl.codegen.msl_emitter import make_silu_kernel

    msl = make_silu_kernel(block_size=256)
    pipeline = runner.compile_msl_to_metallib(msl, "silu_kernel")

    n = 512
    input_data = [float(i) * 0.01 - 2.5 for i in range(n)]  # range [-2.5, 2.6]
    in_buf = runner.make_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buf(n)

    n_groups = (n + 255) // 256
    runner.dispatch(pipeline, [in_buf, out_buf, n_buf], n_groups, 256)

    result = runner.read_buffer(out_buf, n)

    # Verify against Python SiLU: x / (1 + exp(-x))
    for i in range(n):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        assert abs(result[i] - expected) < 1e-4, (
            f"index {i}: x={x}, got {result[i]}, expected {expected}"
        )


@requires_metal
def test_integration_reduction_roundtrip(runner):
    """Direct kernel reduction: generate → compile → execute → verify."""
    from triton_msl.codegen.msl_emitter import make_reduce_kernel

    msl = make_reduce_kernel("reduce_sum", "sum", block_size=256)
    pipeline = runner.compile_msl_to_metallib(msl, "reduce_sum")

    n = 1024
    input_data = [float(i + 1) for i in range(n)]
    in_buf = runner.make_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buf(n)

    runner.dispatch(pipeline, [in_buf, out_buf, n_buf], 1, 256)

    result = runner.read_buffer(out_buf, 1)
    expected = sum(input_data)  # n*(n+1)/2 = 524800.0
    assert abs(result[0] - expected) < 1.0, (
        f"Sum: got {result[0]}, expected {expected}"
    )


@requires_metal
def test_integration_matmul_roundtrip(runner):
    """Direct kernel matmul: generate → compile → execute → verify."""
    from triton_msl.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 32, 32, 32
    msl = make_matmul_kernel(block_m=32, block_n=32, block_k=32)
    pipeline = runner.compile_msl_to_metallib(msl, "matmul_kernel")

    # Identity-like A (diagonal 1s), simple B
    A_data = [0.0] * (M * K)
    for i in range(min(M, K)):
        A_data[i * K + i] = 1.0
    B_data = [float(i % 10) * 0.1 for i in range(K * N)]

    A_buf = runner.make_buffer(A_data)
    B_buf = runner.make_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buf(M)
    N_buf = runner.make_uint_buf(N)
    K_buf = runner.make_uint_buf(K)

    threads_per_tg = 32 * 32
    runner.dispatch(pipeline, [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
                     1, threads_per_tg)

    result = runner.read_buffer(C_buf, M * N)

    # C = I @ B = B
    for i in range(M * N):
        assert abs(result[i] - B_data[i]) < 1e-3, (
            f"index {i}: got {result[i]}, expected {B_data[i]}"
        )


@requires_metal
def test_integration_flash_attention_roundtrip(runner):
    """Direct kernel flash attention: generate → compile → execute → verify."""
    from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 16
    head_dim = 64
    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16)
    pipeline = runner.compile_msl_to_metallib(msl, "flash_attention")

    total = seq_len * head_dim
    Q_data = [0.01] * total
    K_data = [0.01] * total
    V_data = [float(i % head_dim) * 0.01 for i in range(total)]

    Q_buf = runner.make_buffer(Q_data)
    K_buf = runner.make_buffer(K_data)
    V_buf = runner.make_buffer(V_data)
    O_buf = runner.make_empty_buffer(total)
    sl_buf = runner.make_uint_buf(seq_len)

    runner.dispatch(pipeline, [Q_buf, K_buf, V_buf, O_buf, sl_buf], 1, 256)

    result = runner.read_buffer(O_buf, total)

    # With uniform Q and K, attention weights should be ~uniform
    # so output ≈ mean(V) per position
    # Just check output is finite and non-zero
    assert all(math.isfinite(v) for v in result[:head_dim]), "Output should be finite"
    assert any(abs(v) > 1e-6 for v in result[:head_dim]), "Output should be non-zero"


@requires_triton
@requires_metal
def test_integration_backend_driver():
    """Test MetalDriver can detect the device and create targets."""
    from triton_msl.backend.driver import MetalDriver

    driver = MetalDriver()
    assert driver.is_active(), "Metal should be active on macOS"

    target = driver.get_current_target()
    assert "apple" in target.arch.lower() or "m" in target.arch.lower()

    device = driver.get_active_torch_device()
    assert "mps" in str(device)


@requires_triton
@requires_metal
def test_integration_backend_compiler():
    """Test MetalBackend compilation stages work."""
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    # Test options parsing
    opts = MetalOptions()
    assert opts.warp_size == 32
    assert opts.max_threadgroup_memory == 32768

    # Test hash is deterministic
    h1 = opts.hash()
    h2 = opts.hash()
    assert h1 == h2

    # Test make_metallib stage directly
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel
    msl_src = make_vector_add_kernel()
    metallib_bytes = MetalBackend.make_metallib(
        msl_src, {"name": "test_va"}, opts
    )
    assert isinstance(metallib_bytes, bytes)
    assert len(metallib_bytes) > 0
    # Metallib files start with 'MTLB' magic bytes.
    assert metallib_bytes[:4] == b"MTLB"


# ---------------------------------------------------------------------------
# Triton JIT tests (only run if Triton is installed)
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_triton_jit_vector_add():
    """@triton.jit vector_add compiles and runs via Metal backend.

    Uses CPU tensors because MPS tensors require deeper integration
    with PyTorch's MPS command queue for post-kernel operations.
    """
    import torch

    @triton.jit
    def vector_add_kernel(a_ptr, b_ptr, out_ptr, n,
                           BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, a + b, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    out = torch.zeros(n)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](a, b, out, n, BLOCK_SIZE=256)

    expected = a + b
    assert torch.allclose(out, expected, atol=1e-5)


@requires_triton
@requires_metal
def test_triton_jit_sum_reduction():
    """@triton.jit sum reduction compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def sum_kernel(input_ptr, output_ptr, n_elements,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(input_ptr + offsets, mask=mask, other=0.0)
        result = triton.language.sum(x, axis=0)
        if pid == 0:
            triton.language.store(output_ptr, result)

    n = 256
    a = torch.randn(n)
    out = torch.zeros(1)

    sum_kernel[(1,)](a, out, n, BLOCK_SIZE=256)

    expected = a.sum()
    assert torch.allclose(out, expected.unsqueeze(0), atol=1e-4), (
        f"Sum: got {out.item()}, expected {expected.item()}"
    )


@requires_triton
@requires_metal
def test_triton_jit_softmax():
    """@triton.jit fused softmax compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def softmax_kernel(input_ptr, output_ptr, n_cols,
                       BLOCK_SIZE: triton.language.constexpr):
        row_idx = triton.language.program_id(0)
        row_start = row_idx * n_cols
        offsets = triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row = triton.language.load(
            input_ptr + row_start + offsets, mask=mask, other=-float('inf')
        )
        row_max = triton.language.max(row, axis=0)
        shifted = row - row_max
        exp_vals = triton.language.exp(shifted)
        exp_sum = triton.language.sum(exp_vals, axis=0)
        softmax_out = exp_vals / exp_sum
        triton.language.store(
            output_ptr + row_start + offsets, softmax_out, mask=mask
        )

    n_rows, n_cols = 4, 64
    a = torch.randn(n_rows, n_cols)
    out = torch.zeros(n_rows, n_cols)

    softmax_kernel[(n_rows,)](a, out, n_cols, BLOCK_SIZE=128)

    expected = torch.softmax(a, dim=1)
    assert torch.allclose(out, expected, atol=1e-4)


@requires_triton
@requires_metal
def test_triton_jit_matmul():
    """@triton.jit tiled matmul compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: triton.language.constexpr,
        BLOCK_N: triton.language.constexpr,
        BLOCK_K: triton.language.constexpr,
    ):
        pid_m = triton.language.program_id(0)
        pid_n = triton.language.program_id(1)
        offs_m = pid_m * BLOCK_M + triton.language.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + triton.language.arange(0, BLOCK_N)
        offs_k = triton.language.arange(0, BLOCK_K)
        a_ptrs = (a_ptr + offs_m[:, None] * stride_am
                  + offs_k[None, :] * stride_ak)
        b_ptrs = (b_ptr + offs_k[:, None] * stride_bk
                  + offs_n[None, :] * stride_bn)
        acc = triton.language.zeros((BLOCK_M, BLOCK_N), dtype=triton.language.float32)
        for k in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
            b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            a = triton.language.load(a_ptrs, mask=a_mask, other=0.0)
            b = triton.language.load(b_ptrs, mask=b_mask, other=0.0)
            acc += triton.language.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = (c_ptr + offs_m[:, None] * stride_cm
                  + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        triton.language.store(c_ptrs, acc, mask=c_mask)

    # Test multiple sizes: single-tile and multi-tile.
    for size in [32, 64, 128]:
        M = N = K = size
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        c = torch.zeros(M, N)

        grid = (M // 32, N // 32)
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        )

        expected = a @ b
        assert torch.allclose(c, expected, atol=1e-2), (
            f"{size}x{size} max error: {(c - expected).abs().max().item()}"
        )


@requires_metal
@requires_triton
def test_triton_jit_silu():
    """@triton.jit SiLU (x * sigmoid(x)) compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(x_ptr + offsets, mask=mask)
        output = x * triton.language.sigmoid(x)
        triton.language.store(output_ptr + offsets, output, mask=mask)

    n = 1024
    x = torch.randn(n)
    output = torch.empty(n)
    silu_kernel[(triton.cdiv(n, 256),)](x, output, n, BLOCK_SIZE=256)

    expected = torch.nn.functional.silu(x)
    assert torch.allclose(output, expected, atol=1e-5), (
        f"SiLU max error: {(output - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_sigmoid():
    """@triton.jit sigmoid compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def sigmoid_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(x_ptr + offsets, mask=mask)
        output = triton.language.sigmoid(x)
        triton.language.store(output_ptr + offsets, output, mask=mask)

    n = 1024
    x = torch.randn(n)
    output = torch.empty(n)
    sigmoid_kernel[(triton.cdiv(n, 256),)](x, output, n, BLOCK_SIZE=256)

    expected = torch.sigmoid(x)
    assert torch.allclose(output, expected, atol=1e-5), (
        f"Sigmoid max error: {(output - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_gelu():
    """@triton.jit GELU compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(x_ptr + offsets, mask=mask)
        # GELU tanh approximation using exp-based tanh:
        # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        e2z = triton.language.exp(2.0 * z)
        tanh_z = (e2z - 1.0) / (e2z + 1.0)
        output = 0.5 * x * (1.0 + tanh_z)
        triton.language.store(output_ptr + offsets, output, mask=mask)

    n = 1024
    x = torch.randn(n)
    output = torch.empty(n)
    gelu_kernel[(triton.cdiv(n, 256),)](x, output, n, BLOCK_SIZE=256)

    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(output, expected, atol=1e-5), (
        f"GELU max error: {(output - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_vector_add_fp16():
    """@triton.jit vector_add with FP16 tensors."""
    import torch

    @triton.jit
    def add_kernel_fp16(a_ptr, b_ptr, out_ptr, n,
                        BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, a + b, mask=mask)

    n = 1024
    a = torch.randn(n, dtype=torch.float16)
    b = torch.randn(n, dtype=torch.float16)
    out = torch.zeros(n, dtype=torch.float16)

    add_kernel_fp16[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK_SIZE=256)

    expected = a + b
    assert torch.allclose(out, expected, atol=1e-3), (
        f"FP16 add max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_non_power_of_2():
    """@triton.jit with non-power-of-2 tensor sizes (edge case masking)."""
    import torch

    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, a + b, mask=mask)

    for n in [100, 300, 500, 997]:
        a = torch.randn(n)
        b = torch.randn(n)
        out = torch.zeros(n)

        add_kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK_SIZE=256)

        expected = a + b
        assert torch.allclose(out, expected, atol=1e-5), (
            f"n={n} max error: {(out - expected).abs().max().item()}"
        )


@requires_metal
@requires_triton
def test_triton_jit_elementwise_mul():
    """@triton.jit elementwise multiply (2 inputs, 1 output)."""
    import torch

    @triton.jit
    def mul_kernel(a_ptr, b_ptr, out_ptr, n,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, a * b, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    out = torch.empty(n)
    mul_kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK_SIZE=256)

    expected = a * b
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Mul max error: {(out - expected).abs().max().item()}"
    )




@requires_metal
@requires_triton
def test_triton_jit_fused_add_relu():
    """@triton.jit fused add + ReLU (2 inputs, 1 output, select op)."""
    import torch

    @triton.jit
    def fused_add_relu_kernel(a_ptr, b_ptr, out_ptr, n,
                              BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        x = a + b
        zero = 0.0
        output = triton.language.where(x > zero, x, zero)
        triton.language.store(out_ptr + offsets, output, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    out = torch.empty(n)
    fused_add_relu_kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK_SIZE=256)

    expected = torch.nn.functional.relu(a + b)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Fused add+ReLU max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_fused_mul_add():
    """@triton.jit fused multiply-add: out = a * b + c."""
    import torch

    @triton.jit
    def fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        c = triton.language.load(c_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, a * b + c, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    c = torch.randn(n)
    out = torch.empty(n)
    fma_kernel[(triton.cdiv(n, 256),)](a, b, c, out, n, BLOCK_SIZE=256)

    expected = a * b + c
    assert torch.allclose(out, expected, atol=1e-5), (
        f"FMA max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_squared_diff():
    """@triton.jit squared difference: out = (a - b)^2."""
    import torch

    @triton.jit
    def sq_diff_kernel(a_ptr, b_ptr, out_ptr, n,
                       BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        diff = a - b
        triton.language.store(out_ptr + offsets, diff * diff, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    out = torch.empty(n)
    sq_diff_kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK_SIZE=256)

    expected = (a - b) ** 2
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Squared diff max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_max_reduction():
    """@triton.jit max reduction compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def max_kernel(input_ptr, output_ptr, n_elements,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(input_ptr + offsets, mask=mask,
                                  other=-float('inf'))
        result = triton.language.max(x, axis=0)
        if pid == 0:
            triton.language.store(output_ptr, result)

    n = 256
    a = torch.randn(n)
    out = torch.zeros(1)

    max_kernel[(1,)](a, out, n, BLOCK_SIZE=256)

    expected = a.max()
    assert torch.allclose(out, expected.unsqueeze(0), atol=1e-5), (
        f"Max: got {out.item()}, expected {expected.item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_negation():
    """@triton.jit unary negation: out = -x."""
    import torch

    @triton.jit
    def neg_kernel(x_ptr, out_ptr, n,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, -x, mask=mask)

    n = 1024
    x = torch.randn(n)
    out = torch.empty(n)
    neg_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    expected = -x
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Negation max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_exp_log():
    """@triton.jit exp and log: out = log(exp(x)) ≈ x for moderate x."""
    import torch

    @triton.jit
    def exp_log_kernel(x_ptr, out_ptr, n,
                       BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        e = triton.language.exp(x)
        result = triton.language.log(e)
        triton.language.store(out_ptr + offsets, result, mask=mask)

    n = 1024
    # Use moderate range to avoid overflow in exp
    x = torch.randn(n).clamp(-5, 5)
    out = torch.empty(n)
    exp_log_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    # log(exp(x)) ≈ x
    assert torch.allclose(out, x, atol=1e-5), (
        f"exp→log max error: {(out - x).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_leaky_relu():
    """@triton.jit leaky ReLU: out = x if x > 0 else 0.01 * x."""
    import torch

    @triton.jit
    def leaky_relu_kernel(x_ptr, out_ptr, n,
                          BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        zero = 0.0
        output = triton.language.where(x > zero, x, 0.01 * x)
        triton.language.store(out_ptr + offsets, output, mask=mask)

    n = 1024
    x = torch.randn(n)
    out = torch.empty(n)
    leaky_relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    expected = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Leaky ReLU max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_clamp():
    """@triton.jit clamp: out = clamp(x, -1, 1)."""
    import torch

    @triton.jit
    def clamp_kernel(x_ptr, out_ptr, n,
                     BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        lo = -1.0
        hi = 1.0
        # Double where: clamp to [lo, hi]
        clamped = triton.language.where(x < lo, lo, x)
        clamped = triton.language.where(clamped > hi, hi, clamped)
        triton.language.store(out_ptr + offsets, clamped, mask=mask)

    n = 1024
    x = torch.randn(n) * 3  # wider range so clamping actually activates
    out = torch.empty(n)
    clamp_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    expected = x.clamp(-1, 1)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Clamp max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_scalar_multiply():
    """@triton.jit scalar multiply: out = 2.5 * x."""
    import torch

    @triton.jit
    def scale_kernel(x_ptr, out_ptr, n,
                     BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, 2.5 * x, mask=mask)

    n = 1024
    x = torch.randn(n)
    out = torch.empty(n)
    scale_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    expected = 2.5 * x
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Scalar mul max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_rms_norm():
    """@triton.jit RMS normalization: out = x / sqrt(mean(x^2) + eps) * weight."""
    import torch

    @triton.jit
    def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, n_cols,
                        eps: triton.language.constexpr,
                        BLOCK_SIZE: triton.language.constexpr):
        row_idx = triton.language.program_id(0)
        row_start = row_idx * n_cols
        offsets = triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = triton.language.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        w = triton.language.load(weight_ptr + offsets, mask=mask, other=0.0)
        # RMS: sqrt(mean(x^2) + eps)
        x_sq = x * x
        mean_sq = triton.language.sum(x_sq, axis=0) / n_cols
        rms = 1.0 / triton.language.sqrt(mean_sq + eps)
        out = x * rms * w
        triton.language.store(out_ptr + row_start + offsets, out, mask=mask)

    n_rows, n_cols = 4, 64
    x = torch.randn(n_rows, n_cols)
    weight = torch.randn(n_cols)
    out = torch.zeros(n_rows, n_cols)

    rms_norm_kernel[(n_rows,)](x, weight, out, n_cols, eps=1e-6, BLOCK_SIZE=128)

    # Reference RMS norm
    rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-6)
    expected = x / rms * weight
    assert torch.allclose(out, expected, atol=1e-4), (
        f"RMS norm max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_layer_norm():
    """@triton.jit layer normalization: out = (x - mean) / sqrt(var + eps) * w + b."""
    import torch

    @triton.jit
    def layer_norm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_cols,
                          eps: triton.language.constexpr,
                          BLOCK_SIZE: triton.language.constexpr):
        row_idx = triton.language.program_id(0)
        row_start = row_idx * n_cols
        offsets = triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = triton.language.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        w = triton.language.load(w_ptr + offsets, mask=mask, other=0.0)
        b = triton.language.load(b_ptr + offsets, mask=mask, other=0.0)
        # Mean
        mean = triton.language.sum(x, axis=0) / n_cols
        centered = x - mean
        # Variance
        var = triton.language.sum(centered * centered, axis=0) / n_cols
        inv_std = 1.0 / triton.language.sqrt(var + eps)
        out = centered * inv_std * w + b
        triton.language.store(out_ptr + row_start + offsets, out, mask=mask)

    # n_cols must equal BLOCK_SIZE to avoid masked-thread variance contamination
    # (Triton's masked loads set other=0, making centered = -mean for idle threads)
    n_rows, n_cols = 4, 128
    x = torch.randn(n_rows, n_cols)
    w = torch.randn(n_cols)
    b = torch.randn(n_cols)
    out = torch.zeros(n_rows, n_cols)

    layer_norm_kernel[(n_rows,)](x, w, b, out, n_cols, eps=1e-6, BLOCK_SIZE=128)

    expected = torch.layer_norm(x, [n_cols], weight=w, bias=b, eps=1e-6)
    assert torch.allclose(out, expected, atol=1e-4), (
        f"Layer norm max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_min_reduction():
    """@triton.jit min reduction compiles and runs via Metal backend."""
    import torch

    @triton.jit
    def min_kernel(input_ptr, output_ptr, n_elements,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = triton.language.load(input_ptr + offsets, mask=mask,
                                  other=float('inf'))
        result = triton.language.min(x, axis=0)
        if pid == 0:
            triton.language.store(output_ptr, result)

    n = 256
    a = torch.randn(n)
    out = torch.zeros(1)

    min_kernel[(1,)](a, out, n, BLOCK_SIZE=256)

    expected = a.min()
    assert torch.allclose(out, expected.unsqueeze(0), atol=1e-5), (
        f"Min: got {out.item()}, expected {expected.item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_softmax_large():
    """@triton.jit softmax with larger rows (stress test)."""
    import torch

    @triton.jit
    def softmax_kernel(input_ptr, output_ptr, n_cols,
                       BLOCK_SIZE: triton.language.constexpr):
        row_idx = triton.language.program_id(0)
        row_start = row_idx * n_cols
        offsets = triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row = triton.language.load(
            input_ptr + row_start + offsets, mask=mask, other=-float('inf')
        )
        row_max = triton.language.max(row, axis=0)
        shifted = row - row_max
        exp_vals = triton.language.exp(shifted)
        exp_sum = triton.language.sum(exp_vals, axis=0)
        softmax_out = exp_vals / exp_sum
        triton.language.store(
            output_ptr + row_start + offsets, softmax_out, mask=mask
        )

    # Test with larger dimensions
    for n_rows, n_cols, block in [(8, 128, 128), (16, 256, 256)]:
        a = torch.randn(n_rows, n_cols)
        out = torch.zeros(n_rows, n_cols)

        softmax_kernel[(n_rows,)](a, out, n_cols, BLOCK_SIZE=block)

        expected = torch.softmax(a, dim=1)
        assert torch.allclose(out, expected, atol=1e-4), (
            f"{n_rows}x{n_cols} softmax max error: "
            f"{(out - expected).abs().max().item()}"
        )


@requires_metal
@requires_triton
def test_triton_jit_swiglu():
    """@triton.jit SwiGLU activation: out = silu(gate) * up."""
    import torch

    @triton.jit
    def swiglu_kernel(gate_ptr, up_ptr, out_ptr, n,
                      BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        gate = triton.language.load(gate_ptr + offsets, mask=mask)
        up = triton.language.load(up_ptr + offsets, mask=mask)
        # SiLU(gate) * up
        silu_gate = gate * triton.language.sigmoid(gate)
        triton.language.store(out_ptr + offsets, silu_gate * up, mask=mask)

    n = 1024
    gate = torch.randn(n)
    up = torch.randn(n)
    out = torch.empty(n)
    swiglu_kernel[(triton.cdiv(n, 256),)](gate, up, out, n, BLOCK_SIZE=256)

    expected = torch.nn.functional.silu(gate) * up
    assert torch.allclose(out, expected, atol=1e-5), (
        f"SwiGLU max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_weighted_sum():
    """@triton.jit weighted sum: out = a * w_a + b * w_b (linear combination)."""
    import torch

    @triton.jit
    def weighted_sum_kernel(a_ptr, b_ptr, out_ptr, w_a, w_b, n,
                            BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = triton.language.load(a_ptr + offsets, mask=mask)
        b = triton.language.load(b_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, a * w_a + b * w_b, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    out = torch.empty(n)
    w_a, w_b = 0.6, 0.4
    weighted_sum_kernel[(triton.cdiv(n, 256),)](a, b, out, w_a, w_b, n, BLOCK_SIZE=256)

    expected = a * w_a + b * w_b
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Weighted sum max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_residual_add():
    """@triton.jit residual connection: out = x + residual."""
    import torch

    @triton.jit
    def residual_kernel(x_ptr, residual_ptr, out_ptr, n,
                        BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        residual = triton.language.load(residual_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, x + residual, mask=mask)

    n = 1024
    x = torch.randn(n)
    residual = torch.randn(n)
    out = torch.empty(n)
    residual_kernel[(triton.cdiv(n, 256),)](x, residual, out, n, BLOCK_SIZE=256)

    expected = x + residual
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Residual add max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_type_cast_fp16():
    """@triton.jit FP16 input → FP32 compute → FP16 output."""
    import torch

    @triton.jit
    def cast_compute_kernel(x_ptr, y_ptr, out_ptr, n,
                            BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask).to(triton.language.float32)
        y = triton.language.load(y_ptr + offsets, mask=mask).to(triton.language.float32)
        result = (x * y + x) * 0.5
        triton.language.store(out_ptr + offsets, result.to(triton.language.float16), mask=mask)

    n = 1024
    x = torch.randn(n, dtype=torch.float16)
    y = torch.randn(n, dtype=torch.float16)
    out = torch.empty(n, dtype=torch.float16)
    cast_compute_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK_SIZE=256)

    expected = ((x.float() * y.float() + x.float()) * 0.5).half()
    assert torch.allclose(out, expected, atol=1e-3), (
        f"FP16 cast max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_matmul_rectangular():
    """@triton.jit matmul with non-square matrices."""
    import torch

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: triton.language.constexpr,
        BLOCK_N: triton.language.constexpr,
        BLOCK_K: triton.language.constexpr,
    ):
        pid_m = triton.language.program_id(0)
        pid_n = triton.language.program_id(1)
        offs_m = pid_m * BLOCK_M + triton.language.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + triton.language.arange(0, BLOCK_N)
        offs_k = triton.language.arange(0, BLOCK_K)
        a_ptrs = (a_ptr + offs_m[:, None] * stride_am
                  + offs_k[None, :] * stride_ak)
        b_ptrs = (b_ptr + offs_k[:, None] * stride_bk
                  + offs_n[None, :] * stride_bn)
        acc = triton.language.zeros((BLOCK_M, BLOCK_N), dtype=triton.language.float32)
        for k in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
            b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            a = triton.language.load(a_ptrs, mask=a_mask, other=0.0)
            b = triton.language.load(b_ptrs, mask=b_mask, other=0.0)
            acc += triton.language.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = (c_ptr + offs_m[:, None] * stride_cm
                  + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        triton.language.store(c_ptrs, acc, mask=c_mask)

    # Rectangular: 64x96 @ 96x128
    M, N, K = 64, 128, 96
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.zeros(M, N)

    grid = (M // 32, N // 32)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )

    expected = a @ b
    assert torch.allclose(c, expected, atol=1e-2), (
        f"Rectangular matmul max error: {(c - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_online_softmax():
    """@triton.jit online (numerically stable) softmax with 3-pass pattern."""
    import torch

    @triton.jit
    def online_softmax_kernel(input_ptr, output_ptr, n_cols,
                              BLOCK_SIZE: triton.language.constexpr):
        row_idx = triton.language.program_id(0)
        row_start = row_idx * n_cols
        offsets = triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row = triton.language.load(
            input_ptr + row_start + offsets, mask=mask, other=-float('inf')
        )
        # Numerically stable: subtract max, exp, normalize
        row_max = triton.language.max(row, axis=0)
        numerator = triton.language.exp(row - row_max)
        denominator = triton.language.sum(numerator, axis=0)
        softmax_out = numerator / denominator
        triton.language.store(
            output_ptr + row_start + offsets, softmax_out, mask=mask
        )

    n_rows, n_cols = 8, 64
    a = torch.randn(n_rows, n_cols)
    out = torch.zeros(n_rows, n_cols)

    online_softmax_kernel[(n_rows,)](a, out, n_cols, BLOCK_SIZE=128)

    expected = torch.softmax(a, dim=1)
    assert torch.allclose(out, expected, atol=1e-4), (
        f"Online softmax max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_abs_value():
    """@triton.jit absolute value using tl.abs."""
    import torch

    @triton.jit
    def abs_kernel(x_ptr, out_ptr, n,
                   BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        triton.language.store(out_ptr + offsets, triton.language.abs(x), mask=mask)

    n = 1024
    x = torch.randn(n)
    out = torch.empty(n)
    abs_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    expected = x.abs()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Abs max error: {(out - expected).abs().max().item()}"
    )


@requires_metal
@requires_triton
def test_triton_jit_hardswish():
    """@triton.jit HardSwish: out = x * clamp(x/6 + 0.5, 0, 1)."""
    import torch

    @triton.jit
    def hardswish_kernel(x_ptr, out_ptr, n,
                         BLOCK_SIZE: triton.language.constexpr):
        pid = triton.language.program_id(0)
        offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = triton.language.load(x_ptr + offsets, mask=mask)
        inner = x / 6.0 + 0.5
        zero = 0.0
        one = 1.0
        clamped = triton.language.where(inner < zero, zero, inner)
        clamped = triton.language.where(clamped > one, one, clamped)
        result = x * clamped
        triton.language.store(out_ptr + offsets, result, mask=mask)

    n = 1024
    x = torch.randn(n) * 5
    out = torch.empty(n)
    hardswish_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)

    expected = torch.nn.functional.hardswish(x)
    assert torch.allclose(out, expected, atol=1e-4), (
        f"HardSwish max error: {(out - expected).abs().max().item()}"
    )
