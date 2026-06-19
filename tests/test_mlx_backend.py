"""Tests for the Triton → MLX backend (Path C: zero-copy Metal dispatch).

Tests the full pipeline: @triton.jit → TTIR → TTGIR → MSL → mx.fast.metal_kernel()
"""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

import triton
import triton.language as tl
from triton_msl.mlx.msl_extractor import extract_msl_for_mlx, MSLExtraction
from triton_msl.mlx.mlx_launcher import MLXLauncher
import triton_msl.mlx as tmlx


# ─── MSL Extractor Unit Tests ─────────────────────────────────────────────────

class TestMSLExtractor:
    """Unit tests for MSL body extraction."""

    def test_extract_elementwise(self):
        """1D elementwise kernel: ptr args + scalar + thread vars."""
        msl = """
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(
    volatile device float* a_ptr [[buffer(0)]],
    volatile device float* b_ptr [[buffer(1)]],
    volatile device float* out_ptr [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]]
) {
    int c_0 = 256;
    int r_1 = pid * c_0;
    int r_2 = r_1 + lid;
    bool mask_3 = r_2 < n;
    float val_4 = mask_3 ? a_ptr[r_2] : 0.0f;
    float val_5 = mask_3 ? b_ptr[r_2] : 0.0f;
    float r_6 = val_4 + val_5;
    if (mask_3) { out_ptr[r_2] = r_6; }
}
"""
        ext = extract_msl_for_mlx(msl, output_arg_indices=[2])
        assert ext.kernel_name == "add_kernel"
        assert ext.input_names == ["a_ptr", "b_ptr"]
        assert ext.output_names == ["out_ptr"]
        assert ext.scalar_names == ["n"]
        assert ext.scalar_types == ["int"]
        assert not ext.uses_simd
        assert not ext.uses_2d
        assert "thread_position_in_grid" in ext.body

    def test_extract_with_simd(self):
        """Kernel with SIMD intrinsics (sgitg, tiisg)."""
        msl = """
#include <metal_stdlib>
using namespace metal;

kernel void reduce_kernel(
    volatile device float* x_ptr [[buffer(0)]],
    volatile device float* out_ptr [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_0[4];
    float val = x_ptr[tid];
    float s = simd_sum(val);
    if (tiisg == 0) shared_0[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pid == 0 && lid == 0) out_ptr[0] = shared_0[0];
}
"""
        ext = extract_msl_for_mlx(msl, output_arg_indices=[1])
        assert ext.uses_simd
        assert "simdgroup_index_in_threadgroup" in ext.body
        assert "thread_index_in_simdgroup" in ext.body
        assert "threadgroup float shared_0[4]" in ext.body

    def test_extract_2d_kernel(self):
        """2D kernel with pid3/lid3."""
        msl = """
#include <metal_stdlib>
using namespace metal;

kernel void kernel_2d(
    volatile device float* in_ptr0 [[buffer(0)]],
    volatile device float* out_ptr0 [[buffer(1)]],
    constant int& ynumel [[buffer(2)]],
    constant int& xnumel [[buffer(3)]],
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 tid3 [[thread_position_in_grid]]
) {
    uint pid = pid3.x;
    uint lid = lid3.x;
    uint tid = tid3.x;
    uint pid_y = pid3.y;
    out_ptr0[pid * 64 + lid] = in_ptr0[pid_y * 64 + lid];
}
"""
        ext = extract_msl_for_mlx(msl, output_arg_indices=[1])
        assert ext.uses_2d
        assert ext.scalar_names == ["ynumel", "xnumel"]
        # 2D should replace pid3.x with __pid_x
        assert "__pid_x" in ext.body
        assert "__pid_y" in ext.body

    def test_extract_all_outputs(self):
        """When no output_arg_indices, all ptr args become outputs."""
        msl = """
#include <metal_stdlib>
using namespace metal;

kernel void k(
    volatile device float* a [[buffer(0)]],
    volatile device float* b [[buffer(1)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]]
) {
    a[tid] = b[tid];
}
"""
        ext = extract_msl_for_mlx(msl, output_arg_indices=None)
        assert ext.input_names == []
        assert ext.output_names == ["a", "b"]

    def test_header_excludes_metal_stdlib(self):
        """Header should not include metal_stdlib (MLX provides it)."""
        msl = """
#include <metal_stdlib>
using namespace metal;

kernel void k(
    volatile device float* a [[buffer(0)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]]
) {
    a[tid] = 1.0f;
}
"""
        ext = extract_msl_for_mlx(msl)
        assert "metal_stdlib" not in ext.header


# ─── MLX Launcher Tests ───────────────────────────────────────────────────────

class TestMLXLauncher:
    """Tests for MLXLauncher dispatch with pre-extracted MSL."""

    @pytest.fixture
    def add_extraction(self):
        """Extraction for a simple add kernel."""
        msl = """
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(
    volatile device float* a_ptr [[buffer(0)]],
    volatile device float* b_ptr [[buffer(1)]],
    volatile device float* out_ptr [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]]
) {
    int c_0 = 256;
    int r_1 = pid * c_0;
    int r_2 = r_1 + lid;
    bool mask_3 = r_2 < n;
    float val_4 = mask_3 ? static_cast<float>(a_ptr[(0 + r_2)]) : static_cast<float>(0.0f);
    float val_5 = mask_3 ? static_cast<float>(b_ptr[(0 + r_2)]) : static_cast<float>(0.0f);
    float r_6 = val_4 + val_5;
    if (mask_3) { out_ptr[(0 + r_2)] = r_6; }
}
"""
        return extract_msl_for_mlx(msl, output_arg_indices=[2])

    def test_launcher_add(self, add_extraction):
        N = 1024
        np.random.seed(42)
        a = mlx.array(np.random.randn(N).astype(np.float32))
        b = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((N,))

        launcher = MLXLauncher(add_extraction, block_size=256)
        outputs = launcher((4,), a, b, out, N)

        result = np.array(outputs[0])
        ref = np.array(a) + np.array(b)
        np.testing.assert_allclose(result, ref, atol=1e-5)

    def test_launcher_different_sizes(self, add_extraction):
        """Test with non-power-of-2 size (needs masking)."""
        N = 1000
        a = mlx.array(np.random.randn(N).astype(np.float32))
        b = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((N,))

        launcher = MLXLauncher(add_extraction, block_size=256)
        # ceil(1000/256) = 4 threadgroups
        outputs = launcher((4,), a, b, out, N)

        result = np.array(outputs[0])
        ref = np.array(a) + np.array(b)
        np.testing.assert_allclose(result, ref, atol=1e-5)


# ─── End-to-End triton_call() Tests ───────────────────────────────────────────

class TestTritonCallElementwise:
    """End-to-end tests: @triton.jit → triton_call() → MLX output."""

    def test_vector_add(self):
        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        N = 1024
        x = mlx.array(np.random.randn(N).astype(np.float32))
        y = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((N,))

        (result,) = tmlx.triton_call(add_kernel, x, y, out, N, grid=(4,), BLOCK=256)
        np.testing.assert_allclose(np.array(result), np.array(x) + np.array(y), atol=1e-5)

    def test_scalar_mul(self):
        @triton.jit
        def scale_kernel(x_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x * scale, mask=mask)

        N = 1024
        x = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((N,))

        (result,) = tmlx.triton_call(scale_kernel, x, out, 3.14, N, grid=(4,), BLOCK=256)
        np.testing.assert_allclose(np.array(result), np.array(x) * 3.14, atol=1e-4)

    def test_relu(self):
        @triton.jit
        def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            out = tl.where(x > 0, x, 0.0)
            tl.store(out_ptr + offs, out, mask=mask)

        N = 1024
        x = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((N,))

        (result,) = tmlx.triton_call(relu_kernel, x, out, N, grid=(4,), BLOCK=256)
        np.testing.assert_allclose(np.array(result), np.maximum(np.array(x), 0), atol=1e-5)

    def test_fused_add_relu(self):
        @triton.jit
        def fused_add_relu(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            z = x + y
            out = tl.where(z > 0, z, 0.0)
            tl.store(out_ptr + offs, out, mask=mask)

        N = 1024
        x = mlx.array(np.random.randn(N).astype(np.float32))
        y = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((N,))

        (result,) = tmlx.triton_call(fused_add_relu, x, y, out, N, grid=(4,), BLOCK=256)
        ref = np.maximum(np.array(x) + np.array(y), 0)
        np.testing.assert_allclose(np.array(result), ref, atol=1e-5)


class TestTritonCallReductions:
    """Reduction kernels: softmax, sum."""

    def test_softmax(self):
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

        rows, cols = 8, 128
        np.random.seed(42)
        x_np = np.random.randn(rows, cols).astype(np.float32)
        x = mlx.array(x_np.flatten())
        out = mlx.zeros((rows * cols,))

        (result,) = tmlx.triton_call(softmax_kernel, x, out, cols, grid=(rows,), BLOCK=128)

        from scipy.special import softmax as scipy_softmax
        ref = scipy_softmax(x_np, axis=1).flatten()
        np.testing.assert_allclose(np.array(result), ref, atol=1e-5)

    def test_sum_reduction(self):
        @triton.jit
        def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            total = tl.sum(x, axis=0)
            # Only thread 0 writes the scalar result
            tl.store(out_ptr + pid, total)

        N = 128
        np.random.seed(42)
        x = mlx.array(np.random.randn(N).astype(np.float32))
        out = mlx.zeros((1,))

        (result,) = tmlx.triton_call(sum_kernel, x, out, N, grid=(1,), BLOCK=128)
        ref = float(np.sum(np.array(x)))
        np.testing.assert_allclose(np.array(result)[0], ref, atol=1e-3)


class TestTritonCallCaching:
    """Verify compilation caching works."""

    def test_second_call_uses_cache(self):
        @triton.jit
        def add_k(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        N = 256
        x = mlx.array(np.ones(N, dtype=np.float32))
        y = mlx.array(np.ones(N, dtype=np.float32))
        out = mlx.zeros((N,))

        # Clear cache for this test
        tmlx._compile_cache.clear()

        # First call: compiles
        (r1,) = tmlx.triton_call(add_k, x, y, out, N, grid=(1,), BLOCK=256)
        cache_size_after_first = len(tmlx._compile_cache)

        # Second call: should use cache (same signature)
        (r2,) = tmlx.triton_call(add_k, x, y, out, N, grid=(1,), BLOCK=256)
        cache_size_after_second = len(tmlx._compile_cache)

        assert cache_size_after_first == 1
        assert cache_size_after_second == 1  # No new entries
        np.testing.assert_allclose(np.array(r1), np.array(r2))


class TestMLXAvailable:
    """Test the mlx_available() utility."""

    def test_mlx_available(self):
        assert tmlx.mlx_available() is True
