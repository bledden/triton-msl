"""Stress tests for triton-msl.

Tests large tensors, repeated dispatches, and edge cases to verify
no crashes, memory leaks, or data corruption under sustained load.

Requires: TRITON_MSL_DEBUG=0 (or unset) for clean output.
"""

import pytest
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
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
    tl.store(out_ptr + row * n_cols + offsets, x_exp / x_sum, mask=mask)


@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(x, axis=0))


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


@pytest.mark.stress
class TestLargeTensors:
    """Verify correctness on large inputs (16M+ elements)."""

    def test_vector_add_16m(self):
        n = 16 * 1024 * 1024
        x = torch.randn(n, device="cpu")
        y = torch.randn(n, device="cpu")
        out = torch.empty(n, device="cpu")
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        expected = x + y
        assert torch.allclose(out, expected, atol=1e-5), f"max diff: {(out - expected).abs().max()}"

    def test_softmax_large(self):
        rows, cols = 4096, 1024
        x = torch.randn(rows, cols, device="cpu")
        out = torch.empty_like(x)
        _softmax_kernel[(rows,)](x, out, cols, BLOCK_SIZE=1024)
        expected = torch.softmax(x, dim=1)
        assert torch.allclose(out, expected, atol=1e-5), f"max diff: {(out - expected).abs().max()}"


@pytest.mark.stress
class TestRepeatedDispatch:
    """Verify no memory leak or crash under repeated dispatch."""

    def test_100_dispatches(self):
        n = 1024
        x = torch.randn(n, device="cpu")
        y = torch.randn(n, device="cpu")
        out = torch.empty(n, device="cpu")
        grid = (1,)
        for i in range(100):
            _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        expected = x + y
        assert torch.allclose(out, expected, atol=1e-5)

    def test_varying_sizes(self):
        """Dispatch with different tensor sizes to exercise buffer pool."""
        for n in [64, 256, 1024, 4096, 16384, 65536]:
            x = torch.randn(n, device="cpu")
            y = torch.randn(n, device="cpu")
            out = torch.empty(n, device="cpu")
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            _add_kernel[grid](x, y, out, n, BLOCK_SIZE=min(n, 1024))
            expected = x + y
            assert torch.allclose(out, expected, atol=1e-5), f"Failed at n={n}"


@pytest.mark.stress
class TestEdgeCases:
    """Edge cases that might trigger Metal-specific issues."""

    def test_single_element(self):
        x = torch.tensor([3.14], device="cpu")
        y = torch.tensor([2.71], device="cpu")
        out = torch.empty(1, device="cpu")
        _add_kernel[(1,)](x, y, out, 1, BLOCK_SIZE=1024)
        assert torch.allclose(out, x + y, atol=1e-5)

    def test_non_power_of_2_size(self):
        n = 1337
        x = torch.randn(n, device="cpu")
        y = torch.randn(n, device="cpu")
        out = torch.empty(n, device="cpu")
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        expected = x + y
        assert torch.allclose(out, expected, atol=1e-5)

    def test_fp16_large(self):
        n = 1024 * 1024
        x = torch.randn(n, device="cpu", dtype=torch.float16)
        y = torch.randn(n, device="cpu", dtype=torch.float16)
        out = torch.empty(n, device="cpu", dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        expected = x + y
        assert torch.allclose(out, expected, atol=1e-2), f"max diff: {(out - expected).abs().max()}"
