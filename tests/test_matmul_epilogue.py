"""Fused matmul + pointwise/broadcast epilogue (#158).

A matmul whose result feeds a pointwise/broadcast epilogue (scale, bias,
activation, chains) must COMPUTE the epilogue — not drop it (pre-#157) and not
refuse (post-#157). Softmax keeps its own path. Unsupported epilogues still
refuse loudly (integrity).
"""
import numpy as np
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

M = N = K = 32


def _ab():
    a = (torch.randn(M, K) * 0.3)
    b = (torch.randn(K, N) * 0.3)
    return a, b


if HAS:
    @triton.jit
    def _mm_scale(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], acc * 3.0 + 1.0)

    @triton.jit
    def _mm_relu(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], tl.maximum(acc, 0.0))

    @triton.jit
    def _mm_bias_relu(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        bias = tl.load(Bias + on)              # (N,)
        acc = acc + bias[None, :]
        tl.store(C + om[:, None] * N + on[None, :], tl.maximum(acc, 0.0))


@requires_metal
def test_matmul_scale_bias_const():
    a, b = _ab(); c = torch.zeros(M, N)
    _mm_scale[(1,)](a, b, c, M=M, N=N, K=K)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy() * 3.0 + 1.0,
                               atol=2e-2, rtol=2e-2)


@requires_metal
def test_matmul_relu():
    a, b = _ab(); c = torch.zeros(M, N)
    _mm_relu[(1,)](a, b, c, M=M, N=N, K=K)
    np.testing.assert_allclose(c.numpy(), np.maximum((a @ b).numpy(), 0.0),
                               atol=2e-2, rtol=2e-2)


@requires_metal
def test_matmul_bias_relu_linear_layer():
    a, b = _ab(); bias = torch.randn(N) * 0.3; c = torch.zeros(M, N)
    _mm_bias_relu[(1,)](a, b, bias, c, M=M, N=N, K=K)
    ref = np.maximum((a @ b).numpy() + bias.numpy()[None, :], 0.0)
    np.testing.assert_allclose(c.numpy(), ref, atol=2e-2, rtol=2e-2)


if HAS:
    @triton.jit
    def _mm_chain(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        # chained: scale, exp-ish (bounded), clamp
        acc = tl.maximum(acc * 2.0 - 0.5, 0.0)
        acc = tl.minimum(acc, 5.0)
        tl.store(C + om[:, None] * N + on[None, :], acc)

    @triton.jit
    def _mm_rowreduce(A, B, C, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        acc = acc - tl.sum(acc, axis=1)[:, None]   # reduce, NOT softmax
        tl.store(C + om[:, None] * N + on[None, :], acc)


@requires_metal
def test_matmul_chained_pointwise():
    a, b = _ab(); c = torch.zeros(M, N)
    _mm_chain[(1,)](a, b, c, M=M, N=N, K=K)
    mm = (a @ b).numpy()
    ref = np.minimum(np.maximum(mm * 2.0 - 0.5, 0.0), 5.0)
    np.testing.assert_allclose(c.numpy(), ref, atol=2e-2, rtol=2e-2)


@requires_metal
def test_matmul_unsupported_epilogue_still_refuses():
    # A non-softmax reduce epilogue is not representable -> must REFUSE loudly
    # (never silently drop), the #157 integrity boundary.
    from triton_metal.errors import MetalNonRecoverableError
    a, b = _ab(); c = torch.zeros(M, N)
    with pytest.raises(Exception) as ei:
        _mm_rowreduce[(1,)](a, b, c, M=M, N=N, K=K)
    assert ("epilogue" in str(ei.value).lower()
            or "MetalNonRecoverable" in type(ei.value).__name__)


if HAS:
    @triton.jit
    def _mm_scale_runtime_arg(A, B, C, alpha, M: tl.constexpr, N: tl.constexpr,
                              K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(a, b) * alpha)


@requires_metal
def test_matmul_runtime_scalar_epilogue_refuses_not_silentwrong():
    # An epilogue that scales by a RUNTIME scalar arg (not a constant) cannot be
    # lowered by the per-element emitter (the scalar is a kernel-arg leaf). It
    # must REFUSE LOUDLY (MetalNonRecoverableError), never silently resolve the
    # scalar to 0 (-> wrong output) nor crash the MSL compiler. #158 integrity.
    from triton_metal.errors import MetalNonRecoverableError
    a, b = _ab(); c = torch.zeros(M, N)
    with pytest.raises(MetalNonRecoverableError):
        _mm_scale_runtime_arg[(1,)](a, b, c, 2.5, M=M, N=N, K=K)
