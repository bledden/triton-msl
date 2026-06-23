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
def test_matmul_rowreduce_epilogue_correct():
    # matmul + row-reduce-subtract epilogue (acc - sum(acc, axis=1)). The prebuilt
    # matmul template can't represent the reduce and previously REFUSED here; re-audit
    # #6 showed that same prebuilt path was also SILENTLY DROPPING simpler epilogues
    # (looped matmul + fma/scale/relu returned the bare dot). The fix routes every
    # matmul-with-compute-epilogue to the generic op-by-op lowerer, which applies
    # this correctly — correct compute replacing a limitation-refusal.
    a, b = _ab(); c = torch.zeros(M, N)
    _mm_rowreduce[(1,)](a, b, c, M=M, N=N, K=K)
    mm = (a @ b).numpy()
    ref = mm - mm.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(c.numpy(), ref, atol=2e-2, rtol=2e-2)


if HAS:
    @triton.jit
    def _mm_scale_runtime_arg(A, B, C, alpha, M: tl.constexpr, N: tl.constexpr,
                              K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(a, b) * alpha)


@requires_metal
def test_matmul_runtime_scalar_epilogue_correct():
    # An epilogue that scales by a RUNTIME scalar arg. The epilogue TEMPLATE couldn't
    # lower the kernel-arg scalar (a splat leaf) and previously refused; the generic
    # op-by-op lowerer uses the scalar arg directly, so matmul-with-runtime-scalar
    # now computes correctly instead of refusing (re-audit #6). Never silently
    # resolves the scalar to 0.
    a, b = _ab(); c = torch.zeros(M, N)
    _mm_scale_runtime_arg[(1,)](a, b, c, 2.5, M=M, N=N, K=K)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy() * 2.5, atol=2e-2, rtol=2e-2)


if HAS:
    @triton.jit
    def _mm_kloop_fma(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0); pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            acc += tl.dot(tl.load(A + rm[:, None] * sam + kk[None, :] * sak),
                          tl.load(B + kk[:, None] * sbk + rn[None, :] * sbn))
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, tl.math.fma(acc, 2.0, 1.0))


@requires_metal
def test_matmul_kloop_epilogue_not_dropped():
    # re-audit #6: a LOOPED matmul (dot inside scf.for) with a trailing epilogue was
    # claimed by the inline-dot template, which stored the raw accumulator and SILENTLY
    # DROPPED the fma (returned the bare dot, maxerr ~16-33). The dot lives in the
    # scf.for region while the epilogue is top-level, so this routes to the generic
    # lowerer which applies it. Pins fma specifically (the op that slipped the epilogue
    # allow-list with no emission branch).
    a, b = _ab(); c = torch.zeros(M, N)
    _mm_kloop_fma[(1, 1)](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                          b.stride(1), c.stride(0), c.stride(1), BM=M, BN=N, BK=K)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy() * 2.0 + 1.0, atol=2e-2, rtol=2e-2)
