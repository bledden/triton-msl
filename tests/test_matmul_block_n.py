"""Integrity: the pid-tiled K-loop matmul must compute ALL BLOCK_N columns.

A pre-existing bug had `_lower_k_loop_dot_inline` emit only `sgitg*8` = 32
output columns regardless of BLOCK_N (4 simdgroups x 8). For BLOCK_N > 32 — the
STANDARD Triton matmul tiling (64/128) — columns 32+ were silently never
computed: a reachable silent-wrong. These tests dispatch real @triton.jit
tiled matmuls at several BLOCK_N and compare every column against numpy.
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

if HAS:
    @triton.jit
    def _mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, K, BK):
            a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
            b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
            acc += tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], acc)


def _matmul(M, N, K, BM, BN, BK):
    a = torch.randn(M, K, dtype=torch.float32) * 0.1
    b = torch.randn(K, N, dtype=torch.float32) * 0.1
    c = torch.zeros(M, N, dtype=torch.float32)
    grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)
    _mm[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    return c.numpy(), (a @ b).numpy()


@requires_metal
@pytest.mark.parametrize("BN", [32, 64, 128])
def test_kloop_matmul_all_columns_correct(BN):
    # square-ish tiled matmul; the bug left cols >= 32 wrong for BN > 32.
    M, K, BK = 64, 64, 32
    N = BN * 2  # two column tiles, so pid_n tiling is exercised too
    got, ref = _matmul(M, N, K, BM=32, BN=BN, BK=BK)
    np.testing.assert_allclose(got, ref, atol=1e-2, rtol=1e-2)


if HAS:
    @triton.jit
    def _mm_f16out(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, K, BK):
            a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
            b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
            acc += tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.float16))


@requires_metal
@pytest.mark.parametrize("BN", [32, 64])
def test_kloop_matmul_fp16_output_correct(BN):
    # fp16 OUTPUT goes through the threadgroup-staged convert path, which used
    # to write every simdgroup's accumulator to the same tg_out slot (a race
    # that corrupted ALL columns, even at BLOCK_N=32).
    M, K, BK = 64, 64, 32
    N = BN * 2
    a = torch.randn(M, K, dtype=torch.float32) * 0.1
    b = torch.randn(K, N, dtype=torch.float32) * 0.1
    c = torch.zeros(M, N, dtype=torch.float16)
    grid = ((M + 32 - 1) // 32, (N + BN - 1) // BN)
    _mm_f16out[grid](a, b, c, M, N, K, BM=32, BN=BN, BK=BK)
    np.testing.assert_allclose(c.numpy().astype(np.float32), (a @ b).numpy(),
                               atol=2e-2, rtol=2e-2)
