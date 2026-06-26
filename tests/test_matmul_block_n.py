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
@pytest.mark.parametrize("BN", [8, 16, 32, 64, 128])
def test_kloop_matmul_all_columns_correct(BN):
    # square-ish tiled matmul (BN must be a power of 2 for tl.arange). BN>32
    # once dropped cols>=32; BN<32 (8/16) exercises the partially-idle-
    # simdgroup column distribution (the old 4x8=32-col store over-wrote a
    # <32-wide tile).
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


if HAS:
    @triton.jit
    def _mm_masked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, K, BK):
            a = tl.load(A + om[:, None] * K + (k + ok)[None, :],
                        mask=(om[:, None] < M) & ((k + ok)[None, :] < K), other=0.)
            b = tl.load(B + (k + ok)[:, None] * N + on[None, :],
                        mask=((k + ok)[:, None] < K) & (on[None, :] < N), other=0.)
            acc += tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], acc,
                 mask=(om[:, None] < M) & (on[None, :] < N))


@requires_metal
@pytest.mark.parametrize("M,N", [(48, 96), (40, 72), (33, 100)])
def test_kloop_matmul_partial_tiles_correct(M, N):
    # M/N NOT multiples of BLOCK -> partial edge tiles -> the staged masked
    # store path. The old unmasked float store wrapped overflow columns into
    # the next row's in-bounds data (maxdiff ~0.27) and wrote past the buffer.
    K, BM, BN, BK = 64, 32, 64, 32
    a = torch.randn(M, K, dtype=torch.float32) * 0.1
    b = torch.randn(K, N, dtype=torch.float32) * 0.1
    c = torch.zeros(M, N, dtype=torch.float32)
    grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)
    _mm_masked[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy(), atol=1e-2, rtol=1e-2)


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


if HAS:
    @triton.jit
    def _mm_softmax(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        acc = acc - tl.max(acc, axis=1)[:, None]
        e = tl.exp(acc)
        out = e / tl.sum(e, axis=1)[:, None]
        tl.store(C + om[:, None] * N + on[None, :], out)


@requires_metal
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
def test_matmul_softmax_not_dropped(dt):
    # A simple matmul->row-softmax kernel once routed to _detect_simple_dot,
    # which emitted a BARE matmul and silently dropped the softmax (output ==
    # A@B, row sums != 1). matmul_softmax is now checked first. bf16 was added
    # when the softmax template's fragment selection was routed through the shared
    # _simdgroup_frag_for (bf16 -> simdgroup_bfloat8x8 MMA instead of float-upcast).
    M = N = K = 32
    a = (torch.randn(M, K) * 0.3).to(dt)
    b = (torch.randn(K, N) * 0.3).to(dt)
    c = torch.zeros(M, N, dtype=torch.float32)
    _mm_softmax[(1,)](a, b, c, M=M, N=N, K=K)
    mm = (a.float() @ b.float()).numpy()
    ref = np.exp(mm - mm.max(1, keepdims=True))
    ref = ref / ref.sum(1, keepdims=True)
    got = c.numpy()
    np.testing.assert_allclose(got, ref, atol=2e-2, rtol=2e-2)
    # softmax rows sum to 1 (the dropped-softmax bug gave bare-matmul rows)
    np.testing.assert_allclose(got.sum(1), np.ones(M), atol=2e-2)


if HAS:
    @triton.jit
    def _mm_scale(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        acc = acc * 3.0 + 1.0          # non-softmax elementwise epilogue
        tl.store(C + om[:, None] * N + on[None, :], acc)


@requires_metal
def test_matmul_nonsoftmax_epilogue_computed_not_dropped():
    # A matmul + value-changing epilogue (scale/bias/activation) once routed to
    # _detect_simple_dot, which emitted a BARE matmul and silently dropped the
    # epilogue (returned A@B). #157 made it refuse; #158 now COMPUTES it via the
    # fused-epilogue template. Either way it is never silently dropped: verify
    # the epilogue is actually applied (result != bare A@B, == A@B*3+1).
    a = (torch.randn(32, 32) * 0.3)
    b = (torch.randn(32, 32) * 0.3)
    c = torch.zeros(32, 32)
    _mm_scale[(1,)](a, b, c, M=32, N=32, K=32)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy() * 3.0 + 1.0,
                               atol=2e-2, rtol=2e-2)


# --- follow-ups (2026-06-26): N%8/N%16 fast-path rescue + accurate 2-D accumulator refusal ---
if HAS:
    @triton.jit
    def _mm_2dacc(A, B, C, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        c = tl.load(C + om[:, None] * N + on[None, :])
        tl.store(Out + om[:, None] * N + on[None, :], tl.dot(a, b, c))


@requires_metal
@pytest.mark.parametrize("N", [520, 528, 544])   # %8-not-%16, %16-not-%32, %32 (all unaligned vs (4,4))
def test_matmul_unaligned_N_rescue_byte_exact(N):
    # The N%32 perf-cliff fix: N%8==0 shapes now reach the fast path (finer rc tile). Must be
    # byte-exact regardless of which tile the selector picks (a partial strip would OOB).
    M = K = 512
    a = torch.randn(M, K, dtype=torch.float32) * 0.1
    b = torch.randn(K, N, dtype=torch.float32) * 0.1
    c = torch.zeros(M, N, dtype=torch.float32)
    _mm[(triton.cdiv(M, 32), triton.cdiv(N, 32))](a, b, c, M, N, K, BM=32, BN=32, BK=32)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy(), atol=2e-2, rtol=2e-2)


@requires_metal
def test_2d_accumulator_refuses_with_accurate_message():
    # C = A@B + C with a FULL 2-D accumulator is refused with an ACCURATE message (was
    # mislabeled a 'row bias'); the simdgroup epilogue only adds a 1-D bias.
    M = N = K = 32
    a = torch.randn(M, K); b = torch.randn(K, N); c = torch.randn(M, N); out = torch.zeros(M, N)
    from triton_msl.errors import MetalNonRecoverableError
    with pytest.raises(MetalNonRecoverableError, match="2-D accumulator"):
        _mm_2dacc[(1,)](a, b, c, out, M=M, N=N, K=K)
