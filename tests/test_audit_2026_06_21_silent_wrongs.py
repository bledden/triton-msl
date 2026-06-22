"""Regression tests for the 3 silent-wrongs found by the 2026-06-21 dual-lens audit.

Each was a PRIME-DIRECTIVE violation (wrong numbers, no loud error) that the
correctness-only gates missed. These tests close that gap:
  1. 3D reduce with a pre-reduce elementwise op (tl.sum(a*s)) — the template
     dropped the op; both 3D-reduce paths mis-handle it -> must REFUSE loudly.
  2. fp16/bf16 simple-dot (no K-loop) output — the epilogue raced on a shared
     tg_out slot -> wrong/nondeterministic; must be correct + stable.
  3. bf16 FlashAttention at head_dim 32/64 — no dtype gate -> dispatched wrong;
     must REFUSE loudly.
"""
import math
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")


# --- #1: 3D reduce with a pre-reduce op must refuse; plain must still work ----
@triton.jit
def _reduce3d_scaled(a_ptr, out_ptr, s, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    bb = tl.arange(0, B); rr = tl.arange(0, R); cc = tl.arange(0, C)
    a = tl.load(a_ptr + bb[:, None, None] * R * C + rr[None, :, None] * C + cc[None, None, :])
    acc = tl.sum(a * s, axis=2)             # pre-reduce *s — must NOT be silently dropped
    tl.store(out_ptr + bb[:, None] * R + rr[None, :], acc)


@triton.jit
def _reduce3d_plain(a_ptr, out_ptr, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    bb = tl.arange(0, B); rr = tl.arange(0, R); cc = tl.arange(0, C)
    a = tl.load(a_ptr + bb[:, None, None] * R * C + rr[None, :, None] * C + cc[None, None, :])
    acc = tl.sum(a, axis=2)                 # direct load -> validated template path
    tl.store(out_ptr + bb[:, None] * R + rr[None, :], acc)


@requires
def test_3d_reduce_with_pre_op_refuses():
    B, R, C = 2, 4, 4
    a = torch.arange(B * R * C, device="mps", dtype=torch.float32).reshape(B, R, C)
    o = torch.empty(B, R, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _reduce3d_scaled[(1,)](a, o, 2.0, B=B, R=R, C=C)
        torch.mps.synchronize()


@requires
def test_3d_reduce_plain_still_correct():
    B, R, C = 2, 4, 4
    a = torch.arange(B * R * C, device="mps", dtype=torch.float32).reshape(B, R, C)
    o = torch.empty(B, R, device="mps")
    _reduce3d_plain[(1,)](a, o, B=B, R=R, C=C)
    torch.mps.synchronize()
    torch.testing.assert_close(o, a.sum(-1), rtol=1e-4, atol=1e-4)


# --- #2: fp16/bf16 simple-dot (no K-loop) output must be correct + stable -----
@triton.jit
def _dot32(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, BK)
    a = tl.load(a_ptr + om[:, None] * BK + ok[None, :])
    b = tl.load(b_ptr + ok[:, None] * BN + on[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + om[:, None] * BN + on[None, :], c.to(c_ptr.dtype.element_ty))


@requires
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_simple_dot_nonfloat_output_correct_and_stable(dtype):
    # The old shared-tg_out epilogue raced across simdgroups -> wrong AND
    # nondeterministic. Check correctness across several seeds.
    for seed in range(3):
        torch.manual_seed(seed)
        A = torch.randn(32, 32, device="mps", dtype=dtype)
        B = torch.randn(32, 32, device="mps", dtype=dtype)
        C = torch.empty(32, 32, device="mps", dtype=dtype)
        _dot32[(1,)](A, B, C, BM=32, BN=32, BK=32)
        torch.mps.synchronize()
        torch.testing.assert_close(C.float(), A.float() @ B.float(), rtol=2e-2, atol=2e-2)


# --- #3: bf16 FlashAttention (head_dim 32/64) must refuse loudly --------------
@triton.jit
def _fa_bf16(Q, K, V, Out, sqz, sqh, sqm, sqk, skz, skh, skn, skk,
            svz, svh, svn, svk, soz, soh, som, sok, Z, H, N_CTX,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
            IS_CAUSAL: tl.constexpr):
    sm = tl.program_id(0); hz = tl.program_id(1); oz = hz // H; oh = hz % H
    om = sm * BLOCK_M + tl.arange(0, BLOCK_M); on = tl.arange(0, BLOCK_N); od = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + oz * sqz + oh * sqh + om[:, None] * sqm + od[None, :] * sqk,
                mask=om[:, None] < N_CTX, other=0.0)
    mi = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    li = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for snn in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + oz * skz + oh * skh + (snn + on)[:, None] * skn + od[None, :] * skk,
                    mask=(snn + on)[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q.to(tl.bfloat16), tl.trans(k).to(tl.bfloat16)).to(tl.float32) * (1.0 / math.sqrt(HEAD_DIM))
        mij = tl.max(qk, 1); mn = tl.maximum(mi, mij); al = tl.exp(mi - mn); p = tl.exp(qk - mn[:, None])
        li = li * al + tl.sum(p, 1); acc = acc * al[:, None]
        v = tl.load(V + oz * svz + oh * svh + (snn + on)[:, None] * svn + od[None, :] * svk,
                    mask=(snn + on)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); mi = mn
    tl.store(Out + oz * soz + oh * soh + om[:, None] * som + od[None, :] * sok,
             (acc / li[:, None]).to(Out.dtype.element_ty), mask=om[:, None] < N_CTX)


@requires
@pytest.mark.parametrize("head_dim", [32, 64])
def test_bf16_flash_attention_refuses(head_dim):
    # CONTRACT test: bf16 FA must refuse (never silently mis-compute). CAVEAT
    # (verified by fault injection 2026-06-22): bf16 FA is double-guarded — this
    # passes even with the bf16 dtype gate disabled, because the matmul "constexpr
    # M/N" backstop also refuses this kernel. So this asserts the contract, NOT the
    # dtype gate's sole necessity. No kernel has been found that makes bf16 FA
    # dispatch a wrong result; the gate is defense-in-depth. (The audit reported a
    # bf16-FA dispatch-wrong but it could not be reproduced here across several FA
    # kernel structures — see audit memory.)
    Z, H, N, D = 1, 1, 64, head_dim
    q = torch.randn(Z, H, N, D, device="mps"); k = torch.randn(Z, H, N, D, device="mps")
    v = torch.randn(Z, H, N, D, device="mps"); o = torch.empty_like(q)
    with pytest.raises(MetalNonRecoverableError):
        _fa_bf16[(N // 32, Z * H)](q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                                   Z, H, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=D, IS_CAUSAL=False)
        torch.mps.synchronize()
