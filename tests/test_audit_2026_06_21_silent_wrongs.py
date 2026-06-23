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


# --- #3: bf16 FlashAttention must NEVER dispatch a wrong result ---------------
@triton.jit
def _fa_bf16(Q, K, V, Out, sqz, sqh, sqm, sqk, skz, skh, skn, skk,
            svz, svh, svn, svk, soz, soh, som, sok, Z, H, N_CTX,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
            IS_CAUSAL: tl.constexpr, QKB: tl.constexpr, PVB: tl.constexpr):
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
        if QKB:
            qk = tl.dot(q.to(tl.bfloat16), tl.trans(k).to(tl.bfloat16)).to(tl.float32)
        else:
            qk = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32))
        qk = qk * (1.0 / math.sqrt(HEAD_DIM))
        mij = tl.max(qk, 1); mn = tl.maximum(mi, mij); al = tl.exp(mi - mn); p = tl.exp(qk - mn[:, None])
        li = li * al + tl.sum(p, 1); acc = acc * al[:, None]
        v = tl.load(V + oz * svz + oh * svh + (snn + on)[:, None] * svn + od[None, :] * svk,
                    mask=(snn + on)[:, None] < N_CTX, other=0.0)
        if PVB:
            acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32)
        else:
            acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        mi = mn
    tl.store(Out + oz * soz + oh * soh + om[:, None] * som + od[None, :] * sok,
             (acc / li[:, None]).to(Out.dtype.element_ty), mask=om[:, None] < N_CTX)


@requires
@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize("qkb,pvb", [(1, 0), (0, 1), (1, 1)])   # QK / PV / both dots bf16
def test_bf16_flash_attention_never_dispatches_wrong(head_dim, qkb, pvb):
    # CONTRACT: a bf16 FlashAttention kernel must NEVER silently dispatch a wrong
    # result. Verified (2026-06-22 fuzz, gate disabled) that across head_dim
    # {16,32,64} x {QK,PV,both}-bf16, EVERY variant refuses — none dispatch wrong.
    # bf16 FA is guarded redundantly: the bf16 dtype gate (clearest message), the
    # matmul "constexpr M/N" backstop (head_dim 32/64), and the <32-min-dot-tile
    # guard (head_dim 16). This test asserts the contract holds for each variant;
    # if it ever DISPATCHES, the result must match torch (so even a future
    # bf16-FA-compute path can't be silently wrong). The audit's reported
    # dispatch-wrong was NOT reproducible across these structures (likely an
    # artifact of the thermal-loaded 45-agent run).
    Z, H, N, D = 1, 1, 64, head_dim
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device="mps"); k = torch.randn(Z, H, N, D, device="mps")
    v = torch.randn(Z, H, N, D, device="mps"); o = torch.empty_like(q)
    try:
        _fa_bf16[(N // 32, Z * H)](q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                                   Z, H, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=D, IS_CAUSAL=False,
                                   QKB=qkb, PVB=pvb)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # loud refusal satisfies the never-silent-wrong contract
    # If it DID dispatch (no current path does), the result must be correct — never
    # silently wrong. bf16 -> generous tol.
    ref = (torch.softmax((q.float() * (1.0 / math.sqrt(D))) @ k.float().transpose(-2, -1), -1)
           @ v.float())
    rel = (o.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < 5e-2, f"SILENT-WRONG bf16 FA head_dim={D} qkb={qkb} pvb={pvb}: rel_err {rel:.3e}"


# --- Re-audit #3 (2026-06-22) follow-ups -------------------------------------
@requires
def test_bf16_flash_attention_strictly_refuses():
    # Strict refusal pin (re-audit #3 test-gap): bf16 FA (head_dim=64, both dots bf16)
    # MUST raise, never dispatch. Catches a regression where bf16 FA starts dispatching.
    # (bf16 FA is multiply-guarded — the dtype gate AND the matmul/block backstops — so
    # this pins the never-silent-wrong contract, not the dtype gate alone; a gate-only-
    # load-bearing test is impossible since the backstops refuse anyway. See audit memory.)
    Z, H, N, D = 1, 1, 64, 64
    q = torch.randn(Z, H, N, D, device="mps"); k = torch.randn(Z, H, N, D, device="mps")
    v = torch.randn(Z, H, N, D, device="mps"); o = torch.empty_like(q)
    with pytest.raises(MetalNonRecoverableError):
        _fa_bf16[(N // 32, Z * H)](q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                                   Z, H, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=D, IS_CAUSAL=False,
                                   QKB=1, PVB=1)
        torch.mps.synchronize()


@triton.jit
def _argmax1d(a, o, N: tl.constexpr):
    tl.store(o + tl.program_id(0), tl.argmax(tl.load(a + tl.arange(0, N)), axis=0))


@requires
def test_argmax_over_threadgroup_refuses_cleanly():
    # Re-audit #3: argmax/argmin over a 1-D tile > threadgroup needs the multipass
    # path, which has no tuple-reduce aggregation -> emitted UNCOMPILABLE MSL (a raw
    # MetalCompilationError). Must now be a clean MetalNonRecoverableError instead.
    N = 2048
    a = torch.randn(N, device="mps"); o = torch.empty(1, device="mps", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError):
        _argmax1d[(1,)](a, o, N=N); torch.mps.synchronize()


@requires
def test_argmax_within_threadgroup_correct():
    N = 1024
    torch.manual_seed(0)
    a = torch.randn(N, device="mps"); o = torch.empty(1, device="mps", dtype=torch.int32)
    _argmax1d[(1,)](a, o, N=N); torch.mps.synchronize()
    assert o[0].item() == a.cpu().argmax().item()


# --- Re-audit #4 (2026-06-22): integer shift >= bitwidth -> CUDA-clamped ------
@triton.jit
def _shl_k(a, b, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o + i, tl.load(a + i) << tl.load(b + i))


@triton.jit
def _shr_k(a, b, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o + i, tl.load(a + i) >> tl.load(b + i))


@requires
def test_shift_by_ge_bitwidth_matches_cuda():
    # An out-of-range shift (amount >= 32) is UB in MSL/ARM (mod-masks the amount);
    # CUDA/PTX clamps to a DEFINED result. We match CUDA: << and logical >> -> 0,
    # arithmetic >> -> sign fill. In-range shifts are unchanged.
    N = 4
    amt = torch.tensor([32, 64, 33, 3], device="mps", dtype=torch.int32)  # 3 OOR + 1 in-range
    # shli: 1 << amt  -> [0, 0, 0, 8]
    a = torch.ones(N, device="mps", dtype=torch.int32); o = torch.empty(N, device="mps", dtype=torch.int32)
    _shl_k[(1,)](a, amt, o, N=N); torch.mps.synchronize()
    assert o.cpu().tolist() == [0, 0, 0, 8], f"shli not CUDA-clamped: {o.cpu().tolist()}"
    # arithmetic shrsi: -8 >> amt -> sign fill (-1) for OOR; -8>>3 == -1 in-range
    a = torch.full((N,), -8, device="mps", dtype=torch.int32); o = torch.empty(N, device="mps", dtype=torch.int32)
    _shr_k[(1,)](a, amt, o, N=N); torch.mps.synchronize()
    assert o.cpu().tolist() == [-1, -1, -1, -1], f"shrsi not sign-filled: {o.cpu().tolist()}"


# --- Re-audit #5 (2026-06-22): reduce+cooperative-op interaction + epilogue NaN ---
@triton.jit
def _scan_then_reduce(a, o, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o, tl.sum(tl.cumsum(tl.load(a + i), 0), 0))


@triton.jit
def _hist_then_reduce(a, o, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o, tl.sum(tl.histogram(tl.load(a + i), B), 0))


@requires
def test_reduce_plus_cooperative_op_over_threadgroup_refuses():
    # A reduce paired with a cooperative shared-memory op (scan/histogram/...) over a
    # tile > num_threads took the multipass-reduce dispatch, which under-computes the
    # cooperative op's one-element-per-thread staging (scan+reduce gave 33024 vs 131328;
    # histogram re-counted each bin num_warps times). Must refuse loudly.
    N = 512
    a = torch.ones(N, device="mps"); o = torch.empty(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _scan_then_reduce[(1,)](a, o, N=N); torch.mps.synchronize()
    ai = torch.zeros(N, device="mps", dtype=torch.int32); oi = torch.empty(1, device="mps", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError):
        _hist_then_reduce[(1,)](ai, oi, N=N, B=16); torch.mps.synchronize()


@requires
def test_reduce_plus_scan_small_still_computes():
    # Below the threadgroup (no multipass) the combination is correct — must NOT over-refuse.
    N = 64
    a = torch.ones(N, device="mps"); o = torch.empty(1, device="mps")
    _scan_then_reduce[(1,)](a, o, N=N); torch.mps.synchronize()
    assert abs(o.item() - N * (N + 1) / 2) < 1.0


@triton.jit
def _mm_relu_nanprop(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        av = tl.load(a + rm[:, None] * sam + kk[None, :] * sak)
        bv = tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn)
        acc += tl.dot(av, bv)
    r = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, r)


@requires
def test_matmul_epilogue_propagates_nan():
    # The fused matmul-epilogue mapped maximumf/minimumf to NaN-quiet fmax/fmin,
    # dropping NaN under propagate_nan=ALL (relu of a NaN accumulator gave 0.0).
    import math
    M = N = K = 32
    A = torch.randn(M, K, device="mps"); A[0, 0] = float("nan")
    B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _mm_relu_nanprop[(1, 1)](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
                             BM=M, BN=N, BK=K); torch.mps.synchronize()
    assert math.isnan(C[0, 0].item()), f"epilogue dropped NaN: {C[0, 0].item()}"
