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
def _mm_relu_nanprop(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # NON-looped single dot -> the fused-epilogue template (where the NaN-propagation
    # fix lives). A LOOPED matmul + epilogue is refused, not computed (re-audit #6).
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    acc = tl.dot(tl.load(a + om[:, None] * K + ok[None, :]),
                 tl.load(b + ok[:, None] * N + on[None, :]))
    r = tl.maximum(acc, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(c + om[:, None] * N + on[None, :], r)


@requires
def test_matmul_epilogue_propagates_nan():
    # The fused matmul-epilogue mapped maximumf/minimumf to NaN-quiet fmax/fmin,
    # dropping NaN under propagate_nan=ALL (relu of a NaN accumulator gave 0.0).
    import math
    M = N = K = 32
    A = torch.randn(M, K, device="mps"); A[0, 0] = float("nan")
    B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _mm_relu_nanprop[(1,)](A, B, C, M=M, N=N, K=K); torch.mps.synchronize()
    assert math.isnan(C[0, 0].item()), f"epilogue dropped NaN: {C[0, 0].item()}"


# --- Re-audit #6 (2026-06-22): scan-in-loop race / splat-reduce / split gap ----
@triton.jit
def _cumsum_twice(a, o, N: tl.constexpr):
    i = tl.arange(0, N); x = tl.load(a + i)
    for _ in range(2):
        x = tl.cumsum(x, 0)
    tl.store(o + i, x)


@requires
@pytest.mark.parametrize("B", [512, 1024])
def test_cumsum_in_loop_no_race(B):
    # tl.cumsum inside an scf.for reused the scan shared buffer across iterations with
    # NO trailing barrier when total==block_size -> non-deterministic huge errors.
    # The trailing barrier after the scan loop fixes it; run 3x and require determinism.
    a = torch.ones(B, device="mps")
    ref = a.clone()
    for _ in range(2):
        ref = torch.cumsum(ref, 0)
    outs = []
    for _ in range(3):
        o = torch.empty(B, device="mps")
        _cumsum_twice[(1,)](a, o, N=B); torch.mps.synchronize()
        outs.append((o - ref).abs().max().item())
    assert all(e < 1.0 for e in outs), f"cumsum-in-loop wrong/racy: {outs}"


@triton.jit
def _scan_then_gather(a, idx, o, N: tl.constexpr):
    i = tl.arange(0, N)
    s = tl.cumsum(tl.load(a + i), 0)
    tl.store(o + i, tl.gather(s, tl.load(idx + i), axis=0))


@requires
def test_scan_then_gather_correct():
    # Two cooperative ops sharing threadgroup buffers; the scan trailing barrier
    # ensures the scan finishes reading before the gather staging reuses memory.
    N = 512
    torch.manual_seed(0)
    a = torch.ones(N, device="mps")
    idx = torch.randint(0, N, (N,), device="mps", dtype=torch.int32)
    o = torch.empty(N, device="mps")
    _scan_then_gather[(1,)](a, idx, o, N=N); torch.mps.synchronize()
    ref = torch.cumsum(a.cpu(), 0)[idx.cpu().long()]
    torch.testing.assert_close(o.cpu(), ref, rtol=1e-4, atol=1e-4)


@triton.jit
def _splat_sum(o, N: tl.constexpr):
    tl.store(o, tl.sum(tl.full((N,), 3.0, tl.float32), 0))


@requires
@pytest.mark.parametrize("N", [8, 16, 64])
def test_reduce_over_splat_counts_N_not_threads(N):
    # tl.sum over a tt.splat of N had no make_range to pin block_size to N, so the
    # cross-lane reduce summed num_threads copies (768 vs 24). Tail lanes now masked.
    o = torch.empty(1, device="mps")
    _splat_sum[(1,)](o, N=N); torch.mps.synchronize()
    assert abs(o.item() - N * 3.0) < 0.5, f"splat sum N={N}: {o.item()} != {N*3.0}"


@triton.jit
def _split_kernel(a, o0, o1, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(a + 2 * i)
    y = tl.load(a + 2 * i + 1)
    lo, hi = tl.split(tl.join(x, y))
    tl.store(o0 + i, lo); tl.store(o1 + i, hi)


@requires
def test_split_over_threadgroup_refuses():
    # GPU-execution coverage for the tt.split >1024 guard (was assertion-only).
    N = 1024  # join -> 2048 interleaved > 1024-thread cap
    a = torch.arange(2 * N, device="mps", dtype=torch.float32)
    o0 = torch.empty(N, device="mps"); o1 = torch.empty(N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _split_kernel[(1,)](a, o0, o1, N=N); torch.mps.synchronize()


# --- Re-audit #8 (confirming audit, 2026-06-23): 3 more, all REFUSE now ---------
@triton.jit
def _kloop_where(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, tl.where(acc > 0, acc, 0.0))


@requires
def test_kloop_matmul_where_epilogue_refuses():
    # A K-loop matmul + relu-via-tl.where silently dropped the where (stored the raw
    # dot, 505 negatives). tl.where is a top-level arith.select -> now refuses (a masked
    # LOAD's select lives in the loop region, so masked matmuls are NOT over-refused).
    M = N = K = 32
    a = torch.randn(M, K, device="mps"); b = torch.randn(K, N, device="mps"); c = torch.empty(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _kloop_where[(1, 1)](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                             BM=M, BN=N, BK=K); torch.mps.synchronize()


@triton.jit
def _fused_rowbias(a, bias, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    bv = tl.load(bias + om)[:, None]
    acc = tl.dot(tl.load(a + om[:, None] * K + ok[None, :]),
                 tl.load(b + ok[:, None] * N + on[None, :]), tl.broadcast_to(bv, (M, N)))
    tl.store(c + om[:, None] * N + on[None, :], acc * 2.0)


@requires
def test_fused_row_bias_matmul_refuses():
    # A fused per-ROW bias (M-length bias as the dot accumulator) is mis-computed
    # (re-audit #8: M=64 row 40 grossly wrong; direct repro wrong even at M=32). Refuses.
    M, N, K = 64, 32, 32
    a = torch.randn(M, K, device="mps"); b = torch.randn(K, N, device="mps")
    bias = torch.arange(M, device="mps", dtype=torch.float32); c = torch.empty(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _fused_rowbias[(1,)](a, bias, b, c, M=M, N=N, K=K); torch.mps.synchronize()


@triton.jit
def _join_split(a, o0, o1, N: tl.constexpr):
    i = tl.arange(0, N)
    lo, hi = tl.split(tl.join(tl.load(a + 2 * i), tl.load(a + 2 * i + 1)))
    tl.store(o0 + i, lo); tl.store(o1 + i, hi)


@requires
@pytest.mark.parametrize("N", [32, 256])
def test_split_refuses_at_small_sizes(N):
    # tt.split de-interleave mis-matches the store index -> wrong at ALL sizes (the old
    # guard only refused >1024). Now refuses everywhere (re-audit #8).
    a = torch.arange(2 * N, device="mps", dtype=torch.float32)
    o0 = torch.empty(N, device="mps"); o1 = torch.empty(N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _join_split[(1,)](a, o0, o1, N=N); torch.mps.synchronize()


# --- Re-audit #9 (2026-06-23): softmax fp16/bf16 output + i32-reduce precision ---
@triton.jit
def _softmax_row(a, o, NC, BLOCK: tl.constexpr):
    r = tl.program_id(0); cc = tl.arange(0, BLOCK); m = cc < NC
    x = tl.load(a + r * NC + cc, mask=m, other=-float("inf"))
    x = x - tl.max(x, 0); e = tl.exp(x); s = e / tl.sum(e, 0)
    tl.store(o + r * NC + cc, s.to(o.dtype.element_ty), mask=m)


@requires
@pytest.mark.parametrize("out_dt", [torch.float32, torch.float16, torch.bfloat16])
def test_softmax_nonfp32_output_correct(out_dt):
    # The vectorized float4 store reinterpret-cast a half*/bfloat* output -> NaN
    # (fp16) / compile crash (bf16). Now non-fp32 output takes the scalar store with a
    # proper cast. (Vectorize only when ALL ptr args are fp32.)
    import math
    R, NC, BLOCK = 4, 64, 64
    torch.manual_seed(0)
    a = torch.randn(R, NC, device="mps"); o = torch.empty(R, NC, device="mps", dtype=out_dt)
    _softmax_row[(R,)](a, o, NC, BLOCK=BLOCK); torch.mps.synchronize()
    ref = torch.softmax(a, dim=1).to(out_dt)
    err = (o.float() - ref.float()).abs().max().item()
    assert not math.isnan(err) and err < 8e-2, f"softmax {out_dt} output wrong: {err}"


@triton.jit
def _isum(a, o, N: tl.constexpr):
    tl.store(o, tl.sum(tl.load(a + tl.arange(0, N)), 0))


@triton.jit
def _imax(a, o, N: tl.constexpr):
    tl.store(o, tl.max(tl.load(a + tl.arange(0, N)), 0))


@requires
def test_i32_reduce_exact_above_2_24():
    # i32 reductions were emitted through the float-hardcoded threadgroup_reduce,
    # silently losing precision above 2^24. Now reduced in the int type.
    N = 256
    a = torch.full((N,), 100000, device="mps", dtype=torch.int32); o = torch.empty(1, device="mps", dtype=torch.int32)
    _isum[(1,)](a, o, N=N); torch.mps.synchronize()
    assert o.item() == N * 100000, f"i32 sum lost precision: {o.item()} != {N*100000}"
    a2 = torch.arange(N, device="mps", dtype=torch.int32) + (1 << 25); o2 = torch.empty(1, device="mps", dtype=torch.int32)
    _imax[(1,)](a2, o2, N=N); torch.mps.synchronize()
    assert o2.item() == (N - 1) + (1 << 25), f"i32 max lost precision: {o2.item()}"


# --- Re-audit #10 (2026-06-23): bool-xori/atomic-min-max + i64 atomic refuse ----
@triton.jit
def _amin(a, o, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N); tl.atomic_min(o + (i % K), tl.load(a + i))


@triton.jit
def _amax(a, o, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N); tl.atomic_max(o + (i % K), tl.load(a + i))


@requires
def test_fp32_atomic_min_max_correct():
    # arith.xori on i1 emitted `mask ^ -1` (an i1 all-ones constant rendered as -1),
    # always truthy -> both the signed-min and unsigned-min/max branches of the float
    # atomic sign-bit decomposition ran on every element (re-audit #10). Now the i1
    # constant renders 0/1, so the mask is correct.
    N, K = 128, 4
    torch.manual_seed(1); a = torch.randn(N, device="mps")
    o = torch.full((K,), 1e30, device="mps"); _amin[(1,)](a, o, N=N, K=K); torch.mps.synchronize()
    ref = torch.full((K,), 1e30)
    for idx in range(N):
        ref[idx % K] = min(ref[idx % K], a.cpu()[idx])
    torch.testing.assert_close(o.cpu(), ref, rtol=1e-4, atol=1e-4)
    an = -torch.rand(N, device="mps") - 0.1
    o2 = torch.full((K,), -1e30, device="mps"); _amax[(1,)](an, o2, N=N, K=K); torch.mps.synchronize()
    ref2 = torch.full((K,), -1e30)
    for idx in range(N):
        ref2[idx % K] = max(ref2[idx % K], an.cpu()[idx])
    torch.testing.assert_close(o2.cpu(), ref2, rtol=1e-4, atol=1e-4)


@triton.jit
def _aadd64(a, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.atomic_add(o + (i * 0), tl.load(a + i))


@requires
def test_i64_atomic_refuses():
    # Metal has no 64-bit device atomic; the int path truncated to 32 bits (wrote 0).
    a = torch.full((8,), 1 << 40, device="mps", dtype=torch.int64)
    o = torch.zeros(1, device="mps", dtype=torch.int64)
    with pytest.raises(MetalNonRecoverableError):
        _aadd64[(1,)](a, o, N=8); torch.mps.synchronize()


# --- Re-audit #10 part 2 (2026-06-23): argmin/max sub-warp + topk refuse --------
@triton.jit
def _argmin1d(a, o, N: tl.constexpr):
    tl.store(o, tl.argmin(tl.load(a + tl.arange(0, N)), 0))


@triton.jit
def _argmax1d(a, o, N: tl.constexpr):
    tl.store(o, tl.argmax(tl.load(a + tl.arange(0, N)), 0))


@requires
@pytest.mark.parametrize("N", [8, 16, 32, 64])
def test_argminmax_subwarp_correct(N):
    # For N<32 (sub-warp) the simd_shuffle_down tree read INACTIVE lanes (garbage 0),
    # so an all-positive argmin collapsed to index 0 (re-audit #10). Now the take is
    # guarded on the source thread being a real element (lid + _d < block_size).
    torch.manual_seed(N)
    a = torch.rand(N, device="mps") + 0.1
    o = torch.empty(1, device="mps", dtype=torch.int32)
    _argmin1d[(1,)](a, o, N=N); torch.mps.synchronize()
    assert o.item() == a.cpu().argmin().item(), f"argmin N={N}: {o.item()}"
    o2 = torch.empty(1, device="mps", dtype=torch.int32)
    _argmax1d[(1,)](a, o2, N=N); torch.mps.synchronize()
    assert o2.item() == a.cpu().argmax().item(), f"argmax N={N}: {o2.item()}"


@triton.jit
def _sort1d(a, o, N: tl.constexpr):
    tl.store(o + tl.arange(0, N), tl.sort(tl.load(a + tl.arange(0, N))))


@triton.jit
def _topk1d(a, o, N: tl.constexpr, K: tl.constexpr):
    tl.store(o + tl.arange(0, K), tl.topk(tl.load(a + tl.arange(0, N)), K, 0))


@triton.jit
def _topk2d(a, o, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N); kk = tl.arange(0, K)
    tl.store(o + i[:, None] * K + kk[None, :], tl.topk(tl.load(a + i[:, None] * N + j[None, :]), K, 1))


@requires
def test_topk_refuses_full_sort_correct():
    # tl.topk (K<N) mis-computed to duplicated values in BOTH the template and generic
    # paths (re-audit #10); refuse it. A full tl.sort (K==N) stays correct.
    torch.manual_seed(0)
    a = torch.randn(16, device="mps"); o = torch.empty(16, device="mps")
    _sort1d[(1,)](a, o, N=16); torch.mps.synchronize()
    torch.testing.assert_close(o.cpu(), a.cpu().sort().values, rtol=1e-4, atol=1e-4)
    for kernel, args in [(_topk1d, dict(N=16, K=4)), (_topk2d, dict(M=2, N=16, K=4))]:
        sz = 4 if kernel is _topk1d else (2, 4)
        a2 = torch.randn(16 if kernel is _topk1d else (2, 16), device="mps")
        o2 = torch.empty(sz, device="mps")
        with pytest.raises(MetalNonRecoverableError):
            kernel[(1,)](a2, o2, **args); torch.mps.synchronize()


# --- Re-audit #11 (2026-06-23): K-loop dot twins + 2D-reduce gap + layernorm arg ---
@triton.jit
def _kloop_bias(a, bias, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.load(bias + rn)[None, :] + tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc = tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                     tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn), acc)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc)


@requires
def test_kloop_dot_fused_bias_refuses():
    # K-loop matmul with a non-zero accumulator init (fused bias) had the init silently
    # DROPPED by the inline template (computed A@B only) — an unguarded twin of the
    # non-looped fused-bias class (re-audit #11). Now refuses.
    M = N = K = 32
    a = torch.randn(M, K, device="mps"); b = torch.randn(K, N, device="mps")
    bias = torch.randn(N, device="mps"); c = torch.empty(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _kloop_bias[(1, 1)](a, bias, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                            BM=M, BN=N, BK=16); torch.mps.synchronize()


@triton.jit
def _tile_loop_dot(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); rm = pm * BM + tl.arange(0, BM); rk = tl.arange(0, BK)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        acc = tl.dot(tl.load(a + rm[:, None] * sam + rk[None, :] * sak),
                     tl.load(b + rk[:, None] * sbk + rn[None, :] * sbn))
        tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc)


@requires
def test_tile_loop_dot_with_store_in_body_refuses():
    # A tt.dot in a TILE-iteration loop (store INSIDE the loop body) was mis-lowered to
    # the K-loop template (re-audit #11). Now detected (store-in-loop) and refused.
    M, K, N = 16, 16, 32
    a = torch.randn(M, K, device="mps"); b = torch.randn(K, N, device="mps"); c = torch.empty(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _tile_loop_dot[(1,)](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                             BM=16, BN=16, BK=16); torch.mps.synchronize()


@triton.jit
def _ln_2scalar(X, O, M, N, BLOCK: tl.constexpr):
    r = tl.program_id(0); c = tl.arange(0, BLOCK); m = c < N
    x = tl.load(X + r * N + c, mask=m, other=0.0); mu = tl.sum(x, 0) / N
    xc = tl.where(m, x - mu, 0.0); var = tl.sum(xc * xc, 0) / N; inv = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(O + r * N + c, xc * inv, mask=m)


@requires
def test_layernorm_two_scalar_args_correct():
    # _detect_layer_norm grabbed the FIRST scalar arg as the row length, so a kernel
    # passing both M and N normalized only the first M of N (re-audit #11: zeros 448/512).
    # With the scalar-args==1 guard it falls to the generic lowerer, which is correct.
    M, N = 8, 64
    X = torch.randn(M, N, device="mps"); O = torch.empty(M, N, device="mps")
    _ln_2scalar[(M,)](X, O, M, N, BLOCK=64); torch.mps.synchronize()
    ref = (X.cpu() - X.cpu().mean(1, keepdim=True)) / (X.cpu().var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    torch.testing.assert_close(O.cpu(), ref, rtol=2e-3, atol=2e-3)


# --- Re-audit #12 (2026-06-23): dot acc-init twins + 3D-i64 dtype twin ----------
@triton.jit
def _nonlooped_acc(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    acc = tl.dot(tl.load(a + om[:, None] * K + ok[None, :]),
                 tl.load(b + ok[:, None] * N + on[None, :]), tl.full((M, N), 5.0, tl.float32))
    tl.store(c + om[:, None] * N + on[None, :], acc)


@triton.jit
def _kloop_acc(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 7.0, tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc = tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                     tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn), acc)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc)


@requires
def test_dot_nonzero_acc_init_refuses_both_forms():
    # A non-zero accumulator init (fused bias / tl.full) is silently dropped by the
    # inline simdgroup template. Both the NON-LOOPED (init is a dot operand) and the
    # K-LOOP (init is an scf.for iter-arg) forms must refuse (re-audit #12 — twins; the
    # K-loop one was an INEFFECTIVE #11 guard that missed the loop-carried init).
    a = torch.randn(32, 32, device="mps"); b = torch.randn(32, 32, device="mps"); c = torch.empty(32, 32, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _nonlooped_acc[(1,)](a, b, c, M=32, N=32, K=32); torch.mps.synchronize()
    c2 = torch.empty(32, 32, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _kloop_acc[(1, 1)](a, b, c2, 32, 32, 32, *a.stride(), *b.stride(), *c2.stride(),
                           BM=32, BN=32, BK=16); torch.mps.synchronize()


@triton.jit
def _reduce3d_i64(a, o, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    bb = tl.arange(0, B); rr = tl.arange(0, R); cc = tl.arange(0, C)
    tl.store(o + bb[:, None] * R + rr[None, :],
             tl.sum(tl.load(a + bb[:, None, None] * R * C + rr[None, :, None] * C + cc[None, None, :]), 2))


@requires
def test_3d_reduce_i64_refuses():
    # The 3D reduce/argminmax template computes in float32 and the generic path
    # truncates to 32 bits — neither handles i64 (re-audit #12: non-pow2 i64 sum wrong).
    B, R, C = 2, 4, 4
    a = torch.full((B, R, C), (1 << 40) + 1234567, device="mps", dtype=torch.int64)
    o = torch.empty(B, R, device="mps", dtype=torch.int64)
    with pytest.raises(MetalNonRecoverableError):
        _reduce3d_i64[(1,)](a, o, B=B, R=R, C=C); torch.mps.synchronize()


# --- Re-audit #12 part 2 (2026-06-23): softmax/layernorm sole-scalar-is-row-COUNT ---
@triton.jit
def _softmax_count_scalar(x, o, M, N: tl.constexpr):
    r = tl.program_id(0); c = tl.arange(0, N)
    v = tl.load(x + r * N + c); v = v - tl.max(v, 0); e = tl.exp(v)
    tl.store(o + r * N + c, e / tl.sum(e, 0))


@triton.jit
def _layernorm_count_scalar(x, o, M, N: tl.constexpr):
    r = tl.program_id(0); c = tl.arange(0, N)
    v = tl.load(x + r * N + c); mu = tl.sum(v, 0) / N; vc = v - mu
    var = tl.sum(vc * vc, 0) / N
    tl.store(o + r * N + c, vc / tl.sqrt(var + 1e-5))


@requires
def test_softmax_layernorm_count_scalar_correct():
    # softmax/layernorm(x, out, M, N: constexpr) has ONE runtime scalar M = the row
    # COUNT (row length N is constexpr). The template grabbed M as the row length and
    # normalized the wrong span (re-audit #12). The row-stride-is-constexpr guard now
    # declines to the generic lowerer, which is correct.
    import torch.nn.functional as F
    X = torch.randn(4, 64, device="mps")
    O = torch.empty(4, 64, device="mps")
    _softmax_count_scalar[(4,)](X, O, 4, N=64); torch.mps.synchronize()
    torch.testing.assert_close(O.cpu(), F.softmax(X.cpu(), dim=1), rtol=2e-3, atol=2e-3)
    O2 = torch.empty(4, 64, device="mps")
    _layernorm_count_scalar[(4,)](X, O2, 4, N=64); torch.mps.synchronize()
    ref = (X.cpu() - X.cpu().mean(1, keepdim=True)) / (X.cpu().var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    torch.testing.assert_close(O2.cpu(), ref, rtol=2e-3, atol=2e-3)


# --- Re-audit #13 (2026-06-23): strided dot acc-init twin + scan i64 twin ---------
@triton.jit
def _strided_dot_bias(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 2.0, tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc)


@requires
def test_strided_dot_nonzero_acc_init_refuses():
    # The acc-init guard lived only in _detect_simple_dot, which early-returns on stride
    # args — so the STRIDED template silently dropped a non-zero const init (re-audit #13,
    # twin). Now mirrored onto the strided path (scoped to the accumulator tile).
    a = torch.randn(32, 32, device="mps"); b = torch.randn(32, 32, device="mps"); c = torch.empty(32, 32, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _strided_dot_bias[(1, 1)](a, b, c, 32, 32, 32, *a.stride(), *b.stride(), *c.stride(),
                                  BM=32, BN=32, BK=16); torch.mps.synchronize()


@triton.jit
def _cumsum_1d(a, o, N: tl.constexpr):
    tl.store(o + tl.arange(0, N), tl.cumsum(tl.load(a + tl.arange(0, N)), 0))


@requires
def test_cumsum_i64_no_truncation():
    # _lower_scan mapped any int to i32 while its sibling _lower_reduce branches to
    # long/ulong — so i64 cumsum wrapped at 2^31 (re-audit #13). Now i64-aware.
    N = 8
    a = torch.full((N,), 800000000, device="mps", dtype=torch.int64)
    o = torch.empty(N, device="mps", dtype=torch.int64)
    _cumsum_1d[(1,)](a, o, N=N); torch.mps.synchronize()
    # compare values (torch.cumsum promotes int dtypes, so compare as lists)
    assert o.cpu().tolist() == a.cpu().cumsum(0).tolist()
    # i32 stays correct (no over-widening regression)
    a2 = torch.arange(1, N + 1, device="mps", dtype=torch.int32)
    o2 = torch.empty(N, device="mps", dtype=torch.int32)
    _cumsum_1d[(1,)](a2, o2, N=N); torch.mps.synchronize()
    assert o2.cpu().tolist() == a2.cpu().cumsum(0).tolist()


# --- Re-audit #14 (2026-06-23): matmul-template edges + scan/argmax twins ---------
@triton.jit
def _strided_pid_bias(a, b, c, sam, sak, sbk, sbn, scm, scn, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 5.0, tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc)


@triton.jit
def _masked_k_dot(a, b, c, M: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, K):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, BK); mk = ok < K
    av = tl.load(a + om[:, None] * K + ok[None, :], mask=mk[None, :], other=0.0)
    bv = tl.load(b + ok[:, None] * N + on[None, :], mask=mk[:, None], other=0.0)
    tl.store(c + om[:, None] * N + on[None, :], tl.dot(av, bv))


@triton.jit
def _batch_pid_dot(a, b, c, sb, sam, sak, sbk, sbn, scm, scn, K,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(2); pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        acc += tl.dot(tl.load(a + pb * sb + rm[:, None] * sam + kk[None, :] * sak),
                      tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(c + pb * sb + rm[:, None] * scm + rn[None, :] * scn, acc)


@requires
def test_matmul_template_edges_refuse():
    # Three matmul fast-path edges the simdgroup/strided templates mis-handle, now
    # refused (re-audit #14): strided pid K-loop with non-zero acc init (the #13 strided
    # mirror was dead — tt.dot only in scf.for region_ops); masked input load (padded K,
    # template strides by tile width not runtime K); 3-D-grid batched matmul (program_id(2)
    # batch axis dropped).
    a = torch.randn(32, 32, device="mps"); b = torch.randn(32, 32, device="mps"); c = torch.empty(32, 32, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _strided_pid_bias[(1, 1)](a, b, c, *a.stride(), *b.stride(), *c.stride(), 32,
                                  BM=32, BN=32, BK=16); torch.mps.synchronize()
    a2 = torch.randn(16, 3, device="mps"); b2 = torch.randn(3, 16, device="mps"); c2 = torch.empty(16, 16, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _masked_k_dot[(1,)](a2, b2, c2, M=16, N=16, BK=8, K=3); torch.mps.synchronize()
    a3 = torch.randn(2, 16, 16, device="mps"); b3 = torch.randn(16, 16, device="mps"); c3 = torch.empty(2, 16, 16, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _batch_pid_dot[(1, 1, 2)](a3, b3, c3, a3.stride(0), a3.stride(1), a3.stride(2),
                                  b3.stride(0), b3.stride(1), c3.stride(1), c3.stride(2), 16,
                                  BM=16, BN=16, BK=16); torch.mps.synchronize()


@triton.jit
def _mixed_scan_comb(c1, v1, c2, v2):
    return c1 + c2, v1 + v2


@triton.jit
def _mixed_scan(a, b, oc, osu, N: tl.constexpr):
    i = tl.arange(0, N); cnt = tl.load(a + i); val = tl.load(b + i)
    cc, vv = tl.associative_scan((cnt, val), 0, _mixed_scan_comb)
    tl.store(oc + i, cc); tl.store(osu + i, vv)


@requires
def test_mixed_dtype_multivalue_scan_refuses():
    # A multi-value scan staged every slot with operand-0's dtype, truncating the others
    # (re-audit #14: fp32 sum slot -> i32 -> zeros). Mixed-dtype now refuses.
    a = torch.ones(8, device="mps", dtype=torch.int32)
    b = (torch.arange(8, device="mps", dtype=torch.float32) + 1) * 0.1
    oc = torch.empty(8, device="mps", dtype=torch.int32); osu = torch.empty(8, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _mixed_scan[(1,)](a, b, oc, osu, N=8); torch.mps.synchronize()


@triton.jit
def _argmax2d_idx(x, oi, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M); c = tl.arange(0, N)
    _v, idx = tl.max(tl.load(x + r[:, None] * N + c[None, :]), 1, return_indices=True)
    tl.store(oi + r, idx)


@requires
def test_2d_argmax_axis1_indices_correct():
    # 2D argmax axis=1 returned correct values but indices uniformly 0 (re-audit #14).
    # The index along axis=1 IS the column position; now correct.
    M, N = 4, 8
    X = torch.full((M, N), -9.0, device="mps")
    for r in range(M):
        X[r, r] = 10.0 + r
    oi = torch.empty(M, device="mps", dtype=torch.int32)
    _argmax2d_idx[(1,)](X, oi, M=M, N=N); torch.mps.synchronize()
    assert oi.cpu().tolist() == [0, 1, 2, 3]


# --- Re-audit #14 #2 (2026-06-23): in-loop 2D-reduce under-fill -------------------
@triton.jit
def _inloop_2dreduce(a, o, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    rm = tl.arange(0, M); acc = tl.zeros((M,), tl.float32)
    for t in range(0, T):
        cols = tl.arange(0, N)
        x = tl.load(a + t * M * N + rm[:, None] * N + cols[None, :])
        acc += tl.sum(x, axis=1)
    tl.store(o + rm, acc)


@requires
def test_inloop_2d_reduce_underfill_refuses_or_correct():
    # An in-loop 2D reduce whose tile UNDER-fills the (256-min) threadgroup (M*N < 256)
    # had its staged reduce corrupted by the surplus threads -> silent-wrong (re-audit
    # #14, the under-fill twin of the >block_size over-fill guard). Small tiles now
    # refuse; tiles that fill the group (M*N >= 256) still compute correctly.
    T = 3
    # under-fill (M*N=128) -> must refuse, never silent-wrong
    a = torch.randn(T, 8, 16, device="mps"); o = torch.empty(8, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _inloop_2dreduce[(1,)](a, o, M=8, N=16, T=T); torch.mps.synchronize()
    # filled (M*N=512) -> correct
    a2 = torch.randn(T, 8, 64, device="mps"); o2 = torch.empty(8, device="mps")
    _inloop_2dreduce[(1,)](a2, o2, M=8, N=64, T=T); torch.mps.synchronize()
    torch.testing.assert_close(o2.cpu(), a2.cpu().sum(dim=(0, 2)), rtol=2e-3, atol=2e-3)


@triton.jit
def _noloop_2dreduce(a, o, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M); cols = tl.arange(0, N)
    tl.store(o + rm, tl.sum(tl.load(a + rm[:, None] * N + cols[None, :]), axis=1))


@requires
def test_noloop_2d_reduce_small_tile_not_over_refused():
    # The no-loop 2D reduce has block_size == total (no surplus), so a small tile is
    # correct and must NOT be caught by the in-loop under-fill guard (depth-gated).
    a = torch.randn(8, 16, device="mps"); o = torch.empty(8, device="mps")
    _noloop_2dreduce[(1,)](a, o, M=8, N=16); torch.mps.synchronize()
    torch.testing.assert_close(o.cpu(), a.cpu().sum(1), rtol=2e-3, atol=2e-3)


# --- Reduce fuzzer findings (2026-06-23): bf16 3D reduce + fp16/bf16 in-loop 2D ----
@triton.jit
def _reduce3d_bf16(a, o, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    b = tl.arange(0, B); r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(a + b[:, None, None] * R * C + r[None, :, None] * C + c[None, None, :])
    tl.store(o + b[:, None] * R + r[None, :], tl.sum(x, 2))


@requires
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_3d_reduce_fp16_bf16_compiles_correct(dtype):
    # The 3D reduce template accumulates in float; storing to a bf16 output without a
    # cast failed the MSL compile ('assigning to bfloat from float' — MSL has no implicit
    # float->bfloat; half is implicit). Now cast on store; both fp16 and bf16 correct.
    B, R, C = 2, 4, 4
    a = torch.randn(B, R, C, device="mps", dtype=dtype)
    o = torch.empty(B, R, device="mps", dtype=dtype)
    _reduce3d_bf16[(1,)](a, o, B=B, R=R, C=C); torch.mps.synchronize()
    torch.testing.assert_close(o.float().cpu(), a.float().cpu().sum(2), rtol=3e-2, atol=3e-2)


@triton.jit
def _inloop_2dreduce_dt(a, o, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, M); acc = tl.zeros((M,), tl.float32)
    for t in range(0, T):
        j = tl.arange(0, N)
        acc += tl.sum(tl.load(a + t * M * N + i[:, None] * N + j[None, :]), 1)
    tl.store(o + i, acc)


@requires
def test_inloop_2d_reduce_fp16_refuses_fp32_ok():
    # An in-loop 2D reduce of fp16/bf16 input was mis-staged across iterations (T>=2):
    # every row collapsed to the first row's value (fp32 fine). Refuse fp16/bf16; fp32
    # (filled tile) still computes.
    M, N, T = 8, 64, 3
    for dt in (torch.float16, torch.bfloat16):
        a = torch.randn(T, M, N, device="mps", dtype=dt); o = torch.empty(M, device="mps", dtype=dt)
        with pytest.raises(MetalNonRecoverableError):
            _inloop_2dreduce_dt[(1,)](a, o, M=M, N=N, T=T); torch.mps.synchronize()
    a2 = torch.randn(T, M, N, device="mps"); o2 = torch.empty(M, device="mps")
    _inloop_2dreduce_dt[(1,)](a2, o2, M=M, N=N, T=T); torch.mps.synchronize()
    torch.testing.assert_close(o2.cpu(), a2.cpu().sum(dim=(0, 2)), rtol=2e-3, atol=2e-3)


# --- Reduce-surface probe findings (2026-06-23): argmax-index + int-max-over-splat ---
@triton.jit
def _argmax2d_both(x, ov, oi, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    v, idx = tl.max(tl.load(x + i[:, None] * N + j[None, :]), 1, return_indices=True)
    tl.store(ov + i, v); tl.store(oi + i, idx)


@triton.jit
def _argmax2d_ax0(x, oi, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    _v, idx = tl.max(tl.load(x + i[:, None] * N + j[None, :]), 0, return_indices=True)
    tl.store(oi + j, idx)


@requires
def test_argmax2d_index_layout_refuses_broken_cases():
    # The reduce-surface probe found two 2D argmax INDEX-layout silent-wrongs (the value
    # is convert_layout'd before its store but the index is not): axis=1 with BOTH value
    # and index consumed broadcasts row-0's index; axis=0 on a SQUARE tile broadcasts
    # column-0's index. Both now refuse; rectangular axis=0 still computes.
    X = torch.randn(8, 8, device="mps")
    ov = torch.empty(8, device="mps"); oi = torch.empty(8, device="mps", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError):
        _argmax2d_both[(1,)](X, ov, oi, M=8, N=8); torch.mps.synchronize()
    oi2 = torch.empty(8, device="mps", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError):
        _argmax2d_ax0[(1,)](X, oi2, M=8, N=8); torch.mps.synchronize()  # square
    # rectangular axis=0 is correct (not over-refused)
    Xr = torch.randn(4, 8, device="mps"); oi3 = torch.empty(8, device="mps", dtype=torch.int32)
    _argmax2d_ax0[(1,)](Xr, oi3, M=4, N=8); torch.mps.synchronize()
    assert oi3.cpu().tolist() == Xr.cpu().argmax(0).tolist()


@triton.jit
def _intmax_splat(o, V: tl.constexpr, N: tl.constexpr):
    tl.store(o, tl.max(tl.full((N,), V, tl.int32), 0))


@requires
def test_int_max_over_splat_not_over_refused():
    # The tail-mask guard refused an integer max/min over a tl.full/splat (no make_range
    # to pin block_size) because it lacked an int identity. Now uses numeric_limits;
    # int max over a splat computes (== the splat value).
    o = torch.empty(1, device="mps", dtype=torch.int32)
    _intmax_splat[(1,)](o, V=7, N=32); torch.mps.synchronize()
    assert o.item() == 7


# --- Reduce-probe grind (2026-06-24): and/or + custom-2value + ND refuse ----------
@triton.jit
def _and_comb(a, b):
    return a & b


@triton.jit
def _and_reduce(a, o, N: tl.constexpr):
    tl.store(o, tl.reduce(tl.load(a + tl.arange(0, N)), 0, _and_comb))


@requires
def test_bitwise_and_reduce_never_silent_sum():
    # A pure bitwise and/or reduce silently defaulted to SUM (reduce-probe). It's now
    # detected as combine_op=and/or (only when no arithmetic/comparison combine matched,
    # so an incidental select-mask andi in a sum/cmpi reduce is NOT mis-routed — that
    # broke training). It computes via simd_and where the route supports it, else refuses
    # loudly — NEVER a silent SUM (which would give 96 here, not 3).
    a = torch.full((32,), 3, device="mps", dtype=torch.int32)
    o = torch.empty(1, device="mps", dtype=torch.int32)
    try:
        _and_reduce[(1,)](a, o, N=32); torch.mps.synchronize()
        assert o.item() == 3, f"SILENT-WRONG and-reduce: {o.item()} (sum would be 96)"
    except MetalNonRecoverableError:
        pass  # loud refusal satisfies the never-silent-wrong contract


@triton.jit
def _sum2_comb(a, b, c, d):
    return a + c, b + d


@triton.jit
def _custom_2value(a, b, oc, od, N: tl.constexpr):
    i = tl.arange(0, N)
    rc, rd = tl.reduce((tl.load(a + i), tl.load(b + i)), 0, _sum2_comb)
    tl.store(oc, rc); tl.store(od, rd)


@requires
def test_custom_2value_reduce_refuses():
    # A custom 2-value reduce (non-argmax body) was mis-routed to the argminmax path and
    # returned a single-element result (reduce-probe). Now refuses (only value+index
    # argmax/argmin 2-tuples are handled).
    a = torch.randn(64, device="mps"); b = torch.randn(64, device="mps")
    oc = torch.empty(1, device="mps"); od = torch.empty(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _custom_2value[(1,)](a, b, oc, od, N=64); torch.mps.synchronize()


@triton.jit
def _nd_sum(a, o, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    i = tl.arange(0, A); j = tl.arange(0, B); k = tl.arange(0, C); l = tl.arange(0, D)
    x = tl.load(a + i[:, None, None, None] * B * C * D + j[None, :, None, None] * C * D
                + k[None, None, :, None] * D + l[None, None, None, :])
    ii = tl.arange(0, A); kk = tl.arange(0, C); ll = tl.arange(0, D)
    tl.store(o + ii[:, None, None] * C * D + kk[None, :, None] * D + ll[None, None, :],
             tl.sum(x, 1))


@requires
def test_nd_nonxor_reduce_refuses_sort_ok():
    # A general N-D (rank>=4) non-xor reduce mis-compacts results for all axes
    # (reduce-probe). Refuse non-xor ND; tl.sort's xor ND-decomposition is unaffected.
    a = torch.randn(2, 2, 2, 2, device="mps"); o = torch.empty(2, 2, 2, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _nd_sum[(1,)](a, o, A=2, B=2, C=2, D=2); torch.mps.synchronize()
    # sort still works (xor ND reduce)
    a2 = torch.randn(2, 16, device="mps"); o2 = torch.empty(2, 16, device="mps")
    _sort_rows[(1,)](a2, o2, M=2, N=16); torch.mps.synchronize()
    torch.testing.assert_close(o2.cpu(), a2.cpu().sort(1).values, rtol=1e-4, atol=1e-4)


@triton.jit
def _sort_rows(a, o, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    tl.store(o + i[:, None] * N + j[None, :], tl.sort(tl.load(a + i[:, None] * N + j[None, :])))


# --- #3 codegen fix (2026-06-24): multi-program 3D reduce computes (pid offset) -----
@triton.jit
def _reduce3d_multiprog(a, o, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.arange(0, M); j = tl.arange(0, N); k = tl.arange(0, K)
    off = pid * M * N * K + i[:, None, None] * N * K + j[None, :, None] * K + k[None, None, :]
    x = tl.load(a + off)
    oi = tl.arange(0, M); oj = tl.arange(0, N)
    tl.store(o + pid * M * N + oi[:, None] * N + oj[None, :], tl.sum(x, 2))


@requires
def test_multiprogram_3d_reduce_computes_all_batches():
    # The 3D reduce template ignored program_id, so a multi-program dispatch silently
    # computed only batch-0 (reduce-probe #3). The template now offsets staging/store by
    # pid*total (a no-op for grid=1), so every batch computes; FA is no longer
    # mis-detected as a 3D reduce (the detector now rejects tt.dot kernels).
    P, M, N, K = 3, 4, 2, 2
    a = torch.randn(P, M, N, K, device="mps")
    o = torch.empty(P, M, N, device="mps")
    _reduce3d_multiprog[(P,)](a, o, M=M, N=N, K=K); torch.mps.synchronize()
    torch.testing.assert_close(o.cpu(), a.cpu().sum(3), rtol=2e-3, atol=2e-3)
    assert o[1].abs().sum().item() > 0, "batch 1 left untouched — pid offset missing"


# --- #7 + no-loop sibling (2026-06-24): combined 2-D reduce refuse -----------------
@triton.jit
def _combined_2d_noloop(a, o, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    x = tl.load(a + i[:, None] * N + j[None, :])
    tl.store(o + i, tl.sum(x, 1) + tl.max(x, 1))


@triton.jit
def _combined_2d_inloop(a, o, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, M); acc = tl.zeros((M,), tl.float32)
    for t in range(0, T):
        j = tl.arange(0, N)
        x = tl.load(a + t * M * N + i[:, None] * N + j[None, :])
        acc += tl.sum(x, 1) + tl.max(x, 1)
    tl.store(o + i, acc)


@triton.jit
def _separate_2d(a, os_, om, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    x = tl.load(a + i[:, None] * N + j[None, :])
    tl.store(os_ + i, tl.sum(x, 1)); tl.store(om + i, tl.max(x, 1))


@requires
def test_combined_2d_reduce_refuses_both_forms():
    # Two 2-D axis reduces COMBINED in one arithmetic expression (sum(x,1)+max(x,1))
    # mis-compute a tail subset of rows — the combined result isn't convert_layout'd
    # before the store (reduce-probe #7 in-loop form + its no-loop sibling found by the
    # confirming re-audit). Both refuse now. Separate stores (each reduce stored on its
    # own) still compute correctly — that's the supported alternative.
    M, N = 8, 64
    a = torch.randn(M, N, device="mps"); o = torch.empty(M, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _combined_2d_noloop[(1,)](a, o, M=M, N=N); torch.mps.synchronize()
    a2 = torch.randn(3, M, N, device="mps"); o2 = torch.empty(M, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _combined_2d_inloop[(1,)](a2, o2, M=M, N=N, T=3); torch.mps.synchronize()
    # separate stores compute correctly (not over-refused)
    os_ = torch.empty(M, device="mps"); om = torch.empty(M, device="mps")
    _separate_2d[(1,)](a, os_, om, M=M, N=N); torch.mps.synchronize()
    torch.testing.assert_close(os_.cpu(), a.cpu().sum(1), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(om.cpu(), a.cpu().max(1).values, rtol=2e-3, atol=2e-3)
