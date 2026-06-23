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
