"""Seeded property-based differential fuzzer — the systemic "find a 5th silent-wrong".

The audit + the sibling sweep found four silent-wrongs, all at routing boundaries
nobody enumerated. Hand-written tests (even the structured differential harness)
only cover cells someone thought to list — the harness itself missed the float-store
OOB because it only swept S in {32,64}.

This fuzzer removes the human from cell-selection: each seed generates a RANDOM config
(op x dtype x out-dtype x shapes incl odd/boundary/large x block x pre-op) over a set
of CORRECTLY-MASKED templates, runs it on Metal, and asserts the one invariant:

    correct (matches torch) OR loud MetalNonRecoverableError — never silently wrong,
    never a cryptic crash, never an OOB write (canary-checked for the store paths).

The templates are deliberately simple + fully masked so a failure is a BACKEND bug,
not a template bug. On failure the seed + decoded config are printed -> reproducible.

torch is the reference (the semantics Triton targets); for the ops here (matmul,
reduce, elementwise) the torch reference is unimpeachable.
"""
import math
import random
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

_TOL = {torch.float32: 3e-3, torch.float16: 3e-2, torch.bfloat16: 6e-2}
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_POW2 = [8, 16, 32, 64]
_SENT = 2048.0   # exactly representable in fp32/fp16/bf16, far from any randn value


def _clear():
    import os, shutil
    # Clear ONLY the triton-msl codegen cache (force re-codegen). Do NOT delete
    # ~/.triton/cache: it is content-addressed and shared, and deleting it per-test
    # races a sibling test's in-flight make_metallib pipeline -> nondeterministic
    # FileNotFoundError mislabeled as a "cryptic crash" (2026-06-22 re-audit).
    shutil.rmtree(os.path.expanduser("~/.cache/triton_msl"), ignore_errors=True)


# --------------------------------------------------------------------------- #
# Correctly-masked templates (the kernels under test)
# --------------------------------------------------------------------------- #
@triton.jit
def _mm(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a_ = tl.load(a + rm[:, None] * sam + kk[None, :] * sak,
                     mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b_ = tl.load(b + kk[:, None] * sbk + rn[None, :] * sbn,
                     mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a_, b_)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(c.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _single_dot(a, b, c, S: tl.constexpr):
    om = tl.arange(0, S); on = tl.arange(0, S); ok = tl.arange(0, S)
    av = tl.load(a + om[:, None] * S + ok[None, :])
    bv = tl.load(b + ok[:, None] * S + on[None, :])
    tl.store(c + om[:, None] * S + on[None, :], tl.dot(av, bv).to(c.dtype.element_ty))


@triton.jit
def _reduce(a, o, s, R, C, BC: tl.constexpr, OP: tl.constexpr, SCALE: tl.constexpr):
    r = tl.program_id(0); cc = tl.arange(0, BC)
    if OP == 0:
        other = 0.0
    elif OP == 1:
        other = float("-inf")
    else:
        other = float("inf")
    x = tl.load(a + r * C + cc, mask=cc < C, other=other)
    if SCALE:
        x = x * s
    if OP == 0:
        v = tl.sum(x, 0)
    elif OP == 1:
        v = tl.max(x, 0)
    else:
        v = tl.min(x, 0)
    tl.store(o + r, v)


@triton.jit
def _ew(a, b, o, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK); m = i < n
    x = tl.load(a + i, mask=m, other=0.0); y = tl.load(b + i, mask=m, other=0.0)
    if OP == 0:
        z = x + y
    elif OP == 1:
        z = x * y
    elif OP == 2:
        z = tl.maximum(x, y)
    elif OP == 3:
        z = tl.where(x > y, x, y)
    else:
        z = tl.exp(x.to(tl.float32))   # Triton frontend rejects exp on fp16/bf16; exp in fp32
    tl.store(o + i, z, mask=m)


# --------------------------------------------------------------------------- #
# Per-op runners: return (metal_out_float, torch_ref_float, canary_bad_or_None)
# or raise MetalNonRecoverableError (a clean refusal == pass).
# --------------------------------------------------------------------------- #
def _run_matmul(rng):
    dt = rng.choice(_DTYPES); out = rng.choice(_DTYPES)
    M = rng.choice([8, 16, 31, 32, 33, 48, 64, 100, 127, 128, 257, 512])
    N = rng.choice([8, 16, 31, 32, 48, 64, 96, 128, 200, 256])
    K = rng.choice([8, 16, 24, 32, 40, 64, 128, 256])
    BM = rng.choice(_POW2); BN = rng.choice(_POW2); BK = rng.choice([8, 16, 32])
    A = torch.randn(M, K, device="mps", dtype=dt); B = torch.randn(K, N, device="mps", dtype=dt)
    C = torch.empty(M, N, device="mps", dtype=out)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
              C.stride(0), C.stride(1), BM=BM, BN=BN, BK=BK)
    return C.float(), A.float() @ B.float(), None, max(_TOL[dt], _TOL[out]), \
        f"matmul {dt}->{out} M{M} N{N} K{K} BM{BM} BN{BN} BK{BK}"


def _run_single_dot(rng):
    dt = rng.choice(_DTYPES); out = rng.choice(_DTYPES); S = rng.choice(_POW2)
    A = torch.randn(S, S, device="mps", dtype=dt); B = torch.randn(S, S, device="mps", dtype=dt)
    PAD = 64
    Cbuf = torch.empty(S * S + PAD, device="mps", dtype=out); Cbuf[S * S:] = _SENT
    C = Cbuf[:S * S].view(S, S)
    _single_dot[(1,)](A, B, C, S=S)
    canary = int((Cbuf[S * S:] != _SENT).sum().item())
    return C.float(), A.float() @ B.float(), canary, max(_TOL[dt], _TOL[out]), \
        f"single_dot {dt}->{out} S{S}"


def _run_reduce(rng):
    dt = rng.choice(_DTYPES); op = rng.choice([0, 1, 2]); scale = rng.choice([False, True])
    R = rng.choice([1, 4, 8, 16]); C = rng.choice([7, 8, 16, 31, 32, 64, 100, 128])
    BC = 1 << (C - 1).bit_length()          # smallest pow2 >= C
    a = torch.randn(R, C, device="mps", dtype=dt); o = torch.empty(R, device="mps", dtype=torch.float32)
    s = 2.0
    _reduce[(R,)](a, o, s, R, C, BC=BC, OP=op, SCALE=scale)
    x = a.float() * (s if scale else 1.0)
    ref = x.sum(-1) if op == 0 else (x.max(-1).values if op == 1 else x.min(-1).values)
    return o.float(), ref, None, _TOL[dt], f"reduce {dt} op{op} scale{scale} R{R} C{C} BC{BC}"


def _run_ew(rng):
    dt = rng.choice(_DTYPES); op = rng.choice([0, 1, 2, 3, 4])
    n = rng.choice([7, 8, 31, 32, 100, 128, 257, 1000])
    BLOCK = 1 << (n - 1).bit_length()
    a = torch.randn(n, device="mps", dtype=dt); b = torch.randn(n, device="mps", dtype=dt)
    o = torch.empty(n, device="mps", dtype=dt)
    _ew[(1,)](a, b, o, n, OP=op, BLOCK=BLOCK)
    af, bf = a.float(), b.float()
    ref = {0: af + bf, 1: af * bf, 2: torch.maximum(af, bf),
           3: torch.where(af > bf, af, bf), 4: torch.exp(af)}[op]
    return o.float(), ref, None, _TOL[dt], f"ew {dt} op{op} n{n} BLOCK{BLOCK}"


@triton.jit
def _scan(a, o, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(o + i, tl.cumsum(tl.load(a + i), 0))


@triton.jit
def _transpose(a, o, S: tl.constexpr):
    i = tl.arange(0, S); j = tl.arange(0, S)
    x = tl.load(a + i[:, None] * S + j[None, :])
    tl.store(o + i[:, None] * S + j[None, :], tl.trans(x))


@triton.jit
def _f2i(a, o, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK); m = i < n
    tl.store(o + i, tl.load(a + i, mask=m, other=0.0).to(tl.int32), mask=m)


@triton.jit
def _atomic(a, o, n, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK); m = i < n
    tl.atomic_add(o + (i % K), tl.load(a + i, mask=m, other=0.0), mask=m)


def _run_scan(rng):
    # fp32 only: fp16 cumsum accumulation error vs an fp32 reference is ambiguous
    # (not a silent-wrong), and would false-positive. BLOCK=2048 (>1024) should refuse.
    dt = torch.float32; BLOCK = rng.choice([8, 32, 256, 1024, 2048])
    a = torch.randn(BLOCK, device="mps", dtype=dt)
    o = torch.empty(BLOCK, device="mps", dtype=dt)
    _scan[(1,)](a, o, BLOCK=BLOCK)
    return o.float(), a.float().cumsum(0), None, 1e-3, f"scan {dt} BLOCK{BLOCK}"


def _run_transpose(rng):
    dt = rng.choice(_DTYPES); S = rng.choice(_POW2)
    a = torch.randn(S, S, device="mps", dtype=dt); o = torch.empty(S, S, device="mps", dtype=dt)
    _transpose[(1,)](a, o, S=S)
    return o.float(), a.float().t().contiguous(), None, _TOL[dt], f"transpose {dt} S{S}"


def _run_f2i(rng):
    n = rng.choice([7, 8, 32, 100, 128, 257]); BLOCK = 1 << (n - 1).bit_length()
    a = (torch.randn(n, device="mps") * 50.0)   # fractional, +/- ~150 -> tests trunc-toward-zero
    o = torch.empty(n, device="mps", dtype=torch.int32)
    _f2i[(1,)](a, o, n, BLOCK=BLOCK)
    # values ~+/-150 ints; any trunc/rounding error is an absolute diff >= 1
    # (rel >= ~1/150), so a small rel tol catches it while exact matches give rel 0.
    return o.float(), a.to(torch.int32).float(), None, 1e-3, f"f2i n{n} BLOCK{BLOCK}"


def _run_atomic(rng):
    dt = rng.choice(_DTYPES); n = rng.choice([8, 32, 64, 128]); K = rng.choice([1, 2, 4, 8])
    # bf16 atomic-add accumulates in bf16; cap additions-per-bucket (n/K) so the
    # accumulation error stays well inside tol and a loose bound can't HIDE a real
    # mis-accumulation (2026-06-22 re-audit P4). fp16/fp32 are fine at any K.
    if dt == torch.bfloat16:
        K = max(K, 4, 1 << (max(1, n // 32) - 1).bit_length())
    BLOCK = 1 << (n - 1).bit_length()
    a = torch.randn(n, device="mps", dtype=dt); o = torch.zeros(K, device="mps", dtype=dt)
    _atomic[(1,)](a, o, n, K, BLOCK=BLOCK)
    ref = torch.zeros(K, dtype=torch.float32)
    ref.scatter_add_(0, torch.arange(n) % K, a.float().cpu())
    return o.float(), ref.to("mps"), None, _TOL[dt] * 4, f"atomic {dt} n{n} K{K}"


@triton.jit
def _gather1d_fuzz(src, idx, o, S: tl.constexpr):
    i = tl.arange(0, S)
    s = tl.load(src + i)
    ix = tl.load(idx + i)
    tl.store(o + i, tl.gather(s, ix, axis=0))


@triton.jit
def _gather2d_fuzz(src, idx, out, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    s = tl.load(src + i[:, None] * N + j[None, :])
    ix = tl.load(idx + i[:, None] * N + j[None, :])
    r = tl.gather(s, ix, axis=0)
    tl.store(out + i[:, None] * N + j[None, :], r)


@triton.jit
def _cat_fuzz(a, b, out, N: tl.constexpr):
    i = tl.arange(0, N)
    av = tl.load(a + i)
    bv = tl.load(b + i)
    z = tl.cat(av, bv, can_reorder=True)
    j = tl.arange(0, 2 * N)
    tl.store(out + j, z)


_GATHER_SIZES_1D = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
_GATHER_SHAPES_2D = [
    (4, 8), (8, 8), (8, 16), (16, 16), (16, 32), (32, 16),
    (32, 32), (32, 64), (64, 32), (64, 64),
]


def _run_gather1d(rng):
    """1-D tt.gather: S elements staged one-per-thread; S >1024 must refuse."""
    S = rng.choice(_GATHER_SIZES_1D)
    src = torch.arange(S, device="mps", dtype=torch.float32) + 0.5
    idx = torch.randint(0, S, (S,), device="mps", dtype=torch.int32)
    o = torch.empty(S, device="mps", dtype=torch.float32)
    _gather1d_fuzz[(1,)](src, idx, o, S=S)
    ref = src.cpu()[idx.cpu().long()]
    return o.float().cpu(), ref, None, 1e-4, f"gather1d S={S}"


def _run_gather2d(rng):
    """2-D tt.gather axis=0: out[i,j]=src[idx[i,j],j]; M*N >1024 must refuse."""
    M, N = rng.choice(_GATHER_SHAPES_2D)
    src = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N) + 0.5
    idx = torch.randint(0, M, (M, N), dtype=torch.int32, device="mps")
    out = torch.empty(M, N, dtype=torch.float32, device="mps")
    _gather2d_fuzz[(1,)](src, idx, out, M=M, N=N)
    # Reference via torch.gather (equivalent to loop over rows/cols for axis=0)
    ref = torch.gather(src.cpu(), 0, idx.cpu().long())
    return out.float().cpu(), ref, None, 1e-4, f"gather2d M={M} N={N}"


def _run_cat(rng):
    """tl.cat(a,b,can_reorder=True): out[:N]=a, out[N:]=b.
    When 2*N > num_warps*32 the wrapping-loop staging uses lid (wrong) instead of
    _loop_e, so out[N:] silently reads OOB from a instead of copying b.
    """
    N = rng.choice([8, 16, 32, 64, 128, 256, 512, 1024])
    dt = rng.choice([torch.float32])   # fp32 only: sign distinguishes a vs b cleanly
    # Use distinctly-signed a/b so positional errors are unmistakable
    a = (torch.arange(N, dtype=dt, device="mps") + 100.0)
    b = -(torch.arange(N, dtype=dt, device="mps") + 1.0)
    out = torch.zeros(2 * N, dtype=dt, device="mps")
    _cat_fuzz[(1,)](a, b, out, N=N)
    # Reference: cat([a,b]) along dim=0
    ref = torch.cat([a.cpu(), b.cpu()])
    return out.float().cpu(), ref.float(), None, 1e-4, f"cat N={N}"


_RUNNERS = [_run_matmul, _run_single_dot, _run_reduce, _run_ew,
            _run_scan, _run_transpose, _run_f2i, _run_atomic,
            _run_gather1d, _run_gather2d, _run_cat]


@requires
@pytest.mark.parametrize("seed", range(200))
def test_fuzz(seed):
    rng = random.Random(seed)
    runner = rng.choice(_RUNNERS)
    _clear()
    try:
        out, ref, canary, tol, desc = runner(rng)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # loud refusal satisfies the contract
    except Exception as e:
        pytest.fail(f"[seed {seed}] cryptic crash (not a clean refusal): "
                    f"{type(e).__name__}: {str(e)[:200]}")
    if canary is not None and canary != 0:
        pytest.fail(f"[seed {seed}] OOB WRITE ({canary} bytes past buffer): {desc}")
    rel = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < tol, f"[seed {seed}] SILENT-WRONG: rel_err {rel:.3e} > {tol}: {desc}"


# --- The 5th silent-wrong (found by this fuzzer, 2026-06-22): a 2-D tt.trans of
#     a block > the 1024-thread threadgroup silently mis-computed (the shared
#     exchange maps one element per thread; elements past 1024 were left
#     untransposed). Now refuses. Pinned explicitly. -----------------------------
@triton.jit
def _trans_sq(a, o, S: tl.constexpr):
    i = tl.arange(0, S); j = tl.arange(0, S)
    tl.store(o + i[:, None] * S + j[None, :], tl.trans(tl.load(a + i[:, None] * S + j[None, :])))


@requires
def test_transpose_within_threadgroup_correct():
    S = 32  # 1024 elements == cap
    torch.manual_seed(0)
    a = torch.randn(S, S, device="mps"); o = torch.empty(S, S, device="mps")
    _trans_sq[(1,)](a, o, S=S); torch.mps.synchronize()
    torch.testing.assert_close(o, a.t().contiguous(), rtol=1e-4, atol=1e-4)


@requires
def test_transpose_over_threadgroup_refuses():
    S = 64  # 4096 elements > 1024-thread cap -> must refuse, never silently transpose only the first 1024
    a = torch.randn(S, S, device="mps"); o = torch.empty(S, S, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _trans_sq[(1,)](a, o, S=S)
        torch.mps.synchronize()


# --- The 6th + 7th silent-wrongs (found by the re-audit, 2026-06-22) ----------
@triton.jit
def _gather1d(src, idx, o, S: tl.constexpr):
    i = tl.arange(0, S); s = tl.load(src + i); ix = tl.load(idx + i)
    tl.store(o + i, tl.gather(s, ix, axis=0))


@requires
def test_gather1d_within_threadgroup_correct():
    # 6th: 1-D gather at S=512 (<=1024) silently mis-gathered (under MEPT the
    # one-element-per-thread staging filled only ~128 slots; indices past that read
    # uninitialized memory). Now gather forces one-element-per-thread (has_barrier_ops).
    S = 512
    torch.manual_seed(0)
    src = torch.arange(S, device="mps", dtype=torch.float32) + 0.5
    idx = torch.randint(0, S, (S,), device="mps", dtype=torch.int32)
    o = torch.empty(S, device="mps")
    _gather1d[(1,)](src, idx, o, S=S); torch.mps.synchronize()
    torch.testing.assert_close(o.cpu(), src.cpu()[idx.cpu().long()], rtol=1e-4, atol=1e-4)


@requires
def test_gather1d_over_threadgroup_refuses():
    S = 2048  # > 1024-thread cap: one-element-per-thread staging can't cover -> must refuse
    src = torch.arange(S, device="mps", dtype=torch.float32)
    idx = torch.arange(S, device="mps", dtype=torch.int32)
    o = torch.empty(S, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _gather1d[(1,)](src, idx, o, S=S)
        torch.mps.synchronize()


@triton.jit
def _q_fp16(a, o, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(a + i).to(tl.float16)
    tl.store(o + i, (x * 1.0).to(tl.float32))   # *1.0 forces USE of the fp16 register


@requires
def test_truncf_scalar_quantizes():
    # 7th: an explicit mid-computation `.to(tl.float16)` was a passthrough — 2049.0
    # stayed 2049.0 instead of quantizing to fp16's 2048.0. Scalar path now quantizes.
    a = torch.zeros(4, device="mps"); a[:3] = torch.tensor([2049.0, 4097.0, 100.0])
    o = torch.empty(4, device="mps")
    _q_fp16[(1,)](a, o, N=4); torch.mps.synchronize()
    torch.testing.assert_close(o[:3].cpu(), a[:3].half().float().cpu(), rtol=1e-4, atol=1e-4)


# --- Re-audit 2: CLASS-A gather + cat coverage (2026-06-22) ------------------
#
# 1-D gather: already pinned above (tests 6+7). Added to the fuzz runner.
#
# 2-D gather: sizes within the 1024-thread cap are correct; larger are refused.
@triton.jit
def _gather2d_pin(src, idx, out, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M); j = tl.arange(0, N)
    s = tl.load(src + i[:, None] * N + j[None, :])
    ix = tl.load(idx + i[:, None] * N + j[None, :])
    tl.store(out + i[:, None] * N + j[None, :], tl.gather(s, ix, axis=0))


@requires
def test_gather2d_within_threadgroup_correct():
    for M, N in [(4, 8), (8, 16), (16, 32), (32, 32)]:
        torch.manual_seed(0)
        src = (torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N) + 0.5)
        idx = torch.randint(0, M, (M, N), dtype=torch.int32, device="mps")
        out = torch.empty(M, N, dtype=torch.float32, device="mps")
        _gather2d_pin[(1,)](src, idx, out, M=M, N=N); torch.mps.synchronize()
        ref = torch.gather(src.cpu(), 0, idx.cpu().long())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4), \
            f"2D gather M={M} N={N} SILENT-WRONG"


@requires
def test_gather2d_over_threadgroup_refuses():
    for M, N in [(32, 64), (64, 32), (64, 64)]:
        src = torch.zeros(M, N, device="mps")
        idx = torch.zeros(M, N, dtype=torch.int32, device="mps")
        out = torch.empty(M, N, device="mps")
        with pytest.raises(MetalNonRecoverableError):
            _gather2d_pin[(1,)](src, idx, out, M=M, N=N); torch.mps.synchronize()


# --- The 8th/9th/10th silent-wrongs (re-audit #2, 2026-06-22): tt.cat / tt.join /
#     tt.split threadgroup-limit (CLASS A) + truncf wrap-loop quantize (CLASS B) ---
@triton.jit
def _cat_reorder(a, b, o, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o + tl.arange(0, 2 * N), tl.cat(tl.load(a + i), tl.load(b + i), can_reorder=True))


@triton.jit
def _cat_concat(a, b, o, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o + tl.arange(0, 2 * N), tl.cat(tl.load(a + i), tl.load(b + i), can_reorder=False))


@requires
@pytest.mark.parametrize("fn", [_cat_reorder, _cat_concat])
def test_cat_within_threadgroup_correct(fn):
    # 2*N <= 1024: must concat correctly. A POISONED guard region after the output
    # catches an OOB write that would otherwise be masked (the cat bug was hidden by
    # an OOB read landing in the adjacent buffer).
    N = 256
    torch.manual_seed(0)
    a = torch.arange(N, device="mps", dtype=torch.float32)
    b = torch.arange(N, 2 * N, device="mps", dtype=torch.float32)
    PAD = 64
    obuf = torch.full((2 * N + PAD,), float("nan"), device="mps"); obuf[2 * N:] = 1234.0
    o = obuf[:2 * N]
    fn[(1,)](a, b, o, N=N); torch.mps.synchronize()
    assert int((obuf[2 * N:] != 1234.0).sum().item()) == 0, "OOB write past cat output"
    torch.testing.assert_close(o.cpu(), torch.cat([a.cpu(), b.cpu()]), rtol=1e-4, atol=1e-4)


@requires
@pytest.mark.parametrize("fn", [_cat_reorder, _cat_concat])
def test_cat_over_threadgroup_refuses(fn):
    N = 1024  # 2*N = 2048 > 1024-thread cap -> must refuse, never leave the high half unwritten
    a = torch.arange(N, device="mps", dtype=torch.float32)
    b = torch.arange(N, 2 * N, device="mps", dtype=torch.float32)
    o = torch.empty(2 * N, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        fn[(1,)](a, b, o, N=N); torch.mps.synchronize()


@triton.jit
def _q_fp16_big(a, o, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(a + i).to(tl.float16)
    tl.store(o + i, (x * 2.0).to(tl.float32))   # *2 forces USE of the fp16 register


@requires
@pytest.mark.parametrize("N", [64, 4096])   # 64 = scalar, 4096 = wrap-loop
def test_truncf_quantizes_scalar_and_wraploop(N):
    # 10th: `.to(tl.float16)` mid-compute must quantize in BOTH scalar and wrap-loop
    # mode (the fix had over-excluded wrap-loop). 2049.0 -> fp16 2048.0 -> *2 = 4096.0.
    a = torch.full((N,), 2049.0, device="mps")
    o = torch.empty(N, device="mps")
    _q_fp16_big[(1,)](a, o, N=N); torch.mps.synchronize()
    assert abs(o[0].item() - 4096.0) < 1.0, f"N={N} did not quantize: {o[0].item()}"
