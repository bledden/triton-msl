"""Structured differential sweep — the systemic anti-silent-wrong defense.

Motivation (2026-06-21 audit): three silent-wrongs shipped because the tests
covered the configurations the author *thought of*, while the bugs lived at
ROUTING BOUNDARIES nobody enumerated — a single dot with no K-loop (a separate
code path from the K-loop dot), bf16 reaching FlashAttention, a reduce with a
pre-reduce elementwise op. Per-feature tests can't cover the combinatorial
reachable-path space.

This harness instead enumerates the routing axes explicitly — op-pattern ×
dtype × output-dtype × {K-loop vs single dot} × {aligned / unaligned / boundary}
× {with / without a pre-op} — and asserts the ONE invariant that is the whole
contract:

    every cell is either numerically correct OR raises MetalNonRecoverableError
    (a loud refusal) — NEVER silently wrong, NEVER a cryptic crash.

A new silent-wrong = a cell that computes a wrong number without refusing = a
failure here. A loud refusal is a PASS (the backend chose not to mis-compute).
A non-refusal crash is a FAIL (cryptic failure is not the contract).
"""
import math
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

# Relative-error tolerance by dtype (fp32 exact; fp16/bf16 by mantissa width).
_TOL = {torch.float32: 2e-3, torch.float16: 2e-2, torch.bfloat16: 4e-2}
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _clear():
    import os
    import shutil
    for p in ("~/.cache/triton_msl", "~/.triton/cache"):
        shutil.rmtree(os.path.expanduser(p), ignore_errors=True)


def _invariant(run, reference, dtype, *, must=None):
    """Core check. ``run`` dispatches the kernel and returns the output tensor;
    ``reference`` returns the torch ground truth.

    must=None  -> "correct OR clean refusal" (the general sweep; NEW silent-wrongs
                  surface here as a divergence with no refusal).
    must="ok"  -> must compute correctly (no refusal allowed).
    must="ref" -> must refuse loudly (no silent compute allowed).
    """
    try:
        out = run()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        if must == "ok":
            pytest.fail("backend REFUSED a config that must compute correctly")
        return "refused"
    except Exception as e:  # any other exception = cryptic crash, not the contract
        pytest.fail(f"cryptic crash (not a clean MetalNonRecoverableError): "
                    f"{type(e).__name__}: {str(e)[:200]}")
    if must == "ref":
        pytest.fail("backend SILENTLY COMPUTED a config that must refuse loudly "
                    "(silent-wrong risk)")
    ref = reference()
    denom = max(ref.float().abs().max().item(), 1e-9)
    rel = (out.float() - ref.float()).abs().max().item() / denom
    assert rel < _TOL[dtype], (
        f"SILENT-WRONG: rel_err {rel:.3e} exceeds tol {_TOL[dtype]} with NO refusal")
    return "correct"


# ===========================================================================
# Matmul — K-loop path (the standard whole-kernel matmul / generic k-loop dot)
# ===========================================================================
@triton.jit
def _mm_kloop(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM); on = pn * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a + om[:, None] * K + (k + ok)[None, :]),
                      tl.load(b + (k + ok)[:, None] * N + on[None, :]))
    tl.store(c + om[:, None] * N + on[None, :], acc.to(c.dtype.element_ty))


@requires
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),       # aligned
    (2032, 64, 64),     # large unaligned-M (M%16, fast-path enablement)
    (2040, 64, 64),     # large unaligned-M (M%8)
    (96, 96, 96),       # mid unaligned
    (48, 64, 64),       # small unaligned
    (64, 128, 256),     # rectangular, K>>
])
def test_matmul_kloop(dtype, M, N, K):
    _clear()
    torch.manual_seed(0)
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    C = torch.empty(M, N, device="mps", dtype=dtype)
    BM = 32 if M % 32 == 0 else (16 if M % 16 == 0 else 8)
    _invariant(
        lambda: (_mm_kloop[(triton.cdiv(M, BM), triton.cdiv(N, 32))](
            A, B, C, M, N, K, BM=BM, BN=32, BK=32), C)[1],
        lambda: A.float() @ B.float(), dtype, must="ok")


# ===========================================================================
# Single dot, NO K-loop (the _lower_simple_dot_inline path — the #2 boundary
# the existing BLOCK=64 parity test never hit)
# ===========================================================================
@triton.jit
def _mm_single(a, b, c, S: tl.constexpr):
    om = tl.arange(0, S); on = tl.arange(0, S); ok = tl.arange(0, S)
    av = tl.load(a + om[:, None] * S + ok[None, :])
    bv = tl.load(b + ok[:, None] * S + on[None, :])
    tl.store(c + om[:, None] * S + on[None, :], tl.dot(av, bv).to(c.dtype.element_ty))


@triton.jit
def _mm_single_rect(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    av = tl.load(a + om[:, None] * K + ok[None, :])
    bv = tl.load(b + ok[:, None] * N + on[None, :])
    tl.store(c + om[:, None] * N + on[None, :], tl.dot(av, bv).to(c.dtype.element_ty))


@requires
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("out_dtype", _DTYPES)
@pytest.mark.parametrize("S", [8, 16, 32, 64])   # powers of 2 incl. non-32-multiples (partial-tile store)
def test_matmul_single_dot_no_kloop(dtype, out_dtype, S):
    # Non-32-multiple S exercises the PARTIAL-TILE store: an unmasked simdgroup_store
    # writes a full 8x8 and overflows the buffer. So check BOTH the values AND a canary
    # buffer placed right after C — an OOB write corrupts adjacent memory even when C's
    # own values happen to land correct (2026-06-21 sibling-divergence finding).
    _clear()
    torch.manual_seed(0)
    A = torch.randn(S, S, device="mps", dtype=dtype)
    B = torch.randn(S, S, device="mps", dtype=dtype)
    PAD = 64
    SENT = 2048.0   # exactly representable in fp32/fp16/bf16 (2^11) and far from any randn@randn value
    Cbuf = torch.empty(S * S + PAD, device="mps", dtype=out_dtype)
    Cbuf[S * S:] = SENT            # canary sentinel
    C = Cbuf[:S * S].view(S, S)
    tol = max(_TOL[dtype], _TOL[out_dtype])
    try:
        _mm_single[(1,)](A, B, C, S=S); torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # clean refusal acceptable
    except Exception as e:
        pytest.fail(f"cryptic crash: {type(e).__name__}: {str(e)[:200]}")
    canary_bad = int((Cbuf[S * S:] != SENT).sum().item())
    assert canary_bad == 0, (
        f"OOB WRITE: single-dot {dtype}->{out_dtype} S={S} clobbered "
        f"{canary_bad}/{PAD} bytes past the output buffer")
    ref = A.float() @ B.float()
    rel = (C.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < tol, f"SILENT-WRONG single-dot {dtype}->{out_dtype} S={S}: rel_err {rel:.3e}"


@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,N,K", [(32, 16, 32), (16, 32, 32), (16, 64, 32), (32, 16, 16)])
def test_matmul_single_dot_rect(dtype, M, N, K):
    # Non-square partial tiles: M%32==0 & N%32!=0 (partial cols), and vice-versa. This is
    # the exact OOB shape the sibling-divergence sweep found in the unmasked float store.
    _clear()
    torch.manual_seed(0)
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    PAD = 64
    SENT = 2048.0
    Cbuf = torch.empty(M * N + PAD, device="mps", dtype=dtype)
    Cbuf[M * N:] = SENT
    C = Cbuf[:M * N].view(M, N)
    try:
        _mm_single_rect[(1,)](A, B, C, M=M, N=N, K=K); torch.mps.synchronize()
    except MetalNonRecoverableError:
        return
    except Exception as e:
        pytest.fail(f"cryptic crash: {type(e).__name__}: {str(e)[:200]}")
    canary_bad = int((Cbuf[M * N:] != SENT).sum().item())
    assert canary_bad == 0, f"OOB WRITE: rect dot {dtype} {M}x{N}x{K} clobbered {canary_bad}/{PAD} past buffer"
    ref = A.float() @ B.float()
    rel = (C.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < _TOL[dtype], f"SILENT-WRONG rect dot {dtype} {M}x{N}x{K}: rel_err {rel:.3e}"


# ===========================================================================
# 2D reduce, with and without a pre-reduce elementwise op
# ===========================================================================
@triton.jit
def _reduce2d(a, o, s, C: tl.constexpr, OP: tl.constexpr, SCALE: tl.constexpr):
    r = tl.program_id(0); cc = tl.arange(0, C)
    x = tl.load(a + r * C + cc)
    if SCALE:
        x = x * s
    if OP == 0:
        v = tl.sum(x, 0)
    else:
        v = tl.max(x, 0)
    tl.store(o + r, v)


@requires
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op", [0, 1])         # 0=sum, 1=max
@pytest.mark.parametrize("scale", [False, True])
def test_reduce_2d(dtype, op, scale):
    _clear()
    torch.manual_seed(0)
    R, C = 8, 64
    a = torch.randn(R, C, device="mps", dtype=dtype)
    o = torch.empty(R, device="mps", dtype=torch.float32)
    s = 2.0
    def run():
        _reduce2d[(R,)](a, o, s, C=C, OP=op, SCALE=scale)
        return o
    def ref():
        x = a.float() * (s if scale else 1.0)
        return x.sum(-1) if op == 0 else x.max(-1).values
    _invariant(run, ref, dtype)   # correct OR refuse; pre-op must not be silently dropped


# ===========================================================================
# 3D reduce, with and without a pre-op (the #1 boundary — pre-op must refuse)
# ===========================================================================
@triton.jit
def _reduce3d(a, o, s, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr):
    bb = tl.arange(0, B); rr = tl.arange(0, R); cc = tl.arange(0, C)
    x = tl.load(a + bb[:, None, None] * R * C + rr[None, :, None] * C + cc[None, None, :])
    if SCALE:
        x = x * s
    tl.store(o + bb[:, None] * R + rr[None, :], tl.sum(x, axis=2))


@requires
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scale", [False, True])
def test_reduce_3d(dtype, scale):
    _clear()
    torch.manual_seed(0)
    B, R, C = 2, 4, 4
    a = torch.arange(B * R * C, device="mps", dtype=dtype).reshape(B, R, C)
    o = torch.empty(B, R, device="mps", dtype=torch.float32)
    s = 2.0
    def run():
        _reduce3d[(1,)](a, o, s, B=B, R=R, C=C, SCALE=scale)
        return o
    def ref():
        x = a.float() * (s if scale else 1.0)
        return x.sum(-1)
    # No pre-op: must compute correctly. Pre-op: both reduce paths mis-handle it,
    # so it MUST refuse (never silently drop the op).
    _invariant(run, ref, dtype, must=("ref" if scale else "ok"))


# ===========================================================================
# FlashAttention pattern × dtype (the #3 boundary — bf16 must refuse)
# ===========================================================================
@triton.jit
def _fa(Q, K, V, Out, sqz, sqh, sqm, sqk, skz, skh, skn, skk, svz, svh, svn, svk,
        soz, soh, som, sok, Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr, BF16: tl.constexpr):
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
        if BF16:
            qk = tl.dot(q.to(tl.bfloat16), tl.trans(k).to(tl.bfloat16)).to(tl.float32)
        else:
            qk = tl.dot(q, tl.trans(k).to(q.dtype))
        qk = qk * (1.0 / math.sqrt(HEAD_DIM))
        mij = tl.max(qk, 1); mn = tl.maximum(mi, mij); al = tl.exp(mi - mn); p = tl.exp(qk - mn[:, None])
        li = li * al + tl.sum(p, 1); acc = acc * al[:, None]
        v = tl.load(V + oz * svz + oh * svh + (snn + on)[:, None] * svn + od[None, :] * svk,
                    mask=(snn + on)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); mi = mn
    tl.store(Out + oz * soz + oh * soh + om[:, None] * som + od[None, :] * sok,
             (acc / li[:, None]).to(Out.dtype.element_ty), mask=om[:, None] < N_CTX)


@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_flash_attention_dtype(dtype, head_dim):
    _clear()
    torch.manual_seed(0)
    Z, H, N, D = 1, 1, 64, head_dim
    bf16 = dtype == torch.bfloat16
    base = torch.float16 if dtype == torch.float16 else torch.float32
    q = torch.randn(Z, H, N, D, device="mps", dtype=base)
    k = torch.randn(Z, H, N, D, device="mps", dtype=base)
    v = torch.randn(Z, H, N, D, device="mps", dtype=base)
    o = torch.empty_like(q)
    def run():
        _fa[(N // 32, Z * H)](q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                              Z, H, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=D, BF16=bf16)
        return o
    def ref():
        return (torch.softmax((q.float() * (1.0 / math.sqrt(D))) @ k.float().transpose(-2, -1), -1)
                @ v.float())
    # bf16 FA MUST refuse loudly (the attention lowering is fp16/fp32 only — this pins
    # the #3 fix). For fp32/fp16 the contract is only never-silent-wrong: the backend
    # may compute OR refuse (this exact kernel shape happens to hit the matmul "constexpr
    # M/N" refusal — a legitimate loud refusal). fp32/fp16 FA *correctness* is covered by
    # test_flash_attention.py with a detector-recognized FA kernel.
    _invariant(run, ref, dtype, must=("ref" if bf16 else None))


# ===========================================================================
# Associative scan / cumsum (a past silent-wrong lived at block > 1024)
# ===========================================================================
@triton.jit
def _cumsum(a, o, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(o + i, tl.cumsum(tl.load(a + i), 0))


@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("BLOCK", [64, 256, 1024, 2048])   # 2048 > the 1024 boundary
def test_cumsum(dtype, BLOCK):
    _clear()
    torch.manual_seed(0)
    a = torch.randn(BLOCK, device="mps", dtype=dtype)
    o = torch.empty(BLOCK, device="mps", dtype=dtype)
    _invariant(lambda: (_cumsum[(1,)](a, o, BLOCK=BLOCK), o)[1],
               lambda: a.float().cumsum(0), dtype)


# ===========================================================================
# Atomic add (fp16/bf16 atomics are a known-tricky surface)
# ===========================================================================
@triton.jit
def _atomic_add(a, o, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK); m = i < n
    tl.atomic_add(o + (i % 4), tl.load(a + i, mask=m, other=0.0), mask=m)


@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_atomic_add(dtype):
    _clear()
    torch.manual_seed(0)
    n = 64
    a = torch.randn(n, device="mps", dtype=dtype)
    o = torch.zeros(4, device="mps", dtype=dtype)
    def ref():
        r = torch.zeros(4, dtype=torch.float32)
        idx = (torch.arange(n) % 4)
        r.scatter_add_(0, idx, a.float().cpu())
        return r.to("mps")
    _invariant(lambda: (_atomic_add[(1,)](a, o, n, BLOCK=n), o)[1], ref, dtype)
