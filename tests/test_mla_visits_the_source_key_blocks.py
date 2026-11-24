"""The MLA forwards visit exactly the key blocks their source's loop visits.

``tests.test_fa_asymmetric._mla_prescaled_fwd`` (nope and rope: two QK dots) loops
``for s in range(0, N_CTX, BN)`` and puts the causal mask on the SCORE. A key the mask hides gets
probability zero, and its value still enters ``tt.dot(p, v, acc * alpha)``: 0 * NaN and 0 * inf
are NaN. The MLA route emitted the causal work-skip (``if (kv_start > q_start + BM - 1u) break;``),
so query rows 0-63 never read keys 64-127 and returned 1.0 where the source gives NaN.

``tests.test_fa_asymmetric._mla_fwd`` (one QK dot, Q.K and V head dimensions different) clamps its
own loop at ``min((start_m + 1) * BLOCK_M, N_CTX)``. Its route keeps the work-skip, at the source's
own stop and key block: with BLOCK_N = 32 a 64-key tile would read 32 keys the source never loads,
so rows 0-31 must not see key 40 at all.

Q and K are zero (every score is zero) and V is one, except V[key, 0], which is NaN or +inf. By the
source, feature 0 of a row whose loop loads ``key`` is NaN (a NaN value, or an infinity times the
probability zero of a hidden key) or +inf (an infinity at a key the row sees); a row whose loop does
not load the key averages ones. Finite random controls compare with float64 causal attention.
"""

import math

import pytest
import torch

from tests.test_fa_tail_semantics import _executed_source, executed  # noqa: F401
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal device")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from tests.test_fa_asymmetric import _mla_fwd, _mla_prescaled_fwd  # noqa: E402

N, H, BM, BN = 128, 1, 32, 32
DN, DR, DV = 64, 64, 128  # two-dot: nope, rope, value
DQK, DV1 = 128, 64  # one-dot: Q.K, value
SHADER = "simdgroup-MMA FlashAttention-2"
TYPES = {"f32": torch.float32, "f16": torch.float16}


@triton.jit
def _mla_loop_fwd(
    Qn,
    Qr,
    Kn,
    Kr,
    V,
    Out,
    qnz,
    qnh,
    qnm,
    qnk,
    qrz,
    qrh,
    qrm,
    qrk,
    knz,
    knh,
    knn,
    knk,
    krz,
    krh,
    krn,
    krk,
    vz,
    vh,
    vn,
    vk,
    oz,
    oh,
    om,
    ok_,
    Z,
    H,
    N_CTX,
    BM: tl.constexpr,
    BN: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    DV: tl.constexpr,
    CA: tl.constexpr,
    CLAMP: tl.constexpr,
    LO: tl.constexpr,
):
    # _mla_prescaled_fwd with its key loop as a parameter: CLAMP stops it at the query tile's end,
    # LO is its first key.
    pm = tl.program_id(0)
    hz = tl.program_id(1)
    z = hz // H
    h = hz % H
    m = pm * BM + tl.arange(0, BM)
    n0 = tl.arange(0, BN)
    dn = tl.arange(0, DN)
    dr = tl.arange(0, DR)
    dv = tl.arange(0, DV)
    scale = 1.0 / tl.sqrt(float(DN + DR))
    qn = tl.load(Qn + z * qnz + h * qnh + m[:, None] * qnm + dn[None, :] * qnk) * scale
    qr = tl.load(Qr + z * qrz + h * qrh + m[:, None] * qrm + dr[None, :] * qrk) * scale
    acc = tl.zeros([BM, DV], dtype=tl.float32)
    mi = tl.full([BM], float("-inf"), tl.float32)
    li = tl.zeros([BM], tl.float32)
    hi = N_CTX
    if CLAMP:
        hi = min((pm + 1) * BM, N_CTX)
    for s in range(LO, hi, BN):
        kn = tl.load(Kn + z * knz + h * knh + (s + n0)[:, None] * knn + dn[None, :] * knk)
        kr = tl.load(Kr + z * krz + h * krh + (s + n0)[:, None] * krn + dr[None, :] * krk)
        qk = tl.dot(qn, tl.trans(kn).to(qn.dtype)) + tl.dot(qr, tl.trans(kr).to(qr.dtype))
        if CA:
            qk = tl.where(m[:, None] >= (s + n0)[None, :], qk, float("-inf"))
        mij = tl.max(qk, 1)
        mnew = tl.maximum(mi, mij)
        al = tl.exp(mi - mnew)
        p = tl.exp(qk - mnew[:, None])
        li = li * al + tl.sum(p, 1)
        acc = acc * al[:, None]
        v = tl.load(V + z * vz + h * vh + (s + n0)[:, None] * vn + dv[None, :] * vk)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        mi = mnew
    acc = acc / li[:, None]
    tl.store(Out + z * oz + h * oh + m[:, None] * om + dv[None, :] * ok_, acc.to(Out.dtype.element_ty))


def _inputs(form, dtype, n=N, seed=None, key=None, bad=None):
    """(Q roles, K roles, V) for a form: zeros and ones with V[key, 0] = bad, or seeded random."""
    t = TYPES[dtype]
    widths = (DQK, DQK, DV1) if form == "one-dot" else (DN, DR, DN, DR, DV)
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        return tuple(torch.randn((1, H, n, d), generator=g).to(t).to("mps") for d in widths)
    x = [torch.zeros((1, H, n, d), dtype=t) for d in widths[:-1]] + [torch.ones((1, H, n, widths[-1]), dtype=t)]
    if key is not None:
        x[-1][0, :, key, 0] = bad
    return tuple(a.to("mps") for a in x)


def _launch(form, x, causal=True, clamp=False, lo=0):
    """form: "two-dot" (_mla_prescaled_fwd), "one-dot" (_mla_fwd) or "loop" (_mla_loop_fwd)."""
    n, dv = x[0].shape[2], x[-1].shape[3]
    o = torch.full((1, H, n, dv), -99.0, dtype=x[-1].dtype, device="mps")
    st = [s for t in (*x, o) for s in t.stride()]
    grid = (n // BM, H)
    if form == "one-dot":
        compiled = _mla_fwd[grid](*x, o, *st, 1, H, n, BM, BN, DQK, DV1, causal)
    elif form == "two-dot":
        compiled = _mla_prescaled_fwd[grid](*x, o, *st, 1, H, n, BM, BN, DN, DR, DV, causal)
    else:
        compiled = _mla_loop_fwd[grid](*x, o, *st, 1, H, n, BM, BN, DN, DR, DV, causal, clamp, lo)
    torch.mps.synchronize()
    return o[0, 0].cpu().double(), compiled


def _visited(n, clamp, lo=0):
    """visited[r, key]: the source's loop for row r's query tile loads ``key``."""
    rows, keys = torch.arange(n)[:, None], torch.arange(n)[None, :]
    hi = torch.clamp(rows // BM * BM + BM, max=n) if clamp else torch.full_like(rows, n)
    first = keys // BN * BN  # the start of the source block holding the key
    return (first >= lo) & (first < hi)


def _source(x, causal, clamp, lo=0):
    """float64: the source's attention over exactly the keys its loop loads.

    A key the row's mask keeps adds p * v. A key its loop loads and its mask hides adds 0 * v
    (NaN where v is NaN or infinite); a key its loop never loads adds nothing. A row whose every
    loaded score is hidden has an empty running maximum, exp(-inf - -inf) = NaN: the row is NaN.
    """
    q, k = (torch.cat(x[:2], -1), torch.cat(x[2:4], -1)) if len(x) == 5 else x[:2]
    q, k, v = (t[0, 0].cpu().double() for t in (q, k, x[-1]))
    s = q @ k.T / math.sqrt(q.shape[-1])
    n = s.shape[0]
    visited = _visited(n, clamp, lo)
    keep = visited & (torch.ones(n, n, dtype=torch.bool).tril() if causal else True)
    p = torch.softmax(s.masked_fill(~keep, -math.inf), -1)  # NaN on a row with no kept key
    return torch.where(visited[:, :, None], p[:, :, None] * v[None, :, :], 0.0).sum(1)


@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("form", ["two-dot", "one-dot"])
def test_a_hidden_key_value_reaches_every_row_whose_source_loop_loads_it(form, dtype, executed):
    """NaN and +inf at keys 20 and 40 (the first 64-key tile) and 70 and 100 (the second)."""
    tol = 1e-5 if dtype == "f32" else 1e-3
    rows = torch.arange(N)
    loads = _visited(N, clamp=form == "one-dot")
    sources = set()
    for key in (20, 40, 70, 100):
        for bad in (math.nan, math.inf):
            executed.clear()
            got, compiled = _launch(form, _inputs(form, dtype, key=key, bad=bad))
            sources.add(_executed_source(executed, compiled))
            seen = loads[:, key]
            want_nan = seen if math.isnan(bad) else seen & (rows < key)
            want_inf = torch.zeros(N, dtype=torch.bool) if math.isnan(bad) else seen & (rows >= key)
            f0 = got[:, 0]
            case = f"{form} {dtype} key={key} value={bad}"
            assert torch.equal(f0.isnan(), want_nan), f"{case}: NaN rows {f0.isnan().nonzero().flatten().tolist()}"
            assert torch.equal(f0 == math.inf, want_inf), (
                f"{case}: +inf rows {(f0 == math.inf).nonzero().flatten().tolist()}"
            )
            rest = ~(want_nan | want_inf)
            assert bool(((f0[rest] - 1.0).abs() <= tol).all()), f"{case}: rows that never load the key"
            assert bool(((got[:, 1:] - 1.0).abs() <= tol).all()), f"{case}: features 1.."
    (source,) = sources
    assert SHADER in source
    assert "break;" not in source and "skip fully-masked blocks" not in source
    if form == "two-dot":
        # Every key block, as the source's loop: no stop before N_CTX.
        assert "uint n_full = N_CTX / BN;" in source and "source_kv_stop" not in source
    else:
        # The clamped source's own stop: the work past it is still skipped, and the tail tile
        # stages only the keys of the source blocks the loop visits.
        assert "const uint source_kv_stop = min(q_start + BM, N_CTX); // source KV loop stop" in source
        assert "uint n_full = source_kv_stop / BN;" in source
        assert "if (n_full * BN < source_kv_stop) {" in source


@pytest.mark.parametrize("n", [96, 128])
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("form", ["two-dot", "one-dot"])
def test_finite_causal_mla_matches_float64_attention(form, dtype, n, executed):
    x = _inputs(form, dtype, n=n, seed=11)
    got, compiled = _launch(form, x)
    assert SHADER in _executed_source(executed, compiled)
    err = float((got - _source(x, causal=True, clamp=form == "one-dot")).abs().max())
    assert err <= (1e-4 if dtype == "f32" else 5e-3), f"{form} {dtype} N={n}: max error {err}"


@pytest.mark.parametrize(
    "causal,clamp,lo",
    [(True, True, 0), (False, True, 0), (True, False, BN)],
    ids=["clamped-causal", "clamped-unmasked", "loop-from-BN"],
)
def test_a_two_dot_key_loop_other_than_every_block_is_correct_or_refused(causal, clamp, lo):
    """The two-dot route replays range(0, N_CTX, BN) only. Another loop must be refused, or computed
    as its source computes it: a NaN in V at key 40 reaches rows 0-31 only if their loop loads it."""
    x = list(_inputs("two-dot", "f32", seed=3))
    v = x[-1].cpu()
    v[0, 0, 40, 0] = math.nan
    x[-1] = v.to("mps")
    try:
        got, _ = _launch("loop", tuple(x), causal=causal, clamp=clamp, lo=lo)
    except MetalNonRecoverableError:
        return
    want = _source(tuple(x), causal, clamp, lo)
    assert torch.equal(got.isnan(), want.isnan())
    fin = ~want.isnan()
    assert float((got[fin] - want[fin]).abs().max()) <= 1e-4
