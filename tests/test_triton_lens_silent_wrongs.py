"""Regressions for the two BLOCKER silent-wrongs the 2026-06-25 Triton-lens audit found
on core language semantics the matmul/reduce campaign never touched:

  B1. tl.reduce with a CUSTOM associative combine was string-sniffed and silently
      mis-computed — a product (arith.mulf) combine returned the SUM; a max-by-magnitude
      (where(abs(a)>abs(b),a,b)) returned the plain MAX. Now refused loudly; plain
      sum/max/min still compute.
  B2. an scf.if yielding an INTEGER produced inside the branch (an inner scf.for
      accumulator) was declared `float` — silent precision loss for i32 > 2^24 and i64.
      Now derives the dtype from the IR result type + has a long/ulong branch.
"""
import pytest
import torch

try:
    import triton
    import triton.language as tl
    _HAS = True
except Exception:
    _HAS = False

from triton_msl.errors import MetalNonRecoverableError

requires = pytest.mark.skipif(not _HAS, reason="triton not available")

if _HAS:
    @triton.jit
    def _mul(a, b):
        return a * b

    @triton.jit
    def _k_prod(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _mul))

    @triton.jit
    def _maxmag(a, b):
        return tl.where(tl.abs(a) > tl.abs(b), a, b)

    @triton.jit
    def _k_maxmag(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _maxmag))

    @triton.jit
    def _k_sum(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.sum(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_max(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.max(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_if_i32(COND, OUT):
        c = tl.load(COND)
        acc = 1
        if c > 0:
            for _ in range(10):
                acc += 2000003          # 1 + 10*2000003 = 20000031, odd, > 2^24
        tl.store(OUT, acc)

    @triton.jit
    def _k_if_i64(COND, INIT, OUT):
        c = tl.load(COND)
        acc = tl.load(INIT)             # i64 (forces the scf.if result dtype to i64)
        if c > 0:
            for _ in range(10):
                acc += 1600000003       # 16000000031, well beyond fp32 mantissa
        tl.store(OUT, acc)


def _clear():
    import os
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")


@requires
def test_product_reduce_refuses_not_sum():
    _clear()
    X = torch.tensor([1., 2., 3., 4., 1., 1., 1., 1.], device="mps")
    OUT = torch.zeros(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_prod[(1,)](X, OUT, N=8); torch.mps.synchronize()


@requires
def test_max_by_magnitude_reduce_refuses_not_plain_max():
    _clear()
    X = torch.tensor([1., -5., 3., -2., 4., -9., 0.5, 2.], device="mps")
    OUT = torch.zeros(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_maxmag[(1,)](X, OUT, N=8); torch.mps.synchronize()


@requires
def test_plain_sum_max_still_compute():
    _clear()
    X = torch.tensor([1., -5., 3., -2., 4., -9., 0.5, 2.], device="mps")
    for kern, ref in ((_k_sum, X.sum().item()), (_k_max, X.max().item())):
        OUT = torch.zeros(1, device="mps")
        kern[(1,)](X, OUT, N=8); torch.mps.synchronize()
        assert abs(OUT.item() - ref) < 1e-3, (OUT.item(), ref)


@requires
def test_scf_if_int_accumulator_not_float_rounded():
    _clear()
    COND = torch.ones(1, dtype=torch.int32, device="mps")
    OUT = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_if_i32[(1,)](COND, OUT); torch.mps.synchronize()
    assert OUT.item() == 1 + 10 * 2000003   # 20000031 exactly, no fp32 rounding


@requires
def test_scf_if_i64_accumulator_not_truncated():
    _clear()
    COND = torch.ones(1, dtype=torch.int32, device="mps")
    INIT = torch.ones(1, dtype=torch.int64, device="mps")
    OUT = torch.zeros(1, dtype=torch.int64, device="mps")
    try:
        _k_if_i64[(1,)](COND, INIT, OUT); torch.mps.synchronize()
    except MetalNonRecoverableError:
        return   # refusing is acceptable (correct-or-refuse); silently truncating is not
    assert OUT.item() == 1 + 10 * 1600000003   # 16000000031 exactly, no i64->i32/float loss


# --- gather i64 truncation (Triton-lens re-audit 2026-06-25; the systematic-lead sibling) ---
if _HAS:
    @triton.jit
    def _k_gather1d(SRC, IDX, OUT, S: tl.constexpr, I: tl.constexpr):
        src = tl.load(SRC + tl.arange(0, S))
        idx = tl.load(IDX + tl.arange(0, I))
        tl.store(OUT + tl.arange(0, I), tl.gather(src, idx, axis=0))


@requires
def test_gather_i64_not_truncated_to_i32():
    # gather of an int64 source whose values exceed 2^31 must NOT collapse to int32
    # (3_000_000_000 truncated to int32 reads back -1_294_967_296). NB: torch-MPS's own
    # gather is buggy for int64, so the reference is computed by direct indexing.
    _clear()
    S = I = 8
    base = 3_000_000_000
    SRC = torch.arange(S, dtype=torch.int64, device="mps") + base
    IDX = torch.tensor([7, 0, 3, 5, 1, 6, 2, 4], dtype=torch.int32, device="mps")
    OUT = torch.zeros(I, dtype=torch.int64, device="mps")
    _k_gather1d[(1,)](SRC, IDX, OUT, S=S, I=I); torch.mps.synchronize()
    expected = [base + j for j in IDX.tolist()]           # src[idx], i64-exact
    assert OUT.tolist() == expected, (OUT.tolist(), expected)


# --- join/cat i64 truncation + cmp+select reduce inversion (re-audit round 2, 2026-06-25) ---
if _HAS:
    @triton.jit
    def _k_join(A, B, OUT, N: tl.constexpr):
        a = tl.load(A + tl.arange(0, N)); b = tl.load(B + tl.arange(0, N))
        tl.store(OUT + tl.arange(0, 2 * N), tl.reshape(tl.join(a, b), (2 * N,)))

    @triton.jit
    def _k_cat(A, B, OUT, N: tl.constexpr):
        a = tl.load(A + tl.arange(0, N)); b = tl.load(B + tl.arange(0, N))
        tl.store(OUT + tl.arange(0, 2 * N), tl.cat(a, b, can_reorder=True))

    @triton.jit
    def _min_via_gt(a, b):
        return tl.where(a > b, b, a)        # semantic MIN (predicate says gt)

    @triton.jit
    def _max_via_lt(a, b):
        return tl.where(a < b, b, a)        # semantic MAX (predicate says lt)

    @triton.jit
    def _k_reduce_cmbsel(X, OUT, N: tl.constexpr, which: tl.constexpr):
        v = tl.load(X + tl.arange(0, N))
        r = tl.reduce(v, 0, _min_via_gt) if which == 0 else tl.reduce(v, 0, _max_via_lt)
        tl.store(OUT, r)


@requires
def test_join_i64_not_truncated():
    _clear()
    N, base = 8, 3_000_000_001
    A = torch.arange(N, dtype=torch.int64, device="mps") + base
    B = torch.arange(N, dtype=torch.int64, device="mps") + base + 100
    OUT = torch.zeros(2 * N, dtype=torch.int64, device="mps")
    _k_join[(1,)](A, B, OUT, N=N); torch.mps.synchronize()
    exp = torch.stack([A.cpu(), B.cpu()], dim=1).reshape(2 * N)   # CPU reference
    assert (OUT.cpu() == exp).all(), (OUT[:4].tolist(), exp[:4].tolist())


@requires
def test_cat_i64_not_truncated():
    _clear()
    N, base = 8, (2 ** 40) + 7
    A = torch.arange(N, dtype=torch.int64, device="mps") + base
    B = torch.arange(N, dtype=torch.int64, device="mps") + base + 100
    OUT = torch.zeros(2 * N, dtype=torch.int64, device="mps")
    _k_cat[(1,)](A, B, OUT, N=N); torch.mps.synchronize()
    exp = torch.cat([A.cpu(), B.cpu()])
    assert (OUT.cpu() == exp).all(), (OUT[:4].tolist(), exp[:4].tolist())


@requires
def test_reduce_cmp_select_not_inverted():
    # where(a>b,b,a) is a MIN and where(a<b,b,a) is a MAX — classifying by the cmpf
    # predicate alone inverted them (MIN computed as MAX). Must compute correctly (or refuse).
    _clear()
    X = torch.tensor([1., -5., 3., -2., 4., -9., 0.5, 2.])
    for which, ref in ((0, X.min().item()), (1, X.max().item())):
        OUT = torch.zeros(1, device="mps")
        try:
            _k_reduce_cmbsel[(1,)](X.to("mps"), OUT, N=8, which=which); torch.mps.synchronize()
        except MetalNonRecoverableError:
            continue   # refusing is acceptable (correct-or-refuse)
        assert abs(OUT.item() - ref) < 1e-3, (which, OUT.item(), ref)


# --- argmax/argmin i64 (re-audit round 2 sibling: SIMD-shuffle value staging truncated i64) ---
if _HAS:
    @triton.jit
    def _k_argmax1d(X, OUT, N: tl.constexpr):
        tl.store(OUT, tl.argmax(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_argmax2d(X, OUT, M: tl.constexpr, N: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N)
        tl.store(OUT + om, tl.argmax(tl.load(X + om[:, None] * N + on[None, :]), 1))


@requires
def test_argmax_i64_1d_refuses_not_wrong_index():
    # 1-D argmax over i64 must REFUSE (the SIMD-shuffle reduction has no 64-bit path; a
    # 32-bit staging silently truncated and returned the wrong index). i32 must still work.
    _clear()
    X = torch.tensor([3_000_000_005, 1, 3_000_000_009, 2, 3_000_000_001, 0, 7, 3],
                     dtype=torch.int64, device="mps")
    OUT = torch.zeros(1, dtype=torch.int32, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_argmax1d[(1,)](X, OUT, N=8); torch.mps.synchronize()
    Xi = X.to(torch.int32); OUT2 = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_argmax1d[(1,)](Xi, OUT2, N=8); torch.mps.synchronize()   # i32 still computes
    assert OUT2.item() == int(Xi.cpu().argmax())


@requires
def test_argmax_i64_2d_correct_index():
    # 2-D argmax (shared-memory path) handles i64 at full width -> correct index.
    _clear()
    M, N = 4, 32
    X = (torch.arange(M * N, dtype=torch.int64, device="mps") + 3_000_000_000).reshape(M, N)
    OUT = torch.zeros(M, dtype=torch.int32, device="mps")
    _k_argmax2d[(1,)](X, OUT, M=M, N=N); torch.mps.synchronize()
    assert OUT.cpu().tolist() == X.cpu().argmax(1).tolist()


# --- round 3: convert_layout i64 / integer cmp+select / Welford-misroute (2026-06-25) ---
if _HAS:
    @triton.jit
    def _k_2dsum(X, OUT, M: tl.constexpr, N: tl.constexpr):
        om = tl.arange(0, M); on = tl.arange(0, N)
        tl.store(OUT + on, tl.sum(tl.load(X + om[:, None] * N + on[None, :]), 0))

    @triton.jit
    def _imax(a, b):
        return tl.where(a > b, a, b)

    @triton.jit
    def _imin(a, b):
        return tl.where(a > b, b, a)

    @triton.jit
    def _k_int_cmpsel(X, OUT, N: tl.constexpr, w: tl.constexpr):
        v = tl.load(X + tl.arange(0, N))
        tl.store(OUT, tl.reduce(v, 0, _imax) if w == 0 else tl.reduce(v, 0, _imin))

    @triton.jit
    def _tmax(a1, b1, c1, a2, b2, c2):
        return tl.maximum(a1, a2), tl.maximum(b1, b2), tl.maximum(c1, c2)

    @triton.jit
    def _k_3tuple(X, Y, Z, OA, OB, OC, N: tl.constexpr):
        x = tl.load(X + tl.arange(0, N)); y = tl.load(Y + tl.arange(0, N)); z = tl.load(Z + tl.arange(0, N))
        a, b, c = tl.reduce((x, y, z), 0, _tmax)
        tl.store(OA, a); tl.store(OB, b); tl.store(OC, c)


@requires
def test_2d_int64_reduce_not_truncated_by_convert_layout():
    _clear()
    M = N = 16
    X = ((1 << 34) + torch.arange(M * N, dtype=torch.int64, device="mps")).reshape(M, N)
    OUT = torch.zeros(N, dtype=torch.int64, device="mps")
    _k_2dsum[(1,)](X, OUT, M=M, N=N); torch.mps.synchronize()
    assert (OUT.cpu() == X.cpu().sum(0)).all(), (OUT[0].item(), X.cpu().sum(0)[0].item())


@requires
def test_integer_cmp_select_reduce_not_sum():
    _clear()
    torch.manual_seed(0)
    X = torch.randint(-100000, 100000, (128,), dtype=torch.int32, device="mps")
    for w, ref in ((0, int(X.cpu().max())), (1, int(X.cpu().min()))):
        OUT = torch.zeros(1, dtype=torch.int32, device="mps")
        try:
            _k_int_cmpsel[(1,)](X, OUT, N=128, w=w); torch.mps.synchronize()
        except MetalNonRecoverableError:
            continue   # refuse is acceptable; silently summing is not
        assert OUT.item() == ref, (w, OUT.item(), ref)


@requires
def test_custom_3tuple_reduce_refuses_not_welford():
    _clear()
    N = 128
    X = torch.randn(N, device="mps"); Y = torch.randn(N, device="mps"); Z = torch.randn(N, device="mps")
    O = [torch.zeros(1, device="mps") for _ in range(3)]
    with pytest.raises(MetalNonRecoverableError):
        _k_3tuple[(1,)](X, Y, Z, *O, N=N); torch.mps.synchronize()
