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
import numpy as np
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
@pytest.mark.parametrize("N", [8, 2048])   # single-pass + multipass
def test_product_reduce_computes_not_sum(N):
    # A product reduce (a * b combine) now COMPUTES correctly (via simd_product); it was
    # previously refused. Must be the PRODUCT, never silently folded as a sum.
    _clear()
    import numpy as _np
    # values very close to 1 so the product stays finite even over 2048 elements
    vals = (_np.random.RandomState(0).rand(N) * 0.02 + 0.99).astype(_np.float32)
    X = torch.tensor(vals, device="mps")
    OUT = torch.zeros(1, device="mps")
    _k_prod[(1,)](X, OUT, N=N); torch.mps.synchronize()
    ref = float(_np.prod(vals))
    assert abs(OUT.item() - ref) / abs(ref) < 1e-3, (N, OUT.item(), ref)
    assert abs(OUT.item() - float(vals.sum())) > 1e-2   # definitely not the sum


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


# --- round 4: unsigned max/min, NaN-propagate, ge/le predicate (re-audit 2026-06-25) ---
if _HAS:
    @triton.jit
    def _k_umax(X, O, N: tl.constexpr):
        tl.store(O, tl.max(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_umin(X, O, N: tl.constexpr):
        tl.store(O, tl.min(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _nanprop(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _k_nanprop(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _nanprop))

    @triton.jit
    def _ge(a, b):
        return tl.where(a >= b, a, b)

    @triton.jit
    def _k_ge(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _ge))


@requires
@pytest.mark.parametrize("N", [64, 2048])   # small-N simd path + large-N multipass path
def test_uint32_max_min_unsigned_not_signed(N):
    _clear()
    vals = np.arange(N, dtype=np.uint32); vals[0] = 0xFFFFFFFF; vals[1] = 0x80000000
    X = torch.tensor(vals, dtype=torch.uint32, device="mps")
    Omax = torch.zeros(1, dtype=torch.uint32, device="mps")
    _k_umax[(1,)](X, Omax, N=N); torch.mps.synchronize()
    assert Omax.item() == int(vals.max()), (N, Omax.item(), int(vals.max()))
    Omin = torch.zeros(1, dtype=torch.uint32, device="mps")
    _k_umin[(1,)](X, Omin, N=N); torch.mps.synchronize()
    assert Omin.item() == int(vals.min()), (N, Omin.item(), int(vals.min()))


@requires
@pytest.mark.parametrize("N", [64, 2048])   # single-pass simd + multipass
def test_uint64_max_min_unsigned_not_signed(N):
    # uint64 is the SIGNLESS i64 in Triton; a signed `long` reduce reads 2^64-1 as -1 and
    # silently loses the max (and 2^63 wins the min). Must compute UNSIGNED (audit 2026-06-27
    # BLOCKER). The 32-bit twin already worked; this is the missed 64-bit case.
    _clear()
    vals = np.full(N, 5, dtype=np.uint64)
    vals[0] = 2 ** 64 - 1          # all-ones: max if unsigned, -1 (loses) if signed
    vals[1] = 2 ** 63              # high bit: would wrongly win a signed min
    vals[2] = 1                    # the true unsigned min
    X = torch.tensor(vals, dtype=torch.uint64, device="mps")
    Omax = torch.zeros(1, dtype=torch.uint64, device="mps")
    _k_umax[(1,)](X, Omax, N=N); torch.mps.synchronize()
    assert int(Omax.item()) == int(vals.max()) == 2 ** 64 - 1, (N, int(Omax.item()))
    Omin = torch.zeros(1, dtype=torch.uint64, device="mps")
    _k_umin[(1,)](X, Omin, N=N); torch.mps.synchronize()
    assert int(Omin.item()) == int(vals.min()) == 1, (N, int(Omin.item()))


@requires
@pytest.mark.parametrize("N", [128, 2048])   # single-pass + multipass
def test_nan_propagating_max_computes_and_propagates(N):
    # A NaN-PROPAGATING max (tl.maximum propagate_nan=ALL, == inductor triton_helpers.maximum)
    # now COMPUTES correctly (was refused): finite inputs -> the true max; a NaN anywhere ->
    # NaN (propagates, matching numpy/torch). The cross-thread simd step (NaN-quiet) carries
    # an any-NaN side-channel so propagation survives the threadgroup reduction.
    import math
    _clear()
    x = torch.randn(N)
    O = torch.zeros(1, device="mps")
    _k_nanprop[(1,)](x.to("mps"), O, N=N); torch.mps.synchronize()
    assert abs(O.item() - x.max().item()) < 1e-3, (N, O.item(), x.max().item())
    # a NaN present anywhere must propagate to the result
    _clear()
    x2 = torch.randn(N); x2[40] = float("nan")
    O2 = torch.zeros(1, device="mps")
    _k_nanprop[(1,)](x2.to("mps"), O2, N=N); torch.mps.synchronize()
    assert math.isnan(O2.item()), (N, O2.item())


@requires
def test_float_ge_max_not_over_refused():
    _clear()
    x = torch.randn(128)
    O = torch.zeros(1, device="mps")
    _k_ge[(1,)](x.to("mps"), O, N=128); torch.mps.synchronize()   # ge/le must compute, not refuse
    assert abs(O.item() - x.max().item()) < 1e-3, (O.item(), x.max().item())


# --- Phase-1 structural classifier: custom combine refuses, 1-D bitwise computes (2026-06-25) ---
if _HAS:
    @triton.jit
    def _relusum(a, b):
        return a + tl.where(b > 0, b, 0.0)

    @triton.jit
    def _k_relusum(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _relusum))

    @triton.jit
    def _and2(a, b):
        return a & b

    @triton.jit
    def _k_and(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _and2))

    @triton.jit
    def _k_xor(X, O, N: tl.constexpr):
        tl.store(O, tl.xor_sum(tl.load(X + tl.arange(0, N)), 0))


@requires
def test_sum_of_relu_custom_combine_refuses():
    # a + relu(b) is a custom (non-canonical) combine — the structural classifier refuses it
    # (the yielded addf's operands are not the two block args). Sniffing flipped it to MAX.
    _clear()
    x = torch.randn(64, device="mps"); O = torch.zeros(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_relusum[(1,)](x, O, N=64); torch.mps.synchronize()


@requires
def test_1d_bitwise_and_xor_compute():
    # 1-D all()/any()/xor must COMPUTE (was over-refused via a KeyError in threadgroup_reduce).
    _clear()
    import numpy as _np
    x = torch.randint(0, 255, (64,), dtype=torch.int32, device="mps")
    OA = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_and[(1,)](x, OA, N=64); torch.mps.synchronize()
    assert OA.item() == int(_np.bitwise_and.reduce(x.cpu().numpy()))
    OX = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_xor[(1,)](x, OX, N=64); torch.mps.synchronize()
    assert OX.item() == int(_np.bitwise_xor.reduce(x.cpu().numpy()))


# --- structural-classifier validation: block-arg pick refuses, i64 and/or computes (2026-06-25) ---
if _HAS:
    @triton.jit
    def _first(a, b):
        return a

    @triton.jit
    def _k_first(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _first))

    @triton.jit
    def _andc(a, b):
        return a & b

    @triton.jit
    def _k_and_i64(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _andc))


@requires
@pytest.mark.parametrize("N", [64, 1024])   # single-pass simd + multipass
def test_first_pick_combine_refuses_not_sum(N):
    # `return a` (a first/last/identity pick) is a non-canonical combine — must REFUSE
    # (an empty combine region after terminator-strip used to default to SUM).
    _clear()
    x = torch.arange(1, N + 1, dtype=torch.float32, device="mps")
    O = torch.zeros(1, device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _k_first[(1,)](x, O, N=N); torch.mps.synchronize()


@requires
def test_i64_and_reduce_computes():
    # i64 bitwise-and reduce computes (was over-refused — combine table lacked and/or).
    _clear()
    import numpy as _np
    vals = (_np.random.RandomState(0).randint(0, 2 ** 40, 64)).astype(_np.int64)
    X = torch.tensor(vals, dtype=torch.int64, device="mps")
    O = torch.zeros(1, dtype=torch.int64, device="mps")
    _k_and_i64[(1,)](X, O, N=64); torch.mps.synchronize()
    assert O.item() == int(_np.bitwise_and.reduce(vals))


# --- multipass and/or/xor consistency (final validation re-audit, 2026-06-25) ---
if _HAS:
    @triton.jit
    def _xorc(a, b):
        return a ^ b

    @triton.jit
    def _k_xor_big(X, O, N: tl.constexpr):
        tl.store(O, tl.reduce(tl.load(X + tl.arange(0, N)), 0, _xorc))


@requires
@pytest.mark.parametrize("N", [64, 2048])   # single-pass + multipass must AGREE
def test_xor_reduce_both_paths_compute(N):
    # and/or/xor must compute on BOTH the single-pass and multipass paths (the multipass
    # whitelist used to refuse them at N>=1024 while single-pass computed — inconsistent).
    _clear()
    import numpy as _np
    vals = _np.random.RandomState(0).randint(0, 2 ** 28, N).astype(_np.int32)
    X = torch.tensor(vals, dtype=torch.int32, device="mps")
    O = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_xor_big[(1,)](X, O, N=N); torch.mps.synchronize()
    assert O.item() == int(_np.bitwise_xor.reduce(vals))


# --- confirming re-audit 2026-06-27: unsigned argminmax + fused-argminmax barrier ---
if _HAS:
    @triton.jit
    def _k_argmax(X, O, N: tl.constexpr):
        tl.store(O, tl.argmax(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_argmin(X, O, N: tl.constexpr):
        tl.store(O, tl.argmin(tl.load(X + tl.arange(0, N)), 0))

    @triton.jit
    def _k_dual_argminmax(X, Omin, Omax, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M); rn = tl.arange(0, N)
        x = tl.load(X + rm[:, None] * N + rn[None, :])
        tl.store(Omin + rm, tl.argmin(x, axis=1))
        tl.store(Omax + rm, tl.argmax(x, axis=1))


@requires
@pytest.mark.parametrize("dt,np_dt,hi", [
    (torch.uint32, np.uint32, 0xFFFFFFFF),
    (torch.uint16, np.uint16, 0xFFFF),
    (torch.uint8, np.uint8, 0xFF),
])
def test_unsigned_argminmax_not_signed(dt, np_dt, hi):
    # argmax/argmin over unsigned ints must compare UNSIGNED — uintN is the signless iN, so a
    # signed compare reads the high-bit value as negative and picks the wrong index (audit
    # 2026-06-27 BLOCKER, twin of the uint64 max/min one).
    _clear()
    vals = np.array([10, 20, 30, 40, hi, 5, 60, 70], dtype=np_dt)   # argmax=4 (hi), argmin=5 (5)
    X = torch.tensor(vals, dtype=dt, device="mps")
    Omax = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_argmax[(1,)](X, Omax, N=8); torch.mps.synchronize()
    assert int(Omax.item()) == int(np.argmax(vals)) == 4, (np_dt, int(Omax.item()))
    Omin = torch.zeros(1, dtype=torch.int32, device="mps")
    _k_argmin[(1,)](X, Omin, N=8); torch.mps.synchronize()
    assert int(Omin.item()) == int(np.argmin(vals)) == 5, (np_dt, int(Omin.item()))


@requires
def test_fused_2d_argmin_argmax_no_race():
    # Two 2-D argminmax in one kernel: the first op's result arrays must NOT be clobbered by
    # the second op's re-stage (missing-barrier race, twin of the 1ddac9b reduce barrier).
    _clear()
    for seed in range(8):
        xv = np.random.RandomState(seed).randn(8, 128).astype(np.float32)
        X = torch.tensor(xv, device="mps")
        Omin = torch.zeros(8, dtype=torch.int32, device="mps")
        Omax = torch.zeros(8, dtype=torch.int32, device="mps")
        _k_dual_argminmax[(1,)](X, Omin, Omax, M=8, N=128); torch.mps.synchronize()
        np.testing.assert_array_equal(Omin.cpu().numpy(), xv.argmin(1), err_msg=f"argmin seed {seed}")
        np.testing.assert_array_equal(Omax.cpu().numpy(), xv.argmax(1), err_msg=f"argmax seed {seed}")
