"""The neighbours of a reduction of a row-blocked value: correct, or refused.

tests/test_reduce_of_row_reduction.py covers a row reduction and one consumer. This file
covers the forms found next to it on 2026-11-18, each of which returned wrong values with
no error (or failed in the Metal compiler) before:

  * a three-dimensional tile reduced along one axis and stored as a two-dimensional tile
    (the store wrote nothing, or wrote through the wrong argument)
  * tl.argmax / tl.argmin of a row reduction on a tile of at most 64 lanes (a lane that
    held zero won, so all-negative totals answered 0)
  * the totals of a tile laid along the OTHER axis of a two-dimensional store, and the
    outer product of the totals with themselves, also stored to a tile of another shape
  * the index of each row's largest element used as a load address (plain gather), and
    the other users of such an address (scatter, atomic, masked store, second load)
  * two tiles of different shapes in one program: the smaller one stored or atomically
    updated past its end, a ramp tile that is both stored and reduced, two loaded tiles
    of different widths on one row range
  * a one-column or one-row tile combined with its own single total and reduced again,
    alone and beside a wider tile on the same range

and one form that computed before and must keep computing: the rows of a matrix product
normalised by their own largest element and total, where the product's row range also
addresses a wider operand tile (a product is laid out by its own lowering, so it is not
one of the spread tiles the cases above gather or refuse).

Every case runs on the device against a numpy float64 (or exact integer) reference
computed from the input data alone. The data is signed (and, for the index reductions,
also all negative and all positive), so a lane holding zero or the first row again cannot
pass for the answer, and the expected index is never 0 where the form allows. Every
program is defined: one program, every stored cell written once, the output separate
from the inputs. Every output is filled with a sentinel and compared whole, so a store
outside the expected cells fails the test too, and every input is compared with its
original after the launch.

tl.arange needs a power-of-two length, so the tile sides are powers of two; the shapes
are square, wide, tall and one-lane-per-row, and origins are shifted by odd offsets
where the form takes one.
"""

import numpy as np
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.errors import MetalNonRecoverableError

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _three_d_last(O, Y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
        """An (A, B, C) tile summed along its last axis, output argument first."""
        a = tl.arange(0, A)
        b = tl.arange(0, B)
        c = tl.arange(0, C)
        v = tl.load(Y + a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :])
        tl.store(O + a[:, None] * B + b[None, :], tl.sum(v, axis=2))

    @triton.jit
    def _three_d(Y, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, MODE: tl.constexpr):
        """An (A, B, C) tile reduced along AX, input argument first. MODE 0 sum, 1 the load
        shifted by 3 cells, 2 the store shifted by 5 cells, 3 the sum doubled, 4 argmax,
        5 the last axis read backwards."""
        a = tl.arange(0, A)
        b = tl.arange(0, B)
        c = tl.arange(0, C)
        if MODE == 1:
            v = tl.load(Y + 3 + a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :])
        elif MODE == 5:
            v = tl.load(Y + a[:, None, None] * (B * C) + b[None, :, None] * C + (C - 1 - c)[None, None, :])
        else:
            v = tl.load(Y + a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :])
        if MODE == 4:
            r = tl.argmax(v, axis=AX).to(tl.float32)
        else:
            r = tl.sum(v, axis=AX)
        if MODE == 3:
            r = r * 2.0
        if AX == 2:
            dst = O + a[:, None] * B + b[None, :]
        elif AX == 1:
            dst = O + a[:, None] * C + c[None, :]
        else:
            dst = O + b[:, None] * C + c[None, :]
        if MODE == 2:
            dst = dst + 5
        tl.store(dst, r)

    @triton.jit
    def _arg_of_rows(O, Y, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, MODE: tl.constexpr):
        """The index of the largest / smallest of the reductions of an (R, C) tile along AX."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        if MODE == 0:
            tl.store(O, tl.argmax(tl.sum(v, axis=AX), axis=0))
        elif MODE == 1:
            tl.store(O, tl.argmin(tl.sum(v, axis=AX), axis=0))
        elif MODE == 2:
            tl.store(O, tl.argmax(tl.min(v, axis=AX), axis=0))
        elif MODE == 3:
            tl.store(O, tl.argmin(tl.max(v, axis=AX), axis=0))
        elif MODE == 4:
            top, where = tl.max(tl.sum(v, axis=AX), axis=0, return_indices=True)
            tl.store(O, where)
        else:
            low, where = tl.min(tl.sum(v, axis=AX), axis=0, return_indices=True)
            tl.store(O, where)

    @triton.jit
    def _across(O, Y, N: tl.constexpr, AX: tl.constexpr, MODE: tl.constexpr):
        """The totals of a square tile along AX in a two-dimensional store. 0: laid along the
        OTHER axis; 1: their outer product with themselves; 2: laid along their own axis."""
        i = tl.arange(0, N)
        v = tl.load(Y + i[:, None] * N + i[None, :])
        r = tl.sum(v, axis=AX)
        tile = O + i[:, None] * N + i[None, :]
        if MODE == 0:
            if AX == 1:
                tl.store(tile, v + r[None, :])
            else:
                tl.store(tile, v + r[:, None])
        elif MODE == 1:
            tl.store(tile, r[:, None] * r[None, :])
        else:
            if AX == 1:
                tl.store(tile, v + r[:, None])
            else:
                tl.store(tile, v + r[None, :])

    @triton.jit
    def _outer(O, Y, R: tl.constexpr, C: tl.constexpr, MODE: tl.constexpr):
        """The reductions of an (R, C) tile met with themselves across a square tile."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        if MODE == 0:
            x = tl.sum(v, 1)
            tl.store(O + rows[:, None] * R + rows[None, :], x[:, None] - 2.0 * x[None, :])
        elif MODE == 1:
            x = tl.max(v, 0)
            tl.store(O + cols[:, None] * C + cols[None, :], x[:, None] - 2.0 * x[None, :])
        elif MODE == 2:
            x = tl.sum(v, 1)
            tl.store(O + rows[:, None] * R + rows[None, :], tl.where(x[:, None] > x[None, :], x[:, None], x[None, :]))
        elif MODE == 3:
            tl.store(O + rows[:, None] * C + cols[None, :], v - tl.max(v, 1)[:, None] + tl.min(v, 0)[None, :])
        elif MODE == 4:
            x = tl.sum(v, 0)
            m = x[:, None] - x[None, :]
            tl.store(O + cols, tl.sum(m, axis=1))
        else:
            x = tl.sum(v, 1)
            m = x[:, None] - 2.0 * x[None, :]
            tl.store(O + rows, tl.max(m, axis=0))

    @triton.jit
    def _gather(O, Y, P, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, MODE: tl.constexpr):
        """The index of the largest element of each row (AX 1) or column (AX 0) of an (R, C)
        tile, used in an address."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        a = tl.argmax(v, axis=AX)
        if AX == 1:
            idx = rows
        else:
            idx = cols
        if MODE == 0:  # plain gather
            tl.store(O + idx, tl.load(P + a))
        elif MODE == 1:  # each line's own largest element, by gather
            if AX == 1:
                tl.store(O + idx, tl.load(Y + rows * C + a))
            else:
                tl.store(O + idx, tl.load(Y + a * C + cols))
        elif MODE == 2:  # the gathered values summed
            tl.store(O, tl.sum(tl.load(P + a), axis=0))
        elif MODE == 3:  # scatter of one value per line
            tl.store(O + a, idx.to(tl.float32))
        elif MODE == 4:  # scatter of one number
            tl.store(O + a, 1.0)
        elif MODE == 5:  # store masked by the index
            tl.store(O + idx, idx.to(tl.float32), mask=a > 0)
        elif MODE == 6:  # atomic scatter
            tl.atomic_add(O + a, idx.to(tl.float32))
        elif MODE == 7:  # two loads through the index
            tl.store(O + idx, tl.load(P + a) + tl.load(P + a + 1))
        elif MODE == 8:  # masked gather
            tl.store(O + idx, tl.load(P + a, mask=a > 0, other=-3.0))
        else:  # the index scaled and shifted
            tl.store(O + idx, tl.load(P + a * 2 + 1))

    @triton.jit
    def _two_tiles(O, Y, R: tl.constexpr, C: tl.constexpr, R2: tl.constexpr, C2: tl.constexpr, MODE: tl.constexpr):
        """An (R, C) tile and an (R2, C2) tile in one program."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        r2 = tl.arange(0, R2)
        c2 = tl.arange(0, C2)
        second = O + R * C + r2[:, None] * C2 + c2[None, :]
        if MODE == 0:  # both stored
            tl.store(O + rows[:, None] * C + cols[None, :], v)
            tl.store(second, tl.load(Y + r2[:, None] * C2 + c2[None, :]) * 2.0)
        elif MODE == 1:  # the second one a ramp
            tl.store(O + rows[:, None] * C + cols[None, :], v)
            tl.store(second, (r2[:, None] * 100 + c2[None, :]).to(tl.float32))
        elif MODE == 2:  # only the second one stored
            w = tl.load(Y + r2[:, None] * C2 + c2[None, :])
            tl.store(O + r2[:, None] * C2 + c2[None, :], w + tl.sum(v))
        elif MODE == 3:  # the second one updated atomically
            tl.store(O + rows[:, None] * C + cols[None, :], v)
            tl.atomic_add(second, tl.load(Y + r2[:, None] * C2 + c2[None, :]) * 2.0)
        elif MODE == 4:  # the second one stored and its rows totalled
            w = tl.load(Y + r2[:, None] * C2 + c2[None, :])
            tl.store(O + R2 + r2[:, None] * C2 + c2[None, :], w + tl.sum(v))
            tl.store(O + r2, tl.sum(w, axis=1))
        elif MODE == 5:  # a ramp tile stored and its rows totalled
            z = (r2[:, None] * 100 + c2[None, :]).to(tl.float32) + tl.sum(v)
            tl.store(O + R2 + r2[:, None] * C2 + c2[None, :], z)
            tl.store(O + r2, tl.sum(z, axis=1))
        elif MODE == 6:  # the second one less its row totals
            w = tl.load(Y + r2[:, None] * C2 + c2[None, :])
            tl.store(O + r2[:, None] * C2 + c2[None, :], w - tl.sum(w, axis=1)[:, None])
        elif MODE == 7:  # the second one less its column maxima
            w = tl.load(Y + r2[:, None] * C2 + c2[None, :])
            tl.store(O + r2[:, None] * C2 + c2[None, :], w - tl.max(w, axis=0)[None, :] + tl.sum(v))
        else:  # the row totals met across an (R, R) tile that is stored AND reduced
            x = tl.sum(v, axis=1)
            d = x[:, None] - 2.0 * x[None, :]
            tl.store(O + R + rows[:, None] * R + rows[None, :], d)
            tl.store(O + rows, tl.sum(d, axis=1))

    @triton.jit
    def _two_loaded(O, Y, Z, R: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, MODE: tl.constexpr):
        """An (R, C1) and an (R, C2) loaded tile on one row range."""
        rows = tl.arange(0, R)
        c1 = tl.arange(0, C1)
        c2 = tl.arange(0, C2)
        a = tl.load(Y + rows[:, None] * C1 + c1[None, :])
        b = tl.load(Z + rows[:, None] * C2 + c2[None, :])
        if MODE == 0:
            tl.store(O + rows, tl.argmax(a, axis=1).to(tl.float32))
            tl.store(O + R + rows, tl.argmin(b, axis=1).to(tl.float32))
        elif MODE == 1:
            tl.store(O + rows[:, None] * C1 + c1[None, :], tl.cumsum(a, axis=1))
            tl.store(O + R * C1 + rows, tl.max(b, axis=1))
        elif MODE == 2:
            tl.store(O + rows[:, None] * C1 + c1[None, :], a * 2.0)
            tl.store(O + R * C1 + rows[:, None] * C2 + c2[None, :], b + 1.0)
        elif MODE == 3:
            tl.store(O + rows[:, None] * C1 + c1[None, :], a - tl.sum(a, axis=1)[:, None])
            tl.store(O + R * C1 + rows, tl.max(b, axis=1))
        elif MODE == 4:
            tl.store(O, tl.sum(a))
            tl.store(O + 1, tl.max(b))
        elif MODE == 5:
            tl.store(O + c1, tl.sum(a, axis=0))
            tl.store(O + C1 + c2, tl.max(b, axis=0))
        elif MODE == 6:
            tl.store(O, tl.sum(tl.max(tl.cumsum(a, axis=1), axis=1) + tl.max(b, axis=1), axis=0))
        elif MODE == 7:
            tl.store(O + rows, tl.argmax(a, axis=1).to(tl.float32) + tl.max(b))
        elif MODE == 8:
            tl.store(O + rows, tl.sum(a, axis=1) + tl.max(b))
        elif MODE == 9:
            tl.store(O + rows, tl.max(tl.cumsum(a, axis=1), axis=1) + tl.max(b))

    @triton.jit
    def _unit_axis(O, Y, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
        """An (R, 1) or a (1, C) tile less half its own total (one number), then used.
        AX is the long axis."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        AX: tl.constexpr = 0 if C == 1 else 1
        w = v - tl.sum(v, axis=AX, keep_dims=True) * 0.5
        if AX == 0:
            unit = cols
        else:
            unit = rows
        if MODE == 0:
            tl.store(O + unit, tl.sum(w, axis=AX))
        elif MODE == 1:
            tl.store(O + unit, tl.argmax(w, axis=AX).to(tl.float32))
        elif MODE == 2:
            tl.store(O + rows[:, None] * C + cols[None, :], w)
        elif MODE == 3:
            tl.store(O + unit, tl.min(w, axis=AX))
        elif MODE == 4:
            tl.store(O, tl.sum(tl.sum(w, axis=AX), axis=0))
        elif MODE == 5:
            tl.store(O + unit, tl.sum(w * tl.max(v, axis=AX, keep_dims=True), axis=AX))
        elif MODE == 6 or MODE == 7:  # the long range also addresses a wider stored tile
            k = tl.arange(0, K)
            if AX == 0:
                tl.store(O + rows[:, None] * K + k[None, :], v + k[None, :].to(tl.float32))
            else:
                tl.store(O + k[:, None] * C + cols[None, :], v + k[:, None].to(tl.float32))
            if MODE == 6:
                tl.store(O + R * C * K + unit, tl.sum(w, axis=AX))
            else:
                tl.store(O + R * C * K + unit, tl.argmax(w, axis=AX).to(tl.float32))
        elif MODE == 8:  # less the total of a second, shorter one-column tile
            k = tl.arange(0, K)
            u = tl.load(Y + k[:, None] + unit[None, :])
            tl.store(O + unit, tl.sum(v - tl.sum(u, axis=0, keep_dims=True), axis=AX))

    @triton.jit
    def _product_rows(O, A, X, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        """The rows of the (M, N) product of an (M, K) and a (K, N) tile, each less its
        largest element, exponentiated and divided by its total."""
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        x = tl.load(X + rk[:, None] * N + rn[None, :])
        c = tl.dot(a, x)
        p = tl.exp(c - tl.max(c, axis=1)[:, None])
        tl.store(O + rm[:, None] * N + rn[None, :], p / tl.sum(p, axis=1)[:, None])


_GUARD = 8
_INT_SENTINEL = -777777
_RTOL = _ATOL = 5e-4
_MARGIN = 2e-3  # a float32 total may differ from the float64 one by far less than this


def _f32(rng, *shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _cells(total, values, at=0, base=None):
    """The whole expected output: `values` from cell `at`, the sentinel (NaN) elsewhere."""
    want = np.full(total + _GUARD, np.nan) if base is None else np.array(base, dtype=np.float64)
    flat = np.atleast_1d(np.asarray(values, dtype=np.float64)).reshape(-1)
    want[at : at + flat.size] = flat
    return want


def _launch(kernel, total, inputs, dtype=None, out_first=True, **constexprs):
    """One program of `kernel` over a sentinel-filled output of `total` + guard cells.
    Every input must come back as it went in."""
    dev = [torch.from_numpy(x.copy()).to("mps") for x in inputs]
    if dtype is None:
        out = torch.full((total + _GUARD,), float("nan"), dtype=torch.float32, device="mps")
    else:
        out = torch.full((total + _GUARD,), _INT_SENTINEL, dtype=dtype, device="mps")
    args = (out, *dev) if out_first else (*dev, out)
    kernel[(1,)](*args, **constexprs)
    torch.mps.synchronize()
    for sent, held in zip(inputs, dev):
        np.testing.assert_array_equal(held.cpu().numpy(), sent)
    return out.cpu().numpy()


def _close(got, want):
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=_RTOL, atol=_ATOL, equal_nan=True)


def _leads(values, largest=True):
    """True when the largest (smallest) value leads the next one by more than the margin."""
    order = np.sort(np.atleast_1d(values))
    if order.size < 2:
        return True
    return bool((order[-1] - order[-2] if largest else order[1] - order[0]) > _MARGIN)


# ---------------------------------------------------------------------------
# A three-dimensional tile reduced along one axis, stored as a two-dimensional tile.
# ---------------------------------------------------------------------------

_3D_SHAPES = [(4, 4, 4), (2, 8, 8), (16, 16, 4), (2, 4, 8), (8, 2, 4)]
_3D_MODES = {"sum": 0, "load_shifted": 1, "store_shifted": 2, "argmax": 4}


def _three_d_case(A, B, C, ax, mode):
    rng = np.random.default_rng([A, B, C, ax, mode, 6])
    y = _f32(rng, A * B * C + _GUARD)
    assert np.unique(y).size == y.size  # no ties: an index does not depend on a tie rule
    shift = 3 if mode == 1 else 0
    v = y[shift : shift + A * B * C].astype(np.float64).reshape(A, B, C)
    if mode == 5:
        v = v[:, :, ::-1]
    r = v.argmax(ax) if mode == 4 else v.sum(ax)
    if mode == 3:
        r = r * 2.0
    return y, r.reshape(-1), (5 if mode == 2 else 0)


@requires
@pytest.mark.parametrize("A,B,C", _3D_SHAPES)
def test_three_d_sum_along_the_last_axis_stored_as_a_tile_computes(A, B, C):
    """The reported form: `tl.store(O + a[:, None] * B + b[None, :], tl.sum(v, axis=2))`
    with the output as the first argument stored nothing."""
    y, cells, _ = _three_d_case(A, B, C, 2, 0)
    got = _launch(_three_d_last, cells.size, [y], A=A, B=B, C=C)
    _close(got, _cells(cells.size, cells))


@requires
@pytest.mark.parametrize("kind", list(_3D_MODES))
@pytest.mark.parametrize("ax", [2, 1, 0])
@pytest.mark.parametrize("A,B,C", _3D_SHAPES)
def test_three_d_reduction_stored_as_a_tile_computes(kind, ax, A, B, C):
    """Each axis, the input as the first argument; the tile or the store at an odd origin."""
    mode = _3D_MODES[kind]
    y, cells, at = _three_d_case(A, B, C, ax, mode)
    total = cells.size + at
    got = _launch(_three_d, total, [y], out_first=False, A=A, B=B, C=C, AX=ax, MODE=mode)
    _close(got, _cells(total, cells, at=at))


@requires
@pytest.mark.parametrize("ax", [2, 1, 0])
@pytest.mark.parametrize("A,B,C", [(4, 4, 4), (2, 8, 8)])
def test_three_d_reduction_then_an_operation_is_refused(ax, A, B, C):
    """The 3-D reduce template stores the reduction itself: an operation between the
    reduction and the store would be dropped, so the form is refused."""
    y, cells, _ = _three_d_case(A, B, C, ax, 3)
    with pytest.raises(
        MetalNonRecoverableError, match=r"3-D reduce template: the stored value is not the reduction result itself"
    ):
        _launch(_three_d, cells.size, [y], out_first=False, A=A, B=B, C=C, AX=ax, MODE=3)


@requires
@pytest.mark.parametrize("ax", [1, 0])
@pytest.mark.parametrize("A,B,C", [(4, 4, 4), (2, 8, 8)])
def test_three_d_reduction_of_a_tile_read_backwards_is_refused(ax, A, B, C):
    """An axis that is kept, read backwards: not one contiguous row-major tile."""
    y, cells, _ = _three_d_case(A, B, C, ax, 5)
    with pytest.raises(
        MetalNonRecoverableError,
        match=r"3-D reduce template: the load address is not one contiguous row-major tile of \d+ elements",
    ):
        _launch(_three_d, cells.size, [y], out_first=False, A=A, B=B, C=C, AX=ax, MODE=5)


@requires
@pytest.mark.parametrize("A,B,C", [(4, 4, 4), (2, 8, 8), (16, 16, 4)])
def test_three_d_sum_along_an_axis_read_backwards_computes(A, B, C):
    """The reduced axis read backwards: the same totals."""
    y, cells, _ = _three_d_case(A, B, C, 2, 5)
    got = _launch(_three_d, cells.size, [y], out_first=False, A=A, B=B, C=C, AX=2, MODE=5)
    _close(got, _cells(cells.size, cells))


# ---------------------------------------------------------------------------
# The index of the largest / smallest row reduction.
# ---------------------------------------------------------------------------

_ARG_KINDS = {
    "argmax_of_sums": 0,
    "argmin_of_sums": 1,
    "argmax_of_minima": 2,
    "argmin_of_maxima": 3,
    "index_of_max_of_sums": 4,
    "index_of_min_of_sums": 5,
}
_ARG_SIGNS = {"negative": -2.0, "mixed": 0.0, "positive": 2.0}
# At most 64 lanes first (the reported class), then wider tiles.
_ARG_SHAPES = [(4, 4), (2, 8), (8, 2), (4, 8), (8, 8), (16, 4), (4, 16), (2, 32), (32, 2), (16, 16), (64, 4), (4, 64)]


def _arg_case(kind, sign, ax, R, C):
    largest = kind in ("argmax_of_sums", "argmax_of_minima", "index_of_max_of_sums")
    for seed in range(200):
        rng = np.random.default_rng([R, C, ax, seed, 9])
        y = (rng.uniform(-1.0, 1.0, size=(R, C)) + _ARG_SIGNS[sign]).astype(np.float32)
        v = y.astype(np.float64)
        r = v.min(ax) if kind == "argmax_of_minima" else v.max(ax) if kind == "argmin_of_maxima" else v.sum(ax)
        want = int(r.argmax() if largest else r.argmin())
        if want == 0 or not _leads(r, largest):
            continue  # an answer of 0 must be a wrong answer, and the winner must be clear
        return y, want
    raise AssertionError("no seed gives a clear winner away from index 0")


def _arg_params():
    out = []
    for R, C in _ARG_SHAPES:
        for ax in (1, 0):
            for kind in _ARG_KINDS:
                largest = kind in ("argmax_of_sums", "argmax_of_minima", "index_of_max_of_sums")
                # the sign with which a lane holding zero would win, and the mixed one
                for sign in ("negative" if largest else "positive", "mixed"):
                    out.append(pytest.param(kind, sign, ax, R, C, id=f"{kind}-{sign}-ax{ax}-{R}x{C}"))
    return out


@requires
@pytest.mark.parametrize("kind,sign,ax,R,C", _arg_params())
def test_index_of_the_extreme_row_reduction_is_exact(kind, sign, ax, R, C):
    """All-negative totals answered 0 for tl.argmax (all-positive for tl.argmin) on tiles of
    at most 64 lanes: a lane outside the reduced value held zero and won."""
    y, want = _arg_case(kind, sign, ax, R, C)
    got = _launch(_arg_of_rows, 1, [y], dtype=torch.int32, R=R, C=C, AX=ax, MODE=_ARG_KINDS[kind])
    expect = np.full(1 + _GUARD, _INT_SENTINEL, dtype=np.int64)
    expect[0] = want
    np.testing.assert_array_equal(got.astype(np.int64), expect)


# ---------------------------------------------------------------------------
# The totals laid across a two-dimensional store.
# ---------------------------------------------------------------------------


@requires
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["other_axis", "outer_product", "own_axis"])
@pytest.mark.parametrize("ax", [1, 0])
@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_totals_of_a_square_tile_laid_across_a_store_compute(mode, ax, N):
    """`v + r[None, :]` for the axis-1 totals r, and `r[:, None] * r[None, :]`."""
    y = _f32(np.random.default_rng([N, ax, mode, 10]), N, N)
    v = y.astype(np.float64)
    r = v.sum(ax)
    along_rows = r[:, None] if ax == 1 else r[None, :]
    across = r[None, :] if ax == 1 else r[:, None]
    want = (v + across, np.outer(r, r), v + along_rows)[mode]
    got = _launch(_across, N * N, [y], N=N, AX=ax, MODE=mode)
    _close(got, _cells(N * N, want))


_OUTER_SHAPES = [(8, 2), (2, 8), (4, 4), (16, 4), (4, 16), (8, 16), (16, 8), (16, 16), (4, 64), (64, 4), (32, 8)]
_OUTER_KINDS = {
    "row_sums_met_in_a_row_square": 0,
    "column_maxima_met_in_a_column_square": 1,
    "larger_of_two_row_sums": 2,
    "tile_less_row_maxima_plus_column_minima": 3,
    "sum_of_column_sum_differences": 4,
    "max_of_row_sum_differences": 5,
}


def _outer_reference(mode, v):
    s1, s0, x0 = v.sum(1), v.sum(0), v.max(0)
    return (
        lambda: s1[:, None] - 2.0 * s1[None, :],
        lambda: x0[:, None] - 2.0 * x0[None, :],
        lambda: np.maximum(s1[:, None], s1[None, :]),
        lambda: v - v.max(1)[:, None] + v.min(0)[None, :],
        lambda: (s0[:, None] - s0[None, :]).sum(1),
        lambda: (s1[:, None] - 2.0 * s1[None, :]).max(0),
    )[mode]()


def _outer_params():
    out = []
    for kind, mode in _OUTER_KINDS.items():
        for R, C in _OUTER_SHAPES:
            side = (R, C, R, 1, C, R)[mode]
            if side * side > 1024:
                continue  # the square tile must fit one thread group
            out.append(pytest.param(mode, R, C, id=f"{kind}-{R}x{C}"))
    return out


@requires
@pytest.mark.parametrize("mode,R,C", _outer_params())
def test_reductions_met_across_a_tile_of_another_shape_compute(mode, R, C):
    """`x[:, None] - 2 * x[None, :]` for the totals x of an (R, C) tile is an (R, R) tile:
    its lanes are not the lanes of the tile the totals came from."""
    y = _f32(np.random.default_rng([R, C, mode, 11]), R, C)
    want = _outer_reference(mode, y.astype(np.float64))
    got = _launch(_outer, want.size, [y], R=R, C=C, MODE=mode)
    _close(got, _cells(want.size, want))


# ---------------------------------------------------------------------------
# The index of each line's largest element, used in an address.
# ---------------------------------------------------------------------------

_GATHER_SHAPES = [(16, 16), (8, 16), (32, 32), (4, 64), (64, 4), (2, 512)]
_POINTER_USER = (
    r"ttg.convert_layout of a pointer tensor \(addresses computed from a reduction result\) that is "
    r"used by something other than a plain tl.load"
)
_ROW_BLOCKED_EFFECT = (
    r"tt\.(store|atomic_rmw) of \d+ elements whose address, mask or value is computed from a row-wise "
    r"reduction \(held one row per \d+ lanes\) while another operand is held one element per lane"
)


def _gather_case(R, C, ax, own_winner=False, first_at_zero=False):
    """An (R, C) tile and a table of distinct values. `own_winner`: every line's largest
    element sits in a position of its own (needs at least as many positions as lines), so
    a scatter through the indices writes every cell once. `first_at_zero`: line 0 has its
    largest element at position 0 and line 1 at position 1, so a mask `index > 0` is
    false for one line and true for another."""
    rng = np.random.default_rng([R, C, ax, 12])
    y = _f32(rng, R, C)
    n, span = (R, C) if ax == 1 else (C, R)
    if own_winner:
        assert span >= n
        where = rng.permutation(span)[:n]
        for i in range(n):
            if ax == 1:
                y[i, where[i]] = 5.0 + i
            else:
                y[where[i], i] = 5.0 + i
    if first_at_zero:
        y[0, 0], y[1, 1] = 9.0, 8.0
    assert np.unique(y).size == y.size
    p = (rng.permutation(2 * span + _GUARD) - span).astype(np.float32)
    a = y.astype(np.float64).argmax(ax)
    assert not first_at_zero or (a[0] == 0 and a[1] == 1)
    return y, p, a, n, span


@requires
@pytest.mark.parametrize("mode", [0, 1, 2, 8, 9], ids=["table", "own_tile", "summed", "masked", "scaled_index"])
@pytest.mark.parametrize("ax", [1, 0])
@pytest.mark.parametrize("R,C", _GATHER_SHAPES)
def test_gather_through_the_index_of_the_largest_computes(mode, ax, R, C):
    """`tl.load(P + tl.argmax(v, axis))` and `tl.load(Y + rows * C + argmax)`, stored per
    line or summed, failed in the Metal compiler."""
    y, p, a, n, _span = _gather_case(R, C, ax, first_at_zero=mode == 8)
    table = p.astype(np.float64)
    want = (
        lambda: table[a],
        lambda: y.astype(np.float64).max(ax),
        lambda: table[a].sum(),
        lambda: np.where(a > 0, table[a], -3.0),
        lambda: table[a * 2 + 1],
    )[(0, 1, 2, 8, 9).index(mode)]()
    want = np.atleast_1d(want)
    got = _launch(_gather, want.size, [y, p], R=R, C=C, AX=ax, MODE=mode)
    _close(got, _cells(want.size, want))


@requires
@pytest.mark.parametrize("R,C", [(16, 4), (4, 8), (8, 8)])
def test_gather_of_each_rows_own_largest_in_a_narrow_tile_is_refused(R, C):
    """The same gather out of the tile itself where the row addresses are not declared."""
    y, p, _a, n, _span = _gather_case(R, C, 1)
    with pytest.raises(MetalNonRecoverableError, match=r"1-D store layout was not declared by its producer"):
        _launch(_gather, n, [y, p], R=R, C=C, AX=1, MODE=1)


def _scatter_reference(mode, a, n, span):
    """The expected output of the scatter / masked-store / atomic forms over NaN sentinels."""
    if mode == 5:
        return np.where(a > 0, np.arange(n, dtype=np.float64), np.nan), n
    want = np.full(span, np.nan)
    want[a] = np.arange(n) if mode in (3, 6) else 1.0
    return want, span


@requires
@pytest.mark.parametrize("mode", [3, 4, 6, 7], ids=["scatter", "scatter_one_number", "atomic", "two_loads"])
@pytest.mark.parametrize("ax,R,C", [(1, 16, 16), (1, 8, 16), (1, 4, 64), (0, 16, 16), (0, 64, 4), (0, 32, 8)])
def test_another_user_of_the_index_address_is_refused(mode, ax, R, C):
    """A scatter, an atomic update or a second load through the address computed from the
    index of the largest: lanes outside its elements would hold a stray address."""
    y, p, _a, n, span = _gather_case(R, C, ax, own_winner=True)
    with pytest.raises(MetalNonRecoverableError, match=_POINTER_USER):
        _launch(_gather, max(n, span), [y, p], R=R, C=C, AX=ax, MODE=mode)


@requires
@pytest.mark.parametrize("mode", [3, 5, 6], ids=["scatter", "masked_store", "atomic"])
@pytest.mark.parametrize("R,C", [(2, 8), (4, 4)])
def test_one_value_per_lane_through_a_row_blocked_address_or_mask_is_refused(mode, R, C):
    """The index of each row's largest is held one row per C lanes, the stored value one
    element per lane: lanes 0..R-1 would all use the first row's index."""
    y, p, _a, n, span = _gather_case(R, C, 1, own_winner=mode != 5, first_at_zero=mode == 5)
    with pytest.raises(MetalNonRecoverableError, match=_ROW_BLOCKED_EFFECT):
        _launch(_gather, max(n, span), [y, p], R=R, C=C, AX=1, MODE=mode)


@requires
@pytest.mark.parametrize(
    "mode,ax,R,C",
    [
        pytest.param(4, 1, 2, 8, id="scatter_one_number-ax1-2x8"),
        pytest.param(4, 1, 4, 4, id="scatter_one_number-ax1-4x4"),
        pytest.param(4, 0, 8, 2, id="scatter_one_number-ax0-8x2"),
        pytest.param(4, 0, 16, 4, id="scatter_one_number-ax0-16x4"),
        pytest.param(3, 0, 8, 2, id="scatter-ax0-8x2"),
        pytest.param(3, 0, 4, 4, id="scatter-ax0-4x4"),
        pytest.param(3, 0, 16, 4, id="scatter-ax0-16x4"),
        pytest.param(6, 0, 4, 4, id="atomic-ax0-4x4"),
        pytest.param(5, 0, 16, 16, id="masked_store-ax0-16x16"),
        pytest.param(5, 0, 8, 2, id="masked_store-ax0-8x2"),
        pytest.param(5, 0, 4, 64, id="masked_store-ax0-4x64"),
        pytest.param(5, 1, 16, 16, id="masked_store-ax1-16x16"),
        pytest.param(5, 1, 64, 4, id="masked_store-ax1-64x4"),
        pytest.param(5, 1, 32, 8, id="masked_store-ax1-32x8"),
    ],
)
def test_scatter_through_the_index_of_the_largest_computes(mode, ax, R, C):
    """The scatter forms that are served: every cell the indices name is written once, and
    no other cell."""
    y, p, a, n, span = _gather_case(R, C, ax, own_winner=mode != 5, first_at_zero=mode == 5)
    want, total = _scatter_reference(mode, a, n, span)
    if mode == 6:  # an atomic add starts from a defined value, guard cells included
        start = np.full(total + _GUARD, 0.5, dtype=np.float32)
        want = np.where(np.isnan(want), 0.5, want + 0.5)
        got = _launch_into(_gather, start, [y, p], R=R, C=C, AX=ax, MODE=mode)
        _close(got, _cells(total, want, base=start))
        return
    got = _launch(_gather, total, [y, p], R=R, C=C, AX=ax, MODE=mode)
    _close(got, _cells(total, want))


def _launch_into(kernel, start, inputs, **constexprs):
    """One program of `kernel` over an output that starts as `start`."""
    dev = [torch.from_numpy(x.copy()).to("mps") for x in inputs]
    out = torch.from_numpy(start.copy()).to("mps")
    kernel[(1,)](out, *dev, **constexprs)
    torch.mps.synchronize()
    for sent, held in zip(inputs, dev):
        np.testing.assert_array_equal(held.cpu().numpy(), sent)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# Two tiles of different shapes in one program.
# ---------------------------------------------------------------------------

_TWO_TILES = [
    (8, 16, 8, 8),
    (8, 16, 4, 8),
    (8, 16, 4, 4),
    (8, 16, 16, 4),
    (16, 16, 8, 8),
    (4, 64, 4, 4),
    (8, 16, 2, 16),
    (8, 16, 8, 4),
    (16, 4, 4, 16),
    (16, 4, 4, 4),
]
_TWO_TILE_KINDS = {
    "both_stored": 0,
    "second_a_ramp": 1,
    "second_alone": 2,
    "second_atomic": 3,
    "second_stored_and_totalled": 4,
    "ramp_stored_and_totalled": 5,
    "second_less_row_totals": 6,
    "second_less_column_maxima": 7,
}


def _two_tiles_reference(mode, y, R, C, R2, C2):
    """(expected cells, cell values before the launch or None for the sentinel)."""
    m = y.astype(np.float64)
    w = m[: R2 * C2]
    w2 = w.reshape(R2, C2)
    ramp = (np.arange(R2)[:, None] * 100.0 + np.arange(C2)[None, :]).reshape(-1)
    z2 = (ramp + m.sum()).reshape(R2, C2)
    return (
        lambda: np.concatenate([m, 2.0 * w]),
        lambda: np.concatenate([m, ramp]),
        lambda: w + m.sum(),
        lambda: np.concatenate([m, 2.0 * w + 0.25]),
        lambda: np.concatenate([w2.sum(1), w + m.sum()]),
        lambda: np.concatenate([z2.sum(1), z2.reshape(-1)]),
        lambda: (w2 - w2.sum(1)[:, None]).reshape(-1),
        lambda: (w2 - w2.max(0)[None, :] + m.sum()).reshape(-1),
    )[mode]()


@requires
@pytest.mark.parametrize("kind", list(_TWO_TILE_KINDS))
@pytest.mark.parametrize("R,C,R2,C2", _TWO_TILES)
def test_a_second_tile_of_another_shape_computes(kind, R, C, R2, C2):
    """The smaller tile was stored (or atomically updated) on the lanes of the larger one:
    cells past its end were written, and a tile that is stored and reduced lost rows."""
    mode = _TWO_TILE_KINDS[kind]
    y = _f32(np.random.default_rng([R, C, R2, C2, 13]), R * C)
    want = _two_tiles_reference(mode, y, R, C, R2, C2)
    if mode == 3:  # the atomic add starts from a defined value, guard cells included
        start = np.full(want.size + _GUARD, 0.25, dtype=np.float32)
        got = _launch_into(_two_tiles, start, [y], R=R, C=C, R2=R2, C2=C2, MODE=mode)
        _close(got, _cells(want.size, want, base=start))
        return
    got = _launch(_two_tiles, want.size, [y], R=R, C=C, R2=R2, C2=C2, MODE=mode)
    _close(got, _cells(want.size, want))


def _met_and_reduced_reference(y, R, C):
    x = y.astype(np.float64).reshape(R, C).sum(1)
    d = x[:, None] - 2.0 * x[None, :]
    return np.concatenate([d.sum(1), d.reshape(-1)])


@requires
@pytest.mark.parametrize("R,C", [(8, 16), (16, 16), (16, 4), (4, 4), (8, 2)])
def test_row_totals_met_across_a_tile_that_is_stored_and_reduced_compute(R, C):
    y = _f32(np.random.default_rng([R, C, 14]), R * C)
    want = _met_and_reduced_reference(y, R, C)
    got = _launch(_two_tiles, want.size, [y], R=R, C=C, R2=4, C2=4, MODE=8)
    _close(got, _cells(want.size, want))


@requires
def test_row_totals_met_across_a_tile_read_on_two_lane_spacings_is_refused():
    """The (4, 4) tile of the met totals is stored on the lanes of the (4, 64) tile and
    reduced on its own compact lanes: no one placement of the totals serves both."""
    R, C = 4, 64
    y = _f32(np.random.default_rng([R, C, 14]), R * C)
    want = _met_and_reduced_reference(y, R, C)
    with pytest.raises(
        MetalNonRecoverableError,
        match=r"tt.broadcast of a reduction result of shape \(4, 1\) over a \(4, 4\) tile: its consumers read "
        r"the rows of the target tile on different lanes",
    ):
        _launch(_two_tiles, want.size, [y], R=R, C=C, R2=4, C2=4, MODE=8)


_TWO_LOADED = [(16, 16, 32), (8, 32, 16), (16, 4, 64), (8, 2, 8), (4, 4, 16), (16, 4, 16), (4, 64, 4), (2, 8, 128)]
_TWO_LOADED_KINDS = {
    "index_of_each_row": 0,
    "cumsum_tile_and_row_maxima": 1,
    "both_tiles_stored": 2,
    "tile_less_row_totals_and_row_maxima": 3,
    "column_reductions": 5,
}


@requires
@pytest.mark.parametrize("kind", list(_TWO_LOADED_KINDS))
@pytest.mark.parametrize("R,C1,C2", _TWO_LOADED)
def test_two_loaded_tiles_on_one_row_range_compute(kind, R, C1, C2):
    """Two loaded tiles of different widths share `rows`: the index of each row's largest
    element of the narrower tile was stored with the wider tile's block width."""
    mode = _TWO_LOADED_KINDS[kind]
    rng = np.random.default_rng([R, C1, C2, 15])
    y, z = _f32(rng, R * C1), _f32(rng, R * C2)
    assert np.unique(y).size == y.size and np.unique(z).size == z.size
    a = y.astype(np.float64).reshape(R, C1)
    b = z.astype(np.float64).reshape(R, C2)
    want = {
        0: lambda: np.concatenate([a.argmax(1), b.argmin(1)]),
        1: lambda: np.concatenate([np.cumsum(a, 1).reshape(-1), b.max(1)]),
        2: lambda: np.concatenate([(a * 2.0).reshape(-1), (b + 1.0).reshape(-1)]),
        3: lambda: np.concatenate([(a - a.sum(1)[:, None]).reshape(-1), b.max(1)]),
        5: lambda: np.concatenate([a.sum(0), b.max(0)]),
    }[mode]()
    got = _launch(_two_loaded, want.size, [y, z], R=R, C1=C1, C2=C2, MODE=mode)
    _close(got, _cells(want.size, want))


_SPREAD = (
    r"{op} over a \({M}, {N}\) tile whose row range also addresses a wider tile: its rows are held {D} lanes apart"
)


@requires
@pytest.mark.parametrize(
    "kind", ["index_plus_whole_maximum", "row_totals_plus_whole_maximum", "largest_running_total_plus_whole_maximum"]
)
@pytest.mark.parametrize(
    "R,C1,C2",
    [(16, 16, 32), (16, 4, 64), (8, 2, 8), (4, 4, 16), (16, 4, 16), (2, 8, 128), (8, 2, 16), (8, 8, 8), (2, 4, 32)],
)
def test_a_row_reduction_of_the_narrower_of_two_tiles_plus_the_wider_tiles_maximum_computes(kind, R, C1, C2):
    """The narrower tile's rows sit as far apart as the wider tile's: its row reduction
    read the lanes of a compact tile, and took elements of the wrong rows."""
    mode = {"index_plus_whole_maximum": 7, "row_totals_plus_whole_maximum": 8}.get(kind, 9)
    rng = np.random.default_rng([R, C1, C2, 16])
    y, z = _f32(rng, R * C1), _f32(rng, R * C2)
    assert np.unique(y).size == y.size
    a = y.astype(np.float64).reshape(R, C1)
    top = z.astype(np.float64).max()
    want = {7: lambda: a.argmax(1) + top, 8: lambda: a.sum(1) + top, 9: lambda: np.cumsum(a, 1).max(1) + top}[mode]()
    got = _launch(_two_loaded, R, [y, z], R=R, C1=C1, C2=C2, MODE=mode)
    _close(got, _cells(R, want))


@requires
@pytest.mark.parametrize("R,C1,C2", [(16, 4, 16), (16, 4, 64), (2, 8, 128), (8, 2, 16)])
def test_a_scan_of_the_narrower_of_two_tiles_whose_rows_are_spread_is_refused(R, C1, C2):
    """tl.cumsum reads element (i, j) on lane i * C1 + j; the rows of the narrower tile
    sit C2 lanes apart. Only a single-value reduction gathers such a tile."""
    rng = np.random.default_rng([R, C1, C2, 17])
    y, z = _f32(rng, R * C1), _f32(rng, R * C2)
    with pytest.raises(MetalNonRecoverableError, match=_SPREAD.format(op="tt.scan", M=R, N=C1, D=C2)):
        _launch(_two_loaded, 1, [y, z], R=R, C1=C1, C2=C2, MODE=6)


def _unit_axis_case(R, C, seed):
    """A one-column or one-row tile of distinct signed values whose largest element is
    not the first, with float64 `w` = the tile less half its total."""
    y = _f32(np.random.default_rng([R, C, seed]), R * C)
    assert np.unique(y).size == y.size
    top = int(y.argmax())
    if top == 0:
        y[0], y[-1] = y[-1], y[0]
    m = y.astype(np.float64)
    return y, m, m - m.sum() * 0.5


_UNIT_AXIS_KINDS = {
    "total": 0,
    "index_of_the_largest": 1,
    "stored_as_a_tile": 2,
    "smallest": 3,
    "total_of_the_total": 4,
    "total_of_the_tile_times_its_largest": 5,
}
_UNIT_AXIS_SHAPES = [(64, 1), (16, 1), (8, 1), (4, 1), (2, 1), (1, 64), (1, 16), (1, 8), (1, 4), (1, 2)]


@requires
@pytest.mark.parametrize("kind", list(_UNIT_AXIS_KINDS))
@pytest.mark.parametrize("R,C", _UNIT_AXIS_SHAPES)
def test_a_one_column_or_one_row_tile_less_its_own_total_computes(kind, R, C):
    """`v - tl.sum(v, axis, keep_dims=True) * 0.5` of an (R, 1) or (1, C) tile, reduced
    again along its long axis: lane i holds element i (the one total is on every lane)."""
    mode = _UNIT_AXIS_KINDS[kind]
    y, m, w = _unit_axis_case(R, C, 18)
    want = {
        0: lambda: [w.sum()],
        1: lambda: [w.argmax()],
        2: lambda: w,
        3: lambda: [w.min()],
        4: lambda: [w.sum()],
        5: lambda: [(w * m.max()).sum()],
    }[mode]()
    assert mode != 1 or want[0] > 0
    got = _launch(_unit_axis, len(want), [y], R=R, C=C, K=2, MODE=mode)
    _close(got, _cells(len(want), want))


def _beside_a_wider_tile(y, R, C, K):
    k = np.arange(K, dtype=np.float64)
    m = y.astype(np.float64).reshape(R, C)
    return (m + k[None, :] if C == 1 else m + k[:, None]).reshape(-1)


@requires
@pytest.mark.parametrize(
    "R,C,K", [(64, 1, 4), (16, 1, 8), (8, 1, 2), (4, 1, 16), (2, 1, 2), (1, 16, 8), (1, 8, 2), (1, 4, 16), (1, 2, 2)]
)
def test_a_one_column_or_one_row_tile_beside_a_wider_tile_on_its_range_computes(R, C, K):
    """The tile's long range also addresses a stored tile K wide (or K tall), so its
    elements sit K lanes apart. The total of the tile less half its total is gathered."""
    y, _m, w = _unit_axis_case(R, C, 19)
    want = np.concatenate([_beside_a_wider_tile(y, R, C, K), [w.sum()]])
    got = _launch(_unit_axis, want.size, [y], R=R, C=C, K=K, MODE=6)
    _close(got, _cells(want.size, want))


@requires
@pytest.mark.parametrize("R,K", [(64, 4), (16, 8), (8, 2), (4, 16)])
def test_the_index_of_the_largest_of_a_one_column_tile_whose_rows_are_spread_is_refused(R, K):
    """An index reduction carries two values per lane and is not gathered: refused."""
    y, _m, _w = _unit_axis_case(R, 1, 20)
    with pytest.raises(MetalNonRecoverableError, match=_SPREAD.format(op="tt.reduce", M=R, N=1, D=K)):
        _launch(_unit_axis, R * K + 1, [y], R=R, C=1, K=K, MODE=7)


@requires
@pytest.mark.parametrize("R,K", [(64, 4), (16, 8), (8, 2), (2, 2)])
def test_a_one_column_tile_less_the_total_of_a_shorter_one_computes(R, K):
    """The K-row tile's rows sit R / K lanes apart; its gathered copy was then staged as
    if it were still spread, so its total took the wrong lanes."""
    y, m, _w = _unit_axis_case(R, 1, 21)
    want = [(m - m[:K].sum()).sum()]
    got = _launch(_unit_axis, 1, [y], R=R, C=1, K=K, MODE=8)
    _close(got, _cells(1, want))


# ---------------------------------------------------------------------------
# A matrix product: laid out by its own lowering, not by its operands' row ranges.
# ---------------------------------------------------------------------------


@requires
@pytest.mark.parametrize("M,N,K", [(32, 32, 64), (16, 32, 64), (16, 64, 32), (16, 32, 16), (16, 16, 16), (32, 32, 32)])
def test_the_rows_of_a_product_normalised_by_their_own_reductions_compute(M, N, K):
    """The row range of the product also addresses the (M, K) operand, wider than the
    product when K > N. The row reductions read the product where its lowering put it,
    and a product larger than the thread group is divided by its row totals in threadgroup
    memory: both computed before 2026-11-18 and are neither gathered nor refused."""
    rng = np.random.default_rng([M, N, K, 23])
    a, x = _f32(rng, M, K), _f32(rng, K, N)
    c = a.astype(np.float64) @ x.astype(np.float64)
    p = np.exp(c - c.max(1, keepdims=True))
    want = p / p.sum(1, keepdims=True)
    assert np.unique(c.argmax(1)).size > 1  # the rows differ in where their largest element is
    got = _launch(_product_rows, M * N, [a, x], M=M, N=N, K=K)
    _close(got, _cells(M * N, want))
