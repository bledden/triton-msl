"""A consumer of a row reduction: correct, or refused.

`tl.sum(v, axis=1)` of an (R, C) tile is a 1-D value of R elements that the generic
lowering holds "blocked": row r sits on lane r * C, one row per block of C lanes. A
consumer that took lanes 0..R-1 instead read R copies of the first row, and a second
reduction (`tl.sum(tl.sum(v, axis=1), axis=0)`) returned a wrong number with no error.

2026-11-18: every 1-D value records how it is held and with which block width, the second
reduction takes one lane of each block, and a reduction whose input mapping is not proved
is refused. This file runs the consumers of such a value on the device against a numpy
float64 (or exact integer) reference, over both axes, sum and max, square and non-square
tiles, tiles one row or one column wide, tiles of more than 1024 lanes, keep_dims forms,
a row reduction carried through a loop, index ranges that do not start at zero and
integer element types, and pins the refusals:

  * reduce over a row-blocked value whose block width is not one known number
  * reduce over a value computed from a row-wise reduce whose thread-to-element mapping
    is not proved
  * ttg.convert_layout of a value declared blocked whose width and reduce provenance do
    not agree (width 1, or two widths that meet)

The data is signed, so that a lane holding zero, or the first row again, cannot pass for
the largest or the smallest value. Every program is defined: one program, all its cells
distinct, the output separate from the inputs. Every output is filled with a sentinel and
compared whole, so a store outside the expected cells fails the test too.
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
    def _consume(O, Y, P, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, OP: tl.constexpr, MODE: tl.constexpr):
        """One reduction of an (R, C) tile along AX (OP 0 sum, 1 max), then one consumer."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        if OP == 0:
            r = tl.sum(v, axis=AX)
            keep = r > 0.0
        else:
            r = tl.max(v, axis=AX)
            keep = r > 0.9
        if AX == 1:
            idx = rows
        else:
            idx = cols
        tile = O + rows[:, None] * C + cols[None, :]
        if MODE == 0:
            tl.store(O + idx, r)
        elif MODE == 1:
            tl.store(O, tl.sum(r, axis=0))
        elif MODE == 2:
            tl.store(O, tl.max(r, axis=0))
        elif MODE == 3:
            tl.store(O, tl.min(r, axis=0))
        elif MODE == 4:
            tl.store(O, tl.argmax(r, axis=0).to(tl.float32))
        elif MODE == 5:
            tl.store(O + idx, r + tl.load(P + idx))
        elif MODE == 6:
            tl.store(O + idx, r + idx.to(tl.float32))
        elif MODE == 7:
            if AX == 1:
                tl.store(tile, v * r[:, None])
            else:
                tl.store(tile, v * r[None, :])
        elif MODE == 8:
            tl.store(O + idx, tl.where(keep, r, 0.0))
        elif MODE == 9:
            tl.store(O + idx, (r * 8.0).to(tl.int32).to(tl.float32))
        elif MODE == 10:
            tl.store(O + idx, tl.cumsum(r, axis=0))
        elif MODE == 11:
            tl.store(O + idx, tl.sin(r))
        elif MODE == 12:
            tl.store(O, tl.max(tl.sin(r), axis=0))
        elif MODE == 13:
            tl.store(tile, v * tl.sum(r, axis=0))
        elif MODE == 14:
            tl.store(O, tl.sum(r + tl.load(P + idx), axis=0))
        elif MODE == 15:
            tl.store(O, tl.max((r * 8.0).to(tl.int32), axis=0).to(tl.float32))
        elif MODE == 16:
            tl.store(O, tl.sum(tl.where(keep, r, 0.0), axis=0))
        elif MODE == 17:
            tl.store(O, tl.sum(tl.load(P + idx), axis=0) + tl.max(r, axis=0))
        elif MODE == 18:
            tl.store(O, tl.load(P + tl.argmax(r, axis=0)))
        elif MODE == 19:
            tl.store(O + idx, r - tl.max(r, axis=0))
        elif MODE == 20:
            tl.store(O, tl.sum(idx, axis=0).to(tl.float32) + tl.max(r, axis=0))
        elif MODE == 21:
            tl.store(O, tl.max(tl.where(tl.load(P + idx) > 0.0, r, 0.0), axis=0))

    @triton.jit
    def _both_axes(O, Y, N: tl.constexpr, MODE: tl.constexpr):
        """A square tile reduced along both axes, the two results met in one value."""
        i = tl.arange(0, N)
        v = tl.load(Y + i[:, None] * N + i[None, :])
        a = tl.sum(v, axis=0)
        b = tl.sum(v, axis=1)
        if MODE == 0:
            tl.store(O + i, a + 2.0 * b)
        else:
            tl.store(O, tl.max(a + 2.0 * b, axis=0))

    @triton.jit
    def _two_widths(O, Y, Z, P, R: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, MODE: tl.constexpr):
        """Row reductions of two tiles of different widths in one program."""
        rows = tl.arange(0, R)
        c1 = tl.arange(0, C1)
        c2 = tl.arange(0, C2)
        a = tl.load(Y + rows[:, None] * C1 + c1[None, :])
        b = tl.load(Z + rows[:, None] * C2 + c2[None, :])
        ra = tl.sum(a, axis=1)
        rb = tl.max(b, axis=1)
        if MODE == 0:  # each reduced on its own: both widths are known
            tl.store(O, tl.sum(ra, axis=0) + tl.sum(rb, axis=0))
        elif MODE == 1:
            tl.store(O + rows, ra)
            tl.store(O + R + rows, rb)
        elif MODE == 2:
            tl.store(O, tl.sum(ra, axis=0))
            tl.store(O + 1, tl.max(rb, axis=0))
        elif MODE == 3:
            tl.store(O, tl.sum(ra + tl.load(P + rows), axis=0))
        elif MODE == 4:
            tl.store(O, tl.sum(rb + tl.load(P + rows), axis=0))
        elif MODE == 5:  # the two meet in one value: no one width
            tl.store(O, tl.sum(tl.where(tl.load(P + rows) > 0.0, ra, rb), axis=0))
        elif MODE == 6:
            tl.store(O, tl.max(tl.where(tl.load(P + rows) > 0.0, ra, rb), axis=0))
        elif MODE == 7:
            tl.store(O, tl.sum(tl.where(ra > rb, 1.0, 0.0), axis=0))
        elif MODE == 8:
            tl.store(O, tl.sum(tl.abs(ra) + tl.exp(rb), axis=0))
        elif MODE == 9:
            tl.store(O + rows, tl.where(tl.load(P + rows) > 0.0, ra, rb))

    @triton.jit
    def _keep(O, Y, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, OP: tl.constexpr, MODE: tl.constexpr):
        """keep_dims forms of a reduction of an (R, C) tile along AX."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        if OP == 0:
            r = tl.sum(v, axis=AX, keep_dims=True)
        else:
            r = tl.max(v, axis=AX, keep_dims=True)
        if AX == 1:
            idx = rows
        else:
            idx = cols
        tile = O + rows[:, None] * C + cols[None, :]
        if MODE == 0:
            tl.store(tile, v * r)
        elif MODE == 1:
            tl.store(O, tl.sum(r))
        elif MODE == 2:
            tl.store(O, tl.max(r))
        elif MODE == 3:
            tl.store(O + idx, tl.reshape(r, (R if AX == 1 else C,)))
        elif MODE == 4:
            tl.store(O, tl.sum(tl.reshape(r, (R if AX == 1 else C,)), axis=0))
        elif MODE == 5:
            r1 = tl.sum(v, axis=AX)
            tl.store(O + idx, r1 * tl.sum(r1, axis=0, keep_dims=True))
        elif MODE == 6:
            r1 = tl.max(v, axis=AX)
            tl.store(O + idx, r1 - tl.max(r1, axis=0, keep_dims=True))
        elif MODE == 7:  # the kept axis squeezed by a sum, then reduced
            tl.store(O, tl.max(tl.sum(r, axis=AX), axis=0))
        elif MODE == 8:  # the kept axis squeezed by a sum, then stored
            tl.store(O + idx, tl.sum(r, axis=AX))
        elif MODE == 9:  # the other axis reduced with keep_dims too: one number for the tile
            tl.store(tile, v * tl.sum(r, axis=1 - AX, keep_dims=True))

    @triton.jit
    def _loop_carry(
        O, Y, ITERS, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, OP: tl.constexpr, MODE: tl.constexpr
    ):
        """A reduction along AX accumulated over ITERS tiles, then consumed."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        if AX == 1:
            idx = rows
            acc = tl.zeros((R,), dtype=tl.float32)
        else:
            idx = cols
            acc = tl.zeros((C,), dtype=tl.float32)
        for k in range(ITERS):
            v = tl.load(Y + k * (R * C) + rows[:, None] * C + cols[None, :])
            if OP == 0:
                acc += tl.sum(v, axis=AX)
            else:
                acc = tl.maximum(acc, tl.max(v, axis=AX))
        if MODE == 0:
            tl.store(O + idx, acc)
        elif MODE == 1:
            tl.store(O, tl.sum(acc, axis=0))
        elif MODE == 2:
            tl.store(O, tl.max(acc, axis=0))
        elif MODE == 3:
            tl.store(O + idx, acc - tl.min(acc, axis=0))
        elif MODE == 4:
            tl.store(O, tl.argmax(acc, axis=0).to(tl.float32))
        elif MODE == 5:
            tl.store(O, tl.sum(acc * 2.0 + idx.to(tl.float32), axis=0))
        elif MODE == 6:
            tl.store(O + idx, acc + idx.to(tl.float32))

    @triton.jit
    def _shifted(
        O,
        Y,
        P,
        R: tl.constexpr,
        C: tl.constexpr,
        R0: tl.constexpr,
        C0: tl.constexpr,
        AX: tl.constexpr,
        OP: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """The tile's index ranges start at R0 and C0, not at zero."""
        rows = tl.arange(R0, R0 + R)
        cols = tl.arange(C0, C0 + C)
        v = tl.load(Y + rows[:, None] * (C0 + C) + cols[None, :])
        if OP == 0:
            r = tl.sum(v, axis=AX)
        else:
            r = tl.max(v, axis=AX)
        if AX == 1:
            idx = rows
        else:
            idx = cols
        if MODE == 0:
            tl.store(O + idx, r)
        elif MODE == 1:
            tl.store(O, tl.sum(r, axis=0))
        elif MODE == 2:
            tl.store(O, tl.max(r, axis=0))
        elif MODE == 3:
            tl.store(O + idx, r + idx.to(tl.float32))
        elif MODE == 4:
            tl.store(O, tl.sum(idx, axis=0).to(tl.float32) + tl.max(r, axis=0))
        elif MODE == 5:
            tl.store(O + idx, r + tl.load(P + idx))
        elif MODE == 6:
            tl.store(O, tl.sum(r + tl.load(P + idx), axis=0))
        elif MODE == 7:
            tl.store(O, tl.min(r * idx.to(tl.float32), axis=0))

    @triton.jit
    def _consume_int(O, Y, P, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr, OP: tl.constexpr, MODE: tl.constexpr):
        """An integer tile reduced along AX (OP 0 sum, 1 max, 2 min), then one consumer."""
        rows = tl.arange(0, R)
        cols = tl.arange(0, C)
        v = tl.load(Y + rows[:, None] * C + cols[None, :])
        if OP == 0:
            r = tl.sum(v, axis=AX)
        elif OP == 1:
            r = tl.max(v, axis=AX)
        else:
            r = tl.min(v, axis=AX)
        if AX == 1:
            idx = rows
        else:
            idx = cols
        if MODE == 0:
            tl.store(O + idx, r)
        elif MODE == 1:
            tl.store(O, tl.sum(r, axis=0))
        elif MODE == 2:
            tl.store(O, tl.max(r, axis=0))
        elif MODE == 3:
            tl.store(O, tl.min(r, axis=0))
        elif MODE == 4:
            tl.store(O, tl.argmax(r, axis=0))
        elif MODE == 5:
            tl.store(O + idx, r + tl.load(P + idx))
        elif MODE == 6:
            tl.store(O + idx, r + idx)
        elif MODE == 7:
            tl.store(O, tl.sum(r * 2 + idx, axis=0))
        elif MODE == 8:
            tl.store(O + idx, tl.where(r > 1, r, 0))
        elif MODE == 9:
            tl.store(O, tl.xor_sum(r, axis=0))


_GUARD = 8
_INT_SENTINEL = -777777
_RTOL = _ATOL = 5e-4
_MARGIN = 2e-3  # a float32 total may differ from the float64 one by far less than this


def _f32(rng, *shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _cells(total, values, at=0):
    """The whole expected output: `values` from cell `at`, the sentinel (NaN) elsewhere."""
    want = np.full(total + _GUARD, np.nan)
    flat = np.atleast_1d(np.asarray(values, dtype=np.float64)).reshape(-1)
    want[at : at + flat.size] = flat
    return want


def _launch(kernel, total, inputs, dtype=None, **constexprs):
    """One program of `kernel` over a sentinel-filled output of `total` + guard cells."""
    dev = [torch.from_numpy(x).to("mps") if isinstance(x, np.ndarray) else x for x in inputs]
    if dtype is None:
        out = torch.full((total + _GUARD,), float("nan"), dtype=torch.float32, device="mps")
    else:
        out = torch.full((total + _GUARD,), _INT_SENTINEL, dtype=dtype, device="mps")
    kernel[(1,)](out, *dev, **constexprs)
    torch.mps.synchronize()
    return out.cpu().numpy()


def _close(got, want):
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=_RTOL, atol=_ATOL, equal_nan=True)


def _clear(values, of=None):
    """True when no value is within the margin of `of` (default: of an integer)."""
    values = np.atleast_1d(values)
    near = np.abs(values - np.round(values)) if of is None else np.abs(values - of)
    return bool(near.size == 0 or near.min() > _MARGIN)


def _unique_top(values):
    """True when the largest value leads the second largest by more than the margin."""
    top = np.sort(np.atleast_1d(values))[::-1]
    return bool(top.size < 2 or top[0] - top[1] > _MARGIN)


# ---------------------------------------------------------------------------
# One reduction, one consumer.
# ---------------------------------------------------------------------------

# kind -> (MODE, number of shapes per axis and op). The first eleven are the plain
# consumers; the rest put a second operation between the reduction and its consumer.
_KINDS = {
    "store": (0, 3),
    "sum": (1, 3),
    "max": (2, 3),
    "min": (3, 3),
    "argmax": (4, 3),
    "add_load": (5, 3),
    "add_index": (6, 3),
    "broadcast_back": (7, 3),
    "where": (8, 3),
    "to_int": (9, 3),
    "sin": (11, 3),
    "max_of_sin": (12, 1),
    "total_back_into_tile": (13, 1),
    "sum_of_add_load": (14, 1),
    "max_of_to_int": (15, 1),
    "sum_of_where": (16, 1),
    "sum_of_load_plus_max": (17, 1),
    "argmax_as_address": (18, 1),
    "minus_own_max": (19, 1),
    "sum_of_index_plus_max": (20, 1),
    "max_of_where_on_load": (21, 1),
}
_CUMSUM = 10

# Tiles of at most 1024 lanes. The first list is where a second reduction along axis 1
# has more than one row and more than one lane per row; the second is the edges.
_SHAPES = [(16, 16), (32, 32), (8, 16), (64, 16), (16, 64), (4, 256), (2, 512), (256, 4), (1, 64)]
_EDGES = [(64, 1), (1, 16), (16, 1), (4, 8), (16, 4), (8, 8), (2, 2)]


def _consume_reference(kind, v, p, ax, op):
    """(expected cells, whether the data keeps clear of every rounding-sensitive edge)."""
    r = v.sum(ax) if op == 0 else v.max(ax)
    n = r.size
    idx = np.arange(n)
    thr = 0.0 if op == 0 else 0.9
    kept = np.where(r > thr, r, 0.0)
    ints = np.trunc(r * 8.0)
    if kind == "store":
        return r, True
    if kind == "sum":
        return r.sum(), True
    if kind == "max":
        return r.max(), True
    if kind == "min":
        return r.min(), True
    if kind == "argmax":
        return float(r.argmax()), _unique_top(r)
    if kind == "add_load":
        return r + p[:n], True
    if kind == "add_index":
        return r + idx, True
    if kind == "broadcast_back":
        return v * (r[:, None] if ax == 1 else r[None, :]), True
    if kind == "where":
        return kept, _clear(r, of=thr)
    if kind == "to_int":
        return ints, _clear(r * 8.0)
    if kind == "cumsum":
        return np.cumsum(r), True
    if kind == "sin":
        return np.sin(r), True
    if kind == "max_of_sin":
        return np.sin(r).max(), True
    if kind == "total_back_into_tile":
        return v * r.sum(), True
    if kind == "sum_of_add_load":
        return (r + p[:n]).sum(), True
    if kind == "max_of_to_int":
        return ints.max(), _clear(r * 8.0)
    if kind == "sum_of_where":
        return kept.sum(), _clear(r, of=thr)
    if kind == "sum_of_load_plus_max":
        return p[:n].sum() + r.max(), True
    if kind == "argmax_as_address":
        return p[r.argmax()], _unique_top(r)
    if kind == "minus_own_max":
        return r - r.max(), True
    if kind == "sum_of_index_plus_max":
        return idx.sum() + r.max(), True
    if kind == "max_of_where_on_load":
        return np.where(p[:n] > 0.0, r, 0.0).max(), _clear(p[:n], of=0.0)
    raise AssertionError(kind)


def _consume_case(kind, ax, op, R, C):
    """Inputs and the expected cells, from the first seed that keeps clear of the edges."""
    for seed in range(40):
        rng = np.random.default_rng([R, C, ax, op, seed])
        y, p = _f32(rng, R, C), _f32(rng, max(R, C) + _GUARD)
        cells, clear = _consume_reference(kind, y.astype(np.float64), p.astype(np.float64), ax, op)
        if clear:
            return y, p, np.asarray(cells, dtype=np.float64)
    raise AssertionError("no seed keeps clear of the rounding-sensitive edges")


def _width_one(ax, R, C):
    """An (R, 1) tile reduced along its one-lane axis: the value is declared blocked with
    width 1, which no row-wise reduce produces, and its conversion is refused."""
    return ax == 1 and C == 1 and R > 1


def _consume_params():
    computes, refuses = [], []
    combo = 0
    for kind, (mode, count) in _KINDS.items():
        for ax in (1, 0):
            for op in (0, 1):
                if count == 3:
                    shapes = [
                        _SHAPES[combo % len(_SHAPES)],
                        _SHAPES[(combo + 4) % len(_SHAPES)],
                        _EDGES[combo % len(_EDGES)],
                    ]
                elif combo % 3:
                    shapes = [_SHAPES[combo % len(_SHAPES)]]
                else:
                    shapes = [_EDGES[combo % len(_EDGES)]]
                combo += 1
                for R, C in shapes:
                    case = pytest.param(
                        kind, mode, ax, op, R, C, id=f"{kind}-ax{ax}-{'sum' if op == 0 else 'max'}-{R}x{C}"
                    )
                    if _width_one(ax, R, C) and kind != "broadcast_back":
                        refuses.append(case)
                    else:
                        computes.append(case)
    return computes, refuses


_CONSUME_COMPUTES, _CONSUME_WIDTH_ONE = _consume_params()


def _run_consume(kind, mode, ax, op, R, C):
    y, p, cells = _consume_case(kind, ax, op, R, C)
    total = R * C if cells.ndim == 2 else cells.size
    got = _launch(_consume, total, [y, p], R=R, C=C, AX=ax, OP=op, MODE=mode)
    return got, _cells(total, cells)


@requires
@pytest.mark.parametrize("kind,mode,ax,op,R,C", _CONSUME_COMPUTES)
def test_consumer_of_a_reduction_computes(kind, mode, ax, op, R, C):
    """Each consumer of `tl.sum` / `tl.max` of a tile along either axis matches numpy."""
    got, want = _run_consume(kind, mode, ax, op, R, C)
    _close(got, want)


@requires
@pytest.mark.parametrize("kind,mode,ax,op,R,C", _CONSUME_WIDTH_ONE)
def test_consumer_of_a_one_lane_row_reduction_is_refused(kind, mode, ax, op, R, C):
    """An (R, 1) tile reduced along axis 1 is declared blocked with width 1: refused."""
    with pytest.raises(MetalNonRecoverableError, match=r"declared blocked \(width 1\), but reduce provenance is"):
        _run_consume(kind, mode, ax, op, R, C)


@requires
@pytest.mark.parametrize("ax,op,R,C", [(0, 0, 1, 64), (0, 1, 1, 16)])
def test_cumsum_of_a_one_row_tile_reduction_computes(ax, op, R, C):
    got, want = _run_consume("cumsum", _CUMSUM, ax, op, R, C)
    _close(got, want)


@requires
@pytest.mark.parametrize("ax,op,R,C", [(1, 0, 16, 16), (0, 0, 16, 16), (1, 1, 8, 16), (0, 1, 64, 16), (1, 0, 4, 8)])
def test_cumsum_of_a_reduction_is_refused(ax, op, R, C):
    """`tl.cumsum` of a reduction result has no declared layout to store from: refused."""
    with pytest.raises(MetalNonRecoverableError, match="1-D store layout was not declared by its producer"):
        _run_consume("cumsum", _CUMSUM, ax, op, R, C)


# ---------------------------------------------------------------------------
# Tiles of more than 1024 lanes.
# ---------------------------------------------------------------------------


@requires
@pytest.mark.parametrize("kind", ["store", "sum", "max", "argmax", "add_load"])
@pytest.mark.parametrize("op", [0, 1], ids=["sum", "max"])
def test_one_row_of_2048_lanes_computes(kind, op):
    """A (1, 2048) tile reduced along axis 1 is one number; its consumers match numpy."""
    got, want = _run_consume(kind, _KINDS[kind][0], 1, op, 1, 2048)
    _close(got, want)


@requires
@pytest.mark.parametrize("kind", ["store", "sum", "argmax"])
@pytest.mark.parametrize("ax", [1, 0])
@pytest.mark.parametrize("R,C", [(32, 64), (2, 1024), (512, 4)])
def test_many_rows_over_1024_lanes_is_refused(kind, ax, R, C):
    """More than one row and more than 1024 lanes: no lowering, and no wrong number."""
    with pytest.raises(MetalNonRecoverableError, match="Refusing"):
        _run_consume(kind, _KINDS[kind][0], ax, 0, R, C)


# ---------------------------------------------------------------------------
# Two reductions meeting in one value.
# ---------------------------------------------------------------------------


@requires
@pytest.mark.parametrize("N", [16, 32])
def test_reduce_of_both_axes_met_is_refused(N):
    """`tl.max(tl.sum(v, 0) + 2 * tl.sum(v, 1), axis=0)`: one operand is held one element
    per lane and the other one row per block, so no lane mapping is proved."""
    rng = np.random.default_rng([N, 1])
    with pytest.raises(
        MetalNonRecoverableError, match=r"reduce \('max'\) over a \d+-element value computed from a row-wise reduce"
    ):
        _launch(_both_axes, 1, [_f32(rng, N, N)], N=N, MODE=1)


@requires
@pytest.mark.parametrize("N", [16, 32])
def test_store_of_both_axes_met_is_refused(N):
    rng = np.random.default_rng([N, 0])
    with pytest.raises(
        MetalNonRecoverableError, match="ttg.convert_layout cannot prove its source's thread-to-element mapping"
    ):
        _launch(_both_axes, N, [_f32(rng, N, N)], N=N, MODE=0)


@requires
@pytest.mark.parametrize("mode", [0, 1], ids=["store", "max"])
@pytest.mark.parametrize("N", [2, 4, 8])
def test_both_axes_met_in_a_small_tile_compute(mode, N):
    """A tile of at most 64 lanes lowers the same two programs, and matches numpy."""
    rng = np.random.default_rng([N, 2])
    y = _f32(rng, N, N)
    met = y.astype(np.float64).sum(0) + 2.0 * y.astype(np.float64).sum(1)
    cells = met if mode == 0 else [met.max()]
    got = _launch(_both_axes, len(cells), [y], N=N, MODE=mode)
    _close(got, _cells(len(cells), cells))


_TWO_WIDTHS = [(16, 16, 32), (8, 32, 16), (16, 4, 64), (32, 8, 16)]


def _two_widths_case(R, C1, C2):
    for seed in range(40):
        rng = np.random.default_rng([R, C1, C2, seed])
        a, b, p = _f32(rng, R, C1), _f32(rng, R, C2), _f32(rng, R + _GUARD)
        ra, rb = a.astype(np.float64).sum(1), b.astype(np.float64).max(1)
        if _clear(ra - rb, of=0.0) and _clear(p[:R].astype(np.float64), of=0.0):
            return a, b, p, ra, rb
    raise AssertionError("no seed keeps clear of the rounding-sensitive edges")


@requires
@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("R,C1,C2", _TWO_WIDTHS[:2])
def test_two_tile_widths_reduced_apart_compute(mode, R, C1, C2):
    """Two row reductions of different block widths, each consumed on its own."""
    a, b, p, ra, rb = _two_widths_case(R, C1, C2)
    load = p[:R].astype(np.float64)
    cells = {
        0: [ra.sum() + rb.sum()],
        1: np.concatenate([ra, rb]),
        2: [ra.sum(), rb.max()],
        3: [(ra + load).sum()],
        4: [(rb + load).sum()],
    }[mode]
    total = len(cells)
    got = _launch(_two_widths, total, [a, b, p], R=R, C1=C1, C2=C2, MODE=mode)
    _close(got, _cells(total, cells))


@requires
@pytest.mark.parametrize("mode,op", [(5, "sum"), (6, "max"), (7, "sum"), (8, "sum")])
@pytest.mark.parametrize("R,C1,C2", _TWO_WIDTHS[1:3])
def test_reduce_of_two_tile_widths_met_is_refused(mode, op, R, C1, C2):
    """A value selected or summed from row reductions of two block widths has no one
    width, so there is no one lane of each block to take."""
    a, b, p, _, _ = _two_widths_case(R, C1, C2)
    with pytest.raises(
        MetalNonRecoverableError,
        match=rf"reduce \('{op}'\) over a row-blocked value whose block width is not one known number",
    ):
        _launch(_two_widths, 1, [a, b, p], R=R, C1=C1, C2=C2, MODE=mode)


@requires
@pytest.mark.parametrize("R,C1,C2", _TWO_WIDTHS[2:])
def test_store_of_two_tile_widths_met_is_refused(R, C1, C2):
    a, b, p, _, _ = _two_widths_case(R, C1, C2)
    with pytest.raises(MetalNonRecoverableError, match=r"declared blocked \(width None\), but reduce provenance is"):
        _launch(_two_widths, R, [a, b, p], R=R, C1=C1, C2=C2, MODE=9)


@requires
@pytest.mark.parametrize("mode", [5, 6, 7, 8, 9])
def test_two_tile_widths_met_in_a_small_tile_compute(mode):
    """With (4, 8) and (4, 16) tiles the same programs lower, and match numpy."""
    R, C1, C2 = 4, 8, 16
    a, b, p, ra, rb = _two_widths_case(R, C1, C2)
    picked = np.where(p[:R].astype(np.float64) > 0.0, ra, rb)
    cells = {
        5: [picked.sum()],
        6: [picked.max()],
        7: [float((ra > rb).sum())],
        8: [(np.abs(ra) + np.exp(rb)).sum()],
        9: picked,
    }[mode]
    total = len(cells)
    got = _launch(_two_widths, total, [a, b, p], R=R, C1=C1, C2=C2, MODE=mode)
    _close(got, _cells(total, cells))


# ---------------------------------------------------------------------------
# keep_dims forms.
# ---------------------------------------------------------------------------

_KEEP_SHAPES = [(16, 16), (8, 16), (4, 64), (1, 64), (4, 8), (32, 32), (64, 4), (2, 512)]


def _keep_reference(mode, v, ax, op):
    r = v.sum(ax, keepdims=True) if op == 0 else v.max(ax, keepdims=True)
    if mode == 0:
        return v * r
    if mode == 1:
        return r.sum()
    if mode == 2:
        return r.max()
    if mode == 3:
        return r.reshape(-1)
    if mode == 4:
        return r.sum()
    if mode == 5:
        r1 = v.sum(ax)
        return r1 * r1.sum()
    if mode == 6:
        r1 = v.max(ax)
        return r1 - r1.max()
    if mode == 7:
        return r.sum(ax).max()
    if mode == 8:
        return r.sum(ax)
    if mode == 9:
        return v * r.sum(1 - ax, keepdims=True)
    raise AssertionError(mode)


def _keep_params():
    computes, refuses = [], []
    combo = 0
    for mode in range(10):
        for ax in (1, 0):
            for op in (0, 1):
                if mode in (5, 6) and op == 1:
                    continue  # these two fix their own reduction
                shapes = [_KEEP_SHAPES[combo % len(_KEEP_SHAPES)]]
                if mode == 8 and ax == 1:
                    shapes += [(1, 64), (8, 16)] if op == 0 else [(1, 16), (64, 4)]
                combo += 1
                for R, C in shapes:
                    case = pytest.param(
                        mode, ax, op, R, C, id=f"mode{mode}-ax{ax}-{'sum' if op == 0 else 'max'}-{R}x{C}"
                    )
                    # `tl.sum(r, axis=1)` of the kept (R, 1) tile is the one-lane row
                    # reduction again: declared blocked with width 1, and stored
                    if mode == 8 and ax == 1 and R > 1:
                        refuses.append(case)
                    else:
                        computes.append(case)
    return computes, refuses


_KEEP_COMPUTES, _KEEP_WIDTH_ONE = _keep_params()


def _run_keep(mode, ax, op, R, C):
    rng = np.random.default_rng([R, C, ax, op, 77])
    y = _f32(rng, R, C)
    cells = np.asarray(_keep_reference(mode, y.astype(np.float64), ax, op))
    total = cells.size
    got = _launch(_keep, total, [y], R=R, C=C, AX=ax, OP=op, MODE=mode)
    return got, _cells(total, cells)


@requires
@pytest.mark.parametrize("mode,ax,op,R,C", _KEEP_COMPUTES)
def test_keep_dims_reduction_consumers_compute(mode, ax, op, R, C):
    """A reduction taken with keep_dims=True, used as a tile operand, squeezed and reduced
    again, or reduced along the other axis, matches numpy."""
    got, want = _run_keep(mode, ax, op, R, C)
    _close(got, want)


@requires
@pytest.mark.parametrize("mode,ax,op,R,C", _KEEP_WIDTH_ONE)
def test_store_of_a_squeezed_keep_dims_row_reduction_is_refused(mode, ax, op, R, C):
    """`tl.store(O + rows, tl.sum(tl.sum(v, axis=1, keep_dims=True), axis=1))`: the second
    sum is over an (R, 1) tile, so its result is declared blocked with width 1: refused."""
    with pytest.raises(MetalNonRecoverableError, match=r"declared blocked \(width 1\), but reduce provenance is"):
        _run_keep(mode, ax, op, R, C)


# ---------------------------------------------------------------------------
# A row reduction carried through a loop, then consumed.
# ---------------------------------------------------------------------------

_ITERS = 3
_LOOP_SHAPES = [(16, 16), (8, 32), (32, 8), (4, 64), (64, 4), (32, 32), (2, 128), (16, 64), (1, 256)]


def _loop_case(ax, op, R, C, need_unique_top=False):
    for seed in range(40):
        rng = np.random.default_rng([R, C, ax, op, 5, seed])
        y = _f32(rng, _ITERS, R, C)
        t = y.astype(np.float64)
        acc = t.sum(1 + ax).sum(0) if op == 0 else np.maximum(0.0, t.max(1 + ax).max(0))
        if not need_unique_top or _unique_top(acc):
            return y, acc
    raise AssertionError("no seed gives one clear largest element")


def _loop_params(modes, axes, start):
    out = []
    combo = start
    for mode in modes:
        for ax in axes:
            for op in (0, 1):
                for R, C in (_LOOP_SHAPES[combo % len(_LOOP_SHAPES)], _LOOP_SHAPES[(combo + 4) % len(_LOOP_SHAPES)]):
                    out.append(
                        pytest.param(mode, ax, op, R, C, id=f"mode{mode}-ax{ax}-{'sum' if op == 0 else 'max'}-{R}x{C}")
                    )
                combo += 1
    return out


def _run_loop(mode, ax, op, R, C):
    y, acc = _loop_case(ax, op, R, C, need_unique_top=mode == 4)
    n = acc.size
    cells = {
        0: acc,
        1: acc.sum(),
        2: acc.max(),
        3: acc - acc.min(),
        4: float(acc.argmax()),
        5: (acc * 2.0 + np.arange(n)).sum(),
        6: acc + np.arange(n),
    }[mode]
    total = n if mode in (0, 3, 6) else 1
    got = _launch(_loop_carry, total, [y, _ITERS], R=R, C=C, AX=ax, OP=op, MODE=mode)
    return got, _cells(total, cells)


@requires
@pytest.mark.parametrize("mode,ax,op,R,C", _loop_params((0, 1, 2, 4), (1,), 0) + _loop_params((0, 3), (0,), 2))
def test_loop_carried_reduction_consumers_compute(mode, ax, op, R, C):
    """Row totals (or maxima) accumulated over three tiles, then stored or reduced, match
    numpy. Along axis 0 they are also compared with their own minimum; along axis 1 that
    program uses the loop result twice, and upstream Triton's
    TritonGPUOptimizeThreadLocality pass asserts on it before this backend runs."""
    got, want = _run_loop(mode, ax, op, R, C)
    _close(got, want)


@requires
@pytest.mark.parametrize("op", [0, 1], ids=["sum", "max"])
@pytest.mark.parametrize("R,C", [(16, 16), (8, 32), (64, 4)])
def test_reduce_of_loop_carried_rows_with_their_index_is_refused(op, R, C):
    """`tl.sum(acc * 2 + rows)` after the loop: the carried row totals meet the row index,
    which is held one element per lane, so no lane mapping is proved."""
    with pytest.raises(
        MetalNonRecoverableError, match=r"reduce \('sum'\) over a \d+-element value computed from a row-wise reduce"
    ):
        _run_loop(5, 1, op, R, C)


@requires
@pytest.mark.parametrize("op", [0, 1], ids=["sum", "max"])
@pytest.mark.parametrize("R,C", [(16, 16), (4, 64), (64, 4), (8, 32)])
def test_store_of_loop_carried_rows_plus_their_index_is_refused(op, R, C):
    """`tl.store(O + rows, acc + rows)` after the loop: the same meeting, stored. The sum
    has no declared layout although a row-wise reduce feeds it: refused."""
    with pytest.raises(MetalNonRecoverableError, match="source layout is unresolved despite reduce provenance"):
        _run_loop(6, 1, op, R, C)


# ---------------------------------------------------------------------------
# Index ranges that do not start at zero.
# ---------------------------------------------------------------------------

_SHIFTS = [
    (16, 16, 16, 16),
    (8, 32, 8, 32),
    (32, 8, 32, 8),
    (4, 64, 4, 64),
    (16, 16, 32, 64),
    (64, 4, 64, 4),
    (1, 64, 1, 64),
    (32, 32, 32, 32),
]


def _shifted_params():
    out = []
    combo = 0
    for mode in range(8):
        for ax in (1, 0):
            for op in (0, 1):
                R, C, R0, C0 = _SHIFTS[combo % len(_SHIFTS)]
                combo += 1
                if (mode + ax + op) % 2:
                    continue
                out.append(
                    pytest.param(
                        mode,
                        ax,
                        op,
                        R,
                        C,
                        R0,
                        C0,
                        id=f"mode{mode}-ax{ax}-{'sum' if op == 0 else 'max'}-{R}x{C}-from-{R0}-{C0}",
                    )
                )
    return out


@requires
@pytest.mark.parametrize("mode,ax,op,R,C,R0,C0", _shifted_params())
def test_reduction_of_a_tile_from_a_shifted_arange_computes(mode, ax, op, R, C, R0, C0):
    """`tl.arange(R0, R0 + R)` rows and `tl.arange(C0, C0 + C)` columns: the reduction is
    over the far corner of the matrix, and the index operand is not the lane number."""
    rng = np.random.default_rng([R, C, R0, C0, ax, op])
    y, p = _f32(rng, R0 + R, C0 + C), _f32(rng, R0 + R + C0 + C + _GUARD)
    v, load = y.astype(np.float64)[R0:, C0:], p.astype(np.float64)
    r = v.sum(ax) if op == 0 else v.max(ax)
    first = R0 if ax == 1 else C0
    idx = first + np.arange(r.size)
    scalar = {
        1: r.sum(),
        2: r.max(),
        4: idx.sum() + r.max(),
        6: (r + load[idx]).sum(),
        7: (r * idx).min(),
    }
    vector = {0: r, 3: r + idx, 5: r + load[idx]}
    if mode in scalar:
        total, want = 1, _cells(1, scalar[mode])
    else:
        total = first + r.size
        want = _cells(total, vector[mode], at=first)
    got = _launch(_shifted, total, [y, p], R=R, C=C, R0=R0, C0=C0, AX=ax, OP=op, MODE=mode)
    _close(got, want)


# ---------------------------------------------------------------------------
# Integer element types (exact).
# ---------------------------------------------------------------------------

_INT_SHAPES = [(16, 16), (8, 16), (32, 32), (4, 8), (64, 16), (16, 4), (2, 512), (1, 64)]
_INT_MODES = {
    "store": 0,
    "sum": 1,
    "max": 2,
    "min": 3,
    "argmax": 4,
    "add_load": 5,
    "add_index": 6,
    "sum_of_scaled_plus_index": 7,
    "where": 8,
    "xor_sum": 9,
}


def _int_case(kind, dtype, ax, op, R, C):
    for seed in range(40):
        rng = np.random.default_rng([R, C, ax, op, 9, seed])
        y = rng.integers(-9, 10, size=(R, C)).astype(dtype)
        p = rng.integers(-9, 10, size=max(R, C) + _GUARD).astype(dtype)
        v = y.astype(np.int64)
        r = (v.sum(ax), v.max(ax), v.min(ax))[op]
        n = r.size
        idx = np.arange(n)
        if kind == "argmax" and np.count_nonzero(r == r.max()) != 1:
            continue  # one largest element: the result does not depend on a tie rule
        cells = {
            "store": r,
            "sum": r.sum(),
            "max": r.max(),
            "min": r.min(),
            "argmax": r.argmax(),
            "add_load": r + p[:n].astype(np.int64),
            "add_index": r + idx,
            "sum_of_scaled_plus_index": (r * 2 + idx).sum(),
            "where": np.where(r > 1, r, 0),
            "xor_sum": np.bitwise_xor.reduce(r),
        }[kind]
        return y, p, np.atleast_1d(np.asarray(cells, dtype=np.int64))
    raise AssertionError("no seed gives one clear largest element")


def _int_params():
    out = []
    combo = 0
    for dtype in ("int32", "int64", "int16", "int8"):
        for kind in _INT_MODES:
            if kind == "argmax" and dtype == "int64":
                continue  # argmax over a 64-bit integer value is refused for its own reason
            if dtype in ("int16", "int8") and kind not in ("store", "sum", "max", "argmax", "add_load"):
                continue
            for ax in (1, 0):
                op = combo % 3
                R, C = _INT_SHAPES[combo % len(_INT_SHAPES)]
                combo += 1
                out.append(
                    pytest.param(
                        kind, dtype, ax, op, R, C, id=f"{dtype}-{kind}-ax{ax}-{('sum', 'max', 'min')[op]}-{R}x{C}"
                    )
                )
    return out


@requires
@pytest.mark.parametrize("kind,dtype,ax,op,R,C", _int_params())
def test_integer_reduction_consumers_are_exact(kind, dtype, ax, op, R, C):
    """Integer tiles: the consumers of a sum / max / min along either axis, exactly."""
    y, p, cells = _int_case(kind, np.dtype(dtype), ax, op, R, C)
    wide = dtype == "int64"
    got = _launch(
        _consume_int,
        cells.size,
        [y, p],
        dtype=torch.int64 if wide else torch.int32,
        R=R,
        C=C,
        AX=ax,
        OP=op,
        MODE=_INT_MODES[kind],
    )
    want = np.full(cells.size + _GUARD, _INT_SENTINEL, dtype=np.int64)
    want[: cells.size] = cells
    np.testing.assert_array_equal(got.astype(np.int64), want)
