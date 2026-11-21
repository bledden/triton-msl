"""A signed sum, difference or product that leaves its type's range wraps, as the source says.

`x + 1 > x` is false where `x` is the largest int32: the sum wraps to the smallest. The
shader compiler reads a signed sum, difference or product as one that never overflows,
so it answered true there, turned `(x * 3) // 3` back into `x`, and took a branch on
`S + 1 > 3` for the largest `S`. Nothing was refused and nothing was reported. Signed
32- and 64-bit sums, differences and products are now computed on the unsigned type,
which is the same bits and wraps.

Narrower and unsigned types were right before and are controls here.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
GRID = 2
# One thread per lane, and a lane loop.
WIDTHS = [64, 2048]
SIGNED = [np.int32, np.int64]
CONTROLS = [np.int8, np.int16, np.uint8, np.uint32, np.uint64]
FORMS = [
    "sum above its operand",
    "product divided back",
    "difference below its operand",
    "sum carried round a loop",
    "sum of a reduction",
    "sum choosing a value",
    "product of the lane index",
    "negation above zero",
    "sum masking a load",
    "square at least zero",
    "sum stored",
    "product divided by its factor",
    "sum doubled and compared",
]
SUM_OF_A_REDUCTION = FORMS.index("sum of a reduction")
# The value every lane adds, subtracts or multiplies by; every fifth lane has 2.
OTHERS = ["one", "the largest", "near the largest"]


@triton.jit
def _wrapping(X, T, O, K, N: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    x = tl.load(X + pid * N + n)
    t = tl.load(T + pid * N + n)
    o = O + pid * N + n
    if FORM == 0:
        tl.store(o, (x + t > x).to(tl.int32))
    elif FORM == 1:
        tl.store(o, ((x * 3) // 3).to(tl.int32))
    elif FORM == 2:
        tl.store(o, (x - t < x).to(tl.int32))
    elif FORM == 3:
        acc = x
        for _ in range(K):
            acc = acc + t
        tl.store(o, (acc > x).to(tl.int32))
    elif FORM == 4:
        s = tl.sum(x, axis=0)
        tl.store(o, (s + t > s).to(tl.int32))
    elif FORM == 5:
        tl.store(o, tl.where(x + 1 > x, 5, 9))
    elif FORM == 6:
        m = n.to(X.dtype.element_ty) * t
        tl.store(o, (m + t > m).to(tl.int32))
    elif FORM == 7:
        tl.store(o, ((0 - x) > 0).to(tl.int32))
    elif FORM == 8:
        tl.store(o, tl.load(T + pid * N + n, mask=x + t > x, other=7).to(tl.int32))
    elif FORM == 9:
        tl.store(o, (x * x >= 0).to(tl.int32))
    elif FORM == 10:
        tl.store(o, (x + t).to(tl.int32))
    elif FORM == 11:
        tl.store(o, ((x * t) // t).to(tl.int32))
    else:
        tl.store(o, ((x + t) * 2 - t > x).to(tl.int32))


@triton.jit
def _branching(O, S, T, N: tl.constexpr, FORM: tl.constexpr):
    n = tl.arange(0, N)
    if FORM == 0:
        if S + 2 > 3:
            tl.store(O + n, n + 1)
    elif FORM == 1:
        if S + T > 3:
            tl.store(O + n, n + 1)
    elif FORM == 2:
        if S - T > 2147483000:
            tl.store(O + n, n + 1)
    elif FORM == 3:
        if (S * 2) // 2 == S:
            tl.store(O + n, n + 1)
    else:
        if S * T > S:
            tl.store(O + n, n + 1)


@triton.jit
def _floats(X, T, O, N: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    x = tl.load(X + pid * N + n)
    t = tl.load(T + pid * N + n)
    if FORM == 0:
        tl.store(O + pid * N + n, x + t)
    elif FORM == 1:
        tl.store(O + pid * N + n, x - t)
    else:
        tl.store(O + pid * N + n, x * t)


def _wrap(values, kind):
    """Python integers brought into ``kind`` as its arithmetic brings them."""
    info = np.iinfo(kind)
    exact = np.asarray(values, dtype=object).ravel()
    return np.array([(int(v) - info.min) % (1 << info.bits) + info.min for v in exact], dtype=object)


def _toward_zero(top, bottom):
    quotient = np.array([abs(int(a)) // abs(int(b)) for a, b in zip(top, bottom)], dtype=object)
    return np.where(np.array([(a < 0) != (b < 0) for a, b in zip(top, bottom)]), -quotient, quotient)


def _inputs(kind, width, other):
    info = np.iinfo(kind)
    lanes = GRID * width
    pattern = [0, 5, 2, 3, info.max, info.max - 1, info.max - 50, info.min, info.min + 1, info.max // 2 + 1]
    pattern += [info.max // 2 + 6, info.max // 3 + 2, 46341 % (info.max + 1), 7, info.max - 3, 11, 13]
    x = np.array((pattern * (lanes // len(pattern) + 1))[:lanes], dtype=kind)
    t = np.full(lanes, {"one": 1, "the largest": info.max, "near the largest": info.max - 2}[other], dtype=kind)
    t[1::5] = 2
    return x, t


def _expected(form, x, t, trips, kind, width):
    out = np.empty(x.size, dtype=np.int32)
    for program in range(GRID):
        cells = slice(program * width, (program + 1) * width)
        xs, ts = x[cells].astype(object), t[cells].astype(object)
        if form == 0:
            r = _wrap(xs + ts, kind) > xs
        elif form == 1:
            r = _toward_zero(_wrap(xs * 3, kind), np.full(width, 3, dtype=object))
        elif form == 2:
            r = _wrap(xs - ts, kind) < xs
        elif form == 3:
            acc = xs
            for _ in range(trips):
                acc = _wrap(acc + ts, kind)
            r = acc > xs
        elif form == 4:
            total = _wrap([sum(int(v) for v in xs)], kind)[0]
            r = _wrap(total + ts, kind) > total
        elif form == 5:
            r = np.where(_wrap(xs + 1, kind) > xs, 5, 9)
        elif form == 6:
            m = _wrap(np.arange(width).astype(object) * ts, kind)
            r = _wrap(m + ts, kind) > m
        elif form == 7:
            r = _wrap(0 - xs, kind) > 0
        elif form == 8:
            r = np.where(_wrap(xs + ts, kind) > xs, ts, 7)
        elif form == 9:
            r = _wrap(xs * xs, kind) >= 0
        elif form == 10:
            r = _wrap(xs + ts, kind)
        elif form == 11:
            r = _toward_zero(_wrap(xs * ts, kind), ts)
        else:
            r = _wrap(_wrap(_wrap(xs + ts, kind) * 2, kind) - ts, kind) > xs
        out[cells] = _wrap([int(v) for v in r], np.int32).astype(np.int64)
    return out


def _computes(kind, width, form, other, device, trips=3):
    x0, t0 = _inputs(kind, width, other)
    x, t = torch.tensor(x0, device=device), torch.tensor(t0, device=device)
    out = torch.full((GRID * width,), -9, dtype=torch.int32, device=device)
    _wrapping[(GRID,)](x, t, out, trips, N=width, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    assert np.array_equal(out.cpu().numpy(), _expected(form, x0, t0, trips, kind, width))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("other", OTHERS)
@pytest.mark.parametrize("form", range(len(FORMS)), ids=FORMS)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("kind", SIGNED, ids=["int32", "int64"])
def test_signed_arithmetic_that_leaves_its_range_wraps(kind, width, form, other, device):
    _computes(kind, width, form, other, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("other", OTHERS)
@pytest.mark.parametrize("form", range(len(FORMS)), ids=FORMS)
@pytest.mark.parametrize("kind", CONTROLS, ids=[np.dtype(kind).name for kind in CONTROLS])
def test_narrow_and_unsigned_arithmetic_wraps_as_it_did(kind, form, other, device):
    if form == SUM_OF_A_REDUCTION and np.iinfo(kind).bits < 32:
        pytest.skip("a sum of a narrow type is a sum of int32")
    _computes(kind, 64, form, other, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trips", [0, 2, 5, 40])
@pytest.mark.parametrize("kind", SIGNED, ids=["int32", "int64"])
def test_a_sum_carried_round_a_loop_wraps_on_every_trip(kind, trips, device):
    _computes(kind, 64, FORMS.index("sum carried round a loop"), "near the largest", device, trips=trips)


LARGEST, SMALLEST = 2**31 - 1, -(2**31)
# (form, S, T, whether the source takes the branch)
BRANCHES = [
    (0, 5, 0, True),
    (0, LARGEST, 0, False),
    (0, LARGEST - 1, 0, False),
    (0, LARGEST - 2, 0, True),
    (0, SMALLEST, 0, False),
    (1, 5, LARGEST, False),
    (1, LARGEST, 5, False),
    (1, LARGEST, SMALLEST, False),
    (1, 2, 2, True),
    (2, 5, SMALLEST, False),
    (2, -2, SMALLEST, True),
    (2, LARGEST, -5, False),
    (3, 5, 0, True),
    (3, LARGEST, 0, False),
    (3, 2**30, 0, False),
    (3, 2**30 - 1, 0, True),
    (4, 46341, 46341, False),
    (4, 46340, 46340, True),
    (4, LARGEST, 2, False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form,s,t,taken", BRANCHES, ids=[f"form{f}:{s}:{t}" for f, s, t, _ in BRANCHES])
def test_a_branch_on_scalar_arithmetic_goes_the_way_the_wrapped_value_says(form, s, t, taken, device):
    width = 64
    out = torch.full((width,), -9, dtype=torch.int32, device=device)
    _branching[(1,)](out, s, t, N=width, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    want = np.arange(1, width + 1, dtype=np.int32) if taken else np.full(width, -9, dtype=np.int32)
    assert np.array_equal(out.cpu().numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", range(3), ids=["sum", "difference", "product"])
@pytest.mark.parametrize("width", WIDTHS)
def test_float_arithmetic_is_computed_on_floats(width, form, device):
    lanes = GRID * width
    x0 = ((np.arange(lanes) % 23) * np.float32(0.375) - np.float32(3.25)).astype(np.float32)
    x0[3::7] = np.float32(5.0e9)
    t0 = ((np.arange(lanes) % 5) * np.float32(-1.5) + np.float32(0.25)).astype(np.float32)
    want = [x0 + t0, x0 - t0, x0 * t0][form]
    x, t = torch.tensor(x0, device=device), torch.tensor(t0, device=device)
    out = torch.full((lanes,), -9.0, dtype=torch.float32, device=device)
    _floats[(GRID,)](x, t, out, N=width, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    assert np.array_equal(out.cpu().numpy(), want)
