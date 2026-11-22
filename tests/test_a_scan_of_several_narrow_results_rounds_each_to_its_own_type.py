"""A scan over a tuple rounds each of its float16 or bfloat16 results to that result's own type.

Rounding every float16 and bfloat16 result where it is produced left one case refused: an
operation with several results that computes a narrow value by arithmetic. ``tl.associative_scan``
over a tuple is one, and upstream's own ``roll`` scan over bfloat16 was then refused (56 cases of
``test_scan2d``). Each result has a type of its own, and the first result's says nothing of the
others: every one is now rounded to its own.

The references are written here with numpy. A two-wide scan performs its combine once, so its
result does not depend on the order a longer scan would combine in; the operands are chosen so
that float32 holds each product and each sum exactly, and the one rounding is the narrow type's.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

TORCH = {"float16": torch.float16, "bfloat16": torch.bfloat16}
BITS = {"float16": 11, "bfloat16": 8}


def _narrow(values, dtype):
    """float32 values rounded once, to nearest even, to ``dtype``; returned as float32."""
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float32)
    bits = values.view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & ~np.uint64(0xFFFF)
    return rounded.astype(np.uint32).view(np.float32)


def _values(rng, shape, dtype):
    """Values of ``dtype`` between 1 and 2 with every bit of the significand in use."""
    bits = BITS[dtype]
    significand = rng.integers(1 << (bits - 1), 1 << bits, size=shape)
    return (significand / float(1 << (bits - 1))).astype(np.float32)


def test_the_reference_rounding_is_the_type_s_own():
    for dtype in TORCH:
        values = (_values(np.random.default_rng(3), (4096,), dtype) * np.float32(1.3)).astype(np.float32)
        want = torch.tensor(values).to(TORCH[dtype]).to(torch.float32).numpy()
        assert np.array_equal(_narrow(values, dtype), want)
        assert not np.array_equal(want, values)


@triton.jit
def _roll(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


@triton.jit
def _affine(a1, b1, a2, b2):
    return a1 * a2, b1 * a2 + b2


@triton.jit
def _count_and_product(n1, v1, n2, v2):
    return n1 + n2, v1 * v2


@triton.jit
def _product_and_count(v1, n1, v2, n2):
    return v1 * v2, n1 + n2


@triton.jit
def _scan_kernel(
    X, Y, ZA, ZB, ZN, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, REVERSE: tl.constexpr, FORM: tl.constexpr
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    at = rm[:, None] * N + rn[None, :]
    x = tl.load(X + at)
    y = tl.load(Y + at)
    if FORM == 0:
        _, z, _ = tl.associative_scan((1 + 0 * x, 0 * x, x), axis=AXIS, combine_fn=_roll, reverse=REVERSE)
        tl.store(ZB + at, z.to(tl.float32))
    elif FORM == 1:
        a, b = tl.associative_scan((x, y), axis=AXIS, combine_fn=_affine, reverse=REVERSE)
        tl.store(ZA + at, a.to(tl.float32))
        tl.store(ZB + at, b.to(tl.float32))
    elif FORM == 3:
        # An integer that no float holds: 2**24 + 1 in every cell, 2**25 + 2 where two are added.
        odd = tl.full((M, N), 16777217, tl.int32)
        v, n = tl.associative_scan((x, odd), axis=AXIS, combine_fn=_product_and_count, reverse=REVERSE)
        tl.store(ZN + at, n)
        tl.store(ZB + at, v.to(tl.float32))
    else:
        ones = tl.full((M, N), 1, tl.int32)
        n, v = tl.associative_scan((ones, x), axis=AXIS, combine_fn=_count_and_product, reverse=REVERSE)
        tl.store(ZN + at, n)
        tl.store(ZB + at, v.to(tl.float32))


def _launch(device, dtype, x, y, axis, reverse, form):
    shape = x.shape
    tx = torch.tensor(x, device=device).to(TORCH[dtype])
    ty = torch.tensor(y, device=device).to(TORCH[dtype])
    assert np.array_equal(tx.to(torch.float32).cpu().numpy(), x)
    za = torch.full(shape, -7.0, dtype=torch.float32, device=device)
    zb = torch.full(shape, -7.0, dtype=torch.float32, device=device)
    zn = torch.full(shape, -7, dtype=torch.int32, device=device)
    _scan_kernel[(1,)](tx, ty, za, zb, zn, M=shape[0], N=shape[1], AXIS=axis, REVERSE=reverse, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    return za.cpu().numpy(), zb.cpu().numpy(), zn.cpu().numpy()


def _along(array, axis, reverse):
    """The two elements of a two-wide scan in the order the scan meets them."""
    first, second = (array[:, 0], array[:, 1]) if axis == 1 else (array[0, :], array[1, :])
    return (second, first) if reverse else (first, second)


def _place(first, second, axis, reverse, shape):
    out = np.empty(shape, dtype=np.asarray(first).dtype)
    if reverse:
        first, second = second, first
    if axis == 1:
        out[:, 0], out[:, 1] = first, second
    else:
        out[0, :], out[1, :] = first, second
    return out


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis", [1, 0])
@pytest.mark.parametrize("shape", [(8, 32), (32, 16), (16, 64), (4, 4)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_upstream_s_roll_over_a_narrow_type_moves_every_element_one_place(dtype, shape, axis, reverse, device):
    x = _values(np.random.default_rng(17), shape, dtype)
    _, got, _ = _launch(device, dtype, x, x, axis, reverse, 0)
    want = np.zeros(shape, dtype=np.float32)
    if axis == 1 and not reverse:
        want[:, 1:] = x[:, :-1]
    elif axis == 1:
        want[:, :-1] = x[:, 1:]
    elif not reverse:
        want[1:, :] = x[:-1, :]
    else:
        want[:-1, :] = x[1:, :]
    assert np.array_equal(got, want)


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis,shape", [(1, (64, 2)), (1, (16, 2)), (0, (2, 64)), (0, (2, 16))])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_both_results_of_a_two_wide_scan_are_rounded_to_the_narrow_type(dtype, axis, shape, reverse, device):
    rng = np.random.default_rng(29)
    x, y = _values(rng, shape, dtype), _values(rng, shape, dtype)
    got_a, got_b, _ = _launch(device, dtype, x, y, axis, reverse, 1)
    a1, a2 = _along(x, axis, reverse)
    b1, b2 = _along(y, axis, reverse)
    product = _narrow(a1 * a2, dtype)
    carried = _narrow(_narrow(b1 * a2, dtype) + b2, dtype)
    # The products are not values of the type until they are rounded, so the test can tell.
    assert not np.array_equal(product, a1 * a2)
    assert not np.array_equal(carried, b1 * a2 + b2)
    assert np.array_equal(got_a, _place(a1, product, axis, reverse, shape))
    assert np.array_equal(got_b, _place(b1, carried, axis, reverse, shape))


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis,shape", [(1, (64, 2)), (0, (2, 64))])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_a_narrow_result_behind_an_integer_one_is_rounded_to_its_own_type(dtype, axis, shape, reverse, device):
    x = _values(np.random.default_rng(31), shape, dtype)
    _, got_v, got_n = _launch(device, dtype, x, x, axis, reverse, 2)
    v1, v2 = _along(x, axis, reverse)
    product = _narrow(v1 * v2, dtype)
    assert not np.array_equal(product, v1 * v2)
    assert np.array_equal(got_v, _place(v1, product, axis, reverse, shape))
    ones = np.ones(v1.shape, dtype=np.int32)
    assert np.array_equal(got_n, _place(ones, ones + ones, axis, reverse, shape))


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis,shape", [(1, (64, 2)), (0, (2, 64))])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_an_integer_result_behind_a_narrow_one_is_left_an_integer(dtype, axis, shape, reverse, device):
    x = _values(np.random.default_rng(37), shape, dtype)
    _, got_v, got_n = _launch(device, dtype, x, x, axis, reverse, 3)
    v1, v2 = _along(x, axis, reverse)
    assert np.array_equal(got_v, _place(v1, _narrow(v1 * v2, dtype), axis, reverse, shape))
    odd = np.full(v1.shape, 16777217, dtype=np.int32)
    assert int(np.float32(16777217)) != 16777217
    assert np.array_equal(got_n, _place(odd, odd + odd, axis, reverse, shape))
