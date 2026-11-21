"""The launch check divides the smallest 64-bit integer as the shader does.

The lane-order check reads a branch on the launch's integer scalars before the launch and
leaves what is under the way not taken out of its comparison. Its signed quotient took the
absolute value of both operands first, and the smallest 64-bit value has none: ``-2**63 // 2``
was ``+2**62`` in the check and ``-2**62`` in the shader. A kernel that reverses a row in
place under ``else`` of ``if S // T > 1`` was then launched with ``S = -2**63``: the check
read the branch as taken, the shader took the other one. The remainder followed the same
quotient, and the ceiling quotient negated its dividend the same way.

The quotient is now computed with no absolute value and no negation. What the source does not
define (a zero divisor, the smallest value over -1) is still not read.

The references here are Python's unbounded integers, not the checked code.
"""

import itertools

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401
from triton_msl.backend import _lane_order
from triton_msl.errors import MetalNonRecoverableError

MIN, MAX = -(2**63), 2**63 - 1
GRID = 2
WIDTHS = [64, 2048]
MEET = "two lanes of one program reach the same element and one of them stores it"
MAY = (
    "two lanes of one program reach the same element, one of them storing it, unless a branch, or a mask "
    "computed from values the check cannot know before the launch, keeps them apart"
)


def _wrap(value, bits):
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def _truncated(top, bottom):
    quotient = abs(top) // abs(bottom)
    return -quotient if (top < 0) != (bottom < 0) else quotient


def _exact(name, top, bottom, bits):
    if name == "sdiv":
        return _truncated(top, bottom)
    if name == "srem":
        return top - _truncated(top, bottom) * bottom
    if name == "fdiv":
        return top // bottom
    if name == "cdiv":
        return -((-top) // bottom)
    unsigned_top, unsigned_bottom = top % (1 << bits), bottom % (1 << bits)
    return _wrap(unsigned_top // unsigned_bottom if name == "udiv" else unsigned_top % unsigned_bottom, bits)


def _edges(bits):
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return [low, low + 1, low + 2, -17, -4, -3, -2, -1, 0, 1, 2, 3, 4, 17, high - 1, high]


def _undefined(name, top, bottom, bits):
    if _wrap(bottom, bits) == 0:
        return True
    return name in ("sdiv", "srem", "fdiv", "cdiv") and bottom == -1 and top == -(1 << (bits - 1))


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("name", ["sdiv", "srem", "fdiv", "cdiv", "udiv", "urem"])
def test_every_quotient_and_remainder_at_the_edges_of_the_type_is_the_integer_one(name, bits):
    pairs = [pair for pair in itertools.product(_edges(bits), repeat=2) if not _undefined(name, *pair, bits)]
    assert len(pairs) >= 239
    tops = np.array([top for top, _ in pairs], dtype=np.int64)
    bottoms = np.array([bottom for _, bottom in pairs], dtype=np.int64)
    together = _lane_order._binary(name, bits, tops, bottoms)
    for index, (top, bottom) in enumerate(pairs):
        want = _exact(name, top, bottom, bits)
        assert int(together[index]) == want, (name, bits, top, bottom)
        alone = _lane_order._binary(name, bits, np.array([top], dtype=np.int64), np.array([bottom], dtype=np.int64))
        assert int(alone[0]) == want, (name, bits, top, bottom)


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("name", ["sdiv", "srem", "fdiv", "cdiv"])
def test_the_smallest_value_over_minus_one_is_still_not_read(name, bits):
    low = -(1 << (bits - 1))
    with pytest.raises(_lane_order.Unproved) as caught:
        _lane_order._binary(name, bits, np.array([5, low], dtype=np.int64), np.array([2, -1], dtype=np.int64))
    assert caught.value.detail == "quotient outside the type"
    # One above it, and the smallest value over another divisor, are defined.
    near = _lane_order._binary(name, bits, np.array([low + 1, low], dtype=np.int64), np.array([-1, 1], dtype=np.int64))
    assert [int(v) for v in near] == [_exact(name, low + 1, -1, bits), _exact(name, low, 1, bits)]


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("name", ["sdiv", "srem", "fdiv", "cdiv", "udiv", "urem"])
def test_a_zero_divisor_is_still_not_read(name, bits):
    low = -(1 << (bits - 1))
    with pytest.raises(_lane_order.Unproved) as caught:
        _lane_order._binary(name, bits, np.array([5, low], dtype=np.int64), np.array([2, 0], dtype=np.int64))
    assert caught.value.detail == "division by zero"


CONDITIONS = ["S // T > 1", "S // T < -1", "S % T == -2", "S % T == 5", "S // T == 1", "S % T == 0"]


@triton.jit
def _reversal(Z, F, S, T, N: tl.constexpr, COND: tl.constexpr, UNDER_ELSE: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if COND == 0:
        c = S // T > 1
    elif COND == 1:
        c = S // T < -1
    elif COND == 2:
        c = S % T == -2
    elif COND == 3:
        c = S % T == 5
    elif COND == 4:
        c = S // T == 1
    else:
        c = S % T == 0
    if UNDER_ELSE:
        if c:
            tl.store(z + n, tl.load(z + n) + 1)
        else:
            tl.store(F + pid, 1)
            tl.store(z + (N - 1 - n), tl.load(z + n))
    else:
        if c:
            tl.store(F + pid, 1)
            tl.store(z + (N - 1 - n), tl.load(z + n))
        else:
            tl.store(z + n, tl.load(z + n) + 1)


def _taken(cond, s, t):
    quotient = _truncated(s, t)
    remainder = s - quotient * t
    return [quotient > 1, quotient < -1, remainder == -2, remainder == 5, quotient == 1, remainder == 0][cond]


# The smallest value as the dividend and as the divisor, and its neighbours.
SCALARS = [
    (MIN, 2),
    (MIN, -2),
    (MIN, 3),
    (MIN, -3),
    (MIN, MAX),
    (MIN, MIN),
    (MIN, MIN + 1),
    (MIN + 1, MIN),
    (5, MIN),
    (-5, MIN),
    (MAX, MIN),
    (MIN + 1, 2),
    (MIN + 2, -3),
    (MAX, 2),
    (MAX, -3),
]


def _start(width):
    return ((np.arange(GRID * width) * 5) % 11 + 2).astype(np.int32)


def test_the_scalars_take_both_ways_of_every_condition():
    for cond in range(len(CONDITIONS)):
        assert {_taken(cond, s, t) for s, t in SCALARS} == {False, True}, CONDITIONS[cond]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("under_else", [False, True], ids=["under the branch", "under its else"])
@pytest.mark.parametrize("s,t", SCALARS, ids=[f"{s}:{t}" for s, t in SCALARS])
@pytest.mark.parametrize("cond", range(len(CONDITIONS)), ids=CONDITIONS)
@pytest.mark.parametrize("width", WIDTHS)
def test_a_reversal_under_a_quotient_of_the_smallest_integer_is_counted_where_the_shader_performs_it(
    width, cond, s, t, under_else, device
):
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.zeros(GRID, dtype=torch.int32, device=device)

    def launch():
        _reversal[(GRID,)](z, flag, s, t, N=width, COND=cond, UNDER_ELSE=under_else)

    if _taken(cond, s, t) != under_else:
        # The shader would reverse the row in place: refused, and nothing is written.
        with pytest.raises(MetalNonRecoverableError) as caught:
            launch()
        assert f"for the memory of argument(s) Z: {MEET}." in str(caught.value)
        assert np.array_equal(z.cpu().numpy(), start)
    else:
        # The shader takes the way that adds one: the result shows which way it went.
        launch()
        if device == "mps":
            torch.mps.synchronize()
        assert np.array_equal(z.cpu().numpy(), start + 1)
    assert not flag.cpu().numpy().any()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("under_else", [False, True], ids=["under the branch", "under its else"])
@pytest.mark.parametrize(
    "s,t", [(MIN, -1), (MIN, 0), (5, 0)], ids=["the smallest over -1", "the smallest over 0", "over 0"]
)
@pytest.mark.parametrize("cond", range(len(CONDITIONS)), ids=CONDITIONS)
def test_a_quotient_of_64_bits_the_source_does_not_define_is_not_read(cond, s, t, under_else, device):
    start = _start(64)
    z = torch.tensor(start, device=device)
    flag = torch.zeros(GRID, dtype=torch.int32, device=device)
    with pytest.raises(MetalNonRecoverableError) as caught:
        _reversal[(GRID,)](z, flag, s, t, N=64, COND=cond, UNDER_ELSE=under_else)
    assert f"for the memory of argument(s) Z: {MAY}." in str(caught.value)
    assert np.array_equal(z.cpu().numpy(), start)
    assert not flag.cpu().numpy().any()
