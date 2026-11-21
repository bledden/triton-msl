"""A branch on the launch's integer scalars is read before the launch.

The lane-order check counted every access under an ``if`` as performed, whichever way the
branch went, so a kernel that reverses a row in place only ``if S > 0`` was refused with
``S = 0`` too, where the source stores nothing that two lanes share. A condition that follows
nothing but integer scalar arguments and constants is one number for every program, lane and
trip. The check now computes it as the shader does, wrapping at the type's width: what is
under the way not taken is left out of the comparison, and what is under the way taken is
counted without the "unless a branch ..." hedge.

A scalar that a counted loop advances by one fixed amount each trip is such a number too
(its start plus the amount times the trips). Every other condition (a loaded value, the
program id, a loop's trip, another carried value, a float, a division by zero, the smallest
value over -1) stays unread, and what is under it is counted in as before.

Every launch here keeps to its own cells of ``Z`` and one cell of ``F``.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
GRID = 2
# One thread per lane, and a lane loop.
WIDTHS = [64, 2048]
MEET = "two lanes of one program reach the same element and one of them stores it"
MAY = (
    "two lanes of one program reach the same element, one of them storing it, unless a branch, or a mask "
    "computed from values the check cannot know before the launch, keeps them apart"
)
LARGEST, SMALLEST = 2**31 - 1, -(2**31)
CONDITIONS = [
    "S > 0",
    "S + T > 3",
    "S * T > 10",
    "S - T > 2147483000",
    "(S > 0) & (T > 0)",
    "S % 4 == 2",
    "(S >> 1) > 0",
    "minimum(S, T) > 2",
    "S // T > 1",
    "S == T",
]
QUOTIENT = CONDITIONS.index("S // T > 1")


@triton.jit
def _reversal(Z, F, S, T, N: tl.constexpr, COND: tl.constexpr, UNDER_ELSE: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if COND == 0:
        c = S > 0
    elif COND == 1:
        c = S + T > 3
    elif COND == 2:
        c = S * T > 10
    elif COND == 3:
        c = S - T > 2147483000
    elif COND == 4:
        c = (S > 0) & (T > 0)
    elif COND == 5:
        c = S % 4 == 2
    elif COND == 6:
        c = (S >> 1) > 0
    elif COND == 7:
        c = tl.minimum(S, T) > 2
    elif COND == 8:
        c = S // T > 1
    else:
        c = S == T
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


@triton.jit
def _nested(Z, F, S, T, N: tl.constexpr, OUTER_LOADED: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if OUTER_LOADED:
        outer = tl.load(F + pid) > S
    else:
        outer = S > 0
    if outer:
        if T > 0:
            tl.store(z + (N - 1 - n), tl.load(z + n))
        else:
            tl.store(z + n, tl.load(z + n) + 1)


@triton.jit
def _one_after_another(Z, S, T, N: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if S > 0:
        tl.store(z + n, tl.load(z + n) + 1)
    if T > 0:
        tl.store(z + (N - 1 - n), tl.load(z + n))


@triton.jit
def _unread(Z, F, S, T, K, N: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if FORM == 0:
        if tl.load(F + pid) > S:
            tl.store(z + (N - 1 - n), tl.load(z + n))
    elif FORM == 1:
        if pid > S:
            tl.store(z + (N - 1 - n), tl.load(z + n))
    elif FORM == 2:
        for i in range(K):
            if i > S:
                tl.store(z + (N - 1 - n), tl.load(z + n))
    elif FORM == 3:
        carried = S
        for i in range(K):
            carried = carried + i
        if carried > 100:
            tl.store(z + (N - 1 - n), tl.load(z + n))
    elif FORM == 4:
        carried = S
        i = 0
        while i < K:
            carried = carried * 2 + T
            i += 1
        if carried > 100:
            tl.store(z + (N - 1 - n), tl.load(z + n))
    else:
        if S // T > 1:
            tl.store(z + (N - 1 - n), tl.load(z + n))


@triton.jit
def _carried(Z, S, T, K, N: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    carried = S
    for _ in range(K):
        carried = carried + T
    if carried > 100:
        tl.store(z + (N - 1 - n), tl.load(z + n))
    else:
        tl.store(z + n, tl.load(z + n) + 1)


@triton.jit
def _under_a_float(Z, S, N: tl.constexpr, MEETING: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if S > 0.5:
        if MEETING:
            tl.store(z + (N - 1 - n), tl.load(z + n))
        else:
            tl.store(z + n, tl.load(z + n) + 1)


@triton.jit
def _under_a_flag(Z, S, N: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    z = Z + pid * N
    if S:
        tl.store(z + (N - 1 - n), tl.load(z + n))
    else:
        tl.store(z + n, tl.load(z + n) + 1)


def _toward_zero(top, bottom):
    quotient = abs(top) // abs(bottom)
    return -quotient if (top < 0) != (bottom < 0) else quotient


def _taken(cond, s, t):
    """The condition as the source computes it: at 32 bits, or at 64 where a scalar needs them."""
    bits = 32 if all(SMALLEST <= v <= LARGEST for v in (s, t)) else 64

    def wrap(value):
        return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))

    return [
        lambda: s > 0,
        lambda: wrap(s + t) > 3,
        lambda: wrap(s * t) > 10,
        lambda: wrap(s - t) > 2147483000,
        lambda: s > 0 and t > 0,
        lambda: s - _toward_zero(s, 4) * 4 == 2,
        lambda: (s >> 1) > 0,
        lambda: min(s, t) > 2,
        lambda: _toward_zero(s, t) > 1,
        lambda: s == t,
    ][cond]()


# Scalars of 32 bits, with sums, differences and products that leave the type.
NARROW = [
    (5, 7),
    (0, 4),
    (-5, 2),
    (2, 2),
    (6, -3),
    (1, 9),
    (-6, 4),
    (LARGEST, 2),
    (2, LARGEST),
    (LARGEST, SMALLEST),
    (SMALLEST, LARGEST),
    (46341, 46341),
    (-2, SMALLEST),
]
# Scalars of 64 bits.
WIDE = [(2**40 + 3, 2**32 + 5), (-(2**33), 2**40), (2**62, 2**62), (2**62 + 5, -(2**62))]


def _start(width):
    return ((np.arange(GRID * width) * 5) % 11 + 2).astype(np.int32)


def _refused(reason, launch, z, start, flag=None):
    with pytest.raises(MetalNonRecoverableError) as caught:
        launch()
    assert (
        f"lane order: this launch is not proved to keep the source's order for the memory of argument(s) Z: {reason}."
        in str(caught.value)
    )
    assert np.array_equal(z.cpu().numpy(), start)
    assert flag is None or not flag.cpu().numpy().any()


def _computed(launch, z, want, device, flag=None):
    launch()
    if device == "mps":
        torch.mps.synchronize()
    assert np.array_equal(z.cpu().numpy(), want)
    assert flag is None or not flag.cpu().numpy().any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("under_else", [False, True], ids=["under the branch", "under its else"])
@pytest.mark.parametrize("s,t", NARROW + WIDE, ids=[f"{s}:{t}" for s, t in NARROW + WIDE])
@pytest.mark.parametrize("cond", range(len(CONDITIONS)), ids=CONDITIONS)
@pytest.mark.parametrize("width", WIDTHS)
def test_an_in_place_reversal_under_a_scalar_branch_is_counted_only_where_the_launch_performs_it(
    width, cond, s, t, under_else, device
):
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.zeros(GRID, dtype=torch.int32, device=device)

    def launch():
        _reversal[(GRID,)](z, flag, s, t, N=width, COND=cond, UNDER_ELSE=under_else)

    if _taken(cond, s, t) != under_else:
        _refused(MEET, launch, z, start, flag)
    else:
        _computed(launch, z, start + 1, device, flag)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("under_else", [False, True], ids=["under the branch", "under its else"])
@pytest.mark.parametrize("s,t", [(5, 0), (SMALLEST, -1)], ids=["over zero", "the smallest over -1"])
@pytest.mark.parametrize("width", WIDTHS)
def test_a_quotient_the_source_does_not_define_is_not_read(width, s, t, under_else, device):
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.zeros(GRID, dtype=torch.int32, device=device)
    _refused(
        MAY,
        lambda: _reversal[(GRID,)](z, flag, s, t, N=width, COND=QUOTIENT, UNDER_ELSE=under_else),
        z,
        start,
        flag,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "s,t,outcome",
    [(5, 3, "refused"), (5, -2, "bumped"), (-5, 3, "untouched"), (-5, -2, "untouched")],
    ids=["both taken", "inner not taken", "outer not taken", "neither taken"],
)
def test_a_branch_inside_a_branch_is_read_with_the_one_around_it(s, t, outcome, width, device):
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.zeros(GRID, dtype=torch.int32, device=device)

    def launch():
        _nested[(GRID,)](z, flag, s, t, N=width, OUTER_LOADED=False)

    if outcome == "refused":
        _refused(MEET, launch, z, start)
    else:
        _computed(launch, z, start + (outcome == "bumped"), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "s,t,outcome",
    [(5, 3, "refused"), (5, -2, "bumped"), (-5, 3, "refused"), (-5, -2, "untouched")],
    ids=["both taken", "first taken", "second taken", "neither taken"],
)
def test_a_branch_after_a_branch_is_read_by_itself(s, t, outcome, width, device):
    start = _start(width)
    z = torch.tensor(start, device=device)

    def launch():
        _one_after_another[(GRID,)](z, s, t, N=width)

    if outcome == "refused":
        _refused(MEET, launch, z, start)
    else:
        _computed(launch, z, start + (outcome == "bumped"), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("held", [0, 50], ids=["outer not taken", "outer taken"])
def test_an_inner_branch_that_is_not_taken_leaves_its_access_out_under_an_unread_one(held, width, device):
    # The outer condition is loaded: the check cannot know it. The inner one is a scalar.
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.full((GRID,), held, dtype=torch.int32, device=device)
    _nested[(GRID,)](z, flag, 10, -2, N=width, OUTER_LOADED=True)
    if device == "mps":
        torch.mps.synchronize()
    assert np.array_equal(z.cpu().numpy(), start + (held > 10))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("held", [0, 50], ids=["outer not taken", "outer taken"])
def test_an_inner_branch_that_is_taken_under_an_unread_one_is_hedged(held, width, device):
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.full((GRID,), held, dtype=torch.int32, device=device)
    _refused(MAY, lambda: _nested[(GRID,)](z, flag, 10, 3, N=width, OUTER_LOADED=True), z, start)


UNREAD = [
    "a loaded value",
    "the program id",
    "a loop's trip",
    "a sum of a loop's trips",
    "a value a while loop carries",
    "a quotient over zero",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("s", [1000, -1000], ids=["would not be taken", "would be taken"])
@pytest.mark.parametrize("form", range(len(UNREAD)), ids=UNREAD)
@pytest.mark.parametrize("width", WIDTHS)
def test_a_branch_the_check_cannot_read_is_counted_whichever_way_it_goes(width, form, s, device):
    start = _start(width)
    z = torch.tensor(start, device=device)
    flag = torch.zeros(GRID, dtype=torch.int32, device=device)
    _refused(MAY, lambda: _unread[(GRID,)](z, flag, s, 0, 2, N=width, FORM=form), z, start)


# (S, T, trips, whether the source takes the branch): the sum wraps at 32 bits, and a loop
# with a count below one makes no trip.
CARRIED = [
    (0, 1000, 2, True),
    (1000, -600, 2, False),
    (90, 7, 0, False),
    (200, 7, 0, True),
    (90, 3, 4, True),
    (98, 0, 4, False),
    (LARGEST, 1, 2, False),
    (SMALLEST, -1, 2, True),
    (5, SMALLEST, 2, False),
    (5, SMALLEST, 3, False),
    (7, 65536, 65536, False),
    (7, 65537, 65536, True),
    (200, 3, -5, True),
    (0, 300, -5, False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("s,t,trips,taken", CARRIED, ids=[f"{s}:{t}:{k}" for s, t, k, _ in CARRIED])
@pytest.mark.parametrize("width", WIDTHS)
def test_a_scalar_that_a_counted_loop_advances_by_a_fixed_amount_is_read(width, s, t, trips, taken, device):
    start = _start(width)
    z = torch.tensor(start, device=device)

    def launch():
        _carried[(GRID,)](z, s, t, trips, N=width)

    if taken:
        _refused(MEET, launch, z, start)
    else:
        _computed(launch, z, start + 1, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("s", [0.25, 0.75], ids=["not taken", "taken"])
@pytest.mark.parametrize("width", WIDTHS)
def test_a_branch_on_a_float_scalar_is_not_read(width, s, device):
    start = _start(width)
    z = torch.tensor(start, device=device)
    _refused(MAY, lambda: _under_a_float[(GRID,)](z, s, N=width, MEETING=True), z, start)
    # Lanes that keep to their own cells under it compute, as they did.
    _computed(lambda: _under_a_float[(GRID,)](z, s, N=width, MEETING=False), z, start + (s > 0.5), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", WIDTHS)
def test_a_branch_on_a_boolean_scalar_is_read(width, device):
    start = _start(width)
    z = torch.tensor(start, device=device)
    _refused(MEET, lambda: _under_a_flag[(GRID,)](z, True, N=width), z, start)
    _computed(lambda: _under_a_flag[(GRID,)](z, False, N=width), z, start + 1, device)
