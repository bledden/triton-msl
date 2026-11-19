"""One atomic in a ``while`` loop is ordered by its trips, as in a counted loop.

The lanes of one atomic operation are not ordered by the source, so a kernel whose only
access to a memory is one atomic is safe in any thread order. That stops being true inside
a loop: a cell reached by one lane on the first trip and another lane on the second is
updated in that order by the source, and the shader's threads each run their trips without
waiting for the others. The launch check knew this for a counted loop and answered "atomic
only" for a ``while`` loop, where an exchange, a compare-and-swap and an add whose returned
value is used then returned values the source does not compute (in a kernel of more than
1024 lanes, where two lanes reach one cell on different trips). Such a launch is now
refused; an update that commutes with itself still runs, and so does any atomic whose
address is the same on every trip and keeps each lane to cells of its own.
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
SIZES = [64, 1024, 2048, 4096]
UNBOUNDED = (
    "an access sits in a loop whose trips are not known before the launch, and its lanes are not shown "
    "to keep to elements of their own"
)


@triton.jit
def _exchange_while(Z, TRIPS, N: tl.constexpr):
    # Trip 0 the upper half of the lanes, trip 1 the lower half, onto the same cells.
    n = tl.arange(0, N)
    j = 0
    while j < TRIPS:
        tl.atomic_xchg(Z + n % (N // 2), n * 2 + j, mask=(n // (N // 2)) == (1 - j))
        j += 1


@triton.jit
def _compare_and_swap_while(Z, TRIPS, N: tl.constexpr):
    # One lane per cell on every trip; the second trip swaps what the first one left.
    n = tl.arange(0, N)
    j = 0
    while j < TRIPS:
        cell = (n + j * (N // 2)) % N
        tl.atomic_cas(Z + cell, cell * 3 * (1 - j) + (1000 + cell) * j, 1000 + n + 4000 * j)
        j += 1


@triton.jit
def _add_carried_while(Z, TRIPS, N: tl.constexpr):
    # The value an add returns goes into the next trip's update.
    n = tl.arange(0, N)
    j = 0
    carried = n * 0
    while j < TRIPS:
        carried = tl.atomic_add(Z + (n + j * (N // 2)) % N, carried + n + 1)
        j += 1


@triton.jit
def _add_while(Z, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    j = 0
    while j < TRIPS:
        tl.atomic_add(Z + (n + j * (N // 2)) % N, n + 1)
        j += 1


@triton.jit
def _max_while(Z, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    j = 0
    while j < TRIPS:
        tl.atomic_max(Z + n % (N // 2), n * 2 + j)
        j += 1


@triton.jit
def _exchange_once(Z, N: tl.constexpr):
    n = tl.arange(0, N)
    tl.atomic_xchg(Z + n, n * 2 + 1)


@triton.jit
def _exchange_own_cell_while(Z, R, TRIPS, N: tl.constexpr):
    # Each lane exchanges its own cell on every trip; what a trip returns goes into the next.
    n = tl.arange(0, N)
    j = 0
    old = n * 0
    while j < TRIPS:
        old = tl.atomic_xchg(Z + n, old * 2 + n + j)
        j += 1
    tl.store(R + n, old)


@triton.jit
def _compare_and_swap_own_cell_while(Z, R, TRIPS, N: tl.constexpr):
    # The second trip swaps only where the first one did.
    n = tl.arange(0, N)
    j = 0
    old = n * 0
    while j < TRIPS:
        old = tl.atomic_cas(Z + n, n * 3 * (1 - j) + (1000 + n) * j, 1000 + n + 4000 * j)
        j += 1
    tl.store(R + n, old)


@triton.jit
def _add_carried_own_cell_while(Z, R, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    j = 0
    carried = n * 0
    while j < TRIPS:
        carried = tl.atomic_add(Z + n, carried + n + 1)
        j += 1
    tl.store(R + n, carried)


@triton.jit
def _exchange_half_width_while(Z, R, W, TRIPS, N: tl.constexpr):
    # Half as many cells as the kernel has lanes: lanes 0 to N / 2 - 1 each exchange their
    # own cell, and what a trip returns goes into the next.
    n = tl.arange(0, N)
    h = tl.arange(0, N // 2)
    j = 0
    old = h * 0
    while j < TRIPS:
        old = tl.atomic_xchg(Z + h, old * 2 + h + j)
        j += 1
    tl.store(R + h, old)
    tl.store(W + n, n + 1)


def _exchange_own_cell(cells, trips):
    lane, old = np.arange(len(cells)), np.zeros(len(cells), dtype=np.int64)
    for trip in range(trips):
        cells, old = old * 2 + lane + trip, cells
    return cells, old


def _compare_and_swap_own_cell(cells, trips):
    lane, old = np.arange(len(cells)), np.zeros(len(cells), dtype=np.int64)
    for trip in range(trips):
        expected = lane * 3 * (1 - trip) + (1000 + lane) * trip
        cells, old = np.where(cells == expected, 1000 + lane + 4000 * trip, cells), cells
    return cells, old


def _add_carried_own_cell(cells, trips):
    lane, old = np.arange(len(cells)), np.zeros(len(cells), dtype=np.int64)
    for _ in range(trips):
        cells, old = cells + old + lane + 1, cells
    return cells, old


ORDERED = [_exchange_while, _compare_and_swap_while, _add_carried_while]
OWN_CELL = [
    (_exchange_own_cell_while, _exchange_own_cell),
    (_compare_and_swap_own_cell_while, _compare_and_swap_own_cell),
    (_add_carried_own_cell_while, _add_carried_own_cell),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trips", [1, 2])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kernel", ORDERED, ids=["exchange", "compare-and-swap", "add carried"])
def test_an_ordered_atomic_in_a_while_loop_is_refused_and_nothing_is_changed(kernel, size, trips, device):
    start = torch.arange(size, dtype=torch.int32) * 3
    z = start.to(device)
    with pytest.raises(MetalNonRecoverableError) as caught:
        kernel[(1,)](z, trips, N=size)
    message = str(caught.value)
    assert "lane order:" in message
    assert f"for the memory of argument(s) Z: {UNBOUNDED}" in message
    assert torch.equal(z.cpu(), start)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trips", [0, 1, 2, 3])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kernel, source", OWN_CELL, ids=["exchange", "compare-and-swap", "add carried"])
def test_an_ordered_atomic_of_each_lane_on_its_own_cell_computes_in_a_while_loop(kernel, source, size, trips, device):
    # The address is the same on every trip and no two lanes share a cell, so each cell's
    # updates are one lane's, in the order of its trips, however many there are.
    start = np.arange(size, dtype=np.int64) * 3
    z = torch.from_numpy(start.astype(np.int32)).to(device)
    r = torch.full((size,), -7, dtype=torch.int32, device=device)
    kernel[(1,)](z, r, trips, N=size)
    cells, returned = source(start, trips)
    assert np.array_equal(z.cpu().numpy(), cells)
    assert np.array_equal(r.cpu().numpy(), returned)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trips", [0, 1, 2, 3])
@pytest.mark.parametrize("size", [16] + SIZES)
def test_an_exchange_of_half_the_lanes_carried_by_a_while_loop_computes(size, trips, device):
    # The exchange's value depends on what the exchange returned on the trip before. Lane i
    # makes element i's exchange and keeps what it returned, so the value is held element
    # by lane on every trip.
    start = np.arange(size // 2, dtype=np.int64) * 3
    z = torch.from_numpy(start.astype(np.int32)).to(device)
    r = torch.full((size // 2,), -7, dtype=torch.int32, device=device)
    w = torch.full((size,), -7, dtype=torch.int32, device=device)
    _exchange_half_width_while[(1,)](z, r, w, trips, N=size)
    cells, returned = _exchange_own_cell(start, trips)
    assert np.array_equal(z.cpu().numpy(), cells)
    assert np.array_equal(r.cpu().numpy(), returned)
    assert np.array_equal(w.cpu().numpy(), np.arange(size) + 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
def test_an_add_in_a_while_loop_still_computes(device, size):
    start = np.arange(size, dtype=np.int64) * 3
    z = torch.from_numpy(start.astype(np.int32)).to(device)
    _add_while[(1,)](z, 2, N=size)
    lane = np.arange(size)
    want = start.copy()
    for trip in range(2):
        np.add.at(want, (lane + trip * (size // 2)) % size, lane + 1)
    assert np.array_equal(z.cpu().numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
def test_a_maximum_in_a_while_loop_still_computes(device, size):
    start = np.arange(size, dtype=np.int64) * 3
    z = torch.from_numpy(start.astype(np.int32)).to(device)
    _max_while[(1,)](z, 2, N=size)
    lane = np.arange(size)
    want = start.copy()
    for trip in range(2):
        np.maximum.at(want, lane % (size // 2), lane * 2 + trip)
    assert np.array_equal(z.cpu().numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
def test_an_exchange_outside_a_loop_still_computes(device, size):
    z = (torch.arange(size, dtype=torch.int32) * 3).to(device)
    _exchange_once[(1,)](z, N=size)
    assert torch.equal(z.cpu(), torch.arange(size, dtype=torch.int32) * 2 + 1)
