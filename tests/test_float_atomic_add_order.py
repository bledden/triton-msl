"""Float atomic adds that reach one cell from different lanes in different operations are refused.

Two ``tl.atomic_add`` operations (two statements, or one in a loop) are applied lane by lane:
each thread runs its operations without waiting for the other threads to finish the first
one. The source finishes each operation over every lane before the next. An integer cell is
exact in either order. A float cell receives the same terms in another order, and the order
of float additions is their rounding (``1e8 + 1 - 1e8`` is 0.0 in the source's order and 1.0
in the other), so such a launch is refused before it is submitted. Where each cell is reached
by one lane only, that lane's own order is the source's and the launch computes, in a ``while``
loop as well; so does one float atomic add outside a loop, whose lanes the source does not order.

The check compares addresses it can compute before the launch. A mask made of loaded values
is counted as taken, and an index that is loaded is not followed at all, so a launch that
only such a mask or index keeps to one lane per cell is refused too, and the refusal says
which of the two it is. A branch on the launch's integer scalars is read: the add under it
is counted where the branch is taken and left out where it is not.
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
SIZES = [64, 256, 1024, 2048, 4096]
REFUSAL = "lane order"


@triton.jit
def _two_statements(Z, V, TRIPS, N: tl.constexpr):
    # The upper half of the lanes first, then the lower half, onto the same cells.
    n = tl.arange(0, N)
    half = N // 2
    v = tl.load(V + n)
    tl.atomic_add(Z + n % half, v, mask=n >= half)
    tl.atomic_add(Z + n % half, v, mask=n < half)


@triton.jit
def _counted_loop(Z, V, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    half = N // 2
    v = tl.load(V + n)
    for j in range(TRIPS):
        tl.atomic_add(Z + n % half, v, mask=(n // half) == (1 - j))


@triton.jit
def _while_loop(Z, V, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    half = N // 2
    v = tl.load(V + n)
    j = 0
    while j < TRIPS:
        tl.atomic_add(Z + n % half, v, mask=(n // half) == (1 - j))
        j += 1


@triton.jit
def _own_cell_two_statements(Z, V, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    tl.atomic_add(Z + n, tl.load(V + n))
    tl.atomic_add(Z + n, tl.load(V + N + n))


@triton.jit
def _own_cell_counted_loop(Z, V, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    for j in range(TRIPS):
        tl.atomic_add(Z + n, tl.load(V + j * N + n))


@triton.jit
def _own_cell_while_loop(Z, V, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    j = 0
    while j < TRIPS:
        tl.atomic_add(Z + n, tl.load(V + j * N + n))
        j += 1


@triton.jit
def _one_statement(Z, V, TRIPS, N: tl.constexpr):
    n = tl.arange(0, N)
    tl.atomic_add(Z + n % (N // 2), tl.load(V + n))


@triton.jit
def _two_statements_under_a_loaded_mask(Z, V, K, N: tl.constexpr):
    n = tl.arange(0, N)
    half = N // 2
    keep = tl.load(K + n) != 0
    v = tl.load(V + n)
    tl.atomic_add(Z + n % half, v, mask=keep)
    tl.atomic_add(Z + n % half, tl.load(V + N + n), mask=keep)


@triton.jit
def _second_add_under_a_branch(Z, V, S, N: tl.constexpr):
    n = tl.arange(0, N)
    tl.atomic_add(Z + n, tl.load(V + n))
    if S > 0:
        tl.atomic_add(Z + (N - 1 - n), tl.load(V + N + n))


@triton.jit
def _counted_loop_through_a_loaded_index(Z, V, I, N: tl.constexpr):
    n = tl.arange(0, N)
    cell = tl.load(I + n)
    for j in range(2):
        tl.atomic_add(Z + cell, tl.load(V + j * N + n))


MAY_MEET = (
    "two lanes of one program reach the same element, one of them storing it, unless a branch, or a mask "
    "computed from values the check cannot know before the launch, keeps them apart"
)
MEET = "two lanes of one program reach the same element and one of them stores it"
NOT_COMPUTED = "an address depends on a value the check cannot compute before the launch"

KERNELS = [_two_statements, _counted_loop, _while_loop]
IDS = ["two statements", "counted loop", "while loop"]


def _run(kernel, start, values, size, device):
    z = torch.from_numpy(start.copy()).to(device)
    kernel[(1,)](z, torch.from_numpy(values).to(device), 2, N=size)
    return z.cpu().numpy()


def _refused(kernel, start, values, size, device):
    """The launch is refused before anything is submitted: the cells keep what they held."""
    z = torch.from_numpy(start.copy()).to(device)
    with pytest.raises(MetalNonRecoverableError, match=REFUSAL):
        kernel[(1,)](z, torch.from_numpy(values).to(device), 2, N=size)
    assert np.array_equal(z.cpu().numpy(), start)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kernel", KERNELS, ids=IDS)
def test_float_adds_from_different_lanes_in_different_operations_are_refused(kernel, size, device):
    half = size // 2
    start = np.full(half, 1e8, dtype=np.float32)
    values = np.concatenate([np.full(half, -1e8), np.ones(half)]).astype(np.float32)
    _refused(kernel, start, values, size, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kernel", KERNELS, ids=IDS)
def test_they_are_refused_where_no_order_rounds_as_well(kernel, size, device):
    # The check reads addresses, not values: it cannot know that these sums are exact.
    half = size // 2
    start = (np.arange(half) % 7).astype(np.float32)
    values = (np.arange(size) % 13 - 6).astype(np.float32)
    _refused(kernel, start, values, size, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kernel", KERNELS, ids=IDS)
def test_the_integer_twins_are_exact(kernel, size, device):
    half = size // 2
    start = (np.arange(half) * 3).astype(np.int32)
    values = (np.arange(size) * 5 - 7).astype(np.int32)
    got = _run(kernel, start, values, size, device)
    assert np.array_equal(got, start + values[half:] + values[:half])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "kernel", [_own_cell_two_statements, _own_cell_counted_loop], ids=["two statements", "counted loop"]
)
def test_float_adds_of_each_lane_to_its_own_cell_keep_the_source_s_order(kernel, size, device):
    # 1e8 + 1 - 1e8: 0.0 in the source's order, 1.0 in the other.
    start = np.full(size, 1e8, dtype=np.float32)
    values = np.concatenate([np.ones(size), np.full(size, -1e8)]).astype(np.float32)
    got = _run(kernel, start, values, size, device)
    assert np.array_equal(got, (start + values[:size]) + values[size:])
    assert np.all(got == 0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
def test_a_float_add_of_each_lane_to_its_own_cell_computes_in_a_while_loop(size, device):
    # The address is the same on every trip, so one comparison stands for all of them.
    start = np.full(size, 1e8, dtype=np.float32)
    values = np.concatenate([np.ones(size), np.full(size, -1e8)]).astype(np.float32)
    got = _run(_own_cell_while_loop, start, values, size, device)
    assert np.array_equal(got, (start + values[:size]) + values[size:])
    assert np.all(got == 0.0)


def _refused_for(reason, kernel, start, values, extra, size, device):
    z = torch.from_numpy(start.copy()).to(device)
    if isinstance(extra, np.ndarray):
        extra = torch.from_numpy(extra).to(device)
    with pytest.raises(MetalNonRecoverableError) as caught:
        kernel[(1,)](z, torch.from_numpy(values).to(device), extra, N=size)
    assert (
        f"lane order: this launch is not proved to keep the source's order for the memory of argument(s) Z: {reason}."
        in str(caught.value)
    )
    assert np.array_equal(z.cpu().numpy(), start)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [64, 2048])
def test_a_loaded_mask_that_keeps_the_lanes_apart_is_not_read(size, device):
    # The lower half of the lanes only: one lane for each cell. The check cannot know.
    start = np.full(size // 2, 1e8, dtype=np.float32)
    values = np.concatenate([np.ones(size), np.full(size, -1e8)]).astype(np.float32)
    keep = (np.arange(size) < size // 2).astype(np.int32)
    _refused_for(MAY_MEET, _two_statements_under_a_loaded_mask, start, values, keep, size, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [64, 2048])
@pytest.mark.parametrize("scalar", [0, -3])
def test_an_add_under_a_branch_on_a_scalar_that_is_not_taken_is_left_out(scalar, size, device):
    start = np.full(size, 1e8, dtype=np.float32)
    values = np.concatenate([np.ones(size), np.full(size, -1e8)]).astype(np.float32)
    z = torch.from_numpy(start.copy()).to(device)
    _second_add_under_a_branch[(1,)](z, torch.from_numpy(values).to(device), scalar, N=size)
    assert np.array_equal(z.cpu().numpy(), start + values[:size])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [64, 2048])
@pytest.mark.parametrize("scalar", [2, 2**31 - 1])
def test_an_add_under_a_branch_on_a_scalar_that_is_taken_is_counted(scalar, size, device):
    start = np.full(size, 1e8, dtype=np.float32)
    values = np.concatenate([np.ones(size), np.full(size, -1e8)]).astype(np.float32)
    _refused_for(MEET, _second_add_under_a_branch, start, values, scalar, size, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [64, 2048])
def test_a_loaded_index_is_not_followed(size, device):
    # The index is the identity: each lane adds to its own cell on both trips.
    start = np.full(size, 1e8, dtype=np.float32)
    values = np.concatenate([np.ones(size), np.full(size, -1e8)]).astype(np.float32)
    _refused_for(
        NOT_COMPUTED, _counted_loop_through_a_loaded_index, start, values, np.arange(size, dtype=np.int32), size, device
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
def test_one_float_add_whose_lanes_meet_still_computes(size, device):
    # One operation: the source does not order its lanes. These sums are exact in any order.
    half = size // 2
    start = (np.arange(half) % 7).astype(np.float32)
    values = (np.arange(size) % 13 - 6).astype(np.float32)
    got = _run(_one_statement, start, values, size, device)
    assert np.array_equal(got, start + values[:half] + values[half:])
