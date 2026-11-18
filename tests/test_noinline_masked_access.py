"""A masked load or store inside a function that is not inlined keeps its mask.

Such a function takes scalars, so its mask is one bit. The function's lowering used to
write the load and the store without it: the store landed where the source stores
nothing, and the load read memory where the source reads ``other``.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

POISON = -777.0


@triton.jit(noinline=True)
def _put(O, i, v, m):
    tl.store(O + i, v, mask=m)


@triton.jit
def _store_either(X, O, N, K):
    pid = tl.program_id(0)
    v = tl.load(X + pid)
    _put(O, pid, v * 2.0, pid < N)
    _put(O, pid + K, v * 3.0, pid >= N)


@triton.jit(noinline=True)
def _get(X, i, m):
    return tl.load(X + i, mask=m, other=-5.0)


@triton.jit
def _load_or_other(X, O, N):
    pid = tl.program_id(0)
    tl.store(O + pid, _get(X, pid, pid < N) * 2.0)


@triton.jit(noinline=True)
def _get_unset(X, i, m):
    return tl.load(X + i, mask=m)


@triton.jit
def _load_kept_or_nothing(X, O, N):
    pid = tl.program_id(0)
    m = pid < N
    tl.store(O + pid, _get_unset(X, pid, m) + 1.0, mask=m)


CASES = [(grid, kept) for grid in (4, 16, 33) for kept in sorted({0, 1, grid // 2, grid - 1, grid})]


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("grid,kept", CASES)
def test_a_masked_store_in_a_called_function_stores_only_where_the_mask_is_on(grid, kept, device):
    host = torch.arange(grid, dtype=torch.float32) + 1.0
    out = torch.full((2 * grid,), POISON, device=device)
    _store_either[(grid,)](host.to(device), out, kept, grid)
    want = torch.full((2 * grid,), POISON)
    want[:kept] = host[:kept] * 2.0
    want[grid + kept :] = host[kept:] * 3.0
    assert torch.equal(out.cpu(), want)


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("grid,kept", CASES)
def test_a_masked_load_in_a_called_function_gives_other_where_the_mask_is_off(grid, kept, device):
    # The input holds only the cells the mask keeps: the rest must not be read.
    host = torch.arange(max(kept, 1), dtype=torch.float32) + 1.0
    out = torch.full((grid + 8,), POISON, device=device)
    _load_or_other[(grid,)](host.to(device), out, kept)
    want = torch.full((grid + 8,), POISON)
    want[:grid] = -10.0
    want[:kept] = host[:kept] * 2.0
    assert torch.equal(out.cpu(), want)


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("grid,kept", CASES)
def test_a_masked_load_without_other_feeds_a_store_under_the_same_mask(grid, kept, device):
    host = torch.arange(max(kept, 1), dtype=torch.float32) + 1.0
    out = torch.full((grid + 8,), POISON, device=device)
    _load_kept_or_nothing[(grid,)](host.to(device), out, kept)
    want = torch.full((grid + 8,), POISON)
    want[:kept] = host[:kept] + 1.0
    assert torch.equal(out.cpu(), want)
