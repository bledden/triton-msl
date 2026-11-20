"""A row-wise sum is used the same way however many operations stand between it and its use.

`tl.sum(tile, axis=1)` is held one row per block of lanes, and each use of it has to know
that. The walk that finds it out gave up 24 operations deep: `v + 1` written 25 times
between the sum and a store was refused ("ttg.convert_layout cannot prove its source's
thread-to-element mapping"), although the same kernel with 24 additions computed. The
walk now meets every value once, so the answer no longer depends on the length of the
chain, and a chain whose every step uses the value three times does not take three times
as long per step to compile.
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
DTYPES = [torch.float32, torch.int32]
GRID = 3
ROOM = 2048
# (rows, columns) of the tile summed
TILES = [(2, 512), (256, 4), (32, 32)]
LENGTHS = [1, 25, 60, 130]
STORED, ADDED, SUMMED, LARGEST, SPREAD, GATHERED, MASKED = range(7)
FORMS = ["stored", "added", "summed", "largest", "spread over the tile", "gathered through", "stored under its mask"]
ONES, INDEXED, BRANCHED = range(3)
CHAINS = ["plus one", "plus its index", "branched"]


@triton.jit
def _behind(X, Y, Z, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr, FORM: tl.constexpr, CHAIN: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.arange(0, R * C)
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    tile = tl.load(X + r[:, None] * C + c[None, :] + pid)
    v = tl.sum(tile, axis=1)
    for i in tl.static_range(K):
        if CHAIN == 0:
            v = v + 1
        elif CHAIN == 1:
            v = v + (r + i).to(X.dtype.element_ty)
        else:
            v = tl.where(v > 40, v - 7, v + 3)
    z = Z + pid * 2048
    if FORM == 0:
        tl.store(z + r, v)
        tl.store(Y + pid * R * C + w, tl.load(X + pid * 8 + w) + 1)
    elif FORM == 1:
        tl.atomic_add(z + r, v)
        tl.store(Y + pid * R * C + w, tl.load(X + pid * 8 + w) + 1)
    elif FORM == 2:
        tl.store(z, tl.sum(v, axis=0))
    elif FORM == 3:
        tl.store(z, tl.max(v, axis=0))
    elif FORM == 4:
        tl.store(Y + pid * R * C + r[:, None] * C + c[None, :], tile + v[:, None])
    elif FORM == 5:
        tl.store(z + r, tl.load(X + v.to(tl.int32) % 64))
    else:
        tl.store(z + r, v, mask=v > 40)


@triton.jit
def _two_sums(X, Z, R: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, R)
    tile = tl.load(X + r[:, None] * R + r[None, :] + pid)
    v = tl.sum(tile, axis=1) + tl.sum(tile, axis=0)
    for _ in tl.static_range(K):
        v = v + 1
    tl.store(Z + pid * 2048 + r, v)


def _chain(v, rows, length, chain, kind):
    for step in range(length):
        if chain == ONES:
            v = v + kind(1)
        elif chain == INDEXED:
            v = v + (rows + step).astype(kind)
        else:
            v = np.where(v > 40, v - kind(7), v + kind(3))
    return v.astype(kind)


def _inputs(rows, cols, kind):
    lanes = rows * cols
    x = ((np.arange(2 * lanes + 8 * GRID) * 7 + 3) % 13).astype(kind)
    y = np.full(GRID * lanes, -7, dtype=kind)
    z = ((np.arange(GRID * ROOM) * 5) % 11 + 2).astype(kind)
    return x, y, z


def _expected(rows, cols, length, form, chain, kind):
    x, y, z = _inputs(rows, cols, kind)
    lanes = rows * cols
    r, c = np.arange(rows), np.arange(cols)
    for pid in range(GRID):
        tile = x[r[:, None] * cols + c[None, :] + pid]
        v = _chain(tile.sum(1).astype(kind), r, length, chain, kind)
        cells = pid * ROOM + r
        block = slice(pid * lanes, (pid + 1) * lanes)
        if form in (STORED, ADDED):
            z[cells] = v if form == STORED else z[cells] + v
            y[block] = x[pid * 8 : pid * 8 + lanes] + kind(1)
        elif form == SUMMED:
            z[pid * ROOM] = v.sum(dtype=kind)
        elif form == LARGEST:
            z[pid * ROOM] = v.max()
        elif form == SPREAD:
            y[block] = (tile + v[:, None]).reshape(-1)
        elif form == GATHERED:
            z[cells] = x[v.astype(np.int64) % 64]
        else:
            z[cells[v > 40]] = v[v > 40]
    return y, z


def _launch(rows, cols, length, form, chain, dtype, device):
    kind = np.float32 if dtype == torch.float32 else np.int32
    x, y, z = (torch.tensor(values, device=device) for values in _inputs(rows, cols, kind))
    _behind[(GRID,)](x, y, z, R=rows, C=cols, K=length, FORM=form, CHAIN=chain)
    if device == "mps":
        torch.mps.synchronize()
    return y.cpu().numpy(), z.cpu().numpy(), _expected(rows, cols, length, form, chain, kind)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("chain", range(len(CHAINS)), ids=CHAINS)
@pytest.mark.parametrize("form", range(len(FORMS)), ids=FORMS)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rows,cols", TILES)
def test_a_row_sum_behind_many_operations_is_used_as_behind_one(device, dtype, chain, form, length, rows, cols):
    y, z, (want_y, want_z) = _launch(rows, cols, length, form, chain, dtype, device)
    assert np.array_equal(z, want_z)
    assert np.array_equal(y, want_y)


def _two_sums_outcome(rows, length, dtype, device):
    kind = np.float32 if dtype == torch.float32 else np.int32
    x0 = ((np.arange(2 * rows * rows + GRID) * 7 + 3) % 13).astype(kind)
    z0 = ((np.arange(GRID * ROOM) * 5) % 11 + 2).astype(kind)
    want = z0.copy()
    r = np.arange(rows)
    for pid in range(GRID):
        tile = x0[r[:, None] * rows + r[None, :] + pid]
        want[pid * ROOM + r] = tile.sum(1) + tile.sum(0) + kind(length)
    x, z = torch.tensor(x0, device=device), torch.tensor(z0, device=device)
    try:
        _two_sums[(GRID,)](x, z, R=rows, K=length)
    except MetalNonRecoverableError:
        assert np.array_equal(z.cpu().numpy(), z0)
        return "refused"
    if device == "mps":
        torch.mps.synchronize()
    assert np.array_equal(z.cpu().numpy(), want)
    return "computed"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("rows", [16, 32])
def test_a_row_sum_and_a_column_sum_together_are_right_or_refused_at_every_length(device, dtype, rows):
    """Two sums held in different ways meet in one value: no length of chain makes that provable."""
    outcomes = {length: _two_sums_outcome(rows, length, dtype, device) for length in (0, 1, 25, 60)}
    assert len(set(outcomes.values())) == 1, outcomes
