"""The transpose of a tile is stored one lane per element, also when it has one column.

A store through a pointer of one column used to take its lanes from the kernel's widest
block: "the lane at the start of each row of that block". The transpose of a 1xN tile is
an (N, 1) value in a kernel whose widest block is the 1xN tile itself, where that rule
names lane 0 alone, so one element was stored and the other N - 1 cells kept what they
held. The lanes are now read from how the pointer's index is written, and the lowering
knows where the lanes hold a transposed tile; a transpose it cannot place is refused.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
POISON = -7.0
# Tiles the lowering computes, with one row, one column and neither.
TILES = [(1, 8), (1, 64), (1, 512), (8, 1), (64, 1), (2, 4), (4, 2), (4, 4), (2, 16), (8, 8), (16, 32)]


@triton.jit
def _transpose(X, Z, R: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.arange(0, R)
    b = tl.arange(0, C)
    t = tl.load(X + a[:, None] * C + b[None, :] + pid)
    tl.store(Z + pid * 2 * R * C + b[:, None] * R + a[None, :], tl.trans(t))


@triton.jit
def _transpose_beside_a_block(X, Y, Z, R: tl.constexpr, C: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.arange(0, R)
    b = tl.arange(0, C)
    w = tl.arange(0, W)
    t = tl.load(X + a[:, None] * C + b[None, :] + pid)
    tl.store(Y + pid * 2 * W + w, tl.load(X + w + pid) + 1)
    tl.store(Z + pid * 2 * R * C + b[:, None] * R + a[None, :], tl.trans(t))


def _transposed(x, rows, cols, grid):
    total = rows * cols
    want = torch.full((grid * 2 * total,), POISON)
    for program in range(grid):
        tile = x[program : program + total].reshape(rows, cols)
        want[program * 2 * total : program * 2 * total + total] = tile.T.reshape(-1)
    return want


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid", [1, 3])
@pytest.mark.parametrize("rows,cols", TILES)
def test_a_transposed_tile_is_stored_whole(device, grid, rows, cols):
    total = rows * cols
    x = torch.randperm(2 * total + 8, generator=torch.Generator().manual_seed(total)).float()
    z = torch.full((grid * 2 * total,), POISON, device=device)
    _transpose[(grid,)](x.to(device), z, rows, cols)
    assert torch.equal(z.cpu(), _transposed(x, rows, cols, grid))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid", [1, 3])
@pytest.mark.parametrize("narrow", [False, True], ids=["as wide", "half as wide"])
@pytest.mark.parametrize("rows,cols", TILES)
def test_a_transposed_tile_beside_a_block_no_wider_is_stored_whole(device, grid, narrow, rows, cols):
    total = rows * cols
    wide = total // 2 if narrow else total
    x = torch.randperm(2 * total + 8, generator=torch.Generator().manual_seed(total)).float()
    y = torch.full((grid * 2 * wide,), POISON, device=device)
    z = torch.full((grid * 2 * total,), POISON, device=device)
    _transpose_beside_a_block[(grid,)](x.to(device), y, z, rows, cols, wide)
    assert torch.equal(z.cpu(), _transposed(x, rows, cols, grid))
    got = y.cpu().reshape(grid, 2 * wide)
    for program in range(grid):
        assert torch.equal(got[program, :wide], x[program : program + wide] + 1)
    assert torch.all(got[:, wide:] == POISON)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,cols", TILES)
def test_a_transposed_tile_beside_a_wider_block_is_right_or_refused(device, rows, cols):
    # The exchange the transpose is lowered to has one cell a lane: in a kernel of more
    # lanes than the tile has elements the lowering cannot place it.
    total = rows * cols
    wide = 2 * total
    x = torch.randperm(2 * wide + 8, generator=torch.Generator().manual_seed(total)).float()
    y = torch.full((2 * wide,), POISON, device=device)
    z = torch.full((2 * total,), POISON, device=device)
    try:
        _transpose_beside_a_block[(1,)](x.to(device), y, z, rows, cols, wide)
    except MetalNonRecoverableError:
        assert torch.all(y == POISON) and torch.all(z == POISON)
        return
    assert torch.equal(z.cpu(), _transposed(x, rows, cols, 1))
    assert torch.equal(y.cpu()[:wide], x[:wide] + 1)
