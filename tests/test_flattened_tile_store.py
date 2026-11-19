"""A tile flattened to one dimension is used whole beside a block of more lanes.

`tl.reshape(t, [R * C])` of an RxC tile is R * C elements, in a kernel that runs as many
lanes as its widest block. A lane past the tile holds the tile's index taken round again,
so its load was guarded and gave zero where the index was written without a modulo, and
the store through `tl.arange(0, R * C)` then took its lanes from the wide block: zeros
were stored over part of the tile. A load whose index is written with a modulo is in
range on every lane and was guarded the same way. The flattened tile is now proved to be
held one element a lane on the tile's own lanes, and what is not proved is refused.
"""

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
POISON = -7
# (rows, columns, lanes of the block beside the tile)
WIDE = [(16, 16, 1024), (2, 64, 256), (16, 16, 512), (4, 64, 1024), (32, 32, 2048), (2, 128, 1024)]
SMALL = [(8, 8, 128), (8, 8, 1024), (4, 16, 128), (16, 4, 128), (2, 2, 16), (2, 2, 8), (2, 4, 32), (4, 4, 64)]
STORED, PLUS_A_LOAD, MASKED, PLUS_ITS_RANGE, COMPUTED, ADDED, SUMMED = range(7)
ROWS, ROWS_AND_COLUMNS, ROWS_PLUS_A_LOAD = range(3)


@triton.jit
def _flattened(X, Y, Z, R: tl.constexpr, C: tl.constexpr, W: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.arange(0, R)
    b = tl.arange(0, C)
    f = tl.arange(0, R * C)
    w = tl.arange(0, W)
    z = Z + pid * 2 * R * C
    tl.store(Y + pid * 2 * W + w, tl.load(X + w + pid) + 1)
    o = a[:, None] * C + b[None, :]
    if FORM == 4:
        flat = tl.reshape((o * 3 + pid).to(X.dtype.element_ty), [R * C])
    else:
        flat = tl.reshape(tl.load(X + o + pid), [R * C])
    if FORM == 0 or FORM == 4:
        tl.store(z + f, flat)
    elif FORM == 1:
        tl.store(z + f, flat + tl.load(X + 64 + f + pid))
    elif FORM == 2:
        tl.store(z + f, flat, mask=f < R * C - 1)
    elif FORM == 3:
        tl.store(z + f, flat * 2 + f.to(X.dtype.element_ty))
    elif FORM == 5:
        tl.atomic_add(z + f, flat)
    else:
        tl.store(z, tl.sum(flat))


@triton.jit
def _circular(X, Y, Z, R: tl.constexpr, C: tl.constexpr, W: tl.constexpr, SHIFT: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.arange(0, R)
    b = tl.arange(0, C)
    f = tl.arange(0, R * C)
    w = tl.arange(0, W)
    z = Z + pid * 2 * R * C
    tl.store(Y + pid * 2 * W + w, tl.load(X + w + pid) + 1)
    if FORM == 1:
        t = tl.load(X + ((a[:, None] + SHIFT) % R) * C + (b[None, :] + SHIFT) % C + pid)
    else:
        t = tl.load(X + ((a[:, None] + SHIFT) % R) * C + b[None, :] + pid)
    if FORM == 2:
        tl.store(z + f, tl.reshape(t, [R * C]) + tl.load(X + 64 + f + pid))
    else:
        tl.store(z + f, tl.reshape(t, [R * C]))


@triton.jit
def _row_as_a_column(X, Z, K: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.arange(0, 1)
    b = tl.arange(0, K)
    t = tl.load(X + a[:, None] * K + b[None, :] + pid)
    tl.store(Z + pid * 2 * K + b[:, None] + a[None, :], tl.reshape(t, [K, 1]))


@triton.jit
def _range_as_a_row_beside_a_block(X, Y, Z, K: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.arange(0, 1)
    b = tl.arange(0, K)
    f = tl.arange(0, K)
    w = tl.arange(0, W)
    tl.store(Y + pid * 2 * W + w, tl.load(X + w + pid) + 1)
    v = tl.load(X + f + pid)
    tl.store(Z + pid * 2 * K + a[:, None] * K + b[None, :], tl.reshape(v, [1, K]))


def _inputs(total, wide, dtype, device):
    x = torch.randperm(2 * wide + 2 * total + 128, generator=torch.Generator().manual_seed(total + wide)).to(dtype)
    y = torch.full((GRID * 2 * wide,), POISON, dtype=dtype, device=device)
    z = torch.full((GRID * 2 * total,), POISON, dtype=dtype, device=device)
    return x, y, z


def _block_is_whole(x, y, wide):
    got = y.cpu().reshape(GRID, 2 * wide)
    for program in range(GRID):
        assert torch.equal(got[program, :wide], x[program : program + wide] + 1)
    assert torch.all(got[:, wide:] == POISON)


def _flattened_result(x, rows, cols, form):
    total = rows * cols
    want = torch.full((GRID, 2 * total), POISON, dtype=x.dtype)
    cells = torch.arange(total)
    for program in range(GRID):
        tile = x[program : program + total]
        if form == COMPUTED:
            tile = (cells * 3 + program).to(x.dtype)
        if form in (STORED, COMPUTED):
            want[program, :total] = tile
        elif form == PLUS_A_LOAD:
            want[program, :total] = tile + x[64 + program : 64 + program + total]
        elif form == MASKED:
            want[program, : total - 1] = tile[: total - 1]
        elif form == PLUS_ITS_RANGE:
            want[program, :total] = tile * 2 + cells.to(x.dtype)
        elif form == ADDED:
            want[program, :total] = tile + POISON
        else:
            want[program, 0] = tile.sum()
    return want.reshape(-1)


def _circular_result(x, rows, cols, shift, form):
    total = rows * cols
    want = torch.full((GRID, 2 * total), POISON, dtype=x.dtype)
    row = (torch.arange(rows)[:, None] + shift) % rows
    col = torch.arange(cols)[None, :]
    if form == ROWS_AND_COLUMNS:
        col = (col + shift) % cols
    cells = (row * cols + col).reshape(-1)
    for program in range(GRID):
        want[program, :total] = x[cells + program]
        if form == ROWS_PLUS_A_LOAD:
            want[program, :total] += x[64 + program : 64 + program + total]
    return want.reshape(-1)


def _runs_flattened(rows, cols, wide, form, dtype, device, refusal_allowed):
    x, y, z = _inputs(rows * cols, wide, dtype, device)
    try:
        _flattened[(GRID,)](x.to(device), y, z, rows, cols, wide, form)
    except MetalNonRecoverableError:
        assert refusal_allowed
        assert torch.all(y == POISON) and torch.all(z == POISON)
        return
    assert torch.equal(z.cpu(), _flattened_result(x, rows, cols, form))
    _block_is_whole(x, y, wide)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize(
    "form", [STORED, PLUS_A_LOAD, MASKED, PLUS_ITS_RANGE], ids=["stored", "plus a load", "masked", "plus its range"]
)
@pytest.mark.parametrize("rows,cols,wide", WIDE)
def test_a_flattened_tile_beside_a_wider_block_is_stored_whole(device, dtype, form, rows, cols, wide):
    _runs_flattened(rows, cols, wide, form, dtype, device, refusal_allowed=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "form", [STORED, PLUS_A_LOAD, MASKED, PLUS_ITS_RANGE], ids=["stored", "plus a load", "masked", "plus its range"]
)
@pytest.mark.parametrize("rows,cols,wide", SMALL)
def test_a_small_flattened_tile_beside_a_wider_block_is_right_or_refused(device, form, rows, cols, wide):
    _runs_flattened(rows, cols, wide, form, torch.float32, device, refusal_allowed=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("rows,cols,wide", WIDE + SMALL)
def test_a_computed_flattened_tile_beside_a_wider_block_is_right_or_refused(device, dtype, rows, cols, wide):
    _runs_flattened(rows, cols, wide, COMPUTED, dtype, device, refusal_allowed=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("rows,cols,wide", WIDE + SMALL)
def test_a_flattened_tile_beside_a_wider_block_is_added_whole(device, dtype, rows, cols, wide):
    _runs_flattened(rows, cols, wide, ADDED, dtype, device, refusal_allowed=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("rows,cols,wide", [tile for tile in SMALL if tile != (8, 8, 1024)])
def test_a_small_flattened_tile_beside_a_wider_block_is_summed_whole(device, dtype, rows, cols, wide):
    _runs_flattened(rows, cols, wide, SUMMED, dtype, device, refusal_allowed=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,cols,wide", WIDE + [(8, 8, 1024)])
def test_a_flattened_tile_beside_a_much_wider_block_is_summed_whole_or_refused(device, rows, cols, wide):
    _runs_flattened(rows, cols, wide, SUMMED, torch.float32, device, refusal_allowed=True)


def _runs_circular(rows, cols, wide, shift, form, device, refusal_allowed):
    x, y, z = _inputs(rows * cols, wide, torch.float32, device)
    try:
        _circular[(GRID,)](x.to(device), y, z, rows, cols, wide, shift, form)
    except MetalNonRecoverableError:
        assert refusal_allowed
        assert torch.all(y == POISON) and torch.all(z == POISON)
        return
    assert torch.equal(z.cpu(), _circular_result(x, rows, cols, shift, form))
    _block_is_whole(x, y, wide)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift", [0, 3])
@pytest.mark.parametrize(
    "form", [ROWS, ROWS_AND_COLUMNS, ROWS_PLUS_A_LOAD], ids=["rows", "rows and columns", "rows plus a load"]
)
@pytest.mark.parametrize("rows,cols,wide", WIDE)
def test_a_tile_loaded_round_its_rows_is_flattened_and_stored_whole(device, shift, form, rows, cols, wide):
    _runs_circular(rows, cols, wide, shift, form, device, refusal_allowed=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift", [0, 3])
@pytest.mark.parametrize(
    "form", [ROWS, ROWS_AND_COLUMNS, ROWS_PLUS_A_LOAD], ids=["rows", "rows and columns", "rows plus a load"]
)
@pytest.mark.parametrize("rows,cols,wide", SMALL)
def test_a_small_tile_loaded_round_its_rows_is_right_or_refused(device, shift, form, rows, cols, wide):
    _runs_circular(rows, cols, wide, shift, form, device, refusal_allowed=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("count", [8, 64, 512])
def test_one_row_reshaped_to_one_column_is_stored_whole(device, dtype, count):
    x, _, z = _inputs(count, count, dtype, device)
    _row_as_a_column[(GRID,)](x.to(device), z, count)
    want = torch.full((GRID, 2 * count), POISON, dtype=dtype)
    for program in range(GRID):
        want[program, :count] = x[program : program + count]
    assert torch.equal(z.cpu(), want.reshape(-1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float32", "int32"])
@pytest.mark.parametrize("count,wide", [(8, 64), (4, 64), (8, 128), (16, 64), (64, 1024)])
def test_a_range_reshaped_to_one_row_beside_a_wider_block_is_stored_whole(device, dtype, count, wide):
    x, y, z = _inputs(count, wide, dtype, device)
    _range_as_a_row_beside_a_block[(GRID,)](x.to(device), y, z, count, wide)
    want = torch.full((GRID, 2 * count), POISON, dtype=dtype)
    for program in range(GRID):
        want[program, :count] = x[program : program + count]
    assert torch.equal(z.cpu(), want.reshape(-1))
    _block_is_whole(x, y, wide)
