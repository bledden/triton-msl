"""The row-wise sort template stands only for a kernel that is exactly a row-wise sort.

The template is a whole kernel: it loads row ``lid`` of the input, sorts it and stores it,
and nothing else of the source reaches the shader. It used to be chosen from a signature (a
hypercube reshape and an xor reduction) that a flip of a 2xN tile, a sort whose result is
changed before the store, masked accesses and offset addresses all fit, and those kernels
then ran as a plain sort. The template is now chosen by a proof of the whole kernel; every
other kernel goes to the ordinary lowering, which computes it or refuses it.

The ordinary lowering runs a tile of at most 1024 elements. Beside the plain form the proof
reads four more things, each a fact the template is then handed: a store that converts the
sorted values, a load that takes each column from one fixed place of its row, masks that
are true on every cell, and a sort of a sort, up to four. A sort of more than 1024 elements
in those forms takes the template. At most 1024 elements it keeps the ordinary lowering,
and a larger kernel that is anything more is refused, with the reason.

The template's network is also the source's own, step for step. It used to swap a pair
that was out of order, comparing signed: unsigned values above the signed range were
sorted as negative ones, and where neither of a pair is greater (a NaN, a zero of each
sign) the source leaves one value in both cells while the template kept both.
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
POISON = -7
SHAPES = [(4, 8), (2, 16), (16, 16), (8, 64), (256, 16)]
DTYPES = [torch.float32, torch.int32]


@triton.jit
def _plain(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.load(X + o), dim=1, descending=D))


@triton.jit
def _padded_rows(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * (C + 1) + c, tl.sort(tl.load(X + r * (C + 2) + c), dim=1, descending=D))


@triton.jit
def _per_program(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + pid * R * C + o, tl.sort(tl.load(X + o + pid), dim=1, descending=D))


@triton.jit
def _plus_one(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.load(X + o), dim=1, descending=D) + 1)


@triton.jit
def _flipped(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.flip(tl.sort(tl.load(X + o), dim=1, descending=D), 1))


@triton.jit
def _masked_load(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    c = tl.arange(0, C)[None, :]
    o = tl.arange(0, R)[:, None] * C + c
    tl.store(Z + o, tl.sort(tl.load(X + o, mask=c < C - 1, other=-3), dim=1, descending=D))


@triton.jit
def _masked_store(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    c = tl.arange(0, C)[None, :]
    o = tl.arange(0, R)[:, None] * C + c
    tl.store(Z + o, tl.sort(tl.load(X + o), dim=1, descending=D), mask=c < C - 1)


@triton.jit
def _offsets(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o + 5, tl.sort(tl.load(X + o + 3), dim=1, descending=D))


@triton.jit
def _sorted_twice(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.sort(tl.load(X + o), dim=1, descending=D), dim=1, descending=1 - D))


@triton.jit
def _negated(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(-tl.load(X + o), dim=1, descending=D))


@triton.jit
def _flip_two_rows(X, Z, C: tl.constexpr, AXIS: tl.constexpr):
    # A flip of a 2xC tile has the sort's reshape and its xor reduction.
    o = tl.arange(0, 2)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.flip(tl.load(X + o), AXIS))


@triton.jit
def _reversed_columns(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * C + c, tl.sort(tl.load(X + r * C + (C - 1 - c)), dim=1, descending=D))


@triton.jit
def _masked_everywhere(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.load(X + o, mask=o < R * C, other=0), dim=1, descending=D), mask=o < R * C)


@triton.jit
def _pair_swapped_columns(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * C + c, tl.sort(tl.load(X + r * C + (c ^ 1)), dim=1, descending=D))


@triton.jit
def _shifted_load(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.load(X + o + 3), dim=1, descending=D))


@triton.jit
def _one_column(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * C + c, tl.sort(tl.load(X + r * C + c * 0 + 1), dim=1, descending=D))


@triton.jit
def _four_sorts(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    twice = tl.sort(tl.sort(tl.load(X + o), dim=1, descending=D), dim=1, descending=1 - D)
    tl.store(Z + o, tl.sort(tl.sort(twice, dim=1, descending=1 - D), dim=1, descending=D))


@triton.jit
def _five_sorts(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    twice = tl.sort(tl.sort(tl.load(X + o), dim=1, descending=D), dim=1, descending=1 - D)
    four = tl.sort(tl.sort(twice, dim=1, descending=1 - D), dim=1, descending=D)
    tl.store(Z + o, tl.sort(four, dim=1, descending=1 - D))


@triton.jit
def _all_together(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    every = r * C + c < R * C
    loaded = tl.load(X + r * C + (C - 1 - c), mask=every, other=0)
    tl.store(Z + r * C + c, tl.sort(tl.sort(loaded, dim=1, descending=1 - D), dim=1, descending=D), mask=every)


@triton.jit
def _stored_as(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr, AS: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.load(X + o), dim=1, descending=D).to(AS))


@triton.jit
def _half_of_the_columns(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * C + c, tl.sort(tl.load(X + r * C + c // 2 * 2), dim=1, descending=D))


@triton.jit
def _columns_that_follow_the_row(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    o = r * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.sort(tl.load(X + (o ^ r)), dim=1, descending=D))


@triton.jit
def _columns_scaled_by_the_row(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * C + c, tl.sort(tl.load(X + r * C + c * r), dim=1, descending=D))


@triton.jit
def _partly_masked(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    most = o < R * C - 1
    tl.store(Z + o, tl.sort(tl.load(X + o, mask=most, other=0), dim=1, descending=D), mask=most)


@triton.jit
def _in_its_own_argument(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(X + o, tl.sort(tl.load(X + o), dim=1, descending=D))


@triton.jit
def _a_converted_value_for_no_cell(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    o = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    loaded = tl.load(X + o, mask=o < R * C, other=o.to(tl.float32))
    tl.store(Z + o, tl.sort(loaded, dim=1, descending=D))


@triton.jit
def _rows_that_share_a_loaded_cell(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * C + c, tl.sort(tl.load(X + r * (C - 1) + c), dim=1, descending=D))


@triton.jit
def _rows_that_share_a_stored_cell(X, Z, R: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(Z + r * (C - 1) + c, tl.sort(tl.load(X + r * C + c), dim=1, descending=D))


def _ordered(tile, descending):
    return torch.sort(tile, dim=1, descending=bool(descending)).values


def _input(count, dtype):
    order = torch.randperm(count, generator=torch.Generator().manual_seed(count))
    return (order - count // 3).to(dtype)


def _is_the_template(launch):
    msl = launch.asm["msl"]
    return "_mmarg0" in msl and " v[_i] = " in msl


def _launch(kernel, grid, x, z, *constants):
    """The launch, or None when it is refused; a refusal leaves the output as it was."""
    before = z.clone()
    try:
        return kernel[grid](x, z, *constants)
    except MetalNonRecoverableError:
        assert torch.equal(z, before)
        return None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "i32"])
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_a_plain_sort_takes_the_template(device, dtype, descending, rows, cols):
    total = rows * cols
    x = _input(total, dtype)
    z = torch.full((total + 8,), POISON, dtype=dtype, device=device)
    launch = _plain[(1,)](x.to(device), z, rows, cols, descending)
    assert _is_the_template(launch)
    assert torch.equal(z[:total].cpu().reshape(rows, cols), _ordered(x.reshape(rows, cols), descending))
    assert torch.all(z[total:] == POISON)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "i32"])
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_a_sort_of_padded_rows_takes_the_template(device, dtype, descending, rows, cols):
    x = _input(rows * (cols + 2), dtype)
    z = torch.full((rows * (cols + 1),), POISON, dtype=dtype, device=device)
    launch = _padded_rows[(1,)](x.to(device), z, rows, cols, descending)
    assert _is_the_template(launch)
    got = z.cpu().reshape(rows, cols + 1)
    assert torch.equal(got[:, :cols], _ordered(x.reshape(rows, cols + 2)[:, :cols], descending))
    assert torch.all(got[:, cols] == POISON)


def _want_per_program(x, rows, cols, descending, grid):
    total = rows * cols
    return torch.cat([_ordered(x[p : p + total].reshape(rows, cols), descending).reshape(-1) for p in range(grid)])


def _want_masked_load(x, rows, cols, descending):
    tile = x[: rows * cols].reshape(rows, cols).clone()
    tile[:, cols - 1] = -3
    return _ordered(tile, descending).reshape(-1)


def _want_masked_store(x, rows, cols, descending):
    tile = _ordered(x[: rows * cols].reshape(rows, cols), descending)
    tile[:, cols - 1] = POISON
    return tile.reshape(-1)


# name, kernel, grid, first written cell, the written cells as the source defines them
OTHER_KERNELS = [
    ("per_program", _per_program, 3, 0, lambda x, r, c, d: _want_per_program(x, r, c, d, 3)),
    ("plus_one", _plus_one, 1, 0, lambda x, r, c, d: (_ordered(x[: r * c].reshape(r, c), d) + 1).reshape(-1)),
    ("flipped", _flipped, 1, 0, lambda x, r, c, d: _ordered(x[: r * c].reshape(r, c), d).flip(1).reshape(-1)),
    ("masked_load", _masked_load, 1, 0, _want_masked_load),
    ("masked_store", _masked_store, 1, 0, _want_masked_store),
    ("offsets", _offsets, 1, 5, lambda x, r, c, d: _ordered(x[3 : 3 + r * c].reshape(r, c), d).reshape(-1)),
    ("negated", _negated, 1, 0, lambda x, r, c, d: _ordered(-x[: r * c].reshape(r, c), d).reshape(-1)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "i32"])
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("case", OTHER_KERNELS, ids=[case[0] for case in OTHER_KERNELS])
def test_a_kernel_that_is_more_than_a_sort_is_not_run_as_one(case, rows, cols, descending, dtype, device):
    _, kernel, grid, first, want_of = case
    total = rows * cols
    x = _input(total + 8, dtype)
    want = want_of(x, rows, cols, descending)
    z = torch.full((first + want.numel() + 8,), POISON, dtype=dtype, device=device)
    launch = _launch(kernel, (grid,), x.to(device), z, rows, cols, descending)
    if launch is None:
        return
    assert not _is_the_template(launch)
    got = z.cpu()
    assert torch.equal(got[first : first + want.numel()], want)
    assert torch.all(got[:first] == POISON) and torch.all(got[first + want.numel() :] == POISON)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", OTHER_KERNELS, ids=[case[0] for case in OTHER_KERNELS])
def test_the_other_kernels_compute_on_a_small_tile(case, device):
    # The test above accepts a refusal; this one says the small tiles are not refused.
    _, kernel, grid, first, want_of = case
    x = _input(4 * 8 + 8, torch.float32)
    want = want_of(x, 4, 8, 0)
    z = torch.full((first + want.numel() + 8,), POISON, dtype=torch.float32, device=device)
    kernel[(grid,)](x.to(device), z, 4, 8, 0)
    assert torch.equal(z.cpu()[first : first + want.numel()], want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "i32"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("cols", [4, 8, 16, 64, 512])
def test_a_flip_of_two_rows_is_not_a_sort(device, dtype, axis, cols):
    # Along axis 1, from 8 columns up, this came back sorted; along axis 0 it never did.
    x = _input(2 * cols, dtype)
    z = torch.full((2 * cols + 8,), POISON, dtype=dtype, device=device)
    launch = _flip_two_rows[(1,)](x.to(device), z, cols, axis)
    assert not _is_the_template(launch)
    assert torch.equal(z[: 2 * cols].cpu().reshape(2, cols), x.reshape(2, cols).flip(axis))
    assert torch.all(z[2 * cols :] == POISON)


TOO_LARGE = "2D kernel with cooperative ops exceeds 1024 elements"
LARGER = [(64, 32), (512, 8), (4, 512)]


def _tile(x, rows, cols):
    return x[: rows * cols].reshape(rows, cols)


# name, kernel, output dtype, the source's result for an int32 input of rows * cols + 8 cells
BESIDE_PLAIN = [
    ("converting_store", _plain, torch.float32, lambda x, r, c, d: _ordered(_tile(x, r, c), d).to(torch.float32)),
    ("reversed_columns", _reversed_columns, torch.int32, lambda x, r, c, d: _ordered(_tile(x, r, c), d)),
    ("pair_swapped_columns", _pair_swapped_columns, torch.int32, lambda x, r, c, d: _ordered(_tile(x, r, c), d)),
    ("shifted_load", _shifted_load, torch.int32, lambda x, r, c, d: _ordered(_tile(x[3:], r, c), d)),
    ("one_column", _one_column, torch.int32, lambda x, r, c, d: _tile(x, r, c)[:, 1:2].expand(r, c)),
    ("masked_everywhere", _masked_everywhere, torch.int32, lambda x, r, c, d: _ordered(_tile(x, r, c), d)),
    ("sorted_twice", _sorted_twice, torch.int32, lambda x, r, c, d: _ordered(_tile(x, r, c), 1 - d)),
    ("four_sorts", _four_sorts, torch.int32, lambda x, r, c, d: _ordered(_tile(x, r, c), d)),
    ("all_together", _all_together, torch.float32, lambda x, r, c, d: _ordered(_tile(x, r, c), d).to(torch.float32)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", LARGER)
@pytest.mark.parametrize("case", BESIDE_PLAIN, ids=[case[0] for case in BESIDE_PLAIN])
def test_a_larger_sort_in_a_form_the_proof_reads_takes_the_template(case, rows, cols, descending, device):
    _, kernel, out_dtype, want_of = case
    total = rows * cols
    x = _input(total + 8, torch.int32)
    z = torch.full((total + 8,), POISON, dtype=out_dtype, device=device)
    launch = kernel[(1,)](x.to(device), z, rows, cols, descending)
    assert _is_the_template(launch)
    assert torch.equal(z[:total].cpu().reshape(rows, cols), want_of(x, rows, cols, descending))
    assert torch.all(z[total:] == POISON)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", [(16, 16), (32, 32)])
@pytest.mark.parametrize("case", BESIDE_PLAIN, ids=[case[0] for case in BESIDE_PLAIN])
def test_the_same_sort_of_at_most_1024_elements_keeps_the_ordinary_lowering(case, rows, cols, descending, device):
    _, kernel, out_dtype, want_of = case
    total = rows * cols
    x = _input(total + 8, torch.int32)
    z = torch.full((total + 8,), POISON, dtype=out_dtype, device=device)
    launch = kernel[(1,)](x.to(device), z, rows, cols, descending)
    assert not _is_the_template(launch)
    assert torch.equal(z[:total].cpu().reshape(rows, cols), want_of(x, rows, cols, descending))
    assert torch.all(z[total:] == POISON)


# name, kernel, grid, input dtype, output dtype, input cells for each cell of the tile
MORE_THAN_THESE = [
    ("partly_masked", _partly_masked, 1, torch.int32, torch.int32, 1),
    ("a_column_masked_off_the_load", _masked_load, 1, torch.int32, torch.int32, 1),
    ("a_column_masked_off_the_store", _masked_store, 1, torch.int32, torch.int32, 1),
    ("shifted_store", _offsets, 1, torch.int32, torch.int32, 1),
    ("plus_one", _plus_one, 1, torch.int32, torch.int32, 1),
    ("flipped", _flipped, 1, torch.int32, torch.int32, 1),
    ("negated", _negated, 1, torch.int32, torch.int32, 1),
    ("per_program", _per_program, 3, torch.int32, torch.int32, 1),
    ("five_sorts", _five_sorts, 1, torch.int32, torch.int32, 1),
    ("half_of_the_columns", _half_of_the_columns, 1, torch.int32, torch.int32, 1),
    ("columns_that_follow_the_row", _columns_that_follow_the_row, 1, torch.int32, torch.int32, 1),
    ("columns_scaled_by_the_row", _columns_scaled_by_the_row, 1, torch.int32, torch.int32, 2),
    ("float_to_integer", _plain, 1, torch.float32, torch.int32, 1),
    ("integer_to_bfloat16", _plain, 1, torch.int32, torch.bfloat16, 1),
    ("in_its_own_argument", _in_its_own_argument, 1, torch.int32, torch.int32, 1),
    # Rows one cell closer than they are wide: the last cell of a row is the next row's first.
    ("rows_that_share_a_loaded_cell", _rows_that_share_a_loaded_cell, 1, torch.int32, torch.int32, 1),
    ("rows_that_share_a_stored_cell", _rows_that_share_a_stored_cell, 1, torch.int32, torch.int32, 1),
    # The mask leaves no cell out, so the converted value is given to none; a conversion
    # that is not the stored one is refused wherever it stands.
    ("a_converted_value_for_no_cell", _a_converted_value_for_no_cell, 1, torch.float32, torch.float32, 1),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", LARGER)
@pytest.mark.parametrize("case", MORE_THAN_THESE, ids=[case[0] for case in MORE_THAN_THESE])
def test_a_larger_kernel_that_is_more_than_these_forms_is_refused_with_the_reason(case, rows, cols, descending, device):
    _, kernel, grid, in_dtype, out_dtype, cells = case
    total = rows * cols
    x = _input(cells * total + 8, in_dtype).to(device)
    z = torch.full((grid * total + 8,), POISON, dtype=out_dtype, device=device)
    held = x.clone()
    with pytest.raises(MetalNonRecoverableError) as caught:
        kernel[(grid,)](x, z, rows, cols, descending)
    assert TOO_LARGE in str(caught.value)
    assert torch.all(z == POISON) and torch.equal(x, held)


def _values(kind, count, special):
    """``count`` values of ``kind`` with the ``special`` ones among them; no zero of the other sign, no NaN."""
    rng = np.random.default_rng(count + np.dtype(kind).itemsize)
    if np.issubdtype(kind, np.floating):
        values = (rng.integers(-200, 200, count) * 0.25).astype(kind)
    else:
        info = np.iinfo(kind)
        values = rng.integers(max(info.min, -100), min(info.max, 100), count, endpoint=True).astype(kind)
    for position, value in enumerate(special):
        values[3 + position :: 11 + 2 * position] = value
    return values


# input type, output type, values the conversion rounds or reads by their sign
CONVERSIONS = [
    (np.int32, np.float32, [16777217, 16777219, -16777217, 33554434, 2**31 - 1, -(2**31)]),
    (np.int64, np.float32, [2**40 + 2**16, 2**40 + 3 * 2**16, 2**53 + 1, -(2**62) - 2**38, 16777217]),
    (np.uint32, np.float32, [2**32 - 1, 2**31, 2**31 + 129, 16777217]),
    (np.int32, np.float16, [2049, 2051, 65520, -100000, 4097]),
    # 65505 to 65519 round to the largest float16; Metal's own long to half gives infinity there.
    (np.int64, np.float16, [65505, 65519, -65512, 65520, 65504, 2049, 2**40]),
    (np.int8, np.float32, [-128, 127, -1]),
    (np.uint8, np.float16, [255, 128, 129]),
    (np.int32, np.int64, [-1, 2**31 - 1, -(2**31)]),
    (np.int8, np.int32, [-128, -1, 127]),
    (np.uint8, np.int32, [255, 128]),
    (np.int16, np.int64, [-32768, 32767, -1]),
    (np.float16, np.float32, [65504.0, -65504.0, 0.333]),
    (np.float32, np.float16, [2049.0, 2051.0, 65520.0, -100000.0, 0.1]),
    (np.int64, np.int32, [2**40 + 5, -(2**40) - 5, 2**31]),
    (np.int32, np.int8, [300, -300, 128]),
]


def _name(kind):
    return np.dtype(kind).name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", [(16, 16), (64, 32), (4, 512)])
@pytest.mark.parametrize("case", CONVERSIONS, ids=[f"{_name(case[0])}-{_name(case[1])}" for case in CONVERSIONS])
def test_a_converting_store_converts_as_the_source_does(case, rows, cols, descending, device):
    in_kind, out_kind, special = case
    total = rows * cols
    x = _values(in_kind, total, special)
    z = torch.from_numpy(np.full(total + 8, POISON, dtype=out_kind)).to(device)
    launch = _plain[(1,)](torch.from_numpy(x).to(device), z, rows, cols, descending)
    # A store that keeps the low bits of an integer is the plain form.
    narrowed = np.issubdtype(out_kind, np.integer) and np.dtype(out_kind).itemsize < np.dtype(in_kind).itemsize
    assert _is_the_template(launch) == (narrowed or total > 1024)
    ordered = np.sort(x.reshape(rows, cols), axis=1)
    with np.errstate(over="ignore"):
        want = (ordered[:, ::-1] if descending else ordered).astype(out_kind)
    got = z.cpu().numpy()
    assert np.array_equal(got[:total].reshape(rows, cols).view(np.uint8), want.view(np.uint8))
    assert np.all(got[total:] == POISON)


# input type, the type the sorted values are read as, output type
READ_AS = [
    (np.int32, tl.uint32, np.uint32, np.float32),
    (np.uint32, tl.int32, np.int32, np.float32),
    (np.int32, tl.uint32, np.uint32, np.int64),
    (np.uint32, tl.int32, np.int32, np.int64),
    (np.int8, tl.uint8, np.uint8, np.int32),
    (np.uint8, tl.int8, np.int8, np.float32),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", [(16, 16), (64, 32), (512, 8)])
@pytest.mark.parametrize("case", READ_AS, ids=[f"{_name(c[0])}-as-{_name(c[2])}-{_name(c[3])}" for c in READ_AS])
def test_the_conversion_reads_the_sorted_value_by_its_own_sign_not_the_comparison_s(
    case, rows, cols, descending, device
):
    # Sorted as the input's type compares, then widened or made a float as the other sign.
    in_kind, read_as, read_kind, out_kind = case
    total = rows * cols
    info = np.iinfo(in_kind)
    x = _values(in_kind, total, [info.max, info.min, info.max // 2 + 1, info.max - 5])
    if info.min == 0:
        x[1::3] = info.max - x[1::3]  # a third of the values above the signed range
    z = torch.from_numpy(np.full(total + 8, POISON, dtype=out_kind)).to(device)
    launch = _stored_as[(1,)](torch.from_numpy(x).to(device), z, rows, cols, descending, read_as)
    assert _is_the_template(launch) == (total > 1024)
    ordered = np.sort(x.reshape(rows, cols), axis=1)
    want = (ordered[:, ::-1] if descending else ordered).astype(read_kind).astype(out_kind)
    got = z.cpu().numpy()
    assert np.array_equal(got[:total].reshape(rows, cols).view(np.uint8), want.view(np.uint8))
    assert np.all(got[total:] == POISON)


UNSIGNED = [torch.uint8, torch.uint16, torch.uint32]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", UNSIGNED, ids=["u8", "u16", "u32"])
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", [(4, 8), (16, 16)])
def test_an_unsigned_sort_compares_unsigned(device, dtype, descending, rows, cols):
    total = rows * cols
    top = torch.iinfo(dtype).max
    order = torch.randperm(total, generator=torch.Generator().manual_seed(total))
    # Half of the values are above the signed range of the type.
    x = torch.where(order % 2 == 0, order, top - order).to(dtype)
    z = torch.zeros(total, dtype=dtype, device=device)
    launch = _plain[(1,)](x.to(device), z, rows, cols, descending)
    assert _is_the_template(launch)
    want = np.sort(x.numpy().reshape(rows, cols), axis=1)
    assert np.array_equal(z.cpu().numpy().reshape(rows, cols), want[:, ::-1] if descending else want)


def _network(tile, descending):
    """``triton.language.sort`` along the rows of ``tile``, one compare-and-swap step at a time."""
    rows, cols = tile.shape
    dims, stages = (rows * cols).bit_length() - 1, cols.bit_length() - 1
    held = tile.reshape([2] * dims).copy()
    coordinate = np.indices([2] * dims)
    for stage in range(1, stages + 1):
        direction = coordinate[dims - stage - 1] if stage < stages else descending
        for swap in range(stage - 1, -1, -1):
            axis = dims - 1 - swap
            partner = np.flip(held, axis)
            with np.errstate(invalid="ignore"):
                take = (held > partner) != ((coordinate[axis] ^ direction) != 0)
            held = np.where(take, partner, held)
    return held.reshape(rows, cols)


def _with_pairs_neither_greater(kind, count):
    values = np.random.default_rng(count).permutation(count).astype(np.float32) - count / 3
    pick = np.random.default_rng(count + 1).permutation(count)
    if kind == "nan":
        values[pick[: count // 5]] = np.nan
    else:
        values[pick[: count // 2]] = 0.0
        values[pick[count // 2 : 3 * count // 4]] = -0.0
    return values


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["nan", "zeros"])
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", [(16, 16), (64, 32), (4, 512)])
@pytest.mark.parametrize(
    "kernel", [_reversed_columns, _pair_swapped_columns, _shifted_load, _sorted_twice, _four_sorts]
)
def test_the_order_of_the_loaded_columns_and_of_the_sorts_is_the_source_s(kernel, rows, cols, descending, kind, device):
    # Where neither of a pair is greater, the network's result follows the order its
    # values arrive in and the direction of every sort before the last.
    total = rows * cols
    x = _with_pairs_neither_greater(kind, total + 8)
    z = torch.full((total + 8,), float(POISON), device=device)
    launch = kernel[(1,)](torch.from_numpy(x).to(device), z, rows, cols, descending)
    assert _is_the_template(launch) == (total > 1024)
    tile = x[:total].reshape(rows, cols)
    if kernel is _reversed_columns:
        want = _network(tile[:, ::-1], descending)
    elif kernel is _pair_swapped_columns:
        want = _network(tile[:, np.arange(cols) ^ 1], descending)
    elif kernel is _shifted_load:
        want = _network(x[3 : 3 + total].reshape(rows, cols), descending)
    elif kernel is _sorted_twice:
        want = _network(_network(tile, descending), 1 - descending)
    else:
        want = _network(_network(tile, descending), 1 - descending)
        want = _network(_network(want, 1 - descending), descending)
    got = z.cpu().numpy()
    assert np.array_equal(got[:total].view(np.uint32), want.reshape(-1).view(np.uint32))
    assert np.all(got[total:] == POISON)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["nan", "zeros"])
@pytest.mark.parametrize("descending", [0, 1])
@pytest.mark.parametrize("rows,cols", [(4, 8), (16, 16), (8, 64)])
@pytest.mark.parametrize("kernel,first,start", [(_plain, 0, 0), (_offsets, 5, 3)], ids=["template", "ordinary"])
def test_where_neither_of_a_pair_is_greater_the_result_is_the_source_s(
    kernel, first, start, rows, cols, descending, kind, device
):
    total = rows * cols
    x = _with_pairs_neither_greater(kind, total + 8)
    z = torch.full((first + total + 8,), float(POISON), device=device)
    launch = kernel[(1,)](torch.from_numpy(x).to(device), z, rows, cols, descending)
    assert _is_the_template(launch) == (kernel is _plain)
    want = _network(x[start : start + total].reshape(rows, cols), descending).reshape(-1)
    got = z.cpu().numpy()[first : first + total]
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
