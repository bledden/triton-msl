"""The row-wise sort template stands only for a kernel that is exactly a row-wise sort.

The template is a whole kernel: it loads row ``lid`` of the input, sorts it and stores it,
and nothing else of the source reaches the shader. It used to be chosen from a signature (a
hypercube reshape and an xor reduction) that a flip of a 2xN tile, a sort whose result is
changed before the store, masked accesses and offset addresses all fit, and those kernels
then ran as a plain sort. The template is now chosen by a proof of the whole kernel; every
other kernel goes to the ordinary lowering, which computes it or refuses it.

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
def _flip_two_rows(X, Z, C: tl.constexpr):
    # A flip of a 2xC tile along its rows has the sort's reshape and its xor reduction.
    o = tl.arange(0, 2)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Z + o, tl.flip(tl.load(X + o), 0))


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
    ("sorted_twice", _sorted_twice, 1, 0, lambda x, r, c, d: _ordered(x[: r * c].reshape(r, c), 1 - d).reshape(-1)),
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
@pytest.mark.parametrize("cols", [4, 8, 16, 64, 512])
def test_a_flip_of_two_rows_is_not_a_sort(device, dtype, cols):
    x = _input(2 * cols, dtype)
    z = torch.full((2 * cols + 8,), POISON, dtype=dtype, device=device)
    launch = _launch(_flip_two_rows, (1,), x.to(device), z, cols)
    if launch is None:
        return
    assert not _is_the_template(launch)
    assert torch.equal(z[: 2 * cols].cpu().reshape(2, cols), x.reshape(2, cols).flip(0))
    assert torch.all(z[2 * cols :] == POISON)


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
