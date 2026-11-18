"""A load over a block narrower than the kernel stays inside the elements the source loads.

A kernel runs as many lanes as its widest block. A range of 64 beside a block of 128 is
written from the lane number, so on lanes 64 to 127 the index is one the source never
forms. A load through it used to run on those lanes too and read up to a block's worth of
elements past the ones the source loads, past the end of the tensor when the tensor ends
there. The values were not used, so no result shows it: these tests read the shader text
for the condition and check the values beside it.
"""

import re

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
POISON = -777.0


@triton.jit
def _sum_narrow(X, Y, NARROW: tl.constexpr, WIDE: tl.constexpr):
    a = tl.load(X + tl.arange(0, NARROW))
    w = tl.arange(0, WIDE)
    tl.store(Y + w, tl.load(Y + w) + tl.sum(a))


@triton.jit
def _copy_narrow(X, Y, Z, NARROW: tl.constexpr, WIDE: tl.constexpr):
    n = tl.arange(0, NARROW)
    w = tl.arange(0, WIDE)
    tl.store(Z + n, tl.load(X + n) * 2)
    tl.store(Y + w, tl.load(Y + w) + 1)


@triton.jit
def _copy_upper(X, Y, Z, NARROW: tl.constexpr, WIDE: tl.constexpr):
    # The range starts at NARROW: the index is the lane number plus NARROW.
    n = tl.arange(NARROW, 2 * NARROW)
    w = tl.arange(0, WIDE)
    tl.store(Z + n, tl.load(X + n) * 2)
    tl.store(Y + w, tl.load(Y + w) + 1)


@triton.jit
def _copy_masked(X, Y, Z, COUNT, NARROW: tl.constexpr, WIDE: tl.constexpr):
    n = tl.arange(0, NARROW)
    w = tl.arange(0, WIDE)
    tl.store(Z + n, tl.load(X + n, mask=n < COUNT, other=5.0) * 2)
    tl.store(Y + w, tl.load(Y + w) + 1)


@triton.jit
def _gather_narrow(X, INDEX, Y, Z, NARROW: tl.constexpr, WIDE: tl.constexpr):
    # The address of the second load is computed from the first: neither runs past the range.
    n = tl.arange(0, NARROW)
    w = tl.arange(0, WIDE)
    tl.store(Z + n, tl.load(X + tl.load(INDEX + n)))
    tl.store(Y + w, tl.load(Y + w) + 1)


@triton.jit
def _tile_narrow(X, Y, Z, R: tl.constexpr, C: tl.constexpr, WIDE: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    w = tl.arange(0, WIDE)
    tl.store(Z + r * C + c, tl.load(X + r * C + c) + 1)
    tl.store(Y + w, tl.load(Y + w) + 1)


@triton.jit
def _one_for_all(X, Y, WIDE: tl.constexpr):
    # A range of one element stands for every lane of the wide block.
    one = tl.load(X + tl.arange(0, 1))
    w = tl.arange(0, WIDE)
    tl.store(Y + w, tl.load(Y + w) + one)


def _at_the_end(values, device):
    """``values`` as the last elements of a storage, 16 bytes of poison before them."""
    storage = torch.full((values.numel() + 4,), POISON, device=device)
    storage[4:] = values.to(device)
    return storage[4:]


def _loads_of(msl, name):
    return [line.strip() for line in msl.splitlines() if re.search(rf"= .*\b{name}\[", line) and "if (" not in line]


def _held(line, bound):
    """The load on ``line`` is made only where an index is below ``bound``."""
    return re.search(rf"= \(\(\w+ < {bound}u\)(?: && [^?]*)?\) \? static_cast<\w+>\(", line) is not None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("narrow,wide", [(64, 128), (8, 512), (2, 1024), (64, 2048)])
def test_narrow_load_feeding_a_store_stays_in_its_range(device, narrow, wide):
    x = _at_the_end(torch.arange(narrow, dtype=torch.float32) + 1, device)
    y = torch.arange(wide, dtype=torch.float32, device=device)
    z = torch.full((wide,), POISON, device=device)
    launch = _copy_narrow[(1,)](x, y, z, narrow, wide)
    assert torch.equal(z[:narrow].cpu(), (torch.arange(narrow, dtype=torch.float32) + 1) * 2)
    assert torch.all(z[narrow:] == POISON)
    assert torch.equal(y.cpu(), torch.arange(wide, dtype=torch.float32) + 1)
    loads = _loads_of(launch.asm["msl"], "X")
    assert loads and all(_held(line, narrow) for line in loads), loads
    wide_loads = _loads_of(launch.asm["msl"], "Y")
    assert wide_loads and not any(" ? " in line for line in wide_loads), wide_loads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("narrow,wide", [(64, 128), (8, 512), (32, 1024)])
def test_narrow_load_feeding_a_sum_stays_in_its_range(device, narrow, wide):
    x = _at_the_end(torch.arange(narrow, dtype=torch.float32) + 1, device)
    y = torch.arange(wide, dtype=torch.float32, device=device)
    launch = _sum_narrow[(1,)](x, y, narrow, wide)
    assert torch.equal(y.cpu(), torch.arange(wide, dtype=torch.float32) + narrow * (narrow + 1) / 2)
    loads = _loads_of(launch.asm["msl"], "X")
    assert loads and all(_held(line, narrow) for line in loads), loads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("narrow,wide", [(64, 256), (8, 512)])
def test_narrow_load_through_a_range_that_does_not_start_at_zero(device, narrow, wide):
    x = _at_the_end(torch.arange(2 * narrow, dtype=torch.float32) + 1, device)
    y = torch.zeros(wide, device=device)
    z = torch.full((wide,), POISON, device=device)
    launch = _copy_upper[(1,)](x, y, z, narrow, wide)
    assert torch.equal(z[narrow : 2 * narrow].cpu(), (torch.arange(narrow, 2 * narrow, dtype=torch.float32) + 1) * 2)
    assert torch.all(z[:narrow] == POISON) and torch.all(z[2 * narrow :] == POISON)
    assert torch.all(y == 1)
    loads = _loads_of(launch.asm["msl"], "X")
    assert loads and all(_held(line, 2 * narrow) for line in loads), loads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [0, 5, 64])
def test_masked_narrow_load_keeps_its_mask_and_its_other_value(device, count):
    narrow, wide = 64, 128
    x = _at_the_end(torch.arange(narrow, dtype=torch.float32) + 1, device)
    y = torch.zeros(wide, device=device)
    z = torch.full((wide,), POISON, device=device)
    launch = _copy_masked[(1,)](x, y, z, count, narrow, wide)
    expected = torch.full((narrow,), 10.0)
    expected[:count] = (torch.arange(count, dtype=torch.float32) + 1) * 2
    assert torch.equal(z[:narrow].cpu(), expected)
    assert torch.all(z[narrow:] == POISON)
    loads = _loads_of(launch.asm["msl"], "X")
    assert loads and all(_held(line, narrow) for line in loads), loads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("narrow,wide", [(64, 128), (16, 1024)])
def test_load_through_a_loaded_narrow_index_stays_in_its_range(device, narrow, wide):
    table = torch.arange(3 * narrow, dtype=torch.float32) * 3
    index = torch.randperm(3 * narrow, generator=torch.Generator().manual_seed(narrow))[:narrow].to(torch.int32)
    x = table.to(device)
    i = torch.full((narrow + 4,), 10**6, dtype=torch.int32, device=device)
    i[4:] = index.to(device)
    y = torch.zeros(wide, device=device)
    z = torch.full((wide,), POISON, device=device)
    launch = _gather_narrow[(1,)](x, i[4:], y, z, narrow, wide)
    assert torch.equal(z[:narrow].cpu(), table[index.long()])
    assert torch.all(z[narrow:] == POISON)
    for name in ("X", "INDEX"):
        loads = _loads_of(launch.asm["msl"], name)
        assert loads and all(_held(line, narrow) for line in loads), (name, loads)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,cols,wide", [(4, 8, 128), (2, 16, 64), (8, 4, 1024), (4, 4, 2048)])
def test_narrow_tile_load_stays_in_its_rows(device, rows, cols, wide):
    x = _at_the_end(torch.arange(rows * cols, dtype=torch.float32), device)
    y = torch.zeros(wide, device=device)
    z = torch.full((wide,), POISON, device=device)
    launch = _tile_narrow[(1,)](x, y, z, rows, cols, wide)
    assert torch.equal(z[: rows * cols].cpu(), torch.arange(rows * cols, dtype=torch.float32) + 1)
    assert torch.all(z[rows * cols :] == POISON)
    assert torch.all(y == 1)
    loads = _loads_of(launch.asm["msl"], "X")
    assert loads and all(_held(line, rows) for line in loads), loads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wide", [128, 2048])
def test_a_range_of_one_element_is_loaded_on_every_lane(device, wide):
    x = _at_the_end(torch.tensor([7.0]), device)
    y = torch.arange(wide, dtype=torch.float32, device=device)
    _one_for_all[(1,)](x, y, wide)
    assert torch.equal(y.cpu(), torch.arange(wide, dtype=torch.float32) + 7)
