"""A store or an atomic over a block narrower than the kernel's widest is performed once for each element.

A kernel runs one lane for each element of its widest block. A block with fewer elements is
held by more than one lane, so a ``tl.store`` or ``tl.atomic_*`` over it has to be given to
exactly one lane for each element: an atomic add performed by every lane that holds the
element adds too often, and a store placed by the wrong lane writes another element's value
or writes outside the block.

The kernels below each do one narrow operation beside a store of the kernel's widest block.
Every buffer is longer than the cells the kernel should write and starts filled with a
sentinel or with known values; the whole buffer is compared with a value worked out on the
host with numpy, so a cell written twice, written with a neighbour's value or written past
the end shows.

The cases are grouped as:

* a narrow two-dimensional store in a wider two-dimensional kernel (a compact tile, one
  column, one row, values that come from a row total, a column total or a loaded vector);
* a narrow one-dimensional store beside a wider block;
* atomic add, max, exchange and compare-and-swap over a narrow block;
* an atomic on one cell whose returned value the widest block uses;
* narrow updates in place, which a launch accepts, and a narrow copy onto memory its input
  also reaches one cell along, which a launch refuses;
* the forms where the lane that holds each element is not proved, which are refused when
  the kernel is compiled and leave every buffer as it was.
"""

import re

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
SENTINEL = -77
PAD = 37

# (rows, columns of the kernel's block, rows, columns of the narrow tile). The last five run 2048 or 4096 lanes.
TILES = [
    (8, 16, 4, 16),
    (8, 16, 8, 8),
    (8, 16, 4, 4),
    (8, 16, 2, 8),
    (16, 8, 4, 2),
    (16, 8, 8, 4),
    (4, 64, 2, 16),
    (32, 32, 8, 16),
    (32, 32, 16, 4),
    (32, 32, 4, 32),
    (32, 32, 32, 8),
    (32, 64, 8, 16),
    (64, 64, 16, 16),
    (64, 64, 2, 64),
    (64, 64, 64, 8),
    (64, 64, 8, 32),
]
# The tiles of kernels that run no more than 1024 lanes.
SMALL_TILES = [tile for tile in TILES if tile[0] * tile[1] <= 1024]
# (rows, columns of the kernel's block, cells of the narrow column or row).
COLUMNS = [(8, 16, 4), (8, 16, 2), (16, 8, 4), (16, 8, 8), (4, 8, 2), (32, 32, 8), (32, 32, 16), (2, 64, 1)]
LARGE_COLUMNS = [(64, 64, 16), (64, 64, 32), (32, 64, 8)]
ROWS = [(8, 16, 4), (8, 16, 8), (16, 8, 2), (16, 8, 4), (4, 8, 2), (32, 32, 8), (32, 32, 16), (2, 64, 16)]
LARGE_ROWS = [(64, 64, 16), (64, 64, 32), (32, 64, 8)]
# (cells of the kernel's block, cells of the narrow vector).
VECTORS = [
    (64, 16),
    (128, 64),
    (128, 8),
    (1024, 64),
    (1024, 512),
    (2048, 64),
    (2048, 1024),
    (4096, 2048),
    (4096, 64),
]
# (rows, columns of the kernel's block, cells of the narrow vector).
VECTORS_IN_A_TILE = [(8, 16, 8), (8, 16, 16), (8, 16, 64), (16, 8, 32), (32, 32, 64), (32, 32, 512)]
LARGE_VECTORS_IN_A_TILE = [(64, 64, 256), (64, 64, 2048)]
# (cells of the kernel's one-dimensional block, rows, columns of the narrow tile).
TILES_IN_A_VECTOR = [
    (64, 4, 8),
    (64, 8, 4),
    (256, 4, 8),
    (256, 8, 16),
    (256, 16, 4),
    (1024, 8, 8),
    (1024, 16, 32),
    (2048, 16, 16),
    (4096, 16, 32),
    (4096, 64, 8),
    (4096, 2, 64),
]
# (rows, columns) of a kernel whose reduction leaves one value for each row or column.
BLOCKS = [(4, 8), (8, 16), (16, 8), (2, 64), (32, 32), (8, 8), (16, 16)]
# (planes, rows, columns) of a three-dimensional block.
CUBES = [(2, 4, 8), (4, 4, 4), (4, 8, 2), (8, 8, 8), (2, 16, 32), (16, 8, 8), (16, 16, 16)]
CELL_BLOCKS = [1, 2, 64, 1024, 2048, 4096]


# --- A narrow two-dimensional store in a wider two-dimensional kernel ------------------------------------------------


@triton.jit
def _tile_copy(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * W + c, tl.load(X + r * W + c) * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_copy_plain_columns(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * W + c, tl.load(X + r * W + c) * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_ramp(O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * W + c, (r * SC + c + 1).to(tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_flipped(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + (SR - 1 - r) * W + c, tl.load(X + r * W + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _square_one_range(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, S: tl.constexpr):
    i = tl.arange(0, S)
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + i[:, None] * W + i[None, :])
    tl.store(O + i[:, None] * W + i[None, :], x * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_selected_constants(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * W + c)
    tl.store(O + r * W + c, tl.where(x > 0, tl.full([SR, SC], 7, tl.int32), tl.zeros([SR, SC], tl.int32)))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_carried_through_a_loop(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    acc = tl.zeros([SR, SC], tl.int32)
    for k in range(3):
        acc += tl.load(X + r * W + c + k)
    tl.store(O + r * W + c, acc)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_running_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * W + c, tl.cumsum(tl.load(X + r * W + c), 1))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_in_a_vector(X, O, Q, BLOCK: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    i = tl.arange(0, BLOCK)
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    tl.store(Q + i, i.to(tl.int32))
    tl.store(O + r * SC + c, tl.load(X + r * SC + c) * 2 + 1)


@triton.jit
def _tile_wider_than_the_block(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * SC + c, tl.load(X + r * SC + c) * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_in_a_cube(X, O, Q, BA: tl.constexpr, BB: tl.constexpr, BC: tl.constexpr):
    a = tl.arange(0, BA)[:, None, None]
    b = tl.arange(0, BB)[None, :, None]
    c = tl.arange(0, BC)[None, None, :]
    tl.store(Q + a * (BB * BC) + b * BC + c, (a * (BB * BC) + b * BC + c).to(tl.int32))
    r = tl.arange(0, BB)[:, None]
    k = tl.arange(0, BC)[None, :]
    tl.store(O + r * BC + k, tl.load(X + r * BC + k) * 2 + 1)


@triton.jit
def _column_copy(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r, tl.load(X + r) * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _column_ramp(O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r, (r * 3 + 1).to(tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_copy(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + c, tl.load(X + c) * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_ramp(O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + c, (c * 3 + 1).to(tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_row_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * W + c)
    tl.store(O + r, tl.sum(x, 1)[:, None])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_column_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * W + c)
    tl.store(O + c, tl.sum(x, 0)[None, :])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _leading_row_totals(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * BC + C)
    tl.store(O + r, tl.sum(x, 1)[:, None])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _leading_column_totals(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + R * BC + c)
    tl.store(O + c, tl.sum(x, 0)[None, :])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_row_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    totals = tl.sum(tl.load(X + r * W + c), 1)[:, None]
    tl.store(O + r * W + c, totals + c)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_column_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    totals = tl.sum(tl.load(X + r * W + c), 0)[None, :]
    tl.store(O + r * W + c, totals + r)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_wide_row_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * W + C)
    tl.store(O + r * W + c, tl.sum(x, 1)[:, None] + c)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _column_from_vector(V, O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.load(V + tl.arange(0, SR))
    tl.store(O + r, (v * 2)[:, None])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_from_vector(V, O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.load(V + tl.arange(0, SC))
    tl.store(O + c, (v * 2)[None, :])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_row_vector(V, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.load(V + tl.arange(0, SC))
    tl.store(O + r * W + c, v[None, :] + r * 100)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_column_vector(V, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.load(V + tl.arange(0, SR))
    tl.store(O + r * W + c, v[:, None] + c * 100)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


# --- A narrow one-dimensional store beside a wider block -------------------------------------------------------------


@triton.jit
def _vector_copy(X, O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.store(O + j, tl.load(X + j) * 2 + 1)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_copy_in_a_tile(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + j, tl.load(X + j) * 2 + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _vector_folded(X, O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr, HALF: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.store(O + (j % HALF), tl.load(X + j), mask=j >= HALF)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_folded_in_a_tile(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SMALL: tl.constexpr, HALF: tl.constexpr):
    j = tl.arange(0, SMALL)
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + (j % HALF), tl.load(X + j), mask=j >= HALF)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _vector_folded_in_a_loop(O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr, HALF: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    for k in range(3):
        tl.store(O + (j % HALF) + k * HALF, (j + 1 + k).to(tl.int32), mask=j >= HALF)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_scattered(IDX, X, O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.store(O + tl.load(IDX + j), tl.load(X + j))
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_strided(X, O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.store(O + j * 2, tl.load(X + j))
    tl.store(Q + w, w.to(tl.int32))


# --- Atomics over a narrow block -------------------------------------------------------------------------------------


@triton.jit
def _tile_add(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(O + r * W + c, tl.load(X + r * W + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_add_ramp(O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(O + r * W + c, (r * SC + c + 1).to(tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_max(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_max(O + r * W + c, tl.load(X + r * W + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_exchange(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_xchg(O + r * W + c, tl.load(X + r * W + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_compare_and_swap(X, E, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_cas(O + r * W + c, tl.load(E + r * W + c), tl.load(X + r * W + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_add_rows_share(O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(tl.broadcast_to(O + c, [SR, SC]), tl.full([SR, SC], 1, tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_add_columns_share(O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(tl.broadcast_to(O + r, [SR, SC]), tl.full([SR, SC], 1, tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_add_one_cell(O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(O + tl.zeros([SR, SC], tl.int32), tl.full([SR, SC], 1, tl.int32))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_add_in_a_vector(X, O, Q, BLOCK: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    i = tl.arange(0, BLOCK)
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    tl.store(Q + i, i.to(tl.int32))
    tl.atomic_add(O + r * SC + c, tl.load(X + r * SC + c))


@triton.jit
def _column_add(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(O + r, tl.load(X + r))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _column_max(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_max(O + r, tl.load(X + r))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_add(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(O + c, tl.load(X + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_exchange(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_xchg(O + c, tl.load(X + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_totals_added_as_a_column(X, O, N, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * BC + c)
    tl.atomic_add(O + r, tl.sum(x, 1)[:, None], mask=r < N)


@triton.jit
def _column_totals_added_as_a_row(X, O, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * BC + c)
    tl.atomic_add(O + c, tl.sum(x, 0)[None, :])


@triton.jit
def _row_totals_exchanged_old_stored(X, O, P, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.arange(0, BR)
    c = tl.arange(0, BC)
    x = tl.load(X + r[:, None] * BC + c[None, :])
    old = tl.atomic_xchg(O + r, tl.sum(x, 1))
    tl.store(P + r, old)


@triton.jit
def _vector_added_as_a_column(V, O, Q, BR: tl.constexpr, BC: tl.constexpr):
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.load(V + tl.arange(0, BR))
    tl.atomic_add(O + R, (v * 2)[:, None])
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _vector_add(X, O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.atomic_add(O + j, tl.load(X + j))
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_max(X, O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.atomic_max(O + j, tl.load(X + j))
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_add_old_stored(X, O, P, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    old = tl.atomic_add(O + j, tl.load(X + j))
    tl.store(P + j, old)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_exchange_old_stored(X, O, P, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    old = tl.atomic_xchg(O + j, tl.load(X + j))
    tl.store(P + j, old)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_compare_and_swap_old_stored(X, E, O, P, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    old = tl.atomic_cas(O + j, tl.load(E + j), tl.load(X + j))
    tl.store(P + j, old)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _vector_add_folded_in_a_tile(X, O, Q, BR: tl.constexpr, BC: tl.constexpr, SMALL: tl.constexpr, HALF: tl.constexpr):
    j = tl.arange(0, SMALL)
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.atomic_add(O + (j % HALF), tl.load(X + j))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _histogram(IDX, H, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.atomic_add(H + tl.load(IDX + j), tl.full([SMALL], 1, tl.int32))
    tl.store(Q + w, w.to(tl.int32))


# --- An atomic on one cell whose returned value the widest block uses ------------------------------------------------


@triton.jit
def scalar_exchange_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    old = tl.atomic_xchg(S, 5)
    tl.store(Q + w, w.to(tl.int32) + old)


@triton.jit
def scalar_compare_and_swap_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    old = tl.atomic_cas(S, 3, 4)
    tl.store(Q + w, w.to(tl.int32) + old)


@triton.jit
def scalar_max_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    old = tl.atomic_max(S, 9)
    tl.store(Q + w, w.to(tl.int32) + old)


@triton.jit
def scalar_min_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    old = tl.atomic_min(S, -9)
    tl.store(Q + w, w.to(tl.int32) + old)


@triton.jit
def scalar_add_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    old = tl.atomic_add(S, 5)
    tl.store(Q + w, w.to(tl.int32) + old)


@triton.jit
def scalar_max_twice_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    a = tl.atomic_max(S, 7)
    b = tl.atomic_max(S, 11)
    tl.store(Q + w, w + a * 100 + b)


@triton.jit
def scalar_exchange_in_a_branch_used(S, Q, N, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    if N > 2:
        old = tl.atomic_xchg(S, 5)
    else:
        old = tl.atomic_xchg(S, 8) + 100
    tl.store(Q + w, w + old)


@triton.jit
def scalar_exchange_in_a_loop_used(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int32)
    for k in range(3):
        acc += tl.atomic_xchg(S, 5 + k)
    tl.store(Q + w, acc)


@triton.jit
def _scalar_compare_and_swap_chain(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    tl.atomic_cas(S, 3, 4)
    tl.atomic_cas(S, 4, 6)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _scalar_compare_and_swap_in_a_loop(S, Q, BLOCK: tl.constexpr):
    w = tl.arange(0, BLOCK)
    for t in range(3):
        tl.atomic_cas(S, 3 + t, 4 + t)
    tl.store(Q + w, w)


# --- Narrow updates in place -----------------------------------------------------------------------------------------


@triton.jit
def row_rmw_narrow(O, Q, BR: tl.constexpr, BC: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + c, tl.load(O + c) + c + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def column_rmw_narrow(O, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r, tl.load(O + r) + r + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def tile_update_narrow(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * W + c, tl.load(O + r * W + c) + tl.load(X + r * W + c))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def vector_rmw_narrow(O, Q, BLOCK: tl.constexpr, SMALL: tl.constexpr):
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.store(O + j, tl.load(O + j) * 2 + 1)
    tl.store(Q + w, w.to(tl.int32))


@triton.jit
def _two_column_updates(X, O, P, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * BC + C)
    tl.store(O + r, tl.load(O + r) + tl.sum(x, 1)[:, None])
    k = tl.arange(0, SR)[:, None]
    tl.store(P + k, tl.load(P + k) + 1)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


# --- Forms refused when the kernel is compiled -----------------------------------------------------------------------


@triton.jit
def _tile_reshaped_from_a_vector(V, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.load(V + tl.arange(0, SR * SC))
    tl.store(O + r * W + c, tl.reshape(v, [SR, SC]))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_transposed(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    rt = tl.arange(0, SC)[:, None]
    ct = tl.arange(0, SR)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + rt * W + ct, tl.trans(tl.load(X + r * W + c)))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_gathered(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * W + c)
    tl.store(O + r * W + c, tl.gather(x, tl.broadcast_to(SC - 1 - c, [SR, SC]), 1))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_vector_running_totals(
    X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr
):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.cumsum(tl.load(X + tl.arange(0, SC)), 0)
    tl.store(O + r * W + c, v[None, :] + r * 100)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_vector_less_running_totals(
    X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr
):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + tl.arange(0, SC))
    tl.store(O + r * W + c, (x * 3 - tl.cumsum(x, 0))[None, :] + r * 100)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_column_vector_running_totals(
    X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr
):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    v = tl.cumsum(tl.load(X + tl.arange(0, SR)), 0)
    tl.store(O + r * W + c, v[:, None] + c * 100)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _tile_from_row_running_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + r * W + c, tl.cumsum(tl.load(X + c), 1) + r * 100)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_running_totals(X, O, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    tl.store(O + c, tl.cumsum(tl.load(X + c), 1))
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _column_totals_stored_as_a_column(X, O, BR: tl.constexpr, BC: tl.constexpr):
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)
    x = tl.load(X + R * BC + C[None, :])
    tl.store(O + C[:, None], tl.sum(x, 0)[:, None])


@triton.jit
def _row_totals_stored_as_a_row(X, O, BR: tl.constexpr, BC: tl.constexpr):
    R = tl.arange(0, BR)
    C = tl.arange(0, BC)[None, :]
    x = tl.load(X + R[:, None] * BC + C)
    tl.store(O + R[None, :], tl.sum(x, 1)[None, :])


@triton.jit
def _tile_exchange_old_stored(X, O, P, Q, W, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    c = tl.arange(0, SC)[None, :]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    old = tl.atomic_xchg(O + r * W + c, tl.load(X + r * W + c))
    tl.store(P + r * W + c, old)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _column_exchange_old_stored(X, O, P, Q, BR: tl.constexpr, BC: tl.constexpr, SR: tl.constexpr):
    r = tl.arange(0, SR)[:, None]
    R = tl.arange(0, BR)[:, None]
    C = tl.arange(0, BC)[None, :]
    old = tl.atomic_xchg(O + r, tl.load(X + r))
    tl.store(P + r, old)
    tl.store(Q + R * BC + C, (R * BC + C).to(tl.int32))


@triton.jit
def _row_totals_exchanged_as_a_column_old_stored(X, O, P, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    c = tl.arange(0, BC)[None, :]
    x = tl.load(X + r * BC + c)
    old = tl.atomic_xchg(O + r, tl.sum(x, 1)[:, None])
    tl.store(P + r, old)


# --- Helpers ---------------------------------------------------------------------------------------------------------


def _values(count, seed):
    """``count`` different signed values, so a cell that holds another element's value is a wrong cell."""
    return (np.random.default_rng(seed).permutation(count) - count // 3).astype(np.int32)


def _blank(count):
    """A buffer of ``count`` sentinels."""
    return np.full(count, SENTINEL, dtype=np.int32)


def _ramp(count):
    """What the store of the kernel's widest block leaves: 0 to ``count`` - 1, then sentinels."""
    want = _blank(count + PAD)
    want[:count] = np.arange(count)
    return want


def _grid(cells, rows, pitch):
    """The first ``rows`` rows of ``cells``, laid ``pitch`` cells apart, as a view."""
    return cells[: rows * pitch].reshape(rows, pitch)


def _run(kernel, device, arrays, *scalars, times=1, **constants):
    """Launch ``kernel`` ``times`` times on fresh tensors of ``arrays`` and return what each tensor then holds."""
    tensors = [torch.from_numpy(array.copy()).to(device) for array in arrays]
    for _ in range(times):
        kernel[(1,)](*tensors, *scalars, **constants)
    return [tensor.cpu().numpy() for tensor in tensors]


def _same(got, want):
    np.testing.assert_array_equal(got, want)


def _refused(kernel, device, arrays, *scalars, match, **constants):
    """The launch is refused with a message that matches ``match``, and every buffer is as it was."""
    tensors = [torch.from_numpy(array.copy()).to(device) for array in arrays]
    with pytest.raises(MetalNonRecoverableError, match=match):
        kernel[(1,)](*tensors, *scalars, **constants)
    for tensor, array in zip(tensors, arrays):
        _same(tensor.cpu().numpy(), array)


def _tile_case(kernel, device, br, bc, sr, sc, seed, expected, **constants):
    """Run a kernel that stores an ``sr`` by ``sc`` tile at pitch ``2 * bc`` and compare every cell.

    ``expected(x, flat)`` returns the tile from the input, given as a grid at the same pitch and as it lies in memory.
    """
    w = 2 * bc
    x = _values(br * w + PAD, seed)
    arrays = [x, _blank(br * w + PAD), _blank(br * bc + PAD)]
    _, o, q = _run(kernel, device, arrays, w, BR=br, BC=bc, SR=sr, SC=sc, **constants)
    want = _blank(br * w + PAD)
    _grid(want, br, w)[:sr, :sc] = expected(_grid(x, br, w), x)
    _same(o, want)
    _same(q, _ramp(br * bc))


def _line_case(kernel, device, br, bc, count, seed, expected, **constants):
    """Run a kernel that stores ``count`` cells at the start of its output and compare every cell.

    ``expected(x, flat)`` returns the cells from the input, given as a ``br`` by ``bc`` grid and as it lies in memory.
    """
    x = _values(br * bc + PAD, seed)
    arrays = [x, _blank(br * bc + PAD), _blank(br * bc + PAD)]
    _, o, q = _run(kernel, device, arrays, BR=br, BC=bc, **constants)
    want = _blank(br * bc + PAD)
    want[:count] = expected(_grid(x, br, bc), x)
    _same(o, want)
    _same(q, _ramp(br * bc))


# --- A narrow two-dimensional store in a wider two-dimensional kernel ------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_copy_writes_each_cell_of_the_tile_once(br, bc, sr, sc, device):
    """A tile narrower than the block on one side or on both is copied cell for cell and nothing else is written."""
    _tile_case(_tile_copy, device, br, bc, sr, sc, 1, lambda x, flat: x[:sr, :sc] * 2 + 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_copy_with_a_plain_column_range(br, bc, sr, sc, device):
    """The same copy with the column range left one-dimensional and broadcast against the row range."""
    _tile_case(_tile_copy_plain_columns, device, br, bc, sr, sc, 2, lambda x, flat: x[:sr, :sc] * 2 + 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_of_values_made_from_its_own_ranges(br, bc, sr, sc, device):
    """A tile whose values are computed from its row and column numbers alone."""
    w = 2 * bc
    o, q = _run(_tile_ramp, device, [_blank(br * w + PAD), _blank(br * bc + PAD)], w, BR=br, BC=bc, SR=sr, SC=sc)
    want = _blank(br * w + PAD)
    _grid(want, br, w)[:sr, :sc] = np.arange(sr * sc).reshape(sr, sc) + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_stored_with_its_rows_reversed(br, bc, sr, sc, device):
    """The address counts rows down while the value counts them up; each row lands on its mirror row."""
    _tile_case(_tile_flipped, device, br, bc, sr, sc, 3, lambda x, flat: x[:sr, :sc][::-1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "br,bc,s", [(8, 16, 4), (8, 16, 8), (16, 8, 4), (4, 64, 2), (32, 32, 8), (32, 32, 16), (64, 64, 16), (64, 64, 32)]
)
def test_square_tile_whose_rows_and_columns_come_from_one_range(br, bc, s, device):
    """One ``tl.arange`` supplies both the row and the column index of a square tile."""
    w = 2 * bc
    x = _values(br * w + PAD, 4)
    _, o, q = _run(_square_one_range, device, [x, _blank(br * w + PAD), _blank(br * bc + PAD)], w, BR=br, BC=bc, S=s)
    want = _blank(br * w + PAD)
    _grid(want, br, w)[:s, :s] = _grid(x, br, w)[:s, :s] * 2 + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_of_constants_selected_by_a_loaded_tile(br, bc, sr, sc, device):
    """``tl.where`` between two constant tiles, chosen by the sign of the loaded tile."""
    _tile_case(_tile_selected_constants, device, br, bc, sr, sc, 5, lambda x, flat: np.where(x[:sr, :sc] > 0, 7, 0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_accumulated_through_a_loop(br, bc, sr, sc, device):
    """A tile carried through a loop that adds three shifted loads is stored once, after the loop."""

    def expected(x, flat):
        w = x.shape[1]
        return sum(flat[k : k + sr * w].reshape(sr, w)[:, :sc] for k in range(3))

    _tile_case(_tile_carried_through_a_loop, device, br, bc, sr, sc, 6, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_tile_of_running_totals_along_each_row(br, bc, sr, sc, device):
    """``tl.cumsum`` along the columns of the tile, stored at the tile's own cells."""
    _tile_case(_tile_running_totals, device, br, bc, sr, sc, 7, lambda x, flat: np.cumsum(x[:sr, :sc], 1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,sr,sc", TILES_IN_A_VECTOR)
def test_tile_stored_in_a_kernel_whose_widest_block_is_a_vector(block, sr, sc, device):
    """A two-dimensional tile beside a one-dimensional block with more elements."""
    x = _values(block + PAD, 8)
    _, o, q = _run(_tile_in_a_vector, device, [x, _blank(block + PAD), _blank(block + PAD)], BLOCK=block, SR=sr, SC=sc)
    want = _blank(block + PAD)
    want[: sr * sc] = x[: sr * sc] * 2 + 1
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "br,bc,sr,sc",
    [(8, 4, 2, 8), (16, 4, 2, 16), (32, 4, 4, 16), (8, 8, 2, 16), (4, 32, 16, 4), (64, 16, 4, 64), (256, 16, 8, 64)],
)
def test_tile_with_a_side_longer_than_the_same_side_of_the_block(br, bc, sr, sc, device):
    """A tile with fewer elements than the block but more columns, or more rows, than the block has."""
    cells = br * bc
    x = _values(cells + PAD, 9)
    arrays = [x, _blank(cells + PAD), _blank(cells + PAD)]
    _, o, q = _run(_tile_wider_than_the_block, device, arrays, BR=br, BC=bc, SR=sr, SC=sc)
    want = _blank(cells + PAD)
    want[: sr * sc] = x[: sr * sc] * 2 + 1
    _same(o, want)
    _same(q, _ramp(cells))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ba,bb,bc", CUBES)
def test_tile_stored_in_a_kernel_whose_widest_block_has_three_dimensions(ba, bb, bc, device):
    """A ``bb`` by ``bc`` tile, from ranges of its own, beside a ``ba`` by ``bb`` by ``bc`` block."""
    cells = ba * bb * bc
    x = _values(cells + PAD, 10)
    _, o, q = _run(_tile_in_a_cube, device, [x, _blank(cells + PAD), _blank(cells + PAD)], BA=ba, BB=bb, BC=bc)
    want = _blank(cells + PAD)
    want[: bb * bc] = x[: bb * bc] * 2 + 1
    _same(o, want)
    _same(q, _ramp(cells))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", COLUMNS + LARGE_COLUMNS)
def test_column_copy_writes_each_row_once(br, bc, sr, device):
    """A tile of one column: one cell for each of ``sr`` rows, fewer rows than the block has or as many."""
    _line_case(_column_copy, device, br, bc, sr, 11, lambda x, flat: flat[:sr] * 2 + 1, SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", COLUMNS + LARGE_COLUMNS)
def test_column_of_values_made_from_its_own_range(br, bc, sr, device):
    """A one-column tile whose values are computed from the row number alone."""
    o, q = _run(_column_ramp, device, [_blank(br * bc + PAD), _blank(br * bc + PAD)], BR=br, BC=bc, SR=sr)
    want = _blank(br * bc + PAD)
    want[:sr] = np.arange(sr) * 3 + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS + LARGE_ROWS)
def test_row_copy_writes_each_column_once(br, bc, sc, device):
    """A tile of one row: one cell for each of ``sc`` columns."""
    _line_case(_row_copy, device, br, bc, sc, 12, lambda x, flat: flat[:sc] * 2 + 1, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS + LARGE_ROWS)
def test_row_of_values_made_from_its_own_range(br, bc, sc, device):
    """A one-row tile whose values are computed from the column number alone."""
    o, q = _run(_row_ramp, device, [_blank(br * bc + PAD), _blank(br * bc + PAD)], BR=br, BC=bc, SC=sc)
    want = _blank(br * bc + PAD)
    want[:sc] = np.arange(sc) * 3 + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "br,bc,sr,sc",
    [
        (8, 16, 4, 16),
        (8, 16, 8, 8),
        (8, 16, 4, 4),
        (8, 16, 2, 8),
        (16, 8, 4, 2),
        (16, 8, 8, 4),
        (4, 64, 2, 16),
        (32, 32, 16, 4),
    ],
)
def test_row_totals_of_a_tile_stored_as_a_column(br, bc, sr, sc, device):
    """The total of each row of a narrow tile, stored one cell for each row."""
    w = 2 * bc
    x = _values(br * w + PAD, 13)
    arrays = [x, _blank(br * bc + PAD), _blank(br * bc + PAD)]
    _, o, q = _run(_tile_row_totals, device, arrays, w, BR=br, BC=bc, SR=sr, SC=sc)
    want = _blank(br * bc + PAD)
    want[:sr] = _grid(x, br, w)[:sr, :sc].sum(1)
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_column_totals_of_a_tile_stored_as_a_row(br, bc, sr, sc, device):
    """The total of each column of a narrow tile, stored one cell for each column."""
    w = 2 * bc
    x = _values(br * w + PAD, 14)
    arrays = [x, _blank(br * bc + PAD), _blank(br * bc + PAD)]
    _, o, q = _run(_tile_column_totals, device, arrays, w, BR=br, BC=bc, SR=sr, SC=sc)
    want = _blank(br * bc + PAD)
    want[:sc] = _grid(x, br, w)[:sr, :sc].sum(0)
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", [(8, 16, 4), (8, 16, 2), (16, 8, 4), (16, 8, 8), (4, 8, 2), (2, 64, 1)])
def test_totals_of_the_leading_rows_stored_as_a_column(br, bc, sr, device):
    """The first ``sr`` rows of the block, every column of them, totalled and stored one cell for each row."""
    _line_case(_leading_row_totals, device, br, bc, sr, 15, lambda x, flat: x[:sr].sum(1), SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS)
def test_totals_of_the_leading_columns_stored_as_a_row(br, bc, sc, device):
    """The first ``sc`` columns of the block, every row of them, totalled and stored one cell for each column."""
    _line_case(_leading_column_totals, device, br, bc, sc, 16, lambda x, flat: x[:, :sc].sum(0), SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_tile_whose_value_comes_from_its_row_totals(br, bc, sr, sc, device):
    """Each cell of the tile is the total of its row plus its column number."""
    _tile_case(
        _tile_from_row_totals,
        device,
        br,
        bc,
        sr,
        sc,
        17,
        lambda x, flat: x[:sr, :sc].sum(1)[:, None] + np.arange(sc)[None, :],
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_tile_whose_value_comes_from_its_column_totals(br, bc, sr, sc, device):
    """Each cell of the tile is the total of its column plus its row number."""
    _tile_case(
        _tile_from_column_totals,
        device,
        br,
        bc,
        sr,
        sc,
        18,
        lambda x, flat: x[:sr, :sc].sum(0)[None, :] + np.arange(sr)[:, None],
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_tile_whose_value_comes_from_totals_over_the_width_of_the_block(br, bc, sr, sc, device):
    """The row totals are taken over all ``bc`` columns of the block and stored over the tile's ``sc`` columns."""
    _tile_case(
        _tile_from_wide_row_totals,
        device,
        br,
        bc,
        sr,
        sc,
        19,
        lambda x, flat: x[:sr, :bc].sum(1)[:, None] + np.arange(sc)[None, :],
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", COLUMNS + LARGE_COLUMNS + [(8, 16, 8), (32, 32, 32), (64, 64, 64)])
def test_column_whose_value_is_a_loaded_vector(br, bc, sr, device):
    """A one-dimensional load of ``sr`` cells, turned into a column and stored one cell for each row."""
    _line_case(_column_from_vector, device, br, bc, sr, 20, lambda x, flat: flat[:sr] * 2, SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS + LARGE_ROWS + [(8, 16, 16), (32, 32, 32), (64, 64, 64)])
def test_row_whose_value_is_a_loaded_vector(br, bc, sc, device):
    """A one-dimensional load of ``sc`` cells, turned into a row and stored one cell for each column."""
    _line_case(_row_from_vector, device, br, bc, sc, 21, lambda x, flat: flat[:sc] * 2, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_whose_value_is_a_loaded_vector_along_its_columns(br, bc, sr, sc, device):
    """A loaded vector of ``sc`` cells is repeated down the rows of the tile with the row number added."""
    _tile_case(
        _tile_from_row_vector,
        device,
        br,
        bc,
        sr,
        sc,
        22,
        lambda x, flat: flat[None, :sc] + np.arange(sr)[:, None] * 100,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_whose_value_is_a_loaded_vector_along_its_rows(br, bc, sr, sc, device):
    """A loaded vector of ``sr`` cells is repeated across the columns of the tile with the column number added."""
    _tile_case(
        _tile_from_column_vector,
        device,
        br,
        bc,
        sr,
        sc,
        23,
        lambda x, flat: flat[:sr, None] + np.arange(sc)[None, :] * 100,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_tile_whose_value_is_the_running_totals_of_a_vector_along_its_columns(br, bc, sr, sc, device):
    """The running totals of ``sc`` loaded cells are repeated down the rows of the tile with the row number added."""
    _tile_case(
        _tile_from_vector_running_totals,
        device,
        br,
        bc,
        sr,
        sc,
        24,
        lambda x, flat: np.cumsum(flat[:sc])[None, :] + np.arange(sr)[:, None] * 100,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SMALL_TILES)
def test_tile_whose_value_is_a_vector_less_its_running_totals_along_its_columns(br, bc, sr, sc, device):
    """The running totals and the vector they are taken from meet cell by cell before they are laid along the columns."""
    _tile_case(
        _tile_from_vector_less_running_totals,
        device,
        br,
        bc,
        sr,
        sc,
        25,
        lambda x, flat: (flat[:sc] * 3 - np.cumsum(flat[:sc]))[None, :] + np.arange(sr)[:, None] * 100,
    )


# --- A narrow one-dimensional store beside a wider block -------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_copy_beside_a_wider_vector(block, small, device):
    """``small`` cells are copied in a kernel that runs ``block`` lanes."""
    x = _values(block + PAD, 30)
    _, o, q = _run(_vector_copy, device, [x, _blank(block + PAD), _blank(block + PAD)], BLOCK=block, SMALL=small)
    want = _blank(block + PAD)
    want[:small] = x[:small] * 2 + 1
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,small", VECTORS_IN_A_TILE + LARGE_VECTORS_IN_A_TILE)
def test_vector_copy_beside_a_wider_tile(br, bc, small, device):
    """``small`` cells are copied in a kernel whose widest block is two-dimensional."""
    _line_case(_vector_copy_in_a_tile, device, br, bc, small, 31, lambda x, flat: flat[:small] * 2 + 1, SMALL=small)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_masked_vector_store_at_computed_cells(block, small, device):
    """The upper half of the narrow vector is stored over the cells of the lower half; the mask drops the rest."""
    half = small // 2
    x = _values(block + PAD, 32)
    arrays = [x, _blank(block + PAD), _blank(block + PAD)]
    _, o, q = _run(_vector_folded, device, arrays, BLOCK=block, SMALL=small, HALF=half)
    want = _blank(block + PAD)
    want[:half] = x[half:small]
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,small", VECTORS_IN_A_TILE + LARGE_VECTORS_IN_A_TILE)
def test_masked_vector_store_at_computed_cells_beside_a_wider_tile(br, bc, small, device):
    """The same masked store in a kernel whose widest block is two-dimensional."""
    half = small // 2
    constants = dict(SMALL=small, HALF=half)
    _line_case(_vector_folded_in_a_tile, device, br, bc, half, 33, lambda x, flat: flat[half:small], **constants)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_masked_vector_store_inside_a_loop(block, small, device):
    """Three trips of a loop each store the masked half of the narrow vector at the next run of cells."""
    half = small // 2
    arrays = [_blank(4 * block + PAD), _blank(block + PAD)]
    o, q = _run(_vector_folded_in_a_loop, device, arrays, BLOCK=block, SMALL=small, HALF=half)
    want = _blank(4 * block + PAD)
    want[: 3 * half] = (np.arange(half)[None, :] + half + 1 + np.arange(3)[:, None]).reshape(-1)
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_store_at_loaded_cells(block, small, device):
    """The narrow vector is stored at cells read from an index buffer, each cell named once."""
    x = _values(block + PAD, 34)
    index = np.zeros(block + PAD, dtype=np.int32)
    index[:small] = np.random.default_rng(35).permutation(small)
    index[small:] = small + np.arange(block + PAD - small) % (block - small)
    arrays = [index, x, _blank(block + PAD), _blank(block + PAD)]
    _, _, o, q = _run(_vector_scattered, device, arrays, BLOCK=block, SMALL=small)
    want = _blank(block + PAD)
    want[index[:small]] = x[:small]
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_store_at_every_second_cell(block, small, device):
    """The narrow vector is stored at a stride of two; the cells between stay as they were."""
    x = _values(block + PAD, 36)
    arrays = [x, _blank(2 * block + PAD), _blank(block + PAD)]
    _, o, q = _run(_vector_strided, device, arrays, BLOCK=block, SMALL=small)
    want = _blank(2 * block + PAD)
    want[: 2 * small : 2] = x[:small]
    _same(o, want)
    _same(q, _ramp(block))


# --- Atomics over a narrow block -------------------------------------------------------------------------------------

# The tiles at which an atomic whose value is a loaded tile is given its lanes; the rest of TILES are refused below.
ATOMIC_TILES = [
    (8, 16, 4, 16),
    (8, 16, 8, 8),
    (8, 16, 4, 4),
    (8, 16, 2, 8),
    (16, 8, 4, 2),
    (16, 8, 8, 4),
    (4, 64, 2, 16),
    (32, 32, 8, 16),
    (32, 32, 16, 4),
    (32, 32, 4, 32),
    (32, 64, 8, 16),
    (64, 64, 2, 64),
]


def _tile_atomic_case(kernel, device, br, bc, sr, sc, seed, combine):
    """Run an atomic over an ``sr`` by ``sc`` tile of known cells; ``combine(old, new)`` is what each cell becomes."""
    w = 2 * bc
    x = _values(br * w + PAD, seed)
    start = _values(br * w + PAD, seed + 100)
    _, o, q = _run(kernel, device, [x, start, _blank(br * bc + PAD)], w, BR=br, BC=bc, SR=sr, SC=sc)
    want = start.copy()
    tile = _grid(want, br, w)[:sr, :sc]
    tile[...] = combine(tile, _grid(x, br, w)[:sr, :sc])
    _same(o, want)
    _same(q, _ramp(br * bc))


def _line_atomic_case(kernel, device, br, bc, count, seed, combine, **constants):
    """Run an atomic over the first ``count`` cells; ``combine(old, new)`` is what each cell becomes."""
    x = _values(br * bc + PAD, seed)
    start = _values(br * bc + PAD, seed + 100)
    _, o, q = _run(kernel, device, [x, start, _blank(br * bc + PAD)], BR=br, BC=bc, **constants)
    want = start.copy()
    want[:count] = combine(want[:count], x[:count])
    _same(o, want)
    _same(q, _ramp(br * bc))


def _vector_atomic_case(kernel, device, block, small, seed, combine):
    """Run an atomic over ``small`` cells beside a block of ``block``; ``combine(old, new)`` is what each becomes."""
    x = _values(block + PAD, seed)
    start = _values(block + PAD, seed + 100)
    _, o, q = _run(kernel, device, [x, start, _blank(block + PAD)], BLOCK=block, SMALL=small)
    want = start.copy()
    want[:small] = combine(want[:small], x[:small])
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", ATOMIC_TILES)
def test_tile_atomic_add_adds_each_cell_once(br, bc, sr, sc, device):
    """Every cell of the tile gains its own addend once, however many lanes hold the element."""
    _tile_atomic_case(_tile_add, device, br, bc, sr, sc, 40, lambda old, new: old + new)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", ATOMIC_TILES)
def test_tile_atomic_max_uses_each_cell_s_own_value(br, bc, sr, sc, device):
    """Every cell of the tile becomes the larger of itself and its own operand, not a neighbour's."""
    _tile_atomic_case(_tile_max, device, br, bc, sr, sc, 41, np.maximum)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", ATOMIC_TILES)
def test_tile_atomic_exchange_places_each_cell_s_own_value(br, bc, sr, sc, device):
    """Every cell of the tile is replaced by its own operand."""
    _tile_atomic_case(_tile_exchange, device, br, bc, sr, sc, 42, lambda old, new: new)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_atomic_add_of_values_made_from_its_own_ranges(br, bc, sr, sc, device):
    """The addend is computed from the tile's row and column numbers; every cell gains it once."""
    w = 2 * bc
    start = _values(br * w + PAD, 43)
    o, q = _run(_tile_add_ramp, device, [start, _blank(br * bc + PAD)], w, BR=br, BC=bc, SR=sr, SC=sc)
    want = start.copy()
    _grid(want, br, w)[:sr, :sc] += np.arange(sr * sc).reshape(sr, sc) + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_compare_and_swap_replaces_only_the_cells_that_match(br, bc, sr, sc, device):
    """Two cells in three hold the value compared against and are replaced; the others keep theirs."""
    w = 2 * bc
    x = _values(br * w + PAD, 44)
    start = _values(br * w + PAD, 45)
    compared = np.where(np.arange(br * w + PAD) % 3 == 0, start + 1, start).astype(np.int32)
    arrays = [x, compared, start, _blank(br * bc + PAD)]
    _, _, o, q = _run(_tile_compare_and_swap, device, arrays, w, BR=br, BC=bc, SR=sr, SC=sc)
    want = start.copy()
    tile = _grid(want, br, w)[:sr, :sc]
    match = _grid(compared, br, w)[:sr, :sc] == tile
    tile[match] = _grid(x, br, w)[:sr, :sc][match]
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
@pytest.mark.parametrize("shared", ["rows", "columns", "cell"])
def test_tile_atomic_add_whose_elements_share_cells(shared, br, bc, sr, sc, device):
    """The tile's rows, its columns or all of its elements name the same cells; each element still adds once."""
    kernel = {"rows": _tile_add_rows_share, "columns": _tile_add_columns_share, "cell": _tile_add_one_cell}[shared]
    start = _values(br * bc + PAD, 46)
    o, q = _run(kernel, device, [start, _blank(br * bc + PAD)], BR=br, BC=bc, SR=sr, SC=sc)
    want = start.copy()
    if shared == "rows":
        want[:sc] += sr
    elif shared == "columns":
        want[:sr] += sc
    else:
        want[0] += sr * sc
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "block,sr,sc", [(64, 4, 8), (64, 8, 4), (256, 4, 8), (256, 8, 16), (256, 16, 4), (1024, 8, 8), (4096, 2, 64)]
)
def test_tile_atomic_add_in_a_kernel_whose_widest_block_is_a_vector(block, sr, sc, device):
    """A two-dimensional atomic add beside a one-dimensional block with more elements."""
    x = _values(block + PAD, 47)
    start = _values(block + PAD, 48)
    _, o, q = _run(_tile_add_in_a_vector, device, [x, start, _blank(block + PAD)], BLOCK=block, SR=sr, SC=sc)
    want = start.copy()
    want[: sr * sc] += x[: sr * sc]
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", COLUMNS + LARGE_COLUMNS)
def test_column_atomic_add_adds_each_row_once(br, bc, sr, device):
    """An atomic add over a one-column tile: each of ``sr`` cells gains its addend once, not once for each column."""
    _line_atomic_case(_column_add, device, br, bc, sr, 50, lambda old, new: old + new, SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", COLUMNS + LARGE_COLUMNS)
def test_column_atomic_max(br, bc, sr, device):
    """An atomic max over a one-column tile."""
    _line_atomic_case(_column_max, device, br, bc, sr, 51, np.maximum, SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS + LARGE_ROWS)
def test_row_atomic_add_adds_each_column_once(br, bc, sc, device):
    """An atomic add over a one-row tile: each of ``sc`` cells gains its addend once, not once for each row."""
    _line_atomic_case(_row_add, device, br, bc, sc, 52, lambda old, new: old + new, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS + LARGE_ROWS)
def test_row_atomic_exchange(br, bc, sc, device):
    """An atomic exchange over a one-row tile."""
    _line_atomic_case(_row_exchange, device, br, bc, sc, 53, lambda old, new: new, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS)
@pytest.mark.parametrize("short", [0, 1])
def test_row_totals_added_atomically_as_a_column(short, br, bc, device):
    """Each row's total is added to that row's cell once; with ``short`` the mask leaves the last row out."""
    x = _values(br * bc + PAD, 54)
    start = _values(br * bc + PAD, 55)
    _, o = _run(_row_totals_added_as_a_column, device, [x, start], br - short, BR=br, BC=bc)
    want = start.copy()
    want[: br - short] += _grid(x, br, bc).sum(1)[: br - short]
    _same(o, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS)
def test_column_totals_added_atomically_as_a_row(br, bc, device):
    """Each column's total is added to that column's cell once."""
    x = _values(br * bc + PAD, 56)
    start = _values(br * bc + PAD, 57)
    _, o = _run(_column_totals_added_as_a_row, device, [x, start], BR=br, BC=bc)
    want = start.copy()
    want[:bc] += _grid(x, br, bc).sum(0)
    _same(o, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS)
def test_row_totals_exchanged_and_the_old_values_stored(br, bc, device):
    """A one-dimensional exchange of the row totals; the value each cell held before is stored beside it."""
    x = _values(br * bc + PAD, 58)
    start = _values(br * bc + PAD, 59)
    _, o, p = _run(_row_totals_exchanged_old_stored, device, [x, start, _blank(br * bc + PAD)], BR=br, BC=bc)
    want, old = start.copy(), _blank(br * bc + PAD)
    want[:br] = _grid(x, br, bc).sum(1)
    old[:br] = start[:br]
    _same(o, want)
    _same(p, old)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS + [(64, 64), (8, 256), (256, 8)])
def test_loaded_vector_added_atomically_as_a_column(br, bc, device):
    """A one-dimensional load of ``br`` cells, turned into a column and added once to each row's cell."""
    v = _values(br * bc + PAD, 60)
    start = _values(br * bc + PAD, 61)
    _, o, q = _run(_vector_added_as_a_column, device, [v, start, _blank(br * bc + PAD)], BR=br, BC=bc)
    want = start.copy()
    want[:br] += v[:br] * 2
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_atomic_add_beside_a_wider_vector(block, small, device):
    """An atomic add over ``small`` cells in a kernel that runs ``block`` lanes."""
    _vector_atomic_case(_vector_add, device, block, small, 62, lambda old, new: old + new)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_atomic_max_beside_a_wider_vector(block, small, device):
    """An atomic max over ``small`` cells in a kernel that runs ``block`` lanes."""
    _vector_atomic_case(_vector_max, device, block, small, 63, np.maximum)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
@pytest.mark.parametrize("operation", ["add", "exchange"])
def test_vector_atomic_whose_old_values_are_stored(operation, block, small, device):
    """The value each cell held before a narrow atomic add or exchange is stored at the same narrow range."""
    kernel = {"add": _vector_add_old_stored, "exchange": _vector_exchange_old_stored}[operation]
    x = _values(block + PAD, 64)
    start = _values(block + PAD, 65)
    arrays = [x, start, _blank(block + PAD), _blank(block + PAD)]
    _, o, p, q = _run(kernel, device, arrays, BLOCK=block, SMALL=small)
    want, old = start.copy(), _blank(block + PAD)
    want[:small] = start[:small] + x[:small] if operation == "add" else x[:small]
    old[:small] = start[:small]
    _same(o, want)
    _same(p, old)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_compare_and_swap_whose_old_values_are_stored(block, small, device):
    """Two cells in three match and are replaced; the value every cell held before is stored."""
    x = _values(block + PAD, 66)
    start = _values(block + PAD, 67)
    compared = np.where(np.arange(block + PAD) % 3 == 0, start + 1, start).astype(np.int32)
    arrays = [x, compared, start, _blank(block + PAD), _blank(block + PAD)]
    _, _, o, p, q = _run(_vector_compare_and_swap_old_stored, device, arrays, BLOCK=block, SMALL=small)
    want, old = start.copy(), _blank(block + PAD)
    match = compared[:small] == start[:small]
    want[:small][match] = x[:small][match]
    old[:small] = start[:small]
    _same(o, want)
    _same(p, old)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,small", VECTORS_IN_A_TILE + [(64, 64, 64)] + LARGE_VECTORS_IN_A_TILE)
def test_vector_atomic_add_at_cells_two_elements_share(br, bc, small, device):
    """Two elements of the narrow vector name each cell; the cell gains both addends, each once."""
    half = small // 2
    x = _values(br * bc + PAD, 68)
    start = _values(br * bc + PAD, 69)
    arrays = [x, start, _blank(br * bc + PAD)]
    _, o, q = _run(_vector_add_folded_in_a_tile, device, arrays, BR=br, BC=bc, SMALL=small, HALF=half)
    want = start.copy()
    want[:half] += x[:half] + x[half:small]
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_histogram_of_a_narrow_vector(block, small, device):
    """One is added at the bin each of ``small`` loaded indices names; each bin counts its elements once."""
    index = (np.arange(block + PAD) * 5 % 7).astype(np.int32)
    start = _values(block + PAD, 70)
    _, h, q = _run(_histogram, device, [index, start, _blank(block + PAD)], BLOCK=block, SMALL=small)
    want = start.copy()
    want[:7] += np.bincount(index[:small], minlength=7).astype(np.int32)
    _same(h, want)
    _same(q, _ramp(block))


# --- An atomic on one cell whose returned value is used --------------------------------------------------------------


def _cell(start):
    """The buffer an atomic on one cell works on: the cell, then sentinels."""
    return np.array([start, SENTINEL, SENTINEL], dtype=np.int32)


def _shifted(block, by):
    """What a store of the lane number plus ``by`` over the kernel's block leaves in a blank buffer."""
    want = _blank(block + PAD)
    want[:block] = np.arange(block) + by
    return want


def _cell_case(kernel, device, block, start, end, by, *scalars):
    """Run ``kernel`` on a cell holding ``start``: the cell ends at ``end`` and every lane has added ``by``."""
    s, q = _run(kernel, device, [_cell(start), _blank(block + PAD)], *scalars, BLOCK=block)
    _same(s, _cell(end))
    _same(q, _shifted(block, by))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
def test_scalar_exchange_used(block, device):
    """One exchange happens, and every lane of the block sees the value the cell held before it."""
    _cell_case(scalar_exchange_used, device, block, 3, 5, 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
@pytest.mark.parametrize("start", [3, 7])
def test_scalar_compare_and_swap_used(start, block, device):
    """The swap happens only when the cell holds the compared value; every lane sees what the cell held."""
    _cell_case(scalar_compare_and_swap_used, device, block, start, 4 if start == 3 else start, start)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
@pytest.mark.parametrize("start", [3, 12])
def test_scalar_max_used(start, block, device):
    """Every lane sees the value the cell held before the maximum, whether or not the maximum changed it."""
    _cell_case(scalar_max_used, device, block, start, max(start, 9), start)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
def test_scalar_min_used(block, device):
    """Every lane sees the value the cell held before the minimum."""
    _cell_case(scalar_min_used, device, block, 3, -9, 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [1, 2, 64])
def test_scalar_add_used(block, device):
    """One addition happens, not one for each lane, and every lane sees the value the cell held before it."""
    _cell_case(scalar_add_used, device, block, 3, 8, 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [1024, 2048, 4096])
def test_scalar_add_used_is_refused_at_the_thread_count_and_above(block, device):
    """An addition would repeat on each pass over the lanes, so these blocks are refused, as they were before."""
    arrays = [_cell(3), _blank(block + PAD)]
    _refused(scalar_add_used, device, arrays, match="a scalar non-idempotent atomic_add", BLOCK=block)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
def test_scalar_max_twice_used(block, device):
    """Two maxima on the same cell run in the source's order; the second returns what the first left."""
    _cell_case(scalar_max_twice_used, device, block, 3, 11, 3 * 100 + 7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
@pytest.mark.parametrize("n", [1, 5])
def test_scalar_exchange_in_a_branch_used(n, block, device):
    """Only the exchange of the branch taken happens, and its returned value reaches every lane."""
    _cell_case(scalar_exchange_in_a_branch_used, device, block, 3, 5 if n > 2 else 8, 3 if n > 2 else 103, n)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [1, 2, 64, 1024])
def test_scalar_exchange_in_a_loop_used(block, device):
    """Three exchanges in a loop: each trip returns what the trip before left, and every lane totals all three."""
    s, q = _run(scalar_exchange_in_a_loop_used, device, [_cell(3), _blank(block + PAD)], BLOCK=block)
    want = _blank(block + PAD)
    want[:block] = 3 + 5 + 6
    _same(s, _cell(7))
    _same(q, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", CELL_BLOCKS)
@pytest.mark.parametrize(
    "kernel", [_scalar_compare_and_swap_chain, _scalar_compare_and_swap_in_a_loop], ids=["chain", "loop"]
)
def test_scalar_compare_and_swap_steps_each_happen_once(kernel, block, device):
    """Each swap matches what the one before it left, so the cell ends at the last value only if each ran once."""
    _cell_case(kernel, device, block, 3, 6, 0)


# --- Narrow updates in place, and shared memory ----------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", ROWS + LARGE_ROWS)
def test_row_rmw_narrow_is_accepted_and_updates_each_cell_once(br, bc, sc, device):
    """A one-row tile loads and stores the same cells; three launches each add once to each cell."""
    start = _values(br * bc + PAD, 80)
    o, q = _run(row_rmw_narrow, device, [start, _blank(br * bc + PAD)], times=3, BR=br, BC=bc, SC=sc)
    want = start.copy()
    want[:sc] += 3 * (np.arange(sc) + 1)
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", COLUMNS + LARGE_COLUMNS)
def test_column_rmw_narrow_is_accepted_and_updates_each_cell_once(br, bc, sr, device):
    """A one-column tile loads and stores the same cells; three launches each add once to each cell."""
    start = _values(br * bc + PAD, 81)
    o, q = _run(column_rmw_narrow, device, [start, _blank(br * bc + PAD)], times=3, BR=br, BC=bc, SR=sr)
    want = start.copy()
    want[:sr] += 3 * (np.arange(sr) + 1)
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_update_narrow_is_accepted_and_updates_each_cell_once(br, bc, sr, sc, device):
    """A narrow tile loads and stores the same cells of the output; three launches each add the input tile once."""
    w = 2 * bc
    x = _values(br * w + PAD, 82)
    start = _values(br * w + PAD, 83)
    arrays = [x, start, _blank(br * bc + PAD)]
    _, o, q = _run(tile_update_narrow, device, arrays, w, times=3, BR=br, BC=bc, SR=sr, SC=sc)
    want = start.copy()
    _grid(want, br, w)[:sr, :sc] += 3 * _grid(x, br, w)[:sr, :sc]
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", VECTORS)
def test_vector_rmw_narrow_is_accepted_and_updates_each_cell_once(block, small, device):
    """A narrow vector loads and stores the same cells; an update done twice in one launch would show."""
    start = _values(block + PAD, 84)
    o, q = _run(vector_rmw_narrow, device, [start, _blank(block + PAD)], times=3, BLOCK=block, SMALL=small)
    want = start.copy()
    want[:small] = ((start[:small] * 2 + 1) * 2 + 1) * 2 + 1
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", [(8, 16, 4), (8, 16, 2), (16, 8, 4), (16, 8, 8), (4, 8, 2), (2, 64, 1)])
def test_two_column_updates_in_one_kernel(br, bc, sr, device):
    """One column gains its rows' totals and another gains one, each through a load and store of its own cells."""
    cells = br * bc + PAD
    x = _values(cells, 85)
    start_o, start_p = _values(cells, 86), _values(cells, 87)
    _, o, p, q = _run(_two_column_updates, device, [x, start_o, start_p, _blank(cells)], times=3, BR=br, BC=bc, SR=sr)
    want_o, want_p = start_o.copy(), start_p.copy()
    want_o[:sr] += 3 * _grid(x, br, bc)[:sr].sum(1)
    want_p[:sr] += 3
    _same(o, want_o)
    _same(p, want_p)
    _same(q, _ramp(br * bc))


def _shared(kernel, device, x, shift, others, *scalars, **constants):
    """Launch ``kernel`` with one tensor as its input and, ``shift`` cells further on, as its output."""
    base = torch.from_numpy(x.copy()).to(device)
    rest = [torch.from_numpy(array.copy()).to(device) for array in others]
    kernel[(1,)](base, base[shift:], *rest, *scalars, **constants)
    return [base.cpu().numpy()] + [tensor.cpu().numpy() for tensor in rest]


def _shared_refused(kernel, device, x, shift, others, *scalars, match, **constants):
    """The same launch is refused with a message that matches ``match``, and every buffer is as it was."""
    base = torch.from_numpy(x.copy()).to(device)
    rest = [torch.from_numpy(array.copy()).to(device) for array in others]
    with pytest.raises(MetalNonRecoverableError, match=match):
        kernel[(1,)](base, base[shift:], *rest, *scalars, **constants)
    _same(base.cpu().numpy(), x)
    for tensor, array in zip(rest, others):
        _same(tensor.cpu().numpy(), array)


# The tiles at which a copy between views of one tensor, one cell apart, reaches the launch check.
SHARED_TILES = [
    (8, 16, 4, 16),
    (8, 16, 8, 8),
    (8, 16, 4, 4),
    (8, 16, 2, 8),
    (16, 8, 4, 2),
    (16, 8, 8, 4),
    (4, 64, 2, 16),
    (32, 32, 8, 16),
    (32, 32, 16, 4),
    (32, 32, 4, 32),
    (64, 64, 2, 64),
]
SHARED_VECTORS = [(64, 16), (128, 64), (1024, 64), (4096, 64), (4096, 2048)]
SHARED_ROWS = [(8, 16, 4), (16, 8, 4), (32, 32, 16), (64, 64, 32)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_copy_onto_its_own_cells_is_accepted(br, bc, sr, sc, device):
    """The input and the output are the same tensor: each cell of the tile is read and written by its own lane."""
    w = 2 * bc
    x = _values(br * w + PAD, 90)
    o, q = _shared(_tile_copy, device, x, 0, [_blank(br * bc + PAD)], w, BR=br, BC=bc, SR=sr, SC=sc)
    want = x.copy()
    _grid(want, br, w)[:sr, :sc] = _grid(x, br, w)[:sr, :sc] * 2 + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", SHARED_VECTORS)
def test_vector_copy_onto_its_own_cells_is_accepted(block, small, device):
    """The input and the output are the same tensor, beside a wider vector."""
    x = _values(block + PAD, 91)
    o, q = _shared(_vector_copy, device, x, 0, [_blank(block + PAD)], BLOCK=block, SMALL=small)
    want = x.copy()
    want[:small] = x[:small] * 2 + 1
    _same(o, want)
    _same(q, _ramp(block))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", SHARED_ROWS)
def test_row_copy_onto_its_own_cells_is_accepted(br, bc, sc, device):
    """The input and the output are the same tensor, for a one-row tile."""
    x = _values(br * bc + PAD, 92)
    o, q = _shared(_row_copy, device, x, 0, [_blank(br * bc + PAD)], BR=br, BC=bc, SC=sc)
    want = x.copy()
    want[:sc] = x[:sc] * 2 + 1
    _same(o, want)
    _same(q, _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", SHARED_TILES)
def test_tile_copy_between_shifted_views_of_one_tensor_is_refused(br, bc, sr, sc, device):
    """The output starts one cell after the input, so one lane's store is another's load: a lane order refusal."""
    w = 2 * bc
    x = _values(br * w + PAD, 93)
    others = [_blank(br * bc + PAD)]
    _shared_refused(_tile_copy, device, x, 1, others, w, match="lane order:", BR=br, BC=bc, SR=sr, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,small", SHARED_VECTORS)
def test_vector_copy_between_shifted_views_of_one_tensor_is_refused(block, small, device):
    """A narrow vector copied one cell along its own storage is refused by the lane order check."""
    x = _values(block + PAD, 94)
    _shared_refused(_vector_copy, device, x, 1, [_blank(block + PAD)], match="lane order:", BLOCK=block, SMALL=small)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sc", SHARED_ROWS)
def test_row_copy_between_shifted_views_of_one_tensor_is_refused(br, bc, sc, device):
    """A one-row tile copied one cell along its own storage is refused by the lane order check."""
    x = _values(br * bc + PAD, 95)
    _shared_refused(_row_copy, device, x, 1, [_blank(br * bc + PAD)], match="lane order:", BR=br, BC=bc, SC=sc)


# --- Forms refused when the kernel is compiled -----------------------------------------------------------------------

# Tiles, in kernels of no more than 1024 lanes, that are narrower than the block on at least one side.
REFUSED_TILES = [
    (8, 16, 4, 16),
    (8, 16, 8, 8),
    (8, 16, 4, 4),
    (16, 8, 4, 2),
    (16, 8, 8, 4),
    (32, 32, 8, 16),
    (32, 32, 16, 4),
    (32, 32, 32, 8),
]
UNKNOWN_PLACE = "its address, value or mask depends on {}, whose place on the lanes is not known"
ELEMENT_BY_LANE = "its address, value or mask depends on a one-dimensional value held element by lane"
DIFFERENT_STRIDES = "its row indices are held at different lane strides"
COLUMNS_REPEAT = "its column indices repeat every"
OTHER_AXIS = "an index of the other axis reaches it"
VALUE_READ = "whose returned value is read: one lane performs it for each element"
VALUE_READ_IN_A_LOOP = "an atomic on one element, inside a loop, whose returned value is read"


def _why(operation, lanes, reason):
    """The stable part of the refusal of ``operation`` over a narrow block in a kernel of ``lanes`` lanes."""
    return rf"an? {operation} over a \d+x\d+ block in a kernel that runs {lanes} lanes: {re.escape(reason)}"


def _tile_refused(kernel, device, br, bc, sr, sc, match, buffers=3):
    """A kernel over an ``sr`` by ``sc`` tile at pitch ``2 * bc`` is refused and none of its buffers is written."""
    w = 2 * bc
    arrays = [_values(br * w + PAD, 100)] + [_blank(br * w + PAD) for _ in range(buffers - 1)]
    _refused(kernel, device, arrays, w, match=match, BR=br, BC=bc, SR=sr, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_reshaped_from_a_vector_is_refused(br, bc, sr, sc, device):
    """``tl.reshape`` of a loaded vector into the tile: where its elements lie on the lanes is not known."""
    match = _why("store", br * bc, UNKNOWN_PLACE.format("tt.reshape"))
    _tile_refused(_tile_reshaped_from_a_vector, device, br, bc, sr, sc, match)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "br,bc,sr,sc",
    [
        (8, 16, 8, 8),
        (8, 16, 4, 4),
        (8, 16, 2, 8),
        (16, 8, 4, 2),
        (16, 8, 8, 4),
        (32, 32, 8, 16),
        (32, 32, 16, 4),
        (32, 32, 4, 32),
        (32, 32, 32, 8),
    ],
)
def test_tile_transposed_is_refused(br, bc, sr, sc, device):
    """``tl.trans`` of a loaded tile, stored at the transposed cells."""
    match = _why("store", br * bc, UNKNOWN_PLACE.format("tt.trans"))
    _tile_refused(_tile_transposed, device, br, bc, sr, sc, match)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", REFUSED_TILES)
def test_tile_gathered_is_refused(br, bc, sr, sc, device):
    """``tl.gather`` along the columns of a loaded tile."""
    match = _why("store", br * bc, UNKNOWN_PLACE.format("tt.gather"))
    _tile_refused(_tile_gathered, device, br, bc, sr, sc, match)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", REFUSED_TILES)
def test_tile_from_the_running_totals_of_a_vector_along_its_rows_is_refused(br, bc, sr, sc, device):
    """A one-dimensional ``tl.cumsum`` of ``sr`` cells is held by column, which is not where the tile's rows look."""
    match = _why("store", br * bc, ELEMENT_BY_LANE)
    _tile_refused(_tile_from_column_vector_running_totals, device, br, bc, sr, sc, match)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", REFUSED_TILES)
def test_tile_from_the_running_totals_of_a_row_is_refused(br, bc, sr, sc, device):
    """The running totals of a one-row tile hold their rows at another lane stride than the tile's address does."""
    match = _why("store", br * bc, DIFFERENT_STRIDES)
    _tile_refused(_tile_from_row_running_totals, device, br, bc, sr, sc, match)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", [(32, 32, 8, 16), (32, 32, 4, 32), (32, 32, 32, 8)])
def test_row_totals_of_a_tile_are_refused_where_the_strides_differ(br, bc, sr, sc, device):
    """At these tiles the totals and the column they are stored to hold their rows at different lane strides."""
    w = 2 * bc
    arrays = [_values(br * w + PAD, 101), _blank(br * bc + PAD), _blank(br * bc + PAD)]
    match = _why("store", br * bc, DIFFERENT_STRIDES)
    _refused(_tile_row_totals, device, arrays, w, match=match, BR=br, BC=bc, SR=sr, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", [(32, 32, 8), (32, 32, 16)])
def test_totals_of_the_leading_rows_are_refused_where_the_strides_differ(br, bc, sr, device):
    """The leading rows of a 32 by 32 block, totalled and stored as a column."""
    arrays = [_values(br * bc + PAD, 102), _blank(br * bc + PAD), _blank(br * bc + PAD)]
    match = _why("store", br * bc, DIFFERENT_STRIDES)
    _refused(_leading_row_totals, device, arrays, match=match, BR=br, BC=bc, SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", [(32, 32, 32, 8), (64, 64, 16, 16), (64, 64, 64, 8), (64, 64, 8, 32)])
@pytest.mark.parametrize("kernel", [_tile_add, _tile_max, _tile_exchange], ids=["add", "max", "exchange"])
def test_tile_atomic_is_refused_where_the_strides_differ(kernel, br, bc, sr, sc, device):
    """At these tiles an atomic whose value is a loaded tile is refused, and the cells keep what they held."""
    _tile_refused(kernel, device, br, bc, sr, sc, _why("atomic", br * bc, DIFFERENT_STRIDES))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,sr,sc", [(1024, 16, 32), (2048, 16, 16), (4096, 16, 32), (4096, 64, 8)])
def test_tile_atomic_add_beside_a_vector_is_refused_where_the_strides_differ(block, sr, sc, device):
    """A two-dimensional atomic add beside a one-dimensional block, at tiles whose rows the lanes do not line up."""
    arrays = [_values(block + PAD, 103), _values(block + PAD, 104), _blank(block + PAD)]
    match = _why("atomic", block, DIFFERENT_STRIDES)
    _refused(_tile_add_in_a_vector, device, arrays, match=match, BLOCK=block, SR=sr, SC=sc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS)
def test_column_totals_stored_as_a_column_are_refused(br, bc, device):
    """Totals along the rows are held by column; stored down a column, the lanes that hold them repeat too widely."""
    arrays = [_values(br * bc + PAD, 105), _blank(br * bc + PAD)]
    match = _why("store", br * bc, COLUMNS_REPEAT)
    _refused(_column_totals_stored_as_a_column, device, arrays, match=match, BR=br, BC=bc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS)
def test_row_totals_stored_as_a_row_are_refused(br, bc, device):
    """Totals along the columns are held by row; stored along a row, a row index reaches a block of one row."""
    arrays = [_values(br * bc + PAD, 106), _blank(br * bc + PAD)]
    match = _why("store", br * bc, OTHER_AXIS)
    _refused(_row_totals_stored_as_a_row, device, arrays, match=match, BR=br, BC=bc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", REFUSED_TILES)
def test_running_totals_of_a_row_are_refused(br, bc, sr, sc, device):
    """``tl.cumsum`` along a one-row tile, stored at the row's own cells."""
    _tile_refused(_row_running_totals, device, br, bc, sr, sc, _why("store", br * bc, OTHER_AXIS))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", TILES)
def test_tile_exchange_whose_old_values_are_stored_is_refused(br, bc, sr, sc, device):
    """One lane exchanges each element and the old value is not handed to the element's other lanes."""
    lanes = br * bc
    match = rf"an atomic over \[{sr}, {sc}\] in a kernel that runs {lanes} lanes, {re.escape(VALUE_READ)}"
    _tile_refused(_tile_exchange_old_stored, device, br, bc, sr, sc, match, buffers=4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr", [(8, 16, 4), (8, 16, 8), (16, 8, 8), (32, 32, 8), (32, 32, 32), (64, 64, 16)])
def test_column_exchange_whose_old_values_are_stored_is_refused(br, bc, sr, device):
    """The same refusal for a one-column tile, with as many rows as the block or fewer."""
    arrays = [_values(br * bc + PAD, 107)] + [_blank(br * bc + PAD) for _ in range(3)]
    match = rf"an atomic over \[{sr}, 1\] in a kernel that runs {br * bc} lanes, {re.escape(VALUE_READ)}"
    _refused(_column_exchange_old_stored, device, arrays, match=match, BR=br, BC=bc, SR=sr)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc", BLOCKS)
def test_row_totals_exchanged_as_a_column_whose_old_values_are_stored_are_refused(br, bc, device):
    """The same refusal when the exchanged value is the block's row totals."""
    arrays = [_values(br * bc + PAD, 108), _blank(br * bc + PAD), _blank(br * bc + PAD)]
    match = rf"an atomic over \[{br}, 1\] in a kernel that runs {br * bc} lanes, {re.escape(VALUE_READ)}"
    _refused(_row_totals_exchanged_as_a_column_old_stored, device, arrays, match=match, BR=br, BC=bc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [2048, 4096])
def test_scalar_exchange_in_a_loop_used_is_refused_above_the_thread_count(block, device):
    """With more lanes than threads, the value each trip returns is not kept for the lanes that run later."""
    arrays = [_cell(3), _blank(block + PAD)]
    _refused(scalar_exchange_in_a_loop_used, device, arrays, match=VALUE_READ_IN_A_LOOP, BLOCK=block)


# --- An output that does not start on a 16-byte boundary -------------------------------------------------------------

# Such an output is compiled apart: the row range of its address is written a second time. At these tiles the two are
# held at different lane strides and the kernel is refused when it is lowered.
UNALIGNED_REFUSED_TILES = [(32, 32, 32, 8), (64, 64, 16, 16), (64, 64, 64, 8), (64, 64, 8, 32)]
UNALIGNED_TILES = [tile for tile in TILES if tile not in UNALIGNED_REFUSED_TILES]


def _tile_copy_into_a_view(device, br, bc, sr, sc, cells):
    """The tile copy with its output ``cells`` cells into a separate tensor: the launch, the tensors and the input."""
    w = 2 * bc
    x = _values(br * w + PAD, 96)
    arrays = (x, _blank(br * w + PAD + cells), _blank(br * bc + PAD))
    tensors = [torch.from_numpy(array.copy()).to(device) for array in arrays]

    def launch():
        _tile_copy[(1,)](tensors[0], tensors[1][cells:], tensors[2], w, BR=br, BC=bc, SR=sr, SC=sc)

    return launch, tensors, x


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cells", [1, 3])
@pytest.mark.parametrize("br,bc,sr,sc", UNALIGNED_TILES)
def test_tile_copy_into_an_output_off_a_16_byte_boundary(br, bc, sr, sc, cells, device):
    """The copy into a view that starts 4 or 12 bytes into its tensor stores the same tile."""
    w = 2 * bc
    launch, tensors, x = _tile_copy_into_a_view(device, br, bc, sr, sc, cells)
    launch()
    want = _blank(br * w + PAD + cells)
    _grid(want[cells:], br, w)[:sr, :sc] = _grid(x, br, w)[:sr, :sc] * 2 + 1
    _same(tensors[1].cpu().numpy(), want)
    _same(tensors[2].cpu().numpy(), _ramp(br * bc))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cells", [1, 3])
@pytest.mark.parametrize("br,bc,sr,sc", UNALIGNED_REFUSED_TILES)
def test_tile_copy_into_an_output_off_a_16_byte_boundary_is_refused_at_four_tiles(br, bc, sr, sc, cells, device):
    """At these tiles the copy is refused when it is lowered and writes nothing; on an aligned output it computes."""
    w = 2 * bc
    launch, tensors, x = _tile_copy_into_a_view(device, br, bc, sr, sc, cells)
    with pytest.raises(MetalNonRecoverableError, match=_why("store", br * bc, DIFFERENT_STRIDES)):
        launch()
    _same(tensors[1].cpu().numpy(), _blank(br * w + PAD + cells))
    _same(tensors[2].cpu().numpy(), _blank(br * bc + PAD))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("br,bc,sr,sc", [(32, 32, 32, 8), (64, 64, 16, 16), (64, 64, 64, 8)])
def test_tile_copy_between_shifted_views_is_refused_when_lowered_at_these_tiles(br, bc, sr, sc, device):
    """The output one cell after the input is off a 16-byte boundary, so at these tiles the lowering refuses first."""
    w = 2 * bc
    x = _values(br * w + PAD, 96)
    others = [_blank(br * bc + PAD)]
    match = _why("store", br * bc, DIFFERENT_STRIDES)
    _shared_refused(_tile_copy, device, x, 1, others, w, match=match, BR=br, BC=bc, SR=sr, SC=sc)
