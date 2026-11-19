"""A loop-carried tile of more cells than threads is updated in place, or the kernel is refused.

A 32 x 64 tile does not fit a 1024-thread group as one cell per thread. It is held in
threadgroup memory, and an elementwise operation between it and one value per row
overwrites its cells. Three things were wrong there. ``other - tile`` and ``other / tile``
were computed as ``tile - other`` and ``tile / other``. Any second operand of the tile's
shape was read as one value per row, a whole tile too. And the old cells stayed readable
by name after they were overwritten. A fourth: a value computed one cell per thread from the
tile (``tl.maximum(acc, 40.0)``) was never written back to it, so the next trip and the
loop's result read the cells as they stood before, 4096 of 4096 wrong in a 32 x 64 tile.

The update in place is now emitted with its operands in the source's order and only where
the IR shows it is the source's value: the other operand is one value per row, the tile is
made anew before every run of the update, and nothing reads the old cells after it. Every
other form is refused.

No mask is used and every tensor holds whole blocks: every address is inside its tensor.
"""

import re

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import triton_msl  # noqa: F401
from triton_msl.backend.compiler import MetalBackend
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
ROWS, DIM, KEYS = 32, 64, 32
TRIPS = 3


@triton.jit
def _carried_tile(
    Q, K, V, Out, NK, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, FORM: tl.constexpr, LAST: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    ds = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * D + ds[None, :])
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.full((BM, D), 1.0, tl.float32)
    for start in range(0, NK, BN):
        keys = start + cols
        k = tl.load(K + keys[:, None] * D + ds[None, :])
        score = tl.dot(q, tl.trans(k))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp(maximum - newmax) + 0.5
        p = tl.exp(score - newmax[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        v = tl.load(V + keys[:, None] * D + ds[None, :])
        if FORM == 0:
            acc = acc * alpha[:, None]
        elif FORM == 1:
            acc = alpha[:, None] - acc
        elif FORM == 2:
            acc = acc - alpha[:, None]
        elif FORM == 3:
            acc = alpha[:, None] / acc
        elif FORM == 4:
            acc = 2.0 - acc
        elif FORM == 5:
            acc = 1.0 / acc
        elif FORM == 6:
            acc = -acc
        elif FORM == 7:
            acc = acc * denom[:, None]
        elif FORM == 8:  # the other operand is a whole tile
            acc = acc - q
        elif FORM == 9:  # the old tile is read after the new one is made
            t = acc - alpha[:, None]
            acc = acc - t
        elif FORM == 10:  # two updates share one broadcast column
            acc = (acc * alpha[:, None]) * alpha[:, None]
        acc += tl.dot(p, v)
        if FORM == 11:
            acc = alpha[:, None] - acc
        elif FORM == 12:  # a value of one cell per thread is handed on to the tile
            acc = tl.maximum(acc, 40.0)
        maximum = newmax
    if LAST == 0:
        acc = acc / denom[:, None]
    elif LAST == 1:
        acc = denom[:, None] / acc
    else:
        acc = denom[:, None] - acc
    tl.store(Out + rows[:, None] * D + ds[None, :], acc)


UPDATES = {
    0: "tile * column",
    1: "column - tile",
    2: "tile - column",
    3: "column / tile",
    4: "constant - tile",
    5: "constant / tile",
    6: "-tile",
    7: "tile * carried column",
    11: "column - tile, after the dot",
}
REFUSED = {
    8: "its other operand is not shown to be one value per row",
    9: "the tile's cells are also read by",
    10: "tt.broadcast of a reduction result",
    12: "handed on as a value computed one cell per thread",
}
LASTS = {0: "tile / column", 1: "column / tile", 2: "column - tile"}


def _model(q, k, v, form, last):
    out = np.zeros_like(q)
    size = np.zeros_like(q)
    for first in range(0, q.shape[0], ROWS):
        rows = q[first : first + ROWS]
        maximum = np.full(ROWS, -np.inf)
        denom = np.zeros(ROWS)
        acc = np.ones((ROWS, DIM))
        for start in range(0, k.shape[0], KEYS):
            score = rows @ k[start : start + KEYS].T
            newmax = np.maximum(maximum, score.max(1))
            alpha = (np.exp(maximum - newmax) + 0.5)[:, None]
            p = np.exp(score - newmax[:, None])
            denom = denom * alpha[:, 0] + p.sum(1)
            if form in (0, 1, 2, 3, 4, 5, 6, 7):
                acc = [
                    acc * alpha,
                    alpha - acc,
                    acc - alpha,
                    alpha / acc,
                    2.0 - acc,
                    1.0 / acc,
                    -acc,
                    acc * denom[:, None],
                ][form]
            acc = acc + p @ v[start : start + KEYS]
            if form == 11:
                acc = alpha - acc
            maximum = newmax
        column = denom[:, None]
        out[first : first + ROWS] = [acc / column, column / acc, column - acc][last]
        size[first : first + ROWS] = np.maximum(np.abs(acc), np.abs(column))
    return out, size


def _launch(form, last, device):
    random = np.random.default_rng(12 + form)
    count, keys = 2 * ROWS, TRIPS * KEYS
    arrays = (random.random((count, DIM)) * 0.25, random.random((keys, DIM)) * 0.25, random.random((keys, DIM)) + 0.5)
    q, k, v = (torch.tensor(array, dtype=torch.float32, device=device) for array in arrays)
    out = torch.full((count, DIM), -777.0, dtype=torch.float32, device=device)
    _carried_tile[(2,)](q, k, v, out, keys, D=DIM, BM=ROWS, BN=KEYS, FORM=form, LAST=last)
    if device == "mps":
        torch.mps.synchronize()
    return out.cpu().numpy().astype(np.float64), tuple(t.cpu().numpy().astype(np.float64) for t in (q, k, v))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("last", sorted(LASTS), ids=lambda last: "then " + LASTS[last])
@pytest.mark.parametrize("form", sorted(UPDATES), ids=lambda form: UPDATES[form])
def test_a_carried_tile_is_updated_with_its_operands_in_the_source_order(form, last, device):
    got, inputs = _launch(form, last, device)
    want, size = _model(*inputs, form, last)
    assert np.all(np.abs(got - want) <= 3e-4 * (np.abs(want) + size) + 1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", sorted(REFUSED), ids=lambda form: REFUSED[form])
def test_an_update_the_text_cannot_do_in_place_is_refused(form, device):
    with pytest.raises(MetalNonRecoverableError, match=re.escape(REFUSED[form])):
        _launch(form, 0, device)


def _text(form, last):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    signature = {"Q": "*fp32", "K": "*fp32", "V": "*fp32", "Out": "*fp32", "NK": "i32"}
    constants = {"D": DIM, "BM": ROWS, "BN": KEYS, "FORM": form, "LAST": last}
    source = ASTSource(fn=_carried_tile, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options)


@pytest.mark.parametrize("sign,form,last", [("-", 1, 0), ("/", 3, 0), ("-", 0, 2), ("/", 0, 1)])
def test_the_text_of_an_update_reads_the_tile_as_the_operand_the_source_names(sign, form, last):
    """Read from the text, no launch."""
    column_first = (
        rf"(\w+)\[_sb\] = (?:tm_fdiv\()?smem_(?:bcast_)?\d+\[_sb / {DIM}u\](?: {re.escape(sign)}|,) \1\[_sb\]\)?;"
    )
    assert re.search(column_first, _text(form, last))
