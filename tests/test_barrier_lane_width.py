"""``tl.debug_barrier()`` orders lanes; Metal's ``threadgroup_barrier`` orders threads.

A source program finishes everything above a barrier, for every lane, before anything
below it starts. The shader's barrier gives that order between threads, so the two agree
only while each lane has a thread of its own. Two things used to break that silently:

* Metal runs at most 1024 threads in a threadgroup. A barrier kernel that kept one lane
  per thread at 2048 lanes ran the first 1024 and never ran the rest (a plain
  ``load, barrier, store`` left the upper half of its output untouched).
* Under a lane loop (the wrapping loop, or the phase loop of a multipass reduction) a
  thread runs its second lane after it has already passed the barrier for its first. An
  in-place reversal over 2048 lanes kept half its elements, and ``store, barrier, load a
  neighbour`` ahead of a reduce was wrong from 256 lanes.

What the tree does now, and what this file holds it to:

* one lane per thread, at most 1024 lanes: the barrier is exact and the launch computes
  the source's result, in place or not;
* one lane per thread, more than 1024 lanes: the lowering refuses
  (``1024-thread threadgroup cap``);
* a lane loop where two lanes of a program meet at an element one of them stores: the
  launch refuses before anything is submitted (``lane order:``);
* a lane loop where no two lanes meet (separate input and output): the barrier decides
  nothing, and the launch computes.

Which of these a width lands in is not a property of the source alone. It follows the
layout Triton picks (``sizePerThread``), and that follows what the launch knows about the
pointers: a tensor whose first element is 16-byte aligned is specialised as such, and
contiguous accesses to it get several elements per thread. The comments beside each table
below say why the widths sit where they do; every boundary here was established by
launching on the device, at ``num_warps=4`` unless a test says otherwise.

Every reference is numpy applying the source's order: whole-block statements, each load
above the barrier complete (fancy indexing copies) before any store below it.
"""

import platform
import re

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

CAP = "1024-thread threadgroup cap"
ORDER = "lane order:"
GUARD = 3  # cells past what a kernel may touch; they must come back as they went in
SPARE = 5  # a masked kernel's length is this much short of three blocks


# --- kernels: BARRIER is a constexpr, so the control without it is the same source ------


@triton.jit
def _load_store(A, O, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(A + i)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + i, x * 2.0)


@triton.jit
def _barrier_last(A, O, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(O + i, tl.load(A + i) * 2.0)
    if BARRIER:
        tl.debug_barrier()


@triton.jit
def _barrier_each_trip(A, O, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for trip in range(3):
        acc += tl.load(A + trip * BLOCK + i)
        if BARRIER:
            tl.debug_barrier()
    tl.store(O + i, acc)


@triton.jit
def _masked(A, O, n, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A + i, mask=i < n, other=0.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + i, x + 1.0, mask=i < n)


@triton.jit
def _masked_reverse_copy(A, O, n, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    j = tl.arange(0, BLOCK)
    count = tl.minimum(BLOCK, n - start)
    v = tl.load(A + start + j, mask=j < count, other=0.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + start + (count - 1 - j), v, mask=j < count)


@triton.jit
def _reverse_copy(A, O, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(A + i)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + (BLOCK - 1 - i), v)


@triton.jit
def _sum_beside(A, O, S, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(A + i)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + i, x + 1.0)
    tl.store(S, tl.sum(x, axis=0))


@triton.jit
def _masked_sum(A, O, S, n, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A + i, mask=i < n, other=0.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + i, x + 1.0, mask=i < n)
    tl.store(S + pid, tl.sum(x, axis=0))


@triton.jit
def _masked_max_scale(A, O, n, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A + i, mask=i < n, other=0.0)
    top = tl.max(x, axis=0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + i, x + top, mask=i < n)


# The kernels from here to the tiles load a tensor they also store, at elements other
# lanes reach. The barrier stands between every load of that tensor and every store to
# it, which is an order a shader can keep; a lane's own order is not enough.


@triton.jit
def _exchange(X, Y, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(X + i, tl.load(Y + i) + 1.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(Y + i, tl.load(X + (i + 1) % BLOCK))


@triton.jit
def _shift_in_place(X, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + (i + 1) % BLOCK)
    if BARRIER:
        tl.debug_barrier()
    tl.store(X + i, v)


@triton.jit
def _reverse_in_place(X, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    if BARRIER:
        tl.debug_barrier()
    tl.store(X + (BLOCK - 1 - i), v)


@triton.jit
def _masked_reverse_in_place(X, n, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    j = tl.arange(0, BLOCK)
    count = tl.minimum(BLOCK, n - start)
    v = tl.load(X + start + j, mask=j < count, other=0.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(X + start + (count - 1 - j), v, mask=j < count)


@triton.jit
def _reverse_sum(X, S, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    if BARRIER:
        tl.debug_barrier()
    tl.store(X + (BLOCK - 1 - i), v)
    tl.store(S, tl.sum(v, axis=0))


@triton.jit
def _shift_sum(X, Y, S, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(X + i, tl.load(Y + i) + 1.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(S, tl.sum(tl.load(X + (i + 1) % BLOCK) * (i % 7).to(tl.float32), axis=0))


@triton.jit
def _max_then_shift(A, X, O, BLOCK: tl.constexpr, BARRIER: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a = tl.load(A + i)
    top = tl.max(a, axis=0)
    tl.store(X + i, a)
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + i, tl.load(X + (i + 1) % BLOCK) + top)


@triton.jit
def _tile(A, O, M: tl.constexpr, N: tl.constexpr, BARRIER: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    x = tl.load(A + r[:, None] * N + c[None, :])
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + r[:, None] * N + c[None, :], x * 2.0)


@triton.jit
def _tile_row_sum(A, O, S, M: tl.constexpr, N: tl.constexpr, BARRIER: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    x = tl.load(A + r[:, None] * N + c[None, :])
    if BARRIER:
        tl.debug_barrier()
    tl.store(O + r[:, None] * N + c[None, :], x + 1.0)
    tl.store(S + r, tl.sum(x, axis=1))


@triton.jit
def _tile_exchange(X, Y, M: tl.constexpr, N: tl.constexpr, BARRIER: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    at = r[:, None] * N + c[None, :]
    tl.store(X + at, tl.load(Y + at) + 1.0)
    if BARRIER:
        tl.debug_barrier()
    tl.store(Y + at, tl.load(X + ((r[:, None] + 1) % M) * N + (c[None, :] + 1) % N))


@triton.jit
def _tile_transpose_in_place(X, M: tl.constexpr, N: tl.constexpr, BARRIER: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    v = tl.load(X + r[:, None] * N + c[None, :])
    if BARRIER:
        tl.debug_barrier()
    tl.store(X + c[None, :] * N + r[:, None], v)


# --- inputs and the source's result -----------------------------------------------------
#
# A build returns (arrays in argument order, scalar arguments, grid, constexprs); the
# matching reference rewrites copies of the arrays the way the source does. Values are
# small integers, so sums are exact in float32 whatever order they are taken in.


def _data(count):
    return ((np.arange(count, dtype=np.int64) * 37 + 11) % 97).astype(np.float32)


def _fill(count):
    return np.full(count, -7.0, dtype=np.float32)


def _length(w):
    return 3 * w - SPARE


def _b_pair(w):
    return {"A": _data(w + GUARD), "O": _fill(w + GUARD)}, (), (1,), {"BLOCK": w}


def _r_load_store(t, w):
    i = np.arange(w)
    x = t["A"][i]
    t["O"][i] = x * 2.0


def _b_each_trip(w):
    return {"A": _data(3 * w + GUARD), "O": _fill(w + GUARD)}, (), (1,), {"BLOCK": w}


def _r_each_trip(t, w):
    i = np.arange(w)
    acc = np.zeros(w, np.float32)
    for trip in range(3):
        acc = acc + t["A"][trip * w + i]
    t["O"][i] = acc


def _b_masked_pair(w):
    n = _length(w)
    return {"A": _data(n + GUARD), "O": _fill(n + GUARD)}, (n,), (3,), {"BLOCK": w}


def _masked_block(t, w, program):
    """(indexes, mask, loaded values with ``other`` where masked) of one program's block."""
    i = program * w + np.arange(w)
    m = i < _length(w)
    return i, m, np.where(m, t["A"][np.where(m, i, 0)], 0.0).astype(np.float32)


def _r_masked(t, w):
    for program in range(3):
        i, m, x = _masked_block(t, w, program)
        t["O"][i[m]] = (x + 1.0)[m]


def _r_masked_reverse_copy(t, w):
    for program in range(3):
        start = program * w
        count = min(w, _length(w) - start)
        j = np.arange(count)
        v = t["A"][start + j]
        t["O"][start + (count - 1 - j)] = v


def _r_reverse_copy(t, w):
    i = np.arange(w)
    v = t["A"][i]
    t["O"][w - 1 - i] = v


def _b_sum_beside(w):
    return {"A": _data(w + GUARD), "O": _fill(w + GUARD), "S": _fill(1 + GUARD)}, (), (1,), {"BLOCK": w}


def _r_sum_beside(t, w):
    i = np.arange(w)
    x = t["A"][i]
    t["O"][i] = x + 1.0
    t["S"][0] = x.sum(dtype=np.float32)


def _b_masked_sum(w):
    n = _length(w)
    return {"A": _data(n + GUARD), "O": _fill(n + GUARD), "S": _fill(3 + GUARD)}, (n,), (3,), {"BLOCK": w}


def _r_masked_sum(t, w):
    for program in range(3):
        i, m, x = _masked_block(t, w, program)
        t["O"][i[m]] = (x + 1.0)[m]
        t["S"][program] = x.sum(dtype=np.float32)


def _r_masked_max_scale(t, w):
    for program in range(3):
        i, m, x = _masked_block(t, w, program)
        t["O"][i[m]] = (x + x.max())[m]


def _b_exchange(w):
    return {"X": _fill(w + GUARD), "Y": _data(w + GUARD)}, (), (1,), {"BLOCK": w}


def _r_exchange(t, w):
    i = np.arange(w)
    t["X"][i] = t["Y"][i] + 1.0
    t["Y"][i] = t["X"][(i + 1) % w]


def _b_one(w):
    return {"X": _data(w + GUARD)}, (), (1,), {"BLOCK": w}


def _r_shift_in_place(t, w):
    i = np.arange(w)
    v = t["X"][(i + 1) % w]
    t["X"][i] = v


def _r_reverse_in_place(t, w):
    i = np.arange(w)
    v = t["X"][i]
    t["X"][w - 1 - i] = v


def _b_masked_one(w):
    n = _length(w)
    return {"X": _data(n + GUARD)}, (n,), (3,), {"BLOCK": w}


def _r_masked_reverse_in_place(t, w):
    for program in range(3):
        start = program * w
        count = min(w, _length(w) - start)
        j = np.arange(count)
        v = t["X"][start + j]
        t["X"][start + (count - 1 - j)] = v


def _b_reverse_sum(w):
    return {"X": _data(w + GUARD), "S": _fill(1 + GUARD)}, (), (1,), {"BLOCK": w}


def _r_reverse_sum(t, w):
    i = np.arange(w)
    v = t["X"][i]
    t["X"][w - 1 - i] = v
    t["S"][0] = v.sum(dtype=np.float32)


def _b_shift_sum(w):
    return {"X": _fill(w + GUARD), "Y": _data(w + GUARD), "S": _fill(1 + GUARD)}, (), (1,), {"BLOCK": w}


def _r_shift_sum(t, w):
    i = np.arange(w)
    t["X"][i] = t["Y"][i] + 1.0
    t["S"][0] = (t["X"][(i + 1) % w] * (i % 7).astype(np.float32)).sum(dtype=np.float32)


def _b_max_then_shift(w):
    return {"A": _data(w + GUARD), "X": _fill(w + GUARD), "O": _fill(w + GUARD)}, (), (1,), {"BLOCK": w}


def _r_max_then_shift(t, w):
    i = np.arange(w)
    a = t["A"][i]
    top = a.max()
    t["X"][i] = a
    t["O"][i] = t["X"][(i + 1) % w] + top


def _b_tile(shape):
    m, n = shape
    return {"A": _data(m * n + GUARD), "O": _fill(m * n + GUARD)}, (), (1,), {"M": m, "N": n}


def _r_tile(t, shape):
    m, n = shape
    at = np.arange(m)[:, None] * n + np.arange(n)[None, :]
    x = t["A"][at]
    t["O"][at] = x * 2.0


def _b_tile_row_sum(shape):
    m, n = shape
    return {"A": _data(m * n + GUARD), "O": _fill(m * n + GUARD), "S": _fill(m + GUARD)}, (), (1,), {"M": m, "N": n}


def _r_tile_row_sum(t, shape):
    m, n = shape
    at = np.arange(m)[:, None] * n + np.arange(n)[None, :]
    x = t["A"][at]
    t["O"][at] = x + 1.0
    t["S"][np.arange(m)] = x.sum(axis=1, dtype=np.float32)


def _b_tile_exchange(shape):
    m, n = shape
    return {"X": _fill(m * n + GUARD), "Y": _data(m * n + GUARD)}, (), (1,), {"M": m, "N": n}


def _r_tile_exchange(t, shape):
    m, n = shape
    r = np.arange(m)[:, None]
    c = np.arange(n)[None, :]
    at = r * n + c
    t["X"][at] = t["Y"][at] + 1.0
    t["Y"][at] = t["X"][((r + 1) % m) * n + (c + 1) % n]


def _b_tile_one(shape):
    m, n = shape
    return {"X": _data(m * n + GUARD)}, (), (1,), {"M": m, "N": n}


def _r_tile_transpose_in_place(t, shape):
    m, n = shape
    r = np.arange(m)[:, None]
    c = np.arange(n)[None, :]
    v = t["X"][r * n + c]
    t["X"][c * n + r] = v


CASES = {
    "load_store": (_load_store, _b_pair, _r_load_store),
    "barrier_last": (_barrier_last, _b_pair, _r_load_store),
    "barrier_each_trip": (_barrier_each_trip, _b_each_trip, _r_each_trip),
    "masked": (_masked, _b_masked_pair, _r_masked),
    "masked_reverse_copy": (_masked_reverse_copy, _b_masked_pair, _r_masked_reverse_copy),
    "reverse_copy": (_reverse_copy, _b_pair, _r_reverse_copy),
    "sum_beside": (_sum_beside, _b_sum_beside, _r_sum_beside),
    "masked_sum": (_masked_sum, _b_masked_sum, _r_masked_sum),
    "masked_max_scale": (_masked_max_scale, _b_masked_pair, _r_masked_max_scale),
    "exchange": (_exchange, _b_exchange, _r_exchange),
    "shift_in_place": (_shift_in_place, _b_one, _r_shift_in_place),
    "reverse_in_place": (_reverse_in_place, _b_one, _r_reverse_in_place),
    "masked_reverse_in_place": (_masked_reverse_in_place, _b_masked_one, _r_masked_reverse_in_place),
    "reverse_sum": (_reverse_sum, _b_reverse_sum, _r_reverse_sum),
    "shift_sum": (_shift_sum, _b_shift_sum, _r_shift_sum),
    "max_then_shift": (_max_then_shift, _b_max_then_shift, _r_max_then_shift),
    "tile": (_tile, _b_tile, _r_tile),
    "tile_row_sum": (_tile_row_sum, _b_tile_row_sum, _r_tile_row_sum),
    "tile_exchange": (_tile_exchange, _b_tile_exchange, _r_tile_exchange),
    "tile_transpose_in_place": (_tile_transpose_in_place, _b_tile_one, _r_tile_transpose_in_place),
}


# --- launching --------------------------------------------------------------------------


def _place(values, device, aligned):
    """(storage, the tensor the kernel gets). Unaligned is a view one element into the storage."""
    lead = 0 if aligned else 1
    storage = torch.from_numpy(np.concatenate([_fill(lead), values])).clone().to(device)
    tensor = storage[lead:]
    # The launch specialises on this, and the layout (so the dispatch) follows it.
    assert (tensor.data_ptr() % 16 == 0) == aligned, f"allocation at {tensor.data_ptr():#x} is not as assumed"
    return storage, tensor


def _launch(name, width, device, barrier=1, warps=4, aligned=True):
    """Launch one case; (what each storage holds afterwards, the source's result, what went in)."""
    kernel, build, reference = CASES[name]
    arrays, scalars, grid, constexprs = build(width)
    lead = _fill(0 if aligned else 1)
    before = {k: np.concatenate([lead, v]) for k, v in arrays.items()}
    want = {k: v.copy() for k, v in arrays.items()}
    reference(want, width)
    want = {k: np.concatenate([lead, v]) for k, v in want.items()}
    placed = {k: _place(v, device, aligned) for k, v in arrays.items()}

    def read():
        if device == "mps":
            torch.mps.synchronize()
        return {k: storage.detach().cpu().numpy() for k, (storage, _) in placed.items()}

    try:
        kernel[grid](*(t for _, t in placed.values()), *scalars, BARRIER=barrier, num_warps=warps, **constexprs)
    except MetalNonRecoverableError:
        # A refusal comes before anything is submitted: nothing may have been written.
        _same(read(), before, "after a refusal")
        raise
    return read(), want, before


def _same(got, want, what):
    for k in want:
        differ = np.flatnonzero(got[k] != want[k])
        assert differ.size == 0, (
            f"{what}: {k}[{differ[0]}] is {got[k][differ[0]]}, the source gives {want[k][differ[0]]} "
            f"({differ.size} of {want[k].size} cells differ)"
        )


def _computes(name, width, device, **how):
    got, want, before = _launch(name, width, device, **how)
    # Inputs and guard cells are compared too, and a kernel that did nothing must not pass.
    assert any(not np.array_equal(want[k], before[k]) for k in want)
    _same(got, want, f"{name} over {width} lanes on {device}")


def _refuses(name, width, device, text, **how):
    with pytest.raises(MetalNonRecoverableError, match=re.escape(text)):
        _launch(name, width, device, **how)


def _lowered(name, width, barrier, warps=4, aligned=True):
    """Lower one case as a launch with (or without) 16-byte aligned tensors would, without a device."""
    kernel, build, _ = CASES[name]
    arrays, scalars, _, constexprs = build(width)
    signature = {k: "*fp32" for k in arrays} | {k: "i32" for k in kernel.arg_names[len(arrays) :][: len(scalars)]}
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": warps})
    # A launch tells Triton which pointers are 16-byte aligned. A hand-built source that
    # says nothing is the unaligned launch: one element per thread, another dispatch.
    attrs = {(index,): backend.parse_attr("D") for index in range(len(arrays))} if aligned else {}
    src = ASTSource(kernel, signature=signature, constexprs=constexprs | {"BARRIER": barrier}, attrs=attrs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    lowerer = GenericLowerer(walk_ttgir(mod, options), options)
    lowerer.lower()
    return lowerer


def _lanes(width):
    return width[0] * width[1] if isinstance(width, tuple) else width


def _ids(value):
    return "x".join(map(str, value)) if isinstance(value, tuple) else None


def _each(names, widths):
    return [(name, width) for name in names for width in widths]


DEVICES = ["mps", "cpu"]
NARROW = (16, 32, 64, 128, 256, 512, 1024)
PAST_CAP = (2048, 4096, 8192)
TILES_NARROW = ((4, 8), (16, 16), (16, 32), (32, 16), (32, 32), (16, 64))
TILES_PAST_CAP = ((32, 64), (64, 32), (64, 64), (64, 128))

# Separate input and output, no reduction. Aligned contiguous accesses get several
# elements per thread from Triton; the lowering answers a barrier there with one lane per
# thread (it has no lane loop it could prove), so these compute up to Metal's 1024
# threads and are refused past it.
SEPARATE = ("load_store", "barrier_last", "barrier_each_trip", "masked", "masked_reverse_copy")
# The same dispatch, with a tensor that is both loaded and stored.
SHARED = ("exchange", "shift_in_place", "masked_reverse_in_place")
# A reversed address is not contiguous, so Triton gives one element per thread and the
# kernel wraps over 1024 threads past 1024 lanes: a lane loop, never the cap.
REVERSED_SHARED = ("reverse_in_place", "reverse_sum")
# A reduction of several elements per thread runs as phases over num_warps * 32 threads,
# so the lane loop starts just above that: 128 lanes at four warps.
REDUCED_SHARED = ("shift_sum", "max_then_shift")
REDUCED_SEPARATE = ("sum_beside", "masked_sum", "masked_max_scale")


# --- one lane per thread: the barrier is exact ------------------------------------------


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(SEPARATE + ("reverse_copy",) + REDUCED_SEPARATE, NARROW))
def test_barrier_between_separate_tensors_computes(device, name, width):
    _computes(name, width, device)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width", _each(SHARED + REVERSED_SHARED, NARROW) + _each(REDUCED_SHARED, (16, 32, 64, 128))
)
def test_in_place_program_the_barrier_defines_computes(device, name, width):
    """Launches a kernel whose stored tensor is also loaded."""
    _computes(name, width, device)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(("tile", "tile_row_sum"), TILES_NARROW), ids=_ids)
def test_tile_barrier_between_separate_tensors_computes(device, name, width):
    _computes(name, width, device)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width",
    _each(("tile_exchange",), TILES_NARROW) + _each(("tile_transpose_in_place",), ((4, 4), (16, 16), (32, 32))),
    ids=_ids,
)
def test_in_place_tile_the_barrier_defines_computes(device, name, width):
    """Launches a kernel whose stored tensor is also loaded."""
    _computes(name, width, device)


# --- one lane per thread would need more than 1024 threads ------------------------------


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(SEPARATE, PAST_CAP))
def test_barrier_wider_than_the_threadgroup_refuses(device, name, width):
    _refuses(name, width, device, CAP)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(SHARED, PAST_CAP))
def test_in_place_barrier_wider_than_the_threadgroup_refuses(device, name, width):
    """Attempts to launch a kernel whose stored tensor is also loaded; the lowering refuses first."""
    _refuses(name, width, device, CAP)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(("tile",), TILES_PAST_CAP), ids=_ids)
def test_tile_barrier_wider_than_the_threadgroup_refuses(device, name, width):
    _refuses(name, width, device, CAP)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width", _each(("tile_exchange",), TILES_PAST_CAP) + [("tile_transpose_in_place", (64, 64))], ids=_ids
)
def test_in_place_tile_barrier_wider_than_the_threadgroup_refuses(device, name, width):
    """Attempts to launch a kernel whose stored tensor is also loaded; the lowering refuses first."""
    _refuses(name, width, device, CAP)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", [(32, 64), (64, 64)], ids=_ids)
def test_tile_barrier_beside_a_row_reduction_past_the_cap_refuses(device, width):
    # Not this fix: a row reduction of a tile over 1024 lanes was refused before it, with
    # or without the barrier, in words of its own. It must never start computing wrongly.
    with pytest.raises(MetalNonRecoverableError):
        _launch("tile_row_sum", width, device)


# --- a lane loop, and two lanes meet at a stored element --------------------------------


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width", _each(REVERSED_SHARED, PAST_CAP) + _each(REDUCED_SHARED, (256, 512, 1024) + PAST_CAP)
)
def test_barrier_under_a_lane_loop_where_lanes_meet_refuses(device, name, width):
    """Attempts to launch a kernel whose stored tensor is also loaded."""
    _refuses(name, width, device, ORDER)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", REDUCED_SHARED)
@pytest.mark.parametrize("warps, widest", [(1, 32), (8, 256), (32, 1024)])
def test_the_lane_loop_of_a_reduction_starts_above_the_warps(device, name, warps, widest):
    """Launches a kernel whose stored tensor is also loaded."""
    _computes(name, widest, device, warps=warps)
    _refuses(name, 2 * widest, device, ORDER, warps=warps)


# --- a lane loop, and no two lanes meet: the barrier decides nothing ---------------------


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width", _each(REDUCED_SEPARATE, (256, 512, 1024) + PAST_CAP) + _each(("reverse_copy",), PAST_CAP)
)
def test_barrier_under_a_lane_loop_between_separate_tensors_computes(device, name, width):
    _computes(name, width, device)


@pytest.mark.parametrize(
    "name, width", _each(REDUCED_SEPARATE, (256, 2048)) + _each(("reverse_copy",) + REVERSED_SHARED, (2048,))
)
def test_those_widths_do_run_a_lane_loop(name, width):
    """The cases above are only about a lane loop if fewer threads than lanes are dispatched."""
    lowerer = _lowered(name, width, barrier=1)
    assert lowerer._actual_dispatch_threads < width
    assert lowerer._actual_dispatch_threads <= 1024


# --- tensors that do not start on a 16-byte boundary ------------------------------------
#
# Nothing tells Triton these accesses can be grouped, so every kernel gets one element
# per thread: one lane per thread up to 1024 lanes whatever it reduces, and the wrapping
# loop over 1024 threads past that. The cap is never reached; what the launch does past
# 1024 lanes depends only on whether lanes meet.


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(SEPARATE + REDUCED_SEPARATE, (1024, 2048, 4096)))
def test_unaligned_barrier_between_separate_tensors_computes(device, name, width):
    _computes(name, width, device, aligned=False)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", [(32, 64), (64, 64)], ids=_ids)
def test_unaligned_tile_barrier_between_separate_tensors_computes(device, width):
    _computes("tile", width, device, aligned=False)


@pytest.mark.parametrize("name, width", _each(SEPARATE + SHARED, (2048,)) + [("tile", (32, 64))], ids=_ids)
def test_unaligned_widths_past_the_cap_do_run_a_lane_loop(name, width):
    lowerer = _lowered(name, width, barrier=1, aligned=False)
    assert lowerer._actual_dispatch_threads < _lanes(width)
    assert lowerer._actual_dispatch_threads <= 1024


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(SHARED + REVERSED_SHARED + REDUCED_SHARED, (256, 1024)))
def test_unaligned_in_place_program_computes_with_a_thread_per_lane(device, name, width):
    """Launches a kernel whose stored tensor is also loaded."""
    _computes(name, width, device, aligned=False)


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width",
    _each(SHARED + REVERSED_SHARED + REDUCED_SHARED, (2048, 4096)) + [("tile_exchange", (32, 64))],
    ids=_ids,
)
def test_unaligned_in_place_program_under_the_wrapping_loop_refuses(device, name, width):
    """Attempts to launch a kernel whose stored tensor is also loaded."""
    _refuses(name, width, device, ORDER, aligned=False)


# --- controls: the same kernels without the barrier -------------------------------------


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, width", _each(SEPARATE, PAST_CAP) + _each(("tile",), TILES_PAST_CAP), ids=_ids)
def test_without_the_barrier_the_refused_widths_compute(device, name, width):
    # Separate tensors: the program is defined without the barrier, and a lane loop runs it.
    _computes(name, width, device, barrier=0)


@pytest.mark.parametrize(
    "name, width",
    _each(SEPARATE + SHARED, (2048, 8192))
    + _each(("tile", "tile_exchange"), ((32, 64), (64, 128)))
    + [("tile_transpose_in_place", (64, 64))],
    ids=_ids,
)
def test_only_the_barrier_makes_the_lowering_refuse(name, width):
    with pytest.raises(MetalNonRecoverableError, match=re.escape(CAP)):
        _lowered(name, width, barrier=1)
    lowerer = _lowered(name, width, barrier=0)
    assert lowerer._actual_dispatch_threads < _lanes(width)
    assert lowerer._actual_dispatch_threads <= 1024


@pytest.mark.parametrize("name, width", _each(REVERSED_SHARED, (2048, 8192)) + _each(REDUCED_SHARED, (256, 8192)))
def test_without_the_barrier_the_lane_order_widths_still_lower(name, width):
    assert _lowered(name, width, barrier=0)._actual_dispatch_threads <= 1024


@requires_mps
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, width",
    _each(("exchange", "shift_in_place", "reverse_in_place", "shift_sum"), (64, 2048)) + [("tile_exchange", (16, 32))],
    ids=_ids,
)
def test_without_the_barrier_an_in_place_launch_refuses(device, name, width):
    """Attempts to launch a kernel whose stored tensor is also loaded.

    The barrier is what lets the launch above compute: with it, every read is before it
    and every store after. Without it nothing keeps a lane from storing an element
    another lane has yet to load, at any width.
    """
    _refuses(name, width, device, ORDER, barrier=0)
