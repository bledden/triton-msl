"""Launches the lane-order check lets through compute what the source computes.

A Triton program finishes each operation over every lane before the next one starts.
The ordinary lowering runs one lane through the whole body, and its threads do not wait
for each other, so a launch is only the source's when no two lanes of a program meet at
an element one of them stores. The launcher decides that from the launch arguments.

The reference is the kernel's own Python source, run with ``Source`` in the place of
``triton.language``: every load, store and reduction is done in numpy over the whole
block before the next one starts. It runs on host arrays that share memory exactly as
the launched tensors do, so the one source gives the expected result for separate
tensors and for every in-place layout. Values are small whole numbers unless a case
divides, so float32 sums and products are exact and the comparison is exact.

This file holds launches that must compute: separate tensors (never refused, never
changed), and shared storage where each lane keeps to its own elements, where a barrier
separates the loads from the stores, where one lane does all the work, and where atomic
updates commute. Stores of one element are in test_lane_order_single_store.py and the
launches that must be refused in test_lane_order_refusals.py.
"""

import collections
import platform
import types

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from tests.test_loop_pointer_contract import _conditional_pointer, _pointer_flow, _while_pointer

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available()),
    reason="Metal GPU required",
)
pytestmark = requires_mps

DEVICES = ("mps", "cpu")

# A pointer argument that is a view of pointer argument ``of``'s storage from element
# ``start`` on, read as ``dtype`` when one is given.
Shared = collections.namedtuple("Shared", "of start dtype", defaults=(0, None))
Case = collections.namedtuple("Case", "name kernel args grid rtol constants")


def case(name, kernel, args, grid=1, rtol=0.0, **constants):
    return Case(name, kernel, args, grid, rtol, constants)


def whole(count, seed=0, low=1, high=9, dtype=np.float32):
    return np.random.default_rng(seed).integers(low, high, count).astype(dtype)


def blank(count, dtype=np.float32):
    return np.full(count, -7, dtype=dtype)


def distinct(count, seed=0, dtype=np.float32):
    return np.random.default_rng(seed).permutation(count).astype(dtype)


# ------------------------------------------------------------- the source, in numpy
class _Lanes(np.ndarray):
    """A block of values, with the tensor methods the kernels call."""

    def to(self, dtype):
        return np.asarray(self).astype(dtype).view(_Lanes)

    def broadcast_to(self, *shape):
        return np.broadcast_to(np.asarray(self), shape).view(_Lanes)


def _lanes(values):
    return np.asarray(values).view(_Lanes)


class _Pointer:
    """Elements of a host array, counted from where the launched view starts."""

    __array_ufunc__ = None  # so that lanes + pointer is the pointer's addition, not numpy's

    def __init__(self, memory, offset=0):
        self.memory, self.offset = memory, offset

    def __add__(self, more):
        return _Pointer(self.memory, self.offset + np.asarray(more, dtype=np.int64))

    __radd__ = __add__

    def to(self, dtype):  # a pointer read as an integer, or as a pointer again, is the same place
        return self


class Source:
    """``triton.language`` for one program: each operation finishes over every lane."""

    constexpr, float32, int32, int64, int1 = int, np.float32, np.int32, np.int64, np.bool_

    @staticmethod
    def pointer_type(dtype):
        return dtype

    def __init__(self, program):
        self.program = program

    def program_id(self, axis):
        return self.program if axis == 0 else 0

    @staticmethod
    def arange(start, end):
        return _lanes(np.arange(start, end, dtype=np.int64))

    @staticmethod
    def zeros(shape, dtype):
        return _lanes(np.zeros(shape, dtype))

    @staticmethod
    def full(shape, value, dtype):
        return _lanes(np.full(shape, value, dtype))

    @staticmethod
    def broadcast_to(value, shape):
        return _lanes(np.broadcast_to(np.asarray(value), shape))

    @staticmethod
    def cast(value, dtype):
        return _lanes(np.asarray(value).astype(dtype))

    @staticmethod
    def where(condition, one, other):
        return _lanes(np.where(condition, one, other))

    @staticmethod
    def sum(value, axis):
        return _lanes(np.sum(value, axis=axis))

    @staticmethod
    def max(value, axis):
        return _lanes(np.max(value, axis=axis))

    @staticmethod
    def argmax(value, axis):
        return _lanes(np.argmax(value, axis=axis))

    @staticmethod
    def cumsum(value, axis):
        return _lanes(np.cumsum(value, axis=axis))

    @staticmethod
    def debug_barrier():
        pass

    @staticmethod
    def _cells(pointer, mask, *values):
        mask = True if mask is None else np.asarray(mask, dtype=bool)
        at, keep, *values = np.broadcast_arrays(np.asarray(pointer.offset, dtype=np.int64), mask, *values)
        inside = (at[keep] >= 0) & (at[keep] < pointer.memory.size)
        assert inside.all(), "the kernel reaches outside the tensor it was given"
        return (at, keep, *values)

    def load(self, pointer, mask=None, other=0, **hints):
        at, keep = self._cells(pointer, mask)
        out = np.full(at.shape, other, pointer.memory.dtype)
        out[keep] = pointer.memory[at[keep]]
        return _lanes(out)

    def store(self, pointer, value, mask=None, **hints):
        at, keep, value = self._cells(pointer, mask, np.asarray(value))
        pointer.memory[at[keep]] = value[keep].astype(pointer.memory.dtype)

    def _atomic(self, combine, pointer, value, mask):
        """The cell's value before the update is the source's only where no two lanes share a cell."""
        at, keep, value = self._cells(pointer, mask, np.asarray(value))
        old = np.zeros(at.shape, pointer.memory.dtype)
        old[keep] = pointer.memory[at[keep]]
        combine.at(pointer.memory, at[keep], value[keep].astype(pointer.memory.dtype))
        return _lanes(old)

    def atomic_add(self, pointer, value, mask=None):
        return self._atomic(np.add, pointer, value, mask)

    def atomic_max(self, pointer, value, mask=None):
        return self._atomic(np.maximum, pointer, value, mask)

    def atomic_min(self, pointer, value, mask=None):
        return self._atomic(np.minimum, pointer, value, mask)


def build(args, device):
    """(host values, launch values): arrays and tensors that share memory the same way."""
    hosts, tensors = [None] * len(args), [None] * len(args)
    for position, value in enumerate(args):
        if isinstance(value, np.ndarray):
            hosts[position] = _Pointer(value.copy())
            tensors[position] = torch.from_numpy(value.copy()).to(device)
        elif not isinstance(value, Shared):
            hosts[position] = tensors[position] = value
    for position, value in enumerate(args):
        if isinstance(value, Shared):
            memory, tensor = hosts[value.of].memory, tensors[value.of]
            if value.dtype is not None:
                memory = memory.view(value.dtype)
                tensor = tensor.view(torch.from_numpy(np.zeros(1, value.dtype)).dtype)
            hosts[position] = _Pointer(memory, value.start)
            tensors[position] = tensor[value.start :]
    return hosts, tensors


def launch(one, tensors, device):
    one.kernel[(one.grid,)](*tensors, **one.constants)
    if device == "mps":
        torch.mps.synchronize()


def source(one, hosts):
    """Run the kernel's source over the host arrays, one program after another."""
    written = one.kernel.fn
    for program in range(one.grid):
        scope = {**written.__globals__, "tl": Source(program)}
        for name, value in written.__globals__.items():
            if isinstance(value, triton.runtime.JITFunction):  # a function the kernel may call
                scope[name] = types.FunctionType(value.fn.__code__, scope, name)
        types.FunctionType(written.__code__, scope, written.__name__)(*hosts, **one.constants)


def same(one, hosts, tensors):
    for position, value in enumerate(one.args):
        if not isinstance(value, np.ndarray):
            continue
        got, want = tensors[position].detach().cpu().numpy(), hosts[position].memory
        if one.rtol:
            np.testing.assert_allclose(got, want, rtol=one.rtol, atol=0, err_msg=f"argument {position}")
        else:
            np.testing.assert_array_equal(got, want, err_msg=f"argument {position}")


def computes(one, device):
    hosts, tensors = build(one.args, device)
    launch(one, tensors, device)
    source(one, hosts)
    same(one, hosts, tensors)


def compiled(kernel):
    """The binary of ``kernel``'s most recent compilation."""
    device = list(kernel.device_caches.values())[0]
    return list(device[0].values())[-1]


# ------------------------------------------------------------ each lane its own element
@triton.jit
def _scale(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(Y + i, tl.load(X + i, mask=m) * 2.0, mask=m)


@triton.jit
def _chain(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, v + 1.0)
    u = tl.load(Y + i)
    tl.store(Y + i, u * 2.0)


@triton.jit
def _old_and_new(X, Z, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Z + i, v + 1.0)
    w = tl.load(Z + i)
    tl.store(Y + i, v + 2.0 * w)


@triton.jit
def _store_then_load(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a = tl.load(Y + i)
    tl.store(X + i, a + 3.0)
    tl.store(Y + i, tl.load(X + i) + 1.0)


@triton.jit
def _normalised(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, v / tl.sum(v, axis=0))


@triton.jit
def _scanned(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.cumsum(tl.load(X + i), axis=0))


@triton.jit
def _two_stores(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, v + 1.0)
    tl.store(Y + i, v + 2.0)


@triton.jit
def _branch(X, Y, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    if N > 0:
        tl.store(Y + i, tl.load(X + i) + 1.0)
    else:
        tl.store(Y + i, tl.load(X + i) - 1.0)


@triton.jit
def _chosen(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, tl.where(v > 4.5, v, -v))


@triton.jit
def _running(X, Y, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(K):
        acc += tl.load(X + i)
        tl.store(Y + i, acc)


@triton.jit
def _running_blocks(X, Y, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(K):
        acc += tl.load(X + i + k * BLOCK)
        tl.store(Y + i + k * BLOCK, acc)


@triton.jit
def _next_block(X, Y, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    for k in range(K):
        v = tl.load(X + i + k * BLOCK)
        tl.store(Y + i + (k + 1) * BLOCK, v + 1.0)


@triton.jit
def _carry_sum(X, Z, Y, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(K):
        v = tl.load(X + i)
        tl.store(Z + i, v + 1.0)
        acc += v * tl.sum(v, axis=0)
    tl.store(Y + i, acc)


@triton.jit
def _rows(X, Y, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, v * 2.0 + tl.max(v, axis=0))


@triton.jit
def _less_row_max(X, Y, R: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    v = tl.load(X + a)
    tl.store(Y + a, v - tl.max(v, axis=1)[:, None])


@triton.jit
def _affine_tile(X, Y, R: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Y + a, tl.load(X + a) * 2.0 + 1.0)


@triton.jit
def _scaled_by_cell(S, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    s = tl.load(S)
    tl.store(Y + i, tl.load(X + i) * s)


@triton.jit
def _wrapped(X, Y, M, BLOCK: tl.constexpr):
    a = tl.arange(0, BLOCK) % M
    tl.store(Y + a, tl.load(X + a) * 2.0)


@triton.jit
def _widened(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i).to(tl.int32) * 2)


@triton.jit
def _compared(Z, X, Y, SIZE: tl.constexpr):
    off = tl.arange(0, SIZE)
    x = tl.load(X)
    y = tl.load(Y + off)
    tl.store(Z + off, x <= y)


@triton.jit
def _added(Z, X, Y, SIZE: tl.constexpr):
    off = tl.arange(0, SIZE)
    tl.store(Z + off, tl.load(X + off) + tl.load(Y + off))


# ------------------------------------------------- a loaded value used after its store
@triton.jit
def _sum_after_store(X, Z, Q, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Z + i, v * 2.0)
    tl.store(Q, tl.sum(v, axis=0))


@triton.jit
def _reused_after_sum(X, Z, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Z + i, v * 2.0)
    s = tl.sum(v, axis=0)
    tl.store(Y + i, v + s)


@triton.jit
def _two_reductions(X, Z, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    s = tl.sum(v, axis=0)
    tl.store(Z + i, v / s)
    m = tl.max(v, axis=0)
    tl.store(Y + i, v - m)


@triton.jit
def _reused_each_trip(X, Z, Y, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    for _ in range(K):
        v = tl.load(X + i)
        tl.store(Z + i, v * 2.0)
        s = tl.sum(v, axis=0)
        tl.store(Y + i, v + s)


@triton.jit
def _rows_reused(X, Z, Y, R: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    v = tl.load(X + a)
    tl.store(Z + a, v * 2.0)
    s = tl.sum(v, axis=1)
    tl.store(Y + a, v + s[:, None])


@triton.jit
def _scan_after_store(X, Z, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Z + i, v * 2.0)
    tl.store(Y + i, tl.cumsum(v, axis=0))


@triton.jit
def _argmax_after_store(X, Z, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Z + i, -v)
    tl.store(Y, tl.argmax(v, axis=0).to(tl.float32))


@triton.jit
def _row_sums_after_store(X, Z, Y, R: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    v = tl.load(X + a)
    tl.store(Z + a, v * 2.0)
    tl.store(Y + tl.arange(0, R), tl.sum(v, axis=1))


@triton.jit
def _column_max_after_store(X, Z, Y, R: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    v = tl.load(X + a)
    tl.store(Z + a, v * 2.0)
    tl.store(Y + tl.arange(0, C), tl.max(v, axis=0))


# ----------------------------------------------------------- loads, a barrier, stores
@triton.jit
def _reversed_below_barrier(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.debug_barrier()
    tl.store(Y + (BLOCK - 1 - i), v)


@triton.jit
def _reversed_and_used(X, Z, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.debug_barrier()
    tl.store(Z + (BLOCK - 1 - i), v)
    tl.store(Y + i, v * 2.0)


@triton.jit
def _neighbour_below_barrier(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(X + i, tl.load(Y + i) + 1.0)
    tl.debug_barrier()
    tl.store(Y + i, tl.load(X + (i + 1) % BLOCK))


@triton.jit
def _reversed_and_summed(X, Z, Q, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.debug_barrier()
    tl.store(Z + (BLOCK - 1 - i), v)
    tl.store(Q, tl.sum(v, axis=0))


@triton.jit
def _neighbour_sum(X, Y, Q, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(X + i, tl.load(Y + i) + 1.0)
    tl.debug_barrier()
    tl.store(Q, tl.sum(tl.load(X + (i + 1) % BLOCK) * (i % 7).to(tl.float32), axis=0))


@triton.jit
def _max_then_neighbour(A, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a = tl.load(A + i)
    s = tl.max(a, axis=0)
    tl.store(X + i, a)
    tl.debug_barrier()
    tl.store(Y + i, tl.load(X + (i + 1) % BLOCK) + s)


@triton.jit
def _tile3(X, Y, R: tl.constexpr):
    a = tl.arange(0, R)[:, None, None] * R * R + tl.arange(0, R)[None, :, None] * R + tl.arange(0, R)[None, None, :]
    tl.store(Y + a, tl.load(X + a) * 2.0)


# ------------------------------------------------------- one lane, loops and pointers
@triton.jit
def _halving(C, X, n):
    k = n
    while k > 0:
        tl.store(C, tl.load(C) + tl.load(X + k))
        k = k // 2


@triton.jit
def _walk(A, O, R, BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    pointer = A + x
    acc = tl.zeros((BLOCK,), tl.float32)
    i = 0
    while i < R:
        acc += tl.load(pointer)
        pointer += BLOCK
        i += 1
    tl.store(O + x, acc)


@triton.jit
def _gathered(X, P, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + tl.load(P + i))
    tl.store(Y + i, v)


@triton.jit
def _loaded_trips(T, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    n = tl.load(T)
    for k in range(n):
        tl.store(Y + i + k * BLOCK, tl.load(X + i + k * BLOCK) + 1.0)


# ------------------------------------------------------------------- atomic updates
@triton.jit
def _atomic_max(A, O, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_max(O + (i % K), tl.load(A + i))


@triton.jit
def _atomic_min(A, O, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_min(O + (i % K), tl.load(A + i))


@triton.jit
def _atomic_max_masked(A, O, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_max(O + (i % K), tl.load(A + i), mask=(i % 3) != 0)


@triton.jit
def _add_then_add(A, Bv, O, P, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_add(O + (i % K), tl.load(A + i))
    tl.atomic_add(P + ((i + 1) % K), tl.load(Bv + i))


@triton.jit
def _add_then_max(A, Bv, O, P, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_add(O + (i % K), tl.load(A + i))
    tl.atomic_max(P + ((i + 1) % K), tl.load(Bv + i))


@triton.jit
def _max_then_min(A, Bv, O, P, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    tl.atomic_max(O + (i % K), tl.load(A + i))
    tl.atomic_min(P + ((i + 1) % K), tl.load(Bv + i))


@triton.jit
def _add_returning(A, O, R, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    old = tl.atomic_add(O + (i % K), tl.load(A + i))
    tl.store(R + i, old)


@triton.jit
def _max_returning(A, O, R, N: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, N)
    old = tl.atomic_max(O + (i % K), tl.load(A + i))
    tl.store(R + i, old)


# ---------------------------------------------------------------------- the launches
def _own_element():
    out = []
    for block in (16, 256, 1024, 8192):
        x, y = whole(block), blank(block)
        out += [
            case(f"chain, separate, {block}", _chain, [x, y], BLOCK=block),
            case(f"chain, in place, {block}", _chain, [x, Shared(0)], BLOCK=block),
            case(f"old and new value, separate, {block}", _old_and_new, [x, y, y], BLOCK=block),
            case(f"old and new value, stored over the input, {block}", _old_and_new, [x, Shared(0), y], BLOCK=block),
            case(f"store then load, separate, {block}", _store_then_load, [y, x], BLOCK=block),
            case(f"store then load, one tensor, {block}", _store_then_load, [x, Shared(0)], BLOCK=block),
            case(f"two stores, in place, {block}", _two_stores, [x, Shared(0)], BLOCK=block),
            case(f"chosen, in place, {block}", _chosen, [x, Shared(0)], BLOCK=block),
            case(f"scaled by a cell, separate, {block}", _scaled_by_cell, [whole(1, 5), x, y], BLOCK=block),
            # The scalar is cell 0 of the storage; the block is read and written from cell 1 on.
            case(
                f"scaled by a cell of its own storage, {block}",
                _scaled_by_cell,
                [whole(block + 1), Shared(0, 1), Shared(0, 1)],
                BLOCK=block,
            ),
        ]
    for block in (64, 1024, 4096):
        x, y = whole(block), blank(block)
        out += [
            case(f"branch taken, in place, {block}", _branch, [x, Shared(0), 1], BLOCK=block),
            case(f"branch not taken, in place, {block}", _branch, [x, Shared(0), 0], BLOCK=block),
            case(f"scan, separate, {block}", _scanned, [x, y], BLOCK=block),
            case(f"scan, in place, {block}", _scanned, [x, Shared(0)], BLOCK=block),
            case(f"normalised, separate, {block}", _normalised, [x, y], rtol=1e-6, BLOCK=block),
            case(f"normalised, in place, {block}", _normalised, [x, Shared(0)], rtol=1e-6, BLOCK=block),
            case(f"wrapped at the block, in place, {block}", _wrapped, [x, Shared(0), block], BLOCK=block),
            case(f"wrapped past the block, separate, {block}", _wrapped, [x, y, block + 5], BLOCK=block),
            case(
                f"widened, separate, {block}",
                _widened,
                [whole(block, dtype=np.int16), blank(block, np.int32)],
                BLOCK=block,
            ),
        ]
    # Programs over a length that is not a multiple of the block; the tail stays as it was.
    for count, block in ((1000, 256), (2500, 2048), (5000, 16), (20000, 8192)):
        grid = -(-count // block)
        x, y = whole(count + 3), blank(count + 3)
        out += [
            case(f"scale, separate, {count} by {block}", _scale, [x, y, count], grid, BLOCK=block),
            case(f"scale, in place, {count} by {block}", _scale, [x, Shared(0), count], grid, BLOCK=block),
            case(
                f"scale, two halves of one storage, {count} by {block}",
                _scale,
                [whole(2 * count), Shared(0, count), count],
                grid,
                BLOCK=block,
            ),
        ]
    for block in (64, 1024, 8192):
        x = whole(4 * block)
        out += [
            case(f"running sum, separate, {block}", _running, [x, blank(block), 3], BLOCK=block),
            case(f"running sum, in place, {block}", _running, [x, Shared(0), 3], BLOCK=block),
            case(f"running sum over blocks, in place, {block}", _running_blocks, [x, Shared(0), 3], BLOCK=block),
            case(f"next block, separate, {block}", _next_block, [x, blank(4 * block), 3], BLOCK=block),
            # Each trip loads what the trip before stored, lane by lane.
            case(f"next block, in place, {block}", _next_block, [x, Shared(0), 3], BLOCK=block),
            case(f"rows, separate, {block}", _rows, [x, blank(4 * block)], 3, BLOCK=block),
            case(f"rows, in place, {block}", _rows, [x, Shared(0)], 3, BLOCK=block),
        ]
    for block in (64, 256, 4096):
        x = whole(block)
        out += [
            case(f"carried sum, separate, {block}", _carry_sum, [x, blank(block), blank(block), 2], BLOCK=block),
            case(
                f"carried sum, stored over the input, {block}", _carry_sum, [x, Shared(0), blank(block), 2], BLOCK=block
            ),
        ]
    for rows, columns in ((8, 8), (16, 64), (32, 32)):
        x, y = whole(rows * columns), blank(rows * columns)
        out += [
            case(f"less the row maximum, separate, {rows}x{columns}", _less_row_max, [x, y], R=rows, C=columns),
            case(f"less the row maximum, in place, {rows}x{columns}", _less_row_max, [x, Shared(0)], R=rows, C=columns),
            case(f"tile, in place, {rows}x{columns}", _affine_tile, [x, Shared(0)], R=rows, C=columns),
        ]
    for size in (4, 8):
        x = whole(size**3)
        out += [
            case(f"cube, separate, {size}", _tile3, [x, blank(size**3)], R=size),
            case(f"cube, in place, {size}", _tile3, [x, Shared(0)], R=size),
        ]
    # The shapes of upstream's binary-operator tests, as upstream launches them.
    for dtype in (np.int16, np.int64, np.float32):
        x, y = np.arange(128).astype(dtype), np.arange(128)[::-1].astype(dtype)
        out += [
            case(
                f"compared with a scalar load, {dtype.__name__}", _compared, [np.zeros(128, bool), x + 60, y], SIZE=128
            ),
            case(f"added, {dtype.__name__}", _added, [blank(128, dtype), x, y], SIZE=128),
            case(f"added into an operand, {dtype.__name__}", _added, [Shared(1), x, y], SIZE=128),
        ]
    return out


def _used_after_store():
    out = []
    for block in (64, 1024, 4096):
        x, y, q = whole(block), blank(block), blank(1)
        out += [
            case(f"sum after the store, separate, {block}", _sum_after_store, [x, y, q], BLOCK=block),
            case(f"sum after the store, in place, {block}", _sum_after_store, [x, Shared(0), q], BLOCK=block),
            case(f"reused after a sum, separate, {block}", _reused_after_sum, [x, y, y], BLOCK=block),
            case(f"reused each trip, in place, {block}", _reused_each_trip, [x, Shared(0), y, 2], BLOCK=block),
            case(f"scan after the store, in place, {block}", _scan_after_store, [x, Shared(0), y], BLOCK=block),
        ]
    # One pass over the block: the lane still holds what it loaded.
    for block in (64, 256):
        x, y = whole(block), blank(block)
        out += [
            case(
                f"reused after a sum, stored over the input, {block}", _reused_after_sum, [x, Shared(0), y], BLOCK=block
            ),
            case(f"reused after a sum, one tensor, {block}", _reused_after_sum, [x, Shared(0), Shared(0)], BLOCK=block),
            case(f"two reductions, separate, {block}", _two_reductions, [x, y, y], rtol=1e-6, BLOCK=block),
            case(
                f"two reductions, stored over the input, {block}",
                _two_reductions,
                [x, Shared(0), y],
                rtol=1e-6,
                BLOCK=block,
            ),
        ]
    for block in (64, 256, 1024):
        x = distinct(block)
        out.append(
            case(
                f"argmax after the store, in place, {block}", _argmax_after_store, [x, Shared(0), blank(1)], BLOCK=block
            )
        )
    for rows, columns in ((8, 8), (16, 64), (32, 32)):
        x, y = whole(rows * columns), blank(rows * columns)
        out += [
            case(f"rows reused, in place, {rows}x{columns}", _rows_reused, [x, Shared(0), y], R=rows, C=columns),
            case(
                f"row sums after the store, in place, {rows}x{columns}",
                _row_sums_after_store,
                [x, Shared(0), y],
                R=rows,
                C=columns,
            ),
            case(
                f"column maximum after the store, in place, {rows}x{columns}",
                _column_max_after_store,
                [x, Shared(0), y],
                R=rows,
                C=columns,
            ),
        ]
    return out


def _below_a_barrier():
    out = []
    # Every lane is a thread here, so the barrier holds every store until every load is done.
    for block in (64, 256, 1024):
        x, y = whole(block), blank(block)
        out += [
            case(f"reversed below a barrier, separate, {block}", _reversed_below_barrier, [x, y], BLOCK=block),
            case(f"reversed below a barrier, in place, {block}", _reversed_below_barrier, [x, Shared(0)], BLOCK=block),
            case(f"reversed and used, in place, {block}", _reversed_and_used, [x, Shared(0), y], BLOCK=block),
            case(
                f"reversed and summed, in place, {block}", _reversed_and_summed, [x, Shared(0), blank(1)], BLOCK=block
            ),
            case(f"neighbour below a barrier, {block}", _neighbour_below_barrier, [y, x], BLOCK=block),
        ]
    for block in (64, 128):
        x, y = whole(block), blank(block)
        out += [
            case(f"neighbour sum, {block}", _neighbour_sum, [y, x, blank(1)], BLOCK=block),
            case(f"maximum then neighbour, {block}", _max_then_neighbour, [x, y, blank(block)], BLOCK=block),
        ]
    # More lanes than threads: the barrier orders nothing, and with separate tensors nothing has to be ordered.
    for block in (2048, 4096, 8192):
        x, y = whole(block), blank(block)
        out += [
            case(f"reversed below a barrier, separate, {block}", _reversed_below_barrier, [x, y], BLOCK=block),
            case(f"reversed and used, separate, {block}", _reversed_and_used, [x, y, y], BLOCK=block),
            case(f"reversed and summed, separate, {block}", _reversed_and_summed, [x, y, blank(1)], BLOCK=block),
        ]
    return out


def _one_lane_and_pointers():
    out = [
        case("halving walk, separate", _halving, [whole(1, 3), whole(41), 40]),
        case("halving walk, the total in cell 0", _halving, [Shared(1), whole(41), 40]),
        case("while walk, separate", _walk, [whole(96), blank(32), 3], BLOCK=32),
        case("gathered, separate, 64", _gathered, [whole(64), distinct(64, 1, np.int32), blank(64)], BLOCK=64),
        case(
            "gathered, separate, 1024", _gathered, [whole(1024), distinct(1024, 1, np.int32), blank(1024)], BLOCK=1024
        ),
        case(
            "chosen pointer, first, separate", _conditional_pointer, [whole(32), whole(32, 1), blank(32), 0], BLOCK=32
        ),
        case(
            "chosen pointer, second, separate", _conditional_pointer, [whole(32), whole(32, 1), blank(32), 2], BLOCK=32
        ),
    ]
    a, b = whole(96), whole(96, 1)
    for scalar in (False, True):
        for mode in (0, 1, 2):
            out.append(
                case(
                    f"pointer flow, separate, mode {mode}, scalar {scalar}",
                    _pointer_flow,
                    [a, b, blank(64), 2],
                    BLOCK=32,
                    SCALAR=scalar,
                    MODE=mode,
                )
            )
        out += [
            # Only the first argument's blocks are loaded, and the stores land in the second's.
            case(
                f"pointer flow, stored over the unread input, scalar {scalar}",
                _pointer_flow,
                [a, b, Shared(1), 2],
                BLOCK=32,
                SCALAR=scalar,
                MODE=0,
            ),
            case(
                f"while pointer, separate, scalar {scalar}",
                _while_pointer,
                [a, b, blank(64), 2],
                BLOCK=32,
                SCALAR=scalar,
            ),
        ]
    return out


def _atomics():
    out = []
    for lanes in (64, 8192):
        for cells in (1, 37):
            a, b, o = whole(lanes), whole(lanes, 1), whole(cells, 2)
            name = f"{lanes} lanes on {cells}"
            out += [
                case(f"atomic max, {name}", _atomic_max, [a, o], N=lanes, K=cells),
                case(f"atomic min, {name}", _atomic_min, [a, o], N=lanes, K=cells),
                case(f"atomic max, masked, {name}", _atomic_max_masked, [a, o], N=lanes, K=cells),
                # Integer adds are exact in any order. The float twin is refused: the order of
                # float additions is their rounding (tests/test_lane_order_refusals.py).
                case(
                    f"add then add, one tensor, int32, {name}",
                    _add_then_add,
                    [
                        whole(lanes, dtype=np.int32),
                        whole(lanes, 1, dtype=np.int32),
                        whole(cells, 2, dtype=np.int32),
                        Shared(2),
                    ],
                    N=lanes,
                    K=cells,
                ),
                case(f"add then add, separate, {name}", _add_then_add, [a, b, o, o], N=lanes, K=cells),
                case(f"add then max, separate, {name}", _add_then_max, [a, b, o, o], N=lanes, K=cells),
                case(f"max then min, separate, {name}", _max_then_min, [a, b, o, o], N=lanes, K=cells),
            ]
        # A lane a cell: the value an atomic returns is the source's.
        a, o = whole(lanes), whole(lanes, 2)
        out += [
            case(f"atomic add returning, {lanes} lanes", _add_returning, [a, o, blank(lanes)], N=lanes, K=lanes),
            case(f"atomic max returning, {lanes} lanes", _max_returning, [a, o, blank(lanes)], N=lanes, K=lanes),
        ]
    return out


COMPUTED = _own_element() + _used_after_store() + _below_a_barrier() + _one_lane_and_pointers() + _atomics()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("one", COMPUTED, ids=[one.name for one in COMPUTED])
def test_the_launch_computes_what_the_source_computes(one, device):
    computes(one, device)
