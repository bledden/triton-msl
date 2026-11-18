"""Launches the lane-order check refuses, and the same kernels computing with separate tensors.

A Triton program finishes each operation over every lane before the next one starts; the
ordinary shader keeps only each lane's own order. Every launch here lets two lanes of one
program meet at an element one of them stores, so the shader's result depends on which
thread runs first: before the check these launches returned values the source does not
compute (an in-place reversal kept half its elements). The launcher now raises before
anything is submitted, with a message that says which arguments share the memory and why
no proof was found, and the tensors are left as they were.

Each refused launch has a partner: the same kernel on tensors that do not share (or, where
the meeting is inside one tensor, at a width where the lanes do not meet). The partner
must compute what the source computes, so a refusal cannot hide a broken kernel. The
reference is the source interpreter of test_lane_order_launch.py.

The reason texts are written out here, not imported, so that this file is collected on
a tree without the check and its tests fail one by one there.
"""

import collections

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from tests.test_lane_order_launch import (
    DEVICES,
    Shared,
    _add_then_max,
    _conditional_pointer,
    _gathered,
    _loaded_trips,
    _max_then_min,
    _max_then_neighbour,
    _neighbour_sum,
    _pointer_flow,
    _reused_after_sum,
    _reversed_and_summed,
    _reversed_and_used,
    _reversed_below_barrier,
    _running_blocks,
    _scale,
    _scaled_by_cell,
    _walk,
    _while_pointer,
    _widened,
    _wrapped,
    blank,
    build,
    case,
    computes,
    distinct,
    launch,
    requires_mps,
    same,
    whole,
)
from tests.test_lane_order_single_store import _increment_copy, _scale_and_bump
from triton_msl.errors import MetalNonRecoverableError

pytestmark = requires_mps

MEET = "two lanes of one program reach the same element and one of them stores it"
NARROW = "an access narrower than the block reaches an element another lane also reaches"
ORDER = "accesses the shader performs in another order than the source reach the same element"
OPAQUE = "an address depends on a value the check cannot compute before the launch"
SIZE = "the arguments that share memory differ in element size or are not aligned to it"
EVALUATE = "an address uses an operation or an argument the check does not evaluate"
UNBOUNDED = "an access sits in a loop whose trips are not known before the launch"
UNTRACED = "the shader does not perform these accesses one by one in the source's order"
BOUNDS = "a loop around the accesses has bounds that are not fixed before the launch"
BUDGET = "there are too many addresses to compare"
CALL = "the kernel's accesses could not be described (a call this plan cannot follow)"
STORAGE = "an argument's storage could not be read, so it may share memory with any other"

# ``names``: the arguments the message must name, or None where it is not pinned (a pointer
# the check could not follow is named with the group although it has its own storage).
Refusal = collections.namedtuple("Refusal", "case reason names partner")


# ----------------------------------------------------------------------------- kernels
@triton.jit
def _reversed(X, Y, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < N
    tl.store(Y + (N - 1 - i), tl.load(X + i, mask=m), mask=m)


@triton.jit
def _rows_reversed(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    row = tl.program_id(0) * BLOCK
    tl.store(Y + row + (BLOCK - 1 - i), tl.load(X + row + i))


@triton.jit
def _transposed(X, Y, R: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, R)[None, :]
    tl.store(Y + c * R + r, tl.load(X + r * R + c))


@triton.jit
def _cube_reversed(X, Y, R: tl.constexpr):
    a = tl.arange(0, R)[:, None, None] * R * R + tl.arange(0, R)[None, :, None] * R + tl.arange(0, R)[None, None, :]
    tl.store(Y + (R * R * R - 1 - a), tl.load(X + a) * 2.0)


@triton.jit
def _crossed(A, Bv, Y, Z, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(A + i))
    tl.store(Z + (BLOCK - 1 - i), tl.load(Bv + i))


@triton.jit
def _through_an_integer(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    p = X.to(tl.int64).to(tl.pointer_type(tl.float32))
    tl.store(Y + i, tl.load(p + i) + 1.0)


@triton.jit
def _scalar_of_stored(X, Z, S, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Z + i, v * 2.0)
    s = tl.load(S)
    tl.store(Y + i, v + s)


@triton.jit
def _shifted_by_float(X, Y, S, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    d = tl.cast(S, tl.int32)
    tl.store(Y + i + d, tl.load(X + i) + 1.0)


@triton.jit
def _folded_trips(P, O, X, Q, BLOCK: tl.constexpr, FIRST: tl.constexpr, SMALL: tl.constexpr, FOLD: tl.constexpr):
    t = tl.arange(0, FIRST)
    j = tl.arange(0, SMALL)
    w = tl.arange(0, BLOCK)
    tl.store(P + t, tl.load(X + t))
    for k in range(2):
        keep = ((j < FOLD) & (k == 1)) | ((j >= FOLD) & (k == 0))
        tl.store(O + j % FOLD, tl.load(X + j), mask=keep)
    tl.store(Q + w, tl.load(X + w))


@triton.jit(noinline=True)
def _d9(X, Y, k):
    tl.store(Y + k, tl.load(X + k) + 1.0)


@triton.jit(noinline=True)
def _d8(X, Y, k):
    _d9(X, Y, k)


@triton.jit(noinline=True)
def _d7(X, Y, k):
    _d8(X, Y, k)


@triton.jit(noinline=True)
def _d6(X, Y, k):
    _d7(X, Y, k)


@triton.jit(noinline=True)
def _d5(X, Y, k):
    _d6(X, Y, k)


@triton.jit(noinline=True)
def _d4(X, Y, k):
    _d5(X, Y, k)


@triton.jit(noinline=True)
def _d3(X, Y, k):
    _d4(X, Y, k)


@triton.jit(noinline=True)
def _d2(X, Y, k):
    _d3(X, Y, k)


@triton.jit(noinline=True)
def _d1(X, Y, k):
    _d2(X, Y, k)


@triton.jit(noinline=True)
def _d0(X, Y, k):
    _d1(X, Y, k)


@triton.jit
def _one_call(X, Y):
    _d9(X, Y, tl.program_id(0))


@triton.jit
def _nine_calls(X, Y):
    _d1(X, Y, tl.program_id(0))


@triton.jit
def _call_beside_a_block(X, Y, C, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * 2.0)
    _d9(C, C, tl.program_id(0))  # every lane runs the callee: its load and its store of one cell


@triton.jit
def _ten_calls_beside_a_block(X, Y, C, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * 2.0)
    _d0(C, C, tl.program_id(0))


# ------------------------------------------------------------------------ the launches
def _meeting():
    out = []
    for block in (16, 64, 1024, 8192):
        x, count = whole(block), block - 3
        out.append(
            Refusal(
                case(f"reversed in place, {block}", _reversed, [x, Shared(0), count], BLOCK=block),
                MEET,
                "X, Y",
                case(f"reversed, separate, {block}", _reversed, [x, blank(block), count], BLOCK=block),
            )
        )
    for block in (64, 4096):
        x, y, longer = whole(block), blank(block), whole(block + 1)
        a, b, cells = whole(block), whole(block, 1), whole(37, 2)
        scaled = case(f"scale, separate, {block}", _scale, [longer, blank(block + 1), block], BLOCK=block)
        crossed = case(f"crossed stores, separate, {block}", _crossed, [a, b, y, y], BLOCK=block)
        halved = case(f"wrapped at half the block, separate, {block}", _wrapped, [x, y, block // 2], BLOCK=block)
        rows = case(
            f"rows reversed, separate, {block}", _rows_reversed, [whole(3 * block), blank(3 * block)], 3, BLOCK=block
        )
        integer = case(
            f"through an integer, separate, {block}", _through_an_integer, [longer, blank(block)], BLOCK=block
        )
        out += [
            Refusal(
                case(f"scale onto the next cell, {block}", _scale, [longer, Shared(0, 1), block], BLOCK=block),
                MEET,
                "X, Y",
                scaled,
            ),
            Refusal(
                case(f"scale from the next cell, {block}", _scale, [Shared(1, 1), longer, block], BLOCK=block),
                MEET,
                "X, Y",
                scaled,
            ),
            Refusal(
                case(f"crossed stores into one tensor, {block}", _crossed, [a, b, y, Shared(2)], BLOCK=block),
                MEET,
                "Y, Z",
                crossed,
            ),
            Refusal(
                case(
                    f"wrapped at half the block, in place, {block}", _wrapped, [x, Shared(0), block // 2], BLOCK=block
                ),
                MEET,
                "X, Y",
                halved,
            ),
            Refusal(
                case(f"rows reversed in place, {block}", _rows_reversed, [whole(3 * block), Shared(0)], 3, BLOCK=block),
                MEET,
                "X, Y",
                rows,
            ),
            Refusal(
                case(
                    f"through an integer, onto the next cell, {block}",
                    _through_an_integer,
                    [longer, Shared(0, 1)],
                    BLOCK=block,
                ),
                MEET,
                "X, Y",
                integer,
            ),
            # Two atomic updates that do not commute with each other, on one tensor's cells.
            Refusal(
                case(f"add then max, one tensor, {block}", _add_then_max, [a, b, cells, Shared(2)], N=block, K=37),
                MEET,
                "O, P",
                case(f"add then max, separate, {block}", _add_then_max, [a, b, cells, cells], N=block, K=37),
            ),
            Refusal(
                case(f"max then min, one tensor, {block}", _max_then_min, [a, b, cells, Shared(2)], N=block, K=37),
                MEET,
                "O, P",
                case(f"max then min, separate, {block}", _max_then_min, [a, b, cells, cells], N=block, K=37),
            ),
        ]
    for size in (16, 32):
        x = whole(size * size)
        out.append(
            Refusal(
                case(f"transposed in place, {size}", _transposed, [x, Shared(0)], R=size),
                MEET,
                "X, Y",
                case(f"transposed, separate, {size}", _transposed, [x, blank(size * size)], R=size),
            )
        )
    for size in (4, 8):
        x = whole(size**3)
        out.append(
            Refusal(
                case(f"cube reversed in place, {size}", _cube_reversed, [x, Shared(0)], R=size),
                MEET,
                "X, Y",
                case(f"cube reversed, separate, {size}", _cube_reversed, [x, blank(size**3)], R=size),
            )
        )
    return out


def _above_the_threads():
    """A barrier orders threads. With more lanes than threads it does not order the lanes."""
    out = []
    for block in (2048, 4096, 8192):
        x, y, q = whole(block), blank(block), blank(1)
        out += [
            Refusal(
                case(
                    f"reversed below a barrier, in place, {block}", _reversed_below_barrier, [x, Shared(0)], BLOCK=block
                ),
                MEET,
                "X, Y",
                case(f"reversed below a barrier, separate, {block}", _reversed_below_barrier, [x, y], BLOCK=block),
            ),
            Refusal(
                case(f"reversed and used, in place, {block}", _reversed_and_used, [x, Shared(0), y], BLOCK=block),
                MEET,
                "X, Z",
                case(f"reversed and used, separate, {block}", _reversed_and_used, [x, y, y], BLOCK=block),
            ),
            Refusal(
                case(f"reversed and summed, in place, {block}", _reversed_and_summed, [x, Shared(0), q], BLOCK=block),
                MEET,
                "X, Z",
                case(f"reversed and summed, separate, {block}", _reversed_and_summed, [x, y, q], BLOCK=block),
            ),
        ]
    # The lanes meet inside one tensor, whatever the others are: a lane loads, below the barrier, the
    # cell its neighbour stored above it. The reduction makes each thread walk several lanes from 256 on.
    narrow = whole(64)
    for block in (256, 4096):
        x, y = whole(block), blank(block)
        out += [
            Refusal(
                case(f"neighbour sum, {block}", _neighbour_sum, [y, x, blank(1)], BLOCK=block),
                MEET,
                "X",
                case("neighbour sum, 64", _neighbour_sum, [blank(64), narrow, blank(1)], BLOCK=64),
            ),
            Refusal(
                case(f"maximum then neighbour, {block}", _max_then_neighbour, [x, y, blank(block)], BLOCK=block),
                MEET,
                "X",
                case("maximum then neighbour, 64", _max_then_neighbour, [narrow, blank(64), blank(64)], BLOCK=64),
            ),
        ]
    return out


def _one_cell():
    out = []
    one_lane = case("scaled then bumped, one lane", _scale_and_bump, [whole(1, 3), whole(4), blank(4)], BLOCK=1)
    # With one lane the call is in order. On a cpu tensor this partner also needs the cell the called
    # function stored to be copied back, beside the output the entry function stored.
    one_program = case(
        "a call beside a block, one lane", _call_beside_a_block, [whole(4), blank(4), whole(4, 3)], 4, BLOCK=1
    )
    for block in (64, 4096):
        x, y, cell = whole(block), blank(block), whole(1, 4)
        out += [
            # The scalar every lane loads is a cell some lane stores.
            Refusal(
                case(f"scaled by a cell it stores, {block}", _scaled_by_cell, [x, Shared(0), Shared(0)], BLOCK=block),
                NARROW,
                "S, X, Y",
                case(f"scaled by a cell, separate, {block}", _scaled_by_cell, [cell, x, y], BLOCK=block),
            ),
            Refusal(
                case(f"scalar of a stored cell, {block}", _scalar_of_stored, [x, y, Shared(1), y], BLOCK=block),
                NARROW,
                "Z, S",
                case(f"scalar of a cell, separate, {block}", _scalar_of_stored, [x, y, cell, y], BLOCK=block),
            ),
            # Separate tensors: lane 0 stores the cell every other lane still has to load.
            Refusal(
                case(f"scaled then bumped, {block}", _scale_and_bump, [whole(1, 3), x, y], BLOCK=block),
                NARROW,
                "S",
                one_lane,
            ),
            # Separate tensors: every lane runs the called function, so every lane loads and stores its cell.
            Refusal(
                case(
                    f"a call beside a block, {block}",
                    _call_beside_a_block,
                    [whole(4 * block), blank(4 * block), whole(4, 3)],
                    4,
                    BLOCK=block,
                ),
                NARROW,
                "C",
                one_program,
            ),
        ]
    # Inductor's counter copy, incremented in place: every lane loads the cell and every lane stores it.
    for block in (2, 16, 2048):
        count = whole(1, 3, dtype=np.int64)
        out.append(
            Refusal(
                case(f"increment copied over its input, {block}", _increment_copy, [count, Shared(0), 1], XBLOCK=block),
                NARROW,
                "in_ptr0, out_ptr1",
                case(
                    f"increment copied, separate, {block}",
                    _increment_copy,
                    [count, blank(1, np.int64), 1],
                    XBLOCK=block,
                ),
            )
        )
    return out


def _folded():
    """One store in a loop, narrower than the block: a cell's two trips are two lanes' stores.

    The source's second trip stores last. The shader's threads each run both trips, and
    nothing makes the thread of the first trip's store finish before the other's second.
    """
    out = []
    for block, small in ((256, 64), (2048, 64), (4096, 1024)):
        # The narrow store before the loop has the looped store's width, half of it, or twice it.
        for first in (small, small // 2, 2 * small):
            x = distinct(block)
            args = [blank(first), blank(small), x, blank(block)]
            sizes = dict(BLOCK=block, FIRST=first, SMALL=small)
            name = f"{small} of {block} after {first}"
            out.append(
                Refusal(
                    case(f"folded trips, {name}", _folded_trips, args, FOLD=small // 2, **sizes),
                    NARROW,
                    "O",
                    case(f"unfolded trips, {name}", _folded_trips, args, FOLD=small, **sizes),
                )
            )
    return out


def _other_reasons():
    out = []
    # The reduction makes each thread walk several lanes, so the lane's loaded value is loaded again after the store.
    for block in (1024, 4096):
        x, y = whole(block), blank(block)
        separate = case(f"reused after a sum, separate, {block}", _reused_after_sum, [x, y, y], BLOCK=block)
        out += [
            Refusal(
                case(
                    f"reused after a sum, stored over the input, {block}",
                    _reused_after_sum,
                    [x, Shared(0), y],
                    BLOCK=block,
                ),
                ORDER,
                "X, Z",
                separate,
            ),
            Refusal(
                case(
                    f"reused after a sum, one tensor, {block}",
                    _reused_after_sum,
                    [x, Shared(0), Shared(0)],
                    BLOCK=block,
                ),
                ORDER,
                "X, Z, Y",
                separate,
            ),
        ]
    for block in (64, 4096):
        x, y, longer, places = whole(block), blank(block), whole(block + 1), distinct(block, 1, np.int32)
        narrow, wide = whole(block, dtype=np.int16), whole(block, dtype=np.int32)
        out += [
            Refusal(
                case(f"gathered in place, {block}", _gathered, [x, places, Shared(0)], BLOCK=block),
                OPAQUE,
                "X, Y",
                case(f"gathered, separate, {block}", _gathered, [x, places, y], BLOCK=block),
            ),
            Refusal(
                case(
                    f"shifted by a float, in place, {block}", _shifted_by_float, [longer, Shared(0), 1.0], BLOCK=block
                ),
                OPAQUE,
                "X, Y",
                case(
                    f"shifted by a float, separate, {block}",
                    _shifted_by_float,
                    [longer, blank(block + 1), 1.0],
                    BLOCK=block,
                ),
            ),
            # Sixteen-bit loads over the bytes of the thirty-two-bit cells the kernel stores.
            Refusal(
                case(f"widened over its own bytes, {block}", _widened, [Shared(1, 0, np.int16), wide], BLOCK=block),
                SIZE,
                "X, Y",
                case(f"widened, separate, {block}", _widened, [narrow, blank(block, np.int32)], BLOCK=block),
            ),
        ]
    a, b, y = whole(97), whole(97, 1), blank(64)
    walked = case("while walk, separate", _walk, [a, blank(32), 3], BLOCK=32)
    chosen = [
        case(f"chosen pointer, separate, {n}", _conditional_pointer, [whole(33), whole(33, 1), blank(33), n], BLOCK=32)
        for n in (0, 2)
    ]
    out += [
        Refusal(
            case("while walk onto the next cell", _walk, [a, Shared(0, 1), 3], BLOCK=32), UNBOUNDED, "A, O", walked
        ),
        Refusal(
            case(
                "chosen pointer, onto the next cell",
                _conditional_pointer,
                [whole(33), whole(33, 1), Shared(0, 1), 0],
                BLOCK=32,
            ),
            UNTRACED,
            None,
            chosen[0],
        ),
        Refusal(
            case(
                "chosen pointer, onto the other's next cell",
                _conditional_pointer,
                [whole(33), whole(33, 1), Shared(1, 1), 2],
                BLOCK=32,
            ),
            UNTRACED,
            None,
            chosen[1],
        ),
        # No partner: the kernel is refused with separate tensors too (reported, not asserted).
        Refusal(
            case(
                "loaded trip count, onto the next cell",
                _loaded_trips,
                [np.array([3], np.int32), whole(193), Shared(1, 1)],
                BLOCK=64,
            ),
            BOUNDS,
            "X, Y",
            None,
        ),
        Refusal(
            case(
                "running sum over 300 blocks, onto the next cell",
                _running_blocks,
                [whole(8192 * 300 + 1), Shared(0, 1), 300],
                BLOCK=8192,
            ),
            BUDGET,
            "X, Y",
            case(
                "running sum over 300 blocks, separate",
                _running_blocks,
                [whole(8192 * 300), blank(8192 * 300), 300],
                BLOCK=8192,
            ),
        ),
        # No partner and no values: the source divides by zero here, so only the refusal is defined.
        Refusal(
            case("wrapped at zero, in place", _wrapped, [whole(64), Shared(0), 0], BLOCK=64), EVALUATE, "X, Y", None
        ),
        # Ten calls down the plan stops following; every lane of the block runs the callee's load and store.
        Refusal(
            case(
                "ten calls beside a block",
                _ten_calls_beside_a_block,
                [whole(256), blank(256), whole(4, 3)],
                4,
                BLOCK=64,
            ),
            CALL,
            "",
            case("nine calls, in place", _nine_calls, [whole(8), Shared(0)], 4),
        ),
    ]
    for scalar in (False, True):
        for mode in (0, 1, 2):
            out.append(
                Refusal(
                    case(
                        f"pointer flow, onto the next cell, mode {mode}, scalar {scalar}",
                        _pointer_flow,
                        [a, b, Shared(0, 1), 2],
                        BLOCK=32,
                        SCALAR=scalar,
                        MODE=mode,
                    ),
                    UNTRACED,
                    None,
                    case(
                        f"pointer flow, separate, mode {mode}, scalar {scalar}",
                        _pointer_flow,
                        [a, b, y, 2],
                        BLOCK=32,
                        SCALAR=scalar,
                        MODE=mode,
                    ),
                )
            )
        out.append(
            Refusal(
                case(
                    f"while pointer, onto the next cell, scalar {scalar}",
                    _while_pointer,
                    [a, b, Shared(0, 1), 2],
                    BLOCK=32,
                    SCALAR=scalar,
                ),
                UNBOUNDED,
                None,
                case(
                    f"while pointer, separate, scalar {scalar}", _while_pointer, [a, b, y, 2], BLOCK=32, SCALAR=scalar
                ),
            )
        )
    return out


REFUSED = _meeting() + _above_the_threads() + _one_cell() + _folded() + _other_reasons()
PARTNERS = list({row.partner.name: row.partner for row in REFUSED if row.partner is not None}.values())
PARTNERS += [
    case("one call, separate", _one_call, [whole(8), blank(8)], 4),
    case("one call, in place", _one_call, [whole(8), Shared(0)], 4),
    case("nine calls, separate", _nine_calls, [whole(8), blank(8)], 4),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row", REFUSED, ids=[row.case.name for row in REFUSED])
def test_the_launch_is_refused_and_nothing_is_changed(row, device):
    hosts, tensors = build(row.case.args, device)
    with pytest.raises(MetalNonRecoverableError) as caught:
        launch(row.case, tensors, device)
    message = str(caught.value)
    assert "lane order:" in message
    assert f": {row.reason}. The source finishes each operation" in message
    if row.names is not None:
        named = f"argument(s) {row.names}" if row.names else "its arguments"
        assert f"for the memory of {named}: {row.reason}" in message
    same(row.case, hosts, tensors)  # the host arrays were never run: every tensor is as it was built


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("one", PARTNERS, ids=[one.name for one in PARTNERS])
def test_the_refused_kernel_computes_where_no_lanes_meet(one, device):
    computes(one, device)


def test_every_reason_a_launch_can_reach_is_covered():
    reached = {row.reason for row in REFUSED}
    assert reached == {MEET, NARROW, ORDER, OPAQUE, SIZE, EVALUATE, UNBOUNDED, UNTRACED, BOUNDS, BUDGET, CALL}


# ------------------------------------------------------ a storage that cannot be read
class _Unreadable:
    """An argument with a pointer whose storage raises: nothing says what memory it covers."""

    dtype = torch.float32

    def data_ptr(self):
        return 4096

    def untyped_storage(self):
        raise RuntimeError("no storage")


@pytest.mark.parametrize("device", DEVICES)
def test_an_argument_whose_storage_cannot_be_read_shares_with_every_other(device):
    """No launch reaches this: the launcher needs the storage itself. The check is asked directly."""
    from triton_msl.backend._lane_order import LaunchCheck, unproved

    one = case("scale, separate", _scale, [whole(64), blank(64), 64], BLOCK=64)
    one.kernel.device_caches.clear()
    _, tensors = build(one.args, device)
    launch(one, tensors, device)
    cache = list(one.kernel.device_caches.values())[0]
    check = LaunchCheck(list(cache[0].values())[-1].run._access_plan)
    x, y, count = tensors
    assert unproved(check, [x, y, count], (1, 1, 1)) is None
    assert unproved(check, [_Unreadable(), y, count], (1, 1, 1)) == (STORAGE, (0, 1))
    assert unproved(check, [x, _Unreadable(), count], (1, 1, 1)) == (STORAGE, (0, 1))
