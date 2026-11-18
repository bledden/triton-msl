"""A store of one element is performed by lane 0 only, and the launch check knows it.

Every lane of a block evaluates a one-element store's address and value, so before the
change every thread stored the cell: a counter incremented once in the source could be
incremented by several threads, each after another's store. The shader now performs
such a store in lane 0 alone and the access plan lists it, so the launch check lets a
launch through when the cell is the only thing two lanes could have shared.

The compute cases use the source interpreter of test_lane_order_launch.py. The shader
assertions are deliberately few: the guard in front of the store, and the store's
place in the plan. The last cases are two kernels as inductor writes them for a
training step, launched with the arguments aliased as inductor passes them.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from tests.test_lane_order_launch import (
    DEVICES,
    Shared,
    _chain,
    blank,
    build,
    case,
    compiled,
    computes,
    launch,
    requires_mps,
    whole,
)

pytestmark = requires_mps


@triton.jit
def _counter(C, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * 2.0)
    c = tl.load(C)
    tl.store(C, c + 1.0)


@triton.jit
def _counter_first(C, X, Y, BLOCK: tl.constexpr):
    c = tl.load(C)
    tl.store(C, c + 1.0)
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * 2.0)


@triton.jit
def _accumulate(Q, X, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Q, tl.load(Q) + tl.sum(v, axis=0))


@triton.jit
def _accumulate_early(Q, X, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    old = tl.load(Q)
    v = tl.load(X + i)
    tl.store(Q, old + tl.sum(v, axis=0))


@triton.jit
def _scale_and_bump(S, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    s = tl.load(S)
    tl.store(Y + i, tl.load(X + i) * s)
    tl.store(S, s + 1.0)


@triton.jit
def _twice(C, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1.0)
    tl.store(C, 1.0)
    tl.store(C, 2.0)


@triton.jit
def _chain_cell(C, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1.0)
    a = tl.load(C)
    tl.store(C, a + 1.0)
    b = tl.load(C)
    tl.store(C, b * 2.0)


@triton.jit
def _loop_counter(C, X, Y, K, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1.0)
    for _ in range(K):
        c = tl.load(C)
        tl.store(C, c + 1.0)


@triton.jit
def _masked_counter(C, X, Y, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1.0)
    c = tl.load(C)
    tl.store(C, c + 1.0, mask=N > 0)


@triton.jit
def _program_counters(C, X, Y, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    i = p * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * 2.0)
    c = tl.load(C + p)
    tl.store(C + p, c + 1.0)


@triton.jit
def _program_sums(Q, X, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    i = p * BLOCK + tl.arange(0, BLOCK)
    tl.store(Q + p, tl.load(Q + p) + tl.sum(tl.load(X + i), axis=0))


@triton.jit
def _whole_counter(C, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1.0)
    c = tl.load(C)
    tl.store(C, c + 1)


@triton.jit
def _largest_row_sum(Q, X, R: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    v = tl.load(X + a)
    tl.store(Q, tl.load(Q) + tl.max(tl.sum(v, axis=1), axis=0))


@triton.jit
def _cell_from_wide(C, X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y, tl.sum(v, axis=0) + tl.load(C))


@triton.jit
def _row_max_to_first(X, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)
    a = r[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(Y + r * C, tl.max(tl.load(X + a), axis=1))


@triton.jit
def _increment_copy(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):  # as inductor writes a batch counter
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp3, None)


@triton.jit
def _batch_statistics(  # as inductor writes a batch norm's reduction with its running statistics
    in_out_ptr0,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    out_ptr2,
    out_ptr4,
    out_ptr6,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 16
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy="evict_last")
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index % 256
        r0_2 = r0_index // 256
        tmp0 = tl.load(
            in_out_ptr0 + (r0_1 + 256 * x0 + 4096 * r0_2), r0_mask & xmask, eviction_policy="evict_first", other=0.0
        )
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r0_1 + 256 * x0 + 4096 * r0_2), tmp2, r0_mask & xmask)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy="evict_last")
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy="evict_last")
    tmp10 = tl.full([1, 1], 2048.0, tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tl.full([1, 1], 1.0004885197850513, tl.float32)
    tmp16 = tmp11 * tmp15
    tmp17 = tl.full([1, 1], 0.1, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp20 = tl.full([1, 1], 0.9, tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp4 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp26, xmask)


# ---------------------------------------------------------------------- the launches
def _one_element():
    out = []
    for block in (1, 2, 64, 1024, 8192):
        c, x, y = whole(1, 3), whole(block), blank(block)
        count = whole(1, 3, dtype=np.int64)
        out += [
            case(f"counter, separate, {block}", _counter, [c, x, y], BLOCK=block),
            case(f"counter beside in-place work, {block}", _counter, [c, x, Shared(1)], BLOCK=block),
            case(f"counter first, beside in-place work, {block}", _counter_first, [c, x, Shared(1)], BLOCK=block),
            case(f"accumulate, separate, {block}", _accumulate, [c, x], BLOCK=block),
            case(f"accumulate into cell 0 of the input, {block}", _accumulate, [Shared(1), x], BLOCK=block),
            case(
                f"accumulate early, into cell 0 of the input, {block}", _accumulate_early, [Shared(1), x], BLOCK=block
            ),
            case(f"twice, {block}", _twice, [c, x, Shared(1)], BLOCK=block),
            case(f"chain on one cell, {block}", _chain_cell, [c, x, Shared(1)], BLOCK=block),
            case(f"counter in a loop, {block}", _loop_counter, [c, x, Shared(1), 3], BLOCK=block),
            case(f"masked counter, on, {block}", _masked_counter, [c, x, Shared(1), 1], BLOCK=block),
            case(f"masked counter, off, {block}", _masked_counter, [c, x, Shared(1), 0], BLOCK=block),
            case(f"whole-number counter, {block}", _whole_counter, [count, x, Shared(1)], BLOCK=block),
            case(
                f"a counter per program, {block}",
                _program_counters,
                [whole(3, 3), whole(3 * block), Shared(1)],
                3,
                BLOCK=block,
            ),
            case(f"a sum per program, {block}", _program_sums, [whole(3, 3), whole(3 * block)], 3, BLOCK=block),
            case(f"cell from the block, separate, {block}", _cell_from_wide, [c, x, blank(1)], BLOCK=block),
            case(f"sum into cell 0 of the block, {block}", _cell_from_wide, [c, x, Shared(1)], BLOCK=block),
            case(f"increment copied, separate, {block}", _increment_copy, [count, blank(1, np.int64), 1], XBLOCK=block),
        ]
    # One lane: nothing can meet.
    c, x = whole(1, 3), whole(4)
    out += [
        case("counter on cell 0 of the work, one lane", _counter, [Shared(1), x, Shared(1)], BLOCK=1),
        case("counter first on cell 0 of the work, one lane", _counter_first, [Shared(1), x, Shared(1)], BLOCK=1),
        case("scaled then bumped, separate, one lane", _scale_and_bump, [c, x, blank(4)], BLOCK=1),
        case("scaled then bumped, one cell, one lane", _scale_and_bump, [Shared(1), x, Shared(1)], BLOCK=1),
        case(
            "increment copied over its input, one lane",
            _increment_copy,
            [whole(1, 3, dtype=np.int64), Shared(0), 1],
            XBLOCK=1,
        ),
    ]
    for rows, columns in ((8, 8), (16, 64), (32, 32)):
        x = whole(rows * columns)
        out += [
            case(f"largest row sum, {rows}x{columns}", _largest_row_sum, [whole(1, 3), x], R=rows, C=columns),
            case(
                f"row maximum to the row's first cell, {rows}x{columns}",
                _row_max_to_first,
                [x, Shared(0)],
                R=rows,
                C=columns,
            ),
        ]
    return out


ONE_ELEMENT = _one_element()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("one", ONE_ELEMENT, ids=[one.name for one in ONE_ELEMENT])
def test_a_store_of_one_element_computes_what_the_source_computes(one, device):
    computes(one, device)


# ------------------------------------------------------------- the shader and its plan
# (the launch, the one-element stores' places among the kernel's accesses, the stored cells)
GUARDED = [
    (case("counter, 64", _counter, [whole(1), whole(64), blank(64)], BLOCK=64), [3], ["C[0] = "]),
    (case("counter, one lane", _counter, [whole(1), whole(1), blank(1)], BLOCK=1), [1, 3], ["Y[0] = ", "C[0] = "]),
    (case("counter, 4096", _counter, [whole(1), whole(4096), blank(4096)], BLOCK=4096), [3], ["C[0] = "]),
    (case("accumulate, 64", _accumulate, [whole(1), whole(64)], BLOCK=64), [2], ["Q[0] = "]),
    (case("twice, 64", _twice, [whole(1), whole(64), blank(64)], BLOCK=64), [2, 3], ["C[0] = ", "C[0] = "]),
    (case("sum into a cell, 64", _cell_from_wide, [whole(1), whole(64), blank(1)], BLOCK=64), [2], ["Y[0] = "]),
]


def _launched(one):
    """The binary a launch of ``one`` compiles, and the tensors it ran on."""
    one.kernel.device_caches.clear()
    _, tensors = build(one.args, "mps")
    launch(one, tensors, "mps")
    return compiled(one.kernel)


@pytest.mark.parametrize("one, single, stores", GUARDED, ids=[row[0].name for row in GUARDED])
def test_a_store_of_one_element_is_guarded_and_listed_in_the_plan(one, single, stores):
    binary = _launched(one)
    assert binary.run._access_plan["single"] == single
    shader = [line.strip() for line in binary.asm["msl"].splitlines()]
    found = [line for line in shader if any(store in line for store in stores)]
    assert len(found) == len(stores), found
    for line in found:
        # Lane 0 alone: by its thread index, or by its element where a thread walks several lanes.
        assert line.startswith(("if (lid == 0u) { ", "if (_loop_e == 0u) { ")), line
    if one.constants["BLOCK"] <= 1024:
        assert sum("lid == 0u" in line for line in shader) == len(stores)


def test_a_store_of_the_whole_block_is_not_guarded():
    one = case("chain, 64", _chain, [whole(64), blank(64)], BLOCK=64)
    binary = _launched(one)
    assert binary.run._access_plan["single"] == []
    assert "lid == 0u" not in binary.asm["msl"]


def test_one_cell_stored_by_every_lane_is_not_a_store_of_one_element():
    """Inductor's counter copy broadcasts the cell's address over the block: every lane stores it."""
    count = whole(1, 3, dtype=np.int64)
    binary = _launched(case("increment copied, 16", _increment_copy, [count, blank(1, np.int64), 1], XBLOCK=16))
    assert binary.run._access_plan["single"] == []


# ------------------------------------------------- inductor's batch-norm training step
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("aliased", (True, False), ids=("statistics updated in place", "statistics to new tensors"))
def test_inductor_batch_statistics_with_running_statistics(aliased, device):
    """Each of the 16 programs is one channel: it reads and writes its own cell of every statistic."""
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(8 * 16 * 256, generator=generator)
    bias = torch.randn(16, generator=generator)
    variance = torch.rand(16, generator=generator) + 0.5
    mean = torch.randn(16, generator=generator)
    gx, gbias, gvariance, gmean = (value.clone().to(device) for value in (x, bias, variance, mean))
    batch_mean, inverse = torch.zeros(16, device=device), torch.zeros(16, device=device)
    new_variance = gvariance if aliased else torch.zeros(16, device=device)
    new_mean = gmean if aliased else torch.zeros(16, device=device)
    _batch_statistics[(16,)](
        gx, gbias, gvariance, gmean, batch_mean, inverse, new_variance, new_mean, 16, 2048, XBLOCK=1, R0_BLOCK=64
    )
    if device == "mps":
        torch.mps.synchronize()
    shifted = x.numpy().astype(np.float64).reshape(8, 16, 256) + bias.numpy().astype(np.float64)[None, :, None]
    want_mean, want_variance = shifted.mean(axis=(0, 2)), shifted.var(axis=(0, 2))
    expected = (
        (gx, shifted.reshape(-1)),
        (batch_mean, want_mean),
        (inverse, 1 / np.sqrt(want_variance + 1e-5)),
        (new_variance, want_variance * 1.0004885197850513 * 0.1 + variance.numpy().astype(np.float64) * 0.9),
        (new_mean, want_mean * 0.1 + mean.numpy().astype(np.float64) * 0.9),
    )
    for got, want in expected:
        np.testing.assert_allclose(got.cpu().numpy(), want, rtol=2e-4, atol=2e-5)
    if not aliased:
        np.testing.assert_array_equal(gvariance.cpu().numpy(), variance.numpy())
        np.testing.assert_array_equal(gmean.cpu().numpy(), mean.numpy())
