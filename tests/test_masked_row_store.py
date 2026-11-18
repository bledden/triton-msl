"""A masked store of one value per row of a two-dimensional block stores only the rows the mask keeps.

The kernels are the shape torch.compile writes for a reduction over the last axis: XBLOCK
rows per program, a row total stored through ``out_ptr0 + xindex`` under ``xmask``. When
the row count is not a multiple of XBLOCK the last program's mask excludes the rows past
the end. The store used to ignore the mask and write them, past the end of the output.

Each row's store is also given to the row's first lane alone, so the form that reads the
output cell before it stores it (an in-place accumulation) reads it before any store.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

POISON = -777.0
SHAPES = [(2, 32), (4, 32), (4, 256), (8, 8), (8, 64), (16, 64), (32, 32), (64, 16), (2, 512), (64, 8)]


@triton.jit
def _row_sum(out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    tmp0 = tl.load(in_ptr0 + (rindex + rnumel * xindex), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (xindex), tmp4, xmask)


@triton.jit
def _row_sum_looped(out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        tmp0 = tl.load(in_ptr0 + (rindex + rnumel * xindex), rmask & xmask, other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (xindex), tmp2, xmask)


@triton.jit
def _row_max(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    a = tl.load(in_ptr0 + (rindex + RBLOCK * xindex), xmask & (rindex < RBLOCK), other=0.0)
    tl.store(out_ptr0 + (xindex), tl.max(a, 1)[:, None], xmask)


@triton.jit
def _row_sum_in_place(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    tmp0 = tl.load(in_ptr0 + (rindex + rnumel * xindex), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.load(in_out_ptr0 + (xindex), xmask)
    tmp6 = tmp5 + tmp4
    tl.store(in_out_ptr0 + (xindex), tmp6, xmask)


def _launcher(kernel):
    device = list(kernel.device_caches.values())[0]
    return list(device[0].values())[-1].run


def _rows(xblock):
    """Row counts that fill the last program, leave one row of it, and leave all but one."""
    return [xblock, 2 * xblock - 1, 3 * xblock + 1]


def _data(rows, columns, device):
    generator = torch.Generator().manual_seed(rows * 1009 + columns)
    host = torch.rand(rows, columns, generator=generator) * 2.0 - 1.0
    return host, host.to(device)


def _output(rows, device):
    """An output of ``rows`` cells that is the head of a longer buffer of sentinels."""
    whole = torch.full((rows + 96,), POISON, device=device)
    return whole, whole[:rows]


CASES = [(xb, rb, rows, rn) for xb, rb in SHAPES for rows in _rows(xb) for rn in (rb, rb - 3)]
IDS = [f"x{xb}-r{rb}-rows{rows}-cols{rn}" for xb, rb, rows, rn in CASES]


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("xb,rb,rows,rn", CASES, ids=IDS)
def test_row_total_stores_only_the_rows_inside_the_range(xb, rb, rows, rn, device):
    host, data = _data(rows, rn, device)
    whole, out = _output(rows, device)
    _row_sum[(-(-rows // xb),)](out, data, rows, rn, XBLOCK=xb, RBLOCK=rb)
    got = whole.cpu()
    torch.testing.assert_close(got[:rows], host.double().sum(1).float(), rtol=1e-5, atol=1e-5)
    assert torch.equal(got[rows:], torch.full((96,), POISON))


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("xb,rb,rows,rn", CASES, ids=IDS)
def test_looped_row_total_stores_only_the_rows_inside_the_range(xb, rb, rows, rn, device):
    columns = rn if rn == rb else 2 * rb + rn  # the short case takes three trips
    host, data = _data(rows, columns, device)
    whole, out = _output(rows, device)
    _row_sum_looped[(-(-rows // xb),)](out, data, rows, columns, XBLOCK=xb, RBLOCK=rb)
    got = whole.cpu()
    torch.testing.assert_close(got[:rows], host.double().sum(1).float(), rtol=1e-5, atol=1e-5)
    assert torch.equal(got[rows:], torch.full((96,), POISON))


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("xb,rb", SHAPES, ids=[f"x{xb}-r{rb}" for xb, rb in SHAPES])
def test_row_maximum_stores_only_the_rows_inside_the_range(xb, rb, device):
    for rows in _rows(xb):
        # The load reads whole blocks of rows; only rows inside the range are kept.
        host, data = _data(-(-rows // xb) * xb, rb, device)
        whole, out = _output(rows, device)
        _row_max[(-(-rows // xb),)](out, data, rows, XBLOCK=xb, RBLOCK=rb)
        got = whole.cpu()
        assert torch.equal(got[:rows], host[:rows].max(1).values)
        assert torch.equal(got[rows:], torch.full((96,), POISON))


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("xb,rb,rows,rn", CASES, ids=IDS)
def test_row_total_added_in_place_computes_and_stays_inside_the_range(xb, rb, rows, rn, device):
    host, data = _data(rows, rn, device)
    whole, out = _output(rows, device)
    start = torch.arange(rows, dtype=torch.float32) * 0.5 - 3.0
    out.copy_(start.to(device))
    for _ in range(3):
        _row_sum_in_place[(-(-rows // xb),)](out, data, rows, rn, XBLOCK=xb, RBLOCK=rb)
    got = whole.cpu()
    want = (start.double() + 3.0 * host.double().sum(1)).float()
    torch.testing.assert_close(got[:rows], want, rtol=1e-5, atol=1e-5)
    assert torch.equal(got[rows:], torch.full((96,), POISON))


def test_the_in_place_form_is_proved_by_one_lane_per_row_and_not_when_the_input_shares_the_memory():
    from triton_msl.backend._lane_order import LaunchCheck

    xb, rb, rows = 4, 32, 8
    _, data = _data(rows, rb, "mps")
    out = torch.zeros(rows, device="mps")
    _row_sum_in_place[(rows // xb,)](out, data, rows, rb, XBLOCK=xb, RBLOCK=rb)
    plan = _launcher(_row_sum_in_place)._access_plan
    # The store of the row totals and the load its value alone depends on.
    assert plan["rows"] == [2, 1]
    grid = (rows // xb, 1, 1)
    answers = dict((members, answer) for answer, members in LaunchCheck(plan).verdicts([out, data, rows, rb], grid))
    assert answers == {(0,): "lanes", (1,): "no-write"}
    # With the input in the same memory as the totals, a row's first lane stores a cell
    # that other lanes of the block read: no proof, so the launch is refused.
    shared = torch.zeros(rows * rb, device="mps")
    found = LaunchCheck(plan).verdicts([shared[:rows], shared, rows, rb], grid)
    assert [answer for answer, _ in found] == ["lanes-meet-narrow"]
