"""The lane-order check compares a large grid a run of programs at a time.

When an address does not follow the program id by one translation (a row and a column
taken from the flat index with ``%`` and ``//``, as torch.compile writes a slice update),
the check walks every program. A cell is compared inside its own program only, so the
grid is cut into runs that each fit one comparison. The whole launch used to be refused
once the grid held more addresses than one comparison takes.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401
from triton_msl.backend import _lane_order
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
MEET = "two lanes of one program reach the same element and one of them stores it"
BUDGET = "there are too many addresses to compare"


@triton.jit
def _slice_add(in_out_ptr0, in_ptr0, xnumel, W, XBLOCK: tl.constexpr):
    # x[:, 1:] += y: the row and the column come from the flat index.
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % W
    x1 = xindex // W
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + (1 + x0 + (W + 1) * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (1 + x0 + (W + 1) * x1), tmp2, xmask)


@triton.jit
def _reversed_from(in_out_ptr0, xnumel, W, T, XBLOCK: tl.constexpr):
    # Each lane adds one to its own cell below T; from T on a program stores its block reversed.
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % W
    x1 = xindex // W
    own = x0 + W * x1
    cell = tl.where(xindex >= T, (2 * xoffset + XBLOCK - 1) - own, own)
    tmp0 = tl.load(in_out_ptr0 + own, xmask)
    tl.store(in_out_ptr0 + cell, tmp0 + 1.0, xmask)


def _launcher(kernel):
    device = list(kernel.device_caches.values())[0]
    return list(device[0].values())[-1].run


def _counted(monkeypatch, kernel):
    """Record the comparisons the check makes, by the programs each one covers."""
    for device in kernel.device_caches.values():
        for compiled in device[0].values():
            # An answer kept from an earlier launch of the same sizes would make none.
            if getattr(compiled.run, "_lane_check", None) is not None:
                compiled.run._lane_check.cache.clear()
    runs = []
    inner = _lane_order.LaunchCheck._lanes

    def counting(self, ordinals, origins, arguments, grid, programs, *rest):
        runs.append(programs)
        return inner(self, ordinals, origins, arguments, grid, programs, *rest)

    monkeypatch.setattr(_lane_order.LaunchCheck, "_lanes", counting)
    return runs


def _slice_case(rows, width, device):
    host = torch.arange(rows * (width + 1), dtype=torch.float32).reshape(rows, width + 1) % 1021.0
    add = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) % 7.0 + 1.0
    want = host.clone()
    want[:, 1:] += add
    return host.clone().to(device), add.to(device), want


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,width", [(512, 512), (1500, 1500), (2100, 2047)])
def test_an_in_place_slice_update_computes_above_one_comparison(rows, width, device):
    x, y, want = _slice_case(rows, width, device)
    count = rows * width
    grid = (-(-count // 1024),)
    assert (2 * count > _lane_order.BUDGET) == (rows > 512)
    _slice_add[grid](x, y, count, width, XBLOCK=1024)
    assert torch.equal(x.cpu(), want)
    check = _lane_order.LaunchCheck(_launcher(_slice_add)._access_plan)
    answers = sorted(answer for answer, _ in check.verdicts([x, y, count, width], grid + (1, 1)))
    assert answers == ["lanes", "no-write"]


@pytest.mark.parametrize("device", DEVICES)
def test_the_grid_is_compared_in_runs_that_cover_every_program(device, monkeypatch):
    rows, width, block = 96, 100, 64
    count = rows * width
    programs = count // block
    monkeypatch.setattr(_lane_order, "BUDGET", 16 * 2 * block)
    monkeypatch.setattr(_lane_order, "SLICES", programs)
    runs = _counted(monkeypatch, _slice_add)
    x, y, want = _slice_case(rows, width, device)
    _slice_add[(programs,)](x, y, count, width, XBLOCK=block)
    assert torch.equal(x.cpu(), want)
    cut = [run for run in runs if run is not None and run != (0, programs)]
    covered = sorted(run for run in cut if run[1] - run[0] <= 16)
    assert covered and covered[0][0] == 0 and covered[-1][1] == programs
    assert all(before[1] == after[0] for before, after in zip(covered, covered[1:]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("small", [False, True], ids=["one comparison's size", "a small one"])
def test_a_meeting_in_the_last_program_is_found(small, device, monkeypatch):
    rows, width, block = (96, 100, 64) if small else (1536, 1500, 1024)
    count = rows * width
    programs = count // block
    if small:
        monkeypatch.setattr(_lane_order, "BUDGET", 16 * 2 * block)
        monkeypatch.setattr(_lane_order, "SLICES", programs)
    assert 2 * count > _lane_order.BUDGET
    host = torch.arange(count, dtype=torch.float32) % 509.0
    # Nothing reversed: every lane keeps to its own cell.
    x = host.clone().to(device)
    _reversed_from[(programs,)](x, count, width, count, XBLOCK=block)
    assert torch.equal(x.cpu(), host + 1.0)
    # The last program alone reverses its block in place.
    x = host.clone().to(device)
    with pytest.raises(MetalNonRecoverableError, match="lane order: .*" + MEET):
        _reversed_from[(programs,)](x, count, width, count - block, XBLOCK=block)
    assert torch.equal(x.cpu(), host)
    # Onto another tensor's memory the reversal is the source's.
    out = torch.zeros(count, device=device)
    _reversed_from_to[(programs,)](x, out, count, width, count - block, XBLOCK=block)
    want = host + 1.0
    want[count - block :] = torch.flip(want[count - block :], [0])
    assert torch.equal(out.cpu(), want)


@triton.jit
def _reversed_from_to(in_ptr0, out_ptr0, xnumel, W, T, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % W
    x1 = xindex // W
    own = x0 + W * x1
    cell = tl.where(xindex >= T, (2 * xoffset + XBLOCK - 1) - own, own)
    tmp0 = tl.load(in_ptr0 + own, xmask)
    tl.store(out_ptr0 + cell, tmp0 + 1.0, xmask)


@pytest.mark.parametrize("device", DEVICES)
def test_a_grid_of_more_runs_than_one_launch_is_given_refuses_before_comparing_them(device, monkeypatch):
    rows, width, block = 96, 100, 64
    count = rows * width
    programs = count // block
    monkeypatch.setattr(_lane_order, "BUDGET", 16 * 2 * block)
    monkeypatch.setattr(_lane_order, "SLICES", 4)
    runs = _counted(monkeypatch, _slice_add)
    x, y, _ = _slice_case(rows, width, device)
    before = x.clone()
    with pytest.raises(MetalNonRecoverableError, match="lane order: .*" + BUDGET):
        _slice_add[(programs,)](x, y, count, width, XBLOCK=block)
    assert torch.equal(x, before)
    # One comparison that fits, and with it the count of the rest.
    assert sum(1 for run in runs if run is not None and run[1] - run[0] <= 16) == 1
