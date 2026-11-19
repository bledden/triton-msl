"""Plain loads and stores in a ``while`` loop, each lane on its own cell.

The launch check that orders a kernel's lanes admits a ``while`` loop whose loads and stores
keep each lane to its own cell on every trip. Until now only atomics in such a loop had a
test. A loop that reverses its cells in place on every trip is refused when it is launched,
and nothing is written.

One lane per element; a mask keeps to the first M of them: every address is inside its
tensor.
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
SIZES = [16, 64, 1024, 2048]
PROGRAMS = 3
OWN_CELL, MASKED, STORE_ONLY, OTHER_TENSOR, REVERSED_IN_PLACE = 0, 1, 2, 3, 4
FORMS = {
    OWN_CELL: "load and store of the lane's own cell",
    MASKED: "the same under a mask",
    STORE_ONLY: "store only",
    OTHER_TENSOR: "another tensor read at other lanes' cells",
}


@triton.jit
def _plain(Z, R, TRIPS, M, N: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    zb = Z + pid * N
    rb = R + pid * N
    j = 0
    if FORM == 0:
        while j < TRIPS:
            x = tl.load(zb + n)
            tl.store(zb + n, x * 2 + 1 + j)
            j += 1
    elif FORM == 1:
        while j < TRIPS:
            x = tl.load(zb + n, mask=n < M, other=0)
            tl.store(zb + n, x * 2 + 1 + j, mask=n < M)
            j += 1
    elif FORM == 2:
        while j < TRIPS:
            tl.store(zb + n, n * 7 + j)
            j += 1
    elif FORM == 3:
        while j < TRIPS:
            x = tl.load(zb + (N - 1 - n))
            tl.store(rb + n, x + j)
            j += 1
    else:
        while j < TRIPS:
            x = tl.load(zb + (N - 1 - n))
            tl.store(zb + n, x)
            j += 1


def _model(z, r, size, trips, masked, form):
    z = z.astype(np.int64)
    r = r.astype(np.int64)
    n = np.arange(size)
    for pid in range(PROGRAMS):
        cells = pid * size + n
        for j in range(trips):
            if form == OWN_CELL:
                z[cells] = z[cells] * 2 + 1 + j
            elif form == MASKED:
                z[cells[:masked]] = z[cells[:masked]] * 2 + 1 + j
            elif form == STORE_ONLY:
                z[cells] = n * 7 + j
            else:
                r[cells] = z[pid * size + (size - 1 - n)] + j
    return z, r


def _tensors(size, device):
    z0 = ((np.arange(PROGRAMS * size) * 3) % 97).astype(np.int32)
    r0 = np.full(PROGRAMS * size, -7, dtype=np.int32)
    return z0, r0, torch.from_numpy(z0.copy()).to(device), torch.from_numpy(r0.copy()).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trips", [0, 1, 3])
@pytest.mark.parametrize("form", sorted(FORMS), ids=lambda form: FORMS[form])
@pytest.mark.parametrize("size", SIZES)
def test_each_lane_on_its_own_cell_computes(size, form, trips, device):
    masked = size - 5
    z0, r0, z, r = _tensors(size, device)
    _plain[(PROGRAMS,)](z, r, trips, masked, N=size, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    want_z, want_r = _model(z0, r0, size, trips, masked, form)
    assert np.array_equal(z.cpu().numpy(), want_z)
    assert np.array_equal(r.cpu().numpy(), want_r)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", SIZES)
def test_a_reversal_in_place_on_every_trip_is_refused_and_nothing_is_changed(size, device):
    z0, r0, z, r = _tensors(size, device)
    with pytest.raises(MetalNonRecoverableError):
        _plain[(PROGRAMS,)](z, r, 2, size, N=size, FORM=REVERSED_IN_PLACE)
    assert np.array_equal(z.cpu().numpy(), z0)
    assert np.array_equal(r.cpu().numpy(), r0)
