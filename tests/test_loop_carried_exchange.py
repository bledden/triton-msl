"""A loop hands its carried values on all at once.

``a, b = b, a`` in a loop gives each the other's old value: the values a loop yields are
simultaneous. The carried variables were written one after another, in order, so the second
read the first as just written. After one trip of the exchange both held the old ``b``; a
three-way rotation and ``c = b; b = a; a = a + 1`` went the same way, in a counted loop and
in a while loop. A yielded value that is (or reads) a carried variable written before it is
now copied before any of them is written.

A while loop that carries a vector of more elements than the group has threads did not
compile (the vector is held several elements per thread). It is refused by name.

One lane per element and no mask: every address is inside its tensor.
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
DTYPES = {"float32": torch.float32, "int32": torch.int32}
PROGRAMS = 4
WHILE, COUNTED = 0, 1
LOOPS = {WHILE: "while loop", COUNTED: "counted loop"}
FORMS = {
    0: "a, b = b, a",
    1: "a, b = b, a + b",
    2: "a, b = a + b, a",
    3: "a, b, c = b, c, a",
    4: "a, b, c = a + 1, a, b",
    5: "a, b, c = b, c, c + 1",
}


@triton.jit
def _exchange(X, Z, N: tl.constexpr, FORM: tl.constexpr, LOOP: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    a = tl.load(X + n)
    b = tl.load(X + n + N)
    c = tl.load(X + n + 2 * N)
    if LOOP == 0:
        j = 0
        while j < pid + 1:
            if FORM == 0:
                t = a
                a = b
                b = t
            elif FORM == 1:
                t = a + b
                a = b
                b = t
            elif FORM == 2:
                t = a + b
                b = a
                a = t
            elif FORM == 3:
                t = a
                a = b
                b = c
                c = t
            elif FORM == 4:
                c = b
                b = a
                a = a + 1
            elif FORM == 5:
                a = b
                b = c
                c = c + 1
            j += 1
    else:
        for _ in range(pid + 1):
            if FORM == 0:
                t = a
                a = b
                b = t
            elif FORM == 1:
                t = a + b
                a = b
                b = t
            elif FORM == 2:
                t = a + b
                b = a
                a = t
            elif FORM == 3:
                t = a
                a = b
                b = c
                c = t
            elif FORM == 4:
                c = b
                b = a
                a = a + 1
            elif FORM == 5:
                a = b
                b = c
                c = c + 1
    tl.store(Z + pid * N + n, a + b * 100 + c * 10000)


def _model(x, count, form):
    out = np.zeros(PROGRAMS * count, dtype=np.int64)
    n = np.arange(count)
    for pid in range(PROGRAMS):
        a, b, c = (x[n + part * count].astype(np.int64) for part in range(3))
        for _ in range(pid + 1):
            if form == 0:
                a, b = b, a
            elif form == 1:
                a, b = b, a + b
            elif form == 2:
                a, b = a + b, a
            elif form == 3:
                a, b, c = b, c, a
            elif form == 4:
                a, b, c = a + 1, a, b
            elif form == 5:
                a, b, c = b, c, c + 1
        out[pid * count + n] = a + b * 100 + c * 10000
    return out


def _launch(count, form, loop, dtype, device):
    # Small whole numbers: every result is exact in float32.
    x0 = (np.arange(3 * count) * 7 + 3) % 5
    x = torch.tensor(x0, dtype=DTYPES[dtype], device=device)
    z = torch.full((PROGRAMS * count,), -7, dtype=DTYPES[dtype], device=device)
    _exchange[(PROGRAMS,)](x, z, N=count, FORM=form, LOOP=loop)
    if device == "mps":
        torch.mps.synchronize()
    return z.cpu().numpy().astype(np.int64), _model(x0, count, form)


COMPUTED = [(count, loop) for count in (64, 256, 1024) for loop in sorted(LOOPS) if (count, loop) != (256, WHILE)]
COMPUTED.remove((1024, WHILE))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("form", sorted(FORMS), ids=lambda form: FORMS[form])
@pytest.mark.parametrize("count,loop", COMPUTED, ids=[f"{count} elements, {LOOPS[loop]}" for count, loop in COMPUTED])
def test_each_carried_value_gets_the_value_the_source_hands_on(count, loop, form, dtype, device):
    for _ in range(2):
        got, want = _launch(count, form, loop, dtype, device)
        assert np.array_equal(got, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("form", [0, 4], ids=lambda form: FORMS[form])
@pytest.mark.parametrize("count", [256, 1024])
def test_a_while_loop_carrying_more_elements_than_threads_is_refused(count, form, dtype, device):
    with pytest.raises(MetalNonRecoverableError, match="a while loop carries a value held several elements"):
        _launch(count, form, WHILE, dtype, device)


@triton.jit
def _one_trip_late(Z, R, TRIPS, N: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    cells = Z + pid * N + n
    prev = n * 0
    j = 0
    while j < TRIPS:
        x = tl.load(cells)
        tl.store(cells, prev + n + j)
        prev = x
        j += 1
    tl.store(R + pid * N + n, prev)


def _launch_one_trip_late(count, trips, device):
    z0 = ((np.arange(PROGRAMS * count) * 3) % 97).astype(np.int32)
    z = torch.tensor(z0, device=device)
    r = torch.full((PROGRAMS * count,), -7, dtype=torch.int32, device=device)
    _one_trip_late[(PROGRAMS,)](z, r, trips, N=count)
    if device == "mps":
        torch.mps.synchronize()
    return z0, z.cpu().numpy(), r.cpu().numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trips", [0, 1, 3])
@pytest.mark.parametrize("count", [16, 64])
def test_a_loaded_value_handed_to_the_next_trip_computes(count, trips, device):
    z0, z, r = _launch_one_trip_late(count, trips, device)
    want_z = z0.astype(np.int64)
    want_r = np.zeros(PROGRAMS * count, dtype=np.int64)
    n = np.arange(count)
    for pid in range(PROGRAMS):
        cells = pid * count + n
        prev = n * 0
        for j in range(trips):
            x = want_z[cells].copy()
            want_z[cells] = prev + n + j
            prev = x
        want_r[cells] = prev
    assert np.array_equal(z, want_z)
    assert np.array_equal(r, want_r)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [1024, 2048])
def test_a_while_loop_handed_more_elements_than_threads_is_refused(count, device):
    """The carried value starts one element per thread; the load handed on is held several per thread.

    This did not compile, and the failure named no reason.
    """
    with pytest.raises(MetalNonRecoverableError, match="a while loop carries a value held several elements"):
        _launch_one_trip_late(count, 2, device)
