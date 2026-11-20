"""A row-wise sum chosen by a branch or carried by a while loop keeps its place on the lanes.

A row-wise sum of an N x (W / N) tile in a W-lane kernel is held one row per W / N lanes.
Where a narrow store or an atomic of N cells reads it, a layout conversion moves it to one
row per lane. The conversion asked where its source was held, and for the result of an
``if`` or of a ``while`` loop nothing answered: the rows were taken from the first N lanes,
which all hold row 0 (and row 1 from lane W / N on). With 2 rows on 1024 lanes both cells
got row 0; with 256 rows, cells 0 to 3 got row 0, cells 4 to 7 row 1.

The producer walk and the layout resolver now follow a branch's result to the values its
two sides yield, and a while loop's result to the value it starts from and the value its
body hands on. The conversion then moves the rows from the lanes that hold them, or the
kernel is refused where the two sources are held differently.

The walk also stopped after 256 values. A row-wise sum behind 130 additions was then not
found, and the conversion took the rows from the first N lanes as above. The walk now goes
on to every producer; the long chain is refused, as a chain of 60 additions was.

The same holds where the other side of the branch is a constant, which sits on every lane
alike, and for a counted ``for`` loop: one that starts from zeros and adds sums, and one
that starts from a sum and adds a number. The resolver has to look at both sides of the
branch and at both the start and the handed-on value of the loop; looking at the constant
alone it would call the value uniform and take the rows from the first N lanes.

An operation the resolver does not know (a correctly rounded division or square root) is
not taken as uniform: a row-wise sum behind one is computed or refused.

No mask is used and every tensor holds every cell a program names: every address is inside
its tensor.
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
DTYPES = {"float32": (torch.float32, np.float32), "int32": (torch.int32, np.int32)}
PROGRAMS = 3
LANES = 1024
STORE, ATOMIC_ADD = 0, 1
BRANCH, LOOP_OF_SUMS, LOOP_FROM_SUM, LOOP_FROM_LOAD = 0, 1, 2, 3
FORMS = {
    BRANCH: "a branch chooses the sum",
    LOOP_OF_SUMS: "a while loop adds sums",
    LOOP_FROM_SUM: "a while loop starts from the sum",
    LOOP_FROM_LOAD: "a while loop carries a plain load",
}
USES = {STORE: "narrow store", ATOMIC_ADD: "atomic add"}


@triton.jit
def _rows(X, Y, Z, N: tl.constexpr, W: tl.constexpr, FORM: tl.constexpr, USE: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.arange(0, W)
    n = tl.arange(0, N)
    c = tl.arange(0, W // N)
    wide = tl.load(X + pid * 8 + w)
    if FORM == 0:
        if pid % 2 == 0:
            v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
        else:
            v = (n * 2 + 1).to(X.dtype.element_ty)
    else:
        if FORM == 2:
            v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
        else:
            v = tl.load(X + n + 11)
        j = 0
        while j < pid + 1:
            if FORM == 1:
                v = v + tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + j), axis=1)
            else:
                v = v + (n + 1).to(X.dtype.element_ty)
            j += 1
    cells = Z + pid * 2 * N + n
    if USE == 0:
        tl.store(cells, v)
    else:
        tl.atomic_add(cells, v)
    tl.store(Y + pid * W + w, wide + 1)


@triton.jit
def _sum_then_additions(X, Y, Z, N: tl.constexpr, W: tl.constexpr, STEPS: tl.constexpr, USE: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.arange(0, W)
    n = tl.arange(0, N)
    c = tl.arange(0, W // N)
    wide = tl.load(X + pid * 8 + w)
    v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
    for i in tl.static_range(STEPS):
        v = v + (n + i).to(X.dtype.element_ty)
    cells = Z + pid * 2 * N + n
    if USE == 0:
        tl.store(cells, v)
    else:
        tl.atomic_add(cells, v)
    tl.store(Y + pid * W + w, wide + 1)


ZEROS_OR_SUM, SUM_OR_ZEROS, FULL_OR_SUM, SUM_OR_FULL, FOR_OF_SUMS, FOR_FROM_SUM, FOR_FROM_SUM_OF_SUMS = range(7)
CONSTANT_FORMS = {
    ZEROS_OR_SUM: "a branch chooses zeros or the sum",
    SUM_OR_ZEROS: "a branch chooses the sum or zeros",
    FULL_OR_SUM: "a branch chooses fives or the sum",
    SUM_OR_FULL: "a branch chooses the sum or fives",
    FOR_OF_SUMS: "a for loop adds sums to zeros",
    FOR_FROM_SUM: "a for loop adds numbers to the sum",
    FOR_FROM_SUM_OF_SUMS: "a for loop adds sums to the sum",
}
DIVISION, SQUARE_ROOT = 0, 1
ROUNDED = {DIVISION: "rounded division", SQUARE_ROOT: "rounded square root"}


@triton.jit
def _rows_beside_a_constant(X, Y, Z, N: tl.constexpr, W: tl.constexpr, FORM: tl.constexpr, USE: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.arange(0, W)
    n = tl.arange(0, N)
    c = tl.arange(0, W // N)
    wide = tl.load(X + pid * 8 + w)
    if FORM == 0:
        if pid % 2 == 0:
            v = tl.zeros([N], X.dtype.element_ty)
        else:
            v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
    elif FORM == 1:
        if pid % 2 == 0:
            v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
        else:
            v = tl.zeros([N], X.dtype.element_ty)
    elif FORM == 2:
        if pid % 2 == 0:
            v = tl.full([N], 5, X.dtype.element_ty)
        else:
            v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
    elif FORM == 3:
        if pid % 2 == 0:
            v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
        else:
            v = tl.full([N], 5, X.dtype.element_ty)
    elif FORM == 4:
        v = tl.zeros([N], X.dtype.element_ty)
        for j in range(pid + 1):
            v = v + tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + j), axis=1)
    elif FORM == 5:
        v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
        for j in range(pid + 1):
            v = v + (j + 2)
    else:
        v = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1)
        for j in range(pid + 1):
            v = v + tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + j), axis=1)
    cells = Z + pid * 2 * N + n
    if USE == 0:
        tl.store(cells, v)
    else:
        tl.atomic_add(cells, v)
    tl.store(Y + pid * W + w, wide + 1)


@triton.jit
def _sum_then_rounded(X, Y, Z, N: tl.constexpr, W: tl.constexpr, FORM: tl.constexpr, USE: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.arange(0, W)
    n = tl.arange(0, N)
    c = tl.arange(0, W // N)
    wide = tl.load(X + pid * 8 + w)
    total = tl.sum(tl.load(X + n[:, None] * (W // N) + c[None, :] + pid), axis=1).to(tl.float32)
    if FORM == 0:
        v = tl.div_rn(total, tl.full([N], 4.0, tl.float32)).to(X.dtype.element_ty)
    else:
        v = tl.sqrt_rn(total * total).to(X.dtype.element_ty)
    cells = Z + pid * 2 * N + n
    if USE == 0:
        tl.store(cells, v)
    else:
        tl.atomic_add(cells, v)
    tl.store(Y + pid * W + w, wide + 1)


def _model_beside_a_constant(x, z, rows, form, use):
    z = z.copy()
    y = np.zeros(PROGRAMS * LANES, dtype=x.dtype)
    n = np.arange(rows)
    w = np.arange(LANES)
    tile = n[:, None] * (LANES // rows) + np.arange(LANES // rows)[None, :]
    for pid in range(PROGRAMS):
        total = x[tile + pid].sum(1)
        if form in (ZEROS_OR_SUM, FULL_OR_SUM):
            v = np.full(rows, 0 if form == ZEROS_OR_SUM else 5) if pid % 2 == 0 else total
        elif form in (SUM_OR_ZEROS, SUM_OR_FULL):
            v = total if pid % 2 == 0 else np.full(rows, 0 if form == SUM_OR_ZEROS else 5)
        else:
            v = np.zeros(rows) if form == FOR_OF_SUMS else total
            for j in range(pid + 1):
                v = v + (j + 2 if form == FOR_FROM_SUM else x[tile + j].sum(1))
        cells = pid * 2 * rows + n
        z[cells] = v if use == STORE else z[cells] + v
        y[pid * LANES + w] = x[pid * 8 + w] + 1
    return y, z


def _model_of_rounded(x, z, rows, form, use):
    z = z.copy()
    y = np.zeros(PROGRAMS * LANES, dtype=x.dtype)
    n = np.arange(rows)
    w = np.arange(LANES)
    tile = n[:, None] * (LANES // rows) + np.arange(LANES // rows)[None, :]
    for pid in range(PROGRAMS):
        total = x[tile + pid].sum(1).astype(np.float32)
        # Whole numbers below 2**12: the quotient by 4 and the root of the square are exact.
        v = (total / np.float32(4) if form == DIVISION else total).astype(x.dtype)
        cells = pid * 2 * rows + n
        z[cells] = v if use == STORE else z[cells] + v
        y[pid * LANES + w] = x[pid * 8 + w] + 1
    return y, z


def _model(x, z, rows, form, use):
    z = z.copy()
    y = np.zeros(PROGRAMS * LANES, dtype=x.dtype)
    n = np.arange(rows)
    w = np.arange(LANES)
    tile = n[:, None] * (LANES // rows) + np.arange(LANES // rows)[None, :]
    for pid in range(PROGRAMS):
        if form == BRANCH:
            v = x[tile + pid].sum(1) if pid % 2 == 0 else n * 2 + 1
        else:
            v = x[tile + pid].sum(1) if form == LOOP_FROM_SUM else x[n + 11]
            for j in range(pid + 1):
                v = v + (x[tile + j].sum(1) if form == LOOP_OF_SUMS else n + 1)
        cells = pid * 2 * rows + n
        z[cells] = v if use == STORE else z[cells] + v
        y[pid * LANES + w] = x[pid * 8 + w] + 1
    return y, z


def _model_of_additions(x, z, rows, steps, use):
    z = z.copy()
    y = np.zeros(PROGRAMS * LANES, dtype=x.dtype)
    n = np.arange(rows)
    w = np.arange(LANES)
    tile = n[:, None] * (LANES // rows) + np.arange(LANES // rows)[None, :]
    for pid in range(PROGRAMS):
        v = x[tile + pid].sum(1) + steps * n + steps * (steps - 1) // 2
        cells = pid * 2 * rows + n
        z[cells] = v if use == STORE else z[cells] + v
        y[pid * LANES + w] = x[pid * 8 + w] + 1
    return y, z


def _launch_other(kernel, model, rows, form, use, dtype, device):
    torch_type, numpy_type = DTYPES[dtype]
    x0 = ((np.arange(2 * LANES + 8 * PROGRAMS) * 7 + 3) % 13).astype(numpy_type)
    z0 = ((np.arange(PROGRAMS * 2 * rows) * 5) % 11 + 2).astype(numpy_type)
    x = torch.tensor(x0, device=device)
    z = torch.tensor(z0, device=device)
    y = torch.full((PROGRAMS * LANES,), -7, dtype=torch_type, device=device)
    kernel[(PROGRAMS,)](x, y, z, N=rows, W=LANES, FORM=form, USE=use)
    if device == "mps":
        torch.mps.synchronize()
    return (y.cpu().numpy(), z.cpu().numpy()), model(x0, z0, rows, form, use)


def _launch(rows, form, use, dtype, device, steps=None):
    torch_type, numpy_type = DTYPES[dtype]
    # Small whole numbers: every sum is exact in float32.
    x0 = ((np.arange(2 * LANES + 8 * PROGRAMS) * 7 + 3) % 13).astype(numpy_type)
    z0 = ((np.arange(PROGRAMS * 2 * rows) * 5) % 11 + 2).astype(numpy_type)
    x = torch.tensor(x0, device=device)
    z = torch.tensor(z0, device=device)
    y = torch.full((PROGRAMS * LANES,), -7, dtype=torch_type, device=device)
    if steps is None:
        _rows[(PROGRAMS,)](x, y, z, N=rows, W=LANES, FORM=form, USE=use)
    else:
        _sum_then_additions[(PROGRAMS,)](x, y, z, N=rows, W=LANES, STEPS=steps, USE=use)
    if device == "mps":
        torch.mps.synchronize()
    got = (y.cpu().numpy(), z.cpu().numpy())
    if steps is None:
        return got, _model(x0, z0, rows, form, use)
    return got, _model_of_additions(x0, z0, rows, steps, use)


COMPUTED = [
    (rows, form, use)
    for rows, form in [
        (2, BRANCH),
        (256, BRANCH),
        (2, LOOP_OF_SUMS),
        (2, LOOP_FROM_SUM),
        (256, LOOP_FROM_SUM),
        (2, LOOP_FROM_LOAD),
        (256, LOOP_FROM_LOAD),
    ]
    for use in sorted(USES)
    # A store of 256 loaded cells from 1024 lanes is refused for another reason, as it was.
    if (rows, form, use) != (256, LOOP_FROM_LOAD, STORE)
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize(
    "rows,form,use", COMPUTED, ids=[f"{rows} rows, {FORMS[form]}, {USES[use]}" for rows, form, use in COMPUTED]
)
def test_each_cell_gets_its_own_row(rows, form, use, dtype, device):
    for _ in range(2):
        (y, z), (want_y, want_z) = _launch(rows, form, use, dtype, device)
        assert np.array_equal(z, want_z.astype(z.dtype))
        assert np.array_equal(y, want_y.astype(y.dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("use", sorted(USES), ids=lambda use: USES[use])
def test_a_loop_whose_start_and_sums_are_held_differently_is_refused(use, dtype, device):
    """256 loaded cells sit one per lane; 256 sums of 4 sit one per 4 lanes."""
    with pytest.raises(MetalNonRecoverableError, match="cannot prove its source's thread-to-element mapping"):
        _launch(256, LOOP_OF_SUMS, use, dtype, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("use", sorted(USES), ids=lambda use: USES[use])
@pytest.mark.parametrize("rows", [2, 256])
@pytest.mark.parametrize("steps", [0, 130, 400])
def test_a_sum_behind_many_additions_is_found(steps, rows, use, dtype, device):
    """With no addition the rows are computed. Behind 130 or 400 they are computed or refused."""
    try:
        (y, z), (want_y, want_z) = _launch(rows, None, use, dtype, device, steps=steps)
    except MetalNonRecoverableError as refusal:
        assert steps and "cannot prove its source's thread-to-element mapping" in str(refusal)
        return
    assert np.array_equal(z, want_z.astype(z.dtype))
    assert np.array_equal(y, want_y.astype(y.dtype))


# A branch that yields 256 cells on 1024 lanes beside a constant is refused for another reason.
BESIDE_A_CONSTANT = [
    (rows, form, use)
    for form in sorted(CONSTANT_FORMS)
    for rows in ([2, 256] if form >= FOR_OF_SUMS else [2, 128])
    for use in sorted(USES)
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize(
    "rows,form,use",
    BESIDE_A_CONSTANT,
    ids=[f"{rows} rows, {CONSTANT_FORMS[form]}, {USES[use]}" for rows, form, use in BESIDE_A_CONSTANT],
)
def test_each_cell_gets_its_own_row_beside_a_constant(rows, form, use, dtype, device):
    for _ in range(2):
        (y, z), (want_y, want_z) = _launch_other(
            _rows_beside_a_constant, _model_beside_a_constant, rows, form, use, dtype, device
        )
        assert np.array_equal(z, want_z.astype(z.dtype))
        assert np.array_equal(y, want_y.astype(y.dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("use", sorted(USES), ids=lambda use: USES[use])
@pytest.mark.parametrize("rows", [2, 256])
@pytest.mark.parametrize("form", sorted(ROUNDED), ids=lambda form: ROUNDED[form])
def test_a_sum_behind_an_operation_the_resolver_does_not_know_is_not_taken_as_uniform(form, rows, use, dtype, device):
    """Computed or refused: never the rows of the first lanes."""
    try:
        (y, z), (want_y, want_z) = _launch_other(_sum_then_rounded, _model_of_rounded, rows, form, use, dtype, device)
    except MetalNonRecoverableError as refusal:
        assert "cannot prove its source's thread-to-element mapping" in str(refusal)
        return
    assert np.array_equal(z, want_z.astype(z.dtype))
    assert np.array_equal(y, want_y.astype(y.dtype))
