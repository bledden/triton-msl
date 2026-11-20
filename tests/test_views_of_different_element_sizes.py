"""Views of one storage with different element sizes are compared by the bytes they reach.

The lane-order check compares the cells two arguments of one storage reach, and it used
to count them in elements of one size: a float32 input and an int16 output that were
views of one buffer were refused wherever they lay, although the kernel on blocks that
share no byte computes what it computes on two buffers. The check now counts in the
smallest element size among the views:

* views that share no byte compute, a few bytes apart or touching, on the device and on
  host tensors, with a narrow view at an odd byte among them;
* a byte that views of two sizes both reach, one of them storing it, is refused whichever
  lanes reach it, and the storage is left as it was: the output begun inside the input's
  last element, the input begun inside the output's last element, both at the same byte,
  and blocks that meet in a later program only;
* a row sum stored over the bytes it summed computes at any element size, as it does at
  one: the shader has loaded every element before its barrier and stores after it;
* bytes that two views of different sizes only load are no meeting;
* host arrays with storages of their own that overlap in memory are compared the same
  way, and a host array that is not at a multiple of its element size is refused: the
  comparison has no whole number of its elements to count from;
* every byte counts toward the limit on the addresses compared at a time;
* views of one size and two types keep the comparison by lane they had.

Every launch keeps to its storage: each view has the cells its kernel reaches, masks keep
to N, and the storage ends with a guard.

The reason text is written out here, not imported, so that the file is collected on a
tree without the change and its tests fail one by one there.
"""

import collections

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from tests.test_lane_order_launch import DEVICES, requires_mps
from triton_msl.backend import _lane_order
from triton_msl.errors import MetalNonRecoverableError

pytestmark = requires_mps

SIZE = (
    "arguments of different element sizes reach the same bytes of the memory they share, or one of them "
    "is not aligned to its element size"
)
MANY = "there are too many addresses to compare"
GUARD = 64  # bytes after the last view
GAP = 24  # bytes between views that lie apart


@triton.jit
def _scaled(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(Y + i, tl.load(X + i, mask=m) * 2 + 1, mask=m)


@triton.jit
def _read_back(X, Y, Z, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, v * 2 + 1)
    u = tl.load(X + i)
    tl.store(Z + i, u + v)


@triton.jit
def _row_sum(X, Y, Z, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    tl.store(Y + r, tl.sum(x, 1))
    tl.store(Z + r[:, None] * C + c[None, :], x + 3)


@triton.jit
def _summed_trips(X, Y, Z, TRIPS, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.int32)
    for t in range(TRIPS):
        x = tl.load(X + t * BLOCK + i)
        acc += x.to(tl.int32)
        tl.store(Z + t * BLOCK + i, x + 1)
    tl.store(Y + i, acc)


@triton.jit
def _scaled_by_first(X, Y, Z, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    s = tl.load(X)
    tl.store(Z + i, tl.load(Y + i) * s)


@triton.jit
def _counted(X, Y, Z, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i)
    tl.atomic_add(Y + i, x)
    tl.store(Z + i, x * 2)


# What each kernel leaves in the arguments it stores, from the cells it reaches.
def _scaled_model(x, y, n):
    return {1: x[:n] * 2 + 1}


def _read_back_model(x, y, z):
    return {1: x * 2 + 1, 2: x + x}


def _row_sum_model(rows):
    def model(x, y, z):
        return {1: x.reshape(rows, -1).sum(1), 2: x + 3}

    return model


def _summed_trips_model(x, y, z, trips):
    return {1: x.reshape(trips, -1).sum(0), 2: x + 1}


def _scaled_by_first_model(x, y, z):
    return {2: y * x[0]}


def _counted_model(x, y, z):
    return {1: y + x, 2: x * 2}


# ``cells`` is how many elements the kernel reaches through each pointer argument.
Form = collections.namedtuple("Form", "name kernel constants scalars cells model types")

THREE = [
    ("float32", "int16", "uint8"),
    ("int16", "float32", "int32"),
    ("uint8", "int32", "int16"),
    ("int32", "int16", "int16"),
    ("float32", "float16", "float32"),
]


def _forms(block):
    rows = 8
    return [
        Form(f"read back, {block}", _read_back, dict(BLOCK=block), (), (block,) * 3, _read_back_model, THREE),
        Form(
            f"row sum, {rows} by {block // rows}",
            _row_sum,
            dict(R=rows, C=block // rows),
            (),
            (block, rows, block),
            _row_sum_model(rows),
            THREE,
        ),
        Form(
            f"summed trips, {block}",
            _summed_trips,
            dict(BLOCK=block),
            (3,),
            (3 * block, block, 3 * block),
            _summed_trips_model,
            THREE,
        ),
        Form(
            f"scaled by the first cell, {block}",
            _scaled_by_first,
            dict(BLOCK=block),
            (),
            (1, block, block),
            _scaled_by_first_model,
            THREE,
        ),
        Form(
            f"counted, {block}",
            _counted,
            dict(BLOCK=block),
            (),
            (block,) * 3,
            _counted_model,
            [("int16", "int32", "uint8"), ("uint8", "int32", "float32"), ("float32", "int32", "int16")],
        ),
    ]


FORMS = [form for block in (64, 1024) for form in _forms(block)]
THREE_VIEWS = [(form, types) for form in FORMS for types in form.types]
PAIRS = [
    ("float32", "int16"),
    ("float32", "uint8"),
    ("int16", "float32"),
    ("uint8", "float32"),
    ("float16", "float32"),
    ("float32", "float16"),
    ("int32", "int64"),
    ("int64", "int32"),
    ("int32", "int8"),
]
# (elements, block): several programs with a masked tail, whole blocks, and one program.
EXTENTS = [(300, 128), (256, 128), (2048, 1024), (100, 128)]
ONE_PROGRAM = [(128, 128), (100, 128), (1024, 1024), (40, 64)]


def _elementwise(types, count, block):
    name = f"{types[0]} to {types[1]}, {count} of {block}"
    return Form(name, _scaled, dict(BLOCK=block), (count,), (count, count), _scaled_model, [types])


def _width(name):
    return np.dtype(name).itemsize


def _torch(name):
    return torch.from_numpy(np.zeros(1, name)).dtype


def _apart(form, types, gap=GAP, first=0):
    """Byte at which each view starts: one after another, ``gap`` bytes between them."""
    starts, at = [], first
    for cells, name in zip(form.cells, types):
        at += -at % 8 if starts else -at % _width(name)
        starts.append(at)
        at += cells * _width(name) + gap
    return starts


def _values(cells, name, salt):
    return ((np.arange(cells) + salt) % 5 + 1).astype(name)


def _build(form, types, starts, device):
    """(views, storage, its bytes before the launch): every argument a view of one storage."""
    ends = [start + cells * _width(name) for start, cells, name in zip(starts, form.cells, types)]
    total = max(ends) + GUARD
    host = np.full(total + -total % 8, 0x11, dtype=np.uint8)
    # The last argument first: where views share bytes the earlier argument's values stand.
    for position, start, cells, name in reversed(list(zip(range(len(types)), starts, form.cells, types))):
        assert start >= 0 and start % _width(name) == 0
        host[start : start + cells * _width(name)] = _values(cells, name, position).view(np.uint8)
    storage = torch.from_numpy(host.copy()).to(device)
    views = [storage.view(_torch(name))[start // _width(name) :] for start, name in zip(starts, types)]
    return views, storage, host


def _launch(form, views, device):
    grid = -(-form.scalars[0] // form.constants["BLOCK"]) if form.kernel is _scaled else 1
    form.kernel[(grid,)](*views, *form.scalars, **form.constants)
    if device == "mps":
        torch.mps.synchronize()


def _expected(form, types, starts, host):
    """The storage after the launch: the model's values in the stored views, the rest as it was."""
    before = [
        host[start : start + cells * _width(name)].view(name).astype(np.float64)
        for start, cells, name in zip(starts, form.cells, types)
    ]
    want = host.copy()
    for position, value in form.model(*before, *form.scalars).items():
        name, start = types[position], starts[position]
        written = np.asarray(value).astype(name)
        want[start : start + written.size * _width(name)] = written.view(np.uint8)
    return want


def _computes(form, types, starts, device):
    views, storage, host = _build(form, types, starts, device)
    _launch(form, views, device)
    np.testing.assert_array_equal(storage.cpu().numpy(), _expected(form, types, starts, host))


def _refuses(form, types, starts, device, names, reason=SIZE):
    views, storage, host = _build(form, types, starts, device)
    with pytest.raises(MetalNonRecoverableError) as caught:
        _launch(form, views, device)
    message = str(caught.value)
    assert "lane order:" in message
    assert f"for the memory of argument(s) {names}: {reason}. The source finishes each operation" in message
    np.testing.assert_array_equal(storage.cpu().numpy(), host)


def _ids(rows):
    return [f"{row[0].name}, {'/'.join(row[1])}" for row in rows]


def _inside(form, types, other):
    """The views apart, but the last one begun inside the last element of view ``other``."""
    starts = _apart(form, types)
    last = starts[other] + form.cells[other] * _width(types[other]) - 1
    starts[2] = last - last % _width(types[2])
    if other == 0:
        starts[1] = starts[2] + form.cells[2] * _width(types[2]) + GAP
        starts[1] += -starts[1] % 8
    return starts


ELEMENTWISE = [(_elementwise(types, count, block), types) for types in PAIRS for count, block in EXTENTS]
IN_ONE_PROGRAM = [(_elementwise(types, count, block), types) for types in PAIRS for count, block in ONE_PROGRAM]


# ------------------------------------------------------------------- no byte shared
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gap", (GAP, 0), ids=("apart", "touching"))
@pytest.mark.parametrize(("form", "types"), ELEMENTWISE, ids=_ids(ELEMENTWISE))
def test_an_output_of_another_element_size_beside_the_input_computes(form, types, gap, device):
    _computes(form, types, _apart(form, types, gap), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gap", (GAP, 0), ids=("apart", "touching"))
@pytest.mark.parametrize(("form", "types"), ELEMENTWISE, ids=_ids(ELEMENTWISE))
def test_an_output_of_another_element_size_before_the_input_computes(form, types, gap, device):
    cells = form.cells[0]
    gap += -(cells * _width(types[1]) + gap) % 8
    _computes(form, types, [cells * _width(types[1]) + gap, 0], device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gap", (GAP, 0), ids=("apart", "touching"))
@pytest.mark.parametrize(("form", "types"), THREE_VIEWS, ids=_ids(THREE_VIEWS))
def test_three_views_of_one_storage_compute(form, types, gap, device):
    _computes(form, types, _apart(form, types, gap), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("first", (1, 3))
@pytest.mark.parametrize("other", ("int16", "float32", "int64"))
def test_a_narrow_view_at_an_odd_byte_does_not_put_the_wider_ones_out_of_step(other, first, device):
    # The lowest pointer of the group is the byte view, at a byte no wider view could start at.
    for types in (("uint8", other), ("int8", other)):
        form = _elementwise(types, 300, 128)
        _computes(form, types, _apart(form, types, first=first), device)
    form = FORMS[0]
    types = ("uint8", other, "int16")
    _computes(form, types, _apart(form, types, first=first), device)


# --------------------------------------------------- a byte reached at two element sizes
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("form", "types"), IN_ONE_PROGRAM, ids=_ids(IN_ONE_PROGRAM))
def test_an_output_begun_inside_the_inputs_last_element_refuses(form, types, device):
    wide, narrow = _width(types[0]), _width(types[1])
    start = form.cells[0] * wide - min(wide, narrow)
    start -= start % narrow
    _refuses(form, types, [0, start], device, "X, Y")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("form", "types"), IN_ONE_PROGRAM, ids=_ids(IN_ONE_PROGRAM))
def test_an_input_begun_inside_the_outputs_last_element_refuses(form, types, device):
    wide, narrow = _width(types[0]), _width(types[1])
    start = form.cells[1] * narrow - min(wide, narrow)
    start -= start % wide
    _refuses(form, types, [start, 0], device, "X, Y")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("form", "types"), ELEMENTWISE, ids=_ids(ELEMENTWISE))
def test_the_same_bytes_at_two_element_sizes_refuse(form, types, device):
    _refuses(form, types, [0, 0], device, "X, Y")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", (64, 1024))
@pytest.mark.parametrize("types", [("float32", "int16"), ("int32", "uint8"), ("int64", "int32")])
def test_blocks_that_meet_in_a_later_program_only_refuse(types, block, device):
    # The output starts where the first program's input block ends. Program 0 reaches
    # no byte twice; program 1 stores inside the block it loads, since the two views
    # move on by different numbers of bytes from program to program.
    form = _elementwise(types, 2 * block, block)
    _refuses(form, types, [0, block * _width(types[0])], device, "X, Y")


# The last argument is stored by every kernel here; begin it inside an earlier view of
# another element size that the kernel reaches. The row sum has loaded its input
# before anything is stored, so there it is the other output.
BEGUN_INSIDE = [
    (form, types, other)
    for form, types in THREE_VIEWS
    for other in (0, 1)
    if _width(types[other]) != _width(types[2]) and not (form.kernel is _row_sum and other == 0)
]
SUMMED_OVER = [
    (form, types) for form, types in THREE_VIEWS if form.kernel is _row_sum and _width(types[0]) != _width(types[2])
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("form", "types", "other"), BEGUN_INSIDE, ids=_ids(BEGUN_INSIDE))
def test_a_stored_view_begun_inside_a_view_of_another_size_refuses(form, types, other, device):
    _refuses(form, types, _inside(form, types, other), device, "X, Y, Z")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("form", "types"), SUMMED_OVER, ids=_ids(SUMMED_OVER))
def test_a_row_sum_stored_over_the_bytes_it_summed_computes(form, types, device):
    _computes(form, types, _inside(form, types, 0), device)


# ------------------------------------------------------- one size, two element types
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("count", "block"), EXTENTS)
@pytest.mark.parametrize("types", [("int32", "float32"), ("float32", "int32"), ("int16", "float16")])
def test_the_same_cells_at_one_size_and_two_types_still_compute(types, count, block, device):
    # Each lane loads and stores its own cell: the comparison by lane, as before.
    form = _elementwise(types, count, block)
    _computes(form, types, [0, 0], device)


# ------------------------------------------------------------- bytes that are only loaded
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", (64, 1024))
@pytest.mark.parametrize(
    "types",
    [
        ("uint8", "int32", "int16"),
        ("int32", "int16", "int16"),
        ("int16", "float32", "int32"),
        ("float32", "float16", "float32"),
    ],
)
def test_two_loaded_views_over_the_same_bytes_compute(types, block, device):
    # The scale is loaded from inside the block it multiplies, at another element size;
    # the product is stored apart. Nothing stores a byte the two loads share.
    form = next(form for form in _forms(block) if form.kernel is _scaled_by_first)
    starts = _apart(form, types)
    starts[0] = starts[1]
    _computes(form, types, starts, device)


# --------------------------------------------------- host arrays that overlap in memory
def _host_arrays(types, cells, lap, odd=False):
    """(memory, X, Y, byte Y starts at): X then Y, two arrays, Y begun ``lap`` bytes inside X."""
    wide, narrow = _width(types[0]), _width(types[1])
    raw = np.full(cells * (wide + narrow) + GUARD, 0x11, dtype=np.uint8)
    memory = raw[(raw.ctypes.data + 1) % 2 :] if odd else raw
    start = cells * wide - lap
    x = memory[: cells * wide].view(types[0])
    y = memory[start : start + cells * narrow].view(types[1])
    x[:] = _values(cells, types[0], 0)
    return memory, torch.from_numpy(x), torch.from_numpy(y), start


HOST_PAIRS = [("float32", "int16"), ("int16", "float32"), ("int32", "uint8"), ("uint8", "int32"), ("int64", "int32")]


@pytest.mark.parametrize("types", HOST_PAIRS, ids="/".join)
def test_host_arrays_that_overlap_compute_where_the_kernel_stops_short_of_the_shared_bytes(types):
    wide, narrow = _width(types[0]), _width(types[1])
    memory, x, y, start = _host_arrays(types, 100, 8)
    count = 100 - 8 // wide
    want = memory.copy()
    stored = (memory[: count * wide].view(types[0]).astype(np.float64) * 2 + 1).astype(types[1])
    want[start : start + count * narrow] = stored.view(np.uint8)
    _scaled[(1,)](x, y, count, BLOCK=128)
    np.testing.assert_array_equal(memory, want)


@pytest.mark.parametrize("types", HOST_PAIRS, ids="/".join)
def test_host_arrays_that_overlap_refuse_where_the_kernel_reaches_the_shared_bytes(types):
    memory, x, y, _ = _host_arrays(types, 100, 8)
    before = memory.copy()
    with pytest.raises(MetalNonRecoverableError, match="lane order: .*" + SIZE):
        _scaled[(1,)](x, y, 100, BLOCK=128)
    np.testing.assert_array_equal(memory, before)


@pytest.mark.parametrize("types", HOST_PAIRS, ids="/".join)
def test_a_host_array_at_no_multiple_of_its_element_size_refuses(types):
    # The same launch as the one that computes above, every address one byte on.
    memory, x, y, _ = _host_arrays(types, 100, 8, odd=True)
    assert memory.ctypes.data % 2 == 1
    before = memory.copy()
    with pytest.raises(MetalNonRecoverableError, match="lane order: .*" + SIZE):
        _scaled[(1,)](x, y, 100 - 8 // _width(types[0]), BLOCK=128)
    np.testing.assert_array_equal(memory, before)


# ------------------------------------------------------------------ the comparison's limit
@pytest.mark.parametrize("device", DEVICES)
def test_every_byte_of_an_element_counts_toward_the_limit(device, monkeypatch):
    # 128 elements of four bytes and 128 of one: 640 bytes to compare, 256 elements.
    types = ("float32", "uint8")
    form = _elementwise(types, 128, 128)
    # A layout of its own for each device: an answer once found for a layout is kept.
    starts = _apart(form, types, gap=40 + 8 * DEVICES.index(device))
    monkeypatch.setattr(_lane_order, "BUDGET", 639)
    _refuses(form, types, starts, device, "X, Y", MANY)
    monkeypatch.setattr(_lane_order, "BUDGET", 640)
    _computes(form, types, starts, device)


# ------------------------------------------------ an access into arguments of two sizes
def test_an_access_into_arguments_of_two_element_sizes_is_not_counted():
    # No source reaches this, since a pointer chosen between two arguments has one
    # element type and a cast pointer is an address the check does not compute. The
    # comparison does not rely on that.
    check = object.__new__(_lane_order.LaunchCheck)
    check.bases = [frozenset((0, 1)), frozenset((1,)), None]
    assert check._scale(0, {}) == 1
    assert check._scale(0, {0: 2, 1: 2}) == 2
    assert check._scale(1, {0: 4, 1: 1}) == 1
    for ordinal in (0, 2):
        with pytest.raises(_lane_order.Unproved) as caught:
            check._scale(ordinal, {0: 4, 1: 1})
        assert caught.value.reason == "element-size"
