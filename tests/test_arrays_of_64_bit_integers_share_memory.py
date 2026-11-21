"""A kernel of 64-bit integers reuses its threadgroup arrays as a kernel of 32-bit integers does.

The pass that lets two threadgroup arrays share memory read each declaration against a
list of types, and `long` and `ulong` were not on it. Every array of 64-bit integers
stayed apart: two sorts of 256 int64 lanes kept forty arrays, 61440 bytes where 32768
are allowed, and a flip and a sum of 512 lanes kept as many. The launch failed while
the pipeline was being built, with a RuntimeError that named the byte count or, further
past the limit, only an interrupted connection to the shader compiler. The count that
refuses a kernel before a pipeline is asked for did not know the width of `long` or
`ulong` either, so it let these through.

Arrays of every type the lowerer declares are now reused under the same proofs, and the
count knows eight bytes for `long` and `ulong`. A kernel whose arrays still do not fit
is refused as the out-of-resources error an autotuner already understands.
"""

import re

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from triton.runtime.errors import OutOfResources

import triton_msl  # noqa: F401
from tests import test_while_loops_that_share_lanes as shared_lanes
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen import _lowerer_helpers
from triton_msl.codegen._lowerer_helpers import _alias_shared_memory
from triton_msl.errors import MetalNonRecoverableError, MetalResourceError

needs_metal = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
GRID = shared_lanes.GRID
BODIES = shared_lanes.BODIES
WIDE = [np.int64, np.uint64]
WIDE_IDS = ["int64", "uint64"]
# Every type a threadgroup array is declared with, and its bytes.
DECLARED = {"float": 4, "int": 4, "uint": 4, "long": 8, "ulong": 8, "half": 2, "short": 2, "ushort": 2}
DECLARED.update({"char": 1, "uchar": 1, "bool": 1})
TWICE, FOUR_TIMES = range(2)


@triton.jit
def _sorted_again(X, Z, R: tl.constexpr, C: tl.constexpr, FORM: tl.constexpr):
    # Each program owns R * C cells of X and of Z.
    pid = tl.program_id(0)
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    o = pid * R * C + r * C + c
    x = tl.load(X + o)
    if FORM == 0:
        tl.store(Z + o, tl.sort(tl.sort(x, descending=1)))
    else:
        tl.store(Z + o, tl.sort(tl.sort(tl.sort(tl.sort(x), descending=1)), descending=1))


@triton.jit
def _largest_three_times(X, Z, N: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    x = tl.load(X + pid * N + n)
    y = tl.flip(x, 0) // 2 + tl.max(x, axis=0) // 2
    w = tl.flip(y, 0) // 2 + tl.max(y, axis=0) // 2
    tl.store(Z + pid * N + n, tl.flip(w, 0) // 2 + tl.max(w, axis=0) // 2)


@triton.jit
def _two_types(X, Y, Z, N: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    x = tl.load(X + pid * N + n)
    y = tl.load(Y + pid * N + n)
    a = tl.flip(y, 0) + 1
    b = tl.flip(x, 0) + 1
    c = tl.flip(a, 0) * 3
    d = tl.flip(b, 0) * 5
    tl.store(Z + pid * N + n, tl.flip(d, 0) + tl.flip(c, 0).to(X.dtype.element_ty))


def _launch_looped(kernel, kind, lanes, trips, device, **constants):
    z0, r0 = shared_lanes._cells(lanes)
    z0, r0 = z0.astype(kind), r0.astype(kind)
    z, r = torch.tensor(z0, device=device), torch.tensor(r0, device=device)
    kernel[(GRID,)](z, r, trips, N=lanes, **constants)
    if device == "mps":
        torch.mps.synchronize()
    return z.cpu().numpy(), r.cpu().numpy()


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("counted", [0, 1], ids=["while", "counted"])
@pytest.mark.parametrize("body", range(len(BODIES)), ids=BODIES)
@pytest.mark.parametrize("trips", [0, 2, 5])
@pytest.mark.parametrize("lanes", [256, 512, 1024])
@pytest.mark.parametrize("kind", WIDE, ids=WIDE_IDS)
def test_a_loop_over_a_block_of_64_bit_lanes_computes(kind, device, counted, body, trips, lanes):
    """The kernels of the int32 tests, on 64-bit cells: the same values, eight bytes a lane."""
    z, r = _launch_looped(shared_lanes._looped, kind, lanes, trips, device, BODY=body, COUNTED=counted)
    want_z, want_r = shared_lanes._looped_expected(lanes, trips, body)
    assert z.dtype == r.dtype == kind
    assert np.array_equal(z, want_z.astype(kind))
    assert np.array_equal(r, want_r.astype(kind))


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lanes", [64, 256, 1024])
@pytest.mark.parametrize("kind", WIDE, ids=WIDE_IDS)
def test_the_largest_of_a_block_taken_three_times_keeps_all_64_bits(kind, lanes, device):
    x0 = _sort_cells(kind, lanes, 1)
    x = torch.tensor(x0, device=device)
    z = torch.zeros_like(x)
    _largest_three_times[(GRID,)](x, z, N=lanes)
    if device == "mps":
        torch.mps.synchronize()
    want = []
    for program in range(GRID):
        cells = [int(v) for v in x0[program * lanes : (program + 1) * lanes]]
        for _ in range(3):
            top = max(cells)
            # ``//`` rounds toward zero in the kernel.
            cells = [int(v / 2) if abs(v) < 2 else (abs(v) // 2) * (1 if v > 0 else -1) for v in reversed(cells)]
            half = (abs(top) // 2) * (1 if top > 0 else -1)
            cells = [v + half for v in cells]
        want += cells
    assert np.array_equal(z.cpu().numpy(), np.array(want, dtype=object).astype(kind))


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lanes", [64, 256, 1024])
@pytest.mark.parametrize("narrow", [np.int32, np.uint32], ids=["int32", "uint32"])
@pytest.mark.parametrize("kind", WIDE, ids=WIDE_IDS)
def test_arrays_of_64_bit_and_of_32_bit_integers_in_one_kernel_stay_apart(kind, narrow, lanes, device):
    """Shared, either the long values would lose their upper half or the others would be read as halves."""
    x0 = (np.arange(GRID * lanes).astype(kind) * kind(40503) + kind(1 << 40)) % kind(1 << 60)
    y0 = (np.arange(GRID * lanes) * 7 % 1013).astype(narrow)
    x, y = torch.tensor(x0, device=device), torch.tensor(y0, device=device)
    z = torch.zeros_like(x)
    _two_types[(GRID,)](x, y, z, N=lanes)
    if device == "mps":
        torch.mps.synchronize()
    xs, ys = x0.reshape(GRID, lanes), y0.reshape(GRID, lanes).astype(kind)
    want = (xs[:, ::-1] + kind(1)) * kind(5) + (ys[:, ::-1] + kind(1)) * kind(3)
    assert np.array_equal(z.cpu().numpy(), want.reshape(-1))


def _sort_cells(kind, rows, columns):
    info = np.iinfo(kind)
    total = GRID * rows * columns
    rng = np.random.default_rng(total + info.bits + (info.min < 0))
    x = rng.integers(info.min, info.max, total, endpoint=True, dtype=kind)
    x[3::17] = info.max
    x[5::19] = info.min
    x[4::9] = x[2::9][: x[4::9].size]
    # Past what a float32, a float64 or an int32 holds: an array that held any of them would lose these.
    x[7::23] = (1 << 53) + 1
    x[8::29] = (1 << 62) + (1 << 31) + 1
    x[9::31] = info.max - 1
    return x


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", [TWICE, FOUR_TIMES], ids=["sorted twice", "sorted four times"])
@pytest.mark.parametrize("shape", [(16, 16), (32, 16), (4, 64), (64, 4)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", WIDE, ids=WIDE_IDS)
def test_a_sort_of_a_sort_of_64_bit_integers_computes(kind, shape, form, device):
    rows, columns = shape
    x0 = _sort_cells(kind, rows, columns)
    x = torch.tensor(x0, device=device)
    z = torch.zeros_like(x)
    _sorted_again[(GRID,)](x, z, R=rows, C=columns, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    want = np.sort(x0.reshape(GRID * rows, columns), axis=1)
    if form == FOUR_TIMES:
        want = want[:, ::-1]
    assert np.array_equal(z.cpu().numpy(), want.reshape(-1))
    assert np.array_equal(x.cpu().numpy(), x0)


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", [0, 1], ids=["a flip on some trips", "a sum in the test"])
@pytest.mark.parametrize("lanes,required", [(512, 59392), (1024, 131072)])
def test_int64_arrays_that_cannot_fit_are_refused_as_out_of_resources(device, form, lanes, required):
    """Counted at four bytes, the 512 lanes would pass the count and fail in the pipeline."""
    z0, r0 = shared_lanes._cells(lanes)
    z0, r0 = z0.astype(np.int64), r0.astype(np.int64)
    z, r = torch.tensor(z0, device=device), torch.tensor(r0, device=device)
    with pytest.raises(OutOfResources, match="threadgroup memory bytes") as refusal:
        shared_lanes._too_many_arrays[(GRID,)](z, r, 2, N=lanes, FORM=form)
    assert (refusal.value.required, refusal.value.limit) == (required, 32768)
    assert np.array_equal(z.cpu().numpy(), z0)
    assert np.array_equal(r.cpu().numpy(), r0)


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", [0, 1], ids=["a flip on some trips", "a sum in the test"])
@pytest.mark.parametrize("trips", [0, 2, 5])
def test_the_same_int64_kernels_compute_where_their_arrays_fit(device, form, trips):
    lanes = 256
    z0, r0 = shared_lanes._cells(lanes)
    z, r = _launch_looped(shared_lanes._too_many_arrays, np.int64, lanes, trips, device, FORM=form)
    want = r0.astype(np.int64)
    n = np.arange(lanes)
    for pid in range(GRID):
        x = z0[pid * 4 * lanes + n].astype(np.int64)
        for j in range(trips):
            if form == 0:
                x = (x[::-1] + 1) if j % 2 == 0 else (x * 2 % 1009)
                x = x + x.sum() % 7
            else:
                x = (x[::-1] + j) % 1009
        want[pid * 4 * lanes + n] = x
    assert np.array_equal(z, z0)
    assert np.array_equal(r, want)


def _text(kernel, signature, constants):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(fn=kernel, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options)


def _arrays(text):
    return sorted(re.findall(r"^\s*threadgroup\s+(\w+)\s+\w+\[(\d+)\];", text, re.M))


@pytest.mark.parametrize("counted", [0, 1], ids=["while", "counted"])
@pytest.mark.parametrize("body", range(len(BODIES)), ids=BODIES)
@pytest.mark.parametrize("lanes", [256, 1024])
def test_a_loop_keeps_as_many_arrays_of_int64_as_of_int32(lanes, body, counted):
    """Read from the text, no launch."""
    constants = {"N": lanes, "BODY": body, "COUNTED": counted}
    narrow = _arrays(_text(shared_lanes._looped, {"Z": "*i32", "R": "*i32", "TRIPS": "i32"}, constants))
    wide = _arrays(_text(shared_lanes._looped, {"Z": "*i64", "R": "*i64", "TRIPS": "i32"}, constants))
    # A sum of int64 lanes is staged a lane to a cell, so one array may be longer; none is one more.
    assert {kind for kind, _ in narrow} == {"int"}
    assert {kind for kind, _ in wide} == {"long"}
    assert len(wide) == len(narrow) <= 3
    assert all(int(count) <= lanes for _, count in wide)


@pytest.mark.parametrize("form", [TWICE, FOUR_TIMES], ids=["sorted twice", "sorted four times"])
@pytest.mark.parametrize("wide,narrow", [("i64", "i32"), ("u64", "u32")])
def test_sorts_keep_as_many_arrays_of_64_bit_integers_as_of_32_bit(wide, narrow, form):
    """Read from the text, no launch. A sorted value is held as its bits, in ``long`` or ``int``."""
    constants = {"R": 16, "C": 16, "FORM": form}
    short = _arrays(_text(_sorted_again, {"X": "*" + narrow, "Z": "*" + narrow}, constants))
    long = _arrays(_text(_sorted_again, {"X": "*" + wide, "Z": "*" + wide}, constants))
    assert short == [("int", "128"), ("int", "256")]
    assert long == [("long", "128"), ("long", "256")]
    assert _lowerer_helpers._check_generic_shared_memory_budget(_text_of(long)) <= 32768


@pytest.mark.parametrize("lanes", [64, 256, 1024])
def test_three_largest_values_of_uint64_lanes_are_staged_in_one_array_of_ulong(lanes):
    """Read from the text, no launch. The largest value is staged as unsigned, the flips as bits."""
    arrays = _arrays(_text(_largest_three_times, {"X": "*u64", "Z": "*u64"}, {"N": lanes}))
    assert [kind for kind, _ in arrays] == ["long", "long", "ulong"]


@pytest.mark.parametrize("lanes", [64, 256, 1024])
@pytest.mark.parametrize("narrow", ["i32", "u32"])
@pytest.mark.parametrize("wide", ["i64", "u64"])
def test_a_kernel_of_two_widths_keeps_two_arrays_of_each(wide, narrow, lanes):
    """Read from the text, no launch."""
    arrays = _arrays(_text(_two_types, {"X": "*" + wide, "Y": "*" + narrow, "Z": "*" + wide}, {"N": lanes}))
    assert [kind for kind, _ in arrays] == ["int", "int", "long", "long"]


def _text_of(arrays):
    lines = "\n".join(f"    threadgroup {kind} shared_{i}[{count}];" for i, (kind, count) in enumerate(arrays))
    return "kernel void probe() {\n" + lines + "\n}"


BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"


def _two(first, second, boundary=BARRIER):
    zero = {"bool": "false"}
    return f"""kernel void probe(uint lid [[thread_index_in_threadgroup]]) {{
    threadgroup {first} a[128];
    threadgroup {second} b[128];
    a[lid] = {zero.get(first, "1")};
    {first} old = a[127 - lid];
    {boundary}
    b[lid] = {zero.get(second, "2")};
}}"""


@pytest.mark.parametrize("kind", sorted(DECLARED))
def test_two_arrays_of_one_type_share_whatever_the_type(kind):
    aliases = {}
    text = _alias_shared_memory(_two(kind, kind), allocation_aliases=aliases)
    assert _arrays(text) == [(kind, "128")]
    assert aliases == {"a": "a", "b": "a"}
    assert "b[lid]" not in text
    # ... and only across a barrier, as before.
    apart = _alias_shared_memory(_two(kind, kind, boundary=""), allocation_aliases=aliases)
    assert _arrays(apart) == [(kind, "128"), (kind, "128")]
    assert aliases.get("b", "b") == "b"


@pytest.mark.parametrize(
    "first,second",
    [("long", "ulong"), ("ulong", "long"), ("long", "int"), ("int", "long"), ("ulong", "uint"), ("long", "float")],
)
def test_arrays_of_two_types_stay_apart(first, second):
    """An array holds the bits of its own type; one of another type is another array."""
    aliases = {}
    text = _alias_shared_memory(_two(first, second), allocation_aliases=aliases)
    assert _arrays(text) == sorted([(first, "128"), (second, "128")])
    assert aliases.get("b", "b") == "b"
    assert "b[lid]" in text


def test_three_arrays_share_by_type():
    text = f"""kernel void probe(uint lid [[thread_index_in_threadgroup]]) {{
    threadgroup long a[128];
    threadgroup int b[128];
    threadgroup long c[64];
    a[lid] = 1;
    long old = a[127 - lid];
    {BARRIER}
    b[lid] = 2;
    int kept = b[127 - lid];
    {BARRIER}
    c[lid % 64] = old + kept;
}}"""
    aliases = {}
    shared = _alias_shared_memory(text, allocation_aliases=aliases)
    assert _arrays(shared) == [("int", "128"), ("long", "128")]
    assert aliases == {"a": "a", "b": "b", "c": "a"}


@pytest.mark.parametrize("kind,width", sorted({**DECLARED, "bfloat": 2}.items()))
def test_the_count_knows_the_bytes_of_every_type(kind, width):
    most = 32768 // width
    assert _lowerer_helpers._check_generic_shared_memory_budget(_text_of([(kind, most)])) == 32768
    with pytest.raises(MetalResourceError) as refusal:
        _lowerer_helpers._check_generic_shared_memory_budget(_text_of([(kind, most), ("bool", 1)]))
    assert (refusal.value.required, refusal.value.limit) == (32769, 32768)
    assert refusal.value.resource == "threadgroup memory bytes"
    assert not isinstance(refusal.value, MetalNonRecoverableError)
