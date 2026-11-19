"""A float16 or bfloat16 atomic add finds its word from the element's own address.

Metal has no 16-bit device atomic, so the ordinary lowering updates a 16-bit float cell
by a compare-and-swap of the 4-byte word that holds it and keeps the other half of the
word. It used to find that word from the element's index, ``(atomic_uint*)Z + (i >> 1)``
with the half chosen by ``i & 1``, which is the right word only when ``Z`` itself starts a
word. A tensor that is a view starting an odd number of elements into its storage is bound
two bytes into a word: every add then landed on the cell before the one named, on the
device and on a host tensor alike. The same text turned a negative index into an unsigned
one, far outside the storage.

The word is now the one the element's address lies in. These tests compare the whole
storage, so an update beside the view is seen.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import triton_msl  # noqa: F401
from triton_msl.backend.compiler import MetalBackend

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
OFFSETS = [0, 1, 2, 3]
SIZES = [16, 256, 2048]
GUARD = 8


@triton.jit
def _add(Z, R, IDX, TRIPS, BACK, N: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    cells = Z + pid * N
    one = tl.full([N], 1, Z.dtype.element_ty)
    if FORM == 0:  # each lane its own cell
        tl.atomic_add(cells + n, one * 3)
    elif FORM == 1:  # the value the add returns is stored
        old = tl.atomic_add(cells + n, one * 3)
        tl.store(R + pid * N + n, old)
    elif FORM == 2:  # in a `while` loop
        j = 0
        while j < TRIPS:
            tl.atomic_add(cells + n, one * 3)
            j += 1
    elif FORM == 3:  # two lanes on each loaded cell
        tl.atomic_add(cells + tl.load(IDX + n), one)
    elif FORM == 4:  # masked
        tl.atomic_add(cells + n, one * 5, mask=n % 3 != 0)
    else:  # 5: cells before the tensor's first element, inside its storage
        tl.atomic_add(cells + n - BACK, one * 3)


FORMS = {0: "own cell", 1: "result stored", 2: "while loop", 3: "lanes meet", 4: "masked", 5: "negative offset"}


def _dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _expected(form, start, old, offset, size, programs, extra, trips, back):
    """The storages the source leaves, in float64 (every value is a small integer)."""
    cells, returned = start.copy(), old.copy()
    view = cells[offset : offset + programs * size + extra]
    lanes = np.arange(size)
    for program in range(programs):
        row = view[program * size : (program + 1) * size]
        if form == 0:
            row += 3
        elif form == 1:
            returned[offset + program * size : offset + (program + 1) * size] = row
            row += 3
        elif form == 2:
            row += 3 * trips
        elif form == 3:
            np.add.at(row, lanes * 7 % (size // 2), 1)
        elif form == 4:
            row[lanes % 3 != 0] += 5
        else:
            first = offset + program * size - back
            cells[first : first + size] += 3
    return cells, returned


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("extra", [0, 1], ids=["even length", "odd length"])
@pytest.mark.parametrize("programs", [1, 3])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("form", sorted(FORMS), ids=[FORMS[form] for form in sorted(FORMS)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_a_16_bit_float_atomic_add_on_a_view_updates_the_cells_it_names(
    dtype, form, offset, size, programs, extra, device
):
    back = min(offset, 3) if form == 5 else 0
    if form == 5 and back == 0:
        pytest.skip("no element before the first one")
    trips = 2
    elements = programs * size + extra
    total = offset + elements + GUARD
    # bfloat16 holds the integers up to 256 exactly: 60 at the start and at most 6 added.
    start = (np.arange(total) * 3 % 61).astype(np.float64)
    old = np.full(total, -7.0)
    index = (np.arange(size) * 7 % (size // 2)).astype(np.int32)
    try:
        cells = torch.tensor(start, dtype=_dtype(dtype), device=device)
        returned = torch.tensor(old, dtype=_dtype(dtype), device=device)
    except (RuntimeError, TypeError) as error:  # a type this torch build has no device tensor for
        pytest.skip(str(error))
    _add[(programs,)](
        cells[offset : offset + elements],
        returned[offset : offset + elements],
        torch.tensor(index, device=device),
        trips,
        back,
        N=size,
        FORM=form,
    )
    if device == "mps":
        torch.mps.synchronize()
    want_cells, want_returned = _expected(form, start, old, offset, size, programs, extra, trips, back)
    np.testing.assert_array_equal(cells.float().cpu().numpy().astype(np.float64), want_cells)
    np.testing.assert_array_equal(returned.float().cpu().numpy().astype(np.float64), want_returned)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_the_word_of_a_16_bit_float_atomic_is_taken_from_the_address(dtype):
    """Not from the index: an unsigned index shifted right names another word for a view
    two bytes into a word and for an index below zero. Read from the text, no launch."""
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    signature = {"Z": "*" + dtype, "R": "*" + dtype, "IDX": "*i32", "TRIPS": "i32", "BACK": "i32"}
    source = ASTSource(fn=_add, signature=signature, constexprs={"N": 16, "FORM": 5})
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    module = backend.make_msl(module, metadata, options)
    narrow = "half" if dtype == "fp16" else "bfloat"
    assert f"device {narrow}* eptr_" in module
    assert "reinterpret_cast<size_t>(eptr_" in module
    assert "_eidx_" not in module
