"""An integer next to the largest float16 becomes the largest float16, not infinity.

65504 is the largest float16. 65505 to 65519 round to it, and 65520 is the first integer
that rounds to infinity. Metal's own conversion of a ``long`` to ``half`` gives infinity
from 65505 on. A narrow result is rounded to its type where it is produced, and for an
int64 that was this conversion: an int64 of 65505 to 65519, loaded, summed or computed,
was stored as infinity where the source stores 65504. Nothing was refused and nothing was
reported. An integer now becomes a float16 by way of ``float``, which holds every integer
below 65520 exactly, so the one rounding is the one to float16.

To bfloat16 an integer is converted directly, as before: by way of ``float`` it would be
rounded twice. Other integer types, and int64 to bfloat16 and float32, are controls here.

No mask is used and every tensor holds whole blocks: every address is inside its tensor.
"""

import re

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
ROUTES = [
    "loaded",
    "sum of a block",
    "sum of each row",
    "running sum",
    "largest of a block",
    "chosen by a condition",
    "carried round a loop",
    "times a float",
    "lane index plus a scalar",
    "outer sum",
    "int32 widened",
    "sum plus a load",
    "largest of each row",
    "loaded by a lane loop",
    "sum of each column",
    "under a branch on a scalar",
]
N, GRID, R, C, LOOPED = 64, 4, 8, 16, 2048
# name: (Triton's type, torch's type, significant bits, the first magnitude that is infinity)
FLOATS = {
    "float16": (tl.float16, torch.float16, 11, 1 << 16),
    "bfloat16": (tl.bfloat16, torch.bfloat16, 8, 0),
    "float32": (tl.float32, torch.float32, 24, 0),
}
# int64 to float16 is the conversion that was wrong.
PAIRS = [
    ("int64", "float16"),
    ("int32", "float16"),
    ("uint32", "float16"),
    ("uint64", "float16"),
    ("int64", "bfloat16"),
    ("int64", "float32"),
]


@triton.jit
def _routes(X, T, H, Z, S, ROUTE: tl.constexpr, OUT: tl.constexpr, N: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    if ROUTE == 0:
        tl.store(Z + pid * N + n, tl.load(X + pid * N + n).to(OUT))
    elif ROUTE == 1:
        tl.store(Z + pid, tl.sum(tl.load(X + pid * N + n), axis=0).to(OUT))
    elif ROUTE == 2:
        r = tl.arange(0, R)
        x = tl.load(X + pid * R * C + r[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(Z + pid * R + r, tl.sum(x, axis=1).to(OUT))
    elif ROUTE == 3:
        tl.store(Z + pid * N + n, tl.cumsum(tl.load(X + pid * N + n), axis=0).to(OUT))
    elif ROUTE == 4:
        tl.store(Z + pid, tl.max(tl.load(X + pid * N + n), axis=0).to(OUT))
    elif ROUTE == 5:
        x = tl.load(X + pid * N + n)
        tl.store(Z + pid * N + n, tl.where(x > 9, x, x + 1).to(OUT))
    elif ROUTE == 6:
        acc = tl.load(X + pid * N + n)
        t = tl.load(T + pid * N + n)
        for _ in range(S):
            acc = acc + t
        tl.store(Z + pid * N + n, acc.to(OUT))
    elif ROUTE == 7:  # the source converts the integer to the float's type before it multiplies
        tl.store(Z + pid * N + n, tl.load(X + pid * N + n) * tl.load(H + pid * N + n))
    elif ROUTE == 8:
        tl.store(Z + pid * N + n, (n.to(X.dtype.element_ty) + S).to(OUT))
    elif ROUTE == 9:
        r = tl.arange(0, R)
        c = tl.arange(0, C)
        x = tl.load(X + r)[:, None] + tl.load(T + c)[None, :]
        tl.store(Z + r[:, None] * C + c[None, :], x.to(OUT))
    elif ROUTE == 10:
        tl.store(Z + pid * N + n, (n + S).to(X.dtype.element_ty).to(OUT))
    elif ROUTE == 11:
        x = tl.load(X + pid * N + n)
        tl.store(Z + pid * N + n, (tl.sum(x, axis=0) + tl.load(T + pid * N + n)).to(OUT))
    elif ROUTE == 12:
        r = tl.arange(0, R)
        x = tl.load(X + pid * R * C + r[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(Z + pid * R + r, tl.max(x, axis=1).to(OUT))
    elif ROUTE == 13:
        w = tl.arange(0, 2048)
        tl.store(Z + w, tl.load(X + w).to(OUT))
    elif ROUTE == 14:
        c = tl.arange(0, C)
        x = tl.load(X + pid * R * C + tl.arange(0, R)[:, None] * C + c[None, :])
        tl.store(Z + pid * C + c, tl.sum(x, axis=0).to(OUT))
    else:
        x = tl.load(X + pid * N + n)
        if S > 3:
            x = x + S
        tl.store(Z + pid * N + n, x.to(OUT))


@triton.jit
def _flat(X, Z, N: tl.constexpr):
    n = tl.program_id(0) * N + tl.arange(0, N)
    tl.store(Z + n, tl.load(X + n).to(Z.dtype.element_ty))


def _rounded(value, name):
    """A Python integer rounded once, to nearest even, to the float type ``name``."""
    _, _, bits, infinite = FLOATS[name]
    size = abs(int(value))
    if size.bit_length() > bits:
        shift = size.bit_length() - bits
        kept, rest = size >> shift, size & ((1 << shift) - 1)
        if rest > (1 << (shift - 1)) or (rest == (1 << (shift - 1)) and kept & 1):
            kept += 1
        size = kept << shift
    result = float("inf") if infinite and size >= infinite else float(size)
    return -result if value < 0 else result


# The largest float16 and what is next to it, with values of other sizes between them.
NEAR = [65505, 65519, 65512, -65505, -65519, 65504, 65520, 65503, -65520, 13, 65536, 70000, -65512, 65510, 2049, 2051]
NEAR += [65518, -65506, 10, -11, 65521, 65511, 2047, -65536]
TARGETS = [65505, 65519, -65512, 65520, 65504, -65519, 65510, 65536]


def _case(route, kind):
    """The inputs of one route, its scalar and grid, and the integers the source converts."""
    name = ROUTES[route]
    signed = np.dtype(kind).kind == "i"
    fit = (lambda value: value) if signed else abs
    low = -70000 if signed else 0

    def near(count):
        return np.array([fit(NEAR[place % len(NEAR)]) for place in range(count)], dtype=kind)

    x, t, s, grid = near(GRID * N), np.ones(GRID * N, dtype=kind), 0, GRID
    if name == "loaded":
        want = list(x)
    elif name == "loaded by a lane loop":
        x, grid = near(LOOPED), 1
        want = list(x)
    elif name in ("sum of a block", "largest of a block"):
        summed = name == "sum of a block"
        x = np.full(GRID * N, 1 if summed else low, dtype=kind)
        for program in range(GRID):
            x[program * N + 5] = fit(TARGETS[program]) - (N - 1 if summed else 0)
        blocks = x.reshape(GRID, N).astype(object)
        want = list(blocks.sum(axis=1) if summed else blocks.max(axis=1))
    elif name in ("sum of each row", "largest of each row", "sum of each column"):
        x = np.full(GRID * R * C, low if name == "largest of each row" else 1, dtype=kind)
        tiles = x.reshape(GRID, R, C)
        for program in range(GRID):
            for place in range(C if name == "sum of each column" else R):
                target = fit(TARGETS[(program + place) % len(TARGETS)])
                if name == "sum of each row":
                    tiles[program, place, 3] = target - (C - 1)
                elif name == "largest of each row":
                    tiles[program, place, 3] = target
                else:
                    tiles[program, 3, place] = target - (R - 1)
        exact = tiles.astype(object)
        if name == "sum of each row":
            want = list(exact.sum(axis=2).ravel())
        elif name == "largest of each row":
            want = list(exact.max(axis=2).ravel())
        else:
            want = list(exact.sum(axis=1).ravel())
    elif name == "running sum":
        x = np.zeros(GRID * N, dtype=kind)
        x[::N] = 65500
        x[1::2] = 1
        want = list(np.cumsum(x.reshape(GRID, N).astype(object), axis=1).ravel())
    elif name == "chosen by a condition":
        want = [int(v) if v > 9 else int(v) + 1 for v in x]
    elif name == "carried round a loop":
        s = 3
        x = x - np.array(3, dtype=kind)
        want = [int(v) + 3 for v in x]
    elif name == "times a float":
        want = list(x)
    elif name in ("lane index plus a scalar", "int32 widened"):
        s = 65490
        want = [place + s for _ in range(GRID) for place in range(N)]
    elif name == "outer sum":
        x = np.array([fit(v) for v in (65500, -65500, 0, 65490, 3, -65530, 65504, 100)], dtype=kind)
        t, grid = np.arange(C).astype(kind) + np.array(4, dtype=kind), 1
        want = [int(a) + int(b) for a in x for b in t]
    elif name == "sum plus a load":
        x = np.ones(GRID * N, dtype=kind)
        t = (np.arange(GRID * N) % 40 + 65500 - N - 5).astype(kind)
        want = [N + int(v) for v in t]
    else:
        s = 5
        x = x - np.array(5, dtype=kind)
        want = [int(v) + 5 for v in x]
    return x, t, s, grid, [int(v) for v in want]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", range(len(ROUTES)), ids=ROUTES)
@pytest.mark.parametrize("kind,name", PAIRS, ids=[f"{kind} to {name}" for kind, name in PAIRS])
def test_an_integer_becomes_the_float_nearest_to_it_on_every_route(kind, name, route, device):
    x, t, s, grid, integers = _case(route, kind)
    triton_type, torch_type, _, _ = FLOATS[name]
    want = np.array([_rounded(value, name) for value in integers], dtype=np.float64)
    if name == "float16":
        # The case has values that were wrong, and values that are infinity in the source.
        assert any(65504 < abs(value) < 65520 for value in integers)
        assert np.isinf(want).any() or ROUTES[route] in ("sum plus a load",)
    try:
        tensors = [torch.from_numpy(values).to(device) for values in (x, t)]
        ones = torch.ones(max(x.size, 8), dtype=torch_type, device=device)
        out = torch.full((len(integers) + 8,), 7.0, dtype=torch_type, device=device)
    except (RuntimeError, TypeError) as error:  # a type this torch build has no such tensor for
        pytest.skip(str(error))
    _routes[(grid,)](*tensors, ones, out, s, ROUTE=route, OUT=triton_type, N=N, R=R, C=C)
    if device == "mps":
        torch.mps.synchronize()
    got = out.float().cpu().numpy().astype(np.float64)
    np.testing.assert_array_equal(got[: len(integers)], want)
    np.testing.assert_array_equal(got[len(integers) :], 7.0)


def _every_integer_round_a_tie(kind):
    """Every integer of magnitude up to 70000, and those round each tie of each float type."""
    info = np.iinfo(kind)
    values = set(range(max(info.min, -70000), min(info.max, 70000) + 1))
    for bits in (8, 11, 24):
        for exponent in range(bits, 64):
            unit = 1 << (exponent - bits + 1)
            half = unit >> 1
            for kept in ((1 << (bits - 1)), (1 << (bits - 1)) + 1, (1 << bits) - 2, (1 << bits) - 1):
                for rest in (0, half - 1, half, half + 1, unit - 1):
                    values.update((kept * unit + rest, -(kept * unit + rest)))
    values |= {info.max, info.min, info.max - 1, info.min + 1}
    return sorted(value for value in values if info.min <= value <= info.max)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(FLOATS))
@pytest.mark.parametrize("kind", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
def test_every_integer_round_a_tie_is_rounded_once(kind, name, device):
    block = 1024
    values = _every_integer_round_a_tie(np.dtype(kind).type)
    x = np.zeros(-(-len(values) // block) * block, dtype=kind)
    x[: len(values)] = np.array(values, dtype=kind)
    want = np.array([_rounded(value, name) for value in x.astype(object)], dtype=np.float64)
    try:
        out = torch.full((x.size,), 7.0, dtype=FLOATS[name][1], device=device)
        source = torch.from_numpy(x).to(device)
    except (RuntimeError, TypeError) as error:  # a type this torch build has no such tensor for
        pytest.skip(str(error))
    _flat[(x.size // block,)](source, out, N=block)
    if device == "mps":
        torch.mps.synchronize()
    np.testing.assert_array_equal(out.float().cpu().numpy().astype(np.float64), want)


def _text(kind, narrow):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(fn=_flat, signature={"X": "*" + kind, "Z": "*" + narrow}, constexprs={"N": 64})
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options)


@pytest.mark.parametrize("kind", ["i32", "i64", "u32", "u64"])
def test_the_text_converts_an_integer_to_float16_by_way_of_float_and_to_bfloat16_directly(kind):
    """Read from the text, no launch."""
    half, brain = _text(kind, "fp16"), _text(kind, "bf16")
    # ``val_`` names the loaded integer; an unsigned one is first read as unsigned.
    loaded = r"(static_cast<u\w+>\()?val_"
    assert re.search(r"static_cast<half>\(static_cast<float>\(" + loaded, half)
    assert not re.search(r"static_cast<half>\(" + loaded, half)
    assert re.search(r"static_cast<bfloat>\(" + loaded, brain)
    assert "static_cast<bfloat>(static_cast<float>(" not in brain
