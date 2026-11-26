"""A float constant reaches the shader as the value it has in its own type.

MLIR prints a float constant with as many digits as tell it from its neighbours IN ITS OWN TYPE
(``9.997550e-02`` for float16 0.1, ``1.000980e-01`` for bfloat16 0.1), and prints one that is not
finite as its bit pattern (``0x7C00``). The walker handed both on as they were printed: a
double that is not the constant, and an integer. The ordinary lowering decoded them; a called
(noinline) function, the constant dot template and several detectors did not. In a called
function a float16 0.1 was stored as 0.099975586 rounded again and a float16 infinity as
31744.0.

The walker now gives every float constant as a Python float holding exactly the value of the
constant's type, so no reader has a decimal or a bit pattern to decode. These tests read that
from the walker for every type, then store constants through the two lowerings that print one
and compare the bits on the GPU. The stores write 16 elements of a view 64 elements into a
storage of 256 that the test owns.
"""

import math
import struct

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.mlir_walker import MLIRWalker

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")

INF = float("inf")
NAN = float("nan")
TORCH = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ELEM = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}
# Not dyadic, the smallest normal, the largest finite, and the values MLIR prints as bits. Two
# constants are absent because Triton's own IR does not hold them: ``tl.full(shape, -0.0)`` is a
# constant +0.0 in the TTGIR, and a bfloat16 constant below the float16 range (1e-7, 1e-30) is a constant 0.0.
VALUES = {
    "fp32": [0.1, 1.0 / 3.0, 1e-4, 0.3001, 1.17549435e-38, 3.4028234663852886e38, INF, -INF, NAN],
    "fp16": [0.1, 1.0 / 3.0, 1e-4, 0.3001, 6.103515625e-05, 65504.0, INF, -INF, NAN],
    "bf16": [0.1, 1.0 / 3.0, 1e-4, 0.3001, 1e-6, 3.3895313892515355e38, INF, -INF, NAN],
}
# Subnormal constants: held exactly by the walker; the GPU's handling of them is not this test's subject.
SUBNORMAL = {"fp32": [1e-45, 1e-40], "fp16": [6e-8, 3e-5], "bf16": []}


def _in_type(value, kind):
    """The float32 holding ``value`` rounded once to ``kind`` (float16 and bfloat16 values are float32 values)."""
    if kind == "fp32":
        return np.float32(value)
    if kind == "fp16":
        return np.float32(np.float16(value))
    if math.isnan(value) or math.isinf(value):
        return np.float32(value)
    # bfloat16 from the double, in one rounding: the two nearest bfloat16 values around it.
    bits = struct.unpack("<I", struct.pack("<f", np.float32(value)))[0] & 0xFFFF0000
    low = struct.unpack("<f", struct.pack("<I", bits))[0]
    high = struct.unpack("<f", struct.pack("<I", bits + 0x10000))[0]
    if abs(low) > abs(value):  # float32 rounding went past the value: step back
        low, high = struct.unpack("<f", struct.pack("<I", bits - 0x10000))[0], low
        bits -= 0x10000
    below, above = abs(value - low), abs(high - value)
    if below < above or (below == above and not (bits >> 16) & 1):
        return np.float32(low)
    return np.float32(high)


def _same(got, expected):
    got, expected = np.float32(got), np.float32(expected)
    return (np.isnan(got) and np.isnan(expected)) or got.tobytes() == expected.tobytes()


@triton.jit
def _fill(X, V: tl.constexpr, DT: tl.constexpr):
    i = tl.arange(0, 16)
    tl.store(X + i, tl.full((16,), V, DT))


@triton.jit
def _choose(X, V: tl.constexpr, DT: tl.constexpr):
    # The constant as an operand of an operation that changes no bit of it.
    i = tl.arange(0, 16)
    x = tl.load(X + i)
    tl.store(X + i, tl.where(i < 8, tl.full((16,), V, DT), x))


def _graph(kernel, kind, constants):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    signature = {name: ("constexpr" if name in constants else "*" + kind) for name in kernel.arg_names}
    context = ir.context()
    ir.load_dialects(context)
    module = ASTSource(kernel, signature, constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttgir(backend.make_ttir(module, {}, options), {}, options)
    return MLIRWalker(module, options).walk()


def _float_constants(graph):
    def walk(ops):
        for op in ops or ():
            yield op
            yield from walk(op.region_ops)
            yield from walk(op.else_ops)

    return [op for op in walk(graph.ops) if op.op == "arith.constant" and op.elem_type in ("f16", "bf16", "f32")]


CASES = [(kind, value) for kind in VALUES for value in VALUES[kind] + SUBNORMAL[kind]]


@pytest.mark.parametrize("kind,value", CASES, ids=lambda item: repr(item))
def test_the_walker_gives_a_float_constant_as_its_own_value(kind, value):
    constants = _float_constants(
        _graph(_fill, kind, dict(V=value, DT=getattr(tl, TORCH[kind].__str__().split(".")[1])))
    )
    assert constants and all(op.elem_type == ELEM[kind] for op in constants)
    for op in constants:
        held = op.attrs["value"]
        assert type(held) is float, f"{held!r} is not a number of the constant's type"
        assert _same(held, _in_type(value, kind)), f"{held!r} is not {kind} {value!r}"
        # float32 keeps the decimal MLIR printed, which names the float32; the narrow types are exact.
        assert kind == "fp32" or math.isnan(held) or float(np.float32(held)) == held


@requires_gpu
@pytest.mark.parametrize("kernel", [_fill, _choose], ids=["store", "select"])
@pytest.mark.parametrize(
    "kind,value", [(kind, value) for kind in VALUES for value in VALUES[kind]], ids=lambda item: repr(item)
)
def test_a_constant_is_stored_with_the_bits_of_its_type(kind, value, kernel):
    dtype = TORCH[kind]
    storage = torch.full((256,), 7.0, dtype=dtype, device="mps")
    kernel[(1,)](storage[64:], V=value, DT=getattr(tl, str(dtype).split(".")[1]))
    torch.mps.synchronize()
    got = storage.float().cpu().numpy()
    stored = 16 if kernel is _fill else 8
    expected = _in_type(value, kind)
    assert all(_same(cell, expected) for cell in got[64 : 64 + stored]), f"{got[64]!r} stored for {kind} {value!r}"
    assert (got[:64] == 7.0).all() and (got[64 + stored :] == 7.0).all()


# ---- a called function has its own lowering of constants -----------------------------------------


PLUS_INFINITY = tl.constexpr(INF)
MINUS_INFINITY = tl.constexpr(-INF)


@triton.jit(noinline=True)
def _put(X, k):
    tl.store(X + k, 0.1)
    tl.store(X + k + 1, 0.3001)
    tl.store(X + k + 2, PLUS_INFINITY)
    tl.store(X + k + 3, MINUS_INFINITY)
    tl.store(X + k + 4, 65504.0)
    tl.store(X + k + 5, 1e-4)


@triton.jit
def _calls(X):
    _put(X, tl.program_id(0) * 8)


PUT = [0.1, 0.3001, INF, -INF, 65504.0, 1e-4]


@requires_gpu
def test_a_called_function_stores_a_constant_with_the_bits_of_its_type():
    """float32 only: a called function that stores a float16 or bfloat16 constant is refused at compile."""
    kind = "fp32"
    dtype = TORCH[kind]
    storage = torch.full((256,), 7.0, dtype=dtype, device="mps")
    _calls[(1,)](storage[64:])
    torch.mps.synchronize()
    got = storage.float().cpu().numpy()
    for at, value in enumerate(PUT):
        assert _same(got[64 + at], _in_type(value, kind)), f"{got[64 + at]!r} stored for {kind} {value!r}"
    assert (got[:64] == 7.0).all() and (got[70:] == 7.0).all()
