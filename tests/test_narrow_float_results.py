"""A float16 or bfloat16 result is a value of its own type before anything reads it.

The ordinary lowering holds every float in a Metal ``float`` register and narrowed a value
only where it was stored. One operation on loaded values is then the source's value. Two
are not: ``(x + y) - x`` subtracted from the unrounded sum, a float16 sum that overflowed
went on as a float32 number, an integer became bfloat16 by way of float32, a float16
constant was read as the float32 number its decimal names, and a fused multiply-add was
rounded twice. A division by a constant of a value that came from a half was a separate
fault: the Metal compiler narrowed it to a half multiply by the rounded reciprocal.

Each model below rounds where the source's types do: after every operation declared in the
narrow type, and to float32 after every operation the source promotes. Every operation is
exact in float64 or rounded once from it, so the models are the source's arithmetic.

No mask is used and every tensor holds whole blocks: every address is inside its tensor.
"""

from fractions import Fraction

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
from triton_msl.codegen._access_plan import text_order
from triton_msl.codegen._narrow_float import FDIV_HELPER, PDIV_HELPER
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
DTYPES = ["float16", "bfloat16"]
BLOCK = 256
# Significant bits of each type.
BITS = {"float16": 11, "bfloat16": 8}


@triton.jit
def _forms(X, Y, W, I, Z, TRIPS, N: tl.constexpr, FORM: tl.constexpr):
    n = tl.program_id(0) * N + tl.arange(0, N)
    x = tl.load(X + n)
    y = tl.load(Y + n)
    if FORM == 0:
        z = (x + y) - x
    elif FORM == 1:
        z = (x * y) / y
    elif FORM == 2:
        z = (x * 0.1) * 10.0
    elif FORM == 3:  # the source promotes a float16 division by a literal to float32
        z = x / 3.0
    elif FORM == 4:
        z = (x + x) / 3.0
    elif FORM == 5:
        z = tl.div_rn(x.to(tl.float32), tl.full((N,), 3.0, tl.float32))
    elif FORM == 6:
        z = tl.div_rn(x.to(tl.float32), tl.full((N,), 3.0, tl.float32)).to(X.dtype.element_ty) * 3.0
    elif FORM == 7:
        z = tl.div_rn((x + x).to(tl.float32), tl.full((N,), 10.0, tl.float32))
    elif FORM == 8:
        z = tl.where((x + y) > x, x, y)
    elif FORM == 9:  # Z is float32
        z = (x * y).to(tl.float32)
    elif FORM == 10:
        z = tl.load(I + n).to(X.dtype.element_ty)
    elif FORM == 11:  # Z is float32
        z = (tl.load(I + n).to(X.dtype.element_ty) - 2048.0).to(tl.float32)
    elif FORM == 12:
        z = x
        for _ in range(TRIPS):
            z += y
    elif FORM == 13:
        z = tl.math.fma(x, y, tl.load(W + n))
    else:  # 14, Z is float32: the sum of the block, in every cell
        z = tl.sum(x, axis=0).to(tl.float32) + tl.zeros((N,), tl.float32)
    tl.store(Z + n, z)


FORMS = {
    0: "(x + y) - x",
    1: "(x * y) / y",
    2: "(x * 0.1) * 10.0",
    3: "x / 3.0",
    4: "(x + x) / 3.0",
    5: "div_rn(x, 3.0)",
    6: "div_rn(x, 3.0) narrowed, times 3.0",
    7: "div_rn(x + x, 10.0)",
    8: "(x + y) > x",
    9: "x * y widened",
    10: "integer converted",
    11: "integer converted, minus 2048, widened",
    12: "carried by a loop",
    13: "fma",
    14: "sum widened",
}
WIDE = {9, 11, 14}
EVERY_VALUE = {3, 4, 5, 6, 7}


@triton.jit
def _narrow_accumulator(A, B, C, K, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float16)
    for k in range(0, K, BK):
        a = tl.load(A + m[:, None] * SA + (k + kk)[None, :])
        b = tl.load(B + (k + kk)[:, None] * SB + n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.float16)
    tl.store(C + m[:, None] * SC + n[None, :], acc)


def _torch_type(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _narrowed(name, values):
    """Float64 values rounded once, to nearest even, to the type ``name``."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore"):
        if name == "float16":
            return values.astype(np.float16).astype(np.float64)
        if name == "float32":
            return values.astype(np.float32).astype(np.float64)
    mantissa, exponent = np.frexp(values)
    return np.ldexp(np.rint(mantissa * 256.0), exponent - 8)


def _exactly(name, value: Fraction) -> float:
    """A rational rounded once to the type: for an integer or a product float64 does not hold."""
    if value == 0:
        return 0.0
    bits = BITS[name]
    size = abs(value)
    exponent = size.numerator.bit_length() - size.denominator.bit_length()
    if Fraction(2) ** exponent > size:
        exponent -= 1
    unit = Fraction(2) ** (exponent - bits + 1)
    count = size / unit
    whole = count.numerator // count.denominator
    rest = count - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    result = float(whole * unit)
    if name == "float16" and result >= 65520.0:
        result = float("inf")
    return -result if value < 0 else result


def _floats(name, form):
    """The x, y and w of a form, as float64 values the type holds."""
    random = np.random.default_rng(100 + form)
    count = 4 * BLOCK
    if form in EVERY_VALUE and name == "float16":
        patterns = np.arange(1 << 16, dtype=np.uint16)
        x = patterns.view(np.float16).astype(np.float64)
        x = x[np.isfinite(x)]
        x = np.concatenate([x, x[: -len(x) % BLOCK]])
    elif form in EVERY_VALUE:
        x = random.uniform(-900.0, 900.0, 16 * BLOCK)
    elif form in (0, 8):
        # x is a multiple of 8 beside which y is lost or rounded
        x = random.integers(-2000, 2000, count) * 8.0 if name == "float16" else random.integers(-250, 250, count) * 8.0
    elif form == 12:
        x = np.full(count, 2048.0)
    elif form == 13:
        # an odd product one bit too long for the type: it lies midway between two values
        bits = BITS[name]
        low = 1 << (bits // 2)
        first = random.integers(low // 2, low, count) * 2 + 1
        second = ((1 << bits) + random.integers(0, 1 << (bits - 2), count)) // first
        second |= 1
        x, w = first.astype(np.float64), random.choice([1.0, -1.0], count) * 2.0 ** random.integers(-14, -12, count)
        return x, second.astype(np.float64), w
    elif form == 14:
        x = np.full(count, 4000.0)
    else:
        x = random.uniform(-100.0, 100.0, count)
    x = _narrowed(name, x)
    if form in (0, 8):
        y = random.uniform(-1.0, 1.0, len(x))
    elif form == 12:
        y = np.full(len(x), 3.0)
    else:
        y = random.uniform(0.01, 3.0, len(x))
    return x, _narrowed(name, y), np.zeros(len(x))


def _integers(name, kind):
    """Integers one above or below a value midway between two values of the type."""
    bits = BITS[name]
    # A float16 holds no integer of 65520 or more.
    top = 16 if name == "float16" else {"int32": 30, "int64": 62, "uint32": 32}[kind]
    values = [33685505, 2049, 2051, 4097, 100, 0]
    for mantissa in range(1 << (bits - 1), 1 << bits, 7):
        for shift in range(1, top - bits, 2):
            for nudge in (-1, 1):
                values.append((((2 * mantissa + 1) << shift) >> 1) + nudge)
    if kind != "uint32":
        values = [value if place % 2 else -value for place, value in enumerate(values)]
    values = [value for value in values if abs(value) < (1 << top)]
    values += values[: -len(values) % BLOCK]
    return values


def _expected(form, name, x, y, w, integers, trips):
    r = lambda values: _narrowed(name, values)  # noqa: E731
    f = lambda values: _narrowed("float32", values)  # noqa: E731
    with np.errstate(over="ignore", invalid="ignore"):
        if form == 0:
            return r(r(x + y) - x)
        if form == 1:
            return r(r(x * y) / y)
        if form == 2:
            return r(r(x * r(0.1)) * 10.0)
        if form == 3:
            return r(f(x / 3.0))
        if form == 4:
            return r(f(r(x + x) / 3.0))
        if form == 5:
            return r(f(x / 3.0))
        if form == 6:
            return r(r(f(x / 3.0)) * 3.0)
        if form == 7:
            return r(f(r(x + x) / 10.0))
        if form == 8:
            return np.where(r(x + y) > x, x, y)
        if form == 9:
            return r(x * y)
        converted = np.array([_exactly(name, Fraction(value)) for value in integers or ()])
        if form == 10:
            return converted
        if form == 11:
            return r(converted - 2048.0)
        if form == 12:
            for _ in range(trips):
                x = r(x + y)
            return x
        if form == 13:
            return np.array([_exactly(name, Fraction(a) * Fraction(b) + Fraction(c)) for a, b, c in zip(x, y, w)])
        return np.repeat(r(x.reshape(-1, BLOCK).sum(axis=1)), BLOCK)


def _launch(form, name, device, x, y, w, integers, kind="int32", trips=0):
    narrow = _torch_type(name)
    try:
        tensors = [torch.tensor(values, dtype=narrow, device=device) for values in (x, y, w)]
        whole = torch.tensor(np.array(integers, dtype=np.int64), device="cpu").to(getattr(torch, kind)).to(device)
        out = torch.full((len(x),), -7.0, dtype=torch.float32 if form in WIDE else narrow, device=device)
    except (RuntimeError, TypeError) as error:  # a type this torch build has no such tensor for
        pytest.skip(str(error))
    _forms[(len(x) // BLOCK,)](*tensors, whole, out, trips, N=BLOCK, FORM=form)
    if device == "mps":
        torch.mps.synchronize()
    return out.float().cpu().numpy().astype(np.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "form", [form for form in sorted(FORMS) if form not in (10, 11, 13)], ids=lambda form: FORMS[form]
)
@pytest.mark.parametrize("name", DTYPES)
def test_a_narrow_result_is_rounded_to_its_type_before_it_is_read(name, form, device):
    if form == 14 and name == "bfloat16":
        pytest.skip("no bfloat16 sum differs from its float32 one whatever order it is added in")
    x, y, w = _floats(name, form)
    trips = 8 if form == 12 else 0
    got = _launch(form, name, device, x, y, w, [0] * len(x), trips=trips)
    np.testing.assert_array_equal(got, _expected(form, name, x, y, w, None, trips))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", [10, 11], ids=lambda form: FORMS[form])
@pytest.mark.parametrize("kind", ["int32", "int64", "uint32"])
@pytest.mark.parametrize("name", DTYPES)
def test_an_integer_is_converted_to_a_narrow_float_once(name, kind, form, device):
    """33685505 is 33816576 in bfloat16. By way of float32 it is 33554432."""
    integers = _integers(name, kind)
    if form == 11:
        integers = [2049 + place for place in range(BLOCK)]
    zeros = np.zeros(len(integers))
    got = _launch(form, name, device, zeros, zeros, zeros, integers, kind=kind)
    np.testing.assert_array_equal(got, _expected(form, name, zeros, zeros, zeros, integers, 0))


@pytest.mark.parametrize("device", DEVICES)
def test_a_float16_fused_multiply_add_is_rounded_once(device):
    x, y, w = _floats("float16", 13)
    got = _launch(13, "float16", device, x, y, w, [0] * len(x))
    want = _expected(13, "float16", x, y, w, None, 0)
    # The witnesses tell the fused value from the float one rounded again.
    twice = _narrowed("float16", _narrowed("float32", x * y + w))
    assert (twice != want).sum() > len(x) // 4
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("device", DEVICES)
def test_a_bfloat16_fused_multiply_add_is_refused(device):
    x, y, w = _floats("bfloat16", 13)
    with pytest.raises(MetalNonRecoverableError, match="bfloat16 fused multiply-add is not implemented"):
        _launch(13, "bfloat16", device, x, y, w, [0] * len(x))


@pytest.mark.parametrize("device", DEVICES)
def test_a_matmul_that_accumulates_in_float16_is_refused(device):
    """The source rounds its accumulator on every trip of the K loop. The matmul template
    rounds a float accumulator once, after the last trip."""
    size = 64
    a, b = (torch.ones(size, size, dtype=torch.float16, device=device) for _ in range(2))
    c = torch.zeros(size, size, dtype=torch.float16, device=device)
    with pytest.raises(MetalNonRecoverableError, match="accumulator carried over the K loop is rounded to f16"):
        _narrow_accumulator[(2, 2)](a, b, c, size, size, size, size, BM=32, BN=32, BK=32)


def _text(form, narrow):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    wide = "fp32" if form in WIDE else narrow
    signature = {"X": "*" + narrow, "Y": "*" + narrow, "W": "*" + narrow, "I": "*i32", "Z": "*" + wide, "TRIPS": "i32"}
    source = ASTSource(fn=_forms, signature=signature, constexprs={"N": BLOCK, "FORM": form})
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options)


@pytest.mark.parametrize("narrow,metal", [("fp16", "half"), ("bf16", "bfloat")])
def test_the_text_rounds_each_narrow_result_where_it_is_produced(narrow, metal):
    """Read from the text, no launch."""
    rounded = f"static_cast<float>(static_cast<{metal}>("
    assert _text(0, narrow).count(rounded) >= 2
    assert _text(12, narrow).count(rounded) >= 1
    assert f"static_cast<{metal}>(" in _text(10, narrow)
    assert "tm_fdiv(" in _text(3, narrow) and FDIV_HELPER in _text(3, narrow)
    assert "tm_pdiv(" in _text(5, narrow) and PDIV_HELPER in _text(5, narrow)
    assert "tm_fdiv(" not in _text(0, narrow)


def test_the_text_of_a_float16_fused_multiply_add_is_the_half_one():
    assert "fma(static_cast<half>(" in _text(13, "fp16")


def test_the_guarded_quotients_hide_no_barrier_from_the_text_proof():
    """The order proof gives up every barrier of a shader that has a ``return``: a thread
    that returned early would not reach one. The two quotient helpers return a float and
    are no way out of the kernel."""
    kernel = (
        "kernel void k(volatile device float* A [[buffer(0)]]) {\n"
        "    float a = A[0];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    A[0] = a;\n"
        "}\n"
    )
    assert len(text_order(["A"], kernel)["barriers"]) == 1
    for helper in (FDIV_HELPER, PDIV_HELPER):
        assert len(text_order(["A"], helper + "\n" + kernel)["barriers"]) == 1
    other = "static inline float other(float a) { if (a > 0.0f) return a; return 0.0f; }\n"
    assert text_order(["A"], other + kernel)["barriers"] == []
