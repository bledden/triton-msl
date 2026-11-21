"""A float16 or bfloat16 value is a value of its own type.

The ordinary lowering holds every float in a Metal ``float`` register, whatever the
source's type, and a store narrows it. That is the source's value after one operation
on loaded values and not after two: ``(x + y) - x`` in float16 subtracts from the
rounded sum, and a ``float`` register still holds the unrounded one. The interpreter
and LLVM's types round after every operation.

So each operation whose declared result is ``f16`` or ``bf16``, and whose float32
result need not be a value of that type, is rounded through the narrow type where it
is produced. The register stays ``float``: every later reader sees a value the narrow
type holds. An operation declared ``f32`` is left alone, so promotion in the source and
whatever the Metal compiler may do inside one float32 expression are unchanged.

What is rounded is each RESULT. A reduction, a scan and a dot still accumulate in
float32 inside and their result is converted: a float16 sum that overflows is infinity
to its reader, not a float32 number.
"""

import re
import struct

from triton_msl.errors import MetalNonRecoverableError

NARROW = {"f16": "half", "bf16": "bfloat"}

# Operations whose float32 result is not always a value of the narrow type.
# Not here, because a value of the type goes in and comes out: load, select, negate,
# abs, copysign, the minima and maxima, clamp, floor, ceil, trunc, round, and the
# wrappers (splat, broadcast, reshape, the loop-carried arguments).
_ARITH = frozenset(
    {
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.remf",
        "arith.sitofp",
        "arith.uitofp",
        "tt.precise_sqrt",
        "tt.precise_divf",
        "tt.extern_elementwise",
        "tt.dot",
    }
)
# math.fma is here because its own lowering computes it in the narrow type.
_EXACT_MATH = frozenset(
    {"math.absf", "math.copysign", "math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven", "math.fma"}
)
# A conversion to the narrow type that the lowering may have passed through as an alias.
_CASTS = frozenset({"arith.truncf", "tt.fp_to_fp"})
_REGIONS = frozenset({"tt.reduce", "tt.scan"})

_LVALUE = re.compile(r"[A-Za-z_]\w*(\[[^\[\]]+\])?")

# The quotient of a float division by a constant, as IEEE gives it. Without the volatile
# the Metal compiler narrows ``half(float(x) / 3.0f)`` to a half multiply by the rounded
# reciprocal: 19,836 of the 63,488 finite float16 values of x then differ from x / 3.
FDIV_HELPER = "static inline float tm_fdiv(float a, float b) { volatile float q = a / b; return q; }"
# The same for the division the source asks to be rounded to nearest (tl.div_rn).
PDIV_HELPER = "static inline float tm_pdiv(float a, float b) { volatile float q = precise::divide(a, b); return q; }"


def without_helpers(msl: str, keep_positions: bool = False) -> str:
    """The shader text without the guarded quotients, for the proofs that read it.

    Each takes two floats and returns one: no thread leaves the kernel in it and no
    argument buffer is named in it.
    """
    for helper in (FDIV_HELPER, PDIV_HELPER):
        msl = msl.replace(helper, " " * len(helper) if keep_positions else "")
    return msl


def narrow_constant(value: float, elem_type: str) -> float:
    """The value a decimal constant of a narrow type names.

    MLIR prints a float16 or bfloat16 constant with as many digits as tell it from its
    neighbours in its own type (``9.997550e-02`` for float16 0.1). Read as a float32
    literal that decimal is another number. This returns the narrow value itself.
    """
    if elem_type == "f16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _walk(ops):
    for op in ops or ():
        yield op
        yield from _walk(op.region_ops)
        yield from _walk(op.else_ops)


def holds_narrow(lowerer) -> bool:
    """Whether the kernel has a 16-bit float anywhere: an argument or a value."""
    cached = getattr(lowerer, "_holds_narrow_float", None)
    if cached is None:
        graph = lowerer.graph
        cached = any(arg.elem_type in NARROW for arg in graph.args) or any(
            op.elem_type in NARROW for op in _walk(graph.ops)
        )
        lowerer._holds_narrow_float = cached
    return cached


def fdiv(lowerer, a: str, b: str, exact: bool = False) -> str:
    """``a / b`` in a kernel that holds 16-bit floats: the guarded quotient."""
    helper = PDIV_HELPER if exact else FDIV_HELPER
    if helper not in lowerer.kb._device_functions:
        lowerer.kb._device_functions.append(helper)
    return f"tm_pdiv({a}, {b})" if exact else f"tm_fdiv({a}, {b})"


def int_to_narrow(elem_type: str, expr: str) -> str:
    """An integer converted to the narrow type ``elem_type``, as a value of that type.

    To bfloat16 the integer is converted directly. Through ``float`` it is rounded
    twice: 33685505 is 33816576 in bfloat16, and 33554432 by way of float32.

    To float16 it goes through ``float``. Metal's own conversion of a ``long`` to
    ``half`` gives infinity for 65505 to 65519, which round to 65504, the largest
    float16. ``float`` holds every integer below 65520 exactly, so the one rounding is
    the one to float16, and it takes a larger integer to a value that is infinity in
    float16 as the integer is.
    """
    if elem_type == "f16":
        return f"static_cast<half>(static_cast<float>({expr}))"
    return f"static_cast<{NARROW[elem_type]}>({expr})"


def int_to_float(elem_type: str, expr: str) -> str:
    """An integer converted to a float of ``elem_type``, held as ``float``."""
    if elem_type not in NARROW:
        return f"static_cast<float>({expr})"
    return f"static_cast<float>({int_to_narrow(elem_type, expr)})"


def fused_multiply_add(ssa, a: str, b: str, c: str) -> str:
    """``fma(a, b, c)`` for a result of ``ssa``'s type, held as ``float``.

    A float16 one is Metal's own half fma: the exact a * b + c rounded once. The float
    fma of the same operands rounded to half afterwards is rounded twice. Metal has no
    bfloat fma (it computes in float), so a bfloat16 one is refused.
    """
    if ssa.elem_type == "f16":
        return f"static_cast<float>(fma(static_cast<half>({a}), static_cast<half>({b}), static_cast<half>({c})))"
    if ssa.elem_type == "bf16":
        raise MetalNonRecoverableError(
            "math.fma: a bfloat16 fused multiply-add is not implemented. Metal computes it in float, and that "
            "result rounded to bfloat16 is rounded twice. Write `a * b + c` for the unfused value. Refused.",
            op_name=ssa.op,
        )
    return f"fma({a}, {b}, {c})"


def _rounds(ssa) -> bool:
    op = ssa.op
    if op in _ARITH or op in _CASTS:
        return True
    if op.startswith("math."):
        return op not in _EXACT_MATH
    if op in _REGIONS:
        return any(inner.op in _ARITH or inner.op.startswith("math.") for inner in _walk(ssa.region_ops))
    return False


def _cast(narrow: str, expr: str) -> str:
    return f"static_cast<float>(static_cast<{narrow}>({expr}))"


def round_result(lowerer, ssa):
    """Round the result ``ssa`` just produced to its declared narrow type, in place."""
    narrow = NARROW.get(ssa.elem_type)
    if narrow is None or not _rounds(ssa):
        return
    if ssa.result_ids and len(ssa.result_ids) > 1:
        raise MetalNonRecoverableError(
            f"{ssa.op} with {len(ssa.result_ids)} results computes a {ssa.elem_type} value by arithmetic: "
            "the conversion of each result to its own type is not implemented, so the kernel is refused.",
            op_name=ssa.op,
        )
    kb = lowerer.kb
    name = lowerer.env.get(ssa.id)
    array = lowerer.env_array.get(ssa.id)
    shared = getattr(lowerer, "_shared_mem_descs", {}).get(ssa.id)
    sources = [lowerer.env.get(operand) for operand in ssa.operand_ids or ()]
    aliased = name is not None and name in sources
    if ssa.op in _CASTS and not aliased:
        # The conversion's own lowering produced a narrow value.
        return
    if array is not None:
        var, count, ty = array
        if ty in ("half", "bfloat"):
            return
        if ty != "float":
            raise MetalNonRecoverableError(
                f"{ssa.op}: a {ssa.elem_type} result held as `{ty}` cannot be rounded to its type.", op_name=ssa.op
            )
        if aliased:
            fresh = lowerer._var_array("nr", [_cast(narrow, f"{var}[{i}]") for i in range(count)], "float")
            lowerer.env[ssa.id] = fresh
            lowerer.env_array[ssa.id] = (fresh, count, "float")
            return
        for i in range(count):
            kb.raw_line(f"    {var}[{i}] = {_cast(narrow, f'{var}[{i}]')};")
        return
    if shared is not None:
        shared_name, shape, _ = shared
        if aliased and ssa.op in _CASTS:
            raise MetalNonRecoverableError(
                f"{ssa.op}: a conversion to {ssa.elem_type} of a value held in threadgroup memory was passed "
                "through unrounded, and rounding the shared cells would change its other readers. Refused.",
                op_name=ssa.op,
            )
        total = 1
        for dim in shape:
            total *= dim
        step = lowerer.effective_block_size
        kb.raw_line(f"    for (uint _nr = lid; _nr < {total}u; _nr += {step}u) {{")
        kb.raw_line(f"        {shared_name}[_nr] = {_cast(narrow, f'{shared_name}[_nr]')};")
        kb.raw_line("    }")
        kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        if isinstance(name, str) and name != shared_name and _LVALUE.fullmatch(name):
            kb.raw_line(f"    {name} = {_cast(narrow, name)};")
        return
    if not isinstance(name, str):
        raise MetalNonRecoverableError(
            f"{ssa.op}: the {ssa.elem_type} result has no register to round to its type. Refused.", op_name=ssa.op
        )
    if aliased or not _LVALUE.fullmatch(name):
        # A passed-through operand or an expression: its rounded value gets its own register.
        fresh = lowerer._next_var("nr")
        kb.raw_line(f"    float {fresh} = {_cast(narrow, name)};")
        lowerer.env[ssa.id] = fresh
        return
    kb.raw_line(f"    {name} = {_cast(narrow, name)};")
