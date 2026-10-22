"""Optimized-stage semantic checks for the C++ masked-load lowering (CPU only).

The structural pins in ``test_cpp_memory_adversarial.py`` describe the
pre-optimization lowering shape (one guarded block per load, loads on the true
edge). An optimizer may legally merge blocks, complement predicates, compose
branch conditions with ``select``, forward stores and share reads. This file
checks the property those pins exist for, on any lowering of the same corpus:

  on every control path, a memory read executes only where the source performs
  that read (pointer and memory version), volatile reads are exactly preserved,
  and every store writes the source's value in the source's order.

Method: one small concrete-control / symbolic-memory evaluator.  Scalar inputs
(lane id, extent, flag) are enumerated over a grid that covers every predicate
outcome of the corpus, so control flow is concrete and needs no path solver;
memory cells hold symbolic values, so no numeric oracle is invented.  The same
symbolic memory model runs a hand-written per-lane reference of each corpus
kernel.  Undefined source values compare as "unspecified": an optimized store of
an undefined source value may store anything or be dropped, never required.

Bounded domain (stated, not universal equivalence):
  * kernels of the adversarial corpus (f32 elements, splat pointers, scalar
    ``extent``/``flag`` arguments, masks over the lane id) plus the lane-indexed
    cases below (f32 and i32 elements, i32/i64 offsets, strides, constant
    offsets), which exercise the GEP forms an optimizer rewrites;
  * the LLVM subset listed in ``_SUPPORTED``; any other instruction, an
    unrecognised operand spelling or a runaway loop reports UNSUPPORTED and the
    test FAILS with that message, it never passes silently;
  * integers carry their LLVM width: values are kept as unsigned bit patterns
    of the instruction's width, arithmetic wraps modulo 2**width, ``trunc``
    keeps the low bits, ``zext``/``sext`` extend by zero/sign, signed and
    unsigned comparisons read the pattern accordingly, and an ``nsw``/``nuw``
    overflow or a non-disjoint ``or disjoint`` (LLVM poison) is UNSUPPORTED;
  * distinct kernel arguments are distinct memory objects (no aliasing between
    ``in`` and ``out``), lanes are evaluated independently.

Expected per helper (fresh process each, see run_cpp_semantics.py):
  accepted masked+volatile helper      all corpus cases PASS
  full native pipeline (with InstCombine)  all PASS at native LLVM (its Metal
                                         parser failure is not this file's scope)
  removal-only pipeline                all PASS
  original unsafe helper (cpp-build)   corpus cases with a dynamic mask FAIL
                                         (eager read on the false mask) -- the
                                         negative control
"""

import re
import struct

import pytest
import triton  # noqa: F401  (backend discovery before triton_msl imports)

cpp = pytest.importorskip("triton_msl._triton_msl_cpp")

from test_cpp_memory_adversarial import (  # noqa: E402  corpus is single-sourced
    CACHE_MODIFIED_LOAD,
    CONJOINED_MASK,
    DEPENDENT_OTHER,
    LOAD_THEN_STORE_DIFFERENT_MASKS,
    NEGATED_MASK,
    NESTED_IF,
    NESTED_IF_NO_OTHER,
    RUNTIME_LOOP,
    SAME_PTR_SAME_MASK_DIFFERENT_OTHER,
    SAME_PTR_UNEQUAL_MASKS,
    STORE_BETWEEN_SAME_PTR,
    TWO_PTRS_SAME_MASK,
    VOLATILE_LOAD,
    ZERO_TRIP_CONST_LOOP,
    function_body,
)

# --------------------------------------------------------------------------- values and memory


class Unsupported(Exception):
    """IR outside the proved subset. Tests turn this into a loud failure."""


UNDEF = ("undef",)


def fadd(a, b):
    if a is UNDEF or b is UNDEF:
        return UNDEF
    if isinstance(a, float) and isinstance(b, float):
        return a + b
    return ("fadd",) + tuple(sorted((a, b), key=repr))  # commutative, never reassociated


def fmul(a, b):
    if a is UNDEF or b is UNDEF:
        return UNDEF
    if isinstance(a, float) and isinstance(b, float):
        return a * b
    return ("fmul",) + tuple(sorted((a, b), key=repr))


def fsub(a, b):
    if a is UNDEF or b is UNDEF:
        return UNDEF
    if isinstance(a, float) and isinstance(b, float):
        return a - b
    return ("fsub", a, b)


class Memory:
    """Cells keyed by (object, byte offset). Every read/write is an event."""

    def __init__(self):
        self.cells = {}  # cell -> (value, version)
        self.events = []

    def load(self, cell, volatile=False):
        value, version = self.cells.get(cell, (("M", cell, 0), 0))
        self.events.append(("load", cell, version, volatile))
        return value

    def store(self, cell, value):
        _, version = self.cells.get(cell, (None, 0))
        self.cells[cell] = (value, version + 1)
        self.events.append(("store", cell, value))


# --------------------------------------------------------------------------- reference semantics (per lane)


def _lane_reference(name, lid, extent, flag, mem):
    """Per-lane meaning of each corpus kernel, written beside its TTGIR name.

    ``mload(cell, mask, other)`` is Triton's masked load: the read happens iff
    the mask is true, otherwise the value is ``other`` (UNDEF when absent).
    Pointers are splat, so every lane addresses element 0; ``p1`` is +1 element.
    """
    IN, OUT = ("in", 0), ("out", 0)
    IN1 = ("in", 4)
    m = lid < extent

    def mload(cell, mask, other=UNDEF, volatile=False):
        return mem.load(cell, volatile) if mask else other

    def mstore(cell, value, mask=True):
        if mask:
            mem.store(cell, value)

    if name == "NESTED_IF":
        x = mload(IN, m, 7.0) if flag > 0 else 7.0
        mstore(OUT, x)
    elif name == "NESTED_IF_NO_OTHER":
        x = mload(IN, m) if flag > 0 else 7.0
        mstore(OUT, x)
    elif name == "RUNTIME_LOOP":
        c = 7.0
        for _ in range(0, flag):
            c = fadd(c, mload(IN, m, 7.0))
        mstore(OUT, c)
    elif name == "ZERO_TRIP_CONST_LOOP":
        mstore(OUT, 7.0)
    elif name == "DEPENDENT_OTHER":
        x = mload(IN, m, 7.0)
        o = fmul(x, 2.0)
        mstore(OUT, mload(IN, lid >= extent, o))
    elif name == "LOAD_THEN_STORE_DIFFERENT_MASKS":
        x = mload(IN, m, 7.0)
        mstore(OUT, x, lid < extent - 1)
    elif name == "NEGATED_MASK":
        mstore(OUT, mload(IN, not m, 7.0))
    elif name == "CONJOINED_MASK":
        mstore(OUT, mload(IN, m and lid >= 4, 7.0))
    elif name == "STORE_BETWEEN_SAME_PTR":
        x = mload(IN, m, 7.0)
        mstore(IN, 7.0, m)
        y = mload(IN, m, 7.0)
        mstore(OUT, fadd(x, y))
    elif name == "TWO_PTRS_SAME_MASK":
        mstore(OUT, fadd(mload(IN, m, 7.0), mload(IN1, m, 7.0)))
    elif name == "SAME_PTR_UNEQUAL_MASKS":
        mstore(OUT, fadd(mload(IN, m, 7.0), mload(IN, lid < extent + 1, 7.0)))
    elif name == "SAME_PTR_SAME_MASK_DIFFERENT_OTHER":
        mstore(OUT, fadd(mload(IN, m, 7.0), mload(IN, m, 9.0)))
    elif name == "VOLATILE_LOAD":
        mstore(OUT, mload(IN, m, 7.0, volatile=True))
    elif name == "CACHE_MODIFIED_LOAD":
        mstore(OUT, mload(IN, m, 7.0))  # cache/eviction hints: documented no-ops
    # Lane-indexed: element lid of each buffer (4 bytes per element).
    elif name in ("LANE_INDEXED_F32", "LANE_INDEXED_I64_OFFSETS"):
        mstore(("out", 4 * lid), mload(("in", 4 * lid), m, 7.0), m)
    elif name == "STRIDED_2_F32":
        mstore(("out", 4 * lid), mload(("in", 8 * lid), m, 7.0), m)
    elif name == "LANE_PLUS_CONST_F32":
        mstore(("out", 4 * lid), mload(("in", 4 * lid + 4), m, 7.0), m)
    elif name == "LANE_INDEXED_I32":
        mstore(("out", 4 * lid), mload(("in", 4 * lid), m, 7), m)  # i32 element, `other` 7
    else:
        raise KeyError(name)


def _kernel(body, element="f32", zero="7.0"):
    head = NESTED_IF[: NESTED_IF.index("    %c0 = ")]
    if element != "f32":
        head = head.replace("!tt.ptr<f32>", f"!tt.ptr<{element}>").replace("tensor<128xf32>", f"tensor<128x{element}>")
        head = head.replace("dense<7.0>", f"dense<{zero}>")
    return head + body + "    tt.return\n  }\n}"


LANE_INDEXED_F32 = _kernel("""    %pp = tt.addptr %p, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %qq = tt.addptr %q, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %x = tt.load %pp, %m, %z : tensor<128x!tt.ptr<f32>>
    tt.store %qq, %x, %m : tensor<128x!tt.ptr<f32>>
""")
LANE_INDEXED_I64_OFFSETS = _kernel("""    %i64 = arith.extsi %i : tensor<128xi32> to tensor<128xi64>
    %pp = tt.addptr %p, %i64 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
    %qq = tt.addptr %q, %i64 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
    %x = tt.load %pp, %m, %z : tensor<128x!tt.ptr<f32>>
    tt.store %qq, %x, %m : tensor<128x!tt.ptr<f32>>
""")
STRIDED_2_F32 = _kernel("""    %two = arith.constant dense<2> : tensor<128xi32>
    %s = arith.muli %i, %two : tensor<128xi32>
    %pp = tt.addptr %p, %s : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %qq = tt.addptr %q, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %x = tt.load %pp, %m, %z : tensor<128x!tt.ptr<f32>>
    tt.store %qq, %x, %m : tensor<128x!tt.ptr<f32>>
""")
LANE_PLUS_CONST_F32 = _kernel("""    %c1 = arith.constant dense<1> : tensor<128xi32>
    %i1 = arith.addi %i, %c1 : tensor<128xi32>
    %pp = tt.addptr %p, %i1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %qq = tt.addptr %q, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %x = tt.load %pp, %m, %z : tensor<128x!tt.ptr<f32>>
    tt.store %qq, %x, %m : tensor<128x!tt.ptr<f32>>
""")
LANE_INDEXED_I32 = _kernel(
    """    %pp = tt.addptr %p, %i : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %qq = tt.addptr %q, %i : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %x = tt.load %pp, %m, %z : tensor<128x!tt.ptr<i32>>
    tt.store %qq, %x, %m : tensor<128x!tt.ptr<i32>>
""",
    element="i32",
    zero="7",
)

CORPUS = {
    "NESTED_IF": NESTED_IF,
    "NESTED_IF_NO_OTHER": NESTED_IF_NO_OTHER,
    "RUNTIME_LOOP": RUNTIME_LOOP,
    "ZERO_TRIP_CONST_LOOP": ZERO_TRIP_CONST_LOOP,
    "DEPENDENT_OTHER": DEPENDENT_OTHER,
    "LOAD_THEN_STORE_DIFFERENT_MASKS": LOAD_THEN_STORE_DIFFERENT_MASKS,
    "NEGATED_MASK": NEGATED_MASK,
    "CONJOINED_MASK": CONJOINED_MASK,
    "STORE_BETWEEN_SAME_PTR": STORE_BETWEEN_SAME_PTR,
    "TWO_PTRS_SAME_MASK": TWO_PTRS_SAME_MASK,
    "SAME_PTR_UNEQUAL_MASKS": SAME_PTR_UNEQUAL_MASKS,
    "SAME_PTR_SAME_MASK_DIFFERENT_OTHER": SAME_PTR_SAME_MASK_DIFFERENT_OTHER,
    "VOLATILE_LOAD": VOLATILE_LOAD,
    "CACHE_MODIFIED_LOAD": CACHE_MODIFIED_LOAD,
    "LANE_INDEXED_F32": LANE_INDEXED_F32,
    "LANE_INDEXED_I64_OFFSETS": LANE_INDEXED_I64_OFFSETS,
    "STRIDED_2_F32": STRIDED_2_F32,
    "LANE_PLUS_CONST_F32": LANE_PLUS_CONST_F32,
    "LANE_INDEXED_I32": LANE_INDEXED_I32,
}

# Every predicate in the corpus is decided by these: lid<N, lid<N-1, lid<N+1,
# lid>=4, flag>0, and loop trip counts 0..2.
GRID = [(lid, extent, flag) for lid in range(6) for extent in range(6) for flag in (-1, 0, 1, 2)]

# --------------------------------------------------------------------------- LLVM subset evaluator

_SUPPORTED = (
    "addrspacecast",
    "load",
    "store",
    "icmp",
    "and",
    "or",
    "xor",
    "add",
    "sub",
    "mul",
    "shl",
    "lshr",
    "ashr",
    "select",
    "phi",
    "getelementptr",
    "fadd",
    "fmul",
    "fsub",
    "br",
    "ret",
    "zext",
    "sext",
    "trunc",
)
_SIZES = {"float": 4, "i32": 4, "i8": 1, "i64": 8, "i16": 2}
_ICMP = {
    "eq": lambda a, b, kind: a == b,
    "ne": lambda a, b, kind: a != b,
    "slt": lambda a, b, kind: _signed(a, kind) < _signed(b, kind),
    "sle": lambda a, b, kind: _signed(a, kind) <= _signed(b, kind),
    "sgt": lambda a, b, kind: _signed(a, kind) > _signed(b, kind),
    "sge": lambda a, b, kind: _signed(a, kind) >= _signed(b, kind),
    "ult": lambda a, b, kind: a < b,
    "ule": lambda a, b, kind: a <= b,
    "ugt": lambda a, b, kind: a > b,
    "uge": lambda a, b, kind: a >= b,
}
_STEP_BUDGET = 20000
_WIDTH = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


def _wrap(value, kind):
    """Unsigned bit pattern of ``value`` at the width of ``kind``."""
    return int(value) & ((1 << _WIDTH[kind]) - 1)


def _signed(pattern, kind):
    bits = _WIDTH[kind]
    return pattern - (1 << bits) if pattern >> (bits - 1) else pattern


def _int_literal(token, kind):
    if token in ("undef", "poison"):
        return UNDEF
    if kind == "i1":
        return {"true": 1, "false": 0}[token]
    return _wrap(int(token), kind)


def _split_blocks(body):
    """[(label, [instruction lines])]; the unnamed entry block gets LLVM's implicit number."""
    header = body[: body.index("{") + 1]
    args = re.findall(r"(?:ptr(?: addrspace\(\d+\))?|i32|i64|float) (%[\w.]+)", header)
    unnamed = sum(1 for a in args if a[1:].isdigit())
    text = body[len(header) :]
    blocks, label, lines = [], str(unnamed), []
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line == "}":
            continue
        m = re.match(r"^([\w.]+):$", line)
        if m:
            blocks.append((label, lines))
            label, lines = m.group(1), []
        else:
            lines.append(line)
    blocks.append((label, lines))
    return args, dict(blocks), blocks[0][0]


def _float_literal(token):
    if token in ("undef", "poison"):
        return UNDEF
    if token.startswith("0x"):
        return struct.unpack(">d", struct.pack(">Q", int(token, 16)))[0]
    return float(token)


def _elem_size(type_text):
    m = re.match(r"\[(\d+) x (\w+)\]", type_text)
    if m:
        return int(m.group(1)) * _SIZES[m.group(2)]
    if type_text in _SIZES:
        return _SIZES[type_text]
    raise Unsupported(f"getelementptr element type {type_text}")


def evaluate_llvm(body, lid, extent, flag, mem):
    """Run one lane of the kernel body over ``mem``; returns the final SSA environment.

    Integers are unsigned bit patterns at their instruction width (see module
    docstring); memory events are recorded in ``mem``."""
    args, blocks, entry = _split_blocks(body)
    env = {}
    scalars = {}
    for index, name in enumerate(args):
        if index == 0:
            env[name] = ("in", 0)
        elif index == 1:
            env[name] = ("out", 0)
        elif index == 2:
            scalars[name] = _wrap(extent, "i32")
        elif index == 3:
            scalars[name] = _wrap(flag, "i32")
        elif name == "%pid":
            env[name] = 0
        elif name == "%lid":
            env[name] = _wrap(lid, "i32")
        else:
            raise Unsupported(f"argument {name}")

    def value(token, kind):
        if token.startswith("%"):
            if token not in env:
                raise Unsupported(f"use of undefined value {token}")
            return env[token]
        if kind == "float":
            return _float_literal(token)
        return _int_literal(token, kind)

    def truth(token):
        v = value(token, "i1")
        if v is UNDEF:
            raise Unsupported("branch or select on an undefined i1")
        return bool(v)

    label, previous, steps = entry, None, 0
    while True:
        lines = blocks[label]
        for line in lines:
            steps += 1
            if steps > _STEP_BUDGET:
                raise Unsupported("step budget exceeded (runaway loop)")
            m = re.match(r"^(?:(%[\w.]+) = )?(\w+)(?: (.*))?$", line)
            if not m:
                raise Unsupported(line)
            dest, op, rest = m.group(1), m.group(2), m.group(3) or ""
            if op not in _SUPPORTED:
                raise Unsupported(line)
            if op == "addrspacecast":
                src = re.match(r"ptr(?: addrspace\(\d+\))? (%[\w.]+) to ptr(?: addrspace\(\d+\))?$", rest)
                if not src:
                    raise Unsupported(line)
                env[dest] = env[src.group(1)]
            elif op == "load":
                m2 = re.match(r"(volatile )?(float|i32), ptr(?: addrspace\(\d+\))? (%[\w.]+)(?:, align \d+)?$", rest)
                if not m2:
                    raise Unsupported(line)
                pointer = m2.group(3)
                if pointer in scalars:
                    if m2.group(1) or m2.group(2) != "i32":
                        raise Unsupported(line)
                    env[dest] = scalars[pointer]
                else:
                    # Device element loads: float or i32 cells; the cell value is
                    # whatever was stored there or a symbolic memory token.
                    env[dest] = mem.load(env[pointer], volatile=bool(m2.group(1)))
            elif op == "store":
                m2 = re.match(
                    r"(volatile )?(float|i32) (\S+), ptr(?: addrspace\(\d+\))? (%[\w.]+)(?:, align \d+)?$", rest
                )
                if not m2 or m2.group(1):
                    raise Unsupported(line)
                mem.store(env[m2.group(4)], value(m2.group(3), m2.group(2)))
            elif op == "icmp":
                m2 = re.match(r"(\w+) (i32|i64|i1) (\S+), (\S+)$", rest)
                if not m2 or m2.group(1) not in _ICMP:
                    raise Unsupported(line)
                kind = m2.group(2)
                a, b = value(m2.group(3), kind), value(m2.group(4), kind)
                if a is UNDEF or b is UNDEF:
                    raise Unsupported(f"comparison of an undefined integer: {line}")
                env[dest] = int(_ICMP[m2.group(1)](a, b, kind))
            elif op in ("and", "or", "xor"):
                m2 = re.match(r"(disjoint )?(i1|i8|i16|i32|i64) (\S+), (\S+)$", rest)
                if not m2 or (m2.group(1) and op != "or"):
                    raise Unsupported(line)
                kind = m2.group(2)
                a, b = value(m2.group(3), kind), value(m2.group(4), kind)
                if a is UNDEF or b is UNDEF:
                    raise Unsupported(f"logic on an undefined integer: {line}")
                if m2.group(1) and a & b:
                    raise Unsupported(f"poison: 'or disjoint' operands overlap ({a:#x} & {b:#x}): {line}")
                env[dest] = {"and": a & b, "or": a | b, "xor": a ^ b}[op]
            elif op in ("add", "sub", "mul"):
                m2 = re.match(r"((?:(?:nsw|nuw) )*)(i8|i16|i32|i64) (\S+), (\S+)$", rest)
                if not m2:
                    raise Unsupported(line)
                flags, kind = m2.group(1).split(), m2.group(2)
                a, b = value(m2.group(3), kind), value(m2.group(4), kind)
                if a is UNDEF or b is UNDEF:
                    raise Unsupported(f"arithmetic on an undefined integer: {line}")
                exact_unsigned = {"add": a + b, "sub": a - b, "mul": a * b}[op]
                sa, sb = _signed(a, kind), _signed(b, kind)
                exact_signed = {"add": sa + sb, "sub": sa - sb, "mul": sa * sb}[op]
                result = _wrap(exact_unsigned, kind)
                if "nuw" in flags and exact_unsigned != result:
                    raise Unsupported(f"poison: nuw overflow: {line}")
                if "nsw" in flags and _signed(result, kind) != exact_signed:
                    raise Unsupported(f"poison: nsw overflow: {line}")
                env[dest] = result
            elif op in ("shl", "lshr", "ashr"):
                m2 = re.match(r"((?:(?:nsw|nuw|exact) )*)(i8|i16|i32|i64) (\S+), (\S+)$", rest)
                if not m2:
                    raise Unsupported(line)
                flags, kind = m2.group(1).split(), m2.group(2)
                a, b = value(m2.group(3), kind), value(m2.group(4), kind)
                if a is UNDEF or b is UNDEF:
                    raise Unsupported(f"shift of an undefined integer: {line}")
                if b >= _WIDTH[kind]:
                    raise Unsupported(f"poison: shift amount {b} >= width: {line}")
                if op == "shl":
                    result = _wrap(a << b, kind)
                    if "nuw" in flags and (result >> b) != a:
                        raise Unsupported(f"poison: shl nuw overflow: {line}")
                    if "nsw" in flags and _signed(result, kind) != _signed(a, kind) * (1 << b):
                        raise Unsupported(f"poison: shl nsw overflow: {line}")
                elif op == "lshr":
                    result = a >> b
                    if "exact" in flags and (result << b) != a:
                        raise Unsupported(f"poison: lshr exact drops bits: {line}")
                else:
                    result = _wrap(_signed(a, kind) >> b, kind)
                    if "exact" in flags and _wrap(_signed(result, kind) << b, kind) != a:
                        raise Unsupported(f"poison: ashr exact drops bits: {line}")
                env[dest] = result
            elif op in ("zext", "sext", "trunc"):
                m2 = re.match(r"(i1|i8|i16|i32|i64) (\S+) to (i1|i8|i16|i32|i64)$", rest)
                if not m2:
                    raise Unsupported(line)
                source, target = m2.group(1), m2.group(3)
                v = value(m2.group(2), source)
                if v is UNDEF:
                    raise Unsupported(f"cast of an undefined integer: {line}")
                if op == "trunc":
                    if _WIDTH[target] >= _WIDTH[source]:
                        raise Unsupported(line)
                    env[dest] = _wrap(v, target)  # low bits
                else:
                    if _WIDTH[target] <= _WIDTH[source]:
                        raise Unsupported(line)
                    env[dest] = _wrap(_signed(v, source) if op == "sext" else v, target)
            elif op == "select":
                m2 = re.match(r"i1 (\S+), (float|i1|i32) (\S+), (?:float|i1|i32) (\S+)$", rest)
                if not m2:
                    raise Unsupported(line)
                cond = truth(m2.group(1))
                env[dest] = value(m2.group(3), m2.group(2)) if cond else value(m2.group(4), m2.group(2))
            elif op == "phi":
                m2 = re.match(r"(float|i32|i1|i64) (.*)$", rest)
                if not m2:
                    raise Unsupported(line)
                incoming = dict((lab, val) for val, lab in re.findall(r"\[ (\S+), %([\w.]+) \]", m2.group(2)))
                if previous not in incoming:
                    raise Unsupported(f"phi without incoming edge from {previous}: {line}")
                env[dest] = value(incoming[previous], m2.group(1))
            elif op == "getelementptr":
                m2 = re.match(
                    r"(?:inbounds )?(\[\d+ x \w+\]|\w+), ptr(?: addrspace\(\d+\))? (%[\w.]+), (i32|i64) (\S+)$", rest
                )
                if not m2:
                    raise Unsupported(line)
                base = env[m2.group(2)]
                index = value(m2.group(4), m2.group(3))
                if index is UNDEF:
                    raise Unsupported(f"getelementptr with an undefined index: {line}")
                env[dest] = (base[0], base[1] + _elem_size(m2.group(1)) * _signed(index, m2.group(3)))
            elif op in ("fadd", "fmul", "fsub"):
                m2 = re.match(r"(?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc) )*float (\S+), (\S+)$", rest)
                if not m2:
                    raise Unsupported(line)
                if any(flag_ in rest.split(" float ")[0] for flag_ in ("fast", "reassoc")):
                    raise Unsupported(f"reassociating fast-math flag: {line}")
                env[dest] = {"fadd": fadd, "fmul": fmul, "fsub": fsub}[op](
                    value(m2.group(1), "float"), value(m2.group(2), "float")
                )
            elif op == "br":
                m2 = re.match(r"i1 (\S+), label %([\w.]+), label %([\w.]+)$", rest)
                if m2:
                    target = m2.group(2) if truth(m2.group(1)) else m2.group(3)
                else:
                    m3 = re.match(r"label %([\w.]+)$", rest)
                    if not m3:
                        raise Unsupported(line)
                    target = m3.group(1)
                previous, label = label, target
                break
            elif op == "ret":
                if rest != "void":
                    raise Unsupported(line)
                return env
        else:
            raise Unsupported(f"block {label} falls through without a terminator")


# --------------------------------------------------------------------------- equivalence


def compare_events(reference, candidate):
    """Violations of the memory contract between a reference trace and a candidate trace."""
    problems = []
    ref_loads = [event[1:] for event in reference if event[0] == "load"]
    cand_loads = [event[1:] for event in candidate if event[0] == "load"]
    ref_reads = {(cell, version) for cell, version, _volatile in ref_loads}
    for cell, version, _volatile in cand_loads:
        if (cell, version) not in ref_reads:
            problems.append(f"read of {cell} (version {version}) that the source does not perform")
    ref_volatile = [(cell, version) for cell, version, volatile in ref_loads if volatile]
    cand_volatile = [(cell, version) for cell, version, volatile in cand_loads if volatile]
    if ref_volatile != cand_volatile:
        problems.append(f"volatile reads differ: source {ref_volatile} candidate {cand_volatile}")
    ref_stores = [event[1:] for event in reference if event[0] == "store"]
    cand_stores = [event[1:] for event in candidate if event[0] == "store"]
    i = j = 0
    while i < len(ref_stores) or j < len(cand_stores):
        if i < len(ref_stores) and ref_stores[i][1] is UNDEF:
            # Unspecified source value: the candidate may store anything there or nothing.
            if j < len(cand_stores) and cand_stores[j][0] == ref_stores[i][0]:
                j += 1
            i += 1
            continue
        if i >= len(ref_stores):
            problems.append(f"extra store {cand_stores[j]} not in the source")
            j += 1
            continue
        if j >= len(cand_stores):
            problems.append(f"missing store {ref_stores[i]}")
            i += 1
            continue
        if ref_stores[i] != cand_stores[j]:
            problems.append(f"store differs: source {ref_stores[i]} candidate {cand_stores[j]}")
        i += 1
        j += 1
    return problems


def check_case(name, body, grid=GRID):
    """All violations over the grid, each tagged with its lane inputs."""
    violations = []
    for lid, extent, flag in grid:
        ref, cand = Memory(), Memory()
        _lane_reference(name, lid, extent, flag, ref)
        evaluate_llvm(body, lid, extent, flag, cand)
        for problem in compare_events(ref.events, cand.events):
            violations.append(f"lid={lid} extent={extent} flag={flag}: {problem}")
    return violations


def lower(text):
    return function_body(cpp.run_to_llvm(text))


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_lowering_preserves_masked_memory_semantics(record_property, name):
    """Any lowering stage: reads only where the source reads, volatile exact, stores exact."""
    body = lower(CORPUS[name])
    record_property("float_loads", str(len(re.findall(r"\bload (?:volatile )?float\b", body))))
    record_property("cond_branches", str(len(re.findall(r"\bbr i1\b", body))))
    try:
        violations = check_case(name, body)
    except Unsupported as error:
        pytest.fail(f"UNSUPPORTED IR for {name}: {error}\n{body}")
    assert not violations, f"{name}: {len(violations)} violations, first: {violations[0]}\n{body}"


def test_reference_reads_happen_only_under_true_masks():
    """Self-check of the reference: a false mask never reads; a true mask reads once per load."""
    for name in CORPUS:
        for lid, extent, flag in GRID:
            mem = Memory()
            _lane_reference(name, lid, extent, flag, mem)
            reads = [e for e in mem.events if e[0] == "load"]
            if name in ("NEGATED_MASK",):
                assert bool(reads) == (lid >= extent)
            elif name in ("CONJOINED_MASK",):
                assert bool(reads) == (lid < extent and lid >= 4)
            elif name in ("DEPENDENT_OTHER", "SAME_PTR_UNEQUAL_MASKS"):
                pass  # complementary / overlapping masks: covered by the structural pins' intent
            elif name == "ZERO_TRIP_CONST_LOOP":
                assert not reads
            elif name == "RUNTIME_LOOP":
                assert len(reads) == (max(flag, 0) if lid < extent else 0)
            elif name.startswith("NESTED_IF"):
                assert bool(reads) == (flag > 0 and lid < extent)
            else:
                assert bool(reads) == (lid < extent)
            if name.startswith(("LANE", "STRIDED")):
                assert all(cell[1] % 4 == 0 for _k, cell, *_r in mem.events)


# ---- negative controls: deliberate defects must be caught -------------------


def _load_guarding_branch(body):
    """The first `br i1` with a float load in exactly one successor block, either edge.

    Shape-agnostic: the accepted lowering puts the load on the true edge, the
    full pipeline may put it on the false edge of a complemented predicate.
    """
    blocks = {}
    for m in re.finditer(r"^([\w.]+):[^\n]*\n(.*?)(?=^[\w.]+:|\Z)", body, re.M | re.S):
        blocks[m.group(1)] = m.group(2)
    for m in re.finditer(r"br i1 (%[\w.]+), label %([\w.]+), label %([\w.]+)", body):
        hits = [t for t in (m.group(2), m.group(3)) if re.search(r"\bload (?:volatile )?float\b", blocks.get(t, ""))]
        if len(hits) == 1:
            return m, hits[0]
    return None, None


def test_guard_inversion_is_caught():
    """Swapping the guard's edges makes the read happen exactly when the source does not."""
    body = lower(NEGATED_MASK)
    m, _ = _load_guarding_branch(body)
    assert m is not None, body
    inverted = body.replace(m.group(0), f"br i1 {m.group(1)}, label %{m.group(3)}, label %{m.group(2)}")
    assert any("does not perform" in v for v in check_case("NEGATED_MASK", inverted))


def test_volatile_loss_is_caught():
    body = lower(VOLATILE_LOAD)
    assert "load volatile float" in body
    assert any(
        "volatile reads differ" in v
        for v in check_case("VOLATILE_LOAD", body.replace("load volatile float", "load float"))
    )


def test_intervening_store_merge_is_caught():
    body = lower(STORE_BETWEEN_SAME_PTR)
    loads = re.findall(r"(%[\w.]+) = load float, ptr (%[\w.]+)", body)
    if len(loads) < 2:
        pytest.skip("lowering already forwarded the store; nothing to merge")
    first, second = loads[0][0], loads[1][0]
    merged = re.sub(rf"{re.escape(second)} = load float[^\n]*\n", "", body)
    merged = re.sub(rf"(?<![\w.]){re.escape(second)}(?![\w.])", first, merged)
    assert any("store differs" in v for v in check_case("STORE_BETWEEN_SAME_PTR", merged))


def test_corrupted_other_is_caught():
    body = lower(NEGATED_MASK)
    assert "7.000000e+00" in body
    assert any("store differs" in v for v in check_case("NEGATED_MASK", body.replace("7.000000e+00", "8.000000e+00")))


def test_eager_load_defect_is_caught():
    """The original defect shape: the read hoisted above its guard. Built from whatever
    lowering is loaded by moving the guarded block's prefix through its load (address
    computation included) in front of the guarding branch; the real unsafe helper is
    the fresh-process control in run_cpp_semantics.py."""
    body = lower(NEGATED_MASK)
    m, target = _load_guarding_branch(body)
    assert m is not None, body
    block = re.search(
        rf"^{re.escape(target)}:[^\n]*\n((?:.*?\n)*?\s*%[\w.]+ = load (?:volatile )?float[^\n]*\n)", body, re.M
    )
    assert block, body
    prefix = block.group(1)
    hoisted = body.replace(prefix, "", 1).replace("  " + m.group(0), prefix.strip() + "\n  " + m.group(0), 1)
    assert hoisted != body
    violations = check_case("NEGATED_MASK", hoisted)
    assert any("does not perform" in v for v in violations), hoisted


def test_unsupported_ir_is_reported_not_passed():
    body = lower(NEGATED_MASK).replace("ret void", "call void @llvm.trap()\n  ret void")
    with pytest.raises(Unsupported):
        check_case("NEGATED_MASK", body)


# ---- integer width pins: no advertised operation falls through to Python semantics ----


def _run(lines, lid=0, extent=1, flag=1):
    body = (
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2, "
        "ptr addrspace(2) %3, i32 %pid, i32 %lid) {\n" + "\n".join("  " + l for l in lines) + "\n  ret void\n}\n"
    )
    return evaluate_llvm(body, lid, extent, flag, Memory())


@pytest.mark.parametrize(
    "line,expected",
    [
        ("%v = trunc i64 4294967296 to i32", 0),  # the primary's mutant: low bits only
        ("%v = trunc i64 4294967297 to i32", 1),
        ("%v = trunc i32 -1 to i1", 1),
        ("%v = sext i32 -1 to i64", (1 << 64) - 1),  # sign-extended pattern of -1
        ("%v = zext i32 -1 to i64", 4294967295),  # zero-extended pattern
        ("%v = zext i1 true to i32", 1),
        ("%v = add i32 4294967295, 1", 0),  # modular
        ("%v = sub i32 0, 1", 4294967295),
        ("%v = mul i32 65536, 65536", 0),
        ("%v = add i64 4294967295, 1", 4294967296),  # no 32-bit clamp at i64
        ("%v = or disjoint i32 2, 1", 3),
        ("%v = icmp ult i64 4294967296, 1", 0),  # unsigned at the instruction width
        ("%v = icmp ugt i32 -1, 0", 1),
        ("%v = icmp slt i32 -1, 0", 1),
        ("%v = icmp sgt i64 4294967295, 0", 1),  # positive at i64, would be -1 at i32
        ("%v = shl i32 1, 31", 2147483648),
        ("%v = shl i32 1, 32", None),  # poison: amount >= width
        ("%v = shl i32 3, 31", 2147483648),  # wraps
        ("%v = lshr i32 -1, 31", 1),
        ("%v = ashr i32 -8, 1", 4294967292),  # sign-preserving: -4
        ("%v = ashr i64 -8, 1", (1 << 64) - 4),
    ],
)
def test_integer_operations_use_llvm_widths(line, expected):
    if expected is None:
        with pytest.raises(Unsupported):
            _run([line])
        return
    assert _run([line])["%v"] == expected


@pytest.mark.parametrize(
    "line",
    [
        "%v = add nsw i32 2147483647, 1",
        "%v = sub nsw i32 -2147483648, 1",
        "%v = mul nsw i32 65536, 65536",
        "%v = add nuw i32 4294967295, 1",
        "%v = or disjoint i32 3, 1",
        "%v = trunc i32 5 to i32",
        "%v = zext i32 5 to i32",
        "%v = shl nuw i32 2147483648, 1",
        "%v = shl nsw i32 1073741824, 1",
        "%v = lshr exact i32 3, 1",
    ],
)
def test_poison_producing_or_ill_formed_integer_operations_are_unsupported(line):
    with pytest.raises(Unsupported):
        _run([line])


def test_sext_and_zext_choose_different_pointers():
    """A sign-dependent index must move the pointer exactly as LLVM would."""
    lines = [
        "%.generic = addrspacecast ptr addrspace(1) %0 to ptr",
        "%m1 = add i32 0, -1",
        "%wide = {cast} i32 %m1 to i64",
        "%p = getelementptr float, ptr %.generic, i64 %wide",
        "%v = load float, ptr %p, align 4",
    ]
    for cast, offset in (("sext", -4), ("zext", 4 * 4294967295)):
        mem = Memory()
        body = (
            "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2, "
            "ptr addrspace(2) %3, i32 %pid, i32 %lid) {\n"
            + "\n".join("  " + l.replace("{cast}", cast) for l in lines)
            + "\n  ret void\n}\n"
        )
        evaluate_llvm(body, 0, 1, 1, mem)
        assert mem.events == [("load", ("in", offset), 0, False)], (cast, mem.events)


def test_truncation_mutant_moves_the_pointer_and_is_caught():
    """The primary's mutant on NEGATED_MASK: `trunc i64 4294967296 to i32` is 0, so the
    select picks step 1 and the read lands on element 1 instead of element 0."""
    body = lower(NEGATED_MASK)
    load = re.search(r"(%[\w.]+) = load float, ptr (%[\w.]+), align 4", body)
    assert load, body
    mutant = body.replace(
        load.group(0),
        "%narrow = trunc i64 4294967296 to i32\n  %zero = icmp eq i32 %narrow, 0\n"
        "  %step = select i1 %zero, i32 1, i32 0\n"
        f"  %pointer = getelementptr float, ptr {load.group(2)}, i32 %step\n"
        f"  {load.group(1)} = load float, ptr %pointer, align 4",
    )
    violations = check_case("NEGATED_MASK", mutant)
    assert violations and all("('in', 4)" in v for v in violations), violations[:3]
