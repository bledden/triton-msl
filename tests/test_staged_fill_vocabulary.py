"""The staged fill's vocabulary: carried tensors, per-addition sign extension, narrow extensions.

When a 2-D tile is larger than the threadgroup, the generic lowerer re-reads (or stores) it
cooperatively and rebuilds each element's mask and address from ``(_fill_row, _fill_col)``.
Every value in that rebuilt text has to be the same for every element of the tile, or a
function of the fill coordinates. Three things were not:

* a tensor a loop carries (``kn = offs_n`` ... ``kn += BLOCK_N``) has no defining operation
  and was taken for a block-constant scalar, so the fill named ONE lane's register for the
  whole tile and read the wrong key rows;
* the pointer additions of an address were flattened into one sum. With a bare program id in
  it (``arange - program_id``) that sum is ``uint``, and a negative row offset beside a base
  chosen by a select or a branch landed 2**32 elements away;
* an extension from 8 or 16 bits was passed through, so a narrow sum that wraps in the source
  did not wrap in the rebuilt text.

A kernel whose address may be out of bounds is never launched here. The address tests lower
the kernel on the CPU, take the emitted index expression and evaluate it under Metal's integer
rules against the source's pointer arithmetic. Device launches are kept to kernels whose every
address (right or wrong) stays inside padded buffers.
"""

import math
import re
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import triton_msl
from triton_msl.backend.compiler import MetalBackend
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

BM = BN = 32
HEAD_DIM = 64
_POISON = -16384.0
_KEY_LOOP = r"for \(int (\w+) = \w+; \1 < max_n; \1 \+= \w+\)"
_OFFSET_REFUSAL = "cannot structurally resolve an offset term of a shared-memory-staged tt.dot operand"
_MASK_REFUSAL = "the per-element mask is not a single structurally-resolvable row/col bounds comparison"


def test_the_tree_under_test_is_the_one_imported():
    """The package imported is the one beside this file, not an installed copy."""
    imported = Path(triton_msl.__file__).resolve()
    print(f"triton_msl imported from {imported}")
    assert imported.parent.parent == Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# CPU-only lowering
# --------------------------------------------------------------------------


def _emit(fn, signature, constants, reorder_broadcast=True):
    """Lower ``fn`` to MSL on the CPU through the production admission route.

    ``reorder_broadcast=False`` runs the production TTIR passes without Triton's
    reorder-broadcast pass. That pass folds an operation on splats into a splat of a
    scalar operation, which removes every all-constant narrow sum before the lowerer
    sees it; without it the tensor-level ``arith.extsi``/``arith.extui`` reaches the
    staged fill, which is valid input for the lowerer all the same.
    """
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(fn=fn, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    if reorder_broadcast:
        module = backend.make_ttir(module, metadata, options)
    else:
        _make_ttir_without_reorder_broadcast(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options)


def _make_ttir_without_reorder_broadcast(mod, metadata, options):
    """``MetalBackend.make_ttir`` with ``add_reorder_broadcast`` left out."""
    pm = ir.pass_manager(mod.context)
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, "make_ttir_without_reorder_broadcast")
    return mod


# --------------------------------------------------------------------------
# The emitted staged lines
# --------------------------------------------------------------------------

_SUBSCRIPT = re.compile(r"\b(\w+)\[([^\[\]]*)\]")


def _staged_loads(msl):
    """``(pointer, mask or None, index)`` of every cooperative fill of a staged load."""
    found = []
    for line in msl.splitlines():
        line = line.strip()
        m = re.match(r"smem_\d+\[_sa\] = (.*);$", line)
        if not m:
            continue
        rhs = m.group(1)
        mask = None
        if " ? (" in rhs:
            head, rhs = rhs.split(" ? (", 1)
            mask = head[head.count("(") - head.count(")") :]
        sub = _SUBSCRIPT.search(rhs)
        assert sub, f"no subscript in staged load: {line}"
        found.append((sub.group(1), mask, sub.group(2)))
    return found


def _staged_stores(msl):
    """``(pointer, mask or None, index)`` of every cooperative store of a staged tile."""
    found = []
    for line in msl.splitlines():
        m = re.match(r"(?:if \((.*)\) )?(\w+)\[([^\[\]]*)\] = smem_iter_\d+\[_st\];$", line.strip())
        if m:
            found.append((m.group(2), m.group(1), m.group(3)))
    return found


def _only(entries, pointer):
    hits = [entry for entry in entries if entry[0] == pointer]
    assert len(hits) == 1, f"expected one staged line through {pointer}, found {[e[0] for e in entries]}"
    return hits[0]


# --------------------------------------------------------------------------
# An evaluator for the emitted integer expressions, under Metal/C++ rules
# --------------------------------------------------------------------------

_C_TYPES = {
    "char": (8, True),
    "uchar": (8, False),
    "short": (16, True),
    "ushort": (16, False),
    "int": (32, True),
    "uint": (32, False),
    "long": (64, True),
    "ulong": (64, False),
}
_TYPE_NAMES = "|".join(sorted(_C_TYPES, key=len, reverse=True))
_TOKEN = re.compile(r"\s*(?:(\d+)([uU]?)|([A-Za-z_]\w*)|(<=|>=|==|!=|[-+*/()<>]))")


def _wrap(value, ctype):
    bits, signed = _C_TYPES[ctype]
    value &= (1 << bits) - 1
    return value - (1 << bits) if signed and value >> (bits - 1) else value


def _common(a, b):
    """The usual arithmetic conversions, on type names."""
    a, b = ("int" if _C_TYPES[t][0] < 32 else t for t in (a, b))
    if a == b:
        return a
    (bits_a, signed_a), (bits_b, signed_b) = _C_TYPES[a], _C_TYPES[b]
    if signed_a == signed_b:
        return a if bits_a >= bits_b else b
    unsigned, signed = (b, a) if signed_a else (a, b)
    return unsigned if _C_TYPES[unsigned][0] >= _C_TYPES[signed][0] else signed


class _NotBlockConstant(AssertionError):
    """An emitted staged expression names a value the evaluator cannot prove block-constant."""


class _Emitted:
    """The integer declarations of one emitted kernel, and an evaluator over them.

    A name is a kernel argument or fill coordinate bound by the caller, or a register
    declared exactly once and never assigned again, whose initializer is evaluated in
    turn. Anything else (a register a loop updates, a per-lane index) raises: a staged
    fill may not name it. Arithmetic wraps at the operand type's width, mixed operands
    follow the usual arithmetic conversions, a conversion to a signed type keeps the low
    bits. The value of an index expression is its mathematical value, which is what
    pointer arithmetic adds: an ``int`` or ``long`` index is sign extended, a ``uint``
    index zero extended.
    """

    def __init__(self, msl):
        self.types, self.inits, self.unstable = {}, {}, set()
        for line in msl.splitlines():
            line = line.strip()
            m = re.match(rf"(?:constant )?({_TYPE_NAMES})&? (\w+) \[\[", line)
            if m:
                self.types[m.group(2)] = m.group(1)
                continue
            m = re.match(rf"({_TYPE_NAMES}) (\w+) = (.+);$", line)
            if m:
                ctype, name, init = m.groups()
                if name in self.types:
                    assert self.types[name] == ctype, f"{name} declared as {self.types[name]} and {ctype}"
                    if self.inits.get(name) != init:
                        self.unstable.add(name)
                self.types[name] = ctype
                self.inits[name] = init
                continue
            m = re.match(rf"for \(({_TYPE_NAMES}) (\w+) = ", line)
            if m:
                self.types[m.group(2)] = m.group(1)
                self.unstable.add(m.group(2))
                continue
            m = re.match(r"(\w+) (?:[-+*/]?=) ", line)
            if m:
                self.unstable.add(m.group(1))

    def value(self, expr, **bound):
        """The mathematical value of ``expr`` with the given names bound."""
        self._bound, self._cache = bound, {}
        return self._evaluate(expr)[1]

    def _evaluate(self, expr):
        tokens, pos = [], 0
        while pos < len(expr):
            m = _TOKEN.match(expr, pos)
            if not m:
                if not expr[pos:].strip():
                    break
                raise _NotBlockConstant(f"cannot read {expr[pos : pos + 24]!r} in {expr!r}")
            tokens.append(m.groups())
            pos = m.end()
        self._tokens, self._at = tokens, 0
        result = self._comparison()
        assert self._at == len(tokens), f"trailing text in {expr!r}"
        return result

    def _peek(self, offset=0):
        at = self._at + offset
        if at >= len(self._tokens):
            return None
        number, _suffix, name, symbol = self._tokens[at]
        return name or symbol or number

    def _take(self, expected=None):
        token = self._tokens[self._at]
        text = token[2] or token[3] or token[0]
        assert expected is None or text == expected, f"expected {expected!r}, found {text!r}"
        self._at += 1
        return token

    def _comparison(self):
        left = self._sum()
        if self._peek() in ("<", "<=", ">", ">=", "==", "!="):
            symbol = self._take()[3]
            right = self._sum()
            ctype = _common(left[0], right[0])
            a, b = _wrap(left[1], ctype), _wrap(right[1], ctype)
            verdict = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[symbol]
            return ("int", int(verdict))
        return left

    def _sum(self):
        left = self._product()
        while self._peek() in ("+", "-"):
            symbol = self._take()[3]
            right = self._product()
            ctype = _common(left[0], right[0])
            a, b = _wrap(left[1], ctype), _wrap(right[1], ctype)
            left = (ctype, _wrap(a + b if symbol == "+" else a - b, ctype))
        return left

    def _product(self):
        left = self._unary()
        while self._peek() in ("*", "/"):
            symbol = self._take()[3]
            right = self._unary()
            ctype = _common(left[0], right[0])
            a, b = _wrap(left[1], ctype), _wrap(right[1], ctype)
            if symbol == "*":
                left = (ctype, _wrap(a * b, ctype))
            else:
                assert b != 0, "division by zero in an emitted expression"
                quotient = abs(a) // abs(b)
                left = (ctype, _wrap(quotient if (a < 0) == (b < 0) else -quotient, ctype))
        return left

    def _unary(self):
        if self._peek() == "-":
            self._take()
            ctype, value = self._unary()
            ctype = _common(ctype, ctype)
            return (ctype, _wrap(-value, ctype))
        if self._peek() == "(" and self._peek(1) in _C_TYPES and self._peek(2) == ")":
            self._take()
            ctype = self._take()[2]
            self._take(")")
            return (ctype, _wrap(self._unary()[1], ctype))
        return self._primary()

    def _primary(self):
        number, suffix, name, symbol = self._take()
        if number:
            value = int(number)
            return ("uint" if suffix else "int" if value < 1 << 31 else "long", value)
        if symbol == "(":
            inner = self._comparison()
            self._take(")")
            return inner
        assert name, f"unexpected {symbol!r}"
        if name == "static_cast":
            self._take("<")
            name = self._take()[2]
            self._take(">")
        if name in _C_TYPES:
            self._take("(")
            inner = self._comparison()
            self._take(")")
            return (name, _wrap(inner[1], name))
        return self._name(name)

    def _name(self, name):
        if name in self._bound:
            assert name in self.types, f"{name} is bound by the test but not declared in the kernel"
            return (self.types[name], _wrap(self._bound[name], self.types[name]))
        if name in self._cache:
            return self._cache[name]
        if name in self.unstable or name not in self.inits:
            raise _NotBlockConstant(f"the staged text names {name}, which is not one value for the whole tile")
        saved = self._tokens, self._at
        ctype = self.types[name]
        result = (ctype, _wrap(self._evaluate(self.inits[name])[1], ctype))
        self._tokens, self._at = saved
        self._cache[name] = result
        return result


def _s32(value):
    return _wrap(value, "int")


def _s64(value):
    return _wrap(value, "long")


def test_the_evaluator_follows_metal_integer_rules():
    """The evaluator itself: mixed signedness, wrap points and narrow conversions."""
    kernel = _Emitted("constant int& s [[buffer(0)]],\nuint pid [[threadgroup_position_in_grid]],\nint r_1 = pid * s;")
    assert kernel.value("pid * (-(s))", pid=1, s=128) == 2**32 - 128  # uint: zero extended as an index
    assert kernel.value("(long)((int)(pid * (-(s))))", pid=1, s=128) == -128
    assert kernel.value("r_1 + (int)pid", pid=3, s=-5) == -12
    assert kernel.value("(int)((char)(100 + 100))") == -56
    assert kernel.value("(int)((uchar)(200 + 100))") == 44
    assert kernel.value("(int)((short)(30000 + 30000))") == -5536
    assert kernel.value("(long)((ulong)((long)(-1)) * (ulong)((long)(65536)))") == -65536
    assert kernel.value("(((uint)(64) - (uint)(0)) / (uint)(32))") == 2
    assert kernel.value("(long)(-1) < (long)(5)") == 1
    assert kernel.value("(ulong)((long)(-1)) < (ulong)((long)(5))") == 0
    with pytest.raises(_NotBlockConstant):
        kernel.value("iter_7 + 1")


# --------------------------------------------------------------------------
# An attention-shaped kernel: Q, K and V tiles of 32 x 64 are staged cooperatively
# --------------------------------------------------------------------------


@triton.jit
def _attn(
    Q, K, V, Out, M, N, sq, sk, sv, so, start, max_n, step,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    KEYS: tl.constexpr, ROWS: tl.constexpr, MASK: tl.constexpr, ADDR: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    if ROWS == "minus":  # row -1 in program 1: a bare program id beside a negative offset
        offs_m = tl.arange(0, BLOCK_M) - pid
    elif ROWS == "plus-minus-one":
        offs_m = pid + tl.arange(0, BLOCK_M) - 1
    elif ROWS == "wrap32":  # a 32-bit product that wraps to 0 in the source
        offs_m = tl.arange(0, BLOCK_M) * 65536 * 65536 + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    rows64 = offs_m.to(tl.int64)
    M64 = M.to(tl.int64)
    N64 = N.to(tl.int64)

    if MASK == "lt64":
        rmask = rows64[:, None] < M64
    elif MASK == "le64":
        rmask = rows64[:, None] <= M64 - 1
    elif MASK == "gt64-swapped":
        rmask = M64 > rows64[:, None]
    elif MASK == "ge64-neg":
        rmask = (0 - offs_m[:, None]).to(tl.int64) >= 1 - M64
    elif MASK == "eq64":
        rmask = rows64[:, None] == M64 - 1
    elif MASK == "ne64":
        rmask = rows64[:, None] != M64
    elif MASK == "wide-prod":  # a 64-bit product: every row but 0 compares as >= 2**32
        rmask = (rows64 * 65536 * 65536)[:, None] < M64
    else:
        rmask = offs_m[:, None] < M

    if ADDR == "where-base":
        q_base = tl.where(start >= 0, Q + start * sq, Q)
        o_base = tl.where(start >= 0, Out + start * so, Out)
    elif ADDR == "if-base":
        if start >= 0:
            q_base = Q + start * sq
            o_base = Out + start * so
        else:
            q_base = Q + start * sq + 0 * so
            o_base = Out + start * so + 0 * sq
    elif ADDR == "scalar-base":
        q_base = Q + start * sq
        o_base = Out + start * so
    else:
        q_base = Q
        o_base = Out
    if ADDR == "sum64":
        q_ptrs = Q + (start.to(tl.int64) + offs_m)[:, None] * sq + offs_d[None, :]
        o_ptrs = Out + (start.to(tl.int64) + offs_m)[:, None] * so + offs_d[None, :]
    elif ADDR == "split":  # two pointer additions, the scalar offset in the second
        q_ptrs = (Q + offs_m[:, None] * sq) + (start * sq + offs_d[None, :])
        o_ptrs = (Out + offs_m[:, None] * so) + (start * so + offs_d[None, :])
    else:
        q_ptrs = q_base + offs_m[:, None] * sq + offs_d[None, :]
        o_ptrs = o_base + offs_m[:, None] * so + offs_d[None, :]

    q = tl.load(q_ptrs, mask=rmask, other=0.0) * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    if KEYS == "carried64":
        carried = offs_n.to(tl.int64)
    elif KEYS == "carried8":
        carried = offs_n.to(tl.int8)
    elif KEYS == "down" or KEYS == "negative-step":
        carried = offs_n + (max_n - BLOCK_N)
    elif KEYS == "offset-init":
        carried = offs_n + start
    else:
        carried = offs_n
    if KEYS == "carried2d":  # the carried index already has its row axis
        carried_rows = offs_n[:, None]
    for start_n in range(0, max_n, BLOCK_N):
        if KEYS == "induction":
            kn = start_n + offs_n
        elif KEYS == "carried8":
            kn = carried.to(tl.int32)
        else:
            kn = carried
        if KEYS == "carried":  # a 32-bit carried index under a 64-bit comparison
            kmask = kn[:, None] < N64
            cmask = kn[None, :] < N64
        else:
            kmask = kn[:, None] < N
            cmask = kn[None, :] < N
        if KEYS == "carried2d":
            k = tl.load(K + carried_rows * sk + offs_d[None, :], mask=carried_rows < N, other=0.0)
        else:
            k = tl.load(K + kn[:, None] * sk + offs_d[None, :], mask=kmask, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(cmask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if KEYS == "carried2d":
            v = tl.load(V + carried_rows * sv + offs_d[None, :], mask=carried_rows < N, other=0.0)
            carried_rows += BLOCK_N
        else:
            v = tl.load(V + kn[:, None] * sv + offs_d[None, :], mask=kmask, other=0.0)
        acc += tl.dot(p, v)
        m_i = m_new
        if KEYS == "swapped":
            carried = BLOCK_N + carried
        elif KEYS == "down":
            carried -= BLOCK_N
        elif KEYS == "negative-step":
            carried += -BLOCK_N
        elif KEYS == "argument-step":
            carried += step
        elif KEYS == "doubled":
            carried = carried * 2 + BLOCK_N
        elif KEYS == "two-additions":
            carried += BLOCK_N // 2
            carried += BLOCK_N // 2
        else:
            carried += BLOCK_N
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc, mask=rmask)


def _attn_signature(**overrides):
    signature = {name: "*fp32" for name in ("Q", "K", "V", "Out")}
    signature.update({name: "i32" for name in ("M", "N", "sq", "sk", "sv", "so", "start", "max_n", "step")})
    signature.update(overrides)
    return signature


def _attn_constants(**overrides):
    constants = {
        "SCALE": 1.0 / math.sqrt(HEAD_DIM),
        "BLOCK_M": BM,
        "BLOCK_N": BN,
        "D": HEAD_DIM,
        "KEYS": "induction",
        "ROWS": "canonical",
        "MASK": "lt",
        "ADDR": "canonical",
    }
    constants.update(overrides)
    return constants


def _emit_attn(signature=None, **constants):
    return _emit(_attn, _attn_signature(**(signature or {})), _attn_constants(**constants))


def _attn_scalars(n_keys, m_rows):
    """The integer arguments of a launch. None is 1, which Triton would fold into the kernel."""
    tiles = math.ceil(n_keys / BN) * BN
    return (m_rows, n_keys, HEAD_DIM, HEAD_DIM, HEAD_DIM, HEAD_DIM, 0, tiles, BN)


def _warm_attn(keys, n_keys, m_rows=40):
    """Compile, without launching, the very kernel ``_launch_attn`` runs; returns its MSL.

    The launch route specializes on argument divisibility, which ``_emit`` does not, so
    its text differs from ``_emit``'s in register numbering.
    """
    pointer = torch.float32
    warm = _attn.warmup(
        pointer, pointer, pointer, pointer, *_attn_scalars(n_keys, m_rows),
        grid=(2,), **_attn_constants(KEYS=keys), num_warps=4,
    )  # fmt: skip
    return warm.asm["msl"]


def _launch_attn(keys, n_keys, m_rows=40):
    """Two programs of ``_attn`` on the device; returns (Out, float64 reference, handle).

    Every buffer is padded to whole tiles plus one, so an address stays inside its
    buffer whether or not the staged fill names the right row.
    """
    tiles = math.ceil(n_keys / BN) * BN
    gen = torch.Generator().manual_seed(1117 + n_keys)
    q = torch.randn(2 * BM, HEAD_DIM, generator=gen)
    k = torch.randn(tiles + BN, HEAD_DIM, generator=gen)
    v = torch.randn(tiles + BN, HEAD_DIM, generator=gen)
    out = torch.full((2 * BM, HEAD_DIM), _POISON)
    qd, kd, vd, od = (t.to("mps") for t in (q, k, v, out))
    handle = _attn[(2,)](qd, kd, vd, od, *_attn_scalars(n_keys, m_rows), **_attn_constants(KEYS=keys), num_warps=4)
    torch.mps.synchronize()
    scores = (q[:m_rows].double() / math.sqrt(HEAD_DIM)) @ k[:n_keys].double().T
    reference = torch.softmax(scores, dim=-1) @ v[:n_keys].double()
    return od.cpu(), reference, handle


def _assert_attn_launch_stays_in_bounds(keys, n_keys, m_rows=40):
    """Before a launch: every staged address of the same kernel, evaluated, is inside its buffer.

    A fill that names a value the evaluator cannot follow fails here, before the device runs.
    Returns the MSL it evaluated, for comparison with the launched kernel's.
    """
    tiles = math.ceil(n_keys / BN) * BN
    msl = _warm_attn(keys, n_keys, m_rows)
    kernel = _Emitted(msl)
    loops = re.findall(_KEY_LOOP, msl)
    assert len(loops) == 1, loops
    rows = {"Q": 2 * BM, "K": tiles + BN, "V": tiles + BN, "Out": 2 * BM}
    lines = _staged_loads(msl) + _staged_stores(msl)
    assert sorted(pointer for pointer, _mask, _index in lines) == ["K", "Out", "Q", "V"]
    scalars = {"M": m_rows, "N": n_keys, "sq": HEAD_DIM, "sk": HEAD_DIM, "sv": HEAD_DIM, "so": HEAD_DIM}
    scalars.update(start=0, max_n=tiles, step=BN)
    for pointer, _mask, index in lines:
        for pid in (0, 1):
            for trip in range(tiles // BN):
                for lane in (0, 1, BM - 1):
                    for col in (0, HEAD_DIM - 1):
                        bound = {"pid": pid, loops[0]: BN * trip, "_fill_row": lane, "_fill_col": col}
                        element = kernel.value(index, **bound, **scalars)
                        assert 0 <= element < rows[pointer] * HEAD_DIM, (pointer, pid, trip, lane, col, element)
    return msl


def _assert_attn_result(out, reference, m_rows=40):
    assert bool((out[m_rows:] == _POISON).all()), "a row past M was stored"
    got = out[:m_rows].double()
    assert bool(torch.isfinite(got).all())
    worst = float((got - reference).abs().max())
    assert worst < 2e-4, f"max |device - float64 reference| = {worst}"


def _assert_staged_route(msl, pointers, evaluated):
    """The tiles really were filled cooperatively, each from its fill coordinates."""
    assert msl == evaluated, "the launch ran another kernel than the one whose addresses were evaluated"
    loads = {pointer: index for pointer, _mask, index in _staged_loads(msl)}
    for pointer in pointers:
        assert pointer in loads, f"no cooperative fill through {pointer}"
        assert "_fill_row" in loads[pointer] and "_fill_col" in loads[pointer], loads[pointer]
    assert any("_fill_row" in index for _pointer, _mask, index in _staged_stores(msl)), "no cooperative store"


# --------------------------------------------------------------------------
# (e) Control: an ordinary staged kernel
# --------------------------------------------------------------------------


@requires_mps
@pytest.mark.parametrize("n_keys", [45, 64])
def test_control_induction_key_index_computes_on_the_staged_route(n_keys):
    """No carried tensor, no chosen base: still staged, still right."""
    evaluated = _assert_attn_launch_stays_in_bounds("induction", n_keys)
    out, reference, handle = _launch_attn("induction", n_keys)
    _assert_attn_result(out, reference)
    _assert_staged_route(handle.asm["msl"], ("Q", "K", "V"), evaluated)


# --------------------------------------------------------------------------
# (a) A key index a loop carries as a tensor
# --------------------------------------------------------------------------


def _key_index(keys, lane, trip, start, max_n):
    """The source's key index for staged row ``lane`` on loop trip ``trip``."""
    if keys == "negative-step":
        return lane + (max_n - BN) - BN * trip
    return lane + BN * trip + (start if keys == "offset-init" else 0)


@requires_mps
@pytest.mark.parametrize("n_keys", [45, 64, 97])
@pytest.mark.parametrize("keys", ["carried32", "carried", "carried64", "swapped", "negative-step", "carried2d"])
def test_carried_key_index_computes_on_the_staged_route(keys, n_keys):
    """``kn = offs_n`` ... ``kn += BLOCK_N``, as 32-bit, 32-bit under a 64-bit mask and 64-bit.

    ``negative-step`` walks the key tiles from the last to the first (``kn += -BLOCK_N``).
    ``carried2d`` carries the index with its row axis (``offs_n[:, None]``).

    45 and 97 keys are not tile multiples; 97 needs four trips. The unfixed lowering
    read one lane's key row for the whole K and V tile.
    """
    evaluated = _assert_attn_launch_stays_in_bounds(keys, n_keys)
    out, reference, handle = _launch_attn(keys, n_keys)
    _assert_attn_result(out, reference)
    _assert_staged_route(handle.asm["msl"], ("Q", "K", "V"), evaluated)


@pytest.mark.parametrize(
    "keys", ["induction", "carried32", "carried", "carried64", "swapped", "offset-init", "negative-step", "carried2d"]
)
def test_carried_key_index_fill_is_the_source_row_on_every_trip(keys):
    """CPU-only: the emitted K and V fills, evaluated for every trip, row and column.

    The source forms ``K + kn[:, None] * sk + offs_d[None, :]``: one addition of the
    row offset (32-bit, or 64-bit for the 64-bit spelling) and one of the column.
    """
    msl = _emit_attn(KEYS=keys)
    kernel = _Emitted(msl)
    loops = re.findall(_KEY_LOOP, msl)
    assert len(loops) == 1, loops
    wide = keys == "carried64"
    for pointer, stride in (("K", "sk"), ("V", "sv")):
        _pointer, mask, index = _only(_staged_loads(msl), pointer)
        assert mask is not None
        for trip in range(4):
            for lane in (0, 1, 31):
                for col in (0, 63):
                    for stride_value in (64, -64, 1 << 26):
                        bound = {loops[0]: BN * trip, "_fill_row": lane, "_fill_col": col, stride: stride_value}
                        bound.update(start=7, max_n=128)
                        kn = _key_index(keys, lane, trip, 7, 128)
                        row = _s64(kn * stride_value) if wide else _s32(kn * stride_value)
                        assert kernel.value(index, **bound) == row + col, (pointer, trip, lane, col, stride_value)
                        for n in (1, 45, 64, 97):
                            assert kernel.value(mask, N=n, **bound) == int(kn < n), (pointer, trip, lane, n)


@pytest.mark.parametrize("keys", ["doubled", "two-additions", "down", "argument-step", "carried8"])
def test_carried_tensor_outside_the_vocabulary_is_refused(keys):
    """A carried tensor the fill does not render is refused, not read from one lane's register.

    ``doubled`` is not ``init + trip * step`` at all. The others are, but lie outside
    what the fill renders today: two additions per trip, a subtraction (``kn -= BLOCK_N``;
    ``kn += -BLOCK_N`` is rendered), a step that is a kernel argument splatted inside the
    loop, and an 8-bit carried index. The unfixed lowering emitted every one of them
    with one lane's register for the tile.
    """
    with pytest.raises(MetalNonRecoverableError, match=_OFFSET_REFUSAL):
        _emit_attn(KEYS=keys)


@requires_mps
@pytest.mark.parametrize("keys", ["down", "doubled"])
def test_carried_tensor_outside_the_vocabulary_is_refused_at_the_launch(keys):
    """The same refusal on the launch route: nothing runs and the output is not written."""
    out = torch.full((2 * BM, HEAD_DIM), _POISON, device="mps")
    q = torch.randn(2 * BM, HEAD_DIM, device="mps")
    k = torch.randn(3 * BN, HEAD_DIM, device="mps")
    with pytest.raises(MetalNonRecoverableError, match=_OFFSET_REFUSAL):
        _attn[(2,)](q, k, k.clone(), out, *_attn_scalars(64, 40), **_attn_constants(KEYS=keys), num_warps=4)
    assert bool((out == _POISON).all())


# --------------------------------------------------------------------------
# (b) Each pointer addition is sign extended on its own (never launched)
# --------------------------------------------------------------------------

_ROW = {
    "canonical": lambda pid, lane: _s32(pid * BM + lane),
    "minus": lambda pid, lane: _s32(lane - pid),
    "plus-minus-one": lambda pid, lane: _s32(pid + lane - 1),
    "wrap32": lambda pid, lane: _s32(lane * 65536 * 65536 + pid * BM + lane),
}


@pytest.mark.parametrize("rows", ["minus", "plus-minus-one"])
@pytest.mark.parametrize(
    "addr,start_type",
    [
        ("where-base", "i32"),
        ("if-base", "i32"),
        ("where-base", "i64"),
        ("if-base", "i64"),
        ("scalar-base", "i32"),
        ("split", "i32"),
    ],
)
def test_negative_row_offset_beside_a_program_id_is_sign_extended(addr, start_type, rows):
    """A select-chosen, branch-chosen or scalar-offset base, rows ``arange - program_id``.

    The source adds ``offs_m * stride`` (32-bit, sign extended) and then the column to
    the base (``split``: the column and a scalar offset together, as one 32-bit sum).
    The unfixed text was one ``uint`` sum: for program 1, row 0, stride 128 it is
    2**32 - 128 where the source's offset is -128. Evaluated, never launched.
    """
    msl = _emit_attn(signature={"start": start_type}, ROWS=rows, ADDR=addr)
    kernel = _Emitted(msl)
    stores = _staged_stores(msl)
    assert len(stores) == 1
    q_load = _staged_loads(msl)[0]
    checked = 0
    for (pointer, _mask, index), argument, stride in ((stores[0], "Out", "so"), (q_load, "Q", "sq")):
        if addr in ("scalar-base", "split"):
            assert pointer == argument
        else:  # the index is relative to the chosen pointer, which holds the start
            assert pointer not in ("Q", "K", "V", "Out"), pointer
        for pid in (0, 1, 2, 7):
            for lane in (0, 1, 31):
                for col in (0, 63):
                    for stride_value in (128, -128, 1 << 20, 0x7FFFFFFF):
                        for start in (0, 3, -2):
                            bound = {"pid": pid, "_fill_row": lane, "_fill_col": col, "start": start}
                            bound["sq" if stride == "so" else "so"] = 64
                            bound[stride] = stride_value
                            expected = _s32(_ROW[rows](pid, lane) * stride_value)
                            if addr == "scalar-base":
                                expected += _s32(start * stride_value) + col
                            elif addr == "split":
                                expected += _s32(start * stride_value + col)
                            else:
                                expected += col
                            got = kernel.value(index, **bound)
                            assert got == expected, (pointer, pid, lane, col, stride_value, start, got, expected)
                            checked += 1
    assert checked == 2 * 4 * 3 * 2 * 4 * 3


@requires_mps
def test_emitted_select_base_index_agrees_with_metal_pointer_arithmetic():
    """The same index through Metal's own compiler: pointer arithmetic only, nothing dereferenced.

    Confirms the evaluator above on the real target for the select-chosen base: program
    1, staged row 0 (source row -1), column 0, stride 128 is 128 elements BEFORE the base.
    """
    msl = _emit_attn(ROWS="minus", ADDR="where-base")
    pointer, _mask, index = _staged_stores(msl)[0]
    assert pointer != "Out"
    probe = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void probe(device long* res [[buffer(0)]], device float* base [[buffer(1)]],"
        " uint tid [[thread_position_in_grid]]) {\n"
        "    if (tid != 0) return;\n"
        "    uint pid = 1u; uint _fill_row = 0u; uint _fill_col = 0u; int so = 128;\n"
        f"    device float* p = base + ({index});\n"
        "    res[0] = (long)(p - base);\n}\n"
    )
    library = torch.mps.compile_shader(probe)
    result = torch.zeros(1, dtype=torch.int64, device="mps")
    library.probe(result, torch.zeros(1, device="mps"), threads=1)
    torch.mps.synchronize()
    evaluated = _Emitted(msl).value(index, pid=1, _fill_row=0, _fill_col=0, so=128)
    assert int(result[0]) == evaluated == -128


# --------------------------------------------------------------------------
# (d) 64-bit predicates and sums with program-id rows (CPU-only, attention-shaped)
# --------------------------------------------------------------------------

_ROW_MASK = {
    "lt64": lambda row, m: row < m,
    "le64": lambda row, m: row <= m - 1,
    "gt64-swapped": lambda row, m: m > row,
    "ge64-neg": lambda row, m: _s32(0 - row) >= 1 - m,
    "eq64": lambda row, m: row == m - 1,
    "ne64": lambda row, m: row != m,
    "wide-prod": lambda row, m: _s64(row * 65536 * 65536) < m,
}


@pytest.mark.parametrize("rows", ["minus", "canonical", "wrap32"])
@pytest.mark.parametrize("mask", sorted(_ROW_MASK))
def test_64bit_row_predicates_keep_their_predicate_and_operands(mask, rows):
    """The staged store's and the staged Q load's mask, evaluated against the source."""
    msl = _emit_attn(ROWS=rows, MASK=mask)
    kernel = _Emitted(msl)
    stores = _staged_stores(msl)
    assert len(stores) == 1
    masks = [stores[0][1], _only(_staged_loads(msl), "Q")[1]]
    assert all(masks), masks
    for emitted in masks:
        for pid in (0, 1, 3):
            for lane in range(BM):
                for m in (1, 5, 40, 0, -3, 2**31 - 1, -(2**31)):
                    row = _ROW[rows](pid, lane)
                    got = kernel.value(emitted, pid=pid, _fill_row=lane, M=m)
                    assert got == int(_ROW_MASK[mask](row, m)), (emitted, pid, lane, m, row)


@pytest.mark.parametrize("rows", ["wrap32", "minus", "canonical"])
def test_32bit_rows_under_a_64bit_sum_wrap_where_the_source_wraps(rows):
    """``Out + (start64 + offs_m)[:, None] * so + offs_d[None, :]``: rows wrap at 32 bits first."""
    msl = _emit_attn(signature={"start": "i64"}, ROWS=rows, ADDR="sum64")
    kernel = _Emitted(msl)
    stores = _staged_stores(msl)
    assert len(stores) == 1
    for (pointer, _mask, index), argument, stride in ((stores[0], "Out", "so"), (_staged_loads(msl)[0], "Q", "sq")):
        assert pointer == argument
        for pid in (0, 1, 3):
            for lane in (0, 1, 31):
                for col in (0, 63):
                    for stride_value in (128, -128, 1 << 20):
                        for start in (0, 5, 1 << 33):
                            bound = {"pid": pid, "_fill_row": lane, "_fill_col": col, "start": start}
                            bound["sq" if stride == "so" else "so"] = 64
                            bound[stride] = stride_value
                            expected = _s64((start + _ROW[rows](pid, lane)) * stride_value) + col
                            assert kernel.value(index, **bound) == expected, (pointer, pid, lane, col, start)


# --------------------------------------------------------------------------
# (d, e) A non-attention kernel whose masked A tile is staged cooperatively
# --------------------------------------------------------------------------


@triton.jit
def _single_dot(
    A, B, C, M, sam, sbk, scm, big,
    ROWS: tl.constexpr, MASK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    rk = tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)
    if ROWS == "rows64":
        r = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    elif ROWS == "pid64":
        r = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    elif ROWS == "minus1":
        r = pid * BLOCK_M + tl.arange(0, BLOCK_M) - 1
    else:
        r = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    r64 = r.to(tl.int64)
    M64 = M.to(tl.int64)
    if MASK == "le64":
        m = r64[:, None] <= M64 - 1
    elif MASK == "gt64-swapped":
        m = M64 > r64[:, None]
    elif MASK == "eq64":
        m = r64[:, None] == M64 - 1
    elif MASK == "ne64":  # loads rows past M: the buffer is padded for them
        m = r64[:, None] != M64
    elif MASK == "zext":
        m = r.to(tl.uint32).to(tl.int64)[:, None] < M64
    elif MASK == "ult":
        m = r64.to(tl.uint64)[:, None] < M64.to(tl.uint64)
    elif MASK == "wide":
        m = (r64 * 65536 * 65536)[:, None] < M64
    elif MASK == "big":
        m = r64[:, None] < big - (2**31 + 1000) + M64
    elif MASK == "and":
        m = (r[:, None] < M) & (rk[None, :] < BLOCK_K)
    else:
        m = r[:, None] < M
    a = tl.load(A + r[:, None] * sam + rk[None, :], mask=m, other=0.0)
    b = tl.load(B + rk[:, None] * sbk + rn[None, :])
    out = tl.exp(tl.dot(a, b) * 0.25)
    orow = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(C + orow[:, None] * scm + rn[None, :], out, mask=orow[:, None] < M)


_DOT_ROW = {
    "canonical": lambda pid, lane: pid * BM + lane,
    "rows64": lambda pid, lane: pid * BM + lane,
    "pid64": lambda pid, lane: pid * BM + lane,
    "minus1": lambda pid, lane: pid * BM + lane - 1,
}
_DOT_MASK = {
    "lt": lambda row, m: row < m,
    "le64": lambda row, m: row <= m - 1,
    "gt64-swapped": lambda row, m: m > row,
    "eq64": lambda row, m: row == m - 1,
    "ne64": lambda row, m: row != m,
    "zext": lambda row, m: (row & 0xFFFFFFFF) < m,
    "ult": lambda row, m: (row & 0xFFFFFFFFFFFFFFFF) < m,
    "wide": lambda row, m: _s64(row * 65536 * 65536) < m,
    "big": lambda row, m: row < m,
}


def _single_dot_scalars(m_rows):
    """The integer arguments of a launch: M, three strides and a 64-bit bound past 2**31."""
    return (m_rows, BN, BN, BN, 2**31 + 1000)


def _single_dot_constants(rows, mask):
    return {"ROWS": rows, "MASK": mask, "BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BN}


def _launch_single_dot(rows, mask, m_rows, programs=2):
    """``C = exp(0.25 * A_rows @ B)``; A is a view one row into a padded buffer.

    The buffer holds a row before the view and every tile row after it, so row -1 and
    a dropped mask both stay inside it.
    """
    gen = torch.Generator().manual_seed(3)
    a_buf = torch.randn(programs * BM + 2, BN, generator=gen)
    b = torch.randn(BN, BN, generator=gen)
    c = torch.full((programs * BM, BN), -1.0)
    a_dev, b_dev, c_dev = a_buf.to("mps"), b.to("mps"), c.to("mps")
    a_view = a_dev[1 : 1 + m_rows]
    assert (a_view.stride(0), b_dev.stride(0), c_dev.stride(0)) == (BN, BN, BN)
    handle = _single_dot[(programs,)](
        a_view, b_dev, c_dev, *_single_dot_scalars(m_rows), **_single_dot_constants(rows, mask), num_warps=4
    )
    torch.mps.synchronize()
    reference = torch.full((programs * BM, BN), -1.0, dtype=torch.float64)
    for out_row in range(min(m_rows, programs * BM)):
        row = _DOT_ROW[rows](*divmod(out_row, BM))
        a_row = a_buf[1 + row].double() if _DOT_MASK[mask](row, m_rows) else torch.zeros(BN, dtype=torch.float64)
        reference[out_row] = torch.exp(0.25 * (a_row @ b.double()))
    return c_dev.cpu().double(), reference, handle


def _emit_single_dot(rows, mask):
    signature = {name: "*fp32" for name in ("A", "B", "C")}
    signature.update({name: "i32" for name in ("M", "sam", "sbk", "scm")}, big="i64")
    return _emit(_single_dot, signature, _single_dot_constants(rows, mask))


def _assert_single_dot_launch_stays_in_bounds(rows, mask, m_rows, programs=2):
    """Before a launch: the staged A address, evaluated, is inside the padded buffer.

    Evaluates the very kernel the launch runs (compiled, not launched) and returns its MSL.
    """
    pointer = torch.float32
    warm = _single_dot.warmup(
        pointer, pointer, pointer, *_single_dot_scalars(m_rows),
        grid=(programs,), **_single_dot_constants(rows, mask), num_warps=4,
    )  # fmt: skip
    msl = warm.asm["msl"]
    kernel = _Emitted(msl)
    pointer, _mask, index = _staged_loads(msl)[0]
    assert pointer == "A"
    for pid in range(programs):
        for lane in range(BM):
            for col in (0, BN - 1):
                element = kernel.value(index, pid=pid, _fill_row=lane, _fill_col=col, sam=BN)
                assert -BN <= element < (programs * BM + 1) * BN, (pid, lane, col, element)
    return msl


def _assert_single_dot(rows, mask, m_rows):
    evaluated = _assert_single_dot_launch_stays_in_bounds(rows, mask, m_rows)
    got, reference, handle = _launch_single_dot(rows, mask, m_rows)
    assert handle.asm["msl"] == evaluated, "the launch ran another kernel than the one whose address was evaluated"
    torch.testing.assert_close(got, reference, rtol=2e-4, atol=1e-5)
    loads = _staged_loads(handle.asm["msl"])
    pointer, emitted_mask, index = loads[0]
    assert emitted_mask is not None and "_fill_row" in emitted_mask
    assert "_fill_row" in index and "_fill_col" in index


@requires_mps
@pytest.mark.parametrize("m_rows", [45, 64])
def test_control_single_dot_staged_load_computes(m_rows):
    """The plain 32-bit spelling of the non-attention kernel: staged, masked, right."""
    _assert_single_dot("canonical", "lt", m_rows)


@requires_mps
@pytest.mark.parametrize(
    "rows,mask",
    [
        ("canonical", "wide"),
        ("rows64", "wide"),
        ("pid64", "wide"),
        ("minus1", "wide"),
        ("minus1", "zext"),
        ("minus1", "ult"),
        ("pid64", "le64"),
        ("pid64", "gt64-swapped"),
        ("pid64", "eq64"),
        ("rows64", "ne64"),
        ("minus1", "big"),
    ],
)
def test_single_dot_staged_load_64bit_masks_compute(rows, mask):
    """A non-attention staged load with 64-bit masks: products, unsigned compares, swapped bounds."""
    _assert_single_dot(rows, mask, 45)


@pytest.mark.parametrize("rows,mask", [("canonical", "and"), ("rows64", "and"), ("pid64", "zext")])
def test_single_dot_masks_outside_the_vocabulary_are_refused(rows, mask):
    """A conjunction, and a zero extension of a 64-bit program-id row, refuse to stage.

    Not new with the change: these refuse on the trees before it as well.
    """
    with pytest.raises(MetalNonRecoverableError, match=_MASK_REFUSAL):
        _emit_single_dot(rows, mask)


# --------------------------------------------------------------------------
# (c) Extensions from 8 and 16 bits
# --------------------------------------------------------------------------


@triton.jit
def _narrow_dot(
    A, B, C, M, MC, sam, sbk, scm, x, y,
    OFF: tl.constexpr, MASK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    rk = tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)
    r = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    if OFF == "s8":
        r = r + (tl.full([BLOCK_M], x, tl.int8) + tl.full([BLOCK_M], y, tl.int8)).to(tl.int32)
    elif OFF == "u8":
        r = r + (tl.full([BLOCK_M], x, tl.uint8) + tl.full([BLOCK_M], y, tl.uint8)).to(tl.int32)
    elif OFF == "s16":
        r = r + (tl.full([BLOCK_M], x, tl.int16) + tl.full([BLOCK_M], y, tl.int16)).to(tl.int32)
    elif OFF == "u16":
        r = r + (tl.full([BLOCK_M], x, tl.uint16) + tl.full([BLOCK_M], y, tl.uint16)).to(tl.int32)
    elif OFF == "s8-difference":
        r = r + (tl.full([BLOCK_M], x, tl.int8) - tl.full([BLOCK_M], y, tl.int8)).to(tl.int32)
    elif OFF == "s8-then-s16":
        inner = (tl.full([BLOCK_M], x, tl.int8) + tl.full([BLOCK_M], y, tl.int8)).to(tl.int16)
        r = r + (inner + tl.full([BLOCK_M], y, tl.int16)).to(tl.int32)
    elif OFF == "s8-to-64":
        r = r + (tl.full([BLOCK_M], x, tl.int8) + tl.full([BLOCK_M], y, tl.int8)).to(tl.int64)
    elif OFF == "u16-to-64":
        r = r + (tl.full([BLOCK_M], x, tl.uint16) + tl.full([BLOCK_M], y, tl.uint16)).to(tl.int64)
    elif OFF == "s8-product":
        r = r + (tl.full([BLOCK_M], x, tl.int8) * tl.full([BLOCK_M], y, tl.int8)).to(tl.int32)
    elif OFF == "s8-row-index":
        r = r + (tl.arange(0, BLOCK_M).to(tl.int8) + tl.full([BLOCK_M], x, tl.int8)).to(tl.int32)
    if MASK == "lt64":
        m = r.to(tl.int64)[:, None] < M.to(tl.int64)
    else:
        m = r[:, None] < M
    a = tl.load(A + r[:, None] * sam + rk[None, :], mask=m, other=0.0)
    b = tl.load(B + rk[:, None] * sbk + rn[None, :])
    out = tl.exp(tl.dot(a, b) * 0.25)
    orow = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(C + orow[:, None] * scm + rn[None, :], out, mask=orow[:, None] < MC)


def _narrow_signature():
    signature = {name: "*fp32" for name in ("A", "B", "C")}
    signature.update({name: "i32" for name in ("M", "MC", "sam", "sbk", "scm", "x", "y")})
    return signature


def _narrow_constants(off, mask="lt"):
    return {"OFF": off, "MASK": mask, "BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BN}


def _s8(value):
    return _wrap(value, "char")


def _s16(value):
    return _wrap(value, "short")


# The source's row offset: each narrow operation wraps at its own width before the extension.
_NARROW_OFFSET = {
    "s8": lambda x, y: _s8(x + y),
    "u8": lambda x, y: (x + y) & 0xFF,
    "s16": lambda x, y: _s16(x + y),
    "u16": lambda x, y: (x + y) & 0xFFFF,
    "s8-difference": lambda x, y: _s8(x - y),
    "s8-then-s16": lambda x, y: _s16(_s8(x + y) + _s16(y)),
    "s8-to-64": lambda x, y: _s8(x + y),
    "u16-to-64": lambda x, y: (x + y) & 0xFFFF,
}
_NARROW_VALUES = [
    (3, 4),
    (100, 100),
    (-100, -100),
    (-100, 100),
    (127, 1),
    (200, 100),
    (200, 200),
    (30000, 30000),
    (40000, 40000),
]


@pytest.mark.parametrize("mask", ["lt", "lt64"])
@pytest.mark.parametrize("off", sorted(_NARROW_OFFSET))
def test_narrow_block_constant_sum_wraps_at_its_own_width(off, mask):
    """A sum of block constants at 8 or 16 bits, sign or zero extended into the row index.

    ``int8(100) + int8(100)`` is -56 in the source; the unfixed text added the two
    registers at 32 bits and got 200. Evaluated against the source for the staged
    load's mask and address, with the offset reaching 32 and 64 bits.
    """
    msl = _emit(_narrow_dot, _narrow_signature(), _narrow_constants(off, mask), reorder_broadcast=False)
    kernel = _Emitted(msl)
    pointer, emitted_mask, index = _staged_loads(msl)[0]
    assert pointer == "A" and emitted_mask is not None
    wide = off.endswith("-to-64")
    for x, y in _NARROW_VALUES:
        offset = _NARROW_OFFSET[off](x, y)
        for pid in (0, 1):
            for lane in (0, 31):
                for stride in (32, -32, 1 << 24):
                    row = pid * BM + lane + offset
                    bound = {"x": x, "y": y, "pid": pid, "_fill_row": lane, "_fill_col": 5, "sam": stride}
                    expected = (_s64(row * stride) if wide else _s32(row * stride)) + 5
                    assert kernel.value(index, **bound) == expected, (x, y, pid, lane, stride)
                    for m in (1, 40, 300):
                        assert kernel.value(emitted_mask, M=m, **bound) == int(row < m), (x, y, pid, lane, m)


@pytest.mark.parametrize(
    "off,reorder_broadcast",
    [("s8-product", False), ("s8-row-index", False), ("s8-row-index", True)],
)
def test_narrow_value_outside_the_vocabulary_is_refused(off, reorder_broadcast):
    """A narrow product, and a narrow value holding the row index (a truncation), refuse."""
    with pytest.raises(MetalNonRecoverableError, match=_OFFSET_REFUSAL):
        _emit(_narrow_dot, _narrow_signature(), _narrow_constants(off), reorder_broadcast=reorder_broadcast)


@requires_mps
@pytest.mark.parametrize(
    "off,x,y,pad",
    [
        ("s8", 100, 100, 400),  # -56 in the source, 200 unwrapped
        ("u8", 200, 200, 400),  # 144; -112 when added as signed registers
        ("s8-difference", -100, 100, 400),  # 56, -200 unwrapped
        ("s8-then-s16", 100, 100, 400),  # 44, 300 unwrapped
        ("s8-to-64", 100, 100, 400),
        ("s16", 32766, 2, 33000),  # -32768, 32768 unwrapped
        ("u16", 32868, 32868, 400),  # 200; -65336 when added as signed registers
    ],
)
def test_narrow_block_constant_sum_computes_on_the_device(off, x, y, pad, monkeypatch):
    """The supported narrow forms, launched: the rows read are the source's wrapped rows.

    The kernel is compiled with the reorder-broadcast pass left out (see ``_emit``), which
    is the only way these forms reach the staged fill today. A is a view ``pad`` rows into
    a buffer with ``pad`` rows on each side, so the wrapped and the unwrapped row are both
    inside it; the address of the compiled kernel is evaluated before the launch all the same.
    """
    programs = 2
    offset = _NARROW_OFFSET[off](x, y)
    m_bound = offset + 40  # staged rows 40.. of the 64 are masked off
    scalars = (m_bound, programs * BM, BN, BN, BN, x, y)
    monkeypatch.setattr(MetalBackend, "make_ttir", staticmethod(_make_ttir_without_reorder_broadcast))
    fp32 = torch.float32
    warm = _narrow_dot.warmup(fp32, fp32, fp32, *scalars, grid=(programs,), **_narrow_constants(off), num_warps=4)
    msl = warm.asm["msl"]  # compiled, not launched: the very kernel the launch below runs
    kernel = _Emitted(msl)
    pointer, _mask, index = _staged_loads(msl)[0]
    assert pointer == "A"
    for pid in range(programs):
        for lane in range(BM):
            for col in (0, BN - 1):
                element = kernel.value(index, pid=pid, _fill_row=lane, _fill_col=col, sam=BN, x=x, y=y)
                assert -pad * BN <= element < (pad + programs * BM) * BN, (pid, lane, col, element)

    gen = torch.Generator().manual_seed(11)
    a_buf = torch.randn(2 * pad + programs * BM, BN, generator=gen)
    b = torch.randn(BN, BN, generator=gen)
    a_dev, b_dev = a_buf.to("mps"), b.to("mps")
    c_dev = torch.full((programs * BM, BN), -1.0, device="mps")
    a_view = a_dev[pad : pad + programs * BM]
    assert (a_view.stride(0), b_dev.stride(0), c_dev.stride(0)) == (BN, BN, BN)
    handle = _narrow_dot[(programs,)](a_view, b_dev, c_dev, *scalars, **_narrow_constants(off), num_warps=4)
    torch.mps.synchronize()
    assert handle.asm["msl"] == msl, "the launch ran another kernel than the one whose address was evaluated"

    reference = torch.empty(programs * BM, BN, dtype=torch.float64)
    for out_row in range(programs * BM):
        row = out_row + offset
        a_row = a_buf[pad + row].double() if row < m_bound else torch.zeros(BN, dtype=torch.float64)
        reference[out_row] = torch.exp(0.25 * (a_row @ b.double()))
    torch.testing.assert_close(c_dev.cpu().double(), reference, rtol=2e-4, atol=1e-5)
    launched = _staged_loads(handle.asm["msl"])[0]
    assert launched[0] == "A" and "_fill_row" in launched[2]
    # The narrow sum reached the fill as a narrow sum (not folded to one scalar beforehand).
    assert re.search(r"\((?:u?char|u?short)\)\(", launched[2]), launched[2]
