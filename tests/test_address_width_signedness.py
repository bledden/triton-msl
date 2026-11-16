"""Address-arithmetic WIDTH and SIGNEDNESS in the matmul templates (packet 825, W2 F-7).

The strided matmul template rendered every runtime stride as ``(uint)s`` and indexed
with ``uint`` variables, so an emitted address

    A[m * (uint)sam + k * (uint)sak]

(a) truncated an i64 stride to 32 bits, (b) evaluated the sum as UNSIGNED, and
(c) zero-extended a negative stride into pointer arithmetic (``A[(uint)(-6)]`` is
``A + 4294967290``, not ``A - 6``). Negative offsets relative to the bound base are
reachable: the driver binds each argument at its view's byte offset inside a
full-storage mirror on the host path.

Every test here is CPU-only: it lowers a real ``@triton.jit`` kernel through the
production admission route (``make_ttir`` -> ``make_ttgir`` -> ``make_msl``) and
asserts on the emitted MSL text. Several also EVALUATE the emitted index expression
under Metal/C++ integer rules and compare it against a Triton-semantics oracle, so
the pins are not merely textual.
"""

import re

import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402
from triton._C.libtriton import ir  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from triton_msl.backend.compiler import MetalBackend  # noqa: E402
from triton_msl.errors import MetalNonRecoverableError  # noqa: E402


def _emit(fn, signature, constants=None):
    """Same production admission route as tests/test_codegen_admission_semantics.py."""
    constants = constants or {}
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(fn=fn, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    emitted = backend.make_msl(module, metadata, options)
    # These tests evaluate address arithmetic, not variable spelling. Validate
    # positional bindings before translating compiler-owned argument identifiers
    # back to fixture names for the existing independent integer evaluator.
    # Shadowing itself is checked on unmodified output in test_matmul_symbol_hygiene.
    runtime = [name for name, dtype in signature.items() if dtype != "constexpr" and name not in constants]
    if "_mmarg0 [[buffer(0)]]" in emitted:
        mapping = {}
        for i, name in enumerate(runtime):
            alias = f"_mmarg{i}"
            pointer = signature[name].startswith("*")
            declared = alias if pointer else alias + "_buf"
            # Fused templates bind scalar references directly.
            if not pointer and f"& {alias} [[buffer({i})]]" in emitted:
                declared = alias
            assert f"{declared} [[buffer({i})]]" in emitted
            mapping[alias] = name
            mapping[alias + "_buf"] = name + "_buf"
        emitted = re.sub(r"\b_mmarg\d+(?:_buf)?\b", lambda match: mapping[match.group()], emitted)
    return emitted


# --------------------------------------------------------------------------
# A tiny evaluator for the emitted index expressions, under Metal/C++ rules.
# --------------------------------------------------------------------------


class _U32(int):
    """32-bit unsigned value; * and + wrap mod 2**32 exactly as MSL's ``uint`` does."""

    def __new__(cls, v):
        return super().__new__(cls, int(v) & 0xFFFFFFFF)

    def __mul__(self, other):
        return _U32(int(self) * int(other))

    __rmul__ = __mul__

    def __add__(self, other):
        return _U32(int(self) + int(other))

    __radd__ = __add__


class _U64(int):
    """64-bit unsigned value; * and + wrap mod 2**64 exactly as MSL's ``ulong`` does."""

    def __new__(cls, v):
        return super().__new__(cls, int(v) & 0xFFFFFFFFFFFFFFFF)

    def __mul__(self, other):
        return _U64(int(self) * int(other))

    __rmul__ = __mul__

    def __add__(self, other):
        return _U64(int(self) + int(other))

    __radd__ = __add__


def _s32(v):
    """``as_type<int>`` — reinterpret the 32 bits as a signed int (then used at 64 bits)."""
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _s64(v):
    """``as_type<long>`` — reinterpret the 64 bits as a signed long."""
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v


def _eval_index(expr, **vals):
    """Evaluate one emitted MSL index expression with the given variable values.

    Models the exact MSL integer semantics the emitter relies on: ``uint``/``ulong``
    arithmetic wraps mod 2**32 / 2**64, ``as_type`` reinterprets those bits as signed,
    and the remaining ``(long)`` conversions are exact (proved in the emitter's
    docstring: the only signed ``+`` left adds two values in [-2**31, 2**31)).
    """
    py = expr.replace("(ulong)as_type<int>(", "_U64_s32(")
    py = py.replace("as_type<long>(", "_s64(").replace("as_type<int>(", "_s32(")
    py = re.sub(r"\(ulong\)\(([^()]*)\)", r"_U64(\1)", py)
    py = re.sub(r"\(ulong\)([A-Za-z_]\w*)", r"_U64(\1)", py)
    py = py.replace("(long)", "")
    py = re.sub(r"\(uint\)\(([^()]*)\)", r"_U32(\1)", py)
    py = re.sub(r"\(uint\)([A-Za-z_]\w*)", r"_U32(\1)", py)
    env = {
        "_U32": _U32,
        "_U64": _U64,
        "_s32": _s32,
        "_s64": _s64,
        "_U64_s32": lambda v: _U64(_s32(v)),
    }
    env.update(vals)
    return int(eval(py, {"__builtins__": {}}, env))  # noqa: S307 - machine-generated text


def _index_expr(msl, ptr, occurrence=0):
    """Inner text of the ``occurrence``-th ``ptr[...]`` subscript in ``msl``."""
    hits = []
    start = 0
    needle = ptr + "["
    while True:
        i = msl.find(needle, start)
        if i < 0:
            break
        j = msl.index("]", i)
        # Whole-identifier match only: ``tg_A[...]`` must not answer for ``A[...]``.
        if i == 0 or not (msl[i - 1].isalnum() or msl[i - 1] == "_"):
            hits.append(msl[i + len(needle) : j])
        start = j
    assert len(hits) > occurrence, f"no {ptr}[...] #{occurrence} in emitted MSL"
    return hits[occurrence]


def _tri_i32_two_links(terms):
    """Triton oracle: one tt.addptr per source ``+``; each i32 term wraps at 32 bits
    and is SIGN-extended into the 64-bit pointer add."""
    return sum(_s32((i * s) & 0xFFFFFFFF) for i, s in terms)


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

_STRIDES = ("sam", "sak", "sbk", "sbn", "scm", "scn")


@triton.jit
def _strided_mm(A, B, C, sam, sak, sbk, sbn, scm, scn):
    """Tutorial-style strided matmul; every inner stride is a runtime arg, so the
    simdgroup path declines and the stride-aware SCALAR template claims it."""
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak)
    b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, tl.dot(a, b))


@triton.jit
def _strided_mm_fused(A, B, C, sam, sak, sbk, sbn, scm, scn):
    """Same addresses, parenthesised: ONE tt.addptr carries the whole i32 sum."""
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    a = tl.load(A + (rm[:, None] * sam + rk[None, :] * sak))
    b = tl.load(B + (rk[:, None] * sbk + rn[None, :] * sbn))
    tl.store(C + (rm[:, None] * scm + rn[None, :] * scn), tl.dot(a, b))


@triton.jit
def _tutorial_mm(A, B, C, M, N, K, sam, sbk, scm, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Contiguous inner dims + runtime ROW strides: the simdgroup K-loop template."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    a_ptr = A + rm[:, None] * sam + rk[None, :]
    b_ptr = B + rk[:, None] * sbk + rn[None, :]
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptr), tl.load(b_ptr))
        a_ptr += BK
        b_ptr += BK * sbk
    tl.store(C + rm[:, None] * scm + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _row_sort_baked(OUT, INP, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    tl.store(OUT + i[:, None] * N + j[None, :], tl.sort(tl.load(INP + i[:, None] * N + j[None, :])))


def _strided_msl(stride_ty="i32", **overrides):
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: overrides.get(n, stride_ty) for n in _STRIDES})
    return _emit(_strided_mm, sig)


# --------------------------------------------------------------------------
# i32: 32-bit wrapping product, signed 32-bit index, 64-bit pointer add
# --------------------------------------------------------------------------


def test_i32_strides_emit_signed_32bit_index_not_unsigned():
    msl = _strided_msl("i32")
    assert "_sum +=" in msl, "kernel must still be admitted to the strided scalar template"
    a = _index_expr(msl, "A")
    assert a == "(long)as_type<int>((uint)m * (uint)(sam)) + (long)as_type<int>((uint)k * (uint)(sak))", a
    # The defect form is gone everywhere in the kernel.
    assert "m * (uint)sam" not in msl and "(uint)sam +" not in msl
    for s in _STRIDES:
        assert f"* (uint){s}" not in msl, f"bare unsigned stride {s} still multiplies an index"


def test_i32_negative_stride_indexes_backwards_not_four_billion_forwards():
    """The bound pointer sits inside a full-storage mirror, so a negative kernel
    offset is a real address. ``(uint)(-6)`` used to make it ``base + 4294967290``."""
    a = _index_expr(_strided_msl("i32"), "A")
    got = _eval_index(a, m=1, k=0, sam=-6, sak=1)
    assert got == -6, got
    assert _eval_index(a, m=3, k=2, sam=-6, sak=-1) == _tri_i32_two_links(((3, -6), (2, -1))) == -20


def test_i32_arithmetic_still_wraps_at_32_bits_per_addptr_link():
    """A blanket ``long`` cast would be WRONG: an i32 product must wrap at 32 bits
    and only then sign-extend."""
    a = _index_expr(_strided_msl("i32"), "A")
    m, sam = 3, 0x4000_0000  # 3 * 2**30 = 0xC000_0000 -> negative as i32
    assert _eval_index(a, m=m, k=0, sam=sam, sak=1) == _tri_i32_two_links(((m, sam), (0, 1)))
    assert _eval_index(a, m=m, k=0, sam=sam, sak=1) == -0x4000_0000


def test_i32_separate_addptr_links_are_extended_independently():
    """``A + x + y`` is two tt.addptr ops: each i32 term wraps and sign-extends on its
    own, and the two are added at 64 bits (no second 32-bit wrap)."""
    a = _index_expr(_strided_msl("i32"), "A")
    assert a.count("as_type<int>") == 2
    terms = ((1, 0x7FFF_FFFF), (1, 0x7FFF_FFFF))
    assert _eval_index(a, m=1, k=1, sam=0x7FFF_FFFF, sak=0x7FFF_FFFF) == _tri_i32_two_links(terms)
    assert _eval_index(a, m=1, k=1, sam=0x7FFF_FFFF, sak=0x7FFF_FFFF) == 2 * 0x7FFF_FFFF


def test_i32_single_addptr_link_sums_before_the_sign_extend():
    """Parenthesised in the source, the whole offset is ONE i32 addptr: the sum wraps
    at 32 bits before the extend, so the emission must not split it."""
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in _STRIDES})
    a = _index_expr(_emit(_strided_mm_fused, sig), "A")
    assert a == "(long)as_type<int>((uint)m * (uint)(sam) + (uint)k * (uint)(sak))", a
    assert a.count("as_type<int>") == 1
    got = _eval_index(a, m=1, k=1, sam=0x7FFF_FFFF, sak=0x7FFF_FFFF)
    assert got == _s32((0x7FFF_FFFF + 0x7FFF_FFFF) & 0xFFFFFFFF) == -2


# --------------------------------------------------------------------------
# i64: promoted, full-width arithmetic
# --------------------------------------------------------------------------


def test_i64_strides_emit_64bit_arithmetic_not_a_truncated_uint():
    msl = _strided_msl("i64")
    a = _index_expr(msl, "A")
    assert a == "as_type<long>((ulong)m * (ulong)(sam) + (ulong)k * (ulong)(sak))", a
    assert "(uint)sam" not in msl and "as_type<int>" not in a
    assert "long sam = sam_buf[0];" in msl


def test_i64_stride_beyond_32_bits_is_not_truncated():
    """The value needs more than 32 bits: the old ``(uint)`` cast dropped the top half."""
    a = _index_expr(_strided_msl("i64"), "A")
    sam = 5_000_000_000
    assert _eval_index(a, m=3, k=0, sam=sam, sak=1) == 3 * sam
    assert (3 * sam) & 0xFFFFFFFF != 3 * sam  # the truncation this pins would be visible


def test_i64_negative_stride_indexes_backwards():
    a = _index_expr(_strided_msl("i64"), "A")
    assert _eval_index(a, m=2, k=1, sam=-(1 << 33), sak=-7) == -(1 << 34) - 7


def test_i64_products_and_sums_of_two_runtime_strides_stay_exact():
    b = _index_expr(_strided_msl("i64"), "B")
    assert b == "as_type<long>((ulong)k * (ulong)(sbk) + (ulong)n * (ulong)(sbn))", b
    assert _eval_index(b, k=7, n=9, sbk=3_000_000_000, sbn=-4_000_000_000) == 7 * 3_000_000_000 - 9 * 4_000_000_000


# --------------------------------------------------------------------------
# Mixed and unprovable widths
# --------------------------------------------------------------------------


def test_mixed_i32_i64_across_separate_addptr_links_is_emitted_per_term():
    """Each ``tt.addptr`` link is extended on its own, so a row term at i64 and a col
    term at i32 are each rendered at THEIR width. The links are attributed by the
    ``tt.expand_dims`` axis reachable in each offset (index ssa ids cannot be used —
    one reused ``tl.arange`` CSEs to a single ``tt.make_range`` for both axes)."""
    a = _index_expr(_strided_msl("i32", sak="i64"), "A")
    assert a == "as_type<long>((ulong)as_type<int>((uint)m * (uint)(sam)) + (ulong)k * (ulong)(sak))", a
    assert _eval_index(a, m=2, k=3, sam=-5, sak=5_000_000_000) == -10 + 3 * 5_000_000_000


def test_mixed_i32_i64_inside_ONE_addptr_link_refuses():
    """Parenthesised and mixed, Triton promotes the i32 product with an ``arith.extsi``
    INSIDE an i64 ``arith.addi``: that product still wraps at 32 bits, which a
    single-width fused model would not reproduce. Must refuse, not guess."""
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in _STRIDES})
    sig["sak"] = "i64"
    with pytest.raises(MetalNonRecoverableError, match="integer width of the operand address arithmetic"):
        _emit(_strided_mm_fused, sig)


def test_mixed_widths_across_different_operands_are_each_emitted_at_their_own_width():
    """A is i64 and C is i32: per-ROLE proof, not a whole-kernel guess."""
    msl = _strided_msl("i32", sam="i64", sak="i64")
    assert _index_expr(msl, "A") == "as_type<long>((ulong)m * (ulong)(sam) + (ulong)k * (ulong)(sak))"
    assert _index_expr(msl, "C") == (
        "(long)as_type<int>((uint)m * (uint)(scm)) + (long)as_type<int>((uint)n * (uint)(scn))"
    )


def test_unsigned_declared_strides_refuse():
    """``tl.uint32`` reaches the walker as signless ``i32``; the unsignedness shows up
    as ``arith.extui`` on the way to the pointer, which nothing here reproduces."""
    with pytest.raises(MetalNonRecoverableError):
        _strided_msl("u32")


def test_unsigned_extension_is_not_accepted_as_a_proven_width():
    """Defence in depth for the above: the width prover itself must reject an
    ``arith.extui`` link even if the stride tracer ever learns to classify one."""
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "u32" for n in _STRIDES})
    source = ASTSource(fn=_strided_mm, signature=sig, constexprs={})
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    lw = generic_lowerer.GenericLowerer(walk_ttgir(module, options), options)
    assert "arith.extui" in lw.graph.mod_text, "the u32 kernel must actually contain extui"
    assert lw._dot_offset_arith_widths() is None


# --------------------------------------------------------------------------
# Extents: signed, full width
# --------------------------------------------------------------------------


def test_extents_are_signed_and_full_width_so_a_negative_extent_masks_everything():
    msl = _strided_msl("i32")
    assert "long _M = (long)32;" in msl and "long _N = (long)32;" in msl
    assert "uint _M" not in msl and "uint _N" not in msl and "uint _K" not in msl
    assert "if (m >= _M || n >= _N) continue;" in msl
    # uint index vs long extent promotes the index: the comparison is SIGNED, so a
    # negative extent excludes every row instead of admitting all of them.
    assert (0 >= -1) is True and (0 >= (0xFFFFFFFF)) is False


def test_runtime_extents_sign_extend():
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in ("M", "N", "K")})
    sig.update({n: "i32" for n in ("sam", "sbk", "scm")})
    consts = {"BM": 32, "BN": 32, "BK": 32}
    sig.update({k: "constexpr" for k in consts})
    msl = _emit(_tutorial_mm, sig, consts)
    assert "long _M = (long)M;" in msl and "long _N = (long)N;" in msl and "long _K = (long)K;" in msl
    assert "uint _M = (uint)M" not in msl


def test_pid_mapping_still_replays_the_source_32bit_tile_arithmetic():
    """``_M`` is now 64-bit, so the grid mapping narrows explicitly — the bits it
    computes on are identical to the old ``uint _M``."""
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in ("M", "N", "K")})
    sig.update({n: "i32" for n in ("sam", "sbk", "scm")})
    consts = {"BM": 32, "BN": 32, "BK": 32}
    sig.update({k: "constexpr" for k in consts})
    msl = _emit(_tutorial_mm, sig, consts)
    assert "long _M = (long)M;" in msl
    assert "as_type<int>(_M" not in msl and "(_M + 32u" not in msl


# --------------------------------------------------------------------------
# Currently-correct cases are preserved
# --------------------------------------------------------------------------


def test_canonical_i32_simdgroup_matmul_address_text_is_unchanged():
    """The dense/contiguous i32 fast path is the common case: its emitted addresses
    must be byte-identical to before the repair (only the extent DECLARATIONS move)."""
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in ("M", "N", "K")})
    sig.update({n: "i32" for n in ("sam", "sbk", "scm")})
    consts = {"BM": 32, "BN": 32, "BK": 32}
    sig.update({k: "constexpr" for k in consts})
    msl = _emit(_tutorial_mm, sig, consts)
    assert "simdgroup_multiply_accumulate" in msl
    assert "simdgroup_load(a_frag, A + (row_base + 0u) * (uint)sam + k, (uint)sam);" in msl
    assert "A[gr * (uint)sam + gc]" in msl


def test_i64_row_stride_on_the_simdgroup_path_is_not_truncated():
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in ("M", "N", "K")})
    sig.update({n: "i64" for n in ("sam", "sbk", "scm")})
    consts = {"BM": 32, "BN": 32, "BK": 32}
    sig.update({k: "constexpr" for k in consts})
    msl = _emit(_tutorial_mm, sig, consts)
    assert "(uint)sam" not in msl and "(uint)sbk" not in msl and "(uint)scm" not in msl
    assert "simdgroup_load(a_frag, A + (row_base + 0u) * (ulong)sam + k, (ulong)sam);" in msl
    assert "A[gr * (ulong)sam + gc]" in msl


# --------------------------------------------------------------------------
# Row-per-thread (row-wise sort / topk) template
# --------------------------------------------------------------------------


def test_row_sort_baked_stride_offset_is_widened():
    msl = _emit(
        _row_sort_baked, {"OUT": "*fp32", "INP": "*fp32", "M": "constexpr", "N": "constexpr"}, {"M": 2, "N": 16}
    )
    assert "Bitonic sort of 16" in msl, "row-sort template did not claim the kernel"
    assert "long _row_off = (long)lid * (long)(16);" in msl
    assert "long _row_off_z = (long)lid * (long)(16);" in msl
    assert "(uint)16" not in msl


def test_row_sort_runtime_stride_refuses_because_its_width_is_unproven():
    """The detector's runtime-stride branch is currently unreachable (it needs the
    stride argument DIRECTLY inside an ``arith.muli``, but Triton always ``tt.splat``s
    a scalar first), so this exercises the template helper on a real lowerer."""
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)

    @triton.jit
    def _rs(X, sxm, Z, szm, M: tl.constexpr, N: tl.constexpr):
        i = tl.arange(0, M)
        j = tl.arange(0, N)
        tl.store(Z + i[:, None] * szm + j[None, :], tl.sort(tl.load(X + i[:, None] * sxm + j[None, :])))

    sig = {"X": "*fp32", "sxm": "i32", "Z": "*fp32", "szm": "i32", "M": "constexpr", "N": "constexpr"}
    consts = {"M": 2, "N": 16}
    source = ASTSource(fn=_rs, signature=sig, constexprs=consts)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    lw = generic_lowerer.GenericLowerer(walk_ttgir(module, options), options)
    assert lw._detect_row_wise_sort() is None, "runtime-stride branch became reachable; wire a width proof"
    with pytest.raises(MetalNonRecoverableError, match="cannot prove the integer width"):
        lw._row_stride_offset_expr("lid", "sxm", "row-wise sort input")


# --------------------------------------------------------------------------
# Fused matmul+softmax / matmul+epilogue template (packet 825 extension)
#
# `_lower_matmul_softmax_template._addr` used to build `f"({row}) * {row_stride}"`
# with NO cast. With an i32 stride (`constant int&`) and a `uint` row index that is
# a 32-bit UNSIGNED product, zero-extended at the subscript: a negative stride
# addressed ~4 GiB forward and a product past 2**31 lost its sign. (It did not
# truncate an i64 stride — `uint * long` promotes — but that was luck.)
# --------------------------------------------------------------------------


@triton.jit
def _fused_softmax_s(A, B, C, sam, sbk, scm):
    """Fused matmul+softmax with DEDICATED runtime row strides (contiguous inner)."""
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    z = tl.dot(tl.load(A + rm[:, None] * sam + rk[None, :]), tl.load(B + rk[:, None] * sbk + rn[None, :]))
    m = tl.max(z, 1)
    e = tl.exp(z - m[:, None])
    s = tl.sum(e, 1)
    tl.store(C + rm[:, None] * scm + rn[None, :], e / s[:, None])


@triton.jit
def _fused_relu_s(A, B, C, sam, sbk, scm):
    """Fused matmul+ReLU epilogue with dedicated runtime row strides."""
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    z = tl.dot(tl.load(A + rm[:, None] * sam + rk[None, :]), tl.load(B + rk[:, None] * sbk + rn[None, :]))
    tl.store(C + rm[:, None] * scm + rn[None, :], tl.maximum(z, 0.0))


def _fused_msl(fn, stride_ty="i32"):
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: stride_ty for n in ("sam", "sbk", "scm")})
    return _emit(fn, sig)


@pytest.mark.parametrize("fn,route", [(_fused_softmax_s, "softmax"), (_fused_relu_s, "epilogue")])
def test_fused_template_i32_row_stride_is_signed_not_unsigned(fn, route):
    msl = _fused_msl(fn, "i32")
    assert "tg_C" in msl, f"{route} route did not claim the kernel"
    a = _index_expr(msl, "A")
    assert a == (
        "(long)as_type<int>((uint)(mstrip + r) * (uint)(sam)) + (long)as_type<int>((uint)(kk + c) * (uint)(1))"
    ), a
    # The uncast product is gone from every address in the kernel.
    assert "(mstrip + r) * sam" not in msl and "(kk + r) * sbk" not in msl
    assert "(mstrip + row) * scm" not in msl


@pytest.mark.parametrize("fn,route", [(_fused_softmax_s, "softmax"), (_fused_relu_s, "epilogue")])
def test_fused_template_negative_i32_row_stride_indexes_backwards(fn, route):
    msl = _fused_msl(fn, "i32")
    a = _index_expr(msl, "A")
    # row 2 of a backwards-striding A, column 3, unit inner stride.
    got = _eval_index(a, mstrip=0, r=2, kk=0, c=3, sam=-64)
    assert got == _tri_i32_two_links(((2, -64), (3, 1))) == -125, got


@pytest.mark.parametrize("fn,route", [(_fused_softmax_s, "softmax"), (_fused_relu_s, "epilogue")])
def test_fused_template_i32_product_wraps_at_32_bits(fn, route):
    a = _index_expr(_fused_msl(fn, "i32"), "A")
    sam = 0x4000_0000
    assert _eval_index(a, mstrip=0, r=3, kk=0, c=0, sam=sam) == _tri_i32_two_links(((3, sam), (0, 1)))
    assert _eval_index(a, mstrip=0, r=3, kk=0, c=0, sam=sam) == -0x4000_0000


@pytest.mark.parametrize("fn,route", [(_fused_softmax_s, "softmax"), (_fused_relu_s, "epilogue")])
def test_fused_template_i64_row_stride_is_not_truncated(fn, route):
    """The row term is i64 and the folded unit col term stays i32 — two links, two
    widths, each rendered at its own."""
    msl = _fused_msl(fn, "i64")
    a = _index_expr(msl, "A")
    assert a == (
        "as_type<long>((ulong)(mstrip + r) * (ulong)(sam) + (ulong)as_type<int>((uint)(kk + c) * (uint)(1)))"
    ), a
    sam = 5_000_000_000
    assert _eval_index(a, mstrip=0, r=3, kk=0, c=7, sam=sam) == 3 * sam + 7
    assert "long sam = " in msl or "constant long& sam" in msl


@pytest.mark.parametrize("fn,route", [(_fused_softmax_s, "softmax"), (_fused_relu_s, "epilogue")])
def test_fused_template_canonical_positive_values_unchanged(fn, route):
    """Same numbers as the old uncast form for every positive in-range stride."""
    a = _index_expr(_fused_msl(fn, "i32"), "A")
    b = _index_expr(_fused_msl(fn, "i32"), "B")
    for mstrip, r, kk, c, sam in ((0, 0, 0, 0, 64), (32, 5, 16, 3, 64), (0, 31, 24, 7, 4096)):
        assert _eval_index(a, mstrip=mstrip, r=r, kk=kk, c=c, sam=sam) == (mstrip + r) * sam + (kk + c)
    for kk, r, c, sbk in ((0, 0, 0, 64), (16, 3, 9, 64), (24, 7, 31, 4096)):
        assert _eval_index(b, kk=kk, r=r, c=c, sbk=sbk) == (kk + r) * sbk + c


def _corpus_module():
    """Import the sibling admission-semantics test module BY PATH.

    Works whether or not ``tests/`` (or the repo root) is on ``sys.path`` — the
    evidence records runs under both PYTHONPATH conventions."""
    import importlib.util
    import os
    import sys

    name = "test_codegen_admission_semantics"
    if name in sys.modules:
        return sys.modules[name]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fused_corpus_kernel_still_admitted_with_its_new_address_text():
    """The admission-semantics corpus kernel (`_fused`, strides = the runtime dim
    args) stays on the template; this records the address text it now emits."""
    corpus = _corpus_module()

    msl = corpus._fused_msl()
    assert "simdgroup_multiply_accumulate" in msl
    assert (
        "tg_A[i] = float(_mmarg0[(long)as_type<int>((uint)(mstrip + r) * (uint)(_mmarg6))"
        " + (long)as_type<int>((uint)(kk + c) * (uint)(1))]);"
    ) in msl
    assert (
        "tg_B[i] = float(_mmarg1[(long)as_type<int>((uint)(kk + r) * (uint)(_mmarg5)) + (long)as_type<int>((uint)c * (uint)(1))]);"
    ) in msl
    assert (
        "_mmarg3[(long)as_type<int>((uint)(mstrip + row) * (uint)(_mmarg5))"
        " + (long)as_type<int>((uint)col * (uint)(1))] = (float)tg_C[i];"
    ) in msl


def test_fused_template_baked_constexpr_dims_still_admitted():
    """The constexpr-dim ReLU epilogue (tests/test_matmul_epilogue.py::_mm_relu shape)
    keeps its literal strides, now widened through the same prover."""

    @triton.jit
    def _mm_relu(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.maximum(tl.dot(a, b), 0.0))

    sig = {n: "*fp32" for n in ("A", "B", "C")}
    consts = {"M": 32, "N": 32, "K": 32}
    sig.update({k: "constexpr" for k in consts})
    msl = _emit(_mm_relu, sig, consts)
    a = _index_expr(msl, "A")
    assert a == (
        "(long)as_type<int>((uint)(mstrip + r) * (uint)(32)) + (long)as_type<int>((uint)(kk + c) * (uint)(1))"
    ), a
    assert _eval_index(a, mstrip=0, r=5, kk=8, c=3) == 5 * 32 + 11


# ==========================================================================
# Packet 834 task 2 — i64 arithmetic must be DEFINED, not signed-overflow UB
#
# `(long)a * (long)b` is UB on overflow in MSL/C++: the compiler may assume it
# never happens, while Triton's `arith.muli` wraps mod 2**64. Every i64 step is
# now unsigned (`ulong`, defined mod 2**64) with one `as_type<long>` bit-cast.
# ==========================================================================

_M64 = (1 << 64) - 1


def _tri_i64_one_link(terms):
    """Triton oracle, ONE addptr link at i64: the sum wraps mod 2**64, then the
    pointer add is also mod 2**64 — so the link's signed value is the offset."""
    return _s64(sum(i * st for i, st in terms) & _M64)


def _tri_i64_two_links(terms):
    """Triton oracle, one addptr per term: each term wraps mod 2**64 on its own and
    the pointer additions are mod 2**64."""
    return _s64(sum(_s64((i * st) & _M64) for i, st in terms) & _M64)


def test_i64_addresses_use_defined_unsigned_arithmetic_everywhere():
    """No emitted i64 address may contain a signed 64-bit product."""
    for msl in (_strided_msl("i64"), _fused_msl(_fused_softmax_s, "i64")):
        assert "as_type<long>((ulong)" in msl
        assert re.search(r"\(long\)[A-Za-z_(][^*+;]*\* \(long\)", msl) is None, (
            "a signed 64-bit product survived; that is UB on overflow"
        )


def test_i64_single_link_terms_that_cancel_mod_2_64_give_a_small_offset():
    """Two huge i64 terms in ONE addptr link cancel mod 2**64 to an in-allocation
    offset. Signed arithmetic would have overflowed (UB); the unsigned form is exact."""
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i64" for n in _STRIDES})
    a = _index_expr(_emit(_strided_mm_fused, sig), "A")
    assert a == "as_type<long>((ulong)m * (ulong)(sam) + (ulong)k * (ulong)(sak))", a
    cases = [
        ((1, 1 << 63), (1, 1 << 63)),  # 2**63 + 2**63 == 2**64 == 0
        ((3, (1 << 63) + 7), (1, 1 << 63)),  # 3*(2**63+7) + 2**63 == 21 (mod 2**64)
        ((2, -(1 << 62)), (2, 1 << 62)),  # exact cancellation of huge opposites
    ]
    for (m, sam), (k, sak) in cases:
        want = _tri_i64_one_link(((m, sam), (k, sak)))
        assert _eval_index(a, m=m, k=k, sam=sam, sak=sak) == want, (m, sam, k, sak)
    assert _tri_i64_one_link(((1, 1 << 63), (1, 1 << 63))) == 0


def test_i64_separate_links_wrap_per_link_like_triton():
    """Each link wraps mod 2**64 on its own before the pointer add."""
    a = _index_expr(_strided_msl("i64"), "A")
    for m, sam, k, sak in (
        (3, (1 << 62) + 5, 2, -(1 << 62)),
        (5, -(1 << 63), 5, 1 << 63),
        (7, 6_000_000_000, 9, -11),
    ):
        want = _tri_i64_two_links(((m, sam), (k, sak)))
        assert _eval_index(a, m=m, k=k, sam=sam, sak=sak) == want, (m, sam, k, sak)


def test_i32_fused_link_sum_wraps_at_32_bits_then_sign_extends():
    """The i32 fused-link form was already defined (`uint` sum then `as_type<int>`);
    pin it with two terms that cancel mod 2**32."""
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in _STRIDES})
    a = _index_expr(_emit(_strided_mm_fused, sig), "A")
    assert a == "(long)as_type<int>((uint)m * (uint)(sam) + (uint)k * (uint)(sak))", a
    for m, sam, k, sak in ((1, 1 << 31, 1, 1 << 31), (3, 0x5555_5555, 1, 0x4000_0001), (2, -7, 3, 5)):
        want = _s32((m * sam + k * sak) & 0xFFFFFFFF)
        assert _eval_index(a, m=m, k=k, sam=sam, sak=sak) == want, (m, sam, k, sak)
    assert _s32(((1 << 31) + (1 << 31)) & 0xFFFFFFFF) == 0


def test_fused_template_single_link_i32_sum_wraps_before_the_extend():
    """Same proof on the fused matmul+softmax route."""

    @triton.jit
    def _fs(A, B, C, sam, sbk, scm):
        rm = tl.arange(0, 32)
        rn = tl.arange(0, 32)
        rk = tl.arange(0, 32)
        z = tl.dot(tl.load(A + (rm[:, None] * sam + rk[None, :])), tl.load(B + (rk[:, None] * sbk + rn[None, :])))
        m = tl.max(z, 1)
        e = tl.exp(z - m[:, None])
        s = tl.sum(e, 1)
        tl.store(C + (rm[:, None] * scm + rn[None, :]), e / s[:, None])

    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in ("sam", "sbk", "scm")})
    a = _index_expr(_emit(_fs, sig), "A")
    assert a == "(long)as_type<int>((uint)(mstrip + r) * (uint)(sam) + (uint)(kk + c) * (uint)(1))", a
    for r, c, sam in ((1, 0, 1 << 31), (3, 5, 0x5555_5555), (2, 7, -9)):
        want = _s32((r * sam + c) & 0xFFFFFFFF)
        assert _eval_index(a, mstrip=0, r=r, kk=0, c=c, sam=sam) == want, (r, c, sam)
    assert _s32((1 * (1 << 31)) & 0xFFFFFFFF) == -(1 << 31)


def test_fused_template_single_link_mixed_width_refuses():
    """`A + (rm*sam_i64 + rk)` promotes the i32 index term with an `arith.extsi`
    INSIDE the i64 `addi`. The single-width fused model cannot reproduce that, so it
    refuses rather than emit an address that does not wrap where the source does."""

    @triton.jit
    def _fs64(A, B, C, sam, sbk, scm):
        rm = tl.arange(0, 32)
        rn = tl.arange(0, 32)
        rk = tl.arange(0, 32)
        z = tl.dot(tl.load(A + (rm[:, None] * sam + rk[None, :])), tl.load(B + (rk[:, None] * sbk + rn[None, :])))
        m = tl.max(z, 1)
        e = tl.exp(z - m[:, None])
        s = tl.sum(e, 1)
        tl.store(C + (rm[:, None] * scm + rn[None, :]), e / s[:, None])

    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i64" for n in ("sam", "sbk", "scm")})
    with pytest.raises(MetalNonRecoverableError, match="integer width of the operand address arithmetic"):
        _emit(_fs64, sig)


# ==========================================================================
# Packet 834 task 1 — simdgroup leading-dimension ABI proof + scalar fallback
# ==========================================================================


def _kernels(msl):
    """Split emitted MSL into {kernel name: body text}."""
    out = {}
    parts = msl.split("kernel void ")
    for part in parts[1:]:
        name = part.split("(", 1)[0].strip()
        out[name] = part
    return out


def _guard_expr(body):
    """The text of ``bool _simd_ok = ...;`` in one kernel body."""
    i = body.index("bool _simd_ok =")
    return body[i : body.index(";", i) + 1]


def _eval_guard(body, *, M, N, K, lda, ldb, ldc):
    """Evaluate the emitted guard under C semantics.

    ``lda``/``ldb``/``ldc`` are the ULONG values the kernel would see (that is what
    ``(ulong)(uint)s`` produces for a negative i32 stride). Python integers are
    unbounded, which is exactly the point: the emitted 64-bit check is claimed not
    to overflow, so evaluating it exactly must give the same verdict."""
    py = _guard_expr(body)
    py = py[py.index("=") + 1 :].rstrip(";")
    py = " ".join(py.split())  # the emitted guard spans several indented lines
    py = py.replace("&&", "and").replace("(long)", "")
    py = re.sub(r"(0x[0-9a-fA-F]+)(UL|L)\b", r"\1", py)
    py = re.sub(r"\b(\d+)L\b", r"\1", py)
    env = {"_M": M, "_N": N, "_K": K, "_lda": lda, "_ldb": ldb, "_ldc": ldc}
    return bool(eval(py, {"__builtins__": {}}, env))  # noqa: S307 - machine-generated text


def _kloop_msl(stride_ty="i32"):
    sig = {n: "*fp32" for n in ("A", "B", "C")}
    sig.update({n: "i32" for n in ("M", "N", "K")})
    sig.update({n: stride_ty for n in ("sam", "sbk", "scm")})
    consts = {"BM": 32, "BN": 32, "BK": 32}
    sig.update({k: "constexpr" for k in consts})
    return _emit(_tutorial_mm, sig, consts)


def test_kloop_emits_the_uniform_layout_guard_in_both_kernels():
    ks = _kernels(_kloop_msl("i32"))
    assert "_tutorial_mm" in ks and "_tutorial_mm__mmdirect" in ks
    for name, body in ks.items():
        g = _guard_expr(body)
        for cond in (
            "_M > 0L && _N > 0L && _K > 0L",
            "_lda <= 0x7fffffffUL && _ldb <= 0x7fffffffUL && _ldc <= 0x7fffffffUL",
            "(_M - 1L) * (long)_lda + (_K - 1L) < 0x80000000L",
            "(_K - 1L) * (long)_ldb + (_N - 1L) < 0x80000000L",
            "(_M - 1L) * (long)_ldc + (_N - 1L) < 0x80000000L",
        ):
            assert cond in g, (name, cond, g)


def test_layout_guard_is_uniform_across_the_threadgroup():
    """The verdict must read only kernel scalars: a per-thread or per-program term
    would let some threads return before a barrier that the others reach."""
    for body in _kernels(_kloop_msl("i32")).values():
        decl = body[body.index("ulong _lda") : body.index("bool _simd_ok =")] + _guard_expr(body)
        for forbidden in ("tiitg", "sgitg", "pid3", "pid_m", "pid_n", "row_base", "col_base", "laneid"):
            assert forbidden not in decl, (forbidden, decl)


def test_layout_guard_admits_the_canonical_dense_layout():
    body = _kernels(_kloop_msl("i32"))["_tutorial_mm"]
    assert _eval_guard(body, M=1024, N=1024, K=1024, lda=1024, ldb=1024, ldc=1024)
    assert _eval_guard(body, M=32, N=32, K=32, lda=32, ldb=32, ldc=32)


def test_layout_guard_rejects_a_negative_i32_row_stride():
    """`(ulong)(uint)(-64)` is 4294967232, past the 2**31-1 leading-dim limit."""
    body = _kernels(_kloop_msl("i32"))["_tutorial_mm"]
    assert not _eval_guard(body, M=64, N=64, K=64, lda=(-64) & 0xFFFFFFFF, ldb=64, ldc=64)
    assert not _eval_guard(body, M=64, N=64, K=64, lda=64, ldb=(-64) & 0xFFFFFFFF, ldc=64)
    assert not _eval_guard(body, M=64, N=64, K=64, lda=64, ldb=64, ldc=(-64) & 0xFFFFFFFF)


def test_layout_guard_rejects_a_positive_stride_whose_product_crosses_2_31():
    """A positive stride is NOT by itself proof: `(M-1)*lda` can still pass the signed
    boundary, after which the source's i32 arithmetic goes negative while
    `simdgroup_load` keeps walking forward."""
    body = _kernels(_kloop_msl("i32"))["_tutorial_mm"]
    M, lda = 1 << 20, 1 << 12  # (2**20-1) * 2**12 == 4294963200 >= 2**31
    assert (M - 1) * lda >= (1 << 31)
    assert not _eval_guard(body, M=M, N=64, K=64, lda=lda, ldb=64, ldc=64)
    # ... and admits the same shape once the product fits.
    assert _eval_guard(body, M=M, N=64, K=64, lda=1 << 10, ldb=64, ldc=64)


def test_layout_guard_rejects_64bit_and_negative_extents():
    body = _kernels(_kloop_msl("i64"))["_tutorial_mm"]
    assert not _eval_guard(body, M=64, N=64, K=64, lda=1 << 32, ldb=64, ldc=64)
    assert not _eval_guard(body, M=-1, N=64, K=64, lda=64, ldb=64, ldc=64)
    assert not _eval_guard(body, M=64, N=64, K=0, lda=64, ldb=64, ldc=64)
    assert not _eval_guard(body, M=(1 << 31), N=64, K=64, lda=64, ldb=64, ldc=64)
    assert _eval_guard(body, M=64, N=64, K=64, lda=64, ldb=64, ldc=64)


def test_layout_guard_bound_cannot_overflow_its_own_64bit_check():
    """Under conditions 1-3 every product in condition 4 is at most (2**31-1)**2."""
    assert (0x7FFFFFFF) ** 2 + 0x7FFFFFFF < (1 << 63)


@pytest.mark.parametrize("kernel", ["_tutorial_mm", "_tutorial_mm__mmdirect"])
def test_kloop_scalar_fallback_shape(kernel):
    body = _kernels(_kloop_msl("i32"))[kernel]
    i = body.index("if (!_simd_ok) {")
    fb = body[i : body.index("return;", i)]
    assert "for (uint _fe = tiitg; _fe < 1024u; _fe += 128u) {" in fb
    assert "uint m = row_base + _fe / 32u;" in fb
    assert "uint n = col_base + _fe % 32u;" in fb
    # signed, full-width tile masks (a negative extent masks everything)
    assert "if ((long)m >= _M || (long)n >= _N) continue;" in fb
    # full K range: no 8-multiple assumption, so a K tail is covered
    assert "for (long k = 0L; k < _K; k++) {" in fb
    assert "% 8" not in fb
    # fp32 accumulator and the terminal output cast
    assert "float _facc = 0.0f;" in fb and "= (float)_facc;" in fb
    # width-faithful addresses
    assert "A[(long)as_type<int>((uint)m * (uint)(sam)) + (long)as_type<int>((uint)k * (uint)(1))]" in fb
    assert "B[(long)as_type<int>((uint)k * (uint)(sbk)) + (long)as_type<int>((uint)n * (uint)(1))]" in fb
    assert "C[(long)as_type<int>((uint)m * (uint)(scm)) + (long)as_type<int>((uint)n * (uint)(1))]" in fb


def test_kloop_fallback_addresses_are_width_faithful_for_i64_strides():
    body = _kernels(_kloop_msl("i64"))["_tutorial_mm"]
    i = body.index("if (!_simd_ok) {")
    fb = body[i : body.index("return;", i)]
    assert "as_type<long>((ulong)m * (ulong)(sam) + (ulong)as_type<int>((uint)k * (uint)(1)))" in fb
    a = "as_type<long>((ulong)m * (ulong)(sam) + (ulong)as_type<int>((uint)k * (uint)(1)))"
    assert _eval_index(a, m=3, k=7, sam=-(1 << 33)) == -3 * (1 << 33) + 7
    assert _eval_index(a, m=2, k=0, sam=5_000_000_000) == 10_000_000_000


def test_mmdirect_fallback_adds_no_threadgroup_memory():
    """The direct twin exists to keep a zero-threadgroup footprint (Metal caps
    occupancy by a kernel's compile-time threadgroup usage), so its fallback must be
    scalar, not staged."""
    body = _kernels(_kloop_msl("i32"))["_tutorial_mm__mmdirect"]
    code = [ln for ln in body.splitlines() if not ln.strip().startswith("//")]
    assert not any(ln.strip().startswith("threadgroup ") for ln in code), code
    assert "if (!_simd_ok) {" in body


def test_direct_fast_path_entry_condition_is_unchanged_under_the_guard():
    """The existing tile/K-multiple predicate is untouched; the layout proof is an
    additional precondition reached before it."""
    body = _kernels(_kloop_msl("i32"))["_tutorial_mm"]
    assert "if (row_base + 32u <= _M && col_base + 32u <= _N && (_K % 8u) == 0u) {" in body
    assert body.index("bool _simd_ok =") < body.index("if (row_base + 32u <= _M")
    assert body.index("if (!_simd_ok) {") < body.index("threadgroup float tg_A")


def test_guarded_simdgroup_path_text_is_otherwise_unchanged():
    """Under the proof every address is in [0, 2**31), so the hot path keeps its
    existing `uint` addressing verbatim."""
    msl = _kloop_msl("i32")
    assert "simdgroup_load(a_frag, A + (row_base + 0u) * (uint)sam + k, (uint)sam);" in msl
    assert "A[gr * (uint)sam + gc]" in msl
    assert "simdgroup_multiply_accumulate" in msl
