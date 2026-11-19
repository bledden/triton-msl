"""The varlen templates address the cell the source's 32-bit offset names, on either side of the view.

The source forms ``(start + row) * stride + head * stride + depth * stride`` in the width of
its cumulative lengths and adds it to the pointer; a 32-bit offset is sign-extended there. A
32-bit start near ``2**31`` times an even row stride wraps into the storage, and a start just
below zero addresses rows before a view. Both are defined launches: every address is inside
the storage. The templates held the start in 32 unsigned bits and indexed unsigned (a start
below zero landed about 2**32 elements past the view), then in 64 exact bits (a product that
wraps in the source landed about 2**31 rows away). They now add in 32 wrapping bits and index
with the sum sign-extended, and the launcher holds each storage to the 2**31 elements on
either side of the tensor's first one in which that is the source's cell.

Sections: the emitted address expressions evaluated on the host for every loaded and stored
cell (no device), the launcher's storage window on synthetic launches (no device), the
contract each lowering publishes (no device), and launches on the GPU. Each GPU launch runs
the host evaluation first, so nothing is launched that addresses outside its storage.
"""

import ast
import math
import re

import pytest
import torch

from tests.test_attention_template_scalar_domain import _reason
from tests.test_flash_attention import _flash_attn_fwd
from tests.test_template_scalar_abi import _lower
from tests.test_varlen_fa_routing import _varlen_causal_fwd, _varlen_fwd

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0
SIGNED_CELLS, UNSIGNED = "attention_signed_cells_v1", "attention_unsigned_scalars_v1"
HEADS, MAXS, PAD = 2, 64, 8
LENS_Q, LENS_K = (40, 24), (45, 29)

# name: (cumulative-length dtype, first start, the view row the source's offset names for it)
# The pointer add sign-extends a 32-bit offset whatever the tensor's sign (the compiled
# source's reading, and the template's). Read zero-extended, the last uint32 case would
# address outside every storage and the source would define nothing for it.
STARTS = {
    "plain": (torch.int32, 0, 0),
    "int32_below_zero": (torch.int32, -5, -5),
    "int64_below_zero": (torch.int64, -5, -5),
    "int32_product_wraps_from_the_bottom": (torch.int32, -(2**31) + 10, 10),
    "int32_product_wraps_from_the_middle": (torch.int32, 2**30 + 10, 10),
    "int32_product_wraps_to_below_zero": (torch.int32, 2**31 - 5, -5),
    "uint32_product_wraps": (torch.uint32, 2**31 + 10, 10),
    "uint32_product_wraps_to_below_zero": (torch.uint32, 2**32 - 5, -5),
}
ELEMENTS = [(64, torch.float32), (64, torch.float16), (128, torch.float32), (128, torch.float16)]
ELEMENT_IDS = ["d64_fp32", "d64_fp16", "d128_fp32", "d128_fp16"]
SIGNATURE = {torch.float32: "fp32", torch.float16: "fp16", torch.int32: "i32", torch.uint32: "u32", torch.int64: "i64"}
BITS = {torch.int32: 32, torch.uint32: 32, torch.int64: 64}
DEPTHS = lambda dim: (0, 1, 7, 8, dim - 1)  # noqa: E731


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _cumulative(dtype, start, lengths):
    """The cumulative lengths as the tensor holds them: sums wrap at the tensor's width."""
    values = [start, start + lengths[0], start + lengths[0] + lengths[1]]
    if dtype == torch.uint32:
        return [v % 2**32 for v in values]
    return [_signed(v, BITS[dtype]) for v in values]


# ---------------------------------------------------------------------------------------
# The emitted address expressions, evaluated on the host
# ---------------------------------------------------------------------------------------


class _Msl:
    """An MSL integer value: ``uint``, ``int`` or ``long``, wrapping, with C's conversions."""

    def __init__(self, value, kind):
        self.kind = kind
        bits = 64 if kind == "long" else 32
        self.value = value % (1 << bits) if kind == "uint" else _signed(value, bits)

    def _join(self, other, op):
        kinds = (self.kind, other.kind)
        kind = "long" if "long" in kinds else "uint" if "uint" in kinds else "int"
        return _Msl(op(self.value, other.value), kind)

    def __add__(self, other):
        return self._join(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._join(other, lambda a, b: a * b)


def _index_expression(text, tensor):
    """The one expression ``tensor`` is indexed with in the shader text."""
    found = set()
    for match in re.finditer(rf"(?<![\w]){tensor}\[", text):
        depth, end = 1, match.end()
        while depth:
            depth += {"[": 1, "]": -1}.get(text[end], 0)
            end += 1
        found.add(text[match.end() : end - 1])
    assert len(found) == 1, (tensor, found)
    return found.pop()


def _evaluate(node, names):
    """One address expression: sums, products, ``uint``/``int``/``long`` casts, names, literals."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = _evaluate(node.left, names), _evaluate(node.right, names)
        return left + right if isinstance(node.op, ast.Add) else left * right
    if isinstance(node, ast.Call) and node.func.id in ("uint", "int", "long") and len(node.args) == 1:
        return _Msl(_evaluate(node.args[0], names).value, node.func.id)
    if isinstance(node, ast.Name):
        return names[node.id]
    raise AssertionError(ast.dump(node))


def _template_offsets(text, tensor, side, stored_start, strides, rows, dim):
    """{(row, head, depth): element offset} the template forms for ``tensor`` over ``rows``."""
    cu = "CUQ" if side == "q" else "CUK"
    held = re.search(rf"device const (int|uint|long)\* {cu} ", text).group(1)
    kind, cast = re.search(rf"(uint|int|long) {side}_start = (uint|int|long)\({cu}\[bb\]\);", text).groups()
    start = _Msl(_Msl(_Msl(stored_start, held).value, cast).value, kind)
    # A literal ``8u`` becomes the name ``u8``.
    tree = ast.parse(re.sub(r"\b(\d+)u\b", r"u\1", _index_expression(text, tensor)), mode="eval").body
    prefix = tensor[0].lower()
    scope = {f"{prefix}_{axis}": _Msl(stride, "uint") for axis, stride in zip(("st", "sh", "sk"), strides)}
    scope[f"{side}_start"] = start
    row_names = {"qr", "qrow"} if side == "q" else {"kr", "krow"}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - {"uint", "int", "long"}
    scope.update({name: _Msl(int(name[1:]), "uint") for name in used if re.fullmatch(r"u\d+", name)})
    assert not used - set(scope) - row_names - {"h", "c", "d", "dd", "et"}, used
    offsets = {}
    for row in range(rows):
        for head in range(HEADS):
            for depth in DEPTHS(dim):
                names = dict(scope, h=_Msl(head, "uint"), et=_Msl(depth // 8, "uint"))
                names.update({name: _Msl(row, "uint") for name in row_names})
                inner = depth % 8 if "et" in used else depth
                names.update({name: _Msl(inner, "uint") for name in ("c", "d", "dd")})
                offsets[row, head, depth] = _evaluate(tree, names).value
    return offsets


def _source_offsets(dtype, stored_start, strides, rows, dim):
    """The same cells as the source addresses them.

    ``Q + (start + row) * stride + head * stride + depth * stride`` is three pointer adds. The
    first offset has the width of the cumulative lengths, the other two are 32-bit products
    of 32-bit arguments; each wraps at its width and is sign-extended where it is added.
    """
    wrap = lambda v: _signed(v, BITS[dtype])  # noqa: E731
    offsets = {}
    for row in range(rows):
        for head in range(HEADS):
            for depth in DEPTHS(dim):
                offsets[row, head, depth] = (
                    wrap(wrap(stored_start + row) * strides[0])
                    + _signed(head * strides[1], 32)
                    + _signed(depth * strides[2], 32)
                )
    return offsets


def _lowered(kernel, dim, elem, dtype):
    const = dict(SCALE=dim**-0.5, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=dim)
    sig = {}
    for name in kernel.arg_names:
        if name not in const:
            pointer = "*" + SIGNATURE[elem] if name in ("Q", "K", "V", "Out") else "*" + SIGNATURE[dtype]
            sig[name] = pointer if name in ("Q", "K", "V", "Out", "cu_q", "cu_k") else "i32"
    lowerer = _lower(kernel, sig, const)
    return lowerer.lower(), lowerer._tail_access_bounds


def _layout(start_name, side, dim):
    """Storage rows, view rows and cumulative lengths of one launch; the named side takes the start."""
    dtype, start, first = STARTS[start_name]
    plan = {}
    for tensor, lengths in (("q", LENS_Q), ("k", LENS_K)):
        taken = (start, first) if tensor == side else (0, 0)
        plan[tensor] = dict(
            cu=_cumulative(dtype, taken[0], lengths),
            first=taken[1],
            lengths=lengths,
            storage_rows=PAD + max(taken[1], 0) + sum(lengths) + PAD,
        )
    return dtype, plan, (HEADS * dim, dim, 1)


def _assert_addresses_are_the_source_s(start_name, side, dim, elem, causal):
    """Every cell the template loads or stores is the source's, and inside the storage."""
    dtype, plan, strides = _layout(start_name, side, dim)
    text, _contract = _lowered(_varlen_causal_fwd if causal else _varlen_fwd, dim, elem, dtype)
    for tensor, rows_of in (("Q", "q"), ("Out", "q"), ("K", "k"), ("V", "k")):
        part = plan[rows_of]
        before, after = PAD * strides[0], (part["storage_rows"] - PAD) * strides[0]
        for sequence in range(2):
            stored_start, rows = part["cu"][sequence], part["lengths"][sequence]
            source = _source_offsets(dtype, stored_start, strides, rows, dim)
            first_row = part["first"] + sum(part["lengths"][:sequence])
            for (row, head, depth), offset in source.items():
                assert offset == (first_row + row) * strides[0] + head * strides[1] + depth * strides[2]
                assert -before <= offset < after
            assert _template_offsets(text, tensor, rows_of, stored_start, strides, rows, dim) == source, tensor


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim,elem", ELEMENTS, ids=ELEMENT_IDS)
@pytest.mark.parametrize("side", ["q", "k"])
@pytest.mark.parametrize("start_name", sorted(STARTS))
def test_every_emitted_address_is_the_source_s_cell(start_name, side, dim, elem, causal):
    _assert_addresses_are_the_source_s(start_name, side, dim, elem, causal)


def test_the_model_reads_an_unsigned_index_and_an_exact_start_as_metal_does():
    # The two earlier forms of the template, on the launches each got wrong.
    unsigned = "device const int* CUK [[buffer(5)]],\n uint k_start = uint(CUK[bb]);\n K[(k_start + krow) * k_st + h * k_sh + dd * k_sk]"
    exact = "device const int* CUK [[buffer(5)]],\n long k_start = long(CUK[bb]);\n K[(k_start + krow) * k_st + h * k_sh + dd * k_sk]"
    strides = (128, 64, 1)
    assert _template_offsets(unsigned, "K", "k", -5, strides, 1, 64)[0, 0, 0] == 2**32 - 5 * 128
    assert _template_offsets(exact, "K", "k", -5, strides, 1, 64)[0, 0, 0] == -5 * 128
    assert _template_offsets(unsigned, "K", "k", -(2**31) + 10, strides, 1, 64)[0, 0, 0] == 10 * 128
    assert _template_offsets(exact, "K", "k", -(2**31) + 10, strides, 1, 64)[0, 0, 0] == (-(2**31) + 10) * 128


# ---------------------------------------------------------------------------------------
# The launcher's storage window on a synthetic launch
# ---------------------------------------------------------------------------------------


class _Tensor:
    """What the launcher reads of a tensor argument: its storage and where it starts in it."""

    def __init__(self, before, after, element_size=2):
        self._before, self._bytes, self._size = before, (before + after) * element_size, element_size

    def untyped_storage(self):
        storage = self

        class Storage:
            def nbytes(self):
                return storage._bytes

        return Storage()

    def element_size(self):
        return self._size

    def storage_offset(self):
        return self._before


def _window(tag, before, after, element_size=2):
    return _reason((64,), descriptor=(tag, (1,), (), None), pointer=_Tensor(before, after, element_size))


@pytest.mark.parametrize("element_size", [2, 4])
@pytest.mark.parametrize(
    "before,after",
    [(0, 1), (0, 2**31), (2**31, 2**31), (2**31, 1), (4096, 2**30 + 1), (1, 2**31)],
)
def test_signed_cells_launch_inside_2_31_elements_either_way(before, after, element_size):
    assert _window(SIGNED_CELLS, before, after, element_size) is None


@pytest.mark.parametrize("before,after", [(0, 2**31 + 1), (2**31 + 1, 64), (2**31 + 1, 2**31 + 1), (0, 2**32)])
def test_signed_cells_refuse_a_storage_beyond_2_31_elements_either_way(before, after):
    reason = _window(SIGNED_CELLS, before, after)
    assert reason is not None
    assert f"argument 0 has {before} elements of storage before its first one and {after} from it on" in reason
    assert "at most 2**31 elements either way" in reason


@pytest.mark.parametrize(
    "before,after",
    [(0, 1), (0, 2**31), (0, 2**32), (1, 2**30), (2**31, 2**30), (2**32 - 2**30, 2**30), (4096, 64)],
)
def test_unsigned_cells_launch_on_a_whole_storage_or_a_view_of_at_most_2_30_elements(before, after):
    assert _window(UNSIGNED, before, after) is None


@pytest.mark.parametrize("before,after", [(1, 2**30 + 1), (8, 2**31), (2**31, 2**31)])
def test_unsigned_cells_refuse_a_view_a_wrapped_offset_could_land_before(before, after):
    reason = _window(UNSIGNED, before, after)
    assert reason is not None
    assert f"argument 0 is a view {before} elements into its storage with {after} from its first one on" in reason
    assert "at most 2**30 elements from its first one on" in reason


@pytest.mark.parametrize("before,after", [(0, 2**32 + 1), (2**32, 1), (2**32, 2**30)])
def test_unsigned_cells_refuse_a_storage_beyond_2_32_elements(before, after):
    reason = _window(UNSIGNED, before, after)
    assert reason is not None and f"argument 0 has a storage of {before + after} elements" in reason


def test_the_two_windows_are_the_ones_the_index_types_give():
    import triton_msl.backend.driver as driver

    assert driver._ATTENTION_SIGNED_CELL_ELEMENTS == 2**31
    assert driver._ATTENTION_TEMPLATE_ELEMENTS == 2**32
    assert driver._ATTENTION_TEMPLATE_VIEW_ELEMENTS == 2**30


def test_a_wrapped_offset_inside_a_view_s_storage_needs_an_unwrapped_one_outside_it():
    """The reason a view of at most 2**30 elements is safe to index unsigned, on small numbers.

    Scaled to 8 bits: strides below 2**7, a wrapped offset is one of at least 2**7, and the
    claim is that a walk up from zero that reaches one also reaches one in [2**6, 2**7).
    """
    for row_stride in range(1, 2**7):
        for depth_stride in (1, 3, 2**6 + 1):
            for rows in range(1, 6):
                for depth in range(1, 4):
                    cells = [r * row_stride + d * depth_stride for r in range(rows) for d in range(depth)]
                    if any(cell >= 2**7 for cell in cells):
                        assert any(2**6 <= cell < 2**7 for cell in cells), (row_stride, depth_stride, rows, depth)


# ---------------------------------------------------------------------------------------
# The contract each lowering publishes
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim,elem", ELEMENTS, ids=ELEMENT_IDS)
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint32, torch.int64], ids=["int32", "uint32", "int64"])
def test_varlen_lowering_publishes_signed_cells(dtype, dim, elem, causal):
    text, contract = _lowered(_varlen_causal_fwd if causal else _varlen_fwd, dim, elem, dtype)
    assert contract[0] == SIGNED_CELLS and contract[3] is None
    for tensor in ("Q", "K", "V", "Out"):
        assert _index_expression(text, tensor).startswith("int(")


def test_dense_lowering_keeps_the_unsigned_contract():
    const = dict(BLOCK_M=32, BLOCK_N=32, HEAD_DIM=64, IS_CAUSAL=False, Z=1)
    const.update(stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1)
    sig = {n: "*fp32" if n in ("Q", "K", "V", "Out") else "i32" for n in _flash_attn_fwd.arg_names if n not in const}
    lowerer = _lower(_flash_attn_fwd, sig, const)
    lowerer.lower()
    assert lowerer._tail_access_bounds[0] == UNSIGNED


# ---------------------------------------------------------------------------------------
# GPU: the launches above compute the source's rows
# ---------------------------------------------------------------------------------------


@pytest.fixture
def fa_hits(monkeypatch):
    """Record whether the FlashAttention replacement dispatch ran (the driver imports it per call)."""
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits, real = [], fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim,elem", ELEMENTS, ids=ELEMENT_IDS)
@pytest.mark.parametrize("side", ["q", "k"])
@pytest.mark.parametrize("start_name", sorted(STARTS))
def test_varlen_start_computes_the_source_s_rows(start_name, side, dim, elem, causal, device, fa_hits):
    # Nothing is launched unless every address was shown to be the source's and inside storage.
    _assert_addresses_are_the_source_s(start_name, side, dim, elem, causal)
    dtype, plan, _strides = _layout(start_name, side, dim)
    scale, tol = 1.0 / math.sqrt(dim), 2e-2 if elem == torch.float16 else 1e-3
    gen = torch.Generator().manual_seed(1140)
    q = torch.randn(plan["q"]["storage_rows"], HEADS, dim, generator=gen).to(elem)
    k, v = [torch.randn(plan["k"]["storage_rows"], HEADS, dim, generator=gen).to(elem) for _ in range(2)]
    o = torch.full(q.shape, POISON, dtype=elem, device=device)
    cq, ck = [torch.tensor(plan[t]["cu"], dtype=dtype, device=device) for t in ("q", "k")]
    qd, kd, vd, od = [t[PAD:] for t in (q.to(device), k.to(device), v.to(device), o)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    (_varlen_causal_fwd if causal else _varlen_fwd)[(2, 2 * HEADS)](
        qd, kd, vd, od, cq, ck, *st(qd), *st(kd), *st(vd), *st(od), HEADS, MAXS, scale, 32, 32, dim
    )
    if device == "mps":
        torch.mps.synchronize()
        assert fa_hits == [True]
    expected = torch.full(o.shape, POISON, dtype=torch.float64)
    for b in range(2):
        q0, k0 = [PAD + plan[t]["first"] + sum(plan[t]["lengths"][:b]) for t in ("q", "k")]
        q1, k1 = q0 + LENS_Q[b], k0 + LENS_K[b]
        scores = scale * torch.einsum("qhd,khd->hqk", q[q0:q1].double(), k[k0:k1].double())
        if causal:
            scores = scores.masked_fill(torch.arange(q1 - q0)[:, None] < torch.arange(k1 - k0)[None, :], float("-inf"))
        expected[q0:q1] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), v[k0:k1].double())
    got = o.cpu().double()
    assert torch.equal(got != POISON, expected != POISON)
    live = expected != POISON
    torch.testing.assert_close(got[live], expected[live], atol=tol, rtol=tol)
