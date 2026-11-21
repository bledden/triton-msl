"""Launch-time tile contracts of the FlashAttention, MLA and varlen templates.

The templates store only the rows below the sequence length and, on the MLA route, attend only
the keys below it. A source kernel with an UNMASKED store writes every row of every launched
tile, and an MLA source attends whole key tiles. The two agree only when the runtime length is a
whole number of tiles, which is known only at launch: the lowering hands the launcher a
descriptor (``fa_whole_row_tiles_v1`` / ``mla_whole_tiles_v1``) and the launcher refuses a
launch that would lose rows. The varlen templates attend ``range(0, max_seqlen, 32)`` whole key
tiles masked by the sequence's key count, exactly as the source loop does.

Sections: the two driver reasons on synthetic launches (no device), ``_varlen_kmax_expr`` (no
device), the descriptors the lowering emits (no device), and the launches on the GPU against
float64 references of the SOURCE semantics with poisoned outputs.
"""

import math
import re

import pytest
import torch
import triton

from tests.test_fa_biased_routing import _biased_fa
from tests.test_flash_attention import _flash_attn_fwd
from tests.test_template_scalar_abi import _lower
from triton_msl.backend._launch_signature import scalar_bytes
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0
FA_TAG, MLA_TAG = "fa_whole_row_tiles_v1", "mla_whole_tiles_v1"


def _tile_contract(lowerer):
    """The tile contract of a lowered attention template: it rides inside the scalar-domain one."""
    tag, _unsigned, _signed, inner = lowerer._tail_access_bounds
    assert tag == "attention_unsigned_scalars_v1"
    return inner


def _variant(kernel, *edits):
    """A fresh JITFunction of ``kernel`` with each (old, new) applied exactly once."""
    fn = triton.JITFunction(kernel.fn)
    src = fn.src
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    fn._unsafe_update_src(src)
    return fn


# ---------------------------------------------------------------------------------------
# _fa_whole_row_tiles_reason: the sequence length is argument 2 of a synthetic launch
# ---------------------------------------------------------------------------------------


def _fa_reason(rows, tiles, block=32, declared="i32", payload=None, descriptor=None):
    from triton_msl.backend.driver import _fa_whole_row_tiles_reason

    kargs = [object(), object(), rows]
    signatures = ["*fp32", "*fp32", declared]
    if payload is None:
        payload = scalar_bytes(rows, declared)
    return _fa_whole_row_tiles_reason(
        (FA_TAG, 2, block) if descriptor is None else descriptor,
        kargs,
        signatures,
        [None, None, payload],
        (tiles, 2, 1),
    )


@pytest.mark.parametrize("rows,tiles", [(96, 3), (64, 2), (32, 1), (100, 3), (96, 2)])
def test_fa_rows_whole_or_beyond_the_launch_pass(rows, tiles):
    # The source stores tiles * 32 rows; the template stores min(rows, tiles * 32). They agree
    # whenever no launched tile crosses the length (fewer launched tiles store fewer rows on both).
    assert _fa_reason(rows, tiles) is None


@pytest.mark.parametrize("declared", ["i32", "u32", "i64", "u64"])
def test_fa_partial_last_tile_refuses(declared):
    reason = _fa_reason(80, 3, declared=declared)
    assert reason is not None
    assert "all 96 rows of its 3 launched 32-row tiles" in reason
    assert "only the 80 rows below the sequence length" in reason


@pytest.mark.parametrize("rows", [0, -7])
def test_fa_zero_or_negative_length_refuses(rows):
    reason = _fa_reason(rows, 1)
    assert reason is not None and "only the 0 rows" in reason


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        "fa_whole_row_tiles_v1",
        (FA_TAG, 2),
        (FA_TAG, 2, 32, 0),
        ("mla_whole_tiles_v1", 2, 32),
        (FA_TAG, True, 32),
        (FA_TAG, 2.0, 32),
        (FA_TAG, 2, 0),
        (FA_TAG, 2, -32),
        (FA_TAG, 2, 32.0),
        (FA_TAG, 3, 32),
        (FA_TAG, -1, 32),
    ],
    ids=repr,
)
def test_fa_malformed_descriptor_refuses(descriptor):
    from triton_msl.backend.driver import _fa_whole_row_tiles_reason

    kargs, signatures = [object(), object(), 96], ["*fp32", "*fp32", "i32"]
    payloads = [None, None, scalar_bytes(96, "i32")]
    reason = _fa_whole_row_tiles_reason(descriptor, kargs, signatures, payloads, (3, 1, 1))
    assert reason is not None and reason.startswith("the store-domain descriptor is missing or malformed")


def test_fa_list_descriptor_is_accepted():
    assert _fa_reason(96, 3, descriptor=[FA_TAG, 2, 32]) is None


@pytest.mark.parametrize("declared", ["fp32", "*fp32", "i16", "i1"])
def test_fa_non_integer_length_signature_refuses(declared):
    from triton_msl.backend.driver import _fa_whole_row_tiles_reason

    reason = _fa_whole_row_tiles_reason(
        (FA_TAG, 2, 32), [object(), object(), 96], ["*fp32", "*fp32", declared], [None, None, b"\0" * 4], (3, 1, 1)
    )
    assert reason is not None and reason.startswith("the sequence length is not an integer argument")


@pytest.mark.parametrize(
    "payload",
    [scalar_bytes(96, "i32"), scalar_bytes(80, "i64"), None, bytearray(scalar_bytes(80, "i32")), "80"],
    ids=["other_value", "other_width", "missing", "bytearray", "str"],
)
def test_fa_payload_mismatch_refuses(payload):
    from triton_msl.backend.driver import _fa_whole_row_tiles_reason

    reason = _fa_whole_row_tiles_reason(
        (FA_TAG, 2, 32), [object(), object(), 80], ["*fp32", "*fp32", "i32"], [None, None, payload], (3, 1, 1)
    )
    assert reason == "the sequence length changed after its captured launch payload"


def test_fa_reads_the_captured_value_not_a_subclass_int():
    class _Lying(int):
        def __int__(self):
            return 96

    # The payload packs the integer value (80); __int__ would claim whole tiles.
    reason = _fa_reason(_Lying(80), 3)
    assert reason is not None and "only the 80 rows" in reason


@pytest.mark.parametrize("bad", ["grid", "value"])
def test_fa_unevaluable_launch_refuses(bad):
    from triton_msl.backend.driver import _fa_whole_row_tiles_reason

    if bad == "grid":
        reason = _fa_whole_row_tiles_reason(
            (FA_TAG, 2, 32),
            [object(), object(), 96],
            ["*fp32", "*fp32", "i32"],
            [None, None, scalar_bytes(96, "i32")],
            None,
        )
    else:
        reason = _fa_whole_row_tiles_reason(
            (FA_TAG, 2, 32), [object(), object(), 80.5], ["*fp32", "*fp32", "i32"], [None, None, b"\0" * 4], (3, 1, 1)
        )
    assert reason is not None and reason.startswith("the store domain could not be evaluated")


# ---------------------------------------------------------------------------------------
# _mla_whole_tiles_reason
# ---------------------------------------------------------------------------------------


_DEFAULT = object()


def _mla_reason(rows, block=32, declared="i32", payload=None, descriptor=_DEFAULT):
    from triton_msl.backend.driver import _mla_whole_tiles_reason

    if payload is None:
        payload = scalar_bytes(rows, declared)
    return _mla_whole_tiles_reason(
        (MLA_TAG, 1, block) if descriptor is _DEFAULT else descriptor,
        [object(), rows],
        ["*fp16", declared],
        [None, payload],
    )


@pytest.mark.parametrize("rows,block", [(32, 32), (64, 32), (96, 32), (128, 64), (64, 64)])
def test_mla_whole_tiles_pass(rows, block):
    assert _mla_reason(rows, block) is None


@pytest.mark.parametrize("rows,block", [(48, 32), (80, 32), (33, 32), (1, 32), (96, 64), (0, 32), (-32, 32)])
def test_mla_partial_zero_or_negative_length_refuses(rows, block):
    reason = _mla_reason(rows, block)
    assert reason is not None
    assert f"the sequence length {rows} is not a whole number of {block}-row tiles" in reason


@pytest.mark.parametrize("declared", ["u32", "i64", "u64"])
def test_mla_other_integer_widths(declared):
    assert _mla_reason(64, declared=declared) is None
    assert "48 is not a whole number" in _mla_reason(48, declared=declared)


@pytest.mark.parametrize(
    "descriptor",
    [None, (MLA_TAG, 1), (FA_TAG, 1, 32), (MLA_TAG, False, 32), (MLA_TAG, 1, 0), (MLA_TAG, 1, -32), (MLA_TAG, 2, 32)],
    ids=repr,
)
def test_mla_malformed_descriptor_refuses(descriptor):
    reason = _mla_reason(64, descriptor=descriptor)
    assert reason == "the tile-domain descriptor is missing or malformed"


def test_mla_non_integer_signature_and_payload_mismatch_refuse():
    from triton_msl.backend.driver import _mla_whole_tiles_reason

    assert _mla_whole_tiles_reason((MLA_TAG, 1, 32), [object(), 64], ["*fp16", "fp32"], [None, b"\0" * 4]) == (
        "the sequence length is not an integer argument"
    )
    assert _mla_reason(48, payload=scalar_bytes(64, "i32")) == (
        "the sequence length changed after its captured launch payload"
    )
    reason = _mla_whole_tiles_reason((MLA_TAG, 1, 32), [object(), 64.5], ["*fp16", "i32"], [None, b"\0" * 4])
    assert reason.startswith("the tile domain could not be evaluated")


# ---------------------------------------------------------------------------------------
# _varlen_kmax_expr: the keys the varlen source attends
# ---------------------------------------------------------------------------------------


def _eval_kmax(expr, maxs, seqlen_k):
    """Evaluate the emitted MSL ``(c) ? a : b`` expression on integers (uint wrap where it occurs)."""
    m = re.fullmatch(r"\(\((.*)\) \? (.*) : (.*)\)", expr)
    assert m, expr

    def py(text):
        text = re.sub(r"\b(\d+)u\b", r"\1", text)
        text = re.sub(r"\bint\(", "_int(", re.sub(r"\buint\(", "_uint(", text)).replace("/", "//")
        # The text is the module's own emitted expression; eval sees no builtins, only these names.
        names = {"__builtins__": {}, "_uint": lambda x: x % 2**32, "_int": int, "min": min}
        return eval(text, names | {"MAXS": maxs, "seqlen_k": seqlen_k})

    return py(m.group(2)) if py(m.group(1)) else py(m.group(3))


def test_varlen_kmax_without_a_bound_is_the_key_count():
    from triton_msl.codegen._msl_templates import _varlen_kmax_expr

    assert _varlen_kmax_expr(None) == "seqlen_k"
    assert _varlen_kmax_expr("") == "seqlen_k"


def test_varlen_kmax_rounds_the_bound_up_to_whole_key_tiles():
    from triton_msl.codegen._msl_templates import _varlen_kmax_expr

    expr = _varlen_kmax_expr("MAXS")
    assert expr == "((int(MAXS) <= 0) ? 0u : min(seqlen_k, ((uint(MAXS) - 1u) / 32u + 1u) * 32u))"
    for seqlen_k in (0, 1, 33, 40, 60, 64, 70, 100, 200):
        for maxs in range(-3, 140):
            # The source: for start_n in range(0, max_seqlen, 32), keys masked by kn < seqlen_k.
            attended = len([kn for s in range(0, maxs, 32) for kn in range(s, s + 32) if kn < seqlen_k])
            assert _eval_kmax(expr, maxs, seqlen_k) == attended, (maxs, seqlen_k)


# ---------------------------------------------------------------------------------------
# The descriptors the lowering hands the launcher (no device)
# ---------------------------------------------------------------------------------------

_FA_STORE = "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)"
_FA_UNMASKED = "tl.store(o_ptrs, acc.to(Out.dtype.element_ty))"


def _lower_dense(fn, head_dim=64, n_ctx=None):
    const = dict(BLOCK_M=32, BLOCK_N=32, HEAD_DIM=head_dim, IS_CAUSAL=False, Z=1)
    const.update(stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1)
    if n_ctx is not None:
        const["N_CTX"] = n_ctx
    sig = {n: "*fp32" if n in ("Q", "K", "V", "Out") else "i32" for n in fn.arg_names if n not in const}
    return _lower(fn, sig, const)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_unmasked_dense_store_carries_the_row_tile_descriptor(head_dim):
    lowerer = _lower_dense(_variant(_flash_attn_fwd, (_FA_STORE, _FA_UNMASKED)), head_dim)
    lowerer.lower()
    tag, index, block = _tile_contract(lowerer)
    assert (tag, block) == (FA_TAG, 32)
    assert [a.name for a in lowerer.graph.args if a.index == index] == ["N_CTX"]


def test_masked_dense_store_carries_no_row_tile_descriptor():
    lowerer = _lower_dense(_flash_attn_fwd)
    lowerer.lower()
    assert _tile_contract(lowerer) is None


# A mask on the columns alone bounds no row: the source still stores every launched row.
_FA_COLUMN_MASKS = {
    "head_dim": "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)",
    "literal": "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_d[None, :] < 1000)",
}
_FA_ROWS_AND_COLUMNS = (
    "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))"
)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("columns", sorted(_FA_COLUMN_MASKS))
def test_column_only_dense_store_mask_carries_the_row_tile_descriptor(columns, head_dim):
    lowerer = _lower_dense(_variant(_flash_attn_fwd, (_FA_STORE, _FA_COLUMN_MASKS[columns])), head_dim)
    lowerer.lower()
    tag, index, block = _tile_contract(lowerer)
    assert (tag, block) == (FA_TAG, 32)
    assert [a.name for a in lowerer.graph.args if a.index == index] == ["N_CTX"]


def test_row_and_column_dense_store_mask_carries_no_row_tile_descriptor():
    lowerer = _lower_dense(_variant(_flash_attn_fwd, (_FA_STORE, _FA_ROWS_AND_COLUMNS)))
    lowerer.lower()
    assert _tile_contract(lowerer) is None


def test_unmasked_dense_store_without_a_length_argument_refuses():
    # N_CTX == 1 is specialized away (the template still routes it); a larger constant N_CTX
    # does not take the template at all, so it never reaches this contract.
    lowerer = _lower_dense(_variant(_flash_attn_fwd, (_FA_STORE, _FA_UNMASKED)), n_ctx=1)
    with pytest.raises(MetalNonRecoverableError, match="unmasked store and no runtime sequence-length argument"):
        lowerer.lower()


def test_mla_lowering_carries_the_whole_tile_descriptor():
    from tests.test_mla_value_paths import _mla_value_path

    const = dict(BM=32, BN=32, DN=128, DR=64, DV=128, MODE=0, Z=1, qnk=1, qrk=1, knk=1, krk=1, vk=1, ok_=1)
    sig = {n: "*fp16" if n in ("Qn", "Qr", "Kn", "Kr", "V", "Out") else "i32" for n in _mla_value_path.arg_names}
    sig = {n: t for n, t in sig.items() if n not in const}
    lowerer = _lower(_mla_value_path, sig, const)
    lowerer.lower()
    tag, index, block = _tile_contract(lowerer)
    assert (tag, block) == (MLA_TAG, 32)
    assert [a.name for a in lowerer.graph.args if a.index == index] == ["N_CTX"]


# ---------------------------------------------------------------------------------------
# GPU: whole tiles run on the template, a partial last tile refuses at launch
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


@pytest.fixture
def mla_hits(monkeypatch):
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits, real = [], fa_dispatch._dispatch_mla

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "_dispatch_mla", spy)
    return hits


def _assert_poisoned_match(got, expected, tol):
    """Unwritten cells stay poison, written cells match: a missing or extra write both fail."""
    got, expected = got.cpu().double(), expected.double()
    written = expected != POISON
    assert torch.equal(got != POISON, written), f"{int(((got != POISON) != written).sum())} cells differ in domain"
    torch.testing.assert_close(got[written], expected[written], atol=tol, rtol=tol)


def _dense_source(q, k, v, n_ctx, rows, causal):
    """Float64 semantics of tests/test_flash_attention.py::_flash_attn_fwd for ``rows`` launched rows.

    Q, K and V loads are masked to zero at rows >= N_CTX. The K/V loop runs whole 32-key tiles
    up to ``hi`` and has no score mask, so a zero key in the last tile still takes part (score 0).
    """
    qs = q[:, :, :rows].double().clone()
    qs[:, :, n_ctx:] = 0
    qs = qs * q.shape[-1] ** -0.5
    out = torch.empty(qs.shape, dtype=torch.float64)
    for pid in range(rows // 32):
        hi = min((pid + 1) * 32, n_ctx) if causal else n_ctx
        span = triton.cdiv(hi, 32) * 32
        kk, vv = k[:, :, :span].double().clone(), v[:, :, :span].double().clone()
        kk[:, :, n_ctx:] = 0
        vv[:, :, n_ctx:] = 0
        s = qs[:, :, pid * 32 : (pid + 1) * 32] @ kk.transpose(-1, -2)
        if causal:
            rm = torch.arange(pid * 32, pid * 32 + 32)
            s = s.masked_fill(rm[:, None] < torch.arange(span)[None, :], float("-inf"))
        out[:, :, pid * 32 : (pid + 1) * 32] = torch.softmax(s, -1) @ vv
    return out


def _launch_dense(fn, q, k, v, out, n_ctx, tiles, causal):
    fn[(tiles, q.shape[0] * q.shape[1])](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        q.shape[0],
        q.shape[1],
        n_ctx,
        BLOCK_M=32,
        BLOCK_N=32,
        HEAD_DIM=q.shape[-1],
        IS_CAUSAL=causal,
    )
    torch.mps.synchronize()


@requires_mps
@pytest.mark.parametrize(
    "dtype,head_dim,causal,n_ctx,tiles",
    [
        (torch.float32, 64, False, 96, 3),  # whole tiles
        (torch.float32, 64, True, 96, 3),
        (torch.float16, 128, False, 64, 2),
        (torch.float32, 64, False, 80, 2),  # 64 launched rows below N_CTX = 80: no tile crosses it
        (torch.float32, 64, False, 100, 3),
    ],
)
def test_unmasked_dense_store_on_whole_tiles_computes(dtype, head_dim, causal, n_ctx, tiles, fa_hits):
    _all_launched_rows_compute(_FA_UNMASKED, dtype, head_dim, causal, n_ctx, tiles, fa_hits)


def _all_launched_rows_compute(store, dtype, head_dim, causal, n_ctx, tiles, fa_hits):
    fn = _variant(_flash_attn_fwd, (_FA_STORE, store))
    gen = torch.Generator().manual_seed(1117 + head_dim + n_ctx)
    alloc = max(tiles * 32, triton.cdiv(n_ctx, 32) * 32)
    q, k, v = [torch.randn(1, 2, alloc, head_dim, generator=gen).to(dtype) for _ in range(3)]
    out = torch.full((1, 2, alloc + 32, head_dim), POISON, dtype=dtype)
    expected = out.double().clone()
    # The source stores every launched row; rows past them keep the poison.
    expected[:, :, : tiles * 32] = _dense_source(q, k, v, n_ctx, tiles * 32, causal)
    out = out.to("mps")
    _launch_dense(fn, *[x.to("mps") for x in (q, k, v)], out[:, :, :alloc], n_ctx, tiles, causal)
    assert fa_hits == [True]
    _assert_poisoned_match(out, expected, 2e-2 if dtype == torch.float16 else 2e-3)


@requires_mps
@pytest.mark.parametrize("head_dim,causal,n_ctx", [(64, False, 80), (128, False, 80), (64, True, 65), (64, False, 33)])
def test_unmasked_dense_store_over_a_partial_tile_refuses_at_launch(head_dim, causal, n_ctx, fa_hits):
    _partial_tile_refuses_at_launch(_FA_UNMASKED, head_dim, causal, n_ctx, fa_hits)


def _partial_tile_refuses_at_launch(store, head_dim, causal, n_ctx, fa_hits):
    fn = _variant(_flash_attn_fwd, (_FA_STORE, store))
    tiles = triton.cdiv(n_ctx, 32)
    q, k, v = [torch.randn(1, 2, tiles * 32, head_dim, device="mps") for _ in range(3)]
    out = torch.full_like(q, POISON)
    with pytest.raises(MetalNonRecoverableError, match="Refusing unmasked FlashAttention store launch") as info:
        _launch_dense(fn, q, k, v, out, n_ctx, tiles, causal)
    assert f"all {tiles * 32} rows of its {tiles} launched 32-row tiles" in str(info.value)
    assert fa_hits == []
    assert bool((out.cpu() == POISON).all())


@requires_mps
@pytest.mark.parametrize("columns", sorted(_FA_COLUMN_MASKS))
@pytest.mark.parametrize(
    "dtype,head_dim,causal,n_ctx,tiles",
    [
        (torch.float32, 64, False, 96, 3),
        (torch.float16, 128, True, 64, 2),
        (torch.float32, 64, False, 80, 2),  # 64 launched rows below N_CTX = 80: no tile crosses it
    ],
)
def test_column_only_dense_store_mask_on_whole_tiles_computes(dtype, head_dim, causal, n_ctx, tiles, columns, fa_hits):
    _all_launched_rows_compute(_FA_COLUMN_MASKS[columns], dtype, head_dim, causal, n_ctx, tiles, fa_hits)


@requires_mps
@pytest.mark.parametrize("columns", sorted(_FA_COLUMN_MASKS))
@pytest.mark.parametrize("head_dim,causal,n_ctx", [(64, False, 80), (128, False, 80), (64, True, 65)])
def test_column_only_dense_store_mask_over_a_partial_tile_refuses_at_launch(head_dim, causal, n_ctx, columns, fa_hits):
    _partial_tile_refuses_at_launch(_FA_COLUMN_MASKS[columns], head_dim, causal, n_ctx, fa_hits)


@requires_mps
def test_row_and_column_dense_store_mask_over_a_partial_tile_computes(fa_hits):
    # The row clip is there, and the column term is true on every column: the template's store domain.
    gen = torch.Generator().manual_seed(1120)
    q, k, v = [torch.randn(1, 2, 96, 64, generator=gen) for _ in range(3)]
    out = torch.full((1, 2, 128, 64), POISON)
    expected = out.double().clone()
    expected[:, :, :80] = _dense_source(q, k, v, 80, 96, False)[:, :, :80]
    out = out.to("mps")
    fn = _variant(_flash_attn_fwd, (_FA_STORE, _FA_ROWS_AND_COLUMNS))
    _launch_dense(fn, *[x.to("mps") for x in (q, k, v)], out[:, :, :96], 80, 3, False)
    assert fa_hits == [True]
    _assert_poisoned_match(out, expected, 2e-3)


@requires_mps
def test_masked_dense_store_over_a_partial_tile_still_computes(fa_hits):
    # The canonical boundary mask is the template's own store domain: no launch refusal.
    gen = torch.Generator().manual_seed(1118)
    q, k, v = [torch.randn(1, 2, 96, 64, generator=gen) for _ in range(3)]
    out = torch.full((1, 2, 128, 64), POISON)
    expected = out.double().clone()
    expected[:, :, :80] = _dense_source(q, k, v, 80, 96, False)[:, :, :80]
    out = out.to("mps")
    _launch_dense(_flash_attn_fwd, *[x.to("mps") for x in (q, k, v)], out[:, :, :96], 80, 3, False)
    assert fa_hits == [True]
    _assert_poisoned_match(out, expected, 2e-3)


_B_OUT = "tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[:, None])"
_B_LSE = "tl.store(lse_ptrs, lse, mask=mask_m)"
_BIASED_UNMASKED = {
    "out": [(_B_OUT, "tl.store(o_ptrs, normalize.to(q_block.type.element_ty))")],
    "lse": [(_B_LSE, "tl.store(lse_ptrs, lse)")],
    "both": [(_B_OUT, "tl.store(o_ptrs, normalize.to(q_block.type.element_ty))"), (_B_LSE, "tl.store(lse_ptrs, lse)")],
}


def _launch_biased(fn, n_ctx, tiles, dim=32, seed=1119):
    gen = torch.Generator().manual_seed(seed)
    z, h, rows = 1, 2, tiles * 32
    q, k, v = [torch.randn(z, h, rows, dim, generator=gen) for _ in range(3)]
    b = torch.randn(z, h, rows, rows, generator=gen)
    mask = (torch.rand(z, h, rows, generator=gen) < 0.25).to(torch.uint8)
    o = torch.full((z, h, rows + 32, dim), POISON, device="mps")
    lse = torch.full((z, h, rows + 32), POISON, device="mps")
    dev = [x.to("mps") for x in (q, k, v, b, mask)]
    ov, lv = o[:, :, :rows], lse[:, :, :rows]
    st = lambda t: tuple(t.stride())  # noqa: E731
    args = [ov, *st(ov), lv, *st(lv)]
    for t in dev:
        args += [t, *st(t)]
    fn[(tiles, z * h)](*args, dim**-0.5, -1e9, z, h, n_ctx, dim, 32, 32)
    torch.mps.synchronize()
    raw = dim**-0.5 * (q.double() @ k.double().transpose(-1, -2)) + b.double()
    raw = raw.masked_fill(mask.bool()[:, :, None, :], float("-inf"))
    return o, lse, torch.softmax(raw, -1) @ v.double(), torch.logsumexp(raw, -1)


@requires_mps
@pytest.mark.parametrize("which", sorted(_BIASED_UNMASKED))
def test_unmasked_biased_store_on_whole_tiles_computes(which, fa_hits):
    o, lse, o_ref, lse_ref = _launch_biased(_variant(_biased_fa, *_BIASED_UNMASKED[which]), 64, 2)
    assert fa_hits == [True]
    o_exp = torch.full(o.shape, POISON, dtype=torch.float64)
    lse_exp = torch.full(lse.shape, POISON, dtype=torch.float64)
    o_exp[:, :, :64], lse_exp[:, :, :64] = o_ref, lse_ref
    _assert_poisoned_match(o, o_exp, 2e-3)
    _assert_poisoned_match(lse, lse_exp, 2e-3)


@requires_mps
@pytest.mark.parametrize("which", sorted(_BIASED_UNMASKED))
def test_unmasked_biased_store_over_a_partial_tile_refuses_at_launch(which, fa_hits):
    with pytest.raises(MetalNonRecoverableError, match="Refusing unmasked FlashAttention store launch"):
        _launch_biased(_variant(_biased_fa, *_BIASED_UNMASKED[which]), 48, 2)
    assert fa_hits == []


# ---------------------------------------------------------------------------------------
# GPU: MLA (fp16, BN 32) on whole and partial tiles
# ---------------------------------------------------------------------------------------

_MLA_QN = "qn = tl.load(qn_ptr) * scale"
_MLA_QR = "qr = tl.load(Qr + z * qrz + h * qrh + m[:, None] * qrm + dr[None, :] * qrk) * scale"
_MLA_KN = "kn = tl.load(Kn + z * knz + h * knh + (s + n0)[:, None] * knn + dn[None, :] * knk)"
_MLA_KR = "kr = tl.load(Kr + z * krz + h * krh + (s + n0)[:, None] * krn + dr[None, :] * krk)"
_MLA_V = "v = tl.load(V + z * vz + h * vh + (s + n0)[:, None] * vn + dv[None, :] * vk)"
_MLA_O = "tl.store(out_ptr, acc.to(Out.dtype.element_ty))"
_ROW_MASK = ", mask=m[:, None] < N_CTX, other=0.0)"
_KEY_MASK = ", mask=(s + n0)[:, None] < N_CTX, other=0.0)"
_MLA_MASKS = {
    "q": [
        (_MLA_QN, _MLA_QN.replace("(qn_ptr)", "(qn_ptr" + _ROW_MASK)),
        (_MLA_QR, _MLA_QR.replace(") * scale", _ROW_MASK + " * scale")),
    ],
    "k": [(_MLA_KN, _MLA_KN[:-1] + _KEY_MASK), (_MLA_KR, _MLA_KR[:-1] + _KEY_MASK)],
    "v": [(_MLA_V, _MLA_V[:-1] + _KEY_MASK)],
    "o": [(_MLA_O, _MLA_O[:-1] + ", mask=m[:, None] < N_CTX)")],
}
MLA_DIMS = dict(DN=128, DR=64, DV=128)


def _mla_kernel(masked):
    from tests.test_mla_value_paths import _mla_value_path

    return _variant(_mla_value_path, *[e for key in masked for e in _MLA_MASKS[key]])


def _mla_run(fn, n_ctx, masked, seed, select=None, block_n=32):
    """Launch MODE 0 with ``n_ctx`` and return (out, float64 expectation of the SOURCE).

    ``select(row, start, column)`` is a select the source applies to each score tile:
    the query row in its tile, the first key of the key tile and the key's column in it.
    """
    z, nh, alloc = 1, 2, 160
    gen = torch.Generator().manual_seed(seed)
    t = {
        n: (torch.randn(z, nh, alloc, d, generator=gen) * 0.5).half()
        for n, d in (("qn", 128), ("qr", 64), ("kn", 128), ("kr", 64), ("v", 128))
    }
    out = torch.full((z, nh, alloc, 128), POISON, dtype=torch.float16, device="mps")
    tiles = triton.cdiv(n_ctx, 32)
    rows = tiles * 32  # the expectation is for whole 32-key tiles (block_n = 32)
    q = torch.cat([t["qn"], t["qr"]], -1).double()[:, :, :rows].clone() / 192**0.5
    k = torch.cat([t["kn"], t["kr"]], -1).double()[:, :, :rows].clone()
    v = t["v"].double()[:, :, :rows].clone()
    # A masked load reads zero past N_CTX, a plain one reads storage; the masked keys still take part.
    for key, x in (("q", q), ("k", k), ("v", v)):
        if key in masked:
            x[:, :, n_ctx:] = 0
    expected = torch.full(out.shape, POISON, dtype=torch.float64)
    keep = n_ctx if "o" in masked else rows
    scores = q @ k.transpose(-1, -2)
    if select is not None:
        at = torch.arange(rows)
        kept = select(at[:, None] % 32, at[None, :] // 32 * 32, at[None, :] % 32).expand(rows, rows)
        scores = scores.masked_fill(~kept, float("-inf"))
    expected[:, :, :keep] = (torch.softmax(scores, -1) @ v)[:, :, :keep]
    d = {n: x.to("mps") for n, x in t.items()}
    st = lambda x: tuple(x.stride())  # noqa: E731
    fn[(tiles, z * nh)](
        d["qn"], d["qr"], d["kn"], d["kr"], d["v"], out,
        *st(d["qn"]), *st(d["qr"]), *st(d["kn"]), *st(d["kr"]), *st(d["v"]), *st(out),
        z, nh, n_ctx, 32, block_n, 128, 64, 128, 0,
    )  # fmt: skip
    torch.mps.synchronize()
    return out, expected


@requires_mps
@pytest.mark.parametrize("masked", ["", "o", "kv", "qkvo"])
@pytest.mark.parametrize("n_ctx", [64, 96])
def test_mla_whole_tiles_compute(n_ctx, masked, mla_hits):
    out, expected = _mla_run(_mla_kernel(masked), n_ctx, masked, seed=1120 + n_ctx)
    assert mla_hits == [True]
    _assert_poisoned_match(out, expected, 3e-2)


@requires_mps
@pytest.mark.parametrize("masked", ["", "o", "kv", "qkvo"])
@pytest.mark.parametrize("n_ctx", [48, 80])
def test_mla_partial_tile_refuses_at_launch(n_ctx, masked, mla_hits):
    with pytest.raises(MetalNonRecoverableError, match="Refusing MLA attention launch") as info:
        _mla_run(_mla_kernel(masked), n_ctx, masked, seed=1121)
    assert f"the sequence length {n_ctx} is not a whole number of 32-row tiles" in str(info.value)
    assert mla_hits == []


# ---------------------------------------------------------------------------------------
# GPU: varlen attends whole key tiles of range(0, max_seqlen, 32)
# ---------------------------------------------------------------------------------------


def _varlen_source(q, k, v, cu_q, cu_k, heads, scale, maxs):
    """Float64 semantics of tests/test_varlen_fa_routing.py::_varlen_fwd for one launch."""
    if maxs <= 0:
        # range(0, max_seqlen, 32) runs no tile: every stored row is acc / l_i = 0 / 0.
        return torch.full(q.shape, float("nan"), dtype=torch.float64)
    out = torch.zeros(q.shape, dtype=torch.float64)
    keys = triton.cdiv(maxs, 32) * 32
    for b in range(len(cu_q) - 1):
        qs, qe, ks, ke = (int(x) for x in (cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1]))
        n = min(ke - ks, keys)
        for h in range(heads):
            s = (q[qs:qe, h].double() @ k[ks : ks + n, h].double().T) * scale
            out[qs:qe, h] = torch.softmax(s, -1) @ v[ks : ks + n, h].double()
    return out


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "lens_q,lens_k,maxs,dim",
    [
        ([48, 20], [60, 50], 40, 64),  # attends 60 and 50 keys: 40 rounds up to 64
        ([40], [100], 33, 64),  # attends 64 of 100 keys
        ([40], [100], 33, 128),
        ([48, 20], [60, 50], 64, 32),  # whole tiles: the bound is exact
        ([40], [100], 96, 64),
    ],
)
def test_varlen_attends_whole_key_tiles_below_max_seqlen(lens_q, lens_k, maxs, dim, dtype, fa_hits):
    out, expected = _run_varlen(lens_q, lens_k, maxs, dim, dtype)
    assert fa_hits == [True]
    _assert_poisoned_match(out, expected, 2e-2 if dtype == torch.float16 else 1e-3)


@requires_mps
@pytest.mark.parametrize("maxs", [0, -5])
def test_varlen_without_a_key_tile_stores_nan(maxs, fa_hits):
    out, expected = _run_varlen([48, 20], [60, 50], maxs, 64, torch.float32)
    assert fa_hits == [True]
    out = out.cpu().double()
    assert torch.equal(out != POISON, expected != POISON)
    assert bool(out[:68].isnan().all()) and bool(expected[:68].isnan().all())


def _run_varlen(lens_q, lens_k, maxs, dim, dtype, cu_dtypes=(torch.int32, torch.int32)):
    from tests.test_varlen_fa_routing import _varlen_fwd

    heads, scale = 2, 1.0 / math.sqrt(dim)
    gen = torch.Generator().manual_seed(1122 + maxs + dim)
    cu_q = torch.tensor([0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=cu_dtypes[0])
    cu_k = torch.tensor([0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=cu_dtypes[1])
    tq, tk = int(cu_q[-1]), int(cu_k[-1])
    q, k, v = (torch.randn(t, heads, dim, generator=gen).to(dtype) for t in (tq, tk, tk))
    out = torch.full((tq + 32, heads, dim), POISON, dtype=dtype, device="mps")
    ov = out[:tq]
    qd, kd, vd = q.to("mps"), k.to("mps"), v.to("mps")
    _varlen_fwd[(triton.cdiv(max(lens_q), 32), len(lens_q) * heads)](
        qd, kd, vd, ov, cu_q.to("mps"), cu_k.to("mps"),
        *qd.stride(), *kd.stride(), *vd.stride(), *ov.stride(),
        heads, maxs, scale, 32, 32, dim,
    )  # fmt: skip
    torch.mps.synchronize()
    expected = torch.full(out.shape, POISON, dtype=torch.float64)
    expected[:tq] = _varlen_source(q, k, v, cu_q, cu_k, heads, scale, maxs)
    return out, expected
