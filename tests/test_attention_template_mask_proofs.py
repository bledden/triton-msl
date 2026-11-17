"""Attention templates mask the cells the source masks, or refuse.

Every attention template rebuilds its masks from the route's own description: zero for the
operand rows past a sequence boundary, negative infinity for the keys past it, the lower
triangle when the route is causal. The lowering proved which coordinate each source mask
compared and against which bound, but not where that comparison lands in the tensor it masks,
which branch of a score select keeps the score, what the select or a masked load supplies
outside the mask, or that the V load is masked at all. On a square tile a transposed broadcast
has the same coordinate, the same bound and the same shape, so a source that masked other
cells ran on the template and returned the canonical kernel's result.

Each case below is one line of a routed kernel changed. A change that masks other cells must
refuse before anything is dispatched; an equivalent respelling must still take the template
and match a reference of the source.

Sections: the driver reason for an unmasked V load on synthetic launches (no device), the
descriptor the lowering emits (no device), then launches on the GPU for the varlen, causal
varlen, biased, triangle, MLA and backward routes.
"""

import math

import pytest
import torch
import triton

import tests.test_fa_bwd_address_witness as bwd
from tests.test_fa_biased_routing import _biased_fa, _biased_tri_fa
from tests.test_fa_launch_tile_contracts import FA_TAG, _assert_poisoned_match, _mla_run, _variant
from tests.test_mla_value_paths import _mla_value_path
from tests.test_template_scalar_abi import _lower
from tests.test_varlen_fa_routing import _ref_varlen, _run_varlen, _varlen_causal_fwd, _varlen_fwd
from triton_msl.backend._launch_signature import scalar_bytes
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

KEY_TAG = "fa_whole_key_tiles_v1"
POISON = -16384.0
NINF = 'float("-inf")'
AXIS = "does not vary along the template's axis for this role"
BOUNDARY = "the mask is not exactly the template's boundary for this role"
KEEPS = "the score select does not keep the score inside the key boundary and supply -inf"


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


# ---------------------------------------------------------------------------------------
# _fa_whole_key_tiles_reason: the sequence length is argument 2 of a synthetic launch
# ---------------------------------------------------------------------------------------


def _key_reason(keys, block=32, declared="i32", payload=None, descriptor=None):
    from triton_msl.backend.driver import _fa_whole_key_tiles_reason

    if payload is None:
        payload = scalar_bytes(keys, declared)
    return _fa_whole_key_tiles_reason(
        (KEY_TAG, 2, block, None) if descriptor is None else descriptor,
        [object(), object(), keys],
        ["*fp32", "*fp32", declared],
        [None, None, payload],
    )


@pytest.mark.parametrize("keys,block", [(32, 32), (64, 32), (96, 32), (128, 64), (0, 32)])
def test_whole_key_tiles_pass(keys, block):
    # The template supplies zero for the V rows past the length; whole tiles hold none.
    assert _key_reason(keys, block) is None


@pytest.mark.parametrize("declared", ["i32", "u32", "i64", "u64"])
@pytest.mark.parametrize("keys,past", [(48, 16), (33, 31), (95, 1), (1, 31)])
def test_partial_last_key_tile_refuses(keys, past, declared):
    reason = _key_reason(keys, declared=declared)
    assert reason is not None
    assert f"a last key tile that holds {past} rows past the sequence length {keys}" in reason
    assert "mask the V load with the key boundary or launch whole key tiles" in reason


def test_inner_contract_rides_along():
    # An unmasked store's row contract is unwrapped by the launcher after this one passes.
    assert _key_reason(64, descriptor=(KEY_TAG, 2, 32, (FA_TAG, 2, 32))) is None
    assert _key_reason(64, descriptor=[KEY_TAG, 2, 32, None]) is None


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        KEY_TAG,
        (KEY_TAG, 2, 32),
        (KEY_TAG, 2, 32, None, None),
        (FA_TAG, 2, 32, None),
        (KEY_TAG, True, 32, None),
        (KEY_TAG, 2.0, 32, None),
        (KEY_TAG, 2, 0, None),
        (KEY_TAG, 2, -32, None),
        (KEY_TAG, 2, 32.0, None),
        (KEY_TAG, 3, 32, None),
        (KEY_TAG, -1, 32, None),
    ],
    ids=repr,
)
def test_malformed_key_descriptor_refuses(descriptor):
    from triton_msl.backend.driver import _fa_whole_key_tiles_reason

    reason = _fa_whole_key_tiles_reason(
        descriptor, [object(), object(), 64], ["*fp32", "*fp32", "i32"], [None, None, scalar_bytes(64, "i32")]
    )
    assert reason is not None and reason.startswith("the key-domain descriptor is missing or malformed")


@pytest.mark.parametrize("declared", ["fp32", "*fp32", "i16", "i1"])
def test_non_integer_key_length_refuses(declared):
    reason = _key_reason(64, declared=declared, payload=b"\0" * 4)
    assert reason is not None and reason.startswith("the sequence length is not an integer argument")


@pytest.mark.parametrize("payload", [None, b"", scalar_bytes(96, "i32"), bytearray(scalar_bytes(64, "i32"))], ids=repr)
def test_key_length_payload_mismatch_refuses(payload):
    from triton_msl.backend.driver import _fa_whole_key_tiles_reason

    reason = _fa_whole_key_tiles_reason(
        (KEY_TAG, 2, 32, None), [object(), object(), 64], ["*fp32", "*fp32", "i32"], [None, None, payload]
    )
    assert reason == "the sequence length changed after its captured launch payload"


def test_key_length_is_the_captured_value_not_a_subclass_int():
    class Whole(int):
        def __mod__(self, other):
            return 0

    reason = _key_reason(Whole(48))
    assert reason is not None and "16 rows past the sequence length 48" in reason


def test_unevaluable_key_launch_refuses():
    from triton_msl.backend.driver import _fa_whole_key_tiles_reason

    reason = _fa_whole_key_tiles_reason((KEY_TAG, 2, 32, None), [object(), object(), 64], None, None)
    assert reason is not None and reason.startswith("the key domain could not be evaluated (TypeError)")


# ---------------------------------------------------------------------------------------
# Lowering: an unmasked biased V load carries the key-tile contract
# ---------------------------------------------------------------------------------------

B_Q = "q_block = tl.load(q_ptrs, mask_m[:, None])"
B_K = "kt_block = tl.load(kt_ptrs, mask_n[None, :])"
B_B = "b_block = tl.load(b_ptrs, mask_m[:, None] & mask_n[None, :])"
B_MSEL = "scores = tl.where(m_block[None, :], neg_inf, scores)"
B_BSEL = "scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, neg_inf)"
B_V = "v_block = tl.load(v_ptrs, mask_n[:, None])"
B_OUT = "tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[:, None])"
B_LSE = "tl.store(lse_ptrs, lse, mask=mask_m)"
B_V_UNMASKED = (B_V, "v_block = tl.load(v_ptrs)")


def _lower_biased(*edits):
    fn = _variant(_biased_fa, *edits)
    const = dict(DIM=32, BLOCK_M=32, BLOCK_N=32, BAD_TEMP=False, I3D=False)
    sig = {}
    for name in fn.arg_names:
        if name in const:
            continue
        if name.endswith("_ptr"):
            sig[name] = "*i8" if name == "mask_ptr" else "*fp32"
        else:
            sig[name] = "fp32" if name in ("sm_scale", "neg_inf") else "i32"
    lowerer = _lower(fn, sig, const)
    lowerer.lower()
    tag, _unsigned, _signed, inner = lowerer._tail_access_bounds
    assert tag == "attention_unsigned_scalars_v1"
    return inner, fn.arg_names.index("N")


def test_masked_biased_v_load_carries_no_key_tile_descriptor():
    inner, _n = _lower_biased()
    assert inner is None


def test_unmasked_biased_v_load_carries_the_key_tile_descriptor():
    inner, n = _lower_biased(B_V_UNMASKED)
    assert inner == (KEY_TAG, n, 32, None)


def test_unmasked_biased_v_load_and_store_carry_both_descriptors():
    inner, n = _lower_biased(B_V_UNMASKED, (B_OUT, "tl.store(o_ptrs, normalize.to(q_block.type.element_ty))"))
    assert inner == (KEY_TAG, n, 32, (FA_TAG, n, 32))


# ---------------------------------------------------------------------------------------
# GPU: varlen (lengths that are not tile multiples; query and key lengths differ)
# ---------------------------------------------------------------------------------------

VL_Q = "q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)"
VL_K = "k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)"
VL_SEL = f"qk = tl.where(kn[None, :] < seqlen_k, qk, {NINF})"
VL_V = "v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)"
VL_OUT = "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)"
VC_TRI = "causal = offs_m[:, None] >= kn[None, :]"
VC_SEL = f"qk = tl.where((kn[None, :] < seqlen_k) & causal, qk, {NINF})"
LENS_Q, LENS_K = [37, 50, 21], [45, 29, 60]
IND = "\n        "

# name -> (edit, head dim, refusal fragment). A transposed operand mask only broadcasts on a
# square tile, so those run at head dim 32 (the varlen template serves it).
VARLEN_REFUSED = {
    "select_dropped": ((VL_SEL, "qk = qk * 1.0"), 64, "expected the key-boundary select on the score, found 0"),
    "select_one_key_short": ((VL_SEL, f"qk = tl.where(kn[None, :] < seqlen_k - 1, qk, {NINF})"), 64, BOUNDARY),
    "select_polarity": ((VL_SEL, f"qk = tl.where(kn[None, :] < seqlen_k, {NINF}, qk)"), 64, KEEPS),
    "select_inverted_polarity": ((VL_SEL, f"qk = tl.where(kn[None, :] >= seqlen_k, qk, {NINF})"), 64, KEEPS),
    "select_fills_zero": ((VL_SEL, "qk = tl.where(kn[None, :] < seqlen_k, qk, 0.0)"), 64, KEEPS),
    "select_fills_finite": ((VL_SEL, "qk = tl.where(kn[None, :] < seqlen_k, qk, -1e9)"), 64, KEEPS),
    "select_transposed": ((VL_SEL, f"qk = tl.where(kn[:, None] < seqlen_k, qk, {NINF})"), 64, AXIS),
    "select_on_the_row_bound": ((VL_SEL, f"qk = tl.where(offs_m[:, None] < seqlen_q, qk, {NINF})"), 64, "leaves ['Q']"),
    "select_key_boundary_twice": (
        (VL_SEL, VL_SEL + IND + f"qk = tl.where(seqlen_k > kn[None, :], qk, {NINF})"),
        64,
        "leaves ['K', 'K']",
    ),
    "q_mask_transposed": ((VL_Q, "q = tl.load(q_ptrs, mask=offs_m[None, :] < seqlen_q, other=0.0)"), 32, AXIS),
    "k_mask_transposed": ((VL_K, "k = tl.load(k_ptrs, mask=kn[None, :] < seqlen_k, other=0.0)"), 32, AXIS),
    "v_mask_transposed": ((VL_V, "v = tl.load(v_ptrs, mask=kn[None, :] < seqlen_k, other=0.0)"), 32, AXIS),
    "store_mask_transposed": (
        (VL_OUT, "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[None, :] < seqlen_q)"),
        32,
        AXIS,
    ),
    "v_unmasked": ((VL_V, "v = tl.load(v_ptrs)"), 64, "the V load is unmasked"),
}

VARLEN_ADMITTED = {
    "canonical": None,
    "select_bound_first": (VL_SEL, f"qk = tl.where(seqlen_k > kn[None, :], qk, {NINF})"),
    "select_inverted": (VL_SEL, f"qk = tl.where(kn[None, :] >= seqlen_k, {NINF}, qk)"),
    "select_compare_then_expand": (VL_SEL, f"qk = tl.where((kn < seqlen_k)[None, :], qk, {NINF})"),
    "select_with_the_row_boundary": (
        VL_SEL,
        f"qk = tl.where((kn[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q), qk, {NINF})",
    ),
    "select_then_row_select": (VL_SEL, VL_SEL + IND + f"qk = tl.where(offs_m[:, None] < seqlen_q, qk, {NINF})"),
    "q_mask_bound_first": (VL_Q, "q = tl.load(q_ptrs, mask=seqlen_q > offs_m[:, None], other=0.0)"),
    "k_mask_bound_first": (VL_K, "k = tl.load(k_ptrs, mask=seqlen_k > kn[:, None], other=0.0)"),
    "v_mask_bound_first": (VL_V, "v = tl.load(v_ptrs, mask=seqlen_k > kn[:, None], other=0.0)"),
    "store_mask_bound_first": (
        VL_OUT,
        "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=seqlen_q > offs_m[:, None])",
    ),
    "store_mask_compare_then_expand": (
        VL_OUT,
        "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m < seqlen_q)[:, None])",
    ),
}

CAUSAL_REFUSED = {
    "triangle_transposed": ((VC_TRI, "causal = offs_m[None, :] >= kn[:, None]"), "not the template's lower triangle"),
    "triangle_row_plus_1": (
        (VC_TRI, "causal = (offs_m + 1)[:, None] >= kn[None, :]"),
        "not the template's lower triangle",
    ),
    "triangle_key_plus_1": (
        (VC_TRI, "causal = offs_m[:, None] >= (kn + 1)[None, :]"),
        "not the template's lower triangle",
    ),
    "triangle_or_boundary": ((VC_SEL, f"qk = tl.where((kn[None, :] < seqlen_k) | causal, qk, {NINF})"), "Score"),
    "triangle_without_the_boundary": ((VC_SEL, f"qk = tl.where(causal, qk, {NINF})"), "leaves ['causal']"),
    "triangle_and_its_transpose": (
        (VC_SEL, f"qk = tl.where((kn[None, :] < seqlen_k) & causal & (offs_m[None, :] >= kn[:, None]), qk, {NINF})"),
        "not the template's lower triangle",
    ),
    "boundary_one_key_short": ((VC_SEL, f"qk = tl.where((kn[None, :] < seqlen_k - 1) & causal, qk, {NINF})"), BOUNDARY),
    "boundary_transposed": ((VC_SEL, f"qk = tl.where((kn[:, None] < seqlen_k) & causal, qk, {NINF})"), AXIS),
    "boundary_is_the_row_bound": (
        (VC_SEL, f"qk = tl.where((offs_m[:, None] < seqlen_q) & causal, qk, {NINF})"),
        "leaves ['Q', 'causal']",
    ),
    "select_polarity": ((VC_SEL, f"qk = tl.where((kn[None, :] < seqlen_k) & causal, {NINF}, qk)"), KEEPS),
    "select_fills_zero": ((VC_SEL, "qk = tl.where((kn[None, :] < seqlen_k) & causal, qk, 0.0)"), KEEPS),
}

CAUSAL_ADMITTED = {
    "canonical": None,
    "triangle_first": (VC_SEL, f"qk = tl.where(causal & (kn[None, :] < seqlen_k), qk, {NINF})"),
    "with_the_row_boundary": (
        VC_SEL,
        f"qk = tl.where((kn[None, :] < seqlen_k) & causal & (offs_m[:, None] < seqlen_q), qk, {NINF})",
    ),
    "row_boundary_first": (
        VC_SEL,
        f"qk = tl.where((offs_m[:, None] < seqlen_q) & (causal & (kn[None, :] < seqlen_k)), qk, {NINF})",
    ),
    "boundary_bound_first": (VC_SEL, f"qk = tl.where((seqlen_k > kn[None, :]) & causal, qk, {NINF})"),
    "boundary_compare_then_expand": (VC_SEL, f"qk = tl.where((kn < seqlen_k)[None, :] & causal, qk, {NINF})"),
    "two_selects": (
        VC_SEL,
        f"qk = tl.where(kn[None, :] < seqlen_k, qk, {NINF})" + IND + f"qk = tl.where(causal, qk, {NINF})",
    ),
    "two_selects_triangle_first": (
        VC_SEL,
        f"qk = tl.where(causal, qk, {NINF})" + IND + f"qk = tl.where(kn[None, :] < seqlen_k, qk, {NINF})",
    ),
}


def _varlen_refuses(kernel, edit, dim, fragment, fa_hits):
    with pytest.raises(MetalNonRecoverableError) as info:
        _run_varlen(_variant(kernel, edit), LENS_Q, LENS_K, 2, dim, torch.float32, 1.0 / math.sqrt(dim))
    assert fragment in str(info.value)
    assert fa_hits == []


def _varlen_computes(kernel, edit, dim, causal, fa_hits):
    fn = _variant(kernel, *([] if edit is None else [edit]))
    scale = 1.0 / math.sqrt(dim)
    q, k, v, o, cu_q, cu_k = _run_varlen(fn, LENS_Q, LENS_K, 2, dim, torch.float32, scale, seed=1123)
    assert fa_hits == [True]
    torch.testing.assert_close(o, _ref_varlen(q, k, v, cu_q, cu_k, 2, dim, scale, causal=causal), atol=1e-3, rtol=1e-3)


@requires_mps
@pytest.mark.parametrize("name", sorted(VARLEN_REFUSED))
def test_varlen_mask_on_other_cells_refuses(name, fa_hits):
    edit, dim, fragment = VARLEN_REFUSED[name]
    _varlen_refuses(_varlen_fwd, edit, dim, fragment, fa_hits)


@requires_mps
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("name", sorted(VARLEN_ADMITTED))
def test_varlen_equivalent_mask_computes(name, dim, fa_hits):
    _varlen_computes(_varlen_fwd, VARLEN_ADMITTED[name], dim, False, fa_hits)


@requires_mps
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("name", sorted(CAUSAL_REFUSED))
def test_causal_varlen_mask_on_other_cells_refuses(name, dim, fa_hits):
    edit, fragment = CAUSAL_REFUSED[name]
    _varlen_refuses(_varlen_causal_fwd, edit, dim, fragment, fa_hits)


@requires_mps
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("name", sorted(CAUSAL_ADMITTED))
def test_causal_varlen_equivalent_mask_computes(name, dim, fa_hits):
    _varlen_computes(_varlen_causal_fwd, CAUSAL_ADMITTED[name], dim, True, fa_hits)


# ---------------------------------------------------------------------------------------
# GPU: biased forward on a partial tile (48 of 64 launched rows and keys)
# ---------------------------------------------------------------------------------------


def _launch_biased(fn, n_ctx, tiles=2, dim=32, seed=1124):
    """Launch the 2-D biased kernel; returns (out, lse, expected out, expected lse) with poison."""
    gen = torch.Generator().manual_seed(seed)
    z, h, rows = 1, 2, tiles * 32
    q, k, v = [torch.randn(z, h, rows, dim, generator=gen) for _ in range(3)]
    b = torch.randn(z, h, rows, rows, generator=gen)
    mask = (torch.rand(z, h, rows, generator=gen) < 0.25).to(torch.uint8)
    mask[..., 0] = 0  # no query row is left without a key
    o = torch.full((z, h, rows, dim), POISON, device="mps")
    lse = torch.full((z, h, rows), POISON, device="mps")
    dev = [x.to("mps") for x in (q, k, v, b, mask)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    args = [o, *st(o), lse, *st(lse)]
    for t in dev:
        args += [t, *st(t)]
    fn[(tiles, z * h)](*args, dim**-0.5, -1e9, z, h, n_ctx, dim, 32, 32)
    torch.mps.synchronize()
    n = n_ctx
    raw = dim**-0.5 * (q.double() @ k.double().transpose(-1, -2)) + b.double()
    raw = raw.masked_fill(mask.bool()[:, :, None, :], float("-inf"))[:, :, :n, :n]
    o_exp = torch.full(o.shape, POISON, dtype=torch.float64)
    lse_exp = torch.full(lse.shape, POISON, dtype=torch.float64)
    o_exp[:, :, :n] = torch.softmax(raw, -1) @ v.double()[:, :, :n]
    lse_exp[:, :, :n] = torch.logsumexp(raw, -1)
    return o, lse, o_exp, lse_exp


BIASED_REFUSED = {
    "mask_select_transposed": (
        (B_MSEL, "scores = tl.where(m_block[:, None], neg_inf, scores)"),
        "the loaded mask is not broadcast along the score's key axis",
    ),
    "boundary_select_transposed": (
        (B_BSEL, "scores = tl.where(mask_m[None, :] & mask_n[:, None], scores, neg_inf)"),
        AXIS,
    ),
    "q_mask_transposed": ((B_Q, "q_block = tl.load(q_ptrs, mask_m[None, :], other=0.0)"), AXIS),
    "k_mask_transposed": ((B_K, "kt_block = tl.load(kt_ptrs, mask_n[:, None], other=0.0)"), AXIS),
    "bias_mask_transposed": (
        (B_B, "b_block = tl.load(b_ptrs, mask_m[None, :] & mask_n[:, None], other=0.0)"),
        AXIS,
    ),
    "v_mask_transposed": ((B_V, "v_block = tl.load(v_ptrs, mask_n[None, :], other=0.0)"), AXIS),
    "store_mask_transposed": (
        (B_OUT, "tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[None, :])"),
        "a non-tile-boundary output store mask",
    ),
}

BIASED_ADMITTED = {
    "canonical": None,
    "q_mask_bound_first": (B_Q, "q_block = tl.load(q_ptrs, (N > m_idxs)[:, None])"),
    "k_mask_bound_first": (B_K, "kt_block = tl.load(kt_ptrs, (N > (n_idxs + start_n))[None, :])"),
    "v_mask_bound_first": (B_V, "v_block = tl.load(v_ptrs, (N > (n_idxs + start_n))[:, None])"),
    "store_mask_bound_first": (
        B_OUT,
        "tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=(N > m_idxs)[:, None])",
    ),
    "lse_mask_bound_first": (B_LSE, "tl.store(lse_ptrs, lse, mask=N > m_idxs)"),
    "boundary_select_keys_first": (
        B_BSEL,
        "scores = tl.where(mask_n[None, :] & mask_m[:, None], scores, neg_inf)",
    ),
    "q_other_zero": (B_Q, "q_block = tl.load(q_ptrs, mask_m[:, None], other=0.0)"),
    "v_other_zero": (B_V, "v_block = tl.load(v_ptrs, mask_n[:, None], other=0.0)"),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(BIASED_REFUSED))
def test_biased_mask_on_other_cells_refuses(name, fa_hits):
    edit, fragment = BIASED_REFUSED[name]
    with pytest.raises(MetalNonRecoverableError) as info:
        _launch_biased(_variant(_biased_fa, edit), 48)
    assert fragment in str(info.value)
    assert fa_hits == []


@requires_mps
@pytest.mark.parametrize("name", sorted(BIASED_ADMITTED))
def test_biased_equivalent_mask_computes(name, fa_hits):
    edit = BIASED_ADMITTED[name]
    o, lse, o_exp, lse_exp = _launch_biased(_variant(_biased_fa, *([] if edit is None else [edit])), 48)
    assert fa_hits == [True]
    _assert_poisoned_match(o, o_exp, 2e-3)
    _assert_poisoned_match(lse, lse_exp, 2e-3)


@requires_mps
def test_unmasked_biased_v_load_on_whole_key_tiles_computes(fa_hits):
    o, lse, o_exp, lse_exp = _launch_biased(_variant(_biased_fa, B_V_UNMASKED), 64)
    assert fa_hits == [True]
    _assert_poisoned_match(o, o_exp, 2e-3)
    _assert_poisoned_match(lse, lse_exp, 2e-3)


@requires_mps
@pytest.mark.parametrize("n_ctx", [48, 33])
def test_unmasked_biased_v_load_over_a_partial_key_tile_refuses_at_launch(n_ctx, fa_hits):
    fn = _variant(_biased_fa, B_V_UNMASKED)
    with pytest.raises(MetalNonRecoverableError, match="Refusing unmasked FlashAttention V load launch") as info:
        _launch_biased(fn, n_ctx)
    assert f"{64 - n_ctx} rows past the sequence length {n_ctx}" in str(info.value)
    assert fa_hits == []
    # The same compiled kernel still runs the launches the template reproduces.
    o, lse, o_exp, lse_exp = _launch_biased(fn, 64)
    assert fa_hits == [True]
    _assert_poisoned_match(o, o_exp, 2e-3)


# ---------------------------------------------------------------------------------------
# GPU: the triangle kernel (3-D grid, mask shared across heads) on the same template
# ---------------------------------------------------------------------------------------

T_MSEL = "scores = tl.where(m_block[None, :], neg_inf, scores)"
T_Q = "q_block = tl.load(q_ptrs, mask_j[:, None])"


def _launch_triangle(fn, n=48, dim=32, seed=1125):
    gen = torch.Generator().manual_seed(seed)
    hc, hh, inst, rows = 4, 2, 3, 64
    q, k, v = [torch.randn(hc, inst, rows, dim, generator=gen) for _ in range(3)]
    b = torch.randn(hc, rows, rows, generator=gen)
    mask = (torch.rand(hc // hh, inst, rows, generator=gen) < 0.25).to(torch.uint8)
    mask[..., 0] = 0
    o = torch.full((hc, inst, rows, dim), POISON, device="mps")
    lse = torch.full((hc, inst, rows), POISON, device="mps")
    dev = [x.to("mps") for x in (q, k, v, b, mask)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    args = [o, *st(o), lse, *st(lse)]
    for t in dev:
        args += [t, *st(t)]
    fn[(triton.cdiv(n, 32), inst, hc)](*args, dim**-0.5, -1e9, n, hh, dim, 32, 32)
    torch.mps.synchronize()
    raw = dim**-0.5 * (q.double() @ k.double().transpose(-1, -2)) + b.double()[:, None]
    masked = mask.bool()[torch.arange(hc) // hh]
    raw = raw.masked_fill(masked[:, :, None, :], float("-inf"))[:, :, :n, :n]
    o_exp = torch.full(o.shape, POISON, dtype=torch.float64)
    lse_exp = torch.full(lse.shape, POISON, dtype=torch.float64)
    o_exp[:, :, :n] = torch.softmax(raw, -1) @ v.double()[:, :, :n]
    lse_exp[:, :, :n] = torch.logsumexp(raw, -1)
    return o, lse, o_exp, lse_exp


@requires_mps
def test_triangle_mask_select_transposed_refuses(fa_hits):
    fn = _variant(_biased_tri_fa, (T_MSEL, "scores = tl.where(m_block[:, None], neg_inf, scores)"))
    with pytest.raises(MetalNonRecoverableError, match="not broadcast along the score's key axis"):
        _launch_triangle(fn)
    assert fa_hits == []


@requires_mps
@pytest.mark.parametrize(
    "edit", [None, (T_Q, "q_block = tl.load(q_ptrs, (N > j_idxs)[:, None])")], ids=["canonical", "q"]
)
def test_triangle_equivalent_mask_computes(edit, fa_hits):
    o, lse, o_exp, lse_exp = _launch_triangle(_variant(_biased_tri_fa, *([] if edit is None else [edit])))
    assert fa_hits == [True]
    _assert_poisoned_match(o, o_exp, 2e-3)
    _assert_poisoned_match(lse, lse_exp, 2e-3)


# ---------------------------------------------------------------------------------------
# GPU: MLA (fp16). The canonical source has no masks; a causal select is one added line.
# ---------------------------------------------------------------------------------------

MLA_QK = "qk = tl.dot(qn, tl.trans(kn).to(qn.dtype)) + tl.dot(qr, tl.trans(kr).to(qr.dtype))"
MLA_KN = "kn = tl.load(Kn + z * knz + h * knh + (s + n0)[:, None] * knn + dn[None, :] * knk)"


def _mla_select(expr):
    return (MLA_QK, MLA_QK + IND + f"qk = tl.where({expr}, qk, {NINF})")


def _mla_causal_run(fn, n_ctx=64, seed=1126):
    z, nh = 1, 2
    gen = torch.Generator().manual_seed(seed)
    t = {
        n: (torch.randn(z, nh, n_ctx, d, generator=gen) * 0.5).half()
        for n, d in (("qn", 128), ("qr", 64), ("kn", 128), ("kr", 64), ("v", 128))
    }
    out = torch.full((z, nh, n_ctx, 128), POISON, dtype=torch.float16, device="mps")
    q = torch.cat([t["qn"], t["qr"]], -1).double() / 192**0.5
    k = torch.cat([t["kn"], t["kr"]], -1).double()
    rows = torch.arange(n_ctx)
    scores = (q @ k.transpose(-1, -2)).masked_fill(rows[:, None] < rows[None, :], float("-inf"))
    expected = torch.softmax(scores, -1) @ t["v"].double()
    d = {n: x.to("mps") for n, x in t.items()}
    st = lambda x: tuple(x.stride())  # noqa: E731
    fn[(n_ctx // 32, z * nh)](
        d["qn"], d["qr"], d["kn"], d["kr"], d["v"], out,
        *st(d["qn"]), *st(d["qr"]), *st(d["kn"]), *st(d["kr"]), *st(d["v"]), *st(out),
        z, nh, n_ctx, 32, 32, 128, 64, 128, 0,
    )  # fmt: skip
    torch.mps.synchronize()
    return out, expected


@requires_mps
def test_mla_transposed_causal_select_refuses(mla_hits):
    fn = _variant(_mla_value_path, _mla_select("m[None, :] >= (s + n0)[:, None]"))
    with pytest.raises(MetalNonRecoverableError):
        _mla_causal_run(fn)
    assert mla_hits == []


@requires_mps
def test_mla_causal_select_computes(mla_hits):
    out, expected = _mla_causal_run(_variant(_mla_value_path, _mla_select("m[:, None] >= (s + n0)[None, :]")))
    assert mla_hits == [True]
    _assert_poisoned_match(out, expected, 3e-2)


@requires_mps
def test_mla_key_mask_bound_first_computes(mla_hits):
    fn = _variant(_mla_value_path, (MLA_KN, MLA_KN[:-1] + ", mask=N_CTX > (s + n0)[:, None], other=0.0)"))
    out, expected = _mla_run(fn, 64, "", seed=1127)
    assert mla_hits == [True]
    _assert_poisoned_match(out, expected, 3e-2)


# ---------------------------------------------------------------------------------------
# GPU: biased backward on a partial tile (48 rows): what a masked load supplies, and the lse mask
# ---------------------------------------------------------------------------------------

OTHER = "every masked load's `other` being absent or zero"
KV_Q = "q_block = tl.load(q_ptrs, mask_j[:, None])"
KV_L = "row_max = tl.load(l_ptrs, mask=mask_j)"
KV_DO = "do = tl.load(do_ptrs, mask_j[:, None])"
KV_D = "delta = tl.load(d_ptrs, mask_j)"
Q_K = "k_block = tl.load(k_ptrs, mask_k[:, None])"
Q_VT = "vt_block = tl.load(vt_ptrs, mask_k[None, :])"
NAN = "other=float('nan'))"

BACKWARD_REFUSED = {
    "kv_q_other_nan": ("kv", (KV_Q, KV_Q[:-1] + ", " + NAN), OTHER),
    "kv_do_other_nan": ("kv", (KV_DO, KV_DO[:-1] + ", " + NAN), OTHER),
    "kv_delta_other_nan": ("kv", (KV_D, KV_D[:-1] + ", " + NAN), OTHER),
    "kv_lse_other_minus_inf": ("kv", (KV_L, KV_L[:-1] + ", other=float('-inf'))"), OTHER),
    "kv_lse_unmasked": (
        "kv",
        (KV_L, "row_max = tl.load(l_ptrs)"),
        "the lse load masked by exactly the query-row boundary",
    ),
    "q_k_other_nan": ("q", (Q_K, Q_K[:-1] + ", " + NAN), OTHER),
    "q_vt_other_nan": ("q", (Q_VT, Q_VT[:-1] + ", " + NAN), OTHER),
}

BACKWARD_ADMITTED = {
    "kv_q_other_zero": ("kv", (KV_Q, KV_Q[:-1] + ", other=0.0)")),
    "kv_lse_mask_bound_first": ("kv", (KV_L, "row_max = tl.load(l_ptrs, mask=N > (j_idxs + start_j))")),
    "q_k_other_zero": ("q", (Q_K, Q_K[:-1] + ", other=0.0)")),
}


@pytest.fixture
def bwd_hits(fa_hits):
    return fa_hits


@requires_mps
@pytest.mark.parametrize("name", sorted(BACKWARD_REFUSED))
def test_backward_load_that_supplies_another_value_refuses(name, bwd_hits):
    kind, edit, fragment = BACKWARD_REFUSED[name]
    with pytest.raises(MetalNonRecoverableError) as info:
        bwd._launch(kind, bwd._variant(bwd.KERNELS[kind], edit), bwd._inputs(kind, seed=1128))
    assert fragment in str(info.value)
    assert bwd_hits == []


@requires_mps
@pytest.mark.parametrize("name", sorted(BACKWARD_ADMITTED))
def test_backward_equivalent_load_computes(name, bwd_hits):
    kind, edit = BACKWARD_ADMITTED[name]
    t = bwd._inputs(kind, seed=1129)
    outputs = bwd._launch(kind, bwd._variant(bwd.KERNELS[kind], edit), t)
    assert bwd_hits == [True]
    for got, expected in zip(outputs, bwd._expected(kind, {}, t)):
        got = got.cpu().double()
        written = expected != bwd.POISON
        assert torch.equal(got != bwd.POISON, written)
        scale = max(1.0, float(expected[written].abs().max()))
        torch.testing.assert_close(got[written], expected[written], atol=2e-4 * scale, rtol=2e-3)
