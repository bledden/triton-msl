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
from triton.compiler.errors import CompilationError

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


def _lower_biased(*edits, types=None):
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
    sig.update(types or {})
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


@pytest.fixture
def biased_makers(monkeypatch):
    """Record which attention template the biased lowering asks for, and its head-dim slice."""
    import triton_msl.codegen._msl_templates as templates

    class Hits(list):
        """The (template, key block or slice) pairs; ``calls`` keeps each maker call for replay."""

    hits = Hits()
    hits.calls = []
    for name, label in (
        ("make_flash_attention_kernel_simdgroup", "simd"),
        ("make_flash_attention_kernel_tiled", "tiled"),
    ):
        real = getattr(templates, name)

        def spy(*args, _real=real, _label=label, **kwargs):
            hits.append((_label, kwargs.get("sentinel_kv_block") if _label == "simd" else kwargs.get("Dc")))
            hits.calls.append((_real, args, kwargs))
            return _real(*args, **kwargs)

        monkeypatch.setattr(templates, name, spy)
    return hits


def _lower_biased_block(float_type, dim, block_n):
    # A launch on contiguous tensors specializes the head-dim strides to the constant 1.
    const = dict(DIM=dim, BLOCK_M=32, BLOCK_N=block_n, BAD_TEMP=False, I3D=False, o_sk=1, q_sk=1, k_sk=1, v_sk=1)
    sig = {}
    for name in _biased_fa.arg_names:
        if name in const:
            continue
        if name.endswith("_ptr"):
            sig[name] = "*i8" if name == "mask_ptr" else "*fp32" if name == "lse_ptr" else f"*{float_type}"
        else:
            sig[name] = "fp32" if name in ("sm_scale", "neg_inf") else "i32"
    _lower(_variant(_biased_fa), sig, const).lower()


@pytest.mark.parametrize(
    "float_type,dim,block_n,expected",
    [("fp32", dim, block_n, ("simd", block_n)) for dim in (32, 64, 128) for block_n in (8, 16, 32, 64, 128, 256)]
    + [
        ("fp16", 32, 32, ("tiled", 32)),
        ("fp16", 64, 64, ("simd", 64)),
        ("fp16", 128, 32, ("tiled", 64)),
        ("fp16", 32, 128, ("tiled", 16)),
        ("fp16", 64, 128, ("tiled", 8)),
        ("bf16", 32, 128, ("tiled", 16)),
        ("bf16", 64, 64, ("tiled", 32)),
        ("bf16", 128, 64, ("tiled", 16)),
    ],
)
def test_biased_key_block_template_and_slice(float_type, dim, block_n, expected, biased_makers):
    """fp32 runs on the 64-wide MMA template for every key block that divides 64 or is a whole
    number of 64-key tiles. A half type rounds its probabilities per source block, so it takes
    that template only at a 64-key block and otherwise the tiled one with a slice that fits."""
    _lower_biased_block(float_type, dim, block_n)
    assert biased_makers == [expected]


@pytest.mark.parametrize(
    "float_type,dim,block_n,size",
    [("fp16", 128, 128, 37120), ("bf16", 128, 128, 37120), ("fp16", 32, 256, 45312), ("bf16", 64, 256, 49408)],
)
def test_biased_key_block_too_large_for_threadgroup_memory_refuses(float_type, dim, block_n, size, biased_makers):
    # The size named is the working set at the narrowest slice the template stages.
    with pytest.raises(MetalNonRecoverableError, match=f"needs at least {size} bytes of threadgroup memory"):
        _lower_biased_block(float_type, dim, block_n)
    assert biased_makers == []


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        (dict(runtime_neg_inf=False), "requires the biased runtime-sentinel route"),
        (dict(bias=False, mask=False), "requires the biased runtime-sentinel route"),
        (dict(sentinel_kv_block=128, causal=True), "several tiles requires a non-causal source"),
        (dict(sentinel_kv_block=48), "positive divisor of the staging block"),
        (dict(sentinel_kv_block=96), "positive divisor of the staging block"),
        (dict(sentinel_kv_block=0), "positive divisor of the staging block"),
    ],
    ids=lambda v: "-".join(f"{k}={x}" for k, x in v.items()) if isinstance(v, dict) else None,
)
def test_simd_maker_refuses_a_sentinel_block_it_cannot_walk(kwargs, fragment, biased_makers):
    """The route's own maker call with one argument changed: the maker, not only the gate in
    the lowering, refuses a source key block its 64-wide tile cannot replay."""
    _lower_biased_block("fp32", 64, 32)
    ((real, args, route_kwargs),) = biased_makers.calls
    assert biased_makers == [("simd", 32)]
    with pytest.raises(ValueError, match=fragment):
        real(*args, **{**route_kwargs, **kwargs})


def test_biased_lse_of_another_float_type_refuses():
    with pytest.raises(MetalNonRecoverableError, match="log-sum-exp tensor is not fp32, fp16 or bf16"):
        _lower_biased(types={"lse_ptr": "*fp64"})


# ---------------------------------------------------------------------------------------
# GPU: varlen (lengths that are not tile multiples; query and key lengths differ)
# ---------------------------------------------------------------------------------------

VL_Q = "q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)"
VL_K = "k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)"
VL_SEL = f"qk = tl.where(kn[None, :] < seqlen_k, qk, {NINF})"
VL_V = "v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)"
VL_OUT = "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)"
VL_ALPHA = "alpha = tl.exp(m_i - m_new)"
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
    # The finite floats that equal the fp16, bf16 and fp32 bit words of -inf.
    "select_fills_fp16_inf_word": ((VL_SEL, "qk = tl.where(kn[None, :] < seqlen_k, qk, 64512.0)"), 64, KEEPS),
    "select_fills_bf16_inf_word": ((VL_SEL, "qk = tl.where(kn[None, :] < seqlen_k, qk, 65408.0)"), 64, KEEPS),
    "select_fills_fp32_inf_word": ((VL_SEL, "qk = tl.where(kn[None, :] < seqlen_k, qk, 4286578688.0)"), 64, KEEPS),
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
    "length_signed_in_one_mask_unsigned_in_another": (
        (VL_V, "v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k.to(tl.uint32), other=0.0)"),
        64,
        "compares one sequence length as signed in one mask and unsigned in another",
    ),
    "select_off_the_score_path": (
        (VL_ALPHA, VL_ALPHA + IND + f"alpha = tl.where(offs_m < seqlen_q, alpha, {NINF})"),
        64,
        "a sequence-boundary select is not on the path from the QK dot to the softmax",
    ),
    "store_of_rank_3": (
        (
            VL_OUT,
            "tl.store(o_ptrs[:, :, None], acc.to(Out.dtype.element_ty)[:, :, None], "
            "mask=(offs_m[:, None] < seqlen_q)[:, :, None])",
        ),
        64,
        "Out value path is not exactly reproduced by the selected template (a masked tensor of rank 3",
    ),
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
    "select_fills_fp16_inf_word": (
        (VC_SEL, "qk = tl.where((kn[None, :] < seqlen_k) & causal, qk, 64512.0)"),
        KEEPS,
    ),
    "second_select_fills_fp16_inf_word": (
        (VC_SEL, f"qk = tl.where(kn[None, :] < seqlen_k, qk, {NINF})" + IND + "qk = tl.where(causal, qk, 64512.0)"),
        KEEPS,
    ),
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
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize(
    "cu_q,cu_k",
    [([0, 37, 58, 53], [0, 45, 74, 144]), ([0, 37, 87, 108], [0, 50, 124, 114]), ([0, 37, 37, 58], [0, 45, 45, 115])],
    ids=["negative_query_length", "negative_key_length", "empty_sequence"],
)
def test_varlen_decreasing_cumulative_lengths_select_nothing(cu_q, cu_k, dim, causal, fa_hits):
    """The source's signed ``index < length`` selects no row and no key of a negative length.

    A sequence with rows and no keys has every score -inf: the source's own result is NaN.
    A negative query length read as unsigned stored 128 rows past the last sequence.
    """
    heads, scale = 2, 1.0 / math.sqrt(dim)
    gen = torch.Generator().manual_seed(1131)
    # Out is longer than the sequences: a row stored past them lands in poison, not outside it.
    rows = max(cu_q) + 160
    q = torch.randn(rows, heads, dim, generator=gen)
    k, v = [torch.randn(max(cu_k), heads, dim, generator=gen) for _ in range(2)]
    o = torch.full((rows, heads, dim), POISON, device="mps")
    cq, ck = [torch.tensor(c, dtype=torch.int32, device="mps") for c in (cu_q, cu_k)]
    qd, kd, vd = [t.to("mps") for t in (q, k, v)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    (_varlen_causal_fwd if causal else _varlen_fwd)[(4, 3 * heads)](
        qd, kd, vd, o, cq, ck, *st(qd), *st(kd), *st(vd), *st(o), heads, 100, scale, 32, 32, dim
    )
    torch.mps.synchronize()
    assert fa_hits == [True]
    expected = torch.full(o.shape, POISON, dtype=torch.float64)
    for b in range(3):
        (q0, q1), (k0, k1) = cu_q[b : b + 2], cu_k[b : b + 2]
        if q1 <= q0:
            continue
        if k1 <= k0:
            expected[q0:q1] = float("nan")
            continue
        scores = scale * torch.einsum("qhd,khd->hqk", q[q0:q1].double(), k[k0:k1].double())
        if causal:
            scores = scores.masked_fill(torch.arange(q1 - q0)[:, None] < torch.arange(k1 - k0)[None, :], float("-inf"))
        expected[q0:q1] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), v[k0:k1].double())
    got = o.cpu().double()
    assert torch.equal(got != POISON, expected != POISON)
    assert torch.equal(got.isnan(), expected.isnan())
    live = (expected != POISON) & ~expected.isnan()
    torch.testing.assert_close(got[live], expected[live], atol=1e-3, rtol=1e-3)


def _varlen_whole_span(cu_q, cu_k, dtype, dim, causal, fa_hits):
    """Launch 64 rows per sequence over the 64 keys the loop visits; compare with the source.

    A length that is above every launched index (wrapped, or beyond 32 bits) computes every
    row of the launched tiles over every key the loop visits.
    """
    heads, span, scale = 2, 64, 1.0 / math.sqrt(dim)
    gen = torch.Generator().manual_seed(1132)
    q = torch.randn(max(cu_q[:-1]) + span + 8, heads, dim, generator=gen)
    k, v = [torch.randn(max(cu_k[:-1]) + span + 8, heads, dim, generator=gen) for _ in range(2)]
    o = torch.full(q.shape, POISON, device="mps")
    cq, ck = [torch.tensor(c, dtype=dtype).to("mps") for c in (cu_q, cu_k)]
    qd, kd, vd = [t.to("mps") for t in (q, k, v)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    (_varlen_causal_fwd if causal else _varlen_fwd)[(span // 32, (len(cu_q) - 1) * heads)](
        qd, kd, vd, o, cq, ck, *st(qd), *st(kd), *st(vd), *st(o), heads, span, scale, 32, 32, dim
    )
    torch.mps.synchronize()
    assert fa_hits == [True]
    expected = torch.full(o.shape, POISON, dtype=torch.float64)
    for b in range(len(cu_q) - 1):
        (q0, q1), (k0, k1) = cu_q[b : b + 2], cu_k[b : b + 2]
        rows, keys = [min(hi - lo, span) if hi >= lo else span for lo, hi in ((q0, q1), (k0, k1))]
        scores = scale * torch.einsum("qhd,khd->hqk", q[q0 : q0 + rows].double(), k[k0 : k0 + keys].double())
        if causal:
            scores = scores.masked_fill(torch.arange(rows)[:, None] < torch.arange(keys)[None, :], float("-inf"))
        expected[q0 : q0 + rows] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), v[k0 : k0 + keys].double())
    got = o.cpu().double()
    assert torch.equal(got != POISON, expected != POISON)
    live = expected != POISON
    torch.testing.assert_close(got[live], expected[live], atol=1e-3, rtol=1e-3)


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.uint32, torch.uint64], ids=["uint32", "uint64"])
@pytest.mark.parametrize(
    "cu_q,cu_k",
    [([0, 50, 20], [0, 45, 115]), ([0, 50, 90], [0, 70, 30])],
    ids=["wrapped_query_length", "wrapped_key_length"],
)
def test_varlen_unsigned_decreasing_cumulative_lengths_wrap(cu_q, cu_k, dtype, dim, causal, fa_hits):
    """Unsigned cumulative lengths compare ``index < length`` unsigned: a decrease wraps."""
    _varlen_whole_span(cu_q, cu_k, dtype, dim, causal, fa_hits)


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64], ids=["int64", "uint64"])
@pytest.mark.parametrize(
    "cu_q,cu_k",
    [([0, 2**32 + 5], [0, 45]), ([0, 50], [0, 2**32 + 7]), ([0, 2**31 + 5], [0, 2**33])],
    ids=["query_length", "key_length", "both_lengths"],
)
def test_varlen_length_beyond_32_bits_is_above_every_launched_index(cu_q, cu_k, dtype, dim, causal, fa_hits):
    """A 64-bit length of 2**32 + 5 is not its low word, 5."""
    _varlen_whole_span(cu_q, cu_k, dtype, dim, causal, fa_hits)


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("side", ["query", "key"])
@pytest.mark.parametrize(
    "dtype,cu",
    [(torch.int32, [-(2**31) + 10, 2**31 - 10]), (torch.int64, [-(2**62), 2**62 + 100])],
    ids=["int32", "int64"],
)
def test_varlen_length_whose_difference_overflows_keeps_the_wrapped_sign(dtype, cu, side, dim, causal, fa_hits):
    """``(2**31 - 10) - (-2**31 + 10)`` wraps to -20 in the source's int32: a negative length.

    The start is never dereferenced, since no index is below a negative length. A wide
    subtraction saw 2**32 - 20 instead and computed every launched row over every key.
    """
    heads, span = 2, 64
    cu_q, cu_k = (cu, [0, 45]) if side == "query" else ([0, 40], cu)
    gen = torch.Generator().manual_seed(1133)
    q = torch.randn(3 * span, heads, dim, generator=gen)
    k, v = [torch.randn(3 * span, heads, dim, generator=gen) for _ in range(2)]
    o = torch.full(q.shape, POISON, device="mps")
    cq, ck = [torch.tensor(c, dtype=dtype, device="mps") for c in (cu_q, cu_k)]
    qd, kd, vd = [t.to("mps") for t in (q, k, v)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    (_varlen_causal_fwd if causal else _varlen_fwd)[(span // 32, heads)](
        qd, kd, vd, o, cq, ck, *st(qd), *st(kd), *st(vd), *st(o), heads, span, 1.0 / math.sqrt(dim), 32, 32, dim
    )
    torch.mps.synchronize()
    assert fa_hits == [True]
    expected = torch.full(o.shape, POISON, dtype=torch.float64)
    if side == "key":
        expected[:40] = float("nan")  # 40 rows and no key: every score is -inf
    got = o.cpu().double()
    assert torch.equal(got != POISON, expected != POISON)
    assert torch.equal(got.isnan(), expected.isnan())


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "dim,elem,tol",
    [(64, torch.float32, 1e-3), (128, torch.float32, 1e-3), (128, torch.float16, 2e-2)],
    ids=["d64_fp32", "d128_fp32", "d128_fp16"],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_varlen_negative_start_addresses_rows_before_the_view(dtype, dim, elem, tol, causal, fa_hits):
    """The first sequence starts 8 rows before the tensors, which are views 8 rows into storage.

    Every address the source forms is inside the storage, so its result is defined. A start
    held in 32 unsigned bits put the sequence about 2**32 elements past the view.
    """
    heads, pad, scale = 2, 8, 1.0 / math.sqrt(dim)
    cu_q, cu_k = [-pad, 32, 96], [-pad, 42, 106]
    gen = torch.Generator().manual_seed(1134)
    # Storage row r + pad is view row r; the last pad rows stay poison.
    q = torch.randn(96 + 2 * pad, heads, dim, generator=gen).to(elem)
    k, v = [torch.randn(106 + 2 * pad, heads, dim, generator=gen).to(elem) for _ in range(2)]
    o = torch.full(q.shape, POISON, dtype=elem, device="mps")
    cq, ck = [torch.tensor(c, dtype=dtype, device="mps") for c in (cu_q, cu_k)]
    qd, kd, vd, od = [t[pad:] for t in (q.to("mps"), k.to("mps"), v.to("mps"), o)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    (_varlen_causal_fwd if causal else _varlen_fwd)[(2, 2 * heads)](
        qd, kd, vd, od, cq, ck, *st(qd), *st(kd), *st(vd), *st(od), heads, 64, scale, 32, 32, dim
    )
    torch.mps.synchronize()
    assert fa_hits == [True]
    expected = torch.full(o.shape, POISON, dtype=torch.float64)
    for b in range(2):
        (q0, q1), (k0, k1) = [(c[b] + pad, c[b + 1] + pad) for c in (cu_q, cu_k)]
        scores = scale * torch.einsum("qhd,khd->hqk", q[q0:q1].double(), k[k0:k1].double())
        if causal:
            scores = scores.masked_fill(torch.arange(q1 - q0)[:, None] < torch.arange(k1 - k0)[None, :], float("-inf"))
        expected[q0:q1] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), v[k0:k1].double())
    got = o.cpu().double()
    assert torch.equal(got != POISON, expected != POISON)
    live = expected != POISON
    torch.testing.assert_close(got[live], expected[live], atol=tol, rtol=tol)


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
    # The Q, K and bias loads masked along the other axis are computed as the source wrote
    # them: tests/test_load_masks_along_the_other_axis.py.
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


def _launch_biased_sentinel(n_ctx, dim, sentinel, block_n=32, seed=1130, dtype=torch.float32):
    """The 2-D biased kernel with every key of head 1 masked; returns outputs and the SOURCE result.

    The source fills masked keys and the padding keys of its last ``block_n``-key iteration with
    the finite runtime sentinel (in exp2 units), so each of them counts in the softmax: a fully
    masked row averages V over the keys the loop visits. The V load is masked with no ``other``,
    so a padding key that carries weight leaves the source's Out undefined: the last value
    returned marks the rows whose Out the source defines. Lse is defined everywhere.
    """
    gen = torch.Generator().manual_seed(seed)
    z, h, tiles = 1, 2, triton.cdiv(n_ctx, 32)
    rows, keys = tiles * 32, triton.cdiv(n_ctx, block_n) * block_n
    q = torch.randn(z, h, rows, dim, generator=gen).to(dtype)
    k, v = [torch.randn(z, h, keys, dim, generator=gen).to(dtype) for _ in range(2)]
    b = torch.randn(z, h, rows, keys, generator=gen).to(dtype)
    mask = (torch.rand(z, h, keys, generator=gen) < 0.25).to(torch.uint8)
    mask[:, 0, 0] = 0  # head 0 keeps a key, so its padding keys carry no weight at -1e9
    mask[:, 1] = 1
    o = torch.full((z, h, rows, dim), POISON, dtype=dtype, device="mps")
    lse = torch.full((z, h, rows), POISON, device="mps")
    dev = [x.to("mps") for x in (q, k, v, b, mask)]
    st = lambda t: tuple(t.stride())  # noqa: E731
    args = [o, *st(o), lse, *st(lse)]
    for t in dev:
        args += [t, *st(t)]
    _biased_fa[(tiles, z * h)](*args, dim**-0.5, sentinel, z, h, n_ctx, dim, 32, block_n)
    torch.mps.synchronize()
    n, ln2 = n_ctx, math.log(2.0)
    scores = (dim**-0.5 * (q.double() @ k.double().transpose(-1, -2)) + b.double()) / ln2
    scores = scores.masked_fill(mask.bool()[:, :, None, :], sentinel)
    scores[..., n:] = sentinel
    scores = scores[:, :, :n]
    top = scores.max(-1, keepdim=True).values
    weights = torch.exp2(scores - top)
    values = v.double().clone()
    values[:, :, n:] = 0.0
    o_exp = torch.full(o.shape, POISON, dtype=torch.float64)
    lse_exp = torch.full(lse.shape, POISON, dtype=torch.float64)
    o_exp[:, :, :n] = (weights @ values) / weights.sum(-1, keepdim=True)
    lse_exp[:, :, :n] = top[..., 0] * ln2 + torch.log(weights.sum(-1))
    defined = torch.ones(z, h, rows, dtype=torch.bool)
    if keys > n:
        defined[:] = sentinel != 0.0
        defined[:, 1] = sentinel == float("-inf")  # no key at all: NaN whatever V holds
    return o, lse, o_exp, lse_exp, defined


def _assert_biased_sentinel(result, tol):
    """Out has the source's store domain everywhere and its values where the source defines them."""
    o, lse, o_exp, lse_exp, defined = result
    got = o.cpu().double()
    assert torch.equal(got != POISON, o_exp != POISON)
    live = (o_exp != POISON) & defined[..., None]
    torch.testing.assert_close(got[live], o_exp[live], atol=tol, rtol=tol, equal_nan=True)
    got, live = lse.cpu().double(), lse_exp != POISON
    assert torch.equal(got != POISON, live)
    torch.testing.assert_close(got[live], lse_exp[live], atol=tol, rtol=tol, equal_nan=True)


@requires_mps
@pytest.mark.parametrize("sentinel", [-1e9, 0.0])
@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize(
    "n_ctx,block_n",
    [(n, 32) for n in (32, 48, 64, 80, 96, 160)]
    + [(n, 16) for n in (16, 40, 48, 100)]
    + [(n, 64) for n in (32, 96)]
    + [(n, 8) for n in (20, 72)]
    + [(n, 128) for n in (40, 64, 96, 128, 160, 300)]
    + [(300, 256)],
)
def test_biased_finite_sentinel_counts_the_keys_the_source_visits(n_ctx, block_n, dim, sentinel, fa_hits):
    """A 64-wide template tile over an odd number of 32-key source iterations visits 32 keys
    the source never does; counted as sentinels they halved a fully masked row's output.
    A 128-key source iteration visits padding keys past the template's last partial tile."""
    result = _launch_biased_sentinel(n_ctx, dim, sentinel, block_n)
    assert fa_hits == [True]
    _assert_biased_sentinel(result, 2e-3)


@requires_mps
@pytest.mark.parametrize("sentinel", [float("-inf"), -1e9, 0.0])
@pytest.mark.parametrize("n_ctx", [40, 96, 128, 160])
@pytest.mark.parametrize(
    "dtype,dim,block_n",
    [
        (torch.float16, 32, 128),
        (torch.float16, 64, 128),
        (torch.bfloat16, 32, 128),
        (torch.bfloat16, 64, 64),
    ],
    ids=["fp16_d32_bn128", "fp16_d64_bn128", "bf16_d32_bn128", "bf16_d64_bn64"],
)
def test_biased_half_key_block_that_needs_a_narrower_slice(dtype, dim, block_n, n_ctx, sentinel, fa_hits):
    """The tiled template staged a full head-dim slice per key block and asked Metal for more
    than 32 KB of threadgroup memory: pipeline creation failed. A narrower slice fits."""
    result = _launch_biased_sentinel(n_ctx, dim, sentinel, block_n, dtype=dtype)
    assert fa_hits == [True]
    _assert_biased_sentinel(result, 2.5e-2 if dtype == torch.float16 else 8e-2)


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
    with pytest.raises(
        MetalNonRecoverableError, match="query index runs along the rows and key index along the columns"
    ):
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


# The key bound as a select on the scores. Along the rows of the square tile it is
# `s + row < N_CTX`: other cells than the bound along the columns, and the same answer
# only because a launch has whole key tiles, where both hold in every cell.
MLA_KEY_BOUNDS = {
    "along_the_columns": ("(s + n0)[None, :] < N_CTX", lambda n: lambda row, start, column: start + column < n),
    "along_the_rows": ("(s + n0)[:, None] < N_CTX", lambda n: lambda row, start, column: start + row < n),
    "along_the_rows_spread": (
        "tl.broadcast_to((s + n0)[:, None], (BM, BN)) < N_CTX",
        lambda n: lambda row, start, column: start + row < n,
    ),
}


@requires_mps
@pytest.mark.parametrize("n_ctx", [32, 64, 96, 160])
@pytest.mark.parametrize("name", sorted(MLA_KEY_BOUNDS))
def test_mla_key_bound_select_computes_along_either_axis(name, n_ctx, mla_hits):
    expr, select = MLA_KEY_BOUNDS[name]
    fn = _variant(_mla_value_path, _mla_select(expr))
    out, expected = _mla_run(fn, n_ctx, "", seed=1130 + n_ctx, select=select(n_ctx))
    assert mla_hits == [True]
    _assert_poisoned_match(out, expected, 3e-2)


@requires_mps
@pytest.mark.parametrize("n_ctx", [16, 48, 80])
@pytest.mark.parametrize("name", sorted(MLA_KEY_BOUNDS))
def test_mla_key_bound_select_on_a_partial_tile_refuses_at_launch(name, n_ctx, mla_hits):
    # Here the two bounds differ: along the rows the last rows of a tile lose its keys.
    fn = _variant(_mla_value_path, _mla_select(MLA_KEY_BOUNDS[name][0]))
    with pytest.raises(MetalNonRecoverableError, match="Refusing MLA attention launch") as info:
        _mla_run(fn, n_ctx, "", seed=1131)
    assert f"the sequence length {n_ctx} is not a whole number of 32-row tiles" in str(info.value)
    assert mla_hits == []


@requires_mps
@pytest.mark.parametrize("name", ["along_the_rows", "along_the_rows_spread"])
def test_mla_key_bound_along_the_rows_needs_a_square_tile(name, mla_hits):
    # 64 keys do not fit along 32 rows: the frontend rejects the source before the proof reads it.
    fn = _variant(_mla_value_path, _mla_select(MLA_KEY_BOUNDS[name][0]))
    with pytest.raises(CompilationError, match="incompatible dimensions|Cannot broadcast"):
        _mla_run(fn, 128, "", seed=1133, block_n=64)
    assert mla_hits == []


MLA_AXES = "query index runs along the rows and key index along the columns"
MLA_BOUND = "a score mask equivalent to either the N_CTX boundary or the canonical within-sequence lower triangle"
MLA_SELECT_REFUSED = {
    # The query bound along the columns is not the key bound along the rows.
    "query_bound_along_the_columns": ("tl.where(m[None, :] < N_CTX, qk, NINF)", MLA_AXES),
    # An axis the proof cannot read.
    "key_bound_reshaped": ("tl.where(tl.reshape(s + n0, (BN, 1)) < N_CTX, qk, NINF)", MLA_AXES),
    # Bounds that do not hold in every cell of a whole tile.
    "one_key_short": ("tl.where((s + n0)[:, None] < N_CTX - 1, qk, NINF)", MLA_BOUND),
    "a_constant_bound": ("tl.where((s + n0)[:, None] < 40, qk, NINF)", MLA_BOUND),
    "the_column_in_the_tile": ("tl.where(n0[:, None] < N_CTX, qk, NINF)", MLA_BOUND),
    "one_key_further": ("tl.where((s + n0 + 1)[:, None] < N_CTX, qk, NINF)", MLA_BOUND),
    "kept_where_past_the_bound": (
        "tl.where((s + n0)[:, None] < N_CTX, NINF, qk)",
        "a score mask of where\\(proven_mask, score, -inf\\)",
    ),
    "another_fill": (
        "tl.where((s + n0)[:, None] < N_CTX, qk, 0.0)",
        "a score mask of where\\(proven_mask, score, -inf\\)",
    ),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(MLA_SELECT_REFUSED))
def test_mla_select_on_other_cells_refuses(name, mla_hits):
    line, reason = MLA_SELECT_REFUSED[name]
    fn = _variant(_mla_value_path, (MLA_QK, MLA_QK + IND + "qk = " + line.replace("NINF", NINF)))
    with pytest.raises(MetalNonRecoverableError, match=reason):
        _mla_run(fn, 64, "", seed=1132)
    assert mla_hits == []


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
BIAS = "the Bias load masked by the query-row boundary along its rows and the key boundary along its columns"
MASK_KEYS = "the Mask load masked by exactly the key boundary"
BW_B = "b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]"
BW_M = "m_block = tl.load(mask_ptrs, mask_k, "
BW_B_T = "b_block = tl.load(b_ptrs, mask_j[None, :] & mask_k[:, None], other=0.0"
BW_M_ROWS = "m_block = tl.load(mask_ptrs, mask_j, other=0, "

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
    # No dot orients the Bias and Mask loads: a boundary on the other axis zeroes live cells.
    "kv_bias_mask_transposed": ("kv", (BW_B, BW_B_T), BIAS),
    "q_bias_mask_transposed": ("q", (BW_B, BW_B_T), BIAS),
    "b_bias_mask_transposed": ("b", (BW_B, BW_B_T), BIAS),
    "kv_bias_mask_rows_only_on_keys": ("kv", (BW_B, "b_block = tl.load(b_ptrs, mask_k[:, None]"), BIAS),
    "q_bias_mask_one_key_short": (
        "q",
        (BW_B, "b_block = tl.load(b_ptrs, mask_j[:, None] & ((k_idxs + start_k) < N - 1)[None, :]"),
        BIAS,
    ),
    "q_mask_load_on_the_row_boundary": ("q", (BW_M, BW_M_ROWS), MASK_KEYS),
    "b_mask_load_on_the_row_boundary": ("b", (BW_M, BW_M_ROWS), MASK_KEYS),
    "kv_mask_load_one_key_short": ("kv", (BW_M, "m_block = tl.load(mask_ptrs, k_idxs < N - 1, "), MASK_KEYS),
    "q_vt_other_nan": ("q", (Q_VT, Q_VT[:-1] + ", " + NAN), OTHER),
}

BACKWARD_ADMITTED = {
    "kv_q_other_zero": ("kv", (KV_Q, KV_Q[:-1] + ", other=0.0)")),
    "kv_lse_mask_bound_first": ("kv", (KV_L, "row_max = tl.load(l_ptrs, mask=N > (j_idxs + start_j))")),
    "q_k_other_zero": ("q", (Q_K, Q_K[:-1] + ", other=0.0)")),
    "kv_bias_mask_keys_first": ("kv", (BW_B, "b_block = tl.load(b_ptrs, mask_k[None, :] & mask_j[:, None]")),
    "q_bias_mask_bound_first": (
        "q",
        (BW_B, "b_block = tl.load(b_ptrs, (N > j_idxs)[:, None] & (N > (k_idxs + start_k))[None, :]"),
    ),
    "b_bias_mask_keys_first": ("b", (BW_B, "b_block = tl.load(b_ptrs, mask_k[None, :] & mask_j[:, None]")),
    "q_mask_load_bound_first": ("q", (BW_M, "m_block = tl.load(mask_ptrs, N > (k_idxs + start_k), ")),
    "b_mask_load_other_zero": ("b", (BW_M, "m_block = tl.load(mask_ptrs, mask_k, other=0, ")),
    # Padding read as masked: the bounds select supplies the sentinel there either way.
    "kv_mask_load_other_one": ("kv", (BW_M, "m_block = tl.load(mask_ptrs, mask_k, other=1, ")),
    "q_mask_load_other_one": ("q", (BW_M, "m_block = tl.load(mask_ptrs, mask_k, other=1, ")),
    "b_mask_load_other_one": ("b", (BW_M, "m_block = tl.load(mask_ptrs, mask_k, other=1, ")),
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


@requires_mps
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 3e-2), (torch.bfloat16, 8e-2)], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kind", ["kv", "b"])
def test_backward_mask_load_other_one_computes_at_half_types(kind, dtype, tol, bwd_hits):
    """Padding read as masked on half-typed tensors: the same cells and the same bits as the
    canonical load, and the source's result on the rounded inputs. The dQ kernel of this file
    refuses at half types in its canonical spelling too, so it has no case here."""
    _kind, edit = BACKWARD_ADMITTED[f"{kind}_mask_load_other_one"]

    def inputs():
        t = bwd._inputs(kind, seed=1131)
        # lse stays fp32: the template subtracts the raw fp32 value.
        return {n: x.to(dtype) if x.is_floating_point() and n != "lse" else x for n, x in t.items()}

    t = inputs()
    canonical = [x.cpu() for x in bwd._launch(kind, bwd._variant(bwd.KERNELS[kind]), inputs())]
    outputs = [x.cpu() for x in bwd._launch(kind, bwd._variant(bwd.KERNELS[kind], edit), t)]
    assert bwd_hits == [True, True]
    reference = bwd._expected(kind, {}, {n: x.double() if x.is_floating_point() else x for n, x in t.items()})
    for got, same, expected in zip(outputs, canonical, reference):
        assert torch.equal(got, same)
        got = got.double()
        written = expected != bwd.POISON
        # A half-typed gradient can round to the poison value itself, so the stored domain is
        # checked from outside: nothing lands where the source stores nothing.
        assert torch.all(got[~written] == bwd.POISON)
        scale = max(1.0, float(expected[written].abs().max()))
        torch.testing.assert_close(got[written], expected[written], atol=tol * scale, rtol=tol)
