"""CPU route census for the default-off, triangle-only K-layout candidate."""

import hashlib

import pytest

from tests.test_fa_biased_routing import _biased_fa, _biased_tri_fa
from tests.test_template_scalar_abi import _lower
from triton_msl.backend.compiler import MetalOptions
from triton_msl.codegen import _msl_templates as makers


def triangle(dtype="fp16", contiguous=False, dim=64, bm=32, bn=32, enabled=False):
    constants = dict(DIM=dim, BLOCK_J=bm, BLOCK_K=bn)
    if contiguous:
        constants.update(q_sd=1, k_sd=1, v_sd=1, o_sd=1)
    signature = {
        n: "*fp32"
        if n == "lse_ptr"
        else "*u8"
        if n == "mask_ptr"
        else "*" + dtype
        if n.endswith("_ptr")
        else "fp32"
        if n in ("sm_scale", "neg_inf")
        else "i32"
        for n in _biased_tri_fa.arg_names
        if n not in constants
    }
    obj = _lower(_biased_tri_fa, signature, constants)
    obj.options = MetalOptions(experimental_tiled_k_layout=enabled)
    return obj


def test_option_default_and_cache_hash():
    assert MetalOptions().experimental_tiled_k_layout is False
    assert MetalOptions().hash() != MetalOptions(experimental_tiled_k_layout=True).hash()
    assert (
        MetalOptions(experimental_cooperative_attention=True).hash()
        != MetalOptions(experimental_tiled_k_layout=True).hash()
    )


@pytest.mark.parametrize(
    "contiguous,old_hash,new_hash",
    [
        (
            True,
            "1d0e83f41923c1a46681859d79a7a0fd70050555d6b246f3af6f24ff054c5cec",
            "4c6b20dc73ee35ab639e716abaaeccdd2420d4332d30d1f93a23293a23ec025e",
        ),
        (
            False,
            "36b7f796c604fc7a6b53e8e1107f9ef236ee27997760051c334e1583b0aae50d",
            "3bcfadbb8b7110e14951b8445fe91da16f00ccf9e376f2175165e2726442c222",
        ),
    ],
)
def test_exact_screened_sources_and_geometry(contiguous, old_hash, new_hash):
    off, on = triangle(contiguous=contiguous), triangle(contiguous=contiguous, enabled=True)
    before, after = off.lower(), on.lower()
    assert hashlib.sha256(before.encode()).hexdigest() == old_hash
    assert hashlib.sha256(after.encode()).hexdigest() == new_hash
    assert off.effective_block_size == on.effective_block_size == 1024
    assert off._flash_attention[0] == on._flash_attention[0] == "flash_attention"
    assert off._flash_attention[2] == on._flash_attention[2] == 1024
    assert off._fa_rounding == on._fa_rounding
    assert off._used_pid_axes == on._used_pid_axes == {0, 1, 2}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dtype="bf16"),
        dict(dtype="fp32"),
        dict(contiguous=True, dtype="fp32"),
        dict(dim=32),
        dict(dim=128),
        dict(bm=16),
        dict(bn=16),
        dict(bn=64),
    ],
)
def test_neighbor_triangle_sources_unchanged(kwargs):
    off, on = triangle(**kwargs), triangle(**kwargs, enabled=True)
    assert off.lower() == on.lower()
    assert off.effective_block_size == on.effective_block_size
    assert off._flash_attention == on._flash_attention


@pytest.mark.parametrize("enabled", [0, 1, "1", "True", None, [], {}])
def test_only_bool_is_a_cacheable_option(enabled):
    with pytest.raises(ValueError, match="experimental_tiled_k_layout must be an exact bool"):
        MetalOptions(experimental_tiled_k_layout=enabled)


@pytest.mark.parametrize("role,dtype", [("b_ptr", "fp32"), ("b_ptr", "bf16"), ("o_ptr", "fp32")])
def test_mixed_float_roles_keep_original_layout(role, dtype):
    const = dict(DIM=64, BLOCK_J=32, BLOCK_K=32)
    sig = {
        n: "*fp32"
        if n == "lse_ptr"
        else "*u8"
        if n == "mask_ptr"
        else "*fp16"
        if n.endswith("_ptr")
        else "fp32"
        if n in ("sm_scale", "neg_inf")
        else "i32"
        for n in _biased_tri_fa.arg_names
        if n not in const
    }
    sig[role] = "*" + dtype
    sources = []
    for enabled in (False, True):
        obj = _lower(_biased_tri_fa, sig, const)
        obj.options = MetalOptions(experimental_tiled_k_layout=enabled)
        sources.append(obj.lower())
    assert sources[0] == sources[1]
    assert "dot += qv * tg_KV[c * BN + cj];" not in sources[1]


def test_two_dimensional_biased_source_stays_original(monkeypatch):
    const = dict(DIM=64, BLOCK_M=32, BLOCK_N=32, BAD_TEMP=False, I3D=False)
    sig = {
        n: "*fp32"
        if n == "lse_ptr"
        else "*u8"
        if n == "mask_ptr"
        else "*fp16"
        if n.endswith("_ptr")
        else "fp32"
        if n in ("sm_scale", "neg_inf")
        else "i32"
        for n in _biased_fa.arg_names
        if n not in const
    }
    calls = []
    real = makers.make_flash_attention_kernel_tiled

    def observe(*args, **kwargs):
        calls.append(kwargs.get("transpose_k", False))
        return real(*args, **kwargs)

    monkeypatch.setattr(makers, "make_flash_attention_kernel_tiled", observe)
    outputs = []
    for enabled in (False, True):
        obj = _lower(_biased_fa, sig, const)
        obj.options = MetalOptions(experimental_tiled_k_layout=enabled)
        outputs.append(obj.lower())
    assert outputs[0] == outputs[1]
    assert calls == [False, False]


@pytest.mark.parametrize("mask,plain", [(0, False), (1, False), (3, False), (0, True)])
def test_dense_tiled_caller_stays_original(monkeypatch, mask, plain):
    from tests.test_fa_tail_semantics import _graph
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    calls = []
    real = makers.make_flash_attention_kernel_tiled

    def observe(*args, **kwargs):
        calls.append(kwargs.get("transpose_k", False))
        return real(*args, **kwargs)

    monkeypatch.setattr(makers, "make_flash_attention_kernel_tiled", observe)
    outputs = []
    for enabled in (False, True):
        graph, _ = _graph("fp16", 128, mask, plain_k=plain, plain_v=plain)
        obj = GenericLowerer(graph, MetalOptions(experimental_tiled_k_layout=enabled))
        outputs.append(obj.lower())
    assert outputs[0] == outputs[1]
    assert calls == [False, False]
