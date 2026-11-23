"""Bitwise regression pins for the nonfinite-safe batched alpha rescale."""

import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup


def test_mla_rescale_reference_and_candidate_emission():
    reference = (Path(__file__).parent / "golden/mla_rescale_parent661.msl").read_bytes()
    assert hashlib.sha256(reference).hexdigest() == "683ebb67f17cc9c7d5cbb47f44c12d15c2da1de1dfae9a36fc12934456338bc0"
    candidate = make_flash_attention_kernel_simdgroup(
        head_dim=192, v_head_dim=128, out_dtype="fp16", kernel_name="_mla_value_path", scale=0.0721687824
    )
    assert (
        hashlib.sha256(candidate.encode()).hexdigest()
        == "56d8d1c0966ac29a63579c15daef71897a7ed45ccda68361297a1075d49ca6a3"
    )


@pytest.mark.parametrize("n", [17, 65])
@pytest.mark.parametrize("kind", ["finite", "qnan", "vnonfinite", "signedzero", "subnormal"])
def test_mla_rescale_preserves_parent_bits(n, kind):
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        pytest.skip("requires MPS compile_shader")
    q = np.zeros((1, 2, n, 192), dtype=np.float16)
    k = np.zeros_like(q)
    v = np.ones((1, 2, n, 128), dtype=np.float16)
    if kind == "qnan":
        q[0, 0, 0, 0] = np.nan
    if kind == "vnonfinite":
        v[0, 0, 7, :3] = [np.inf, -np.inf, np.nan]
    if kind == "signedzero":
        v.view(np.uint16)[..., :] = np.tile(np.array([0, 0x8000], dtype=np.uint16), 64)
    if kind == "subnormal":
        v.view(np.uint16)[..., :] = np.tile(np.array([1, 0x8001, 0x400, 0x8400], dtype=np.uint16), 32)
    q, k, v = [torch.from_numpy(a).to("mps") for a in (q, k, v)]
    source = [
        (Path(__file__).parent / "golden/mla_rescale_parent661.msl").read_text(),
        make_flash_attention_kernel_simdgroup(
            head_dim=192, v_head_dim=128, out_dtype="fp16", kernel_name="_mla_value_path", scale=0.0721687824
        ),
    ]
    values = []
    for text in source:
        library = torch.mps.compile_shader(text)
        out = torch.full_like(v, -111)
        library._mla_value_path(
            q,
            k,
            v,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            1,
            2,
            n,
            threads=(((n + 31) // 32) * 256, 2, 1),
            group_size=(256, 1, 1),
        )
        torch.mps.synchronize()
        values.append(out.cpu().numpy())
    assert np.array_equal(values[0].view(np.uint16), values[1].view(np.uint16)), (
        "including signed-zero and NaN payload bits"
    )
    expected = np.ones(values[1].shape, dtype=np.float16)
    if kind == "qnan":
        expected[0, 0, 0, :] = np.nan
    if kind == "vnonfinite":
        expected[0, 0, :, :3] = [np.inf, -np.inf, np.nan]
    if kind == "signedzero":
        expected.fill(0)
    if kind == "subnormal":
        expected.view(np.uint16)[..., :] = np.tile(np.array([1, 0x8001, 0x400, 0x8400], dtype=np.uint16), 32)
    for classify in (np.isnan, np.isposinf, np.isneginf):
        assert np.array_equal(classify(values[1]), classify(expected))
    finite = np.isfinite(expected)
    if kind == "signedzero":
        assert np.all(values[1] == 0), "zero membership must not use a tolerance"
    np.testing.assert_allclose(
        values[1][finite].astype(np.float32),
        expected[finite].astype(np.float32),
        atol=5e-5 if kind == "subnormal" else 0.05,
        rtol=0,
    )
