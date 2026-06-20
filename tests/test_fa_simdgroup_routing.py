"""Routing: simd FA template chosen for contiguous head_dim=128 FA; scalar otherwise."""
import platform
import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Metal only")


def _info(contiguous=True, out_dtype="f32"):
    c = "c1" if contiguous else 5   # non-c1 innermost stride = a runtime arg index
    return {"head_dim": 128, "block_m": 32, "block_n": 32, "out_dtype": out_dtype,
            "causal": False, "scale": 0.0883,
            "strides": {r: ["c1", "c1", 2, c] for r in ("q", "k", "v", "o")}}


def test_eligible_contiguous_fp32():
    from triton_msl.codegen.generic_lowerer import _simd_fa_eligible
    assert _simd_fa_eligible(_info(contiguous=True)) is True


def test_ineligible_noncontiguous():
    from triton_msl.codegen.generic_lowerer import _simd_fa_eligible
    assert _simd_fa_eligible(_info(contiguous=False)) is False
