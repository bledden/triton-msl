"""Dense FlashAttention whose tensors do not share one float element type.

The simdgroup template stages Q through a scalar cast, but it loads K and V and stores Out as
matrices of one element type. A K, V or Out of another float type, or a bfloat Q beside half
tiles, used to reach the Metal compiler and fail there with a template-deduction error. The
lowering now names the refusal. A Q of another type that the cast handles still runs on the
template and matches a float64 reference. A float32 Q beside float16 tiles is not one of those:
the cast would round it to float16, so it runs on the tiled template, which stages Q in float32
(tests/test_where_the_dense_forward_multiplies_by_its_constant.py).
"""

import math

import pytest
import torch

from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0
N = 64
TYPES = {"f": torch.float32, "h": torch.float16, "b": torch.bfloat16}


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


def _launch(dim, base, role, odd):
    gen = torch.Generator().manual_seed(1126)
    cpu = {
        name: torch.randn(1, 2, N, dim, generator=gen).to(TYPES[odd] if name == role else TYPES[base]) for name in "qkv"
    }
    out = torch.full((1, 2, N, dim), POISON, dtype=TYPES[odd] if role == "o" else TYPES[base], device="mps")
    dev = {name: tensor.to("mps") for name, tensor in cpu.items()}
    _flash_attn_fwd[(N // 32, 2)](
        dev["q"], dev["k"], dev["v"], out,
        *dev["q"].stride(), *dev["k"].stride(), *dev["v"].stride(), *out.stride(),
        1, 2, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=dim, IS_CAUSAL=False,
    )  # fmt: skip
    torch.mps.synchronize()
    scores = (cpu["q"].double() @ cpu["k"].double().transpose(-1, -2)) / math.sqrt(dim)
    return out.cpu().double(), torch.softmax(scores, -1) @ cpu["v"].double()


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize(
    "base, role, odd",
    [
        ("f", "k", "h"),
        ("f", "v", "h"),
        ("f", "o", "h"),
        ("h", "k", "f"),
        ("h", "v", "f"),
        ("h", "o", "f"),
        ("f", "k", "b"),
        ("f", "v", "b"),
        ("h", "q", "b"),
    ],
)
def test_mixed_matrix_role_types_refuse_by_name(fa_hits, dim, base, role, odd):
    with pytest.raises(MetalNonRecoverableError, match="one matrix element type"):
        _launch(dim, base, role, odd)
    assert fa_hits == []


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("base, odd, simdgroup", [("f", "h", True), ("h", "f", False), ("f", "b", True)])
def test_query_of_another_float_type_still_computes(fa_hits, dim, base, odd, simdgroup):
    got, expected = _launch(dim, base, "q", odd)
    assert fa_hits == ([True] if simdgroup else [])
    assert not bool((got == POISON).any())
    torch.testing.assert_close(got, expected, atol=3e-3, rtol=3e-3)
