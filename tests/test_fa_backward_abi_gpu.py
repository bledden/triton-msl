"""Focused direct-op GPU ABI proof. Collect/run only under the coordinator's lease."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton.backends
from triton_msl import fa_backward as fa


@pytest.fixture(scope="module", autouse=True)
def metal_window():
    # No device probing at collection: the allocated GPU worker owns this call.
    assert torch.backends.mps.is_available(), "allocated Metal device is unavailable"
    yield
    torch.mps.synchronize()


def inputs(shape, dtype, layout, generator):
    values = []
    for _ in range(3):
        if layout == "slice":
            cpu = torch.randn((*shape[:-1], 128), generator=generator, dtype=dtype) * 0.25
            value = cpu.to("mps")[..., ::2]
        elif layout == "transpose":
            cpu = torch.randn((*shape[:-2], 64, shape[-2]), generator=generator, dtype=dtype) * 0.25
            value = cpu.to("mps").transpose(-1, -2)
        elif layout == "expanded":
            # Expand a leading head axis; sequence rows stay distinct so all
            # three gradient oracles have a nonzero comparison scale.
            cpu = torch.randn((*shape[:-3], 1, shape[-2], 64), generator=generator, dtype=dtype) * 0.25
            value = cpu.to("mps").expand(shape)
        else:
            value = (torch.randn(shape, generator=generator, dtype=dtype) * 0.25).to("mps")
        values.append(value.detach().requires_grad_())
    assert all(t.is_contiguous() == (layout == "contiguous") for t in values)
    return values


def check(shape, dtype, layout, causal):
    generator = torch.Generator(device="cpu").manual_seed(747)
    values = inputs(shape, dtype, layout, generator)
    references = [x.detach().cpu().clone().requires_grad_() for x in values]
    gradient = torch.randn(shape, generator=generator, dtype=dtype)
    # Independent CPU SDPA oracle; no patched package/autograd/runtime endpoint.
    # Half/bfloat inputs intentionally follow the documented FP32 internal math.
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*(x.float() for x in references), is_causal=causal).to(dtype)
        expected.backward(gradient)
    actual = fa.flash_attention(*values, causal=causal)
    actual.backward(gradient.to("mps"))
    torch.mps.synchronize()
    assert actual.shape == shape and actual.dtype == dtype and actual.device == values[0].device
    # Existing direct-op gates use1e-3 FP32 and3e-2 half max-error/peak-gradient.
    # Bfloat uses the same3e-2 bound; these are fixed before the GPU run.
    tolerance = 1e-3 if dtype == torch.float32 else 3e-2
    for name, got, want in [
        ("output", actual, expected),
        *[(name, x.grad, r.grad) for name, x, r in zip(("dQ", "dK", "dV"), values, references)],
    ]:
        assert got is not None and got.dtype == dtype
        got = got.detach().cpu().float()
        want = want.detach().float()
        assert torch.isfinite(got).all() and torch.isfinite(want).all()
        error = (got - want).abs().max().item()
        peak = want.abs().max().item()
        assert peak > 0 and error / peak < tolerance, (name, dtype, layout, causal, error, peak, tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "slice", "transpose", "expanded"])
def test_fixed_self_attention_views_match_cpu_output_and_all_gradients(dtype, causal, layout):
    check((1, 2, 16, 64), dtype, layout, causal)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_rank_two_self_attention_keeps_supported_behavior(dtype, causal):
    check((16, 64), dtype, "contiguous", causal)


@pytest.mark.parametrize("defect", ["equal_numel_peer", "broadcast_peer", "dtype_peer", "cpu_peer"])
def test_real_tensor_invalid_peers_refuse_before_eager_or_metal(monkeypatch, defect):
    q = torch.zeros(2, 16, 64, device="mps")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    if defect == "equal_numel_peer":
        k = torch.zeros(1, 32, 64, device="mps")
    elif defect == "broadcast_peer":
        v = torch.zeros(1, 16, 64, device="mps")
    elif defect == "dtype_peer":
        k = k.half()
    else:
        v = torch.zeros(2, 16, 64, device="cpu")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid peer entered eager/autograd/Metal preparation")

    monkeypatch.setattr(fa._FlashAttentionFn, "apply", forbidden)
    monkeypatch.setattr(fa, "_attention_output", forbidden)
    monkeypatch.setattr(fa, "_logsumexp", forbidden)
    monkeypatch.setattr(fa, "_dispatch_backward", forbidden)
    with pytest.raises((ValueError, TypeError)):
        fa.flash_attention(q, k, v)
