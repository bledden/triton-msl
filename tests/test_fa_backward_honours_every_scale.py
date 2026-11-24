"""``triton_msl.fa_backward.flash_attention`` uses the scale of each call, whatever came before it.

Its forward was torch's MPS ``scaled_dot_product_attention(..., scale=scale)``. torch 2.12.1
keeps the first scale that operation sees for a shape and causal flag and returns that
attention for every later scale. A second scale for the same shape, or a scale the caller's own
code had used first, therefore gave a wrong O and, through ``D = rowsum(dO * O)``, wrong dQ and dK
(dV was right). Everything here is judged against float64 autograd on the CPU.
"""

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal device")

TOL = {torch.float32: 1e-4, torch.float16: 5e-3}


def _inputs(shape, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=g).to(dtype) for _ in range(4))


def _float64(q, k, v, dO, scale, causal):
    a, b, c = (t.double().requires_grad_() for t in (q, k, v))
    s = (a @ b.transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), -math.inf)
    o = torch.softmax(s, -1) @ c
    o.backward(dO.double())
    return dict(O=o.detach(), dQ=a.grad, dK=b.grad, dV=c.grad)


def _device(q, k, v, dO, scale, causal):
    from triton_msl.fa_backward import flash_attention

    qd, kd, vd = (t.to("mps").requires_grad_() for t in (q, k, v))
    o = flash_attention(qd, kd, vd, scale=scale, causal=causal)
    o.backward(dO.to("mps"))
    torch.mps.synchronize()
    return dict(O=o.detach(), dQ=qd.grad, dK=kd.grad, dV=vd.grad)


def _check(case, got, want, tol):
    for name in ("O", "dQ", "dK", "dV"):
        rel = float((got[name].cpu().double() - want[name]).abs().max() / want[name].abs().max())
        assert rel <= tol, f"{case}: {name} relative error {rel:.3g}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_alternating_scales_on_one_shape(causal, dtype):
    x = _inputs((1, 2, 48, 64), 5, dtype)
    for scale in (0.3, None, 0.125, 1.0, 0.3):
        _check(f"scale={scale} causal={causal}", _device(*x, scale, causal), _float64(*x, scale, causal), TOL[dtype])


# The MPS SDPA calls below leave torch's cached SDPA at scale 0.5 for these two shapes (both causal
# flags) for the rest of the process, so no other test may use MPS SDPA on them: a later SDPA
# oracle at another scale would get scale 0.5's attention. Reserved: [1, 3, 208, 64] and its
# folded [3, 208, 64]; no other test uses N = 208.
RESERVED_SDPA_SHAPE = (1, 3, 208, 64)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_a_prior_user_sdpa_call_with_another_scale(causal):
    # The caller's own SDPA calls come first, on the 4-D shape and on the folded [Z*H, N, 64]
    # shape flash_attention works on, with a scale that flash_attention is then not given.
    z, h, n, d = RESERVED_SDPA_SHAPE
    x = _inputs(RESERVED_SDPA_SHAPE, 9, torch.float32)
    q, k, v = (t.to("mps") for t in x[:3])
    F.scaled_dot_product_attention(q, k, v, scale=0.5, is_causal=causal)
    F.scaled_dot_product_attention(*(t.reshape(z * h, n, d) for t in (q, k, v)), scale=0.5, is_causal=causal)
    torch.mps.synchronize()
    for scale in (0.125, None):
        _check(f"after SDPA(0.5): scale={scale}", _device(*x, scale, causal), _float64(*x, scale, causal), 1e-4)


def test_the_forward_does_not_call_torch_sdpa(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("flash_attention's result must not depend on torch's SDPA")

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)
    x = _inputs((2, 32, 64), 13, torch.float32)
    _check("SDPA forbidden", _device(*x, 0.2, True), _float64(*x, 0.2, True), 1e-4)
