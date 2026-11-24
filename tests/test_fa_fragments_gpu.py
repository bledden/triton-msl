"""All retained fragment compositions through public Triton dispatch."""

import pytest
import torch

from test_flash_attention import _flash_attn_fwd
from triton_msl.codegen._fa_fragments import MODES


@pytest.mark.parametrize("n", [32, 64])
def test_fragment_compositions_match_reference_and_each_other(n):
    torch.manual_seed(709 + n)
    q, k, v = [torch.randn(1, 1, n, 64, dtype=torch.float16) * 0.25 for _ in range(3)]
    snapshots = [x.clone() for x in (q, k, v)]
    reference = torch.softmax(q.double() @ k.double().transpose(-1, -2) * 0.125, dim=-1) @ v.double()
    baseline = None
    for mode in MODES:
        backing = torch.full((n * 64 + 32,), 117, dtype=torch.float16)
        output = backing[16:-16].view_as(q)
        output.fill_(float("nan"))
        kernel = _flash_attn_fwd[(n // 32, 1)](
            q,
            k,
            v,
            output,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            1,
            1,
            n,
            BLOCK_M=32,
            BLOCK_N=32,
            HEAD_DIM=64,
            IS_CAUSAL=False,
            experimental_fa_fragment_lowering=mode,
        )
        torch.mps.synchronize()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), reference, rtol=0.004, atol=0.004)
        assert torch.all(backing[:16] == 117) and torch.all(backing[-16:] == 117)
        assert all(torch.equal(x, old) for x, old in zip((q, k, v), snapshots))
        if mode == "off":
            # The default lowering replays each 32-key source block's accumulator update; the
            # experimental lowerings keep one fused update per 64-key tile. They agree with it
            # to rounding (the reference comparison above) and with each other exactly.
            pass
        elif baseline is None:
            baseline = output.clone()
        else:
            assert torch.equal(output, baseline), mode
        metadata = kernel.metadata._asdict() if hasattr(kernel.metadata, "_asdict") else vars(kernel.metadata)
        if mode != "off":
            assert metadata["fa_fragment_lowering"]["selected"] == mode
