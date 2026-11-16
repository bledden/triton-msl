"""Installed device positives for template parameter hygiene; never run old shaders."""

import pytest
import torch

from tests.test_matmul_symbol_hygiene import renamed_tile
from tests.test_template_symbol_hygiene import CASES, _batch, _chained, _norm, _reshape, _soft, _sort
from tests.test_template_scalar_abi import _shape
from tests.test_codegen_admission_semantics import _transpose3


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("fn,sig,constants,maker,collisions", CASES, ids=[c[3] for c in CASES])
def test_renamed_template_matches_independent_oracle_and_preserves_guards(
    tmp_path, monkeypatch, fn, sig, constants, maker, collisions
):
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    calls = []
    original = getattr(GenericLowerer, maker)

    def observe(self, *args, **kwargs):
        calls.append(maker)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GenericLowerer, maker, observe)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    renamed = renamed_tile(tmp_path, fn, tuple(collisions) + tuple(n for n in fn.arg_names if n not in sig))
    renamed.device_caches.clear()
    gen = torch.Generator().manual_seed(9017)
    grid = (1,)
    scalar_args = []
    inputs = []
    output_dtype = torch.float32
    if fn is _batch:
        # Small integer values make all dot products exact in fp32; the CPU
        # oracle uses float64 independently of the shader's fragment layout.
        inputs = [torch.randint(-2, 3, (2, 32, 32), generator=gen).float() for _ in range(2)]
        expected = (inputs[0].double() @ inputs[1].double()).float()
        scalar_args = [1024, 32, 1] * 3
        grid = (1, 1)
    elif fn is _shape:
        x = torch.randint(-3, 4, (4, 4, 4), generator=gen).float()
        inputs = [x]
        mode = constants["MODE"]
        expected = x.flip(1) if mode == 0 else x.sum(2) if mode == 1 else x.argmax(2).int()
        output_dtype = expected.dtype
        scalar_args = [1.25]
    elif fn in (_soft, _norm):
        x = torch.randn((3, 61), generator=gen) * 2 + 1
        inputs = [x]
        if fn is _soft:
            expected = torch.softmax(x.double(), -1).float()
        else:
            xd = x.double()
            centered = xd - xd.mean(-1, keepdim=True)
            expected = (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)).float()
        scalar_args = [61]
        grid = (3,)
    elif fn is _sort:
        x = torch.randn((4, 16), generator=gen)
        inputs = [x]
        expected = torch.sort(x, dim=1).values
    elif fn is _reshape:
        x = torch.arange(64, dtype=torch.float32).reshape(4, 16)
        inputs = [x]
        expected = x.T.contiguous().flatten()
    elif fn is _chained:
        x = torch.randint(-5, 6, (4, 4, 4, 4, 4), generator=gen, dtype=torch.int32)
        inputs = [x]
        expected = x.long().permute(2, 1, 0, 3, 4).sum(3).sum(1).sum(0).int()
        output_dtype = torch.int32
    else:
        assert fn is _transpose3
        x = torch.arange(512, dtype=torch.float32).reshape(4, 8, 16)
        inputs = [x]
        expected = x.permute(1, 0, 2).contiguous()

    guard = 32
    canary = -8191
    input_storage = []
    device_inputs = []
    for cpu in inputs:
        flat = cpu.flatten()
        storage = torch.full((flat.numel() + 2 * guard,), canary, dtype=cpu.dtype, device="mps")
        view = storage[guard:-guard]
        view.copy_(flat)
        input_storage.append(storage)
        device_inputs.append(view)
    out_storage = torch.full((expected.numel() + 2 * guard,), canary, dtype=output_dtype, device="mps")
    out = out_storage[guard:-guard]
    out.fill_(12345)  # Every logical output must overwrite the poison.
    args = [*device_inputs, out, *scalar_args]
    renamed[grid](*args, **constants, num_warps=4)
    torch.mps.synchronize()
    assert calls == [maker], "must exercise the changed emitter through installed production admission"
    actual = out.cpu().reshape(expected.shape)
    if fn in (_soft, _norm):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    else:
        assert torch.equal(actual, expected)
    assert torch.equal(out_storage[:guard].cpu(), torch.full((guard,), canary, dtype=output_dtype))
    assert torch.equal(out_storage[-guard:].cpu(), torch.full((guard,), canary, dtype=output_dtype))
    for storage, cpu in zip(input_storage, inputs):
        want = torch.full((cpu.numel() + 2 * guard,), canary, dtype=cpu.dtype)
        want[guard:-guard] = cpu.flatten()
        assert torch.equal(storage.cpu(), want)
