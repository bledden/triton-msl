"""Dense-transpose raw-buffer dispatch, output layout and alias controls.

Run through the shared project GPU lease in the correctness lane.
"""

import pytest
import torch
import triton

from tests.test_flat_matmul_mapping import _flat
from tests.test_generic_2d_gpu import affine_2d

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


@requires_gpu
@pytest.mark.parametrize("transpose_output", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m,n,k", [(64, 96, 64), (70, 97, 47)])
def test_public_dense_transpose_direct_route(monkeypatch, dtype, m, n, k, transpose_output):
    from triton_msl.backend.driver import _get_utils, _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    gen = torch.Generator().manual_seed(10010)
    ac = torch.randint(-2, 3, (m, k), generator=gen).to(dtype)
    bc = torch.randint(-2, 3, (n, k), generator=gen).to(dtype)
    aa = ac.to("mps")
    owner = bc.to("mps")
    bb = owner.t()
    cc = torch.full((n, m) if transpose_output else (m, n), -8192.0, dtype=dtype, device="mps")
    if transpose_output:
        cc = cc.t()
    count = triton.cdiv(m, 32) * triton.cdiv(n, 32)
    utils = _get_utils()
    rt = _get_compile_shader_runtime()
    real = rt.dispatch
    calls = []

    def observe(lib, name, args, **geometry):
        assert args[0] is aa and args[1] is bb and args[2] is cc
        result = real(lib, name, args, **geometry)
        calls.append((lib, name, geometry))
        return result

    def host_forbidden(*args, **kwargs):
        raise AssertionError("dense transpose must dispatch directly, not host-roundtrip")

    monkeypatch.setattr(rt, "dispatch", observe)
    monkeypatch.setattr(utils, "launch", host_forbidden)
    for _ in range(2):
        compiled = _flat[(count,)](
            aa, bb, cc, m, n, k, *aa.stride(), *bb.stride(), *cc.stride(), BM=32, BN=32, BK=32, MODE=1
        )
        torch.mps.synchronize()
        expected = (ac.double() @ bc.double().t()).to(dtype)
        torch.testing.assert_close(cc.cpu(), expected, rtol=0, atol=0)
        assert torch.equal(aa.cpu(), ac) and torch.equal(owner.cpu(), bc)
    assert len(calls) == 2
    assert all(c[1] == "_flat" and c[2] == dict(threads=(count * 256, 1, 1), group_size=(256, 1, 1)) for c in calls)
    assert any("N-fastest source tile mapping" in text for text in compiled.asm.values() if isinstance(text, str))


@requires_gpu
@pytest.mark.parametrize("kind", ["dense_inplace", "contiguous_offset", "offset_transpose"])
def test_dense_transpose_alias_and_offset_routes(monkeypatch, kind):
    from triton_msl.backend.driver import _get_utils, _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    source = torch.arange(5 * 93, dtype=torch.float32).reshape(5, 93) / 16
    if kind == "dense_inplace":
        owner = source.to("mps")
        x = owner.t()
        out = owner
    else:
        owner = torch.full((5 * 93 + 64,), -8192.0, device="mps")
        original = owner[16 : 16 + 5 * 93].view(5, 93)
        original.copy_(source.to("mps"))
        x = original if kind == "contiguous_offset" else original.t()
        output_owner = torch.full((5 * 93 + 64,), -8192.0, device="mps")
        out = output_owner[8 : 8 + 5 * 93].view(5, 93)
    before = owner.cpu().clone()
    rt = _get_compile_shader_runtime()
    utils = _get_utils()
    real_direct = rt.dispatch
    real_host = utils.launch
    direct = []
    host = []

    def observed_direct(lib, name, args, **geometry):
        assert args[0] is x and args[1] is out
        result = real_direct(lib, name, args, **geometry)
        direct.append(geometry)
        return result

    def observed_host(*args, **kwargs):
        result = real_host(*args, **kwargs)
        host.append(True)
        return result

    monkeypatch.setattr(rt, "dispatch", observed_direct)
    monkeypatch.setattr(utils, "launch", observed_host)
    compiled = affine_2d[(3, 5)](x, out, 93, 1.25, BLOCK=32)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), source * 1.25 + 1.0, rtol=0, atol=0)
    if kind == "offset_transpose":
        assert len(host) == 1 and not direct
    else:
        assert len(direct) == 1 and not host
        group = direct[0]["group_size"]
        assert direct[0]["threads"] == (3 * group[0], 5, 1) and group[1:] == (1, 1)
    if kind == "dense_inplace":
        # Same physical element read then written by exactly one lane: race-free.
        assert x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        torch.testing.assert_close(owner.cpu(), source * 1.25 + 1.0, rtol=0, atol=0)
    else:
        assert torch.equal(owner.cpu(), before)
        actual_owner = output_owner.cpu()
        expected_owner = torch.full_like(actual_owner, -8192.0)
        expected_owner[8 : 8 + 5 * 93] = (source * 1.25 + 1.0).reshape(-1)
        assert torch.equal(actual_owner, expected_owner)
