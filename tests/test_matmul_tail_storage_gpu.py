"""Bounded real-launch acceptance for the partial-K backing-allocation guard."""

import hashlib
import pytest
import torch
import triton
import triton.language as tl

from tests.test_fa_tail_semantics import _executed_source, executed  # noqa: F401
from tests.test_matmul_tail_storage_contract import _masked, _unmasked
from tests.cache_helpers import patch_live_singleton_method
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


@triton.jit
def _reordered(C, B, A, M, N, K, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * SA + rk[None, :]
    bp = B + rk[:, None] * SB + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK
        bp += BK * SB
    tl.store(C + rm[:, None] * SC + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


def _data(size):
    # Integer-valued fp32 products/sums are exact at this bounded geometry.
    return ((torch.arange(size, dtype=torch.int64) % 7) - 3).float()


def _positive(case):
    m = n = 32
    k = 64 if case == "aligned" else 33
    logical_k = k if case == "masked" else triton.cdiv(k, 32) * 32
    reordered = case.startswith("reordered")
    sa = 64 if reordered else k
    offset = case == "reordered_offset"
    negative = case == "reordered_negative"
    a_lead, b_lead, c_lead = (7, 5, 3) if offset else (0, 0, 0)
    sa = -sa if negative else sa
    a_ptr = a_lead + ((m - 1) * -sa if negative else 0)
    a_size = a_lead + (m - 1) * abs(sa) + logical_k
    b_size = b_lead + logical_k * n
    canary = 17391.0
    ah = _data(a_size)
    bh = _data(b_size)
    ch = torch.full((c_lead + m * n + 5,), canary)
    if a_lead:
        ah[:a_lead] = canary
        bh[:b_lead] = canary
    a_indices = a_ptr + torch.arange(m)[:, None] * sa + torch.arange(logical_k)[None, :]
    b_indices = b_lead + torch.arange(logical_k)[:, None] * n + torch.arange(n)[None, :]
    assert int(a_indices.min()) >= 0 and int(a_indices.max()) < len(ah)
    assert int(b_indices.min()) >= 0 and int(b_indices.max()) < len(bh)
    oracle = ah[a_indices].double() @ bh[b_indices].double()
    a, b, c = ah.to("mps"), bh.to("mps"), ch.to("mps")
    a_view, b_view, c_view = a[a_ptr:], b[b_lead:], c[c_lead:]
    if reordered:
        kernel, args = _reordered, (c_view, b_view, a_view, m, n, k, sa, n, n)
    else:
        kernel, args = (_masked if case == "masked" else _unmasked), (a_view, b_view, c_view, m, n, k)
    return kernel, args, (a, b, c), (ah, bh, ch), oracle, c_lead


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["0", "1"])
@pytest.mark.parametrize(
    "case", ["padded", "aligned", "masked", "reordered_padded", "reordered_offset", "reordered_negative"]
)
def test_tail_storage_gpu_positive(case, compile_shader, monkeypatch, executed, record_property):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    kernel, args, backing, before, oracle, c_lead = _positive(case)
    executed.clear()
    handle = kernel[(1, 1)](*args, BM=32, BN=32, BK=32)
    torch.mps.synchronize()
    assert torch.equal(backing[0].cpu().view(torch.int32), before[0].view(torch.int32))
    assert torch.equal(backing[1].cpu().view(torch.int32), before[1].view(torch.int32))
    got = backing[2].cpu()
    assert torch.equal(got[c_lead : c_lead + 1024].reshape(32, 32), oracle.float())
    assert torch.equal(got[:c_lead], before[2][:c_lead])
    assert torch.equal(got[c_lead + 1024 :], before[2][c_lead + 1024 :])
    assert len(executed) == 1
    if case == "aligned" and compile_shader == "0":
        from triton_msl.backend.driver import _PIPELINE_BUCKETS

        entry = _PIPELINE_BUCKETS[id(handle.function)].latest("direct")
        assert entry.primary is handle.function and executed[0]["pipeline"] is entry.direct
        assert handle.name + "__mmdirect" in handle.asm["msl"]
        source, route = handle.asm["msl"], "host_mmdirect"
    elif case == "aligned" and compile_shader == "1":
        from triton_msl.autotuning._fast_matmul_dispatch import _VARIANT_MSL_CACHE

        source = executed[0]["msl"]
        assert source in [handle.metadata.fast_matmul[0], *_VARIANT_MSL_CACHE.values()]
        assert "kernel void simdgroup_matmul_fast(" in source
        route = "compile_shader_fast_matmul"
    else:
        source = _executed_source(executed, handle)
        assert source == handle.asm["msl"]
        route = "compile_shader_source" if "msl" in executed[0] else "host_source"
    record_property("executed_route", route)
    record_property("executed_shader_sha256", hashlib.sha256(source.encode()).hexdigest())
    record_property("output_backing_sha256", hashlib.sha256(got.numpy().tobytes()).hexdigest())
    record_property(
        "input_backing_sha256", ",".join(hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in before[:2])
    )
    descriptor = handle.metadata.batched_dot_bounds
    if case == "masked":
        assert descriptor is None
    else:
        assert descriptor[0] == "matmul_grid_storage_v1"
        assert "unmasked source executes complete BLOCK_K iterations" in handle.asm["msl"]


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["0", "1"])
@pytest.mark.parametrize("bad_role", ["A", "B"])
def test_tail_storage_gpu_refuses_without_dispatch(bad_role, compile_shader, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.backend.driver import MetalUtils, _get_utils, _get_compile_shader_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe source must refuse before any dispatch")

    # Even a broken bounds guard cannot execute the intentionally insufficient
    # allocation: both actual submission paths are replaced by a tripwire.
    monkeypatch.setattr(CompileShaderRuntime, "dispatch", forbidden)
    monkeypatch.setattr(MetalUtils, "launch", forbidden)
    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime, "dispatch", lambda *a: forbidden)
    patch_live_singleton_method(monkeypatch, _get_utils, "launch", lambda *a: forbidden)
    m = n = 32
    k = 33
    a = _data(m * k if bad_role == "A" else (m - 1) * k + 64).to("mps")
    b = _data(k * n if bad_role == "B" else 64 * n).to("mps")
    c = torch.full((m * n,), 17391.0, device="mps")
    before = [x.cpu().view(torch.int32).clone() for x in (a, b, c)]
    with pytest.raises(MetalNonRecoverableError, match="mask the K tail"):
        _unmasked[(1, 1)](a, b, c, m, n, k, BM=32, BN=32, BK=32)
    assert all(torch.equal(x.cpu().view(torch.int32), y) for x, y in zip((a, b, c), before))
