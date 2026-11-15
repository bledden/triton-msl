"""Issue #10's flat grid and transposed B, executed through the installed public route.

The reported kernel uses unmasked tiles. Keep all dimensions tile-aligned here;
tail safety belongs to the separate masked/allocated-tail controls.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def mm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    pm, pn = pid // n_tiles, pid % n_tiles
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * sam + (k + offs_k)[None, :] * sak)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * sbk + offs_n[None, :] * sbn)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc)


@pytest.mark.parametrize("size", [64, 128, 256])
def test_reporter_transposed_b_reaches_gpu(monkeypatch, record_property, size):
    from triton_msl.backend.driver import _get_utils, _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    # Small exact integers make the fp64 oracle independent of reduced dot precision.
    generator = torch.Generator().manual_seed(10010 + size)
    ac = torch.randint(-2, 3, (size, size), generator=generator).float()
    bc = torch.randint(-2, 3, (size, size), generator=generator).float()
    a, owner = ac.to("mps"), bc.to("mps")
    b = owner.t()
    c = torch.full((size, size), -8192.0, device="mps")
    runtime = _get_compile_shader_runtime()
    dispatch = runtime.dispatch
    calls = []
    utils = _get_utils()
    launch = utils.launch
    host_calls = []

    def observe(lib, name, args, **geometry):
        assert args[0] is a and args[1] is b and args[2] is c
        result = dispatch(lib, name, args, **geometry)
        calls.append(geometry)
        return result

    def observe_host(*args, **kwargs):
        # The host-copy route still submits a real Metal pipeline and waits.
        # It is a cost to measure, not a CPU computation fallback.
        result = launch(*args, **kwargs)
        host_calls.append(True)
        return result

    monkeypatch.setattr(runtime, "dispatch", observe)
    monkeypatch.setattr(utils, "launch", observe_host)
    mm[((size // 32) * (size // 32),)](
        a,
        b,
        c,
        size,
        size,
        size,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    torch.mps.synchronize()
    assert len(calls) + len(host_calls) == 1
    record_property("execution_route", "direct_mps" if calls else "metal_host_copy")
    torch.testing.assert_close(c.cpu(), (ac.double() @ bc.double().t()).float(), rtol=0, atol=0)
    assert torch.equal(a.cpu(), ac) and torch.equal(owner.cpu(), bc)
