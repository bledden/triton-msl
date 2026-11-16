"""General full-K direct route: rectangular, padded, aliased and offset inputs."""

import pytest
import torch
import triton
import triton.language as tl

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


@triton.jit
def full_k(
    A,
    B,
    C,
    M,
    N,
    K,
    SAM,
    SAK,
    SBK,
    SBN,
    SCM,
    SCN,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MASK_MN: tl.constexpr,
):
    pid = tl.program_id(0)
    columns = tl.cdiv(N, BN)
    rm = (pid // columns) * BM + tl.arange(0, BM)
    rn = (pid % columns) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        ap = A + rm[:, None] * SAM + (k + rk)[None, :] * SAK
        bp = B + (k + rk)[:, None] * SBK + rn[None, :] * SBN
        if MASK_MN:
            av = tl.load(ap, rm[:, None] < M, other=0.0)
            bv = tl.load(bp, rn[None, :] < N, other=0.0)
        else:
            av, bv = tl.load(ap), tl.load(bp)
        acc += tl.dot(av, bv)
    cp = C + rm[:, None] * SCM + rn[None, :] * SCN
    if MASK_MN:
        tl.store(cp, acc, (rm[:, None] < M) & (rn[None, :] < N))
    else:
        tl.store(cp, acc)


def _run(monkeypatch, record_property, m, n, k, dtype, mask_mn, layout="plain"):
    from triton_msl.backend.driver import _get_utils, _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    padded = triton.cdiv(k, 32) * 32
    gen = torch.Generator().manual_seed(10010 + m + n + k)
    ac = torch.randint(-2, 3, (m, padded), generator=gen).to(dtype)
    bc = torch.randint(-2, 3, (n, padded), generator=gen).to(dtype)
    aa, bb_owner = ac.to("mps"), bc.to("mps")
    cc = torch.full((m, n), -8192.0, dtype=dtype, device="mps")
    owners = []
    if layout == "input_alias":
        assert m == n
        bb_owner = aa
        bc = ac
    elif layout == "output_transpose":
        cc = torch.full((n, m), -8192.0, dtype=dtype, device="mps").t()
    elif layout == "contiguous_offset":
        a_owner = torch.full((m * padded + 16,), 731.0, dtype=dtype, device="mps")
        c_owner = torch.full((m * n + 16,), 731.0, dtype=dtype, device="mps")
        aa = a_owner[8:-8].reshape(m, padded)
        aa.copy_(ac)
        cc = c_owner[8:-8].reshape(m, n)
        owners = [a_owner, c_owner]
    elif layout == "transpose_offset":
        b_owner = torch.full((n * padded + 16,), 731.0, dtype=dtype, device="mps")
        bb_owner = b_owner[8:-8].reshape(n, padded)
        bb_owner.copy_(bc)
        owners = [b_owner]
    elif layout == "holes":
        b_owner = torch.full((n, padded * 2), 731.0, dtype=dtype, device="mps")
        bb_owner = b_owner[:, ::2]
        bb_owner.copy_(bc)
        owners = [b_owner]
    bb = bb_owner.t()
    before = [x.cpu().clone() for x in owners]
    expected = (ac.double() @ bc.double().t()).to(dtype)
    rt, utils = _get_compile_shader_runtime(), _get_utils()
    dispatch, launch = rt.dispatch, utils.launch
    routes = []
    count = triton.cdiv(m, 32) * triton.cdiv(n, 32)
    args = (aa, bb, cc, m, n, k, *aa.stride(), *bb.stride(), *cc.stride())
    options = dict(BM=32, BN=32, BK=32, MASK_MN=mask_mn)
    compiled = full_k.warmup(*args, grid=(count,), **options)
    block = compiled.metadata.block_size
    expected_geometry = (
        dict(threads=(count * block, 1, 1), group_size=(block, 1, 1))
        if compiled.metadata.needs_2d_grid
        else dict(threads=count * block, group_size=block)
    )

    def direct(lib, name, args, **geometry):
        assert args[0] is aa and args[1] is bb and args[2] is cc
        assert geometry == expected_geometry
        result = dispatch(lib, name, args, **geometry)
        routes.append("direct")
        return result

    def host(*args, **kwargs):
        result = launch(*args, **kwargs)
        routes.append("host")
        return result

    monkeypatch.setattr(rt, "dispatch", direct)
    monkeypatch.setattr(utils, "launch", host)
    for poison in (-8192.0, 8192.0):
        cc.fill_(poison)
        handle = full_k[(count,)](
            aa,
            bb,
            cc,
            m,
            n,
            k,
            *aa.stride(),
            *bb.stride(),
            *cc.stride(),
            BM=32,
            BN=32,
            BK=32,
            MASK_MN=mask_mn,
        )
        torch.mps.synchronize()
        assert torch.equal(cc.cpu(), expected)
        assert torch.equal(aa.cpu(), ac) and torch.equal(bb_owner.cpu(), bc)
    assert routes == (["host"] * 2 if layout in ("transpose_offset", "holes") else ["direct"] * 2)
    if k == 1:
        # Equal-to-one specialization folds the loop to a single staged dot,
        # whose masks are replayed verbatim and whose static K tile is 32.
        assert handle.metadata.batched_dot_bounds is None
        assert "smem_dot_" in handle.asm["msl"]
    else:
        assert handle.metadata.batched_dot_bounds[0] == "matmul_full_k_storage_v1"
    for owner, original in zip(owners, before):
        after = owner.cpu()
        if owner is owners[-1] and layout == "contiguous_offset":
            original[8:-8] = expected.reshape(-1)
        assert torch.equal(after, original)
    record_property("route", routes[0])
    record_property("shape", f"{m},{n},{k}")
    record_property("rounded_k", padded)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "m,n,k,mask_mn",
    [
        (32, 96, 64, False),
        (96, 160, 96, False),
        (128, 64, 224, False),
        (192, 320, 128, False),
        (320, 96, 256, False),
        (512, 512, 96, False),
        (1, 7, 1, True),
        (17, 33, 15, True),
        (31, 47, 33, True),
        (70, 97, 47, True),
        (95, 129, 65, True),
    ],
)
def test_full_k_rectangles_and_safe_padded_tails(monkeypatch, record_property, m, n, k, mask_mn, dtype):
    _run(monkeypatch, record_property, m, n, k, dtype, mask_mn)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "layout", ["input_alias", "output_transpose", "contiguous_offset", "transpose_offset", "holes"]
)
def test_full_k_preserves_views_aliases_and_host_fallback(monkeypatch, record_property, dtype, layout):
    _run(monkeypatch, record_property, 96, 96, 64, dtype, False, layout)


@triton.jit
def _tight_masked_dot(A, B, C, M, N, LIMIT: tl.constexpr):
    rm = tl.arange(0, 32)
    rn = tl.program_id(0) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    av = tl.load(A + rm[:, None] * 32 + rk[None, :], rm[:, None] < M, other=0.0)
    bv = tl.load(B + rn[None, :] * 32 + rk[:, None], rn[None, :] < N, other=0.0)
    acc = tl.dot(av, bv)
    # This deliberately is tighter than M. A template clipping only M/N would
    # clobber all the poisoned rows from LIMIT through M-1.
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < LIMIT) & (rn[None, :] < N))


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_dot_tighter_store_keeps_unwritten_rows(monkeypatch, dtype):
    from triton_msl.backend.driver import _get_compile_shader_runtime, _get_utils

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    m, n, limit = 17, 33, 3
    gen = torch.Generator().manual_seed(100103)
    ac = torch.randint(-2, 3, (m, 32), generator=gen).to(dtype)
    bc = torch.randint(-2, 3, (n, 32), generator=gen).to(dtype)
    a, b = ac.to("mps"), bc.to("mps")
    owner = torch.full((m * n + 16,), 731.0, dtype=dtype, device="mps")
    out = owner[8:-8].reshape(m, n)
    rt, utils = _get_compile_shader_runtime(), _get_utils()
    real = rt.dispatch
    calls = []

    def dispatch(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(True)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("proved masked graph should use the direct generic path")

    monkeypatch.setattr(rt, "dispatch", dispatch)
    monkeypatch.setattr(utils, "launch", forbidden)
    for poison in (-8192.0, 8192.0):
        out.fill_(poison)
        handle = _tight_masked_dot[(triton.cdiv(n, 32),)](a, b, out, m, n, LIMIT=limit)
        torch.mps.synchronize()
        expected = torch.full((m, n), poison, dtype=dtype)
        expected[:limit] = (ac[:limit].double() @ bc.double().t()).to(dtype)
        assert torch.equal(out.cpu(), expected)
        actual_owner = owner.cpu()
        assert bool((actual_owner[:8] == 731).all()) and bool((actual_owner[-8:] == 731).all())
        assert torch.equal(a.cpu(), ac) and torch.equal(b.cpu(), bc)
        assert "smem_dot_" in handle.asm["msl"]
    assert len(calls) == 2
