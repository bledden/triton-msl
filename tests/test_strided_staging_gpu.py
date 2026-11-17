"""Race-free, one-program output/input aliases use pre-launch snapshot oracles."""

import pytest
import torch

from tests.test_full_k_direct_gpu import full_k


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("mode", ["c_a", "c_b", "all", "shifted"])
def test_single_program_output_alias_is_snapshot_faithful(mode):
    gen = torch.Generator().manual_seed(2233)
    owner = torch.full((1120,), 173.0, device="mps")
    aa = owner[32:1056].view(32, 32)
    other = torch.randint(-2, 3, (32, 32), generator=gen).float().to("mps")
    if mode == "c_b":
        aa, bo = other, aa
        cc = bo
    else:
        bo = aa if mode == "all" else other
        cc = owner[(48 if mode == "shifted" else 32) : (1072 if mode == "shifted" else 1056)].view(32, 32)
    bb = bo.t()
    for trial in range(2):
        owner.copy_(torch.randint(-2, 3, owner.shape, generator=gen).float())
        ac, bc = aa.cpu().clone(), bb.cpu().clone()
        before = owner.cpu().clone()
        expected = (ac.double() @ bc.double()).float()
        args = (aa, bb, cc, 32, 32, 32, *aa.stride(), *bb.stride(), *cc.stride())
        opts = dict(BM=32, BN=32, BK=32, MASK_MN=True)
        compiled = full_k.warmup(*args, grid=(1,), **opts)
        assert "_stage_a[1056]" in compiled.asm["msl"]
        full_k[(1,)](*args, **opts)
        torch.mps.synchronize()
        assert torch.equal(cc.cpu(), expected), (mode, trial)
        after = owner.cpu()
        offset = cc.storage_offset()
        assert torch.equal(after[:offset], before[:offset])
        assert torch.equal(after[offset + cc.numel() :], before[offset + cc.numel() :])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("kind", ["nan", "infinity", "signed_zero"])
def test_staged_exceptional_values_and_zero_classification(kind):
    ac = torch.ones((32, 32))
    bc = torch.ones((32, 32))
    if kind == "nan":
        ac[2, 4] = float("nan")
        bc[7, 3] = float("nan")
    elif kind == "infinity":
        ac[0, 0], ac[1, 0] = float("inf"), -float("inf")
        bc[0, 1] = 0.0
    else:
        ac[:, ::2], ac[:, 1::2] = 0.0, -0.0
        bc[:, ::2], bc[:, 1::2] = 1.0, -1.0
    expected = (ac.double() @ bc.double()).float()
    aa, bb = ac.to("mps"), bc.t().contiguous().to("mps").t()
    cc = torch.empty_like(aa)
    args = (aa, bb, cc, 32, 32, 32, *aa.stride(), *bb.stride(), *cc.stride())
    opts = dict(BM=32, BN=32, BK=32, MASK_MN=True)
    compiled = full_k.warmup(*args, grid=(1,), **opts)
    assert "_stage_a[1056]" in compiled.asm["msl"]
    for poison in [-8192.0, 8192.0]:
        cc.fill_(poison)
        full_k[(1,)](*args, **opts)
        torch.mps.synchronize()
        actual = cc.cpu()
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isinf(actual), torch.isinf(expected))
        finite = torch.isfinite(expected)
        assert torch.equal(actual[finite], expected[finite])
        defined_sign = ~torch.isnan(expected)
        assert torch.equal(torch.signbit(actual[defined_sign]), torch.signbit(expected[defined_sign]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("shape", [(17, 33, 33), (70, 97, 47), (32, 96, 64), (128, 64, 224)])
@pytest.mark.parametrize("kind", ["precision", "random", "cancellation"])
def test_matrix_fp32_precision_and_source_full_k(shape, kind):
    m, n, k = shape
    kp = (k + 31) // 32 * 32
    gen = torch.Generator().manual_seed(671 + m + n + k)
    a = torch.randn((m, kp), generator=gen) * 0.25
    bo = torch.randn((n, kp), generator=gen) * 0.25
    if kind == "precision":
        a = 1 + torch.randint(1, 4, (m, kp), generator=gen).float() * 2**-13
        bo = 1 + torch.randint(1, 4, (n, kp), generator=gen).float() * 2**-13
    elif kind == "cancellation":
        a[:, 1::2] = a[:, ::2]
        bo[:, 1::2] = -bo[:, ::2]
    expected = a.double() @ bo.double().t()
    bound = (3 * 2**-24 * kp) * (a.double().abs() @ bo.double().abs().t()) + 1e-7
    aa, bb = a.to("mps"), bo.to("mps").t()
    owner = torch.full((m * n + 64,), -32768.0, device="mps")
    cc = owner[32 : 32 + m * n].view(m, n)
    args = (aa, bb, cc, m, n, k, *aa.stride(), *bb.stride(), *cc.stride())
    opts = dict(BM=32, BN=32, BK=32, MASK_MN=True)
    grid = (((m + 31) // 32) * ((n + 31) // 32),)
    compiled = full_k.warmup(*args, grid=grid, **opts)
    assert "simdgroup_multiply_accumulate" in compiled.asm["msl"]
    for poison in [-8192.0, 8192.0]:
        cc.fill_(poison)
        full_k[grid](*args, **opts)
        torch.mps.synchronize()
        actual = cc.cpu().double()
        assert bool(torch.isfinite(actual).all())
        assert bool(((actual - expected).abs() <= bound).all())
        after = owner.cpu()
        assert bool((after[:32] == -32768).all()) and bool((after[-32:] == -32768).all())
        assert torch.equal(aa.cpu(), a) and torch.equal(bb.cpu(), bo.t())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("fallback", ["feature", "width"])
def test_matrix_fallback_executes_same_shader_scalar_branch(fallback):
    # Execute emitted source with only its hardware eligibility forced false.
    from tests.test_strided_staging import _source

    source = _source()
    if fallback == "feature":
        source = source.replace("using namespace metal;", "#undef __HAVE_SIMDGROUP_MATRIX__\nusing namespace metal;")
    else:
        source = source.replace("_matrix_width == 32u", "_matrix_width == 0u")
    lib = torch.mps.compile_shader(source)
    gen = torch.Generator().manual_seed(592)
    a = torch.randint(-2, 3, (17, 64), generator=gen).float()
    bo = torch.randint(-2, 3, (33, 64), generator=gen).float()
    aa, bb = a.to("mps"), bo.to("mps").t()
    cc = torch.full((17, 33), -8192.0, device="mps")
    # constexpr args are absent from the emitted ABI.
    lib.full_k(aa, bb, cc, 17, 33, 47, 64, 64, 33, threads=(512, 1, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()
    assert torch.equal(cc.cpu(), (a.double() @ bo.double().t()).float())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
def test_matrix_branch_has_execution_witness_and_scalar_counterfactual():
    from tests.test_strided_staging import _source

    source = _source()
    # Test-only uniform returns disable both scalar alternatives. Matrix code,
    # including every load, barrier and store, is unchanged. If it did not run,
    # the output remains poisoned. Forcing its guard false is the red control.
    marker = "#endif\n        uint _m0"
    assert source.count(marker) == 1
    witness = source.replace(marker, "#endif\n        return;\n        uint _m0")
    marker2 = "    float _cell[4];\n    for (uint _e = lid;"
    assert witness.count(marker2) == 1
    witness = witness.replace(marker2, "    return;\n" + marker2)
    gen = torch.Generator().manual_seed(810)
    a = torch.randn((70, 96), generator=gen) * 0.125
    bo = torch.randn((97, 96), generator=gen) * 0.125
    expected = a.double() @ bo.double().t()
    bound = 3 * 2**-24 * 96 * (a.double().abs() @ bo.double().abs().t()) + 1e-7
    aa, bb = a.to("mps"), bo.to("mps").t()
    owner = torch.full((70 * 97 + 64,), -16384.0, device="mps")
    cc = owner[32:-32].view(70, 97)
    for code, writes in [
        (witness, True),
        (witness.replace("_matrix_width == 32u", "_matrix_width == 0u"), False),
        (source.replace("_matrix_width == 32u", "_matrix_width == 0u"), True),
    ]:
        lib = torch.mps.compile_shader(code)
        cc.fill_(-8192.0)
        lib.full_k(aa, bb, cc, 70, 97, 65, 96, 96, 97, threads=(12 * 256, 1, 1), group_size=(256, 1, 1))
        torch.mps.synchronize()
        actual = cc.cpu().double()
        if writes:
            assert bool(((actual - expected).abs() <= bound).all())
        else:
            assert bool((actual == -8192.0).all())
        after = owner.cpu()
        assert bool((after[:32] == -16384.0).all() & (after[-32:] == -16384.0).all())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
def test_outer_guard_false_zero_reduction_executes_unstaged_path():
    from tests.test_strided_staging import _source

    source = _source()
    # The real _K > 0 guard is false. Disable only the final scalar fallback
    # in the red control; unchanged source must write zeros, control no values.
    marker = "    float _cell[4];\n    for (uint _e = lid;"
    assert source.count(marker) == 1
    a = torch.ones((32, 32), device="mps")
    b = torch.ones((32, 32), device="mps")
    owner = torch.full((1088,), -16384.0, device="mps")
    c = owner[32:-32].view(32, 32)
    for code, expected in [(source, 0.0), (source.replace(marker, "    return;\n" + marker), -8192.0)]:
        c.fill_(-8192.0)
        lib = torch.mps.compile_shader(code)
        lib.full_k(a, b, c, 32, 32, 0, 32, 32, 32, threads=(256, 1, 1), group_size=(256, 1, 1))
        torch.mps.synchronize()
        assert bool((c.cpu() == expected).all())
        assert bool((owner.cpu()[:32] == -16384.0).all() & (owner.cpu()[-32:] == -16384.0).all())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("k", [64, 96])
@pytest.mark.parametrize("mode", ["c_a", "c_b", "all", "shifted"])
def test_single_program_alias_across_multiple_k_chunks(k, mode):
    gen = torch.Generator().manual_seed(1000 + k)
    owner = torch.randint(-2, 3, (32 * k + 96,), generator=gen).float().to("mps")
    shared = owner[32 : 32 + 32 * k].view(32, k)
    other = torch.randint(-2, 3, (32, k), generator=gen).float().to("mps")
    aa, bo = (other, shared) if mode == "c_b" else (shared, shared if mode == "all" else other)
    offset = 48 if mode == "shifted" else 32
    cc = owner[offset : offset + 1024].view(32, 32)
    before = owner.cpu().clone()
    ac = aa.cpu().clone()
    bc = bo.cpu().clone()
    expected = (ac.double() @ bc.double().t()).float()
    bb = bo.t()
    full_k[(1,)](aa, bb, cc, 32, 32, k, *aa.stride(), *bb.stride(), *cc.stride(), BM=32, BN=32, BK=32, MASK_MN=True)
    torch.mps.synchronize()
    assert torch.equal(cc.cpu(), expected)
    after = owner.cpu()
    assert torch.equal(after[:offset], before[:offset])
    assert torch.equal(after[offset + 1024 :], before[offset + 1024 :])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("kind", ["nan", "infinity", "signed_zero"])
def test_multitile_exceptional_classification(kind):
    a = torch.ones((64, 96))
    bo = torch.ones((96, 96))
    if kind == "nan":
        a[39, 65] = float("nan")
        bo[70, 39] = float("nan")
    elif kind == "infinity":
        a[33, 40] = float("inf")
        a[34, 40] = -float("inf")
        bo[63, 40] = 0.0
    else:
        a[:, ::2] = 0.0
        a[:, 1::2] = -0.0
        bo[:, 1::2] = -1.0
    expected = (a.double() @ bo.double().t()).float()
    aa, bb = a.to("mps"), bo.to("mps").t()
    owner = torch.full((64 * 96 + 64,), -16384.0, device="mps")
    cc = owner[32:-32].view(64, 96)
    for poison in (-8192.0, 8192.0):
        cc.fill_(poison)
        full_k[(6,)](
            aa, bb, cc, 64, 96, 96, *aa.stride(), *bb.stride(), *cc.stride(), BM=32, BN=32, BK=32, MASK_MN=True
        )
        torch.mps.synchronize()
        actual = cc.cpu()
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isinf(actual), torch.isinf(expected))
        finite = torch.isfinite(expected)
        assert torch.equal(actual[finite], expected[finite])
        defined = ~torch.isnan(expected)
        assert torch.equal(torch.signbit(actual[defined]), torch.signbit(expected[defined]))
        after = owner.cpu()
        assert bool((after[:32] == -16384.0).all() & (after[-32:] == -16384.0).all())
