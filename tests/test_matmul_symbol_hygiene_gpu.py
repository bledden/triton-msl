"""Repaired names execute against CPU oracles; never launch an old wrong shader."""

import pytest
import torch
from tests.test_matmul_symbol_hygiene import renamed_kernel, renamed_tile, _tile, _soft, _epilogue, _constant

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("shape", [(32, 32, 32), (64, 96, 64), (96, 64, 96)])
@pytest.mark.parametrize(
    "names",
    [
        ("a", "b", "c", "m", "n", "k"),
        ("i", "r", "gr", "row_base", "col_base", "_K"),
        ("_M", "_N", "_K", "_stage_a", "_matrix_width", "_kb"),
        ("_mmarg1", "_mmarg0", "_mmarg3_buf", "_mmarg5", "_mmarg4", "_mmarg3"),
    ],
)
def test_lower_and_upper_twins_match_independent_oracle(tmp_path, transposed, shape, names):
    m, n, k = shape
    gen = torch.Generator().manual_seed(111 + m + n + k)
    a = torch.randint(-3, 4, (m, k), generator=gen).float()
    b = torch.randint(-3, 4, (k, n), generator=gen).float()
    expected = (a.double() @ b.double()).float()
    aa = a.to("mps")
    bb = b.t().contiguous().to("mps") if transposed else b.to("mps")
    owner = torch.full((m * n + 64,), -16384.0, device="mps")
    cc = owner[32:-32].view(m, n)
    grid = ((m // 32) * (n // 32),)
    for variant in (names, ("AA", "BB", "CC", "MM", "NN", "KK")):
        fn = renamed_kernel(tmp_path, variant, transposed)
        compiled = fn.warmup(aa, bb, cc, m, n, k, grid=grid)
        assert "_mmarg5" in compiled.asm["msl"]
        for poison in (-8192.0, 8192.0):
            cc.fill_(poison)
            fn[grid](aa, bb, cc, m, n, k)
            torch.mps.synchronize()
            assert torch.equal(cc.cpu(), expected)
            after = owner.cpu()
            assert bool((after[:32] == -16384).all() & (after[-32:] == -16384).all())
            assert torch.equal(aa.cpu(), a)
            assert torch.equal(bb.cpu(), b.t() if transposed else b)


@pytest.mark.parametrize(
    "original,names",
    [
        (_tile, ("c", "r", "i")),
        (_soft, ("c", "i", "row")),
        (_epilogue, ("c", "i", "row", "col")),
        (_constant, ("_i",)),
    ],
)
def test_related_templates_execute_hygienic_names(tmp_path, original, names):
    gen = torch.Generator().manual_seed(191)
    a = torch.randn((32, 32), generator=gen) * 0.125
    b = torch.randn((32, 32), generator=gen) * 0.125
    bias = torch.randn((32,), generator=gen) * 0.125
    expected = a.double() @ b.double()
    if original is _soft:
        expected = torch.softmax(expected, dim=1)
    elif original is _epilogue:
        expected = torch.clamp(expected + bias.double(), min=0)
    elif original is _constant:
        expected = torch.full((32, 32), 64.0, dtype=torch.float64)
    owner = torch.full((1088,), -16384.0, device="mps")
    cc = owner[32:-32].view(32, 32)
    aa, bb = a.to("mps"), b.to("mps")
    args = (cc,) if original is _constant else (aa, bb, cc, bias.to("mps")) if original is _epilogue else (aa, bb, cc)
    fn = renamed_tile(tmp_path, original, names)
    for poison in (-8192.0, 8192.0):
        cc.fill_(poison)
        fn[(1,)](*args)
        torch.mps.synchronize()
        torch.testing.assert_close(cc.cpu().double(), expected, rtol=2e-5, atol=2e-6)
        after = owner.cpu()
        assert bool((after[:32] == -16384).all() & (after[-32:] == -16384).all())


def test_pointer_extent_names_in_single_tile_strided_kernel(tmp_path):
    from tests.test_matmul_symbol_hygiene import _tile_bt

    fn = renamed_tile(tmp_path, _tile_bt, ("M", "N", "K"))
    gen = torch.Generator().manual_seed(127)
    a = torch.randint(-2, 3, (32, 32), generator=gen).float()
    bo = torch.randint(-2, 3, (32, 32), generator=gen).float()
    aa, bb = a.to("mps"), bo.to("mps")
    cc = torch.full((32, 32), -8192.0, device="mps")
    fn[(1,)](aa, bb, cc)
    torch.mps.synchronize()
    assert torch.equal(cc.cpu(), (a.double() @ bo.double().t()).float())


def test_mlx_constant_dot_public_route_with_generated_local_name(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from triton_msl.mlx import triton_call

    fn = renamed_tile(tmp_path, _constant, ("_i",))
    (out,) = triton_call(fn, mx.zeros((32, 32)), grid=(1,), output_arg_indices=[0])
    mx.eval(out)
    assert bool(mx.all(out == 64.0).item())
