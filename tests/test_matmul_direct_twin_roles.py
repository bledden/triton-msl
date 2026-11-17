"""The ``__mmdirect`` twin and its launch descriptor bind extents by role, not by the names M/N/K."""

import itertools

import pytest
import torch

from tests.test_matmul_symbol_hygiene import emit_names, renamed_kernel

# renamed_kernel assigns names[3:] to the (rows, cols, depth) roles in that order.
ORDERS = [("A", "B", "C") + order for order in itertools.permutations(("M", "N", "K"))]
NATURAL = ("A", "B", "C", "M", "N", "K")


@pytest.mark.parametrize("names", ORDERS, ids=lambda names: "".join(names[3:]))
def test_twin_text_does_not_depend_on_which_role_is_named_m_n_k(tmp_path, names):
    source = emit_names(tmp_path, names, False)
    twin = source.split("__mmdirect(", 1)[1]
    assert "long _M = (long)(_mmarg3);" in twin
    assert "long _N = (long)(_mmarg4);" in twin
    assert "long _K = (long)_mmarg5;" in twin
    assert source == emit_names(tmp_path, NATURAL, False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("shape", [(64, 32, 64), (32, 64, 64), (64, 64, 32), (96, 64, 32), (64, 96, 128)])
@pytest.mark.parametrize("names", ORDERS, ids=lambda names: "".join(names[3:]))
def test_aligned_non_square_shapes_match_oracle_for_every_name_order(tmp_path, names, shape, device):
    rows, cols, depth = shape
    gen = torch.Generator().manual_seed(211 + rows + 3 * cols + 5 * depth)
    a = torch.randint(-3, 4, (rows, depth), generator=gen).float()
    b = torch.randint(-3, 4, (depth, cols), generator=gen).float()
    expected = (a.double() @ b.double()).float()
    aa, bb = a.to(device), b.to(device)
    owner = torch.full((rows * cols + 8192,), -16384.0, device=device)
    cc = owner[4096:-4096].view(rows, cols)
    grid = ((rows // 32) * (cols // 32),)
    fn = renamed_kernel(tmp_path, names, False)
    for poison in (-8192.0, 8192.0):
        cc.fill_(poison)
        fn[grid](aa, bb, cc, rows, cols, depth)
        if device == "mps":
            torch.mps.synchronize()
        assert torch.equal(cc.cpu(), expected)
        after = owner.cpu()
        assert bool((after[:4096] == -16384).all() & (after[-4096:] == -16384).all())
        assert torch.equal(aa.cpu(), a) and torch.equal(bb.cpu(), b)
