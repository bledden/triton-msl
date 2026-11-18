"""Memory written by a called function or a compare-and-swap comes back to a host tensor.

The list of arguments to copy back after a launch was built from the entry function's
``tl.store`` and ``tl.atomic_*`` read-modify-writes alone. A kernel that stored to one
argument itself and to another through a function that is not inlined, or through
``tl.atomic_cas``, had the second left off the list: a cpu tensor came back unchanged,
with no error. A kernel with no store of its own was right only because an empty list
means copy everything.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")


@triton.jit(noinline=True)
def _bump(X, Y, k):
    tl.store(Y + k, tl.load(X + k) + 1.0)


@triton.jit(noinline=True)
def _bump_through(X, Y, k):
    _bump(X, Y, k)


@triton.jit
def _only_a_call(X, Y, C):
    _bump(C, C, tl.program_id(0))


@triton.jit
def _store_then_call(X, Y, C):
    k = tl.program_id(0)
    tl.store(Y + k, tl.load(X + k) * 2.0)
    _bump(C, C, k)


@triton.jit
def _call_then_store(X, Y, C):
    k = tl.program_id(0)
    _bump(C, C, k)
    tl.store(Y + k, tl.load(X + k) * 2.0)


@triton.jit
def _store_then_nested_call(X, Y, C):
    k = tl.program_id(0)
    tl.store(Y + k, tl.load(X + k) * 2.0)
    _bump_through(C, C, k)


@triton.jit
def _store_then_call_in_a_branch(X, Y, C):
    k = tl.program_id(0)
    tl.store(Y + k, tl.load(X + k) * 2.0)
    if k < 3:
        _bump(C, C, k)


@triton.jit
def _store_then_swap(X, Y, S):
    k = tl.program_id(0)
    tl.store(Y + k, tl.load(X + k) * 2.0)
    tl.atomic_cas(S + k, 0, 9)


@triton.jit
def _swap_then_store(X, Y, S):
    k = tl.program_id(0)
    tl.atomic_cas(S + k, 0, 9)
    tl.store(Y + k, tl.load(X + k) * 2.0)


X0 = torch.tensor([7.0, 6.0, 5.0, 3.0])
C0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
S0 = torch.tensor([0, 2, 0, 4], dtype=torch.int32)
UNSET = torch.full((4,), -7.0)


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize(
    "kernel,stores,bumped",
    [
        (_only_a_call, False, 4),
        (_store_then_call, True, 4),
        (_call_then_store, True, 4),
        (_store_then_nested_call, True, 4),
        (_store_then_call_in_a_branch, True, 3),
    ],
    ids=["only a call", "store then call", "call then store", "nested call", "call in a branch"],
)
def test_a_store_in_a_called_function_comes_back(kernel, stores, bumped, device):
    # A cpu tensor moved to the cpu is the same tensor, so each launch gets copies.
    x, y, c = X0.clone().to(device), UNSET.clone().to(device), C0.clone().to(device)
    for launch in range(1, 4):
        kernel[(4,)](x, y, c)
        want = C0.clone()
        want[:bumped] += launch
        assert torch.equal(c.cpu(), want)
        assert torch.equal(y.cpu(), X0 * 2.0 if stores else UNSET)
        assert torch.equal(x.cpu(), X0)


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("kernel", [_store_then_swap, _swap_then_store], ids=["store then swap", "swap then store"])
def test_a_compare_and_swap_beside_a_store_comes_back(kernel, device):
    x, y, s = X0.clone().to(device), UNSET.clone().to(device), S0.clone().to(device)
    kernel[(4,)](x, y, s)
    assert torch.equal(s.cpu(), torch.tensor([9, 2, 9, 4], dtype=torch.int32))
    assert torch.equal(y.cpu(), X0 * 2.0)
    assert torch.equal(x.cpu(), X0)
