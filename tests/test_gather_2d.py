"""2D tt.gather coverage (roadmap #4).

1D gather already works; 2D was silently mis-computing and is now refused loudly
(see _lower_tt_gather). This suite drives the 2D implementation TDD-style:

  axis=0:  out[i, j] = src[index[i, j], j]
  axis=1:  out[i, j] = src[i, index[i, j]]

Compared against torch.gather on MPS. Refuse-on-budget-overflow stays loud
(never silent-wrong).
"""

import pytest
import torch

try:
    import triton
    import triton.language as tl
    HAS = torch.backends.mps.is_available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="MPS + triton required")
DEV = "mps"


@triton.jit
def _gather2d_axis0(src_ptr, idx_ptr, out_ptr,
                    Ms: tl.constexpr, Mi: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, Mi)[:, None]
    cols = tl.arange(0, N)[None, :]
    srows = tl.arange(0, Ms)[:, None]
    src = tl.load(src_ptr + srows * N + cols)      # [Ms, N]
    idx = tl.load(idx_ptr + rows * N + cols)       # [Mi, N]
    g = tl.gather(src, idx, axis=0)                # out[i,j] = src[idx[i,j], j]
    tl.store(out_ptr + rows * N + cols, g)


@triton.jit
def _gather2d_axis1(src_ptr, idx_ptr, out_ptr,
                    M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    src = tl.load(src_ptr + rows * N + cols)       # [M, N]
    idx = tl.load(idx_ptr + rows * N + cols)       # [M, N]
    g = tl.gather(src, idx, axis=1)                # out[i,j] = src[i, idx[i,j]]
    tl.store(out_ptr + rows * N + cols, g)


@requires
@pytest.mark.parametrize("Ms,Mi,N", [(4, 4, 4), (8, 8, 8), (4, 4, 16)])
def test_gather_2d_axis0(Ms, Mi, N):
    torch.manual_seed(0)
    src = torch.randn(Ms, N, device=DEV)
    idx = torch.randint(0, Ms, (Mi, N), device=DEV, dtype=torch.int32)
    out = torch.empty(Mi, N, device=DEV)
    _gather2d_axis0[(1,)](src, idx, out, Ms, Mi, N)
    torch.mps.synchronize()
    ref = torch.gather(src, 0, idx.long())
    assert torch.allclose(out, ref, atol=1e-5), \
        f"2D gather axis=0 ({Ms},{Mi},{N}): max_diff={(out-ref).abs().max().item():.3e}"


@requires
@pytest.mark.parametrize("Ms,Mi,N", [(4, 8, 4), (2, 8, 4), (8, 16, 2)])
def test_gather_2d_axis0_ragged(Ms, Mi, N):
    """Ragged axis=0: index has more rows than source (the upstream
    [4,4]->[8,4] shape). Source must share the column count and fit the thread
    grid; the index picks any source row. (Shapes kept warp-local: Triton's own
    frontend asserts isWarpLocal() on the gather layout for larger tiles.)"""
    torch.manual_seed(0)
    src = torch.randn(Ms, N, device=DEV)
    idx = torch.randint(0, Ms, (Mi, N), device=DEV, dtype=torch.int32)
    out = torch.empty(Mi, N, device=DEV)
    _gather2d_axis0[(1,)](src, idx, out, Ms, Mi, N)
    torch.mps.synchronize()
    ref = torch.gather(src, 0, idx.long())
    assert torch.allclose(out, ref, atol=1e-5), \
        f"ragged axis=0 ({Ms},{Mi},{N}): max_diff={(out-ref).abs().max().item():.3e}"


@requires
@pytest.mark.parametrize("M,N", [(4, 4), (8, 8), (4, 16)])
def test_gather_2d_axis1(M, N):
    torch.manual_seed(0)
    src = torch.randn(M, N, device=DEV)
    idx = torch.randint(0, N, (M, N), device=DEV, dtype=torch.int32)
    out = torch.empty(M, N, device=DEV)
    _gather2d_axis1[(1,)](src, idx, out, M, N)
    torch.mps.synchronize()
    ref = torch.gather(src, 1, idx.long())
    assert torch.allclose(out, ref, atol=1e-5), \
        f"2D gather axis=1 ({M},{N}): max_diff={(out-ref).abs().max().item():.3e}"


@requires
def test_gather_2d_ragged_refused():
    """A 2D gather whose source differs in shape from the index (ragged gather
    axis) is not yet lowered -- must refuse loudly, never silent-wrong."""
    from triton_msl.errors import MetalNonRecoverableError
    try:
        from triton.compiler.errors import CompilationError as _CErr
    except Exception:
        _CErr = Exception

    @triton.jit
    def _ragged(src_ptr, idx_ptr, out_ptr, Ms: tl.constexpr, Mi: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, Mi)[:, None]; cols = tl.arange(0, N)[None, :]
        srows = tl.arange(0, Ms)[:, None]
        src = tl.load(src_ptr + srows * N + cols)
        idx = tl.load(idx_ptr + rows * N + cols)
        tl.store(out_ptr + rows * N + cols, tl.gather(src, idx, axis=0))

    Ms, Mi, N = 8, 4, 4   # src rows != index rows -> ragged
    src = torch.randn(Ms, N, device=DEV)
    idx = torch.randint(0, Ms, (Mi, N), device=DEV, dtype=torch.int32)
    out = torch.empty(Mi, N, device=DEV)
    with pytest.raises((MetalNonRecoverableError, _CErr)):
        _ragged[(1,)](src, idx, out, Ms, Mi, N)
        torch.mps.synchronize()
