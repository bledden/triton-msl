"""Audit: 1D tt.gather with src size > 1024.

The 1D gather lowering stages the source via:
    if (lid < S) shared[lid] = src_var;

When S > 1024 the threadgroup has at most 1024 threads, so indices >= 1024 in
shared memory are NEVER written. If any index in the index tensor points to
src[1024..S-1], the gather reads uninitialized shared memory and silently
returns wrong numbers.

This test confirms the behavior: either the backend REFUSES loudly, or it
returns CORRECT values. A wrong result with no refusal is a silent-wrong.
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
def _gather1d_large_src(src_ptr, idx_ptr, out_ptr,
                        S: tl.constexpr, I: tl.constexpr):
    """1D gather: out[i] = src[idx[i]], with S=2048 (> 1024 threadgroup cap)."""
    i = tl.arange(0, I)
    src = tl.load(src_ptr + i)       # shape [I] - loads first I of the src
    idx = tl.load(idx_ptr + i)       # shape [I]
    out = tl.gather(src, idx, axis=0)
    tl.store(out_ptr + i, out)


@triton.jit
def _gather1d_large_src_indexed(src_ptr, idx_ptr, out_ptr,
                                S: tl.constexpr, I: tl.constexpr):
    """1D gather: src is S elements; index tensor has I elements pointing into src[0..S-1].

    Specifically tests indices that point INTO the high half (>= 1024) of src.
    """
    i = tl.arange(0, I)
    src = tl.load(src_ptr + tl.arange(0, S))   # src is S=2048 elements
    idx = tl.load(idx_ptr + i)                  # I index values, some >= 1024
    out = tl.gather(src, idx, axis=0)           # gather from the full S-element src
    tl.store(out_ptr + i, out)


@requires
def test_gather1d_src_exceeds_1024():
    """1D gather where src has S=2048 > 1024: must refuse OR produce correct values."""
    torch.manual_seed(42)
    S = 2048
    I = 1024  # index tensor stays at 1024 (fits in one threadgroup)

    src = torch.arange(S, dtype=torch.float32, device=DEV)
    # Indices that specifically point to the high half (>= 1024): these are the
    # elements that would be uninitialized in the faulty staging path.
    idx = torch.randint(1024, S, (I,), device=DEV, dtype=torch.int32)
    out = torch.zeros(I, device=DEV)

    from triton_msl.errors import MetalNonRecoverableError
    try:
        _gather1d_large_src_indexed[(1,)](src, idx, out, S, I)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        # Good: loud refusal instead of silent wrong
        pytest.skip(f"backend refused loudly (correct behavior): {e}")
        return

    # Compute reference on CPU
    ref = src.cpu()[idx.cpu().long()]

    err = (out.cpu() - ref).abs().max().item()
    assert err < 1e-3, (
        f"SILENT-WRONG: 1D gather with S={S}>1024; indices pointing to src[1024..{S-1}] "
        f"returned wrong values (max err={err:.4f}). "
        f"The staging 'if (lid < {S}) shared[lid] = src' only covers lid<1024; "
        f"elements at [1024, {S}) are uninitialized shared memory."
    )


@requires
def test_gather1d_src_2048_indices_in_high_half():
    """Targeted: indices all >= 1024 so EVERY result must read uninitialized memory."""
    torch.manual_seed(0)
    S = 2048
    I = 512  # small enough for one threadgroup

    src = torch.arange(S, dtype=torch.float32, device=DEV) * 1.5 + 0.5
    # ALL indices in [1024, 2047]: every gather result comes from uninitialized shared.
    idx = torch.arange(1024, 1024 + I, device=DEV, dtype=torch.int32)
    out = torch.zeros(I, device=DEV)

    from triton_msl.errors import MetalNonRecoverableError
    try:
        _gather1d_large_src_indexed[(1,)](src, idx, out, S, I)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        pytest.skip(f"backend refused loudly (correct behavior): {e}")
        return

    ref = src.cpu()[idx.cpu().long()]
    err = (out.cpu() - ref).abs().max().item()
    assert err < 1e-3, (
        f"SILENT-WRONG: 1D gather S={S}, all indices in [1024,{1024+I}), "
        f"max_err={err:.4f}. Every result reads from uninitialized shared memory."
    )
