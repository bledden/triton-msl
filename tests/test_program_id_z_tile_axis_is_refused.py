"""A nested program_id(2) tile axis is REFUSED, not emitted minus its z.

`_emit_tile_ids_from_source_grid` maps exactly two tile axes — pid_m <- _ta[0],
pid_n <- _ta[1]. Two axes on any pair emit correctly (x/y, or y/z beside a batch
on x — that is the batched-matmul path, and it is validated). But a THIRD
non-batch tile axis — a program_id(2) folded into a tile coordinate with no batch
pointer offset to absorb it — reaches emission and is then dropped, losing its
coordinate SILENTLY. CodeRabbit flagged exactly this on PR #6.

This pins the refusal: the kernel below puts program_id(2) into the ROW tile
(alongside program_id(0)), so all three axes are tiles and none is a batch. It
must raise `MetalNonRecoverableError`, not compile a kernel that computes the
wrong rows for every z.
"""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.errors import MetalNonRecoverableError

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _mm_three_tile_axes(a, b, c, M, N, K,
                            sam, sak, sbk, sbn, scm, scn,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """program_id(2) is folded into the ROW tile — a genuine third tile axis,
        NOT a batch pointer offset (no `a + pid*stride_batch`). So the grid is
        (z, 1, 1)-tiled on all three axes and z has nowhere to go at emission."""
        pm = tl.program_id(0) * 2 + tl.program_id(2)     # z contributes to rows
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        ap = a + (rm[:, None] * sam + rk[None, :] * sak)
        bp = b + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(bp))
            ap += BK * sak
            bp += BK * sbk
        tl.store(c + (rm[:, None] * scm + rn[None, :] * scn),
                 acc.to(c.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


@requires
def test_a_third_tile_axis_is_refused_not_dropped():
    M = N = K = 64
    BM = BN = BK = 32
    A = torch.randn(M, K, device="mps")
    B = torch.randn(K, N, device="mps")
    C = torch.zeros(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="tile axes"):
        _mm_three_tile_axes[(1, (N + BN - 1) // BN, 2)](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
            C.stride(0), C.stride(1), BM=BM, BN=BN, BK=BK)
        torch.mps.synchronize()
