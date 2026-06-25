"""Regression: batched matmul with a CONSTEXPR-FOLDED batch offset must REFUSE.

A confirming re-audit found the f43af78 B4 guard caught only the runtime-stride
batch spelling (z*sab); the constexpr-folded spelling (z*M*K, M/K constexpr ->
a folded constant referencing no arg) slipped through and SILENTLY computed only
batch 0. The fix: a uniform program_id-derived base advance is a tile advance
ONLY if provably so (multiplies an inferred 2-D stride arg, OR is a folded
multiple equal to a make_range tile extent); otherwise REFUSE (ambiguity = refuse).
The fuzzer only spelled the batch offset with a runtime stride arg, so this form
was structurally untested.
"""
import pytest
import torch

try:
    import triton
    import triton.language as tl
    _HAS = True
except Exception:
    _HAS = False

from triton_msl.errors import MetalNonRecoverableError

requires = pytest.mark.skipif(not _HAS, reason="triton not available")


if _HAS:
    @triton.jit
    def _cbmm(a, b, c, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             sam, sak, sbk, sbn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pz = tl.program_id(0)
        om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, BK)
        ap = a + pz * (M * K) + (om[:, None] * sam + ok[None, :] * sak)
        bp = b + pz * (K * N) + (ok[:, None] * sbk + on[None, :] * sbn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(bp)); ap += BK * sak; bp += BK * sbk
        tl.store(c + pz * (M * N) + (om[:, None] * scm + on[None, :] * scn), acc)


@requires
def test_constexpr_folded_batched_matmul_refuses():
    import os
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    Z = M = N = K = 32
    A = torch.randn(Z, M, K, device="mps"); B = torch.randn(Z, K, N, device="mps")
    C = torch.zeros(Z, M, N, device="mps")
    # The batch offset z*(M*K) is a constexpr-folded constant; the simdgroup template
    # maps only program_id->2-D tile and would drop the batch -> must refuse loudly,
    # NOT silently compute only batch 0.
    with pytest.raises(MetalNonRecoverableError):
        _cbmm[(Z,)](A, B, C, Z, M, N, K,
                    A.stride(1), A.stride(2), B.stride(1), B.stride(2),
                    C.stride(1), C.stride(2), BM=M, BN=N, BK=K)
        torch.mps.synchronize()
