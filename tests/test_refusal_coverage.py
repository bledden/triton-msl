"""Systematic refusal-coverage for the never-silent-wrong integrity contract.

The headline safety claim is "kernels we cannot lower correctly are *refused*
(`MetalNonRecoverableError`), never silent-wrong." This session's history shows
that claim has had real holes that were caught + closed (the tridec in-loop-reduce
under-coverage; the n>1-per-thread store/atomic under-coverage; the
`test_constexpr_if_return` `pid+0` garbage a weak assert masked). Those individual
refusals have their own tests (`test_nover_store_refusal.py`,
`test_atomic_nover_refusal.py`, `test_inloop_reduce_coverage.py`). This file is the
*consolidated* regression guard: it asserts representative unsupported patterns from
the integrity-prescan catalog REFUSE loudly rather than emit (possibly-wrong) output.
A passing test is not the same as a correct kernel — so for these we require a loud
failure, not a quiet success. Serial GPU.
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

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _kloop_constexpr_mn_matmul(A, B, C, K, BM: tl.constexpr, BN: tl.constexpr,
                                   BK: tl.constexpr):
        """A K-loop matmul that tiles its output across programs (program_id on
        axes 0 AND 1) but bakes M/N as constexpr — there is no runtime scalar arg
        named M/N, so the K-loop matmul template cannot recover the true output
        stride (it would guess stride == BLOCK and silently produce wrong output
        for a tiled grid: the `test_dot_mulbroadcasted` class, ~98% mismatch). The
        integrity guard (`_lower_k_loop_dot_inline`) must refuse this loudly."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offm = pid_m * BM + tl.arange(0, BM)
        offn = pid_n * BN + tl.arange(0, BN)
        offk = tl.arange(0, BK)
        a_ptrs = A + offm[:, None] * K + offk[None, :]
        b_ptrs = B + offk[:, None] * BN + offn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, K, BK):     # runtime K -> scf.for -> K-loop matmul path
            acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
            a_ptrs += BK
            b_ptrs += BK * BN
        tl.store(C + offm[:, None] * BN + offn[None, :], acc)

    _MODULE_KERNELS = (_kloop_constexpr_mn_matmul,)
else:
    _MODULE_KERNELS = ()


@pytest.fixture(autouse=True)
def _clear_jit_cache():
    if HAS:
        for _fn in _MODULE_KERNELS:
            _fn.device_caches.clear()
    yield


@requires_metal
def test_kloop_constexpr_mn_matmul_refuses():
    """K-loop matmul tiling the output across programs (program_id axes {0,1}) with
    M/N baked as constexpr (no runtime M/N scalar args) must refuse loudly — the
    template cannot derive the true output stride, and guessing it is a silent-wrong
    (`test_dot_mulbroadcasted`). Backs the integrity guard at
    `_lower_k_loop_dot_inline` (`_lowerer_templates.py`)."""
    BM = BN = BK = 32
    M = BM * 2          # 2x2 program grid -> the constexpr stride guess collapses
    N = BN * 2
    K = BK * 2          # runtime K -> the dot sits in a real scf.for (K-loop path)
    torch.manual_seed(0)
    A = torch.randn(M, K, device="mps", dtype=torch.float32)
    B = torch.randn(K, N, device="mps", dtype=torch.float32)
    C = torch.zeros(M, N, device="mps", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError):
        _kloop_constexpr_mn_matmul[(2, 2)](A, B, C, K, BM=BM, BN=BN, BK=BK)
