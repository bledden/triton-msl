"""Track the former blanket refusal for tiled matmuls without M/N scalar args.

Source-proved leading strides and a proved program grid now recover that capability.
The positive checks every tile against an independent oracle. A CPU collision witness
retains why the historical BN-as-global-stride fixture was undefined for its 2x2 grid;
that raced source must not be used as a numerical positive.

Unsupported-pattern refusal controls remain in their dedicated suites, including
``test_nover_store_refusal``, ``test_atomic_nover_refusal`` and
``test_inloop_reduce_coverage``. This capability transition does not weaken them.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton_msl.errors import MetalNonRecoverableError

    HAS = torch.backends.mps.is_available()
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:

    @triton.jit
    def _kloop_constexpr_mn_matmul(A, B, C, K, LD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """A K-loop whose source literal leading stride is independent of BN.

        No M/N scalar supplies output clips. The 2D grid and actual source
        strides determine every full tile, including multiple column programs.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offm = pid_m * BM + tl.arange(0, BM)
        offn = pid_n * BN + tl.arange(0, BN)
        offk = tl.arange(0, BK)
        a_ptrs = A + offm[:, None] * K + offk[None, :]
        b_ptrs = B + offk[:, None] * LD + offn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, K, BK):  # runtime K -> scf.for -> K-loop matmul path
            acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
            a_ptrs += BK
            b_ptrs += BK * LD
        tl.store(C + offm[:, None] * LD + offn[None, :], acc)

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
def test_kloop_literal_strides_without_mn_scalars_write_every_source_tile(monkeypatch):
    """A proved full-tile source needs no guessed M/N extent.

    The historical fixture used BN32 as B/C leading stride on a 2x2 column
    grid, causing overlapping output writes. Explicit LD64 makes that intended
    64x64 matrix source well-defined while preserving absent M/N scalar args.
    """
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    BM = BN = BK = 32
    M = N = K = 64
    generator = torch.Generator().manual_seed(327264)
    ah = torch.randn(M, K, generator=generator)
    bh = torch.randn(K, N, generator=generator)
    a, b = ah.to("mps"), bh.to("mps")
    owner = torch.full((M * N + 64,), -17391.0, device="mps")
    c = owner[32:-32].view(M, N)
    handle = _kloop_constexpr_mn_matmul[(2, 2)](a, b, c, K, LD=N, BM=BM, BN=BN, BK=BK)
    assert handle.metadata.mm_two_kernel is None
    expected = (ah.double() @ bh.double()).float()
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-3, atol=1e-3)
    assert torch.all(owner[:32].cpu() == -17391.0) and torch.all(owner[-32:].cpu() == -17391.0)
    assert torch.equal(a.cpu(), ah) and torch.equal(b.cpu(), bh)


def test_historical_bn_stride_fixture_has_overlapping_output_coordinates_cpu():
    # The two distinct source cells are owned by different programs in the
    # historical 2x2 grid, so arbitrary matmul values could race at offset32.
    first = (0, 32)  # pid(0,1), local(0,0)
    second = (1, 0)  # pid(0,0), local(1,0)
    offset = lambda cell, leading: cell[0] * leading + cell[1]
    assert first != second and offset(first, 32) == offset(second, 32) == 32
    assert offset(first, 64) != offset(second, 64)
