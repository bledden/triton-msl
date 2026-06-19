"""Base-path BLOCK-wide atomic under-coverage refusal.

A base-path atomic scatter emits PTR[k + lid] op= val (one element per thread).
When the tile exceeds the threadgroup (block_size > num_threads, i.e. n>1
elements per thread) it silently covers only `lid` of each tile-stride and
DROPS the rest — the same silent-wrong class that the store-only guard
(b82136b) left open for tt.atomic_rmw / tt.atomic_cas. The correct n>1 atomic
paths are the MEPT register-array regime and the _loop_e-wrapped emission; when
neither applies the atomic must refuse loudly. Launching with num_warps =
BLOCK/32 (num_threads == BLOCK, n=1) is verified correct.

A pure-elementwise BLOCK-wide atomic is already covered correctly under the
default flag (register-array-scattered) or by the _loop_e wrap, so it never
reaches the base path. The base-path under-cover only happens when a kernel is
forced onto multipass (a 1-D full reduce → block_size = num_threads) AND its
multi-element value is NOT register-array / wrap covered (a rank-changing
tl.reshape defeats the MEPT spine) AND a BLOCK-wide atomic sits in a
control-flow region. `_undercover_atomic` reproduces exactly that structure —
mirroring tests/test_nover_store_refusal.py. Serial GPU.
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
    def _undercover_atomic(X, OUT, OUT2, K, BLOCK: tl.constexpr, H: tl.constexpr):
        o = tl.arange(0, BLOCK)
        v = tl.load(X + o)
        s = tl.sum(v)                            # 1-D full reduce -> multipass,
        #                                          block_size collapses to num_threads
        w = tl.reshape(v + s, (H, BLOCK // H))   # rank-changing reshape defeats
        #                                          the MEPT register-array spine
        rm = tl.arange(0, H)[:, None]
        rn = tl.arange(0, BLOCK // H)[None, :]
        tl.store(OUT2 + rm * (BLOCK // H) + rn, w)
        for k in range(0, K):                    # BLOCK-wide atomic in a control-
            tl.atomic_add(OUT + o, v + s + k)    # flow region -> base path scatter

# All @triton.jit kernels defined in this file.  The autouse fixture below
# clears their in-process JIT caches before each test.
_MODULE_KERNELS = (_undercover_atomic,) if HAS else ()


@pytest.fixture(autouse=True)
def _clear_jit_cache():
    """Triton's in-process JIT cache is keyed on signature/constexprs, NOT on
    num_warps in a way that distinguishes a refusing vs. correct compile across
    tests sharing a kernel — clear device_caches before each test so every test
    compiles fresh against its own launch parameters."""
    if HAS:
        for _fn in _MODULE_KERNELS:
            _fn.device_caches.clear()
    yield


@requires_metal
def test_blockwide_atomic_refuses_when_undercover():
    """BLOCK=256 with num_warps=4 → 128 threads, n=2 per thread. The base path
    would scatter only one element per thread, silently dropping half of each
    tile-stride. Must refuse loudly."""
    BLOCK, H, K = 256, 2, 3
    torch.manual_seed(0)
    X = torch.randn(BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(BLOCK, device="mps", dtype=torch.float32)
    OUT2 = torch.zeros(BLOCK, device="mps", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError):
        _undercover_atomic[(1,)](X, OUT, OUT2, K, BLOCK=BLOCK, H=H, num_warps=4)


@requires_metal
def test_blockwide_atomic_correct_when_matched():
    """BLOCK=256 with num_warps=8 → 256 threads, n=1 per thread. The base path
    covers the whole tile (one element per thread) → correct, no refusal."""
    BLOCK, H, K = 256, 2, 3
    torch.manual_seed(0)
    X = torch.randn(BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(BLOCK, device="mps", dtype=torch.float32)
    OUT2 = torch.zeros(BLOCK, device="mps", dtype=torch.float32)
    _undercover_atomic[(1,)](X, OUT, OUT2, K, BLOCK=BLOCK, H=H, num_warps=8)
    # OUT accumulates v + s + k over all K loop iterations (atomic_add into zeros).
    s = X.sum()
    ref = sum((X + s + k) for k in range(K))
    torch.testing.assert_close(OUT, ref, rtol=1e-3, atol=1e-3)
