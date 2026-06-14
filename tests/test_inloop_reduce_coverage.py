"""In-loop reduction coverage (spec 2026-06-13-inloop-reduce-coverage).

A tt.reduce inside a runtime loop must NEVER silently sum only the first
num_threads elements when block_size > num_threads. Under MEPT=0 (no register
arrays) such a reduce must refuse loudly (Stage B); under the default flag the
common where-on-reduce shape must compute correctly (Stage C). Serial GPU.
"""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    from triton_metal.errors import MetalNonRecoverableError
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _sum_carry_in_loop(X, OUT, C: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, C):
            v = tl.load(X + i * BLOCK + tl.arange(0, BLOCK))
            acc = acc + tl.sum(v)
        tl.store(OUT + tl.arange(0, 1), acc)

    @triton.jit
    def _min_blocksum_in_loop(X, OUT, C: tl.constexpr, BLOCK: tl.constexpr):
        best = tl.zeros((), dtype=tl.float32) + 1e30
        for i in range(0, C):
            v = tl.load(X + i * BLOCK + tl.arange(0, BLOCK))
            s = tl.sum(v)                       # in-loop reduce -> scalar
            best = tl.where(s < best, s, best)  # cmpf + select on reduce result
        tl.store(OUT + tl.arange(0, 1), best)

# All @triton.jit kernels defined in this file.  The autouse fixture below
# clears their in-process JIT caches before each test.  Add any new kernel
# defined in this file to this tuple so the cache flush covers it.
_MODULE_KERNELS = (_sum_carry_in_loop, _min_blocksum_in_loop) if HAS else ()


@pytest.fixture(autouse=True)
def _clear_jit_cache():
    """Triton's in-process JIT cache is keyed on signature/constexprs, NOT on
    TRITON_METAL_MEPT, so a compile made under one flag could be served to a
    test that sets a different flag — a false negative for the pytest.raises
    refusal test.  Clear device_caches (a defaultdict keyed by device; its
    first value is the kernel_cache dict) before each test so every test
    compiles fresh against the current env."""
    if HAS:
        for _fn in _MODULE_KERNELS:
            _fn.device_caches.clear()
    yield


@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512])
def test_inloop_reduce_mept0_refuses(BLOCK, monkeypatch):
    """MEPT=0: an in-loop reduce with block>num_threads is uncovered → refuse
    loudly (was silent-wrong before Stage B)."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    C = 4
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError):
        _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)


@requires_metal
def test_inloop_reduce_small_block_ok(monkeypatch):
    """block_size <= num_threads is fully covered (one elem/thread) → never
    refused, correct under both flags."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    # BLOCK=128 with the default num_warps=4 → 128 threads, so
    # block_size (128) <= num_threads (128): the reduce is NOT refused.
    # Do not lower num_warps here — that would push block_size > num_threads
    # and silently convert this into a false-positive pass of the refusal test.
    BLOCK, C = 128, 4
    torch.manual_seed(0)
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)
    torch.testing.assert_close(OUT[0], X.sum(), rtol=1e-3, atol=1e-3)


@requires_metal
@pytest.mark.parametrize("BLOCK", [128, 256, 512, 1024])
def test_inloop_where_on_reduce_default_correct(BLOCK, monkeypatch):
    """Default flag: a where (cmpf+select) consuming an in-loop reduce result
    is register-array-eligible (Stage C) → correct at full SIMD width."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "1")
    C = 4
    torch.manual_seed(0)
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    _min_blocksum_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)
    ref = X.view(C, BLOCK).sum(dim=1).min()
    torch.testing.assert_close(OUT[0], ref, rtol=1e-4, atol=1e-4)
