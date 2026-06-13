"""fp16/bf16 atomic RMW via word-CAS (Phase 3 feature 1). These refused before
(Metal has no 16-bit device atomic); a neighbor-preserving 32-bit word-CAS makes
them correct. Serial GPU."""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _atomic_add_kernel(X, OUT, N, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.atomic_add(OUT + (offs % 4), x, mask=mask)   # many threads -> 4 slots

    @triton.jit
    def _atomic_neighbor_kernel(X, OUT, N, BLOCK: tl.constexpr):
        # Each thread adds to its OWN even/odd adjacent fp16 slot — checks the
        # word-CAS preserves the neighbor half.
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.atomic_add(OUT + offs, x, mask=mask)


@requires_metal
def test_fp16_atomic_add_accumulates():
    N, BLOCK = 256, 256
    X = torch.randn(N, dtype=torch.float16)
    OUT = torch.zeros(4, dtype=torch.float16)
    _atomic_add_kernel[(1,)](X, OUT, N, BLOCK=BLOCK)
    want = torch.zeros(4, dtype=torch.float32)
    for i in range(N):
        want[i % 4] += X[i].float()
    assert torch.allclose(OUT.float(), want, atol=1e-1, rtol=1e-2), (
        f"got {OUT.float()} want {want}")


@requires_metal
def test_fp16_atomic_neighbor_preservation():
    # Adjacent fp16 slots written by different threads must not corrupt each
    # other (the word-CAS splice keeps the other half).
    N = 256
    X = torch.arange(1, N + 1, dtype=torch.float16) * 0.01
    OUT = torch.zeros(N, dtype=torch.float16)
    _atomic_neighbor_kernel[(1,)](X, OUT, N, BLOCK=256)
    assert torch.allclose(OUT.float(), X.float(), atol=1e-2), (
        f"neighbor corruption: got {OUT.float()[:8]} want {X.float()[:8]}")


@requires_metal
@pytest.mark.parametrize("op", ["max", "min"])
def test_fp16_atomic_max_min_refused_by_frontend(op):
    # NOTE (2026-06-13): upstream Triton's frontend hard-restricts fp16/bf16
    # atomics to `add` ONLY (triton/python/triton/language/semantic.py:1259-1262:
    # "atomic_<op> does not support fp16"). max/min IR for a 16-bit float is
    # therefore UNREACHABLE through the public tl.atomic_max / tl.atomic_min API
    # — the ValueError is raised at trace time, before any backend sees the IR.
    # The upstream test_atomic_rmw corpus reflects this: it parametrizes only
    # ('add', 'float16') / ('add', 'bfloat16'); float max/min appear only for
    # float32/float64. Our backend word-CAS helper (_emit_atomic_rmw_16bit) DOES
    # support max/min in the float domain, so the lowering is forward-compatible
    # if Triton ever lifts the restriction; but it cannot be exercised end-to-end
    # today, so we pin the real boundary here instead of testing an impossible
    # path. (See the design's Out-of-scope / op-coverage notes.)
    import triton.language as tl

    @triton.jit
    def _k(X, OUT, N, OP: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        if OP == 0:
            tl.atomic_max(OUT + (offs % 2), x, mask=mask)
        else:
            tl.atomic_min(OUT + (offs % 2), x, mask=mask)

    N = 256
    X = torch.randn(N, dtype=torch.float16)
    OUT = torch.zeros(2, dtype=torch.float16)
    with pytest.raises(Exception, match="does not support fp16"):
        _k[(1,)](X, OUT, N, OP=(0 if op == "max" else 1), BLOCK=256)


@requires_metal
def test_bf16_atomic_add_accumulates():
    N, BLOCK = 256, 256
    X = torch.randn(N, dtype=torch.bfloat16)
    OUT = torch.zeros(4, dtype=torch.bfloat16)
    _atomic_add_kernel[(1,)](X, OUT, N, BLOCK=BLOCK)
    want = torch.zeros(4, dtype=torch.float32)
    for i in range(N):
        want[i % 4] += X[i].float()
    # bf16 has an 8-bit mantissa — looser tolerance.
    assert torch.allclose(OUT.float(), want, atol=1.0, rtol=5e-2), (
        f"got {OUT.float()} want {want}")
