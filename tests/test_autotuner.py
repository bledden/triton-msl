"""Tests for the Metal kernel autotuning infrastructure."""

import os
import struct
import tempfile

import pytest

# Skip entire module if Metal framework not available
try:
    import Metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

requires_metal = pytest.mark.skipif(
    not HAS_METAL,
    reason="Metal framework not available"
)

from triton_msl.autotuning.autotuner import (
    AutotuneConfig,
    AutotuneResult,
    MetalAutotuner,
    ELEMENTWISE_CONFIGS,
    REDUCTION_CONFIGS,
)


# ---------------------------------------------------------------------------
# AutotuneConfig tests (no GPU needed)
# ---------------------------------------------------------------------------

def test_config_defaults():
    """Config has sensible defaults."""
    cfg = AutotuneConfig()
    assert cfg.block_size == 256
    assert cfg.num_warps == 8  # 256 / 32


def test_config_auto_warps():
    """num_warps is auto-computed from block_size."""
    cfg = AutotuneConfig(block_size=128)
    assert cfg.num_warps == 4
    cfg = AutotuneConfig(block_size=512)
    assert cfg.num_warps == 16


def test_config_to_kwargs():
    """to_kwargs produces correct dict."""
    cfg = AutotuneConfig(block_size=512)
    kw = cfg.to_kwargs()
    assert kw == {"block_size": 512}


def test_config_to_kwargs_with_tiles():
    """to_kwargs includes tile dimensions when set."""
    cfg = AutotuneConfig(block_size=128, tile_m=32, tile_n=32, tile_k=16)
    kw = cfg.to_kwargs()
    assert kw["block_m"] == 32
    assert kw["block_n"] == 32
    assert kw["block_k"] == 16
    assert kw["block_size"] == 128


def test_config_to_kwargs_with_extra():
    """to_kwargs passes through extra params."""
    cfg = AutotuneConfig(block_size=256, extra={"head_dim": 64})
    kw = cfg.to_kwargs()
    assert kw["head_dim"] == 64


def test_config_signature_unique():
    """Different configs have different signatures."""
    c1 = AutotuneConfig(block_size=128)
    c2 = AutotuneConfig(block_size=256)
    assert c1.signature() != c2.signature()


def test_config_signature_deterministic():
    """Same config produces same signature."""
    c1 = AutotuneConfig(block_size=256, tile_m=32)
    c2 = AutotuneConfig(block_size=256, tile_m=32)
    assert c1.signature() == c2.signature()


def test_presets_exist():
    """Preset config lists are non-empty."""
    assert len(ELEMENTWISE_CONFIGS) >= 3
    assert len(REDUCTION_CONFIGS) >= 2
    assert all(isinstance(c, AutotuneConfig) for c in ELEMENTWISE_CONFIGS)


# ---------------------------------------------------------------------------
# GPU autotuning tests
# ---------------------------------------------------------------------------

def make_test_buffers(device, n):
    """Create simple input/output buffers for vector_add autotuning."""
    a_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    b_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    out_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    n_buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )

    # Fill
    for buf_obj, pattern in [(a_buf, 1.0), (b_buf, 2.0)]:
        view = buf_obj.contents().as_buffer(n * 4)
        for i in range(n):
            struct.pack_into("f", view, i * 4, pattern)
    n_view = n_buf.contents().as_buffer(4)
    struct.pack_into("I", n_view, 0, n)

    return [a_buf, b_buf, out_buf, n_buf]


@requires_metal
def test_autotune_vector_add():
    """Autotuner finds best block_size for vector_add."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    device = Metal.MTLCreateSystemDefaultDevice()
    n = 65536
    buffers = make_test_buffers(device, n)

    configs = [
        AutotuneConfig(block_size=128),
        AutotuneConfig(block_size=256),
        AutotuneConfig(block_size=512),
    ]

    # Use temp cache dir to avoid stale results
    cache_dir = tempfile.mkdtemp(prefix="autotune_test_")
    tuner = MetalAutotuner(configs, cache_dir=cache_dir, warmup=5, rep=20)
    result = tuner.tune(make_vector_add_kernel, "vector_add",
                         buffers=buffers, n_elements=n)

    assert isinstance(result, AutotuneResult)
    assert result.best_config.block_size in [128, 256, 512]
    assert result.best_time_us > 0
    assert len(result.all_results) == 3
    assert all(r["status"] == "ok" for r in result.all_results)


@requires_metal
def test_autotune_cache_hit():
    """Second tune() call returns cached result."""
    from triton_msl.codegen.msl_emitter import make_vector_add_kernel

    device = Metal.MTLCreateSystemDefaultDevice()
    n = 4096
    buffers = make_test_buffers(device, n)

    configs = [
        AutotuneConfig(block_size=128),
        AutotuneConfig(block_size=256),
    ]

    cache_dir = tempfile.mkdtemp(prefix="autotune_cache_test_")
    tuner = MetalAutotuner(configs, cache_dir=cache_dir, warmup=3, rep=10)

    # First call: benchmarks
    r1 = tuner.tune(make_vector_add_kernel, "vector_add",
                      buffers=buffers, n_elements=n)

    # Second call: should hit cache (no GPU work)
    r2 = tuner.tune(make_vector_add_kernel, "vector_add",
                      buffers=buffers, n_elements=n)

    assert r1.best_config.block_size == r2.best_config.block_size
    assert r1.best_time_us == r2.best_time_us


@requires_metal
def test_autotune_silu():
    """Autotuner works with silu kernel (different signature)."""
    from triton_msl.codegen.msl_emitter import make_silu_kernel

    device = Metal.MTLCreateSystemDefaultDevice()
    n = 65536

    in_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    out_buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    n_buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    n_view = n_buf.contents().as_buffer(4)
    struct.pack_into("I", n_view, 0, n)

    configs = [
        AutotuneConfig(block_size=256),
        AutotuneConfig(block_size=512),
    ]

    cache_dir = tempfile.mkdtemp(prefix="autotune_silu_test_")
    tuner = MetalAutotuner(configs, cache_dir=cache_dir, warmup=3, rep=10)
    result = tuner.tune(make_silu_kernel, "silu_kernel",
                         buffers=[in_buf, out_buf, n_buf], n_elements=n)

    assert result.best_config.block_size in [256, 512]
    assert result.best_time_us > 0


@requires_metal
def test_autotune_handles_compile_error():
    """Autotuner handles kernel compilation failure gracefully."""
    def bad_kernel(block_size=256):
        if block_size == 512:
            return "THIS IS NOT VALID MSL"
        from triton_msl.codegen.msl_emitter import make_vector_add_kernel
        return make_vector_add_kernel(block_size=block_size)

    device = Metal.MTLCreateSystemDefaultDevice()
    n = 4096
    buffers = make_test_buffers(device, n)

    configs = [
        AutotuneConfig(block_size=256),
        AutotuneConfig(block_size=512),  # will fail to compile
    ]

    cache_dir = tempfile.mkdtemp(prefix="autotune_error_test_")
    tuner = MetalAutotuner(configs, cache_dir=cache_dir, warmup=3, rep=10)
    result = tuner.tune(bad_kernel, "vector_add",
                         buffers=buffers, n_elements=n)

    # Should still return a result (the working config)
    assert result.best_config.block_size == 256
    # The failed config should show error status
    failed = [r for r in result.all_results if r["config"].block_size == 512]
    assert len(failed) == 1
    assert "error" in failed[0]["status"]


# ---------------------------------------------------------------------------
# @triton.autotune + @triton.jit integration tests
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import torch
    HAS_TORCH = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
except ImportError:
    HAS_TORCH = False

requires_triton_msl = pytest.mark.skipif(
    not (HAS_METAL and HAS_TRITON and HAS_TORCH),
    reason="Requires Metal + Triton + PyTorch MPS"
)


@requires_triton_msl
def test_triton_autotune_vector_add():
    """End-to-end: @triton.autotune selects a config for vector_add on Metal."""
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    n = 4096
    x = torch.randn(n, device="cpu", dtype=torch.float32)
    y = torch.randn(n, device="cpu", dtype=torch.float32)
    out = torch.empty(n, device="cpu", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n)

    expected = x + y
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max()}"
    print("@triton.autotune vector_add: PASS")
