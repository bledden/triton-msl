"""Real model validation tests for triton-metal.

Tests HuggingFace models end-to-end via torch.compile with the Metal backend.
Validates correctness against eager execution and measures performance.

Requires: pip install transformers
"""

import os
import sys
import time
import pytest
import torch
import torch.nn as nn

# These tests drive torch.compile, which PyTorch refuses on Python 3.14+ (its
# own platform guard — TorchDynamo's CPython frame-eval hooks aren't ported to
# 3.14 yet). Not a triton-metal bug or an API backfill; an upstream-PyTorch
# capability gap. Gate so they're honest skips, not red failures; auto-lifts
# when PyTorch ships 3.14 Dynamo support. Run on a Python <=3.13 lane (see
# docs/superpowers/specs/2026-05-30-ws0-foundation-design.md, component C3).
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="torch.compile is not supported on Python 3.14+ (PyTorch's own "
    "platform guard; resolves when PyTorch ships 3.14 Dynamo support). "
    "See REFERENCES.md [12].",
)

# Metal/PyObjC is not fork-safe; force single-thread compilation
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

try:
    from transformers import GPT2LMHeadModel, GPT2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


DEVICE = "mps"


@pytest.fixture(autouse=True)
def setup_backend():
    """Register metal triton backend and reset dynamo before each test."""
    import triton_metal.inductor
    triton_metal.inductor.register_metal_triton_backend()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _warmup_and_bench(fn, n_warmup=3, n_iter=10):
    """Warmup and benchmark a function."""
    for _ in range(n_warmup):
        fn()
    if DEVICE == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn()
        if DEVICE == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)


# ---------------------------------------------------------------------------
# GPT-2 Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestGPT2:

    def test_gpt2_tiny_forward(self):
        """GPT-2 tiny (2 layer, 128 dim) forward pass."""
        torch.manual_seed(42)
        config = GPT2Config(
            n_layer=2, n_head=4, n_embd=128,
            vocab_size=1000, n_positions=64,
            attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
        )
        model = GPT2LMHeadModel(config)
        model.eval()
        model = model.to(DEVICE)
        input_ids = torch.randint(0, 1000, (1, 32), device=DEVICE)

        with torch.no_grad():
            expected = model(input_ids).logits
            compiled = torch.compile(model, backend="inductor")
            result = compiled(input_ids).logits

        # HuggingFace GPT-2 accumulates numerical differences across attention layers.
        # Metal compute precision can differ from MPS eager. Use cosine similarity.
        expected_flat = expected.cpu().float().flatten()
        result_flat = result.cpu().float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            expected_flat.unsqueeze(0), result_flat.unsqueeze(0)
        ).item()
        assert cos_sim > 0.95, f"GPT-2 tiny: cosine_sim={cos_sim:.6f} (want > 0.95)"

    def test_gpt2_small_forward(self):
        """GPT-2 small (6 layer, 384 dim) forward pass."""
        config = GPT2Config(
            n_layer=6, n_head=6, n_embd=384,
            vocab_size=5000, n_positions=128,
            attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
        )
        model = GPT2LMHeadModel(config)
        model.eval()
        model = model.to(DEVICE)
        input_ids = torch.randint(0, 5000, (1, 64), device=DEVICE)

        with torch.no_grad():
            expected = model(input_ids).logits
            compiled = torch.compile(model, backend="inductor")
            result = compiled(input_ids).logits

        expected_flat = expected.cpu().float().flatten()
        result_flat = result.cpu().float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            expected_flat.unsqueeze(0), result_flat.unsqueeze(0)
        ).item()
        assert cos_sim > 0.98, f"GPT-2 small: cosine_sim={cos_sim:.6f} (want > 0.98)"

    def test_gpt2_tiny_generation(self):
        """GPT-2 tiny token-by-token generation — verify compiled model produces valid output."""
        torch.manual_seed(42)
        config = GPT2Config(
            n_layer=2, n_head=4, n_embd=128,
            vocab_size=1000, n_positions=64,
            attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
        )
        model = GPT2LMHeadModel(config)
        model.eval()
        model = model.to(DEVICE)
        input_ids = torch.randint(0, 1000, (1, 8), device=DEVICE)

        with torch.no_grad():
            # Compiled single-step forward — verify logits are valid (not all zeros/NaN)
            compiled = torch.compile(model, backend="inductor")
            logits = compiled(input_ids).logits

        # Check logits are not degenerate
        assert not torch.isnan(logits).any(), "Generated NaN logits"
        assert not torch.isinf(logits).any(), "Generated Inf logits"
        assert logits.abs().max() > 0.01, "Logits are all near-zero"
        # Check logits have reasonable variance (not collapsed)
        assert logits.std() > 0.01, f"Logits std too low: {logits.std().item():.6f}"

    @pytest.mark.slow
    def test_gpt2_small_benchmark(self):
        """Benchmark GPT-2 small forward pass: eager vs compiled."""
        config = GPT2Config(
            n_layer=6, n_head=6, n_embd=384,
            vocab_size=5000, n_positions=128,
            attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
        )
        model = GPT2LMHeadModel(config)
        model.eval()
        model = model.to(DEVICE)
        input_ids = torch.randint(0, 5000, (1, 64), device=DEVICE)

        with torch.no_grad():
            # Eager benchmark
            eager_min, eager_avg = _warmup_and_bench(
                lambda: model(input_ids), n_warmup=3, n_iter=10
            )

            # Compiled benchmark
            compiled = torch.compile(model, backend="inductor")
            compiled(input_ids)  # first compile
            comp_min, comp_avg = _warmup_and_bench(
                lambda: compiled(input_ids), n_warmup=3, n_iter=10
            )

        print(f"\nGPT-2 small forward (1x64 tokens):")
        print(f"  Eager:    {eager_min*1000:.1f}ms min, {eager_avg*1000:.1f}ms avg")
        print(f"  Compiled: {comp_min*1000:.1f}ms min, {comp_avg*1000:.1f}ms avg")
        print(f"  Speedup:  {eager_min/comp_min:.2f}x min, {eager_avg/comp_avg:.2f}x avg")


# ---------------------------------------------------------------------------
# Vision Model Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed")
class TestVisionModels:

    def test_resnet18_forward(self):
        """ResNet-18 forward pass."""
        model = resnet18(weights=None)
        model.eval()
        model = model.to(DEVICE)
        x = torch.randn(1, 3, 32, 32, device=DEVICE)

        with torch.no_grad():
            expected = model(x)
            compiled = torch.compile(model, backend="inductor")
            result = compiled(x)

        expected_flat = expected.cpu().float().flatten()
        result_flat = result.cpu().float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            expected_flat.unsqueeze(0), result_flat.unsqueeze(0)
        ).item()
        assert cos_sim > 0.99, f"ResNet-18: cosine_sim={cos_sim:.6f} (want > 0.99)"

    def test_resnet50_forward(self):
        """ResNet-50 forward pass."""
        model = resnet50(weights=None)
        model.eval()
        model = model.to(DEVICE)
        x = torch.randn(1, 3, 32, 32, device=DEVICE)

        with torch.no_grad():
            expected = model(x)
            compiled = torch.compile(model, backend="inductor")
            result = compiled(x)

        expected_flat = expected.cpu().float().flatten()
        result_flat = result.cpu().float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            expected_flat.unsqueeze(0), result_flat.unsqueeze(0)
        ).item()
        assert cos_sim > 0.99, f"ResNet-50: cosine_sim={cos_sim:.6f} (want > 0.99)"

    @pytest.mark.slow
    def test_resnet18_benchmark(self):
        """Benchmark ResNet-18: eager vs compiled."""
        model = resnet18(weights=None)
        model.eval()
        model = model.to(DEVICE)
        x = torch.randn(4, 3, 64, 64, device=DEVICE)

        with torch.no_grad():
            eager_min, eager_avg = _warmup_and_bench(
                lambda: model(x), n_warmup=3, n_iter=10
            )

            compiled = torch.compile(model, backend="inductor")
            compiled(x)  # first compile
            comp_min, comp_avg = _warmup_and_bench(
                lambda: compiled(x), n_warmup=3, n_iter=10
            )

        print(f"\nResNet-18 forward (4x3x64x64):")
        print(f"  Eager:    {eager_min*1000:.1f}ms min, {eager_avg*1000:.1f}ms avg")
        print(f"  Compiled: {comp_min*1000:.1f}ms min, {comp_avg*1000:.1f}ms avg")
        print(f"  Speedup:  {eager_min/comp_min:.2f}x min, {eager_avg/comp_avg:.2f}x avg")


# ---------------------------------------------------------------------------
# Custom Model Tests
# ---------------------------------------------------------------------------


class TestCustomModels:

    def test_mlp_large(self):
        """Large MLP (4 layers, 512 dim)."""
        class LargeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(512, 1024), nn.GELU(),
                    nn.Linear(1024, 1024), nn.GELU(),
                    nn.Linear(1024, 512), nn.GELU(),
                    nn.Linear(512, 10),
                )
            def forward(self, x):
                return self.layers(x)

        model = LargeMLP()
        model.eval()
        model = model.to(DEVICE)
        x = torch.randn(32, 512, device=DEVICE)

        with torch.no_grad():
            expected = model(x)
            compiled = torch.compile(model, backend="inductor")
            result = compiled(x)

        diff = (result.cpu().float() - expected.cpu().float()).abs().max().item()
        assert diff < 1e-3, f"LargeMLP: max_diff={diff:.6f}"

    def test_transformer_encoder_large(self):
        """Transformer encoder with larger config."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.0,
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        model.eval()
        model = model.to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)

        with torch.no_grad():
            expected = model(x)
            compiled = torch.compile(model, backend="inductor")
            result = compiled(x)

        diff = (result.cpu().float() - expected.cpu().float()).abs().max().item()
        assert diff < 0.01, f"TransformerEncoder: max_diff={diff:.6f}"
