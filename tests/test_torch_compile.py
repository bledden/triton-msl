"""Tests for torch.compile integration via triton-msl backend.

Validates that torch.compile(model, backend='inductor') produces correct
results on MPS device by routing through TritonScheduling -> triton-msl -> MSL -> Metal GPU.
"""

import pytest
import torch
import torch.nn as nn

# torch.compile / TorchDynamo rewrites CPython bytecode via interpreter-internal
# frame-evaluation hooks, which are Python-version-specific. Rather than hardcode
# a Python-version gate (PyTorch added 3.14 Dynamo support in 2.10/2.12, so a
# `>= (3,14)` skip went stale the moment torch was upgraded), probe the actual
# capability: skip only if TorchDynamo reports it can't run in this interpreter.
# The gate then auto-lifts the instant the running torch supports the version.
pytestmark = pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile/TorchDynamo unsupported in this interpreter "
    "(torch._dynamo.is_dynamo_supported() is False).",
)


@pytest.fixture(autouse=True)
def setup_backend():
    """Register metal triton backend and reset dynamo before each test.

    Registration also pins inductor to single-process compilation (Metal/PyObjC
    is not fork-safe); the backend owns that requirement, so tests don't set it.
    """
    import triton_msl.inductor
    triton_msl.inductor.register_metal_triton_backend()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


DEVICE = "mps"
ATOL = 1e-3


def _check(model, x, atol=ATOL):
    """Compile model and compare against eager execution."""
    model = model.eval().to(DEVICE)
    if isinstance(x, torch.Tensor):
        x = x.to(DEVICE)
    with torch.no_grad():
        expected = model(x)
        compiled = torch.compile(model, backend="inductor")
        result = compiled(x)
    if isinstance(expected, tuple):
        expected, result = expected[0], result[0]
    diff = (result.cpu().float() - expected.cpu().float()).abs().max().item()
    assert diff < atol, f"max_diff={diff:.6f} exceeds atol={atol}"


# --- Elementwise ---

class TestElementwise:
    def test_identity(self):
        _check(nn.Identity(), torch.randn(1024))

    def test_relu(self):
        _check(nn.ReLU(), torch.randn(1024))

    def test_gelu(self):
        _check(nn.GELU(), torch.randn(1024))

    def test_silu(self):
        _check(nn.SiLU(), torch.randn(1024))

    def test_sigmoid(self):
        _check(nn.Sigmoid(), torch.randn(1024))

    def test_tanh(self):
        _check(nn.Tanh(), torch.randn(1024))

    def test_elu(self):
        _check(nn.ELU(), torch.randn(1024))

    def test_leaky_relu(self):
        _check(nn.LeakyReLU(0.2), torch.randn(1024))

    def test_dropout_in_eval(self):
        _check(nn.Dropout(0.5), torch.randn(1024))


# --- Layer types ---

class TestLayers:
    def test_linear(self):
        _check(nn.Linear(128, 64), torch.randn(32, 128))

    def test_layer_norm(self):
        _check(nn.LayerNorm(128), torch.randn(32, 128))

    def test_batch_norm_2d(self):
        _check(nn.BatchNorm2d(16), torch.randn(2, 16, 8, 8))

    def test_group_norm(self):
        _check(nn.GroupNorm(4, 16), torch.randn(2, 16, 8, 8))

    def test_instance_norm(self):
        _check(nn.InstanceNorm2d(16, affine=True), torch.randn(2, 16, 8, 8))

    def test_embedding(self):
        _check(nn.Embedding(1000, 128), torch.randint(0, 1000, (2, 16)).to(DEVICE))

    def test_conv2d(self):
        _check(nn.Conv2d(3, 16, 3, padding=1), torch.randn(1, 3, 32, 32))

    def test_avg_pool(self):
        _check(nn.AvgPool2d(2), torch.randn(1, 16, 8, 8))

    def test_max_pool(self):
        _check(nn.MaxPool2d(2), torch.randn(1, 16, 8, 8))

    def test_softmax(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)
        _check(M(), torch.randn(32, 128))

    def test_log_softmax(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.log_softmax(x, dim=-1)
        _check(M(), torch.randn(32, 128))


# --- Composite models ---

class TestModels:
    def test_mlp(self):
        _check(
            nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)),
            torch.randn(32, 128),
        )

    def test_large_mlp(self):
        _check(
            nn.Sequential(
                nn.Linear(256, 512), nn.GELU(),
                nn.Linear(512, 512), nn.GELU(),
                nn.Linear(512, 256),
            ),
            torch.randn(16, 256),
        )

    def test_res_block(self):
        class ResBlock(nn.Module):
            def __init__(self, ch=16):
                super().__init__()
                self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(ch)
                self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(ch)
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return torch.relu(out + x)
        _check(ResBlock(), torch.randn(1, 16, 8, 8))

    def test_depthwise_separable(self):
        class DWSep(nn.Module):
            def __init__(self):
                super().__init__()
                self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
                self.pw = nn.Conv2d(32, 64, 1)
                self.bn = nn.BatchNorm2d(64)
            def forward(self, x):
                return torch.relu(self.bn(self.pw(self.dw(x))))
        _check(DWSep(), torch.randn(2, 32, 8, 8))

    def test_convnet(self):
        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(64, 10)
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        _check(ConvNet(), torch.randn(4, 3, 16, 16))

    def test_transformer_block(self):
        class TBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.TransformerEncoderLayer(
                    128, 4, dim_feedforward=256, batch_first=True, dropout=0.0
                )
            def forward(self, x):
                return self.layer(x)
        _check(TBlock(), torch.randn(2, 16, 128))

    def test_multihead_attention(self):
        class MHA(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.0)
            def forward(self, x):
                return self.attn(x, x, x)[0]
        _check(MHA(), torch.randn(2, 16, 128))

    def test_small_gpt(self):
        class SmallGPT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
                enc = nn.TransformerEncoderLayer(
                    128, 4, dim_feedforward=256, batch_first=True, dropout=0.0
                )
                self.transformer = nn.TransformerEncoder(enc, 2)
                self.fc = nn.Linear(128, 1000)
            def forward(self, x):
                return self.fc(self.transformer(self.embedding(x)))
        _check(SmallGPT(), torch.randint(0, 1000, (2, 16)).to(DEVICE), atol=1e-2)

    def test_gpt_4layer(self):
        class GPT(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(5000, 256)
                self.pos = nn.Embedding(128, 256)
                enc = nn.TransformerEncoderLayer(
                    256, 8, dim_feedforward=512, batch_first=True, dropout=0.0
                )
                self.transformer = nn.TransformerEncoder(enc, 4)
                self.head = nn.Linear(256, 5000)
            def forward(self, x):
                pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                x = self.emb(x) + self.pos(pos)
                return self.head(self.transformer(x))
        _check(GPT(), torch.randint(0, 5000, (2, 32)).to(DEVICE), atol=1e-2)

    def test_mini_vit(self):
        class MiniViT(nn.Module):
            def __init__(self, patch=4, d=128):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, d, patch, stride=patch)
                self.cls_token = nn.Parameter(torch.randn(1, 1, d))
                enc = nn.TransformerEncoderLayer(
                    d, 4, dim_feedforward=256, batch_first=True, dropout=0.0
                )
                self.encoder = nn.TransformerEncoder(enc, 2)
                self.head = nn.Linear(d, 10)
            def forward(self, x):
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.encoder(x)
                return self.head(x[:, 0])
        _check(MiniViT(), torch.randn(2, 3, 16, 16))

    def test_lstm(self):
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
                self.fc = nn.Linear(128, 10)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1])
        _check(LSTMModel(), torch.randn(4, 16, 64))

    def test_embedding_bag_mean(self):
        class EmbBag(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10000, 256)
                self.fc = nn.Linear(256, 64)
            def forward(self, x):
                return self.fc(self.emb(x).mean(dim=1))
        _check(EmbBag(), torch.randint(0, 10000, (8, 32)).to(DEVICE))


def test_persistent_reduction_filter_refuses_underfilling():
    """Regression: the Metal persistent-reduction config filter must NEVER return
    an under-filling config — surplus lanes wrap `lid % rnumel` unmasked and
    over-count the reduction denominator (the documented transformer-gradient
    corruption). When no config fills the threadgroup it must refuse loudly, not
    fall back to a rejected config (the old `configs[:1]` re-admitted exactly the
    config the filter exists to remove)."""
    from triton_msl.inductor import _filter_metal_persistent_configs
    from triton_msl.errors import MetalNonRecoverableError

    class FakeConfig:
        def __init__(self, xblock, num_warps):
            self.kwargs = {"XBLOCK": xblock}
            self.num_warps = num_warps

    rnumel = 16
    filled = FakeConfig(xblock=2, num_warps=1)     # 32 elems >= 32 threads -> keep
    underfill = FakeConfig(xblock=1, num_warps=2)  # 16 elems < 64 threads  -> drop

    # Mixed input: keep only the thread-filling config, drop the under-filling one.
    assert _filter_metal_persistent_configs([underfill, filled], rnumel) == [filled]

    # Every config under-fills -> refuse loudly (never re-admit a rejected config).
    with pytest.raises(MetalNonRecoverableError):
        _filter_metal_persistent_configs(
            [underfill, FakeConfig(xblock=1, num_warps=4)], rnumel)

    # Every config exceeds 1024 threads -> refuse loudly too.
    with pytest.raises(MetalNonRecoverableError):
        _filter_metal_persistent_configs([FakeConfig(xblock=128, num_warps=1)], rnumel)
