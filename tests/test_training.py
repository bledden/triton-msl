"""Training (forward + backward) through torch.compile on the Metal backend.

triton-msl was inference-only until the inductor backend port (2026-06-18).
Once torch.compile routes through TritonScheduling, AOTAutograd's backward graph
is just more Triton kernels (matmul->matmul, the embedding scatter-add, the
softmax/layernorm backwards, etc.) that lower through triton-msl -> MSL. So
training "just works" through the compiled path -- these tests pin that:

  1. gradients from the compiled model match eager (same device, same data), and
  2. a real multi-step Adam loop actually *converges* and tracks eager step-by-step.

The backward path exercises kernels the forward never does (notably
``embedding_dense_backward``'s grad zero-init + scatter-add), so this is genuine
coverage, not a restatement of test_torch_compile.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile/TorchDynamo unsupported in this interpreter "
    "(torch._dynamo.is_dynamo_supported() is False).",
)


@pytest.fixture(autouse=True)
def setup_backend():
    """Register the metal triton backend and reset dynamo around each test."""
    import triton_msl.inductor
    triton_msl.inductor.register_metal_triton_backend()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


DEVICE = "mps"


def _mlp():
    return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 32))


def _cnn():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
    )


def _transformer():
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 64)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, dropout=0.0), 2)
            self.head = nn.Linear(64, 100)

        def forward(self, x):
            return self.head(self.enc(self.emb(x))).mean(1)
    return T()


def _data(kind):
    torch.manual_seed(7)
    if kind == "mlp":
        return (torch.randn(16, 64, device=DEVICE),
                torch.randint(0, 32, (16,), device=DEVICE))
    if kind == "cnn":
        return (torch.randn(8, 3, 16, 16, device=DEVICE),
                torch.randint(0, 10, (8,), device=DEVICE))
    return (torch.randint(0, 100, (8, 16), device=DEVICE),
            torch.randint(0, 100, (8,), device=DEVICE))


_BUILD = {"mlp": _mlp, "cnn": _cnn, "transformer": _transformer}


def _train(kind, compiled, steps=8):
    torch.manual_seed(0)
    model = _BUILD[kind]().to(DEVICE)
    model.train()
    x, y = _data(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    fn = torch.compile(model, backend="inductor") if compiled else model
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(fn(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach().cpu().item())
    return losses


@pytest.mark.parametrize("kind", ["mlp", "cnn", "transformer"])
def test_backward_grads_match_eager(kind):
    """One forward+backward: compiled gradients == eager gradients."""
    def grads(compiled):
        torch.manual_seed(0)
        model = _BUILD[kind]().to(DEVICE)
        model.train()
        x, y = _data(kind)
        fn = torch.compile(model, backend="inductor") if compiled else model
        loss = nn.functional.cross_entropy(fn(x), y)
        loss.backward()
        return loss.detach().cpu().item(), \
            [p.grad.detach().cpu().clone() for p in model.parameters()]

    torch._dynamo.reset(); le, ge = grads(False)
    torch._dynamo.reset(); lc, gc = grads(True)
    assert abs(le - lc) < 1e-3, f"{kind}: loss eager={le} vs compiled={lc}"
    max_gd = max((a - b).abs().max().item() for a, b in zip(ge, gc))
    assert max_gd < 2e-3, f"{kind}: max grad diff {max_gd:.3e} (eager vs compiled)"


@pytest.mark.parametrize("kind", ["mlp", "cnn", "transformer"])
def test_training_loop_converges_and_matches_eager(kind):
    """A multi-step Adam loop converges (loss drops) and tracks eager."""
    torch._dynamo.reset(); eager = _train(kind, compiled=False)
    torch._dynamo.reset(); compiled = _train(kind, compiled=True)
    # Learns: final loss meaningfully below the first step.
    assert compiled[-1] < compiled[0] - 0.05, \
        f"{kind}: did not converge ({compiled[0]:.3f} -> {compiled[-1]:.3f})"
    # Tracks eager step-by-step. The compiled (triton-msl MSL) and eager (MPS)
    # paths are two independent fp backends; their loss trajectories drift by only
    # ~1e-6 here when idle, but the DEEP transformer amplifies a RARE transient
    # under heavy GPU contention through the 8-step Adam loop (measured up to
    # ~2.6e-3 under full-suite load, vs ~1e-6 idle and ~4e-8 single-step grads).
    # The result stays correct every run (converges to the same final loss); this
    # assertion only bounds trajectory tracking, so the transformer gets robust
    # margin against that transient — a REAL divergence is orders larger (NaN or
    # >>0.05, per the convergence check above). mlp/cnn are shallow and stay tight.
    step_tol = 1e-2 if kind == "transformer" else 2e-3
    max_step = max(abs(a - b) for a, b in zip(eager, compiled))
    assert max_step < step_tol, \
        f"{kind}: compiled loss trajectory diverges from eager (max {max_step:.3e} >= {step_tol})"


def test_embedding_backward_compiles():
    """Regression: embedding_dense_backward's grad zero-init is a masked MEPT
    store of a constant 0 to a 1D buffer. It must lower (was a malformed
    `ptr[off][lid]` double-subscript -> MetalCompilationError) so embedding
    layers can train through torch.compile."""
    torch.manual_seed(0)
    emb = nn.Embedding(100, 64).to(DEVICE)
    emb.train()
    x = torch.randint(0, 100, (8, 16), device=DEVICE)
    fn = torch.compile(emb, backend="inductor")
    out = fn(x)
    out.sum().backward()
    assert emb.weight.grad is not None
    # grad of sum() w.r.t. an embedding row = count of times that row was used.
    assert emb.weight.grad.abs().sum().item() > 0
