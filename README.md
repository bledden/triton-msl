# triton-metal

Metal (Apple Silicon) backend for [OpenAI Triton](https://github.com/triton-lang/triton) [\[1\]](REFERENCES.md)[\[2\]](REFERENCES.md). Write `@triton.jit` kernels and run them on your Mac's GPU.

```
@triton.jit → Triton TTIR → TTGIR → MSL → metallib → Apple GPU
```

## Status

**Alpha** — actively developed, not yet production-ready.

- **0 failures** across the upstream Triton `test_core.py` suite — 4,326 kernels
  attempted and correct, ~5,016 documented feature-gap skips (each is either a
  *refused* kernel — fails loudly, never silent-wrong — or a hardware-impossible
  case like FP64). Aligned with Triton [\[2\]](REFERENCES.md) release `3.7.0`.
- **507 / 507** project tests (codegen, GPU correctness, integration,
  FlashAttention, MLX backend). FlashAttention path: **11 / 11** at
  HEAD_DIM=32 (see [\[4\]](REFERENCES.md) for the algorithm); **15 / 15** MLX
  backend tests; project test-suite size grew from 434 to 507 since
  `0.1.0-alpha`.
- **32 / 32** `torch.compile` model tests pass on Python ≤ 3.13 (PyTorch
  Inductor [\[12\]](REFERENCES.md)). On Python 3.14 the suite is honestly
  skipped because PyTorch's own platform guard refuses `torch.compile` on
  3.14 — auto-lifts when PyTorch ships 3.14 Dynamo support.
- Triton tutorials 01–03, 05 passing.
- Built against Triton's `TRITON_EXT_ENABLED=1` plugin architecture
  (upstream PR [#9783](https://github.com/triton-lang/triton/pull/9783)).
- **Integrity contract**: kernels we can lower run correctly; kernels we
  cannot are *refused* (`MetalNonRecoverableError`) — never silent-wrong.
  See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) "Lowering paths and
  the integrity model" for the catalog.

See [`REFERENCES.md`](REFERENCES.md) for citations and
[`docs/superpowers/specs/2026-05-30-triton-metal-roadmap.md`](docs/superpowers/specs/2026-05-30-triton-metal-roadmap.md)
for the active pre-1.0 roadmap.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14 (Sonoma) or later
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.10+

## Install

```bash
pip install triton-metal

# Triton is required but installed separately (macOS wheels may not be available)
pip install triton>=3.6.0

# If no Triton wheel exists for your platform, build from source:
# pip install git+https://github.com/triton-lang/triton.git
```

## Quick Start

### @triton.jit

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device="cpu")
y = torch.randn(n, device="cpu")
out = torch.empty(n, device="cpu")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)
print(f"Max error: {(out - (x + y)).abs().max():.2e}")
```

### torch.compile

```python
import torch
import triton_metal.inductor
triton_metal.inductor.register_metal_triton_backend()

model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
)

compiled = torch.compile(model, backend="metal")
x = torch.randn(32, 256)
out = compiled(x)
```

### MLX

```python
import mlx.core as mx
import triton
import triton.language as tl
from triton_metal.mlx import triton_call

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = mx.random.normal((n,))
y = mx.random.normal((n,))
out = mx.zeros((n,))
results = triton_call(add_kernel, x, y, out, n, grid=(4,), BLOCK=256)
```

## What Works

| Category | Operations |
|----------|-----------|
| **Elementwise** | add, sub, mul, div, exp, log, sqrt, abs, neg, SiLU, GELU, sigmoid, tanh, ReLU, leaky ReLU, clamp, FMA |
| **Reductions** | sum, max, min, argmax, argmin, xor_sum |
| **Dot product** | `tl.dot` with strided matmul template, all epilogues (add, softmax, chain-dot, transpose) |
| **Attention** | FlashAttention [\[4\]](REFERENCES.md) (causal + non-causal) at HEAD_DIM=32 via the C++ MLIR→LLVM path; dedicated `qk = q @ trans(k)` lowering in `DotOpToLLVM.cpp`. HEAD_DIM=64 is part of the WS1 register-array spine. |
| **Normalization** | Layer norm, RMS norm, batch norm |
| **Type casts** | FP32, FP16, BF16, INT8, INT16, INT32, bool |
| **Control flow** | `scf.for`, `scf.if`, while loops |
| **Atomics** | atomic_add, atomic_max, atomic_min, atomic_and, atomic_or, atomic_xor, CAS |
| **Tensor ops** | cat, join, split, interleave, reshape, permute, transpose, histogram, gather |
| **torch.compile** | 32 models including MLP, ResBlock, TransformerBlock, SmallGPT, MiniViT, LSTM |
| **MLX** | Zero-copy dispatch via `mx.fast.metal_kernel()` |

## What Doesn't Work

| Feature | Reason |
|---------|--------|
| FP64 | Metal has no FP64 support |
| FP8, TF32 | Not available on Apple GPUs |
| Backward pass / training | Not implemented |
| Multi-GPU | Apple Silicon is single-GPU |
| `tl.dot` with sizePerThread > 1 | Requires 2D cooperative execution model (addressed by the register-array spine — WS1) |
| Unstructured control flow (`cf.cond_br`) | Refused with `MetalNonRecoverableError` (never silent-wrong); a `cf`-dialect lowerer is WS2 |
| `tt.dot_scaled` (microscaling matmul) | No Apple microscaling hardware; refused |

## Performance (M4 Max [\[13\]](REFERENCES.md))

Benchmarks from Triton tutorials, measured 2026-04 (see
`reports/perf_baseline.json`):

| Kernel | Size | Throughput | vs CPU | % of peak |
|--------|------|-----------|--------|-----------|
| Vector add | 16M elements | 137.5 GB/s | 0.93× | ~25% of 546 GB/s |
| Softmax | 8192×1024 | 109.4 GB/s | **1.26×** | ~20% of 546 GB/s |
| Matmul | 512×512 | 826 GFLOP/s | 0.32× | (compute-bound; well below peak) |
| Layer norm | 4096×1024 | 77.5 GB/s | 0.34× | ~14% of 546 GB/s |

**These numbers reflect the *current* state, not the *target*.** The
register-array spine (WS1 of the active roadmap) is explicitly designed to
move them toward the hardware roofline. "Optimal bounds given by the
hardware" is defined empirically by the WS0/C6 profiling+disassembly
harness: *saturate the limiting hardware counter, verified in the
disassembly*. See the roadmap for the methodology.

**Known bottleneck**: ~0.15 ms buffer copy overhead per kernel launch when
using MPS tensors (MPS→CPU→Metal→CPU→MPS). Use CPU tensors for best
performance, or the MLX backend [\[7\]](REFERENCES.md) for zero-copy
dispatch via `mx.fast.metal_kernel`.

## Architecture

```
@triton.jit kernel
    → Triton frontend (Python AST → TTIR)
    → Triton optimizer (TTIR → TTGIR)
    → mlir_walker.py: walk TTGIR module → IRGraph
    → generic_lowerer.py: IRGraph → MSL source
    → xcrun metal: MSL → AIR → metallib
    → driver.py: load metallib, dispatch on GPU
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If you use `triton-metal` in research or technical work, see
[`CITING.md`](CITING.md) for a suggested BibTeX entry. For citations of
the papers and projects this backend builds on (Triton, FlashAttention,
online softmax, MLX, Asahi/`applegpu`, the MSL specification, PyTorch
Inductor), see [`REFERENCES.md`](REFERENCES.md).

## License

[MIT](LICENSE)
