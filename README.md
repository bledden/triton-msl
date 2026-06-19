# triton-msl

Metal (Apple Silicon) backend for [OpenAI Triton](https://github.com/triton-lang/triton) [\[1\]](REFERENCES.md)[\[2\]](REFERENCES.md). Write `@triton.jit` kernels and run them on your Mac's GPU.

```
@triton.jit → Triton TTIR → TTGIR → MSL → metallib → Apple GPU
```

## Status

**Alpha** — actively developed, not yet production-ready.

- **0 failures** across the upstream Triton `test_core.py` suite — 5,560 kernels
  attempted and correct, ~3,634 documented feature-gap skips (each is either a
  *refused* kernel — fails loudly, never silent-wrong — or a hardware-impossible
  case like FP64). Aligned with Triton [\[2\]](REFERENCES.md) release `3.7.0`.
  Measured by `scripts/run_upstream_tests.py` — the single source of truth for this
  count — which runs `--device cpu` (torch references compute on CPU while the Metal
  backend compiles and runs the kernels on the GPU, since upstream `test_core`
  otherwise assumes CUDA). Re-run it to reproduce; counts in this file and
  `CHANGELOG.md` are regenerated from it, not hand-maintained.
- **787 passed / 0 failed** in the project suite (codegen, GPU correctness,
  integration, FlashAttention, MLX backend, fast-matmul / compile_shader
  zero-copy, `torch.compile`, and training). FlashAttention: causal + non-causal
  at **HEAD_DIM 32 / 64 / 128** (head_dim 128 fp32 + fp16 via the head-dim-tiled
  template; see [\[4\]](REFERENCES.md) for the algorithm); **15 / 15** MLX backend
  tests; the project suite grew from 434 → 603 → 716 → ~800 since `0.1.0-alpha`.
  (A further ~20 C++-MLIR-backend tests skip unless that optional extension is
  built.)
- **`torch.compile` routes through triton-msl** on Python 3.10–3.14 (PyTorch
  Inductor [\[12\]](REFERENCES.md)) — inference and training (AOTAutograd
  backward), static and `dynamic=True`; **32 / 32** `torch.compile` model tests
  plus the training suite pass.
- Triton tutorials 01–03, 05 passing.
- Built against Triton's `TRITON_EXT_ENABLED=1` plugin architecture
  (upstream PR [#9783](https://github.com/triton-lang/triton/pull/9783)).
- **Integrity contract**: kernels we can lower run correctly; kernels we
  cannot are *refused* (`MetalNonRecoverableError`) — never silent-wrong.
  See [`docs/SUPPORTED_OPS.md`](docs/SUPPORTED_OPS.md) for the supported
  ops/dtypes matrix + the loud-refusal catalog, and
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) "Lowering paths and the
  integrity model" for the lowering paths.

See [`REFERENCES.md`](REFERENCES.md) for citations and
[`docs/superpowers/specs/2026-05-30-triton-msl-roadmap.md`](docs/superpowers/specs/2026-05-30-triton-msl-roadmap.md)
for the active pre-1.0 roadmap.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14 (Sonoma) or later
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.10+

## Install

```bash
pip install triton-msl

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
import triton_msl.inductor
triton_msl.inductor.register_metal_triton_backend()

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
from triton_msl.mlx import triton_call

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

### MPS tensors — zero-copy

The same `@triton.jit` kernel runs **zero-copy** on `torch` MPS tensors: the driver
dispatches the emitted Metal through `torch.mps.compile_shader`, skipping the host
round-trip (~10× faster on memory-bound kernels; on by default, no code change):

```python
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty(n, device="mps")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)  # runs on the GPU, no copy
```

### Matmul (`tl.dot`)

Aligned fp16/fp32 matmuls (M%32, N%32, K%8) on MPS tensors take a direct
simdgroup-matrix path (~11–12 TFLOP/s on M4 Max), dispatched zero-copy:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b))
        a += BK * sak; b += BK * sbk
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc.to(tl.float16))

M = N = K = 2048
A = torch.randn(M, K, device="mps", dtype=torch.float16)
B = torch.randn(K, N, device="mps", dtype=torch.float16)
C = torch.empty(M, N, device="mps", dtype=torch.float16)
matmul_kernel[(M // 64, N // 64)](
    A, B, C, M, N, K,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    BM=64, BN=64, BK=32)
```

### Integrity contract — refused, never silently wrong

A kernel triton-msl **cannot lower correctly raises `MetalNonRecoverableError`**
rather than returning garbage. For example, a pid-tiled matmul that bakes its M/N
dims as `constexpr` (so the true output strides can't be recovered) is refused:

```python
from triton_msl.errors import MetalNonRecoverableError

@triton.jit
def matmul_baked_dims(a_ptr, b_ptr, c_ptr, K,
                      M: tl.constexpr, N: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a = a_ptr + (offm[:, None] * K + offk[None, :])
    b = b_ptr + (offk[:, None] * BN + offn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b)); a += BK; b += BK * BN
    tl.store(c_ptr + (offm[:, None] * BN + offn[None, :]), acc)

try:
    matmul_baked_dims[(2, 2)](A, B, C, K, M=64, N=64, BM=32, BN=32, BK=32)
except MetalNonRecoverableError as e:
    print("refused (not silent-wrong):", e)
```

See [`docs/SUPPORTED_OPS.md`](docs/SUPPORTED_OPS.md) for the full op/dtype support
matrix and the loud-refusal catalog.

### FlashAttention

A full FlashAttention v2 forward (causal + non-causal) runs through the standard
`@triton.jit` path at **`BLOCK_M = BLOCK_N = 32`** for **head_dim 32, 64, and 128** —
see [`tests/test_flash_attention.py`](tests/test_flash_attention.py) for the kernel and
launch. head_dim 32/64 use the generic lowering; **head_dim 128** is routed to a
head-dim-tiled FA2 MSL template (**fp32 + fp16**) that chunks the head dimension to fit
Metal's 32 KB threadgroup budget. Out-of-range configs are **refused loudly**
(`MetalNonRecoverableError`, never silent-wrong): head_dim > 128, block tiles ≠ 32, bf16
inputs, and any FA-shaped kernel whose strides/scale can't be resolved unambiguously.
Larger blocks and head_dim > 128 are on the roadmap.

### Tuning flags

All default-on; set to `0` to disable (an escape hatch for bisecting a regression):

| Flag | Effect when disabled |
|------|----------------------|
| `TRITON_MSL_COMPILE_SHADER=0` | Use the host-copy driver instead of the zero-copy `compile_shader` dispatch |
| `TRITON_MSL_FAST_MATMUL=0` | Use the generic matmul instead of the fast simdgroup-matrix path |
| `TRITON_MSL_MEPT=0` | Disable the multi-element-per-thread register-array model |
| `TRITON_MSL_LEGACY=1` | Opt **in** to the heuristic legacy text parser (off by default — it can be silent-wrong) |

## What Works

| Category | Operations |
|----------|-----------|
| **Elementwise** | add, sub, mul, div, exp, log, sqrt, abs, neg, SiLU, GELU, sigmoid, tanh, ReLU, leaky ReLU, clamp, FMA |
| **Reductions** | sum, max, min, argmax, argmin, xor_sum |
| **Dot product** | `tl.dot` with strided matmul template, all epilogues (add, softmax, chain-dot, transpose) |
| **Attention** | FlashAttention [\[4\]](REFERENCES.md) (causal + non-causal) at **`BLOCK_M=BLOCK_N=32`, HEAD_DIM 32 / 64 / 128** via the Python MSL path (head_dim 128 routed to a head-dim-tiled FA2 template, fp32 + fp16). Out-of-range configs (head_dim > 128; block tiles ≠ 32; bf16) are refused (`MetalNonRecoverableError`, never silent-wrong); larger blocks/head_dim are on the roadmap. |
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

Current numbers via the zero-copy `compile_shader` path (default-on); see
`reports/perf_baseline.json`. Hardware peak: 546 GB/s memory, 18.4 / 36.9 TFLOP/s
fp32 / fp16.

| Kernel | Size | Throughput | % of peak | vs host-copy path |
|--------|------|-----------|-----------|-------------------|
| Vector add | 16M | 347 GB/s | 64% | **13×** |
| Elementwise | 16M | 315 GB/s | 58% | 13.4× |
| Softmax | 8192×1024 | 232 GB/s | 42% | **17.8×** |
| Reduction | 16M | 235 GB/s | 43% | 8.2× |
| Matmul (fp32) | 2048³ | 11.4 TFLOP/s | 62% of fp32 peak | ~4× generic |
| Matmul (fp16) | 2048³ | 12.4 TFLOP/s | ≈ fp32 rate\* | ~4× generic |

\* fp16 matmul runs at roughly the fp32 matrix-unit rate (float accumulation for
precision); Apple's simdgroup-matrix unit isn't faster for half accumulation, so the
36.9 TFLOP/s fp16 figure is an unreachable vector-ALU peak. The ~58–64% memory-bound
and ~60% fp32-matmul numbers are **near the practical ceilings** for these kernel
classes on this hardware (the raw 546 / 18.4 / 36.9 spec peaks are not reachable by
compute) — see the Phase-5 readiness audit (`docs/audits/`).

**MPS tensors run zero-copy** via `torch.mps.compile_shader` (default-on) — the prior
host-round-trip copy bottleneck is gone. CPU tensors and the MLX backend
[\[7\]](REFERENCES.md) (`mx.fast.metal_kernel`) also dispatch zero-copy.

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

If you use `triton-msl` in research or technical work, see
[`CITING.md`](CITING.md) for a suggested BibTeX entry. For citations of
the papers and projects this backend builds on (Triton, FlashAttention,
online softmax, MLX, Asahi/`applegpu`, the MSL specification, PyTorch
Inductor), see [`REFERENCES.md`](REFERENCES.md).

## License

[MIT](LICENSE)
