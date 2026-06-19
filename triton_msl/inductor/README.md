# triton-msl Inductor Integration

Routes `torch.compile()` through Triton on Apple Silicon:

```
torch.compile(fn) → Dynamo → Inductor → TritonScheduling → Triton code → triton-msl → MSL → Metal GPU
```

Without this, Inductor uses `MetalScheduling` — a limited early prototype with broken reductions and no Triton involvement.

## Usage

```python
import torch
import triton_msl.inductor

triton_msl.inductor.register_metal_triton_backend()

model = torch.compile(my_model)
output = model(input)  # compiles via Triton → Metal
```

Call `register_metal_triton_backend()` **before** the first `torch.compile()` invocation.

Set `TORCHINDUCTOR_COMPILE_THREADS=1` when using Metal — PyObjC is not fork-safe, so Inductor's multi-process compilation crashes.

## How It Works

### Backend Registration

`register_metal_triton_backend()` replaces MPS's default `MetalScheduling` with `TritonScheduling`:

```python
register_backend_for_device("mps", TritonScheduling, PythonWrapperCodegen, None, WrapperFxCodegen)
```

PyTorch's `init_backend_registration()` checks `if get_scheduling_for_device("mps") is None` before registering its default, so pre-registering takes priority. This is the same pattern used by Intel XPU and MTIA.

### Patches Applied

1. **Device op overrides** (`MetalTritonDeviceOpOverrides`): Metal has no CUDA streams — stream-related methods return no-ops.

2. **MPS device interface**: Adds missing `exchange_device`, `maybe_exchange_device`, `set_device`, `get_raw_stream` to `MpsInterface`. MPS is single-device, so these are all no-ops returning 0 or None.

3. **libdevice replacement** (`metal_libdevice.py`): CUDA's `libdevice` maps math functions to `__nv_*` extern calls. Metal has no such library. We provide `@triton.jit` implementations using `tl.math.*` (generates standard MLIR math ops our backend handles). Covers 40+ functions including `rsqrt`, `tanh`, `erf`, `erfinv`, trig functions, etc.

4. **Persistent reduction config filter** (`_patch_persistent_reduction_configs`): Inductor generates configs where `XBLOCK * R0_BLOCK` threads are needed. Metal supports at most 1024 threads per threadgroup. We filter configs that exceed this limit.

5. **Reduction config patching** (`_patch_reduction_configs`): Our MSL lowering uses flat `lid`-based indexing (1 element per thread). When `R0_BLOCK > num_warps * 32`, Triton's blocked layout uses `sizePerThread > 1`, which our lowering doesn't support. Fix: force `XBLOCK=1` (prevents cross-row mixing in Welford reductions) and cap `R0_BLOCK` to thread count.

## Architecture

```
triton_msl/inductor/
├── __init__.py          # Registration hook + Inductor patches
├── metal_libdevice.py   # @triton.jit libdevice replacements for Metal
└── README.md            # This file
```

## Validated Models

All tested with `torch.allclose(compiled(x), eager(x), atol=1e-3)`:

| Category | Models |
|----------|--------|
| Elementwise | Identity, ReLU, GELU, SiLU, Sigmoid, Tanh, ELU, LeakyReLU, Dropout |
| Layers | Linear, LayerNorm, BatchNorm2d, GroupNorm, InstanceNorm, Embedding, Conv2d, AvgPool, MaxPool, Softmax, LogSoftmax |
| Composite | MLP, LargeMLP, ResBlock, DepthwiseSeparable, ConvNet, TransformerBlock, MultiheadAttention, SmallGPT (2L), GPT (4L), MiniViT, LSTM, EmbeddingBag |

32 tests total in `tests/test_torch_compile.py`.

## Known Limitations

1. **Buffer copy overhead**: ~0.15ms per kernel launch (mps → cpu → Metal buffer → kernel → Metal buffer → cpu → mps). This is the CPU intermediate path; direct MPS integration would eliminate it.

2. **XBLOCK=1 for reductions**: We force XBLOCK=1 on all non-persistent reduction configs. This means each threadgroup handles one row — correct but potentially less efficient than multi-row configs on CUDA.

3. **Single-threaded compilation**: Must set `TORCHINDUCTOR_COMPILE_THREADS=1` due to PyObjC fork safety.

4. **No FP64/FP8**: Metal hardware limitation — these dtypes are permanently unsupported.

## Prior Art

- **Intel XPU**: `register_backend_for_device("xpu", TritonScheduling, ...)` — simplest reference
- **MTIA**: Same pattern
- **Intel Extension for PyTorch**: Out-of-tree backend registration reference
