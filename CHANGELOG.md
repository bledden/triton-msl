# Changelog

## Unreleased

### Integrity prescan (silent-wrong → loud refusal)

Added a structural integrity contract: when the compiler recognizes a kernel
but cannot lower it correctly, it raises `MetalNonRecoverableError` (surfaced
as a clear error to the user) instead of emitting wrong numbers. The catalog
lives in `GenericLowerer._refuse_unsafe_unsupported_ops`; the underlying
audit method was *classifying* the skip-listed feature-gap tests as
loud-failure-safe vs silent-wrong, and migrating the latter to refusals.

Cases now refused (each was a silent-wrong producer before the guard):

- `tt.dot_scaled` — microscaling matmul; no Apple hardware (`test_scaled_dot`).
- pid-tiled matmul with constexpr-baked M/N (`test_dot_mulbroadcasted`).
- rank ≥ 3 `tt.trans` with a non-identity permutation (`test_trans_4d`).
- rank ≥ 2 `tt.cat` / `tt.join` (`test_cat_nd`).
- `tt.dot` inside a noinline device function (`test_noinline[shared]`).
- `tt.join` result feeding `tt.dot` (`test_join_with_mma`).
- unstructured kernel-level control flow / `cf.cond_br` (`test_nested_if_else_return`).

### False-pass exposed by the `cf.cond_br` refusal

`test_constexpr_if_return` shares the void-early-return shape of
`test_nested_if_else_return`; before the guard the legacy parser had been
emitting `Out = pid + 0` for it — dropping the `atomic_add`, dropping the
early return, writing out of bounds — but the test only asserts `out >= 0`,
which `pid + 0` happens to satisfy. The garbage went undetected. Now
correctly refused; skip-listed with that rationale. *A passing test is not
the same as a correct kernel.*

### GPU-hang root cause

`test_dot_max_num_imprecise_acc` was investigated for a per-config hang and
*ruled out* as a per-kernel defect: each config either runs correctly or
raises `OutOfResources` cleanly. The apparent hang was reproduced as **GPU
driver-state accumulation** from many back-to-back fp8 dispatches in a
single process — the same environmental class as concurrent test sweeps.
Documented; not a code fix.

### Test suite (as of 2026-05-30, fresh cache, Python 3.14)

- `test_core.py` (upstream Triton): **4,326 passed / 5,016 skipped / 0 failed**.
- Project suite (codegen / GPU correctness / integration / FlashAttention /
  MLX, excluding torch.compile suites blocked on Py 3.14): **507 passed /
  0 failed**.
- FlashAttention: 11/11 at HEAD_DIM=32 (via the C++ MLIR→LLVM path).
- MLX backend: 15/15.
- `torch.compile` suites (32 + 9 tests) are environment-blocked on Python
  3.14 — PyTorch's own platform guard refuses `torch.compile` on 3.14;
  honest skips with `skipif(py>=3.14)`, will auto-lift when PyTorch ships
  3.14 Dynamo support.

### Documentation & roadmap

- Added `REFERENCES.md` and `CITING.md` (citations for Triton [1],
  FlashAttention v1/v2 [4,5], online softmax [6], MLX [7], Asahi/`applegpu`
  [10,11], MSL spec [8], PyTorch Inductor [12], M4 Max hardware [13]).
- Added `docs/superpowers/specs/2026-05-30-triton-metal-roadmap.md`
  (umbrella roadmap: WS0 foundation, WS1 the register-array spine,
  WS2 orthogonal-refusal cleanup, WS3 experimental sub-AIR AGX).
- Added `docs/superpowers/specs/2026-05-30-ws0-foundation-design.md`
  (the first workstream's full design — documentation truth, citations,
  test hygiene, C++ build hardening, integrity single source of truth,
  hardware profiling + disassembly harness).
- `docs/ARCHITECTURE.md` reconciled: corrected stale upstream-test numbers,
  added integrity-model section, refusal catalog, MEPT experimental
  charter, scoped FlashAttention claim to HEAD_DIM=32.

## 0.1.0-alpha (2026-03-10)

First public alpha release of triton-metal.

### Milestone 1: First Kernel on Metal
- `@triton.jit` vector add running on Apple GPU via Metal Shading Language

### Milestone 2: Kernel Coverage
- 28 `@triton.jit` tests: sum, max, min, softmax, matmul, SiLU, sigmoid, GELU, SwiGLU, RMS norm, layer norm, fused add+ReLU, leaky ReLU, clamp, FMA, FP16, negation, exp+log

### Milestone 3: Real Compiler
- Replaced pattern-matching parser with proper MLIR walker + op-by-op generic lowerer
- All kernels route through new pipeline (`mlir_walker.py` + `generic_lowerer.py`)
- Legacy parser (`ttgir_parser.py` + `msl_emitter.py`) kept as safety fallback only

### Milestone 4: Upstream Compatibility
- 4,279 / 9,334 upstream `test_core.py` tests passing (0 failures)
- Completed: atomics, while loops, `tt.dot` (strided matmul + all epilogues), 2D/3D reduce, argmax/argmin, `tt.histogram`, `tt.gather`, `tl.cat`, `tl.join`/`tl.split`, reshape, permute, transpose, `scf.for`/`scf.if`, NaN propagation, floor div, shift ops
- Triton tutorials 01 (vector add), 02 (softmax), 03 (matmul), 05 (layer norm) all passing
- `@triton.autotune` working end-to-end

### Milestone 5: torch.compile
- 32/32 torch.compile tests passing
- Models: Identity, ReLU, GELU, SiLU, Sigmoid, Tanh, ELU, LeakyReLU, Dropout, Linear, LayerNorm, BatchNorm2d, GroupNorm, InstanceNorm, Embedding, Conv2d, AvgPool, MaxPool, Softmax, LogSoftmax, MLP, LargeMLP, ResBlock, DepthwiseSeparable, ConvNet, TransformerBlock, MHA, SmallGPT, GPT, MiniViT, LSTM, EmbeddingBag
- `torch.compile(model, backend="metal")` integration via Triton inductor

### Milestone 6: MLX Backend
- 15/15 MLX backend tests passing
- Zero-copy dispatch via `mx.fast.metal_kernel()`
- API: `triton_metal.mlx.triton_call(kernel_fn, *args, grid=(...), **constexpr_kwargs)`

### Performance (M4 Max)
- Vector add (16M): 137.5 GB/s
- Softmax (8192x1024): 109.4 GB/s (1.26x vs CPU)
- Matmul (512x512): 826 GFLOP/s
- Layer norm (4096x1024): 77.5 GB/s
- MLX dispatch: ~0.12ms (zero-copy, comparable to native MLX)

### Known Limitations
- ~0.15ms buffer copy overhead per kernel launch (MPS tensors)
- No FP64, FP8, or TF32 (Metal hardware limitation)
- No backward pass / training support
- 32x32 matmul tile size (larger tiles would improve throughput)
