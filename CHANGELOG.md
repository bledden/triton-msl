# Changelog

## Unreleased

### FlashAttention — large head_dim + integrity hardening (2026-06-17)

- **head_dim 128 @ `BLOCK_M=BLOCK_N=32`** now supported (fp32 + fp16, causal + non-causal).
  A real `@triton.jit` FlashAttention-2 kernel is routed to a new head-dim-tiled FA2 MSL
  template (`make_flash_attention_kernel_tiled`) that chunks the head dimension to fit
  Metal's 32 KB threadgroup budget (the un-tiled lowering hit `OutOfResources` at 128).
  Routing is a prescan detector with **refuse-on-any-ambiguity** — an FA-shaped kernel whose
  pointers/strides/scale can't be resolved unambiguously is refused, never guessed; the
  detector is robust to Triton's `equal_to_1` arg specialization. Stays FA2 (FA3/FA4 are
  Hopper/Blackwell async-hardware co-designs with no Apple analog).
- **Closed a small-block silent-wrong hole:** FlashAttention at `BLOCK_M`/`BLOCK_N` < 32
  silently mis-computed (rows past the first → garbage) for *any* head_dim — including the
  otherwise-supported 32/64 — which the previous head_dim>64-only guard missed. The prescan
  now refuses min-dot-tile-dim < 32. head_dim > 128, other block sizes, and bf16 matmul
  inputs are refused loudly.
- **Skip-list reclaim:** +28 upstream `test_core` passes recovered from over-broad skips
  (a Gluon-tool base-name collision wrongly skipping core `test_cat`/`test_split`, plus
  `test_tl_range_num_stages`, a uint16/fp16 modulus, `test_pointer_arguments[cpu_pinned]`) —
  each verified a real pass, not a loose-assertion false-pass.
- **Tooling:** `scripts/run_upstream_tests.py` now loads `-p conftest_metal`, so the cited
  source-of-truth command reproduces the documented conformance number.

### Phase 4 — zero-copy execution + fast matmul + Phase-5 readiness audit (2026-06-16)

- **Zero-copy MPS execution** via `torch.mps.compile_shader`: routes emitted MSL through
  PyTorch's compiler so kernels run against MPS tensors without the per-launch host
  round-trip. ~10× on memory-bound kernels (vector_add 28 → ~347 GB/s ≈ 64% of the M4 Max
  546 GB/s roof). Flag `TRITON_METAL_COMPILE_SHADER` (default-on, `=0` escape hatch).
- **Fast simdgroup matmul** (`make_simdgroup_matmul_kernel_fast`) dispatched zero-copy for
  aligned MPS matmuls. Measured at 2048³: fp32 ~9.6–11.5 TFLOP/s (~55–62% of the 18.4
  fp32 peak — competitive with MLX/MPS GEMM), fp16 ~7.8–12, fp16-output ~12.3 — vs the
  ~2.8 generic fallback. Float accumulation (precision); fp16 output via a cast epilogue.
  Flag `TRITON_METAL_FAST_MATMUL`; correctness-gated (test_core dot/matmul on==off identical).
  This is **not** MLX-parity (fp16 runs at ~fp32 rate to keep float accumulation); the
  earlier "~13.8 TFLOP/s MLX parity" docstring claim was an overstatement and is corrected.
- **MEPT** multi-element-per-thread register-array model is the default lowering path.
- **Test suite (Triton 3.7.0):** upstream `test_core.py` **5,559 passed / 0 failed /
  ~3,783 feature-gap skips** (each a loud refusal or HW-impossible); the single source of
  truth is `scripts/run_upstream_tests.py` (`--device cpu`, which loads the `conftest_metal`
  skip plugin), not hand-maintained counts.
  Project suite **754 passed / 0 failed**. FlashAttention causal + non-causal at HEAD_DIM
  32 / 64 / 128 via the **Python/MSL** lowering — the C++ MLIR→LLVM path named in the
  2026-05-30 snapshot below was shelved (AGX compiler blocker; Python/MSL is primary).
- **Phase-5 readiness audit** (dual NVIDIA/Triton + MLX/Apple lens) recorded in
  `docs/audits/2026-06-16-phase5-readiness-audit.md`; remaining pre-1.0 items tracked there.

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

### Hardware profiling harness (WS0/C6)

- Added `benchmarks/hw_harness.py` + `triton_metal/profiling/roofline.py` +
  `triton_metal/profiling/disasm.py`: per-kernel GPU-timestamp timing →
  roofline classification (% of the M4 Max 546 GB/s memory roof / estimated
  compute roof, memory- vs compute-bound), pipeline-reflection occupancy,
  best-effort native-AGX disassembly, and an MLX comparison ratio. Emits
  per-kernel JSON + summary.md + baseline.json. This is the empirical
  backbone for the WS1 perf work ("optimal bounds = saturate the limiting
  counter"). First run surfaced a concrete target: reduce_sum at ~16% of the
  bandwidth roof / 1.3x slower than MLX, vs vector_add (72% of roof, 0.89x
  MLX) and silu (48%, 0.59x MLX).
- Vendored `applegpu` (dougallj — REFERENCES.md [11]) under
  `third_party/applegpu/` for native-AGX disassembly. Honest scope: live GPU
  counters (ALU%/occupancy/registers) are NOT programmatically available on
  Apple Silicon (the device vends only the `timestamp` counter set), and
  applegpu is M1-era so M4/AGX2 disassembly is partial (the harness reports a
  decode-coverage %). `docs/INSTRUMENTS.md` documents the Xcode-capture /
  Instruments path for the counters the programmatic API can't provide. No
  Swift counter-helper was built — it would hit the same Metal API wall.
- Tests: `tests/test_roofline.py` (9), `tests/test_disasm.py` (6, incl.
  fat-header parsing for the 0xCBFEBABE GPU-archive magic + graceful
  degradation).

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
