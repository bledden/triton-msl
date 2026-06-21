# triton-msl Implementation Roadmap

> Last updated: 2026-06-18 (torch.compile / inductor backend port landed). The dated phase tables further
> below were authored **2026-04-09** and are kept as a detailed reference — but much of
> Phase 0–1 + 2A + 6A has since landed. **Read "Current status" first for what's actually
> done vs. remaining.**

This document tracks remaining work for triton-msl, organized into phased delivery with
dependencies, scope estimates, and file-level change lists.

---

## Current status (2026-06-17) — supersedes the 2026-04-09 status line

**0.1.0-alpha; first public release 2026-06-16.** Upstream `test_core.py`: **5,560 passed /
0 failed / 3,782 skipped** (captured 2026-06-17) via `scripts/run_upstream_tests.py` (`--device cpu`, which loads
the `conftest_metal` skip plugin). The skips are hardware-impossible (fp64, fp8/microscaling,
64-bit atomics, TMA, device printf) or unimplemented features — **each refused loudly, never
silent-wrong**. Project suite **799 / 0** (incl. 38 `torch.compile` + real-model tests + 7
training tests, all un-gated); FlashAttention causal + non-causal at head_dim 32 / 64 / 128.
**`torch.compile` routes through triton-msl on Metal** — inference *and* training (forward +
backward), static + dynamic shapes; see below.

### Landed since this roadmap was written (2026-04-09)
- **Phase 0** (foundation) — all items.
- **Phase 1A–1D** — 2D tensor semantics, `ttg.local_alloc` shared memory, generic `tt.dot` via
  simdgroup MMA, K-loop. Matmul + attention now lower generically.
- **MEPT register-array spine** (not in the old plan) — multi-element-per-thread is the DEFAULT
  lowering path (`TRITON_MSL_MEPT=0` escape hatch), retiring the 1-elem-per-thread model.
- **2A** buffer-copy elimination — done via **zero-copy `torch.mps.compile_shader`** (~10× on
  memory-bound; vector_add 28 → 347 GB/s ≈ 64% of the M4 Max 546 GB/s roof).
- **Fast simdgroup matmul** — zero-copy, fp32/fp16/fp16-out, ~55–62% of fp32 peak.
- **6A FlashAttention** — done, incl. **large head_dim** (128 @ BLOCK=32, fp32 + fp16, causal +
  non-causal) via a head-dim-tiled template + refuse-on-ambiguity routing.
- **Integrity contract** — `_refuse_unsafe_unsupported_ops` prescan (~24 guard sites), the
  published refusal catalog (`SUPPORTED_OPS.md`), systematic refusal-coverage tests.
- **Phase-5 honesty audit** (dual NVIDIA/MLX lens) + first public push.
- **2E** (shared-mem aliasing) — partially realized ad hoc (the FA head-dim tiling); a general
  pass remains open.
- **torch.compile / inductor backend port (2026-06-18)** — `torch.compile(model,
  backend="inductor")` on `"mps"` routes through our `TritonScheduling` → triton-msl → MSL.
  **32/32 torch.compile + 6/6 real-model tests (cold & warm); dynamic shapes (`dynamic=True`)
  flow through with a single compiled graph (was roadmap 1H).** torch 2.10+ now ships a *native*
  MPS inductor; the port restores routing through OUR kernels and **closes 4 latent silent-wrong
  bugs** the old path had (device-op-override clobber; Metal fork-unsafe compile subprocesses +
  cache corruption; `_MSL_BY_NAME` cross-graph cache-key collision; the `triton_per_*` softmax
  template mapping `xnumel`=row-count as the row length → 4×-wrong reductions). Closes the
  "torch.compile produces wrong values for unsupported patterns" honesty concern (old 1A/0h).
- **Training / backward pass (2026-06-18)** — falls out of the inductor port: AOTAutograd's
  backward graph is just more Triton kernels (matmul→matmul, the embedding scatter-add,
  softmax/layernorm/attention backwards) that lower through triton-msl. `torch.compile`d
  **MLP, CNN, and transformer (w/ embedding) train + converge, matching eager** (grads + an
  8-step Adam loop, `tests/test_training.py`). Fixed one backward-only codegen gap:
  `embedding_dense_backward`'s grad zero-init (a masked MEPT store of a constant to a 1D buffer)
  emitted a malformed `ptr[off][lid]` — the MEPT scatter now broadcasts splat/constant values.
  The old "custom `autograd.Function` wrappers" framing (Phase 5) is obsolete — AOTAutograd
  handles autograd; we just lower its kernels.

### Remaining — re-prioritized for the current state
1. ~~**Training / backward pass** (old Phase 5)~~ — **DONE** for the torch.compile path (see
   "Landed 2026-06-18"). Remaining sub-items: optimizer-step fusion / `torch.compile`-d
   optimizers, gradient-checkpointing, and larger end-to-end training runs (real datasets).
2. ~~**Dynamic shapes** (1H)~~ — **DONE** via the inductor port (`torch.compile(dynamic=True)`
   verified; single graph across variable seq lengths). Hand-written `@triton.jit` kernels already
   took runtime dims. Remaining sub-item: stress dynamic shapes on larger real models.
3. **Distribution & upstream** — PyPI wheel (6E), official Triton backend submission (6C),
   M1–M4 CI matrix, broader real-model testing beyond `test_core`.
4. **Incremental op coverage** (all refuse loudly today, so safe to defer): noinline-dot (1E),
   `tl.range` loop fusion (1F), 2D gather, multi-program atomics / cooperative sync (1G),
   rank-≥2 cat/join, `map_elementwise` pack/multi-output, the i64 loop-induction-var hang,
   unstructured control flow (`cf.cond_br`).
5. **Performance** — near practical Apple ceilings per the Phase-5 audit (matmul ~60%,
   memory-bound ~64%; **fp16-2× and vectorized-loads explored + declined**). Open: FA perf pass
   (serialized acc-rescale), larger FA blocks / head_dim>128, autotuning beyond 1D blocks (2C),
   shared-mem aliasing (2E), kernel fusion (2B).
6. **Forward-looking / HW-gated**: M5 tensor-op matmul (3A), Metal 4 command model (3B), native
   fp8/microscaling (3C) — await Apple HW/SDK.
7. **Hardware-impossible (never; correctly refused)**: fp64, native fp8 matrix, 64-bit atomics,
   TMA, device printf.

### Plan assumptions that have since changed
- The **C++ MLIR pass / port** (old 6D + the port-first roadmap) was **SHELVED** on an AGX
  compiler blocker — **Python/MSL is the primary path** (language decision 2026-06-11).
- Performance headroom is smaller than the old plan assumed: the Phase-5 audit measured matmul
  ~60% and memory-bound ~64% as near the practical Apple ceiling.

---

### Strategic Principles (from TorchTPU, TurboQuant, kforge research)

1. **Framework compatibility > kernel performance.** Getting more models working at 0.5x native speed is more valuable than peak GFLOP/s. Prioritize by "models unlocked" not "upstream tests passed."
2. **Never silently produce wrong results.** Graceful CPU fallback for unsupported patterns (Phase 0h). TorchTPU falls back for every unlowered op.
3. **MMA intrinsics are our competitive advantage.** `mx.fast.metal_kernel()` cannot access `simdgroup_matrix_multiply_accumulate` or MPP tensor ops. triton-msl CAN. This makes Phase 1C/3A the primary differentiator.
4. **Profile before optimizing kernels.** TurboQuant's biggest win (14x) was architectural, not codegen.
5. **Correctness-first validation.** Tolerance-based testing (cosine similarity > 0.99, or abs/rel error < 0.01 on 100 random inputs).

---

## Table of Contents

1. [Phase 0: In-Progress Items (Landing Now)](#phase-0-in-progress-items-landing-now)
2. [Phase 1: Core Architecture (2D Semantics + Generic Matmul)](#phase-1-core-architecture)
3. [Phase 2: Performance Foundations](#phase-2-performance-foundations)
4. [Phase 3: Metal 4 / M5 Enablement](#phase-3-metal-4--m5-enablement)
5. [Phase 4: Type System Expansion](#phase-4-type-system-expansion)
6. [Phase 5: Training Support](#phase-5-training-support)
7. [Phase 6: Ecosystem Integration](#phase-6-ecosystem-integration)
8. [Dependency Graph](#dependency-graph)
9. [Risk Register](#risk-register)

---

## Phase 0: In-Progress Items (Landing Now)

Items actively being implemented. These should land before any Phase 1 work begins.

| # | Item | What / Why | Scope | Files | Depends On |
|---|------|-----------|-------|-------|------------|
| 0a | **2D tensor shape tracking** | Foundation for 2D semantics. `_prescan_2d_info()` maps `tt.make_range` ops to dimension indices, tracks `_effective_2d_shape`, and propagates through `expand_dims`/`broadcast` chains. Required before any generic 2D codegen. | ~200 LOC (incremental to existing ~70 LOC in `_prescan_2d_info`) | `triton_msl/codegen/generic_lowerer.py` | -- |
| 0b | **Metal 4 device detection** | `device_detect.py` module that probes chip family (M1-M5), infers Metal SDK version (3.0-4.1), and exposes `DeviceInfo` with `supports_metal4`, `supports_tensor_ops`, `has_bfloat16`. Enables conditional codegen for M4+/M5+ features. | ~240 LOC (file being created) | `triton_msl/backend/device_detect.py` | -- |
| 0c | **Missing math ops (log1p, expm1)** | Map `math.log1p` and `math.expm1` to MSL `log(1+x)` / `exp(x)-1` (or precise equivalents). Needed for numerical stability in loss functions. | ~20 LOC | `triton_msl/codegen/generic_lowerer.py` (`_lower_math`) | -- |
| 0d | **tt.extern_elementwise** | Handle `tt.extern_elementwise` by dispatching to a user-provided MSL function name or a built-in libdevice shim. Unblocks upstream tests that use `tl.extra.cuda.libdevice.*`. | ~60 LOC | `triton_msl/codegen/generic_lowerer.py` (`_lower_op_dispatch`), `triton_msl/inductor/metal_libdevice.py` | -- |
| 0e | **scf.for in device functions** | Currently emits `// UNSUPPORTED: scf.for in device func`. Port `_lower_scf_for` logic from `GenericLowerer` into `_DeviceFuncLowerer` so noinline callees can contain loops. | ~80 LOC | `triton_msl/codegen/generic_lowerer.py` (`_DeviceFuncLowerer._lower_op`) | -- |
| 0f | **conftest skip list update** | Update `scripts/conftest_metal.py` skip predicates to reflect newly-passing test categories (atomics, while loops, 2D reduce, etc.). Reduces noise in upstream test runs. | ~30 LOC | `scripts/conftest_metal.py` | -- |
| 0g | **macOS 26 (Tahoe) compatibility** | Verify `device_detect.py` handles macOS 26+ version strings correctly. PyTorch MPS broke on Tahoe due to version parsing ("26.0" not parsed as >= "14.0"). Ensure our Metal version probing, deployment target flags, and `xcrun` invocations work on macOS 26.x. | ~40 LOC | `triton_msl/backend/device_detect.py`, `triton_msl/backend/compiler.py` | 0b |
| 0h | **Graceful fallback for unsupported kernels** | When a Triton kernel fails to compile to MSL (e.g., unsupported 2D patterns from PyTorch 2.11 inductor), fall back to CPU execution with a logged warning instead of silently producing wrong results. Lesson from TorchTPU: every unlowered op falls back to CPU. Currently our torch.compile path produces *wrong values* for unsupported patterns — the worst failure mode. | ~80 LOC | `triton_msl/backend/compiler.py`, `triton_msl/inductor/__init__.py` | -- |
| 0i | **Persistent compilation cache** | Move metallib cache from `/tmp/` to `~/.cache/triton_msl/` so compiled kernels survive reboots. Cache `emit_msl()` output too (TTGIR hash → MSL string). Lesson from TorchTPU (shared compilation cache) and TurboQuant (kernel caching pattern). | ~60 LOC | `triton_msl/backend/compiler.py`, `triton_msl/debug.py` | -- |
| 0j | **sizePerThread handling** | **DONE.** Extract `sizePerThread` from TTGIR blocked layout, multi-pass reduction for reduction kernels. Softmax 18% faster (0.98ms → 0.78ms). | Commits: 47e1810..615b638 | `triton_msl/codegen/mlir_walker.py`, `triton_msl/codegen/generic_lowerer.py` | -- |

**Exit criterion for Phase 0:** All items merged to main, CI green, upstream test count unchanged or increased.

---

## Phase 1: Core Architecture

The critical path. These items transform the lowerer from a 1D-per-thread model into a 2D-capable compiler that can generically lower `tt.dot` without a prebuilt template.

**Primary success metric: torch.compile model coverage.** Phase 1A should be prioritized by restoring 32/32 torch.compile tests (9 regressions from PyTorch 2.11's `triton_per_*` 2D reduction patterns) and unlocking new models, not just upstream test_core.py pass rate. Framework compatibility drives adoption more than benchmark numbers.

### 1A. Full 2D Tensor Semantics in Codegen

| Aspect | Detail |
|--------|--------|
| **What** | Extend the generic lowerer's execution model so that each thread can compute its (row, col) position within a 2D tile. Today `tt.make_range` always maps to `lid` (1D thread index). After this change, a `make_range` on dim 0 maps to `lid / N_cols` and on dim 1 maps to `lid % N_cols`, using the shape information from Phase 0a. `tt.expand_dims` and `tt.broadcast` must become real ops that produce correct indexing expressions instead of passthroughs. |
| **Why** | Without this, any kernel that uses 2D indexing (matmul, 2D convolution, multi-head attention with 2D tiling, `test_bin_op` scalar broadcast) produces incorrect results. The ~780 "numerical mismatch" upstream failures are primarily caused by this. |
| **Scope** | ~400 LOC. Touches `_lower_make_range`, `_lower_expand_dims`, `_lower_broadcast`, `_lower_load`, `_lower_store` (2D pointer arithmetic), and thread launch logic in `KernelBuilder`. |
| **Dependencies** | Phase 0a (2D shape tracking) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (core changes), `triton_msl/codegen/msl_emitter.py` (2D threadgroup dispatch in `KernelBuilder`) |

### 1B. ttg.local_alloc -- Threadgroup Shared Memory

| Aspect | Detail |
|--------|--------|
| **What** | Implement `ttg.local_alloc` as a real op that declares a `threadgroup` shared memory array in MSL, with proper sizing from the tensor type annotation. `ttg.local_store` writes to it, `ttg.local_load` reads from it. Currently all three are passthroughs, which is correct for 1D elementwise kernels but wrong for cooperative algorithms (matmul tiling, cross-warp reductions beyond 1024 elements). |
| **Why** | Required for generic `tt.dot` lowering (tiles of A and B must be staged in threadgroup memory for cooperative loading). Also needed for reductions on tensors larger than one threadgroup. |
| **Scope** | ~250 LOC. New shared memory allocation tracking in `GenericLowerer.__init__`, emitters for alloc/load/store, MSL declaration in kernel prologue. |
| **Dependencies** | Phase 1A (2D semantics for indexing into shared memory) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (`_lower_ttg`), `triton_msl/codegen/msl_emitter.py` (shared memory declarations) |

### 1C. Generic tt.dot via simdgroup_matrix_multiply_accumulate

| Aspect | Detail |
|--------|--------|
| **What** | Replace `_lower_dot_via_prebuilt_template()` (~580 LOC of template string generation) with generic lowering that maps `tt.dot` to Metal's `simdgroup_matrix<float, 8, 8>` MMA intrinsic. The lowerer will: (1) allocate threadgroup memory for A and B tiles via 1B, (2) emit cooperative tile loading using 2D thread positions from 1A, (3) emit the `simdgroup_multiply_accumulate` call, (4) handle the accumulator. This replaces ~3 separate template paths (`_lower_dot_simple_template`, `_lower_dot_constant_template`, strided dot) with one generic path. |
| **Why** | The prebuilt templates are fragile, hard to extend (every new epilogue needs a new template variant), and limited to specific pointer layouts. A generic path lets Triton's optimizer handle fusion and layout -- the lowerer just translates ops. |
| **Scope** | ~500 LOC new generic path, ~-580 LOC removed templates (net ~-80 LOC). High complexity: must handle SIMD group coordination, 8x8 tile decomposition, accumulator initialization. |
| **Dependencies** | Phase 1A (2D semantics), Phase 1B (threadgroup memory) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (new `_lower_dot` implementation, delete `_lower_dot_via_prebuilt_template`, `_lower_dot_simple_template`, `_lower_dot_constant_template`) |

### 1D. K-Loop Handling (scf.for Wrapping tt.dot)

| Aspect | Detail |
|--------|--------|
| **What** | Handle the common pattern where `scf.for` wraps `tt.dot` to iterate over the K dimension. Today this works for the prebuilt template (which hardcodes the K loop) but must work with the generic path from 1C. The `scf.for` body will contain: `tt.load` (load next K-tile of A and B into shared), `tt.dot` (accumulate), `scf.yield` (carry accumulator). The lowerer must emit correct MSL `for` loop with threadgroup barrier between load and compute. |
| **Why** | All real matmul kernels (including Triton tutorial 03) tile the K dimension. Without this, only tiny matmuls that fit in a single tile work. |
| **Scope** | ~150 LOC. Mostly ensuring `_lower_scf_for` correctly handles iter_args that are matrix accumulators, and inserting `threadgroup_barrier(mem_flags::mem_threadgroup)` between the cooperative load and the MMA. |
| **Dependencies** | Phase 1C (generic tt.dot) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (`_lower_scf_for`, barrier insertion logic) |

### 1E. 2D Matmul in Device Functions

| Aspect | Detail |
|--------|--------|
| **What** | Extend `_DeviceFuncLowerer` to support 2D ops and `tt.dot` within noinline device functions. Currently, `scf.for` is unsupported in device funcs (Phase 0e fixes the loop, but 2D ops are still passthroughs). |
| **Why** | Upstream Triton tests use noinline functions containing matmul. Also needed for modular kernel design where matmul is a sub-routine. |
| **Scope** | ~200 LOC. Port 2D indexing and shared memory logic from `GenericLowerer` into `_DeviceFuncLowerer`. |
| **Dependencies** | Phase 0e (scf.for in device funcs), Phase 1C (generic tt.dot) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (`_DeviceFuncLowerer`) |

### 1F. Loop Fusion (tl.range Optimization)

| Aspect | Detail |
|--------|--------|
| **What** | When Triton emits multiple sequential `scf.for` loops that iterate over the same range and access the same data, fuse them into a single loop. This is a post-walk optimization pass on the IRGraph. `tl.range` in user code often results in separate loops for load, compute, and store phases that could be a single loop. |
| **Why** | Reduces kernel launch overhead and improves cache/register utilization. Important for achieving competitive performance on memory-bound kernels. |
| **Scope** | ~200 LOC. New optimization pass in `mlir_walker.py` or a new file `triton_msl/codegen/optimizations.py`. |
| **Dependencies** | Phase 1D (K-loop handling, so we understand the loop IR patterns) |
| **Files** | `triton_msl/codegen/mlir_walker.py` or new `triton_msl/codegen/optimizations.py`, `triton_msl/codegen/generic_lowerer.py` (invoke pass before lowering) |

### 1G. Cooperative Grid Sync

| Aspect | Detail |
|--------|--------|
| **What** | Support `launch_cooperative_grid=True` in `MetalOptions`. Metal does not have a direct equivalent of CUDA's cooperative launch (`cudaLaunchCooperativeKernel`), but threadgroup barriers and device-memory fences (`mem_flags::mem_device`) can approximate single-pass multi-threadgroup synchronization for specific patterns (e.g., persistent kernels). |
| **Why** | Required for persistent kernel patterns used in advanced Triton kernels (FlashAttention, fused optimizers). Also needed for `tt.debug_barrier` and cross-threadgroup reductions. |
| **Scope** | ~150 LOC. Changes to driver dispatch logic, kernel argument additions (global sync counter buffer), MSL fence intrinsics. |
| **Dependencies** | Phase 1B (threadgroup memory, as cooperative sync patterns often use shared memory) |
| **Files** | `triton_msl/backend/driver.py` (cooperative dispatch path), `triton_msl/codegen/generic_lowerer.py` (emit sync primitives), `triton_msl/backend/compiler.py` (propagate option) |

### 1H. Dynamic Shape Support (Full Symbolic Dispatch)

| Aspect | Detail |
|--------|--------|
| **What** | Today, block sizes and tensor shapes are `tl.constexpr` -- they must be known at compile time. Dynamic shape support means passing shape dimensions as runtime scalar arguments, and the lowerer emitting MSL that uses those scalar arguments for bounds checking and indexing instead of hardcoded constants. The driver must also compute grid dimensions dynamically from the runtime shapes. |
| **Why** | Eliminates recompilation when input shapes change (common in inference with variable sequence lengths). Required for `torch.compile` with dynamic shapes. |
| **Scope** | ~300 LOC. Changes span the full pipeline: walker must track which values are symbolic, lowerer must emit runtime expressions instead of constant-folded values, driver must compute grids dynamically. |
| **Dependencies** | Phase 1A (2D semantics, since dynamic shapes affect 2D indexing) |
| **Files** | `triton_msl/codegen/mlir_walker.py` (symbolic value tracking), `triton_msl/codegen/generic_lowerer.py` (symbolic codegen), `triton_msl/backend/driver.py` (dynamic grid calculation), `triton_msl/backend/compiler.py` (metadata for dynamic args) |

**Phase 1 summary:**

| Metric | Value |
|--------|-------|
| Total estimated LOC | ~2,150 new/changed |
| Complexity | High (core compiler changes) |
| Expected upstream test impact | +1,500-2,500 passing (2D semantics fix the majority of failures) |
| Key risk | 2D indexing correctness across all op combinations |

---

## Phase 2: Performance Foundations

These items improve runtime performance without changing correctness. Can be worked in parallel with Phase 1 items that don't share dependencies.

### 2A. Buffer Copy Overhead Elimination on MPS Path

| Aspect | Detail |
|--------|--------|
| **What** | Eliminate the MPS tensor double-copy (`MPS -> CPU -> Metal buffer -> kernel -> Metal buffer -> CPU -> MPS`). Three approaches in priority order: (1) Detect page-aligned CPU tensors and use `newBufferWithBytesNoCopy` (already done for CPU path, extend heuristic). (2) Investigate `MTLSharedEvent` for direct MPS command buffer interop. (3) If PyTorch exposes `MTLBuffer` handles in future versions, use direct buffer sharing. |
| **Why** | The ~0.15ms per-launch copy overhead dominates small kernel latency. Removing it would give 55x improvement for aligned cases (per ARCHITECTURE.md measurements). |
| **Scope** | ~200 LOC. Primarily driver changes, plus alignment detection utilities. |
| **Dependencies** | None (independent of Phase 1) |
| **Files** | `triton_msl/backend/driver.py` (buffer creation/copy-back logic), `triton_msl/buffer_pool.py` (alignment-aware allocation) |

### 2B. Kernel Fusion Opportunities

| Aspect | Detail |
|--------|--------|
| **What** | Fuse consecutive kernel launches when the output of kernel A is the input of kernel B and no other consumer exists. This is a driver-level optimization: instead of commit-wait-commit, encode both kernels into one command buffer and let Metal handle scheduling. The existing batched dispatch (`_batch_mode`) provides the foundation. |
| **Why** | Reduces command buffer overhead and allows Metal's GPU scheduler to overlap execution. Important for torch.compile where the inductor may emit many small kernels. |
| **Scope** | ~250 LOC. Extend batched dispatch to detect fusable sequences, add buffer lifetime tracking. |
| **Dependencies** | None |
| **Files** | `triton_msl/backend/driver.py` (fusion detection in launch path), `triton_msl/inductor/metal_libdevice.py` (hint annotations for fusable sequences) |

### 2C. Autotuning Improvements (Search Space Expansion)

| Aspect | Detail |
|--------|--------|
| **What** | Expand the autotuning search space beyond `num_warps` and `BLOCK_SIZE`. Add: (1) `num_stages` (software pipelining depth, maps to number of in-flight tiles), (2) tile shape configurations for 2D kernels (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), (3) Metal-specific parameters (threadgroup memory usage strategy, SIMD group count). Also add a persistent cache backed by SQLite to avoid re-tuning across runs. |
| **Why** | Current autotuning only explores 1D block sizes. 2D kernels (matmul, attention) have a much richer configuration space where the optimal tile size depends on problem dimensions and hardware. |
| **Scope** | ~300 LOC. Autotuner configuration expansion, SQLite cache. |
| **Dependencies** | Phase 1A (2D semantics, for 2D tile size tuning to be meaningful) |
| **Files** | `triton_msl/autotuning/autotuner.py`, `triton_msl/autotuning/__init__.py`, new `triton_msl/autotuning/cache.py` |

### 2D. Benchmark Suite Expansion

| Aspect | Detail |
|--------|--------|
| **What** | Add benchmarks for: (1) 2D matmul at multiple sizes (128-4096), (2) fused attention (FlashAttention), (3) reduction sweep (1D, 2D, 3D at various shapes), (4) torch.compile model inference latency, (5) comparison against MLX native ops and MPS. Output results as JSON for CI regression tracking. |
| **Why** | Current benchmarks cover basic operations but lack the coverage needed to detect performance regressions or guide optimization work. |
| **Scope** | ~400 LOC. New benchmark scripts plus a regression tracker. |
| **Dependencies** | None (can use existing kernel paths) |
| **Files** | `benchmarks/bench_all.py` (extend), new `benchmarks/bench_matmul_sweep.py`, new `benchmarks/bench_attention.py`, `benchmarks/bench_regression.py` (extend with JSON output) |

### 2E. Shared Memory Lifetime Analysis and Aliasing

| Aspect | Detail |
|--------|--------|
| **What** | Implement shared memory reuse: when multiple `ttg.local_alloc` arrays have non-overlapping lifetimes, alias them to the same physical shared memory. Currently each `local_alloc` gets a unique `threadgroup` array, which accumulates to >32KB for complex kernels like FlashAttention with HEAD_DIM=64 (41KB needed, 32KB limit). Analyze SSA def-use chains to determine which arrays are live simultaneously, then assign overlapping arrays to the same base allocation. |
| **Why** | Metal caps threadgroup memory at 32KB. FlashAttention with HEAD_DIM=64 needs Q(8KB) + K(8KB) + V(8KB) + S(4KB) + dot_result(8KB) = 36KB+. K can be freed before V is loaded, so K and V can alias. This optimization would unlock HEAD_DIM=64 flash attention and larger matmul tiles. |
| **Scope** | ~200 LOC. Lifetime analysis pass after the lowerer's op scan, shared memory allocation planner, alias assignment. |
| **Dependencies** | Phase 1B (ttg.local_alloc as real ops — DONE) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (lifetime analysis + alias assignment) |

**Phase 2 summary:**

| Metric | Value |
|--------|-------|
| Total estimated LOC | ~1,350 |
| Complexity | Medium |
| Parallelizable with | Phase 1 (items 2A, 2B, 2D are fully independent) |

---

## Phase 3: Metal 4 / M5 Enablement

These items target Apple's next-generation GPU architecture and Metal 4 API. They require Phase 0b (device detection) and benefit from Phase 1 (2D semantics).

### 3A. Phase 2 Tensor Op Matmul Kernels (mpptensorop_half8x8)

| Aspect | Detail |
|--------|--------|
| **What** | On M5 hardware with Metal 4.1, use the new GPU tensor op intrinsics (`mpptensorop_half8x8` / equivalent Metal 4.1 API) instead of `simdgroup_matrix_multiply_accumulate`. These are hardware-accelerated matrix ops through the Neural Accelerator pipeline. The codegen must conditionally emit tensor op MMA when `DeviceInfo.supports_tensor_ops` is True, falling back to simdgroup MMA otherwise. |
| **Why** | M5's tensor ops are expected to deliver 2-4x higher matmul throughput vs simdgroup MMA on M4. This is the single biggest performance opportunity for compute-bound workloads. |
| **Scope** | ~350 LOC. New MMA emission path in the generic lowerer, conditional on device detection. Metal 4.1 API surface is speculative (based on WWDC 2026 expectations), so scope may shift. |
| **Dependencies** | Phase 0b (device detection), Phase 1C (generic tt.dot -- the tensor op path replaces the MMA emission, not the surrounding tile/loop logic) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (new MMA emission path in `_lower_dot`), `triton_msl/backend/compiler.py` (Metal 4.1 compilation flags), `triton_msl/backend/device_detect.py` (already has `supports_tensor_ops`) |

### 3B. Metal 4 Command Model (MTL4ArgumentTable, Explicit Barriers)

| Aspect | Detail |
|--------|--------|
| **What** | Metal 4 introduces a new command encoding model: `MTL4ArgumentTable` replaces per-kernel `setBuffer:offset:atIndex:` calls with a pre-built argument table, and explicit barriers replace implicit resource tracking. This reduces CPU-side dispatch overhead and gives finer control over GPU execution ordering. |
| **Why** | Reduces per-kernel CPU overhead from ~50us to potentially ~10us. Also enables advanced patterns like indirect dispatch and persistent kernels that require explicit barrier control. |
| **Scope** | ~400 LOC. New Metal 4 dispatch path in the driver, parallel to the existing Metal 3.x path. Must maintain backward compatibility. |
| **Dependencies** | Phase 0b (device detection, to gate on `supports_metal4`) |
| **Files** | `triton_msl/backend/driver.py` (new `_dispatch_metal4` method alongside existing `_dispatch`), `triton_msl/buffer_pool.py` (argument table allocation) |

### 3C. nvfp4/mxfp8 Quantization Format Support

| Aspect | Detail |
|--------|--------|
| **What** | Support the microscaling FP formats (MXFP8 e4m3/e5m2, NVFP4 e2m1) that Metal 4.1 may expose as native types. If hardware support is unavailable, implement software emulation using INT8 storage + scale factors (block-scaled quantization). The codegen must handle mixed-precision dot products: `tt.dot(fp8_tensor, fp8_tensor) -> fp32_accumulator`. |
| **Why** | MXFP8 and FP4 are the dominant quantization formats for LLM inference (used by GPTQ, AWQ, GGUF). Supporting them is table-stakes for vLLM and llama.cpp integration. |
| **Scope** | ~500 LOC (high if software emulation needed). Type system additions, dequantization kernels, mixed-precision dot product support. |
| **Dependencies** | Phase 3A (tensor op matmul -- FP8 dot products are most useful with tensor ops), Phase 4A (FP8 software emulation as fallback) |
| **Files** | `triton_msl/codegen/msl_types.py` (new types), `triton_msl/codegen/generic_lowerer.py` (mixed-precision load/dot), `triton_msl/codegen/msl_emitter.py` (dequantization helpers) |

### 3D. Backward Compatibility Testing (M1-M4 Fallback Paths)

| Aspect | Detail |
|--------|--------|
| **What** | Systematic testing and fallback codegen for all conditional features: (1) M1/M2: no BF16 in shaders -- auto-promote to FP32. (2) M1-M3: no Metal 4 -- use Metal 3.x command model. (3) M1-M4: no tensor ops -- use simdgroup MMA. Build a CI matrix that tests on M1, M2, M3, M4 (via macOS VMs or physical hardware). |
| **Why** | triton-msl claims "M1 or later" support. Without regression testing on older hardware, Metal 4 codegen changes could silently break M1-M3 users. |
| **Scope** | ~200 LOC (codegen fallback guards) + CI configuration. |
| **Dependencies** | Phase 0b (device detection), Phase 3A/3B (features that need fallbacks) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (conditional emission), `triton_msl/backend/compiler.py` (MSL version flag selection), `.github/workflows/` (CI matrix), new `tests/test_device_compat.py` |

**Phase 3 summary:**

| Metric | Value |
|--------|-------|
| Total estimated LOC | ~1,450 |
| Complexity | High (hardware-dependent, speculative API surface) |
| Key risk | Metal 4.1 API details unknown until WWDC 2026 |
| Mitigation | Design abstractions now (Phase 0b), implement when SDK ships |

---

## Phase 4: Type System Expansion

### 4A. FP8 Software Emulation (e4m3, e5m2)

| Aspect | Detail |
|--------|--------|
| **What** | Implement FP8 types (`float8_e4m3fn`, `float8_e5m2`) as software-emulated types stored in INT8 buffers. Provide: (1) `fp8_to_float` / `float_to_fp8` conversion functions in MSL, (2) load/store that auto-dequantize/quantize, (3) arithmetic that promotes to FP16/FP32, computes, and truncates back. |
| **Why** | FP8 is used by transformer inference libraries (vLLM, TRT-LLM, FlashInfer). Even without hardware FP8, software emulation enables running FP8-quantized models on Apple Silicon. The ~500 upstream tests skipped for FP8 could partially pass. |
| **Scope** | ~300 LOC. MSL helper functions, type system additions, load/store modifications. |
| **Dependencies** | None (independent, but benefits from Phase 1A for 2D FP8 tensors) |
| **Files** | `triton_msl/codegen/msl_types.py` (FP8 type definitions), `triton_msl/codegen/msl_builtins.py` (conversion functions), `triton_msl/codegen/generic_lowerer.py` (FP8-aware load/store/cast), `scripts/conftest_metal.py` (un-skip FP8 tests) |

### 4B. INT4 Weight-Only Quantization

| Aspect | Detail |
|--------|--------|
| **What** | Support INT4 packed weights: two INT4 values per byte, with group-wise FP16 scales. Provide: (1) INT4 unpacking in MSL (bit shifts + masks), (2) dequantization (`int4_val * scale`), (3) integration with `tt.dot` for W4A16 matmul (FP16 activations x INT4 weights). |
| **Why** | INT4 quantization (GPTQ, AWQ) halves model memory, critical on Apple Silicon's unified memory. The existing `make_int4_matmul_kernel` in `msl_emitter.py` is a prebuilt template; this makes it generic. |
| **Scope** | ~250 LOC. MSL unpack helpers, modified load path for packed types, scale application. |
| **Dependencies** | Phase 1C (generic tt.dot, so INT4 matmul can use the generic path) |
| **Files** | `triton_msl/codegen/msl_builtins.py` (INT4 unpack), `triton_msl/codegen/generic_lowerer.py` (packed load), `triton_msl/codegen/msl_types.py` (INT4 type) |

### 4C. Mixed Precision Support Improvements

| Aspect | Detail |
|--------|--------|
| **What** | Improve automatic mixed-precision handling: (1) BF16 compute should use FP32 intermediates (already done for FP16, verify for BF16). (2) INT8 matmul with INT32 accumulator. (3) FP16 x INT8 mixed-type operations. (4) Correct truncation/rounding modes for all cast pairs. |
| **Why** | Real models use many precision combinations. Current type handling is ad-hoc; a systematic approach prevents subtle numerical bugs. |
| **Scope** | ~200 LOC. Audit and fix cast emission for all type pairs. |
| **Dependencies** | Phase 4A (FP8 adds more type pairs to handle) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (`_lower_arith` cast handlers), `triton_msl/codegen/msl_types.py` |

**Phase 4 summary:**

| Metric | Value |
|--------|-------|
| Total estimated LOC | ~750 |
| Complexity | Medium |
| Expected upstream test impact | +200-500 passing (FP8 un-skips) |

---

## Phase 5: Training Support

Training requires backward pass generation and memory management for gradients. This is the largest single feature area and depends on most of Phase 1.

### 5A. Backward Pass / Gradient Support

| Aspect | Detail |
|--------|--------|
| **What** | Generate backward kernels for forward kernels. Two approaches: (1) **Triton-level**: rely on Triton's `@triton.jit` to define both forward and backward kernels manually (user writes both, we just compile both). (2) **Autograd-level**: integrate with PyTorch's autograd to register custom backward functions for Triton-compiled forward ops. Start with approach 1 (lower barrier), then add approach 2. |
| **Why** | Training on Apple Silicon is the most-requested feature. Without backward pass support, triton-msl is inference-only. |
| **Scope** | ~500 LOC for approach 1 (register manual backward kernels with autograd), ~800 LOC additionally for approach 2 (automatic backward generation). |
| **Dependencies** | Phase 1 (2D semantics -- backward matmul needs 2D), Phase 2A (buffer copy elimination -- training is memory-intensive) |
| **Files** | New `triton_msl/training/__init__.py`, new `triton_msl/training/autograd.py`, `triton_msl/backend/driver.py` (gradient buffer management), `triton_msl/inductor/metal_libdevice.py` (backward op registration) |

### 5B. Autograd Integration

| Aspect | Detail |
|--------|--------|
| **What** | Create `torch.autograd.Function` subclasses that wrap Triton kernel pairs (forward + backward). The forward saves tensors for backward via `ctx.save_for_backward`. The backward dispatches the backward kernel. Register these with torch.compile so that `torch.compile(model, backend="metal")` automatically uses Triton kernels for both forward and backward passes. |
| **Why** | Enables `model.train()` with torch.compile on Metal. Without this, users must manually manage gradient computation. |
| **Scope** | ~400 LOC. Autograd function wrappers, context management, inductor integration. |
| **Dependencies** | Phase 5A (backward kernel compilation) |
| **Files** | `triton_msl/training/autograd.py`, `triton_msl/inductor/metal_libdevice.py` (register backward patterns), new `tests/test_training.py` |

### 5C. Memory Management for Training

| Aspect | Detail |
|--------|--------|
| **What** | Training requires: (1) **Gradient accumulation**: sum gradients across microbatches without extra memory allocation (in-place atomic adds to gradient buffers). (2) **Optimizer state**: Metal buffers for Adam/SGD momentum and variance. (3) **Activation checkpointing**: recompute forward activations during backward instead of storing all intermediates. (4) **Memory pool**: extend `MetalBufferPool` with generation-based cleanup (free all buffers from the forward pass after backward completes). |
| **Why** | Apple Silicon's unified memory is typically 16-128 GB. Training a 7B model requires ~28 GB for weights + gradients + optimizer state in FP16. Without careful memory management, OOM on 32 GB machines. |
| **Scope** | ~600 LOC. Buffer pool extensions, gradient accumulation kernels, optimizer state management. |
| **Dependencies** | Phase 5B (autograd integration, to know buffer lifetimes) |
| **Files** | `triton_msl/buffer_pool.py` (generation-based cleanup), `triton_msl/backend/driver.py` (gradient buffer APIs), new `triton_msl/training/memory.py`, new `triton_msl/training/optimizers.py` |

**Phase 5 summary:**

| Metric | Value |
|--------|-------|
| Total estimated LOC | ~2,300 |
| Complexity | Very high |
| Key risk | PyTorch autograd integration is tightly coupled to internal APIs |
| Mitigation | Start with manual backward kernels (5A), defer full autograd until stable |

---

## Phase 6: Ecosystem Integration

External integrations that depend on core features being stable.

### 6A. FlashAttention on Metal

| Aspect | Detail |
|--------|--------|
| **What** | Port the Triton FlashAttention kernel (Dao et al.) to run correctly on Metal. The existing `make_flash_attention_kernel` in `msl_emitter.py` is a prebuilt MSL template. The goal is to have the standard Triton FlashAttention Python code (`@triton.jit` with tl.dot, online softmax, causal masking) compile through the generic lowerer to correct and performant MSL. |
| **Why** | FlashAttention is the most important kernel for transformer inference and training. It exercises every major subsystem: 2D tiling, K-loop, shared memory, reductions, masking, and epilogue fusion. Getting it right validates the entire stack. |
| **Scope** | ~200 LOC (codegen fixes/additions, not a new kernel). The kernel itself is already written in Triton; the work is making the lowerer handle its patterns. |
| **Dependencies** | Phase 1A-1D (2D semantics, shared memory, generic tt.dot, K-loop), Phase 1G (cooperative grid sync for persistent FlashAttention variant) |
| **Files** | `triton_msl/codegen/generic_lowerer.py` (pattern fixes discovered during FlashAttention bring-up), new `tests/test_flash_attention.py`, `benchmarks/bench_attention.py` |

### 6B. vLLM-Metal Integration

| Aspect | Detail |
|--------|--------|
| **What** | Create a vLLM executor backend for Apple Silicon that uses triton-msl for kernel dispatch. Key requirements: (1) PagedAttention kernel running on Metal, (2) KV cache management using Metal buffers, (3) Sampler kernels (top-k, top-p) on Metal, (4) Model runner that detects Apple Silicon and routes through triton-msl. |
| **Why** | vLLM is the dominant LLM serving framework. Metal support would make Apple Silicon viable for local LLM serving with state-of-the-art batching and scheduling. |
| **Scope** | ~1,000 LOC (in a separate repo/fork, plus ~200 LOC adapter in triton-msl). High complexity due to vLLM's internal architecture. |
| **Dependencies** | Phase 6A (FlashAttention -- vLLM's core kernel), Phase 4A/4B (FP8/INT4 for quantized models), Phase 1H (dynamic shapes for variable sequence lengths) |
| **Files** | New `triton_msl/integrations/vllm.py` (adapter), external vLLM fork (executor backend), `triton_msl/backend/driver.py` (KV cache buffer management) |

### 6C. Upstream Triton Contribution (Issue #4824)

| Aspect | Detail |
|--------|--------|
| **What** | Contribute triton-msl as an official third-party backend to the Triton project. Per Triton's backend plugin architecture, this means: (1) Register via `[project.entry-points."triton.backends"]` (already done). (2) Pass the upstream backend conformance tests. (3) Submit a PR to triton-lang/triton adding Metal to the backend registry and CI. (4) Address review feedback on API compatibility. |
| **Why** | Official upstream status means triton-msl gets tested in Triton CI, discovered by `pip install triton`, and maintained alongside Triton core changes. |
| **Scope** | ~100 LOC in triton-msl (conformance fixes), ~200 LOC in upstream Triton (registry, CI config, docs). Majority of effort is review and iteration. |
| **Dependencies** | Phase 1 (core architecture must be solid), target: >6,000 upstream tests passing |
| **Files** | `triton_msl/backend/compiler.py` (API conformance), `triton_msl/backend/driver.py` (API conformance), `pyproject.toml` (metadata for registry) |

### 6D. triton-ext C++ Pass Contribution

| Aspect | Detail |
|--------|--------|
| **What** | Contribute a C++ MLIR pass to triton-lang/triton-ext that performs Metal-specific TTGIR transformations: (1) Layout optimization for Apple GPU SIMD width (32). (2) Threadgroup memory allocation planning. (3) Metal-specific constant folding (e.g., eliminate FP64 operations). This would run in the Triton optimizer before the Python lowerer, improving codegen quality. |
| **Why** | C++ passes are faster and more robust than Python-level pattern matching. Other backends (AMD, Intel) contribute backend-specific passes to triton-ext. |
| **Scope** | ~1,500 LOC C++ (MLIR pass infrastructure + Metal-specific transforms). Very high complexity: requires MLIR C++ development experience and Triton's internal pass pipeline knowledge. |
| **Dependencies** | Phase 6C (upstream relationship established), Phase 1 (architecture stable enough to know what transforms are needed) |
| **Files** | New C++ project in triton-ext (separate repo). `triton_msl/backend/compiler.py` (invoke C++ pass if available) |

### 6E. PyPI Publishing Workflow Testing

| Aspect | Detail |
|--------|--------|
| **What** | Set up and test the `pip install triton-msl` publishing pipeline: (1) GitHub Actions workflow for building sdist and wheel. (2) macOS-only wheel with platform tag `macosx_14_0_arm64`. (3) Test installation on clean macOS VM. (4) Automated version bumping and changelog generation. (5) TestPyPI dry-run before real publish. |
| **Why** | Currently installable via `pip install -e .` only. A published package is required for user adoption and upstream Triton integration. |
| **Scope** | ~150 LOC (CI workflow YAML + publish script). |
| **Dependencies** | None (can be done anytime, but should wait until Phase 1 is stable for a meaningful release) |
| **Files** | New `.github/workflows/publish.yml`, new `.github/workflows/test.yml`, `pyproject.toml` (version/metadata refinement) |

**Phase 6 summary:**

| Metric | Value |
|--------|-------|
| Total estimated LOC | ~3,350 (including external repos) |
| Complexity | Very high (external integrations) |
| Longest pole | 6D (C++ pass, ~1,500 LOC, MLIR expertise required) |

---

## Dependency Graph

```
Phase 0 (In-Progress)
  |
  +--[0a: 2D shape tracking]---------+
  |                                   |
  +--[0b: Metal 4 detection]---+      |
  |                            |      |
  +--[0c: log1p/expm1]        |      |
  |                            |      |
  +--[0d: extern_elementwise]  |      |
  |                            |      |
  +--[0e: scf.for device func]-+      |
  |                            |      |
  +--[0f: conftest update]     |      |
                               |      |
Phase 1 (Core Architecture)    |      |
                               |      |
  [0a]--->[1A: Full 2D semantics]--->[1B: ttg.local_alloc]--->[1C: Generic tt.dot]
                |                                                    |
                |                                                    +--->[1D: K-loop]
                |                                                    |        |
                |                                                    |        +--->[1F: Loop fusion]
                |                                                    |
                +--->[1H: Dynamic shapes]                            +--->[1E: 2D device funcs]
                                                                     |
  [1B]--->[1G: Cooperative grid sync]                                |
                                                                     |
Phase 2 (Performance) [mostly parallel with Phase 1]                 |
                                                                     |
  [2A: Buffer copy elimination] (independent)                        |
  [2B: Kernel fusion] (independent)                                  |
  [1A]--->[2C: Autotuning expansion]                                 |
  [2D: Benchmark suite] (independent)                                |
                                                                     |
Phase 3 (Metal 4 / M5)                                              |
                                                                     |
  [0b]+[1C]--->[3A: Tensor op matmul]--->[3C: nvfp4/mxfp8]          |
  [0b]-------->[3B: Metal 4 command model]                           |
  [0b]+[3A]+[3B]--->[3D: Backward compat testing]                   |
                                                                     |
Phase 4 (Types)                                                      |
                                                                     |
  [4A: FP8 emulation] (independent)--->[4C: Mixed precision]        |
  [1C]--->[4B: INT4 quantization]                                    |
                                                                     |
Phase 5 (Training)                                                   |
                                                                     |
  [Phase 1]+[2A]--->[5A: Backward pass]--->[5B: Autograd]--->[5C: Memory mgmt]
                                                                     |
Phase 6 (Ecosystem)                                                  |
                                                                     |
  [1A-1D]+[1G]--->[6A: FlashAttention]--->[6B: vLLM integration]    |
  [Phase 1]--->[6C: Upstream Triton PR]--->[6D: C++ pass]            |
  [6E: PyPI publishing] (independent)                                |
```

---

## Prioritized Execution Order

For a solo developer, this is the recommended execution order to maximize impact at each step:

| Order | Item | Rationale |
|-------|------|-----------|
| 1 | Phase 0 (all) | Unblock everything; already in progress |
| 2 | 1A: Full 2D semantics | Highest impact: fixes ~780 numerical mismatch failures |
| 3 | 2A: Buffer copy elimination | Quick win, independent, large perf improvement |
| 4 | 1B: ttg.local_alloc | Unblocks generic matmul |
| 5 | 1C: Generic tt.dot | Eliminates ~580 LOC of fragile templates |
| 6 | 1D: K-loop handling | Completes matmul story |
| 7 | 2D: Benchmark suite | Needed before further perf work to measure progress |
| 8 | 6A: FlashAttention | Validates entire 2D + matmul stack, high visibility |
| 9 | 4A: FP8 emulation | Unblocks many skipped tests, needed for ecosystem |
| 10 | 6E: PyPI publishing | Enables user adoption |
| 11 | 1H: Dynamic shapes | Required for torch.compile with real workloads |
| 12 | 6C: Upstream Triton PR | Establishes official backend status |
| 13 | 5A-5B: Training basics | Most-requested feature |
| 14 | 3A-3B: Metal 4 / M5 | Depends on WWDC timeline |
| 15 | 6B: vLLM integration | Depends on FlashAttention + FP8 + INT4 |
| 16 | 6D: C++ pass | Long-term investment, lowest priority |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Metal 4 API surface changes at WWDC | High | Medium | Design abstractions behind `DeviceInfo`, don't hardcode API assumptions |
| 2D indexing causes subtle correctness bugs | High | High | Exhaustive upstream test_core.py validation after each change |
| PyTorch MPS internals change, breaking buffer interop | Medium | Medium | Isolate MPS-specific code in driver.py, test against PyTorch nightly |
| Triton upstream API changes break backend interface | Medium | High | Pin to Triton 3.6.x, test against main weekly |
| FP8 software emulation too slow to be useful | Medium | Low | It's a correctness bridge; real perf comes from Metal 4 hardware FP8 |
| Solo developer bandwidth | High | High | Prioritize ruthlessly: 2D semantics > everything else |
| vLLM internal architecture changes | High | Medium | Start with minimal executor, iterate with upstream |

---

## Total Scope Summary

| Phase | LOC (est.) | Complexity | Key Deliverable |
|-------|-----------|------------|-----------------|
| 0: In-Progress | ~660 | Low-Medium | Foundation ready |
| 1: Core Architecture | ~2,150 | High | 2D semantics, generic matmul |
| 2: Performance | ~1,150 | Medium | Copy elimination, autotuning |
| 3: Metal 4 / M5 | ~1,450 | High | Tensor ops, Metal 4 command model |
| 4: Type System | ~750 | Medium | FP8, INT4 |
| 5: Training | ~2,300 | Very High | Backward pass, autograd |
| 6: Ecosystem | ~3,350 | Very High | FlashAttention, vLLM, upstream |
| **Total** | **~11,810** | | |

Current codebase is ~15,000 LOC. Full roadmap roughly doubles it.
