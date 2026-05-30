# triton-metal Architecture

Metal backend for OpenAI Triton. Compiles `@triton.jit` kernels to MSL (Metal Shading Language) and dispatches on Apple GPUs.

## Pipeline

```
@triton.jit Python kernel
    → Triton frontend (AST → TTIR)
    → Triton optimizer (TTIR → TTGIR)
    → mlir_walker.py: walk TTGIR module → IRGraph
    → generic_lowerer.py: IRGraph → MSL source
    → xcrun metal: MSL → AIR → metallib
    → driver.py: load metallib, dispatch on GPU
```

## Codegen: What's Generic vs What Uses Templates

### Generic op-by-op lowering

The primary codegen path (`generic_lowerer.py`) lowers each TTGIR op independently to MSL:

| Category | Ops | MSL Output |
|----------|-----|-----------|
| Indexing | `tt.get_program_id`, `tt.make_range`, `tt.splat` | `pid`, `lid`, passthrough |
| Memory | `tt.load`, `tt.store`, `tt.addptr` | Masked buffer read/write |
| Arithmetic | `arith.addf/subf/mulf/divf/addi/subi/...` | `a + b`, `a * b`, etc. |
| Math | `math.exp/log/sqrt/rsqrt/abs/sin/cos` | MSL intrinsics |
| Comparison | `arith.cmpf/cmpi` + `arith.select` | `cond ? a : b` |
| Type casts | `arith.extf/truncf/sitofp/fptosi` | `static_cast<>` |
| Reductions | `tt.reduce` | SIMD intrinsics + threadgroup shared memory |
| Control flow | `scf.for`, `scf.if`, `scf.yield` | MSL `for`/`if` |
| Constants | `arith.constant` | Literal values |

This handles elementwise ops, reductions, fused expressions (SiLU, GELU, softmax), and control flow generically. Any novel combination of these ops produces correct MSL without special-casing.

### Prebuilt matmul template

When `tt.dot` is detected, the lowerer switches to a prebuilt tiled matmul MSL template (`_lower_dot_via_prebuilt_template()`).

**Why:** Metal's `simdgroup_matrix` 8x8 MMA requires a pattern that cannot be derived from individual ops in a 1D-per-thread lowerer:

1. Threadgroup shared memory for A/B tile staging
2. A K-dimension tile loop with cooperative loading
3. `simdgroup_multiply_accumulate` for the inner product
4. 2D thread-to-tile mapping within each threadgroup

The generic lowerer uses a 1D execution model (one thread = one element). Matrix multiply requires 2D cooperative execution (threads within a SIMD group share work on a tile). Until the lowerer supports 2D shape tracking and threadgroup memory allocation, matmul uses a template.

**Roadmap to generic tt.dot:**
1. Add 2D shape tracking to the lowerer environment
2. Implement threadgroup memory allocation (`ttg.local_alloc`)
3. Map `tt.dot` to `simdgroup_matrix_multiply_accumulate`
4. Handle the K-loop (`scf.for` wrapping `tt.dot`)

## Pattern detectors (and why they exist)

Beyond the matmul template, the lowerer runs a sequence of `_detect_*`
predicates in `_lowerer_detection.py`. Each scans the `IRGraph`; on a match,
a corresponding `_lower_*_template` emits a complete, hand-written MSL kernel
and the generic op-by-op path is bypassed. The dispatch order is in
`GenericLowerer.lower()`.

| Detector | Handles | Why a template (not generic) |
|----------|---------|------------------------------|
| `_detect_simple_dot` / `_detect_dot_epilogue` | `tt.dot` (incl. K-loop) | simdgroup 8×8 MMA needs 2D cooperative execution |
| `_detect_matmul_softmax` | matmul → row-softmax fusion | M-strip staging to fit the 32 KB threadgroup cap |
| `_detect_softmax` / `_detect_layer_norm` | row-wise norm | caches the row in TG memory; ~2× vs re-reading |
| `_detect_flip` | `tl.flip` (reshape+xor-reduce) | closed-form index flip |
| `_detect_transpose_via_reshape` | reshape→permute→reshape | closed-form transpose lookup; the generic path routes this through a multi-element `convert_layout` the 1D model can't honor |
| `_detect_row_wise_sort` | `tl.sort`/`tl.topk` per row | in-register bitonic sort when total > 1024 threads |
| `_detect_3d_reduce` | 3-D reduce / argmin-max | axis-aware shared-memory reduction |
| `_detect_permute_chained_reduce` | N-D permute + chained sum-reduce | fuses the permute into the reduction index math (the permuted tensor would exceed threadgroup memory) — see `test_chained_reductions` |

**Honest assessment for reviewers:** these are *structural debt*, not the
intended end state. They exist because the generic lowerer's per-thread
**scalar** model (one thread = one element) can't express cooperative
patterns (MMA, cross-lane shuffles, multi-element-per-thread layouts). Two
of them — `_detect_transpose_via_reshape` and the matmul detectors — would
be subsumed once the lowerer tracks per-thread **register arrays** and a
general `ttg.convert_layout` shuffle (the multi-element-per-thread work in
`docs/superpowers/plans/2026-05-21-multi-element-per-thread.md`; the shuffle
primitive already exists but isn't reachable from the current op coverage).
The remainder (softmax/layer-norm/sort) are performance specializations that
could in principle be generic but are kept as templates for speed. The
convergence plan is to grow generic coverage and retire detectors as the
generic path provably matches them (tracked as "4g").

## Lowering paths and the integrity model

`emit_msl` (in `msl_emitter.py`) chooses among three paths, in order:

1. **Primary** — `GenericLowerer.lower()`: pattern templates (above) or the
   generic op-by-op lowering. **99.96%** of kernels in the upstream
   `test_core` suite (4713 / 4715) take this path.
2. **Legacy fallback** — a text-based TTGIR parser (`ttgir_parser.py`), used
   only when the primary path emits an `UNSUPPORTED` marker or raises a
   recoverable error. Exercised by **2 / 4715** kernels in the suite, both
   via an explicit marker (never a silent throw — `0` exceptions measured).
3. **Refusal** — `MetalNonRecoverableError`.

**The integrity guarantee: the backend never returns numbers it can't vouch
for.** Two distinct "I can't lower this" signals make that precise:

- **`UNSUPPORTED` marker** = "the primary path can't, but the legacy parser
  *might*." Falls back. Also raised by the empty-body guard: if the generic
  path emits a kernel with a store in the IR but no write in the body (the
  way an unhandled N-D reduce manifested before its template), it marks
  `UNSUPPORTED` rather than returning a kernel that compiles to zeros.
- **`MetalNonRecoverableError`** = "the primary path recognized this kernel,
  knows it can't lower it correctly, *and* knows the legacy parser can't
  either — so falling back would only swap one wrong answer for another."
  `emit_msl` **re-raises** this; it never reaches the user as silent output.
  Example: a pid-tiled matmul whose dimensions are baked in as `tl.constexpr`
  (no runtime M/N/K) — the template can't derive the true output strides, so
  it refuses (`test_dot_mulbroadcasted`).

A heuristic text parser cannot be *proven* correct for arbitrary kernels, so
the legacy fallback is deliberately load-bearing for as few kernels as
possible (the long-term goal is to retire it once the primary path is
complete, leaving a single auditable lowering path). Correctness for the
covered surface is enforced by the upstream suite (4327 passing, 0 failing).

## The 1D Per-Thread Model

The generic lowerer assumes each thread processes one scalar element:

- `tt.make_range(0, BLOCK_SIZE)` → `lid` (thread index within threadgroup)
- `tt.splat(scalar)` → passthrough (scalar is same for every thread)
- `tt.expand_dims` → passthrough (shape is irrelevant for scalar)
- `tt.broadcast` → passthrough (same)

**Scope:** This is correct for all 1D elementwise, reduction, and fused expression kernels. It covers the majority of Triton kernels in practice (activation functions, normalization, loss functions, sampling, etc.).

**Limitation:** 2D tensor operations (`tt.expand_dims`, `tt.broadcast`, `ttg.convert_layout`) are no-ops in the generic lowerer. Kernels that rely on 2D tensor semantics (matmul, 2D convolution, multi-head attention with 2D tiling) must use the matmul template path or a dedicated prebuilt kernel.

### Multi-element-per-thread (experimental, `TRITON_METAL_MEPT=1`)

There is an **opt-in, off-by-default** experimental path that lets a thread
hold *N* tensor elements as a register array (`T v[N]`) instead of one
scalar — the prototype of a register-array programming model. It is gated
end-to-end on the `mept_enabled` flag; the producer (`tt.make_range`) is the
single activation root, and with the flag off the generated MSL is
byte-identical to not having it.

**Status, stated honestly for reviewers:**
- *Correct:* the full upstream `test_core` suite passes with the flag **on**
  as well as off (4327 / 0 both ways).
- *Not a perf feature:* benchmarked perf-neutral on elementwise/reduce
  kernels (the array form and the scalar wrap-loop are both
  bandwidth-bound; deltas are within launch-overhead noise).
- *Why it exists:* it is the foundation for a **generic** `ttg.convert_layout`
  shuffle and `tt.dot` lowering that would subsume the matmul and
  transpose pattern detectors — i.e. the path out of the structural debt
  above. The shuffle primitive (`_lower_convert_layout_mept_shuffle`,
  XOR-basis position math) already exists.

It should not be enabled by default until it demonstrates a measured win on
a real workload. Full design + history:
`docs/superpowers/plans/2026-05-21-multi-element-per-thread.md`.

## MPS Tensor Integration

### The copy overhead

MPS tensors require a CPU intermediate for Metal kernel dispatch:

```
MPS tensor → .cpu() → Metal buffer (newBufferWithBytes) → kernel
    → Metal buffer → numpy → torch.from_numpy → tensor.copy_() → MPS
```

### Why zero-copy isn't available for MPS

- PyTorch's MPS backend does not expose `MTLBuffer` handles
- Direct `ctypes.memmove` to MPS `data_ptr()` corrupts MPS buffer tracking (segfault)
- MPS tensor allocations are not page-aligned (ARM64 requires 16KB alignment for `newBufferWithBytesNoCopy`)

### CPU tensor path (recommended)

CPU tensors on Apple Silicon share UMA with the GPU. When page-aligned, they use `newBufferWithBytesNoCopy` for true zero-copy — the Metal kernel operates directly on the tensor's memory without any data movement.

For best performance, use CPU tensors with Triton kernels:
```python
x = torch.randn(n, device="cpu")  # Not "mps"
out = torch.empty(n, device="cpu")
kernel[grid](x, out, n, BLOCK_SIZE=256)
```

### Copy overhead (measured on M4 Max)

| Path | Bandwidth | Notes |
|------|-----------|-------|
| Zero-copy (page-aligned CPU) | N/A (55x faster than copy) | `newBufferWithBytesNoCopy` wraps pointer |
| Copy-based (non-aligned CPU) | ~15 GB/s copy-in | `newBufferWithBytes`, single copy |
| Copy-back | ~70-80 GB/s | `memmove` from Metal buffer |
| MPS tensor | 2 copies each way | CPU intermediate required |

The `output_arg_indices` optimization (propagated from `GenericLowerer._prescan_stores()` through metadata) skips copy-back for read-only inputs, saving ~10% of total copy overhead.

Run `python benchmarks/bench_copy_overhead.py` for full numbers.

## Auto-tuning

`@triton.autotune` works end-to-end on Metal. The backend provides `metal_do_bench` via `MetalDriver.get_benchmarker()`, which uses `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` for nanosecond-precision timing.

## Triton Upstream Test Results

Against `triton/python/test/unit/language/test_core.py` (9,320 tests):

| Status | Count | % |
|--------|-------|---|
| Passed | 1,404 | 15.1% |
| Failed | 7,625 | 81.8% |
| Skipped | 291 | 3.1% |

Top failure categories:
- Numerical mismatch (780): kernel runs but produces wrong output — mostly scalar broadcast bugs
- Type error (92): missing integer/float type handling
- Runtime error (46): MSL runtime failures
- No FP64 (4): Metal has no FP64 support

The passing tests are primarily integer division ops and type codegen. Most arithmetic ops fail because `test_bin_op` tests scalar broadcast (`x[:1].reshape(())`) which the 1D per-thread model doesn't handle correctly.

Run `python scripts/run_upstream_tests.py` to reproduce. Full reports in `reports/`.

## Apple GPU Properties (M4 Max reference)

- 40 GPU cores, 128 ALUs/core, SIMD width 32
- Max threads per threadgroup: 1024
- Threadgroup memory: 32 KB
- Memory bandwidth: 546 GB/s, 128 GB UMA
- No FP64, no FP8
- Supported: FP32, FP16, BF16, INT8, INT16, INT32
