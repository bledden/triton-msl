# Fast matmul routing — design (2026-06-15)

> Route eligible clean matmuls to the existing, proven `make_simdgroup_matmul_kernel_fast`
> template (direct device `simdgroup_load`/`simdgroup_store`, register blocking, no
> threadgroup staging, no serial epilogue) instead of the generic dot lowering's
> staged kernel (34 barriers, serial fragment-by-fragment masked epilogue). **Measured
> at 2048³, relerr 0.0 (exact) both dtypes: fp32 11.2 TFLOP/s = 98% of `torch.matmul`;
> fp16 7.8 TFLOP/s = 77% — vs the generic lowering's ~2.8 TFLOP/s (37%).** Phase 4 (perf), the
> matmul lever after compile_shader. Purely additive: ineligible matmuls fall back to
> the generic lowering unchanged. Prime directive: never silent-wrong.

## Problem (profiled + ground-truthed)

A standard `@triton.jit` matmul (tutorial-style: own `BM×BN` tiling, K-loop
`tl.dot`-accumulate, masked store) lowers through the **generic dot lowering** in
`triton_metal/codegen/_lowerer_templates.py`. Dumping its MSL (fp16 2048³, BM=BN=64,
BK=32) confirms the inefficiency directly:

- The kernel stages A/B through `threadgroup half tg_A[512]` / `tg_B` with a barrier
  per K-step, and
- has a **serial epilogue**: each 8×8 accumulator fragment is `simdgroup_store`d into a
  `threadgroup float tg_st[4u*64u]` scratch, then 64 threads copy it out to `C`
  **element-by-element with bounds checks** (`if (gr<_M && gc<_N) c_ptr[...] = half(...)`),
  one fragment at a time — **34 `threadgroup_barrier`s total**.

This is the diagnostic signature observed in profiling (larger tiles make it *slower*,
2.8→2.4→… TFLOP/s): the kernel is barrier/threadgroup-memory bound, not MMA bound. The
MMA path *is* firing (`simdgroup_float8x8` accumulators, `simdgroup_half8x8` fragments) —
it is a template-**quality** problem, not a missed detector.

Two other matmul paths exist and were ruled out as the lever:
- `make_simdgroup_matmul_kernel` (`_msl_templates.py:3628`, the ttgir_parser prebuilt
  `simdgroup_matmul`, 32×32 staged tile, ~7 TFLOP/s claim) — the standard `@triton.jit`
  matmul does **not** route here (verified: emitted kernel is named `mm`, the generic
  lowering, not `simdgroup_matmul`). Swapping it would not help the common case.
- `make_simdgroup_matmul_kernel_fast` (`_msl_templates.py:3547`) — already implements
  the optimization (direct device load/store, register blocking rr=rc=4 → 32×128 tile,
  zero staging, zero epilogue barriers) but is **unwired** (no caller in `triton_metal/`).
  This is the proven asset to route to.

## Approach (confirmed by direct benchmark)

Wire the existing fast template into the generic dot lowering as a **whole-kernel
substitution** gated by an airtight eligibility contract, with the current generic
lowering as the fallback. We do **not** rewrite the generic lowering's body and we do
**not** rewrite the fast template — the fast template is proven (7.8 TFLOP/s, relerr 0.0)
and reused verbatim (only renamed to the user's kernel entry-point name).

Rejected alternatives:
- **(Y) Rewrite the generic lowering's kernel body in place** — replace the staged
  loads + serial epilogue with direct `simdgroup_load`/`simdgroup_store` + register
  blocking parameterized by the user's arbitrary `BM/BN/BK`. Avoids a grid override
  (keeps the user's 2-D grid) but is *new* parameterized MMA codegen for arbitrary tile
  sizes — higher silent-wrong surface than reusing a proven fixed template. Deferred.
- **(Z) Swap the ttgir_parser prebuilt `simdgroup_matmul` → `_fast`** — doesn't help the
  common path (standard matmuls don't route there).

## Architecture

The dispatched threadgroup count in the launcher is `gridX·gridY·gridZ` (collapsed to
1-D unless `needs_2d_grid`), and `grid{X,Y,Z}` come from the **user's** launch lambda
(`grid=(cdiv(M,BM), cdiv(N,BN))`). There is **no existing grid-override machinery**. The
fast template uses its own 32×128 tile (`n_groups = ceil(M/32)·ceil(N/128)`, 128 threads,
1-D `pid [[threadgroup_position_in_grid]]`), independent of the user's `BM/BN`. So wiring
it in requires computing `n_groups` at dispatch from the runtime M/N values and overriding
the user's grid. Components:

1. **Eligibility detector** (in the dot-lowering path, e.g. `_lowerer_templates.py` /
   `_lowerer_detection.py`): recognize the clean single-matmul pattern and verify the
   airtight contract (below). Returns the M/N/K arg indices + dtype on success, `None`
   otherwise.

2. **Whole-kernel substitution:** when eligible, emit `make_simdgroup_matmul_kernel_fast(
   dtype=...)` as the kernel MSL, **renamed** from `simdgroup_matmul_fast` to the user's
   kernel entry-point name (the launcher dispatches by `self.kernel_name`). The substituted
   kernel declares buffers `A,B,C,M,N,K` at `[[buffer(0..5)]]`; the launcher binds the
   full karg list (a_ptr,b_ptr,c_ptr,M,N,K,strides,…) positionally — extra trailing buffer
   bindings are unused by the kernel (safe), but correctness **depends on** the contract's
   contiguity guarantee (strides must be the row-major defaults the template assumes).

3. **Fast-matmul dispatch metadata:** a new metadata descriptor, e.g.
   `fast_matmul = {"m_idx": i, "n_idx": j, "tile_m": 32, "tile_n": 128, "threads": 128}`
   (arg indices into the *non-constexpr* karg list). Threaded through compiler metadata
   the same way `needs_2d_grid`/`output_indices`/`block_size` are.

4. **Launcher grid override** (`driver.py` `MetalLauncher.__call__`): if `fast_matmul` is
   present, compute `n_groups = ceil(int(kargs[m_idx]) / tile_m) · ceil(int(kargs[n_idx]) /
   tile_n)` and dispatch `n_groups` threadgroups of `threads` each (1-D), **ignoring**
   `gridX/gridY/gridZ`. This composes with the compile_shader fast-path
   (`threads = n_groups·tile_threads`, `group_size = threads_per_tg`) and with the existing
   host-round-trip path (`grid = (n_groups, 1, 1)`).

## Eligibility contract (airtight — any miss → fall back to generic lowering)

A wrong matmul is silent-wrong (numerically plausible garbage), so the gate is
conservative; the fallback (generic lowering) is correct for everything excluded.

- **Single clean matmul:** exactly one `tt.dot`, accumulated in a K-loop, with no fused
  epilogue beyond an optional output dtype cast (`acc.to(fp16)`) — no activation, bias,
  mask-on-output, or second consumer of the result.
- **Row-major contiguous A, B, C:** A is `[M,K]` with strides `(K,1)`, B is `[K,N]` with
  strides `(N,1)`, C is `[M,N]` with strides `(N,1)` — i.e. the exact layout the fast
  template indexes (`A[gr*K+gc]`, `B[k*N+gc]`, `C[gr*N+gc]`). Established from the addptr
  pattern in the IR (or refused if not provable). **No transposed/strided operands.**
- **Aligned dims (the template's SIZE CONTRACT):** `M % (8·rr) == 0` (rr=4 → M%32==0),
  `N % (32·rc) == 0` (rc=4 → N%128==0), `K % 8 == 0`. The template does **no boundary
  masking** on `simdgroup_store`; ragged dims must fall back.
- **Input dtype:** fp16 **and** fp32 — the fast template's `dtype` branch handles both
  (`make_simdgroup_matmul_kernel_fast(dtype="fp16"|"fp32")`, benchmarked 2026-06-15: fp32
  11.2 TFLOP/s = 98% of torch, fp16 7.8 TFLOP/s = 77%, both relerr 0.0). bf16 falls back.
- **Output dtype must be fp32.** The template **always** declares `device float* C`
  (`simdgroup_store` of a `simdgroup_float8x8` accumulator writes float). So the bound
  output tensor must be float32: this covers fp32→fp32 (98%) and fp16-input→fp32-output
  (77%). A fp16-**output** matmul (`acc.to(fp16)` into a `half` C) would bind a half
  tensor to a float\* buffer → silent-wrong, so it **falls back** in v1. (A fp16-output
  fast-template variant — cast `float8x8`→`half` on store — is a documented follow-up,
  not v1; the output-dtype check keeps the gate airtight meanwhile.)
- **M, N, K available as runtime scalar args** (so the launcher can read them to compute
  `n_groups`). If M/N are compile-time constants only, fall back (or const-fold — plan
  decides; default fall back).

Anything not provably meeting **all** of the above → generic lowering, unchanged.

## Data flow

Triton compile → dot-lowering: eligible? → **yes:** emit fast template (renamed) + set
`fast_matmul` metadata → launcher computes `n_groups` from runtime M/N → dispatch 1-D
(via compile_shader zero-copy or host-round-trip) → result in C. **no:** generic dot
lowering (current staged kernel) → user's 2-D grid → unchanged.

## Error handling / integrity (prime directive: never silent-wrong)

- The fast path must produce results **identical to tolerance** to the generic lowering
  for every eligible kernel — gated by the full-suite run below before default-on.
- Eligibility is fail-closed: any uncertainty (unprovable strides, ragged dims, unexpected
  ops, missing runtime M/N) → fall back. The excluded set is fully covered by the generic
  lowering.
- Flag `TRITON_METAL_FAST_MATMUL` (default-on once the gate is green; `=0` escape hatch),
  mirroring `TRITON_METAL_MEPT` / `TRITON_METAL_COMPILE_SHADER` — a regression can be
  bisected/disabled without a code change.
- `CODEGEN_VERSION` bumped (cache key) since emitted MSL changes for eligible kernels.

## Testing / validation (correctness FIRST, then perf)

1. **Full-suite parity (the gate):** upstream `test_core` dot/matmul families + the full
   ratchet, **both MEPT flags**, with `TRITON_METAL_FAST_MATMUL` on == off. 0 failed,
   identical to tolerance. No perf claim before this is green.
2. **Parity harness:** a representative matmul set (fp32→fp32 and fp16→fp32, aligned
   2048³/1024²/512² and non-square; plus the MUST-fall-back set: ragged dims, transposed/
   strided operands, bf16, fp16-output) run through both the fast path and the generic
   lowering, asserting identical-to-tolerance outputs and that the ineligible cases take
   the fallback (assert kernel name / metadata).
3. **Real kernels:** any project matmul-bearing kernels (and FlashAttention, which has its
   own path) stay green.
4. **Perf gate (after correctness):** re-bench fast vs generic vs torch at 2048³ for both
   dtypes; assert the eligible class is materially above the generic floor (fp32 target
   ≳ 90% of torch ≈ 11 TFLOP/s; fp16 ≳ 70% ≈ 7.5 TFLOP/s; both ≳ 2× the generic ~2.8).
   Record ON/OFF, both dtypes, in `reports/perf_baseline.json`.
5. **Fallback paths:** ragged/transposed/bf16/fused-epilogue matmuls run correctly via the
   generic lowering; flag-off path identical to pre-change.

## Open items the plan resolves

- Exact detector location + how the clean-matmul + contiguity pattern is proven from the
  TTGIR (addptr stride analysis vs requiring explicit stride args == defaults).
- Renaming mechanism for the substituted MSL entry point (regex on `kernel void
  simdgroup_matmul_fast` → user name; confirm `self.kernel_name`/`_MSL_BY_NAME` threading
  stays consistent for both driver paths).
- `fast_matmul` metadata plumbing (where `needs_2d_grid`/`output_indices` are set in
  `compiler.py` / `msl_emitter.py`) and the launcher read.
- How the output-dtype check is established (the C tensor's dtype + the IR store: direct
  `acc` store → float C eligible; `acc.to(fp16)` store → half C, fall back). Both fp16 and
  fp32 inputs are confirmed supported + benchmarked (2026-06-15); no open question on input
  dtype remains.
- `rr/rc` tile choice (currently 4/4 → 32×128); confirm 7.8 TFLOP/s is the best static
  choice or whether a second tile (e.g. 8×4) helps tall/skinny shapes (follow-up, not v1).

## Out of scope

- Rewriting the generic lowering body (Design Y) — deferred; the fallback stays as-is.
- Further fast-template tuning (rr/rc sweep, double-buffered K-loop on top of direct loads,
  bank-conflict layout) to push past 77% of torch — a separate follow-up; this design's
  lever is the routing (2.8 → 7.8), not template micro-tuning.
- Batched matmul / `tt.dot` with >2 operands / int8 matmul (own path) / FlashAttention
  (own path) — unchanged.
