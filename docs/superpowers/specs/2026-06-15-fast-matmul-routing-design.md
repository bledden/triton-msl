# Fast matmul routing — design (2026-06-15)

> Route eligible clean matmuls to the existing, proven `make_simdgroup_matmul_kernel_fast`
> template (direct device `simdgroup_load`/`simdgroup_store`, register blocking, no
> threadgroup staging, no serial epilogue) instead of the generic dot lowering's
> staged kernel (34 barriers, serial fragment-by-fragment masked epilogue). **Measured
> at 2048³, relerr 0.0 (exact) both dtypes: fp32 11.2 TFLOP/s = 98% of `torch.matmul`;
> fp16 7.8 TFLOP/s = 77% — vs the generic lowering's ~2.8 TFLOP/s (37%).** Phase 4 (perf), the
> matmul lever after compile_shader. Purely additive: ineligible matmuls run the generic
> lowering unchanged. Prime directive: never silent-wrong.

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

This is the diagnostic signature observed in profiling (larger tiles make it *slower*):
the kernel is barrier/threadgroup-memory bound, not MMA bound. The MMA path *is* firing
(`simdgroup_float8x8` accumulators, `simdgroup_half8x8` fragments) — it is a
template-**quality** problem, not a missed detector. The generic kernel itself
(`make_matmul_kernel`, `_msl_templates.py:448`) is **row-major contiguous** (indexes
`A[gr*K+gc]`, `B[gr*N+gc]`, `C[gr*N+gc]`, ignores stride args), uses a 1-D flattened `pid`
grid, bounds-checks edges (handles ragged M/N), and reads runtime `M/N/K` from scalar args
at buffers 3/4/5 (A/B/C at 0/1/2).

The proven asset: `make_simdgroup_matmul_kernel_fast` (`_msl_templates.py:3547`) — direct
device load/store, register blocking (rr=rc=4 → 32×128 tile), zero staging, zero epilogue
barriers — but **unwired** (no caller in `triton_metal/`). It assumes the **same** row-major
layout and the **same** A/B/C@0-2, M/N/K@3-5 arg positions as `make_matmul_kernel`, so it is
correct on exactly the inputs the generic kernel is correct on — *provided* the runtime dims
are aligned (it has **no** boundary handling; see contract).

## Approach: runtime dispatch via the compile_shader path (not compile-time substitution)

The fast template requires aligned dims and has no edge handling. But the matmul's `M/N/K`
are **runtime** scalar args (dynamic shapes) — alignment is **not knowable at compile time**.
A compile-time whole-kernel substitution would bake the fast template into the IR-keyed
cache and then run it for *any* later shape, including ragged ones → out-of-bounds writes /
silent-wrong. (Verified: at M=2050 the fast template's grid rounds M up to 2080 and writes
30 rows past C's end — relerr on the valid region read as 0.0, masking a heap overflow.
This is exactly the trap the runtime gate prevents.)

So we keep the compiled metallib as the **generic kernel (always correct, always the
fallback)** and dispatch the fast template **at runtime** through the existing
`compile_shader` zero-copy path — the infrastructure built in the prior Phase-4 work —
only when the actual runtime tensors/dims satisfy the contract. This is the same lever
("route eligible matmuls to the fast template, fall back otherwise"), moved to runtime
because correctness requires it. It needs no MSL substitution, no kernel rename, no metallib
changes, and no host-path grid override; it is proven feasible (the 7.8 / 11.2 TFLOP/s
benchmarks *were* this `compile_shader`-dispatched fast template).

Rejected alternatives:
- **Compile-time whole-kernel substitution + launcher grid override** — unsafe for
  dynamic shapes (above); committed at compile time but alignment is a runtime fact.
- **Rewrite the generic lowering body** (direct loads/store parameterized by user `BM/BN`)
  — new parameterized MMA codegen, higher silent-wrong surface than reusing a proven
  fixed template. Deferred.
- **Swap the ttgir_parser prebuilt `simdgroup_matmul` → `_fast`** — that path is the
  legacy opt-in fallback (default-refuses); standard matmuls don't route there.

## Architecture

Two halves: a compile-time **detector** (additive metadata only — does not change the
emitted/compiled generic kernel) and a runtime **dispatch gate** in the launcher.

1. **Compile-time eligibility detector** (`GenericLowerer`, in the
   `_lower_dot_simple_template` path — the exact path eligible kernels already take). When
   the structural contract holds, it builds the fast-template MSL via
   `make_simdgroup_matmul_kernel_fast(dtype=<input_dtype>, rr=4, rc=4)` and records a
   **`fast_matmul` descriptor**; the generic kernel MSL is still emitted and returned
   unchanged. Descriptor (all cacheable scalars/str so it survives the `.meta.json`
   disk cache): `(fast_msl, m_idx=3, n_idx=4, k_idx=5, tile_m=32, tile_n=128)`. The fast
   template's entry name stays `simdgroup_matmul_fast` (compile_shader dispatches by
   explicit name — no rename needed).

2. **Metadata plumbing** (mirrors `mm_two_kernel`): `emit_msl` reads
   `metadata["fast_matmul"] = getattr(lowerer, "_fast_matmul", None)`; `pack_metadata`
   appends it to the launcher tuple at **index 7** (after `mm_two_kernel` at 6).

3. **Runtime dispatch gate** (`driver.py` `MetalLauncher.__call__`, inside the existing
   `compile_shader` try-block, as a branch *before* the elementwise 1-D-grid branch since
   the fast path computes its own grid). Read `fast_matmul = kernel_metadata[7] if
   len(kernel_metadata) > 7 else None`. Fire **only** when: `fast_matmul` is present; the
   runtime is available and the fast MSL isn't marked unsupported; every tensor karg is
   MPS; and the runtime dims pass the contract `M = int(kargs[m_idx]); N = int(kargs[n_idx]);
   K = int(kargs[k_idx])` with `M%32==0 and N%32==0 and K%8==0`. Then
   `n_groups = ceil(M/tile_m) * ceil(N/tile_n)`, `lib = rt.get_library(fast_msl)`,
   `rt.dispatch(lib, "simdgroup_matmul_fast", kargs, threads=n_groups*128, group_size=128)`,
   return. Any miss (no descriptor / non-MPS / misaligned / disk-cache-cold so no
   descriptor / exception) → fall through to the existing path (generic metallib, correct).

The generic metallib is **always** the compiled artifact and the fallback; the fast path
only ever *adds* a faster route for MPS tensors with aligned dims.

## Eligibility contract

Split into a compile-time structural gate (decides whether a `fast_matmul` descriptor is
emitted at all) and a runtime gate (decides whether to use it this launch). Both fail
closed; the generic kernel covers everything excluded.

**Compile-time structural (detector returns a descriptor only if ALL hold):**
- The kernel reaches `_lower_dot_simple_template` — i.e. a single `tt.dot`, K-loop
  accumulate, pid-tiled, **not** a 3-D/batched dot, no fused epilogue (matmul+softmax and
  matmul+pointwise-epilogue are detected earlier in `lower()` and take other paths). This
  is the same path whose generic kernel already assumes row-major A/B/C and A/B/C@0-2,
  M/N/K@3-5 — so the fast template inherits an identical layout assumption (no new risk).
- Exactly 3 pointer args (A, B, C) occupying buffers 0/1/2, and ≥3 scalar args; `m_idx=3,
  n_idx=4, k_idx=5` (the positions `make_matmul_kernel` already binds M/N/K to). If the
  layout differs, no descriptor (and the generic kernel would already be wrong — we are
  never worse than it).
- **Input dtype** fp16 or fp32 (`make_simdgroup_matmul_kernel_fast` supports both;
  benchmarked). bf16 → no descriptor.
- **Output dtype fp32.** The fast template always declares `device float* C`. A fp16-output
  matmul (`acc.to(fp16)` into a `half` C) must not use it → no descriptor (fp16-output fast
  variant is a follow-up). fp32→fp32 (98%) and fp16-in→fp32-out (77%) both qualify.

**Runtime (gate, every launch — this is what makes substitution safe):**
- Every tensor karg is an MPS tensor (the compile_shader path requires it).
- `M%32==0 and N%32==0 and K%8==0`, read from the runtime `kargs`. Empirically pinned
  2026-06-15: N%32 (not N%128) is sufficient (partial 128-col tiles handled); but M%32 is
  **mandatory** (M%32≠0 → grid rounds M up → OOB store past C; the M=2050 case looked
  correct by relerr yet overflowed the heap). K%8≠0 and N%32≠0 give wrong results.
- Else → generic metallib (correct, bounds-checked, any shape).

## Data flow

Compile: dot-lowering reaches `_lower_dot_simple_template` → structural gate passes? →
build fast MSL + set `_fast_matmul` descriptor (generic kernel still emitted/compiled) →
`emit_msl` → `pack_metadata` tuple[7]. Launch: launcher reads `fast_matmul`; MPS tensors +
runtime M/N/K aligned? → compile_shader the fast template (cached) + dispatch `n_groups`
1-D grid → result in C. Otherwise → generic metallib (existing path). Flag off, non-MPS,
ragged, or fp16-output → generic, unchanged.

## Error handling / integrity (prime directive: never silent-wrong)

- The compiled artifact never changes; on any doubt the generic kernel runs. The fast path
  is a runtime *acceleration*, gated on the actual tensors/dims.
- Whole runtime attempt is inside the existing `compile_shader` try/except → any exception
  marks the fast MSL unsupported + falls through to the generic path.
- **relerr is not sufficient to validate** the fast path (an OOB write can read as relerr
  0.0). The gate logic (does it fall back on M%32≠0 / N%32≠0 / K%8≠0 / fp16-output /
  non-MPS?) is tested directly, separately from numeric parity.
- Flag `TRITON_METAL_FAST_MATMUL` (default-on once the gate is green; `=0` escape hatch),
  mirroring `TRITON_METAL_COMPILE_SHADER`. Disables both halves (no descriptor emitted /
  runtime branch skipped) so a regression bisects without a code change.
- `CODEGEN_VERSION` bumped (`2026.06.13.2` → next) since metadata/emitted descriptors change.

## Testing / validation (correctness FIRST, then perf)

1. **Full-suite parity (the gate):** upstream `test_core` dot/matmul families + the full
   ratchet, **both MEPT flags**, with `TRITON_METAL_FAST_MATMUL` on == off. 0 failed,
   identical to tolerance. No perf claim before this is green.
2. **Numeric parity harness:** eligible matmuls (fp32→fp32 and fp16→fp32; aligned square
   2048³/1024²/512² and non-square incl. N%32-but-not-128, K a non-128 multiple of 8) run
   through the fast path and the generic kernel — assert identical to tolerance.
3. **Gate-logic tests (NOT relerr — fall-back behavior):** for each MUST-fall-back case —
   M%32≠0, N%32≠0, K%8≠0, fp16-output, bf16, non-MPS tensors, transposed/strided non-pid
   dot (strided generation path), 3-D/batched dot, flag off — assert the generic path runs
   (no `fast_matmul` descriptor emitted, or the runtime branch not taken; observe via a
   dispatch counter / hook), so misalignment can never reach the fast template.
4. **Real kernels:** project matmul-bearing kernels + FlashAttention (own path) stay green.
5. **Perf gate (after correctness):** re-bench fast vs generic vs torch at 2048³ both
   dtypes; assert eligible class materially above the generic floor (fp32 ≳ 90% of torch
   ≈ 11 TFLOP/s; fp16 ≳ 70% ≈ 7.5; both ≳ 2× the generic ~2.8). Record ON/OFF, both
   dtypes, in `reports/perf_baseline.json`.

## Open items the plan resolves

- Exact site within `_lower_dot_simple_template` to run the structural gate + build the
  descriptor (after `out_dtype` is computed; before/after the existing emit).
- How the structural gate proves "reaches `_lower_dot_simple_template`" cleanly (run it
  inside that method so it's by construction, vs a separate predicate).
- `fast_matmul` descriptor exact shape as a cacheable tuple and its `.meta.json`
  round-trip (str + ints); confirm `getattr(metadata, "fast_matmul", None)` in
  `pack_metadata` mirrors `mm_two_kernel`.
- Launcher branch ordering vs the elementwise compile_shader branch and the `mark_unsupported`
  key (the fast MSL string, distinct from the generic `self._msl`).
- Disk-cache-cold behavior: a process that loads a cached metallib without re-running
  `make_msl` — does `fast_matmul` survive via `.meta.json`/`pack_metadata` (preferred) so
  the fast path still fires, or fall back like the `_MSL_BY_NAME` limitation? (Default:
  carry it in the cached metadata tuple so it survives.)

## Out of scope

- Compile-time substitution / rewriting the generic lowering body — rejected/deferred above.
- fp16-**output** fast variant (cast `float8x8`→`half` on store) — follow-up.
- Accelerating **CPU-tensor** matmuls — the fast path is MPS-only (the compile_shader
  route); CPU matmuls stay on the generic kernel (rare, already slow). Acceptable.
- Further fast-template tuning (rr/rc sweep, K-loop double-buffer on top of direct loads,
  bank-conflict layout) past 77%/98% of torch — separate follow-up.
- Batched/3-D dot, int8 matmul (own path), FlashAttention (own path) — unchanged.
