# fp16-output fast matmul — design (2026-06-15)

> Extend the shipped fast-matmul routing to **fp16 output** (`half* C`), so the common
> transformer matmul (fp16-in → fp16-out) gets the zero-copy fast path instead of falling
> back to the copy-bound generic kernel. The fast template accumulates in
> `simdgroup_float8x8` and currently stores to `float* C`; fp16 output needs a
> down-convert on store. **Empirically confirmed (2026-06-15): a per-block cast epilogue
> (store float fragment → threadgroup scratch → cast→half → write `half* C`) is correct
> (relerr 0.0 at 512³/2048³/non-square) and FAST (11.9 TFLOP/s fp16 2048³).** Float
> accumulation is preserved for precision. Phase 4 follow-up; flag-gated under the same
> `TRITON_METAL_FAST_MATMUL`; never silent-wrong.

## Problem

The shipped feature (`docs/.../2026-06-15-fast-matmul-routing-design.md`) routes eligible
matmuls to `make_simdgroup_matmul_kernel_fast` via `compile_shader`, but the
compile-time detector's output-dtype gate **requires fp32 output** — because the template
declares `device float* C` and stores accumulators with `simdgroup_store(float8x8,
float*, N)`. A fp16-output matmul (`acc.to(fp16)` into a `half` C tensor — the dominant
transformer pattern) is excluded and falls back to the generic, copy-bound kernel
(~2.8 TFLOP/s effective).

Metal cannot down-convert directly: both `simdgroup_store(float8x8, half*, …)` and a
`simdgroup_half8x8(float8x8)` cast fail to compile (verified). The result fragment's
values must be read out (via `simdgroup_store` to threadgroup memory) then cast to half
per element.

## Approach (confirmed by probe)

**Extend `make_simdgroup_matmul_kernel_fast` with an `out_dtype` parameter** (default
`"fp32"`, preserving today's behavior byte-for-byte). When `out_dtype="fp16"`:
- Declare `device half* C [[buffer(2)]]` (instead of `float*`).
- Add `uint tiisg [[thread_index_in_simdgroup]]` to the signature and a
  `threadgroup float scratch[4 * 64]` (one 8×8 block per simdgroup; 4 simdgroups).
- Replace the direct store with a **cast epilogue** per accumulator block `c{r}_{c}`:
  `simdgroup_store(c{r}_{c}, scratch + sgitg*64u, 8);` → `simdgroup_barrier(threadgroup)`
  → each lane writes `C[(row_base+{r*8}+i/8)*N + col0 + {c*8} + i%8] = half(scratch[
  sgitg*64 + i])` for `i = tiisg; i < 64; i += 32` → `simdgroup_barrier(threadgroup)`.
- Everything else — the float accumulators, direct device `simdgroup_load` of A/B, the
  register-blocked K-loop, the `if (col0 >= N) return` partial-column guard — is
  identical to the fp32-out path.

Rejected: (B) a separate `_f16out` function (duplicates the entire K-loop — not DRY);
(C) leaving fp16-out on the generic fallback (misses the common case).

## Architecture (what changes, what doesn't)

1. **Template** (`_msl_templates.py:make_simdgroup_matmul_kernel_fast`): new `out_dtype`
   kwarg. `out_dtype="fp32"` → existing code path, unchanged. `out_dtype="fp16"` → `half*
   C` + `tiisg`/`scratch` + cast epilogue. Entry name stays `simdgroup_matmul_fast`.
   **Dispatch contract is identical** for both variants (same `n_groups = ceil(M/(8*rr)) *
   ceil(N/(32*rc))`, 128 threads, same `M%32 / N%32 / K%8` alignment).

2. **Detector** (`_lowerer_templates.py:_maybe_fast_matmul_descriptor`): the output-dtype
   gate today is `out_dtype in ("fp32","f32","float") → else None`. Extend: also accept
   `out_dtype in ("fp16","f16")`, and pass it through to
   `make_simdgroup_matmul_kernel_fast(dtype=<input>, rr, rc, out_dtype=<fp32|fp16>)`. Input
   dtype gate unchanged (fp16/fp32). bf16 output still falls back. The descriptor tuple
   shape `(fast_msl, 3, 4, 5, 32, 128)` is unchanged — only the embedded MSL differs.

3. **Launcher** (`driver.py`): **no change.** It dispatches `fast_msl` by the same name
   with the same contract; `compile_shader` binds the (half) C tensor to the `half*`
   buffer zero-copy. The variant is fixed at compile time from the IR's output dtype (not
   runtime-variable), so no runtime dtype check is needed — consistent with the generic
   kernel.

## Data flow

Compile: detector sees a clean matmul with fp16/fp32 input and **fp16 output** →
generates the fp16-out template variant → records the (unchanged-shape) descriptor →
metadata index 7. Launch (existing path): all-MPS + `M%32/N%32/K%8` aligned → compile the
fp16-out template via compile_shader → dispatch `n_groups` grid → `half` result in C.
Misaligned / non-MPS / bf16-out / flag-off → generic kernel, unchanged.

## Error handling / integrity (never silent-wrong)

- **fp32-out path provably untouched** — the `out_dtype="fp32"` branch emits the identical
  string as today (the new code is gated behind `out_dtype=="fp16"`). A template unit test
  asserts the fp32-out MSL is byte-identical to the pre-change output.
- Float accumulation preserved (precision parity with the generic kernel, which also
  accumulates in float then casts to half on store).
- Same runtime alignment gate (the epilogue writes whole 8×8 blocks with no per-element
  bounds check beyond the `col0 >= N` simdgroup guard, so `M%32 / N%32 / K%8` remain
  mandatory — identical to the fp32-out variant).
- Flag-gated under the existing `TRITON_METAL_FAST_MATMUL` (one flag, both layers).
- `CODEGEN_VERSION` bump (emitted MSL changes for fp16-output matmuls).

## Testing / validation (correctness FIRST, then perf)

1. **Template unit test:** `make_simdgroup_matmul_kernel_fast(dtype="fp16", out_dtype="fp32")`
   == today's output (byte-identical, no regression to the fp32-out path); the
   `out_dtype="fp16"` variant contains `half* C` + the cast epilogue.
2. **Detector test (extend `test_fast_matmul_detect.py`):** an fp16-in→fp16-out matmul now
   emits a descriptor (previously asserted NO descriptor in `test_fp16_output_no_descriptor`
   — that test is updated to assert the descriptor IS now emitted and its MSL is the
   half-output variant). fp32-out still emits its variant; bf16-out still emits none.
3. **Numeric parity (extend `test_fast_matmul_parity.py`):** add fp16-in→fp16-out across
   aligned square + non-square (incl. N%32-not-128); assert fast == torch (fp16 tol) AND
   fast == generic (flag on==off, seeded).
4. **Gate-logic (extend `test_fast_matmul_gate.py`):** an aligned fp16-out MPS matmul now
   DISPATCHES `simdgroup_matmul_fast` (previously fell back); misaligned fp16-out still
   falls back. Via the dispatch spy.
5. **Full ratchet (the gate):** real `--device cpu` test_core dot/matmul subset,
   `TRITON_METAL_FAST_MATMUL` on == off, identical pass/fail — fp16-output matmuls in the
   suite must not regress. (Run via `scripts/run_upstream_tests.py` semantics; see
   [[reference_upstream_ratchet]] — raw pytest hits CUDA asserts.)
6. **Perf (after correctness):** fp16-in→fp16-out 2048³ ≥ ~7 TFLOP/s (≥2× generic ~2.8;
   probe showed 11.9). Record `matmul_2048_fp16out` in `reports/perf_baseline.json`.

## Open items the plan resolves

- Exact epilogue barrier count / scratch layout (per-block reuse with 2 barriers/block —
  16 blocks — vs one larger scratch + single barrier). The probe used per-block (2/block)
  and hit 11.9 TFLOP/s, so per-block is acceptable; the plan keeps it unless a one-barrier
  variant is trivially cleaner.
- Whether `out_dtype` defaults keep all existing callers (e.g. ttgir_parser, tests) at
  fp32-out behavior (yes — default `"fp32"`); confirm no other caller passes a C dtype.
- The `_maybe_fast_matmul_descriptor` change reads the C dtype from `args[2].elem_type`
  (already does, for the fp32 check) — extend the branch, no new IR analysis.

## Out of scope

- **bf16 output** — its common case (bf16-in→bf16-out) needs bf16 *input* fragments the
  fast template lacks (`simdgroup_bfloat8x8` unused); only rare combos would benefit. Deferred.
- Further epilogue micro-tuning, rr/rc sweep, unify-with-#159 — separate follow-ups.
