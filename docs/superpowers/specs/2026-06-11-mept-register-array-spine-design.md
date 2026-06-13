# MEPT register-array spine — design (2026-06-11)

> Make per-thread multi-element values a first-class register-array model in the
> Python/MSL lowerer, so per-element state carries across data-dependent control
> flow. Unblocks tridec BLOCK>=256 (Bug 2), the >1024 threadgroup ceiling,
> FlashAttention HEAD_DIM=64, and chained reductions. Python/MSL per the
> 2026-06-11 language/IR-target decision.

## Decisions (locked in brainstorm)
- **Approach B — unified register-array value model.** Every SSA value is
  `(name, n_elems, ty)`; scalars are `n_elems==1`. Eliminates the `env` (scalar)
  vs `env_array` scoping mismatch that causes Bug 2.
- **Emission picks the cheapest correct form** per value/region:
  - `n_elems==1` -> plain scalar, no loop (byte-identical to today's scalar path).
  - `n_elems>1`, straight-line region -> the existing `_loop_e` wrap-loop
    (constant registers, re-execution) — kept exactly where it wins.
  - `n_elems>1` AND data-dependent control flow in scope -> true register arrays
    `T v[n_elems]` (the only correct form; B's contribution).
- **Rollout:** build behind `TRITON_METAL_MEPT` (off by default). Drive flag-ON to
  5,335/0 + the newly-unlocked tests across the whole corpus, THEN flip default.
  The 5,335/0 scalar path is never at risk until B is proven. Reversible.
- **Out of scope this cycle:** retiring the perf detectors (matmul MMA at MLX
  parity, matmul->softmax, etc.) — they stay as fast-paths. The spine is the
  general *correctness* path; it only retires detectors that exist to patch MEPT
  gaps (e.g. `_detect_permute_chained_reduce`) once the spine subsumes them, as a
  follow-up.

## Architecture
Single value map keyed by SSA id -> `RegVal(name, n_elems, ty, form)` where
`form in {scalar, wraploop, array}`. `_lookup` returns a `RegVal`; a shim keeps
the legacy scalar string API working during migration. The emitter has one
per-op lowering that, given operand `RegVal`s, emits the chosen form.

**Form selection** (per connected region): a region is classified
`needs_array` if it contains a data-dependent `scf.for`/`scf.while`/`scf.if`
whose body references or carries a multi-element value. `needs_array` -> all
multi-element values in that region are `array`; else straight-line multi-element
values are `wraploop`; `n_elems==1` is always `scalar`.

## Components (files)
- `triton_metal/codegen/regval.py` (new): `RegVal` + form-selection classifier
  (`classify_region(ops) -> form`), pure, unit-testable without a GPU.
- `generic_lowerer.py`: unify `env`/`env_array` reads through `_lookup` ->
  `RegVal`; route every `_emit_*` through a single `emit_form(regval, body_fn)`
  helper that materializes scalar / wrap-loop / array. Reuse existing
  `_var_array`, `env_n_elems`, `_track_n_elems`.
- `_lowerer_control.py`: `_lower_scf_for`/`_lower_scf_while`/`scf.if` carry
  `array`-form values as iter-args/results (declare `T v[n]` before the loop;
  the loop body reads/writes `v[e]`; yields update `v`). This is the Bug-2 fix:
  hoisted arrays declared before the loop persist into the body naturally.
- `_lowerer_reduce.py`: reductions fold the per-element array (`for e: acc op= v[e]`)
  then the existing SIMD/threadgroup cross-thread reduce. dot/convert_layout
  already consume arrays via the MMA/shuffle paths — extend to the array form.

## Data flow (the Bug-2 case, post-spine)
```
offs: int[spt]                              # hoisted arange, array form
for (k=0; k<n_tiles; k++) {                 # data-dependent scf.for (scalar k)
    float v[spt];
    for (e<spt) v[e] = mask ? X[k*BLOCK + offs[e]] : 0.0f;
    for (e<spt) partial += v[e];            # per-element fold
}
total = simd_reduce(partial);               # cross-thread
```
`offs[]` is declared once before the loop and visible inside it — no scope
mismatch, no `UNKNOWN_`.

## Error handling / integrity
- The `UNKNOWN_<id>` backstop (already shipped) stays: if any path still emits an
  unresolved value, refuse loudly rather than emit invalid MSL.
- Register budget guard: if `n_elems * live_arrays` exceeds a configurable
  threshold (spill risk), refuse with an actionable message (or fall back to the
  wrap-loop where the region allows) rather than silently spill.
- Flag-OFF path is unchanged and remains the integrity reference (differential:
  flag-ON output must match flag-OFF on all currently-passing kernels).

## Testing / ratchet
- Unit (no GPU): `classify_region` form selection; `RegVal` lookup unification;
  scf.for array-carry MSL shape (text assertions via `emit_msl`).
- Correctness (GPU, serial): the tridec Bug-2 kernel at BLOCK 256/512/1024 ->
  correct (currently refuses); FA HEAD_DIM=64; a chained-reduction case.
- Ratchet: every commit, fresh-cache `test_core` flag-OFF stays 5,335/0; flag-ON
  must monotonically climb toward 5,335 + newly-unlocked, never regress.
- Flip gate: flag-ON full `test_core` >= 5,335/0 AND the newly-unlocked tests pass
  AND project suite green -> flip `TRITON_METAL_MEPT` default to on.

## Milestones (each its own plan)
1. `RegVal` unification + form classifier + scalar-collapse parity (flag-ON ==
   flag-OFF on the scalar corpus; no new passes yet).
2. `scf.for`/`while` array-carry -> Bug-2 BLOCK>=256 correct. **DONE (M2,
   commits 41e5b39->e7f554e): eligibility extended so control-flow kernels
   enter the single-pass register-array form (`mept_arrayform_eligible`);
   hoisted values persist into the loop body via `env_array`. tridec Bug 2
   computes at BLOCK 256/512/1024 (flag-on); flag-off upstream test_core
   5,335/0 unchanged. GPU validation surfaced + fixed two array-path gaps
   (shape-preserving reshape-before-reduce in the eligibility set; replicated-
   layout exact-cover via `block_size // num_threads`). Array iter-arg carry
   (per-element state yielded across iterations) deferred to M3, where chained
   reductions require it — Bug 2's dataflow carries only a scalar partial.**
3. Cooperative ops (reduce/dot/convert_layout) on the array form -> >1024 ceiling
   + chained reductions.
4. FlashAttention HEAD_DIM=64 on the spine.
5. Flip default; then (separate) retire MEPT-gap detectors.

## Risks
- Common-path bloat if scalar-collapse regresses -> the differential gate catches
  it (flag-ON must equal flag-OFF on scalar kernels, byte-for-byte where feasible).
- Register pressure at large `spt` -> the budget guard + wrap-loop fallback.
- Migration churn across ~46 `env[]=` sites -> milestone 1 is exactly this, gated
  by the parity differential before any new behavior lands.
