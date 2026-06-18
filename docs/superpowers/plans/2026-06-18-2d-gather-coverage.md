# 2D `tt.gather` coverage (incremental op coverage / roadmap #4 pick)

> **Executable plan (2026-06-18).** Self-contained. The genuine `test_core` coverage gaps are
> all real features (well-maintained skip-list); 2D gather is the highest-value pick (gather/
> embedding-style lookups are common). 1D gather already works; this adds 2D.

**Goal:** make `tt.gather` lower correctly for 2D `src`/`indices` (axis 0 and 1), turning the
4 conftest-skipped cases green:
`test_gather[src_shape0-indices_shape0-0]`, `[…1-0]`, `[…2-0]`, `[src_shape3-…-1]`
(axis is the trailing index in the test id). Never silent-wrong: refuse loudly if the src
tile exceeds threadgroup budget or the 2D thread mapping can't be resolved.

## Semantics
`out[i, j] = src[index[i, j], j]` for **axis=0**; `out[i, j] = src[i, index[i, j]]` for
**axis=1**. `index` and `out` share shape; `src` may differ along the gather axis.

## Current state to build on
- `triton_metal/codegen/generic_lowerer.py:5433` `_lower_tt_gather` handles **1D**: stage
  `src` (size S) to a `threadgroup` array (`if (lid < S) shared[lid] = src_var;` + barrier),
  then each thread reads `shared[(uint)idx_var]`. One-element-per-thread.
- 2D is refused via `scripts/conftest_metal.py` (`test_gather` skip when
  `len(src_shape) > 1 or len(indices_shape) > 1`).
- 2D thread/position semantics already exist (the lowerer maps a `tt.make_range` on dim 0 →
  row, dim 1 → col; see `_make_range_stride_below` and the 2D indexing in `_lower_load`/
  `_lower_store`). The gather output element's `(row, col)` comes from the same machinery.

## Staged plan

### Task 1 — axis=0, smallest case, test-first
- Add a project test (`tests/test_gather_2d.py`) that builds a small `@triton.jit` gather
  kernel (src `[Ms, N]`, index/out `[Mi, N]`, axis=0) and compares to `torch.gather(src, 0,
  index)` on MPS, tight atol. Start with the smallest skipped shape.
- Extend `_lower_tt_gather`: detect 2D (`len(src_shape)==2`). Stage the full 2D `src` to a
  `threadgroup` array of size `Ms*N` (row-major), barrier. For each output element at
  `(row, col)` (from the 2D thread position), compute the flat source index
  `gathered = idx_value * N + col` (axis=0) and read `shared[gathered]`. The `idx_value` is
  the gather index tensor's per-element value (already an env var, possibly MEPT array).
- **Budget guard:** if `Ms*N*elem_size > 32KB`, raise `MetalNonRecoverableError` (don't stage
  an oversized tile — refuse, don't silent-wrong). The skipped test shapes are small; large
  2D gather stays refused as future work.
- Verify the axis=0 test passes on GPU; commit.

### Task 2 — axis=1
- Same staging; for axis=1 the flat source index is `row * N_src + idx_value`. Add the axis=1
  test case (`src_shape3`, axis=1) + the per-axis index formula (read the gather `axis`
  attribute from the `tt.gather` op). Verify; commit.

### Task 3 — remaining skipped 2D shapes + MEPT interplay
- Cover `src_shape1`/`src_shape2`. Confirm correct interaction with the MEPT register-array
  path (if `out` is MEPT-arrayed, the gather must read per array element); if that path isn't
  covered, refuse rather than mis-emit. Verify each.

### Task 4 — un-skip + verify + docs
- In `scripts/conftest_metal.py`, narrow the `test_gather` 2D skip so the now-supported cases
  run (keep refusing any that exceed budget). 
- Verify: project suite green, the 4 `test_gather` 2D cases pass via the skip-aware ratchet
  (`run_upstream_tests.py`), `test_core` count rises by the reclaimed cases, 0 failed.
- Bump `CODEGEN_VERSION`. Update `docs/SUPPORTED_OPS.md` (gather row: 1D + 2D).

## Risks
- Index bounds: a `index` value ≥ src extent is OOB. Match torch's behavior (it's UB/clamped);
  for safety, the kernel/test should use valid indices, and the lowering should not assume
  in-bounds beyond what the test provides — if masking is needed, refuse rather than guess.
- 2D thread mapping must be the SAME one `_lower_load`/`_lower_store` use, or row/col will be
  transposed — verify against torch on a non-square shape.
- Keep one-clear-purpose: this touches only `_lower_tt_gather` + the conftest skip + tests.
