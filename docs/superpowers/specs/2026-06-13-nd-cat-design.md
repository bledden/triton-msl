# Generic N-D concat (tt.cat rank≥2) — design (2026-06-13)

> Close `test_cat_nd` (~10 cases) with a detector + closed-form direct-copy
> template for `load(N-D) ×2 → tt.cat(dim) → store`, mirroring `_detect_nd_trans`.
> Phase 3 feature 4 (cat_nd only; `join_with_mma` stays refused). Python/MSL.

## Problem
`_lower_tt_cat`/`_lower_tt_join` handle only 1-D concat. Rank≥2 `tt.cat`/`tt.join`
refuse via the catalog check `_check_nd_cat_join` (`refusal_catalog.py:125`).
`test_cat_nd`: `load(N-D via descriptor) ×2 → tl.cat(x,y,dim) → store(N-D)`, same
shape as `test_trans_4d`. (Upstream params: small N-D shapes, various `dim`.)

## Approach — detector + closed-form direct copy (mirrors N-D transpose)
`tl.cat(x, y, dim)` doubles the output along `dim`: `out[O] = x[O]` if
`O[dim] < D` else `y[O with O[dim]-=D]`, where `D = src_shape[dim]`, both inputs
share `src_shape`, output `dst_shape = src_shape` with `dst_shape[dim] = 2*D`.
Emit a strided direct copy (no shared mem, no barrier):

```
for (uint k = lid; k < TOTAL_OUT; k += BLOCK) {
    // O = unflatten(k, dst_shape) (row-major); pick input by O[dim]
    uint od = (k / dst_stride[dim]) % dst_shape[dim];
    uint in_flat = Σ_d ( (d==dim ? (od % D) : O[d]) ) * src_stride[d];
    out[k] = (od < D) ? x[in_flat] : y[in_flat];   // od%D handles the y-half offset
}
```
where `O[d] = (k / dst_stride[d]) % dst_shape[d]`, and for `d==dim` the input
coordinate is `od` if `od < D` else `od - D` (i.e. `od % D` since `dst_shape[dim]=2D`).
Strides are compile-time constants. The strided loop handles `TOTAL_OUT > 1024`.

## The catalog-pre-pass wrinkle (new vs trans)
`_check_nd_cat_join` runs in the refusal pre-pass (`lower()` line 608), BEFORE the
detectors — so it would refuse before `_detect_nd_cat` runs. Fix: make
`_check_nd_cat_join` **defer the clean detectable pattern** — return None when the
op is `tt.cat` and the kernel is exactly `load(s) → [reshape/convert]* → cat →
[reshape/convert]* → store` (the same value-preserving data-flow the detector
requires). It still refuses: any `tt.join` rank≥2 (`join_with_mma` etc.), and any
N-D `tt.cat` with a compute op in the value path (which the direct-copy template
can't handle) — so no silent-wrong slips through.

## Components (files)
- `triton_msl/codegen/_lowerer_detection.py` — `_detect_nd_cat` (mirror
  `_detect_nd_trans`: one `tt.cat` rank≥2, ≥2 loads feeding it, store, no
  reduce/dot/control-flow; data-flow guard load→cat→store via value-preserving
  ops; extract x_arg/y_arg/out_arg, src_shape, dim, elem_type).
- `triton_msl/codegen/_lowerer_templates.py` — `_lower_nd_cat_template` (the
  closed-form strided copy above).
- `triton_msl/codegen/generic_lowerer.py` `lower()` — call `_detect_nd_cat` with
  the other detectors.
- `triton_msl/codegen/refusal_catalog.py` — `_check_nd_cat_join`: defer the clean
  N-D `tt.cat` pattern (return None) so the detector handles it; keep refusing
  non-clean cat + all rank≥2 join.

## Error handling / integrity
- The catalog still refuses any N-D cat the detector can't prove is the clean copy
  (compute in the value path) and all N-D joins — never silent-wrong.
- `dim` is read from the `tt.cat` attrs (or inferred from the shape doubling: the
  axis where `dst_shape[d] == 2*src_shape[d]`). If `dim` can't be resolved, the
  detector returns None → the catalog refuses → loud, correct.

## Testing / ratchet
- **Correctness (GPU, serial):** a project test (`tests/test_nd_cat.py`) over a few
  shapes × dims (incl. dim=0, a middle dim, and the last dim; a shape whose output
  exceeds 1024 to exercise the strided loop), compared to `torch.cat` (exact).
- **Corpus:** un-skip `test_cat_nd` in conftest_metal; the ~10 cases pass. Upstream
  `test_core` ratchets UP.
- **Regression:** 1-D `test_cat`/`test_join` unaffected (detector requires rank≥2);
  the catalog still refuses non-clean N-D cat + N-D join (`join_with_mma` stays
  skipped); project suite green.

## Out of scope
- `join_with_mma` (tt.join → tt.dot, 1 test, matmul-coupled) — stays refused
  (`_check_join_into_dot` untouched).
- N-D `tt.join` via load→join→store — no target test; stays refused.
