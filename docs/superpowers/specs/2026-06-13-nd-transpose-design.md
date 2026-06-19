# Generic N-D transpose (tt.trans rank≥3) — design (2026-06-13)

> Close `test_trans_4d` (96 cases) by adding a detector + closed-form direct-copy
> template for the `load(N-D) → trans → [reshape] → store` pattern, instead of the
> shared-memory exchange (which can't handle the >1024-thread case). Phase 3
> feature 2. Python/MSL per the 2026-06-11 language decision.

## Problem
`_lower_tt_trans` (generic_lowerer.py:4132) implements only 2-D transpose (a
shared-memory exchange). Rank≥3 non-identity permutations refuse loudly
(`rank3_nonidentity_trans`). `test_trans_4d`: dtypes int32/int8, shapes
**(4,4,4,16)=1024** and **(2,2,8,64)=2048**, all 24 permutations of [0,1,2,3] = 96
cases. The 2048-element shapes exceed Metal's 1024-thread limit, and `tt.trans` is
a barrier op so the re-execution wrap-loop is unavailable — the shared-mem path
cannot serve them.

## Approach — detector + closed-form direct copy (no shared mem, no barrier)
Add `_detect_nd_trans` (mirrors the existing `_detect_transpose_via_reshape`) for
the pattern: one `tt.load` of a rank≥3 tensor → one `tt.trans` (any permutation) →
optional `tt.reshape`(s) → one `tt.store` to a flat pointer, with NO `tt.reduce`
and NO control flow. After `add_rewrite_tensor_descriptor_to_pointer`, the load
gives direct pointer access to the N-D input, so the transpose is a pure index
remap — no inter-thread exchange needed. Emit a strided direct copy:

```
for (uint k = lid; k < TOTAL; k += BLOCK) {     // strided loop covers TOTAL > 1024
    // output flat k -> output multi-index -> permuted input flat index
    out_ptr[k] = in_ptr[ SRC_FLAT(k) ];
}
SRC_FLAT(k): O[d] = (k / dst_stride[d]) % dst_shape[d]   for each output axis d
             return Σ_d  O[d] * src_stride[order[d]]
```
- `dst_shape[d] = src_shape[order[d]]`; strides are row-major, compile-time constants.
- `BLOCK = min(TOTAL, 1024)` (or the kernel's thread count) — the strided loop makes
  the >1024 case correct with ≤1024 threads. No shared memory, no barrier.
- Reduces to the 2-D formula (verified: order=[1,0] gives `(k%M)*N + (k/M)`).
- dtype-agnostic: int32/int8 only change the typed pointer declarations.

## Components (files)
- `triton_msl/codegen/_lowerer_detection.py` — `_detect_nd_trans(self) -> dict|None`:
  match the pattern; extract the input base pointer (from the load's ptr operand),
  output base pointer (from the store), `src_shape`, `order` (via the existing
  `_parse_trans_order`), and the element dtype. Return None unless it's a clean
  rank≥3 `load→trans→[reshape]→store` with no reduce/control-flow.
- `triton_msl/codegen/_lowerer_templates.py` — `_lower_nd_trans_template(self, info)
  -> str`: emit the full kernel MSL (strided direct-copy loop), modeling
  `_lower_transpose_via_reshape_template`.
- `triton_msl/codegen/generic_lowerer.py` `lower()` — call `_detect_nd_trans`
  alongside the other detectors (before the generic op-by-op path), returning its
  template MSL when it fires. Order it AFTER `_detect_transpose_via_reshape` and
  `_detect_permute_chained_reduce` (those are more specific) — though they already
  return None for this pattern, so order is for clarity.
- `scripts/conftest_metal.py` — un-skip `test_trans_4d`.

## Error handling / integrity
- The `_lower_tt_trans` rank≥3 refusal STAYS as the integrity backstop: any N-D
  trans the detector does NOT catch (e.g. trans feeding a non-store consumer, or
  with control flow) still refuses rather than silently dropping the permutation.
- The detector is conservative: it returns None on anything it can't prove is the
  clean copy pattern, so it never mis-fires into wrong output.

## Testing / ratchet
- **Correctness (GPU, serial):** a project test (`tests/test_nd_transpose.py`) over
  a representative set of the 24 perms × both shapes × {int32, int8}, comparing to
  `torch.permute(...).reshape(-1)`. Must include a 2048-element (>1024) shape to
  exercise the strided loop, and a non-trivial perm (e.g. (3,1,0,2)).
- **Corpus:** un-skip `test_trans_4d` in conftest_metal; the 96 cases pass.
  Upstream `test_core` ratchets UP, never down.
- **Regression:** flag-default project suite green; the 2-D transpose path and the
  two existing transpose templates unaffected (they return None for this pattern).

## Out of scope
- N-D transpose feeding a reduce (handled by `_detect_permute_chained_reduce`) or a
  2-D-load reshape-permute (handled by `_detect_transpose_via_reshape`) — already
  covered.
- transpose results consumed by something other than a flat store — stays refused.
