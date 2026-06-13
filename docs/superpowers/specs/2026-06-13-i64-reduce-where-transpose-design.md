# i64/u64 reduce / where / transpose — design (2026-06-13)

> Close `test_reduce1d`/`test_reduce2d`/`test_where`/`test_transpose` for int64/
> uint64 (gated behind METAL_TEST_INT64 in conftest_metal). Phase 3 feature 3.
> Python/MSL per the 2026-06-11 language decision.

## Problem
int64/uint64 scalar compute, comparison, load/store, and indexing already work
(~1047 cases pass). But:
- **reduce:** `_lower_reduce` uses `simd_sum`/`simd_max`/`simd_min` (the
  `SIMD_REDUCTIONS` intrinsics), which have NO `long`/`ulong` overload in MSL, and
  the integer path hardcodes `msl_type="int"` / `shared_dtype="i32"` — truncating a
  64-bit value. Result: a loud MSL compile error (correct refusal, but a gap).
- **transpose:** `_lower_tt_trans` uses `shared_dtype = "i32" if not is_float` —
  truncates i64 in the shared-memory exchange.
- **where (select):** `cond ? a : b` is type-agnostic but the lowering may emit an
  `int`-typed temporary for an i64 select. Likely just type-plumbing.

These are gated off by `_I64_UNIMPLEMENTED` in `scripts/conftest_metal.py:480`.

## Approach
Thread the 64-bit MSL type (`long` for int64, `ulong` for uint64) through the three
paths, and replace the unsupported `simd_*` reduction for 64-bit with a
**threadgroup shared-memory tree reduction** (type-agnostic, guaranteed to compile
for `long`/`ulong` — no `simd_sum(long)` and no `simd_shuffle(long)` dependency):

1. **reduce (1-D full):** when the reduce input dtype is i64/u64, take a dedicated
   64-bit path: declare a `long`/`ulong` threadgroup array of size `block_size`,
   write each thread's value, barrier, then a standard stride-halving parallel tree
   reduction in shared memory (`for (s=bs/2; s>0; s>>=1){ if (lid<s) sh[lid]=OP(sh[lid],sh[lid+s]); barrier; }`),
   result in `sh[0]`. Combine op: sum=`+`, max/min=`max`/`min` on the 64-bit type
   (umax/umin use the `ulong` type so the compare is unsigned). This mirrors the
   sequential shared-memory reduce the 2-D path already uses, generalized to 64-bit.
   The float/i32 fast SIMD path is unchanged.
2. **transpose:** in `_lower_tt_trans`, pick `shared_dtype`/`msl_type` = `i64`/`long`
   (or `u64`/`ulong`) when the element dtype is 64-bit. The exchange logic is
   otherwise unchanged.
3. **where:** ensure the select emits the 64-bit type for an i64/u64 result. Verify
   first whether it already works once un-skipped (it may be pure type-plumbing or
   already correct); only touch the select emission if a 64-bit case fails.

## Components (files)
- `triton_metal/codegen/_lowerer_reduce.py` — `_lower_reduce`: add the i64/u64
  branch (64-bit shared-memory tree reduction) before the SIMD path; thread the
  64-bit `msl_type`/`shared_dtype`.
- `triton_metal/codegen/generic_lowerer.py` — `_lower_tt_trans`: 64-bit
  shared_dtype/msl_type for i64/u64 elements.
- `triton_metal/codegen/_lowerer_*` — where/select: only if a 64-bit case fails.
- `scripts/conftest_metal.py` — remove/relax the `_I64_UNIMPLEMENTED` gate for the
  now-passing tests (keep `test_for_iv` skipped — i64 loop induction HANGS, separate
  and harder; keep i64/u64 atomics skipped — hardware-impossible).

## Error handling / integrity
- Multi-value reduces (argmin/argmax) on i64 stay on their existing path; if they
  don't support 64-bit, keep them skipped rather than emit wrong output.
- Overflow: 64-bit accumulators don't need a wider type (already widest); the
  existing narrow-type masking (i8/i16 sum) is unaffected.
- Anything 64-bit the new paths don't cover still refuses/compile-errors loudly
  (never silent-wrong); the conftest gate is only relaxed for tests that pass.

## Testing / ratchet
- **Correctness (GPU, serial):** a project test (`tests/test_i64_ops.py`): int64 +
  uint64 `tl.sum`/`tl.max`/`tl.min` (1-D reduce), int64 transpose, int64 where —
  compared to torch (exact integer equality; values large enough to exceed 32 bits
  to prove no truncation, e.g. > 2^40).
- **Corpus:** relax `_I64_UNIMPLEMENTED` in conftest_metal under the default (no
  METAL_TEST_INT64 needed); the un-gated `test_reduce1d`/`reduce2d`/`where`/
  `transpose` int64/uint64 variants pass. Upstream `test_core` ratchets UP.
- **Regression:** flag-default project suite green; float/i32 reduce + transpose
  paths byte-unchanged (the 64-bit branch is gated on the 64-bit dtype).

## Out of scope
- `test_for_iv` (i64 loop induction variable) — HANGS, separate gnarly issue.
- i64/u64 atomics — no Metal 64-bit device atomic (hardware-impossible).
- i64 argmin/argmax multi-value reduce — only if cheap; else stays skipped.
