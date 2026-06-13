# MEPT: array-wire arith.select (tl.where) — close the in-loop-reduction Bug 2 (2026-06-13)

> Make `arith.select` (`tl.where`) participate in the MEPT register-array regime,
> so a cross-lane reduction (`tl.sum`) **inside** a runtime-bound loop that uses
> masking computes at BLOCK≥256 instead of refusing. This is the remaining tridec
> Bug-2 case: it caps tridec's **relay** megakernel at BLOCK=128. Python/MSL.

## Source
tridec feedback `Bug2 remaining: tl.sum inside a runtime-bound loop` (2026-06-13),
verified against `triton_metal @ 0a1eafb` (CODEGEN_VERSION 2026.06.13). tridec's
relay caps at BLOCK=128 (loud refusal — never silent-wrong); BP is fully unblocked
(no `tl.sum`, runs 256–1024). The ask is **coverage**, not a behavior change.

## Confirmed root cause (controlled experiment, reproduced in our repo)
Varying ONE factor at a time from the known-passing M2 `_sum_in_loop` shape,
BLOCK=256, default flag:

| variant | result |
|---|---|
| A: in-loop `tl.sum`, masked load `other=0.0`, no `where` (M2 shape) | PASS |
| **B: A + `tl.where(m, load, 0.0)`** | **REFUSE** |
| C: A + 0-D `tl.zeros((),fp32)` accumulator | PASS |
| D: A + `range(0, C, BLOCK)` loop form | PASS |
| E: tridec's exact P3 (`where` + 0-D + `range(0,C,BLOCK)`) | REFUSE |

**The trigger is `tl.where` = `arith.select`.** Neither the accumulator form (C)
nor the loop form (D) matters — only the presence of a `select` op. `arith.select`
is in the "deliberately EXCLUDED (not array-wired)" set (`generic_lowerer.py:60`)
and is absent from `_MEPT_SAFE_OPS`. So `_arrayform_op_ok(select)` is False →
`mept_arrayform_eligible` is False → the kernel never enters the single-pass
register-array regime → the hoisted `tl.arange(0, BLOCK)` offsets and masked-load
`other=` are not rematerialized inside the loop body → codegen emits `UNKNOWN_<id>`
→ the integrity backstop refuses. tridec's `REPLY_BUG2` `.metal` dumps named those
same hoisted operands. (tridec's earlier "rematerialization doesn't descend into
loops" hypothesis is one lens; the precise mechanism is the eligibility gate — the
select op disqualifies the whole kernel from the arrayform regime that M2/M3a
built, which is exactly what already rematerializes the arange inside the loop for
the non-`where` case.)

This maps exactly to BP-works / relay-refuses: `_bp_megakernel` has no `tl.sum`;
`_relay_megakernel` has two in-loop `tl.sum` (megakernel.py:427 `mism=tl.sum(...)`
and :439 `w=tl.sum(...)`), both following masked `tl.where` — and both required
per BP iteration (can't be hoisted out).

## Approach — array-wire arith.select
1. Add `arith.select` to `_MEPT_SAFE_OPS` (`generic_lowerer.py:65`) and remove it
   from the "deliberately EXCLUDED" comment (line 60). select is elementwise and
   value-preserving — array-wiring it is the same per-element pattern as the other
   array-wired binaries.
2. Make `_lower_select` (`generic_lowerer.py:3820`) emit the **array form** when its
   operands are `env_array` entries: `T r[n]; for e: r[e] = cond[e] ? a[e] : b[e];`
   (via `_var_array` / `_materialize`, mirroring the array-wired binary dispatch),
   registering `env_array[ssa.id]`. Scalar/broadcast operands stay scalar
   (n_elems==1) — byte-identical to today when not in the array regime. The cond
   may be a per-element mask array or a scalar; handle both (broadcast a scalar
   cond across elements).
3. Result: a reduction-in-loop kernel using `tl.where` becomes
   `mept_arrayform_eligible` → enters single-pass → the arange/`other` operands are
   array-resident inside the loop → the in-loop `tl.reduce` folds correctly (the
   M2/M3 reduce-in-loop path already works once the kernel is eligible).

## Components (files)
- `triton_metal/codegen/generic_lowerer.py` — add `arith.select` to `_MEPT_SAFE_OPS`
  (+ update the line-60 comment); array-wire `_lower_select`.
- (Possibly) the array-wired-op dispatch site that routes binaries — wire select
  through it if that's where array emission is centralized.
- `scripts/conftest_metal.py` — no change expected (the in-loop-reduce shapes are in
  the upstream corpus already-passing for ≤128; this lifts the BLOCK≥256 refusal).

## Error handling / integrity
- The `UNKNOWN_` backstop stays. If array-wiring select still leaves any operand
  unresolved for a shape we don't cover, it refuses loudly (never silent-wrong) —
  same contract tridec relies on.
- Gated on the array regime (`_mept_single_pass`): scalar/flag-off select is
  byte-unchanged (the parity gate proves it).

## Testing / ratchet
- **Correctness (GPU, serial):** a project test resurrecting the A/B/E diagnostic as
  a regression — in-loop `tl.sum` with `tl.where` computes correctly at BLOCK
  256/512/1024 (compare to torch); plus a nested-loop variant (tridec's P4 / relay
  shape: `for leg: for it: total += tl.sum(where(...))`). Values/masks chosen so the
  `where` actually selects.
- **Flag/regression:** flag-default project suite green; parity gate byte-identical
  on the scalar corpus (select array-wiring must not change scalar-kernel MSL).
- **Downstream (offered by tridec):** tridec will test a candidate branch against
  the real relay kernel + `tests/test_megakernel_metal.py` and confirm relay lifts
  to BLOCK=256+. Coordinate via the bug thread.

## Value / priority
This unblocks a real downstream user's flagship kernel (tridec relay → full SIMD
width, 256–1024 vs the current 128 cap) and is a contained fix (one op array-wired).
Higher value than the remaining low-value Phase-3 tail (cat_nd ~10, map_elementwise
~2). Recommend prioritizing it within Phase 3.

## Out of scope
- `arith.cmpf` array-wiring (the other "deliberately excluded" op) — only if a
  needed kernel requires it; not part of this case.
- tridec's `relay256_bisect.py` `D`-control discrepancy (a no-`tl.sum` masked store
  that fails in isolation but works in the real BP kernel) — tridec flagged it as an
  unfaithful minimal kernel; revisit only if a faithful repro shows a real store gap.
