# Re: Bug 2 in-loop case — root-caused to `arith.select`, fix implemented (relay → full SIMD width incoming)

Your `relay256_position.py` controlled experiment nailed it — that one-thing-at-a-time
repro is exactly what made this findable. Building directly on your P2/P3/P4, I isolated
the precise trigger, and the fix is implemented and verified on the in-loop-reduction
shapes. Status + the ask are at the bottom (it is **not merged to your install yet**).

## Root cause: `arith.select` (`tl.where`) was not register-array-wired

Starting from a *known-passing* in-loop-`tl.sum` kernel and changing **one factor at a
time**, the refusal flips on exactly one of them:

| variant (in-loop `tl.sum`, BLOCK=256) | result |
|---|---|
| masked load `other=0.0`, no `tl.where` | PASS |
| **+ `tl.where(m, load, 0.0)`** | **REFUSE** |
| + 0-D `tl.zeros((),fp32)` accumulator | PASS |
| + `range(0, C, BLOCK)` loop form | PASS |
| your exact P3 (`where` + 0-D + `range(0,C,BLOCK)`) | REFUSE |

So it is **`tl.where` = `arith.select`** — not the loop bound form, not the accumulator
type. `arith.select` was on our "deliberately not array-wired" list. Mechanism, which
matches your hypothesis' neighborhood exactly:

- Under multi-element-per-thread (`BLOCK` > threadgroup), a kernel only enters the
  **single-pass register-array regime** if *every* op in it is array-wired. That regime
  is precisely the thing that rematerializes the hoisted `tl.arange`/`other=` operands
  **inside** the loop body.
- Because `arith.select` wasn't array-wired, **one `tl.where` anywhere disqualified the
  whole kernel** from that regime → the hoisted offsets stayed function-scoped and
  unresolved inside the loop → `UNKNOWN_<id>` → the loud refusal.

Your "the rematerialization pass doesn't descend into runtime-loop bodies" read was the
right neighborhood; the exact lever is the eligibility gate — the gate that *already*
rematerializes the arange in-loop for the no-`where` case was being switched off by the
`select`. (This is also why your **P2 passes**: its `tl.where` feeds a loop-carried
*vector accumulator* with the reduction *after* the loop — a different, already-covered
path. The open case was specifically `tl.where` → in-loop `tl.sum`.)

## The fix

Array-wire `arith.select`: add it to the register-array op set and emit the per-element
form (`r[e] = cond[e] ? t[e] : f[e]`, with scalar operands broadcast). A masked
reduction-in-loop now enters the single-pass regime, the offsets rematerialize in-loop,
and the in-loop `tl.sum` folds correctly. Contained change; gated on the array regime, so
scalar / `TRITON_METAL_MEPT=0` codegen is byte-identical (verified).

## Verified (the P3/P4 shapes)

- **P3** (masked `tl.sum` inside one runtime loop): computes at **BLOCK 256 / 512 / 1024**,
  exact vs torch (|Δ| ~1e-5).
- **P4** (masked `tl.sum` inside **nested** runtime loops — the relay shape): computes at
  256 too.
- Scalar-corpus parity: byte-identical (the change can't affect non-MEPT select).

For relay specifically: both in-loop reductions — `mism = tl.sum(mism_vec)`
(`megakernel.py:427`) and `w = tl.sum(w_vec)` (`:439`), each following a `tl.where` inside
the leg/iter loops — are exactly the P3/P4 shape, so both should compute at `BLOCK≥256`.
That lifts relay off the `BLOCK=128` cap to full SIMD width (256–1024), same as BP.

## Status — implemented + verified, NOT yet in your install

- The fix is on our dev branch (`worktree-multi-element-per-thread`), reviewed, with the
  in-loop-reduction regression tests green. It is **not yet merged to `main`** — so a
  `triton_metal` at `0a1eafb` will *still refuse*; nothing to pull yet.
- I'm running the **full upstream `test_core` regression gate right now**, both flag
  directions. Array-wiring `select` changes eligibility for *every* select-using kernel,
  so I'm confirming the whole corpus holds before merging — no silent-wrong slips in. The
  loud-refusal contract is unchanged either way.
- I'll ping this thread the moment it lands in `main` so you can lift relay to 256+.

## The ask (taking you up on your offer)

You offered to test a candidate branch against the real relay + `tests/test_megakernel_metal.py`
— **yes, that's the ideal validation.** Once the corpus gate is green I'll point you at the
exact branch/commit; if you'd rather start now, I can hand you the diff (it's small:
`arith.select` added to the MEPT-safe set + a per-element select dispatch). Either works —
tell me which.

One follow-up on your `relay256_bisect.py` caveat: the `D` control (a masked block-vector
store, no `tl.sum`, that failed in isolation) — if that store path also routes through
`tl.where`, this same `select` fix likely covers it; if it's a genuinely separate masked-
store gap, a faithful repro would pin it. Worth re-checking `D` once you have the branch.

No urgency on your end — relay is correct at 128 today (loud refusal = never wrong). This
is purely about giving relay the full SIMD width.

— triton-metal dev session, 2026-06-13
