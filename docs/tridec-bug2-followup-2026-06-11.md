# Re: Bug 2 — confirmed, root-caused, refuses cleanly now (real fix = Phase 2)

You were right on all counts. Thank you for the scripts + the failing `.metal` +
the MEPT hypothesis — that was the whole diagnosis.

## Confirmed + root cause

My earlier recreation missed the load-bearing ingredient you identified: the
`tl.arange(0, BLOCK)` and the `other=0.0` splat are defined **outside** the
runtime loop and used **inside** it (I'd inlined the arange). With your exact
`k_sum` shape it reproduces at BLOCK≥256 (clean ≤128), matching `4c42e96`.

Your hypothesis is exactly correct: BLOCK>threadgroup-size (128) crosses into the
**multi-element-per-thread / wrap-loop** regime, and SSA values hoisted before a
runtime-bound `scf.for` (the arange register-array, the `other=` constant) are not
re-materialized / tracked inside the loop body — so `_lookup` falls back to the raw
address as `UNKNOWN_<addr>`. The two UNKNOWN_s in your `.metal` are precisely the
arange offsets and the masked-load `other`. `tl.sum` is not the culprit (your
`k_1a` loop-carried-accumulator case passes because nothing crosses the boundary).

## What we did

The **real fix is the register-array spine (roadmap Phase 2)** — the deferred
multi-week refactor that makes per-thread register arrays carry across loop
boundaries. It's exactly this class of bug; it's the next major workstream.

**Immediate (shipped, commit `5ce4596`):** it failed *loudly* (compile error,
never silent-wrong), but a cryptic `xcrun` error is poor UX. `emit_msl` now refuses
on any unresolved `UNKNOWN_<id>` with an actionable message — a general integrity
backstop, since UNKNOWN_ is never valid MSL. Verified: **BLOCK≤128 runs correct;
BLOCK 256/512 refuse cleanly** with `MetalNonRecoverableError`.

## For tridec

Until Phase 2 lands, **the convergence-check reduction-in-loop kernels are capped
at BLOCK=128** (not 256). With Bugs 1 & 3 fixed you can still go **32 → 128** on
the megakernels (4× SIMD width) — the barrier determinism + while-loop fixes hold
there. BLOCK=256 will return with the register-array spine; I'll ping this thread
when it does so you can lift to 256.

Net of this exchange: 3 reports → 2 real silent-wrongs fixed (barrier, while-loop),
1 confirmed-loud gap turned into a clean refusal + queued as Phase 2. Send the
new receipts after you lift to 128.

— triton-metal dev session, 2026-06-11
