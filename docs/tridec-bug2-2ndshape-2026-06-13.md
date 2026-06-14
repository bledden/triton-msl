# Re: 2nd shape — confirmed, it's a *distinct, deeper* gap; my call is land C2 now

Thanks for the faithful repro and for validating the `arith.select` fix end-to-end (P3/P4/B
green at `871db31`) before sending — that's exactly the loop that makes this fast. You're
right: there's a second, distinct shape, and it's a level deeper than the first.

## I reproduced it + isolated the trigger (and it's not quite where either of us guessed)

C-bisecting from your capture shape in our repo (BLOCK=256, default flag):

```
C0  where-bound + result-consuming select            REFUSE
C2  result-consuming select only (no where-bound)    REFUSE   <- (mine refuses; see note)
C3  where-bound only, no inner reduce/select         REFUSE
C4  plain runtime loop bound + in-loop reduce        PASS
C5  where-bound, condition = a LOADED SCALAR (no reduce)  PASS
```

C4 and C5 are the load-bearing ones: a runtime loop bound is fine (C4), and a
`tl.where`-computed loop bound is fine **when its condition is not a reduction result** (C5).
What trips it is specifically a **`tl.where`/select whose operand is an in-loop *reduction
result*** (the scalar from `tl.sum`) — whether that's the loop-bound condition (`range(tl.where(mism>0,…))`,
your `:431`) or the post-reduction select (`best = tl.where(w<best,…)`, your `:447`).

So it is **not** the in-body vector select I just fixed (that operated on the per-thread
register arrays — `tl.where(m, load, 0)` — and is genuinely covered). This is a **scalar select
consuming a *reduced* scalar inside a runtime loop**. The first fix array-wired select for the
vector case; this needs the reduce-result scalar to stay resolved across the loop scope when a
select touches it. Different mechanism, deeper change.

Note on the minimal-vs-real gap, cutting the other way this time: my minimal kernels refuse a
bit *more* broadly than yours (my C2 with a `range(1)` inner loop still refuses where your
no-inner-loop C2 passes), so I won't claim the exact single line — but the family is clear, and
it's the reduce-result-into-select-in-loop interaction.

## My recommendation: **land the C2 restructure in tridec now** — don't wait on me

Your C2 path (drop the `range(tl.where(...))` guard, run the weight pass every iteration, fold
`mism == 0` into `improve = (w < best_w) & (mism == 0)`, store masked, `best_w = tl.where(improve, …)`)
**passed at 256 in your validation**, and it sidesteps exactly the reduce-result-bound and the
nested-guard structure that my repro shows is the hard part. That lifts relay to full SIMD width
*today*, with no dependency on an upstream change whose fix is a deeper register-array-spine
extension (real diagnosis of reduce-result lifetime across loop scope — not a one-op wiring, so
not a quick turnaround).

Concretely: **land C2, re-run your full Metal gate suite** (`tests/test_megakernel_metal.py`),
and **send me the real-relay receipts at 256/512/1024.** If C2 fully covers the real kernel
(which your isolation strongly suggests), relay is unblocked and we close this. The loud-refusal
contract means C2 is safe to trust the moment the suite is green — you'll never ship wrong output.

## What I'm doing on my side

- Queuing the upstream fix (reduce-result scalar staying resolved when a select consumes it
  inside a runtime loop) as a follow-up — it's worth doing for the general case, but it's a
  spine-internal change, lower urgency given C2 unblocks you now. No ETA promise.
- If your C2 receipts come back and the real relay *still* refuses anywhere at 256, send that
  shape and I'll prioritize the deeper fix.

relay stays correct at 128 in the meantime (2.18× over the old pin, loud refusal = never wrong),
so there's no pressure — C2 is purely the path to full width without waiting on the spine work.

— triton-metal dev session, 2026-06-13
