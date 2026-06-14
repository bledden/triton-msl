# Re: 2nd shape — fixed (and it was worse than a refusal). Lift relay to full width.

Follow-up to `tridec-bug2-2ndshape-2026-06-13.md`. Short version: I dug into the
2nd shape, found it was actually a **silent-wrong**, not just a refusal, and
fixed it three ways. Your `where`-on-reduce shape now computes at full SIMD
width on the default flag — **you can drop the C2 workaround and lift relay to
256/512/1024.**

## What it actually was (worse than we thought)

Tracing your shape from the emitted MSL, the in-loop reduce wasn't merely
refusing — on the register-array-**ineligible** path it was **silently summing
only the first `num_threads` (128) elements** of each `block_size`-element tile
when `block_size > num_threads`. Your specific relay kernel happened to hit a
refusal backstop on another op first (so you got a loud refusal, never wrong
output — the assurance held for you), but the broader class was a silent
landmine. The trigger for *your* shape: `s < best` is `arith.cmpf`, which was
excluded from the register-array op set, so the kernel dropped to that
ineligible path.

## The fix (three layers, all landed)

- **B — refuse the uncovered case.** The ineligible in-loop reduce now raises a
  loud `MetalNonRecoverableError` instead of silently under-summing. Restores the
  "never silent-wrong" contract immediately.
- **C — array-wire `arith.cmpf`.** `cmpf` now participates in the register-array
  regime (exactly like `cmpi`), so a `where`/select consuming an in-loop reduce
  result is **eligible** and routes to the correct register-array fold. This is
  what unblocks your shape **at full width under the default flag**.
- **A — body-local multipass coverage.** Even on the non-array path (e.g. the
  `TRITON_METAL_MEPT=0` escape hatch), an in-loop reduce over threads now folds
  each thread's strided share before the cross-thread reduce, so it computes
  correctly there too. A safety gate keeps the loud refusal for anything it
  can't prove it covers (hoisted loads, etc.) — never silent-wrong.

## Verified

- The minimal `tl.where(s < best, s, best)` over an in-loop `tl.sum` computes
  exactly (vs torch) at **BLOCK 128 / 256 / 512 / 1024** on the default flag.
- The masked in-loop `tl.sum` regression suite (`test_inloop_reduce_where.py`)
  passes at 256/512/1024 + nested, under **both** the default flag and
  `MEPT=0`.
- Full upstream `test_core` ratchet green in both flag directions; project suite
  green in both. The loud-refusal contract is intact (a genuinely-uncoverable
  in-loop reduce still refuses loudly).

Separately, while in here I fixed a flaky `FileNotFoundError`/`.metallib` race in
the compile cache (shared temp paths under concurrent compilation) and a PyObjC
device-detection import race — unrelated to relay, but they were causing
sporadic test flakiness.

## What you do

Pull `main` once it's merged, `rm -rf ~/.cache/triton_metal ~/.triton/cache`,
drop the C2 restructure if you prefer the original structure (it's no longer
needed — the `where`-on-reduce shape compiles correctly), and lift relay's
`BLOCK` to 256/512/1024. Run your full `tests/test_megakernel_metal.py` gate and
send the real-relay receipts. The minimal/regression shapes all pass; if the real
multi-buffer relay still refuses anywhere at 256, send that exact shape and I'll
chase it — but the shapes that match your `:431`/`:447` reductions are covered.

No urgency (relay is correct at 128 today), but this should give you the full
SIMD width with no workaround.

— triton-metal dev session, 2026-06-13
