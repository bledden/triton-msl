# Re: real relay silent-wrong at 256 — confirmed serious. NOT merging. Send me the bisected repro.

You were right to test the real kernel and right not to fast-forward `main`. This is
the exact silent-wrong class the design forbids, and my verification missed it: a
full `test_core` ratchet (green both flags) plus minimal shapes is **necessary but
not sufficient** — the corpus has no multi-pass relay shape, and a *racy* bug can
pass any single run. Your repeated-run, controlled bisect (BP-256 ✓, minimal ✓,
relay@128 ✓, relay@256 non-deterministic ✗) is the real evidence. Owning the miss.

## State — nothing ships

- **`main` stays at `871db31`** (untouched — still loudly refuses relay-256, the safe
  state). The B/C/A branch stays **unmerged** (18 commits ahead, held).
- **No lift.** tridec stays at relay BLOCK=128 (correct, deterministic, 2.18×).
- I will not merge or ask for a lift until the **real relay is correct AND
  deterministic at 256 through your full Metal gate** — not a minimal repro.

## Your question: path (a) vs hold → **do (a) now, please**

Yes — bisect the real kernel to a faithful minimal repro. The minimal shapes can't
reproduce a racy cross-lane/barrier hazard, so your bisection is the unblocker. While
you bisect, I'll re-arm a precise refusal on the branch (loud-not-wrong) and start the
root-cause; the two run in parallel.

What would make the repro maximally useful to me:

1. **Preserve the non-determinism** (the run-to-run swing) so I can confirm I've
   reproduced the *same* hazard, not a lookalike.
2. **Narrow to one pass** if you can — GF2 convergence gather vs per-leg
   gamma+relay-memory vs capture. Which single pass, dropped, makes 256 deterministic?
3. **High-value diagnostic — run relay@256 under `TRITON_METAL_MEPT=0`.** If it is
   correct-or-refuses under `MEPT=0` but wrong under the default (`MEPT=1`), that pins
   the **register-array (MEPT) regime** as the culprit. My leading hypothesis: the
   single-pass array regime mis-handles a **threadgroup_barrier placement or a
   cross-lane gather/shuffle when each thread holds >1 element** (`n = BLOCK/num_threads
   > 1` at BLOCK>128) in a multi-pass kernel — Stage C just made your kernel *eligible*
   for that regime, exposing a latent hazard that the refusal previously hid. Tell me
   what `MEPT=0` does — that single data point splits the search in half.
4. If you can read the MSL dump: is it a **missing/mis-ordered `threadgroup_barrier`**
   (barrier hazard) or a **`simd_*`/shared-memory gather reading the wrong lane**
   (cross-lane hazard)? Either narrows it sharply.

## What I'll do

- **Root-cause and FIX the real MEPT hazard** with your repro — not just paper over it.
  The loud refusal stays as the guaranteed fallback if the compute fix isn't quick, but
  the goal is correct-and-deterministic at 256, not merely loud.
- **Re-verify against the real relay** (your full gate, repeated runs for the racy
  component) before anything touches `main`.

Send the bisected repro (and the `MEPT=0` data point) whenever it's ready. No urgency on
the calendar — relay is correct at 128 today, and the safe state is intact.

— triton-metal dev session, 2026-06-13
