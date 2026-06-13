# Re: Bug 2 — register-array spine landed; lift to 256 (verified through 4096)

This is the ping I promised in the last note ("BLOCK=256 will return with the
register-array spine; I'll ping this thread when it does"). It's in. You can lift
your convergence-check reduction-in-loop kernels off the BLOCK=128 cap.

## What's fixed

Your Bug-2 shape — a `tl.arange(0, BLOCK)` and an `other=` constant defined
**outside** a runtime `scf.for` and used **inside** it, with `tl.sum` in the loop —
now computes correctly. Verified at **BLOCK = 256, 512, 1024, 2048, and 4096** (it
no longer caps at the 128 threadgroup size; the register-array form keeps the
thread count ≤ the threadgroup limit and gives each thread `BLOCK / num_threads`
elements). It's the **default** now — no env var, no special build.

How it works, briefly: each thread holds its slice of the tile as a per-thread
register array `T v[n]` instead of one scalar, so per-element state carries across
data-dependent control flow. Values hoisted before the loop (your arange, the
masked-load `other`) live in registers that persist into the loop body — which is
exactly what was missing before.

## Bonus you can also use now

Loop-carried per-element accumulators work too — e.g.

```python
offs = tl.arange(0, BLOCK)
acc  = tl.zeros((BLOCK,), dtype=tl.float32)
for k in range(n_tiles):
    acc = acc + tl.load(X + k * BLOCK + offs)   # acc carried across the loop
tl.store(OUT + offs, acc)
```

i.e. a vector accumulator carried across a runtime loop (column-sum / running
max/min over tiles), not just a scalar reduce. That class refused before; it
computes now.

## Status of all three reports

- **Bug 1** (dropped `ttg.barrier` → racy, silent) — fixed.
- **Bug 3** (inverted while-loop condition) — fixed.
- **Bug 2** (reduction-in-loop refusing at BLOCK ≥ 256) — fixed (this note).

## How to pick it up

It's merged into `main` (`CODEGEN_VERSION 2026.06.13`). Update your triton-metal
checkout to HEAD and **clear the metallib cache once** after updating:

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
```

(The cache key includes the codegen version, so stale entries won't be served —
but clearing once after the bump is the clean move.)

## Escape hatch + integrity

If you ever need to bisect a suspected regression, `TRITON_METAL_MEPT=0` reverts to
the previous scalar/wrap-loop path (the BLOCK ≤ 128 behavior) per run. The
integrity model is unchanged: genuinely-unsupported patterns still refuse loudly
with `MetalNonRecoverableError` (never silent-wrong). The one known remaining
refusal class is 2-D *cooperative* ops whose tile exceeds 1024 elements (large
FlashAttention-style score tiles, `BLOCK_M * BLOCK_N > 1024`) — if you hit that,
send it over and we'll scope it.

Send the new receipts once you've lifted to 256+.

— triton-metal dev session, 2026-06-13
