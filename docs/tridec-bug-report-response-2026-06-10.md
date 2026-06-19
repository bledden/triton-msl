# Response to tridec's three triton-msl bug reports (2026-06-10)

Thanks — this is exactly the kind of real-workload burn-in the project needs. All
three investigated against the current dev branch (`worktree-multi-element-per-thread`,
~100 commits ahead of the `4c42e96` you ran, which IS an ancestor so everything
since applies). Summary:

| Bug | Status on dev branch | Action |
|---|---|---|
| 1 (barrier dropped) | **CONFIRMED + FIXED** | commits `f48f9ed` + `01a1f4f` |
| 2 (sum-in-loop UNKNOWN_ ≥256) | **Not reproducible** | likely already fixed; need your script to be sure |
| 3 (while + scalar-if dropped) | **CONFIRMED + FIXED** (different mechanism than diagnosed) | commit `86d5d10` |

---

## Bug 1 — CONFIRMED, FIXED. You had the root cause exactly right.

`tl.debug_barrier()` arrives as `ttg.barrier` in TTGIR (triton's rename), and the
lowerer only knew `tt.debug_barrier`. But it wasn't the unknown-op *skip* that hid
it — `ttg.barrier` hit `_lower_ttg`'s "other ttg ops: passthrough" branch (before
the generic unknown-op guard), so it was silently dropped.

Fixed three ways:
1. `_lower_ttg` emits `threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)`
   for `ttg.barrier` (full `__syncthreads` semantics — orders both threadgroup AND
   device memory; your cross-lane gather needs the device flag).
2. Added `ttg.barrier` to the `has_barrier_ops` scan.
3. **Your meta-bug point is addressed**: the `ttg.*` passthrough fallthrough now
   *refuses loudly* for any unknown no-result (side-effecting) ttg op, matching the
   generic-dispatch default-deny we'd already added. A future op rename can't
   silently drop a barrier again.

**GPU-verified**: a BLOCK=128 cross-lane write→barrier→gather is now correct and
bit-identical across 5 runs (was racy). **You can lift the BLOCK=32 workaround** —
set `block=128` (or 256, pending bug 2). Regression test:
`tests/test_barrier_lowering.py`.

## Bug 3 — CONFIRMED, FIXED. Real silent-wrong, but the mechanism is subtler than "scalar-if dropped".

It's not the `if` that's dropped — it's the **loop condition that inverts**. A
`while (a < n) and (cond):` whose body modifies a carried var (keeping `cond` live)
mis-compiled because cmpi predicates were keyed by SSA *name*, and `scf.while`'s
before/after regions reuse local names (`%0`). The after-region's `cmpi sge`
overwrote the before-region's `cmpi slt`, so `a < n` emitted as `a >= n` → the loop
broke on iteration 0 and the carried var read its initial value (your "never
updated" symptom). Fix: index predicates positionally by walk order, not by name
(`86d5d10`).

Note: your `tl.where` workaround is still good practice, but the pattern now works.
Verified plain / compound-`and` / `if`-as-select while loops all correct;
`tests/test_while_loop_carried.py`.

## Bug 2 — could not reproduce on the dev branch.

`tl.sum` in a runtime-bound loop, strided AND counted, at BLOCK 128/256/512 — the
emitted MSL is clean (no `UNKNOWN_` identifiers), so `xcrun` compiles fine. Since
this was a *compile-level* failure and the codegen is now clean, I believe it was
fixed by intervening reduce/loop work between `4c42e96` and HEAD. **To be certain I
fixed *your* exact kernel and not an approximation, please share `mk_probe2.py` /
`mk_probe3.py`** (or re-test your `k_sum` family against the dev branch). If it
still repros, send the failing `.metal` and I'll fix the specific structure.

## To pick up the fixes

These are on the dev branch, not yet merged to whatever `4c42e96` was. Once merged,
clear `~/.cache/triton_msl` (the cache key now includes a codegen version, so this
is the last time you'll need to clear manually) and re-run your acceptance test at
BLOCK=128 — both the barrier determinism gate and the while-loop forms should pass.

— triton-msl dev session
