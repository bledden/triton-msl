# triton-metal workplan: n>1 (BLOCK > num_threads) coverage & generality

**Context (2026-06-14):** the in-loop-reduce / silent-wrong arc is closed and merged
(`main @ b82136b`). tridec is **fully unblocked** — BP and relay megakernels run at
`BLOCK=256` (`num_warps = BLOCK/32`, n=1), correct + deterministic, their full Metal gate
4/4. This backlog (from tridec's 2026-06-14 review + my own audit) tracks the residual
generality/integrity work. **None of it blocks tridec.** Ordered by priority.

## P1 — Integrity (ACTIVE): extend the n>1 under-cover refusal to atomics

`b82136b` refuses a BLOCK-wide **store** the base path can't cover at `n>1`
(`block_size > num_threads`), which closed the relay silent-wrong. **Audit confirms the
guard is store-only** (`generic_lowerer.py:_lower_store:2958`); `_lower_atomic_rmw`
(`_lowerer_control.py:663`) and `_lower_atomic_cas` (`:845`) have **no** such guard. A
BLOCK-wide **atomic scatter** at `n>1` on the base path would emit one element per thread
(`r = k + lid`) and silently drop the rest — the same silent-wrong class.

**Fix:** mirror the store refusal in `_lower_atomic_rmw` and `_lower_atomic_cas`: on the
base path (`not _mept_single_pass and not _needs_wrapping`), if the atomic's value/operand
tensor width `> num_threads`, raise `MetalNonRecoverableError` directing the user to
`num_warps = BLOCK/32`. Loads do **not** need their own guard: an under-covered load's
wrong values can only reach output through a store (refused), an atomic (refused after this
fix), or a reduce (covered by Stage A / multipass) — so guarding the output-producing
element-wise ops (store + atomics) closes the class. Verify: a BLOCK-wide atomic at n>1
refuses; the Phase 3 fp16/bf16 atomic tests (n=1) still pass; full ratchet 5531/0 both flags.

## P2 — Generality (deferred, large): full base-path n>1 coverage for general ops

Make the base path emit the strided per-thread element loop (`_loop_e`, like the Stage-A
reduce-cover) for **all** general element-wise loads/stores/atomics, so **any** `num_warps`
works at `BLOCK > num_threads` (not just `num_warps = BLOCK/32`). Removes the refusal's
constraint entirely. **Not needed** — `num_warps = BLOCK/32` (n=1) is correct and the fastest
config measured. Large change touching the core element-wise lowering; high regression risk;
do only if a real kernel needs n>1 on the base path and can't set num_warps.

## P3 — Generality (long-term, largest): register-array eligibility for relay-shaped kernels

Extend the MEPT single-pass eligibility analysis to cover multi-pass kernels with in-loop
reductions + masked captures (the relay shape), so they use the register-array regime at n>1
with arbitrary `num_warps` (like BP, which is already eligible). Largest item; clearly out of
scope for the lift; recorded for completeness.

## Considered & declined: auto-select num_warps to force n=1

tridec suggested: when a kernel is not register-array-eligible and launched with
`BLOCK > num_threads`, auto-pick `num_warps = BLOCK/32` instead of refusing. **Declined for
now** — silently changing `num_warps` changes the threadgroup size (occupancy, shared-memory
sizing) and can mask user intent; an explicit loud refusal with an actionable message is
better DX than implicit adjustment. Revisit only if the refusal proves a recurring annoyance
for other users (tridec sets num_warps explicitly, so no impact).

## Non-item: fp64 on Metal

Metal hardware has no fp64; tridec's fp64 path is hardware-N/A, not a triton-metal gap.
Nothing to do.

---

*Source: tridec backlog `tridec-bug-reports/` (REPLY_RELAY_LIFTED / BISECTION /
REAL_RELAY_SILENT_WRONG, 2026-06-14) + triton-metal audit. Prime directive unchanged:
never silent-wrong — every uncoverable pattern refuses loudly.*
