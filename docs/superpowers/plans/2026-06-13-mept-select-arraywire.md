# MEPT array-wire arith.select (in-loop-reduction Bug 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Array-wire `arith.select` (`tl.where`) so a masked cross-lane reduction (`tl.sum`) inside a runtime-bound loop computes at BLOCK≥256 instead of refusing — closing tridec's remaining Bug-2 case and lifting its relay megakernel to full SIMD width.

**Architecture:** `arith.select` is currently excluded from `_MEPT_SAFE_OPS`, so any kernel using `tl.where` is ineligible for the single-pass register-array regime (confirmed root cause: the controlled A/B/E experiment isolates the trigger to the select op). Add `arith.select` to the safe set and give `_lower_select` an array-form branch (per-element `r[e] = cond[e] ? a[e] : b[e]`), mirroring `_mept_binary_dispatch`. Gated on the array regime — scalar/flag-off select is byte-unchanged.

**Tech Stack:** Python lowerer (`triton_msl/codegen/generic_lowerer.py`, `_lowerer_emission.py`), MSL, pytest (serial GPU), upstream `test_core` ratchet.

**Key risk:** array-wiring select changes eligibility for a *class* of select-using kernels (they may now enter the arrayform regime). The full flag-default `test_core` + project-suite ratchet (Task 2) is the essential no-regression gate — not optional.

> Run from the worktree with `/opt/homebrew/bin/python3`. GPU: SERIAL ONLY, dual-cache-clear before codegen-sensitive runs, `pkill`+recovery on hang.

---

### Task 1: Array-wire arith.select + regression test (RED→GREEN)

**Files:**
- Modify: `triton_msl/codegen/generic_lowerer.py` (`_MEPT_SAFE_OPS` + `_lower_select`)
- Modify: `triton_msl/codegen/_lowerer_emission.py` (add `_mept_select_dispatch`)
- Test: `tests/test_inloop_reduce_where.py` (create)

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_inloop_reduce_where.py` (the confirmed A/B/E shapes as a GPU correctness test):

```python
"""tridec Bug-2 remaining case: a masked cross-lane reduce (tl.sum over
tl.where) inside a runtime-bound loop must compute at BLOCK>=256, not refuse.
The trigger was arith.select not being MEPT-array-wired. Serial GPU."""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _sum_where_in_loop(X, OUT, N, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = tl.zeros((), dtype=tl.float32)
        for i in range(n_tiles):
            idx = i * BLOCK + offs
            m = idx < N
            total += tl.sum(tl.where(m, tl.load(X + idx, mask=m, other=0.0), 0.0))
        tl.store(OUT, total)

    @triton.jit
    def _sum_where_nested(X, OUT, N, n_legs, n_tiles, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = tl.zeros((), dtype=tl.float32)
        for leg in range(n_legs):
            for i in range(n_tiles):
                idx = i * BLOCK + offs
                m = idx < N
                total += tl.sum(tl.where(m, tl.load(X + idx, mask=m, other=0.0), 0.0))
        tl.store(OUT, total)


@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512, 1024])
def test_sum_where_in_loop(BLOCK):
    N = 4096
    X = torch.randn(N); OUT = torch.zeros(1)
    _sum_where_in_loop[(1,)](X, OUT, N, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - X.sum().item()) < 1e-1, (
        f"BLOCK={BLOCK}: got {float(OUT[0])} want {X.sum().item()}")


@requires_metal
def test_sum_where_nested():
    # nested loops (tridec relay shape): total = n_legs * sum(X)
    N, BLOCK, n_legs = 2048, 256, 3
    X = torch.randn(N); OUT = torch.zeros(1)
    _sum_where_nested[(1,)](X, OUT, N, n_legs, (N + BLOCK - 1) // BLOCK, BLOCK=BLOCK)
    assert abs(float(OUT[0]) - n_legs * X.sum().item()) < 2e-1
```

- [ ] **Step 2: Run, verify it FAILS (refusal today)**

```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/test_inloop_reduce_where.py -v 2>&1 | tail -12
```
Expected: FAIL — `MetalNonRecoverableError` (UNKNOWN_, the select disqualifies the kernel from the arrayform regime). The RED.

- [ ] **Step 3: Add `_mept_select_dispatch` to `_lowerer_emission.py`**

Add to the emission mixin (next to `_mept_binary_dispatch`):

```python
    def _mept_select_dispatch(self, ssa, cond_id, t_id, f_id,
                              cond, t, f, ty, dtype) -> bool:
        """MEPT array dispatch for arith.select (ternary). If any operand is a
        register array, emit a per-element select and return True; else False
        (caller does the scalar path). Mirrors _mept_binary_dispatch."""
        if not self.mept_enabled:
            return False
        c_arr = self.env_array.get(cond_id)
        t_arr = self.env_array.get(t_id)
        f_arr = self.env_array.get(f_id)
        arrs = [a for a in (c_arr, t_arr, f_arr) if a is not None]
        if not arrs:
            return False
        ns = {a[1] for a in arrs}
        if len(ns) != 1:
            return False  # mismatched array lengths -> scalar fallback
        n = ns.pop()
        read_c = ((lambda i, an=c_arr[0]: f"{an}[{i}]") if c_arr
                  else (lambda i, cv=cond: cv))
        read_t = ((lambda i, an=t_arr[0]: f"{an}[{i}]") if t_arr
                  else (lambda i, tv=t: tv))
        read_f = ((lambda i, an=f_arr[0]: f"{an}[{i}]") if f_arr
                  else (lambda i, fv=f: fv))
        exprs = [f"({read_c(i)} ? {read_t(i)} : {read_f(i)})" for i in range(n)]
        var_name = self._var_array("r", exprs, ty)
        self.env[ssa.id] = var_name
        self.env_array[ssa.id] = (var_name, n, ty)
        self.env_types[ssa.id] = dtype
        self._propagate_shape_elementwise(ssa)
        return True
```

- [ ] **Step 4: Wire it into `_lower_select`**

In `generic_lowerer.py` `_lower_select`, after the `ty`/`dtype` are computed and BEFORE the existing scalar `self.kb.raw_line(f"    {ty} {var_name} = {cond} ? ...")`, insert:

```python
        if self._mept_select_dispatch(
                ssa, ssa.operand_ids[0], ssa.operand_ids[1],
                ssa.operand_ids[2], cond, true_val, false_val, ty, dtype):
            return
```
(The existing scalar emission + `self.env[ssa.id] = var_name` stay as the fallback for the scalar/non-array case.)

- [ ] **Step 5: Add `arith.select` to `_MEPT_SAFE_OPS`**

In `generic_lowerer.py`, add `"arith.select"` to the `_MEPT_SAFE_OPS` frozenset (near `"arith.cmpi"`), and update the line-60 "Deliberately EXCLUDED (not array-wired)" comment to drop `arith.select` (it IS array-wired now) — leave `arith.cmpf` listed.

- [ ] **Step 6: Run the regression test — verify GREEN**

```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/test_inloop_reduce_where.py -v 2>&1 | tail -8
```
Expected: all pass (BLOCK 256/512/1024 + nested) — the masked in-loop reduce now computes. If still refusing, the kernel isn't reaching `mept_arrayform_eligible`: check that `arith.select` is now accepted by `_arrayform_op_ok` (it routes through `_op_mept_ok` → `_MEPT_SAFE_OPS`) and that `_mept_select_dispatch` actually fires (the cond/operands are env_array inside the loop). Dump MSL if needed.

- [ ] **Step 7: Parity gate (select array-wiring must not change scalar MSL)**

```bash
/opt/homebrew/bin/python3 -m pytest tests/test_mept_parity.py tests/test_regval.py -q 2>&1 | tail -4
```
Expected: pass — `_mept_select_dispatch` returns False for scalar operands, so scalar-corpus MSL is byte-identical.

- [ ] **Step 8: Commit**

```bash
git add triton_msl/codegen/generic_lowerer.py triton_msl/codegen/_lowerer_emission.py tests/test_inloop_reduce_where.py
git commit -m "feat(mept): array-wire arith.select — close tridec in-loop-reduction Bug 2

tl.where (arith.select) was excluded from _MEPT_SAFE_OPS, disqualifying any
masked kernel from the single-pass register-array regime, so a tl.sum inside a
runtime loop (after tl.where) refused at BLOCK>=256 (hoisted arange not
rematerialized -> UNKNOWN_). Add arith.select to the safe set + a per-element
_mept_select_dispatch. Masked in-loop reduce computes at BLOCK 256/512/1024 +
nested. Gated on the array regime; scalar/flag-off select byte-unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Full ratchet (the eligibility-change safety net)

> Array-wiring select changes eligibility for ALL select-using kernels — the full corpus is the essential no-regression check. ~15-min test_core run; serial.

- [ ] **Step 1: Flag-default full upstream test_core (must not regress; should hold or rise)**

```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache
unset TRITON_MSL_MEPT
bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -6
```
Expected: **>= 5335 passed, 0 failed** (the count may RISE if select-using reductions un-block; must NOT fall). Any new failure = a select-using kernel that the arrayform regime now mis-handles → STOP, capture the failing test, diagnose (likely `_mept_select_dispatch` shape/length edge) — do NOT proceed with a regression.

- [ ] **Step 2: Escape-hatch direction (MEPT=0) unchanged**

```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache
TRITON_MSL_MEPT=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -6
```
Expected: 5335 passed / 0 failed (select stays scalar with MEPT=0 — byte-unchanged).

- [ ] **Step 3: Project suite (flag-default)**

```bash
rm -rf ~/.cache/triton_msl ~/.triton/cache
/opt/homebrew/bin/python3 -m pytest tests/ -q -k "not test_mept_m2_bug2_gpu and not test_mept_m3a_itercarry_gpu and not test_mept_m3c_gt1024_gpu" 2>&1 | tail -4
```
Expected: 0 failed (the known autotuner flake aside — re-run alone if it appears).

- [ ] **Step 4: Update memory / notify (tridec)**

Note in `project_phase3_features.md` that the select-fix landed (test_core delta), and that tridec can be told relay should now lift to BLOCK≥256 (they offered to test a candidate branch against the real relay + `tests/test_megakernel_metal.py`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(mept-select): ratchet — flag-default + MEPT=0 test_core hold/rise, project green

Array-wiring arith.select holds the corpus (both flag directions); tridec
in-loop-reduction Bug 2 closed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** add arith.select to _MEPT_SAFE_OPS (Step 5); array-form `_lower_select` via `_mept_select_dispatch` (Steps 3-4); regression test for masked in-loop reduce + nested (Step 1); parity byte-identical (Step 7); the eligibility-change full-corpus ratchet, both flag directions (Task 2). ✓

**Placeholder scan:** `_mept_select_dispatch` + the wiring + the test are complete code. The "dump MSL if still refusing" (Step 6) is a concrete diagnostic, not a placeholder. ✓

**Type consistency:** `_mept_select_dispatch(ssa, cond_id, t_id, f_id, cond, t, f, ty, dtype)` — `ty`/`dtype` are exactly the values `_lower_select` already computes; mirrors `_mept_binary_dispatch`'s env/env_array/shape bookkeeping (`_var_array`, `_propagate_shape_elementwise`). The dispatch returns False on no-array / mismatched-length → scalar fallback unchanged. ✓
