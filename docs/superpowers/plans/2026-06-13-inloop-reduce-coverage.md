# In-loop Reduction Coverage Fix (B+C+A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate a silent-correctness bug where a `tt.reduce` inside a runtime `scf.for`/`scf.if`/`scf.while` body silently sums only the first `num_threads` elements of its block when `block_size > num_threads` and the reduce is not register-array-covered.

**Architecture:** Three layers. **B** — refuse the uncovered in-loop reduce loudly (restores the loud-refusal contract). **C** — array-wire `arith.cmpf` so the common `where`-on-reduce shape becomes register-array-eligible and routes to the already-correct fold (default flag only). **A** — measure the residual surface B refuses; build body-local multipass replay only if that surface is non-empty (it is the only correctness path under `MEPT=0`).

**Tech Stack:** Python MSL-source backend codegen (`triton_metal/codegen/`), pytest against Apple Metal GPU (serial), upstream `test_core` ratchet via `scripts/run_upstream_test.sh`.

**Spec:** `docs/superpowers/specs/2026-06-13-inloop-reduce-coverage-design.md`

**Conventions for every GPU/ratchet run in this plan:**
- Clear the cache first: `rm -rf ~/.cache/triton_metal ~/.triton/cache` (the key includes the effective MEPT flag, but clear anyway when verifying codegen changes).
- Project GPU tests run from the worktree root: `python3 -m pytest tests/<file> -q` (serial; do not parallelize Metal tests).
- Upstream ratchet (default flag): `bash scripts/run_upstream_test.sh unit/language/test_core.py -q`
- Upstream ratchet (escape hatch): `TRITON_METAL_MEPT=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q`
- Baseline to hold/raise: `test_core` **5531 passed / 0 failed** (both flag directions); project suite **0 failed**.

---

## File Structure

- `triton_metal/codegen/generic_lowerer.py` — `__init__` gets `_control_flow_depth`; `_lower_op` increments it around control-flow dispatch; `_MEPT_SAFE_OPS` gains `arith.cmpf`; `_lower_cmpf` gains the MEPT array branch (Tasks 1, 3).
- `triton_metal/codegen/_lowerer_reduce.py` — `_lower_reduce` gains the B refusal for the uncovered 1-D in-loop case; (Task 6, conditional) body-local multipass coverage.
- `tests/test_inloop_reduce_coverage.py` — new GPU regression test (Tasks 1, 3, 6).
- `tests/test_reduceresult_select_DIAG.py` — existing temporary diagnostic; deleted in Task 7 (its assertions are absorbed into the kept test).
- `scripts/conftest_metal.py` — only touched if Task 2/5 finds a now-refused upstream test needs an explicit skip (correctness-honest).

---

## Task 1: B — refuse the uncovered in-loop reduce

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py` (`__init__` ~line 190; `_lower_op` lines 1871-1876)
- Modify: `triton_metal/codegen/_lowerer_reduce.py` (`_lower_reduce`, after the 2D/3D/ND dispatch returns ~line 424, before the i64 path at 431 and the 1-D path at 436)
- Test: `tests/test_inloop_reduce_coverage.py`

- [ ] **Step 1: Write the failing test (B refuses under MEPT=0)**

Create `tests/test_inloop_reduce_coverage.py`:

```python
"""In-loop reduction coverage (spec 2026-06-13-inloop-reduce-coverage).

A tt.reduce inside a runtime loop must NEVER silently sum only the first
num_threads elements when block_size > num_threads. Under MEPT=0 (no register
arrays) such a reduce must refuse loudly (Stage B); under the default flag the
common where-on-reduce shape must compute correctly (Stage C). Serial GPU.
"""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    from triton_metal.errors import MetalNonRecoverableError
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _sum_carry_in_loop(X, OUT, C: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, C):
            v = tl.load(X + i * BLOCK + tl.arange(0, BLOCK))
            acc = acc + tl.sum(v)
        tl.store(OUT + tl.arange(0, 1), acc)


@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512])
def test_inloop_reduce_mept0_refuses(BLOCK, monkeypatch):
    """MEPT=0: an in-loop reduce with block>num_threads is uncovered → refuse
    loudly (was silent-wrong before Stage B)."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    C = 4
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError):
        _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)


@requires_metal
def test_inloop_reduce_small_block_ok(monkeypatch):
    """block_size <= num_threads is fully covered (one elem/thread) → never
    refused, correct under both flags."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    BLOCK, C = 128, 4
    torch.manual_seed(0)
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)
    torch.testing.assert_close(OUT[0], X.sum(), rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run the test, verify it FAILS (currently silent-wrong, not refusing)**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_inloop_reduce_coverage.py -q`
Expected: `test_inloop_reduce_mept0_refuses` FAILS (`DID NOT RAISE MetalNonRecoverableError` — today it returns a wrong value). `test_inloop_reduce_small_block_ok` PASSES.

- [ ] **Step 3: Add the control-flow depth counter**

In `generic_lowerer.py` `__init__`, after the existing env/flag inits (near line 190, right after `self.mept_enabled = ...`), add:

```python
        # Depth of nested control-flow bodies (scf.for/if/while) currently
        # being lowered. An in-loop reduce (depth > 0) that is not register-
        # array-covered cannot use the top-level multipass wrap and would
        # silently under-cover block_size > num_threads — see _lower_reduce.
        self._control_flow_depth = 0
```

In `generic_lowerer.py` `_lower_op` (lines 1871-1876), replace the three control-flow dispatches with depth-tracked versions:

```python
        elif op == "scf.for":
            self._control_flow_depth += 1
            try:
                self._lower_scf_for(ssa)
            finally:
                self._control_flow_depth -= 1
        elif op == "scf.if":
            self._control_flow_depth += 1
            try:
                self._lower_scf_if(ssa)
            finally:
                self._control_flow_depth -= 1
        elif op == "scf.while":
            self._control_flow_depth += 1
            try:
                self._lower_scf_while(ssa)
            finally:
                self._control_flow_depth -= 1
```

- [ ] **Step 4: Add the B refusal in `_lower_reduce`**

In `_lowerer_reduce.py` `_lower_reduce`, immediately after the 2D dispatch block returns (after line ~423, before the `# Cast bool (i1)` block at line 425), insert:

```python
        # Stage B (in-loop reduction coverage): a 1-D full reduce whose tile
        # exceeds the threadgroup (block_size > num_threads) is only correct
        # when its per-thread input already covers the whole tile — either via
        # the register-array fold (_mept_reduce_fold, applied above, which sets
        # input_shape=None) or the top-level multipass wrap (which rebinds the
        # input to a scalar accumulator before reaching here, depth == 0). An
        # in-loop reduce (inside scf.for/if/while, depth > 0) with a raw block
        # tensor and no array cover would emit a one-element-per-thread cross-
        # lane reduce that SILENTLY sums only the first num_threads elements.
        # Refuse loudly instead of returning a wrong result.
        if (mept_arr is None
                and self._control_flow_depth > 0
                and input_shape is not None
                and len(input_shape) == 1
                and input_shape[0] > self.kb.block_size):
            from triton_metal.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError(
                f"Refusing in-loop reduction: a tile of {input_shape[0]} "
                f"elements exceeds the {self.kb.block_size}-thread threadgroup "
                f"and is not register-array-covered, so a cross-lane reduce "
                f"here would sum only the first {self.kb.block_size} elements "
                f"(silent-wrong). Use the default register-array path "
                f"(TRITON_METAL_MEPT unset) or BLOCK <= num_threads.")
```

Note: `mept_arr` and `input_shape` are already in scope here (computed at lines ~392-398 and ~381-383). When `mept_arr` is not None, line ~398 sets `input_shape=None`, so the condition naturally skips the array-covered case.

- [ ] **Step 5: Run the test, verify it PASSES**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_inloop_reduce_coverage.py -q`
Expected: both tests PASS (`_mept0_refuses` now raises `MetalNonRecoverableError`; `_small_block_ok` still correct).

- [ ] **Step 6: Sanity — default flag still computes the eligible plain-sum correctly**

Run:
```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
python3 -c "
import torch, triton, triton.language as tl
from tests.test_inloop_reduce_coverage import _sum_carry_in_loop
X=torch.randn(4*256,device='mps',dtype=torch.float32); OUT=torch.zeros(1,device='mps',dtype=torch.float32)
_sum_carry_in_loop[(1,)](X,OUT,C=4,BLOCK=256)
print('default-flag 256:', 'OK' if torch.allclose(OUT[0],X.sum(),rtol=1e-3,atol=1e-3) else 'WRONG')
"
```
Expected: `default-flag 256: OK` (the eligible register-array path covers it; B must NOT fire here because `mept_arr` is set → `input_shape` is None).

- [ ] **Step 7: Commit**

```bash
git add tests/test_inloop_reduce_coverage.py triton_metal/codegen/generic_lowerer.py triton_metal/codegen/_lowerer_reduce.py
git commit -m "fix(reduce): refuse uncovered in-loop reduction (Stage B) instead of silent-wrong

A tt.reduce inside scf.for/if/while with block_size > num_threads and no
register-array cover silently summed only the first num_threads elements.
Add a control-flow depth counter and refuse loudly in _lower_reduce."
```

---

## Task 2: B verification + residual-surface measurement (this IS Stage A's measurement)

**Files:** none modified (measurement only; may add a skip to `scripts/conftest_metal.py` if a regression is found)

**Sequencing note:** after Stage B but before Stage C, the *default* flag may newly-refuse `cmpf`-bearing in-loop-reduce kernels (B catches them; C restores them correctly in Task 3). So the hard "0 failed" gate for the default flag lands at **Task 3 Step 7** (post-C). Here we *classify*. The `MEPT=0` flag has no such caveat (C is inert there).

- [ ] **Step 1: Full ratchet, default flag — classify any newly-refused tests**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -15`
Expected: at most a handful of new errors, each a `MetalNonRecoverableError` from the Stage-B message. For each, classify: (a) **cmpf-bearing** in-loop reduce → C will restore it (re-checked at Task 3 Step 7); (b) **other-ineligible** in-loop reduce that was previously *silently wrong* → correctly converted to loud (record for a conftest skip / Stage-A candidate); (c) **previously-passing-correctly** → impossible for this shape (would have been silent-wrong), so treat as a **B false-positive** and fix B's condition before continuing. If `0` new errors, even better.

- [ ] **Step 2: Full ratchet, escape hatch**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_MEPT=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -15`
Expected: `0 failed`. Because the `MEPT=0` ratchet passed before (and `MEPT=0` has no array cover), no currently-passing `test_core` kernel can be an uncovered in-loop-reduce-over-threads case — so B fires on nothing here. If any new error appears, it is a B false-positive (Step 1 case c) — investigate and fix before continuing.

- [ ] **Step 3: Project suite, both flags**

Run:
```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/ -q 2>&1 | tail -15
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_MEPT=0 python3 -m pytest tests/ -q 2>&1 | tail -15
```
Expected: `0 failed` both. (`test_inloop_reduce_where.py` — the prior select-fix regression — must still pass: those kernels are MEPT-eligible, so B does not fire.)

- [ ] **Step 4: Record the residual surface**

For any test that newly ERRORS with `MetalNonRecoverableError` from the Stage-B message in Steps 1-3, record it: test id, BLOCK, flag, and whether it was *previously passing* (= B false-positive → fix B's condition) or *previously silently-wrong* (= correctly converted to loud; add a conftest skip with a comment, and it becomes a Stage-A candidate). Write the list into the commit message of Step 5. If there are **zero** newly-refused tests, state that explicitly — it means B regresses nothing and Stage A's only consumers are out-of-corpus large-block kernels (tridec relay, the repro).

- [ ] **Step 5: Commit the measurement record (and any necessary skip)**

```bash
git add -A
git commit -m "test(reduce): Stage B ratchet green both flags; residual surface = <N tests / none>

<paste the residual list or 'no upstream test_core kernel hits the Stage-B
refusal under either flag; Stage-A consumers are out-of-corpus large-block
in-loop reduces (tridec relay, repro)'>"
```

---

## Task 3: C — array-wire `arith.cmpf`

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py` (`_MEPT_SAFE_OPS` ~line 82; the EXCLUDED comment ~lines 60-62/81; `_lower_cmpf` lines 3769-3820)
- Test: `tests/test_inloop_reduce_coverage.py`

- [ ] **Step 1: Write the failing test (default-flag where-on-reduce correctness)**

Append to `tests/test_inloop_reduce_coverage.py` (inside the `if HAS:` block, add the kernel; then the test):

```python
if HAS:
    @triton.jit
    def _min_blocksum_in_loop(X, OUT, C: tl.constexpr, BLOCK: tl.constexpr):
        best = tl.zeros((), dtype=tl.float32) + 1e30
        for i in range(0, C):
            v = tl.load(X + i * BLOCK + tl.arange(0, BLOCK))
            s = tl.sum(v)                       # in-loop reduce -> scalar
            best = tl.where(s < best, s, best)  # cmpf + select on reduce result
        tl.store(OUT + tl.arange(0, 1), best)


@requires_metal
@pytest.mark.parametrize("BLOCK", [128, 256, 512, 1024])
def test_inloop_where_on_reduce_default_correct(BLOCK):
    """Default flag: a where (cmpf+select) consuming an in-loop reduce result
    is register-array-eligible (Stage C) → correct at full SIMD width."""
    C = 4
    torch.manual_seed(0)
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    _min_blocksum_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)
    ref = X.view(C, BLOCK).sum(dim=1).min()
    torch.testing.assert_close(OUT[0], ref, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run, verify it FAILS at BLOCK 256/512/1024**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_inloop_reduce_coverage.py::test_inloop_where_on_reduce_default_correct -q`
Expected: BLOCK 128 PASSES; BLOCK 256/512/1024 FAIL (cmpf makes the kernel ineligible → currently silent-wrong OR, if Stage B's depth check catches it under the default flag... it does NOT: under the default flag the reduce is still ineligible, so today it is silent-wrong, and after Stage B it now REFUSES at 256+ under the default flag too). Either way: not yet correct → FAIL. This is the failing state Stage C fixes.

- [ ] **Step 3: Add `arith.cmpf` to `_MEPT_SAFE_OPS` and fix the comment**

In `generic_lowerer.py`, in the EXCLUDED comment (~line 60), remove `arith.cmpf` from the "Deliberately EXCLUDED" list. Update the comparison comment (~line 81) from `# comparison (only cmpi is array-wired; cmpf is NOT)` to `# comparison (cmpi and cmpf are array-wired)` and add `"arith.cmpf",` right after `"arith.cmpi",` (~line 82):

```python
    # comparison (cmpi and cmpf are array-wired)
    "arith.cmpi",
    "arith.cmpf",
```

- [ ] **Step 4: Add the MEPT array branch to `_lower_cmpf`**

Replace the body of `_lower_cmpf` (lines 3769-3820) with a version that factors the per-element expression into a closure (preserving the exact original scalar text so `MEPT=0` stays byte-identical) and adds the array dispatch, mirroring `_lower_cmpi`:

```python
    def _lower_cmpf(self, ssa: SSAValue):
        """arith.cmpf → float comparison with NaN-aware unordered predicates.

        pred_name (from MLIR text parsing) is the primary predicate source.
        pred_int is used as fallback only. Array path (MEPT) mirrors
        _lower_cmpi: a register-array operand yields a bool[N] per-position
        mask. The scalar fallback emits exactly the original text so MEPT=0
        codegen is byte-identical.
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])

        pred_name = ssa.attrs.get("predicate_name")
        pred_int = ssa.attrs.get("predicate")

        # Per-element boolean expression for operands (av, bv). Returns the
        # SAME unparenthesised text the scalar path historically emitted, so
        # `bool m = {_expr(a, b)};` is byte-identical to the old code.
        def _expr(av, bv):
            if pred_name == "false":
                return "false"
            if pred_name == "true":
                return "true"
            if pred_name == "uno":
                return f"isnan({av}) || isnan({bv})"
            if pred_name == "ord":
                return f"!isnan({av}) && !isnan({bv})"
            if pred_name == "une":
                return f"{av} != {bv}"
            if pred_name and pred_name in CMPF_NAMED:
                op_str = CMPF_NAMED[pred_name]
                if pred_name.startswith("u"):
                    return f"isnan({av}) || isnan({bv}) || ({av} {op_str} {bv})"
                return f"{av} {op_str} {bv}"
            if pred_int is not None and pred_int in CMPF_PREDICATES:
                return f"{av} {CMPF_PREDICATES[pred_int]} {bv}"
            return f"{av} < {bv}"

        # MEPT array path: if either operand is a register array, emit bool[N].
        if self._mept_binary_dispatch(
                ssa, ssa.operand_ids[0], ssa.operand_ids[1], a, b,
                _expr, "bool", "i1"):
            self.env_is_mask[ssa.id] = True
            return

        # Scalar fallback (byte-identical to the original emission).
        var_name = self._next_var("mask")
        self.kb.raw_line(f"    bool {var_name} = {_expr(a, b)};")
        self.env[ssa.id] = var_name
        self.env_is_mask[ssa.id] = True
        self.env_types[ssa.id] = "i1"
        self._propagate_shape_elementwise(ssa)
        self._propagate_bcast_layout_binary(ssa)
```

- [ ] **Step 5: Run the C test, verify it PASSES at all widths**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_inloop_reduce_coverage.py -q`
Expected: all tests PASS, including `test_inloop_where_on_reduce_default_correct` at 128/256/512/1024. (`test_inloop_reduce_mept0_refuses` still passes — under `MEPT=0` cmpf wiring is inert, so that kernel still refuses.)

- [ ] **Step 6: Verify MEPT=0 codegen is byte-identical for a cmpf kernel**

Run:
```bash
cat > /tmp/cmpf_parity.py <<'PY'
import torch, triton, triton.language as tl, sys
@triton.jit
def kf(X, OUT, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK); v = tl.load(X + o)
    tl.store(OUT + o, tl.where(v < 0.0, -v, v))
X=torch.randn(128,device="mps"); OUT=torch.zeros(128,device="mps")
print(kf.warmup(X, OUT, BLOCK=128, grid=(1,)).asm["msl"])
PY
rm -rf ~/.cache/triton_metal ~/.triton/cache; TRITON_METAL_MEPT=0 python3 /tmp/cmpf_parity.py > /tmp/cmpf_after.txt
git stash; rm -rf ~/.cache/triton_metal ~/.triton/cache; TRITON_METAL_MEPT=0 python3 /tmp/cmpf_parity.py > /tmp/cmpf_before.txt; git stash pop
diff /tmp/cmpf_before.txt /tmp/cmpf_after.txt && echo "BYTE-IDENTICAL"
```
Expected: `BYTE-IDENTICAL` (no diff). If the kernel above is MEPT-eligible at BLOCK=128 (one elem/thread, no array), the scalar cmpf path is exercised and must be unchanged.

- [ ] **Step 7: Ratchet both flags**

Run:
```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -8
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_MEPT=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -8
```
Expected: `>= 5531 passed, 0 failed` both directions. cmpf-wiring must not regress (it widens the eligible set; the risk is a too-wide safe set — watch `test_where`, masked-store, `test_reduce`/`test_argmin`/`test_argmax` families especially).

- [ ] **Step 8: Project suite both flags**

Run:
```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/ -q 2>&1 | tail -8
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_MEPT=0 python3 -m pytest tests/ -q 2>&1 | tail -8
```
Expected: `0 failed` both.

- [ ] **Step 9: Commit**

```bash
git add tests/test_inloop_reduce_coverage.py triton_metal/codegen/generic_lowerer.py
git commit -m "feat(mept): array-wire arith.cmpf (Stage C) — routes where-on-reduce to the correct path

cmpf was excluded from _MEPT_SAFE_OPS, forcing where-on-reduce kernels to the
ineligible (silent-under-covering) in-loop reduce path. Wire cmpf like cmpi
(bool[N] per-position mask, NaN-aware predicates preserved). The where-on-reduce
shape now computes at BLOCK 256/512/1024 under the default flag. MEPT=0 codegen
byte-identical. Full ratchet green both directions."
```

---

## Task 4: Stage A decision gate

**Files:** none (decision + report)

- [ ] **Step 1: Re-run the residual surface under the default flag, post-C**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -8`
Confirm `0 failed`. The default-flag residual after C = Task 2's default residual minus any cmpf-bearing kernels now eligible. Record the count.

- [ ] **Step 2: Decide**

- **If both the default-flag residual AND the `MEPT=0` residual (Task 2 Step 2) consist only of out-of-corpus large-block kernels (no real test refused):** Stage A's only consumers are large-block in-loop reduces not present in the corpus. Per the spec's YAGNI gate, **stop here** — document A as designed-but-unbuilt (the spec already contains the approach) and proceed to Task 7. Note in the report that `MEPT=0` large-block in-loop reduces refuse loudly (correct), and the default flag covers them via the register-array path.
- **If the `MEPT=0` ratchet or project suite has real kernels that now refuse (block>num_threads in-loop reduces that previously passed correctly):** those were correct before only if they were array-covered — under `MEPT=0` they cannot be, so a *previously-passing* `MEPT=0` test refusing means it was relying on small blocks; re-examine. If genuinely real consumers exist, **proceed to Task 5/6 to build Stage A**.

- [ ] **Step 3: Record the decision** in the Task 7 summary (build A or defer A, with the evidence).

---

## Task 5: A (conditional) — write the failing coverage test

> Only if Task 4 decided to build Stage A.

**Files:** Test: `tests/test_inloop_reduce_coverage.py`

- [ ] **Step 1: Add a failing MEPT=0 correctness test**

Append:

```python
@requires_metal
@pytest.mark.parametrize("BLOCK", [256, 512])
def test_inloop_reduce_mept0_correct(BLOCK, monkeypatch):
    """Stage A: under MEPT=0, an in-loop reduce with block>num_threads computes
    correctly via body-local multipass coverage (no longer refuses)."""
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")
    C = 4
    torch.manual_seed(0)
    X = torch.randn(C * BLOCK, device="mps", dtype=torch.float32)
    OUT = torch.zeros(1, device="mps", dtype=torch.float32)
    _sum_carry_in_loop[(1,)](X, OUT, C=C, BLOCK=BLOCK)
    torch.testing.assert_close(OUT[0], X.sum(), rtol=1e-3, atol=1e-3)
```

Also delete `test_inloop_reduce_mept0_refuses` (its kernel now computes correctly under A) — Stage B's refusal becomes a true backstop covered by a kernel A cannot handle, which there is no in-corpus example of; document the removal in the commit.

- [ ] **Step 2: Run, verify it FAILS (currently refuses)**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_inloop_reduce_coverage.py::test_inloop_reduce_mept0_correct -q`
Expected: FAIL with `MetalNonRecoverableError` (Stage B still refuses; A not yet built).

---

## Task 6: A (conditional) — body-local multipass coverage

> Only if Task 4 decided to build Stage A.

**Files:** Modify: `triton_metal/codegen/_lowerer_reduce.py` (`_lower_reduce`, replace the Stage-B refusal with coverage for the same condition)

- [ ] **Step 1: Replace the B refusal with body-local strided coverage**

Where Task 1 raised `MetalNonRecoverableError`, instead emit a per-thread strided accumulation over the reduce's input chain, then run the existing cross-thread reduce on the accumulator. Reuse `_get_reduce_combine_info` / `_reduce_identity_combine` for the identity and combine, mirror `_lower_multipass_reduction`'s `_loop_e` wrap and `_collect_tensor_deps` replay, but emit it inline at the in-loop reduce site:

```python
        if (mept_arr is None
                and self._control_flow_depth > 0
                and input_shape is not None
                and len(input_shape) == 1
                and input_shape[0] > self.kb.block_size):
            covered_var = self._cover_inloop_reduce(
                ssa, combine_op, msl_type, total=input_shape[0])
            if covered_var is not None:
                input_var = covered_var
                input_shape = None  # now one partial per thread
            else:
                from triton_metal.errors import MetalNonRecoverableError
                raise MetalNonRecoverableError(
                    "Refusing in-loop reduction the body-local multipass "
                    "cover could not handle (non-replayable input chain).")
```

Implement `_cover_inloop_reduce(self, ssa, combine_op, msl_type, total)` adjacent to `_mept_reduce_fold` in `_lowerer_reduce.py`:
- Collect the reduce input's dependency ops via `_collect_tensor_deps([<reduce input op>], all_preceding_in_body, known_scalars)`. The in-loop body op list is available from the enclosing `_lower_scf_for`; thread it via a new `self._current_loop_body_ops` set when `_control_flow_depth` is bumped (set it in `_lower_scf_for` before lowering body ops, restore after).
- Declare `T acc = <identity>;` (from `_reduce_identity_combine(combine_op, msl_type)`).
- Emit `for (uint _loop_e = lid; _loop_e < {total}u; _loop_e += {self.kb.block_size}u) {`, set `self._needs_wrapping = True`, re-emit the replayed input ops (the existing wrap machinery rewrites `lid → _loop_e`), accumulate `acc = combine(acc, <reduce input var>)`, close the loop, clear `_needs_wrapping`.
- Return `acc`.
- If the input chain cannot be replayed (e.g. it depends on a value not reconstructable per-element), return `None` so the caller refuses (Stage B backstop preserved).

This duplicates no machinery the codebase lacks; it extends the existing `_needs_wrapping`/`_collect_tensor_deps`/`_loop_e` wrap (which already produces correct multipass stores and top-level reduces) to the in-`scf.for` reduce site.

- [ ] **Step 2: Run the A test, verify PASS**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_inloop_reduce_coverage.py -q`
Expected: `test_inloop_reduce_mept0_correct` PASSES at 256/512; all others PASS.

- [ ] **Step 3: Ratchet both flags + project suite**

Run the four commands from Task 3 Steps 7-8.
Expected: `>= 5531 passed, 0 failed` (ratchet, both flags) and `0 failed` (project, both flags). Under `MEPT=0`, any in-corpus in-loop-reduce-over-threads kernel now computes instead of refusing.

- [ ] **Step 4: Commit**

```bash
git add triton_metal/codegen/_lowerer_reduce.py tests/test_inloop_reduce_coverage.py
git commit -m "feat(reduce): body-local multipass coverage for in-loop reductions (Stage A)

An in-loop reduce with block_size > num_threads on the non-array path now
covers the whole tile via a strided per-thread accumulation (the existing
_loop_e wrap + dep replay) before the cross-lane reduce, instead of refusing.
Correct under MEPT=0 at 256/512. Stage B remains the backstop for non-replayable
input chains."
```

---

## Task 7: Finalize — clean up, conftest, memory, tridec follow-up

**Files:**
- Delete: `tests/test_reduceresult_select_DIAG.py`
- Modify (if Task 2 found a real refused upstream test): `scripts/conftest_metal.py`
- Create: `docs/tridec-bug2-2ndshape-followup-2026-06-13.md`

- [ ] **Step 1: Remove the temporary diagnostic**

Run: `git rm -f tests/test_reduceresult_select_DIAG.py 2>/dev/null || rm -f tests/test_reduceresult_select_DIAG.py`
(Its assertions are absorbed into `tests/test_inloop_reduce_coverage.py`.)

- [ ] **Step 2: Final full regression, both flags**

Run:
```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -6
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_MEPT=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -6
rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/ -q 2>&1 | tail -6
```
Expected: `>= 5531 passed, 0 failed` (both ratchet directions); project `0 failed`.

- [ ] **Step 3: Write the tridec follow-up doc**

Create `docs/tridec-bug2-2ndshape-followup-2026-06-13.md` summarizing: the 2nd shape was root-caused to a *silent* in-loop-reduce under-coverage (not just a refusal); Stage B closes the silent-wrong, Stage C unblocks the `where`-on-reduce shape at full width under the default flag (relay can lift to 256+), and whether Stage A was built (per Task 4). Honest caveat: relay's *specific* kernel was already safe (refused); the fix removes the broader silent landmine and unblocks the shape.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(reduce): finalize in-loop reduce coverage — remove DIAG, tridec follow-up doc"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** B (Task 1), B-verify/residual-measurement (Task 2 = spec's Stage-A measurement), C (Task 3), A decision gate (Task 4), A impl (Tasks 5-6, conditional), loud-refusal contract (Task 1 + backstop in Task 6), ratchet both flags (every verify step), byte-identical MEPT=0 (Task 3 Step 6), test matrix incl. 128/256/512/1024 (Tasks 1, 3). All covered.
- **MEPT=0 scope:** captured — B refuses all uncovered in-loop reduces under `MEPT=0`; only A makes them correct there; C is default-flag-only (Task 3 Step 5 note, Task 4 decision).
- **Type/name consistency:** `_control_flow_depth`, `mept_arr`, `input_shape`, `self.kb.block_size`, `_mept_binary_dispatch(ssa, op0, op1, a, b, expr, "bool", "i1")`, `_reduce_identity_combine`, `_get_reduce_combine_info` used consistently and verified against the current source.
