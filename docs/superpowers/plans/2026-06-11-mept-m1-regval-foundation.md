# MEPT Milestone 1: RegVal foundation + parity gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the unified `RegVal` register-array value abstraction + a region form-classifier + a flag-ON==flag-OFF parity differential gate, and prove scalar-collapse is byte-identical to today — WITHOUT changing any emitted MSL.

**Architecture:** New pure module `regval.py` (RegVal dataclass + `region_needs_arrays`). `_lookup` returns a `RegVal` (with a `.name` shim so legacy string callers are untouched). A `materialize` helper emits the scalar form (identical to `_var`) or array form (`_var_array`). Milestone 1 wires ONE op through it to prove the pattern; the broad emit-site migration is later milestones. All behind `TRITON_METAL_MEPT` (off by default); flag-OFF path is byte-unchanged.

**Tech Stack:** Python 3.14 venv `/Users/bledden/Documents/triton-metal/.venv/bin/python`; pytest. Clear `~/.cache/triton_metal ~/.triton/cache` before any codegen verification. GPU serial only.

---

### Task 1: RegVal + region classifier (pure module)

**Files:**
- Create: `triton_metal/codegen/regval.py`
- Test: `tests/test_regval.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_regval.py
from triton_metal.codegen.regval import RegVal, region_needs_arrays

class FakeOp:
    def __init__(self, op, operand_ids=(), region_ops=None):
        self.op = op; self.operand_ids = list(operand_ids); self.region_ops = region_ops

def test_regval_scalar_defaults():
    rv = RegVal(name="v0", n_elems=1, ty="float")
    assert rv.form == "scalar" and rv.is_scalar

def test_regval_array_form():
    rv = RegVal(name="v0", n_elems=4, ty="float", form="array")
    assert not rv.is_scalar and rv.n_elems == 4

def test_region_needs_arrays_straightline_false():
    ops = [FakeOp("tt.load"), FakeOp("arith.addf"), FakeOp("tt.store")]
    assert region_needs_arrays(ops, multi_elem_ids={1, 2}) is False

def test_region_needs_arrays_data_dependent_for_true():
    body = [FakeOp("tt.load"), FakeOp("arith.addf", operand_ids=[7])]
    ops = [FakeOp("scf.for", operand_ids=[7], region_ops=body)]
    assert region_needs_arrays(ops, multi_elem_ids={7}) is True

def test_region_needs_arrays_for_without_multielem_false():
    body = [FakeOp("arith.addi")]
    ops = [FakeOp("scf.for", region_ops=body)]
    assert region_needs_arrays(ops, multi_elem_ids=set()) is False
```

- [ ] **Step 2: Run, expect ImportError**
Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_regval.py -q`
Expected: FAIL (no module `regval`).

- [ ] **Step 3: Implement**
```python
# triton_metal/codegen/regval.py
"""Unified per-thread value model for the MEPT register-array spine.

Every SSA value is a RegVal(name, n_elems, ty, form). Scalars are n_elems==1,
form='scalar'. form selects emission: 'scalar' (no loop), 'wraploop' (the
existing _loop_e re-execution loop), or 'array' (T name[n_elems], the only
correct form when per-element state crosses data-dependent control flow).
See docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md.
"""
from dataclasses import dataclass

_CONTROL_OPS = ("scf.for", "scf.while", "scf.if")


@dataclass
class RegVal:
    name: str
    n_elems: int = 1
    ty: str = ""
    form: str = "scalar"  # 'scalar' | 'wraploop' | 'array'

    @property
    def is_scalar(self) -> bool:
        return self.n_elems == 1 and self.form == "scalar"


def region_needs_arrays(ops, multi_elem_ids) -> bool:
    """True if `ops` contains a data-dependent control-flow op (scf.for/
    while/if) whose body references or carries a multi-element value.

    Such regions cannot use the re-execution wrap-loop (it can't carry
    per-element state across the control-flow loop); they require true
    register arrays. `multi_elem_ids` is the set of SSA ids with n_elems>1.
    """
    multi = set(multi_elem_ids)
    for op in ops:
        if op.op in _CONTROL_OPS:
            body = op.region_ops or []
            if any(oid in multi for oid in (op.operand_ids or [])):
                return True
            for b in body:
                if any(oid in multi for oid in (getattr(b, "operand_ids", None) or [])):
                    return True
                if getattr(b, "id", None) in multi:
                    return True
            # recurse into nested regions
            if region_needs_arrays(body, multi):
                return True
    return False
```

- [ ] **Step 4: Run, expect PASS**
Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_regval.py -q` → 5 passed.

- [ ] **Step 5: Commit**
```bash
git add triton_metal/codegen/regval.py tests/test_regval.py
git commit -m "feat(mept-m1): RegVal value model + region_needs_arrays classifier

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `_lookup_regval` returning RegVal (non-invasive)

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py` (add method near `_lookup`, line ~1627)
- Test: `tests/test_regval.py` (append)

- [ ] **Step 1: Write failing test** (append to tests/test_regval.py)
```python
def test_lookup_regval_scalar_and_array():
    import triton  # noqa: F401
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    lo = GenericLowerer.__new__(GenericLowerer)
    lo.env = {5: "v5"}; lo.env_array = {6: ("a6", 4, "float")}
    lo.env_n_elems = {5: 1, 6: 4}; lo.env_types = {5: "i32", 6: "f32"}
    s = lo._lookup_regval(5)
    assert s.name == "v5" and s.n_elems == 1 and s.is_scalar
    a = lo._lookup_regval(6)
    assert a.name == "a6" and a.n_elems == 4 and a.form == "array"
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: _lookup_regval`)
Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_regval.py::test_lookup_regval_scalar_and_array -q`

- [ ] **Step 3: Implement** — add after `_lookup_array` (line ~1648):
```python
    def _lookup_regval(self, ssa_id):
        """Unified RegVal view over env / env_array. Does not change emission;
        callers migrate to this incrementally (MEPT spine, milestone 1)."""
        from triton_metal.codegen.regval import RegVal
        if ssa_id in getattr(self, "env_array", {}):
            name, n, ty = self.env_array[ssa_id]
            return RegVal(name=name, n_elems=n, ty=ty, form="array")
        name = self.env.get(ssa_id, f"UNKNOWN_{ssa_id}")
        n = self.env_n_elems.get(ssa_id, 1)
        ty = self.env_types.get(ssa_id, "")
        return RegVal(name=name, n_elems=n, ty=ty, form="scalar")
```

- [ ] **Step 4: Run, expect PASS**
Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_regval.py -q` → 6 passed.

- [ ] **Step 5: Commit**
```bash
git add triton_metal/codegen/generic_lowerer.py tests/test_regval.py
git commit -m "feat(mept-m1): _lookup_regval unified view over env/env_array

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: flag-ON==flag-OFF parity differential harness (the gate)

**Files:**
- Create: `tests/test_mept_parity.py`

- [ ] **Step 1: Write the test** (it must PASS now — flag changes nothing yet — and stay green every milestone)
```python
"""MEPT parity gate: flag-ON emitted MSL must equal flag-OFF on the scalar
corpus. The ratchet invariant — the unified model must reproduce today's output
byte-for-byte until a milestone deliberately unlocks new behavior."""
import importlib, os
import pytest
import triton  # noqa: F401
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir


def _emit(fn, sig, cst, mept):
    os.environ["TRITON_METAL_FORCE_PYTHON"] = "1"
    os.environ["TRITON_METAL_MEPT"] = "1" if mept else "0"
    import triton_metal.codegen.generic_lowerer as G
    import triton_metal.codegen.msl_emitter as M
    importlib.reload(G); importlib.reload(M)
    from triton_metal.backend.compiler import MetalBackend
    t = GPUTarget("metal", "apple-m4", 32); be = MetalBackend(t); o = be.parse_options({})
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    ctx = ir.context(); ir.load_dialects(ctx)
    mod = src.make_ir(t, o, be.get_codegen_implementation(o), be.get_module_map(), ctx)
    meta = {}
    mod = be.make_ttir(mod, meta, o); mod = be.make_ttgir(mod, meta, o)
    return M.emit_msl(mod, meta, o)


if True:
    @triton.jit
    def _vadd(X, Y, O, N: tl.constexpr):
        i = tl.arange(0, N); tl.store(O + i, tl.load(X + i) + tl.load(Y + i))

    @triton.jit
    def _vmul_scalar(X, O, N: tl.constexpr):
        i = tl.arange(0, N); tl.store(O + i, tl.load(X + i) * 3.0 + 1.0)


@pytest.mark.parametrize("fn,sig,cst", [
    (_vadd, {"X": "*fp32", "Y": "*fp32", "O": "*fp32"}, dict(N=256)),
    (_vmul_scalar, {"X": "*fp32", "O": "*fp32"}, dict(N=256)),
])
def test_mept_flag_parity_scalar_corpus(fn, sig, cst):
    off = _emit(fn, sig, cst, mept=False)
    on = _emit(fn, sig, cst, mept=True)
    assert on == off, "MEPT flag changed scalar MSL:\n--- OFF ---\n%s\n--- ON ---\n%s" % (off, on)
```

- [ ] **Step 2: Run, expect PASS** (flag is inert on scalar kernels today)
Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_mept_parity.py -q` → 2 passed.
If it FAILS, the flag already diverges on scalars — investigate before proceeding (do not weaken the test).

- [ ] **Step 3: Commit**
```bash
git add tests/test_mept_parity.py
git commit -m "test(mept-m1): flag-ON==flag-OFF parity gate on scalar corpus

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `_materialize` helper (scalar-collapse + array form), proven in isolation

> Scope note: Milestone 1 proves the helper as a unit and keeps the parity gate green; wiring live ops (tt.splat, elementwise) through `_materialize` is Milestone 2+, where the array form is actually exercised. Keeping it isolated here preserves "no new behavior."

**Files:**
- Modify: `triton_metal/codegen/generic_lowerer.py`
- Test: `tests/test_regval.py` (append)

- [ ] **Step 1: Write failing unit test** for the helper
```python
def test_materialize_scalar_collapses_to_plain_var():
    import triton  # noqa: F401
    from triton_metal.codegen.generic_lowerer import GenericLowerer
    from triton_metal.codegen.regval import RegVal
    lo = GenericLowerer.__new__(GenericLowerer)
    lo.kb = type("KB", (), {"lines": [], "raw_line": lambda self, s: self.lines.append(s)})()
    lo._var_counter = 0
    rv = lo._materialize(RegVal(name="", n_elems=1, ty="float", form="scalar"),
                         lambda e: "a + b", base="m")
    assert rv.n_elems == 1 and rv.is_scalar
    joined = "\n".join(lo.kb.lines)
    assert "float m0 = a + b;" in joined and "for (" not in joined  # no array, no loop
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: _materialize`)

- [ ] **Step 3: Implement `_materialize`** (add near `_var_array`, line ~199):
```python
    def _materialize(self, regval, body, base="t"):
        """Emit the cheapest correct form for a value and return its RegVal.

        body(e) -> MSL expression string for element index e.
        scalar (n_elems==1): 'ty name = body(0);' (identical to _var) — the
        scalar-collapse that keeps the common path byte-identical to today.
        array: 'ty name[n]; for-each e: name[e] = body(e);' via _var_array.
        wraploop is handled by callers that already emit the _loop_e loop;
        here body(0) is emitted once inside that loop (scalar-shaped).
        """
        from triton_metal.codegen.regval import RegVal
        name = "%s%d" % (base, self._var_counter); self._var_counter += 1
        if regval.form == "array" and regval.n_elems > 1:
            arr = self._var_array(name, [body(e) for e in range(regval.n_elems)], regval.ty)
            return RegVal(name=arr, n_elems=regval.n_elems, ty=regval.ty, form="array")
        # scalar / wraploop: single expression, no array, no extra loop
        self.kb.raw_line("    %s %s = %s;" % (regval.ty, name, body(0)))
        return RegVal(name=name, n_elems=1, ty=regval.ty, form=regval.form)
```

- [ ] **Step 4: Run unit test, expect PASS**
Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_regval.py -q` → 7 passed.

- [ ] **Step 5: Verify parity gate still green + project suite**
```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_mept_parity.py tests/test_regval.py -q
PYTHONPATH=$PWD TRITON_DEFAULT_BACKEND=metal .venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```
Expected: parity 2 passed, regval 7 passed; project suite 0 failed.

- [ ] **Step 6: Commit**
```bash
git add triton_metal/codegen/generic_lowerer.py tests/test_regval.py
git commit -m "feat(mept-m1): _materialize helper (scalar-collapse + array form)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Gate (end of Milestone 1)
- `tests/test_regval.py` + `tests/test_mept_parity.py` green.
- Project suite 0 failed.
- Fresh-cache flag-OFF `test_core` unchanged at 5,335/0 (the working path is byte-untouched):
  `rm -rf ~/.cache/triton_metal ~/.triton/cache && scripts/run_upstream_test.sh unit/language/test_core.py -q -p no:cacheprovider`
- Deliverable: the RegVal model + classifier + parity gate + `_materialize` exist and are proven on the scalar path; NO new behavior. Milestone 2 (scf.for array-carry → tridec Bug 2) builds on this.
