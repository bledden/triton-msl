# Phase 1 Plan 1: Differential Harness + Family Allowlist + Elementwise Flip

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Data-driven C++ family allowlist + differential C++/Python harness; flip elementwise/cast default-on through C++.

**Architecture:** Extract `_has_complex_ops`'s op set into `cpp_families.py` (family→ops + ENABLED set). Routing: kernel ops ⊆ enabled-union → C++ by default (`TRITON_METAL_FORCE_PYTHON=1` escape). Differential = subprocess per path, byte-compare buffers.

**Tech Stack:** `/Users/bledden/Documents/triton-metal/.venv/bin/python`, pytest; caches `~/.cache/triton_metal` + `~/.triton/cache` cleared, serial GPU. C++ must be built; tests skip if `MetalBackend._has_cpp_passes()` false.

---

### Task 1: data-driven family table

**Files:** Create `triton_metal/backend/cpp_families.py`; Modify `compiler.py:268-330`; Test `tests/test_cpp_families.py`

- [ ] Step 1 failing test:
```python
import triton  # noqa
def test_families_cover_legacy_allowlist():
    from triton_metal.backend.cpp_families import FAMILIES, enabled_ops
    assert {"elementwise"} <= set(FAMILIES)
    assert "tt.load" in enabled_ops()
def test_router_uses_table():
    from triton_metal.backend.compiler import MetalBackend
    assert MetalBackend._has_complex_ops("  %0 = tt.fancy_unknown %a") is True
    assert MetalBackend._has_complex_ops("  %0 = tt.splat %a") is False
```
- [ ] Step 2 run → ImportError.
- [ ] Step 3: `cpp_families.py`: `FAMILIES = {"elementwise": {<every op now in compiler.py allowlist>}}`; `ENABLED = {"elementwise"}`; `def enabled_ops(): return set().union(*(FAMILIES[f] for f in ENABLED))`. In compiler.py replace the literal set with `from .cpp_families import enabled_ops` / `allowed_ops = enabled_ops()`.
- [ ] Step 4 pass; project suite 0 failed. Step 5 commit `refactor: data-driven cpp family allowlist`.

### Task 2: differential harness

**Files:** Create `tests/test_diff_cpp_python.py`
- [ ] Step 1 failing test (no harness yet):
```python
import os, subprocess, sys, tempfile, pytest
import triton  # noqa
from triton_metal.backend.compiler import MetalBackend
pytestmark = pytest.mark.skipif(not MetalBackend._has_cpp_passes(), reason="cpp not built")
KERNEL = '''import os,sys,torch,triton,triton.language as tl,numpy as np
os.environ.setdefault("TRITON_DEFAULT_BACKEND","metal")
@triton.jit
def k(X,O,N: tl.constexpr):
    i=tl.arange(0,N); x=tl.load(X+i); tl.store(O+i,(x*2.0+1.0).to(tl.float16))
x=torch.randn(256); o=torch.zeros(256,dtype=torch.float16)
k[(1,)](x,o,N=256); np.save(sys.argv[1],o.numpy())'''
def run(path, force):
    env=dict(os.environ,PYTHONPATH=os.getcwd(),TRITON_METAL_CACHE_DIR=tempfile.mkdtemp())
    env["TRITON_METAL_FORCE_PYTHON"]="1" if force else "0"
    subprocess.run([sys.executable,"-c",KERNEL,path],check=True,env=env,timeout=120)
def test_elementwise_matches():
    import numpy as np
    a=tempfile.mktemp('.npy'); b=tempfile.mktemp('.npy')
    run(a,True); run(b,False)
    np.testing.assert_array_equal(np.load(a),np.load(b))
```
- [ ] Step 2 run → fails (FORCE_PYTHON not honored; both Python).
- [ ] Step 3: Task 3 makes flag meaningful; this test passes after Task 3. Commit harness with `xfail(strict=False)` until then.

### Task 3: routing flip

**Files:** Modify `compiler.py:203-214`
- [ ] Step 1: replace env gate with: `use_cpp = (os.environ.get("TRITON_METAL_FORCE_PYTHON") != "1" and self._has_cpp_passes())` (drop USE_CPP requirement; allowlist + ops⊆enabled still routes).
- [ ] Step 2: remove xfail; differential passes. Project suite 0 failed.
- [ ] Step 3 commit `feat: elementwise family default-on through C++`.

### Task 4: refusal parity smoke
- [ ] Test: a refused kernel (fp16 atomic) still raises MetalNonRecoverableError default route. Commit.

### Gate: fresh-cache test_core 5,335/0; coverage `reports/cpp_coverage.json` = {"enabled":["elementwise"]} committed.
