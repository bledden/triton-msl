# Phase 0 Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repo-hygiene week from the pre-1.0 roadmap: versioned caches, no silent legacy fallback, no hangs, honest evidence at HEAD, clean tree.

**Architecture:** Small isolated fixes to compiler cache keys, emit_msl fallback policy, scf.for i64 guard, packaging, and docs. Every change TDD'd; gate = fresh-cache `test_core` 5,335/0.

**Tech Stack:** Python 3.14 project venv (`/Users/bledden/Documents/triton-msl/.venv/bin/python`), pytest, `scripts/run_upstream_test.sh`. Caches at `~/.cache/triton_msl` + `~/.triton/cache` — clear both, serial GPU only.

---

### Task 1: CODEGEN_VERSION in MSL cache key

**Files:** Modify `triton_msl/__init__.py`, `triton_msl/backend/compiler.py:1806-1808`; Test `tests/test_cache_versioning.py`

- [ ] Step 1: failing test:
```python
import hashlib
def test_msl_cache_key_includes_codegen_version():
    from triton_msl import CODEGEN_VERSION
    from triton_msl.backend.compiler import _msl_cache_key
    assert CODEGEN_VERSION in _msl_cache_key.__doc__ or _msl_cache_key("x", "h") != hashlib.sha256("xh".encode()).hexdigest()[:16]
```
- [ ] Step 2: run; expect ImportError (no CODEGEN_VERSION/_msl_cache_key).
- [ ] Step 3: add to `triton_msl/__init__.py`: `CODEGEN_VERSION = "2026.06.09"` (bump on lowerer change). In compiler.py replace inline sha with module func `_msl_cache_key(mod_text, opts_hash)` hashing `mod_text + opts_hash + CODEGEN_VERSION + os.environ.get("TRITON_MSL_MEPT","")`; call at 1806.
- [ ] Step 4: test passes; project suite 0 failed.
- [ ] Step 5: `git commit -m "fix: cache key includes CODEGEN_VERSION + MEPT flag"`

### Task 2: metallib caches versioned

**Files:** Modify `compiler.py:1599,1927` (src_hash sites)
- [ ] Step 1-3: append `+ CODEGEN_VERSION` into both `src_hash` inputs (same TDD pattern, assert digest differs from unversioned).
- [ ] Step 4-5: suite green; commit `fix: metallib cache key versioned`.

### Task 3: i64 scf.for refuses (was hang)

**Files:** Modify `triton_msl/codegen/_lowerer_control.py:_lower_scf_for`; Test `tests/test_int64_integrity.py` (append)
- [ ] Step 1: failing test: i64-bound `tl.range` kernel with 10s subprocess timeout → expect `MetalNonRecoverableError`, not timeout.
- [ ] Step 2: run; expect TIMEOUT.
- [ ] Step 3: in `_lower_scf_for`, when `self.env_types.get(start/end) == "i64"`, raise `MetalNonRecoverableError("i64 loop bounds not supported — would hang")`.
- [ ] Step 4: refusal raised; conftest skip for `test_for_iv` stays.
- [ ] Step 5: commit `fix: refuse i64 loop bounds instead of hanging`.

### Task 4: retire legacy fallback

**Files:** Modify `triton_msl/codegen/msl_emitter.py:528-556`; Test `tests/test_legacy_fallback_retired.py`
- [ ] Step 1: failing test: with `TRITON_MSL_LEGACY` unset, emit_msl on UNSUPPORTED graph raises `MetalNonRecoverableError`; with =1, returns MSL.
- [ ] Step 2: run; fails (currently falls back silently).
- [ ] Step 3: wrap legacy block in `if os.environ.get("TRITON_MSL_LEGACY") != "1": raise MetalNonRecoverableError(...)`.
- [ ] Step 4-5: project suite + sweep; commit `fix: legacy parser opt-in only`.

### Task 5: packaging + tree

**Files:** `pyproject.toml:36` → `triton==3.7.0`; `git rm --cached compile_commands.json triton_msl/csrc/.cache -r`; `.gitignore` add both.
- [ ] Step 1-2: `pip check`; commit `chore: pin triton 3.7.0, untrack build artifacts`.

### Task 6: evidence regeneration (after T1-5)

**Files:** delete `benchmarks/PERFORMANCE_METRICS.md`; regen `reports/upstream_test_core.txt` via `scripts/run_upstream_test.sh unit/language/test_core.py -q` (fresh cache); README counts/tutorials/FA-attribution.
- [ ] Step 1: clear caches → sweep; expect ≥5,335/0.
- [ ] Step 2: commit report; sync README; commit `docs: evidence at HEAD`.

### Task 7: name decision (doc only)

- [ ] Step 1: `docs/superpowers/specs/2026-06-09-name-decision.md` — `pip install triton-msl-backend`; README; gated to Phase 5.

### Gate: 5,335/0; project 0 failed; all tasks committed.
