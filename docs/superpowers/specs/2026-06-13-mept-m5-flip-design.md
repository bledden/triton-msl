# MEPT Milestone 5 — flip the register-array default ON (design, 2026-06-13)

> Make the MEPT register-array model the default codegen path, keeping a
> `TRITON_METAL_MEPT=0` escape hatch to the scalar/wrap-loop path. The flip gate
> is already met: flag-ON full upstream `test_core` = 5335/0 (identical to
> flag-OFF) and flag-ON project suite = 619/0, with the newly-unlocked kernels
> (tridec Bug 2 ≥256, loop-carried arrays, >1024) all computing. Python/MSL per
> the 2026-06-11 language decision.

## Context: why the flip is ready
The spine (M1–M3c) made the register-array model produce correct results across
the entire corpus when enabled. Measured 2026-06-13:
- flag-ON upstream `test_core`: **5335 passed / 4007 skipped / 0 failed** — byte-identical to the flag-OFF baseline.
- flag-ON project suite: **619 passed / 0 failed** (after fixing a `FORCE_PYTHON` test-isolation leak, commit `3574aba` — not a codegen issue).
- flag-ON MEPT GPU tests (Bug 2, iter-carry, >1024): all pass.

The spine's correctness goals (Bug 2, chained reductions, >1024 1-D, FA HEAD_DIM=64
— the last already worked via the smem accumulator path) are achieved. The flip is
the payoff: the register-array model becomes the general default.

## Decisions (locked in brainstorm)
- **Default ON + `TRITON_METAL_MEPT=0` escape hatch.** Absent ⇒ register-array model
  ON; `=0` ⇒ scalar/wrap-loop kill-switch. Reversible per-run; the scalar path stays
  as a tested reference. (Not chosen: removing the flag entirely — no escape hatch,
  larger/riskier; not chosen: renaming to an opt-out flag — extra churn.)
- **Detector retirement deferred** to a separate follow-up. M5 is the flip only.

## Architecture — the flip mechanic
Invert the default at *every* flag read so the effective value is ON unless
explicitly disabled:
```python
self.mept_enabled = os.environ.get("TRITON_METAL_MEPT", "1") != "0"
```
- `generic_lowerer.py` (~line 186): `self.mept_enabled` read.
- `compiler.py` `_msl_cache_key`: the cache key must key on the **effective** flag
  (same default logic), so a default run and an explicit `TRITON_METAL_MEPT=1` run
  share a key, and `=0` gets a distinct key.
- Any other read site (the implementation plan greps `TRITON_METAL_MEPT` and flips
  all of them consistently — a partial flip would desync the lowerer from the cache
  key and risk serving a stale-default metallib).

## Cache invalidation
Bump `CODEGEN_VERSION` in `triton_metal/__init__.py` (currently `"2026.06.09"`).
The default codegen changes for MEPT-eligible kernels; bumping the version (a cache
key component) guarantees no metallib compiled under the old default is served.
Belt-and-suspenders alongside the flag-in-key, since silent-wrong from a stale
cache is the worst failure mode.

## Components (files)
- `triton_metal/codegen/generic_lowerer.py` — flip the `mept_enabled` default.
- `triton_metal/backend/compiler.py` — flip the cache-key flag default to match.
- `triton_metal/__init__.py` — bump `CODEGEN_VERSION`.
- `tests/test_unknown_value_backstop.py` — reframe docstring/comments: the
  `_force_mept_off` fixture now documents the **escape-hatch** refusal (`MEPT=0` ⇒
  BLOCK≥256 still refuses). Assertions unchanged.
- `tests/test_mept_m5_default_gpu.py` (new) — lock the **new default**: with NO env
  var set, tridec Bug 2 at BLOCK=256/512 *computes* `X.sum()` correctly (the inverse
  of the escape-hatch refusal test). Serial GPU.
- `docs/...` + memory — mark M5 done.

## Data flow (unchanged emission; only the default route changes)
A MEPT-eligible kernel with no env var now takes the single-pass register-array
path it previously took only under `TRITON_METAL_MEPT=1`. A non-eligible kernel is
unaffected (eligibility still gates emission). `TRITON_METAL_MEPT=0` restores the
exact pre-flip scalar/wrap-loop route.

## Error handling / integrity
Unchanged. The `UNKNOWN_<id>` backstop stays (refuse loud, never silent-wrong). The
escape hatch preserves the scalar reference path. The parity gate
(`test_mept_parity`, `_emit(mept=False/True)`) continues to prove scalar-collapse
equivalence regardless of the default.

## Testing / verification (the gate — both directions must be green)
- **New default (no env var):** full upstream `test_core` = 5335/0; project suite
  = 619/0; Bug 2 / iter-carry / >1024 GPU tests compute. (These GPU tests currently
  require `TRITON_METAL_MEPT=1`; post-flip they pass with no env too — the new
  `test_mept_m5_default_gpu.py` asserts the no-env default explicitly.)
- **Escape hatch (`TRITON_METAL_MEPT=0`):** full upstream `test_core` = 5335/0;
  project suite = 619/0; `test_unknown_value_backstop` refuses at BLOCK≥256.
- Parity gate green.
- Fresh dual-cache-clear before each direction (codegen-sensitive). Serial GPU.

## Risks
- **Stale cache serving old-default metallib** → mitigated by the `CODEGEN_VERSION`
  bump + the flag-in-key.
- **A flag-ON regression not covered by test_core/project suite** → the escape hatch
  (`MEPT=0`) is the per-run rollback; the scalar path remains fully tested.
- **Partial flag-flip (lowerer vs cache key desync)** → the plan flips all read sites
  in one change and verifies both directions.

## Out of scope (follow-ups)
- Retiring MEPT-gap detectors (`_detect_permute_chained_reduce`, matmul→softmax) the
  spine subsumes — separate milestone, verify each is truly redundant first.
- CI matrix running both default and `MEPT=0` (the language decision's "harden the
  MSL path" investment) — separate.
