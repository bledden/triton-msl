# Phase 1: C++ port with corpus ratchet — design (2026-06-09)

> Roadmap phase 1 of `2026-06-09-pre1.0-port-first-roadmap-design.md`.
> Goal: the C++ MLIR path becomes the default lowering for the full corpus;
> the Python MSL emitter freezes as the differential-testing oracle.

## Decisions
- Routing: **C++ default-on per family**. A family ships to the allowlist the
  moment its corpus slice + differential gate pass. Python serves unported
  families during transition; `TRITON_MSL_FORCE_PYTHON=1` escape hatch.
- Refusal parity: every refusal the Python path raises must also refuse in
  C++ (consume refusal_catalog `--json` contract).
- Ratchet: full fresh-cache `test_core` 5,335/0 holds at every family flip;
  count only moves up. C++ coverage tracked in `reports/cpp_coverage.json`.

## Components
1. **Differential harness** (`tests/test_diff_cpp_python.py`): compile + run
   the same kernel via both paths, compare buffers exactly (fp tol per dtype).
2. **Family order**: elementwise/cast → load/store/mask → reduce/scan →
  atomics → softmax/layernorm templates → dot (reuses MMA work).
3. **Allowlist** (`compiler.py _has_complex_ops`) refactored to data-driven
   per-family table; CI asserts table == coverage report.

## Gate
test_core 5,335/0 through C++ for allowlisted families on every commit;
differential pass on full corpus; Python frozen.
