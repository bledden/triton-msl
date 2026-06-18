# CHECKPOINT — start here to resume (last updated 2026-06-18)

Single "start here" pointer after a compaction / fresh session. Read this, confirm the
env, then begin **#1**.

## Where things stand
- `origin/main` @ **07dda07**. Worktree: `.claude/worktrees/multi-element-per-thread`
  (branch `worktree-multi-element-per-thread`). Run all commands from the worktree;
  merge to main via `git -C ~/Documents/triton-metal merge --ff-only worktree-multi-element-per-thread`.
- **Env (local, NOT in git):** python **3.14.4**, torch **2.12.1** (upgraded from 2.9.1 this
  session — `torch.compile` now works on 3.14; do NOT downgrade Python). Rollback if needed:
  `pip install --break-system-packages torch==2.9.1`.
- Health: project suite **754/0**; upstream `test_core` **5,559/0/~3,783** (skip-aware, via
  `scripts/run_upstream_tests.py`, which loads `-p conftest_metal`). FlashAttention causal +
  non-causal at head_dim 32/64/128 (128 = fp32+fp16 via the tiled template — shipped this session).
- Prime directive ALWAYS: never silent-wrong — refuse loudly or fall back, never emit a guessed
  kernel. Every push needs explicit user confirmation (local commits/merges are fine).

## Agreed priority order
**#1 inductor backend port (torch.compile coverage) → #3 training/backward → #4 incremental
op coverage → #2 PyPI publishing.**

## START WITH #1 — inductor backend port
**This is the real torch.compile-coverage work** (NOT "dynamic shapes" — runtime dims already
work for hand-written kernels; see the plan for why). The integration in
`triton_metal/inductor/` is bit-rotted against torch 2.12's inductor API.

- **Full plan:** `docs/superpowers/plans/2026-06-18-inductor-backend-port.md`
- **Diagnosis/context memory:** `project_torchcompile_inductor_state`
- **Current symptom:** `tests/test_torch_compile.py` = 6 passed / 26 failed; failures are
  `torch._inductor.exc.InductorError: NotImplementedError` at `torch/_inductor/codegen/common.py:319`
  (`self.get_backend(device).codegen_node(node)`). **Loud, not silent-wrong.**

### First concrete action (Task 1 of the plan): pin the failure
The torch.compile tests are still gated by a (now-stale) `skipif(sys.version_info >= (3,14))`.
Temporarily bypass it to reproduce, e.g. run one case directly:
```
cd ~/Documents/triton/python/test   # or use the project test; the kernel must be importable
WT=~/Documents/triton-metal/.claude/worktrees/multi-element-per-thread
TORCHDYNAMO_VERBOSE=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH="$WT" python3.14 -m pytest \
  "$WT/tests/test_torch_compile.py::TestModels::test_mlp" -x --no-header -p no:cacheprovider \
  -o addopts=""   # (and comment out the pytestmark skip in test_torch_compile.py to run it)
```
Then diff torch 2.12's `register_backend_for_device` / `TritonScheduling` / `get_backend`
contract against what `triton_metal/inductor/__init__.py::register_metal_triton_backend()`
passes — the registration signature/scheduling drift is the prime suspect. Goal of Task 1: a
one-paragraph root cause. Then proceed through the plan's Tasks 2–5 (fix registration → green
the tiers → dynamic shapes → un-gate the guards + verify; the guards' fix is in Task 5 — flip
`skipif(version>=3.14)` to a torch.compile-availability probe).

Gotchas: `TORCHINDUCTOR_COMPILE_THREADS=1` is required (Metal/PyObjC not fork-safe). Don't
commit the guard-flip until the inductor work is green (it would expose the 26 known-fails into
the suite). Bump `CODEGEN_VERSION` (`triton_metal/__init__.py`) on any codegen change + clear
caches with `find ~/.cache/triton_metal ~/.triton/cache -type f -delete` (not `rm -rf`).

## Also teed up
- **#4 (incremental coverage):** `docs/superpowers/plans/2026-06-18-2d-gather-coverage.md`
  (2D `tt.gather`, axis 0/1). The genuine coverage gaps are all real features — no quick wins.
- Refreshed `docs/ROADMAP.md` ("Current status" section) for the broader picture.
