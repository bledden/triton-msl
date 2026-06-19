# CHECKPOINT — start here to resume (last updated 2026-06-18)

Single "start here" pointer after a compaction / fresh session. Read this, confirm the
env, then begin **#3 (training / backward)** — #1 is DONE.

## Where things stand
- Worktree: `.claude/worktrees/multi-element-per-thread` (branch
  `worktree-multi-element-per-thread`). Run all commands from the worktree; merge to main via
  `git -C ~/Documents/triton-metal merge --ff-only worktree-multi-element-per-thread`.
- Branch is **2 commits ahead of origin/main @ 07dda07** (checkpoint `7fd4927` + inductor port
  `9eb6d31`) — **NOT pushed; not yet merged to main.** Push needs explicit user confirmation.
- **Env (local, NOT in git):** python **3.14.4**, torch **2.12.1** (do NOT downgrade Python).
  Rollback if needed: `pip install --break-system-packages torch==2.9.1`.
- Health: project suite **792/0** (was 754; +38 torch.compile + model tests now un-gated +
  passing); upstream `test_core` **5,559/0/~3,783** (skip-aware, via
  `scripts/run_upstream_tests.py`). FlashAttention causal + non-causal at head_dim 32/64/128.
  **`torch.compile` routes through triton-metal** (static + `dynamic=True`).
- Prime directive ALWAYS: never silent-wrong — refuse loudly or fall back, never emit a guessed
  kernel. Every push needs explicit user confirmation (local commits/merges are fine).

## Agreed priority order
~~#1 inductor backend port~~ **DONE** → **#3 training/backward (NEXT)** → #4 incremental op
coverage → #2 PyPI publishing.

## #1 — inductor / torch.compile coverage — DONE (2026-06-18, commit 9eb6d31)
Not a tier-by-tier port — it was a single registration-ordering bug + 3 latent silent-wrong
bugs exposed once torch.compile ran. Fixed: (1) torch-2.10+ native MPS device-op-override
clobber; (2) Metal fork-unsafe compile subprocesses + cache corruption (pin compile_threads=1);
(3) `_MSL_BY_NAME` cross-graph cache-key collision (re-key by content hash); (4) `triton_per_*`
softmax template mapping `xnumel`=row-count as row length → 4×-wrong reductions (refuse → generic).
Verified 32/32 + 6/6 (cold & warm), dynamic=True single-graph, full suite 792/0. See
`docs/superpowers/plans/2026-06-18-inductor-backend-port.md` (STATUS: DONE) + memory
`project_torchcompile_inductor_state`.

## START WITH #3 — training / backward pass (old Phase 5)
Biggest capability gap: triton-metal is **inference-only** today. See `docs/ROADMAP.md` Phase 5
(autograd.Function wrappers pairing forward + backward Triton kernels, registered so
`torch.compile` uses them for both passes; gradient buffer management; optimizers). Files:
new `triton_metal/training/`, `triton_metal/backend/driver.py` (gradient buffers),
`triton_metal/inductor/` (backward op registration), new `tests/test_training.py`.
First action: scope what a minimal `torch.autograd.Function` + backward-kernel pair looks like
for one op (e.g. a Linear or a matmul), verify the backward lowers + matches eager grads.

## Also teed up
- **#4 (incremental coverage):** `docs/superpowers/plans/2026-06-18-2d-gather-coverage.md`
  (2D `tt.gather`, axis 0/1). The genuine coverage gaps are all real features — no quick wins.
- Refreshed `docs/ROADMAP.md` ("Current status" + "Landed 2026-06-18") for the broader picture.

## Gotchas
- `compile_threads=1` for torch.compile is now enforced by the backend (Metal not fork-safe) —
  tests no longer set the env var.
- Bump `CODEGEN_VERSION` (`triton_metal/__init__.py`) on any codegen change + clear caches:
  `find ~/.cache/triton_metal ~/.triton/cache -type f -delete` AND the inductor cache at
  `$TMPDIR/torchinductor_$USER` when testing torch.compile cold paths.
