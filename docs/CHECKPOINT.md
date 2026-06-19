# CHECKPOINT — start here to resume (last updated 2026-06-18)

Single "start here" pointer after a compaction / fresh session. Read this, confirm the
env, then begin **#4 (incremental op coverage)** — #1 and #3 are DONE.

## Where things stand
- Worktree: `.claude/worktrees/multi-element-per-thread` (branch
  `worktree-multi-element-per-thread`). Run all commands from the worktree; merge to main via
  `git -C ~/Documents/triton-metal merge --ff-only worktree-multi-element-per-thread`.
- `origin/main` @ **1ab416f** (the #1 inductor port is merged + pushed). Branch is **1 commit
  ahead**: training `cb4fac8` (#3) — **NOT pushed/merged.** Push needs explicit user confirmation.
- **Env (local, NOT in git):** python **3.14.4**, torch **2.12.1** (do NOT downgrade Python).
  Rollback if needed: `pip install --break-system-packages torch==2.9.1`.
- Health: project suite **799/0** (was 754; +38 torch.compile + model tests, +7 training tests,
  all un-gated); upstream `test_core` **5,559/0/~3,783** (skip-aware, via
  `scripts/run_upstream_tests.py`). FlashAttention causal + non-causal at head_dim 32/64/128.
  **`torch.compile` routes through triton-metal** — inference + training, static + `dynamic=True`.
- Prime directive ALWAYS: never silent-wrong — refuse loudly or fall back, never emit a guessed
  kernel. Every push needs explicit user confirmation (local commits/merges are fine).

## Agreed priority order
~~#1 inductor backend port~~ **DONE** → ~~#3 training/backward~~ **DONE** → **#4 incremental op
coverage (NEXT)** → #2 PyPI publishing.

## #1 — inductor / torch.compile coverage — DONE (2026-06-18, commit 9eb6d31)
Not a tier-by-tier port — it was a single registration-ordering bug + 3 latent silent-wrong
bugs exposed once torch.compile ran. Fixed: (1) torch-2.10+ native MPS device-op-override
clobber; (2) Metal fork-unsafe compile subprocesses + cache corruption (pin compile_threads=1);
(3) `_MSL_BY_NAME` cross-graph cache-key collision (re-key by content hash); (4) `triton_per_*`
softmax template mapping `xnumel`=row-count as row length → 4×-wrong reductions (refuse → generic).
Verified 32/32 + 6/6 (cold & warm), dynamic=True single-graph, full suite 792/0. See
`docs/superpowers/plans/2026-06-18-inductor-backend-port.md` (STATUS: DONE) + memory
`project_torchcompile_inductor_state`.

## #3 — training / backward — DONE (2026-06-18, commit cb4fac8)
Falls out of the inductor port: AOTAutograd's backward graph is just more Triton kernels that
lower through triton-metal. MLP/CNN/transformer (w/ embedding) train + converge + match eager
(`tests/test_training.py`). Fixed one backward-only gap: `embedding_dense_backward`'s grad
zero-init (masked MEPT store of a constant) emitted a malformed `ptr[off][lid]` — MEPT scatter
now broadcasts splat/constant values. The "custom autograd.Function wrappers" framing is
obsolete. Remaining sub-items (NOT blocking): `torch.compile`-d optimizers, grad checkpointing,
larger real-dataset training runs.

## START WITH #4 — incremental op coverage
The genuine `test_core` coverage gaps are all real features (each refuses loudly today — safe to
defer, never silent-wrong). Highest-value pick: **2D `tt.gather`** (axis 0/1; gather/embedding
lookups are common). Full executable plan:
`docs/superpowers/plans/2026-06-18-2d-gather-coverage.md`. Other gaps: noinline-dot (1E),
`tl.range` loop fusion (1F), multi-program atomics / cooperative sync (1G), rank-≥2 cat/join,
the i64 loop-induction-var hang, unstructured control flow (`cf.cond_br`). See `docs/ROADMAP.md`
"Remaining" #4. First action: follow the 2D-gather plan's Task 1 (test-first, axis=0 smallest
shape) — refuse-on-budget-overflow, never silent-wrong.

## Also note
- Refreshed `docs/ROADMAP.md` ("Current status" + "Landed 2026-06-18") + `docs/SUPPORTED_OPS.md`
  ("Framework integration": torch.compile + training) for the broader picture.

## Gotchas
- `compile_threads=1` for torch.compile is now enforced by the backend (Metal not fork-safe) —
  tests no longer set the env var.
- Bump `CODEGEN_VERSION` (`triton_metal/__init__.py`) on any codegen change + clear caches:
  `find ~/.cache/triton_metal ~/.triton/cache -type f -delete` AND the inductor cache at
  `$TMPDIR/torchinductor_$USER` when testing torch.compile cold paths.
