# CHECKPOINT — start here to resume (last updated 2026-06-18)

Single "start here" pointer after a compaction / fresh session. Read this, confirm the
env, then begin **#2 (PyPI publishing)** — #1, #3 DONE; #4 substantially done (2D gather
landed; remaining coverage items are safely refused, not blocking).

## Where things stand
- Worktree: `.claude/worktrees/multi-element-per-thread` (branch
  `worktree-multi-element-per-thread`). Run all commands from the worktree; merge to main via
  `git -C ~/Documents/triton-metal merge --ff-only worktree-multi-element-per-thread`.
- `origin/main` @ **2ce413b** (#1 + #3 incl. autotune root-fix + #4 gather-refusal all
  merged/pushed). Branch is **ahead** with **2D-gather implementation `1e904e8`** (#4) —
  **NOT pushed/merged.** Push needs explicit user confirmation.
- **#4 2D tt.gather IMPLEMENTED (`1e904e8`):** axis=0 (incl. ragged row counts) + same-shape
  axis=1, via full-tile shared staging; upstream `test_gather[[4,4]->[8,4],0]` now PASSES
  (+1 conformance). Refused loudly (HW/scope): tiles >1024 threads, ragged axis=1, MEPT operands.
  Also fixed the walker to parse the gather `axis` attr. Tests: `tests/test_gather_2d.py` (10).
- **#3 autotune silent-wrong is now ROOT-FIXED (`fac6176`), not just mitigated.** The exact
  culprit was an under-filling persistent-reduction config (`__safe_softmax` XBLOCK=1/rnumel=16/
  num_warps=2 → unmasked duplicate lanes → all-zero softmax → corrupt grads). The filter drops
  `XBLOCK*rnumel < num_warps*32` configs so it can't be generated; verified autotune-ON 15/15 +
  a force-every-config correctness scan = 0 divergent. autotune-off (`77bd87d`) stays as
  determinism + blanket safety, but the bug class is now structurally unreachable regardless.
- KNOWN unrelated flake: `test_fast_matmul_perf::test_fast_matmul_throughput[fp16]` dips below its
  5.5 TFLOP/s floor under **thermal throttling** after long test runs (whole matmul baseline was
  ~30% depressed: fp32 7.92 vs ~10-12 cool). Hand-written compile_shader path, NOT torch.compile —
  unaffected by any inductor/codegen change here. Re-run cool to confirm; do not lower the floor.
- **Env (local, NOT in git):** python **3.14.4**, torch **2.12.1** (do NOT downgrade Python).
  Rollback if needed: `pip install --break-system-packages torch==2.9.1`.
- Health: project suite **799/0** (was 754; +38 torch.compile + model tests, +7 training tests,
  all un-gated); upstream `test_core` **5,559/0/~3,783** (skip-aware, via
  `scripts/run_upstream_tests.py`). FlashAttention causal + non-causal at head_dim 32/64/128.
  **`torch.compile` routes through triton-msl** — inference + training, static + `dynamic=True`.
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

## #3 — training / backward — DONE (2026-06-18, commits cb4fac8 + 77bd87d)
Falls out of the inductor port: AOTAutograd's backward graph is just more Triton kernels that
lower through triton-msl. MLP/CNN/transformer (w/ embedding) train + converge + match eager
(`tests/test_training.py`). Two backward bugs fixed: (1) `embedding_dense_backward`'s grad
zero-init (masked MEPT store of a constant) emitted a malformed `ptr[off][lid]` — MEPT scatter
now broadcasts splat/constant values; (2) **a nondeterministic wrong `head.weight` gradient
(~0.11 on ~1/4 cold runs)** — inductor autotuning was selecting miscompiled/timing-nondeterministic
tile configs on Metal; fixed by `autotune_pointwise = False` in the backend (77bd87d). The
"custom autograd.Function wrappers" framing is obsolete. Remaining sub-items (NOT blocking):
`torch.compile`-d optimizers, grad checkpointing, larger real-dataset training runs.

## #4 — incremental op coverage — 2D GATHER DONE
2D `tt.gather` silent-wrong closed (97a82b3) THEN implemented (1e904e8): axis=0 (incl. ragged
rows) + same-shape axis=1 via full-tile shared staging; upstream `[4,4]->[8,4]` passes.
Remaining #4 coverage gaps (all refuse loudly today — safe to defer, never silent-wrong):
2D-gather tiles >1024 threads (strided multi-pass) + ragged axis=1; noinline-dot (1E);
`tl.range` loop fusion (1F); multi-program atomics / cooperative sync (1G); rank-≥2 cat/join;
the i64 loop-induction-var hang; unstructured control flow (`cf.cond_br`). See `docs/ROADMAP.md`
"Remaining" #4. None block release.

## NEXT: #2 — PyPI publishing
The agreed order's last item. Package + publish the wheel (roadmap 6E). See `docs/ROADMAP.md`
distribution section. First action: audit `pyproject.toml` (metadata, deps, version), build the
wheel, test-install in a clean venv, then TestPyPI before PyPI. Pin torch 2.12 once green.

## Also note
- Refreshed `docs/ROADMAP.md` ("Current status" + "Landed 2026-06-18") + `docs/SUPPORTED_OPS.md`
  ("Framework integration": torch.compile + training) for the broader picture.

## Gotchas
- `compile_threads=1` for torch.compile is now enforced by the backend (Metal not fork-safe) —
  tests no longer set the env var.
- Bump `CODEGEN_VERSION` (`triton_msl/__init__.py`) on any codegen change + clear caches:
  `find ~/.cache/triton_msl ~/.triton/cache -type f -delete` AND the inductor cache at
  `$TMPDIR/torchinductor_$USER` when testing torch.compile cold paths.
