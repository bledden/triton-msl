# Inductor backend port → torch.compile coverage (the real roadmap #1)

> **STATUS: DONE (2026-06-18).** Executed this session. The "bit-rot needs tier-by-tier port"
> hypothesis was wrong: the breakage was a **single registration-ordering bug** (torch 2.12's
> native MPS device-op-overrides clobbering ours), plus three latent silent-wrong bugs exposed
> once torch.compile ran (Metal fork-unsafe compile subprocesses; `_MSL_BY_NAME` cross-graph
> cache-key collision; the `triton_per_*` softmax template mis-mapping `xnumel`=row-count as the
> row length → 4×-wrong reductions). All fixed + verified: **32/32 torch.compile, 6/6 real-model
> (cold & warm), dynamic=True single-graph, full project suite 792/0.** See ROADMAP "Landed
> 2026-06-18" and SUPPORTED_OPS "Framework integration". Tasks below are kept as the audit trail.

> **Post-compaction handoff plan (2026-06-18).** Self-contained: a fresh session should
> read this + memory `project_torchcompile_inductor_state` and execute. This supersedes the
> "dynamic shapes (1H)" framing — see "Why this is #1" below.

**Goal:** restore `torch.compile(model, backend="inductor")` coverage on Metal under the
current torch (2.12.1) — i.e. get `tests/test_torch_compile.py` (and `tests/test_models.py`)
from **6/32 passing to green**, by porting the existing inductor integration in
`triton_msl/inductor/` to torch 2.12's inductor codegen API. Never silent-wrong: a model
that can't lower must fail loudly or fall back, never return wrong values.

## Why this is #1 (the reframe)
- "Dynamic shapes" (roadmap 1H) was framed as runtime-dim support. But **runtime data dims
  already work** for hand-written `@triton.jit` kernels (FA/matmul use runtime N_CTX/M/N/K/
  strides). Its only remaining value is the **torch.compile** path.
- torch.compile was env-blocked (Python 3.14) until **torch was upgraded 2.9.1 → 2.12.1**
  (2026-06-18; torch added 3.14 Dynamo support in 2.10, full in 2.12). It now runs.
- With it running, the real gap surfaced: the **inductor integration is bit-rotted** — 26/32
  torch.compile tests fail with `torch._inductor.exc.InductorError: NotImplementedError` at
  inductor's `common.py:319` (`self.get_backend(device).codegen_node(node)`). The 6 passing
  are simple ops (identity/relu/gelu/silu/linear/lstm). Failures are LOUD, not silent-wrong.
- So #1 = **port the inductor backend**; dynamic shapes is a sub-task of full coverage.

## Current state to build on (do NOT rebuild)
- `triton_msl/inductor/__init__.py` `register_metal_triton_backend()` already:
  registers `TritonScheduling` + `PythonWrapperCodegen` for `"mps"` via
  `register_backend_for_device`; installs `MetalTritonDeviceOpOverrides`; patches
  `MpsInterface` (exchange/set_device/get_raw_stream); caps persistent + non-persistent
  reduction configs to Metal's 1024-thread limit; swaps libdevice → `metal_libdevice`.
- This worked at "32/32" on a **much older torch**. The breakage is inductor-API drift across
  torch 2.9 → 2.12, not a from-scratch gap.
- Env: torch **2.12.1**, py3.14, MPS. `torch.mps.compile_shader` intact; project suite 754/0.

## Staged plan

### Task 1 — Pin the exact codegen_node failure
- Run one failing case with full trace:
  `TORCHDYNAMO_VERBOSE=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=<wt> python3.14 -m pytest
  tests/test_torch_compile.py::TestModels::test_mlp -x` (after temporarily flipping the
  py-3.14 skip guard — see Task 5). Identify WHICH `get_backend`/`codegen_node` path raises
  `NotImplementedError` at `torch/_inductor/codegen/common.py:319` and for which node type.
- Diff torch 2.12's `register_backend_for_device` / `TritonScheduling` / `BackendFeature` API
  against what `register_metal_triton_backend()` passes (signature/positional drift is the
  prime suspect — e.g. a new required backend arg, or a `get_backend` that now expects a
  `BackendFeature`/`codegen_node` the registration doesn't supply).
- Deliverable: a one-paragraph root-cause in the report.

### Task 2 — Fix the registration / scheduling wiring
- Update `register_metal_triton_backend()` to satisfy torch 2.12's `register_backend_for_device`
  + scheduling contract so `get_backend("mps").codegen_node(node)` resolves (no
  `NotImplementedError`). Re-verify the 6 simple cases still pass + the next tier
  (softmax/log_softmax/max_pool) compiles.
- Integrity: if a node genuinely can't lower, it must raise a clear error or fall back to
  eager — never emit wrong values. Add/verify the graceful-CPU-fallback path
  (roadmap 0h / TorchTPU lesson) if inductor doesn't already raise loudly.

### Task 3 — Progressively green the suite
- Work the failing tiers in order: layers (softmax, pooling, norms) → MLPs → conv/resnet →
  transformer/attention → small GPT. Each tier: compile, compare vs eager
  (`atol≈1e-3` / cosine>0.95 for deep models, as the tests already encode). Fix the
  lowering/scheduling gaps each tier exposes. Commit per tier.

### Task 4 — Dynamic shapes through torch.compile
- Only after static torch.compile is green: test `torch.compile(model, dynamic=True)` with a
  variable-seq-length model; ensure the runtime-symbolic dims flow through (the hand-written
  path already supports runtime dims — verify inductor's symbolic ints reach the lowerer and
  the dynamic grid is computed). Close any constant-folding-of-symbolic-dim holes.

### Task 5 — Un-gate the tests + verify + docs
- Flip the skip guard in `test_torch_compile.py` + `test_models.py` from
  `skipif(sys.version_info >= (3,14))` to a **torch.compile-availability probe** (a
  `try: @torch.compile def _p(x): return x+1; _p(torch.zeros(1)); except: skip`) so it
  auto-lifts. (Drafted + reverted this session — the version-hardcode is now stale.)
- Any genuinely-unportable case → mark `xfail` with a precise reason (not a hidden skip).
- Full verify: project suite (was 754/0), torch.compile suite green (or xfail-documented),
  `test_core` ratchet unchanged (5,559/0), bump `CODEGEN_VERSION` if codegen changed.
- Update `docs/SUPPORTED_OPS.md` (torch.compile coverage) + `docs/ROADMAP.md` (mark 1H/6A-area).

## Risks / notes
- Inductor's internal API has no stability guarantee across torch minors — expect more drift
  points beyond `codegen_node`. Pin the torch version in `pyproject` once green.
- `TORCHINDUCTOR_COMPILE_THREADS=1` is required (Metal/PyObjC not fork-safe) — the tests set it.
- Keep the never-silent-wrong contract central: prefer loud failure / eager fallback over a
  guessed kernel at every step.
