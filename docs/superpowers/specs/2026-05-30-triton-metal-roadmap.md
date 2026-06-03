# triton-metal roadmap (umbrella) — 2026-05-30

> Index doc for the pre-1.0 hardening + optimization roadmap. The detailed
> design for the first workstream is in `2026-05-30-ws0-foundation-design.md`.
> Subsequent workstreams will each get their own spec → plan → implementation
> cycle. This umbrella captures the cross-workstream decisions and sequencing
> so they aren't lost.

## Framing

A maintainer-lens and an MLX-lens review of the project (this session's
"review as if you are the triton team and mlx teams") surfaced four asks:

1. Fill the correctness gaps the integrity prescan currently *refuses* (the
   six refusal cases plus `dot_scaled`).
2. Fix or modify the failing test suites in light of how the libraries have
   changed.
3. Optimize the kernels toward the **optimal bounds given by the hardware**.
4. Make the documentation proper, including proper citations of other papers
   and projects.

Three of these (1, 3, 4) converge on a single architectural spine: the
**multi-element-per-thread (MEPT) register-array model** plus a general
`ttg.convert_layout` shuffle. That spine is the same work that lets a thread
hold `v[lid, lid+1024, …]` (the prerequisite for generic matmul, FA HEAD_DIM=64,
`chained_reductions`, `dot_mulbroadcasted`, `join_with_mma`, layout-aware
`trans_4d`), *simultaneously* vectorizes memory access toward the M4 Max
546 GB/s bandwidth roof (today: `vector_add` at 23% of peak), and *retires*
the pattern-detector debt. So "fill gaps" and "optimize" are largely the same
workstream, not two.

Test-suite health and documentation are independent supporting streams.

## Decisions baked in

| Question | Decision | Rationale |
|---|---|---|
| **North star / sequencing** | Spine-first, value-ordered increments. The orthogonal refusals (cf-dialect CF, N-D `cat`/`join`) come after as a cleanup phase. `dot_scaled` stays a documented honest refusal (no Apple microscaling hardware). | Generic matmul (the highest-value gap) requires the spine anyway; building it first means each phase ships a perf win *and* closes a gap cluster. |
| **Implementation substrate** | **Hybrid**: Python `generic_lowerer` for breadth/correctness/bring-up; **C++ MLIR→LLVM path for hot paths** (matmul, FA, reductions, `convert_layout`). Destination is C++-primary for perf-critical kernels; Python shrinks toward fallback over time. | Apple gives no supported writable native-assembly layer; the lowest controllable supported layer is AIR (LLVM bitcode), reachable through the C++ path. Layouts are first-class in MLIR — emitting register-array codegen from MLIR is far more tractable than XOR-basis position math + text. The pattern-detector debt exists because text-emission can't express layout-aware cooperative ops generically; the C++ path dissolves it. |
| **Definition of "optimal bounds"** | Empirical: *push each kernel until the limiting hardware counter saturates, verified in the disassembly.* A hardware-profiling + disassembly harness defines the bound and gates "done." | "Optimal bounds given by the hardware" cannot be guessed as a percentage; it has to be measured at the ISA/counter level. |
| **ISA depth** | Primary: **ship-safe empirical loop** — Metal GPU counters + native-AGX disassembly via `applegpu` (Rosenzweig, dougallj) and the Asahi compiler stack for READ-ONLY analysis; codegen control at the AIR/LLVM-IR level via the C++ path. Experimental side-track: **sub-AIR AGX generation** via the Asahi/`applegpu` toolchain, flag-gated like MEPT — never gates the shippable product. | Apple's AGX ISA is undocumented and has no supported assembler. The supported floor is AIR; below that is reverse-engineered, research-grade, and unsuitable as the primary product surface — but the AGX generation work has genuine ceiling-pushing potential as a research track. |
| **Test policy** | Two buckets, two policies: **(A) upstream `test_core` feature-gap skips** — fix-the-feature as the spine lands, un-skip + verify, skip-list shrinks; never modify-the-test. **(B) torch.compile / `test_models` Python-3.14 failures** — env-gate with `skipif(py>=3.14)` (honest skips, not red), plus a Python ≤3.13 CI lane to keep coverage live until torch ships 3.14 Dynamo support (auto-lifts then). Neither is a "backfill" — bucket B is PyTorch's own platform guard. | The first integrity rule still rules: never silent-wrong, always loud. Skips must be self-justifying. |
| **Citations standard** | A `REFERENCES.md` at the repo root + inline academic-style citations in `docs/` (author, title, year, link). Required references: Triton (Tillet et al.), FlashAttention v1/v2 (Dao et al.), online softmax (Milakov & Gimelshein), MLX, Asahi / `applegpu` (Rosenzweig; dougallj), Metal Shading Language spec, Apple GPU architecture notes. | The user explicitly asked for proper citations of other papers and projects; the project rides on a substantial body of public work and should acknowledge it. |

## Cross-cutting principles

- **Single integrity source of truth.** The refusal catalog and skip-list
  policy must hold across *both* code paths (Python and C++). The integrity
  prescan that lives in `GenericLowerer._refuse_unsafe_unsupported_ops` is
  authoritative; the C++ path must consult the same catalog (or its
  equivalent) so the two paths cannot disagree on what's supported. Never
  silent-wrong. Always loud.
- **Benchmark-tracked.** Every spine phase reports its before/after numbers
  through the harness: limiting counter, % of peak, MLX-comparison ratio,
  native-ISA disassembly snippet. No "perf win" claim without harness output.
- **Documentation moves with code.** A piece of work isn't done until the
  user-facing docs (README, ARCHITECTURE.md, CHANGELOG, REFERENCES.md) reflect
  it. The stale Phase-3 `1,404` table episode (where the docs internally
  contradicted reality and would have lost an external reviewer in five
  minutes) is the failure mode this principle is designed to prevent.
- **PR gating.** Per the user's standing instruction (2026-05-30), local
  commits to the worktree branch proceed without check-in; *PR creation* and
  push-for-review require explicit confirmation.

## Workstreams

### WS0 — Foundation & truth (spec: `2026-05-30-ws0-foundation-design.md`)

Documentation truth pass + citations infrastructure + test hygiene
(env-gating + 3.13 CI lane) + C++ build hardening (pinned Triton commit,
fixed CMake paths, CI-buildable) + the integrity-contract single source of
truth + the **HW profiling + disassembly harness**. Built first because it is
the empirical backbone of WS1 and clears the review-blocking debt cheaply.

### WS1 — The spine (the trunk)

MEPT register-array model + general `convert_layout` shuffle, built in
value-ordered phases. Hybrid substrate. Each phase ships a perf win *and*
closes a cluster of gaps:

| Phase | Substrate | Perf win | Gap cluster closed |
|---|---|---|---|
| A. elementwise arrays *(already landed)* | Python | bandwidth → 546 GB/s | (consumer-side complete) |
| B. reductions | Python → C++ for hot | softmax / layernorm | `chained_reductions`, multi-axis reduce edge cases |
| C. `tt.dot` register arrays | C++ (primary) | matmul off 0.32× | `dot_mulbroadcasted`, `join_with_mma`, FA HEAD_DIM=64 |
| D. general `convert_layout` shuffle | C++ (primary) | layout-aware throughput | `trans_4d`; **retires the `_detect_*` detectors** |

### WS2 — Completeness cleanup (the orthogonal refusals)

The refusals the spine does not cover:

- `cf`-dialect control-flow lowerer (the `cf.cond_br` / `cf.br` cases refused
  by case (e) of the integrity prescan: `test_nested_if_else_return`,
  `test_constexpr_if_return`).
- N-D `tt.cat` / `tt.join` (`test_cat_nd`).
- `tt.dot_scaled` stays a documented honest refusal — Apple has no
  microscaling hardware; "completeness" there would mean slow software
  emulation, which we'd only build if explicitly desired.

### WS3 — Experimental: sub-AIR AGX generation

Flag-gated (`TRITON_METAL_AGX_BACKEND=1`, off by default) research track
emitting native AGX via the Asahi / `applegpu` toolchain, never gating the
shippable product. Same posture MEPT had before it landed.

## Sequencing

1. WS0 (foundation) — write spec, write plan, execute.
2. WS1 (spine), phase by phase, each phase being its own plan.
3. WS2 (orthogonal refusals) as the spine matures and the substrate is in
   place.
4. WS3 in parallel with WS1 once the foundation harness exists.

Each workstream's transition to implementation invokes the
`superpowers:writing-plans` skill.
