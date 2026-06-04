# WS0 — Foundation & truth (design) — 2026-05-30

> First workstream of the umbrella roadmap (`2026-05-30-triton-metal-roadmap.md`).
> Establishes the documentation truth, citation infrastructure, test hygiene,
> a hardened C++ build, a single integrity source of truth, and — most
> importantly — the **hardware profiling + disassembly harness** that defines
> "optimal bounds given by the hardware" as an empirical, measurable bar.
>
> Everything in WS1 (the spine) measures against the harness this workstream
> builds. WS0 is therefore the empirical backbone, not just review hygiene.

## Goals

1. **Documentation is true.** No stale contradictions between docs (e.g. the
   Phase-3 `1,404` upstream-test table contradicting the README's `4,320 / 0
   failures`). A reader who opens any doc sees the current reality.
2. **Citations are proper and complete.** Every paper, project, and
   reverse-engineered toolchain the work depends on is attributed in a
   single `REFERENCES.md`, with inline academic-style citations in the docs
   that lean on each. The user explicitly asked for this; it's a first-class
   deliverable, not an afterthought.
3. **Test suite is honest.** Both buckets of "not passing" are handled
   correctly: feature-gap skips (bucket A) have policy and tooling for
   shrinking as features land; environment-gated failures (bucket B, the
   Python-3.14 / `torch.compile` 37) become honest skips on 3.14 *and* are
   covered by a parallel Python ≤3.13 CI lane that auto-lifts when PyTorch
   ships 3.14 support.
4. **C++ build is buildable.** The C++ path (`triton_metal/csrc/`) configures
   and builds against a pinned Triton commit, without hardcoded
   developer-machine paths. CI runs the C++ tests. Without this, the hybrid
   substrate cannot become the perf spine.
5. **Integrity contract has a single source of truth.** The refusal catalog
   currently in `GenericLowerer._refuse_unsafe_unsupported_ops` (cases a–e)
   becomes structured data consumable by *both* the Python and C++ paths, so
   the two paths cannot drift on what's refused.
6. **Hardware profiling + disassembly harness exists.** Metal GPU counters +
   native-AGX disassembly + roofline analysis, run automatically against a
   suite of representative kernels, producing reports that say which counter
   is the limiting bound for each kernel. This is what makes "optimal bounds
   given by the hardware" a measurable, falsifiable claim.

## Why WS0 first

- **It is the backbone of WS1.** Without the harness, "we sped up matmul"
  is anecdotal; with the harness, we can say "matmul moved from ALU-utilization
  18% to 71%, register spills 14 → 0, MMA-occupancy 0.31 → 0.84, X% of peak
  FP16 FLOPs, Y% of MLX baseline." That is the actual claim the user wants
  to make. Building WS1 without the harness would be optimizing blind.
- **It clears the review-blocking debt cheaply.** The stale `1,404` table,
  the missing citations, the failing torch.compile suite — these are
  presentation issues a reviewer hits first. Fixing them now (before deep
  work) means the next reviewer reads honest, internally consistent docs.
- **It de-risks the hybrid.** The C++ path can't be the perf spine if it
  only builds on one machine. The CMake hardening is gating for everything
  in WS1's hot-path phases.
- **It does not block on WS1.** Most of WS0 is independent of the spine
  work and can land in parallel-friendly increments.

## Components

Each component below is independently designable, testable, and lands as one
or more commits. The spec describes *what* each is; the plan (next skill)
will sequence *how*.

### C1. Documentation truth pass

**What it is.** A pass over every project doc to reconcile claims against
fresh-measurement reality, fix internal contradictions, and re-scope
unverifiable claims.

**How to use it.** Outcome is a state where any doc you open agrees with
every other doc and with the actual measurement artifacts in `reports/`.
Specifically:

- Replace the stale Phase-3 "Triton Upstream Test Results" table in
  `docs/ARCHITECTURE.md` with the current measurement (or remove it, since
  the README + `reports/upstream_test_core.txt` already carry it). The stale
  table is the highest-stakes single bug — a reader sees "1,404 / 7,625 fail
  / 81.8%" and concludes the README is lying. It must go.
- Reconcile the README claims with measurement: the "434/434 project tests"
  and "32/32 torch.compile model tests" are stale (current: 507 passed under
  the relevant suite; torch.compile suites are environment-blocked). State
  these accurately.
- Re-scope claims that read as broader than they are:
  - **FlashAttention**: the project has a *genuinely engineered* FA path
    (dedicated `_is_flash_attention_pattern` detection,
    `csrc/lib/Conversion/DotOpToLLVM.cpp` qk=q@trans(k) handling, an
    aliasing-pass fix, 11/11 passing), but only at **HEAD_DIM=32** and the
    specific block sizes the tests exercise. State that.
  - **Pattern detectors as performance specialization**: the doc already
    correctly calls them "structural debt"; ensure the language stays
    consistent with the roadmap's intent to retire them under WS1.D.
- Add a short "Status & honesty" preamble to ARCHITECTURE.md cross-linking
  the doc to the umbrella roadmap, so a reviewer immediately sees the plan
  for the gaps the doc lists.

**What it depends on.** Read access to `reports/`, this session's fresh
sweep, and the umbrella roadmap.

### C2. Citations infrastructure

**What it is.** A `REFERENCES.md` at the repo root and a citation
convention used in the docs.

**How to use it.**

- `REFERENCES.md` lists every external work the project leans on, in a
  numbered academic style with author, title, venue/year, and URL. Required
  entries at minimum:
  - **Triton** — Tillet, Kung, Cox. "Triton: An Intermediate Language and
    Compiler for Tiled Neural Network Computations." MAPL 2019.
  - **FlashAttention v1** — Dao, Fu, Ermon, Rudra, Ré. "FlashAttention:
    Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
  - **FlashAttention v2** — Dao. "FlashAttention-2: Faster Attention with
    Better Parallelism and Work Partitioning." 2023.
  - **Online softmax** — Milakov, Gimelshein. "Online normalizer
    calculation for softmax." 2018. (The numerical-stability trick the
    softmax/layernorm templates rely on.)
  - **MLX** — Apple ML Research. "MLX: A framework for machine learning
    research on Apple silicon." 2023.
  - **Asahi AGX** — Rosenzweig et al. AGX open-source GPU compiler (Mesa).
  - **`applegpu`** — dougallj. Apple GPU ISA disassembler / reverse
    engineering project.
  - **Metal Shading Language Specification** — Apple, current version.
  - **PyTorch Inductor / TorchDynamo** — Ansel et al. PyTorch 2.0
    paper / docs.
- Citation convention in docs: `[N]` references back to `REFERENCES.md`
  entry N. README + ARCHITECTURE.md + CHANGELOG carry citations where they
  lean on a paper or project (e.g., the online-softmax reference where the
  template is described; the FlashAttention paper where FA support is
  claimed; the Asahi/`applegpu` references where the disassembly harness or
  the AGX research track is described).
- `CITING.md`: short note on how to cite *triton-metal itself* (suggested
  BibTeX), since other projects may want to reference it.

**What it depends on.** Nothing — purely additive. Lands in one commit
plus per-doc inline-citation edits.

### C3. Test hygiene + skip-list policy

**What it is.** A test-suite state where every failure is genuine and every
skip is self-justifying.

**How to use it.**

- **Bucket B (PyTorch / Python 3.14):** add an autouse `skipif` to
  `tests/test_torch_compile.py` and `tests/test_models.py`:
  ```python
  pytestmark = pytest.mark.skipif(
      sys.version_info >= (3, 14),
      reason="torch.compile is not supported on Python 3.14+ "
             "(PyTorch's own platform guard; resolves when PyTorch ships "
             "3.14 Dynamo support — see REFERENCES.md [PyTorch])"
  )
  ```
  Outcome on the current dev/CI machine (Python 3.14): honest skips, not
  red failures.
- **Bucket B coverage preservation:** stand up a CI job on Python 3.13 with
  `triton` + `triton_metal` installed (the same stack the dev venv carries
  on 3.14), so `test_torch_compile.py` and `test_models.py` actually *run*
  in CI. When PyTorch adds 3.14 Dynamo support, the `skipif` lifts
  automatically and the 3.13 lane can be retired or kept as a coverage
  buffer.
- **Bucket A skip-list policy** (codified in `scripts/conftest_metal.py` as
  comments + in `docs/ARCHITECTURE.md` as policy):
  - Every entry in the feature-gap skip-list MUST be one of:
    1. *Refused, integrity-safe*: the lowerer raises
       `MetalNonRecoverableError` for the kernel. Skip exists because the
       upstream test asserts success. (Most current entries.)
    2. *Compile error*: the kernel fails at compile/parse with a clear
       error message. Skip exists for the same reason.
    3. *Genuine HW gap*: the kernel requires hardware Apple doesn't have
       (FP64, FP8 compute, microscaling). Skip is permanent.
  - Forbidden category: *silent-wrong tolerated by a loose assertion* (the
    `test_constexpr_if_return` class). When found, the kernel becomes a
    refusal and the test joins category 1 with the rationale recorded.
  - Each entry MUST carry a one-line rationale comment naming the category.
- **No modify-the-test.** The policy is fix-the-feature, un-skip, verify.
  When an upstream test changes shape because the upstream library evolved,
  document it in the skip-list comment and re-evaluate; do not adjust the
  test body.

**What it depends on.** A working Python ≤3.13 CI lane. Provider choice
(GitHub Actions matrix, or a local CI runner) is a plan-level decision.

### C4. C++ build hardening

**What it is.** Making `triton_metal/csrc/` build on any machine with the
declared toolchain — not just the developer's.

**How to use it.** The path the build takes today:

- `CMakeLists.txt:7` and `:34–36` hardcode
  `$HOME/Documents/triton` and the prebuilt object path
  `build/cmake.macosx-15.0-arm64-cpython-3.14`. This breaks anywhere else.
- `pyproject.toml` declares `triton>=3.6.0` (loose); the C++ build needs a
  specific Triton revision because it links against TritonGPU dialect
  objects.

Hardening:

- **Pin** a specific Triton commit (or a tag if upstream offers a stable
  one) in a top-level `TRITON_PINNED_COMMIT` constant the build script
  references. Document why this commit; bump it as a discrete, tested step.
- **Resolve Triton paths via find logic**: prefer a configurable
  `TRITON_ROOT` cmake variable (env-overridable), fall back to a vendored
  build of the pinned commit, fall back finally to the developer-home path
  with a clear deprecation warning. No silent "works on my machine."
- **Vendor or cache the Triton build artifacts** the C++ extension links
  against, so CI doesn't need to rebuild upstream Triton to build the
  extension.
- **CI integration.** A `cpp-build` job in CI that runs `cmake` + `make` +
  the C++ unit tests for the existing conversion passes. This is the gate
  that proves the build is hardened.

**What it depends on.** Knowledge of which Triton commit the current code
is aligned to (recoverable from the `chore: align with upstream triton 3.7.0`
commit history); a CI provider choice (same lane as C3).

### C5. Integrity-contract single source of truth

**What it is.** The refusal catalog moves from inline code (cases a–e in
`GenericLowerer._refuse_unsafe_unsupported_ops`) to a structured registry
that the C++ path can also consume.

**How to use it.**

- Define a single Python module — `triton_metal/codegen/refusal_catalog.py`
  — that lists every refusal case as a structured record:
  ```python
  REFUSAL_CASES = [
      RefusalCase(
          name="dot_scaled_no_hw",
          op="tt.dot_scaled",
          predicate=is_dot_scaled_op,
          message="Microscaling matmul has no Apple hardware …",
          rationale="no Apple microscaling unit; software emulation would …",
          examples=["test_scaled_dot"],
      ),
      # cases for constexpr-dim matmul, rank>=3 trans, N-D cat/join,
      # noinline-dot, join->dot, unstructured cf.*
  ]
  ```
- `_refuse_unsafe_unsupported_ops` becomes a thin walker over
  `REFUSAL_CASES`. The current cases a–e get one record each, with their
  predicates extracted.
- The C++ path imports the same module at codegen time (or, if Python is
  not in scope from C++, generates a small `refusal_cases.json` artifact
  the C++ path reads at build time) so its TTGIR-level analysis refuses
  the *exact same* set of patterns with the *exact same* messages.
- Documentation: ARCHITECTURE.md "Lowering paths and the integrity model"
  section is regenerated from `REFUSAL_CASES.message` strings, so the doc
  cannot drift from the code.
- Unit test: `test_refusal_catalog.py` round-trips every case (each known
  example kernel triggers exactly its registered case; no case overlaps;
  rationale text is non-empty; doc generation is byte-stable).

**What it depends on.** No external work. Touches `generic_lowerer.py`
modestly. Lands ahead of any C++ work that needs it (so the C++ path can be
consistent from day one).

### C6. Hardware profiling + disassembly harness

> **Implementation reality (2026-06-03), corrected from this spec after
> probing the M4.** Two assumptions in the original C6 design did not survive
> contact with Apple Silicon, and the honest scope is recorded here:
> 1. **Live GPU counters are not programmatically available.** The M4's only
>    counter set is `timestamp` (`MTLDevice.counterSets()` → `['timestamp']`);
>    `StageUtilization`/`Statistic` constants exist but the device does not
>    vend them. So ALU%/occupancy/register *live* counters cannot be sampled
>    via `MTLCounterSampleBuffer` — by pyobjc **or** a Swift helper (it's a
>    Metal public-API wall). The planned Swift-helper "Stage 3" was therefore
>    **not built**; it was reconceived as `docs/INSTRUMENTS.md` documenting
>    the Xcode-capture / Instruments path (the genuine home for those
>    counters). What the harness measures instead: GPU-timestamp timing →
>    roofline, plus reliable pipeline-reflection occupancy.
> 2. **Native-AGX disassembly is best-effort.** A `.metallib` holds AIR, not
>    native code; the native AGX is in the serialized `MTLBinaryArchive`
>    (a `0xCBFEBABE` fat Mach-O with an `applegpu` slice). The vendored
>    `applegpu` decoder is M1-era, so M4/AGX2 decode coverage is partial
>    (~40% measured) and the harness reports the coverage % explicitly rather
>    than pretending completeness.
>
> Net: the harness is real and runs (`reports/hw_harness/<date>/`), giving the
> roofline %, bound, occupancy hint, MLX ratio, and best-effort disasm per
> kernel — the empirical backbone for WS1. The two items above moved to
> `INSTRUMENTS.md` / best-effort rather than being faked.

**What it is.** A tool — `python benchmarks/hw_harness.py <kernel-suite>`
— that runs a representative kernel suite, times each with GPU timestamps,
performs a roofline analysis, reads pipeline-reflection occupancy, attempts
best-effort native-AGX disassembly, compares against MLX, and emits a
structured report. This is the empirical definition of "optimal bounds."

**How to use it.** For each kernel in the suite, the harness produces:

- **Wall-clock min/avg/max** over warm-up + N iterations (matching the
  existing `benchmarks/bench_*` pattern).
- **Metal GPU counters** via `MTLCounterSampleBuffer` (or
  `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` as a baseline if richer
  counters aren't accessible from Python without bridging — choice
  finalized at plan time):
  - ALU active cycles / total cycles → ALU utilization.
  - Memory bytes loaded / stored → effective bandwidth.
  - Threadgroup memory pressure.
  - Register pressure / occupancy (limit reason: register-spill-limited?
    threadgroup-memory-limited? thread-count-limited?).
- **Roofline classification**: bandwidth-bound vs ALU-bound vs
  occupancy-bound, computed from the kernel's arithmetic intensity vs the
  hardware's compute and bandwidth roofs (M4 Max: 546 GB/s; FP32 + FP16
  peak FLOPs; numbers cited in REFERENCES.md from Apple's specs).
- **Native-AGX disassembly** of the produced `metallib`, via the
  `applegpu`/Asahi toolchain (referenced read-only). The harness saves the
  disasm to `reports/disasm/<kernel>.agx.txt` and computes summary metrics
  from it: register count per thread, instruction mix, presence of MMA
  (simdgroup-matrix) instructions, spill counts. This is what tells us
  "the metal compiler did/didn't emit the hardware MMA path" without
  guessing.
- **MLX comparison** (for kernels with an MLX equivalent: matmul, softmax,
  layernorm, attention): the same shape run through MLX, reported as a
  ratio. Not a pass/fail bar — context.
- **Limiting bound**: the harness's call on which counter is saturated and
  therefore which is "the bound." This is the single number that defines
  "done" for each kernel in WS1.

Output formats:

- Per-kernel JSON in `reports/hw_harness/<date>/<kernel>.json`.
- A human-readable markdown summary in `reports/hw_harness/<date>/summary.md`
  with the limiting-bound table.
- A regression baseline (`reports/hw_harness/baseline.json`) that future
  runs diff against; CI flags regressions > 5%.

Kernel suite (initial):

- `vector_add_16M` (memory-bound; 23% of peak today → backbone for
  bandwidth-vectorization gains).
- `softmax_8Kx1K` (memory-bound; 102 GB/s today).
- `layernorm_4Kx1K` (memory-bound; 77.5 GB/s today).
- `matmul_512x512_fp32` (compute-bound; 0.32× CPU today).
- `matmul_1Kx1Kx1K_fp16` (compute-bound; the workhorse).
- `attention_2_2_64_32` (the FA HEAD_DIM=32 case).
- `attention_HEAD_DIM=64` (the WS1.C target; expected to fail/refuse today,
  passes when WS1.C lands).
- `chained_reductions` (the WS1.B target).

Each kernel records the limiting bound under both Python-MSL and (where
present) C++ MLIR→LLVM paths — so we can see whether the C++ path actually
moves the bound.

**What it depends on.** Decisions at plan time on which counter-access API
to bind (raw `Metal` framework via `pyobjc`, or shell out to a small
Swift/Objective-C++ helper); which `applegpu` revision to vendor or pull;
the MLX install in the same env (already present in the project venv).

**Sub-AIR AGX experimental track stub.** Within this component, leave a
hook for the WS3 experimental path: a flag-gated kernel-emit path that, when
`TRITON_METAL_AGX_BACKEND=1`, routes through the Asahi/`applegpu` toolchain
to produce a native AGX `metallib`. WS0 only stubs this (the harness can
*receive* an AGX-path metallib and disasm it); the actual emission lands in
WS3.

## Data flow / integration

WS0 components compose like this:

```
        ┌──────────────────────────────────────────────────────────┐
        │                    HW harness (C6)                        │
        │   counters + disasm + roofline + MLX comparison           │
        │   reports/hw_harness/<date>/{kernel}.json                 │
        └──────────────────────────────────────────────────────────┘
                              ▲
                              │ measures
        ┌──────────────────────────────────────────────────────────┐
        │   Python codegen path  +  C++ codegen path (hardened)     │
        │   (C4 makes both buildable; C5 makes them agree)          │
        └──────────────────────────────────────────────────────────┘
                              ▲
                              │ documents / tests
        ┌──────────────────────────────────────────────────────────┐
        │   Docs truth (C1) + Citations (C2) + Test hygiene (C3)    │
        │   REFERENCES.md, ARCHITECTURE.md, README, CHANGELOG,      │
        │   conftest_metal.py, ws0+roadmap specs                    │
        └──────────────────────────────────────────────────────────┘
```

WS1 enters at the top, runs each spine phase, and gates "done" on the
harness's limiting-bound report.

## Testing plan

- **C1 (docs):** doctest-style consistency check — a small script
  (`scripts/check_doc_consistency.py`) that asserts the README test number
  matches `reports/upstream_test_core.json`'s pass count to within tolerance.
  CI runs it.
- **C2 (citations):** lint that every `[N]` in a doc resolves to an entry in
  `REFERENCES.md`. Same script.
- **C3 (test hygiene):** sweep both buckets on both Python lanes (3.14 dev,
  3.13 CI). Bucket A = the existing upstream sweep (4326 / 0). Bucket B on
  3.13 = `test_torch_compile.py` + `test_models.py` actually run (expected
  pass count to be recorded once the lane is up).
- **C4 (C++ build):** the `cpp-build` CI job — `cmake` + `make` + run the
  C++ unit tests for existing conversion passes (`DotOpToLLVM`, etc.). Plus
  a smoke `python -c "import triton_metal._triton_metal_cpp"` from a fresh
  install.
- **C5 (refusal catalog):** `tests/test_refusal_catalog.py` — round-trip
  each known example kernel, assert correct case fires, assert doc
  generation is byte-stable.
- **C6 (harness):** smoke test on the initial suite — `vector_add_16M`
  produces a JSON, the disasm is non-empty, the limiting bound is decided.
  CI runs the harness on every PR (later) and diffs against the baseline.

## Acceptance criteria for WS0 as a whole

WS0 is complete when:

1. Every project doc agrees with every other doc and with `reports/`. The
   stale Phase-3 table is gone.
2. `REFERENCES.md` exists with at least the listed entries; inline
   citations are present in any doc that leans on a reference.
3. On Python 3.14: 0 hard failures across the full project suite + upstream
   sweep. Bucket-B suites are honest skips with a documented reason. On
   Python 3.13: bucket B *runs* in CI and its pass count is recorded as a
   baseline.
4. `cpp-build` CI job is green. A new dev can clone, install, and build the
   C++ extension without editing CMakeLists or installing Triton at a
   specific path.
5. `REFUSAL_CASES` is the source of truth for the integrity prescan; the
   existing six refusal cases each have a record; the corresponding
   ARCHITECTURE.md section is doc-generated.
6. The HW harness runs on the initial 8-kernel suite, emits per-kernel
   JSON + a summary, decides a limiting bound for each, and saves disasm.
   The baseline is committed; a regression-detector script flags > 5%
   wall-clock changes.

## Scope boundaries (not in WS0)

- No spine work (no MEPT phase B/C/D, no `convert_layout` redesign, no
  `tt.dot` register arrays). That is WS1.
- No new ops, new dtypes, or removed refusals. The refusal *set* doesn't
  change; only its representation does.
- No PR creation (per the user's standing instruction; all WS0 work commits
  locally on the worktree branch, PRs gated by explicit check-in).
- The experimental AGX-generation path (WS3) is only *stubbed* in C6's
  harness for future use; not actually emitting AGX.

## Open questions for plan time

These are deferred to the implementation plan (next skill), not to the
spec:

- **Counter API surface.** Raw `Metal` framework via `pyobjc`, or a small
  Swift/Objective-C++ helper invoked via subprocess? The first is simpler
  to integrate; the second gives richer counter access. Plan decides.
- **CI provider.** GitHub Actions matrix (3.13 + 3.14 macOS runners — but
  macOS runners are slow and the C++/Metal build is non-trivial), or a
  self-hosted runner on the Apple Silicon machine. Plan decides; either
  works for the design.
- **`applegpu` integration depth.** Vendor a pinned revision into
  `third_party/applegpu/`, or shell out to a user-installed copy? Plan
  decides.
- **MLX comparison shapes.** Initial kernel suite shapes are listed above;
  the MLX-equivalent invocations need to be exact (same dtype, same shape,
  same iteration count). Plan codifies them.

## Outcome of WS0

When WS0 lands, the project state is: docs you can trust, citations
befitting the body of work the project rides on, a test suite that's
honest in both directions, a C++ path anyone can build, an integrity
contract that holds across both code paths from one source, and — most
critically — a measurement apparatus that turns "optimal bounds given by
the hardware" from aspiration into a number per kernel. Every subsequent
workstream stands on this foundation.
