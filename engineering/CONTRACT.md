# Project engineering contract

The purpose is to make Triton programs useful on Metal: faithful computation,
competitive execution, predictable resource use, and dependable integration and
installation. No single pass count, benchmark, compiler route, or architecture
represents that objective. The user reports are mandatory regression cases, not
the boundary of our investigation.

## Decisions in force

- On September 26 Blake approved the reviewed discovery-memo F1 v3 repair:
  two added import-machinery tuple captures, miss-time re-reads, ownership of
  observed finders/hooks/resolver for one memo generation, and no new memo for
  unproved captures. Retired finalizers may run during replacement. This narrow
  approval does not waive observation, frame, recursion or scheduling semantics
  for a future native full-observer engine. Installed costs remain measured and
  separately disclosed; this correctness repair is not a performance win.
- The full-byte toolchain, SDK and framework inventories remain the definition of
  those identities. On November 14 Blake replaced the September 21 rule that every
  process must repeat them. The proposal put to him was a persisted per-user identity
  record: the full inventory once and whenever the recorded evidence changes, a
  metadata check afterwards, the full inventory on request; record use on by default;
  the SDK checked by its version files rather than a walk of every file; the compiler
  binaries checked by metadata on their read-only mount rather than re-hashed. His
  answer: "okay well yes you have my full go to address this completely". In force:
  a record may return only a digest that a full-byte inventory produced; any doubt is
  a miss followed by the unchanged full inventory, never an error or a refusal;
  `TRITON_MSL_IDENTITY_RECORD=0` keeps the full inventory in every process. What a
  record does not see is listed in docs/CACHE_CONTRACTS.md, "Persisted identity
  records". Trusting less evidence than that section states, or a shared record
  directory, needs a new decision. This records a design decision for the candidate;
  it is not a timing result, a performance claim or a release approval.
- Host-owned extent borrowing stays opt-in and default-off. Preserve its live-owner,
  allocator/VM proof, self-test and dispatch-lifetime guarantees.
- The C++ route and scalar-CFG optimization stay opt-in and default-off; preserve
  positive pre-adapter proof and unoptimized fallback on refusal.
- The divisible byte-offset adapter extension remains deferred.
- Blake approved the prepared isolated native/pure wheel builds and local installation
  checks on September 25. They completed in the installed-baseline-preparation evidence
  lane, from the frozen September 24 source. This records the granted decision; do not
  request it again. Artifact smoke is not full installed qualification, and injected
  worktree helpers cannot substitute for installed evidence. Commit/push/release remain
  separate decisions.

Improving the machinery does not waive these decisions. The direct instruction behind
the recovery priority is Blake's September 23 statement: "every cell of this system is
non-negotiable to compromise beyond hardware-restricted work". This is a transcript
quote, not a new approval attributed to the reviewer. It resolves conflicts with
historical roadmap priorities without treating all historical guidance as obsolete.

## Eight simultaneous obligations

| Obligation | Evidence needed |
| --- | --- |
| Correctness and numerical fidelity | Independent source-aware oracles; outputs, auxiliary outputs and gradients; justified precision/tolerances; exceptional values, masks, tails and rounding. |
| Memory and execution safety | Bounds, aliases, synchronization, assertions, ownership and error propagation; no fallback replay after possible submission. |
| Useful capability | Positive execution of required programs and source idioms; explicit operation/shape/type coverage and refusal transitions. Safe refusal is not recovered capability. |
| Performance | Cold and warm workflow latency, compilation cost, device execution, throughput and variance. Retain each workload/mode; match inputs, source, numerical contract, stack and hardware. |
| Resource efficiency | Copies, allocations, peak memory, retained memory, compilation/cache footprint, and sustained work. Diagnose power/thermal effects where measurements support it. |
| Compatibility and integration | Hardware/OS/toolchain/Python/framework matrix; JIT, Inductor, MLX and external runtimes; argument ABI, addresses, queues and lifetimes. |
| Distribution and reproducibility | Native and pure installed artifacts, clean/relocated imports, helper selection, dependency pins, identity/invalidation, diagnostics and actual loaded bytes. |
| Maintainability and observability | Explainable route selection, coherent contracts, regression tests that detect the old defect, bounded complexity, and maintained documentation. Another special case needs comparison with fixing the underlying representation. |

These are separate obligations, not weighted scores. An improvement in one cannot
cancel failure in another. Different workloads can need different implementations;
this is not a requirement that one kernel dominate every configuration.

## Evidence and scope

The ledger starts OPEN. Historical ratios are extracted from the existing release
limitations, not newly reproduced measurements. Historical successes do not qualify
the current candidate. All 18 existing warm comparison workloads remain visible in
native and pure modes, alongside cold and external-runtime cases. The corpus must
also cover the operation census and existing model/training suites; the initial rows
are not exhaustive. Add cases when a new defect, idiom or workload exposes a gap.

Baseline families: last correct usable implementation (including 0.2.0 when correct),
released 0.3.0, current candidate, and credible alternatives such as local generated
kernels, MLX/MPS or triton-ext where contracts match. Preserve incorrect baseline
results as failures; do not compute a qualifying speedup against them. A 30% gain
against a regressed baseline does not establish recovery of the earlier user outcome.

Define numerical and performance acceptance before choosing the winning candidate.
Record both absolute costs and relative changes, timing boundaries, repetitions,
ordering, environment and uncertainty. Do not silently choose a convenient comparator,
input precision, tile, cache state or benchmark subset. Use hardware-specific targets;
an M4 floor cannot qualify M1. Missing hardware remains blocked, not passed.

Classify a limitation as: hardware/API restriction, semantic obligation, implementation
gap, integration gap, or unmeasured. A hardware restriction needs a pinned primary-source
reference or discriminating experiment, affected devices and assessment of alternatives
(including emulation and alternate lowering). Hardware absence is not impossibility;
one unsuccessful implementation is not a ceiling. The checker has no hardware waiver.

## Working cycle

1. Read the current dashboard and history. Select a demonstrated problem with broad
   user impact. Keep one primary implementation hypothesis active at a time; independent
   inquiries may proceed alongside it without conflicting source/GPU ownership.
2. State the cause suspected, the observation that would refute it, and the baseline.
   Profile the full path before assigning a slow workflow to any one component.
3. Map changes to all eight obligations. Record why a dimension is unaffected in the
   experiment report rather than claiming it passed without testing. Trace shared helpers
   to their other consumers. Numerical faithfulness and lifetime guarantees are constraints;
   the present checking machinery and its cost are design choices we may improve.
4. Compare fixes to the causal problem. Prefer coherent lowering/layout/ownership models
   over accumulating whole-kernel exceptions where that solves more than one defect.
5. Test positive capability, nearby adverse cases and actual execution routes. Measure
   performance on the unchanged computation. Reuse valid evidence; rerun when the bytes,
   scope or hypothesis change, rather than mechanically rerunning unrelated suites.
6. Review the experiment and patch independently, including assumptions in the harness.
   Update each affected ledger row, including regressions and unknowns. Source changes
   invalidate current receipt bindings until revalidated or explicitly re-established
   with a bounded equivalence argument and independent review.

The next session resumes from `active_investigation` and open obligations, not whichever
component happened to improve last. A checkpoint states what changed, evidence gained,
what remains open, and the next discriminating experiment.

## Executable dashboard

Run `python3 -I scripts/check_engineering.py` from this checkout. It uses only the
standard library and never imports the backend or queries the device. Exit 0 means
the ledger is structurally valid, not release approval. `--release-check` returns 2
while any obligation or workload lacks current evidence; malformed/stale bindings
return 1. There is no overall percentage or compensating score.

For an `evidence_ready` row, attach a JSON receipt with this envelope:

```
{"subject": "row-id", "source_digest": "checker-reported digest",
 "outcome": "pass", "protocol": {"path": "...", "sha256": "..."},
 "results": {"path": "...", "sha256": "..."},
 "review": {"path": "...", "sha256": "..."}}
```

Paths are relative to the repository or absolute local evidence paths. Protocol,
results and independent review are distinct nonempty files. The row's `receipt`
contains the receipt path and its SHA-256. Freeze actual experiment scope, artifact
and environment bindings, acceptance criteria, per-obligation impact and raw-data
references inside these files. The checker checks their existence, integrity and
source binding; it cannot verify that a reviewer was independent, numerical evidence
is sound, or a scope was honestly chosen. Those remain review obligations. Do not
create a passing envelope around a failed, refused or unexecuted required case.

The source digest binds package sources, build metadata, project test sources, all
files under `tests/fixtures/` (regardless of suffix), `.ll` inputs, the checker and its
tests, `AGENTS.md`, and this contract. It also binds a canonical ledger definition:
all top-level fields except `active_investigation`, and all obligation/workload fields
except `state`, `receipt` and `next_experiment`. Case identities, schema, criteria,
scope, historical baselines and new definition fields therefore invalidate old receipts
when edited. The excluded fields track progress and evidence references, not acceptance
requirements; put those in a bound field or the contract. Including receipt hashes
would create a circular dependency, so the mutable evidence index is intentionally
excluded. Status changes still undergo the checker's validation. Raw JSON whitespace
and object-key order do not change the canonical definition; list order does.

The digest excludes build outputs and unrelated documentation. Protocol/results must separately
bind installed helper/binary bytes and runtime environment. Changing an unrelated
test can conservatively stale a receipt; retain the old receipt and explain the
applicable revalidation rather than overwriting history.

This binds the gate definition used by an honest checker; it is not a security root
against someone replacing the checker and forging receipts together. Independent
review and retained file hashes remain necessary.

Release review requires current evidence across the declared contract, no unexplained
regression or lost required capability, and explicit treatment of genuine hardware
restrictions. An empty ledger, new labels or edited criteria cannot satisfy it.
Publication is still Blake's decision. Tooling helps continuity; it cannot replace
engineering judgment or manufacture confidence.
