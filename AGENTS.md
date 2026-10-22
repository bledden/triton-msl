# Engineering continuity

Before changing this project, read `engineering/CONTRACT.md` and run
`python3 -I scripts/check_engineering.py`. Consult `engineering/ledger.json` for
open obligations and the workload baseline. These supplement CONTRIBUTING.md.

Blake's direct instruction in the September 23 recovery discussion was:
"every cell of this system is non-negotiable to compromise beyond hardware-restricted work".
Apply that instruction when historical roadmap priorities conflict with it. Preserve
correctness, numerical fidelity, safety, capability, performance, compatibility,
distribution and maintainability together. Do not infer approval to compromise one.
The standing decisions in `engineering/CONTRACT.md` remain in force, including SDK
byte reads, default-off experimental routes, and deferred byte-offset extension.

For a change, identify the user outcome, falsifiable hypothesis, affected obligations,
baseline and acceptance criteria before implementation. Record actual results and
unresolved effects in the ledger. Do not relabel historical evidence as current,
delete failing workload rows, substitute refusals for required capability, or use
an aggregate improvement to erase individual regressions. An evidence-ready record
requires an independently reviewed, source-bound receipt; the checker verifies
record integrity, not the scientific validity of the experiment.

CPU-only checks must not create a Metal device, including through test collection.
GPU work requires the shared project lease and isolated correctness/performance
runs. Follow the existing validation-lane procedures. Do not execute a known-unsafe
shader to confirm a static defect. Keep benchmark oracles independent of candidates.

Ask Blake specific questions only when an actual product-contract choice, unavailable
resource, or publication decision requires his judgment. Investigate routine engineering
choices autonomously. Existing source and reviews are evidence, not sunk-cost constraints.
