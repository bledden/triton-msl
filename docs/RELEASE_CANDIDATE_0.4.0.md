# 0.4.0rc1 installed-workflow candidate

This is the integrated local candidate on branch
`dev/0.4.0-candidate-20261103`, based on `146e377` plus the development
worktree's working changes and the retained reporter repairs. It is not published.
The original development worktree is unchanged.

## Included work

- Toolchain, cache, ownership, import/identity and launch-contract repairs carried
  forward from the development branch, including native/pure helper selection.
- The pure-Python environment-snapshot and tensor-classification improvements.
- The composed host fast paths: guard reuse, shared binder census, scalar
  eligibility and packed-key reuse. The November 7 wheels contain all four and
  installed correctness passes. The November 7 timing panel was mixed. In the
  November 9 comparison with the November 3 wheels, the native wheel's mean
  call time is higher in all nine cases (+1.47% to +10.07%) and the pure wheel
  is mixed (-8.14% to +13.23%). Neither panel establishes a general speedup,
  and with two workers per arm neither attributes the difference to a cause.
- All seven dense attention fragment compositions, with the existing default-off
  option and source/precision admission checks preserved.
- Opt-in cooperative source-program export and generic backward stride lowering.
- An installed `triton-msl-export` command (also available as
  `python -I -B -m triton_msl.tools.export_source_program`).
- Reporter frontend regression fixtures, stride tests, installed-path test repair
  and public GPU tests for every attention composition.

## Artifacts and actual execution

One sdist produced the CPython 3.14 macOS arm64 native wheel and the pure wheel.
Both were force-installed into separate relocated environments. Tests ran outside
the source checkout and checked the loaded package origin. The native wheel selects
three native helpers; the pure wheel selects none. Pip dependency checks and the
installed exporter pass. The companion Julia/XLA bundle holds the separate adapter
and private plugin changes; they are not silently installed by the Python wheel.

The original November 3 execution and artifacts are retained at:
`~/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/release-candidate-integration-20261103/`.
Read its `RESULTS.json`, `FINDINGS.txt` and `CANDIDATE-MANIFEST.json` for final counts,
source/artifact hashes, logs, dependencies and failure records. The `dist/` directory
holds the installable wheels and sdist. The companion workflow has its own relative
input manifest and reproducible compiler-only build command.

The earlier host-stack wheels and sdist are retained in the sibling
`host-stack-integration-20261107/dist/`. Its `ARTIFACT-RECEIPT.json` retains that earlier source binding and review. Both wheel variants
passed installed CPU tests and normal/relocated artifact checks. The nine public
workflow checks and four transposed-B checks passed in both before/after installs;
all corresponding output hashes match. Timing is retained separately under
`host-stack-installed-20261107-harness/` (descriptive and mixed) and
`installed-current-comparison-20261109-native-access/` (7,200 samples, nine
cases, two workers per arm, current wheels against the November 3 wheels: all
nine native means slower, pure mixed, every output hash equal). Neither is a
qualified performance result. The nine cases do not include transposed-B matmul.

Tested stack: Apple M4 Max, macOS 26.7, Xcode 27.2, Metal Toolchain 27B47,
CPython 3.14.4, Triton 3.7.0+git4da2e268, Torch 2.12.1, NumPy 2.5.1 and
PyObjC 12.1. Julia checks use 1.12.7 and the pinned companion project.

## Dense transposed matrices: November 7 successor

The current wheels and sdist are in `transposed-dense-direct-20261107/dist/`.
Its independently reviewed `ARTIFACT-RECEIPT.json` binds source digest
`1f79f3c2` and the installed native/pure results. The new gate admits only
full-storage, zero-offset dense rank-two fp16/fp32 transposes to the existing
raw-buffer direct MPS route. Original views and aliases are preserved; offset,
partial, overlapping and gappy transposes retain the existing host route.

Four related transposed-B matmul cases show public-call-plus-sync means falling
from 1.08–1.26 ms to 0.21–0.26 ms, about 4.5–5.4x faster, in both installed modes.
All 3,200 samples are retained. This is a descriptive M4 workflow result, not
formal release performance qualification or the unavailable original reporter
program. See that lane's `FINDINGS.txt` and `timing-001/ANALYSIS.json`.

The full installed CPU selection passes 1,318 native and 1,226 pure tests. Eleven
GPU route, output-layout, alias and fallback tests pass in each mode. Sixteen
broader numerical cases pass in all four baseline/candidate installs, with
byte-identical outputs, including cancellation and NaN/Inf cases. The nine-case
public workflow correctness panel also passes on both successor wheels.

## Limits carried forward

- The bounded transpose timing result does not establish a general backend speedup.
  Earlier withheld timing labels remain withheld. Experimental defaults and
  numerical/SDK/ownership safeguards are unchanged.
- M1 has not been tested or emulated. The companion checks actual device identity.
- The first public call in a fresh process took 9.7 to 14.0 s in the November 9
  runs (two observations per arm, confounded with run order). The standalone
  toolchain identity step was 4.65 s. Two inventory changes measured on November 10
  under rules fixed in advance, small-file pooling and invocation-scoped alias
  reuse, did not meet their predeclared SDK-stage speedup thresholds and are not
  part of this candidate.
- The opt-in external training companion now passes whole-layer automatic
  gradients and six SGD updates per orientation using freshly compiled kernels
  from the November 7 host-stack installed wheel; the later transpose gate does
  not change those emitter or external runtime sources. This is the bounded compiler-hook route
  described in [EXTERNAL_TRAINING.md](EXTERNAL_TRAINING.md), not general support
  for differentiating arbitrary external custom calls.
- The regular Python launcher still refuses the pending source-program descriptor.
  External export does not weaken that admission gate.
- Native-wheel validation covers CPython 3.14 on this stack. Other declared Python
  versions, older macOS/toolchains and the complete workload/precision matrix are
  not newly qualified by this candidate.
- The release ledger retains its criteria and historical evidence. The current
  artifact row is independently reviewed and source-bound; 58 other rows remain
  pending. Further source/test/gate changes stale that receipt. Publication is
  separate from local validation.
