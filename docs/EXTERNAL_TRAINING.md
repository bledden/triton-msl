# Tested external automatic training route

The November 7 local companion runs automatic differentiation through the complete
LuxTriangleAttention layer and its external Metal kernels. It uses the explicit
`AttentionLowering.compile_external` entry point, a process-local Reactant compiler
hook. It does not enable general differentiation of arbitrary custom calls, remove
KernelDispatch's protective AD guard, or change the Python wheel's defaults.

## What ran

The November 7 host-stack native wheel (SHA256
`3675a618b98f2f429eb9015d91dad1856a24ef52ae97ee9b04438b7555a69ba2`,
source digest prefix `a5d3e459`) regenerated the forward, dQ, dKV, dBias and
fully-masked dV-fix kernels. Their generated Metal source matched the tested ABI
before fresh binaries were passed to the companion. The training IR contained six
external Metal calls and no outlined reference-attention function. This run does
not bind the later dense-transpose successor wheel (source digest prefix
`1f79f3c2`); that successor did not change the emitter or external-runtime sources,
but the complete training run was not repeated with its wheel.

Both layer orientations passed automatic gradients against an automatically
differentiated kernel-free reference on the same Enzyme/Reactant/PJRT stack, then
six SGD updates with decreasing loss. This reference is not an independent FP64
or finite-difference gradient oracle.
The additional cases cover partial/fully masked rows, all-dead inputs, shared q/k
and gradient accumulation. All 41 retained gradient-array comparisons passed;
maximum absolute error was `1.1920928955078125e-7`. This was a fresh November 7
execution; the retained arrays and freshly built kernel binaries were byte-equal
to the November 3 run, so it does not add a new input population.

The tested scope is Float32, N=32, D=32, H=2, B=1, first-order reverse AD, on the
M4 Max and pinned Julia 1.12.7/Reactant/PJRT stack. Higher-order AD, other shapes,
arbitrary custom calls, M1/M5 and training performance are not established.
The runtime reports its PJRT API versions and retained precompilation warnings;
successful execution here is not a broader plugin compatibility guarantee.

## Local artifacts and use

Evidence lives under
`~/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/training-integration-20261107/`.
The fresh run is `current-run-002/`; its `CURRENT-BINDING.json` binds the installed
wheel, emitted sources, fresh binaries and runtime. The failed first preparation
launch remains in `current-run/` (missing copied fixture, before GPU execution).

`triton-msl-current-training-companion.tar.gz` contains the relative project,
adapter sources, plugin, freshly generated kernels, explicit lowering hook and
tested entry scripts. `CURRENT-COMPANION.json` gives its hash. This companion is
separate from the Python wheel and has not been published or installed globally.
It needs the pinned Julia dependencies; it is not a dependency-free executable.

In a fresh extracted copy with the qualified dependency depot selected, use Julia
with `--startup-file=no --history-file=no --project=workflow-bundle/project` to run
`training_probe.jl` and then `coverage_probe.jl`, serialized under the project GPU
lease. These execute GPU work and produce output files. Preserve the supplied
retained results by using a separate copy. `python3 -I -B verify.py` only checks
the retained relative hashes and numerical records; it executes no training.

The source ownership is explicit: Triton-MSL emits the kernels; KernelDispatch,
LuxTriangleAttention, Reactant and the private PJRT plugin implement the external
runtime and AD integration. The compiler hook is an opt-in companion integration,
not an upstreamed or generally stable Reactant extension API.
