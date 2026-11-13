# Experimental external TriFast export

The closed source-program proof can now select a cooperative 128-thread lowering
with `MetalOptions(experimental_source_program_cooperative=True)`. It shares each
32x32 tile across four SIMD groups instead of assigning one entire row to each
thread. The option is off by default and is part of the compiler cache key. It
preserves the source recurrence, mask, output and saved-LSE computation. A failed
proof remains terminal. This is a lowering of the proved program, not recognition
by a kernel name or source hash.

After installing the candidate wheel, export a retained TTGIR fixture with Triton and the Metal toolchain installed:

```sh
python -I -B -m triton_msl.tools.export_source_program \
  --ttgir tests/fixtures/source_program/masked.ttgir \
  --output /absolute/new/export --cooperative --metal-version 4.0
```

Use `padded.ttgir` for the other retained source branch. `--source-only` emits MSL
and metadata without invoking the Metal compiler. The command never creates a
Metal device. The output directory must be new; partial exports have no completed
manifest. Metal language support depends on the installed SDK; the tested toolchain
used version 4.0. This is not a claim that an older Mac requires Metal 4 hardware.

The original Python reporter frontend also reaches these same shader bytes when
compiled with the option. The regular Python JIT launcher still refuses the pending
source-program admission descriptor. Exporting a metallib for an external consumer
does not remove that gate or automatically provide a Julia/XLA integration.

## External caller contract

Read `manifest.json`, verify the file hashes, and load the named entry. Require the
pipeline's thread execution width to be 32 and its maximum threadgroup size to be
at least the descriptor's 128 threads. Dispatch exactly `(128,1,1)` threads per
group; the scalar control uses `(32,1,1)`. Dispatch groups `(N/32,N,H*batch)`.
Cooperative full-tile N values are 32, 64, 128 and 256, with D=32, H and batch each
1 or 2. Do not treat these bounds as proof for partial tiles or another layout.
The direct C-consumer controls cover H=batch=1. The companion Julia workflow
additionally checks H=batch=2 at N=32/64 and H=3, batch=1 at N=32; these are
explicit tested cases, not a qualification of arbitrary head or batch counts.

The retained reporter ABI has eleven separate bindings. Each tensor is contiguous;
byte sizes below exclude any caller guard regions. Use disjoint buffers for these
controls, keep them alive until completion, and check command completion errors.

| Index | Value | Required bytes |
| --- | --- | --- |
| 0 | output f32 | `batch*H*N*N*32*4` |
| 1 | saved LSE f32 | `batch*H*N*N*4` |
| 2,3,4 | Q,K,V f32 | each `batch*H*N*N*32*4` |
| 5 | bias f32 | `batch*H*N*N*4` |
| 6 | mask u8 | `batch*N*N` |
| 7,8 | scale and masked sentinel f32 | each 4 |
| 9,10 | N and H i32 | each 4 |

The descriptor additionally carries the proved address expressions. An external
consumer must validate its backing sizes, offsets, scalar domain and grid itself;
the exported shader has no host-side safety checks. Shared storage is 13,088 bytes.
The tested sentinel is negative infinity. Source behavior is preserved: a fully
masked first block poisons the online recurrence with NaNs, including when later
blocks have valid elements. This is not silently replaced by a different softmax.

## Evidence and limits

On Apple M4 Max, both retained source branches passed independent output/LSE
checks at N=32/64/128/256, with two random fixtures, all-masked and first-block-masked
fixtures, two output poisons, input witnesses and canaries. The original Python
frontend and retained native IR emitted the same shader sources. The default
scalar emission and metadata remain byte-identical to the pre-change versions.

A separate warm external submit-and-wait experiment found a descriptive saving
for masked N=32: roughly 328 to 183 microseconds, with GPU timestamps roughly
193 to 56 microseconds. The other three branch/size cells had lower averages but
failed the predefined same-source noise screen; no qualified saving is claimed
for them. Compilation, allocation and validation were outside that timer. This
is not full-workflow latency, installed Python performance, a comparison against
MPS/XLA, or M1 evidence. No release obligation is closed by this experiment.

## Candidate integration

The 0.4.0rc1 companion workflow bundle contains the reporter adapter, five kernels,
source and license records, and a private PJRT plugin. It runs the reporter's
forward and explicit backward both eagerly and through Reactant/XLA. Those
adapter/plugin changes are separate projects; installing the Python wheel alone
does not install them. The bundle uses relative paths and an isolated Julia project.

The adapter packs the reporter's head/batch layout and uses a finite masked
sentinel. A separate out-of-place correction handles the reporter's fully masked
dV derivative. This is a disclosed adapter correction, not a change to the
compiler's general softmax semantics. The November 7 opt-in companion additionally
passed first-order automatic differentiation through the complete layer and six
SGD updates per orientation at Float32, N=32, D=32, H=2, B=1 on M4 Max. This uses
an explicit process-local Reactant compiler hook; it does not enable general
differentiation of arbitrary custom calls or remove the protective AD guard.
See [the training route and its exact wheel binding](EXTERNAL_TRAINING.md).
Forward-only and explicit-backward checks remain separate evidence. M4 execution
does not qualify M1 or M5, and no training-performance result is established.
