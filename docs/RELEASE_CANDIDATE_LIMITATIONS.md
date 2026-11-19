# 0.3.0 validation record and limitations

This is the 0.3.0 validation and performance record. In it, "candidate" means the
0.3.0 release candidate, and "public main" or "main" means commit `182c1820`, whose
code equals 0.2.0. 0.4.0 did not rerun these measurements. For what 0.4.0 changes,
read the [0.4.0 release guide](releases/0.4.0.md). A "0.4.0 note" marks a
statement that the 0.4.0 tree changes; the rest was not re-checked line by line.

Version 0.3.0 supersedes the unpublished rc1–rc4 snapshots. Earlier wheel hashes do
not identify the final artifacts; the runtime-payload binding is described below.

## Evidence and remaining limitations

The formatted 0.3.0 candidate at `73ba428` is bound to fresh packet 906 gates: 6,079 project passes / 16 skips and the exact 9,342-node upstream map (5,780 passes / 3,562 skips). Packet 907 builds, reproduces and relocates its native and pure validation artifacts. Packet 909 installed acceptance passes 1,468 native / 252 pure with no failures or skips; all four unchanged M4 Max performance floors pass, and the three-workload cold subset below computes correctly in every arm. Final documentation-only artifacts are rebuilt, reproduced from the sdist and rebound to these tested runtime payloads, with fresh and relocated CPU checks. Formatting changed Python bytes and cache identities, so the earlier runtime-byte-equality claim is not used to inherit the old gates.

The pre-format lineage remains separately attributed: packet 872 tested frozen871, packets 875/876 covered its rc4 artifacts and installed acceptance, and packet 878 re-executed the original and additive-IEEE portability contracts on that source in both local Metal modes against retained remote outputs. Packet 879 supplies the full main-relative warm tables below; packets 890/895 supply the later attention comparisons on pre-format final-version artifacts. These full warm comparisons and vendor receipts were not rerun on formatted bytes. Their historical scopes, failures and limitations remain unchanged; no warm alert is cleared.

### Formatted-candidate revalidation and cold subset (2026-09-16)

The 228-file formatting commit preserved 226 syntax trees excluding source locations; the other two changed only docstring whitespace. The reviewed corpus produced 38 identical emissions covering 35 distinct shaders and 29 unchanged refusal messages. This is finite corpus evidence, not universal emission equivalence. Two source-mutation test fixtures were updated for the formatter's spelling while preserving negative-kernel ASTs, assertions and parameter IDs. The installed flip witness's old literal producer hash was separately rebound to the AST-identical formatted file; all installed tests then reran. No numerical threshold, skip, refusal or capability was removed to obtain the passes.

Packet 909 retained the existing workload workers, inputs, source oracles and acceptance thresholds. Its single cold observations, in seconds, are:

| Workload | Public main | Formatted native wheel | Formatted pure wheel |
|---|---:|---:|---:|
| Add | 0.629 | 9.480 | 9.586 |
| Compiled MLP | 0.861 | 10.024 | 10.053 |
| GPT-2 small | 2.810 | 5.130 | 5.174 |

The timer covers the first synchronized specialization invocation, including compilation and the retained live observer, but excludes imports, input/model construction and oracle evaluation. Add starts in a fresh portable worker. MLP and then GPT-2 share a fresh model worker, so GPT-2 is not a fresh-interpreter start. The original 20 setup warmups and final checks remain, but the full warm-matrix timed loops between rows were omitted. These observations do not reproduce that matrix's entire process history and do not establish a formatting-driven improvement over packet 879. Cold costs remain substantial. The four passing floors are fixed thresholds, not clearance of any main-relative regression.

The 814→871 audit added faithful or fail-closed handling for width/layout guards, partial-K source extents and fast-path fallback, ordered quiet/propagating comparator reductions (including bounded multipass), native callee types, operation-owned transpose order/orientation chains, matmul/flip detector false positives and retained side effects, KDA state validation/copyback, and precise ambiguous diagnostics. This is finite tested coverage, not proof that every source is safe. Remaining unproved nested/MEPT/multipass/layout forms refuse as recorded in the capability register.

In packet 878's pre-format receipt, all 11 scoped exact-byte predicates pass across local native/pure Metal outputs and retained A40/MI300X outputs. Local Metal passes 16/16 original required numerical rows; each remote backend remains 14/16 because the two original default-precision attention rows fail. Separate additive explicit-IEEE contexts pass 2/2 on every arm. No default or tolerance changed. M1 is still user-run after merge. Optional C++ sources are packaged but the route remains deferred, unaudited and off by default. Performance regressions remain open and are quantified below.

### Cross-vendor precision and parity scope (2026-09-15)

The remote A40 and MI300X execute upstream Triton's CUDA/ROCm backends, not triton-msl.
The frozen test sources and inputs are shared with the installed native/pure Metal runs.
All 11 mandatory exact-byte predicates pass between every vendor pair, including both
Metal packaging modes. These are scoped output-byte checks, not universal byte equality,
archive reproducibility, or accuracy claims about every supported kernel.

The original 17-row contract contains 16 required rows and one optional TF32 diagnostic.
Both remote backends pass 14/16 required source-oracle checks. The two f32-intermediate
attention rows (`attention_16x32_narrow0_causal0` and
`attention_32x64_narrow0_causal0`) fail the retained 3e-5 absolute/relative accuracy
criterion under backend-default dot precision. Those failures remain in the original
receipts. Retained compiler IR identifies TF32/XF32 multiplication on the remote arms;
the higher-precision oracle did not model those backend-specific defaults. A matching
post-observation rounding model is diagnostic corroboration, not an independent hardware
oracle or permission to relabel the failed runs as passing.

Two **separate, additive** source contexts specify `input_precision='ieee'` at the dot
operations. Inputs, masks, sentinels, oracle arrays and tolerances are unchanged. Both
contexts pass first/warm/final checks on native Metal, pure Metal, A40 and MI300X, with
maximum error against the oracle below 3.58e-7 and maximum peer difference 1.79e-7.
No production kernel, source default, or numerical threshold changed. The optional TF32
diagnostic errors before execution on Metal and receives no Metal computation credit.
The receipt and actual toolchains are recorded in [`../PORTABILITY.md`](../PORTABILITY.md).

These results establish the named precision-explicit examples, not the reporter's complete
AlphaFold workload, all attention layouts, or default-precision equivalence between vendors.

## Validation observation boundary and opt-out

With the optional native helper available, a recognized unchanged-state saved-identity hit
performs callback-free checks. It omits Python key/descriptor/metadata observation callbacks
and the audit hooks those observations would trigger, including legitimate side effects.
Code that relies on those callbacks to change policy, veto a launch or record every such
observation must use `TRITON_MSL_IDENTITY_FAST_PATH=0`. This runs the complete evaluator on
every invocation and takes effect even when a saved record already exists. Unknown or changed
state falls back to the complete evaluator; the pure-Python installation does not use this
native shortcut.

This is an explicit observation-contract change, not permission to miss supported changes
made between calls. Live configuration, framework/provider and native-library checks remain;
owning-JIT invalidation/recompilation and stale direct-handle refusal are retained. Both
public-launch validations, descriptor/scalar/ABI checks, submission tracking and immediate
assertion observation stay in place. No delayed assertion boundary or unchecked policy reuse
is adopted.

The dictionary watcher certifies the exact-dictionary/exact-string-key domain, never cached
configuration values. The reviewed extension also caches immutable probe-schema decoding and
dictionary-key preflight under a watcher epoch, context serial and live-size check. Every
dictionary value and all residual type/closure/list/native-library checks remain live. Mutation
invalidates preparation and deallocation removes key entries; unavailable watcher slots or
capacity exhaustion falls back to scanning.

The program cache holds at most 16 entries. On a validation miss, before any complete-evaluator
input is read, a bounded cold collector detaches cache-exclusive programs before releasing
them. Finalizers may change policy or re-enter; hot hits never collect. Cyclic or externally
held programs can retain slots and lose the optimization without bypassing validation. The
entry bound is not a byte bound on the referenced object graph. The measured approximately
0.5–0.6 µs saving per validation is a component screen, not a new workload qualification.

Real MLX imports no longer necessarily force the complete evaluator: standard top-level
namespace paths are certified only when parent path, cache epoch, instance fields, method
code and dependencies prove recalculation is a no-op. Changed/unsupported namespaces use
the full path. This removes a host-validation fallback exposure, not a certification of the
whole MLX backend.

Native extensions must obey CPython's watcher-ID ownership rules. A foreign extension
clearing an ID it does not own can silently
defeat invalidation: the cached-hit path cannot detect that misuse. This helper is not a
security boundary against arbitrary native memory or interpreter-state corruption. The
supported native wheel is GIL-enabled CPython 3.14 on macOS 15+ arm64; subinterpreter imports
are unsupported, and the pure-Python wheel remains the fallback for other supported setups.

## Reporter workload coverage

- Issue #11: fp32 32×32 matmul tiles with a K loop, runtime alpha/beta and column bias replay
  the source epilogue. Tests include partial output grids and nontrivial scales. Arbitrary
  tile sizes, dtypes and epilogues are not certified; the literal source uses unmasked input
  loads, so callers must allocate the complete loaded tiles.
- PR #7: independent-length, widened-P/V decode computes with 8-/16-row query tiles, including
  folded Q=1 and fp16 Q scale-rounding. Query and key lengths stay distinct. Causal alignment
  follows the source's top-left comparison, not an inferred bottom-right convention.
- Bias-without-LSE replay covers 32×32 tiles, head dimensions 32/64, fp16/fp32, independent
  lengths, non-unit bias strides and causal masks. Tests reconstruct this feature from the PR
  description, not the reporter's unavailable full biased workload. The later tile/precision and wide-row recoveries below have separate admission proofs;
  arbitrary combinations remain unqualified.
- A runtime **attention score** scale after QK is not admitted merely because the separate
  matmul epilogue now computes. Moving a scale to Q changes rounding and is not universally
  equivalent.

- Low-precision and rectangular K-loop epilogues (issue #11 family) replay for f16/bf16/f32
  storage and 16/32/64 tiles with f32 accumulation and f32 epilogue arithmetic. Activations,
  clamps, fused multiply-adds, row bias and non-tile-boundary masks refuse. Tiles needing more
  than 32 KiB of threadgroup memory after scratch reuse (64×64×64) refuse with a typed capacity
  error; 64×64×32, 64×32×64 and 32×64×64 sit exactly at the limit.
- Biased attention replay also covers 8/16-row query tiles, bf16 storage, explicit f16
  probabilities and D64 causal rows. Separate full-row replay computes tested fp16/D64
  32×64 noncausal (NQ=19/NK=97) and 64×32 causal (NQ=65/NK=97) cases. It distinguishes the native
  f32 -Inf sentinel from finite 64512.0 and preserves all program input reads before output
  stores, including the tested shared Bias/Out allocation. Arbitrary dtype/tile/recurrence
  combinations are not certified. Explicit bf16 dot operands, D128 and post-QK score scaling
  remain named limitations; single-key nk=1 specialization is a separately recorded refusal.
  Accepted 553/557 score reuse reduces this from 130 score evaluations to one per key block
  without changing the tested numerical/alias contracts. Performance is unqualified.

Other limits include unprovable pointer roles, some multi-reduction layouts, narrow backward
delta recurrences and shapes over the threadgroup budget. See the actual diagnostic and
[capability register](CAPABILITY_AND_REFUSAL_CENSUS.md). Correct-or-refuse is the design
objective, not a proven universal absence of defects.

## Corrected silent-wrong: generic two-dimensional reductions (present since 0.2.0)

Before the allocator repair, threadgroup scratch reuse (`_alias_shared_memory`) considered
nonoverlapping textual first/last uses sufficient. Before this candidate the generic axis-1 reducer wrote its result on the line
after its own row loop, so the result array could be merged onto the input array; reducer `k`
then overwrote input slot `k` (row `k // N`) while a reducer in the first SIMD group could still
be reading that row. Measured on one M4 Max: wrong rows in about 2 of 100 launches of a 128×8
row-softmax tile (8,192 threadgroups per launch), on the 0.2.0 source and on the earlier rc4
candidate alike; 64×16 row-sum tiles showed the same family less often. Kernels whose reducers
fit one SIMD group (≤ 32 rows on axis 1, including the 32-row attention tiles) were not observed
to fail, and axis-0 reductions were structurally unaffected by this specific handoff. Generic
axis-1 reductions with more than 32 rows could be exposed when emitted with this alias pattern. The repair holds the result in registers, barriers,
then writes it under a lane guard on both axes; emission pins assert that ordering for both
axes. The 0.2.0 source contains the defect; users of that code reducing tiles with more than
32 rows through this generic alias pattern should treat those results as suspect until they upgrade.

The allocator now requires proved synchronization between reused lifetimes and checks loop
backedges/control flow (529). This closes the additional 516 handoff findings: reduce→scan
(public since 0.2.0) and join→split (local lineage; introduced by 6ab6db7, already in local main)
were witnessed numeric races. The argmin/argmax scratch handoff and two atomic handoff
families were structural concerns, not observed numeric failures. In the bounded 529 replay,
parent D/E cases were wrong 60/60 launches; repaired cases and the small control were wrong
0/80 launches. The 1,230 unique-emission audit found 0 remaining checker flags and no newly
exceeded 32 KiB capacity. Focused 529 tests passed 456 contracts; this is finite corpus evidence,
not a universal absence-of-races proof or final integrated gate.

The independent public output-offset defect C is repaired by 527: axis-1 argmin/argmax 1-D stores
retain program/runtime offsets and remap source pointer/mask coordinates to the result row.
Final focused 51 contracts and 16 adversarial live-launch GPU rows cover both reducers,
8×128/64×16 tiles, multiple programs, masks/strides/selects, negative offset views and
rank-1/axis-0/2-D-store controls with exact sentinels. A nonuniform tensor constant mask could
previously emit as always true; this store now refuses unparsed/nonuniform tensor constants
and unproved data-dependent address leaves. Scalar and true/false splat forms remain admitted.
An additional direct-store reproduction compiled an alternating dense boolean mask as `if (1)`
through the public native-IR compile entry point. The generic constant emitter now refuses
unparsed/nonuniform literals rather than coercing them to true or zero. It preserves decoded
numeric scalar/splats and typed special-float words; it does not implement general nonuniform
tensor constants or certify every constant consumer. The full integrated admission/status
ratchet now passes on 589; this does not prove every source is safe. Public 0.2.0 users should treat results from the affected 512/C/D
source patterns as suspect until using a validated corrected artifact; no advisory was sent.

## Cross-process identity memo feasibility screen — not adopted

Packet 883 measured the existing execution-contract capture at 17.53 s, including 10.03 s for the tree stage and 6.52 s for the first resolver. In three fresh-process optimistic screens with a preloaded manifest, identity work alone took 2.11–2.23 s (median 2.16173 s). Those screens excluded stat traversal, cache I/O, compiler and GPU work, so the tree-only proposal could not meet the declared approximately 1.5 s target under its own optimistic conditions.

No persistent memo was implemented or adopted, and exact871 is unchanged. This is not a finding of no possible benefit, an absolute performance ceiling, or proof that another mechanism cannot improve cold start. The default continues to content-hash the relevant trees once per interpreter. The p879 first-specialization figures are retained pre-format observations; the formatted cold subset is reported separately above.

0.4.0 note: this describes 0.3.0. In 0.4.0, a later process can instead validate and reuse a per-user identity record that holds the identity the full computation produced. `TRITON_MSL_IDENTITY_RECORD=0` keeps the 0.3.0 behaviour. See the [0.4.0 release guide](releases/0.4.0.md#start-up-identity-record).

## Latest composed main-relative evidence: exact 871 (pre-format)

The September 16, 2026 installed-wheel comparisons use public main `182c1820fd24a836d565e1da842f28414de64084` and exact frozen 871 on one M4 Max. Each mode retains 11,520 samples under the unchanged p527 protocol. Main is correct on 18 comparable rows, wrong on six and refuses six; the candidate computes all 30 required rows. Wrong/refusing main is untimed and receives no ratio. The optional TF32 diagnostic errors without execution credit.

Ratios are medians of paired block-median ratios. The table reports excess per request and a separate launch-count column; it does not report per-launch excess. Neither quantity is a GPU timestamp or causal decomposition. Native and pure are separate sessions.

### Native wheel — 13 open alerts above 10%

| Workload | Candidate / main | Excess µs/request | Backend launches/request |
|---|---:|---:|---:|
| add | 1.1984 | 25.9 | 1 |
| softmax | 1.2378 | 24.3 | 1 |
| matmul_ieee | 1.1913 | 26.3 | 1 |
| varlen_48_32 | 0.1154 | -1648.7 | 1 |
| layernorm_1024x1024 | 1.1828 | 25.2 | 1 |
| quant_decode_512x1024 | 1.1943 | 24.4 | 1 |
| fa512_f4_causal0 | 1.1595 | 30.1 | 1 |
| fa512_f4_causal1 | 1.1784 | 33.2 | 1 |
| fa512_f2_causal0 | 1.1507 | 27.4 | 1 |
| fa512_f2_causal1 | 1.1819 | 33.8 | 1 |
| mm2048_f4 | 1.0373 | 58.1 | 1 |
| mm2048_f2 | 1.0363 | 52.8 | 1 |
| bandwidth_16M | 1.0476 | 27.6 | 1 |
| assert_valid | 1.1929 | 23.4 | 1 |
| mla_canonical | 1.2791 | 52.9 | 1 |
| compiled_mlp | 1.1309 | 39.0 | 2 |
| gpt2_small | 1.2127 | 728.4 | 43 |
| attention_forward_backward | 0.9973 | -1.1 | 3 |

This mode has 13 measured alerts above the unchanged 10% threshold. They are open, not accepted or declared irreducible.

### Pure-Python wheel — 14 open alerts above 10%

| Workload | Candidate / main | Excess µs/request | Backend launches/request |
|---|---:|---:|---:|
| add | 1.3768 | 50.4 | 1 |
| softmax | 1.4131 | 50.6 | 1 |
| matmul_ieee | 1.3903 | 57.8 | 1 |
| varlen_48_32 | 0.1315 | -1630.2 | 1 |
| layernorm_1024x1024 | 1.3645 | 51.6 | 1 |
| quant_decode_512x1024 | 1.4166 | 53.8 | 1 |
| fa512_f4_causal0 | 1.3241 | 59.7 | 1 |
| fa512_f4_causal1 | 1.3674 | 65.8 | 1 |
| fa512_f2_causal0 | 1.3275 | 63.2 | 1 |
| fa512_f2_causal1 | 1.3748 | 67.2 | 1 |
| mm2048_f4 | 1.0691 | 109.4 | 1 |
| mm2048_f2 | 1.0611 | 88.7 | 1 |
| bandwidth_16M | 1.1028 | 61.4 | 1 |
| assert_valid | 1.4332 | 48.7 | 1 |
| mla_canonical | 1.4986 | 91.3 | 1 |
| compiled_mlp | 1.2310 | 70.9 | 2 |
| gpt2_small | 1.4109 | 1408.9 | 43 |
| attention_forward_backward | 1.0009 | 0.3 | 3 |

This mode has 14 measured alerts above the unchanged 10% threshold. They are open, not accepted or declared irreducible.

The varlen ratios above are historical, unqualified observations. The October 31
baseline-admissibility investigation subsequently found the old comparator unsafe;
matching numerical outputs in the September run did not establish memory safety.
Do not cite 0.1154/0.1315 as an approximately eightfold speedup. The safe replacement
comparison used a shader byte-identical to the current shader, so it cannot test or
refute the old kernel-speed claim. The original rows remain here as historical
records, not accepted performance evidence.

These alerts are not universal model slowdowns and no exhaustive lower bound or proven irreducibility exists. Public JIT and Inductor have different validator counts. Both public validations and immediate assertion observation remain enabled; no timing recovery may weaken them.

### Cold first-call observations

- Native add: main 0.73 s, candidate 12.07 s; pure add: main 0.63 s, candidate 10.43 s.
- Native compiled MLP: main 1.04 s, candidate 18.17 s; pure: main 1.06 s, candidate 18.24 s.
- Native GPT-2 small: main 2.75 s, candidate 5.38 s; pure: main 2.73 s, candidate 5.30 s.

These first-specialization measurements include compilation, live observation and process residency. They are single cold observations, not whole-application startup times, repeated warm charges, isolated compiler costs or accepted costs. Cold performance is not fixed or qualified.

The separate KDA screen reports decode 1.0785× and prefill **1.1016×** main (paired block-ratio range 1.0721–1.1313). Prefill therefore exceeds the unchanged 10% alert threshold; it is a separate open alert and is not folded into the 13-row native matrix count.

The p877 padded partial-K diagnostic found main numerically wrong at K=33 and therefore did not time it; exact871 used the source-faithful generic fallback. That establishes correctness/fallback behavior, not a relative cost bound. Ordered comparator reductions use bounded order-preserving trees for admitted full logical reductions, including tested 1024/4096 extents; unsupported nested, MEPT or unproved shape/order cases refuse. No claim is made that all ordered-reduction costs are resolved.

All four absolute floors pass. Neither those floors nor these measurements accept the relative regressions, qualify historical throughput claims, or authorize release.

### MLA context length: historical 661 measurements, before P1

The following same-session main comparison used the earlier 661 native wheel, not exact871.
All three lengths passed their source oracle and route checks. Ratios and excesses are
medians of paired block quantities; the separate columns need not add exactly.

| Key length N | 661 / main total latency | Public-return excess | Final-sync excess |
|---:|---:|---:|---:|
| 32 | 1.327× | 53.4 µs | 5.7 µs |
| 512 | 1.305× | 49.8 µs | 20.5 µs |
| 2048 | 1.187× | 56.2 µs | 107.7 µs |

These observations show an approximately fixed public-return charge and a completion-wait
excess that grows with length. Public-return time is not a pure CPU-work measurement, and
final synchronization is not a hardware GPU timestamp. Retained shader differences identify
the nonfinite-safe accumulator rescale's scratch traffic and synchronization as a kernel-side
cost candidate, also present in ordinary attention. A separate, narrowly controlled P1
shader-enqueue-plus-completion comparison reduced latency by about 2.6–5.4% at N=512/2048
while retaining bitwise output agreement on the tested cases. It supports that mechanism;
it does not quantify its share of the full-workload regression.

P1 batches this work only for half-input/default-float accumulation. The fp32-input and
opt-in half-accumulate paths remain unchanged. The latest measured exact871 N=32 full-call ratios
are 1.2791× native and 1.4986× pure in their separate sessions above. The larger-N 661 results
are historical, not fresh larger-N receipts for exact871; no same-session
full-call improvement from 661 to exact871 is established here. Remaining costs
are not established as irreducible or accepted for release.

### Historical half-accumulate docstring is not an accuracy bound

The shipped `make_flash_attention_kernel_simdgroup` docstring retains historical statements
of approximately 4% speedup, 1% maximum absolute error and 0.01% for fp32 accumulation.
**None of those figures is validated for exact871, and none is a universal accuracy
bound.** This qualification applies to those docstring figures as well as the historical
README tuning numbers. Accuracy depends on inputs, shapes and operation order; validate
the opt-in mode on the actual workload. It remains off by default. The 671 P1 results use
default float accumulation and provide no new evidence for the opt-in half-accumulate mode.
Frozen runtime bytes were preserved; retaining the old text is not endorsement of its numbers.

## Assertions and dispatch

Supported retained device assertions stop the workgroup before guarded accesses, record a
launch-local message and raise on the host before results return. Unsupported forms still
refuse. Proven native scalar-true assertions can be discharged; data-dependent checks remain.
Assertion-bearing launches currently synchronize/read back a flag. A separate historical paired A/B against
earlier candidate 439 measured **+15.1% latency for a retained-assertion kernel** and **+11.9%
for GPT-2 small**, with the other nine tested workloads within ±1.2%. These are open alerts,
not merely unmeasured costs. Both runs used the same sources and valid inputs on one M4 Max,
10 blocks of 20 individually alternating pairs, without filtering samples. They do not
establish universal bounds or qualify the published PyTorch/MLX ratios, and are not the
retained exact871 main-relative performance summary above.

The GPT-2 workload checks **2 of its 43** backend launches; one also changes threadgroup
width to preserve assertion execution. Readback waits include preceding GPU work, so their
duration is not an isolated sync penalty. Later diagnostics attribute a launch-validation
component and an immediate-observation component, but neither this historical A/B nor the
total-latency table is a complete additive decomposition of every launch's cost.
The earlier candidate silently omitted retained checks. Those checks will not be removed or
weakened to recover its timing.

A pre-invocation eligibility miss may take a slower supported route. Once invocation has been
attempted, an error raises without fallback replay, even when enqueue cannot be proven.

## Current attention comparators (890/895)

Packet 890 measures the final installed 0.3.0 native and pure wheels against current
PyTorch MPS SDPA and MLX fast SDPA: 42 cases per wheel, 84 passes, 25,200 timed calls.
Both modes use the same input bytes and JIT bodies, AST-equal to checked public main
182c1820fd24a836d565e1da842f28414de64084. M4 Max, macOS 26.6, Python 3.14.4,
Torch 2.12.1, Triton 3.7.0, MLX 0.32.1. This is not the full August environment.

Ratios below are comparator latency / backend latency; above 1 favors triton-msl.
The dense subset is batch 2, 8 heads, D128, N1024/2048/4096. The full 36-row dense
panel also includes D64, N512 and batch-1 README anchors. MLA is batch 1, 8 heads,
split Q/K widths 128+64, V width 128.

| Comparison | Native wheel | Pure wheel |
|---|---:|---:|
| Dense fp16 full / SDPA | 1.94–2.18× | 1.18–1.50× |
| Dense fp32 full / SDPA | 1.36–1.41× | 0.94–1.16× |
| Dense fp16 causal / SDPA | 2.72–3.70× | 2.09–2.58× |
| Dense fp32 causal / SDPA | 2.09–2.53× | 1.59–2.10× |
| Full MLA N1024 / prejoined SDPA | 1.08× | 0.99× |
| Full MLA N2048 / prejoined SDPA | 1.42× | 1.14× |
| Dense median wins / SDPA, full panel | 36/36 | 34/36 |
| Dense median wins / MLX, full panel | 0/36 | 0/36 |

The native wheel takes 23–44% longer than MLX across the full dense panel
(0.69–0.82× speed, rounded), with every observed native dense block favoring MLX.
Historical near-MLX parity is not sustained. MLA depends on size and boundary:
N512 loses to prejoined SDPA, while causal N2048 beats MLX by about 1.24× native
and 1.23× pure. Candidate MLA concatenation is timed; MLX and prejoined SDPA receive
already joined Q/K. The separate split-input SDPA arm includes concatenation;
native full N1024/2048 speedups against it are 1.215×/1.483×.

Method and limits: warmed public calls plus framework synchronization, not isolated
GPU time. MLX uses the optimized `mask="causal"`. Candidate output is preallocated
as in the historical benchmark; comparator API output allocation remains timed.
Six counterbalanced blocks per dense row, eight per MLA, 15 calls/arm/block;
reported ratios are medians of paired block-median ratios. Observed block ranges
are not confidence intervals. Four native dense SDPA comparisons have blocks
crossing parity despite winning medians; native full MLA N1024 versus prejoined
SDPA also crosses parity. Pure results vary substantially, including near-one
fp32 medians. Native and pure ran in separate sessions: do not attribute their
entire difference to missing native helpers.

Every output element is checked against SDPA, plus a sampled float64 CPU oracle
over distributed query positions, every head and all keys; the latter is not a
full-output double oracle. Predeclared fp16 atol/rtol are 5e-3/5e-3; fp32 uses
3e-5/3e-4. Inputs, outputs, actual dispatched shaders and block samples are
retained. This measurement is not an exceptional-value rerun. All existing
main-relative warm alerts and cold costs remain separate and open.

### Same-stack backend attribution (895)

A follow-up compares the installed native final wheel with checked public-main
source 182c1820 on the same current stack, using separate persistent processes
and alternating their timing blocks under one exclusive GPU controller. Twelve
dense D128 rows (batch 2, 8 heads, N1024/2048/4096, both dtypes and causal modes)
plus full/causal MLA N2048 pass on both backends: 28 passing cases, 8,400 timed
calls. Kernel bodies, input bytes, numerical checks and comparator APIs are the
same as 890. The pure wheel was not rerun in this attribution experiment.

| Group | Final/main paired latency ratio |
|---|---:|
| Dense fp16 full | 1.088–1.114× |
| Dense fp16 causal | 1.085–1.223× |
| Dense fp32 full | 1.141–1.165× |
| Dense fp32 causal | 1.148–1.291× |
| MLA N2048 full / causal | 1.125× / 1.131× |

Above 1 means final is slower. These are medians of corresponding block-median
ratios, not ratios of independently aggregated absolute times. Observed paired
blocks cross parity for fp16 N4096 full and fp32 N4096 causal; the other twelve
comparisons are above 1 in all observed blocks. Block ranges are not confidence
intervals. Both backends beat SDPA on all 12 dense medians. Main already trails
MLX on those medians (0.828–0.948× MLX speed); final trails further
(0.713–0.827×). Thus the update contributes to today's MLX gap. This does not
identify the cost of each code hunk or recreate historical MLX: it cannot
apportion the full August-to-current change. The main arm is source-bound to
the named commit, not a new installed PyPI 0.2.0 compatibility test.

## What remains unqualified

- **Performance:** except for the scoped current attention comparisons above, README ratios
  and absolute rates are historical until individually remeasured and tied to this artifact.
  Correctness does not qualify them. The four absolute
  floors (7.0 / 5.5 / 5.5 TFLOP/s and 250 GB/s) are M4 Max sentinels, not M1 budgets. An earlier
  in-suite failure remains unattributed; no threshold was lowered.
- **C++:** optional, unaudited and off by default. Enabling TRITON_MSL_USE_CPP=1 enters an
  unqualified surface; TRITON_MSL_FORCE_PYTHON=1 suppresses it. Default-off is not validation.
- **AlphaFold:** reporter-relevant classes were tested, not the full original workload.
- **Hardware:** local validation uses one M4 Max. Independent M1/M2/M3 acceptance is unrun.
  The M1 kit must name the final version and hashes before it is sent.
- **Exceptional values:** the [axis-to-test map](CAPABILITY_AND_REFUSAL_CENSUS.md) separates
  exact classification, finite controls and order-robust bounds. Welford m2 checks are
  admissibility checks, not an exact non-finite recurrence oracle or exhaustive coverage.

- **In-place outputs:** a matmul whose output overlaps an operand keeps the source's
  load-then-store order by leaving the replacement template; the quantized routes and the
  attention kernels refuse such a launch by name, except the two biased source replays, which
  keep the order behind a barrier and compute in place. The attention refusal compares whole
  storages: an output is refused unless its storage is different from every other tensor
  argument's, and a storage that cannot be read counts as shared. Views of one storage whose
  cells stay apart, such as the slots of one packed buffer, are refused with the rest, because
  the kernel addresses its tensors through its own stride and length arguments, not the views'.
  Write the output to a tensor of its own. An ordinary kernel is checked before every launch:
  where two lanes of one program are not proved to stay apart at an element one of them stores,
  the launch raises `MetalNonRecoverableError` starting `lane order:` and nothing is submitted.
  That refuses an in-place reverse, transpose, shift or gather, a cell every lane loads and the
  program stores, an exchange or compare-and-swap whose lanes meet at a cell, an exchange,
  compare-and-swap or add with a used result inside a `while` loop unless its address is the
  same on every trip and keeps each lane to cells of its own, a store over a
  block narrower than the kernel's widest block that shares a tensor with a block-wide store,
  and some launches the check cannot follow although the shader matched the source in every
  probe; the changelog lists them. A kernel in which each lane loads and stores its own elements
  computes in place. The check does not look at conflicts between different programs of one
  launch, which are races in the source, and is not applied to a launcher built from metadata
  without an access plan. A launch that needs more than 16 comparison runs (`x[:, 1:] += y` at
  8192 by 8192) is refused.
- **Float atomic adds:** two float `tl.atomic_add` operations, as two statements or as one in a
  loop, that reach one cell from different lanes are refused before submission. The shader
  applies them lane by lane, and the order of float additions is their rounding (`1e8 + 1 - 1e8`
  is 0.0 in the source's order and 1.0 in the other). The check reads addresses, not values, so
  a float histogram or scatter-add accumulated over the trips of a loop is refused also where
  no order would round. A mask made of loaded values, a branch and a loaded index are not
  evaluated, so adds that only one of those keeps to one lane per cell are refused too, and
  the message says which. In a `while` loop a float add computes only where its address is the
  same on every trip and keeps each lane to its own cell.
  Integer atomic adds, a float add of each lane to its own cell, and one float add outside a
  loop compute.
- **Row-wise sort:** the sort template runs only a kernel proved to be one unmasked load of a
  tile, `tl.sort` along its last axis and one unmasked store, at addresses of the form
  `base + row * stride + column`. A sort with an offset or a program term in an address, a
  mask, arithmetic on the values or a second rearrangement goes to the ordinary lowering,
  which computes it or refuses it when the kernel is lowered. The ordinary lowering runs a
  tile of at most 1024 elements: a larger sort that is not the plain form is refused with
  that reason. A converting store, a load of reversed columns, a mask that is true everywhere
  and a sort of a sort returned the source's result at 64 by 32 and 512 by 8 before.
- **Blocks narrower than the kernel:** a store or an atomic over a block with fewer elements
  than the kernel's widest block computes only where the lowering proves which lane holds each
  element's address, value and mask. Otherwise the kernel is refused when it is lowered, by
  operation, block and reason. Refused where earlier builds were wrong: a one-dimensional
  running total laid along the rows of a narrow tile, and a narrow tile copy into an output
  that does not start on a 16-byte boundary at some tile shapes. Refused although earlier
  builds returned the source's result in probe launches: an atomic over a narrow block whose
  returned value is stored, and a tile indexed through `tl.reshape` of a range in a
  three-dimensional kernel. A block made inside a called function that is not inlined is
  refused when the kernel is lowered, as before. A tile flattened with `tl.reshape` beside a
  wider block, where the tile is loaded: an atomic add computes; a store computes for the
  tiles of 128 elements and more that were tried and is refused for tiles of 64 elements or
  fewer; a `tl.sum` computes for a tile of up to 64 elements beside a block of up to 128 lanes
  and is refused beside a block of 256 lanes or more. A flattened tile
  whose values are computed from its indices with no load is refused when the kernel is
  lowered. A tile transposed with `tl.trans` computes alone and beside a block no wider, with
  one row or one column as well, and is refused beside a wider block. A kernel that needs one
  thread per lane over
  a block of more than 1024 lanes is refused by block size, also when its mask keeps every
  active lane below 1024.
- **Half accumulators in matmul:** a matmul kernel that keeps a float16 accumulator across its
  K blocks (`acc += tl.dot(a, b, out_dtype=tl.float16)`) is accumulated in float32 and rounded
  to float16 once, at the store, where the source rounds after every block. In a 64 by 64 probe
  with two K blocks 555 of 4096 cells differed from the source's per-block rounding and none
  from the product rounded once, on the replacement template and the ordinary entry alike.
- **Scratch aliasing:** the conservative 529 allocator and 527 store fixes have bounded focused
  evidence and passing combined 589 gates. Cross-vendor/artifact evidence remains separate.

## Shutdown diagnostics

A historical installed-wheel run printed an upstream Triton finalizer AttributeError involving
kernel_unload_hook during teardown, after correct output with exit status zero. Later simpler
RC runs did not reproduce it. It remains disclosed, not suppressed or represented as fixed.
A similar-looking error during execution must not be dismissed as this shutdown issue.

## Reporting

Report minimal source, inputs, version, device and full diagnostics at
https://github.com/bledden/triton-msl/issues. Unexpected refusals and silent wrong values both
matter.
