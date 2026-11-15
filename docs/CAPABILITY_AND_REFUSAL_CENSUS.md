# Capability & refusal evidence register: 0.3.0 record and 0.4.0 transitions

Parts 1 to 3 record the 0.3.0 release-candidate evidence unchanged. Section 2e
lists transitions in 0.4.0. Their evidence is the named code and tests, not a new
census of every kernel.

triton-msl follows a **correct-or-refuse** contract: a kernel it accepts is meant to compute
correctly, and one it cannot prove it can lower safely raises `MetalNonRecoverableError` rather
than returning a wrong answer. This repository-only document records scoped capability and transition evidence. It is not an
exhaustive paired before/after census of all kernels or a claim that every refusal has been probed.
`MANIFEST.in` prunes `docs/`; release-facing limitations must also appear in the actual delivered materials.

> **Scope note.** Capability rows cite the specific evidence that established them. The reviewed
> dispatch, grouped-matmul and #8/#9 changes coexist with 479/483/485/489/493 and the later
> 527/529/553/557/567 repairs in the current development candidate. Historical
> sibling evidence is identified below; it does not by itself imply combined-tree validation.
> Historical 589 integration evidence remains attributed below. Current exact871 gates and installed acceptance are recorded separately; historical packet evidence remains attributed to its original tree.

---

## Part 1 — Verified working (capability map)

| capability | evidence | executed route |
|---|---|---|
| element-wise (vector add) via compile_shader | exact | `vadd` |
| fast matmul fp32 / fp16 | exact vs fp64 ref | `simdgroup_matmul_fast` |
| int8 weight-only matmul | 3.05e-05 | `int8_matmul_fast` |
| FlashAttention forward (MPS + host-tensor paths) | 4–6e-7 | tiled / host-marshalling |
| biased / triangle attention (masked, `-inf`) | 4.8–7.2e-7 | `_biased_fa` |
| NaN-row locality (row-local, no cross-contamination) | bitwise | — |
| reduce (sum, max), cumsum, atomic_add | 6e-6 – 1e-5 | — |
| signed-zero-preserving sum | `-0.0` retained | — |
| grouped matmul (canonical signed-i32 cdiv mapping) | exact vs fp64, incl. partial grid | native coordinate-preserving template; evidence 459, integrated 475 |
| equal Q64/K64, noncausal, HEAD_DIM=64 | fp16 2.44e-4; fp32 5.96e-7 | native launch witnessed in sibling 459; census 461 |
| separate Q32/K64, HEAD_DIM=64 | fp32 only tested, causal/noncausal ≤5.75e-7 | native launch witnessed in sibling 459; census 461 |
| separate Q8/K128, HEAD_DIM=64, BM=BN=32 | fp32 noncausal, 2.38e-7 | native launch witnessed in sibling 459; census 461 |

The historical 479 and 493 upstream gates each reported 5,780 passed / 3,562 skipped,
with all 9,342 node statuses baseline-identical. Those results predate the current
development candidate. Packet 589's historical same-seed project gate passed 4,792 / 12 skipped /
zero failures, with four performance sentinels scheduled separately. Its upstream gate
also passes 5,780 / 3,562 skipped / zero failures; independent complete JUnit/node-map
reparses confirm all 9,342 statuses equal the baseline. Final installed artifact
acceptance is separate from those historical gates.

Current combined-runtime evidence: packet 872 on frozen871 records 6,079 project passes / 16 skips and exact 9,342-node upstream identity (5,780 passes / 3,562 skips). Packet 876 records installed acceptance of 1,468 native / 252 pure passes with no failures/skips. Packet 879 records four passing M4 Max floors and separate 13-native / 14-pure warm alert tables; KDA prefill is a separate 1.1016× alert. These receipts do not expand any capability row's source/shape envelope.

The 814→871 sweep adds pins for width and partial-K extent proofs, ordered comparator reductions, native callee types, operation-owned transpose chains, detector whole-value/side-effect checks, KDA writable-state ABI, and fail-closed unsupported layouts. Nested/MEPT/unproved ordered forms remain refusals rather than inferred identities or reassociated operators.

Scoped recoveries (original focused evidence retained; current combined gates recorded above):

| workload | qualified development envelope | pins |
|---|---|---|
| PR6 N-fastest flat-grid matmul | exact signed-i32 coordinate proof; source grid retained | `test_flat_matmul_mapping.py` |
| Issue11 runtime alpha/beta, K-loop and column bias | fp32 32×32×32 tiles; partial output grids | `test_loop_matmul_epilogue.py`, `test_template_scalar_abi.py` |
| PR7 independent query/key lengths, folded Q1 | fp16/fp32; 32-row generic and 8-/16-row tiled source spelling; widened P/V dot | `test_q1_decode_boundary.py` |
| PR7 bias without LSE | 32×32 tiles, D32/64, fp16/fp32, independent lengths, causal/noncausal, strided bias | `test_biased_decode_replay.py` |
| Issue11 low-precision / rectangular K-loop epilogues | f16/bf16/f32 storage; 16/32/64 tiles; f32 accumulation and epilogue arithmetic; ≤ 32 KiB threadgroup after reuse | `test_epilogue_capability.py` |
| PR7 biased attention precision / query tiles | 8/16/32-row query × 32-key tiles; D32/64; fp16/bf16 storage; f32 or explicit f16 probabilities; top-left causal; independent lengths incl. folded Q=1 | `test_attention_capability.py` |
| PR7 wide biased attention | tested fp16/D64 32×64 noncausal NQ=19/NK=97 and 64×32 causal NQ=65/NK=97; typed f32 -Inf; disjoint/overlapping Bias/Out; zero replay shared bytes | `test_attention_capability.py` (515/523) |
| Argmin/argmax output offsets | axis1 8×128/64×16, 3 programs; runtime offset/stride/select/mask; negative offset views; exact sentinels | `test_arg_store_offsets.py`, p527/final-focused |

---

## Part 2 — Refusal-transition census (A)

The rows below distinguish observed repairs, tighter rejection, and remaining capability limits.
A census-only computation is not a recovery without a same-source baseline observation. "Pinned in" names the tree
that carries the regression test; reviewer-sibling results remain attributed to their original
tree; the current combined-gate receipts above are separate evidence, not relabeled sibling runs.

### 2a. RECOVERED — refusal (or silent-wrong) → computes the source

| what | before | after | evidence / packet | pinned in |
|---|---|---|---|---|
| trifast #1 renamed reduction axis (K-by-name dropped the K-loop) | silent-wrong (err 23.8) | structural `scf.for` upper-bound trace | commit `5b8f815` | committed lineage (`test_issue4_trifast.py`) |
| trifast #2 loop-carried pointer idiom | silent-wrong / compile-err | MEPT pointer-offset carry | `2758c1e`/`392ae75` | committed lineage |
| trifast #4/#5 matmul extent | silent-wrong | structural `_K`/`_M`/`_N` | `8385e6f`/`cd80b44` | committed lineage |
| trifast #6a constant QK-result score scale | refuse | capability (also closed a live varlen silent-wrong) | `b34627d` | committed lineage |
| trifast #6b multi-output copy-back + loop-carried 1-D layout | silent-wrong (O never copied back) | computes (FA lse 4.8e-7; GEMV 3.8e-6) | `4c8cbbf`+`388d1cd` | committed lineage |
| trifast #7 argument-buffer path | refuse | computes | `20d5c7c`→`aa279d1` | committed lineage |
| PR5 1-D store after axis-1 reduce (offset dropped) | silent-wrong | computes (covered by reduce-store + #6b work) | triage 450, confirmed | committed lineage |
| **#9 broadcast index collision — bare / arith / gather** | silent-wrong (2nd broadcast copied 1st's stride) | computes | packets 464/468/472 | integrated; `test_broadcast_index_collision.py` |
| **#9 residual — comparison `(range>0).to(int32)` / select `where(range==0,7,3)`** | silent-wrong (on candidate AND baseline) | computes | packets 468/472 | integrated; `test_broadcast_index_collision.py` |
| PR6 grouped matmul (canonical cdiv mapping) | refuse | computes (full/short-final/column-tail/partial/overflow) | packet 459 | integrated 475: `test_grouped_matmul_mapping.py` |
| **generic 2-D reduction result aliased onto its input (public since 0.2.0)** | silent-wrong (wrong rows in ~2% of launches; tiles with > 32 reducers) | register result → barrier → guarded write, both axes | packets 512 (GPU witnesses on rc4 emission and the `v0.2.0` archive), 517 (independent confirmation) | 511 `_lowerer_reduce.py` hunk; two-axis emission pins `test_reduce_alias_barrier.py` (sibling `daybreak-reduce-pin-517`) |
| Issue11 epilogues: f16/bf16 storage, rectangular 16/32/64 tiles | refuse (nine of ten cases) | computes, dyadic-exact vs fp64 | packets 509, 510 | `test_epilogue_capability.py` (509) |
| PR7 biased attention: 8/16-row query tiles, bf16 storage, explicit f16 probabilities, D64 causal | refuse (all six rows on 509) | computes (≤ 3e-5 vs fp64 oracle; independent attack rows ≤ 9.1e-7 with f32 output) | packets 511, 512 | `test_attention_capability.py` (511) |
| Generic scratch reuse B/D/E/F/G | unsafe textual-lifetime aliases; D/E witnessed wrong, B/F/G structural | synchronization-separated aliases including loop backedges; D/E parent 60/60 wrong versus repair 0/80 wrong; 1,230 emissions, 0 audit flags | 529; focused 456 passes, combined 589 gates pass | `test_shared_pool_epochs.py` |
| Public argmin/argmax axis1 scalar/program output offsets (C) | deterministic wrong addresses | pointer/mask SSA remapped at result row; final 51 focused passes plus 16 exact adversarial GPU rows | 527 | `test_arg_store_offsets.py` |

Historical 461 was capability evidence, not a measured transition. Subsequent 485/493 recover
the previously refusing Q1/small-tile sources and biased-no-LSE source. Candidate pins copied to
the exact pre-493 implementation bite; positive rows execute on 493. These recoveries do not
establish arbitrary combinations of bias, tile sizes, score casts or attention recurrences.

### 2b. TIGHTENED — silent behaviour → honest refusal

| what | before | after | evidence / packet | pinned in |
|---|---|---|---|---|
| **#8 retained `tl.device_assert` (source or framework, including Inductor defaults)** | silently elided at direct generic lowering; blanket refusal in 475 broke 11 framework workloads | supported generic checks use a uniform threadgroup stop and launch-local host error; unsupported forms and callee regions still refuse; proved native scalar-true checks are discharged by 567 | 466/472 refusal, 475 regression, 477 repair and 478/480/498 reviews; historical project gate 589; current combined receipts above | `test_retained_device_assert_refuses.py`, `test_constant_assertions.py`; see [assertion contract](DEVICE_ASSERTIONS.md) |
| dispatch post-submit boundary (completed library call then a later raise) | fail-open → swallowed, fallback re-ran work | `PostSubmitError`, fail-loud, no double-apply | packet 457 | integrated 475; reviewed in 458 |
| MLX direct extraction (`extract_msl_for_mlx`) of a shader that still names original thread parameters | admitted an invalid shader with undefined `pid3`/`_lid3` | refuses before lazy shader construction, naming the parameters | packets 505, 506 | `test_mlx_backend.py` (505) |
| source-replay kernels over the 32 KiB threadgroup budget (e.g. 64×64×64 epilogue = 49,152 B) | lowered, then a plain pipeline-creation `RuntimeError` at load | typed `MetalResourceError` → `OutOfResources` at lowering; 32,768 B admitted | packets 510, 513, 514, 517 | `test_epilogue_capability.py` (513) |
| Blocked 1-D store with a nonuniform/unparsed constant mask or offset | mixed bool mask could emit as uniformly true; offset may emit invalid MSL | precise store-local refusal; true/false splats compute the original predicate | 527, legalIR emission counterexample and boundary pins | `test_arg_store_offsets.py` |
| Blocked 1-D store depending on an unproved loaded/effect-derived tensor address | wrong row could be used without remapping | refuse when no blocked/uniform leaf proof exists; do not re-execute effects | 527 | `test_arg_store_offsets.py` |
| Generic constant lowering, including direct 1-D stores | mixed dense boolean mask compiled as `if (1)`; unknown scalar could default to zero | decoded numeric scalar/splat required; no truthiness or zero fallback for unparsed constants | 531; direct native-IR public compile witness, CPU boundary controls | `test_constant_admission.py` |

(The broader silent-wrong→refuse lineage — uint64 max/min, reduce-combine classifier, dot/reduce
stride families, bf16 FA dtype gate, 3-D reduce pre-op — is committed and catalogued as its own
regression suites; it is summarised here, not re-enumerated.)

### 2c. Remaining limits outside the recovered envelopes

| family | representative reason | documented workaround |
|---|---|---|
| small-query decode beyond the source contract above | unsupported probability narrowing, dimensions or value graph | no blanket workaround claimed |
| bias-without-LSE outside the generic and 515/523 wide replay proofs | tested tiles/dtypes listed above; arbitrary combinations remain unqualified | changing tile size is not universal semantic equivalence |
| runtime scale/bias on a `tt.dot` result | cannot prove finite constant multiply on the score path | a source that pre-scales Q and uses bias as the dot accumulator may be supported; moving operations is not universally numerically equivalent |
| two reductions of one loaded tile | second would reduce over the first's accumulator | load the tile separately per reduction |
| narrow-dtype backward delta (`rowsum(O·dO)` in fp16) | order-dependent | widening the reduction is a source/accuracy change, not a semantics-preserving compiler workaround |
| rank-≥3 `tt.trans` non-identity permutation; `nd` `cat`/`join`; `join`→`dot`; top-level `cf.br` | no proven safe lowering | see `refusal_catalog.py` messages |
| C++ / MLX boundary | C++ route defect; MLX extraction gap | leave `TRITON_MSL_USE_CPP` off (default) |
| **explicit bf16 dot operands** (`p.to(bf16)` or bf16 V/K fed straight into `tl.dot`) | biased replay admits f32 or explicit f16 probabilities only; bf16 dot arithmetic is unproven | f32 probability computation is a different source precision contract; use it only if appropriate for the caller's workload |
| **biased attention outside the proved generic and full-row replay envelopes**, including D128 and explicit bf16 dot arithmetic | 32×64 and 64×32 fp16/D64 cases are now computed by 515/523; arbitrary combinations are untested or separately refused, not certified by those rows | retain the original source's precision/mask semantics; no universal spelling substitution |
| **score scaling after QK** in the biased replay — constant (`score * c` after `tl.dot(q, kᵀ, bias)`) or runtime | value graph outside the replay proof; a constant post-dot scale is not folded into Q because that changes rounding | pre-scaling Q and placing bias in the dot accumulator describes another supported source, not an equivalent automatic rewrite |
| **epilogue operations beyond add/mul/extend/truncate with ≤ 1 column bias** (activation, clamp, fma, exp, row bias) and non-tile-boundary output masks | outside `_loop_epilogue.eligible`'s closed op set / structural-mask proof | no general equivalent rewrite claimed; removing an activation or changing a mask changes the program |
| **source-replay tiles over 32 KiB of threadgroup memory** (64×64×64 epilogue) | typed capacity refusal after scratch reuse | 64×64×32, 64×32×64, 32×64×64 sit exactly at the limit and compute |
| **single-key launches (nk = 1) of the biased attention spelling** | Triton specialises the length-1 argument; the folded graph leaves the replay admission and refuses loudly | key counts 32 and 65 were verified; no claim for every other key count; nk = 1 is a loud refusal, not a wrong answer |

### 2d. Historical refusal site inventory (not re-counted for rc4)

| family | sites | representative reason | working alternative |
|---:|---:|---|---|
| dot / matmul | ~54 | fused epilogue/bias on a `tt.dot` result; oversized cooperative staging; batched dot on host path | scale inputs pre-dot; tiles within the 1024-thread budget |
| reduce / scan | ~53 | two reductions of one tile; unproved axis; in-loop reduction wider than the tile | load the tile separately per reduction |
| attention fwd/bwd | ~36 | runtime scale/bias on score path; backward head_dim over budget; unresolved biased-FA bwd orientation | pre-scale Q + bias accumulator; smaller head_dim/block |
| atomic | ~25 | oversized single-element RMW; unproved scope/ordering; 1-D CAS shape | fewer elements; supported sem/scope pairs |
| pointer / marshalling | ~22 | value role unprovable from pointer provenance; negative strides from offset views (host) | distinct provable pointer roles |
| layout / store | ~15 | conflicting 1-D layouts; store extent disagreement; reshape provenance | consistent declared layouts |
| contract / integrity | ~19 | cache/toolchain identity changed mid-process | fresh process; stable toolchain |
| gather / histogram / cat / split | — | output exceeds the 1024-thread cap; ragged/ unsupported axis | smaller tiles; supported axes |
| C++ / MLX boundary | ~6 | C++ route defect; MLX extraction gap | leave `TRITON_MSL_USE_CPP` off |

**A refusal is designed behaviour, not a crash.** Over-refusal is a real defect class here — report
it. A **wrong number without a refusal** is a P0 and the most valuable thing to send us.

### 2e. 0.4.0 transitions

| what | 0.3.0 | 0.4.0 | code | pinned in |
|---|---|---|---|---|
| Xcode 27 toolchain identity (`LC_DYLD_ENVIRONMENT`, load command 0x27) | every kernel refuses: `cannot establish Metal compiler/SDK identity: unsupported native load command 0x27` | accepted only in a main executable and only for `DYLD_VERSIONED_FRAMEWORK_PATH`; any other embedded variable refuses | `_native_dependencies.py` | `test_native_dependency_contract.py`, `test_toolchain_native_dependencies.py` |
| Metal language-standard probe (issue #12) | a library load alone selected a standard; kernels then failed where pipeline creation fails | pipeline creation required; the probe tries the next older standard | `device_detect.py` | `test_device_probe_contract.py` |
| more than 31 arguments on the host-copy route | refuse | computes with a compiler-certified packed layout; unproved layouts refuse | `driver.py`, `_argument_abi.py` | `test_issue4_trifast.py::test_over_31_buffer_args_refuses_clearly` (historical name) |
| host launch after a failed dispatch with uncertain completion | owners dropped; later launches proceed | owners retained; later host launches raise `PostSubmitError` until completion is proven | `_host_dispatch.py` | `test_host_dispatch_lifetime.py` |
| source-program envelope: seven pointer then four scalar arguments, `tt.dot` with `exp2`, `log` or `maxnumf` | not recorded here | proved program: emitted with a pending descriptor that the Python launcher refuses before dispatch; unproved: compile-time refusal with no fallback; `triton-msl-export` exports proved programs | `codegen/_source_program/entry.py`, `driver.py` | `test_source_program_admission.py`, `test_source_program_export.py`, `test_source_program_integration.py` |
| reporter triangle-attention backward kernels with derived contiguous strides and boolean masks | not recorded here | lower on the default path | `generic_lowerer.py` | `test_trifast_backward_strides.py` |
| full-storage, zero-offset, dense rank-two fp16/fp32 transposes | not admitted to the direct route | direct MPS route; offset, partial, overlapping and gappy transposes keep the host route | `_generic_launch.py` | `test_dense_transpose_admission.py` |
| opt-in C++ route masked `tt.load` | unconditional load then select: inactive lanes could read out of bounds | load control-dependent on the mask | `csrc/lib/Conversion/ElementwiseOpToLLVM.cpp` | `test_cpp_masked_load_lowering.py` |

---

## Part 3 — Exceptional-value tests and supporting controls → test-ID map (B)

This replaces the prior unquantified "exceptional-value contracts" claim with the concrete axes the
suite pins, each mapped to its regression test(s). Rows distinguish exceptional-value behavior from finite capability controls and checker validation;
not all are non-finite GPU coverage. "test ID" = `file::test`. Collection establishes existence,
not execution. Counts for all tests in a file do not equal coverage of the specifically named rows.

| # | axis (exceptional-value behaviour contracted) | test ID(s) |
|--:|---|---|
| 1 | forward NaN-row locality — a NaN query row stays row-local (dense / MLA / varlen) | `test_fa_nonfinite_data.py::test_dense_nan_q_row_is_row_local`, `::test_mla_nan_q_row_is_row_local`, `::test_varlen_nan_q_row_is_row_local` |
| 2 | forward non-finite output stays rowwise (no cross-row contamination) | `test_fa_nonfinite_output.py::test_nonfinite_value_outputs_remain_rowwise` |
| 3 | empty-K sequence preserves the zero-denominator semantics | `test_fa_nonfinite_data.py::test_varlen_empty_k_sequence_preserves_zero_denominator_semantics` |
| 4 | matmul epilogue propagates NaN (does not clamp/mask it away) | `test_audit_2026_06_21_silent_wrongs.py::test_matmul_epilogue_propagates_nan` |
| 5 | NaN-propagating `max` computes and propagates (not nan-skipping) | `test_triton_lens_silent_wrongs.py::test_nan_propagating_max_computes_and_propagates` |
| 6 | FINITE control: `float >= max` not over-refused (random-normal input; no non-finite coverage) | `test_triton_lens_silent_wrongs.py::test_float_ge_max_not_over_refused` |
| 7 | backward NaN-lse row locality + single shared sentinel | `test_fa_bwd_nan_rows.py::test_bwd_kv_nan_rows_match_the_source`, `::test_bwd_q_nan_rows_stay_row_local`, `::test_bwd_b_nan_lse_row_matches_the_source`, `::test_bwd_sentinel_must_be_one_shared_value` |
| 8 | backward tail never reads masked-out padding | `test_fa_bwd_nan_rows.py::test_bwd_tail_never_reads_masked_out_padding` |
| 9 | bf16/fp16 backward NaN-scale propagation + guard | `test_fa_bwd_rounding_replay.py::test_software_bf16_rounding_nan_is_guarded`, `::test_nan_scale_propagates_on_the_bf16_kv_route` |
| 10 | biased-FA literal / loaded `-inf` sentinel is exact or refuses | `test_fa_biased_value_paths.py::test_literal_neg_inf_with_loaded_mask_is_exact`, `::test_loaded_neg_inf_bias_is_exact_or_refuses_before_dispatch`, `::test_runtime_nonfinite_sentinel_is_exact_or_refuses_before_dispatch` |
| 11 | non-canonical NaN scalar stays NaN through the scale chain | `test_fa_scale_chain.py::test_gpu_noncanonical_nan_scalar_stays_nan` |
| 12 | biased bf16 NaN scale covers all rows (tiled dispatch) | `test_fa_tiled_dispatch.py::test_biased_bf16_nan_scale_covers_all_rows` |
| 13 | layernorm non-finite input → poisoned row is **all-NaN** (NaN/+Inf/-Inf), siblings bitwise-exact | `test_welford_nonfinite_inputs.py::test_layernorm_nonfinite_input_makes_row_all_nan_and_siblings_exact[nan|inf|-inf]` |
| 14 | layernorm clean path matches a float64 oracle | `test_welford_nonfinite_inputs.py::test_layernorm_clean_matches_float64_oracle` |
| 15 | Welford m2 under non-finite input is in {NaN, +Inf}, **never negative / -Inf**; mean non-finite | `test_welford_nonfinite_inputs.py::test_welford_nonfinite_input_mean_nonfinite_and_m2_never_negative[nan|inf|-inf]` |
| 16 | Welford clean path matches a float64 reference | `test_welford_nonfinite_inputs.py::test_welford_clean_matches_float64_reference` |
| 17 | checker self-validation rejects wrong classifications (all-+Inf row; m2 = -Inf) | `test_welford_nonfinite_inputs.py::test_checkers_reject_wrong_classifications` |
| 18 | signed-zero preservation in sum identities (threadgroup + public/legacy makers; survives padding) | `test_reduce_zero_identity.py::test_threadgroup_sum_identity_preserves_both_zero_signs`, `::test_sum_zero_sign_survives_compiler_padding`, `::test_public_sum_makers_preserve_zero_and_empty_identities`, `::test_legacy_direct_sum_preserves_zero_and_empty_identities` |
| 19 | KDA prefill preserves the exceptional recurrence | `test_kda.py::test_kda_prefill_preserves_exceptional_recurrence` |
| 20 | INTEGER edge-value control: integer-exact reduce above 2^24 (no fp32 rounding) | `test_audit_2026_06_21_silent_wrongs.py::test_i32_reduce_exact_above_2_24` |
| 21 | INTEGER edge-value control: i64 not truncated to i32 across gather / join / cat / scf.if / 2-D reduce | `test_triton_lens_silent_wrongs.py::test_gather_i64_not_truncated_to_i32`, `::test_join_i64_not_truncated`, `::test_cat_i64_not_truncated`, `::test_scf_if_i64_accumulator_not_truncated`, `::test_2d_int64_reduce_not_truncated_by_convert_layout` |
| 22 | INTEGER edge-value control: unsigned uint32 / uint64 max/min computed unsigned (not signed) | `test_triton_lens_silent_wrongs.py::test_uint32_max_min_unsigned_not_signed`, `::test_uint64_max_min_unsigned_not_signed` |
| 23 | signed-zero CAS comparison control; INTEGER tensor-CAS lane control (not NaN/Inf coverage) | `test_atomic_cas_strong.py::test_float_cas_compares_bits`, `::test_tensor_cas_swaps_each_lane_independently` |

**Admissibility vs. exact-oracle.** Row 13 asserts exact all-NaN membership plus bitwise finite
siblings; rows 14 and 16 compare clean results with float64 references within tolerance. Row 15
asserts an order-robust admissible bound (mean non-finite; m2 NaN or +Inf, never negative/-Inf), not
an exact recurrence oracle. Row 17 validates those checkers rather than running another kernel.
These different strengths must not be collapsed into either universal exactness or universal bounds.

---

*Maintained by the correct-or-refuse campaign. The per-packet evidence for every row lives under
`~/Documents/triton-validation-evidence/`. Combined project/upstream and installed-artifact gates passed on exact871 as recorded above. The numerical validation campaigns used the named rc4 artifacts; final-version 0.3.0 artifacts have subsequently been freshly built and rebound to the tested runtime payloads, with reproducibility and fresh/relocated CPU checks.
Artifact identities, rather than the version string alone, bind this evidence.*
