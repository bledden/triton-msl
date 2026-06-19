# In-loop reduction coverage — silent-wrong fix (B + C + A) — design (2026-06-13)

> A `tt.reduce` (e.g. `tl.sum`) **inside a runtime `scf.for` loop** silently sums
> only the first `num_threads` elements of its block when `block_size >
> num_threads`, on any kernel that is **ineligible** for the register-array
> (MEPT) single-pass regime. This is a silent-correctness bug that violates the
> loud-refusal contract. Fix in three layers: **B** refuse the uncovered case,
> **C** array-wire `arith.cmpf` so the common `where`-on-reduce shape routes to
> the already-correct eligible path, **A** give the ineligible in-loop reduce
> real multipass coverage so it computes correctly. Python/MSL backend.

## Problem (root-caused from emitted MSL)

Repro (`tests/test_reduceresult_select_DIAG.py`): an in-loop `tl.sum` whose
**scalar result** feeds `tl.where(s < best, s, best)`. At BLOCK 256/512 the
result is **wrong**, on both the default flag and `TRITON_MSL_MEPT=0`; BLOCK
64/128 are correct.

Mechanism, confirmed by dumping `compiled.asm["msl"]`:

- The register-array (MEPT) single-pass regime is entered only if **every** op is
  in `_MEPT_SAFE_OPS` (`generic_lowerer.py:65`). `arith.cmpf` is *deliberately
  excluded* (only `arith.cmpi` is wired). So `s < best` (a float compare) makes
  the kernel ineligible → it drops to the "scalar wrap-loop path."
- That path's charter comment claims it is **"always correct."** It is not: with
  128 threads and a 256-element block, the in-loop reduce emits
  `simd_sum(X[r_7 + lid])` — **one element per thread** — covering only 128 of
  256 elements. (The post-loop *store* correctly got a `for (_loop_e = lid;
  _loop_e < 256; _loop_e += 128)` multipass wrap; the in-loop reduce did not.)
- The **eligible** path is correct because it pre-folds a register array
  (`val[2]` per thread → `fold = val[0]+val[1]`) via `_mept_reduce_fold` before
  the cross-thread reduce. Top-level reduces are correct because they route
  through `_lower_multipass_reduction` (a `_loop_e` strided accumulation). The
  **in-loop ineligible reduce alone** has no coverage mechanism.

Why the broader severity: tridec's *specific* relay kernel happens to hit a
refusal backstop on a different op first (loud refuse — never wrong output), so
they are safe-but-blocked. But the **class** — any ineligible kernel with an
in-loop reduce and `block_size > num_threads` — is a silent landmine the minimal
repro lands on directly.

**Scope under `MEPT=0` (verified):** the register-array path is fully disabled,
so under `MEPT=0` **every** in-loop reduce with `block_size > num_threads` is
silent-wrong — even a plain `acc += tl.sum(v)` with no `cmpf` (confirmed: 256/512
wrong, 128 correct). So the true condition is "in-loop reduce, block > threads,
**not array-covered**" (`mept_arr is None`), which the B detection already
captures. Consequences: (1) C (array-wiring) only helps the **default** flag —
under `MEPT=0` C is inert, so B/A carry the `MEPT=0` correctness. (2) **A is the
only path to correctness under `MEPT=0`** (B alone refuses every such kernel
there). (3) B's full-corpus verification run **is** A's residual measurement —
one pass, both flag directions, listing every kernel B refuses. Because the
`MEPT=0` ratchet currently passes, no currently-passing test_core kernel can be
an uncovered in-loop-reduce-over-threads case (an exact test would already fail),
so B is expected not to regress the ratchet; the verification run confirms this
empirically and is mandatory before B is declared done.

## Constraints

- **Loud-refusal contract:** no reachable path may be silently wrong. After this
  work, every in-loop-reduce-over-threads case must be either correct or a loud
  `MetalNonRecoverableError`.
- **Ratchet:** flag-default upstream `test_core` must hold/rise (no regression),
  0 failed, both default and `TRITON_MSL_MEPT=0`; project suite 0 failed.
- **Scalar-path parity:** `MEPT=0` codegen for kernels not exercising this path
  must be byte-identical.
- Python/MSL only. No C++.

## Stage B — Refuse the uncovered in-loop reduce (mandatory, first)

**Goal:** convert the silent-wrong into a loud refusal, immediately, so the tree
is contract-safe at every later commit.

**Where:** `_lower_reduce` (`_lowerer_reduce.py:308`), the 1-D full-reduce path
(the branch at line 436+, and the i64 branch at 431).

**Detection:** refuse when **all** hold:
1. not array-eligible for this reduce: `mept_arr is None`
   (`self.env_array.get(ssa.operand_ids[0])` is None);
2. the reduce input is a raw block tensor that has **not** been folded/covered by
   a preceding mechanism — i.e. it is being lowered inside an `scf.for` body
   (an in-loop reduce), as opposed to a top-level reduce that already routed
   through `_lower_multipass_reduction` (which rebinds the input to a
   `_local_acc_*` scalar);
3. the 1-D input tile exceeds the threadgroup: `input_shape` is 1-D and
   `input_shape[0] > self.kb.block_size`.

The discriminator for (2) is an **in-`scf.for`-body** signal. If no such signal
exists, add a depth counter incremented around the body loop in
`_lower_scf_for` (`_lowerer_control.py:32`) and read it in `_lower_reduce`.
Distinguishing in-loop from top-level is required so the (correct) top-level
multipass reduce is **not** refused.

**Action:** raise `MetalNonRecoverableError` with a clear message naming the
shape (`in-loop reduction with block_size N > num_threads T on a register-array-
ineligible kernel`). This is the same loud-refusal mechanism used elsewhere.

**Risk control:** B must refuse *only* the genuinely-uncovered case. Verify
against the full corpus that B does not refuse any currently-passing kernel
(top-level reduces, array-eligible in-loop reduces, block ≤ num_threads).

## Stage C — Array-wire `arith.cmpf` (routes the common case to the correct path)

**Goal:** make the `where`-on-reduce shape (and any `cmpf`-bearing kernel that is
otherwise all-safe-ops) **eligible**, so its in-loop reduce uses the correct
register-array fold instead of the ineligible path.

**Changes (mirror the existing `cmpi` wiring):**
1. `generic_lowerer.py` — add `"arith.cmpf"` to `_MEPT_SAFE_OPS` (and update the
   "deliberately EXCLUDED" comment to drop cmpf, keeping the rationale honest).
2. `_lower_cmpf` (`generic_lowerer.py:3769`) — add the MEPT array branch exactly
   as `_lower_cmpi` (3746) does: build a per-position expression closure and call
   `self._mept_binary_dispatch(ssa, a_id, b_id, a, b, _make_expr, "bool", "i1")`;
   on success mark `self.env_is_mask[ssa.id] = True` and return. The closure must
   preserve cmpf's NaN-aware predicate forms (`uno`/`ord`/`une`, the unordered
   `isnan(a)||isnan(b)||(a op b)` variants, and the ordered `a op b` forms) for
   each array position — i.e. the same per-element string the scalar path emits,
   parameterised over the two array reads.
3. The scalar fallback (existing body) stays unchanged for the non-array case.

**Why this is correct, not a shortcut:** the eligible register-array path is the
project's *intended* mechanism for `block_size > num_threads`; it is already
tested and correct (proven by `k_persum`/`k_carry`). C widens that correct path
to cover `cmpf`, rather than patching the broken path. The scalar
`where`-on-reduce-result (a scalar select consuming the scalar reduce result)
is handled by the existing scalar `cmpf`/`select` fallbacks inside the eligible
regime (scalars are never arrays), so it stays correct.

**Interaction with B:** once C lands, the repro kernel is eligible → it no longer
reaches B's refusal → it computes correctly at full width. B remains the backstop
for kernels still ineligible for *other* reasons.

## Stage A — Universal coverage for the ineligible in-loop reduce

**Goal:** an ineligible kernel (some op outside `_MEPT_SAFE_OPS` that C does not
cover — e.g. `arith.bitcast`, `math.erf`, a shape op) that *also* has an in-loop
reduce with `block_size > num_threads` should **compute correctly** rather than
refuse (B).

**Gate A on evidence (first task of Stage A):** after B+C are in and green,
measure the residual surface — enumerate which `test_core` (and project) kernels
now hit B's refusal that previously "passed" (silently or otherwise). Concretely:
temporarily log every B-refusal during a full `test_core` run and list the
distinct kernel shapes.
- **If the residual surface is empty / negligible:** B+C already fully resolve
  the bug (no silent-wrong, common case correct, nothing real refused). Record
  the measurement, document A as a designed-but-unbuilt backstop, and stop —
  building body-local multipass replay for zero reachable cases is speculative
  risk on correctness-critical code (YAGNI). Surface the data to the user.
- **If the surface is non-empty:** build A as below.

**Approach (body-local multipass replay):** mirror `_lower_multipass_reduction`
(`_lowerer_reduce.py:140`) inside the `scf.for` body. When an in-loop 1-D reduce
has `block_size > num_threads` and `mept_arr is None`:
1. declare a per-thread accumulator at the reduce's combine identity (reuse
   `_reduce_identity_combine` / the i64 identity logic already in
   `_lower_multipass_reduction`);
2. emit `for (uint _loop_e = lid; _loop_e < total; _loop_e += block_size) { … }`
   re-emitting the reduce's input-dependency chain (the load and any elementwise
   between load and reduce) with the strided index `_loop_e` in place of `lid`,
   accumulating each into the per-thread accumulator;
3. run the existing cross-thread `threadgroup_reduce` (or the i64 tree) on the
   accumulator.

The input-dependency chain is recovered with the same `_collect_tensor_deps`
replay the top-level multipass uses; the index substitution rides the existing
`_needs_wrapping` machinery that already rewrites `lid → _loop_e` for wrapped
ops (it is what produced the correct multipass *store*). A re-implements nothing
the codebase lacks; it extends the existing wrap to the in-`scf.for` reduce.

**Once A lands:** B becomes a true backstop (fires only on a shape A cannot
cover); the contract holds either way.

## Error handling / integrity

- B is the safety floor: any uncovered in-loop reduce refuses loudly. A only ever
  *upgrades* a refusal to a correct result; it never introduces a silent path.
- C cannot produce wrong results: a too-wide `_MEPT_SAFE_OPS` is the documented
  risk, so C is validated by the full ratchet in both flag directions before
  landing, plus targeted cmpf/where regression tests.
- No change touches the C++ path.

## Testing / ratchet

- **Repro → regression (GPU, serial):** promote the DIAG kernel to a kept test
  asserting correctness at BLOCK 128/256/512/1024 for: (i) `where`-on-reduce
  (closes the silent-wrong via C), (ii) a `cmpf`-bearing in-loop reduce, (iii) a
  plain in-loop `tl.sum` (regression guard). Compare to torch (exact/1e-4).
- **B unit:** a kernel that stays ineligible *and* has an in-loop reduce over
  threads refuses with `MetalNonRecoverableError` (loud), and the same kernel at
  BLOCK ≤ num_threads still passes.
- **C parity:** `MEPT=0` codegen byte-identical on a corpus slice; cmpf-heavy
  kernels (`test_where`, masked ops) unaffected.
- **Full ratchet:** upstream `test_core` ≥ baseline, 0 failed, default **and**
  `MEPT=0`; project suite 0 failed. Clear `~/.cache/triton_msl ~/.triton/cache`
  before each verifying run (cache is keyed on the effective MEPT flag).
- **A residual measurement** logged and reported regardless of A's outcome.

## Sequence (never a silent-wrong intermediate)

B (refuse) → C (cmpf eligibility, repro goes correct) → A (measure residual; build
only if reachable). Each stage commits independently; the tree is contract-safe
from the end of B onward.

## Out of scope

- Top-level reduce coverage for `block_size > num_threads` with no per-element
  ops to wrap (separate path; check during A's residual measurement, file
  separately if real).
- Performance of the wrap-loop path (correctness only here).
- C++ lowerer (Python-primary).
