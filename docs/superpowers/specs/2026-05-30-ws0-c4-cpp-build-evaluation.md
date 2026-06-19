# WS0/C4 — C++ build hardening: evaluation of the Triton-link dependency

> The C++ MLIR→LLVM path links against Triton internals. Before "hardening"
> the build we have to decide *what the build should depend on*. This note is
> the evaluation that decision needs. Status: **recommendation pending user
> sign-off** (the user flagged this as needing a real evaluation, not a snap
> pick).

## The constraint (verified empirically, 2026-05-30)

The C++ extension (`triton_msl/csrc/`) needs Triton's TritonGPU / Triton IR
dialect symbols — non-inline op methods (`LocalAllocOp`, `LoadOp`, …), TypeID
symbols, and layout helpers (`LinearLayout`, `LayoutUtils`, `inferDstEncoding`,
`supportMMA`). The current `CMakeLists.txt` gets them by linking **object
files from Triton's build tree** (`…/TritonGPUIR.dir/*.o`, etc.), *not* from
`libtriton.so`.

Why it can't just link `libtriton.so`, measured on the installed
`libtriton.so` (267 MB, Apr 25 build):

```
nm -gU libtriton.so | grep -i tritongpu   →    101   (exported TTG symbols)
nm -gU libtriton.so | wc -l               → 113,545  (total exported)
```

Only ~101 TritonGPU symbols are **exported**; the dialect's full method/TypeID
symbol set is present in the binary but as **local (hidden) symbols**, so it
is unreachable by linking against the `.so`. The `.so` is linked too (line
156) — but only to share the *registered* MLIR dialects in-process (avoiding
duplicate-registration crashes); it does not surface the symbols the
extension's compile units reference. Hence the build-tree `.o` files.

This is the root constraint every option below is evaluated against.

## What this means for an *out-of-tree* backend

In-tree backends (`triton/third_party/{amd,intel,nvidia}`) don't hit this:
they're built *inside* Triton's own CMake, so the dialect objects are right
there. `triton-msl` is **out-of-tree** (a separate pip-installable repo), so
it has to reach into Triton's build products from outside — the unusual
position that creates the coupling.

## Options

### Option A — Require a from-source Triton build; make the paths robust

Keep linking the build-tree `.o` files, but fix the *fragility* (which is
purely in the hardcoded paths, not the approach):

- `TRITON_SRCPATH` / `TRITON_BUILDPATH` honor `$ENV{TRITON_ROOT}` /
  `$ENV{TRITON_BUILD}` (today: only `-D`-overridable, defaults hardcode
  `$HOME/Documents/triton`).
- GLOB the build dir's `cmake.macosx-*-arm64-cpython-*` suffix instead of
  hardcoding `cmake.macosx-15.0-arm64-cpython-3.14` (today it breaks on any
  other macOS/Python version).
- Pin the Triton commit the C++ symbols are ABI-compatible with
  (`TRITON_PINNED_COMMIT`), alongside the LLVM hash it already tracks
  (`87717bf9f81f7b29466c5d9a30a3453bdfc93941`).
- Clear `FATAL_ERROR`s with remediation text when the build tree is absent.

**Pros:** cheapest; matches reality (anyone touching the C++ path has a
from-source Triton build); no binary blobs in the repo; honest about the
dependency.
**Cons:** the C++ path can't be built from a Triton *wheel* — a from-source
Triton build is a hard prerequisite. (The Python MSL path has no such
requirement; it's unaffected.)
**Feasibility:** trivial. Build tree confirmed present locally.

### Option B — Vendor the Triton object files into the repo

Commit (or ship as a release artifact) the specific `.o` files + generated
TableGen headers the extension links.

**Pros:** C++ path builds without a local Triton source build.
**Cons:** the `.o` files are **triple-specific** — baked against the LLVM hash
(`87717bf9…`), the macOS deployment target (15.0), and the Python ABI
(cpython-3.14). They must be regenerated **per platform and on every Triton/LLVM
bump**. Size is large: the `TritonGPUIR.dir` alone is **31 MB**, and we link
four more object dirs (TritonIR, TritonNvidiaGPU, Tools, Analysis) — easily
50–100 MB of arch-specific binary blobs that rot on every bump. Committing that
to git is a poor tradeoff; a release-artifact pipeline is real infra.
**Feasibility:** possible, but high maintenance and repo bloat.

### Option C — Restructure to depend only on `libtriton.so`'s public surface

Stop linking build-tree objects; get everything from the `.so`.

**Pros:** cleanest, most portable, most upstreamable — the "right" end state.
**Cons:** **infeasible today without an upstream change.** The symbols aren't
exported (the measurement above). Reaching this state requires one of:
  - **Upstream Triton exports the TTG dialect symbols** (a visibility / export-
    list change in Triton's build). This is the *correct* long-term fix — it
    helps every out-of-tree backend and aligns with the `TRITON_EXT_ENABLED`
    plugin direction [REFERENCES.md [3]]. But it's a PR against upstream we
    don't control the timeline of.
  - **Rebuild the dialects ourselves** → two copies of the TritonGPU dialect
    registered when `libtriton.so` is also loaded in-process → duplicate-
    registration crash (the exact failure the current `.so` link avoids).
    Not viable.
**Feasibility:** blocked on upstream (option C-upstream) or non-viable (rebuild).

## Recommendation

**A now; C-upstream as the tracked long-term fix.**

- Implement Option A (robust paths + pinned commit + clear errors). It's
  honest, cheap, zero-maintenance-overhead, and unblocks every realistic
  contributor (they have a from-source Triton build). Document the from-source
  requirement prominently for the C++ path; the Python MSL path stays
  wheel-installable and is unaffected.
- File/track an **upstream Triton issue or PR to export the TTG dialect
  symbols** from `libtriton.so` on macOS. If/when it lands, Option C becomes
  feasible and we delete the object-file linking entirely — the genuinely
  best end product, and the most upstreamable. This fits the project's stated
  triton-ext alignment.
- **Reject Option B**: 50–100 MB of triple-specific blobs that rot every bump
  is worse than an honest from-source requirement.

CI (`cpp-build` job) and the Python-3.13 lane are **deferred** in WS0 (optional
automation; the project runs on manual local sweeps today) and will be revisited
once the C6 harness exists and a CI substrate is chosen.

## Open question for sign-off

Does "the best end product" warrant **opening the upstream symbol-export PR
now** (so Option C can land sooner), or do we ship Option A and file the
upstream issue as tracked-but-not-yet-actioned? The first is more work and
puts us on upstream's review timeline; the second keeps momentum here.
