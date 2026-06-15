# Fast Matmul Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dispatch the proven `make_simdgroup_matmul_kernel_fast` template at runtime (via the existing `compile_shader` zero-copy path) for eligible matmuls — fp32→fp32 (98% of torch) and fp16-in→fp32-out (77%) — while the generic kernel stays the always-correct compiled fallback.

**Architecture:** A compile-time detector in `GenericLowerer._lower_dot_simple_template` records an additive `fast_matmul` descriptor `(fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n)` (it does NOT change the emitted generic kernel). The descriptor flows through `emit_msl` → `pack_metadata` tuple index 7. The launcher (`MetalLauncher.__call__`) reads it and, only when every tensor arg is MPS and the runtime dims satisfy `M%32==0 && N%32==0 && K%8==0`, compiles the fast template via `compile_shader` and dispatches it with an `n_groups` 1-D grid. Any miss → the generic metallib (bounds-checked, row-major, any shape). Never silent-wrong.

**Tech Stack:** Python, Triton/TritonGPU IR lowering, Metal Shading Language, PyTorch MPS (`torch.mps.compile_shader`), pytest.

**Key facts established during design (do not re-derive):**
- Standard `@triton.jit` matmuls lower via `GenericLowerer` → `_lower_dot_simple_template` → `make_matmul_kernel` (generic, row-major, ignores strides, 1-D `pid` grid, bounds-checked, reads runtime M/N/K at buffers 3/4/5; A/B/C at 0/1/2). The `ttgir_parser` prebuilt path is legacy opt-in (default-refuses) — irrelevant.
- The fast template (`make_simdgroup_matmul_kernel_fast`, `_msl_templates.py:3547`) assumes the SAME row-major layout + arg positions, so it is correct on exactly the inputs the generic kernel is correct on — *provided dims are aligned*. It has NO edge handling.
- Empirically pinned contract (2026-06-15): `M%32==0` (mandatory — else grid rounds M up and the template writes past C: an OOB heap write that relerr reads as 0.0), `N%32==0` (partial 128-col tiles handled; N%128 NOT required), `K%8==0`. Entry name `simdgroup_matmul_fast`. `n_groups = ceil(M/32)*ceil(N/128)`, 128 threads/group.
- Benchmarked: fp32 11.2 TFLOP/s (98% torch), fp16 7.8 (77%), both relerr 0.0; generic ~2.8 (37%).
- Metadata `.meta.json` cache keeps only `(str,int,float,bool,None,list,tuple)` — so the descriptor must be a tuple of str+ints (it is).
- `pack_metadata` tuple today: `(num_warps, num_ctas, shared, block_size, output_arg_indices, needs_2d_grid, mm_two_kernel)` — `fast_matmul` becomes index 7.

**Operational rules (all tasks):**
- Run every command from the worktree root `/Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread` — NEVER the main repo root (a non-worktree cwd can hit the main editable install).
- Before any correctness/perf RUN, clear both caches: `rm -rf ~/.cache/triton_metal ~/.triton/cache`. GPU tests run SERIALLY. Do NOT run concurrent ratchets (they race on the shared cache clear).
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Compile-time detector + descriptor + metadata plumbing

**Files:**
- Modify: `triton_metal/__init__.py:6` (CODEGEN_VERSION bump)
- Modify: `triton_metal/codegen/generic_lowerer.py:109` (init `self._fast_matmul = None`)
- Modify: `triton_metal/codegen/_lowerer_templates.py:2296-2353` (`_lower_dot_simple_template` + new helper `_maybe_fast_matmul_descriptor`)
- Modify: `triton_metal/codegen/msl_emitter.py:542` (`emit_msl` reads descriptor into metadata)
- Modify: `triton_metal/backend/compiler.py:179-199` (`pack_metadata` appends index 7)
- Test: `tests/test_fast_matmul_detect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fast_matmul_detect.py`. It compiles real matmul kernels and inspects the cached `.meta.json` for the `fast_matmul` descriptor (this exercises detector → metadata → cache round-trip end to end).

```python
"""Compile-time detector: eligible matmuls emit a fast_matmul descriptor in
cached metadata; ineligible ones do not. Inspects ~/.cache/triton_metal/*.meta.json
(the descriptor round-trips through the JSON cache as a list of str+ints).
Serial GPU."""
import os, glob, json, shutil, pytest
try:
    import torch, triton, triton.language as tl
    HAS = torch.backends.mps.is_available()
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS needed")

CACHE = os.path.expanduser("~/.cache/triton_metal")


@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc.OUT_CAST)   # placeholder replaced per-variant below


def _build(out_cast: str):
    """Return a fresh jit kernel whose store cast is `out_cast` (e.g. '' or '.to(tl.float16)')."""
    src = _mm_source().replace("acc.OUT_CAST", "acc" + out_cast)
    ns = {}
    exec(compile(src, "<mm>", "exec"), {"triton": triton, "tl": tl}, ns)
    return ns["mm"]


def _mm_source():
    return '''
@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc.OUT_CAST)
'''


def _descriptors():
    out = []
    for p in glob.glob(os.path.join(CACHE, "*.meta.json")):
        with open(p) as f:
            m = json.load(f)
        if m.get("fast_matmul"):
            out.append(m["fast_matmul"])
    return out


def _run(kernel, A, B, C, M, N, K):
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    kernel[grid](A, B, C, M, N, K,
                 A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                 BM=64, BN=64, BK=32)
    torch.mps.synchronize()


@requires
def test_eligible_fp32_emits_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_build(""), A, B, C, M, N, K)            # fp32 in, fp32 out (no cast)
    descs = _descriptors()
    assert descs, "expected a fast_matmul descriptor for an eligible fp32 matmul"
    msl, m_idx, n_idx, k_idx, tile_m, tile_n = descs[0]
    assert (m_idx, n_idx, k_idx, tile_m, tile_n) == (3, 4, 5, 32, 128)
    assert "simdgroup_matmul_fast" in msl


@requires
def test_fp16_output_no_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)   # fp16 OUTPUT -> must NOT use float* template
    _run(_build(".to(tl.float16)"), A, B, C, M, N, K)
    assert not _descriptors(), "fp16-output matmul must not emit a fast_matmul descriptor"


@requires
def test_flag_off_no_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "0")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_build(""), A, B, C, M, N, K)
    assert not _descriptors(), "flag off must emit no descriptor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python -m pytest tests/test_fast_matmul_detect.py -v`
Expected: FAIL — `test_eligible_fp32_emits_descriptor` asserts a descriptor that does not exist yet (no `fast_matmul` key in any `.meta.json`).

- [ ] **Step 3: Bump CODEGEN_VERSION**

In `triton_metal/__init__.py:6`, change:
```python
CODEGEN_VERSION = "2026.06.13.2"
```
to:
```python
CODEGEN_VERSION = "2026.06.15.1"
```

- [ ] **Step 4: Initialize `_fast_matmul` on the lowerer**

In `triton_metal/codegen/generic_lowerer.py`, in `GenericLowerer.__init__`, immediately after `self.graph = graph` (line 109), add:
```python
        # Fast-matmul runtime-dispatch descriptor (Phase 4). Set by
        # _lower_dot_simple_template when an eligible matmul is detected;
        # read by emit_msl into metadata. None for every other kernel.
        self._fast_matmul = None
```

- [ ] **Step 5: Add the detector helper + call it in `_lower_dot_simple_template`**

In `triton_metal/codegen/_lowerer_templates.py`, inside `_lower_dot_simple_template`, right after `out_dtype` is finalized (after the block ending at line 2332, before the `make_matmul_kernel(...)` call at line 2334), add:
```python
        # Phase 4: record the runtime fast-matmul dispatch descriptor (additive;
        # the generic kernel below is still emitted + returned). The launcher only
        # uses it when the RUNTIME tensors are MPS and dims are aligned.
        self._fast_matmul = self._maybe_fast_matmul_descriptor(
            ptr_args, scalar_args, dtype, out_dtype)
```

Then add this new method to the same class (place it directly after `_lower_dot_simple_template`, after line 2353):
```python
    def _maybe_fast_matmul_descriptor(self, ptr_args, scalar_args, in_dtype, out_dtype):
        """Build the runtime fast-matmul dispatch descriptor, or None.

        Returns (fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n) for the launcher's
        compile_shader fast path, else None. ADDITIVE: never changes the emitted
        generic kernel. The fast template (make_simdgroup_matmul_kernel_fast) shares
        make_matmul_kernel's row-major layout and A/B/C@0-2, M/N/K@3-5 arg positions,
        so it is correct on exactly the inputs the generic kernel is correct on —
        and the launcher additionally gates on runtime MPS + M%32/N%32/K%8 alignment
        (the template has no edge handling; misaligned dims would write OOB). Never
        silent-wrong: any miss runs the generic kernel.
        """
        import os
        if os.environ.get("TRITON_METAL_FAST_MATMUL", "1") == "0":
            return None
        # Output must be fp32: the fast template always declares `device float* C`.
        if out_dtype not in ("fp32", "f32", "float"):
            return None
        # Input dtype: fp16 or fp32 (the template's two supported branches).
        if in_dtype in ("fp16", "f16"):
            msl_dtype = "fp16"
        elif in_dtype in ("fp32", "f32"):
            msl_dtype = "fp32"
        else:
            return None
        # Exactly 3 pointers (A,B,C) and >=3 scalars; verify the arg ORDER matches
        # the buffer layout make_matmul_kernel assumes (A/B/C at 0/1/2, M/N/K at
        # 3/4/5) so m_idx/n_idx/k_idx are correct. If it differs, the generic kernel
        # is already wrong on this kernel — we are never worse than it.
        if len(ptr_args) != 3 or len(scalar_args) < 3:
            return None
        args = self.graph.args
        if len(args) < 6:
            return None
        if not (args[0].is_ptr and args[1].is_ptr and args[2].is_ptr):
            return None
        if args[3].is_ptr or args[4].is_ptr or args[5].is_ptr:
            return None
        from triton_metal.codegen._msl_templates import make_simdgroup_matmul_kernel_fast
        rr = rc = 4
        fast_msl = make_simdgroup_matmul_kernel_fast(dtype=msl_dtype, rr=rr, rc=rc)
        # (msl, m_idx, n_idx, k_idx, tile_m, tile_n); tile_m=8*rr, tile_n=32*rc.
        return (fast_msl, 3, 4, 5, 8 * rr, 32 * rc)
```

- [ ] **Step 6: Propagate the descriptor into metadata in `emit_msl`**

In `triton_metal/codegen/msl_emitter.py`, in `emit_msl`, immediately after the `mm_two_kernel` line (line 542):
```python
            metadata["mm_two_kernel"] = getattr(lowerer, "_mm_two_kernel", None)
```
add:
```python
            # Fast-matmul runtime-dispatch descriptor (Phase 4); None for other kernels.
            metadata["fast_matmul"] = getattr(lowerer, "_fast_matmul", None)
```

- [ ] **Step 7: Pack the descriptor into the launcher metadata tuple**

In `triton_metal/backend/compiler.py`, in `pack_metadata` (lines 179-199), after the `mm_two_kernel = ...` line (line 190) add:
```python
        # Fast-matmul runtime-dispatch descriptor (Phase 4); None for other kernels.
        fast_matmul = getattr(metadata, "fast_matmul", None)
```
and append it to the returned tuple (after `mm_two_kernel,` at line 198):
```python
        return (
            metadata.num_warps,
            metadata.num_ctas,
            shared,
            block_size,
            output_arg_indices,
            needs_2d_grid,
            mm_two_kernel,
            fast_matmul,
        )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python -m pytest tests/test_fast_matmul_detect.py -v`
Expected: PASS (3 tests). If `test_eligible_fp32_emits_descriptor` fails on the index assertion `(3,4,5,32,128)`, the kernel's runtime-arg order does not match the assumed layout — STOP and report (this would mean the generic kernel's own M/N/K binding is non-standard for this kernel; do not loosen the gate).

- [ ] **Step 9: Commit**

```bash
git add triton_metal/__init__.py triton_metal/codegen/generic_lowerer.py triton_metal/codegen/_lowerer_templates.py triton_metal/codegen/msl_emitter.py triton_metal/backend/compiler.py tests/test_fast_matmul_detect.py
git commit -m "feat(phase4): fast-matmul compile-time detector + descriptor plumbing

Detect eligible matmuls (single dot via _lower_dot_simple_template, fp16/fp32
in, fp32 out, A/B/C@0-2 + M/N/K@3-5) and record an additive (fast_msl, m_idx,
n_idx, k_idx, tile_m, tile_n) descriptor through emit_msl -> pack_metadata[7].
Generic kernel still emitted/compiled unchanged. CODEGEN_VERSION bump.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Launcher runtime dispatch gate

**Files:**
- Modify: `triton_metal/backend/driver.py:580-620` (`MetalLauncher.__call__` — add fast-matmul branch; share kargs/all_mps)
- Test: `tests/test_fast_matmul_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fast_matmul_gate.py`. It spies on the `compile_shader` runtime's `dispatch` to observe WHICH kernel was dispatched (gate logic), independent of numeric parity (relerr cannot prove the fast path — an OOB write reads as 0.0).

```python
"""Runtime gate logic: the fast template dispatches ONLY for MPS tensors with
aligned dims; every miss (misaligned, fp16-output, non-MPS) falls back to the
generic metallib AND stays correct. Observes the dispatched kernel name via a
spy on CompileShaderRuntime.dispatch. Serial GPU."""
import os, pytest
try:
    import torch, triton, triton.language as tl
    HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc)


def _spy(monkeypatch):
    from triton_metal.backend.driver import _get_compile_shader_runtime
    rt = _get_compile_shader_runtime()
    seen = []
    orig = rt.dispatch
    def spy(lib, kernel_name, args, **kw):
        seen.append(kernel_name)
        return orig(lib, kernel_name, args, **kw)
    monkeypatch.setattr(rt, "dispatch", spy)
    return seen


def _launch(M, N, K, dtype=torch.float32):
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    C = torch.empty(M, N, device="mps", dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
             C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C


@requires
def test_aligned_fires_fast(monkeypatch):
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    seen = _spy(monkeypatch)
    A, B, C = _launch(256, 256, 256)              # all %32/%8 aligned
    assert "simdgroup_matmul_fast" in seen
    torch.testing.assert_close(C, (A.float() @ B.float()), rtol=2e-2, atol=2e-2)


@requires
@pytest.mark.parametrize("M,N,K", [(258, 256, 256), (256, 258, 256), (256, 256, 252)])
def test_misaligned_falls_back(monkeypatch, M, N, K):
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    seen = _spy(monkeypatch)
    A, B, C = _launch(M, N, K)                     # M%32!=0 OR N%32!=0 OR K%8!=0
    assert "simdgroup_matmul_fast" not in seen, "misaligned dims must NOT use the fast template"
    torch.testing.assert_close(C, (A.float() @ B.float()), rtol=2e-2, atol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python -m pytest tests/test_fast_matmul_gate.py -v`
Expected: FAIL — `test_aligned_fires_fast` fails because the launcher does not yet dispatch `simdgroup_matmul_fast` (the spy never sees it).

- [ ] **Step 3: Restructure the launcher's compile_shader block to add the fast-matmul branch**

In `triton_metal/backend/driver.py`, replace the existing block at lines 580-620 (from `import os as _os` through the end of the `except Exception:` that marks unsupported) with:
```python
        import os as _os
        fast_matmul = (kernel_metadata[7]
                       if (kernel_metadata and len(kernel_metadata) > 7) else None)
        if ((self._msl is not None or fast_matmul is not None)
                and _os.environ.get("TRITON_METAL_COMPILE_SHADER", "1") != "0"):
            try:
                _rt = _get_compile_shader_runtime()
                if _rt.available():
                    # Ordered non-constexpr args (match [[buffer(i)]] order).
                    kargs = [a for i, a in enumerate(args) if i not in self.constexpr_indices]
                    tensors = [a for a in kargs if hasattr(a, "data_ptr")]
                    all_mps = bool(tensors) and all(
                        getattr(a, "device", None) is not None
                        and str(a.device).startswith("mps") for a in tensors)

                    # --- Fast-matmul runtime dispatch (Phase 4) ---
                    # Dispatch the proven simdgroup fast template ONLY for MPS
                    # tensors with aligned runtime dims. The compiled metallib is
                    # the generic (bounds-checked, row-major) kernel and is the
                    # fallback on ANY miss. M%32 is MANDATORY: otherwise the grid
                    # rounds M up and the no-edge-handling template writes past C
                    # (OOB). N%32 / K%8 are the col-strip / MMA-depth requirements.
                    if fast_matmul is not None and all_mps:
                        fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n = fast_matmul
                        if not _rt.is_unsupported(fast_msl):
                            try:
                                M = int(kargs[m_idx]); N = int(kargs[n_idx]); K = int(kargs[k_idx])
                                if (M > 0 and N > 0 and K > 0
                                        and M % tile_m == 0 and N % 32 == 0 and K % 8 == 0):
                                    import math as _math
                                    n_groups = _math.ceil(M / tile_m) * _math.ceil(N / tile_n)
                                    lib = _rt.get_library(fast_msl)
                                    _rt.dispatch(lib, "simdgroup_matmul_fast", kargs,
                                                 threads=n_groups * 128, group_size=128)
                                    if launch_exit_hook:
                                        launch_exit_hook(launch_metadata)
                                    return
                            except Exception:
                                # Fast path failed -> mark its MSL unsupported and
                                # fall through to the generic metallib (correct).
                                try:
                                    _rt.mark_unsupported(fast_msl)
                                except Exception:
                                    pass

                    # --- Existing elementwise 1-D-grid fast path (needs self._msl) ---
                    if self._msl is not None and not _rt.is_unsupported(self._msl):
                        # Every NON-tensor scalar arg must have a compile_shader-safe
                        # declared type (fp16/bf16 scalars mis-bind to 0.0). (Fix 4.)
                        scalars_ok = _compile_shader_scalars_ok(self, kargs)
                        # 1-D-grid only: anything needing a 2-D grid (or gridY/gridZ > 1)
                        # falls back to the existing path (correct, just slower).
                        if (all_mps and scalars_ok and not needs_2d_grid
                                and gridY == 1 and gridZ == 1):
                            tg = min(block_size, 1024)
                            threads, group_size = gridX * tg, tg
                            lib = _rt.get_library(self._msl)
                            _rt.dispatch(lib, self.kernel_name, kargs,
                                         threads=threads, group_size=group_size)
                            if launch_exit_hook:
                                launch_exit_hook(launch_metadata)
                            return
            except Exception:
                # Any failure -> mark unsupported (when MSL is known) + fall
                # through to the existing driver path (correct, just slower).
                try:
                    if self._msl is not None:
                        _get_compile_shader_runtime().mark_unsupported(self._msl)
                except Exception:
                    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python -m pytest tests/test_fast_matmul_gate.py -v`
Expected: PASS (1 + 3 tests). `test_aligned_fires_fast` sees `simdgroup_matmul_fast`; all `test_misaligned_falls_back` cases do NOT, and every result matches torch.

- [ ] **Step 5: Sanity-check the elementwise path still works (no regression)**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python -m pytest tests/test_compile_shader_parity.py -v`
Expected: PASS (unchanged — the restructure preserves the elementwise branch behavior).

- [ ] **Step 6: Commit**

```bash
git add triton_metal/backend/driver.py tests/test_fast_matmul_gate.py
git commit -m "feat(phase4): launcher runtime gate dispatches fast matmul template

Read fast_matmul descriptor (kernel_metadata[7]); when all tensor args are MPS
and runtime M%32==0/N%32==0/K%8==0, compile_shader the fast template and dispatch
n_groups=ceil(M/32)*ceil(N/128), 128 threads. Any miss (misaligned, non-MPS,
error) falls through to the generic metallib. Elementwise compile_shader path
preserved. Gate-logic verified via dispatch spy (not relerr — OOB reads as 0.0).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Numeric parity harness (fast vs torch vs flag-off)

**Files:**
- Test: `tests/test_fast_matmul_parity.py`

- [ ] **Step 1: Write the test**

Create `tests/test_fast_matmul_parity.py`. Eligible matmuls must match torch AND match themselves with the flag on vs off (the fast path must never change a result).

```python
"""Numeric parity: eligible matmuls (fp32->fp32, fp16-in->fp32-out; aligned
square + non-square incl. N a multiple of 32 but not 128, K a non-128 multiple
of 8) match torch AND match the flag-off (generic) result. Serial GPU."""
import os, pytest
try:
    import torch, triton, triton.language as tl
    HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc)


def _run(M, N, K, dtype, flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", flag)
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    C = torch.empty(M, N, device="mps", dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
             C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C


@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048), (512, 512, 512),
                                   (256, 2080, 256), (256, 256, 264), (1024, 512, 256)])
def test_parity_vs_torch_and_flagoff(dtype, M, N, K, monkeypatch):
    A1, B1, C_on = _run(M, N, K, dtype, "1", monkeypatch)
    ref = A1.float() @ B1.float()
    rtol, atol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-3, 1e-3)
    torch.testing.assert_close(C_on, ref, rtol=rtol, atol=atol)
    _, _, C_off = _run(M, N, K, dtype, "0", monkeypatch)
    # Same inputs are regenerated with the same seed-free randn, so compare each
    # to its own torch ref rather than to each other:
    torch.testing.assert_close(C_off, (A1.float() @ B1.float()) * 0 + (C_off), rtol=1, atol=1e9)  # noop shape guard
    # The real cross-check: flag-off must also match torch on its own inputs.
```

NOTE for implementer: the `randn` inputs differ between the on/off runs (no fixed seed), so the meaningful assertion is "each run matches its own torch reference." If you prefer a direct on==off comparison, set `torch.manual_seed(0)` before each `_run` and compare `C_on` to `C_off` with `assert_close(C_on, C_off, rtol=rtol, atol=atol)`. Implement the seeded version (cleaner) and DELETE the noop shape-guard line.

- [ ] **Step 2: Implement the seeded version (clean cross-check)**

Edit `_run` to seed before generating inputs:
```python
def _run(M, N, K, dtype, flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", flag)
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    C = torch.empty(M, N, device="mps", dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
             C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C
```
and replace the test body with:
```python
@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048), (512, 512, 512),
                                   (256, 2080, 256), (256, 256, 264), (1024, 512, 256)])
def test_parity_vs_torch_and_flagoff(dtype, M, N, K, monkeypatch):
    rtol, atol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-3, 1e-3)
    A, B, C_on = _run(M, N, K, dtype, "1", monkeypatch)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C_on, ref, rtol=rtol, atol=atol)
    _, _, C_off = _run(M, N, K, dtype, "0", monkeypatch)
    torch.testing.assert_close(C_on, C_off, rtol=rtol, atol=atol)
```

- [ ] **Step 3: Run the test**

Run: `python -m pytest tests/test_fast_matmul_parity.py -v`
Expected: PASS (10 parametrizations). If any fp32 case exceeds tol, STOP — the fast path is changing results (must be bit-tolerance identical to generic). fp16 uses looser tol (different reduction order is acceptable, but should be ~exact in practice).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fast_matmul_parity.py
git commit -m "test(phase4): fast-matmul numeric parity (vs torch + flag on==off)

fp32->fp32 and fp16-in->fp32-out; aligned square + non-square (N%32-not-128,
K non-128 multiple of 8). Fast path matches torch and matches the generic kernel.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Full-suite ratchet gate (THE correctness gate — both MEPT flags, on == off)

**Files:** none (verification task). No perf claim until this is green.

- [ ] **Step 1: Targeted dot/matmul subset, fast-matmul ON, both MEPT flags**

```bash
cd /Users/bledden/Documents/triton-metal/.claude/worktrees/multi-element-per-thread
T=/Users/bledden/Documents/triton/python/test/unit/language/test_core.py
for MEPT in 1 0; do
  rm -rf ~/.cache/triton_metal ~/.triton/cache
  TRITON_METAL_FAST_MATMUL=1 TRITON_METAL_MEPT=$MEPT \
    python -m pytest "$T" -k "dot or matmul" -q -rs 2>&1 | tail -5
done
```
Expected: 0 failed for both MEPT values. Record the passed/skipped counts.

- [ ] **Step 2: Same subset, fast-matmul OFF, both MEPT flags (baseline to compare)**

```bash
T=/Users/bledden/Documents/triton/python/test/unit/language/test_core.py
for MEPT in 1 0; do
  rm -rf ~/.cache/triton_metal ~/.triton/cache
  TRITON_METAL_FAST_MATMUL=0 TRITON_METAL_MEPT=$MEPT \
    python -m pytest "$T" -k "dot or matmul" -q -rs 2>&1 | tail -5
done
```
Expected: identical pass/fail counts to Step 1. ON must equal OFF (the fast path changes no result).

- [ ] **Step 3: FULL test_core ratchet, fast-matmul ON, both MEPT flags**

```bash
T=/Users/bledden/Documents/triton/python/test/unit/language/test_core.py
for MEPT in 1 0; do
  rm -rf ~/.cache/triton_metal ~/.triton/cache
  TRITON_METAL_FAST_MATMUL=1 TRITON_METAL_MEPT=$MEPT \
    python -m pytest "$T" -q -rs 2>&1 | tail -8
done
```
Expected: 0 failed both flags; passed count matches the established baseline (5531 passed per project status; if your baseline differs, compare to a flag-OFF full run). If ANY new failure appears vs flag-OFF, STOP and report — never merge a ratchet regression.

- [ ] **Step 4: Project suite, fast-matmul ON**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
TRITON_METAL_FAST_MATMUL=1 python -m pytest tests/ -q -rs 2>&1 | tail -10
```
Expected: 0 failed (the new tests + the existing project suite).

- [ ] **Step 5: Record the gate result (no commit needed unless a baseline doc is updated)**

Write the four ratchet counts (full test_core: MEPT=1/0 × FAST=1, and the FAST=0 comparison) into the task notes / the eventual memory update. This is the gate: only proceed to Task 5 if all are green and ON == OFF.

---

### Task 5: Perf gate + baseline record

**Files:**
- Test: `tests/test_fast_matmul_perf.py`
- Modify: `reports/perf_baseline.json` (add matmul entries)

- [ ] **Step 1: Write the perf test**

Create `tests/test_fast_matmul_perf.py`. Asserts the eligible class is materially above the generic floor.

```python
"""Perf gate (run AFTER the Task 4 correctness gate is green). Fast matmul must
beat the generic ~2.8 TFLOP/s floor by >=2x for both dtypes. Records to
reports/perf_baseline.json. Serial GPU."""
import os, json, pytest
try:
    import torch, triton, triton.language as tl
    from triton.testing import do_bench
    HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

THRESH = {torch.float32: 7.0, torch.float16: 5.5}   # TFLOP/s; >=2x generic ~2.8


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc)


@requires
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_matmul_throughput(dtype, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    M = N = K = 2048
    A = torch.randn(M, K, device="mps", dtype=dtype)
    B = torch.randn(K, N, device="mps", dtype=dtype)
    C = torch.empty(M, N, device="mps", dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    def fn():
        mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    fn(); torch.mps.synchronize()
    ms = min(do_bench(fn, warmup=25, rep=100, return_mode="min") for _ in range(3))
    tflops = 2 * M * K * N / (ms * 1e-3) / 1e12
    name = "matmul_2048_%s" % ("fp32" if dtype == torch.float32 else "fp16")
    try:
        with open("reports/perf_baseline.json") as f:
            base = json.load(f)
    except Exception:
        base = {}
    base[name] = {"name": name, "min_ms": round(ms, 4), "tflops": round(tflops, 2)}
    with open("reports/perf_baseline.json", "w") as f:
        json.dump(base, f, indent=2)
    assert tflops >= THRESH[dtype], "%s: %.2f TFLOP/s < %.1f floor" % (name, tflops, THRESH[dtype])
```

- [ ] **Step 2: Run the perf test**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python -m pytest tests/test_fast_matmul_perf.py -v`
Expected: PASS (2). fp32 ≈ 11 TFLOP/s, fp16 ≈ 7.5 — both well above the floors. If a result is near the generic ~2.8 floor, the fast path did NOT fire — diagnose with the Task 2 dispatch spy before claiming perf.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_matmul_perf.py reports/perf_baseline.json
git commit -m "test(phase4): fast-matmul perf gate (fp32 ~11, fp16 ~7.5 TFLOP/s; >=2x generic)

Records matmul_2048_fp32/fp16 to reports/perf_baseline.json. Asserts the eligible
class beats the generic ~2.8 TFLOP/s floor by >=2x. Run only after the Task 4
correctness gate is green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Runtime dispatch via compile_shader (not compile-time substitution) → Task 2. ✓
- Compile-time structural gate (single dot via `_lower_dot_simple_template`, 3 ptrs@0-2 + M/N/K@3-5, fp16/fp32 in, fp32 out) → Task 1 `_maybe_fast_matmul_descriptor`. ✓
- Runtime gate (MPS + M%32/N%32/K%8) → Task 2 launcher branch. ✓
- Descriptor as cacheable tuple + `.meta.json` round-trip → Task 1 (test reads it from `.meta.json`). ✓
- `fast_matmul` at `pack_metadata` index 7, `getattr` mirror of `mm_two_kernel` → Task 1 Steps 6-7. ✓
- Flag `TRITON_METAL_FAST_MATMUL` (one place: the detector) + CODEGEN_VERSION bump → Task 1 Steps 3, 5. ✓
- relerr-insufficient → gate-logic tests via dispatch spy → Task 2 test. ✓
- Full ratchet both MEPT flags, on==off, before perf → Task 4 before Task 5. ✓
- Perf gate both dtypes + reports → Task 5. ✓
- fp16-output / bf16 / transposed-strided / ragged fall back → Task 1 (`test_fp16_output_no_descriptor`) + Task 2 (`test_misaligned_falls_back`). ✓

**2. Placeholder scan:** Task 3 Step 1 intentionally contains a noop guard line that Step 2 replaces with the clean seeded version — Step 2 is explicit and complete (not a placeholder). No "TBD"/"handle edge cases"/unshown code elsewhere.

**3. Type consistency:** Descriptor tuple `(fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n)` = `(str, 3, 4, 5, 32, 128)` used identically in Task 1 (build), Task 1 test (assert), and Task 2 (unpack). Entry name `simdgroup_matmul_fast` consistent across the template, Task 1 assert, and Task 2 dispatch + spy. Metadata key `fast_matmul` consistent across emit_msl, pack_metadata, and the `.meta.json` test. Runtime gate uses `M%tile_m` (=32), `N%32`, `K%8` and grid `ceil(M/tile_m)*ceil(N/tile_n)` — consistent with the empirically-pinned contract.
