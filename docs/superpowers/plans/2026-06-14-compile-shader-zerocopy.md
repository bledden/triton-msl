# Zero-copy MPS execution via torch.mps.compile_shader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route triton-metal's emitted MSL through `torch.mps.compile_shader` when kernel args are torch MPS tensors, dispatching zero-copy (no per-launch host round-trip) — confirmed 10.2× on the memory-bound class — with the existing driver as a safe fallback.

**Architecture:** A small `CompileShaderRuntime` (compile+cache+dispatch) plus a launcher fast-path in `driver.py` that fires when every tensor arg is MPS and the runtime is available; the MSL is threaded from the compiler to the launcher via a name-keyed cache. Behind `TRITON_METAL_COMPILE_SHADER` (default-on once verified, `=0` escape hatch). Any failure falls back to the existing host-round-trip driver.

**Tech Stack:** Python, PyTorch MPS (`torch.mps.compile_shader`), the existing `triton_metal/backend/driver.py` launcher.

**Spec:** `docs/superpowers/specs/2026-06-14-compile-shader-zerocopy-design.md`

**Conventions for every GPU run:** clear caches first — `rm -rf ~/.cache/triton_metal ~/.triton/cache`. GPU tests run from the worktree root, serial. Upstream ratchet: `bash scripts/run_upstream_test.sh unit/language/test_core.py -q` (and `TRITON_METAL_MEPT=0 …`). Baseline to hold: `test_core` **5531/0** both flags; project suite **0 failed**.

---

## File Structure

- `triton_metal/backend/compile_shader_runtime.py` — **new.** `CompileShaderRuntime`: `available()`, `get_library(msl)` (cached on MSL), `dispatch(...)`, `mark_unsupported(msl)`. One clear responsibility: compile + cache + dispatch MSL via `torch.mps.compile_shader`. No driver knowledge.
- `triton_metal/backend/compiler.py` — **modify.** Where `asm["msl"]` is produced, stash the MSL keyed on kernel name into a process-global so the launcher can retrieve it.
- `triton_metal/backend/driver.py` — **modify.** `MetalLauncher.__init__` captures the kernel name; `MetalLauncher.__call__` adds the all-MPS fast-path (eligibility → grid translation → marshal args → dispatch), falling back to the existing body on any miss.
- `tests/test_compile_shader_runtime.py` — **new.** Unit tests for the runtime (availability, cache, dispatch correctness, reduction/shared-mem, multi-dim grid).
- `tests/test_compile_shader_parity.py` — **new.** Parity: representative kernels through BOTH paths give identical-to-tolerance results.

---

## Task 1: CompileShaderRuntime (compile + cache + dispatch)

**Files:**
- Create: `triton_metal/backend/compile_shader_runtime.py`
- Test: `tests/test_compile_shader_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
"""CompileShaderRuntime: compile MSL via torch.mps.compile_shader + dispatch
zero-copy against MPS tensors. Serial GPU."""
import pytest
try:
    import torch
    HAS = hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires_cs = pytest.mark.skipif(not HAS, reason="torch.mps.compile_shader needed")

_VADD = '''#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                 device float* OUT [[buffer(2)]], constant int& N [[buffer(3)]],
                 uint pid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
    uint i = pid*256u + lid; if (i < (uint)N) OUT[i] = A[i] + B[i];
}'''

@requires_cs
def test_available():
    from triton_metal.backend.compile_shader_runtime import CompileShaderRuntime
    assert CompileShaderRuntime().available() is True

@requires_cs
def test_dispatch_vadd_zero_copy():
    import torch
    from triton_metal.backend.compile_shader_runtime import CompileShaderRuntime
    rt = CompileShaderRuntime()
    N = 4096
    A = torch.randn(N, device="mps"); B = torch.randn(N, device="mps"); OUT = torch.empty(N, device="mps")
    lib = rt.get_library(_VADD)
    assert rt.get_library(_VADD) is lib   # cached (same object)
    rt.dispatch(lib, "vadd", [A, B, OUT], [N], threads=N, group_size=256)
    torch.mps.synchronize()
    torch.testing.assert_close(OUT, A + B, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run, verify it fails**

Run: `python3 -m pytest tests/test_compile_shader_runtime.py -q`
Expected: FAIL (`ModuleNotFoundError: triton_metal.backend.compile_shader_runtime`).

- [ ] **Step 3: Implement the runtime**

Create `triton_metal/backend/compile_shader_runtime.py`:

```python
"""Zero-copy MPS execution via torch.mps.compile_shader.

PyTorch (newer versions) can compile a Metal compute library from MSL source
and dispatch its kernels against MPS tensors zero-copy (PyTorch owns the
buffers + the MPS stream). This runtime wraps that: compile (cached on the MSL
string) + dispatch. It has NO triton-metal driver knowledge — the driver
selects when to use it.
"""
from __future__ import annotations


class CompileShaderRuntime:
    """Compile + cache + dispatch MSL via torch.mps.compile_shader."""

    def __init__(self):
        self._lib_cache = {}          # msl_source -> compiled library
        self._unsupported = set()     # msl_source hashes that failed; skip fast-path

    def available(self) -> bool:
        try:
            import torch
            return (hasattr(torch, "mps")
                    and torch.backends.mps.is_available()
                    and hasattr(torch.mps, "compile_shader"))
        except Exception:
            return False

    def is_unsupported(self, msl: str) -> bool:
        return hash(msl) in self._unsupported

    def mark_unsupported(self, msl: str) -> None:
        self._unsupported.add(hash(msl))

    def get_library(self, msl: str):
        """Compile MSL (cached on the source string). Raises on compile error."""
        lib = self._lib_cache.get(msl)
        if lib is None:
            import torch
            lib = torch.mps.compile_shader(msl)
            self._lib_cache[msl] = lib
        return lib

    def dispatch(self, lib, kernel_name: str, tensor_and_scalar_args, scalar_args=None,
                 *, threads, group_size) -> None:
        """Dispatch lib.<kernel_name>(*args, threads=..., group_size=...).

        ``tensor_and_scalar_args`` is the ordered argument list (MPS tensors +
        Python scalars) matching the kernel's [[buffer(i)]] order. ``threads``
        and ``group_size`` are ints (1-D) or tuples (2-D/3-D). PyTorch binds the
        MPS tensors zero-copy and enqueues on the MPS stream.
        """
        if scalar_args is not None:
            args = list(tensor_and_scalar_args) + list(scalar_args)
        else:
            args = list(tensor_and_scalar_args)
        fn = getattr(lib, kernel_name)
        fn(*args, threads=threads, group_size=group_size)
```

(Note: the test calls `dispatch(lib, "vadd", [A,B,OUT], [N], threads=N, group_size=256)` — so `dispatch` accepts the tensor list and an optional scalar list, concatenating in order. Keep this signature.)

- [ ] **Step 4: Run, verify it passes**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_compile_shader_runtime.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Add reduction (shared-mem) + multi-dim grid coverage**

Append two tests asserting (a) a `threadgroup float[] + simd_sum + threadgroup_barrier` blocksum kernel is correct through `dispatch` (group_size=256), and (b) a 2-D grid kernel dispatches with `threads=(4,4), group_size=(4,4)` and `uint2 [[thread_position_in_grid]]`. (Use the exact kernels verified during design — they pass.) Run; both PASS.

- [ ] **Step 6: Commit**

```bash
git add triton_metal/backend/compile_shader_runtime.py tests/test_compile_shader_runtime.py
git commit -m "feat(driver): CompileShaderRuntime — compile+cache+dispatch MSL via torch.mps.compile_shader"
```

---

## Task 2: Thread the MSL source + kernel name to the launcher

**Files:**
- Modify: `triton_metal/backend/compiler.py` (where `asm["msl"]` / `make_msl` produces the MSL)
- Modify: `triton_metal/backend/driver.py` (`MetalLauncher.__init__`)
- Test: extend `tests/test_compile_shader_runtime.py`

The launcher has the kernel name (via `metadata`) and `function` (pipeline) but NOT the MSL. Stash MSL by name at compile time; the launcher retrieves it.

- [ ] **Step 1: Add a process-global MSL registry**

In `triton_metal/backend/compiler.py`, add near the top:

```python
# Kernel-name -> MSL source, populated when MSL is emitted so the driver's
# launcher can retrieve the source for the torch.mps.compile_shader fast-path
# (the launcher only receives the compiled metallib + name, not the source).
_MSL_BY_NAME: dict[str, str] = {}
```

In `make_msl` (the stage that produces the MSL string), after the MSL `src` is built and the kernel name is known, add: `_MSL_BY_NAME[kernel_name] = msl_src`. Find the kernel name the same way the rest of `make_msl`/`make_metallib` derives it (e.g. `metadata.name` or the function name parsed from the MSL `kernel void <name>`). READ `make_msl` to use the exact name variable already in scope.

- [ ] **Step 2: Capture name + MSL in the launcher**

In `driver.py` `MetalLauncher.__init__`, after the existing body, add:

```python
        # Kernel name for the compile_shader fast-path MSL lookup.
        self.kernel_name = getattr(metadata, "name", None)
        from triton_metal.backend.compiler import _MSL_BY_NAME
        self._msl = _MSL_BY_NAME.get(self.kernel_name) if self.kernel_name else None
```

If `metadata` has no `.name` (verify by adding a one-off `print(type(metadata), getattr(metadata,'name',None))` during a manual launch, then remove), fall back to parsing the name from the metallib at `load_binary` and threading it — but first confirm `metadata.name` exists; in current Triton it does.

- [ ] **Step 3: Verify the MSL is retrievable**

Manual check: compile a trivial Triton kernel and confirm `_MSL_BY_NAME` is populated and the launcher's `self._msl` is non-None.

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache
python3 -c "
import torch, triton, triton.language as tl
@triton.jit
def k(X, OUT, N, BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<N
    tl.store(OUT+o, tl.load(X+o,mask=m)*2.0, mask=m)
X=torch.randn(1024,device='mps'); OUT=torch.empty(1024,device='mps')
k[(1,)](X,OUT,1024,BLOCK=1024); torch.mps.synchronize()
from triton_metal.backend.compiler import _MSL_BY_NAME
print('MSL registry keys:', list(_MSL_BY_NAME)[:3], '| non-empty:', bool(_MSL_BY_NAME))
"
```
Expected: non-empty registry with the kernel name.

- [ ] **Step 4: Commit**

```bash
git add triton_metal/backend/compiler.py triton_metal/backend/driver.py
git commit -m "feat(driver): thread MSL source by kernel name to the launcher (compile_shader path)"
```

---

## Task 3: Driver fast-path — eligibility, grid translation, dispatch, fallback

**Files:**
- Modify: `triton_metal/backend/driver.py` (`MetalLauncher.__call__`, top of the body after metadata unpack)
- Test: `tests/test_compile_shader_parity.py`

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_compile_shader_parity.py`:

```python
"""Parity: kernels run identically (to tolerance) with the compile_shader
fast-path ON vs the existing driver. Serial GPU."""
import os, pytest
try:
    import torch, triton, triton.language as tl
    HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")

@triton.jit
def _vadd(A,B,OUT,N,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<N
    tl.store(OUT+o, tl.load(A+o,mask=m)+tl.load(B+o,mask=m), mask=m)

@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_vadd_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    N=4096; A=torch.randn(N,device="mps"); B=torch.randn(N,device="mps"); OUT=torch.empty(N,device="mps")
    _vadd[(triton.cdiv(N,1024),)](A,B,OUT,N,BLOCK=1024); torch.mps.synchronize()
    torch.testing.assert_close(OUT, A+B, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run, verify it passes for flag=0, and (currently) flag=1 just runs the old path**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_compile_shader_parity.py -q`
Expected: PASS both (the flag does nothing yet). This locks correctness before wiring the path.

- [ ] **Step 3: Add the fast-path to `MetalLauncher.__call__`**

In `driver.py`, immediately after the `num_warps`/`block_size`/`needs_2d_grid` unpack (~line 498), insert the fast-path. `self.constexpr_indices`, `self.arg_names` are from `__init__`; `args` are the raw kernel args; `kernel_metadata[4]` is `output_indices`.

```python
        import os as _os
        if (self._msl is not None
                and _os.environ.get("TRITON_METAL_COMPILE_SHADER", "1") != "0"):
            _rt = _get_compile_shader_runtime()
            if _rt.available() and not _rt.is_unsupported(self._msl):
                try:
                    import torch as _torch
                    # Ordered non-constexpr args (match [[buffer(i)]] order).
                    kargs = [a for i, a in enumerate(args) if i not in self.constexpr_indices]
                    tensors = [a for a in kargs if hasattr(a, "data_ptr")]
                    # Eligible only if EVERY tensor arg is an MPS tensor.
                    all_mps = tensors and all(
                        getattr(a, "device", None) is not None
                        and str(a.device).startswith("mps") for a in tensors)
                    if all_mps:
                        tg = num_warps * 32
                        # grid (gridX/Y/Z) = threadgroup counts; threads = grid*tg.
                        if gridY == 1 and gridZ == 1:
                            threads, group_size = gridX * tg, tg
                        else:
                            threads = (gridX * tg, gridY, gridZ)
                            group_size = (tg, 1, 1)
                        lib = _rt.get_library(self._msl)
                        _rt.dispatch(lib, self.kernel_name, kargs,
                                     threads=threads, group_size=group_size)
                        if launch_exit_hook:
                            launch_exit_hook(launch_metadata)
                        return
                except Exception:
                    # Any failure → mark unsupported + fall through to the
                    # existing driver path (correct, just slower). Never wrong.
                    _rt.mark_unsupported(self._msl)
        # ---- existing driver path continues below unchanged ----
```

Add a module-level singleton accessor near `_get_utils`:

```python
_COMPILE_SHADER_RUNTIME = None
def _get_compile_shader_runtime():
    global _COMPILE_SHADER_RUNTIME
    if _COMPILE_SHADER_RUNTIME is None:
        from triton_metal.backend.compile_shader_runtime import CompileShaderRuntime
        _COMPILE_SHADER_RUNTIME = CompileShaderRuntime()
    return _COMPILE_SHADER_RUNTIME
```

IMPORTANT: `kargs` passes tensors AND scalars together in arg order (compile_shader binds tensors zero-copy + scalars as `constant&`). The grid translation must match how the MSL computes indices: the kernel uses `pid [[threadgroup_position_in_grid]]` (0..gridX-1) and `lid [[thread_position_in_threadgroup]]` (0..tg-1), so `threads = gridX*tg`, `group_size = tg`. For multi-dim grids, confirm against `needs_2d_grid` and how the MSL reads program_id(1)/(2); if a multi-dim kernel mis-maps, `mark_unsupported` catches it (wrong result would be caught by the parity gate — see Task 4).

- [ ] **Step 4: Run the parity test — both flags must pass**

Run: `rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/test_compile_shader_parity.py -q`
Expected: PASS both flags (flag=1 now exercises the fast-path, flag=0 the old path; both correct).

- [ ] **Step 5: Commit**

```bash
git add triton_metal/backend/driver.py tests/test_compile_shader_parity.py
git commit -m "feat(driver): compile_shader zero-copy fast-path for all-MPS-tensor launches (flag-gated, fallback-safe)"
```

---

## Task 4: Correctness gate — full suite both flags + real kernels (THE GATE)

**Files:** extend `tests/test_compile_shader_parity.py`; possibly `scripts/conftest_metal.py`

- [ ] **Step 1: Broaden the parity harness**

Add parametrized parity tests (flag 1 vs 0, assert identical to tolerance) over: a reduction (`tl.sum`), softmax, a `tl.where`/masked kernel, an atomic_add scatter, and a 2-D kernel. Each runs the same kernel under both flags and `assert_close`. Run; all PASS — if any flag=1 result differs, that kernel mis-maps through compile_shader: fix the translation or ensure it falls back (mark_unsupported) and is covered by the existing path.

- [ ] **Step 2: Full upstream ratchet, fast-path ON, both MEPT flags**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -8
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_MEPT=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -8
```
Expected: **5531 passed, 0 failed** BOTH (the fast-path must not change any result; anything it can't handle falls back). If any test regresses (wrong value, not a flaky FileNotFoundError), the fast-path produced a wrong result for that kernel — STOP, identify the kernel shape, and make it fall back (tighten eligibility) before proceeding. Correctness is the gate.

- [ ] **Step 3: Ratchet with the fast-path OFF (regression-free escape hatch)**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_COMPILE_SHADER=0 bash scripts/run_upstream_test.sh unit/language/test_core.py -q 2>&1 | tail -6
```
Expected: 5531/0 (the existing path, unchanged).

- [ ] **Step 4: Project suite + real kernels, both flags**

```bash
rm -rf ~/.cache/triton_metal ~/.triton/cache && python3 -m pytest tests/ -q 2>&1 | tail -6
rm -rf ~/.cache/triton_metal ~/.triton/cache && TRITON_METAL_COMPILE_SHADER=0 python3 -m pytest tests/ -q 2>&1 | tail -6
```
Expected: 0 failed both. Includes FlashAttention (11/11) and the relay regression tests. Additionally verify the real relay@256+num_warps=8 (the tridec kernel, via the PYTHONPATH harness) is still bit-identical to @128 with the fast-path ON.

- [ ] **Step 5: Commit the correctness record**

```bash
git add -A
git commit -m "test(driver): compile_shader fast-path full-suite parity green both flags + real kernels"
```

---

## Task 5: Perf gate (after correctness)

**Files:** `benchmarks/`, `reports/perf_baseline.json`

- [ ] **Step 1: Re-bench the memory-bound class via the real Triton path**

Bench vector_add@16M, an elementwise kernel, softmax, and a reduction with the fast-path ON, measuring GB/s (`do_bench`, GB moved / time). Compare to the fast-path OFF (the 28 GB/s floor).

- [ ] **Step 2: Assert the win**

Add a perf test (or extend `tests/test_roofline.py`) asserting vector_add@16M with the fast-path ON achieves **≥ 250 GB/s** (vs ~28 OFF). Record the new numbers (ON and OFF) in `reports/perf_baseline.json`.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/ reports/perf_baseline.json tests/
git commit -m "perf(driver): compile_shader zero-copy — vector_add 28->~280 GB/s (10x); baseline recorded"
```

---

## Task 6: Finalize

- [ ] **Step 1:** Confirm the flag default-on is correct (it is `!= "0"`). Document `TRITON_METAL_COMPILE_SHADER` (and the fallback) in the driver module docstring + `docs/`.
- [ ] **Step 2:** Final full ratchet both MEPT flags + the COMPILE_SHADER flag both ways (the matrix), all green; project suite green.
- [ ] **Step 3:** Update memory ([[project_status]], a new perf note) + commit.

---

## Self-Review notes (addressed)

- **Spec coverage:** runtime (T1), MSL threading (T2), driver fast-path + grid translation + arg marshaling + flag + fallback (T3), full-suite parity gate both flags + real kernels (T4), perf gate ≥250 GB/s (T5), finalize/flag/docs (T6). All spec sections mapped.
- **Correctness-first:** the perf gate (T5) is strictly after the correctness gate (T4); the fast-path is purely additive with a fallback; eligibility requires ALL tensor args MPS; any failure marks-unsupported + falls back.
- **Open items from the spec:** grid translation (T3 step 3, incl. multi-dim + needs_2d_grid), scalar coverage (exercised by the full ratchet T4 — every scalar type in test_core flows through), MSL-source threading (T2), lib cache (T1 get_library), availability detection (T1 available()).
- **Name/type consistency:** `CompileShaderRuntime.available/get_library/dispatch/is_unsupported/mark_unsupported`, `_MSL_BY_NAME`, `self._msl`/`self.kernel_name`, `_get_compile_shader_runtime`, `TRITON_METAL_COMPILE_SHADER` used consistently.
