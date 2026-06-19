# sizePerThread Fix — Proper Thread-to-Element Mapping

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When Triton 3.6.0 generates TTGIR with `sizePerThread > 1`, dispatch the correct number of threads (`num_warps * warp_size`) and have each thread process multiple elements via a loop, instead of over-launching 1 thread per element.

**Architecture:** The fix has three parts: (1) extract `sizePerThread` from the TTGIR layout encoding in the walker, (2) use `num_warps * warp_size` as the thread count when `sizePerThread > 1`, emitting a per-thread element loop in the lowerer, (3) size reduction shared memory arrays by actual SIMD group count (not `block_size / 32`). This replaces the current "always launch `block_size` threads" model with "launch `num_threads` threads, each handling `sizePerThread` elements."

**Tech Stack:** Python, Metal Shading Language, Triton MLIR

---

### Task 1: Extract sizePerThread from TTGIR layout encoding

**Files:**
- Modify: `triton_msl/codegen/mlir_walker.py:60-67` (IRGraph dataclass)
- Modify: `triton_msl/codegen/mlir_walker.py:480-492` (TTGIRWalker.__init__)
- Modify: `triton_msl/codegen/mlir_walker.py:575-585` (walker.walk return)
- Test: `tests/test_mlir_walker.py`

The TTGIR module text contains layout attributes like:
```
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

We need to parse this and expose it on `IRGraph`.

- [ ] **Step 1: Write failing test**

Add to `tests/test_mlir_walker.py`:

```python
def test_walker_extracts_size_per_thread():
    """Walker should extract sizePerThread from TTGIR blocked layout."""
    # This test uses the cached TTGIR from bench_regression softmax
    # which has sizePerThread = [4]
    import os
    cache_dir = os.path.expanduser("~/.triton/cache")
    ttgir_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".ttgir"):
                ttgir_files.append(os.path.join(root, f))
    
    # Find one with sizePerThread > 1
    found = False
    for path in ttgir_files:
        with open(path) as f:
            text = f.read()
        if "sizePerThread = [4]" in text:
            found = True
            break
    
    if not found:
        pytest.skip("No cached TTGIR with sizePerThread > 1")
    
    # Parse the layout from text directly
    from triton_msl.codegen.mlir_walker import _parse_blocked_layout
    layout = _parse_blocked_layout(text)
    assert layout is not None
    assert layout["size_per_thread"] == [4]
    assert layout["threads_per_warp"] == [32]
    assert layout["warps_per_cta"] == [4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_mlir_walker.py::test_walker_extracts_size_per_thread -v`
Expected: FAIL — `_parse_blocked_layout` doesn't exist

- [ ] **Step 3: Implement `_parse_blocked_layout` and add `size_per_thread` to IRGraph**

In `triton_msl/codegen/mlir_walker.py`, add the parsing function near the top (after imports):

```python
import re

def _parse_blocked_layout(mod_text: str) -> dict | None:
    """Extract sizePerThread, threadsPerWarp, warpsPerCTA from TTGIR blocked layout.
    
    Parses: #blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    Returns dict with int lists, or None if not found.
    """
    m = re.search(
        r'#ttg\.blocked<\{sizePerThread\s*=\s*\[([^\]]+)\],\s*'
        r'threadsPerWarp\s*=\s*\[([^\]]+)\],\s*'
        r'warpsPerCTA\s*=\s*\[([^\]]+)\]',
        mod_text,
    )
    if not m:
        return None
    return {
        "size_per_thread": [int(x.strip()) for x in m.group(1).split(",")],
        "threads_per_warp": [int(x.strip()) for x in m.group(2).split(",")],
        "warps_per_cta": [int(x.strip()) for x in m.group(3).split(",")],
    }
```

Add to the `IRGraph` dataclass:

```python
@dataclass
class IRGraph:
    ...
    size_per_thread: list[int] | None = None  # From TTGIR blocked layout
```

In `TTGIRWalker.__init__`, parse it:

```python
self._layout = _parse_blocked_layout(self._mod_text)
```

In `TTGIRWalker.walk()`, attach it to the graph:

```python
graph.size_per_thread = self._layout["size_per_thread"] if self._layout else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_mlir_walker.py::test_walker_extracts_size_per_thread -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add triton_msl/codegen/mlir_walker.py tests/test_mlir_walker.py
git commit -m "feat: extract sizePerThread from TTGIR blocked layout"
```

---

### Task 2: Use num_warps * warp_size for thread count and emit per-thread loop

**Files:**
- Modify: `triton_msl/codegen/generic_lowerer.py:279-314` (block_size / wrapping logic)
- Modify: `triton_msl/codegen/generic_lowerer.py:1739-1748` (_lower_make_range 1D case)
- Test: `tests/test_generic_lowerer.py`

When `sizePerThread > 1`, we need to:
- Set `effective_block_size = num_warps * warp_size` (e.g., 128 instead of 1024)
- Emit a per-thread loop: `for (uint _e = lid; _e < BLOCK_SIZE; _e += NUM_THREADS)`
- Map `tt.make_range` to the loop variable `_e` instead of `lid`

- [ ] **Step 1: Write failing test**

Create a standalone test that compiles a softmax-like kernel with sizePerThread=4 and checks the MSL output has a per-thread loop and correct thread count.

Add to `tests/test_generic_lowerer.py`:

```python
@pytest.mark.skipif(not _has_triton(), reason="Triton not installed")
def test_size_per_thread_emits_element_loop():
    """When sizePerThread > 1, codegen should emit a per-thread element loop
    and use num_warps * warp_size threads instead of block_size threads."""
    import triton
    import triton.language as tl
    
    @triton.jit
    def _softmax(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x = x - x_max
        x_exp = tl.exp(x)
        x_sum = tl.sum(x_exp, axis=0)
        out = x_exp / x_sum
        tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)
    
    import torch
    x = torch.randn(4, 1024, device="cpu")
    out = torch.empty_like(x)
    _softmax[(4,)](x, out, 1024, BLOCK_SIZE=1024)
    
    expected = torch.softmax(x, dim=-1)
    diff = (out - expected).abs().max().item()
    assert diff < 1e-5, f"Softmax incorrect: max_diff={diff}"
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `source .venv/bin/activate && python -m pytest tests/test_generic_lowerer.py::test_size_per_thread_emits_element_loop -v`
Expected: PASS (current code works but with 1024 threads — we'll check perf separately)

This test verifies correctness. We'll add a structural assertion after the implementation.

- [ ] **Step 3: Implement sizePerThread-aware thread dispatch**

In `triton_msl/codegen/generic_lowerer.py`, modify the block_size computation section (around line 279-314):

```python
        # Determine actual thread count vs total elements.
        # When TTGIR specifies sizePerThread > 1, each thread handles multiple
        # elements. Use num_warps * warp_size as the thread count, not block_size.
        size_per_thread = 1
        if self.graph.size_per_thread:
            size_per_thread = 1
            for s in self.graph.size_per_thread:
                size_per_thread *= s
        
        num_threads = self.graph.num_warps * 32  # warp_size = 32
        
        if size_per_thread > 1 and block_size > num_threads:
            # Triton expects num_threads threads, each handling size_per_thread elements.
            # Use the wrapping loop to iterate over all elements.
            self._needs_wrapping = True
            self._total_elements = block_size
            block_size = num_threads
        elif block_size > 1024:
            self._needs_wrapping = True
            self._total_elements = block_size
            block_size = 1024
        
        self.effective_block_size = block_size
```

And in `_lower_make_range` (1D pure case, ~line 1739), when wrapping is active, use the loop variable:

```python
        # Pure 1D case
        lid = self._lid_expr
        if self._needs_wrapping:
            # When wrapping, use the loop variable instead of lid directly
            lid = "_loop_e"
```

- [ ] **Step 4: Run test suite to verify correctness**

Run: `source .venv/bin/activate && python -m pytest tests/test_generic_lowerer.py tests/test_integration.py tests/test_gpu_correctness.py -v --timeout=120`
Expected: All previously-passing tests still pass

- [ ] **Step 5: Commit**

```bash
git add triton_msl/codegen/generic_lowerer.py tests/test_generic_lowerer.py
git commit -m "feat: use num_warps * warp_size thread count when sizePerThread > 1

Triton 3.6.0 generates TTGIR with sizePerThread=4 for softmax kernels,
expecting 128 threads each handling 8 elements. Previously we launched
1024 threads (one per element), causing 32-way cross-warp reductions
instead of 4-way. Now we dispatch num_warps * warp_size threads and
use a per-thread element loop."
```

---

### Task 3: Fix reduction shared memory sizing

**Files:**
- Modify: `triton_msl/codegen/generic_lowerer.py:3593-3596` (n_simd_groups calculation)
- Test: existing softmax correctness + new structural test

The reduction code at line 3595 computes `n_simd_groups = (self.kb.block_size + 31) // 32`. After Task 2, `self.kb.block_size` will be 128 (not 1024), so `n_simd_groups` will correctly be 4 instead of 32. **This should fix automatically** — but we need to verify.

- [ ] **Step 1: Write test that checks shared memory array size**

Add to `tests/test_generic_lowerer.py`:

```python
@pytest.mark.skipif(not _has_triton(), reason="Triton not installed")
def test_softmax_shared_memory_sized_by_warps():
    """Reduction shared memory should be sized by actual warp count, not block_size/32."""
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget
    
    @triton.jit
    def _softmax(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x = x - x_max
        x_exp = tl.exp(x)
        x_sum = tl.sum(x_exp, axis=0)
        out = x_exp / x_sum
        tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)
    
    target = GPUTarget("metal", "apple-m4", 32)
    src = ASTSource(fn=_softmax, signature="*fp32,*fp32,i32", constexprs={"BLOCK_SIZE": 1024})
    compiled = triton_compile(src, target=target)
    msl = compiled.asm.get("msl", "")
    
    # Should have shared arrays sized by num_warps (4), not block_size/32 (32)
    # Look for shared_N[4] not shared_N[32]
    import re
    shared_arrays = re.findall(r'threadgroup float (\w+)\[(\d+)\]', msl)
    for name, size in shared_arrays:
        size = int(size)
        assert size <= 8, (
            f"Shared array {name}[{size}] is too large — should be sized by "
            f"num_warps (4), not block_size/32 (32). Got {size}."
        )
```

- [ ] **Step 2: Run test**

Run: `source .venv/bin/activate && python -m pytest tests/test_generic_lowerer.py::test_softmax_shared_memory_sized_by_warps -v`
Expected: PASS if Task 2's `block_size = num_threads` propagated correctly

- [ ] **Step 3: If test fails, fix n_simd_groups calculation**

If `self.kb.block_size` isn't being set correctly, explicitly compute:

```python
n_simd_groups = (self.kb.block_size + 31) // 32
```

This should already work since Task 2 sets `block_size = num_threads = num_warps * 32`, making `n_simd_groups = num_warps`. Verify and adjust if needed.

- [ ] **Step 4: Run full test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ --ignore=tests/test_torch_compile.py --ignore=tests/test_mps_tensor.py -v --timeout=120`
Expected: All tests pass (torch_compile and mps_tensor excluded — known issues)

- [ ] **Step 5: Commit**

```bash
git add triton_msl/codegen/generic_lowerer.py tests/test_generic_lowerer.py
git commit -m "test: verify reduction shared memory sized by actual warp count"
```

---

### Task 4: Update driver to use metadata block_size correctly

**Files:**
- Modify: `triton_msl/backend/compiler.py:88-99` (pack_metadata)
- Test: benchmark regression test

The compiler's `pack_metadata` already reads `metadata.block_size` which comes from the lowerer's `effective_block_size`. After Task 2, `effective_block_size` will be `num_warps * warp_size` when `sizePerThread > 1`. The driver at line 576 does `threads_per_tg = min(block_size, 1024)`, which will now correctly be 128.

- [ ] **Step 1: Verify the pipeline works end-to-end with benchmark**

Run: `source .venv/bin/activate && python benchmarks/bench_regression.py --json 2>&1 | grep -A5 softmax`
Expected: Softmax throughput should improve (fewer SIMD groups in reduction)

- [ ] **Step 2: Run correctness check**

```bash
source .venv/bin/activate && python -c "
import torch, triton, triton.language as tl
from benchmarks.bench_regression import _softmax_kernel

# Test various sizes
for cols in [128, 256, 512, 1024]:
    x = torch.randn(8, cols, device='cpu')
    out = torch.empty_like(x)
    _softmax_kernel[(8,)](x, out, cols, BLOCK_SIZE=1024)
    expected = torch.softmax(x, dim=-1)
    diff = (out - expected).abs().max().item()
    status = 'PASS' if diff < 1e-5 else f'FAIL (diff={diff})'
    print(f'  cols={cols}: {status}')
"
```

- [ ] **Step 3: Update baseline if performance improved**

Run: `source .venv/bin/activate && python benchmarks/bench_regression.py --update-baseline`

- [ ] **Step 4: Commit**

```bash
git add reports/perf_baseline.json
git commit -m "perf: update baseline after sizePerThread fix

Softmax 8Kx1K: reduction now uses 4 SIMD groups (num_warps=4)
instead of 32, eliminating unnecessary cross-warp synchronization."
```

---

### Task 5: Verify vector_add and other kernels still work

**Files:**
- Test: `tests/test_vector_add.py`, `tests/test_integration.py`, `tests/test_stress.py`

The sizePerThread change should only affect kernels where `sizePerThread > 1`. Vector add typically has `sizePerThread=1` or `num_warps` chosen so that `num_warps * 32 == block_size`. We need to verify no regressions.

- [ ] **Step 1: Run the full Triton-dependent test suite**

Run: `source .venv/bin/activate && python -m pytest tests/test_vector_add.py tests/test_integration.py tests/test_stress.py tests/test_mlx_backend.py tests/test_autotuner.py -v --timeout=120`
Expected: Same pass/fail as before (no new failures)

- [ ] **Step 2: Run bench_regression and compare**

Run: `source .venv/bin/activate && python benchmarks/bench_regression.py --json`
Expected: No regressions on vector_add, reduction, or dispatch overhead

- [ ] **Step 3: Commit final verification**

```bash
git commit --allow-empty -m "test: verify sizePerThread fix causes no regressions"
```

(Only if there were additional fixes needed during verification.)
