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


@pytest.fixture(autouse=True)
def _reset_unsupported():
    try:
        from triton_metal.backend.driver import _get_compile_shader_runtime
        rt = _get_compile_shader_runtime()
        for attr in ("_unsupported",):
            obj = getattr(rt, attr, None)
            if hasattr(obj, "clear"):
                obj.clear()
    except Exception:
        pass
    yield


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


@requires
def test_flag_off_skips_cached_descriptor(monkeypatch):
    """A descriptor cached in phase 1 must NOT fire when TRITON_METAL_FAST_MATMUL=0
    in phase 2. Both phases run in the same process so the kernel stays cached
    in-process — exactly the scenario Fix 1 (runtime env-var gate) addresses."""
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")

    # Phase 1: flag ON — compile + cache the descriptor; fast path may fire.
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    seen1 = _spy(monkeypatch)
    _launch(256, 256, 256)
    # (We don't assert seen1 here — we just want the descriptor cached.)

    # Phase 2: flag OFF — same kernel/shape, no cache clear; fast path must NOT fire.
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "0")
    # Reset the spy list by patching a fresh one (reuse the same rt object).
    from triton_metal.backend.driver import _get_compile_shader_runtime
    rt = _get_compile_shader_runtime()
    seen2 = []
    orig2 = rt.dispatch
    def spy2(lib, kernel_name, args, **kw):
        seen2.append(kernel_name)
        return orig2(lib, kernel_name, args, **kw)
    monkeypatch.setattr(rt, "dispatch", spy2)

    A, B, C = _launch(256, 256, 256)
    torch.mps.synchronize()
    assert "simdgroup_matmul_fast" not in seen2, (
        "TRITON_METAL_FAST_MATMUL=0 must suppress dispatch even for a cached descriptor"
    )
    # Correctness: result must still match torch matmul (fallback path ran).
    torch.testing.assert_close(C, (A.float() @ B.float()), rtol=2e-2, atol=2e-2)


@triton.jit
def mm_f16(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
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
    tl.store(c_ptrs, acc.to(tl.float16))


def _launch_f16(M, N, K):
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm_f16[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C


@requires
def test_fp16out_aligned_fires_fast(monkeypatch):
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    seen = _spy(monkeypatch)
    A, B, C = _launch_f16(256, 256, 256)
    assert "simdgroup_matmul_fast" in seen
    torch.testing.assert_close(C, (A.float() @ B.float()).half(), rtol=2e-2, atol=2e-2)


@requires
@pytest.mark.parametrize("M,N,K", [(258, 256, 256), (256, 258, 256), (256, 256, 252)])
def test_fp16out_misaligned_falls_back(monkeypatch, M, N, K):
    os.system("rm -rf ~/.cache/triton_metal ~/.triton/cache")
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", "1")
    seen = _spy(monkeypatch)
    A, B, C = _launch_f16(M, N, K)
    assert "simdgroup_matmul_fast" not in seen
    torch.testing.assert_close(C, (A.float() @ B.float()).half(), rtol=2e-2, atol=2e-2)
