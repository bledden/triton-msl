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
