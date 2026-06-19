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
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", flag)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    torch.manual_seed(0)
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
    rtol, atol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-3, 1e-3)
    A, B, C_on = _run(M, N, K, dtype, "1", monkeypatch)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C_on, ref, rtol=rtol, atol=atol)
    _, _, C_off = _run(M, N, K, dtype, "0", monkeypatch)
    torch.testing.assert_close(C_on, C_off, rtol=rtol, atol=atol)


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


def _run_f16out(M, N, K, flag, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", flag)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    os.system("rm -rf ~/.cache/triton_msl ~/.triton/cache")
    torch.manual_seed(0)
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)   # fp16 OUTPUT
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    mm_f16[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1), BM=64, BN=64, BK=32)
    torch.mps.synchronize()
    return A, B, C


@requires
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048), (512, 512, 512),
                                   (256, 2080, 256), (1024, 512, 256)])
def test_fp16_output_parity(M, N, K, monkeypatch):
    A, B, C_on = _run_f16out(M, N, K, "1", monkeypatch)
    ref = (A.float() @ B.float()).half()
    torch.testing.assert_close(C_on, ref, rtol=2e-2, atol=2e-2)
    _, _, C_off = _run_f16out(M, N, K, "0", monkeypatch)
    torch.testing.assert_close(C_on, C_off, rtol=2e-2, atol=2e-2)
