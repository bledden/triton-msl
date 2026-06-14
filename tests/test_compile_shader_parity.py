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
