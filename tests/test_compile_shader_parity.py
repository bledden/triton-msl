"""Parity: kernels run identically (to tolerance) with the compile_shader
fast-path ON (TRITON_METAL_COMPILE_SHADER=1) vs OFF (=0). Every kernel must
match torch AND match itself across both flag values. The fast-path must NEVER
change a result; the 2-D-grid kernel MUST fall back under flag=1 yet stay
correct. Serial GPU."""
import os, pytest
try:
    import torch, triton, triton.language as tl
    HAS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS + compile_shader needed")


@triton.jit
def _vadd(A, B, OUT, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < N
    tl.store(OUT + o, tl.load(A + o, mask=m) + tl.load(B + o, mask=m), mask=m)


@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_vadd_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    N = 4096
    A = torch.randn(N, device="mps"); B = torch.randn(N, device="mps"); OUT = torch.empty(N, device="mps")
    _vadd[(triton.cdiv(N, 1024),)](A, B, OUT, N, BLOCK=1024); torch.mps.synchronize()
    torch.testing.assert_close(OUT, A + B, rtol=1e-4, atol=1e-4)


# ---- 1-D reduction: tl.sum over each row -----------------------------------
@triton.jit
def _row_sum(X, OUT, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK); m = o < N
    x = tl.load(X + row * N + o, mask=m, other=0.0)
    tl.store(OUT + row, tl.sum(x, axis=0))


@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_reduction_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    R, N = 32, 512
    X = torch.randn(R, N, device="mps"); OUT = torch.empty(R, device="mps")
    _row_sum[(R,)](X, OUT, N, BLOCK=triton.next_power_of_2(N)); torch.mps.synchronize()
    torch.testing.assert_close(OUT, X.sum(dim=1), rtol=1e-4, atol=1e-3)


# ---- softmax-style: max + exp + sum + div over each row --------------------
@triton.jit
def _softmax(X, OUT, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK); m = o < N
    x = tl.load(X + row * N + o, mask=m, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(OUT + row * N + o, e / s, mask=m)


@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_softmax_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    R, N = 16, 256
    X = torch.randn(R, N, device="mps"); OUT = torch.empty(R, N, device="mps")
    _softmax[(R,)](X, OUT, N, BLOCK=triton.next_power_of_2(N)); torch.mps.synchronize()
    torch.testing.assert_close(OUT, torch.softmax(X, dim=1), rtol=1e-4, atol=1e-4)


# ---- masked tl.where -------------------------------------------------------
@triton.jit
def _where(A, B, COND, OUT, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < N
    c = tl.load(COND + o, mask=m, other=0) != 0
    out = tl.where(c, tl.load(A + o, mask=m), tl.load(B + o, mask=m))
    tl.store(OUT + o, out, mask=m)


@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_where_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    N = 4096
    A = torch.randn(N, device="mps"); B = torch.randn(N, device="mps")
    COND = (torch.rand(N, device="mps") > 0.5).to(torch.int32)
    OUT = torch.empty(N, device="mps")
    _where[(triton.cdiv(N, 1024),)](A, B, COND, OUT, N, BLOCK=1024); torch.mps.synchronize()
    torch.testing.assert_close(OUT, torch.where(COND != 0, A, B), rtol=1e-4, atol=1e-4)


# ---- tl.atomic_add scatter -------------------------------------------------
@triton.jit
def _scatter_add(IDX, VAL, OUT, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < N
    idx = tl.load(IDX + o, mask=m, other=0)
    val = tl.load(VAL + o, mask=m, other=0.0)
    tl.atomic_add(OUT + idx, val, mask=m)


@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_atomic_add_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    N, NBUCKET = 4096, 64
    IDX = torch.randint(0, NBUCKET, (N,), device="mps", dtype=torch.int32)
    VAL = torch.randn(N, device="mps")
    OUT = torch.zeros(NBUCKET, device="mps")
    _scatter_add[(triton.cdiv(N, 1024),)](IDX, VAL, OUT, N, BLOCK=1024); torch.mps.synchronize()
    ref = torch.zeros(NBUCKET, device="mps").index_add_(0, IDX.to(torch.int64), VAL)
    torch.testing.assert_close(OUT, ref, rtol=1e-4, atol=1e-3)


# ---- 2-D-grid kernel: uses tl.program_id(1) -> MUST fall back under flag=1 --
@triton.jit
def _add2d(A, B, OUT, M, N, BLOCK: tl.constexpr):
    rm = tl.program_id(1)
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < N
    base = rm * N + o
    tl.store(OUT + base, tl.load(A + base, mask=m) + tl.load(B + base, mask=m), mask=m)


@requires
@pytest.mark.parametrize("flag", ["1", "0"])
def test_2d_grid_parity(flag, monkeypatch):
    monkeypatch.setenv("TRITON_METAL_COMPILE_SHADER", flag)
    M, N = 8, 4096
    A = torch.randn(M, N, device="mps"); B = torch.randn(M, N, device="mps")
    OUT = torch.empty(M, N, device="mps")
    _add2d[(triton.cdiv(N, 1024), M)](A, B, OUT, M, N, BLOCK=1024); torch.mps.synchronize()
    torch.testing.assert_close(OUT, A + B, rtol=1e-4, atol=1e-4)
