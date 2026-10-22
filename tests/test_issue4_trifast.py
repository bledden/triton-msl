"""Regression tests for GitHub issue #4 (trifast -> Metal porting bugs).

Each test pins a reported silent-wrong or over-refusal to the integrity contract:
a kernel triton-msl accepts must be CORRECT, and one it cannot lower must be REFUSED
(``MetalNonRecoverableError``) -- never silently wrong.

The matmul kernels below rebuild addresses from base each iteration (no loop-carried
pointer) so they isolate issue #1 (reduction-extent arg name) from issue #2.
"""

import pytest

import triton
import triton.language as tl

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from triton_msl.errors import MetalNonRecoverableError

requires = pytest.mark.skipif(not HAS_TORCH, reason="torch needed")


@triton.jit
def _mm_K(
    a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc)


@triton.jit
def _mm_DIM(
    a_ptr, b_ptr, c_ptr, M, N, DIM, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, DIM, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc)


@triton.jit
def _mm_depth(
    a_ptr, b_ptr, c_ptr, M, N, depth, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, depth, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc)


@requires
@pytest.mark.parametrize("kernel", [_mm_K, _mm_DIM, _mm_depth], ids=["K", "DIM", "depth"])
def test_reduction_extent_correct_or_refuse(kernel):
    """#1: the K-loop must reduce over the FULL extent regardless of the reduction
    arg's name, or refuse -- never silently drop the loop and return the first tile."""
    M = N = KK = 64
    torch.manual_seed(0)
    A = torch.randn(M, KK, device="cpu", dtype=torch.float32)
    B = torch.randn(KK, N, device="cpu", dtype=torch.float32)
    C = torch.empty(M, N, device="cpu", dtype=torch.float32)
    try:
        kernel[(M // 32, N // 32)](
            A,
            B,
            C,
            M,
            N,
            KK,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            BM=32,
            BN=32,
            BK=32,
        )
    except MetalNonRecoverableError:
        return  # refused loudly -- contract satisfied
    ref = A @ B
    err = (C - ref).abs().max().item()
    first_tile_err = (C - (A[:, :32] @ B[:32, :])).abs().max().item()
    assert err < 1e-3, (
        f"wrong result (err={err:.3f}); err vs first-tile-only={first_tile_err:.3f} "
        f"({'SILENT-WRONG: dropped the K-loop' if first_tile_err < 1e-3 else 'wrong'})"
    )


@requires
def test_over_31_buffer_args_refuses_clearly(tmp_path):
    """#7: certified packed arguments compute correctly beyond 31 logical slots.

    Keep the historical node name for regression continuity. The host route now
    supports producer-certified packing; unproved layouts still refuse in the
    argument-ABI and host-packed control suites.
    """
    import importlib.util

    n_scalars = 33  # 1 pointer + 33 scalars = 34 args, over the 31-slot limit
    names = [f"s{i}" for i in range(n_scalars)]
    body = " + ".join(names)
    src = (
        "import triton\nimport triton.language as tl\n\n"
        f"@triton.jit\ndef big_kernel(out_ptr, {', '.join(names)}):\n"
        f"    tl.store(out_ptr + tl.program_id(0), {body})\n"
    )
    modfile = tmp_path / "bigk.py"
    modfile.write_text(src)
    spec = importlib.util.spec_from_file_location("bigk", modfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Host path binds the certified packed signature and publishes the exact result.
    out = torch.zeros(1, device="cpu", dtype=torch.int32)
    mod.big_kernel[(1,)](out, *range(n_scalars))
    assert int(out[0]) == sum(range(n_scalars)), "packed >31-arg kernel wrong on host"

    # MPS path packs the scalars into one argument buffer -> computes correctly (issue #4.7).
    if torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"):
        out_m = torch.zeros(1, device="mps", dtype=torch.int32)
        mod.big_kernel[(1,)](out_m, *range(n_scalars))
        torch.mps.synchronize()
        assert int(out_m.cpu()[0]) == sum(range(n_scalars)), "packed >31-arg kernel wrong on MPS"


@triton.jit
def _loopcarry_ptr(in_ptr, out_ptr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    """A pointer advanced across the loop (`p += BLOCK`) -- the tutorial idiom."""
    offs = tl.arange(0, BLOCK)
    p = in_ptr + offs
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(0, STEPS):
        acc += tl.load(p)
        p += BLOCK
    tl.store(out_ptr + offs, acc)


@requires
@pytest.mark.parametrize("nw", [4, 8])
def test_loop_carried_pointer_correct_or_refuse(nw):
    """#2: a loop-carried pointer must compute correctly OR refuse -- it must not lower to
    a per-thread scalar (silent-wrong at num_warps=4, compile error at num_warps=8)."""
    BLOCK, STEPS = 256, 4
    torch.manual_seed(0)
    x = torch.randn(STEPS * BLOCK, device="cpu", dtype=torch.float32)
    out = torch.empty(BLOCK, device="cpu", dtype=torch.float32)
    # Offset-carry now handles BOTH the scalar (num_warps=8, 1 elem/thread) and the MEPT
    # (num_warps=4, per-thread offset array) case -- the loop-carried pointer must compute.
    _loopcarry_ptr[(1,)](x, out, STEPS=STEPS, BLOCK=BLOCK, num_warps=nw)
    assert (out - x.reshape(STEPS, BLOCK).sum(0)).abs().max().item() < 1e-3


@triton.jit
def _mm_square(
    a_ptr, b_ptr, c_ptr, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """Square (N x N) matmul whose output mask bounds BOTH axes by the single extent
    arg N -- the row axis is clipped by N (not by an arg named M)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    mask = (offm[:, None] < N) & (offn[None, :] < N)
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc, mask=mask)


@requires
def test_square_matmul_one_extent_arg_computes():
    """#5: an N x N matmul that bounds both output axes by one extent arg should compute
    correctly (currently refused, forcing PADDED workarounds)."""
    NN = KK = 64
    torch.manual_seed(0)
    A = torch.randn(NN, KK, device="cpu", dtype=torch.float32)
    B = torch.randn(KK, NN, device="cpu", dtype=torch.float32)
    C = torch.empty(NN, NN, device="cpu", dtype=torch.float32)
    _mm_square[(NN // 32, NN // 32)](
        A,
        B,
        C,
        NN,
        KK,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BM=32,
        BN=32,
        BK=32,
    )
    assert (C - A @ B).abs().max().item() < 1e-3


@triton.jit
def _fa_shared_kv(
    Q,
    KV,
    Out,
    sqz,
    sqh,
    sqm,
    sqk,
    skz,
    skh,
    skn,
    skk,
    soz,
    soh,
    som,
    sok,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """FA v2 forward where K and V are BOTH read through the ``KV`` pointer arg (K in
    rows [0, N_CTX), V in rows [N_CTX, 2*N_CTX)) -- shared-KV attention. Routes to the
    simdgroup FA template (HEAD_DIM=128), where Q/K/V/Out bind to distinct buffers."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk)
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(KV + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(KV + off_z * skz + off_h * skh + (N_CTX + start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk)
        acc += tl.dot(p.to(q.dtype), v.to(q.dtype))
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok, acc)


@requires
def test_shared_kv_fa_refuses():
    """#3: an FA kernel that reads V through the same pointer arg as K (shared-KV) must
    refuse -- Q/K/V/Out bind to distinct buffers with independent strides and the template
    cannot alias them, so lowering would be silently wrong."""
    Z, H, N, HD = 1, 1, 64, 128
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, HD, device="cpu", dtype=torch.float32)
    k = torch.randn(Z, H, N, HD, device="cpu", dtype=torch.float32)
    v = torch.randn(Z, H, N, HD, device="cpu", dtype=torch.float32)
    kv = torch.cat([k, v], dim=2).contiguous()  # [Z,H,2N,HD]: K then V, one buffer
    out = torch.empty(Z, H, N, HD, device="cpu", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError):
        _fa_shared_kv[(N // 32, Z * H)](
            q,
            kv,
            out,
            *q.stride(),
            *kv.stride(),
            *out.stride(),
            Z,
            H,
            N,
            BLOCK_M=32,
            BLOCK_N=32,
            HEAD_DIM=HD,
        )
