"""Regression: a cooperative (over-threadgroup) staged fill must honour the mask of
the ``tl.load`` it re-reads, or refuse.

Background (2026-08-26). When a 2-D tile is larger than the threadgroup, the generic
lowerer does not reuse the per-thread loaded value; it re-reads the tile straight from
global memory, rebuilding each element's address from ``(_fill_row, _fill_col)``. That
rebuild recovered the ADDRESS but dropped the load's ``mask=``/``other=``, so the fill
read unconditionally:

    smem[_sa] = K[_fill_col + _fill_row * skt + ... + k_32 * skt];   // no guard

For a varlen kernel the last K/V block of a short sequence starts past the end of the
packed tensor, so this reads OUT OF BOUNDS. The failure is not loud: the masked lanes
still compute ``p == 0`` correctly, but the accumulator then evaluates ``0 * garbage``
and ``0 * NaN == NaN``, so the NaN reaches the output as a SILENT WRONG. Whether the
garbage happens to be NaN depends on what the allocator left behind, which is why it
only showed up after other GPU work had run in the same process.

The cooperative STORE path already resolved its mask structurally (or refused); this is
the same guard on the load side. These tests pin the numeric result for the shape that
overhangs, and pin that the emitted MSL actually carries the guard.
"""

import math

import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _varlen_overhang(
    Q,
    K,
    V,
    Out,
    cu_q,
    cu_k,
    sqt,
    sqh,
    sqd,
    skt,
    skh,
    skd,
    svt,
    svh,
    svd,
    sot,
    soh,
    sod,
    H,
    GROUP,
    max_seqlen,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    """Varlen FA on a GQA head so it lowers generically. With ragged lengths the final
    K/V block of the SHORTER sequence begins past the end of the packed tensor — the
    tile that the staged fill used to read unguarded."""
    sm = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    hkv = h // GROUP
    qs = tl.load(cu_q + b)
    slq = tl.load(cu_q + b + 1) - qs
    ks = tl.load(cu_k + b)
    slk = tl.load(cu_k + b + 1) - ks
    om = sm * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    od = tl.arange(0, D)
    q = tl.load(Q + (qs + om)[:, None] * sqt + h * sqh + od[None, :] * sqd, mask=om[:, None] < slq, other=0.0) * SCALE
    mi = tl.full([BM], float("-inf"), tl.float32)
    li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for sn in range(0, max_seqlen, BN):
        kn = sn + on
        k = tl.load(K + (ks + kn)[:, None] * skt + hkv * skh + od[None, :] * skd, mask=kn[:, None] < slk, other=0.0)
        qk = tl.where(kn[None, :] < slk, tl.dot(q, tl.trans(k).to(q.dtype)), float("-inf"))
        m2 = tl.maximum(mi, tl.max(qk, 1))
        a = tl.exp(mi - m2)
        p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1)
        acc = acc * a[:, None]
        vv = tl.load(V + (ks + kn)[:, None] * svt + hkv * svh + od[None, :] * svd, mask=kn[:, None] < slk, other=0.0)
        acc += tl.dot(p.to(tl.float32), vv.to(tl.float32))
        mi = m2
    tl.store(
        Out + (qs + om)[:, None] * sot + h * soh + od[None, :] * sod,
        (acc / li[:, None]).to(Out.dtype.element_ty),
        mask=om[:, None] < slq,
    )


@triton.jit
def _primer_matmul(
    a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """An unrelated fp32 matmul, run only for its side effect on the allocator."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_mask = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _prime_allocator():
    """Leave real fp32 kernel output in the allocator's free blocks.

    An out-of-bounds read only *shows* when the memory past the tensor holds something
    that corrupts the result. Freshly-mapped pages read as zero, and zero is
    indistinguishable from a correct mask — so an unguarded fill on a clean heap looks
    fine. Running and freeing an unrelated fp32 kernel first puts real float data
    there; reinterpreted as fp16 its low (even-indexed) halves contain NaN bit
    patterns, which is exactly the even-column NaN signature this bug produced.
    """
    dev = "mps"
    M, N, K = 100, 80, 48  # deliberately non-multiples of 32, as in the example
    torch.manual_seed(1234)
    at = torch.randn(M, K, device=dev, dtype=torch.float32)
    bt = torch.randn(K, N, device=dev, dtype=torch.float32)
    ct = torch.zeros(M, N, device=dev, dtype=torch.float32)
    _primer_matmul[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
        at, bt, ct, M, N, K, *at.stride(), *bt.stride(), *ct.stride(), BM=32, BN=32, BK=32
    )
    torch.mps.synchronize()
    del at, bt, ct


def _run(dtype, lens, D=64, nw=4, H=4, Hkv=2, prime=True):
    """Run the kernel and return max|Δ| vs a torch reference.

    Tensors are exact-size allocations (not views into a padded buffer): the overhang
    must read past the END of the Metal buffer, which is where the corruption lives. A
    padded view keeps the read inside the buffer, where it is harmless.
    """
    dev = "mps"
    if prime:
        _prime_allocator()
    torch.manual_seed(0)
    group = H // Hkv
    cu = torch.tensor([0] + list(torch.tensor(list(lens)).cumsum(0)), device=dev, dtype=torch.int32)
    total = int(cu[-1])
    q = torch.randn(total, H, D, device=dev, dtype=dtype)
    k = torch.randn(total, Hkv, D, device=dev, dtype=dtype)
    v = torch.randn(total, Hkv, D, device=dev, dtype=dtype)
    o = torch.zeros(total, H, D, device=dev, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    mx = max(lens)
    _varlen_overhang[(triton.cdiv(mx, 32), len(lens) * H)](
        q,
        k,
        v,
        o,
        cu,
        cu,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        H,
        group,
        mx,
        scale,
        32,
        32,
        D,
        num_warps=nw,
    )
    torch.mps.synchronize()

    ref = torch.zeros_like(o)
    for b in range(len(lens)):
        s, e = int(cu[b]), int(cu[b + 1])
        for h in range(H):
            hk = h // group
            sc = (q[s:e, h].float() * scale) @ k[s:e, hk].float().T
            ref[s:e, h] = (torch.softmax(sc, -1) @ v[s:e, hk].float()).to(ref.dtype)
    return (o.float() - ref.float()).abs().max().item()


@requires_mps
@pytest.mark.parametrize("nw", [4, 8])
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-2), (torch.float32, 1e-3)])
def test_overhanging_kv_block_is_masked_not_oob(dtype, tol, nw):
    # lens=(48, 32): the second sequence is 32 long, so its sn=32 K/V block starts at
    # packed row 80 of an 80-row tensor — entirely out of bounds.
    #
    # NOTE on fp32: unguarded, fp32 does NOT corrupt in this allocation state while
    # fp16 does. That is luck, not correctness — the emitted fill was identical and
    # read out of bounds either way; the fp32 read simply landed on bytes that did not
    # poison the result. fp32's real guarantee is the structural one below (the guard
    # is emitted for every dtype), so this case is a numeric floor, not the proof.
    err = _run(dtype, (48, 32), nw=nw)
    assert err == err, f"staged fill read out of bounds: got NaN (dtype={dtype}, nw={nw})"
    assert err < tol, f"staged fill wrong: err {err:.2e} (dtype={dtype}, nw={nw})"


@requires_mps
@pytest.mark.parametrize("lens", [(64, 16, 33), (32, 32), (17, 63)])
def test_ragged_lengths_stay_finite(lens):
    err = _run(torch.float16, lens)
    assert err == err, f"staged fill read out of bounds for lens={lens}"
    assert err < 2e-2, f"staged fill wrong for lens={lens}: err {err:.2e}"


def _emitted_msl(dtype):
    """Exercise the lowering boundary, never a cached executable or temp file.

    The old newest-file search depended on the linker's disposable .metal input;
    an outer JIT hit could also leave no file in this test's private cache. Build
    this source's TTGIR afresh and inspect the actual generic lowerer's output.
    Numeric allocation/overhang tests above remain independent GPU checks.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    element = {torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    signature = {name: "*" + element for name in ("Q", "K", "V", "Out")}
    signature.update({name: "*i32" for name in ("cu_q", "cu_k")})
    signature.update(
        {
            name: "i32"
            for name in (
                "sqt",
                "sqh",
                "sqd",
                "skt",
                "skh",
                "skd",
                "svt",
                "svh",
                "svd",
                "sot",
                "soh",
                "sod",
                "H",
                "GROUP",
                "max_seqlen",
            )
        }
    )
    source = ASTSource(
        _varlen_overhang, signature=signature, constexprs={"SCALE": 1 / math.sqrt(64), "BM": 32, "BN": 32, "D": 64}
    )
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    module = source.make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return GenericLowerer(walk_ttgir(module, options), options).lower()


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_emitted_msl_guards_every_staged_fill(dtype):
    """Structural pin, and the real guarantee for fp32.

    Every staged fill that re-reads a MASKED load must carry a guard. The numeric tests
    above only catch this when the memory past the tensor happens to hold poisonous
    bytes — which for fp32 it did not. Pinning the emitted guard is what makes fp32
    correct rather than lucky: the read simply does not leave the buffer any more,
    whatever the allocator left behind.
    """
    import re

    src = _emitted_msl(dtype)
    assert src, f"lowering emitted no MSL for _varlen_overhang ({dtype})"

    fills = re.findall(r"^\s*(smem_\w+)\[_sa\] = (.+);$", src, re.M)
    assert fills, "expected cooperative staged fills in the emitted MSL"
    # Only the fills that REBUILD a global address from (_fill_row, _fill_col) re-read
    # memory and so need the load's mask restated. A fill that copies the per-thread
    # value carries the scalar path's own `mask ? ... : other` already.
    rebuilt = [(n, e) for n, e in fills if "_fill_row" in e or "_fill_col" in e]
    assert rebuilt, "expected address-rebuilding staged fills for this kernel"
    unguarded = [(n, e) for n, e in rebuilt if "?" not in e]
    assert not unguarded, "staged fill re-reads a masked load without a guard (out-of-bounds read): " + "; ".join(
        f"{n}[_sa] = {e}" for n, e in unguarded
    )
