"""Recover the twelve real autotuner tiles without weakening the staging guard."""

import math
import re

import pytest
import torch
import triton

from test_fa_tail_semantics import executed, _executed_source  # noqa: F401
from test_flash_attention import _flash_attn_fwd
from triton_msl.errors import MetalNonRecoverableError


def _graph(dtype, bm, bn, causal):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    opts = backend.parse_options({})
    constants = dict(
        Z=1, stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1, BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=64, IS_CAUSAL=causal
    )
    signature = {
        n: ("*" + dtype if n in ("Q", "K", "V", "Out") else "i32")
        for n in _flash_attn_fwd.arg_names
        if n not in constants
    }
    source = ASTSource(_flash_attn_fwd, signature, constexprs=constants)
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(target, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), context)
    metadata = {}
    mod = backend.make_ttir(mod, metadata, opts)
    mod = backend.make_ttgir(mod, metadata, opts)
    return walk_ttgir(mod, opts), opts


@pytest.mark.parametrize("bm,bn", [(32, 64), (64, 32), (64, 64)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_larger_attention_tiles_compute_source(bm, bn, causal, dtype, executed, monkeypatch):
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    graph, opts = _graph(dtype, bm, bn, causal)
    lower = GenericLowerer(graph, opts)
    msl = lower.lower()  # Fresh lowering must compute, not refuse or reuse a JIT cache.
    # 32 by 64 is the matrix shader's own tile, and a contiguous source of one float type takes it
    # (test_a_64_wide_key_block_takes_the_matrix_attention_shader); the tiled shader keeps the rest.
    route_marker = "simdgroup-MMA FlashAttention-2" if (bm, bn) == (32, 64) else "Head-dim-tiled FlashAttention-2"
    assert route_marker in msl
    if (bm, bn) == (32, 64):
        assert lower.effective_block_size == 256
    else:
        assert lower.effective_block_size == 1024
        assert "const uint BM = 32u;" in msl
        assert f"const uint BN = {bn}u;" in msl
        assert f"const uint DC = {32 if bn == 64 else 64}u;" in msl
        arrays = re.findall(r"threadgroup float \w+\[([0-9 *]+)\];", msl)
        assert len(arrays) == 5
        declared_bytes = 4 * sum(math.prod(int(n.strip()) for n in size.split("*")) for size in arrays)
        assert declared_bytes == (24832 if bn == 64 else 20736)
        assert declared_bytes <= 32768
    if bm == 64:
        assert "const uint SOURCE_BM = 64u;" in msl
        assert "query_offset < SOURCE_BM; query_offset += BM" in msl
        if causal:
            assert "min((q_block + 1u) * SOURCE_BM, N_CTX)" in msl
    if not torch.backends.mps.is_available():
        pytest.skip("GPU equivalence requires MPS; lowering contract ran")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    torch.manual_seed(293)
    td = torch.float32 if dtype == "fp32" else torch.float16
    n, d = 65, 64
    q, k, v = [torch.randn(1, 2, n, d).to(td) for _ in range(3)]
    v += 1
    padded = triton.cdiv(n, bn) * bn
    kp, vp = [torch.zeros(1, 2, padded, d) for _ in range(2)]
    kp[:, :, :n], vp[:, :, :n] = k.float(), v.float()
    scores = (q.float() * d**-0.5) @ kp.transpose(-1, -2)
    if causal:
        scores.masked_fill_(torch.arange(padded)[None, :] > torch.arange(n)[:, None], -float("inf"))
    reference = (scores.softmax(-1) @ vp).to(td)
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full_like(q, float("nan"))

    def launch():
        return _flash_attn_fwd[(triton.cdiv(n, bm), 2)](
            q,
            k,
            v,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            1,
            2,
            n,
            BLOCK_M=bm,
            BLOCK_N=bn,
            HEAD_DIM=d,
            IS_CAUSAL=causal,
        )

    compiled = launch()
    torch.mps.synchronize()
    assert route_marker in _executed_source(executed, compiled)
    torch.testing.assert_close(out.cpu(), reference, atol=0.002, rtol=0.002)
    # Splitting query storage must NOT shorten the source program's visited KV
    # range: its first SOURCE_BM=64 queries all visit the NaN in row40, even
    # though the first physical query subtile has only32 rows.
    if (bm, bn, causal, dtype) == (64, 32, True, "fp32"):
        v[:, :, 40] = float("nan")
        executed.clear()
        compiled = launch()
        torch.mps.synchronize()
        assert "Head-dim-tiled FlashAttention-2" in _executed_source(executed, compiled)
        assert torch.isnan(out.cpu()).all()


@pytest.mark.parametrize("block", [32, 64])
@pytest.mark.parametrize("effect", ["rmw", "cas"])
def test_attention_does_not_discard_atomic_effects(block, effect, monkeypatch):
    import sys
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    fn = triton.JITFunction(_flash_attn_fwd.fn)
    update = (
        "tl.atomic_add(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)"
        if effect == "rmw"
        else "tl.atomic_cas(o_ptrs, acc.to(Out.dtype.element_ty), tl.full((BLOCK_M, HEAD_DIM), 0., tl.float32))"
    )
    fn._unsafe_update_src(fn.src + "\n    " + update + "\n")
    monkeypatch.setattr(sys.modules[__name__], "_flash_attn_fwd", fn)
    graph, opts = _graph("fp32", block, block, False)
    expected = "tt.atomic_rmw" if effect == "rmw" else "tt.atomic_cas"
    assert any(op.op == expected for op in graph.ops)
    with pytest.raises(MetalNonRecoverableError, match="effect"):
        GenericLowerer(graph, opts).lower()
