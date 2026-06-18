"""Compile-time FlashAttention pattern detector + param extractor.

Builds the real ``_flash_attn_fwd`` kernel's TTGIR (via the same
make_ir → make_ttir → make_ttgir harness as ``tests/test_mlir_walker.py``),
constructs a ``GenericLowerer`` over the resulting ``IRGraph``, and asserts
``_detect_flash_attention`` returns the FA parameter dict — pointer roles
(kernel-arg indices), the 16 strides (4 per Q/K/V/Out), the constexprs
(BLOCK_M/BLOCK_N/HEAD_DIM/IS_CAUSAL), Z/H/N_CTX arg indices, scale, and
out_dtype.

Detection is pure IR analysis: NO GPU is touched (we stop before MSL
emission / metallib compilation), so these tests run anywhere Triton can
build a module.
"""

import pytest

try:
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")

# The canonical FA kernel under test lives in test_flash_attention.py — reuse it
# rather than copy, so the detector is exercised against the exact IR the rest
# of the suite runs end-to-end on the GPU.
if _HAS_TRITON:
    from tests.test_flash_attention import _flash_attn_fwd


def _build_fa_lowerer(causal=False, head_dim=128, block=32):
    """Compile ``_flash_attn_fwd`` to TTGIR and wrap it in a GenericLowerer.

    Mirrors ``tests/test_mlir_walker.py::_compile_to_ttgir`` then walks the
    module to an IRGraph and instantiates the lowerer (no MSL emission).
    """
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})

    sig = {
        "Q": "*fp32", "K": "*fp32", "V": "*fp32", "Out": "*fp32",
        "stride_qz": "i32", "stride_qh": "i32", "stride_qm": "i32", "stride_qk": "i32",
        "stride_kz": "i32", "stride_kh": "i32", "stride_kn": "i32", "stride_kk": "i32",
        "stride_vz": "i32", "stride_vh": "i32", "stride_vn": "i32", "stride_vk": "i32",
        "stride_oz": "i32", "stride_oh": "i32", "stride_om": "i32", "stride_ok": "i32",
        "Z": "i32", "H": "i32", "N_CTX": "i32",
    }
    constexprs = {
        "BLOCK_M": block, "BLOCK_N": block, "HEAD_DIM": head_dim,
        "IS_CAUSAL": causal,
    }
    src = ASTSource(fn=_flash_attn_fwd, signature=sig, constexprs=constexprs)
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target, options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(), context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)

    graph = walk_ttgir(mod, options)
    return GenericLowerer(graph, options)


@requires_triton
def test_detect_flash_attention_non_causal():
    """Detector returns the full FA param dict for the canonical kernel."""
    lo = _build_fa_lowerer(causal=False, head_dim=128, block=32)
    info = lo._detect_flash_attention()
    assert info is not None, "expected an FA param dict for _flash_attn_fwd"

    # Pointer roles → kernel-arg indices (Q,K,V,Out = 0,1,2,3).
    assert info["q"] == 0
    assert info["k"] == 1
    assert info["v"] == 2
    assert info["out"] == 3
    # 4 distinct pointer roles.
    assert len({info["q"], info["k"], info["v"], info["out"]}) == 4

    # 16 strides: 4 contiguous arg indices per pointer.
    assert info["strides"]["q"] == [4, 5, 6, 7]
    assert info["strides"]["k"] == [8, 9, 10, 11]
    assert info["strides"]["v"] == [12, 13, 14, 15]
    assert info["strides"]["o"] == [16, 17, 18, 19]

    # Z/H/N_CTX arg indices.
    assert info["Z"] == 20
    assert info["H"] == 21
    assert info["N_CTX"] == 22

    # Constexprs from the dot tile shapes.
    assert info["block_m"] == 32
    assert info["block_n"] == 32
    assert info["head_dim"] == 128

    # Non-causal.
    assert info["causal"] is False

    # scale = 1/sqrt(head_dim) ≈ 0.0883883.
    assert abs(info["scale"] - (1.0 / (128 ** 0.5))) < 1e-4

    # fp32 output.
    assert info["out_dtype"] == "f32"


@requires_triton
def test_detect_flash_attention_causal():
    """Causal variant is recognized and flagged causal=True."""
    lo = _build_fa_lowerer(causal=True, head_dim=128, block=32)
    info = lo._detect_flash_attention()
    assert info is not None
    assert info["causal"] is True
    # Roles / strides are unaffected by the causal mask.
    assert info["q"] == 0 and info["k"] == 1 and info["v"] == 2 and info["out"] == 3
    assert info["strides"]["q"] == [4, 5, 6, 7]
    assert info["head_dim"] == 128
    assert info["block_m"] == 32 and info["block_n"] == 32


@requires_triton
def test_fa_pattern_with_unresolvable_stride_refuses():
    """Integrity hard rule: a kernel that IS an FA pattern (>=2 dot + exp +
    max) but whose stride chain can't be resolved must REFUSE, never guess.

    We take the real FA graph and sever the ``stride_qm`` splat from its
    kernel arg, so the Q addressing chain no longer resolves to 4 strides.
    The FA gate still fires (dots/exp/max are untouched), so the detector
    must raise ``MetalNonRecoverableError`` rather than return a partial dict.
    """
    from triton_metal.errors import MetalNonRecoverableError

    lo = _build_fa_lowerer(causal=False, head_dim=128, block=32)

    # Walk the (possibly nested) graph and break the stride_qm splat: clearing
    # its operand makes _stride_from_index_term return None for Q's row stride.
    qm_id = next(a.id for a in lo.graph.args if a.name == "stride_qm")

    def _break(ops):
        for s in ops:
            if s.op == "tt.splat" and s.operand_ids == [qm_id]:
                s.operand_ids = []
            if s.region_ops:
                _break(s.region_ops)
            if s.else_ops:
                _break(s.else_ops)
    _break(lo.graph.ops)

    with pytest.raises(MetalNonRecoverableError):
        lo._detect_flash_attention()


@requires_triton
def test_non_fa_kernel_returns_none():
    """A plain vector-add kernel (no dots) is not an FA pattern → None."""
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_metal.backend.compiler import MetalBackend
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import GenericLowerer

    @triton.jit
    def vector_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    src = ASTSource(fn=vector_add, signature=sig, constexprs={"BLOCK_SIZE": 256})
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target, options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(), context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    graph = walk_ttgir(mod, options)
    lo = GenericLowerer(graph, options)
    assert lo._detect_flash_attention() is None
