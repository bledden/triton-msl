"""CPU native-IR checks for the mask-honoring route after unit specialization."""

import pytest
import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError


def _unit_lowerer():
    from tests.test_full_k_direct_gpu import full_k

    signature = {
        name: "*fp32" if name in ("A", "B", "C") else "i32"
        for name in ("A", "B", "C", "M", "N", "K", "SAM", "SAK", "SBK", "SBN", "SCM", "SCN")
    }
    constants = dict(M=1, K=1, SAK=1, SBK=1, SCN=1, BM=32, BN=32, BK=32, MASK_MN=True)
    signature.update({name: "constexpr" for name in constants})
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    module = ASTSource(full_k, signature, constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    return GenericLowerer(walk_ttgir(module, options), options)


def test_specialized_unit_mask_reaches_only_the_mask_honoring_route(monkeypatch):
    lowerer = _unit_lowerer()
    assert lowerer._template_output_mask_nontrivial(False)
    assert lowerer._masked_dot_generic_eligible()
    assert not lowerer._requires_matmul_template()
    calls = []
    generic = lowerer._lower_dot

    def observe(op):
        calls.append(op.id)
        return generic(op)

    def forbidden(*args):
        raise AssertionError("masked unit dot reached a mask-dropping template")

    monkeypatch.setattr(lowerer, "_lower_dot", observe)
    monkeypatch.setattr(lowerer, "_lower_simple_dot_inline", forbidden)
    source = lowerer.lower()
    assert len(calls) == 1
    assert "smem_dot_" in source and "(int)1" in source and "if (" in source


def test_unit_mask_still_refuses_without_complete_generic_proof(monkeypatch):
    lowerer = _unit_lowerer()
    monkeypatch.setattr(lowerer, "_masked_dot_generic_eligible", lambda: False)
    with pytest.raises(MetalNonRecoverableError, match="non-tile-boundary"):
        lowerer.lower()


def test_unknown_staged_mask_algebra_is_not_admitted():
    from tests.test_masked_dot_loads import _direct_lowerer

    with pytest.raises(
        MetalNonRecoverableError, match="masked.*(not supported|cannot be safely reconstructed|per-element)"
    ):
        _direct_lowerer(mask_mode=2, other_mode=1).lower()
