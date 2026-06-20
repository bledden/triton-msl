"""Default-deny for un-handled side-effecting ops (audit #165).

An unknown op with NO result (negative synthetic id — a store/atomic/scatter/cf
variant the lowerer doesn't model) must REFUSE loudly, not be dropped silently
(which would lose its side effect). An unknown op WITH a result is tolerated
(its UNKNOWN_<id> result fails loud at MSL compile if consumed, or is harmless
if dead), so it stays a comment.
"""
import pytest

from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import SSAValue
from triton_msl.errors import MetalNonRecoverableError


def _mk(op, ssa_id):
    return SSAValue(id=ssa_id, name="v", op=op, operand_ids=[], attrs={},
                    type_str="", elem_type="f32", is_tensor=False)


def _bare_lowerer():
    lo = GenericLowerer.__new__(GenericLowerer)
    lo.env = {}
    lo.env_types = {}
    lo.env_shapes = {}
    return lo


def test_no_result_unknown_op_refuses():
    # Negative id == no result (mlir_walker assigns -counter to result-less ops).
    lo = _bare_lowerer()
    with pytest.raises(MetalNonRecoverableError):
        lo._lower_op_dispatch(_mk("tt.some_unknown_sideeffect", -7))


def test_result_producing_unknown_op_does_not_refuse():
    # A positive id == has a result; tolerated as a comment (fails loud only if
    # consumed). Must NOT raise MetalNonRecoverableError.
    lo = _bare_lowerer()
    import triton_msl.codegen.msl_emitter as _m
    lo.kb = _m.KernelBuilder("k")
    try:
        lo._lower_op_dispatch(_mk("tt.some_unknown_valued", 123))
    except MetalNonRecoverableError:
        pytest.fail("result-producing unknown op should not refuse")


@pytest.mark.skipif(
    __import__("platform").system() != "Darwin",
    reason="Metal backend requires macOS",
)
def test_scan_over_1024_elements_refuses():
    """Regression: tl.cumsum / tl.associative_scan over a >1024-element tile must
    refuse loudly. The scan stages one element per thread through threadgroup memory
    and Metal caps a threadgroup at 1024 threads, so elements past 1024 are left
    uninitialized -> silently-wrong result. A <=1024 scan stays correct."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def kscan(inp, out, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        tl.store(out + i, tl.cumsum(tl.load(inp + i), axis=0))

    # <=1024 stays correct.
    a = torch.randn(1024)
    o = torch.empty(1024)
    kscan[(1,)](a, o, BLOCK=1024)
    assert (o - a.cumsum(0)).abs().max().item() < 1e-3

    # >1024 -> loud refusal (was a silent-wrong path: elements past 1024 dropped).
    a2 = torch.randn(2048)
    o2 = torch.empty(2048)
    with pytest.raises(MetalNonRecoverableError):
        kscan[(1,)](a2, o2, BLOCK=2048)


def test_generic_dot_refuses_large_tile():
    """Regression: a tt.dot reaching the generic per-thread `_lower_dot` fallback
    with a tile dim > 64 must REFUSE. Validated dots (matmul/FA templates) return
    early in lower(); a dot landing here escaped them and the per-thread mapping
    silently mis-computes large tiles. The refusal is on the PRIMITIVE (oversized
    dot tile), not the FA idiom — so a max-less head_dim-128 attention that slips
    past the FA `max`-substring heuristic can't reach the wrong path silently."""
    lo = _bare_lowerer()
    lo._find_op_type_str = lambda oid: "tensor<128x128xf32>"
    lo._lookup = lambda oid: "acc"
    big = SSAValue(id=1, name="d", op="tt.dot", operand_ids=[10, 11, 12],
                   attrs={}, type_str="tensor<128x128xf32>", elem_type="f32",
                   is_tensor=True)
    with pytest.raises(MetalNonRecoverableError):
        lo._lower_dot(big)

    # A <=64 tile must NOT trip the large-tile guard (it may fail later for an
    # unrelated reason in this bare setup, but never via the ">64" refusal).
    lo._find_op_type_str = lambda oid: "tensor<32x32xf32>"
    small = SSAValue(id=2, name="d", op="tt.dot", operand_ids=[10, 11, 12],
                     attrs={}, type_str="tensor<32x32xf32>", elem_type="f32",
                     is_tensor=True)
    try:
        lo._lower_dot(small)
    except MetalNonRecoverableError as e:
        assert "> 64" not in str(e), "32x32 dot wrongly hit the large-tile guard"
    except Exception:
        pass  # any other emission detail in the bare lowerer is irrelevant here
