"""Default-deny for un-handled side-effecting ops (audit #165).

An unknown op with NO result (negative synthetic id — a store/atomic/scatter/cf
variant the lowerer doesn't model) must REFUSE loudly, not be dropped silently
(which would lose its side effect). An unknown op WITH a result is tolerated
(its UNKNOWN_<id> result fails loud at MSL compile if consumed, or is harmless
if dead), so it stays a comment.
"""
import pytest

from triton_metal.codegen.generic_lowerer import GenericLowerer
from triton_metal.codegen.mlir_walker import SSAValue
from triton_metal.errors import MetalNonRecoverableError


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
    import triton_metal.codegen.msl_emitter as _m
    lo.kb = _m.KernelBuilder("k")
    try:
        lo._lower_op_dispatch(_mk("tt.some_unknown_valued", 123))
    except MetalNonRecoverableError:
        pytest.fail("result-producing unknown op should not refuse")
