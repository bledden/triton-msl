"""Unified per-thread value model for the MEPT register-array spine.

Every SSA value is a RegVal(name, n_elems, ty, form). Scalars are n_elems==1,
form='scalar'. ``form`` selects emission: 'scalar' (no loop), 'wraploop' (the
existing _loop_e re-execution loop), or 'array' (T name[n_elems], the only
correct form when per-element state crosses data-dependent control flow).
See docs/superpowers/specs/2026-06-11-mept-register-array-spine-design.md.
"""
from dataclasses import dataclass

_CONTROL_OPS = ("scf.for", "scf.while", "scf.if")


@dataclass
class RegVal:
    name: str
    n_elems: int = 1
    ty: str = ""
    form: str = "scalar"  # 'scalar' | 'wraploop' | 'array'

    @property
    def is_scalar(self) -> bool:
        return self.n_elems == 1 and self.form == "scalar"


def region_needs_arrays(ops, multi_elem_ids) -> bool:
    """True if ``ops`` contains a data-dependent control-flow op (scf.for/
    while/if) whose body references or carries a multi-element value.

    Such regions cannot use the re-execution wrap-loop (it can't carry
    per-element state across the control-flow loop); they require true
    register arrays. ``multi_elem_ids`` is the set of SSA ids with n_elems>1.
    """
    multi = set(multi_elem_ids)
    for op in ops:
        if op.op in _CONTROL_OPS:
            body = op.region_ops or []
            if any(oid in multi for oid in (op.operand_ids or [])):
                return True
            for b in body:
                if any(oid in multi for oid in (getattr(b, "operand_ids", None) or [])):
                    return True
                if getattr(b, "id", None) in multi:
                    return True
            if region_needs_arrays(body, multi):
                return True
    return False
