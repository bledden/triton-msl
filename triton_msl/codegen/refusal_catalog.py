"""Single source of truth for the integrity refusal catalog.

The integrity contract: when the compiler recognizes a kernel it cannot lower
correctly *and* knows the legacy text parser cannot either, it raises
``MetalNonRecoverableError`` rather than emitting silently-wrong output (a
kernel that runs and returns wrong numbers). See
``docs/ARCHITECTURE.md`` → "Lowering paths and the integrity model".

This module is the **one place** that catalog lives, so that:

  * ``GenericLowerer._refuse_unsafe_unsupported_ops`` walks it (Python path);
  * the C++ MLIR→LLVM path can consume the same catalog via
    :func:`export_json` (it must refuse the *exact same* set with the *exact
    same* messages, so the two code paths cannot drift);
  * ``docs/ARCHITECTURE.md`` can be generated from :func:`doc_markdown`;
  * tests can round-trip every case (``tests/test_refusal_catalog.py``).

Adding a new prescan refusal = adding a :class:`RefusalCase` here, nowhere
else. Two refusals are *contextual* (they need local lowering state and live
in the template lowerers, not the prescan); they are recorded here as
:data:`CONTEXTUAL_REFUSALS` metadata for documentation/export completeness
but are not part of the prescan walker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class Violation:
    """A concrete refusal: the user-facing message and the offending op name.

    The message may contain case-specific dynamic detail (e.g. the actual op
    name or tensor rank); it is built inside the case's ``check`` so it is
    byte-identical to what the prescan emitted before this catalog existed.
    """

    message: str
    op_name: str


@dataclass
class RefusalContext:
    """Everything a refusal predicate needs, injected by the lowerer.

    Passing accessors as callables (rather than importing lowerer internals)
    keeps this module free of any import coupling to ``generic_lowerer`` /
    ``mlir_walker``.
    """

    all_ops: list                 # ops walked recursively (incl. region/else)
    top_level_ops: list           # graph.ops only — NO recursion
    called_funcs: list            # graph.called_funcs or []
    find_op_type_str: Callable[[object], Optional[str]]  # ssa_id -> type str
    extract_shape: Callable[[str], Sequence]             # type str -> shape


@dataclass(frozen=True)
class RefusalCase:
    """One prescan refusal. ``check`` returns a :class:`Violation` or None."""

    name: str                     # stable snake_case id
    summary: str                  # one-line human description (docs)
    rationale: str                # WHY refusing is the integrity-correct call
    examples: tuple               # upstream test names that exercise it
    trigger_ops: tuple            # ops that can trigger it (docs / C++ export)
    check: Callable[[RefusalContext], Optional[Violation]] = field(
        default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ContextualRefusal:
    """A refusal that lives in the template lowerers (needs local state), not
    the prescan. Recorded for catalog completeness; has no ``check`` here."""

    name: str
    summary: str
    rationale: str
    examples: tuple
    location: str                 # where the guard actually lives


# ── Shared constants ──────────────────────────────────────────────────────

# Ops with no Apple hardware and no codegen handler — the output tensor is
# never computed, so any emission is silently-wrong. Keep this tight: only ops
# that are both unsupported *and* unsafe to approximate belong here.
UNSAFE_UNSUPPORTED_OPS = frozenset({
    # Microscaling (mxfp) matmul — no Apple hardware (test_scaled_dot).
    "tt.dot_scaled",
})

# Value-passing ops a tt.join result can flow through on its way to a tt.dot.
_JOIN_PASSTHROUGH_OPS = (
    "tt.reshape", "tt.trans", "ttg.convert_layout", "ttg.local_alloc",
    "ttg.local_load", "ttg.memdesc_trans", "tt.broadcast", "tt.expand_dims",
)


# ── Prescan predicates (cases a–e) ─────────────────────────────────────────

def _check_unsafe_unsupported_op(ctx: RefusalContext) -> Optional[Violation]:
    for s in ctx.all_ops:
        if s.op in UNSAFE_UNSUPPORTED_OPS:
            return Violation(
                f"'{s.op}' has no correct lowering on the Metal backend "
                "and cannot be safely approximated (e.g. microscaling "
                "matmul has no Apple hardware). Refusing rather than "
                "emitting wrong numbers.", s.op)
    return None


def _check_noinline_dot(ctx: RefusalContext) -> Optional[Violation]:
    for cf in ctx.called_funcs:
        if any(o.op == "tt.dot" for o in (cf.ops or [])):
            return Violation(
                "tt.dot inside a noinline device function is not "
                "supported — the device-function lowerer cannot emit "
                "cooperative matrix-multiply, so the result would be "
                "zeros (test_noinline[shared]).", "tt.call")
    return None


def _check_nd_cat_join(ctx: RefusalContext) -> Optional[Violation]:
    for s in ctx.all_ops:
        if s.op in ("tt.cat", "tt.join") and s.operand_ids:
            sh = ctx.extract_shape(
                ctx.find_op_type_str(s.operand_ids[0]) or "")
            if sh and len(sh) >= 2:
                return Violation(
                    f"'{s.op}' on a rank-{len(sh)} tensor is not "
                    "supported (the generic handler is 1-D only); an "
                    "N-D concat/join would be laid out incorrectly "
                    "(test_cat_nd).", s.op)
    return None


def _check_join_into_dot(ctx: RefusalContext) -> Optional[Violation]:
    op_by_id = {s.id: s for s in ctx.all_ops}
    join_ids = {s.id for s in ctx.all_ops if s.op == "tt.join"}
    if not join_ids:
        return None

    def reaches_join(sid, depth=0):
        if depth > 16 or sid in join_ids:
            return sid in join_ids
        o = op_by_id.get(sid)
        if not o or o.op not in _JOIN_PASSTHROUGH_OPS or not o.operand_ids:
            return False
        return reaches_join(o.operand_ids[0], depth + 1)

    for s in ctx.all_ops:
        if s.op == "tt.dot" and any(
                reaches_join(oid) for oid in s.operand_ids):
            return Violation(
                "a tt.join result feeding tt.dot is not "
                "supported — the interleaved join layout is not "
                "what the matmul template expects, so the "
                "product would be wrong (test_join_with_mma).", "tt.dot")
    return None


def _check_unstructured_cf(ctx: RefusalContext) -> Optional[Violation]:
    # Only TOP-LEVEL ops: tt.map_elementwise bodies also use cf.cond_br but
    # those live in region_ops and ARE handled
    # (_lower_map_elementwise_cond_br), so they must not be refused.
    for s in ctx.top_level_ops:
        if s.op in ("cf.cond_br", "cf.br"):
            return Violation(
                "unstructured kernel-level control flow "
                f"('{s.op}', produced by a void early `return` mid-kernel) "
                "has no Metal lowering — the branch would be dropped and "
                "the wrong value stored (test_nested_if_else_return). "
                "Structured control flow (scf.if) and value-returning "
                "early returns are supported.", s.op)
    return None


# ── The catalog ────────────────────────────────────────────────────────────

REFUSAL_CASES: List[RefusalCase] = [
    RefusalCase(
        name="unsafe_unsupported_op",
        summary="Op with no Apple hardware and no handler (microscaling matmul)",
        rationale="The op has no Apple GPU hardware and no codegen handler, so "
                  "the result tensor is never computed; any emission is "
                  "silently-wrong. tt.dot_scaled (mxfp microscaling matmul) is "
                  "the canonical case.",
        examples=("test_scaled_dot",),
        trigger_ops=tuple(sorted(UNSAFE_UNSUPPORTED_OPS)),
        check=_check_unsafe_unsupported_op,
    ),
    RefusalCase(
        name="noinline_device_fn_dot",
        summary="tt.dot inside a noinline device function",
        rationale="The device-function lowerer has no cooperative-MMA path, so "
                  "a tt.dot in a @triton.jit(noinline=True) callee returns "
                  "zeros instead of the product.",
        examples=("test_noinline[shared]",),
        trigger_ops=("tt.call", "tt.dot"),
        check=_check_noinline_dot,
    ),
    RefusalCase(
        name="nd_cat_join",
        summary="rank-≥2 tt.cat / tt.join",
        rationale="The generic concat/join handler indexes by the first "
                  "dimension only (1-D); a rank-≥2 operand would be laid out "
                  "incorrectly.",
        examples=("test_cat_nd",),
        trigger_ops=("tt.cat", "tt.join"),
        check=_check_nd_cat_join,
    ),
    RefusalCase(
        name="join_into_dot",
        summary="tt.join result feeding tt.dot",
        rationale="The interleaved layout tt.join produces is not the layout "
                  "the matmul template expects; a join result reaching a tt.dot "
                  "through value-passing ops would compute the wrong product.",
        examples=("test_join_with_mma",),
        trigger_ops=("tt.join", "tt.dot"),
        check=_check_join_into_dot,
    ),
    RefusalCase(
        name="unstructured_kernel_cf",
        summary="top-level cf.cond_br / cf.br (void early return)",
        rationale="A void early `return` mid-kernel lowers to top-level "
                  "cf.cond_br/cf.br; _lower_op_dispatch has no cf-dialect "
                  "handler, so the branch is dropped and the wrong value is "
                  "stored. Structured scf.if and value-returning early returns "
                  "are supported and untouched.",
        examples=("test_nested_if_else_return", "test_constexpr_if_return"),
        trigger_ops=("cf.cond_br", "cf.br"),
        check=_check_unstructured_cf,
    ),
]

# Refusals that live in the template lowerers (they need local lowering state
# — has_M/has_N for the matmul tile, the parsed trans permutation — so they
# can't be a pure predicate over the op graph). Recorded here for catalog
# completeness; the actual guards are at the noted locations.
CONTEXTUAL_REFUSALS: List[ContextualRefusal] = [
    ContextualRefusal(
        name="constexpr_dim_matmul",
        summary="pid-tiled matmul with constexpr-baked M/N",
        rationale="The matmul template needs runtime M/N to derive output "
                  "strides; when they are baked as constexpr it would guess "
                  "_N=BLOCK_N and mis-stride the output (~98% wrong).",
        examples=("test_dot_mulbroadcasted",),
        location="_lowerer_templates.py::_lower_k_loop_dot_inline",
    ),
    ContextualRefusal(
        name="rank3_nonidentity_trans",
        summary="rank-≥3 tt.trans with a non-identity permutation",
        rationale="The generic lowerer only implements 2-D transpose; a "
                  "rank-≥3 non-identity permutation would silently drop the "
                  "permutation.",
        examples=("test_trans_4d",),
        location="generic_lowerer.py::_lower_tt_trans",
    ),
]


# ── Walker + exporters ──────────────────────────────────────────────────────

def check_all(ctx: RefusalContext) -> Optional[Violation]:
    """Return the first :class:`Violation` from the prescan catalog, or None.

    Order matches the historical case order (a→e) so behavior is identical to
    the inline prescan it replaced.
    """
    for case in REFUSAL_CASES:
        violation = case.check(ctx)
        if violation is not None:
            return violation
    return None


def export_json() -> str:
    """Serialize catalog metadata (not the predicates) for the C++ path.

    The C++ MLIR→LLVM path reimplements the predicates over TTGIR but must
    refuse the same named cases with the same rationale/examples — this is the
    shared contract artifact.
    """
    import json
    return json.dumps(
        {
            "prescan": [
                {"name": c.name, "summary": c.summary,
                 "rationale": c.rationale, "examples": list(c.examples),
                 "trigger_ops": list(c.trigger_ops)}
                for c in REFUSAL_CASES
            ],
            "contextual": [
                {"name": c.name, "summary": c.summary,
                 "rationale": c.rationale, "examples": list(c.examples),
                 "location": c.location}
                for c in CONTEXTUAL_REFUSALS
            ],
        },
        indent=2, sort_keys=False,
    )


def doc_markdown() -> str:
    """Render the catalog as the markdown bullet list used in ARCHITECTURE.md.

    Keeping the doc generated from the catalog means the doc cannot drift from
    the code.
    """
    lines = ["The known refusal cases (each was a silent-wrong producer before "
             "the guard):", ""]
    for c in REFUSAL_CASES:
        examples = ", ".join(c.examples)
        lines.append(f"  - **{c.summary}** — {c.rationale} ({examples})")
    lines.append("")
    lines.append("Contextual refusals (located in the template lowerers):")
    lines.append("")
    for c in CONTEXTUAL_REFUSALS:
        examples = ", ".join(c.examples)
        lines.append(f"  - **{c.summary}** — {c.rationale} ({examples})")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - regeneration / export entry point
    # `python -m triton_msl.codegen.refusal_catalog`        -> markdown table
    # `python -m triton_msl.codegen.refusal_catalog --json` -> C++-path export
    import sys
    print(export_json() if "--json" in sys.argv[1:] else doc_markdown())
