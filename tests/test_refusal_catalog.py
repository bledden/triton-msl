"""Unit tests for the refusal catalog (WS0/C5).

These test the catalog *mechanism* directly with lightweight fake ops — no
full compile pipeline — so they pin the single-source-of-truth contract:
every case fires on its trigger, none misfires, metadata is complete, and the
export/doc generators are stable. End-to-end refusal behavior (via emit_msl)
is covered separately in test_generic_lowerer.py.
"""
import json
from dataclasses import dataclass, field

import pytest

from triton_msl.codegen import refusal_catalog as rc


# ── Lightweight fake op + context ───────────────────────────────────────────

@dataclass
class FakeOp:
    op: str
    id: int = 0
    operand_ids: list = field(default_factory=list)
    region_ops: list = field(default_factory=list)
    else_ops: list = field(default_factory=list)
    ops: list = field(default_factory=list)  # for called_funcs


def make_ctx(top_level=None, all_ops=None, called_funcs=None,
             type_strs=None):
    """Build a RefusalContext from fake ops.

    type_strs: dict {ssa_id: type_str} used by the cat/join shape check.
    """
    top_level = top_level or []
    all_ops = all_ops if all_ops is not None else list(top_level)
    type_strs = type_strs or {}

    def find_op_type_str(ssa_id):
        return type_strs.get(ssa_id)

    def extract_shape(type_str):
        # minimal stand-in: type_strs maps to a shape tuple directly when the
        # test wants a rank; here type_str is already the shape tuple or None
        return type_str if isinstance(type_str, (tuple, list)) else ()

    return rc.RefusalContext(
        all_ops=all_ops,
        top_level_ops=top_level,
        called_funcs=called_funcs or [],
        find_op_type_str=find_op_type_str,
        extract_shape=extract_shape,
    )


# ── Catalog integrity ───────────────────────────────────────────────────────

def test_every_case_has_complete_metadata():
    assert rc.REFUSAL_CASES, "catalog must not be empty"
    for c in rc.REFUSAL_CASES:
        assert c.name and c.name.replace("_", "").isalnum()
        assert c.summary.strip()
        assert c.rationale.strip()
        assert c.examples, f"{c.name} must list at least one example test"
        assert c.trigger_ops, f"{c.name} must list trigger ops"
        assert callable(c.check)


def test_case_names_are_unique():
    names = [c.name for c in rc.REFUSAL_CASES] + \
            [c.name for c in rc.CONTEXTUAL_REFUSALS]
    assert len(names) == len(set(names)), "refusal case names must be unique"


def test_clean_graph_is_not_refused():
    # A plain elementwise kernel: load, add, store — nothing refusable.
    ops = [FakeOp("tt.load", id=1), FakeOp("arith.addf", id=2),
           FakeOp("tt.store", id=3)]
    assert rc.check_all(make_ctx(top_level=ops, all_ops=ops)) is None


# ── Each case fires on its trigger ──────────────────────────────────────────

def test_unsafe_unsupported_op_fires():
    ops = [FakeOp("tt.dot_scaled", id=1)]
    v = rc.check_all(make_ctx(top_level=ops, all_ops=ops))
    assert v is not None and v.op_name == "tt.dot_scaled"
    assert "microscaling" in v.message


def test_noinline_dot_fires():
    callee = FakeOp("tt.call", id=0, ops=[FakeOp("tt.dot", id=99)])
    ctx = make_ctx(top_level=[FakeOp("tt.call", id=1)],
                   all_ops=[FakeOp("tt.call", id=1)],
                   called_funcs=[callee])
    v = rc.check_all(ctx)
    assert v is not None and v.op_name == "tt.call"
    assert "noinline" in v.message


def test_nd_cat_join_fires_on_rank2():
    cat = FakeOp("tt.cat", id=5, operand_ids=[10])
    ctx = make_ctx(top_level=[cat], all_ops=[cat],
                   type_strs={10: (32, 16)})  # rank-2 shape
    v = rc.check_all(ctx)
    assert v is not None and v.op_name == "tt.cat"
    assert "rank-2" in v.message


def test_nd_cat_join_allows_rank1():
    cat = FakeOp("tt.cat", id=5, operand_ids=[10])
    ctx = make_ctx(top_level=[cat], all_ops=[cat],
                   type_strs={10: (64,)})  # rank-1 — supported
    assert rc.check_all(ctx) is None


def test_join_into_dot_fires_through_passthrough():
    # join(id=1) -> reshape(id=2) -> dot(id=3) operand chain
    join = FakeOp("tt.join", id=1, operand_ids=[])
    reshape = FakeOp("tt.reshape", id=2, operand_ids=[1])
    dot = FakeOp("tt.dot", id=3, operand_ids=[2])
    ops = [join, reshape, dot]
    v = rc.check_all(make_ctx(top_level=ops, all_ops=ops))
    assert v is not None and v.op_name == "tt.dot"
    assert "join" in v.message


def test_dot_without_join_is_allowed():
    load = FakeOp("tt.load", id=1)
    dot = FakeOp("tt.dot", id=3, operand_ids=[1])
    ops = [load, dot]
    assert rc.check_all(make_ctx(top_level=ops, all_ops=ops)) is None


def test_unstructured_cf_fires_at_top_level():
    ops = [FakeOp("cf.cond_br", id=1)]
    v = rc.check_all(make_ctx(top_level=ops, all_ops=ops))
    assert v is not None and v.op_name == "cf.cond_br"
    assert "control flow" in v.message


def test_cf_inside_region_is_not_refused():
    # cf.cond_br only inside a map_elementwise body (region_ops) — handled
    # elsewhere, must NOT be refused. It appears in all_ops but not top_level.
    inner = FakeOp("cf.cond_br", id=2)
    mapop = FakeOp("tt.map_elementwise", id=1, region_ops=[inner])
    ctx = make_ctx(top_level=[mapop], all_ops=[mapop, inner])
    assert rc.check_all(ctx) is None


# ── Exporters ────────────────────────────────────────────────────────────────

def test_export_json_is_valid_and_complete():
    data = json.loads(rc.export_json())
    assert set(data) == {"prescan", "contextual"}
    assert len(data["prescan"]) == len(rc.REFUSAL_CASES)
    assert len(data["contextual"]) == len(rc.CONTEXTUAL_REFUSALS)
    for entry in data["prescan"]:
        assert entry["name"] and entry["rationale"] and entry["examples"]


def test_doc_markdown_mentions_every_case():
    md = rc.doc_markdown()
    for c in rc.REFUSAL_CASES:
        assert c.summary in md
    for c in rc.CONTEXTUAL_REFUSALS:
        assert c.summary in md


def test_doc_markdown_is_deterministic():
    assert rc.doc_markdown() == rc.doc_markdown()
