"""Each tt.trans consumer must use that operation's proved permutation."""

from tests.source_identity import assert_candidate_source

from pathlib import Path

import pytest
import triton
import triton.language as tl
import triton_msl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tests.test_dot_batched_rank3 import _flat_lowerer
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.codegen.mlir_walker import _ModuleTextIndex


@triton.jit
def _two_orders(A, B, C, N: tl.constexpr):
    offsets = tl.arange(0, N * N * N)
    av = tl.load(A + offsets).reshape((N, N, N))
    bv = tl.load(B + offsets).reshape((N, N, N))
    av = tl.trans(av, (0, 2, 1))
    bv = tl.trans(bv, (1, 0, 2))
    tl.store(C + offsets, (av + bv).reshape((N * N * N,)))


def _two_order_lowerer(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    assert_candidate_source(root, triton_msl)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    source = ASTSource(
        _two_orders,
        {"A": "*fp32", "B": "*fp32", "C": "*fp32", "N": "constexpr"},
        {"N": 32},
    )
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


def test_two_same_rank_transposes_keep_distinct_operation_owned_orders(tmp_path, monkeypatch):
    lowerer = _two_order_lowerer(tmp_path, monkeypatch)
    trans = [op for op in lowerer.graph.ops if op.op == "tt.trans"]
    assert [op.attrs["order"] for op in trans] == [[0, 2, 1], [1, 0, 2]]
    assert [lowerer._parse_trans_order(op, 3) for op in trans] == [[0, 2, 1], [1, 0, 2]]


def test_unknown_transpose_order_cannot_enter_batched_dot():
    lowerer = _flat_lowerer(rank=3, ta=True, tb=True)
    trans = [op for op in lowerer.graph.ops if op.op == "tt.trans"]
    assert len(trans) == 2 and lowerer._detect_batched_dot() is not None
    trans[1].attrs["order"] = None
    assert lowerer._parse_trans_order(trans[1], 3) is None
    assert lowerer._detect_batched_dot() is None


@pytest.mark.parametrize("damage", [[0, 2], [0, 0, 1], [0, 2, 3]])
def test_malformed_operation_owned_order_refuses(damage):
    lowerer = _flat_lowerer(rank=3, ta=True, tb=True)
    trans = next(op for op in lowerer.graph.ops if op.op == "tt.trans")
    trans.attrs["order"] = damage
    assert lowerer._parse_trans_order(trans, 3) is None


def test_canonical_batched_transpose_positive_is_preserved():
    lowerer = _flat_lowerer(rank=3, ta=True, tb=True)
    plan = lowerer._detect_batched_dot()
    assert plan is not None
    assert "round3_batched_dot" in lowerer._lower_batched_dot_template(plan)


def test_quoted_commented_nested_and_private_repeated_ssa_names_are_scoped():
    source = r"""#b = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#loc = loc("%fake = tt.trans %x {order = array<i32: 2, 1, 0>}")
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @entry(%x: tensor<2x2x32xf32, #b>) {
    %t = tt.trans %x {order = array<i32: 1, 0, 2>} : tensor<2x2x32xf32, #b> -> tensor<2x2x32xf32, #b> loc(#loc)
    scf.if %pred {
      %t = tt.trans %x {order = array<i32: 2, 1, 0>} : tensor<2x2x32xf32, #b> -> tensor<32x2x2xf32, #b>
    }
    tt.return
  }
  tt.func private @helper(%x: tensor<2x2x32xf32, #b>) -> tensor<2x32x2xf32, #b> {
    // %t = tt.trans %x {order = array<i32: 2, 0, 1>} : tensor<2x2x32xf32, #b> -> tensor<32x2x2xf32, #b>
    %t = tt.trans %x {order = array<i32: 0, 2, 1>} : tensor<2x2x32xf32, #b> -> tensor<2x32x2xf32, #b>
    tt.return %t : tensor<2x32x2xf32, #b>
  }
}
"""
    records = _ModuleTextIndex(source).trans_orders
    assert [(r["function"], r["result"], r["order"]) for r in records] == [
        ("entry", "t", (1, 0, 2)),
        ("entry", "t", (2, 1, 0)),
        ("helper", "t", (0, 2, 1)),
    ]
