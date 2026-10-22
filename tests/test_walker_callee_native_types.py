"""Callee legacy type fields must agree with native MLIR facts, not regex guesses."""

from tests.source_identity import assert_candidate_source

import os
from pathlib import Path

import pytest
import triton_msl
import triton_msl.codegen.mlir_walker as walker_module

from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import IRSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.mlir_walker import walk_ttgir


def _graph(tmp_path, dtype, space=1):
    root = Path(os.environ.get("EXPECTED_SOURCE", Path(__file__).resolve().parents[1])).resolve()
    assert_candidate_source(root, triton_msl, walker_module)
    tensor = f"tensor<32x{dtype}, #blocked>"
    pointer = f"!tt.ptr<{dtype}, {space}>"
    source = (
        """#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
"""
        + f"""
  tt.func public @entry(%x: {tensor}, %p: {pointer}) {{
    %a:2 = tt.call @identity(%x, %p) : ({tensor}, {pointer}) -> ({tensor}, {pointer})
    tt.return
  }}
  tt.func private @identity(%x: {tensor}, %p: {pointer}) -> ({tensor}, {pointer}) {{
    tt.return %x, %p : {tensor}, {pointer}
  }}
}}
"""
    )
    path = tmp_path / "callee.ttgir"
    path.write_text(source)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    context = ir.context()
    parsed = IRSource(str(path), context, backend)
    assert parsed.module.verify()
    return walk_ttgir(parsed.module, backend.parse_options({}))


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "i8", "i64"])
@pytest.mark.parametrize("space", [1, 3])
def test_callee_arguments_keep_native_tensor_and_pointer_types(tmp_path, dtype, space):
    graph = _graph(tmp_path, dtype, space)
    (helper,) = graph.called_funcs
    assert len(helper.args) == 2
    for arg in helper.args:
        native = helper.result_meta[arg.id].type
        assert arg.type_str == native.raw, "callee argument spelling truncated"
        assert arg.elem_type == dtype, "callee argument silently defaulted to another dtype"
        assert arg.is_ptr == (native.kind == "pointer")
    assert helper.result_meta[helper.args[1].id].type.address_space == space


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "i8", "i64"])
def test_callee_results_keep_native_comma_containing_types(tmp_path, dtype):
    graph = _graph(tmp_path, dtype, 3)
    (helper,) = graph.called_funcs
    returned = helper.ops[-1]
    assert returned.op == "tt.return"
    expected = [helper.result_meta[vid].type.raw for vid in returned.operand_ids]
    assert helper.return_types == expected, "callee result tuple was split inside a tensor or pointer type"


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "i8", "i64"])
@pytest.mark.parametrize("space", [1, 3])
def test_entry_pointer_pointee_matches_native_argument(tmp_path, dtype, space):
    graph = _graph(tmp_path, dtype, space)
    pointer = graph.args[1]
    native = graph.result_meta[pointer.id].type
    assert native.address_space == space
    assert pointer.type_str == native.raw
    assert pointer.elem_type == native.pointee.elem == dtype
    assert pointer.is_ptr


@pytest.mark.parametrize("raw", ["vector<4xf32>", "!tt.ptr<garbage>", "garbage"])
def test_unproved_function_type_never_defaults_to_f32(raw):
    from triton_msl.codegen.result_metadata import parse_type_facts
    from triton_msl.errors import MetalNonRecoverableError

    # Direct contract control: these are not necessarily legal Triton function types.
    with pytest.raises(MetalNonRecoverableError, match="unsupported native function type"):
        walker_module._native_argument_fields(parse_type_facts(raw))
