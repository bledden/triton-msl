"""CPU-only source generation and closed public-dispatch boundary.

Fixtures are exact reporter f32 TTGIR from the red installed-emitter control.
No library is compiled or loaded by these tests. Module contexts remain owned.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from triton._C.libtriton import ir
from triton_msl.backend.compiler import MetalOptions
from triton_msl.backend import _cache_contract, _launch_contract, driver
from triton_msl.codegen import msl_emitter, mlir_walker
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen._source_program import entry, precision, semantics, renderer, uniform
from triton_msl.errors import MetalNonRecoverableError

FIXTURES = Path(__file__).parent / "fixtures/source_program"


def parse(path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(path), ctx)
    mod._retained_context = ctx
    return mod


def emitted(branch="masked"):
    mod = parse(FIXTURES / (branch + ".ttgir"))
    metadata = {"num_warps": 4, "num_ctas": 1, "execution_contract": "controlled"}
    source = msl_emitter.emit_msl(mod, metadata, MetalOptions())
    return mod, source, metadata


@pytest.mark.parametrize("branch", ["padded", "masked"])
def test_real_entry_emits_exact_arithmetic_and_source_bound_descriptor(branch):
    mod, source, metadata = emitted(branch)
    program, receipt = precision.capture(mod, ir, mlir_walker.walk_ttgir, MetalOptions(), semantics, GenericLowerer)
    prototype, _ = renderer.Row(program).generate()
    assert source == prototype.replace(
        "kernel void trifast_expression_" + branch + "(", "kernel void " + metadata["name"] + "(", 1
    )
    assert emitted(branch)[1:] == (source, metadata)
    kind, d = metadata["batched_dot_bounds"]
    assert kind == entry.KIND and d["bindings"] == 11 and d["outputs"] == [0, 1]
    assert d["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert d["entry"] == metadata["name"] and d["threads"] == [32, 1, 1]
    assert {r["root"] for r in d["addresses"]} == set(range(7))
    assert d["native_ieee_enum"] == int(ir.INPUT_PRECISION.IEEE)
    assert metadata["flash_attention"] is None and metadata["argument_abi"] is None
    assert metadata["output_arg_indices"] == [0, 1] and metadata["needs_2d_grid"] is True
    _launch_contract.seal_launch_metadata(metadata)
    packed = json.loads(_launch_contract.validate_launch_metadata(metadata))
    assert len(packed) == 12 and packed[10][0] == entry.KIND


@pytest.mark.parametrize("branch", ["padded", "masked"])
def test_names_are_not_recognition_and_changed_symbols_are_new_source(branch, tmp_path):
    text = (FIXTURES / (branch + ".ttgir")).read_text()
    assert "@_fwd(" in text
    path = tmp_path / "renamed.ttgir"
    path.write_text(text.replace("@_fwd(", "@unrelated_public_name(", 1))
    mod = parse(path)
    metadata = {}
    source = msl_emitter.emit_msl(mod, metadata, MetalOptions())
    assert metadata["name"] == "unrelated_public_name"
    assert "kernel void unrelated_public_name(" in source
    assert (
        metadata["batched_dot_bounds"][1]["source_sha256"]
        != emitted(branch)[2]["batched_dot_bounds"][1]["source_sha256"]
    )


@pytest.mark.parametrize("branch", ["padded", "masked"])
@pytest.mark.parametrize("policy", ["tf32", "tf32x3"])
@pytest.mark.parametrize("dot_index", [0, 1])
def test_native_non_ieee_never_uses_legacy(branch, policy, dot_index, tmp_path, monkeypatch):
    text = (FIXTURES / (branch + ".ttgir")).read_text()
    line = [s for s in text.splitlines() if "= tt.dot " in s][dot_index]
    path = tmp_path / "precision.ttgir"
    path.write_text(text.replace(line, line.replace(" : ", ", inputPrecision = " + policy + " : ", 1), 1))
    mod = parse(path)
    monkeypatch.setenv("TRITON_MSL_LEGACY", "1")
    monkeypatch.setattr(msl_emitter, "_legacy_fallback", lambda *a: pytest.fail("legacy reached"))
    with pytest.raises(MetalNonRecoverableError, match="native precision is not IEEE"):
        msl_emitter.emit_msl(mod, {}, MetalOptions())


@pytest.mark.parametrize("damage", ["address", "bias", "recurrence", "store", "effect", "type", "dominance"])
def test_near_miss_proof_failure_is_terminal(damage, monkeypatch):
    mod = parse(FIXTURES / "masked.ttgir")
    graph = mlir_walker.walk_ttgir(mod, MetalOptions())
    ops = list(uniform.walk(graph.ops))
    if damage == "address":
        op = next(o for o in ops if o.op == "tt.addptr")
        op.operand_ids[1] = graph.args[10].id
    elif damage == "bias":
        op = next(o for o in ops if o.op == "arith.mulf")
        op.op = "arith.addf"
    elif damage == "recurrence":
        op = next(o for o in ops if o.op == "scf.yield")
        op.operand_ids[0], op.operand_ids[1] = op.operand_ids[1], op.operand_ids[0]
    elif damage == "store":
        op = next(o for o in ops if o.op == "tt.store")
        op.operand_ids.pop()
    elif damage == "effect":
        next(o for o in ops if o.op == "tt.return").op = "tt.print"
    elif damage == "type":
        graph.args[7].elem_type = "f16"
    else:
        graph.ops[0], graph.ops[-1] = graph.ops[-1], graph.ops[0]
    assert entry.candidate(graph)
    monkeypatch.setenv("TRITON_MSL_LEGACY", "1")
    monkeypatch.setattr(msl_emitter, "_legacy_fallback", lambda *a: pytest.fail("legacy reached"))
    monkeypatch.setattr(mlir_walker, "walk_ttgir", lambda *a: graph)
    with pytest.raises(MetalNonRecoverableError, match="source-program proof refused"):
        msl_emitter.emit_msl(mod, {}, MetalOptions())


def test_nonmatching_graph_is_not_taken_by_new_entry():
    assert entry.emit(None, SimpleNamespace(args=[]), None, "unused", None) is None
    args = [SimpleNamespace(is_ptr=True)] * 7 + [SimpleNamespace(is_ptr=False)] * 4
    assert entry.emit(None, SimpleNamespace(args=args, ops=[]), None, "unused", None) is None


@pytest.mark.parametrize("branch", ["padded", "masked"])
def test_foreign_native_graph_refuses(branch):
    mod = parse(FIXTURES / (branch + ".ttgir"))
    other = parse(FIXTURES / (branch + ".ttgir"))
    graph = mlir_walker.walk_ttgir(other, MetalOptions())
    with pytest.raises(MetalNonRecoverableError, match="native graph correspondence"):
        entry.emit(mod, graph, MetalOptions(), "name", GenericLowerer)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_runtime_stride_reference_keeps_its_real_emitter_route(contiguous, tiled, tmp_path, monkeypatch):
    import ast
    import importlib.util
    import sys
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend

    source = (Path(__file__).parent / "test_fa_biased_routing.py").read_text()
    (fn,) = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_biased_tri_fa"]
    extracted = "import triton\nimport triton.language as tl\n@triton.jit\n" + "".join(
        source.splitlines(keepends=True)[fn.lineno - 1 : fn.end_lineno]
    )
    path = tmp_path / "reference.py"
    path.write_text(extracted)
    spec = importlib.util.spec_from_file_location("source_program_runtime_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module._biased_tri_fa
    constants = {"DIM": 64, "BLOCK_J": 32, "BLOCK_K": 32}
    if contiguous:
        constants.update({n: 1 for n in ("o_sd", "q_sd", "k_sd", "v_sd")})
    signature = {
        n: (
            "*fp32"
            if n == "lse_ptr"
            else "*u8"
            if n == "mask_ptr"
            else "*fp16"
            if n.endswith("_ptr")
            else "fp32"
            if n in ("sm_scale", "neg_inf")
            else "i32"
        )
        for n in kernel.arg_names
        if n not in constants
    }
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    opts = MetalOptions(experimental_tiled_k_layout=tiled)
    ctx = ir.context()
    ir.load_dialects(ctx)
    src = ASTSource(kernel, signature=signature, constexprs=constants)
    mod = src.make_ir(target, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), ctx)
    md = {}
    mod = backend.make_ttir(mod, md, opts)
    mod = backend.make_ttgir(mod, md, opts)
    assert not entry.candidate(mlir_walker.walk_ttgir(mod, opts))
    parent_metadata = copy.deepcopy(md)
    with monkeypatch.context() as m:
        m.setattr(entry, "emit", lambda *a: None)
        baseline = msl_emitter.emit_msl(mod, parent_metadata, opts)
    current_metadata = copy.deepcopy(md)
    current = msl_emitter.emit_msl(mod, current_metadata, opts)
    assert current == baseline and current_metadata == parent_metadata
    assert current_metadata["batched_dot_bounds"] is None


@pytest.mark.parametrize("compile_shader", ["0", "1"])
@pytest.mark.parametrize(
    "damage",
    ["none", "source_sha256", "entry", "addresses", "threads", "scalar_roles", "full_tile_domain", "runtime_admission"],
)
def test_public_call_refuses_before_runtime_with_real_packed_validator(compile_shader, damage, monkeypatch):
    _, _, metadata = emitted()
    _launch_contract.seal_launch_metadata(metadata)
    expected = _launch_contract.validate_launch_metadata(metadata)
    packed = json.loads(expected)
    if damage != "none":
        packed[10][1][damage] = None
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher._execution_contract = "controlled"
    launcher._packed_contract = expected
    launcher.arg_names = ["p" + str(i) for i in range(7)] + ["scale", "sentinel", "N", "H"]
    launcher.signature = dict(zip(launcher.arg_names, ["*fp32"] * 6 + ["*i1", "fp32", "fp32", "i32", "i32"]))
    events = []
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda stamp: events.append("identity"))
    monkeypatch.setattr(driver, "_get_utils", lambda: pytest.fail("runtime acquired"))
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: pytest.fail("framework acquired"))
    pointer = SimpleNamespace(data_ptr=lambda: 0)
    with pytest.raises(MetalNonRecoverableError):
        launcher(
            1,
            32,
            1,
            None,
            None,
            packed,
            None,
            lambda _: events.append("hook"),
            None,
            *([pointer] * 7),
            1.0,
            float("-inf"),
            32,
            1,
        )
    assert events == (["identity", "hook"] if damage == "none" else ["identity"])
