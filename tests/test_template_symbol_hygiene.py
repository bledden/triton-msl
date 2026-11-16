"""Template locals cannot capture source parameters (real IR, CPU-only)."""

import importlib.util
import re
import sys

import pytest
import triton
import triton.language as tl

from tests.test_matmul_symbol_hygiene import lower, renamed_tile
from tests.test_template_scalar_abi import _lower, _shape, _simple
from tests.test_codegen_admission_semantics import _transpose3


@triton.jit
def _soft(x, out, n):
    row = tl.program_id(0)
    ix = tl.arange(0, 64)
    value = tl.load(x + row * n + ix, ix < n, -float("inf"))
    ee = tl.exp(value - tl.max(value, 0))
    tl.store(out + row * n + ix, ee / tl.sum(ee, 0), ix < n)


@triton.jit
def _norm(x, out, n):
    row = tl.program_id(0)
    ix = tl.arange(0, 64)
    value = tl.load(x + row * n + ix, ix < n, 0.0)
    centered = tl.where(ix < n, value - tl.sum(value, 0) / n, 0.0)
    var = tl.sum(centered * centered, 0) / n
    tl.store(out + row * n + ix, centered * tl.rsqrt(var + 1e-5), ix < n)


@triton.jit
def _sort(x, out):
    ii = tl.arange(0, 4)
    jj = tl.arange(0, 16)
    value = tl.load(x + ii[:, None] * 16 + jj[None, :])
    tl.store(out + ii[:, None] * 16 + jj[None, :], tl.sort(value))


@triton.jit
def _reshape(x, out):
    ii = tl.arange(0, 4)
    jj = tl.arange(0, 16)
    value = tl.load(x + ii[:, None] * 16 + jj[None, :])
    result = value.reshape(4, 2, 4, 2).permute(1, 2, 3, 0).reshape(64)
    tl.store(out + tl.arange(0, 64), result)


@triton.jit
def _chained(x, out):
    ix = tl.arange(0, 1024).reshape(4, 4, 4, 4, 4)
    value = tl.load(x + ix).permute(2, 1, 0, 3, 4)
    result = tl.sum(tl.sum(tl.sum(value, 3), 1), 0)
    tl.store(out + tl.arange(0, 16).reshape(4, 4), result)


@triton.jit
def _batch(a, b, out, ab, am, ak, bb, bk, bn, cb, cm, cn):
    ii = tl.arange(0, 2)
    mm = tl.program_id(0) * 32 + tl.arange(0, 32)
    nn = tl.program_id(1) * 32 + tl.arange(0, 32)
    xx = tl.arange(0, 32)
    av = tl.load(a + ii[:, None, None] * ab + mm[None, :, None] * am + xx[None, None, :] * ak)
    bv = tl.load(b + ii[:, None, None] * bb + xx[None, :, None] * bk + nn[None, None, :] * bn)
    tl.store(out + ii[:, None, None] * cb + mm[None, :, None] * cm + nn[None, None, :] * cn, tl.dot(av, bv))


CASES = [
    (_shape, dict(X="*fp32", Y="*fp32", S="fp32"), dict(MODE=0), "_lower_flip_template", ("_src", "_e", "lid")),
    (_shape, dict(X="*fp32", Y="*fp32", S="fp32"), dict(MODE=1), "_lower_3d_reduce_template", ("_input", "_r", "pid")),
    (
        _shape,
        dict(X="*fp32", Y="*i32", S="fp32"),
        dict(MODE=2),
        "_lower_3d_argminmax_template",
        ("_result_idx", "_r", "tid"),
    ),
    (_soft, dict(x="*fp32", out="*fp32", n="i32"), {}, "_lower_softmax_template", ("row_cache", "i", "row_start")),
    (_norm, dict(x="*fp32", out="*fp32", n="i32"), {}, "_lower_layer_norm_template", ("row_cache", "i", "row_start")),
    (_sort, dict(x="*fp32", out="*fp32"), {}, "_lower_row_wise_sort_template", ("v", "_i")),
    (_reshape, dict(x="*fp32", out="*fp32"), {}, "_lower_transpose_via_reshape_template", ("_row", "k")),
    (_chained, dict(x="*i32", out="*i32"), {}, "_lower_permute_chained_reduce_template", ("_acc", "e")),
    (
        _transpose3,
        dict(In="*fp32", Out="*fp32"),
        dict(PID_AXIS=0, INPUT_PID=False, OUTPUT_PID=False),
        "_lower_nd_trans_template",
        ("_row", "tid"),
    ),
    (
        _batch,
        {n: "*fp32" if i < 3 else "i32" for i, n in enumerate(_batch.arg_names)},
        {},
        "_lower_batched_dot_template",
        ("batch", "q", "c", "kk", "i", "r", "row_base", "col_base", "tiitg", "tg_A", "pid3", "acc0"),
    ),
]


def emit_witness(fn, sig, constants, maker):
    obj = _lower(fn, sig, constants)
    graph_names = [a.name for a in obj.graph.args]
    hits = []
    original = getattr(obj, maker)

    def observe(*args, **kwargs):
        hits.append(maker)
        return original(*args, **kwargs)

    setattr(obj, maker, observe)
    source = obj.lower()
    assert hits == [maker], "must emit the named template through production admission"
    assert [a.name for a in obj.graph.args] == graph_names
    return source, obj


@pytest.mark.parametrize("fn,sig,constants,maker,collisions", CASES, ids=[c[3] for c in CASES])
@pytest.mark.parametrize("spelling", ["locals", "positional", "permuted"])
def test_template_source_names_do_not_capture_locals(tmp_path, fn, sig, constants, maker, collisions, spelling):
    names = tuple(sig)
    renamed_names = (
        collisions
        if spelling == "locals"
        else tuple(
            f"_mmarg{i}" if spelling == "positional" else f"_mmarg{len(names) - 1 - i}" for i in range(len(names))
        )
    )
    all_names = tuple(renamed_names) + tuple(n for n in fn.arg_names if n not in sig)
    renamed = renamed_tile(tmp_path, fn, all_names)
    variant_sig = dict(zip(renamed_names, sig.values()))
    original_source, original_obj = emit_witness(fn, sig, constants, maker)
    variant_source, variant_obj = emit_witness(renamed, variant_sig, constants, maker)
    assert variant_source == original_source
    assert "_mmarg0 [[buffer(0)]]" in original_source
    assert getattr(original_obj, "_batched_dot_bounds", None) == getattr(variant_obj, "_batched_dot_bounds", None)


@pytest.mark.parametrize(
    "names",
    [("A", "B", "C", "M", "N", "K"), ("lid", "tileA", "tileB", "m", "n", "k"), tuple(f"_mmarg{i}" for i in range(6))],
)
def test_legacy_prebuilt_pointer_binding_uses_template_owned_identifiers(tmp_path, names):
    def emit(fn, argument_names):
        obj = _lower(fn, {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(argument_names)}, {})
        before = [a.name for a in obj.graph.args]
        source = obj._lower_dot_simple_template([32], [a for a in obj.graph.args if a.is_ptr], "fp32")
        assert [a.name for a in obj.graph.args] == before
        return source

    reference = emit(lower, lower.arg_names)
    renamed = renamed_tile(tmp_path, lower, names)
    assert emit(renamed, names) == reference
    for i in range(3):
        assert f"_mmarg{i} [[buffer({i})]]" in reference
    # Uppercase template identifiers never survive in pointer arithmetic either.
    assert re.search(r"\b[ABC]\b", reference) is None


@pytest.mark.parametrize("fn,sig,constants,maker,collisions", CASES, ids=[c[3] for c in CASES])
def test_template_mlx_parser_preserves_supported_or_refused_abi(fn, sig, constants, maker, collisions):
    from tests.test_matmul_symbol_hygiene import _extractor

    source, obj = emit_witness(fn, sig, constants, maker)
    output_index = 2 if fn is _batch else 1
    if fn in (_shape, _batch):
        # The existing MLX signature parser does not represent device scalar
        # buffers. Keep its refusal rather than weaken the ABI proof here.
        from triton_msl.errors import MetalNonRecoverableError

        with pytest.raises(MetalNonRecoverableError, match="complete source binding map"):
            _extractor()(
                source, output_arg_indices=[output_index], expected_args=len(sig), expected_signature=list(sig.items())
            )
        return
    result = _extractor()(
        source, output_arg_indices=[output_index], expected_args=len(sig), expected_signature=list(sig.items())
    )
    assert result.output_names == [f"_mmarg{output_index}"]
    assert result.arg_positions == list(range(len(sig)))


def test_legacy_prebuilt_mlx_parser_retains_template_scalar_abi():
    from tests.test_matmul_symbol_hygiene import _extractor

    # The legacy builder owns its fixed uint M/N/K ABI. This test preserves that
    # existing ABI and its parser support; it does not qualify a new source route.
    sig = {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(lower.arg_names)}
    obj = _lower(lower, sig, {})
    expected_sig = [(name, "*fp32" if i < 3 else "u32") for i, name in enumerate(lower.arg_names)]
    source = obj._lower_dot_simple_template([32], [a for a in obj.graph.args if a.is_ptr], "fp32")
    result = _extractor()(source, output_arg_indices=[2], expected_args=6, expected_signature=expected_sig)
    assert result.arg_positions == list(range(6))
    assert result.output_names == ["_mmarg2"]


@pytest.mark.parametrize("entry", ["A", "B", "C", "M", "N", "K"])
def test_legacy_template_binding_preserves_source_entry_name(tmp_path, entry):
    path = tmp_path / ("entry_" + entry + ".py")
    source = lower.src.replace("def lower(", f"def {entry}(", 1)
    path.write_text("import triton\nimport triton.language as tl\n@triton.jit\n" + source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, entry)
    sig = {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(fn.arg_names)}
    obj = _lower(fn, sig, {})
    code = obj._lower_dot_simple_template([32], [a for a in obj.graph.args if a.is_ptr], "fp32")
    assert f"kernel void {entry}(" in code
    assert "_mmarg0 [[buffer(0)]]" in code
