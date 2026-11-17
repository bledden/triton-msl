"""Argument spelling cannot capture a generated matmul local (CPU-only IR emission)."""

import importlib.util
import re
import sys
import tokenize
import io

import pytest
import triton
import triton.language as tl

from tests.test_codegen_admission_semantics import _emit


@triton.jit
def lower(a, b, c, m, n, k):  # contiguous A (M,K), contiguous B (K,N), lower-case extents
    pid = tl.program_id(0)
    cols = tl.cdiv(n, 32)
    rm = (pid // cols) * 32 + tl.arange(0, 32)
    rn = (pid % cols) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for kk in range(0, k, 32):
        x = tl.load(a + rm[:, None] * k + (kk + rk)[None, :])
        y = tl.load(b + (kk + rk)[:, None] * n + rn[None, :])
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * n + rn[None, :], acc, (rm[:, None] < m) & (rn[None, :] < n))


@triton.jit
def lower_bt(a, b, c, m, n, k):  # B stored transposed as (N,K): b[rn*k + kk]
    pid = tl.program_id(0)
    cols = tl.cdiv(n, 32)
    rm = (pid // cols) * 32 + tl.arange(0, 32)
    rn = (pid % cols) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for kk in range(0, k, 32):
        x = tl.load(a + rm[:, None] * k + (kk + rk)[None, :])
        y = tl.load(b + (kk + rk)[:, None] + rn[None, :] * k)
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * n + rn[None, :], acc, (rm[:, None] < m) & (rn[None, :] < n))


def renamed_kernel(tmp_path, names, transposed=False):
    original = lower_bt if transposed else lower
    rename = dict(zip(("a", "b", "c", "m", "n", "k"), names))
    tokens = tokenize.generate_tokens(io.StringIO(original.src).readline)
    source = tokenize.untokenize(
        t._replace(string=rename.get(t.string, t.string)) if t.type == tokenize.NAME else t for t in tokens
    )
    path = tmp_path / ("variant_" + "_".join(names) + ("_bt" if transposed else "") + ".py")
    path.write_text("import triton\nimport triton.language as tl\n@triton.jit\n" + source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return getattr(module, original.__name__)


def emit_names(tmp_path, names, transposed):
    fn = renamed_kernel(tmp_path, names, transposed)
    return _emit(fn, {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(names)}, {})


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "names",
    [
        ("a", "b", "c", "m", "n", "k"),
        ("i", "r", "gr", "row_base", "col_base", "_K"),
        ("_M", "_N", "_K", "_stage_a", "_matrix_width", "_kb"),
        tuple("_mmarg" + str(i) for i in range(6)),
        ("_mmarg1", "_mmarg0", "_mmarg3_buf", "_mmarg5", "_mmarg4", "_mmarg3"),
    ],
)
def test_source_spelling_cannot_change_shader(tmp_path, names, transposed):
    reference = emit_names(tmp_path, ("AA", "BB", "CC", "MM", "NN", "KK"), transposed)
    actual = emit_names(tmp_path, names, transposed)
    assert actual == reference
    for i in range(3):
        assert f"* _mmarg{i} [[buffer({i})]]" in actual
    for i in range(3, 6):
        assert f"int _mmarg{i} = _mmarg{i}_buf[0];" in actual
    if transposed:
        assert "metal::simdgroup_multiply_accumulate" in actual
        assert "(uint)(_mmarg5)" in actual
    else:
        assert "* (uint)_mmarg5 + k" in actual


def test_uppercase_direct_twin_uses_same_hygienic_names(tmp_path):
    source = emit_names(tmp_path, ("A", "B", "C", "M", "N", "K"), False)
    assert "__mmdirect(" in source
    twin = source.split("__mmdirect(", 1)[1]
    assert "long _M = (long)_mmarg3, _N = (long)_mmarg4;" in twin
    assert "ulong _lda = (ulong)((uint)_mmarg5);" in twin
    assert "simdgroup_load(a_frag, _mmarg0 + (ulong)(_sr + 0u) * _lda + k, _lda);" in twin
    assert "(uint)K" not in source


@triton.jit
def _tile(a, b, c):
    rr = tl.arange(0, 32)
    cc = tl.arange(0, 32)
    kk = tl.arange(0, 32)
    aa = tl.load(a + rr[:, None] * 32 + kk[None, :])
    bb = tl.load(b + kk[:, None] * 32 + cc[None, :])
    tl.store(c + rr[:, None] * 32 + cc[None, :], tl.dot(aa, bb))


@triton.jit
def _soft(a, b, c):
    rr = tl.arange(0, 32)
    cc = tl.arange(0, 32)
    kk = tl.arange(0, 32)
    aa = tl.load(a + rr[:, None] * 32 + kk[None, :])
    bb = tl.load(b + kk[:, None] * 32 + cc[None, :])
    dot = tl.dot(aa, bb)
    ee = tl.exp(dot - tl.max(dot, 1)[:, None])
    yy = ee / tl.sum(ee, 1)[:, None]
    tl.store(c + rr[:, None] * 32 + cc[None, :], yy)


@triton.jit
def _epilogue(a, b, c, bias):
    rr = tl.arange(0, 32)
    cc = tl.arange(0, 32)
    kk = tl.arange(0, 32)
    aa = tl.load(a + rr[:, None] * 32 + kk[None, :])
    bb = tl.load(b + kk[:, None] * 32 + cc[None, :])
    dot = tl.dot(aa, bb) + tl.load(bias + cc)[None, :]
    tl.store(c + rr[:, None] * 32 + cc[None, :], tl.maximum(dot, 0.0))


@triton.jit
def _constant(c):
    rr = tl.arange(0, 32)
    cc = tl.arange(0, 32)
    aa = tl.full((32, 32), 1.0, tl.float32)
    bb = tl.full((32, 32), 2.0, tl.float32)
    tl.store(c + rr[:, None] * 32 + cc[None, :], tl.dot(aa, bb))


def renamed_tile(tmp_path, original, names):
    rename = dict(zip(original.arg_names, names))
    tokens = tokenize.generate_tokens(io.StringIO(original.src).readline)
    source = tokenize.untokenize(
        t._replace(string=rename.get(t.string, t.string)) if t.type == tokenize.NAME else t for t in tokens
    )
    path = tmp_path / (original.__name__ + "_" + "_".join(names) + ".py")
    path.write_text("import triton\nimport triton.language as tl\n@triton.jit\n" + source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return getattr(module, original.__name__)


@pytest.mark.parametrize(
    "original,names",
    [
        (_tile, ("c", "r", "i")),
        (_soft, ("c", "i", "row")),
        (_epilogue, ("c", "i", "row", "col")),
        (_constant, ("_i",)),
    ],
)
def test_related_dot_templates_use_hygienic_arguments(tmp_path, original, names):
    original_source = _emit(original, {name: "*fp32" for name in original.arg_names}, {})
    renamed = renamed_tile(tmp_path, original, names)
    renamed_source = _emit(renamed, {name: "*fp32" for name in names}, {})
    assert renamed_source == original_source
    assert "_mmarg0 [[buffer(0)]]" in original_source


def _extractor():
    # The parser is pure Python. Loading the parent mlx package would initialize
    # a native framework, which does not belong in this CPU test.
    import triton_msl
    from pathlib import Path

    path = Path(triton_msl.__file__).parent / "mlx/msl_extractor.py"
    spec = importlib.util.spec_from_file_location("_hygiene_extractor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.extract_msl_for_mlx


def test_mlx_constant_dot_retains_positional_alias_compatibility(tmp_path):
    extract = _extractor()
    fn = renamed_tile(tmp_path, _constant, ("_i",))
    source = _emit(fn, {"_i": "*fp32"}, {})
    result = extract(source, output_arg_indices=[0], expected_args=1, expected_signature=[("_i", "*fp32")])
    assert result.output_names == ["_mmarg0"]
    assert result.arg_positions == [0]
    from triton_msl.errors import MetalNonRecoverableError

    for mutated in (
        source.replace("_mmarg0", "_mmarg1"),
        source.replace("_mmarg0", "arbitrary"),
        source.replace("device float*", "device half*"),
        source.replace("buffer(0)", "buffer(1)"),
    ):
        with pytest.raises(MetalNonRecoverableError):
            extract(mutated, output_arg_indices=[0], expected_args=1, expected_signature=[("_i", "*fp32")])


@triton.jit
def _tile_bt(a, b, c):
    rr = tl.arange(0, 32)
    cc = tl.arange(0, 32)
    kk = tl.arange(0, 32)
    aa = tl.load(a + rr[:, None] * 32 + kk[None, :])
    bb = tl.load(b + cc[None, :] * 32 + kk[:, None])
    tl.store(c + rr[:, None] * 32 + cc[None, :], tl.dot(aa, bb))


@pytest.mark.parametrize("names", [("M", "N", "K"), ("_M", "_N", "_K")])
def test_pointer_names_never_supply_missing_scalar_extents(tmp_path, names):
    reference = _emit(_tile_bt, {name: "*fp32" for name in _tile_bt.arg_names}, {})
    fn = renamed_tile(tmp_path, _tile_bt, names)
    actual = _emit(fn, {name: "*fp32" for name in names}, {})
    assert actual == reference
    assert "long _M = (long)32;" in actual
    assert "long _N = (long)32;" in actual


def test_mlx_rejects_mixed_and_duplicate_binding_schemes():
    from triton_msl.errors import MetalNonRecoverableError

    extract = _extractor()
    template = (
        "kernel void x(device float* FIRST [[buffer(0)]], device float* SECOND [[buffer(1)]]) { SECOND[0] = FIRST[0]; }"
    )
    signature = [("_mmarg1", "*fp32"), ("out", "*fp32")]
    for first, second, accepted in [
        ("_mmarg1", "out", True),
        ("_mmarg0", "_mmarg1", True),
        ("_mmarg1", "_mmarg1", False),
        ("_mmarg0", "out", False),
        ("_mmarg1", "_mmarg0", False),
    ]:
        code = template.replace("FIRST", first).replace("SECOND", second)
        if accepted:
            result = extract(code, output_arg_indices=[1], expected_args=2, expected_signature=signature)
            assert result.arg_positions == [0, 1]
        else:
            with pytest.raises(MetalNonRecoverableError):
                extract(code, output_arg_indices=[1], expected_args=2, expected_signature=signature)
