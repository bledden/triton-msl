"""Bounded cooperative scalar matmul must retain masks, widths and fallback."""

import pytest

from tests.test_codegen_admission_semantics import _emit
from tests.test_full_k_direct_gpu import full_k


def _source(dtype="fp32", bm=32, bn=32, bk=32, width="i32"):
    names = ("A", "B", "C", "M", "N", "K", "SAM", "SAK", "SBK", "SBN", "SCM", "SCN")
    sig = {name: "*" + dtype if name in ("A", "B", "C") else width for name in names}
    const = dict(SAK=1, SBK=1, SCN=1, BM=bm, BN=bn, BK=bk, MASK_MN=True)
    sig.update({name: "constexpr" for name in const})
    return _emit(full_k, sig, const)


def test_staging_retains_full_k_contract_and_scalar_fallback():
    # The tile-aligned direct twin follows the staged entry and has its own
    # tests; the staging contract below is the entry's alone.
    source, twin = _source().split("kernel void full_k__mmdirect(")
    assert "_stage_" not in twin and "threadgroup_barrier" not in twin
    assert "_stage_a[1056]" in source and "_stage_b[1056]" in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 5
    assert "_K_source" in source and "_K <= 2147483647L" in source
    assert "row_base <= 4294967264u" in source and "col_base <= 4294967264u" in source
    assert "(long)row_base < _M" in source and "(long)col_base < _N" in source
    last_barrier = source.rindex("threadgroup_barrier(mem_flags::mem_threadgroup)")
    assert last_barrier < source.index(" = _acc0;")
    # The unstaged loop holds each thread's cells until the threadgroup has loaded.
    assert source.count("threadgroup_barrier(mem_flags::mem_none)") == 1
    loaded = source.index("threadgroup_barrier(mem_flags::mem_none)")
    assert source.index("float _cell[4];") < source.index("_cell[_e / 256u] = _sum;") < loaded
    assert loaded < source.index("] = (float)_cell[_e / 256u];")
    assert "for (uint k = 0u; k < _K; k++)" in source
    assert "if (m >= _M || n >= _N) continue;" in source
    assert "as_type<int>((uint)m * (uint)(_mmarg6))" in source
    assert "_kb + _kk < _K" in source
    assert "simdgroup_float8x8" in source
    assert "_matrix_width == 32u && _K % 32L == 0L" in source
    assert "[[threads_per_simdgroup]]" in source
    assert "[[simdgroup_index_in_threadgroup]]" in source
    assert source.count("#if defined(__HAVE_SIMDGROUP_MATRIX__)") == 2
    assert "_stage_b[(_t % 32u) * 33u + _t / 32u]" in source
    assert "simdgroup_store" in source


@pytest.mark.parametrize(
    "dtype,bm,bn,bk", [("fp16", 32, 32, 32), ("fp32", 16, 32, 32), ("fp32", 32, 16, 32), ("fp32", 32, 32, 16)]
)
def test_other_shapes_and_types_keep_scalar_path(dtype, bm, bn, bk):
    source = _source(dtype, bm, bn, bk)
    assert "_stage_a" not in source
    assert "float _sum" in source


import triton
import triton.language as tl


@triton.jit
def _collision(A, B, C, M, N, K, _kb):
    m = tl.arange(0, 32)
    n = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for k in range(0, K, 32):
        a = tl.load(A + m[:, None] * _kb + (k + rk)[None, :])
        b = tl.load(B + (k + rk)[:, None] + n[None, :] * K)
        acc += tl.dot(a, b)
    tl.store(C + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def test_new_staging_local_never_shadows_a_source_stride():
    source = _emit(_collision, dict(A="*fp32", B="*fp32", C="*fp32", M="i32", N="i32", K="i32", _kb="i32"), {})
    assert "_stage_a" in source
    assert "for (uint _kb" in source
    assert "(uint)(_mmarg6)" in source
    assert "float _sum" in source


@pytest.mark.parametrize("name", ["_matrix_width", "_matrix_sg", "_matrix_row", "_matrix_b"])
def test_matrix_local_collision_uses_hygienic_stride(name, tmp_path):
    # Source renaming creates a real TTGIR argument, not merely a string check.
    fn = _renamed_stride(name, tmp_path)
    sig = dict(A="*fp32", B="*fp32", C="*fp32", M="i32", N="i32", K="i32")
    sig[name] = "i32"
    source = _emit(fn, sig, {})
    assert "simdgroup_float8x8" in source
    assert "_stage_a" in source
    assert "(uint)(_mmarg6)" in source
    assert "float _sum" in source


def test_fragment_ownership_and_scratch_bounds():
    slots = []
    for sg in range(8):
        for fragment in range(2):
            slots.extend(((sg // 4) * 16 + fragment * 8 + r, (sg % 4) * 8 + c) for r in range(8) for c in range(8))
    assert len(slots) == len(set(slots)) == 1024
    assert set(slots) == {(r, c) for r in range(32) for c in range(32)}
    assert max(r * 33 + c for r, c in slots) < 1056


@pytest.mark.parametrize(
    "name", ["simdgroup_load", "simdgroup_store", "simdgroup_float8x8", "simdgroup_multiply_accumulate"]
)
def test_source_names_do_not_shadow_metal_matrix_api(name, tmp_path):
    fn = _renamed_stride(name, tmp_path)
    sig = dict(A="*fp32", B="*fp32", C="*fp32", M="i32", N="i32", K="i32")
    sig[name] = "i32"
    source = _emit(fn, sig, {})
    assert "metal::simdgroup_load(" in source
    assert "metal::simdgroup_store(" in source
    assert "metal::simdgroup_multiply_accumulate(" in source
    assert "metal::simdgroup_float8x8" in source
    assert "(uint)(_mmarg6)" in source


def _renamed_stride(name, tmp_path):
    # A real Python signature keeps Triton's source and parameter metadata aligned.
    import importlib.util
    import inspect
    import sys

    path = tmp_path / "renamed_kernel.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n" + inspect.getsource(_collision.fn).replace("_kb", name)
    )
    spec = importlib.util.spec_from_file_location("renamed_kernel_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._collision
