"""Template-level tests for make_simdgroup_matmul_kernel_fast's out_dtype param.
The fp32-out path must stay BYTE-IDENTICAL to the pre-change golden (no regression);
the fp16-out path must declare half* C + the cast epilogue. No GPU needed."""
import os
from triton_msl.codegen._msl_templates import make_simdgroup_matmul_kernel_fast

GOLD = os.path.join(os.path.dirname(__file__), "golden")


def test_fp32_out_byte_identical_to_golden():
    # Default out_dtype (and explicit "fp32") reproduce the pre-change output exactly.
    for in_dt, fname in [("fp16", "simdgroup_matmul_fast_fp16in_fp32out.msl"),
                         ("fp32", "simdgroup_matmul_fast_fp32in_fp32out.msl")]:
        golden = open(os.path.join(GOLD, fname)).read()
        assert make_simdgroup_matmul_kernel_fast(dtype=in_dt, rr=4, rc=4) == golden
        assert make_simdgroup_matmul_kernel_fast(dtype=in_dt, rr=4, rc=4, out_dtype="fp32") == golden


def test_fp16_out_has_half_C_and_cast_epilogue():
    msl = make_simdgroup_matmul_kernel_fast(dtype="fp16", rr=4, rc=4, out_dtype="fp16")
    assert "device half* C [[buffer(2)]]" in msl
    assert "threadgroup float scratch[4 * 64];" in msl
    assert "uint tiisg [[thread_index_in_simdgroup]]" in msl
    assert "half(scratch[sgitg*64u + i])" in msl
    # accumulators stay float (precision); no direct float-store to C remains.
    assert "simdgroup_float8x8 c0_0(0)" in msl
    assert "simdgroup_store(c0_0, C +" not in msl


def test_bad_out_dtype_raises():
    import pytest
    # fp16/bf16/fp32 are the supported in/out dtypes; anything else must raise.
    with pytest.raises(ValueError):
        make_simdgroup_matmul_kernel_fast(dtype="fp16", out_dtype="int8")
    with pytest.raises(ValueError):
        make_simdgroup_matmul_kernel_fast(dtype="int8", out_dtype="fp32")
    # bf16 in/out is now valid (must NOT raise).
    make_simdgroup_matmul_kernel_fast(dtype="bf16", out_dtype="bf16")
