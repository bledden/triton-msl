"""WS1 Phase C.1: fp16 matmul must be GENUINE.

The "fp16" simdgroup matmul template used to upcast halves to float and run
`simdgroup_float8x8` MMA — i.e. it ran fp32 regardless of dtype (the harness
measured fp16==fp32==7.00 TFLOP/s). These tests assert on the GENERATED MSL so
"fp16" can never silently regress to fp32 again. The accumulator stays float
(fp32 accumulation for precision); only the INPUT fragments become half.
"""
import re

from triton_metal.codegen._msl_templates import make_simdgroup_matmul_kernel


def test_fp16_matmul_uses_half_input_fragments():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    assert "simdgroup_half8x8" in msl, \
        "fp16 MMA must use simdgroup_half8x8 INPUT fragments, not float8x8"


def test_fp16_matmul_does_not_upcast_inputs_before_mma():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    # staging buffers must be half (no float staging), and no float(A[/float(B[
    assert "threadgroup half" in msl, "fp16 must stage inputs as half"
    assert not re.search(r"float\(\s*A\[", msl), "no float() upcast of A"
    assert not re.search(r"float\(\s*B\[", msl), "no float() upcast of B"


def test_fp16_accumulator_stays_float_for_precision():
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    assert "simdgroup_float8x8 acc" in msl, \
        "accumulator must stay simdgroup_float8x8 (fp32 accumulation)"


def test_fp32_matmul_unchanged_still_float_fragments():
    # the fp32 path must remain float fragments (regression guard).
    msl = make_simdgroup_matmul_kernel(dtype="fp32")
    assert "simdgroup_float8x8 a_frag" in msl
    assert "simdgroup_half8x8" not in msl
