"""Source-pattern regressions from the final code review, through production MSL admission.

CPU tests compile real Triton IR, never a fabricated detection graph. GPU tests
are separately named and use the same source; no refusal earns computation credit.
"""

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.errors import MetalNonRecoverableError


def _emit(fn, signature, constants):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(fn=fn, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options)


@triton.jit
def _fused(A, B, Bias, C, M, N, K, MODE: tl.constexpr, PID_AXIS: tl.constexpr, LOAD_MASK: tl.constexpr):
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    if PID_AXIS == 0:
        rm = rm + tl.program_id(0) * 32
    elif PID_AXIS == 1:
        rn = rn + tl.program_id(1) * 32
    elif PID_AXIS == 2:
        rm = rm + tl.program_id(2) * 32
    if LOAD_MASK:
        a = tl.load(A + rm[:, None] * K + rk[None, :], rm[:, None] < M, 0)
    else:
        a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    if MODE == 1:
        z = tl.dot(a, b, tl.load(Bias + rm[:, None] * N + rn[None, :]))
    else:
        z = tl.dot(a, b)
    m = tl.max(z, 1)
    if MODE == 2 or MODE == 4:
        e = tl.exp(m[:, None] - z)
    else:
        e = tl.exp(z - m[:, None])
    s = tl.sum(e, 1)
    if MODE == 3 or MODE == 4:
        y = s[:, None] / e
    else:
        y = e / s[:, None]
    if PID_AXIS >= 0:
        tl.store(C + rm[:, None] * N + rn[None, :], y, (rm[:, None] < M) & (rn[None, :] < N))
    else:
        tl.store(C + rm[:, None] * N + rn[None, :], y)


def _fused_msl(mode=0, axis=-1, load_mask=False):
    signature = {name: "*fp32" for name in ("A", "B", "Bias", "C")}
    signature.update({name: "i32" for name in ("M", "N", "K")})
    constants = dict(MODE=mode, PID_AXIS=axis, LOAD_MASK=load_mask)
    signature.update({name: "constexpr" for name in constants})
    return _emit(_fused, signature, constants)


def test_fused_softmax_canonical_still_emits():
    msl = _fused_msl()
    assert "simdgroup_multiply_accumulate" in msl
    assert "exp(" in msl


@pytest.mark.parametrize(
    "mode,reason",
    [(1, "zero dot accumulator"), (2, "reversed operands"), (3, "reversed operands"), (4, "reversed operands")],
)
def test_fused_softmax_does_not_replace_source_arithmetic(mode, reason):
    with pytest.raises(MetalNonRecoverableError, match=reason):
        _fused_msl(mode=mode)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_fused_softmax_mask_does_not_certify_unemitted_pid_mapping(axis):
    with pytest.raises(MetalNonRecoverableError, match="does not reproduce program-id"):
        _fused_msl(axis=axis)


def test_masked_load_generic_recovery_is_not_removed():
    msl = _fused_msl(axis=0, load_mask=True)
    # This was already a computing generic route, unlike the unmasked-load
    # counterexample. The template-specific guard must not remove it.
    body = msl[msl.index("{", msl.index("kernel void")) :]
    assert "pid" in body
    assert "M" in body


@triton.jit
def _product(a, b):
    return a * b


@triton.jit
def _difference(a, b):
    return a - b


@triton.jit
def _reduce3(X, Z, AXIS: tl.constexpr, KIND: tl.constexpr):
    i = tl.arange(0, 4)
    j = tl.arange(0, 8)
    k = tl.arange(0, 16)
    x = tl.load(X + i[:, None, None] * 128 + j[None, :, None] * 16 + k[None, None, :])
    if KIND == 0:
        y = tl.sum(x, AXIS)
    elif KIND == 1:
        y = tl.max(x, AXIS)
    elif KIND == 2:
        y = tl.min(x, AXIS)
    elif KIND == 3:
        y = tl.reduce(x, AXIS, _product)
    elif KIND == 4:
        y = tl.reduce(x, AXIS, _difference)
    elif KIND == 5:
        y = tl.argmin(x, AXIS)
    else:
        y = tl.argmax(x, AXIS)
    if AXIS == 0:
        tl.store(Z + j[:, None] * 16 + k[None, :], y)
    elif AXIS == 1:
        tl.store(Z + i[:, None] * 16 + k[None, :], y)
    else:
        tl.store(Z + i[:, None] * 8 + j[None, :], y)


def _reduce_msl(kind, axis, dtype):
    return _emit(
        _reduce3,
        dict(X="*" + dtype, Z="*i32" if kind >= 5 else "*" + dtype, AXIS="constexpr", KIND="constexpr"),
        dict(AXIS=axis, KIND=kind),
    )


@pytest.mark.parametrize("kind", [3, 4])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reduce3_never_defaults_a_different_combine_to_sum(kind, axis):
    with pytest.raises(MetalNonRecoverableError, match="substitute a sum"):
        _reduce_msl(kind, axis, "fp32")


@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "i32"])
def test_reduce3_supported_combine_still_emits(kind, axis, dtype):
    msl = _reduce_msl(kind, axis, dtype)
    assert "_result[_r] = acc" in msl
    assert ("acc + val" if kind == 0 else "max(acc, val)" if kind == 1 else "min(acc, val)") in msl


@pytest.mark.parametrize("kind,comparison", [(5, "<"), (6, ">")])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "i32"])
def test_reduce3_argminmax_uses_proven_value_direction(kind, comparison, axis, dtype):
    msl = _reduce_msl(kind, axis, dtype)
    assert f"if (val {comparison} best_v" in msl
    if dtype == "i32":
        assert ("best_v = INT_MAX" if kind == 5 else "best_v = INT_MIN") in msl


@triton.jit
def _transpose3(In, Out, PID_AXIS: tl.constexpr, INPUT_PID: tl.constexpr, OUTPUT_PID: tl.constexpr):
    i = tl.arange(0, 4)
    j = tl.arange(0, 8)
    k = tl.arange(0, 16)
    p = tl.program_id(PID_AXIS)
    off_in = p * 512 if INPUT_PID else 0
    off_out = p * 512 if OUTPUT_PID else 0
    x = tl.load(In + off_in + i[:, None, None] * 128 + j[None, :, None] * 16 + k[None, None, :])
    y = tl.trans(x, (1, 0, 2))
    tl.store(Out + off_out + j[:, None, None] * 64 + i[None, :, None] * 16 + k[None, None, :], y)


def _trans_msl(axis, input_pid, output_pid):
    constants = dict(PID_AXIS=axis, INPUT_PID=input_pid, OUTPUT_PID=output_pid)
    signature = dict(In="*fp32", Out="*fp32", **{name: "constexpr" for name in constants})
    return _emit(_transpose3, signature, constants)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("input_pid,output_pid", [(True, True), (True, False), (False, True)])
def test_nd_transpose_does_not_drop_either_program_address(axis, input_pid, output_pid):
    with pytest.raises(MetalNonRecoverableError, match="N-D transpose.*program-id addressing"):
        _trans_msl(axis, input_pid, output_pid)


def test_nd_transpose_without_live_pid_still_emits():
    msl = _trans_msl(0, False, False)
    assert "_mmarg1[k] = _mmarg0[" in msl
