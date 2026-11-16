"""Column-bias address proof at the real compiler admission boundary."""

import pytest
import triton
import triton.language as tl

from tests.test_codegen_admission_semantics import _emit
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _bias_dot(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STEP: tl.constexpr,
    OFFSET: tl.constexpr,
    MASKED: tl.constexpr,
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    if MASKED:
        bias = tl.load(Bias + OFFSET + rn * STEP, rn < N // 2, -3.0)
    else:
        bias = tl.load(Bias + OFFSET + rn * STEP)
    acc = tl.broadcast_to(bias[None, :], (M, N))
    dot = tl.dot(a, b, acc)
    tl.store(Out + rm[:, None] * N + rn[None, :], tl.maximum(dot, 0.0))


def _source(m=32, step=1, offset=0, masked=False):
    signature = {name: "*fp32" for name in ("A", "B", "Bias", "Out")}
    constants = dict(M=m, N=32, K=32, STEP=step, OFFSET=offset, MASKED=masked)
    signature.update({name: "constexpr" for name in constants})
    return _emit(_bias_dot, signature, constants)


@pytest.mark.parametrize("m", [32, 64])
def test_contiguous_column_bias_keeps_template(m):
    source = _source(m=m)
    assert "_mmarg2[col]" in source  # Bias is argument 2; the template names it positionally


@pytest.mark.parametrize("step,offset,masked", [(2, 0, False), (1, 3, False), (1, 0, True), (2, 3, True)])
def test_noncanonical_bias_uses_generic_source_replay(step, offset, masked):
    source = _source(step=step, offset=offset, masked=masked)
    assert "_mmarg2[col]" not in source and "_mmarg" not in source
    assert "kernel void" in source


@pytest.mark.parametrize("step,offset,masked", [(2, 0, False), (1, 3, False), (1, 0, True), (2, 3, True)])
def test_wide_noncanonical_bias_does_not_lose_source_address(step, offset, masked):
    with pytest.raises(MetalNonRecoverableError, match="column-bias"):
        _source(m=64, step=step, offset=offset, masked=masked)
