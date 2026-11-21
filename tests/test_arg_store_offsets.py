"""Finding C: blocked arg-reduction stores preserve their pointer and mask DAGs."""

from pathlib import Path
import re

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.codegen.generic_lowerer import GenericLowerer
from tests.test_reduce_return_generic import _lower
from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize("infer", [False, True])
def test_blocked_store_preserves_program_offset_at_lowering(tmp_path, monkeypatch, infer):
    if infer:
        original = GenericLowerer._value_1d_layout_of
        monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
        monkeypatch.setattr(
            GenericLowerer,
            "_value_1d_layout_of",
            lambda self, vid, *a, **kw: None
            if original(self, vid, *a, **kw) == "blocked"
            else original(self, vid, *a, **kw),
        )
    source = (Path(__file__).parent / "fixtures/arg_store_offset.ttgir").read_text()
    msl = _lower(source, tmp_path)
    program_offset = re.search(r"int (r_\d+) = \(int\)\(\(uint\)\(pid\) \* \(uint\)\(c_0\)\);", msl).group(1)
    store = next(line for line in msl.splitlines() if " arg1[" in line)
    assert f"arg1[(int({program_offset}) + int((lid / 16u)))]" in store
    assert "lid % 16u == 0u" in store


def test_blocked_store_rejects_unmapped_loaded_address_at_lowering(tmp_path):
    source = (Path(__file__).parent / "fixtures/arg_store_offset.ttgir").read_text()
    source = source.replace(
        "    tt.store %3, %5",
        """    %loaded_rows = tt.load %3 : tensor<64x!tt.ptr<i32>, #blocked1>
    %badptr = tt.addptr %2, %loaded_rows : tensor<64x!tt.ptr<i32>, #blocked1>, tensor<64xi32, #blocked1>
    tt.store %badptr, %5""",
    )
    with pytest.raises(MetalNonRecoverableError, match="row coordinate cannot be reconstructed"):
        _lower(source, tmp_path)


@pytest.mark.parametrize("where", ["mask", "offset"])
def test_blocked_store_rejects_nonuniform_constant_at_lowering(tmp_path, where):
    source = (Path(__file__).parent / "fixtures/arg_store_offset.ttgir").read_text()
    if where == "mask":
        values = ", ".join(["true", "false"] * 32)
        replacement = f"""    %densemask = arith.constant dense<[{values}]> : tensor<64xi1, #blocked1>
    tt.store %3, %5, %densemask"""
    else:
        values = ", ".join(str(i * 2) for i in range(64))
        replacement = f"""    %denseoffset = arith.constant dense<[{values}]> : tensor<64xi32, #blocked1>
    %denseptr = tt.addptr %2, %denseoffset : tensor<64x!tt.ptr<i32>, #blocked1>, tensor<64xi32, #blocked1>
    tt.store %denseptr, %5"""
    source = source.replace("    tt.store %3, %5", replacement)
    with pytest.raises(MetalNonRecoverableError, match="nonuniform or unparsed tensor constant"):
        _lower(source, tmp_path)


def test_blocked_store_replay_keeps_proof_state_local_at_lowering(tmp_path, monkeypatch):
    original = GenericLowerer._lower_store
    seen = []

    def inspect(self, op):
        # Include nested containers in each representation: a shallow dict copy
        # alone would not protect a nested set from in-place mutation.
        before = {key: repr(value) for key, value in vars(self).items() if isinstance(value, (dict, set))}
        graph_before = repr(self.graph)
        result = original(self, op)
        after = {key: repr(value) for key, value in vars(self).items() if isinstance(value, (dict, set))}
        assert before == after
        assert repr(self.graph) == graph_before
        seen.append(op.id)
        return result

    monkeypatch.setattr(GenericLowerer, "_lower_store", inspect)
    source = (Path(__file__).parent / "fixtures/arg_store_offset.ttgir").read_text()
    _lower(source, tmp_path)
    assert len(seen) == 1


@pytest.mark.parametrize("value,emitted", [("true", "1"), ("false", "0")])
def test_blocked_store_accepts_splat_mask_at_lowering(tmp_path, value, emitted):
    source = (Path(__file__).parent / "fixtures/arg_store_offset.ttgir").read_text()
    source = source.replace(
        "    tt.store %3, %5",
        f"""    %splatmask = arith.constant dense<{value}> : tensor<64xi1, #blocked1>
    tt.store %3, %5, %splatmask""",
    )
    msl = _lower(source, tmp_path)
    store = next(line for line in msl.splitlines() if " arg1[" in line)
    assert f"&& {emitted})" in store


@triton.jit
def _reduce_offset(X, O, off, stride, count, M: tl.constexpr, N: tl.constexpr, KIND: tl.constexpr):
    p = tl.program_id(0)
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    x = tl.load(X + p * M * N + r[:, None] * N + c[None, :])
    if KIND == "argmax":
        value = tl.argmax(x, 1)
    elif KIND == "argmin":
        value = tl.argmin(x, 1)
    elif KIND == "max":
        value = tl.max(x, 1)
    else:
        value = tl.sum(x, 1)
    # Two pointer additions plus a runtime stride, comparison/select and mask.
    row = tl.where(r < count, r, 0)
    tl.store(O + off + p * M * stride + row * stride, value, (r < count) & (r >= 1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("M,N", [(8, 128), (64, 16)])
@pytest.mark.parametrize("kind", ["argmax", "argmin", "max", "sum"])
def test_reduction_program_runtime_strided_masked_output(M, N, kind):
    groups, off, stride, sentinel = 3, 7, 3, -999
    # Integral floats make sum an exact control too; varying extrema across rows.
    x = torch.randint(-100, 100, (groups, M, N), generator=torch.Generator().manual_seed(527)).float()
    count = M - 2
    dtype = torch.int32 if kind.startswith("arg") else torch.float32
    expected = torch.full((off + groups * M * stride + 11,), sentinel, dtype=dtype)
    if kind == "argmax":
        values = x.argmax(2)
    elif kind == "argmin":
        values = x.argmin(2)
    elif kind == "max":
        values = x.max(2).values
    else:
        values = x.sum(2)
    for p in range(groups):
        for row in range(1, count):
            expected[off + p * M * stride + row * stride] = values[p, row]
    output = torch.full_like(expected, sentinel, device="mps")
    _reduce_offset[(groups,)](x.to("mps"), output, off, stride, count, M=M, N=N, KIND=kind)
    torch.mps.synchronize()
    assert torch.equal(output.cpu(), expected), "wrong index/value or modified sentinel region"
