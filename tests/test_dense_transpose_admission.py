"""CPU controls: device labels are doubles; never create an MPS tensor."""

import json
from types import SimpleNamespace
import pytest
import torch
from triton_msl.backend._generic_launch import generic_2d_geometry


@pytest.fixture
def case(monkeypatch):
    launcher = SimpleNamespace(
        _msl="checked generic source",
        _msl_block_size=256,
        _execution_contract=json.dumps({"source": {"policy": {"USE_CPP": False}}}),
    )
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("mps")))
    return (
        launcher,
        [torch.zeros(64, 32), torch.zeros(96, 32).t(), torch.zeros(64, 96)],
        ["*fp32"] * 3,
        [4, 1, 0, 256, None, True, None, None, None, None, None, None],
        (6, 1, 1),
    )


@pytest.mark.parametrize("dtype,sig", [(torch.float32, "*fp32"), (torch.float16, "*fp16")])
def test_full_storage_dense_transpose_uses_original_view(case, dtype, sig):
    launcher, args, sigs, meta, grid = case
    args[1] = torch.arange(96 * 32, dtype=dtype).reshape(96, 32).t()
    sigs[1] = sig
    original = args[1]
    storage = original.untyped_storage()
    before = original.clone()
    assert generic_2d_geometry(*case) == ((1536, 1, 1), (256, 1, 1))
    assert args[1] is original and args[1].untyped_storage().data_ptr() == storage.data_ptr()
    assert torch.equal(original, before)
    # The same backing storage may be supplied at multiple source pointer slots.
    args[0] = original
    sigs[0] = sig
    assert generic_2d_geometry(*case) == ((1536, 1, 1), (256, 1, 1))
    assert args[0] is args[1]


@pytest.mark.parametrize(
    "kind",
    ["offset", "partial_storage", "row_holes", "column_holes", "broadcast", "overlap", "rank3", "int32", "int64"],
)
def test_other_noncontiguous_storage_forms_stay_host(case, kind):
    launcher, args, sigs, meta, grid = case
    if kind == "offset":
        value = torch.zeros(96 * 32 + 1)[1:].view(96, 32).t()
    elif kind == "partial_storage":
        value = torch.zeros(97, 32)[:96].t()
    elif kind == "row_holes":
        value = torch.zeros(96, 64)[:, ::2].t()
    elif kind == "column_holes":
        value = torch.zeros(192, 32)[::2].t()
    elif kind == "broadcast":
        value = torch.zeros(1, 96).expand(32, 96)
    elif kind == "overlap":
        value = torch.zeros(96 * 32).as_strided((32, 96), (1, 1))
    elif kind == "rank3":
        value = torch.zeros(2, 96, 32).transpose(1, 2)
    elif kind == "int32":
        value = torch.zeros(96, 32, dtype=torch.int32).t()
        sigs[1] = "*i32"
    elif kind == "int64":
        value = torch.zeros(96, 32, dtype=torch.int64).t()
        sigs[1] = "*i64"
    args[1] = value
    assert not value.is_contiguous()
    assert generic_2d_geometry(*case) is None


def test_original_contiguous_offset_policy_unchanged(case):
    launcher, args, sigs, meta, grid = case
    args[1] = torch.zeros(4096)[16 : 16 + 3072].view(32, 96)
    assert args[1].storage_offset() == 16 and args[1].is_contiguous()
    assert generic_2d_geometry(*case) == ((1536, 1, 1), (256, 1, 1))
