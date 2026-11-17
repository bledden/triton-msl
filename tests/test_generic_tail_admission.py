"""Invocation-local full-K admission; CPU tensors with explicit device doubles."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import triton  # noqa: F401 -- initialize backend discovery before importing driver

from triton_msl.backend import driver, _cache_contract, _launch_contract, _launch_signature
from triton_msl.backend._generic_launch import generic_2d_geometry
from triton_msl.errors import MetalNonRecoverableError


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("mps")))
    launcher = SimpleNamespace(
        _msl="any checked source; no shader recognition",
        _msl_block_size=256,
        _execution_contract=json.dumps({"source": {"policy": {"USE_CPP": False}}}),
    )
    args = [torch.zeros(32 * 64), torch.zeros(96, 64).t(), torch.zeros(32 * 96), 32, 96, 33]
    desc = [
        "matmul_full_k_storage_v1",
        32,
        [0, 1, 2],
        [["arg", i, "i32"] for i in (3, 4, 5)],
        [["literal", v, "i64"] for v in (64, 1, 1, 64, 96, 1)],
        [["i32", "i32", False]] * 3,
    ]
    meta = [4, 1, 0, 256, None, True, None, None, None, None, desc, None]
    return launcher, args, ["*fp32"] * 3 + ["i32"] * 3, meta, (3, 1, 1)


@pytest.mark.parametrize("k", [1, 31, 32, 33, 63, 64])
def test_validated_full_k_preserves_geometry_metadata_and_original_views(case, k):
    launcher, args, sigs, meta, grid = case
    args[-1] = k
    before = copy.deepcopy(meta)
    views = tuple(args[:3])
    assert driver._matmul_tail_bounds_reason(meta[10], args) is None
    assert generic_2d_geometry(*case) is None
    assert generic_2d_geometry(*case, validated_tail_bounds=meta[10]) == ((768, 1, 1), (256, 1, 1))
    assert meta == before and all(x is y for x, y in zip(views, args))


@pytest.mark.parametrize("evidence", ["missing", "boolean", "equal_copy", "previous_call", "unknown", "batched"])
def test_no_blanket_or_stale_descriptor_admission(case, evidence):
    launcher, args, sigs, meta, grid = case
    proof = {"missing": None, "boolean": True}.get(evidence, copy.deepcopy(meta[10]))
    if evidence == "previous_call":
        proof = meta[10]
        meta[10] = copy.deepcopy(proof)
    if evidence in ("unknown", "batched"):
        meta[10][0] = "unknown_v1" if evidence == "unknown" else "batched_dot_host_bounds_v1"
        proof = meta[10]
    assert generic_2d_geometry(*case, validated_tail_bounds=proof) is None


@pytest.mark.parametrize("container", [tuple, list])
def test_empty_checked_address_descriptor_declines_without_indexing(case, container):
    _, _, _, meta, _ = case
    empty = container()
    meta[10] = empty
    assert generic_2d_geometry(*case, validated_tail_bounds=empty) is None
    assert meta[10] is empty


@pytest.mark.parametrize("mutation", ["transpose_offset", "holes", "dtype", "subclass", "template", "cpp", "grid"])
def test_tail_evidence_does_not_bypass_other_route_checks(case, mutation):
    launcher, args, sigs, meta, grid = case
    if mutation == "transpose_offset":
        args[1] = torch.zeros(96 * 64 + 1)[1:].reshape(96, 64).t()
    elif mutation == "holes":
        args[1] = torch.zeros(96, 128)[:, ::2].t()
    elif mutation == "dtype":
        args[1] = args[1].double()
    elif mutation == "subclass":
        args[1] = torch.nn.Parameter(args[1])
    elif mutation == "template":
        meta[7] = ["specialized"]
    elif mutation == "cpp":
        launcher._execution_contract = json.dumps({"source": {"policy": {"USE_CPP": True}}})
    else:
        grid = (3, 1, 2)
    assert generic_2d_geometry(launcher, args, sigs, meta, grid, validated_tail_bounds=meta[10]) is None


def test_storage_aliases_and_contiguous_offsets_keep_actual_views(case):
    launcher, args, sigs, meta, grid = case
    owner = torch.zeros(16000)
    args[:3] = [owner[8:2056], owner[4096:10240], owner[11000:14072]]
    assert driver._matmul_tail_bounds_reason(meta[10], args) is None
    originals = tuple(args[:3])
    assert generic_2d_geometry(*case, validated_tail_bounds=meta[10]) is not None
    assert all(x is y for x, y in zip(originals, args))


def test_each_call_revalidates_live_tail_storage_before_runtime(case, monkeypatch):
    stub, args, sigs, meta, grid = case
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher.__dict__.update(vars(stub))
    launcher.arg_names = ["A", "B", "C", "M", "N", "K"]
    launcher.signature = dict(zip(launcher.arg_names, sigs))
    launcher.constexpr_indices = set()
    launcher._binding_plan = _launch_signature.make_binding_plan(launcher.arg_names, launcher.signature)
    launcher._packed_contract = _launch_contract._canonical(meta)
    launcher.kernel_name = "no_special_shader_name"
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: stub._execution_contract)
    seen = []

    class RuntimeReached(Exception):
        pass

    def reached():
        seen.append(True)
        raise RuntimeReached

    monkeypatch.setattr(driver, "_get_utils", reached)
    with pytest.raises(RuntimeReached):
        launcher(*grid, None, None, meta, None, None, None, *args)
    args[1] = torch.zeros(96, 33).t()  # same live K, insufficient rounded-K storage
    with pytest.raises(MetalNonRecoverableError, match="mask the K tail"):
        launcher(*grid, None, None, meta, None, None, None, *args)
    assert len(seen) == 1


def test_validated_no_wrap_v2_keeps_the_same_generic_geometry(case):
    _, args, _, meta, grid = case
    legacy = meta[10]
    desc = [
        "matmul_grid_storage_v2",
        *legacy[1:3],
        [None, None, legacy[3][2]],
        *legacy[4:],
        [32, 32, ["flat_n", 32], [True, True], [None, legacy[3][1]], [None, None], True, [legacy[3][0], legacy[3][1]]],
    ]
    meta[10] = desc
    assert driver._matmul_tail_bounds_reason(desc, args, grid=grid) is None
    assert generic_2d_geometry(*case, validated_tail_bounds=desc) == ((768, 1, 1), (256, 1, 1))
