"""Exact TriFast narrow-dtype delta reduction for the proved D=32 layout."""

from tests.source_identity import assert_candidate_source

import hashlib
import sys
from pathlib import Path

import pytest
import torch
import triton_msl

sys.path.insert(0, "tests")
from test_fa_bwd_routing import _bwd_q, _build_lowerer

from triton_msl.errors import MetalNonRecoverableError


_WITNESSES = {
    "fp16": (
        [
            1.953125,
            -0.019989013671875,
            0.50390625,
            64.8125,
            0.0011539459228515625,
            0.00754547119140625,
            2.763671875,
            -30.421875,
            -0.42333984375,
            21.46875,
            0.42431640625,
            -0.3603515625,
            0.040618896484375,
            -144.5,
            -60.4375,
            -87.3125,
            -1.3515625,
            0.044830322265625,
            -0.0005955696105957031,
            -13.1640625,
            -0.002254486083984375,
            0.97607421875,
            15.8984375,
            43.65625,
            -0.1627197265625,
            1048.0,
            -30.3125,
            -73.9375,
            -126.25,
            -0.0308380126953125,
            -0.001983642578125,
            -0.00382232666015625,
        ],
        632.5,
    ),
    "bf16": (
        [
            0.0015869140625,
            -624.0,
            -168.0,
            0.00762939453125,
            -0.010498046875,
            -0.130859375,
            -43.5,
            -0.0155029296875,
            -2.09375,
            0.302734375,
            -0.0022735595703125,
            -35.25,
            -0.06884765625,
            -892.0,
            -3.933906555175781e-06,
            -0.00122833251953125,
            -0.00016117095947265625,
            -0.03271484375,
            -19.25,
            3.328125,
            -288.0,
            1576.0,
            5.0625,
            22.625,
            0.01019287109375,
            -1.1328125,
            0.000324249267578125,
            716.0,
            0.53125,
            -148.0,
            0.0301513671875,
            0.0010528564453125,
        ],
        100.0,
    ),
}
_TORCH_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}


@pytest.fixture(autouse=True, scope="module")
def _assert_package_identity():
    assert_candidate_source(Path(__file__).resolve().parents[1], triton_msl)


def _runtime_layout_oracle(values, dtype):
    """D=32 runtime layout: eight-value binary trees, then lane XOR 2,1."""
    values = torch.as_tensor(values, dtype=dtype)
    lanes = []
    for lane in range(4):
        current = list(values[lane * 8 : (lane + 1) * 8])
        while len(current) > 1:
            current = [(current[i].float() + current[i + 1].float()).to(dtype) for i in range(0, len(current), 2)]
        lanes.append(current[0])
    current = torch.stack(lanes)
    lane_ids = torch.arange(4)
    for mask in (2, 1):
        previous = current
        current = (previous.float() + previous[lane_ids ^ mask].float()).to(dtype)
    return current[0]


def _one_value_per_lane_oracle(values, dtype):
    """The disproved feasibility layout, retained as an anti-oracle."""
    current = torch.tensor(values, dtype=dtype)
    lanes = torch.arange(32)
    for mask in (16, 8, 4, 2, 1):
        previous = current
        current = (previous.float() + previous[lanes ^ mask].float()).to(dtype)
    return current[0]


@pytest.mark.parametrize("name", ["fp16", "bf16"])
def test_witness_distinguishes_runtime_tree_from_false_oracles(name):
    values, expected = _WITNESSES[name]
    dtype = _TORCH_DTYPE[name]
    got = _runtime_layout_oracle(values, dtype)
    sequential = torch.tensor(0.0, dtype=dtype)
    for value in values:
        sequential = (sequential.float() + torch.tensor(value)).to(dtype)
    widened = torch.tensor(values).sum().to(dtype)
    assert got.item() == expected
    assert got.item() != sequential.item()
    assert got.item() != widened.item()
    assert got.item() != _one_value_per_lane_oracle(values, dtype).item()


def _signature(dtype, dim=32):
    constexprs = {
        name: (dim if name == "DIM" else 32 if name in ("BLOCK_J", "BLOCK_K") else 64)
        for name in [_bwd_q.arg_names[i] for i in _bwd_q.constexprs]
    }
    signature = {}
    for name in _bwd_q.arg_names:
        if name in constexprs:
            continue
        if name.endswith("_ptr"):
            signature[name] = "*u8" if "mask" in name else ("*fp32" if name.startswith("l_") else f"*{dtype}")
        elif name in ("sm_scale", "neg_inf"):
            signature[name] = "fp32"
        else:
            signature[name] = "i32"
    return signature, constexprs


@pytest.mark.parametrize("name", ["fp16", "bf16"])
def test_d64_narrow_delta_remains_a_precise_refusal(name):
    signature, constexprs = _signature(name, dim=64)
    with pytest.raises(
        MetalNonRecoverableError,
        match="accumulates it in|narrow delta reduction",
    ):
        _build_lowerer(_bwd_q, signature, constexprs).lower()


def _launch_q(fn, name, n):
    dtype = _TORCH_DTYPE[name]
    values, expected = _WITNESSES[name]
    dim = 32
    shape4 = (1, 1, n, dim)
    q = torch.zeros(shape4, dtype=dtype, device="mps")
    # K=1,V=0 makes dP=0 and dQ=-delta for every output element. This
    # independently proves that the exact delta is consumed downstream; a
    # correct delta store paired with a stale fp32 internal value cannot pass.
    k = torch.ones_like(q)
    v = torch.zeros_like(q)
    o = torch.ones_like(q)
    do = torch.tensor(values, dtype=dtype, device="mps").reshape(1, 1, 1, dim).expand(shape4).contiguous()
    bias = torch.zeros((1, n, n), dtype=dtype, device="mps")
    lse = torch.full((1, 1, n), float(torch.tensor(n).log()), device="mps")
    mask = torch.zeros((1, 1, n), dtype=torch.uint8, device="mps")
    delta = torch.full((1, 1, n), -99, dtype=dtype, device="mps")
    dq = torch.full_like(q, -99)
    st = lambda tensor: tuple(tensor.stride())
    fn.device_caches.clear()
    compiled = fn[((n + 31) // 32, 1, 1)](
        delta,
        *st(delta),
        q,
        *st(q),
        k,
        *st(k),
        v,
        *st(v),
        bias,
        *st(bias),
        lse,
        *st(lse),
        mask,
        *st(mask),
        o,
        *st(o),
        do,
        *st(do),
        dq,
        *st(dq),
        1.0,
        -1.0e9,
        n,
        1,
        dim,
        n,
        32,
        32,
    )
    torch.mps.synchronize()
    return compiled, delta, dq, expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize(
    ("name", "n"),
    [("fp16", 32), ("bf16", 32), ("fp16", 33), ("fp16", 1)],
)
def test_unmodified_trifast_q_replays_narrow_runtime_delta(name, n, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    compiled, delta, dq, expected = _launch_q(_bwd_q, name, n)
    dtype = _TORCH_DTYPE[name]
    msl = compiled.asm["msl"]
    shuffle_lines = [line for line in msl.splitlines() if "delta_peer" in line and "simd_shuffle_xor" in line]
    assert len(shuffle_lines) == 2
    assert "eight contiguous D values per lane" in msl
    assert "simd_shuffle_xor(dl, 2u)" in msl
    assert "simd_shuffle_xor(dl, 1u)" in msl
    assert torch.equal(delta.cpu(), torch.full((1, 1, n), expected, dtype=dtype))
    assert torch.equal(dq.cpu(), torch.full_like(dq.cpu(), -expected))
    assert hashlib.sha256(msl.encode()).hexdigest()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_singleton_backward_does_not_exempt_a_tighter_output_mask(monkeypatch, tmp_path):
    from test_fa_bwd_rounding_replay import _variant

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    variant = _variant(
        "_bwd_q",
        [
            (
                "tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None])",
                "tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None] & (d_idxs[None, :] < 16))",
            )
        ],
        "bwd_q_singleton_tighter_store",
        tmp_path,
    )
    with pytest.raises(MetalNonRecoverableError, match="stored under|output store mask"):
        _launch_q(variant, "fp16", 1)
