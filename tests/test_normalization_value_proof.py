"""Whole normalization semantics, not the presence of normalization-like op names.

The five packet189 witnesses plus sum-input, exponent-base and variance/divisor
siblings. Compile-time pins bypass executable caches; supported source variants
must compute their actual math. Canonical controls must use the template.
"""

import importlib
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fa_bwd_routing import _build_lowerer


@triton.jit
def _soft(x_ptr, o_ptr, N, BLOCK: tl.constexpr, MODE: tl.constexpr):
    if MODE == 8:
        r = tl.program_id(1)
    else:
        r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    if MODE == 6:
        x = tl.load(x_ptr + r * N + c, c < N, other=0.0)
    elif MODE == 7:
        x = tl.load(x_ptr + r * N + c + 1, c < N, other=-float("inf"))
    else:
        x = tl.load(x_ptr + r * N + c, c < N, other=-float("inf"))
    z = x - tl.max(x, 0)
    if MODE == 3:
        z = z.to(tl.float16).to(tl.float32)
    if MODE == 5:
        e = tl.exp2(z)
    else:
        e = tl.exp(z)
    if MODE == 4:
        s = tl.sum(e * 2.0, 0)
    else:
        s = tl.sum(e, 0)
    if MODE == 1:
        s = s * 2.0
    if MODE == 2:
        s = s.to(tl.float16).to(tl.float32)
    if MODE == 9:
        # The effect witness launches eight rows; its counter is disjoint from
        # every ordinary output store, so the expected count is deterministic.
        tl.atomic_add(o_ptr + 8 * N, 1.0, sem="relaxed")
    if MODE == 10:
        tl.store(o_ptr + r * N + c, e / s, c < N - 1)
    else:
        tl.store(o_ptr + r * N + c, e / s, c < N)


@triton.jit
def _norm(x_ptr, o_ptr, N, BLOCK: tl.constexpr, MODE: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + r * N + c, c < N, other=0.0)
    mean = tl.sum(x, 0) / N
    if MODE == 1:
        mean = mean * 2.0
    if MODE == 2:
        mean = mean.to(tl.float16).to(tl.float32)
    xc = tl.where(c < N, x - mean, 0.0)
    sq = xc * xc
    if MODE == 3:
        sq = sq + 1.0
    if MODE == 4:
        var = tl.sum(sq, 0) / (N + 1)
    else:
        var = tl.sum(sq, 0) / N
    if MODE == 5:
        y = tl.rsqrt(var + 1.0e-5) * xc
    else:
        y = xc * tl.rsqrt(var + 1.0e-5)
    tl.store(o_ptr + r * N + c, y, c < N)


CASES = [(family, mode) for family in ("soft", "norm") for mode in range(6)]


@pytest.fixture
def routes(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    # ALWAYS_COMPILE bypasses Triton's disk cache, not JITFunction.run's
    # earlier in-memory lookup. Every route-spy row must really compile,
    # including the cancellation witness when canonical MODE=0 ran first.
    for fn in (_soft, _norm):
        fn.device_caches.clear()
    cls = importlib.import_module("triton_msl.codegen.generic_lowerer").GenericLowerer
    hits = []
    for name in ("_lower_softmax_template", "_lower_layer_norm_template"):
        real = getattr(cls, name)

        def spy(self, *args, _name=name, _real=real, **kwargs):
            hits.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, spy)
    return hits


def _positive(family, mode):
    return mode == 0 or (family == "norm" and mode == 5)


@pytest.mark.parametrize("family,mode", CASES)
def test_normalization_proof_at_lowering(routes, family, mode):
    fn = _soft if family == "soft" else _norm
    text = _build_lowerer(fn, {"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32"}, {"BLOCK": 64, "MODE": mode}).lower()
    assert "kernel void" in text
    assert bool(routes) == _positive(family, mode), routes


def _reference(x, family, mode):
    if family == "soft":
        z = x - x.max(-1, keepdim=True).values
        if mode == 3:
            z = z.half().float()
        e = z.exp2() if mode == 5 else z.exp()
        s = e.sum(-1, keepdim=True)
        if mode in (1, 4):
            s = s * 2
        if mode == 2:
            s = s.half().float()
        return e / s
    mean = x.mean(-1, keepdim=True)
    if mode == 1:
        mean = mean * 2
    if mode == 2:
        mean = mean.half().float()
    xc = x - mean
    total = xc.square().sum(-1, keepdim=True)
    if mode == 3:
        total = total + 64  # source adds one to ALL BLOCK lanes, including padding
    var = total / (x.shape[-1] + (mode == 4))
    return xc * torch.rsqrt(var + 1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize("family,mode", CASES)
@pytest.mark.parametrize("n", [47, 64])
def test_normalization_source_math_must_compute(routes, family, mode, n):
    torch.manual_seed(189)
    cpu = torch.randn(8, n) * 3 + 2
    x = cpu.to("mps")
    out = torch.full_like(x, float("nan"))
    fn = _soft if family == "soft" else _norm
    fn[(8,)](x, out, n, 64, mode)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), _reference(cpu, family, mode), rtol=1e-6, atol=1e-6)
    assert len(routes) == int(_positive(family, mode))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_layernorm_centered_variance_not_difference_of_moments(routes):
    # Exactly representable mean and deviations: no oracle ambiguity from a
    # different summation order. E[x*x] - E[x]*E[x] loses the variance entirely.
    cpu = torch.tensor([999.875, 1000.125]).repeat(4, 32)
    x = cpu.to("mps")
    out = torch.full_like(x, float("nan"))
    _norm[(4,)](x, out, 64, 64, 0)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), _reference(cpu, "norm", 0), rtol=1e-6, atol=1e-6)
    assert len(routes) == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize("family", ["soft", "norm"])
def test_runtime_row_length_does_not_extend_source_tile(routes, family):
    fn = _soft if family == "soft" else _norm
    text = _build_lowerer(fn, {"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32"}, {"BLOCK": 64, "MODE": 0}).lower()
    # Do NOT launch the pre-fix out-of-bounds shared-memory kernel. This is a
    # lowering-boundary bite, followed by a hardware preservation pin after fix.
    assert "int _norm_cols = clamp(int(_mmarg2), 0, 64);" in text
    assert "i < (uint)_mmarg2" not in text and "int n_v = _mmarg2 / 4" not in text
    torch.manual_seed(193)
    cpu = torch.randn(4, 80)
    x = cpu.to("mps")
    out = torch.full_like(x, -12345)
    routes.clear()
    fn[(4,)](x, out, 80, 64, 0)
    torch.mps.synchronize()
    prefix = cpu[:, :64]
    if family == "soft":
        expected = torch.softmax(prefix, -1)
    else:
        xc = prefix - prefix.sum(-1, keepdim=True) / 80
        expected = xc * torch.rsqrt(xc.square().sum(-1, keepdim=True) / 80 + 1e-5)
    torch.testing.assert_close(out.cpu()[:, :64], expected, rtol=1e-6, atol=1e-6)
    assert bool((out.cpu()[:, 64:] == -12345).all())
    assert len(routes) == 1


@pytest.mark.parametrize("mode", [6, 7, 8, 9, 10])
def test_normalization_address_fill_and_effect_near_misses(routes, mode):
    text = _build_lowerer(_soft, {"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32"}, {"BLOCK": 64, "MODE": mode}).lower()
    assert "kernel void" in text
    assert not routes


@pytest.mark.parametrize("family", ["soft", "norm"])
def test_normalization_type_proof_uses_native_value_metadata_not_legacy_strings(family):
    from triton_msl.codegen._normalization_proof import prove_normalization

    fn = _soft if family == "soft" else _norm
    lowerer = _build_lowerer(fn, {"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32"}, {"BLOCK": 64, "MODE": 0})
    info = lowerer._detect_softmax() if family == "soft" else lowerer._detect_layer_norm()
    assert info is not None

    def poison(ops):
        for op in ops:
            op.type_str = "unusable legacy type"
            op.elem_type = "unusable legacy element"
            poison(op.region_ops or [])
            poison(op.else_ops or [])

    poison(lowerer.graph.ops)
    for arg in lowerer.graph.args:
        arg.type_str = "unusable legacy type"
        arg.elem_type = "unusable legacy element"
    # Exercise this consumer directly: another detector/emitter's legacy reads
    # remain migration work and must not masquerade as coverage of this slice.
    prove_normalization(lowerer.graph, info, "softmax" if family == "soft" else "layer_norm")


@pytest.mark.parametrize("damage", ["missing", "wrong_owner", "index_width", "projection"])
def test_normalization_native_type_proof_rejects_unproved_facts(damage):
    from triton_msl.codegen._normalization_proof import prove_normalization

    lowerer = _build_lowerer(_soft, {"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32"}, {"BLOCK": 64, "MODE": 0})
    info = lowerer._detect_softmax()
    assert info is not None
    load = next(op for op in lowerer.graph.ops if op.op == "tt.load")
    vid = next(arg.id for arg in lowerer.graph.args if arg.name == "N") if damage == "index_width" else load.id
    meta = lowerer.graph.result_meta[vid]
    if damage == "missing":
        del lowerer.graph.result_meta[vid]
    elif damage == "wrong_owner":
        lowerer.graph.result_meta[vid] = replace(meta, producer_id=meta.producer_id + 1)
    elif damage == "index_width":
        lowerer.graph.result_meta[vid] = replace(meta, type=replace(meta.type, width=64))
    else:
        lowerer.graph.result_meta[vid] = replace(meta, type=replace(meta.type, shape=(1, 64)))
    with pytest.raises(MetalNonRecoverableError, match="native|metadata|projection"):
        prove_normalization(lowerer.graph, info, "softmax")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize("mode", [6, 7, 8, 9, 10])
def test_normalization_address_fill_and_effects_compute(routes, mode):
    """Each row adds a distinct contract absent from the value-only matrix."""
    torch.manual_seed(261)
    n, rows = 47, 8
    cpu = torch.randn(rows * n + int(mode == 7)) * 3 + 2
    source = cpu[int(mode == 7) :].reshape(rows, n)
    x = cpu.to("mps")
    storage = torch.full((rows * n + int(mode == 9),), -12345.0, device="mps")
    if mode == 9:
        storage[-1] = 0
    grid = (1, rows) if mode == 8 else (rows,)
    _soft[grid](x, storage, n, 64, mode)
    torch.mps.synchronize()
    if mode == 6:
        # other=0 participates in both max and denominator across all64 lanes.
        padded = torch.nn.functional.pad(source, (0, 64 - n))
        expected = torch.softmax(padded, dim=-1)[:, :n]
    else:
        expected = torch.softmax(source, dim=-1)
    actual = storage[: rows * n].cpu().reshape(rows, n)
    if mode == 10:
        assert (actual[:, -1] == -12345).all()
        actual, expected = actual[:, :-1], expected[:, :-1]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if mode == 9:
        assert storage[-1].item() == rows
    assert not routes


@pytest.mark.parametrize(
    "kind",
    [
        "missing_return",
        "projection_return",
        "duplicate_operand",
        "wrong_axis",
        "missing_type",
        "unknown_rank",
        "wrong_result_shape",
    ],
)
def test_normalization_combiner_metadata_is_mandatory(routes, kind):
    type_case = kind in ("missing_type", "unknown_rank", "wrong_result_shape")
    lowerer = _build_lowerer(
        _soft, {"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "i32"}, {"BLOCK": 64, "MODE": 1 if type_case else 0}
    )
    reduce = next(o for o in lowerer.graph.ops if o.op == "tt.reduce")
    args = reduce.attrs["block_arg_ids"]
    if kind == "missing_return":
        reduce.attrs.pop("return_ids", None)
    elif kind == "projection_return":
        reduce.attrs["return_ids"] = [args[0]]
    elif kind == "duplicate_operand":
        reduce.region_ops[0].operand_ids = [args[0], args[0]]
    elif kind == "wrong_axis":
        reduce.attrs["axis"] = 1
    elif kind == "missing_type":
        lowerer.graph.result_meta.pop(reduce.operand_ids[0])
    else:
        vid = reduce.operand_ids[0] if kind == "unknown_rank" else reduce.id
        meta = lowerer.graph.result_meta[vid]
        shape = None if kind == "unknown_rank" else (64,)
        lowerer.graph.result_meta[vid] = replace(meta, type=replace(meta.type, shape=shape))
    with pytest.raises(MetalNonRecoverableError):
        lowerer.lower()
    assert not routes
