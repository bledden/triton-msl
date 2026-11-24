"""Overflow, infinities and NaNs come out of the attention forwards where the source puts them.

The source carries a float32 accumulator across its key loop: ``acc = acc * alpha`` at each key
block, then ``tl.dot(p, v)`` added to it. Two things follow that a finite, well-scaled input
never shows.

1. A block whose value product overflows float32 leaves ``+inf`` in the accumulator, and a
   later block's positive ``alpha`` keeps it there. A shader that fuses two 32-key source
   blocks into one 64-key update forms every probability against the later maximum, and the
   same product stays finite.
2. A mask on the SCORE gives a key probability zero; it does not unload its value. The causal
   variable-length source loads V for every key of the sequence, so a future key whose value is
   NaN or infinite contributes ``0 * v``, which is NaN.

Every case here is structured so that each query scores key ``k`` at exactly ``s_k``; the
expectation is the source's recurrence, emulated a block at a time (the emulation of
test_attention_forwards_follow_the_source_recurrence, whose value product is formed term by
term). No case depends on the order in which one block's product is summed: the compiler's IR
is ``tt.dot(p, v, acc * alpha)``, a product that carries the accumulator.
"""

import math

import pytest
import torch

import tests.test_varlen_fa_routing as routing
from tests.test_attention_forwards_follow_the_source_recurrence import (
    MATRIX,
    T,
    _base,
    _emulate,
    _launch,
    _variant,
    variant_dir,  # noqa: F401
)
from tests.test_fa_tail_semantics import _executed_source, executed  # noqa: F401
from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal device")

BIG, INF, NAN = 2.0**127, math.inf, math.nan
# name: ({key: score}, {key: value in features 0 and 9}, needs a V type that holds 2**127)
SCENARIOS = {
    "control": ({0: 0.0}, {}, False),
    "overflow-then-raise-32": ({0: 0.0, 1: 0.0, 32: 1.0}, {0: BIG, 1: BIG}, True),
    "overflow-then-raise-64": ({0: 0.0, 1: 0.0, 64: 1.0}, {0: BIG, 1: BIG}, True),
    "overflow-then-raise-96": ({64: 0.0, 65: 0.0, 96: 1.0}, {64: BIG, 65: BIG}, True),
    "raise-inside-the-block": ({0: 0.0, 1: 0.0, 2: 1.0}, {0: BIG, 1: BIG}, True),
    "raise-inside-a-later-block": ({32: 0.0, 33: 0.0, 40: 1.0}, {32: BIG, 33: BIG}, True),
    "infinity-then-two-raises": ({0: 0.0, 32: 60.0, 64: 120.0}, {0: INF}, False),
    "infinity-at-40": ({0: 0.0, 40: 0.0}, {40: INF}, False),
    "nan-at-20": ({0: 0.0}, {20: NAN}, False),
    "nan-at-40": ({0: 0.0}, {40: NAN}, False),
    "nan-at-100": ({0: 0.0}, {100: NAN}, False),
}
FEATURES = (0, 9)


def _inputs(name, n, heads, dim, q_t, kv_t):
    scores, values, _ = SCENARIOS[name]
    q, k, v = torch.zeros((n, heads, dim)), torch.zeros((n, heads, dim)), torch.ones((n, heads, dim))
    q[:, :, 0], k[:, :, 0] = 8.0, -40.0
    for key, score in scores.items():
        k[key, :, 0] = score
    for key, value in values.items():
        for feature in FEATURES:
            v[key, :, feature] = value
    return q.to(q_t), k.to(kv_t), v.to(kv_t)


def _differs(got, want):
    """Where the NaN mask, the signed-infinity mask or a finite value (1e-4 relative) differs."""
    bad = (got.isnan() != want.isnan()) | (got.isinf() != want.isinf())
    bad |= got.isinf() & want.isinf() & (got.sign() != want.sign())
    finite = got.isfinite() & want.isfinite()
    return bad | (finite & ~((got - want).abs() <= 1e-4 * want.abs() + 1e-4))


def _check(kind, mix, form, bm, bn, causal, variant_dir, executed):
    q_t, kv_t, out_t = (T[x] for x in mix)
    kernel = _variant(_base(kind, causal), form, variant_dir)
    n, heads, dim = 128, 2, 64
    source = None
    for name, (_, _, needs_big) in SCENARIOS.items():
        if needs_big and kv_t == torch.float16:
            continue
        q, k, v = _inputs(name, n, heads, dim, q_t, kv_t)
        executed.clear()
        try:
            got, source = _launch(kind, kernel, q, k, v, out_t, bm, bn, causal, executed)
        except MetalNonRecoverableError:
            pytest.skip("refused: an allowed answer")
        want = torch.stack(
            [_emulate(kind, q[:, h], k[:, h], v[:, h], bm, bn, causal, form, out_t)[0] for h in range(heads)], 1
        )
        bad = _differs(got, want)
        first = tuple(bad.nonzero()[0].tolist()) if bad.any() else None
        assert first is None, (name, int(bad.sum()), first, float(got[first]), float(want[first]))
    return source


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("form", ["wide", "half"])
@pytest.mark.parametrize("bm,bn", [(32, 32), (32, 64), (64, 32), (64, 64)])
@pytest.mark.parametrize("mix", [("f32", "f32", "f32"), ("f16", "f32", "f32"), ("f16", "f16", "f16")], ids="-".join)
def test_the_dense_forward_keeps_overflow_infinities_and_nans(
    mix, bm, bn, form, causal, variant_dir, executed, monkeypatch
):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    source = _check("dense", mix, form, bm, bn, causal, variant_dir, executed)
    if (bm, bn) == (32, 32) and mix[1] == mix[2]:
        # Two source key blocks to the 64-key tile, each its own accumulator update.
        assert MATRIX in source and "// source key block 1 of this tile: acc * alpha, then its value product" in source


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("form", ["wide", "vtype"])
@pytest.mark.parametrize(
    "mix",
    [
        ("f32", "f16", "f16"),  # one-key shader
        ("f16", "f16", "f16"),  # matrix shader
        ("f32", "bf16", "bf16"),
        ("bf16", "bf16", "bf16"),
        ("f32", "f32", "f32"),
        ("f16", "f16", "f32"),
    ],
    ids="-".join,
)
def test_the_variable_length_forward_keeps_overflow_infinities_and_nans(
    mix, form, causal, variant_dir, executed, monkeypatch
):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    _check("varlen", mix, form, 32, 32, causal, variant_dir, executed)


def _dense_overflow(case, width, form, variant_dir, executed):
    n = 32 if case == "first-block-only" else 64
    big = 2.0**120 if case == "smaller" else BIG
    q, k = torch.zeros((n, 2, 64)), torch.zeros((n, 2, 64))
    q[..., 0], k[..., 0] = 8.0, -40.0
    k[0, :, 0] = k[1, :, 0] = 0.0
    if n == 64:
        k[32, :, 0] = 1.0
    v = torch.zeros((n, 2, 64))
    v[0] = v[1] = big
    kernel = _variant(_flash_attn_fwd, form, variant_dir)
    executed.clear()
    got, source = _launch("dense", kernel, q, k, v, torch.float32, 32, width, False, executed)
    assert MATRIX in source
    return got


@pytest.mark.parametrize("form", ["wide", "half"])
def test_a_first_block_that_overflows_float32_stays_infinite_after_a_later_block(
    form, variant_dir, executed, monkeypatch
):
    """Keys 0 and 1 score 0 with value 2**127: probability 1 each, and their float32 sum is +inf.
    Key 32 scores 1 with value 0. With 32-key source blocks the first block's accumulator is
    already +inf when the second multiplies it by exp(-1); with one 64-key block the same two
    products are formed against the maximum 1 and their sum is finite."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    assert bool((_dense_overflow("first-block-only", 32, form, variant_dir, executed) == INF).all())
    assert bool((_dense_overflow("overflow", 32, form, variant_dir, executed) == INF).all())
    e = math.exp(-1.0)
    p = float(torch.tensor(e, dtype=torch.float32).to(torch.float16)) if form == "half" else e
    for case, width, big in (("overflow", 64, BIG), ("smaller", 32, 2.0**120), ("smaller", 64, 2.0**120)):
        got = _dense_overflow(case, width, form, variant_dir, executed)
        # 32-key blocks at 2**120: (2**120 + 2**120) * exp(-1); one 64-key block: 2 * p * big.
        top = 2 * big * (e if width == 32 else p)
        want = top / (1.0 + 2 * e)
        assert float(((got - want).abs() / want).max()) <= 1e-5, (case, width, got.unique(), want)


def _varlen_future(q_t, value, form, variant_dir, executed, lengths=(32,)):
    n = sum(lengths)
    q = torch.zeros((n, 2, 64), dtype=q_t, device="mps")
    k = torch.zeros((n, 2, 64), dtype=torch.float16, device="mps")
    v = torch.ones((n, 2, 64), dtype=torch.float16, device="mps")
    if value is not None:
        v[16, :, 0] = value
    if len(lengths) == 2:
        v[lengths[0] :] = NAN  # another sequence's storage: never loaded for this one
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="mps")
    out = torch.full_like(v, -99.0)
    kernel = _variant(routing._varlen_causal_fwd, form, variant_dir)
    executed.clear()
    compiled = kernel[(1, 2 * len(lengths))](
        q, k, v, out, cu, cu, *q.stride(), *k.stride(), *v.stride(), *out.stride(), 2, 32, 0.125, 32, 32, 64
    )
    torch.mps.synchronize()
    return out.cpu().float(), _executed_source(executed, compiled)


@pytest.mark.parametrize("form", ["wide", "vtype"])
@pytest.mark.parametrize("q_t,matrix", [(torch.float32, False), (torch.float16, True)], ids=["one-key", "matrix"])
def test_a_causal_future_key_still_multiplies_its_value(q_t, matrix, form, variant_dir, executed, monkeypatch):
    """Q and K are zero and V is one. Key 16's value in feature 0 is NaN or +inf. Rows 0-15 mask
    key 16's score (probability 0) and still load its value: 0 * NaN and 0 * inf are NaN. Rows
    16-31 give it a positive probability: NaN, and +inf."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    got, source = _varlen_future(q_t, None, form, variant_dir, executed)
    assert ("simdgroup_multiply_accumulate" in source) == matrix
    assert bool((got == 1.0).all())
    got, _ = _varlen_future(q_t, NAN, form, variant_dir, executed)
    assert bool(got[:, :, 0].isnan().all()) and bool((got[:, :, 1:] == 1.0).all())
    got, _ = _varlen_future(q_t, INF, form, variant_dir, executed)
    assert bool(got[:16, :, 0].isnan().all()) and bool((got[16:, :, 0] == INF).all())
    assert bool((got[:, :, 1:] == 1.0).all())
    # Storage past the sequence's own end belongs to the next sequence and is not loaded.
    got, _ = _varlen_future(q_t, None, form, variant_dir, executed, lengths=(24, 32))
    assert bool((got[:24] == 1.0).all()) and bool(got[24:].isnan().all())


@pytest.mark.parametrize("raise_key", [2, 32, 64])
@pytest.mark.parametrize("block_n", [32, 64, 128])
def test_a_biased_source_key_block_is_one_accumulator_update(block_n, raise_key):
    """The biased source, float32, with every score taken from the bias: keys 0 and 1 score 0
    with value 2**127 and one later key scores 1 (in the source's exp2 units, probability 1/2
    for the first two). Where that key lies in the first source block the two products are
    2**126 each and their sum is finite; where it lies in a later block the first block's
    accumulator is already +inf. A 64-key tile must neither fuse two 32-key blocks nor split
    a 128-key one."""
    import triton

    from tests.test_fa_biased_routing import _biased_fa

    n, dim, ln2 = 128, 64, 0.6931471824645996
    q, k, v = torch.zeros(1, 2, n, dim), torch.zeros(1, 2, n, dim), torch.zeros(1, 2, n, dim)
    bias = torch.full((1, 2, n, n), -40.0 * ln2)
    bias[..., 0] = bias[..., 1] = 0.0
    bias[..., raise_key] = ln2
    v[:, :, 0] = v[:, :, 1] = BIG
    mask = torch.zeros(1, 2, n, dtype=torch.uint8)
    out = torch.full((1, 2, n, dim), -99.0, device="mps")
    lse = torch.full((1, 2, n), -99.0, device="mps")
    args = [out, *out.stride(), lse, *lse.stride()]
    for tensor in (q, k, v, bias, mask):
        tensor = tensor.to("mps")
        args += [tensor, *tensor.stride()]
    _biased_fa[(triton.cdiv(n, 32), 2)](*args, dim**-0.5, -1e9, 1, 2, n, dim, 32, block_n)
    torch.mps.synchronize()
    got = out.cpu()
    if raise_key >= block_n:
        assert bool((got == INF).all()), got.unique()
    else:
        want = BIG / 2  # (2**126 + 2**126) / (1/2 + 1/2 + 1), the other keys below float32's last place
        assert float(((got - want).abs() / want).max()) <= 1e-5, got.unique()
