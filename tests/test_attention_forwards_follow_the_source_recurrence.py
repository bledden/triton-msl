"""The attention forwards compute what the source's own recurrence computes, type by type.

A source decides where its probability is rounded. ``tl.dot(p.to(tl.float32), v.to(tl.float32))``
keeps it float32; ``tl.dot(p.to(v.dtype), v)`` or an explicit ``p.to(tl.float16).to(tl.float32)``
narrows it, after the softmax of that key block has taken the block's maximum. A shader that rounds
a different number (a probability against a running maximum, or against a wider tile's maximum),
rounds when the source does not, or does not round when it does, gives another result.

The first half emulates the source step by step on the CPU: each operation rounded to the type the
source gives it, each dot formed in float64 and rounded to float32, one online-softmax update per
source key block. Random inputs, a float32 Out where the storage types allow one, and a tolerance
that admits a probability sitting on a rounding boundary. The second half is exact witnesses.
"""

import importlib.util
import inspect
import math
import sys

import pytest
import torch

import tests.test_varlen_fa_routing as routing
from tests.test_fa_tail_semantics import _executed_source, executed  # noqa: F401
from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

MATRIX = "simdgroup-MMA FlashAttention-2"
TILED = "Head-dim-tiled FlashAttention-2"
T = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
F32 = torch.float32
WIDE = "acc += tl.dot(p.to(tl.float32), v.to(tl.float32))\n"
PV = {
    "wide": "tl.dot(p.to(tl.float32), v.to(tl.float32))",
    "vtype": "tl.dot(p.to(v.dtype), v)",
    "half": "tl.dot(p.to(tl.float16).to(tl.float32), v.to(tl.float32))",
    "bfloat": "tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))",
}
_VARIANTS = {}


@pytest.fixture(scope="module")
def variant_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pv_variants")


def _variant(base, form, root):
    """``base`` with its P @ V line replaced by ``PV[form]``."""
    if form == "wide":
        return base
    name = f"{base.fn.__name__.strip('_')}_{form}"
    if name not in _VARIANTS:
        src = inspect.getsource(base.fn)
        assert src.count(WIDE) == 1
        path = root / f"{name}.py"
        path.write_text("import triton\nimport triton.language as tl\n\n\n" + src.replace(WIDE, f"acc += {PV[form]}\n"))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _VARIANTS[name] = getattr(module, base.fn.__name__)
    return _VARIANTS[name]


def _rnd(x, dtype):
    return x.to(dtype).double()


def _p_for_dot(p, form, v_dtype):
    if form == "wide" or (form == "vtype" and v_dtype == F32):
        return p
    return _rnd(p, {"vtype": v_dtype, "half": torch.float16, "bfloat": torch.bfloat16}[form])


def _dot(p, v):
    """p @ v term by term: a BLAS product may skip a zero factor, and 0 * NaN is NaN."""
    return (p[:, :, None] * v[None, :, :]).sum(1)


def _recurrence(q, k, v, mask_fn, blocks, form, v_dtype):
    """One query block. q is the source's scaled Q, k the keys as the score dot reads them."""
    rows = q.shape[0]
    m_i = torch.full((rows,), -math.inf, dtype=torch.float64)
    l_i = torch.zeros(rows, dtype=torch.float64)
    acc = torch.zeros((rows, v.shape[1]), dtype=torch.float64)
    risk = torch.zeros(rows, dtype=torch.float64)
    for start, width in blocks:
        kb = torch.zeros((width, k.shape[1]), dtype=torch.float64)
        vb = torch.zeros((width, v.shape[1]), dtype=torch.float64)
        have = max(0, min(width, k.shape[0] - start))
        kb[:have], vb[:have] = k[start : start + have], v[start : start + have]
        qk = _rnd(q @ kb.T, F32)
        mask = mask_fn(start, width)
        if mask is not None:
            qk = torch.where(mask, qk, torch.full_like(qk, -math.inf))
        m_new = torch.maximum(m_i, qk.max(1).values)
        alpha = _rnd(torch.exp(_rnd(m_i - m_new, F32)), F32)
        p = _rnd(torch.exp(_rnd(qk - m_new[:, None], F32)), F32)
        l_i = _rnd(_rnd(l_i * alpha, F32) + _rnd(p.sum(1), F32), F32)
        acc = _rnd(_rnd(acc * alpha[:, None], F32) + _rnd(_dot(_p_for_dot(p, form, v_dtype), vb), F32), F32)
        # What a probability within 5e-6 of a rounding boundary could move the row by.
        swing = (_p_for_dot(p * (1 + 5e-6), form, v_dtype) - _p_for_dot(p * (1 - 5e-6), form, v_dtype)).abs()
        swing = (swing - p * 1e-5).clamp(min=0).nan_to_num()
        risk = risk * alpha + (swing * vb.abs().max(1).values[None, :]).sum(1)
        m_i = m_new
    return _rnd(acc / l_i[:, None], F32), (risk / l_i).nan_to_num(posinf=0.0)


def _emulate(kind, q, k, v, bm, bn, causal, form, out_dtype):
    """q, k, v: [N, D] of one head, in their storage types."""
    n = q.shape[0]
    out = torch.zeros((n, v.shape[1]), dtype=torch.float64)
    risk = torch.zeros(n, dtype=torch.float64)
    # Dense: q * (1 / sqrt(64)) is a float32 product and K is read as float32. Variable length:
    # q * SCALE keeps Q's type and the score dot reads K in that type.
    q_type = F32 if kind == "dense" else q.dtype
    keys = k.double() if kind == "dense" else _rnd(k.double(), q.dtype)
    for start_m in range(0, n, bm):
        rows = torch.arange(start_m, start_m + bm)
        live = rows < n
        qb = torch.zeros((bm, q.shape[1]), dtype=torch.float64)
        qb[live] = q[rows[live]].double()
        qb = _rnd(qb * 0.125, q_type)
        if kind == "dense":
            stop = min(start_m + bm, n) if causal else n

            def mask_fn(s, w, rows=rows):
                return (rows[:, None] >= (s + torch.arange(w))[None, :]) if causal else None
        else:
            stop = n

            def mask_fn(s, w, rows=rows):
                kn = s + torch.arange(w)
                seen = (kn < n)[None, :].expand(bm, w)
                return (seen & (rows[:, None] >= kn[None, :])) if causal else seen

        got, r = _recurrence(qb, keys, v.double(), mask_fn, [(s, bn) for s in range(0, stop, bn)], form, v.dtype)
        out[rows[live]], risk[rows[live]] = got[live], r[live]
    return _rnd(out, out_dtype), risk


def _launch(kind, kernel, q, k, v, out_dtype, bm, bn, causal, executed):
    """q, k, v: [N, H, D] on the CPU. Returns ([N, H, D] float64, executed shader)."""
    n, heads, dim = q.shape
    if kind == "dense":
        Q, K, V = [x.transpose(0, 1).unsqueeze(0).contiguous().to("mps") for x in (q, k, v)]
        O = torch.full((1, heads, n, dim), float("nan"), dtype=out_dtype, device="mps")
        compiled = kernel[(math.ceil(n / bm), heads)](
            Q, K, V, O, *Q.stride(), *K.stride(), *V.stride(), *O.stride(), 1, heads, n,
            BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=dim, IS_CAUSAL=causal,
        )  # fmt: skip
        torch.mps.synchronize()
        return O[0].transpose(0, 1).cpu().double(), _executed_source(executed, compiled)
    Q, K, V = [x.to("mps") for x in (q, k, v)]
    O = torch.full((n, heads, dim), float("nan"), dtype=out_dtype, device="mps")
    cu = torch.tensor([0, n], device="mps", dtype=torch.int32)
    compiled = kernel[(math.ceil(n / 32), heads)](
        Q, K, V, O, cu, cu, *Q.stride(), *K.stride(), *V.stride(), *O.stride(), heads, n, 0.125, 32, 32, dim
    )
    torch.mps.synchronize()
    return O.cpu().double(), _executed_source(executed, compiled)


def _base(kind, causal):
    if kind == "dense":
        return _flash_attn_fwd
    return routing._varlen_causal_fwd if causal else routing._varlen_fwd


DENSE_TILES = [(32, 32), (32, 64), (64, 32), (64, 64)]
CASES = [
    ("dense", mix, form, bm, bn, causal)
    for mix in [("f32", "f32", "f32"), ("f16", "f32", "f32"), ("f16", "f16", "f16"), ("f32", "f16", "f16")]
    for form in PV
    for bm, bn in DENSE_TILES
    for causal in (False, True)
] + [
    ("varlen", mix, form, 32, 32, causal)
    for mix in [
        ("f32", "f32", "f32"),
        ("f16", "f16", "f16"),
        ("f16", "f16", "f32"),
        ("f32", "f16", "f32"),
        ("f16", "f16", "bf16"),
        ("bf16", "bf16", "bf16"),
        ("f32", "bf16", "f32"),
    ]
    for form in PV
    for causal in (False, True)
]


@pytest.mark.parametrize(
    "kind,mix,form,bm,bn,causal",
    CASES,
    ids=[f"{c[0]}-{'-'.join(c[1])}-{c[2]}-{c[3]}x{c[4]}-{int(c[5])}" for c in CASES],
)
def test_the_forward_agrees_with_the_source_recurrence(
    kind, mix, form, bm, bn, causal, variant_dir, executed, monkeypatch
):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    q_t, kv_t, out_t = (T[x] for x in mix)
    n, heads, dim = 128, 2, 64
    g = torch.Generator().manual_seed(7)
    q = (torch.randn((n, heads, dim), generator=g) * 1.7).to(q_t)
    k = (torch.randn((n, heads, dim), generator=g) * 1.7).to(kv_t)
    v = (torch.randn((n, heads, dim), generator=g) * 100).to(kv_t)
    try:
        got, _ = _launch(
            kind, _variant(_base(kind, causal), form, variant_dir), q, k, v, out_t, bm, bn, causal, executed
        )
    except MetalNonRecoverableError:
        pytest.skip("refused")
    per_head = [_emulate(kind, q[:, h], k[:, h], v[:, h], bm, bn, causal, form, out_t) for h in range(heads)]
    want = torch.stack([x[0] for x in per_head], 1)
    risk = torch.stack([x[1] for x in per_head], 1)[:, :, None]
    ulp = {"f32": 0.0, "f16": 2.0**-10, "bf16": 2.0**-7}[mix[2]]
    tol = 3e-6 * float(v.double().abs().max()) + ulp * want.abs() + risk
    err = (got - want).abs()
    assert bool((err <= tol).all()), (float((err / tol).nan_to_num(nan=math.inf).max()), int((~(err <= tol)).sum()))


def test_the_emulation_tells_the_rounding_points_apart():
    """The comparison above means something only if the forms differ by more than its tolerance."""
    n, dim = 128, 64
    g = torch.Generator().manual_seed(7)
    q, k = [(torch.randn((n, 2, dim), generator=g) * 1.7)[:, 0] for _ in range(2)]
    v = (torch.randn((n, 2, dim), generator=g) * 100)[:, 0]
    for kind, bn in (("dense", 32), ("dense", 64), ("varlen", 32)):
        wide, _ = _emulate(kind, q, k, v, 32, bn, False, "wide", F32)
        half, risk = _emulate(kind, q, k, v, 32, bn, False, "half", F32)
        tol = 3e-6 * float(v.abs().max()) + risk[:, None]
        assert int(((wide - half).abs() > 4 * tol).sum()) > 1000
    # A narrowing against a 64-key maximum is not the narrowing against two 32-key maxima.
    two, risk = _emulate("dense", q, k, v, 32, 32, False, "half", F32)
    one, _ = _emulate("dense", q, k, v, 32, 64, False, "half", F32)
    assert int(((two - one).abs() > 2 * (3e-6 * float(v.abs().max()) + risk[:, None])).sum()) > 100


# ---- exact witnesses -------------------------------------------------------------------------
# Q[:, 0] = 8 and the scale is 1/8, so a key's score is its first coordinate.


def _witness(n, heads, dim, q_t, kv_t, scores, values):
    q = torch.zeros((n, heads, dim), dtype=q_t)
    q[..., 0] = 8
    k = torch.zeros((n, heads, dim), dtype=kv_t)
    k[..., 0] = -100
    v = torch.zeros((n, heads, dim), dtype=kv_t)
    for row, (score, value) in enumerate(zip(scores, values)):
        if score is not None:
            k[row, :, 0], v[row] = score, value
    return q, k, v


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("q_t", ["f16", "f32"])
@pytest.mark.parametrize("kv_t,big", [("f16", 65504.0), ("bf16", 65536.0)])
def test_a_narrowed_variable_length_product_does_not_depend_on_key_order(
    kv_t, big, q_t, swapped, variant_dir, executed
):
    """Both probabilities narrow to 1 against the block maximum, whichever key comes first."""
    if q_t == "f16" and kv_t == "bf16":
        q_t = "bf16"
    scores, values = [0.0, -(2.0**-13)], [big, -big]
    if swapped:
        scores, values = scores[::-1], values[::-1]
    q, k, v = _witness(64, 2, 64, T[q_t], T[kv_t], scores, values)
    kernel = _variant(routing._varlen_fwd, "vtype", variant_dir)
    got, source = _launch("varlen", kernel, q, k, v, T[kv_t], 32, 32, False, executed)
    assert torch.equal(got, torch.zeros_like(got)), got.unique()
    if q_t == "f32":
        assert "simdgroup_multiply_accumulate" not in source
        assert "float s_blk[32];" in source and "l_i = l_i * alpha + p_sum;" in source


def _narrowed_ratio(scores, block, grid=2048):
    """3 * sum(round(p)) / sum(p) for keys of value 3, one softmax update per ``block`` keys."""
    m, num, den = -math.inf, 0.0, 0.0
    for start in range(0, len(scores), block):
        chunk = [s for s in scores[start : start + block] if s is not None]
        if not chunk:
            continue
        m_new = max(m, max(chunk))
        alpha = math.exp(m - m_new)
        p = [math.exp(s - m_new) for s in chunk]
        num = num * alpha + sum(round(x * grid) / grid for x in p)
        den = den * alpha + sum(p)
        m = m_new
    return 3 * num / den


def _later_block_scores(first_of_later_block, n):
    scores = [None] * n
    scores[0], scores[1], scores[first_of_later_block] = 0.0, -0.5, 1.0
    return scores


def test_the_later_block_witness_separates_the_candidate_shaders():
    source = _narrowed_ratio(_later_block_scores(32, 64), 32)
    one_maximum = _narrowed_ratio(_later_block_scores(32, 64), 64)
    assert abs(source - one_maximum) > 1e-4 and abs(source - 3) > 5e-5 and abs(one_maximum - 3) > 1e-4


@pytest.mark.parametrize("q_t", ["f32", "f16"])
@pytest.mark.parametrize("bm,bn", [(32, 32), (32, 64), (64, 32), (64, 64)])
def test_a_dense_explicit_half_probability_is_narrowed_in_its_own_key_block(
    bm, bn, q_t, variant_dir, executed, monkeypatch
):
    """Keys 0 and 1 are narrowed against their block's maximum; a later block then raises it.

    At 32-key source blocks the later key is key 32, inside the matrix shader's 64-key tile: one
    maximum over that tile would narrow other numbers, so the shader keeps each block's own.
    """
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    n = 128
    scores = _later_block_scores(bn, n)
    q, k, v = _witness(n, 2, 64, T[q_t], F32, scores, [3.0] * n)
    v[:] = 3
    got, source = _launch(
        "dense", _variant(_flash_attn_fwd, "half", variant_dir), q, k, v, F32, bm, bn, False, executed
    )
    padded = [s if s is not None else -100.0 for s in scores]
    want = _narrowed_ratio(padded, bn)
    assert float((got - want).abs().max()) <= 2e-5, (got.unique(), want)
    if (bm, bn) == (32, 64) and q_t == "f32":
        assert MATRIX in source and "tg_S[pr*BN+cj] = float(half(p));" in source
    if bm == 32:
        assert MATRIX in source
    if (bm, bn) == (32, 32):
        # Two source blocks to a tile: each chunk reads the maximum its own block ended with,
        # and each block has its own alpha.
        assert "float mr = m_was;" in source and "tg_alpha[pr*2u + pc/4u] = exp(m_lo - mr);" in source


@pytest.mark.parametrize("q_t", ["f32", "f16"])
@pytest.mark.parametrize("kv_t", ["f16", "bf16"])
def test_a_variable_length_narrowed_probability_is_narrowed_in_its_own_key_block(kv_t, q_t, variant_dir, executed):
    if q_t == "f16" and kv_t == "bf16":
        q_t = "bf16"
    n = 64
    scores = _later_block_scores(32, n)
    q, k, v = _witness(n, 2, 64, T[q_t], T[kv_t], scores, [3.0] * n)
    v[:] = 3
    got, _ = _launch(
        "varlen", _variant(routing._varlen_fwd, "vtype", variant_dir), q, k, v, F32, 32, 32, False, executed
    )
    padded = [s if s is not None else -100.0 for s in scores]
    want = _narrowed_ratio(padded, 32, grid=2048 if kv_t == "f16" else 256)
    assert abs(want - 3) > 5e-5
    assert float((got - want).abs().max()) <= 2e-5, (got.unique(), want)


@pytest.mark.parametrize("form,want", [("half", 0.0), ("wide", 3.998046875)])
@pytest.mark.parametrize("q_t", ["f32", "f16"])
@pytest.mark.parametrize("bn", [32, 64])
def test_a_dense_explicit_half_cast_before_a_float32_dot_cancels(
    bn, q_t, form, want, variant_dir, executed, monkeypatch
):
    """float32 K, V and Out: the half cast makes both probabilities 1; without it they differ."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    q, k, v = _witness(64, 2, 64, T[q_t], F32, [0.0, -(2.0**-13)], [65504.0, -65504.0])
    got, _ = _launch("dense", _variant(_flash_attn_fwd, form, variant_dir), q, k, v, F32, 32, bn, False, executed)
    assert float((got - want).abs().max()) <= 4e-3 * (want != 0), got.unique()


def test_a_bfloat16_probability_beside_float16_tensors_keeps_its_rounding(variant_dir, executed, monkeypatch):
    """bfloat16 holds 8 bits: exp(-0.5) narrows to 155/256, which the half tiles cannot hold as such."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    n = 64
    q, k, v = _witness(n, 2, 64, torch.float16, torch.float16, [0.0, -0.5], [1024.0, 1024.0])
    got, source = _launch(
        "dense", _variant(_flash_attn_fwd, "bfloat", variant_dir), q, k, v, torch.float16, 32, 64, False, executed
    )
    assert MATRIX in source and "tgP[pr*BN+cj] = float(bfloat(p));" in source
    p = math.exp(-0.5)
    want = float(torch.tensor(1024 * (1 + round(p * 256) / 256) / (1 + p)).to(torch.float16))
    wide = float(torch.tensor(1024.0).to(torch.float16))
    assert want != wide
    assert float((got - want).abs().max()) == 0, got.unique()
