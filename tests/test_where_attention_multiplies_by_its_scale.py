"""Where the biased attention kernels multiply by ``sm_scale``.

Each source multiplies one operand of its first dot by the scale: K in the dK / dV and the dQ
kernel, Q in the dbias and the forward kernel (tests/test_fa_bwd_routing.py,
tests/test_fa_biased_routing.py). The float32 templates used to multiply the finished dot
instead, and every backward template took a plain zero for a key past ``N``. Those are the
source's numbers only while every product is finite:

* an operand times a scale near 3e38 overflows in the source, whatever the dot times the
  scale would have been;
* an infinite scale times a zero element is NaN in the source, where an infinite scale times
  a dot of positive terms is an infinity;
* a key past ``N`` is the load's zero times the scale, NaN when the scale is not finite.

Here the scale is a launch argument that is infinite, NaN, large, zero, negative or tiny, and
every result is compared with an evaluation of the source that multiplies every pair, so zero
times infinity is the NaN it is in the kernel.
"""

import pytest
import torch
import triton

import tests.test_fa_bwd_address_witness as witness
import tests.test_fa_forward_rounding as forward
import tests.test_what_a_masked_backward_load_supplies as supplies
import triton_msl.codegen._msl_templates as templates

requires_mps = supplies.requires_mps
makers = supplies.makers  # the backward makers called, on this module's caches
cold_gpu_caches, maker_spy = forward.cold_gpu_caches, forward.maker_spy
INF, NAN = float("inf"), float("nan")
INV_LN2, LN2 = forward.INV_LN2, forward.LN2
TYPE_ID, TOLERANCE = supplies.TYPE_ID, supplies.TOLERANCE
SCALES = [INF, -INF, NAN, 1e30, 0.0, -0.25, 1e-30]
OVERFLOWING = [3e38, -3e38]  # float32 only: in 16 bits these scales are infinite


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    return tmp_path_factory.mktemp("attention_scale")


def _f32(x):
    return x.float().double()


def _sign(x):
    return torch.where(x < 0, -1.0, 1.0).to(x.dtype)


# ---------------------------------------------------------------------------------------
# dK / dV and dQ: K times the scale
# ---------------------------------------------------------------------------------------


def _positive(t):
    """No term of any sum is negative and no key is masked: an infinite scale gives +inf
    scores, and a key past N its zero times the scale."""
    for name in ("q", "k", "v", "do", "o"):
        t[name] = t[name].abs() + 0.125
    t["mask"][:] = 0


def _mixed(t):
    """Keys and dO positive, every Q.K positive from terms of both signs, one row of V: the
    dot times an infinite scale is one infinity, the dot of Q with K times it is NaN."""
    t["k"] = t["k"].abs() + 0.125
    t["do"] = t["do"].abs() + 0.125
    t["q"][..., 0] = 40.0
    t["v"][0, 0, :] = t["v"][0, 0, 0]
    t["mask"][:] = 0


def _overflowing(t):
    """K times the scale overflows; Q.K times the scale is an ordinary number."""
    t["k"] = _sign(t["k"]) * (1.5 + t["k"].abs())
    t["q"] = _sign(t["q"]) * (1.0 + t["q"].abs()) * 2e-38
    t["mask"][:] = 0


BACKWARD_FORMS = {"random": None, "positive": _positive, "mixed": _mixed, "overflowing": _overflowing}


def _backward_cases(kind):
    cases = []
    for dim, dtype, maker in supplies.SHAPES[kind]:
        for form in BACKWARD_FORMS:
            if form == "overflowing" and dtype != torch.float32:
                continue
            for scale in OVERFLOWING if form == "overflowing" else SCALES:
                cases.append(pytest.param(dim, dtype, maker, form, scale, id=f"{dim}-{TYPE_ID[dtype]}-{form}-{scale}"))
    return cases


@requires_mps
@pytest.mark.parametrize("sentinel", [-1e9, 5.0], ids=["-1e9", "5.0"])
@pytest.mark.parametrize("n", [48, 64])
@pytest.mark.parametrize("dim,dtype,maker,form,scale", _backward_cases("kv"))
def test_dk_dv_score_is_q_dot_k_times_the_scale(dim, dtype, maker, form, scale, n, sentinel, makers):
    supplies._run(
        "kv", {}, sentinel, n, dim, dtype, maker, makers, seed=4200 + n, edit_inputs=BACKWARD_FORMS[form], scale=scale
    )


@requires_mps
@pytest.mark.parametrize("sentinel", [-1e9, 5.0], ids=["-1e9", "5.0"])
@pytest.mark.parametrize("n", [48, 64])
@pytest.mark.parametrize("dim,dtype,maker,form,scale", _backward_cases("q"))
def test_dq_score_and_sum_are_over_k_times_the_scale(dim, dtype, maker, form, scale, n, sentinel, makers):
    """48 leaves sixteen keys of the last tile past N: each is zero times the scale."""
    supplies._run(
        "q", {}, sentinel, n, dim, dtype, maker, makers, seed=4200 + n, edit_inputs=BACKWARD_FORMS[form], scale=scale
    )


# ---------------------------------------------------------------------------------------
# dbias: Q times the scale
# ---------------------------------------------------------------------------------------


def _dbias_source(t, dtype, scale):
    """``_bwd_b`` on the witness module's tensors, every pair multiplied."""
    a = {**witness.CANON, "t": t, "s": {name: x.stride() for name, x in t.items()}}
    s = a["s"]
    r = lambda x: x.to(dtype).double()
    n, dim = witness.N, witness.DIM
    out = torch.full(t["db"].shape, witness.POISON, dtype=torch.float64)
    j = torch.arange(n)[:, None]
    k = torch.arange(n)[:, None]
    d = torch.arange(dim)[None, :]
    sm = r(_f32(torch.tensor(scale, dtype=torch.float64)))  # tl.full([1], sm_scale, the tensors' type)
    for h in range(witness.HC):
        db = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            q = witness._gather(t["q"], a["q"](s["q"], h, i, j, d))
            kk = witness._gather(t["k"], a["k"](s["k"], h, i, k, d))
            v = witness._gather(t["v"], a["v"](s["v"], h, i, k, d))
            do = witness._gather(t["do"], a["do"](s["do"], h, i, j, d))
            bias = witness._gather(t["b"], a["b"](s["b"], h, i, j, k.T))
            lse = witness._gather(t["lse"], a["lse"](s["lse"], h, i, j))
            masked = witness._gather(t["mask"], a["mask"](s["mask"], h, i, k.T)) != 0
            delta = witness._gather(t["delta"], a["delta"](s["delta"], h, i, j))
            scores = _f32(_f32(supplies._mm(r(_f32(q * sm)), kk.T)) + bias)
            p = _f32(torch.exp2(_f32(_f32(scores - lse) * INV_LN2))).masked_fill(masked, 0.0)
            db = _f32(db + _f32(p * _f32(_f32(supplies._mm(do, v.T)) - delta)))
        witness._place(out, a["db"](s["db"], h, j, k.T), db)
    return out


def _dbias_mixed(t):
    """Q and K positive but for one zero column of Q, which an infinite scale turns to NaN."""
    t["q"] = t["q"].abs()
    t["q"][..., 0] = 0.0
    t["k"] = t["k"].abs()


def _dbias_overflowing(t):
    """Q times the scale overflows; Q.K times the scale is an ordinary number."""
    t["q"] = _sign(t["q"]) * (1.5 + t["q"].abs())
    t["k"] = _sign(t["k"]) * (1.0 + t["k"].abs()) * 2e-38


Q_FORMS = {
    "random": (None, SCALES),
    "mixed": (_dbias_mixed, [INF, -INF]),
    "overflowing": (_dbias_overflowing, OVERFLOWING),
}


def _q_cases(scales=()):
    cases = []
    for form, (_, form_scales) in Q_FORMS.items():
        for dtype in TYPE_ID:
            if form == "overflowing" and dtype != torch.float32:
                continue
            for scale in [*form_scales, *(scales if form == "random" else ())]:
                cases.append(pytest.param(dtype, form, scale, id=f"{TYPE_ID[dtype]}-{form}-{scale}"))
    return cases


@requires_mps
@pytest.mark.parametrize("dtype,form,scale", _q_cases())
def test_dbias_score_is_q_times_the_scale_dot_k(dtype, form, scale, makers, monkeypatch):
    monkeypatch.setattr(witness, "SM", scale)
    t = supplies._dbias_inputs(dtype, seed=2031)
    if Q_FORMS[form][0]:
        Q_FORMS[form][0](t)
    (got,) = witness._launch("b", supplies._kernel("b", {}), t)
    supplies._same([got.cpu()], [_dbias_source(t, dtype, scale)], 4e-3)


# ---------------------------------------------------------------------------------------
# The forward kernel: Q times the scale
# ---------------------------------------------------------------------------------------


def _forward_source(p, dtype, wide_product=False, round_p=True):
    """``_biased_tri_fa`` block by block, every pair multiplied, rounding where it rounds: the
    product of Q and the scale unless it is float32 (``wide_product``), P unless ``round_p``
    is false, the output."""
    q, k, v, bias = (p[name].cpu().double() for name in ("q", "k", "v", "bias"))
    n, block, heads = p["N"], p["BK"], p["Hc"]
    r = (lambda x: x) if dtype == torch.float32 else (lambda x: x.to(dtype).double())
    mm = lambda a, b: _f32((a[..., :, None, :] * b[..., None, :, :]).sum(-1))  # [.., J, D] x [.., K, D]
    scale = _f32(torch.tensor(p["sm"], dtype=torch.float64))
    qs = _f32(q * scale) if wide_product else r(_f32(q * r(scale)))
    masked = p["mask"].cpu()[torch.arange(heads) // p["Hh"]].bool()
    m = torch.full((heads, p["I"], n), -INF, dtype=torch.float64)
    denom = torch.zeros(heads, p["I"], n, dtype=torch.float64)
    acc = torch.zeros(heads, p["I"], n, p["DIM"], dtype=torch.float64)
    for k0 in range(0, n, block):
        keys = slice(k0, k0 + block)
        s = _f32(_f32(mm(qs, k[:, :, keys]) + bias[:, None, :, keys]) * INV_LN2)
        s = torch.where(masked[:, :, None, keys], torch.tensor(-1e9, dtype=torch.float64), s)
        m_new = torch.maximum(m, s.amax(-1))
        e = _f32(torch.exp2(_f32(s - m_new[..., None])))
        alpha = _f32(torch.exp2(_f32(m - m_new)))
        denom = _f32(_f32(denom * alpha) + _f32(e.sum(-1)))
        pv = _f32(((r(e) if round_p else e)[..., :, :, None] * v[:, :, keys][..., None, :, :]).sum(-2))
        acc = _f32(_f32(acc * alpha[..., None]) + pv)
        m = m_new
    return r(_f32(acc / denom[..., None])), _f32(_f32(m * LN2) + _f32(torch.log(denom)))


def _strided(p, dtype):
    """Q as every other column of a wider tensor: the tiled maker's route."""
    backing = torch.empty(*p["q"].shape[:-1], p["DIM"] * 2, device="mps", dtype=dtype)
    view = backing[..., ::2]
    view.copy_(p["q"])
    p["q"] = view


def _forward_problem(dtype, form, scale, n, route):
    p = forward._problem(dtype, N=n, seed=5, sm=scale)
    if Q_FORMS[form][0]:
        Q_FORMS[form][0](p)
    if route == "strided":
        _strided(p, dtype)
    return p


@pytest.fixture
def forward_caches(monkeypatch, cache_root):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(cache_root / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root / "triton"))


@requires_mps
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("route", ["contiguous", "strided"])
@pytest.mark.parametrize("dtype,form,scale", _q_cases(scales=[0.1234567]))
def test_forward_score_is_q_times_the_scale_dot_k(dtype, form, scale, route, n, forward_caches):
    p = _forward_problem(dtype, form, scale, n, route)
    o, lse = forward._launch(p, dtype)
    supplies._same([o.cpu(), lse.cpu()], list(_forward_source(p, dtype)), TOLERANCE[dtype])


@requires_mps
@pytest.mark.parametrize("route,maker", [("contiguous", "simdgroup"), ("strided", "tiled")])
def test_the_float32_forward_makers_are_given_the_source_s_product(route, maker, cold_gpu_caches, maker_spy):
    p = _forward_problem(torch.float32, "overflowing", 3e38, 64, route)
    o, lse = forward._launch(p, torch.float32)
    assert maker_spy == [(maker, "float", None)]
    want_o, want_lse = _forward_source(p, torch.float32)
    assert bool(want_o.isnan().all()) and bool(want_lse.isnan().all())  # every Q element overflows
    supplies._same([o.cpu(), lse.cpu()], [want_o, want_lse], TOLERANCE[torch.float32])


# A 16-bit Q converted to float32 before the multiply: the product is float32 and overflows
# where a float32 product does. P is rounded to the tensors' type, or V is converted too.
WIDE_PRODUCT = [
    (
        "q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)",
        "q_block = q_block.to(tl.float32) * tl.full([1], value=sm_scale, dtype=tl.float32)",
    ),
    ("scores = tl.dot(q_block, kt_block, scores)", "scores = tl.dot(q_block, kt_block.to(tl.float32), scores)"),
    (
        "tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_j[:, None])",
        "tl.store(o_ptrs, normalize.to(o_ptr.dtype.element_ty), mask=mask_j[:, None])",
    ),
]
P_ROUNDED = (
    "exp_scores = exp_scores.to(q_block.type.element_ty)",
    "exp_scores = exp_scores.to(v_block.type.element_ty)",
)
P_FLOAT32 = ("acc = tl.dot(exp_scores, v_block, acc)", "acc = tl.dot(exp_scores, v_block.to(tl.float32), acc)")
ROUNDING = {torch.float16: "half", torch.bfloat16: "bfloat"}


@requires_mps
@pytest.mark.parametrize("form,scale", [("random", 0.1234567), ("random", INF), ("mixed", -INF), ("overflowing", 3e38)])
@pytest.mark.parametrize("round_p", [True, False], ids=["P-rounded", "P-float32"])
@pytest.mark.parametrize("dtype", list(ROUNDING), ids=["fp16", "bf16"])
def test_a_float32_product_of_a_16_bit_q_is_staged_by_the_tiled_maker(
    dtype, round_p, form, scale, cold_gpu_caches, maker_spy, monkeypatch
):
    """The simdgroup maker holds a 16-bit Q in 16 bits, which the float32 product does not fit.
    A 16-bit Q times 3e38 overflows in float32 as a float32 Q does."""
    fn = triton.JITFunction(forward._biased_tri_fa.fn)
    src = fn.src
    for old, new in [*WIDE_PRODUCT, P_ROUNDED if round_p else P_FLOAT32]:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    fn._unsafe_update_src(src)
    monkeypatch.setattr(forward, "_biased_tri_fa", fn)
    p = _forward_problem(dtype, form, scale, 64, "contiguous")
    o, lse = forward._launch(p, dtype)
    assert maker_spy == [("tiled", "float", ROUNDING[dtype] if round_p else None)]
    want = _forward_source(p, dtype, wide_product=True, round_p=round_p)
    supplies._same([o.cpu(), lse.cpu()], list(want), TOLERANCE[dtype])


def test_the_simdgroup_maker_stages_a_float32_product_of_float32_q_only():
    """The lowering sends a float32 product of a narrower Q to the tiled maker."""
    with pytest.raises(ValueError, match="staged by the float32 maker only"):
        templates.make_flash_attention_kernel_simdgroup(head_dim=32, out_dtype="fp16", round_q="float")
