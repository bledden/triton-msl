"""Packet 164 — item F-a: the biased forward FA replays the source's rounding points for bf16 / fp16.

The trifast forward (`_biased_tri_fa`, `_biased_value`) rounds at three points that the templates
skipped for narrow input dtypes: ``q * tl.full([1], scale, dtype)`` (the scale rounded to the input
dtype, each product rounded), ``exp_scores.to(dtype)`` before the P @ V dot of EVERY kv block (so
relative to that block's running max), and the output cast. Packet 160 measured the gap against a
faithful oracle: bf16 3.9e-3, fp16 9.8e-4.

Now the value-path verifier records the scale multiply's element type and the P conversion's target
(`round_q`, `round_p`); both makers take them: Q is staged as ``elem(float(Q) * float(elem(scale)))``
(a half×half / bfloat×bfloat product is exact in fp32, so this is the source's single rounding) and
the score tile is no longer scaled again; P is rounded before P @ V. Because the source rounds P per
kv block relative to ITS running max, the replay must walk the source's block width: the tiled maker
does (its block_n is the source's), the simdgroup maker is fixed at 64-wide blocks, so a P-rounding
source with a narrower block routes to the tiled maker (exact; slower — the simdgroup half-block
variant is the ledgered performance follow-up). fp32 sources record no rounding point, only that Q is
multiplied by the scale before the dot (`q_product`), and Q is staged as that float32 product
(tests/test_where_attention_multiplies_by_its_scale.py).

The oracle below is the SOURCE's own online loop, block by block, in fp32 with the same exp2 /
inv_ln2 arithmetic, rounding P per block; only accumulation-order effects remain, which show up as
at most one output-ulp on a rare tie.
"""

import math
import sys

import pytest

try:
    import torch
    import triton

    import triton_msl
    import triton_msl.codegen._msl_templates as M
    from triton_msl.errors import MetalNonRecoverableError

    sys.path.insert(0, "tests")
    from test_fa_biased_routing import _biased_tri_fa
    from test_fa_bwd_routing import _build_lowerer

    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"
INV_LN2 = 1.4426950408889634
LN2 = 0.6931471824645996


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@pytest.fixture
def maker_spy(monkeypatch):
    hits = []
    for name in ("make_flash_attention_kernel_tiled", "make_flash_attention_kernel_simdgroup"):
        real = getattr(M, name)

        def spy(*a, _r=real, _n=name, **k):
            hits.append((_n.replace("make_flash_attention_kernel_", ""), k.get("round_q"), k.get("round_p")))
            return _r(*a, **k)

        monkeypatch.setattr(M, name, spy)
    return hits


def _problem(dtype, DIM=32, N=64, Hc=2, Hh=1, I=1, seed=5, sm=0.1234567):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(Hc, I, N, DIM, device=D).to(dtype) for _ in range(3))
    bias = torch.randn(Hc, N, N, device=D).to(dtype)
    mask = (torch.rand(Hc // Hh, I, N, device=D) < 0.2).to(torch.uint8)
    return dict(q=q, k=k, v=v, bias=bias, mask=mask, sm=sm, N=N, DIM=DIM, Hc=Hc, Hh=Hh, I=I, BK=32)


def _faithful(p, dtype):
    """The SOURCE's online loop (`_biased_tri_fa`), block by block, in fp32:
    q = round_dtype(q * round_dtype(scale)); per kv block: s = bias + q @ kᵀ; s *= inv_ln2; masks →
    neg_inf; m_new = max(m, rowmax s); e = exp2(s − m_new); l = l·exp2(m − m_new) + Σe;
    acc = acc·exp2(m − m_new) + round_dtype(e) @ v; m = m_new. Output round_dtype(acc / l);
    lse = m·ln2 + log l. For fp32 no rounding is applied anywhere."""
    q, k, v, bias, mask = p["q"].float(), p["k"].float(), p["v"].float(), p["bias"].float(), p["mask"]
    N, BK, Hc, Hh, I = p["N"], p["BK"], p["Hc"], p["Hh"], p["I"]
    rd = (lambda t: t) if dtype == torch.float32 else (lambda t: t.to(dtype).float())
    scale = p["sm"] if dtype == torch.float32 else torch.tensor(p["sm"], dtype=dtype).float().item()
    qs = rd(q * scale)
    mh = mask[torch.arange(Hc, device=D) // Hh].bool()  # [Hc, I, N]
    m = torch.full((Hc, I, N), -float("inf"), device=D)
    l = torch.zeros(Hc, I, N, device=D)
    acc = torch.zeros(Hc, I, N, p["DIM"], device=D)
    for k0 in range(0, N, BK):
        kb, vb, bb, mb = k[:, :, k0 : k0 + BK], v[:, :, k0 : k0 + BK], bias[:, k0 : k0 + BK, :], mh[:, :, k0 : k0 + BK]
        s = (
            torch.einsum("hijd,hikd->hijk", qs, kb) + bb.transpose(1, 2)[:, None]
            if False
            else torch.einsum("hijd,hikd->hijk", qs, kb) + bias[:, None, :, k0 : k0 + BK]
        )
        s = s * INV_LN2
        s = s.masked_fill(mb[:, :, None, :], -1e9)
        m_new = torch.maximum(m, s.amax(-1))
        e = torch.exp2(s - m_new[..., None])
        alpha = torch.exp2(m - m_new)
        l = l * alpha + e.sum(-1)
        acc = acc * alpha[..., None] + torch.einsum("hijk,hikd->hijd", rd(e), vb)
        m = m_new
    o = rd(acc / l[..., None])
    lse = m * LN2 + torch.log(l)
    return o, lse


def _launch(p, dtype):
    o = torch.zeros(p["Hc"], p["I"], p["N"], p["DIM"], device=D, dtype=dtype)
    lse = torch.zeros(p["Hc"], p["I"], p["N"], device=D)
    st = lambda t: tuple(t.stride())
    if hasattr(_biased_tri_fa, "device_caches"):
        _biased_tri_fa.device_caches.clear()
    _biased_tri_fa[(triton.cdiv(p["N"], 32), p["I"], p["Hc"])](
        o,
        *st(o),
        lse,
        *st(lse),
        p["q"],
        *st(p["q"]),
        p["k"],
        *st(p["k"]),
        p["v"],
        *st(p["v"]),
        p["bias"],
        *st(p["bias"]),
        p["mask"],
        *st(p["mask"]),
        p["sm"],
        -1e9,
        p["N"],
        p["Hh"],
        p["DIM"],
        32,
        p["BK"],
    )
    torch.mps.synchronize()
    return o.float(), lse


def _ulp(o_ref, dtype):
    bits = {torch.float32: 23, torch.float16: 10, torch.bfloat16: 7}[dtype]
    return o_ref.abs().clamp(min=2.0**-20).log2().floor().exp2() * (2.0**-bits)


@requires_gpu
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize(
    "dtype,maker,rq,rp",
    [
        (torch.float32, "simdgroup", "float", None),  # the float32 product: no rounding point
        (torch.bfloat16, "tiled", "bfloat", "bfloat"),
        (
            torch.float16,
            "tiled",
            "half",
            "half",
        ),  # P rounding with a 32-wide source block: exact only on the tiled maker
    ],
)
def test_biased_forward_replays_source_rounding(cold_gpu_caches, maker_spy, dtype, maker, rq, rp, N):
    """Correct-or-refuse against the SOURCE's online loop: the template must route to the named
    maker with the recorded rounding points and match the source's own rounded semantics — every
    element within one output ulp, and only a rare tie differing at all. On `cbff49c` the bf16 row is
    3.9e-3 off and the fp16 row 9.8e-4 off this oracle, on a large fraction of elements."""
    p = _problem(dtype, N=N)
    o_ref, lse_ref = _faithful(p, dtype)
    o, lse = _launch(p, dtype)
    assert maker_spy and maker_spy[-1] == (maker, rq, rp), maker_spy
    err = (o - o_ref).abs()
    frac = (err > 0).float().mean().item()
    if dtype == torch.float32:
        # no rounding point to replay: fp32 accumulation-order noise only (measured max 3.9e-7)
        assert err.max().item() < 2e-6, f"fp32 N={N}: max |template - source replay| = {err.max():.3e}"
    elif dtype == torch.bfloat16:
        # measured BIT-EXACT at N = 64 and 128 (0 differing elements); allow one output-ulp tie
        assert bool((err <= _ulp(o_ref, dtype) * 1.01).all()) and frac < 0.02, (
            f"bf16 N={N}: max err {err.max():.3e}, {frac * 100:.2f}% differ"
        )
    else:
        # fp16: the source evaluates exp2 on inv_ln2-scaled scores, the template exp on natural
        # scores (the equivalence packet 117 accepted); they differ by a few fp32 ulps BEFORE P is
        # rounded to fp16, so a rare P tie flips and moves one output row by ≤ one fp16 ulp of P
        # (measured: 1–3 P flips, 0.2 % of outputs, max 1.22e-4). A log2-unit template would make
        # this bit-exact too (ledgered). Anything systematic (the pre-164 9.8e-4 on 40 % of
        # elements) fails both bounds.
        assert err.max().item() < 2.5e-4 and frac < 0.01, (
            f"fp16 N={N}: max err {err.max():.3e}, {frac * 100:.2f}% differ (systematic = not replayed)"
        )
    fin = torch.isfinite(lse_ref)
    assert (lse[fin] - lse_ref[fin]).abs().max().item() < 2e-4


@requires_gpu
def test_biased_forward_fp32_records_the_product_only(tmp_path):
    """fp32 sources record no rounding point, only the float32 product q * scale; the emitted MSL
    carries no rounding, stages that product once and does not scale the scores again."""
    fn = _biased_tri_fa
    cex = {n: (32 if n in ("DIM", "BLOCK_J", "BLOCK_K") else 256) for n in [fn.arg_names[i] for i in fn.constexprs]}
    sig = {
        n: ("*u8" if "mask" in n or n == "m_ptr" else "*fp32")
        if n.endswith("_ptr")
        else "fp32"
        if n in ("sm_scale", "neg_inf")
        else "i32"
        for n in fn.arg_names
        if n not in cex
    }
    lw = _build_lowerer(fn, sig, cex)
    msl = lw.lower()
    assert "bfloat(" not in msl and "half(" not in msl
    assert msl.count("* float(float(scale))") == 1 and "s_scale = 1.0f" in msl
    assert getattr(lw, "_fa_rounding", {}) == {"q_product": "float"}


@requires_gpu
@pytest.mark.parametrize("dtype,rq,rp", [(torch.bfloat16, "bfloat", "bfloat"), (torch.float16, "half", "half")])
def test_verifier_records_the_rounding_points(dtype, rq, rp):
    """Direct lowering: for a bf16 / fp16 source the verifier records `round_q` (the scale multiply's
    element type) and `round_p` (the P conversion's target), and the emitted MSL neutralises the
    second scaling."""
    fn = _biased_tri_fa
    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    cex = {n: (32 if n in ("DIM", "BLOCK_J", "BLOCK_K") else 256) for n in [fn.arg_names[i] for i in fn.constexprs]}
    sig = {
        n: ("*u8" if "mask" in n or n == "m_ptr" else ("*fp32" if n.startswith("lse") else f"*{dt}"))
        if n.endswith("_ptr")
        else "fp32"
        if n in ("sm_scale", "neg_inf")
        else "i32"
        for n in fn.arg_names
        if n not in cex
    }
    lw = _build_lowerer(fn, sig, cex)
    msl = lw.lower()
    assert isinstance(msl, str) and "kernel void" in msl
    assert getattr(lw, "_fa_rounding", {}) == {"round_q": rq, "round_p": rp, "scale_chain": (rq,)}
    assert "s_scale = 1.0f" in msl
