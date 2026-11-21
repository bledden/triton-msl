"""Packet 174 (158 row 6) — the biased backward templates replay the source's fp16 / bf16 rounding.

Since packet 148 any dtype conversion on the backward value path refused ("the template computes in
fp32 and does not replay a narrowing"), so the trifast backward was fp32-only on Metal. The trifast
sources round at fixed points for narrow inputs; the proof now records them and the five makers
replay them:

  kv: K staged as dtype(K * dtype(scale)) [round_k]; P rounded before the dV dot ONLY (dS uses the
      fp32 P) [round_p]; dS rounded before the dK dot [round_ds]; dK *= scale in fp32; output casts.
  q:  the same K [round_k] — and because that rounded, scaled K also feeds the dQ dot, no scale is
      applied at the store; dS rounded before the dQ dot [round_ds]; delta stored rounded, used in
      fp32; output cast. The UNMODIFIED trifast `_bwd_q` computes delta = tl.sum(o * do) IN the
      input dtype (a bf16 / fp16 tree-sum: order-dependent rounding). The proved runtime D=32
      specialization is replayed by the simdgroup maker; this file's direct-helper layout remains
      a refusal control. The one-line variant tl.sum((o * do).to(tl.float32), 1) rounds each product
      [round_dprod] and accumulates in fp32.
  b:  Q staged as dtype(Q * dtype(scale)) [round_q]; dS stays fp32 (db += dS); output cast.
Exact widenings (the bias, the loaded delta) are transparent. Every other conversion refuses.

The GPU oracle is the source's own math in fp32 with those roundings applied where the source
applies them. Unlike the forward's online softmax, the backward's block structure only changes fp32
accumulation ORDER, so a whole-matrix formulation is the same function up to fp32 reassociation.
"""

import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

try:
    import torch
    import triton
    import triton.language as tl

    import triton_msl
    from triton_msl.errors import MetalNonRecoverableError

    sys.path.insert(0, "tests")
    from test_fa_bwd_routing import _bwd_b, _bwd_kv, _bwd_q, _build_lowerer

    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"
INV_LN2 = 1.4426950408889634
NEG = -1e9


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


# ------------------------------------------------------------------ source variants (fresh modules)


def _variant(kernel_name, replacements, name, tmp_dir):
    src = Path(__file__).with_name("test_fa_bwd_routing.py").read_text()
    start = src.index(f"@triton.jit\ndef {kernel_name}(")
    end = start + 20 + re.search(r"\n(?=@|def )", src[start + 20 :]).start()
    ks = src[start:end]
    for old, new in replacements:
        assert ks.count(old) == 1, old
        ks = ks.replace(old, new, 1)
    mp = Path(tmp_dir) / f"{name}.py"
    mp.write_text("import triton\nimport triton.language as tl\n\n" + ks + "\n")
    spec = importlib.util.spec_from_file_location(name, mp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, kernel_name)


_Q_F32DELTA = [
    ("delta = tl.sum(o_block * do_block, axis=1)", "delta = tl.sum((o_block * do_block).to(tl.float32), axis=1)")
]


def _q_f32delta(tmp_path):
    return _variant("_bwd_q", _Q_F32DELTA, "bwd_q_f32delta", tmp_path)


def _sig(fn, dt):
    cex = {
        n: (32 if n in ("DIM", "BLOCK_J", "BLOCK_K", "BLOCK_I") else 64)
        for n in [fn.arg_names[i] for i in fn.constexprs]
    }
    sig = {}
    for n in fn.arg_names:
        if n in cex:
            continue
        if n.endswith("_ptr"):
            sig[n] = "*u8" if ("mask" in n or n == "m_ptr") else ("*fp32" if n.startswith("l_") else f"*{dt}")
        elif n in ("sm_scale", "neg_inf"):
            sig[n] = "fp32"
        else:
            sig[n] = "i32"
    return sig, cex


def _lower(fn, dt):
    sig, cex = _sig(fn, dt)
    return _build_lowerer(fn, sig, cex).lower()


_ELEM = {"bf16": "bfloat", "fp16": "half"}
# The uniform scale is rounded to bfloat by a software round-to-nearest-even (integer arithmetic):
# the Metal compiler produced wrong values for `bfloat(scale)` on a kernel-uniform float in the kv
# makers while the identical text in the q makers was fine (bisected on the M4 Max, packet 174 §4).
_SCALE_R = {
    "bfloat": "(isnan(scale) ? scale : as_type<float>((as_type<uint>(scale) + 0x7FFFu + ((as_type<uint>(scale) >> 16u) & 1u)) & 0xFFFF0000u))",
    "half": "float(half(scale))",
}


def _bf16_rne_software(bits):
    """The MSL integer formula, as Python, for the CPU row below."""
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000) & 0xFFFFFFFF


@pytest.mark.parametrize("value", [0.1767767, 1.0, 3.0e38, float("inf"), -float("inf"), 0.0, -0.0])
def test_software_bf16_rounding_of_the_scale_matches_torch(value):
    """CPU: the integer round-to-nearest-even the makers emit for the bf16 scale equals torch's
    fp32→bf16 conversion on finite / infinite values (overflow rounds to inf as torch does)."""
    import struct

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    got = struct.unpack("<f", struct.pack("<I", _bf16_rne_software(bits)))[0]
    want = torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float().item()
    assert got == want or (got != got and want != want), (value, got, want)


def test_software_bf16_rounding_nan_is_guarded():
    """191: the bare integer formula maps a NaN payload (0x7fffffff) to 0x80000000 (−0.0); the
    emitted expression guards it with isnan(scale) so a NaN scale stays NaN. The integer formula's
    defect is pinned as such; the guard is pinned in the emitted text."""
    assert _bf16_rne_software(0x7FFFFFFF) == 0x80000000  # the defect, as a fact about the formula
    assert "(isnan(scale) ? scale : as_type<float>(" in _SCALE_R["bfloat"]


@requires_gpu
def test_nan_scale_propagates_on_the_bf16_kv_route(cold_gpu_caches):
    """GPU: with a NaN scale the source's scaled K is NaN, so P is NaN on every UNMASKED key
    (dV rows for those keys NaN) while masked keys still get the sentinel score (P = 0, dV rows
    zero), and dK is NaN everywhere through its store-time `* scale`. Pre-fix the −0.0 mapping
    produced finite numbers for all of them. (My first cut of this pin expected dV NaN everywhere:
    wrong — the mask select bypasses the NaN score; corrected to the source's semantics.)"""
    p = _problem(torch.bfloat16)
    p["sm"] = float("nan")
    N, DIM, I, Hc, Hh = p["N"], p["DIM"], p["I"], p["Hc"], p["Hh"]
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hc, I, N, DIM, device=D, dtype=torch.bfloat16)
    dv = torch.zeros_like(dk)
    q, k, v, bias, mask, do, lse, delta = (
        p["q"],
        p["k"],
        p["v"],
        p["bias"],
        p["mask"],
        p["do"],
        p["lse"],
        p["delta"].to(torch.bfloat16),
    )
    if hasattr(_bwd_kv, "device_caches"):
        _bwd_kv.device_caches.clear()
    _bwd_kv[(triton.cdiv(N, 32), I, Hc)](
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
        do,
        *st(do),
        dk,
        *st(dk),
        dv,
        *st(dv),
        p["sm"],
        NEG,
        N,
        Hh,
        DIM,
        N,
        32,
        32,
    )
    torch.mps.synchronize()
    assert bool(torch.isnan(dk.float()).all()), "dK: the store-time * scale makes every element NaN"
    unmasked = (~p["mh"])[..., None].expand(-1, -1, -1, DIM)  # key k of (h, i) present
    assert torch.equal(torch.isnan(dv.float()).cpu(), unmasked.cpu()), "dV: NaN exactly on unmasked keys"


def _staged(e, load):
    return f"float({e}(float({load}) * {_SCALE_R[e]}))"


# ------------------------------------------------------------------ direct-lowering rows (CPU)


@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_kv_replays_k_scale_p_and_ds_rounding(dt):
    """kv: K staged as the rounded product with the rounded scale (no scale after the dot), P
    rounded in place AFTER dS is formed (so dS used the fp32 P) and the dV dot reads the rounded P,
    dS rounded, outputs cast. Pre-174: refused at the first narrowing."""
    e = _ELEM[dt]
    msl = _lower(_bwd_kv, dt)
    assert _staged(e, "K[k_base + krow*k_sn + dd*k_sk]") in msl
    assert "tg_P[i]*scale" not in msl and "s*scale" not in msl
    assert f"tg_dS[i] = float({e}(tg_P[i] * (tg_dS[i] - tg_delta[jj])));" in msl
    assert f"tg_P[i] = float({e}(tg_P[i]));" in msl
    # order: dS formed, then P rounded, then the dV dot reads tg_P
    i_ds, i_pr, i_dv = (
        msl.index("tg_dS[i] = float("),
        msl.index(f"tg_P[i] = float({e}(tg_P[i]))"),
        msl.index("simdgroup_load(pf, tg_P"),
    )
    assert i_ds < i_pr < i_dv
    assert f"= {e}(tg_dS[i] * scale);" in msl and f"= {e}(tg_P[i]);" in msl


@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_b_replays_q_scale_rounding_and_keeps_ds_fp32(dt):
    e = _ELEM[dt]
    msl = _lower(_bwd_b, dt)
    assert _staged(e, "Q[q_hbase + ii*q_si + jrow*q_sm + dd*q_sk]") in msl
    assert "s*scale" not in msl and "db_acc[i] += tg_P[i] * (dp - tg_delta[jj]);" in msl
    assert f"= {e}(db_acc[i]);" in msl


@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_q_direct_helper_layout_refuses_the_narrow_delta_accumulation(dt):
    """The direct helper's one-value-per-lane layout is not the runtime specialization.

    A narrow reduction is admitted only when its own result metadata proves the
    eight-values-per-lane runtime tree; a source spelling and D=32 alone are insufficient.
    """
    with pytest.raises(MetalNonRecoverableError, match="eight-values-per-lane") as ei:
        _lower(_bwd_q, dt)
    assert "failed: source-layout,result-layout" in str(ei.value)


@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_q_fp32_delta_variant_replays_product_k_and_ds_rounding(dt, tmp_path):
    """tl.sum((o * do).to(tl.float32), 1): each product rounded before the fp32 rowsum
    (with fp32 accumulation), delta stored rounded, K as the rounded scaled product feeding BOTH dots
    (no store-time scale), dS rounded before the dQ dot, output cast."""
    e = _ELEM[dt]
    msl = _lower(_q_f32delta(tmp_path), dt)
    assert f"dl += float({e}(float(O[o_base + jrow*o_sm + d*o_sk]) * tg_dO[i*D + d]));" in msl
    assert f"= {e}(dl);" in msl
    assert _staged(e, "K[k_base + krow*k_sn + dd*k_sk]") in msl
    assert "tg_P[i]*scale" not in msl and "s*scale" not in msl
    assert f"tg_dS[i] = float({e}(tg_P[i] * (tg_dS[i] - tg_delta[jj])));" in msl
    assert f"= {e}(tg_P[i]);" in msl and "tg_P[i] * scale" not in msl


# Where each fp32 text multiplies by the scale: the operand the source scales, staged before the
# dot. A key past the length in dQ is zero times the scale, as the source's masked load gives it.
_FP32_STAGED = {
    "kv": "tg_K[i] = (krow < N_CTX) ? (float(K[k_base + krow*k_sn + dd*k_sk]) * scale) : 0.0f;",
    "q": "tg_K[i] = (krow < N_CTX) ? (float(K[k_base + krow*k_sn + dd*k_sk]) * scale) : (float(0.0f) * scale);",
    "b": "tg_Q[i] = (jrow < N_CTX) ? (float(Q[q_hbase + ii*q_si + jrow*q_sm + dd*q_sk]) * scale) : 0.0f;",
}


@pytest.mark.parametrize("fn,name", [(_bwd_kv, "kv"), (_bwd_q, "q"), (_bwd_b, "b")])
def test_fp32_sources_emit_no_rounding(fn, name):
    """Control on both sides: fp32 sources record nothing; the text carries no rounding and stages
    the operand times the scale in fp32, with no scale after the dot or at the dQ store."""
    msl = _lower(fn, "fp32")
    assert "bfloat(" not in msl and "half(" not in msl
    assert _FP32_STAGED[name] in msl
    assert "s*scale" not in msl and "tg_P[i]*scale" not in msl and "tg_P[i] * scale" not in msl


@pytest.mark.parametrize(
    "kernel,replacements,label",
    [
        (
            "_bwd_kv",
            [
                (
                    "dscores = sm_value * (dsm_value - delta[:, None])",
                    "dscores = sm_value.to(input_dtype) * (dsm_value - delta[:, None])",
                )
            ],
            "P rounded where it feeds dS",
        ),
        (
            "_bwd_kv",
            [
                (
                    'dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee")',
                    'dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee")\n        dsm_value = dsm_value.to(input_dtype).to(tl.float32)',
                )
            ],
            "dP rounded",
        ),
        (
            "_bwd_b",
            [("db_block += dscores", "db_block += dscores.to(input_dtype)")],
            "dS rounded before the dbias accumulation",
        ),
        (
            "_bwd_kv",
            [
                (
                    'dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee")',
                    'dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), (do.to(tl.float32) * 2.0).to(input_dtype), input_precision="ieee")',
                )
            ],
            "dO scaled and re-rounded",
        ),
    ],
)
def test_unreplayed_roundings_still_refuse(kernel, replacements, label, tmp_path):
    """Roundings the source does NOT make are not admitted: a narrowing where dS would consume a
    rounded P, a rounded dP, a rounded dS in the dbias accumulation, a dO scaled in fp32 and
    re-rounded before the dV dot. Each refuses (bf16) before any maker. (A widening immediately
    narrowed back to the same dtype is folded by the frontend — the program is unchanged — so it
    is not a witness.)"""
    fn = _variant(kernel, replacements, "bwd_mut_" + "".join(ch if ch.isalnum() else "_" for ch in label), tmp_path)
    with pytest.raises(MetalNonRecoverableError):
        _lower(fn, "bf16")


# ------------------------------------------------------------------ GPU rows against the source's math


def _rd(dtype):
    return (lambda t: t) if dtype == torch.float32 else (lambda t: t.to(dtype).float())


def _problem(dtype, Hc=2, Hh=1, I=2, N=64, DIM=32, seed=3):
    torch.manual_seed(seed)
    sm = 1.0 / math.sqrt(DIM)
    q, k, v, do = (torch.randn(Hc, I, N, DIM, device=D).to(dtype) for _ in range(4))
    bias = torch.randn(Hc, N, N, device=D).to(dtype)
    mask = (torch.rand(Hc // Hh, I, N, device=D) < 0.15).to(torch.uint8)
    # lse from the fp32 forward on the rounded inputs (what the forward stores); delta rounded to dtype
    mh = mask[torch.arange(Hc, device=D) // Hh].bool()
    rd = _rd(dtype)
    qs = rd(q.float() * (torch.tensor(sm, dtype=dtype).float().item() if dtype != torch.float32 else sm))
    raw = torch.einsum("hijd,hikd->hijk", qs, k.float()) + bias.float()[:, None]
    raw = raw.masked_fill(mh[:, :, None, :], NEG)
    lse = torch.logsumexp(raw, -1)
    p = torch.exp(raw - lse[..., None])
    o = rd(torch.einsum("hijk,hikd->hijd", rd(p), v.float()))
    delta = (o * do.float()).sum(-1)
    return dict(
        q=q,
        k=k,
        v=v,
        do=do,
        bias=bias,
        mask=mask,
        mh=mh,
        sm=sm,
        lse=lse,
        o=o.to(dtype),
        delta=rd(delta),
        N=N,
        DIM=DIM,
        Hc=Hc,
        Hh=Hh,
        I=I,
    )


def _oracle_kv(p, dtype):
    rd = _rd(dtype)
    q, k, v, do, bias, mh = p["q"].float(), p["k"].float(), p["v"].float(), p["do"].float(), p["bias"].float(), p["mh"]
    sm = torch.tensor(p["sm"], dtype=dtype).float().item() if dtype != torch.float32 else p["sm"]
    ks = rd(k * sm)
    s = torch.einsum("hijd,hikd->hijk", q, ks) + bias[:, None]
    s = s.masked_fill(mh[:, :, None, :], NEG)
    P = torch.exp2((s - p["lse"][..., None]) * INV_LN2)
    dV = torch.einsum("hijk,hijd->hikd", rd(P), do)
    dP = torch.einsum("hijd,hikd->hijk", do, v)
    dS = rd(P * (dP - p["delta"].float()[..., None]))
    dK = torch.einsum("hijk,hijd->hikd", dS, q) * p["sm"]
    return rd(dK), rd(dV)


def _oracle_q(p, dtype):
    rd = _rd(dtype)
    q, k, v, do, bias, mh, o = (
        p["q"].float(),
        p["k"].float(),
        p["v"].float(),
        p["do"].float(),
        p["bias"].float(),
        p["mh"],
        p["o"].float(),
    )
    sm = torch.tensor(p["sm"], dtype=dtype).float().item() if dtype != torch.float32 else p["sm"]
    delta = rd(o * do).sum(-1)  # products rounded, fp32 sum
    ks = rd(k * sm)
    s = torch.einsum("hijd,hikd->hijk", q, ks) + bias[:, None]
    s = s.masked_fill(mh[:, :, None, :], NEG)
    P = torch.exp2((s - p["lse"][..., None]) * INV_LN2)
    dP = torch.einsum("hijd,hikd->hijk", do, v)
    dS = rd(P * (dP - delta[..., None]))
    dQ = torch.einsum("hijk,hikd->hijd", dS, ks)  # the rounded, scaled K; no store scale
    return rd(dQ), rd(delta)


def _oracle_b(p, dtype):
    rd = _rd(dtype)
    q, k, v, do, bias, mh = p["q"].float(), p["k"].float(), p["v"].float(), p["do"].float(), p["bias"].float(), p["mh"]
    sm = torch.tensor(p["sm"], dtype=dtype).float().item() if dtype != torch.float32 else p["sm"]
    qs = rd(q * sm)
    s = torch.einsum("hijd,hikd->hijk", qs, k) + bias[:, None]
    s = s.masked_fill(mh[:, :, None, :], NEG)
    P = torch.exp2((s - p["lse"][..., None]) * INV_LN2)
    dP = torch.einsum("hijd,hikd->hijk", do, v)
    dS = P * (dP - p["delta"].float()[..., None])
    return rd(dS.sum(1))  # sum over the triangle-i axis


def _ulp(ref, dtype):
    bits = {torch.float32: 23, torch.float16: 10, torch.bfloat16: 7}[dtype]
    return ref.abs().clamp(min=2.0**-20).log2().floor().exp2() * (2.0**-bits)


def _check(got, ref, dtype, label):
    """bf16: within one output ulp everywhere, < 2 % of elements differing (fp32 reassociation
    can flip a rare rounding); fp16: within two ulps, < 2 % differing; fp32 rows are controls
    against autograd elsewhere and not exercised here."""
    err = (got.float() - ref).abs()
    frac = (err > 0).float().mean().item()
    tol = _ulp(ref, dtype) * (1.01 if dtype == torch.bfloat16 else 2.01)
    assert bool((err <= tol).all()) and frac < 0.02, (
        f"{label}: max err {err.max():.3e} (ref max {ref.abs().max():.2f}), {frac * 100:.2f}% differ"
    )


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_kv_computes_the_source_rounding_on_gpu(cold_gpu_caches, dtype):
    p = _problem(dtype)
    N, DIM, I, Hc, Hh = p["N"], p["DIM"], p["I"], p["Hc"], p["Hh"]
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hc, I, N, DIM, device=D, dtype=dtype)
    dv = torch.zeros_like(dk)
    q, k, v, bias, mask, do, lse, delta = (
        p["q"],
        p["k"],
        p["v"],
        p["bias"],
        p["mask"],
        p["do"],
        p["lse"],
        p["delta"].to(dtype),
    )
    if hasattr(_bwd_kv, "device_caches"):
        _bwd_kv.device_caches.clear()
    _bwd_kv[(triton.cdiv(N, 32), I, Hc)](
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
        do,
        *st(do),
        dk,
        *st(dk),
        dv,
        *st(dv),
        p["sm"],
        NEG,
        N,
        Hh,
        DIM,
        N,
        32,
        32,
    )
    torch.mps.synchronize()
    dk_ref, dv_ref = _oracle_kv(p, dtype)
    _check(dv, dv_ref, dtype, "dV")
    _check(dk, dk_ref, dtype, "dK")


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_q_fp32_delta_variant_computes_the_source_rounding_on_gpu(cold_gpu_caches, tmp_path, dtype):
    fn = _q_f32delta(tmp_path)
    p = _problem(dtype)
    N, DIM, I, Hc, Hh = p["N"], p["DIM"], p["I"], p["Hc"], p["Hh"]
    st = lambda t: tuple(t.stride())
    dq = torch.zeros(Hc, I, N, DIM, device=D, dtype=dtype)
    delta = torch.zeros(Hc, I, N, device=D, dtype=dtype)
    q, k, v, bias, mask, do, lse, o = p["q"], p["k"], p["v"], p["bias"], p["mask"], p["do"], p["lse"], p["o"]
    fn[(triton.cdiv(N, 32), I, Hc)](
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
        p["sm"],
        NEG,
        N,
        Hh,
        DIM,
        N,
        32,
        32,
    )
    torch.mps.synchronize()
    dq_ref, delta_ref = _oracle_q(p, dtype)
    _check(delta, delta_ref, dtype, "delta")
    _check(dq, dq_ref, dtype, "dQ")


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_b_computes_the_source_rounding_on_gpu(cold_gpu_caches, dtype):
    p = _problem(dtype, I=64)  # _bwd_b loops i over N: I == N
    N, DIM, Hc, Hh = p["N"], p["DIM"], p["Hc"], p["Hh"]
    st = lambda t: tuple(t.stride())
    db = torch.zeros(Hc, N, N, device=D, dtype=dtype)
    q, k, v, bias, mask, do, lse, delta = (
        p["q"],
        p["k"],
        p["v"],
        p["bias"],
        p["mask"],
        p["do"],
        p["lse"],
        p["delta"].to(dtype),
    )
    if hasattr(_bwd_b, "device_caches"):
        _bwd_b.device_caches.clear()
    _bwd_b[(triton.cdiv(N, 32), triton.cdiv(N, 32), Hc)](
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
        do,
        *st(do),
        db,
        *st(db),
        p["sm"],
        NEG,
        Hh,
        N,
        DIM,
        N,
        32,
        32,
    )
    torch.mps.synchronize()
    _check(db, _oracle_b(p, dtype), dtype, "dbias")
