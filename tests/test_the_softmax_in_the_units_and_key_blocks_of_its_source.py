"""The softmax of a forward template, in the units and in the key blocks of its source.

Two things a source's online softmax does that a template in other units, or with other
tile widths, does not do by itself:

* The biased source multiplies its float32 scores by a constant (``scores *= inv_ln2``) and
  takes ``exp2``. A score between about 2.36e38 and 3.4e38 overflows in that multiply and
  the row is NaN; a row whose every score lies below -2.36e38 is NaN the same way. The
  template kept natural units, where both rows are finite, and computed any constant within
  1e-4 of 1 / ln 2 as if it were 1 / ln 2. It now forms the source's product, takes the
  source's exponential and stores the maximum times the source's own constant. An exponent
  that carries a multiplier of its own is refused on this route: it was ignored.

* A source row whose running maximum is still -inf when one of the SOURCE's key blocks ends
  subtracts -inf from -inf and is NaN from there on. A template tile that is wider than the
  source's block saw later keys in time and stayed finite; one that is narrower was NaN
  where the source's block still had a finite key to come. That reaches the biased route
  with finite tensors (the first keys masked with an infinite sentinel) and the dense,
  variable-length and MLA routes with keys that are not finite.

The biased results are compared with the source's own recurrence replayed in float32, block
by block. On the other routes the first ``w`` keys score -inf for every row (the keys are
+inf and Q is negative), and the source's recurrence leaves exactly these rows NaN: every row
when ``w`` reaches its key block, the rows that see no other key when it is causal, and no
row otherwise.
"""

import importlib.util
import inspect
import math
import re
import sys

import pytest

try:
    import numpy as np
    import torch
    import triton

    import tests.test_fa_biased_routing as biased_source
    import tests.test_flash_attention as dense_source
    import tests.test_mla_value_paths as mla_source
    import tests.test_varlen_fa_routing as varlen_source
    import triton_msl.codegen._msl_templates as templates
    from triton_msl.errors import MetalNonRecoverableError

    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_MPS = False

requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Metal GPU needed")
INF = float("inf")
INV_LN2 = 1.4426950408889634
LN2 = 0.6931471824645996
TYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}
TOLERANCE = {"f32": 2e-6, "f16": 3e-3, "bf16": 2e-2}
MAKERS = (
    "make_flash_attention_kernel_tiled",
    "make_flash_attention_kernel_simdgroup",
    "make_varlen_flash_attention",
    "make_varlen_flash_attention_mma",
)
_CONSTANTS = "    inv_ln2: tl.constexpr = 1.4426950408889634\n    ln2: tl.constexpr = 0.6931471824645996\n"
_MULTIPLY = "        if not BAD_TEMP:\n            scores *= inv_ln2  # inv_ln2 + exp2 == natural softmax\n"
_P = "        exp_scores = tl.math.exp2(scores)\n"
_ALPHA = "        exp_scale = tl.math.exp2(scores_max - block_max)\n"
_LSE = "    lse = (scores_max * ln2) + tl.log(sm_denom)\n"
NATURAL = [
    (_MULTIPLY, ""),
    (_P, "        exp_scores = tl.math.exp(scores)\n"),
    (_ALPHA, "        exp_scale = tl.math.exp(scores_max - block_max)\n"),
    (_LSE, "    lse = scores_max + tl.log(sm_denom)\n"),
]
EXPONENT_MULTIPLIER = [
    (_MULTIPLY, ""),
    (_P, "        exp_scores = tl.math.exp(scores * 1.5)\n"),
    (_ALPHA, "        exp_scale = tl.math.exp((scores_max - block_max) * 1.5)\n"),
    (_LSE, "    lse = scores_max + tl.log(sm_denom)\n"),
]


@pytest.fixture(scope="module")
def variant_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("softmax_units")


_VARIANTS = {}


def _variant(name, replacements, root):
    """The biased kernel's own text with the replacements, as a module of its own."""
    fn = biased_source._biased_fa
    if name not in _VARIANTS:
        if not replacements:
            _VARIANTS[name] = fn
        else:
            src = inspect.getsource(fn.fn)
            for old, new in replacements:
                assert src.count(old) == 1, old
                src = src.replace(old, new)
            path = root / f"biased_{name}.py"
            path.write_text("import triton\nimport triton.language as tl\n\n\n" + src)
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = mod
            spec.loader.exec_module(mod)
            _VARIANTS[name] = getattr(mod, fn.fn.__name__)
    return _VARIANTS[name]


def _with_constants(c, k, root):
    text = f"    inv_ln2: tl.constexpr = {c!r}\n    ln2: tl.constexpr = {k!r}\n"
    return _variant(f"c{c!r}".replace(".", "_"), [(_CONSTANTS, text)], root)


@pytest.fixture
def cold_caches(tmp_path, monkeypatch):
    """Nothing compiled before this test is reused, on disk or in the kernels' own caches."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    kernels = [biased_source._biased_fa, dense_source._flash_attn_fwd, mla_source._mla_value_path]
    kernels += [varlen_source._varlen_fwd, varlen_source._varlen_causal_fwd, *_VARIANTS.values()]
    for kernel in kernels:
        kernel.device_caches.clear()


class _Hits(list):
    calls = None


@pytest.fixture
def maker_spy(monkeypatch):
    """(maker, its keyword arguments, the text it returned) for every template made."""
    hits, calls = _Hits(), []
    hits.calls = calls
    for name in MAKERS:
        real = getattr(templates, name)

        def spy(*a, _r=real, _n=name, **k):
            text = _r(*a, **k)
            hits.append((_n, k, text))
            calls.append((_r, a, k))
            return text

        monkeypatch.setattr(templates, name, spy)
    return hits


def _code(msl):
    return re.sub(r"//[^\n]*", "", msl)


def _exponentials(msl):
    return set(re.findall(r"(?<![A-Za-z0-9_])(exp2?)\(", _code(msl)))


# ---------------------------------------------------------------------------------------
# the biased source, replayed
# ---------------------------------------------------------------------------------------


def _replay(q, k, v, b, mask, sm, neg_inf, n, block, c=INV_LN2, lse_unit=LN2, base=2.0):
    """The source's recurrence for one (z, h), block by block, in float32."""
    f32 = np.float32
    power = np.exp2 if base == 2.0 else np.exp
    with np.errstate(all="ignore"):
        qs = (q.astype(f32) * f32(sm)).astype(f32)
        rows, dim = q.shape
        m = np.full(rows, -np.inf, f32)
        den = np.zeros(rows, f32)
        acc = np.zeros((rows, dim), f32)
        for start in range(0, n, block):
            cols = np.arange(start, start + block)
            ok = cols < n
            kb = np.zeros((block, dim), f32)
            vb = np.zeros((block, dim), f32)
            bb = np.zeros((rows, block), f32)
            mb = np.zeros(block, bool)
            kb[ok], vb[ok], bb[:, ok], mb[ok] = k[cols[ok]], v[cols[ok]], b[:, cols[ok]], mask[cols[ok]]
            s = (bb + (qs.astype(np.float64) @ kb.astype(np.float64).T).astype(f32)).astype(f32)
            if c is not None:
                s = (s * f32(c)).astype(f32)
            s = np.where(mb[None, :], f32(neg_inf), s)
            s = np.where(ok[None, :], s, f32(neg_inf))
            m_new = np.maximum(m, s.max(1))
            p = power((s - m_new[:, None]).astype(f32)).astype(f32)
            alpha = power((m - m_new).astype(f32)).astype(f32)
            den = (den * alpha + p.sum(1, dtype=f32)).astype(f32)
            acc = (acc * alpha[:, None] + (p.astype(np.float64) @ vb.astype(np.float64)).astype(f32)).astype(f32)
            m = m_new
        out = (acc / den[:, None]).astype(f32)
        lse = (m * f32(lse_unit) + np.log(den).astype(f32)).astype(f32)
    return out, lse


def _tensors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    make = lambda *shape: rng.standard_normal(shape).astype(np.float32)  # noqa: E731
    return make(n, dim), make(n, dim), make(n, dim), make(n, n), rng.random(n) < 0.25


def _launch(kernel, type_name, tensors, n, dim, block=32, neg_inf=-1e9):
    """Out, Lse and the tensors as the kernel read them, for Z = H = 1."""
    dtype = getattr(torch, TYPES[type_name])
    q, k, v, b, mask = tensors
    put = lambda x: torch.from_numpy(x).to("mps").to(dtype)[None, None]  # noqa: E731
    q_, k_, v_, b_ = put(q), put(k), put(v), put(b)
    m_ = torch.from_numpy(mask.astype(np.uint8)).to("mps")[None, None]
    out = torch.full((1, 1, n, dim), 7.0, device="mps", dtype=dtype)
    lse = torch.full((1, 1, n), 7.0, device="mps")
    st = lambda t: tuple(t.stride())  # noqa: E731
    args = [out, *st(out), lse, *st(lse), q_, *st(q_), k_, *st(k_), v_, *st(v_), b_, *st(b_), m_, *st(m_)]
    kernel[(triton.cdiv(n, 32), 1)](*args, 1.0 / math.sqrt(dim), neg_inf, 1, 1, n, dim, 32, block)
    torch.mps.synchronize()
    back = lambda x: x.float().cpu().numpy()[0, 0]  # noqa: E731
    return back(out), back(lse), (back(q_), back(k_), back(v_), back(b_), mask)


def _same(got, want, tolerance, label):
    got_nan, want_nan = np.isnan(got), np.isnan(want)
    assert (got_nan == want_nan).all(), (
        f"{label}: {int(got_nan.sum())} NaN where the source has {int(want_nan.sum())}, "
        f"{int((got_nan != want_nan).sum())} cells differ"
    )
    finite = ~want_nan
    if finite.any():
        with np.errstate(all="ignore"):
            err = np.abs(got[finite].astype(np.float64) - want[finite].astype(np.float64))
        err = err[~np.isnan(err)]
        scale = max(1.0, float(np.abs(want[finite][np.isfinite(want[finite])]).max(initial=0.0)))
        assert err.size == 0 or err.max() <= tolerance * scale, f"{label}: {err.max():.3e} of {scale:.3e}"


def _check(kernel, type_name, tensors, n, dim=32, block=32, neg_inf=-1e9, tolerance=None, **source):
    out, lse, read = _launch(kernel, type_name, tensors, n, dim, block, neg_inf)
    want_out, want_lse = _replay(*read, 1.0 / math.sqrt(dim), neg_inf, n, block, **source)
    tolerance = TOLERANCE[type_name] if tolerance is None else tolerance
    _same(out, want_out, tolerance, "Out")
    _same(lse, want_lse, tolerance, "Lse")
    return want_out


def _ordinary(q, k, v, b, mask):
    return None


def _above(q, k, v, b, mask):
    b[3, 5] = 3.0e38  # times 1 / ln 2 this is +inf: row 3 is NaN in the source
    mask[5] = False
    return 3


def _under(q, k, v, b, mask):
    b[3, 5] = 2.2e38  # times 1 / ln 2 this is 3.17e38: finite
    mask[5] = False
    return None


def _all_below(q, k, v, b, mask):
    b[4, :] = -3.0e38  # every score of row 4 is -inf after the multiply
    mask[:] = False
    return 4


def _all_under(q, k, v, b, mask):
    b[4, :] = -2.2e38
    mask[:] = False
    return None


def _all_masked(q, k, v, b, mask):
    mask[:] = True  # every key holds the finite sentinel: the stored maximum is the sentinel
    return None


FORMS = {
    "ordinary": _ordinary,
    "above": _above,
    "under": _under,
    "all-below": _all_below,
    "all-under": _all_under,
    "all-masked": _all_masked,
}
UNIT_CASES = [(t, f) for t in ("f32", "bf16") for f in FORMS] + [("f16", "ordinary"), ("f16", "all-masked")]


@requires_mps
@pytest.mark.parametrize("n", [64, 50])
@pytest.mark.parametrize("type_name,form", UNIT_CASES)
def test_biased_forward_overflows_where_its_source_does(type_name, form, n):
    tensors = _tensors(n, 32)
    nan_row = FORMS[form](*tensors)
    want = _check(biased_source._biased_fa, type_name, tensors, n)
    assert np.isnan(want).any(1).tolist() == [r == nan_row for r in range(n)]  # the replay itself


@requires_mps
@pytest.mark.parametrize("c", [1.4427, 1.4428, 1.4426, 1.44274])
def test_a_constant_near_one_over_ln2_is_the_sources_constant(c, variant_dir):
    """Natural units gave 4e-6 to 8e-5 of the largest value here; the source's product, 4e-7."""
    k = float(np.float32(1.0 / c))
    _check(_with_constants(c, k, variant_dir), "f32", _tensors(64, 32), 64, c=c, lse_unit=k)


@requires_mps
@pytest.mark.parametrize("type_name,maker", [("f32", MAKERS[1]), ("bf16", MAKERS[0])])
def test_biased_maker_is_given_the_source_units(type_name, maker, cold_caches, maker_spy):
    _check(biased_source._biased_fa, type_name, _tensors(64, 32), 64)
    assert [name for name, _, _ in maker_spy] == [maker]
    _, kwargs, msl = maker_spy[0]
    c, base, lse_unit = kwargs["score_unit"]
    assert (np.float32(c), base, np.float32(lse_unit)) == (np.float32(INV_LN2), 2.0, np.float32(LN2))
    assert kwargs["neg_inf_unit"] == 1.0
    assert _exponentials(msl) == {"exp2"}
    assert f"s *= {float(c)!r}f;" in msl and f" * {float(lse_unit)!r}f + log(" in msl


@requires_mps
@pytest.mark.parametrize("type_name,maker", [("f32", MAKERS[1]), ("bf16", MAKERS[0])])
def test_a_natural_source_keeps_natural_units(type_name, maker, variant_dir, cold_caches, maker_spy):
    kernel = _variant("natural", NATURAL, variant_dir)
    _check(kernel, type_name, _tensors(64, 32), 64, c=None, lse_unit=1.0, base=math.e)
    assert [name for name, _, _ in maker_spy] == [maker]
    _, kwargs, msl = maker_spy[0]
    assert kwargs["score_unit"] is None and kwargs["neg_inf_unit"] == 1.0
    assert _exponentials(msl) == {"exp"} and "s *= " not in msl


@requires_mps
@pytest.mark.parametrize("c", [1.00005, 0.99995])
def test_a_natural_source_with_a_constant_near_one_takes_it(c, variant_dir, cold_caches, maker_spy):
    """The constant was dropped: 5e-5 of every score."""
    k = float(np.float32(1.0 / c))
    edits = [(old, new) for old, new in NATURAL if old not in (_MULTIPLY, _LSE)]
    edits += [(_CONSTANTS, f"    inv_ln2: tl.constexpr = {c!r}\n    ln2: tl.constexpr = {k!r}\n")]
    kernel = _variant(f"natural{c!r}".replace(".", "_"), edits, variant_dir)
    tensors = _tensors(64, 32)
    tensors[3][:] *= 300.0  # scores of a few hundred, where 5e-5 of one moves a probability
    _check(kernel, "f32", tensors, 64, c=c, lse_unit=k, base=math.e)
    c_given, base, k_given = maker_spy[0][1]["score_unit"]
    assert (np.float32(c_given), base, np.float32(k_given)) == (np.float32(c), math.e, np.float32(k))
    assert _exponentials(maker_spy[0][2]) == {"exp"}


@requires_mps
@pytest.mark.parametrize("type_name", ["f32", "bf16"])
def test_the_stored_maximum_takes_the_sources_own_constant(type_name, variant_dir):
    """Not 1 / c: a source constant 9e-7 away from it, on a maximum near 6e6."""
    k = float(np.float32(LN2 * (1.0 + 9e-7)))
    kernel = _with_constants(INV_LN2, k, variant_dir)
    n, dim = 64, 32
    tensors = _tensors(n, dim)
    tensors[3][:] += 4.0e6
    _, lse, read = _launch(kernel, type_name, tensors, n, dim)
    _, want = _replay(*read, 1.0 / math.sqrt(dim), -1e9, n, 32, lse_unit=k)
    assert np.abs(lse.astype(np.float64) - want).max() <= 1.5
    assert np.abs(want.astype(np.float64) - _replay(*read, 1.0 / math.sqrt(dim), -1e9, n, 32)[1]).min() > 3.0


@requires_mps
@pytest.mark.parametrize("type_name", ["f32", "bf16"])
def test_the_makers_refuse_source_units_beside_a_converted_sentinel(type_name, cold_caches, maker_spy):
    _launch(biased_source._biased_fa, type_name, _tensors(64, 32), 64, 32)
    ((real, args, kwargs),) = maker_spy.calls
    with pytest.raises(ValueError, match="score_unit needs a biased or masked score"):
        real(*args, **{**kwargs, "neg_inf_unit": 0.5})
    assert real(*args, **{**kwargs, "score_unit": None, "neg_inf_unit": 0.5}) != maker_spy[0][2]


@requires_mps
def test_an_exponent_multiplier_is_refused_on_the_biased_route(variant_dir, cold_caches, maker_spy):
    kernel = _variant("exponent", EXPONENT_MULTIPLIER, variant_dir)
    with pytest.raises(MetalNonRecoverableError, match="a multiplier inside the exponent"):
        _launch(kernel, "f32", _tensors(64, 32), 64, 32)
    assert maker_spy == []


# ---------------------------------------------------------------------------------------
# the biased source's key blocks
# ---------------------------------------------------------------------------------------


def _masked(lo, hi):
    def form(q, k, v, b, mask):
        mask[:] = False
        mask[lo:hi] = True
        return -INF  # the sentinel

    return form


def _bias(lo, hi):
    def form(q, k, v, b, mask):
        mask[:] = False
        b[4, lo:hi] = -INF
        return -1e9

    return form


BLOCK_FORMS = {
    "mask-16": _masked(0, 16),
    "mask-32": _masked(0, 32),
    "mask-64": _masked(0, 64),
    "mask-32-64": _masked(32, 64),
    "bias-32": _bias(0, 32),
    "bias-64": _bias(0, 64),
}


@requires_mps
@pytest.mark.parametrize("n", [50, 160])
@pytest.mark.parametrize("form", list(BLOCK_FORMS))
@pytest.mark.parametrize("block", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("type_name", ["f32", "bf16", "f16"])
def test_biased_row_is_nan_where_a_source_block_ends_on_minus_infinity(type_name, block, form, n):
    tensors = _tensors(n, 32)
    neg_inf = BLOCK_FORMS[form](*tensors)
    _check(biased_source._biased_fa, type_name, tensors, n, block=block, neg_inf=neg_inf)


@requires_mps
@pytest.mark.parametrize("block,text", [(16, "(j % 2u) == 1u"), (32, "(j % 4u) == 3u")])
def test_simdgroup_text_ends_the_sources_block(block, text, cold_caches, maker_spy):
    tensors = _tensors(160, 32)
    _check(biased_source._biased_fa, "f32", tensors, 160, block=block)
    assert [name for name, _, _ in maker_spy] == [MAKERS[1]]
    assert text in maker_spy[0][2] and "m_new == -INFINITY" in maker_spy[0][2]


@requires_mps
def test_a_source_block_wider_than_the_tile_takes_the_tiled_maker(cold_caches, maker_spy):
    """A 128-key source block is one accumulator update; the 64-key simdgroup tile would make it
    two (test_attention_forwards_keep_the_source_special_values)."""
    _check(biased_source._biased_fa, "f32", _tensors(160, 32), 160, block=128)
    assert [name for name, _, _ in maker_spy] == [MAKERS[0]]


@requires_mps
def test_a_source_block_below_eight_keys_takes_the_tiled_maker(cold_caches, maker_spy):
    """The simdgroup tile reduces its maximum eight keys at a time: it cannot end a block of 4."""
    _check(biased_source._biased_fa, "f32", _tensors(160, 32), 160, block=4)
    assert [name for name, _, _ in maker_spy] == [MAKERS[0]]


EXPONENT_FORMS = {
    "alpha": [(_ALPHA, "        exp_scale = tl.math.exp((scores_max - block_max) * 0.6931471805599453)\n")],
    "p": [(_P, "        exp_scores = tl.math.exp(scores * 0.6931471805599453)\n")],
}


@requires_mps
@pytest.mark.parametrize("which", list(EXPONENT_FORMS))
def test_one_exponential_spelled_with_a_multiplier_is_refused(which, variant_dir, cold_caches, maker_spy):
    """exp(x * ln 2) beside exp2(x): the bases agree, the operations are not the template's."""
    kernel = _variant(f"exponent_{which}", EXPONENT_FORMS[which], variant_dir)
    with pytest.raises(MetalNonRecoverableError, match="a multiplier inside the exponent"):
        _launch(kernel, "f32", _tensors(64, 32), 64, 32)
    assert maker_spy == []


@requires_mps
def test_simdgroup_text_at_the_sources_own_width_is_unchanged(cold_caches, maker_spy):
    _check(biased_source._biased_fa, "f32", _tensors(160, 32), 160, block=64)
    assert [name for name, _, _ in maker_spy] == [MAKERS[1]]
    assert "m_new == -INFINITY" not in maker_spy[0][2]


# ---------------------------------------------------------------------------------------
# the key blocks of the dense, variable-length and MLA sources
# ---------------------------------------------------------------------------------------


def _nan_rows(out, row_axis):
    nan = torch.isnan(out.float()).movedim(row_axis, 0).flatten(1).cpu()
    assert (nan.all(1) == nan.any(1)).all(), "a row is NaN in part"
    return nan.all(1).tolist()


def _expected_rows(rows, w, block, causal):
    """Rows whose first source block holds no key above -inf."""
    return [w >= block or (causal and r < w) for r in range(rows)]


def _negative(*shape, dtype):
    return -torch.rand(*shape, device="mps", dtype=dtype) - 0.1


@requires_mps
@pytest.mark.parametrize("w", [16, 32, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "type_name,dim,block", [(t, d, b) for t in ("f32", "f16") for d, b in ((64, 32), (64, 64), (128, 32))]
)
def test_dense_row_is_nan_where_a_source_block_ends_on_minus_infinity(type_name, dim, block, causal, w):
    torch.manual_seed(0)
    dtype, n = getattr(torch, TYPES[type_name]), 128
    q = _negative(1, 2, n, dim, dtype=dtype)
    k = torch.randn(1, 2, n, dim, device="mps", dtype=dtype)
    k[:, :, :w] = INF
    v = torch.randn(1, 2, n, dim, device="mps", dtype=dtype)
    out = torch.full_like(q, 7.0)
    strides = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    dense_source._flash_attn_fwd[(n // 32, 2)](
        q, k, v, out, *strides, 1, 2, n, BLOCK_M=32, BLOCK_N=block, HEAD_DIM=dim, IS_CAUSAL=causal
    )
    torch.mps.synchronize()
    assert _nan_rows(out, -2) == _expected_rows(n, w, block, causal)


@requires_mps
@pytest.mark.parametrize("w", [16, 32, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("type_name", ["f32", "f16", "bf16"])
def test_variable_length_row_is_nan_where_a_source_block_ends_on_minus_infinity(type_name, dim, causal, w):
    torch.manual_seed(0)
    dtype, t, heads = getattr(torch, TYPES[type_name]), 128, 2
    kernel = varlen_source._varlen_causal_fwd if causal else varlen_source._varlen_fwd
    cu = torch.tensor([0, t], device="mps", dtype=torch.int32)
    q = _negative(t, heads, dim, dtype=dtype)
    k = torch.randn(t, heads, dim, device="mps", dtype=dtype)
    k[:w] = INF
    v = torch.randn(t, heads, dim, device="mps", dtype=dtype)
    out = torch.full_like(q, 7.0)
    strides = [*q.stride(), *k.stride(), *v.stride(), *out.stride()]
    kernel[(triton.cdiv(t, 32), heads)](q, k, v, out, cu, cu, *strides, heads, t, 1.0 / math.sqrt(dim), 32, 32, dim)
    torch.mps.synchronize()
    assert _nan_rows(out, 0) == _expected_rows(t, w, 32, causal)


@requires_mps
@pytest.mark.parametrize("w", [16, 32, 64])
def test_mla_row_is_nan_where_a_source_block_ends_on_minus_infinity(w):
    torch.manual_seed(0)
    z, heads, n, dtype = 1, 2, 128, torch.float16
    dn, dr, dv = mla_source.DN, mla_source.DR, mla_source.DV
    qn, qr = _negative(z, heads, n, dn, dtype=dtype), _negative(z, heads, n, dr, dtype=dtype)
    kn = torch.randn(z, heads, n, dn, device="mps", dtype=dtype)
    kr = torch.randn(z, heads, n, dr, device="mps", dtype=dtype)
    kn[:, :, :w] = INF
    kr[:, :, :w] = INF
    v = torch.randn(z, heads, n, dv, device="mps", dtype=dtype)
    out = torch.full((z, heads, n, dv), 7.0, device="mps", dtype=dtype)
    strides = [s for t in (qn, qr, kn, kr, v, out) for s in mla_source._st(t)]
    mla_source._mla_value_path[(n // 32, z * heads)](
        qn, qr, kn, kr, v, out, *strides, z, heads, n, 32, 32, dn, dr, dv, 0
    )
    torch.mps.synchronize()
    assert _nan_rows(out, -2) == _expected_rows(n, w, 32, False)


# ---------------------------------------------------------------------------------------
# the makers' text
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit,message",
    [
        ((INV_LN2, 10.0, LN2), "base must be 2 or e"),
        ((INF, 2.0, LN2), "positive and finite"),
        ((INV_LN2, 2.0, 0.0), "positive and finite"),
        ((-INV_LN2, 2.0, LN2), "positive and finite"),
    ],
)
def test_units_no_template_spells_are_an_error(unit, message):
    with pytest.raises(ValueError, match=message):
        templates._fa_source_units(unit)


def test_the_spelling_of_source_units():
    assert templates._fa_source_units((1.5, 2.0, 0.25)) == ("s *= 1.5f;", "exp2", "0.25f")
    assert templates._fa_source_units((1.5, math.e, 0.25))[1] == "exp"


@pytest.mark.parametrize("maker_name", MAKERS[:2])
def test_source_units_need_a_biased_or_masked_score(maker_name):
    maker = getattr(templates, maker_name)
    args = (64, 32, 64) if maker_name == MAKERS[1] else (64, 32, 32)
    with pytest.raises(ValueError, match="score_unit needs a biased or masked score"):
        maker(*args, score_unit=(INV_LN2, 2.0, LN2))


_ARGS = ["device float* p0 [[buffer(0)]]"]
_BINDINGS = {
    **{name: f"p{i}" for i, name in enumerate(("Q", "K", "V", "O", "CUQ", "CUK"))},
    "H": "2u",
    **{f"{t}_{a}": "1u" for t in "qkvo" for a in ("st", "sh", "sk")},
}


def _varlen_text(maker_name, out_dtype, **kw):
    maker = getattr(templates, maker_name)
    return maker(head_dim=64, out_dtype=out_dtype, arg_decls=_ARGS, bindings=dict(_BINDINGS), scale=0.3, **kw)


@pytest.mark.parametrize(
    "maker_name,out_dtype,block,text",
    [
        (MAKERS[3], "fp32", 32, "(m_new == -INFINITY && ((kt + BN) % 32u) != 0u)"),
        (MAKERS[3], "fp32", 64, "(m_new == -INFINITY && ((kt + BN) % 64u) != 0u)"),
        (MAKERS[3], "fp16", 64, "(m_new == -INFINITY && ((kt + BN) % 64u) != 0u)"),
    ],
)
def test_variable_length_text_ends_the_sources_block(maker_name, out_dtype, block, text):
    plain = _varlen_text(maker_name, out_dtype)
    told = _varlen_text(maker_name, out_dtype, source_block=block)
    assert text not in plain and "m_new == -INFINITY" not in plain
    assert told.count(text) == 2  # alpha and P
    assert told.replace(f"({text} ? 1.0f : ", "(").replace(f"({text} ? 0.0f : ", "(") != plain


def test_the_one_key_shader_takes_its_sources_block_whole():
    """At the source's own 32 keys the one-key template takes the block in two passes: the
    maximum first, then alpha and every probability against it. A block that ends on -inf is
    NaN by that arithmetic, with no guard to say when the block ends."""
    plain = _varlen_text(MAKERS[2], "fp32")
    told = _varlen_text(MAKERS[2], "fp32", source_block=32)
    assert "float s_blk[32];" in told and "float s_blk[32];" not in plain
    assert "float alpha = exp2((m_i - m_new) * inv_ln2);" in told
    assert "m_new == -INFINITY" not in told and "m_new == -INFINITY" not in plain


@pytest.mark.parametrize(
    "maker_name,out_dtype,block", [(MAKERS[3], "fp16", 32), (MAKERS[3], "fp32", 16), (MAKERS[2], "fp32", 1)]
)
def test_variable_length_text_at_the_sources_own_width_is_unchanged(maker_name, out_dtype, block):
    assert _varlen_text(maker_name, out_dtype, source_block=block) == _varlen_text(maker_name, out_dtype)


@pytest.mark.parametrize(
    "maker_name,out_dtype,block",
    [(MAKERS[3], "fp16", 48), (MAKERS[3], "fp16", 16), (MAKERS[3], "fp16", 0), (MAKERS[2], "fp32", 0)],
)
def test_a_source_block_the_variable_length_tiles_cannot_end_is_an_error(maker_name, out_dtype, block):
    with pytest.raises(ValueError, match="source_block"):
        _varlen_text(maker_name, out_dtype, source_block=block)


@pytest.mark.parametrize(
    "block,message", [(12, "multiple of 8"), (24, "multiple of 8"), (4, "multiple of 8"), (96, "whole number of tiles")]
)
def test_a_source_block_the_simdgroup_tile_cannot_end_is_an_error(block, message):
    with pytest.raises(ValueError, match=message):
        templates.make_flash_attention_kernel_simdgroup(64, 32, 64, source_block_ends=block)


@pytest.mark.parametrize(
    "block,text", [(32, "(j % 4u) == 3u && m_new == -INFINITY) m_ended = true;"), (128, "% 128u) != 0u")]
)
def test_simdgroup_text_for_a_block_width_alone(block, text):
    plain = templates.make_flash_attention_kernel_simdgroup(64, 32, 64)
    assert templates.make_flash_attention_kernel_simdgroup(64, 32, 64, source_block_ends=64) == plain
    assert text in templates.make_flash_attention_kernel_simdgroup(64, 32, 64, source_block_ends=block)
    assert "m_new == -INFINITY" not in plain
