"""Where the dense, variable-length and MLA forward templates multiply by the softmax constant.

The sources multiply Q by a constant before the first dot product (``q = q * qk_scale``,
``q = q * SCALE``, ``qn = tl.load(...) * scale``). The templates used to multiply the finished
score instead. The two orders agree in real numbers only:

* a Q element near 1e30 against keys near 1e9 overflows the dot product, where Q times a
  constant below one does not: the template's row was NaN and the source's is finite;
* with a constant above one, Q times it overflows in the source and the row is NaN, where the
  dot product times the constant was finite in the template. In a 16-bit product that is a Q
  element near 6e4.

Each kernel here is the source of tests/test_flash_attention.py, tests/test_varlen_fa_routing.py
or tests/test_mla_value_paths.py, as written or with the multiply moved, and every result is
compared with the source evaluated in float64 with its float32 overflow.

Float16 tensors with a float32 product keep the score multiply: half tiles cannot hold the
product, and with magnitudes of at most 65504 neither order can overflow.
"""

import importlib.util
import inspect
import math
import sys

import pytest

try:
    import torch
    import triton

    import tests.test_flash_attention as dense_source
    import tests.test_mla_value_paths as mla_source
    import tests.test_varlen_fa_routing as varlen_source
    import triton_msl.codegen._msl_templates as templates
    from triton_msl.errors import MetalNonRecoverableError

    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_MPS = False

requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Metal GPU needed")
F32_MAX = 3.4028234e38
INF = float("inf")
TYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}
TOLERANCE = {"f32": 2e-4, "f16": 3e-2, "bf16": 5e-2}
MAKERS = (
    "make_flash_attention_kernel_tiled",
    "make_flash_attention_kernel_simdgroup",
    "make_varlen_flash_attention",
    "make_varlen_flash_attention_mma",
)

_QK = "        qk = tl.dot(q, tl.trans(k).to(q.dtype))\n"
DENSE = {
    "canonical": [],  # q * qk_scale, a float32 scalar: the product is float32 at every tensor type
    "typed": [("    q = q * qk_scale\n", "    q = q * (1.0 / HEAD_DIM**0.5)\n")],  # in Q's own type
    "post": [("    q = q * qk_scale\n", ""), (_QK, _QK[:-1] + " * qk_scale\n")],
    "both": [(_QK, _QK[:-1] + " * 0.5\n")],
}
VARLEN = {
    "canonical": [],  # q * SCALE, a Python constant: the product is in Q's own type
    "wide": [("    q = q * SCALE\n", "    q = q.to(tl.float32) * SCALE\n")],
    "post": [("    q = q * SCALE\n", ""), (_QK, _QK[:-1] + " * SCALE\n")],
    "both": [(_QK, _QK[:-1] + " * 0.5\n")],
}
# variant: (the type of Q's product, the constant on the finished score as a multiple of the
# source's constant or as a number)
PLAN = {
    "canonical": ("own", None),
    "typed": ("tensor", None),
    "wide": ("f32", None),
    "post": (None, "constant"),
    "both": ("own", 0.5),
}


@pytest.fixture(scope="module")
def variant_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("forward_constant")


_VARIANTS = {}


def _variant(fn, name, replacements, root):
    """The kernel's own text with the replacements, as a module of its own."""
    key = (fn.fn.__name__, name)
    if key not in _VARIANTS:
        if not replacements:
            _VARIANTS[key] = fn
        else:
            src = inspect.getsource(fn.fn)
            for old, new in replacements:
                assert src.count(old) == 1, old
                src = src.replace(old, new)
            path = root / f"{fn.fn.__name__}_{name}.py"
            path.write_text("import triton\nimport triton.language as tl\n\n\n" + src)
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = mod
            spec.loader.exec_module(mod)
            _VARIANTS[key] = getattr(mod, fn.fn.__name__)
    return _VARIANTS[key]


@pytest.fixture
def cold_caches(tmp_path, monkeypatch):
    """Nothing compiled before this test is reused, on disk or in the kernels' own caches."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    for kernel in (dense_source._flash_attn_fwd, varlen_source._varlen_fwd, *_VARIANTS.values()):
        kernel.device_caches.clear()


@pytest.fixture
def maker_spy(monkeypatch):
    hits = []
    for name in MAKERS:
        real = getattr(templates, name)

        def spy(*a, _r=real, _n=name, **k):
            hits.append((_n, k.get("round_q"), k.get("post_scale")))
            return _r(*a, **k)

        monkeypatch.setattr(templates, name, spy)
    return hits


def _dtype(name):
    return getattr(torch, TYPES[name])


def _source(q, k, v, constant, product, post):
    """Q times the constant in the product's type, the dot product with its float32 overflow,
    the constant on the score, the softmax and P @ V; float64 otherwise."""
    if product is not None:
        q = (q.to(product) * torch.tensor(constant, dtype=product)).to(product)
    s = q.double() @ k.double().T
    s = torch.where(s.abs() > F32_MAX, torch.sign(s) * INF, s)
    if post is not None:
        s = (s * post).float().double()
    return torch.softmax(s, -1) @ v.double()


def _random(q, k, big):
    pass


def _dot_overflows(q, k, big):
    """The dot product overflows float32 before a constant below one and not after it. At
    float16 nothing can overflow float32 and the row is an ordinary one."""
    q[5, :] = 0.0
    q[5, 0] = 1e30 if big else 6e4
    k[:, 0] = (4e8 + torch.arange(k.shape[0]) * 1e7) if big else (100.0 + torch.arange(k.shape[0]) * 3.0)


def _product_overflows(q, k, big):
    """Q times a constant above one overflows its type; the dot product times it would not."""
    q[5, :] = 0.0
    q[5, 0] = 3e38 if big else 6e4
    k[:, 0] = 1e-3 * (1.0 + torch.arange(k.shape[0]))


def _infinite(q, k, big):
    q[5, 3] = INF
    k[:, 3] = k[:, 3].abs() + 0.5


def _nan(q, k, big):
    q[5, 3] = float("nan")


FORMS = {
    "random": _random,
    "dot-overflows": _dot_overflows,
    "product-overflows": _product_overflows,
    "infinite": _infinite,
    "nan": _nan,
}


def _compare(got, want, type_name, label):
    got = got.double()
    nan_got, nan_want = got.isnan().any(-1), want.isnan().any(-1)
    assert torch.equal(nan_got, nan_want), (
        f"{label}: NaN rows {nan_got.nonzero().flatten().tolist()}, the source's "
        f"{nan_want.nonzero().flatten().tolist()}"
    )
    if bool((~nan_want).any()):
        err = float((got[~nan_want] - want[~nan_want]).abs().max())
        assert err < TOLERANCE[type_name], f"{label}: {err:.3e} from the source"


def _plan(variant, own, tensor, constant):
    product, post = PLAN[variant]
    product = {"own": own, "tensor": tensor, "f32": torch.float32, None: None}[product]
    return product, constant if post == "constant" else post


def _run_dense(kernel, variant, type_name, dim, form, q_type=None, literal=None):
    """``literal``: the kernel multiplies by this Python constant, Q in Q's own type."""
    n = 64
    torch.manual_seed(1)
    q, k, v = (torch.randn(n, dim) for _ in range(3))
    FORMS[form](q, k, type_name != "f16")
    q, k, v = q.to(_dtype(q_type or type_name)), k.to(_dtype(type_name)), v.to(_dtype(type_name))
    constant = 1.0 / math.sqrt(dim) if literal is None else literal
    own = torch.float32 if literal is None else q.dtype
    want = _source(q, k, v, constant, *_plan(variant, own, q.dtype, constant))
    dev = [x[None, None].contiguous().to("mps") for x in (q, k, v)]
    out = torch.full((1, 1, n, dim), 77.0, dtype=_dtype(type_name), device="mps")
    args = [*dev, out]
    strides = [s for x in args for s in x.stride()]
    kernel[(n // 32, 1)](*args, *strides, 1, 1, n, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=dim, IS_CAUSAL=False)
    torch.mps.synchronize()
    return out.cpu()[0, 0], want


def _run_varlen(kernel, variant, type_name, dim, form, constant, q_type=None):
    torch.manual_seed(1)
    lens, heads = [48, 32, 17], 2
    total = sum(lens)
    q, k, v = (torch.randn(total, heads, dim) for _ in range(3))
    for h in range(heads):
        FORMS[form](q[:48, h], k[:48, h], type_name != "f16")
    q, k, v = q.to(_dtype(q_type or type_name)), k.to(_dtype(type_name)), v.to(_dtype(type_name))
    want = _source(q[:48, 1], k[:48, 1], v[:48, 1], constant, *_plan(variant, q.dtype, q.dtype, constant))
    dev = [x.contiguous().to("mps") for x in (q, k, v)]
    out = torch.full((total, heads, dim), 77.0, dtype=_dtype(type_name), device="mps")
    cu = torch.tensor([0, 48, 80, 97], dtype=torch.int32, device="mps")
    args = [*dev, out]
    strides = [s for x in args for s in x.stride()]
    kernel[(2, len(lens) * heads)](*args, cu, cu, *strides, heads, 48, constant, 32, 32, dim)
    torch.mps.synchronize()
    return out.cpu()[:48, 1], want


# ---------------------------------------------------------------------------------------
# dense
# ---------------------------------------------------------------------------------------

# The variants of each type that reach a template or the ordinary lowering (a bfloat16 dot
# product and a head dimension of 128 at bfloat16 are refused for other reasons).
DENSE_CASES = [
    (variant, type_name, dim)
    for type_name, variants, dims in (
        ("f32", ("canonical", "typed", "post", "both"), (64, 128)),
        ("f16", ("canonical", "typed", "post", "both"), (64, 128)),
        ("bf16", ("canonical", "both"), (64,)),
    )
    for variant in variants
    for dim in dims
]


@requires_mps
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("variant,type_name,dim", DENSE_CASES)
def test_dense_forward_is_the_source(variant, type_name, dim, form, variant_dir):
    kernel = _variant(dense_source._flash_attn_fwd, variant, DENSE[variant], variant_dir)
    got, want = _run_dense(kernel, variant, type_name, dim, form)
    _compare(got, want, type_name, f"dense {variant} {type_name} D={dim} {form}")


@requires_mps
@pytest.mark.parametrize(
    "variant,type_name,dim,expected",
    [
        ("canonical", "f32", 64, ("make_flash_attention_kernel_simdgroup", "float", None)),
        ("canonical", "f32", 128, ("make_flash_attention_kernel_simdgroup", "float", None)),
        ("typed", "f32", 64, ("make_flash_attention_kernel_simdgroup", "float", None)),
        ("both", "f32", 64, ("make_flash_attention_kernel_simdgroup", "float", 0.5)),
        ("post", "f32", 64, ("make_flash_attention_kernel_simdgroup", None, None)),
        # float16 tensors, float32 product: the half tiles keep the score multiply
        ("canonical", "f16", 64, ("make_flash_attention_kernel_simdgroup", None, None)),
        ("canonical", "f16", 128, ("make_flash_attention_kernel_simdgroup", None, None)),
        ("both", "f16", 64, ("make_flash_attention_kernel_simdgroup", None, None)),
        # float16 tensors, float16 product: staged with its one rounding
        ("typed", "f16", 64, ("make_flash_attention_kernel_simdgroup", "half", None)),
        ("canonical", "bf16", 64, ("make_flash_attention_kernel_tiled", "float", None)),
        ("both", "bf16", 64, ("make_flash_attention_kernel_tiled", "float", 0.5)),
    ],
)
def test_dense_maker_is_given_the_source_product(
    variant, type_name, dim, expected, variant_dir, cold_caches, maker_spy
):
    kernel = _variant(dense_source._flash_attn_fwd, variant, DENSE[variant], variant_dir)
    got, want = _run_dense(kernel, variant, type_name, dim, "dot-overflows")
    assert maker_spy == [expected]
    _compare(got, want, type_name, f"dense {variant} {type_name} D={dim}")


def _beyond_float16(q, k, big):
    """A Q element float16 cannot hold, against keys small enough that the row is ordinary."""
    q[5, :] = 0.0
    q[5, 0] = 1e5
    k[:, 0] = torch.linspace(-1.0, 1.0, k.shape[0]) * 1e-4


@requires_mps
@pytest.mark.parametrize("form", ["random", "beyond-float16"])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("variant,post", [("canonical", None), ("typed", None), ("both", 0.5)])
def test_a_float32_q_beside_float16_keys_is_staged_in_float32(
    variant, post, dim, form, variant_dir, cold_caches, maker_spy, monkeypatch
):
    """The simdgroup maker's half tiles rounded Q to float16: an element of 1e5 became an
    infinity and its row NaN. The tiled maker stages the float32 product."""
    monkeypatch.setitem(FORMS, "beyond-float16", _beyond_float16)
    kernel = _variant(dense_source._flash_attn_fwd, variant, DENSE[variant], variant_dir)
    got, want = _run_dense(kernel, variant, "f16", dim, form, q_type="f32")
    assert maker_spy == [("make_flash_attention_kernel_tiled", "float", post)]
    _compare(got, want, "f16", f"dense {variant} float32 Q, float16 K D={dim} {form}")


# ---------------------------------------------------------------------------------------
# variable length
# ---------------------------------------------------------------------------------------

VARLEN_CONSTANTS = {64: (0.125, 2.0, -0.25), 128: (128**-0.5, 2.0, -0.25)}
VARLEN_CASES = [
    pytest.param(variant, type_name, dim, constant, id=f"{variant}-{type_name}-{dim}-{constant:.4g}")
    for variant in VARLEN
    for type_name in TYPES
    for dim in (64, 128)
    for constant in VARLEN_CONSTANTS[dim]
    # a score constant that is not positive is refused by the score proof, as before
    if not (variant == "post" and constant < 0)
]


@requires_mps
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("variant,type_name,dim,constant", VARLEN_CASES)
def test_variable_length_forward_is_the_source(variant, type_name, dim, constant, form, variant_dir):
    kernel = _variant(varlen_source._varlen_fwd, variant, VARLEN[variant], variant_dir)
    got, want = _run_varlen(kernel, variant, type_name, dim, form, constant)
    _compare(got, want, type_name, f"varlen {variant} {type_name} D={dim} {constant} {form}")


@requires_mps
@pytest.mark.parametrize(
    "variant,type_name,dim,expected",
    [
        ("canonical", "f32", 64, ("make_varlen_flash_attention_mma", "float", None)),
        ("canonical", "f32", 128, ("make_varlen_flash_attention", "float", None)),
        ("canonical", "f16", 64, ("make_varlen_flash_attention_mma", "half", None)),
        ("canonical", "f16", 128, ("make_varlen_flash_attention_mma", "half", None)),
        ("canonical", "bf16", 64, ("make_varlen_flash_attention_mma", "bfloat", None)),
        ("both", "f32", 64, ("make_varlen_flash_attention_mma", "float", 0.5)),
        ("both", "f16", 64, ("make_varlen_flash_attention_mma", "half", 0.5)),
        ("both", "bf16", 64, ("make_varlen_flash_attention_mma", "bfloat", 0.5)),
        ("post", "f32", 64, ("make_varlen_flash_attention_mma", None, None)),
        ("post", "f16", 64, ("make_varlen_flash_attention_mma", None, None)),
        # a float32 product of 16-bit values: the half tiles keep the score multiply, and
        # bfloat16 values, which can overflow, go to the maker with float registers
        ("wide", "f16", 64, ("make_varlen_flash_attention_mma", None, None)),
        ("wide", "bf16", 64, ("make_varlen_flash_attention", "float", None)),
        ("wide", "bf16", 128, ("make_varlen_flash_attention", "float", None)),
    ],
)
def test_variable_length_maker_is_given_the_source_product(
    variant, type_name, dim, expected, variant_dir, cold_caches, maker_spy
):
    kernel = _variant(varlen_source._varlen_fwd, variant, VARLEN[variant], variant_dir)
    got, want = _run_varlen(kernel, variant, type_name, dim, "dot-overflows", 0.125)
    assert maker_spy == [expected]
    _compare(got, want, type_name, f"varlen {variant} {type_name} D={dim}")


# A constant that is not finite in the type of its product: MLIR prints it as its bit pattern,
# and that integer was baked as the scale (2139095040 for a float32 infinity, 31744 for a
# float16 one). 7e4 is an infinity in float16 and 3.4e38 in bfloat16.
NOT_FINITE = [
    pytest.param(type_name, constant, id=f"{type_name}-{constant}")
    for type_name in TYPES
    for constant in (INF, -INF, float("nan"), *({"f16": (7e4,), "bf16": (3.4e38,)}.get(type_name, ())))
]


@requires_mps
@pytest.mark.parametrize("form", ["random", "infinite"])
@pytest.mark.parametrize("type_name,constant", NOT_FINITE)
def test_variable_length_constant_that_is_not_finite_reaches_no_template(
    type_name, constant, form, cold_caches, maker_spy
):
    """The ordinary lowering multiplies Q as written, and refuses the bfloat16 dot product by name."""
    if type_name == "bf16":
        with pytest.raises(MetalNonRecoverableError, match="bf16 dot operands"):
            _run_varlen(varlen_source._varlen_fwd, "canonical", type_name, 64, form, constant)
    else:
        got, want = _run_varlen(varlen_source._varlen_fwd, "canonical", type_name, 64, form, constant)
        _compare(got, want, type_name, f"varlen {type_name} {constant} {form}")
    assert maker_spy == []


def _literal(constant):
    return {INF: 'float("inf")', -INF: 'float("-inf")'}.get(
        constant, 'float("nan")' if constant != constant else repr(constant)
    )


@requires_mps
@pytest.mark.parametrize("type_name,constant", NOT_FINITE)
def test_dense_constant_that_is_not_finite_reaches_no_template(
    type_name, constant, variant_dir, cold_caches, maker_spy
):
    kernel = _variant(
        dense_source._flash_attn_fwd,
        f"literal_{type_name}_{len(_VARIANTS)}",
        [("    q = q * qk_scale\n", f"    q = q * {_literal(constant)}\n")],
        variant_dir,
    )
    if type_name == "bf16":
        with pytest.raises(MetalNonRecoverableError, match="bf16 dot operands"):
            _run_dense(kernel, "canonical", type_name, 64, "random", literal=constant)
    else:
        got, want = _run_dense(kernel, "canonical", type_name, 64, "random", literal=constant)
        _compare(got, want, type_name, f"dense {type_name} {constant}")
    assert maker_spy == []


@requires_mps
@pytest.mark.parametrize("route", ["dense", "varlen"])
@pytest.mark.parametrize("type_name", ["f32", "f16"])
@pytest.mark.parametrize("constant", [INF, -INF, float("nan")], ids=["inf", "-inf", "nan"])
def test_a_score_constant_that_is_not_finite_is_refused(
    route, type_name, constant, variant_dir, cold_caches, maker_spy
):
    """On the finished score the bit pattern passed for a finite positive constant."""
    with pytest.raises(MetalNonRecoverableError, match="tt.dot RESULT"):
        if route == "varlen":
            kernel = _variant(varlen_source._varlen_fwd, "post", VARLEN["post"], variant_dir)
            _run_varlen(kernel, "post", type_name, 64, "random", constant)
        else:
            kernel = _variant(
                dense_source._flash_attn_fwd,
                f"score_literal_{len(_VARIANTS)}",
                [("    q = q * qk_scale\n", ""), (_QK, _QK[:-1] + f" * {_literal(constant)}\n")],
                variant_dir,
            )
            _run_dense(kernel, "post", type_name, 64, "random")
    assert maker_spy == []


@requires_mps
@pytest.mark.parametrize("form", ["random", "beyond-float16"])
@pytest.mark.parametrize("variant,round_q", [("canonical", "float"), ("both", "float"), ("post", None)])
@pytest.mark.parametrize("type_name", ["f16", "bf16"])
def test_variable_length_float32_q_beside_16_bit_keys_is_staged_in_float32(
    type_name, variant, round_q, form, variant_dir, cold_caches, maker_spy, monkeypatch
):
    """The MMA maker's 16-bit tiles rounded a float32 Q; the scalar maker's registers hold it."""
    monkeypatch.setitem(FORMS, "beyond-float16", _beyond_float16)
    kernel = _variant(varlen_source._varlen_fwd, variant, VARLEN[variant], variant_dir)
    got, want = _run_varlen(kernel, variant, type_name, 64, form, 0.125, q_type="f32")
    post = 0.5 if variant == "both" else None
    assert maker_spy == [("make_varlen_flash_attention", round_q, post)]
    _compare(got, want, type_name, f"varlen {variant} float32 Q, {type_name} K {form}")


_ALPHA, _P = "        alpha = tl.exp(m_i - m_new)\n", "        p = tl.exp(qk - m_new[:, None])\n"


@requires_mps
@pytest.mark.parametrize("route", ["dense", "varlen"])
@pytest.mark.parametrize("type_name", ["f32", "f16"])
@pytest.mark.parametrize("multiplier", ['float("inf")', 'float("nan")'], ids=["inf", "nan"])
def test_an_exponent_multiplier_that_is_not_finite_is_refused(
    route, type_name, multiplier, variant_dir, cold_caches, maker_spy
):
    """``exp2((score - max) * c)``: the bit pattern of c passed for a number and was folded into
    the baked scale."""
    replacements = [
        (_ALPHA, f"        alpha = tl.exp2((m_i - m_new) * {multiplier})\n"),
        (_P, f"        p = tl.exp2((qk - m_new[:, None]) * {multiplier})\n"),
    ]
    source = dense_source._flash_attn_fwd if route == "dense" else varlen_source._varlen_fwd
    kernel = _variant(source, f"exp2_{len(_VARIANTS)}", replacements, variant_dir)
    with pytest.raises(MetalNonRecoverableError, match="exponential fact"):
        if route == "dense":
            _run_dense(kernel, "canonical", type_name, 64, "random")
        else:
            _run_varlen(kernel, "canonical", type_name, 64, "random", 0.125)
    assert maker_spy == []


def _plan_of(rounding, info, multiplier, tile, q_elem):
    from types import SimpleNamespace

    from triton_msl.codegen.generic_lowerer import GenericLowerer

    return GenericLowerer._fa_q_product_plan(SimpleNamespace(_fa_rounding=rounding), info, multiplier, tile, q_elem)


def test_the_plan_for_a_product_the_tiles_hold():
    info = {"q_factor": 0.3, "score_scale": {"factor": 0.5}, "scale": 0.15}
    assert _plan_of({"q_product": "float"}, info, 1.0, "float", "f32") == {
        "round_q": "float",
        "scale": 0.3,
        "post_scale": 0.5,
        "scale_chain": (),
    }
    # a score constant of 0.5 and an exponent multiplier of 2: nothing is left for the score
    chained = _plan_of(
        {"round_q": "half", "q_product": "float", "scale_chain": ("half", "bfloat")}, info, 2.0, "half", "f16"
    )
    assert chained == {"round_q": "half", "scale": 0.3, "post_scale": None, "scale_chain": ("half", "bfloat")}
    # no Q constant: the score constant alone, as before
    assert _plan_of({}, {"q_factor": None, "score_scale": {"factor": 0.5}, "scale": 0.5}, 1.0, "float", "f32") == {}
    # float16 values, float32 product, half tiles: the score multiply is the same function
    assert _plan_of({"q_product": "float"}, info, 1.0, "half", "f16") == {}


@pytest.mark.parametrize(
    "rounding,info,multiplier,tile,q_elem,message",
    [
        (
            {"q_product": "float"},
            {"q_factor": 0.3, "score_scale": {"factor": 1.0}},
            1.0,
            "half",
            "f32",
            "cannot be staged in float16 tiles",
        ),
        (
            {"q_product": "float"},
            {"q_factor": 0.3, "score_scale": {"factor": 1.0}},
            1.0,
            "half",
            "bf16",
            "cannot be staged in float16 tiles",
        ),
        (
            {"q_product": "float"},
            {"q_factor": INF, "score_scale": {"factor": 1.0}},
            1.0,
            "float",
            "f32",
            "is not finite",
        ),
        (
            {"q_product": "float"},
            {"q_factor": 0.3, "score_scale": {"factor": INF}},
            1.0,
            "float",
            "f32",
            "is not finite",
        ),
        (
            {"q_product": "float"},
            {"q_factor": 0.3, "score_scale": {"factor": 1.0}},
            float("nan"),
            "float",
            "f32",
            "is not finite",
        ),
        ({}, {"q_factor": 0.3, "score_scale": {"factor": 1.0}}, 1.0, "float", "f32", "was not recorded"),
    ],
)
def test_the_plan_refuses_what_no_tile_holds(rounding, info, multiplier, tile, q_elem, message):
    """No source reaches these as measured (the detectors and the routes turn them away first);
    the plan holds the conditions its own argument needs."""
    with pytest.raises(MetalNonRecoverableError, match=message):
        _plan_of(rounding, info, multiplier, tile, q_elem)


@pytest.mark.parametrize(
    "type_str,value,number",
    [
        ("f32", 0x7F800000, INF),
        ("tensor<32x64xf32>", 0xFF800000, -INF),
        ("f16", 0x7C00, INF),
        ("bf16", 0x7F80, INF),
        ("f64", 0x7FF0000000000000, INF),
        ("f32", 0.125, 0.125),
        ("i32", 3, 3.0),
    ],
)
def test_a_float_constant_printed_as_its_bits_is_read_in_its_type(type_str, value, number):
    from types import SimpleNamespace

    from triton_msl.codegen.generic_lowerer import _constant_number

    assert _constant_number(SimpleNamespace(attrs={"value": value}, type_str=type_str)) == number
    assert math.isnan(_constant_number(SimpleNamespace(attrs={"value": 0x7FC00000}, type_str="f32")))
    assert _constant_number(SimpleNamespace(attrs={"value": True}, type_str="i1")) is None
    assert _constant_number(SimpleNamespace(attrs={"value": "dense"}, type_str="f32")) is None


# ---------------------------------------------------------------------------------------
# MLA
# ---------------------------------------------------------------------------------------


def _run_mla(type_name, dims, form):
    dn, dr, dv = dims
    n = mla_source.N
    torch.manual_seed(1)
    q, k = torch.randn(n, dn + dr), torch.randn(n, dn + dr)
    v = torch.randn(n, dv)
    FORMS[form](q, k, type_name != "f16")
    q, k, v = (x.to(_dtype(type_name)) for x in (q, k, v))
    constant = 1.0 / math.sqrt(dn + dr)
    want = _source(q, k, v, constant, torch.float32, None)
    spread = lambda x: x[None, None].expand(mla_source.Z, mla_source.H, *x.shape).contiguous().to("mps")
    tensors = (spread(q[:, :dn]), spread(q[:, dn:]), spread(k[:, :dn]), spread(k[:, dn:]), spread(v))
    out = torch.full((mla_source.Z, mla_source.H, n, dv), 77.0, dtype=_dtype(type_name), device="mps")
    assert mla_source._run_mla(mla_source._mla_value_path, tensors, out, 0, dims=dims) == "computed"
    return out.cpu()[0, 1], want


@requires_mps
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize(
    "type_name,dims,round_q",
    [("f32", (64, 64, 128), "float"), ("f16", (64, 64, 128), None), ("f16", (128, 64, 128), None)],
)
def test_mla_forward_is_the_source(type_name, dims, round_q, form, cold_caches, maker_spy):
    got, want = _run_mla(type_name, dims, form)
    assert maker_spy == [("make_flash_attention_kernel_simdgroup", round_q, None)]
    _compare(got, want, type_name, f"MLA {type_name} {dims} {form}")


# ---------------------------------------------------------------------------------------
# the makers' text (no device)
# ---------------------------------------------------------------------------------------

_ARGS = [f"    device const float* p{i} [[buffer({i})]]" for i in range(6)]
_BINDINGS = {
    **{name: f"p{i}" for i, name in enumerate(("Q", "K", "V", "O", "CUQ", "CUK"))},
    "H": "2u",
    **{f"{t}_{a}": "1u" for t in "qkvo" for a in ("st", "sh", "sk")},
}


def _varlen_text(maker, out_dtype, **kw):
    return maker(head_dim=64, out_dtype=out_dtype, arg_decls=_ARGS, bindings=dict(_BINDINGS), scale=0.3, **kw)


@pytest.mark.parametrize(
    "tile,round_q,staged",
    [
        ("float", None, "float(X)"),
        ("half", None, "half(X)"),
        ("float", "float", "(float(X) * scale)"),
        ("half", "half", "half(float(X) * float(half(scale)))"),
        ("bfloat", "bfloat", "bfloat(float(X) * float(bfloat(scale)))"),
        ("float", "half", "float(half(float(X) * float(half(scale))))"),
        ("float", "bfloat", "float(bfloat(float(X) * float(bfloat(scale))))"),
        ("bfloat", "half", "bfloat(half(float(X) * float(half(scale))))"),
    ],
)
def test_staged_product_spelling(tile, round_q, staged):
    assert templates._fa_staged_q("X", round_q, tile) == staged


@pytest.mark.parametrize(
    "tile,round_q", [("half", "float"), ("bfloat", "float"), ("half", "bfloat"), ("float", "double")]
)
def test_a_product_its_tile_cannot_hold_is_an_error(tile, round_q):
    with pytest.raises(ValueError):
        templates._fa_staged_q("X", round_q, tile)


def test_score_constant_beside_a_staged_product():
    assert "const float s_scale = 1.0f;" in templates._fa_post_scale_decl(None)
    assert "const float s_scale = 0.5f;" in templates._fa_post_scale_decl(0.5)
    for bad in (INF, float("nan")):
        with pytest.raises(ValueError):
            templates._fa_post_scale_decl(bad)


@pytest.mark.parametrize(
    "maker_name,out_dtype", [("make_varlen_flash_attention", "fp32"), ("make_varlen_flash_attention_mma", "fp32")]
)
def test_variable_length_text_scales_where_it_is_told(maker_name, out_dtype):
    maker = getattr(templates, maker_name)
    plain = _varlen_text(maker, out_dtype)
    staged = _varlen_text(maker, out_dtype, round_q="float")
    both = _varlen_text(maker, out_dtype, round_q="float", post_scale=0.5)
    cell = "s *= {};" if maker_name == "make_varlen_flash_attention" else "*{} : -INFINITY;"
    score = "scale" if maker_name == "make_varlen_flash_attention" else "SCALE"
    assert cell.format(score) in plain and "s_scale" not in plain and "* scale)" not in plain
    for text, post in ((staged, "1.0f"), (both, "0.5f")):
        assert cell.format("s_scale") in text and cell.format(score) not in text
        assert f"const float s_scale = {post};" in text and "* scale)" in text
    with pytest.raises(ValueError):
        _varlen_text(maker, out_dtype, post_scale=0.5)
    with pytest.raises(ValueError):
        _varlen_text(maker, out_dtype, scale_chain=("half",))
    chained = _varlen_text(maker, out_dtype, round_q="float", scale_chain=("half",))
    assert "const float scale = float(half(0.3f))" in chained


@pytest.mark.parametrize("maker_name", ["make_flash_attention_kernel_tiled", "make_flash_attention_kernel_simdgroup"])
def test_dense_text_takes_a_score_constant_only_beside_a_staged_product(maker_name):
    maker = getattr(templates, maker_name)
    kw = dict(head_dim=64, out_dtype="fp32", scale=0.125)
    assert "const float s_scale = 0.5f;" in maker(round_q="float", post_scale=0.5, **kw)
    assert "const float s_scale = 1.0f;" in maker(round_q="float", **kw)
    assert "s_scale" not in maker(**kw)
    with pytest.raises(ValueError):
        maker(post_scale=0.5, **kw)
