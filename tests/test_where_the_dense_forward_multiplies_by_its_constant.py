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

Float16 tensors with a float32 product keep the score multiply only where the constant is a
power of two (between 1e-20 and 1e25 in magnitude): multiplying by it rounds nothing, and the
two orders are one number. Any other constant rounds each product of Q, and terms that cancel
without it do not cancel with it: with 0.3, a Q of [40960, 57344] against a key of
[57344, -40960] scores -64 in the source and 0 on the score multiply. The half-tile makers then
form the float32 product as they load each Q fragment. Outside the bounds (a Q of 2 times 3e38
is an infinity, and times a key of zero NaN) the variable-length route takes the maker with
float registers and the dense and MLA simdgroup routes refuse.
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


def _two_times_zero(q, k, big):
    """Every key is zero: a score is zero, or NaN where Q times the constant is an infinity."""
    q[5, :] = 2.0
    q[6, :] = 1e-3
    k[:, :] = 0.0


def _smallest_values(q, k, big):
    """Rows of the smallest float16 magnitudes: with a small constant the source's product
    underflows float32, and with any constant the scores stay far below one."""
    q[5, :] = 2.0**-24
    q[6, :] = -(2.0**-14)
    k[:, 0] = 2.0**-24


def _cancelling(q, k, big):
    """Two terms that cancel exactly without the constant, as in the reviewer's case. Half the
    keys are those and the rest are zero, so a score that does not cancel moves every weight.

    With these two values the source's score is below -12 for 0.3, 1/sqrt(128) and 1/sqrt(192)
    however a float32 dot product rounds its two terms (each product rounded, or one fused
    into the sum, in either order), and exactly zero for a power of two and for 3. The
    reviewer's own values, 40960 and 57344, score 0 or -4 at 1/sqrt(128) by that choice.
    """
    q[:, :] = 0.0
    q[:, 0], q[:, 1] = CANCEL
    k[:, :] = 0.0
    half = k.shape[0] // 2
    k[:half, 0], k[:half, 1] = CANCEL[1], -CANCEL[0]


CANCEL = (40640.0, 61792.0)
# The constants at which the cancelling form has one result whatever the dot product's rounding.
CANCEL_CONSTANTS = (0.3, 128**-0.5, 192**-0.5, 3.0, 0.125, 0.25)


EDGE_FORMS = {"two-times-zero": _two_times_zero, "smallest-values": _smallest_values, "cancelling": _cancelling}

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
    {**FORMS, **EDGE_FORMS}[form](q, k, type_name != "f16")
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
        {**FORMS, **EDGE_FORMS}[form](q[:48, h], k[:48, h], type_name != "f16")
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
        # float16 tensors, float32 product: the half tiles keep the score multiply for a power
        # of two (1/8) and form the float32 product as they load Q for any other constant
        ("canonical", "f16", 64, ("make_flash_attention_kernel_simdgroup", None, None)),
        ("canonical", "f16", 128, ("make_flash_attention_kernel_simdgroup", "float", None)),
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
    # float16 values, float32 product, half tiles: with a power of two the score multiply is
    # the same function; with 0.3 the makers form the float32 product
    assert _plan_of({"q_product": "float"}, {**info, "q_factor": 0.25}, 1.0, "half", "f16") == {}
    assert _plan_of({"q_product": "float"}, info, 1.0, "half", "f16")["round_q"] == "float"


# ---------------------------------------------------------------------------------------
# a float32 product of float16 values, with a constant the score multiply does not follow
# ---------------------------------------------------------------------------------------

LOW, HIGH = 1e-20, 1e25
OUTSIDE = [3e38, -3e38, 1e30, 1.0001e25, 9.999e-21, -1e-30, 1e-45, 0.0]
POWERS = [0.125, -0.25, 2.0, 2.0**-66, 2.0**83, -(2.0**83), 1.0]
ROUNDING = [0.3, 128**-0.5, 192**-0.5, -0.7, 3.0, 1e10, 1e-10, HIGH, -HIGH, LOW, -LOW]


def _float32_orders(constant, q, k):
    """One score both ways in float32, term by term: the source's, and the score multiply."""
    np = pytest.importorskip("numpy")
    with np.errstate(all="ignore"):
        c = np.float32(constant)
        q = np.asarray(q, dtype=np.float16).astype(np.float32)
        k = np.asarray(k, dtype=np.float16).astype(np.float32)
        source, template = np.float32(0), np.float32(0)
        for i in range(len(q)):
            source = np.float32(source + np.float32(q[i] * c) * k[i])
            template = np.float32(template + q[i] * k[i])
        return float(source), float(np.float32(template * c))


def _float16_vectors(seed, terms):
    """Float16 values of every bit pattern that is finite, with pairs of terms that cancel."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 0x7C00, size=terms, dtype=np.uint16) | (rng.integers(0, 2, size=terms, dtype=np.uint16) << 15)
    k = rng.integers(0, 0x7C00, size=terms, dtype=np.uint16) | (rng.integers(0, 2, size=terms, dtype=np.uint16) << 15)
    q, k = q.view(np.float16).copy(), k.view(np.float16).copy()
    # q[1] * k[1] is minus q[0] * k[0] exactly, as in the reviewer's case
    q[1], k[1] = k[0], -q[0]
    return q, k


@pytest.mark.parametrize("constant", POWERS)
def test_with_a_power_of_two_the_two_orders_are_one_number(constant):
    np = pytest.importorskip("numpy")
    for seed in range(200):
        q, k = _float16_vectors(seed, 2 + seed % 255)
        source, template = _float32_orders(constant, q, k)
        assert np.float32(source).tobytes() == np.float32(template).tobytes() or (source == 0 and template == 0), (
            constant,
            seed,
            source,
            template,
        )


def test_with_another_constant_terms_that_cancel_without_it_do_not_cancel_with_it():
    # the reviewer's case: 0.3, Q [40960, 57344] against K [57344, -40960]
    source, template = _float32_orders(0.3, [40960.0, 57344.0], [57344.0, -40960.0])
    assert (source, template) == (-64.0, 0.0)
    # beside a key of zero, whose score is zero both ways, the attention weights are (0, 1) and (1/2, 1/2)
    assert math.exp(source) < 1e-27
    for constant in (0.3, 128**-0.5, 192**-0.5):
        differing = 0
        for seed in range(200):
            source, template = _float32_orders(constant, *_float16_vectors(seed, 2))
            differing += source != template
        assert differing > 20, constant


def _dot_roundings(constant, x, y):
    """Q [x, y] times the constant against K [y, -x] in float32: each product rounded and then
    summed, the exact sum rounded once, and one product fused into the sum, in either order."""
    np = pytest.importorskip("numpy")
    from fractions import Fraction

    a, b = np.float32(x) * np.float32(constant), np.float32(y) * np.float32(constant)
    first, second = Fraction(float(a)) * Fraction(y), Fraction(float(b)) * Fraction(-x)
    rounded_first, rounded_second = np.float32(a * np.float32(y)), np.float32(b * np.float32(-x))
    return (
        float(rounded_first + rounded_second),
        float(np.float32(float(first + second))),
        float(np.float32(float(Fraction(float(rounded_first)) + second))),
        float(np.float32(float(first + Fraction(float(rounded_second))))),
    )


def test_the_cancelling_form_does_not_depend_on_how_a_dot_product_rounds():
    for constant in (0.3, 128**-0.5, 192**-0.5):
        assert max(_dot_roundings(constant, *CANCEL)) <= -12.0, constant
    for constant in (3.0, 0.125, 0.25):
        assert set(_dot_roundings(constant, *CANCEL)) == {0.0}, constant
    # the reviewer's values at 1/sqrt(128), and these at a constant near the upper bound, do
    assert set(_dot_roundings(128**-0.5, 40960.0, 57344.0)) == {0.0, -4.0}
    near = _dot_roundings(9e24, *CANCEL)
    assert min(near) < 0 < max(near)


def test_outside_the_bounds_the_two_orders_differ_in_kind():
    source, template = _float32_orders(3e38, [2.0] * 4, [0.0] * 4)
    assert math.isnan(source) and template == 0.0
    source, template = _float32_orders(1e36, [65504.0], [2.0**-24])
    assert math.isinf(source) and 0 < template < F32_MAX


def _half_tile_plan(constant, post=1.0, multiplier=1.0):
    info = {"q_factor": constant, "score_scale": {"factor": post}, "scale": constant * post * multiplier}
    return _plan_of({"q_product": "float"}, info, multiplier, "half", "f16")


@pytest.mark.parametrize("constant", POWERS)
def test_half_tiles_keep_the_score_multiply_for_a_power_of_two(constant):
    assert _half_tile_plan(constant) == {}
    if 1e-19 <= abs(constant) <= 1e24:
        assert _half_tile_plan(constant, 0.3) == {}


@pytest.mark.parametrize("constant", ROUNDING)
def test_half_tiles_form_the_float32_product_for_any_other_constant_inside_the_bounds(constant):
    assert _half_tile_plan(constant) == {"round_q": "float", "scale": constant, "post_scale": None, "scale_chain": ()}
    if 1e-19 <= abs(constant) <= 1e24:
        assert _half_tile_plan(constant, 0.5)["post_scale"] == 0.5


@pytest.mark.parametrize("constant", OUTSIDE + [2.0**84, 2.0**-67])
def test_half_tiles_refuse_a_float32_product_outside_the_bounds(constant):
    with pytest.raises(MetalNonRecoverableError, match="multiplies a float16 Q by .* in float32 before the"):
        _half_tile_plan(constant)


@pytest.mark.parametrize(
    "constant,post,multiplier",
    [
        (1.0, 1.0001e25, 1.0),
        (1e20, 1e3, 1e3),
        (1.0, 1.0, 9e-21),
        (1e-10, 1e-11, 1.0),
        (1e-10, 1.0, 0.0),
        (0.3, 1e26, 1.0),
    ],
)
def test_half_tiles_refuse_where_the_whole_constant_is_outside_the_bounds(constant, post, multiplier):
    with pytest.raises(MetalNonRecoverableError, match="multiplies a float16 Q by .* in float32 before the"):
        _half_tile_plan(constant, post, multiplier)
    # the same product staged in float tiles is the source's
    info = {"q_factor": constant, "score_scale": {"factor": post}, "scale": constant * post * multiplier}
    assert _plan_of({"q_product": "float"}, info, multiplier, "float", "f32")["scale"] == constant


def test_the_bounds_and_the_powers_of_two_are_the_ones_tested_here():
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    inside = GenericLowerer._fa_half_tile_constant_in_bounds
    assert inside(HIGH, 1.0) and inside(LOW, 1.0) and inside(-HIGH, 1.0) and inside(-LOW, 1.0)
    assert not inside(math.nextafter(HIGH, INF), 1.0) and not inside(math.nextafter(LOW, 0.0), 1.0)
    assert not inside(1.0, math.nextafter(HIGH, INF)) and not inside(1.0, math.nextafter(LOW, 0.0))
    assert inside(1e30 / 1e10, 1e-10) and not inside(1e30, 1e-10)
    same = GenericLowerer._fa_half_tile_scale_is_the_source
    assert same(0.125, 1.0) and same(-0.125, 0.3) and same(2.0**83, 1.0) and same(2.0**-66, 1.0)
    assert not same(math.nextafter(0.125, 1.0), 1.0) and not same(math.nextafter(0.125, 0.0), 1.0)
    assert not same(0.375, 1.0) and not same(0.3, 1.0) and not same(128**-0.5, 1.0)
    assert not same(2.0**84, 1.0) and not same(2.0**-67, 1.0) and not same(0.125, 1e26)


_FRAGMENT_PRODUCT = """
#include <metal_stdlib>
using namespace metal;
kernel void product(device const half* A [[buffer(0)]], device float* P [[buffer(1)]],
                    device const float* C [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint g = gid.x / 32u;
    const float scale = C[0];
    simdgroup_half8x8 qh;
    simdgroup_float8x8 qf;
    simdgroup_float8x8 qc(scale);
    simdgroup_load(qh, A + g*64u, 8);
    simdgroup_multiply(qf, qh, qc);
    simdgroup_store(qf, P + g*64u, 8);
}
"""


@requires_mps
@pytest.mark.parametrize("constant", [0.3, 128**-0.5, 192**-0.5, -0.7, 3.0000001, 2e-20, 9e24])
def test_a_fragment_times_the_constant_on_a_diagonal_is_the_float32_product(constant):
    """What the half-tile makers load: each cell is float32(q) * constant, rounded once.

    A fragment row that holds an infinity or NaN has no finite cell (infinity times the zeros
    off the diagonal is NaN); the source's row of scores is not finite there either.
    """
    np = pytest.importorskip("numpy")
    groups = 2048
    rng = np.random.default_rng(7)
    values = rng.integers(0, 65536, size=groups * 64, dtype=np.uint16).view(np.float16).copy()
    values[:64] = [40960.0, 57344.0] * 32
    values[64:128] = [65504.0, -65504.0, 2.0**-24, -(2.0**-24), 0.0, -0.0, 1.0, 2.0**-14] * 8
    a = torch.tensor(values).to("mps")
    p = torch.full((groups * 64,), 77.0, dtype=torch.float32, device="mps")
    c = torch.tensor([constant], dtype=torch.float32, device="mps")
    torch.mps.compile_shader(_FRAGMENT_PRODUCT).product(a, p, c, threads=groups * 32, group_size=32)
    torch.mps.synchronize()
    got = p.cpu().numpy().reshape(groups, 8, 8)
    with np.errstate(all="ignore"):
        want = (values.astype(np.float32) * np.float32(constant)).reshape(groups, 8, 8)
    finite_rows = np.isfinite(values).reshape(groups, 8, 8).all(axis=2)
    assert finite_rows.sum() > groups * 6
    # an infinity stays one in its own cell and is NaN in the row's other cells
    assert not np.isfinite(got[~finite_rows]).any(axis=1).any() and np.isnan(got[~finite_rows]).any(axis=1).all()
    got, want = got[finite_rows], want[finite_rows]
    # the same bits, but for a zero's sign (the sum with the zeros off the diagonal is +0)
    assert np.array_equal(got == 0, want == 0)
    assert np.array_equal(got[want != 0].view(np.uint32), want[want != 0].view(np.uint32))


VARLEN_OUTSIDE = [3e38, 1e30, 1.0001e25, 9.999e-21, 1e-30]
# 9e24 and 2e-20 are inside the bounds as float32 values too.
VARLEN_ROUNDING = [0.3, 128**-0.5, 3.0, 9e24, 2e-20]
EDGE = ["cancelling", "two-times-zero", "smallest-values", "random", "infinite", "nan"]


def _edge_cases(constants):
    """Every form at every constant, the cancelling one where its result is one number."""
    return [
        pytest.param(constant, dim, form, id=f"{constant:.5g}-{dim}-{form}")
        for constant in constants
        for dim in (64, 128)
        for form in EDGE
        if form != "cancelling" or constant in CANCEL_CONSTANTS
    ]


@requires_mps
@pytest.mark.parametrize("constant,dim,form", _edge_cases(VARLEN_OUTSIDE + VARLEN_ROUNDING + [0.125, 0.25]))
def test_variable_length_float32_product_of_float16_values_is_the_source_at_any_constant(
    constant, dim, form, variant_dir, cold_caches, maker_spy
):
    kernel = _variant(varlen_source._varlen_fwd, "wide", VARLEN["wide"], variant_dir)
    got, want = _run_varlen(kernel, "wide", "f16", dim, form, constant)
    if constant in VARLEN_OUTSIDE:
        assert maker_spy == [("make_varlen_flash_attention", "float", None)]
    elif constant in VARLEN_ROUNDING:
        assert maker_spy == [("make_varlen_flash_attention_mma", "float", None)]
    else:
        assert maker_spy == [("make_varlen_flash_attention_mma", None, None)]
    if form == "two-times-zero" and constant == 3e38:
        # the reviewer's first case: 2 times 3e38 is an infinity, and times a key of zero NaN
        assert bool(want[5].isnan().all()) and not bool(want[6].isnan().any())
    _compare(got, want, "f16", f"varlen wide f16 D={dim} {constant} {form}")


def test_the_cancelling_form_separates_the_two_orders():
    """The source's result puts every weight on the keys of zero; uniform weights are far from it."""
    torch.manual_seed(1)
    q, k, v = (torch.randn(48, 64) for _ in range(3))
    _cancelling(q, k, False)
    q, k, v = q.half(), k.half(), v.half()
    source = _source(q, k, v, 0.3, torch.float32, None)
    assert torch.allclose(source, v[24:].double().mean(0).expand(48, 64), atol=1e-4)
    uniform = v.double().mean(0).expand(48, 64)
    assert float((source - uniform).abs().max()) > 10 * TOLERANCE["f16"]


@requires_mps
@pytest.mark.parametrize("form", ["cancelling", "two-times-zero", "random"])
# powers of two, the same number in float32: 1.5e26 and 8.5e-22
@pytest.mark.parametrize("post,staged", [(2.0**87, True), (2.0**-70, True), (2.0, False)])
def test_variable_length_whole_constant_outside_the_bounds_is_staged_too(
    post, staged, form, variant_dir, cold_caches, maker_spy, monkeypatch
):
    """Q's constant of 0.125 is inside the bounds; with the score's the whole one may not be."""
    name = f"wide_post_{post}"
    monkeypatch.setitem(PLAN, name, ("f32", post))
    kernel = _variant(
        varlen_source._varlen_fwd, name, [*VARLEN["wide"], (_QK, _QK[:-1] + f" * {post!r}\n")], variant_dir
    )
    got, want = _run_varlen(kernel, name, "f16", 64, form, 0.125)
    if staged:
        # the constant is read from its nine printed digits
        assert [hit[:2] for hit in maker_spy] == [("make_varlen_flash_attention", "float")]
        assert maker_spy[0][2] == pytest.approx(post, rel=1e-7)
    else:
        assert maker_spy == [("make_varlen_flash_attention_mma", None, None)]
    _compare(got, want, "f16", f"varlen wide f16 0.125 then {post} {form}")


@requires_mps
@pytest.mark.parametrize("constant,dim,form", _edge_cases([3e38, 1e-30, 9e24, 2e-20, 0.3, 128**-0.5, 3.0, 0.25]))
def test_dense_float32_product_of_float16_values_is_the_source_or_refused(
    constant, dim, form, variant_dir, cold_caches, maker_spy
):
    kernel = _variant(
        dense_source._flash_attn_fwd,
        f"wide_{constant}",
        [("    q = q * qk_scale\n", f"    q = q.to(tl.float32) * {constant!r}\n")],
        variant_dir,
    )
    if constant in (3e38, 1e-30):
        with pytest.raises(MetalNonRecoverableError, match="multiplies a float16 Q by .* in float32 before the"):
            _run_dense(kernel, "wide", "f16", dim, form, literal=constant)
        assert maker_spy == []
        return
    got, want = _run_dense(kernel, "wide", "f16", dim, form, literal=constant)
    assert maker_spy == [("make_flash_attention_kernel_simdgroup", None if constant == 0.25 else "float", None)]
    _compare(got, want, "f16", f"dense wide f16 D={dim} {constant} {form}")


@requires_mps
@pytest.mark.parametrize("dim,round_q", [(64, None), (128, "float")])
def test_the_canonical_dense_source_at_float16_cancels_as_its_float32_product_does(
    dim, round_q, variant_dir, cold_caches, maker_spy
):
    """1/sqrt(64) is a power of two and 1/sqrt(128) is not."""
    got, want = _run_dense(dense_source._flash_attn_fwd, "canonical", "f16", dim, "cancelling")
    assert maker_spy == [("make_flash_attention_kernel_simdgroup", round_q, None)]
    _compare(got, want, "f16", f"dense canonical f16 D={dim} cancelling")


# ---------------------------------------------------------------------------------------
# MLA
# ---------------------------------------------------------------------------------------


def _run_mla(type_name, dims, form):
    dn, dr, dv = dims
    n = mla_source.N
    torch.manual_seed(1)
    q, k = torch.randn(n, dn + dr), torch.randn(n, dn + dr)
    v = torch.randn(n, dv)
    {**FORMS, **EDGE_FORMS}[form](q, k, type_name != "f16")
    q, k, v = (x.to(_dtype(type_name)) for x in (q, k, v))
    constant = 1.0 / math.sqrt(dn + dr)
    want = _source(q, k, v, constant, torch.float32, None)
    spread = lambda x: x[None, None].expand(mla_source.Z, mla_source.H, *x.shape).contiguous().to("mps")
    tensors = (spread(q[:, :dn]), spread(q[:, dn:]), spread(k[:, :dn]), spread(k[:, dn:]), spread(v))
    out = torch.full((mla_source.Z, mla_source.H, n, dv), 77.0, dtype=_dtype(type_name), device="mps")
    assert mla_source._run_mla(mla_source._mla_value_path, tensors, out, 0, dims=dims) == "computed"
    return out.cpu()[0, 1], want


@requires_mps
@pytest.mark.parametrize("form", [*FORMS, "cancelling"])
@pytest.mark.parametrize(
    "type_name,dims,round_q",
    [("f32", (64, 64, 128), "float"), ("f16", (64, 64, 128), "float"), ("f16", (128, 64, 128), "float")],
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


def test_float16_text_forms_the_float32_product_of_q():
    """A float32 product of float16 values. Where the tile fits in float32 it holds the product,
    and the final store's scratch moves into the dead score tile to make the room. Where it
    does not (a QK head dimension of 128 or 192, beside the float32 P tile), Q is staged as it is and each fragment loaded from
    it is multiplied by the constant on a diagonal. The score is not multiplied again."""
    make = templates.make_flash_attention_kernel_simdgroup
    for dim in (64,):
        kw = dict(head_dim=dim, out_dtype="fp16", scale=0.3)
        plain = make(**kw)
        assert "simdgroup_half8x8 qf, kf;" in plain and "s_scale" not in plain
        assert "threadgroup half tgQ[" in plain and "threadgroup float on_scratch[8 * 64];" in plain
        assert plain.count("tg_S + sgitg*256u") == 4 and "tg_S[sgitg*256u+e]" not in plain
        for text, post in (
            (make(round_q="float", **kw), "1.0f"),
            (make(round_q="float", post_scale=0.5, **kw), "0.5f"),
        ):
            assert f"threadgroup float tgQ[32 * {dim}];" in text and "threadgroup float tgP[" in text
            assert "tgQ[i] = (qr < N_CTX) ? (float(Q[q_base + qr*q_sm + (i%D)*q_sk]) * scale) : float(0);" in text
            assert text.count("simdgroup_float8x8 qf; simdgroup_half8x8 kf;") == 2
            assert "qh" not in text and "simdgroup_multiply(qf" not in text
            # the final store's scratch: this SIMD group's 256 cells of the score tile
            assert "on_scratch" not in text and text.count("tg_S + sgitg*256u") == 5
            assert text.count("tg_S[sgitg*256u+e]") == 1
            assert f"const float s_scale = {post};" in text and "const float scale = 0.3f;" in text
    # 32 KB of threadgroup memory: beside a float32 P tile the float Q tile at 128 no longer fits
    assert 32 * 128 * 4 + 32 * 64 * 4 + 32 * 64 * 4 + 32 * 12 + 32 * 64 + 64 * 8 * 2 > 32768

    for kw in (dict(head_dim=128, out_dtype="fp16", scale=0.3), dict(head_dim=192, out_dtype="fp16", scale=0.3)):
        plain = make(**kw)
        assert "simdgroup_half8x8 qf, kf;" in plain and "qh" not in plain
        for text, post in (
            (make(round_q="float", **kw), "1.0f"),
            (make(round_q="float", post_scale=0.5, **kw), "0.5f"),
        ):
            assert "simdgroup_half8x8 qh, kf; simdgroup_float8x8 qf; simdgroup_float8x8 qc(scale);" in text
            # two key loops (whole blocks and the split tail), four row groups each
            assert text.count("simdgroup_multiply(qf, qh, qc);") == 8 and text.count("simdgroup_load(qh, tgQ") == 8
            assert "simdgroup_load(qf, tgQ" not in text
            assert "tgQ[i] = (qr < N_CTX) ? Q[q_base + qr*q_sm + (i%D)*q_sk] : half(0);" in text
            assert "threadgroup half tgQ[" in text and "threadgroup float on_scratch[8 * 64];" in text
            assert f"const float s_scale = {post};" in text and "const float scale = 0.3f;" in text
    with pytest.raises(ValueError, match="needs float32 score accumulators"):
        make(round_q="float", half_accumulate=True, **kw)
    # a float32 maker stages the product in its float tile, as before
    assert "qh" not in make(head_dim=64, out_dtype="fp32", round_q="float")

    maker = templates.make_varlen_flash_attention_mma
    plain = _varlen_text(maker, "fp16")
    wide = _varlen_text(maker, "fp16", round_q="float")
    assert "simdgroup_half8x8 qf, kf;" in plain and "qh" not in plain
    assert "simdgroup_half8x8 qh, kf; simdgroup_float8x8 qf;" in wide
    assert wide.count("simdgroup_multiply(qf, qh, simdgroup_float8x8(scale));") == 1
    assert "simdgroup_load(qf, tg_Q" not in wide and "* scale)" not in wide
    assert "*s_scale : -INFINITY;" in wide and "const float s_scale = 1.0f;" in wide
    assert "qh" not in _varlen_text(maker, "fp32", round_q="float")
    assert "qh" not in _varlen_text(maker, "fp16", round_q="half")


@pytest.mark.parametrize("maker_name", ["make_flash_attention_kernel_tiled", "make_flash_attention_kernel_simdgroup"])
def test_dense_text_takes_a_score_constant_only_beside_a_staged_product(maker_name):
    maker = getattr(templates, maker_name)
    kw = dict(head_dim=64, out_dtype="fp32", scale=0.125)
    assert "const float s_scale = 0.5f;" in maker(round_q="float", post_scale=0.5, **kw)
    assert "const float s_scale = 1.0f;" in maker(round_q="float", **kw)
    assert "s_scale" not in maker(**kw)
    with pytest.raises(ValueError):
        maker(post_scale=0.5, **kw)
