"""A float16 attention kernel multiplies the probabilities it computed, not their float16 rounding.

The standard forward forms ``tl.dot(p.to(tl.float32), v.to(tl.float32))``: P is float32 at the
product whatever the tensors' type. The matrix shader used to store P as half before that product
at every float16 source it took. Two probabilities that differ in float32 and round to one float16
value then cancel against equal and opposite values, and a result near 4 comes out as 0.

Here Q[:, 0] is 8 and only K's first coordinate is set: 0, -2**-13, then -100. At head dimension
64 the scores are 0, -2**-13 and -100, so the probabilities before normalization are 1,
exp(-2**-13) and about 0. The first two differ by 1.2e-4 and both round to 1 in float16. V's first
two rows are 65504 and -65504. Every output element is 65504 * (1 - p) / (1 + p) = 3.998046870...,
which float16 holds as 3.998046875. Other head dimensions scale the scores by sqrt(64 / dim). The
sequence is a whole number of key blocks, so no padded key is scored.
"""

import importlib.util
import inspect
import math
import sys

import pytest
import torch
import triton

from tests.test_fa_tail_semantics import _executed_source, executed  # noqa: F401
from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.errors import MetalNonRecoverableError

MATRIX = "simdgroup-MMA FlashAttention-2"
TILED = "Head-dim-tiled FlashAttention-2"


def _want(dim, n=64):
    scale = float(torch.tensor(dim**-0.5, dtype=torch.float32))
    p = math.exp(-(2.0**-13) * 8 * scale)
    return 65504.0 * (1.0 - p) / (1.0 + p + (n - 2) * math.exp(-100 * 8 * scale))


def _lowered(kernel, dtype, constants):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    opts = backend.parse_options({})
    signature = {
        n: ("*" + dtype if n in ("Q", "K", "V", "Out") else "i32") for n in kernel.arg_names if n not in constants
    }
    source = ASTSource(kernel, signature, constexprs=constants)
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(target, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), context)
    meta = {}
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    return GenericLowerer(walk_ttgir(mod, opts), opts).lower()


def _constants(bm, bn, dim):
    return dict(
        Z=1, stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1, BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=dim, IS_CAUSAL=False
    )


_NARROWED = {}


def _narrowed(root):
    """The standard forward with the source itself narrowing P: ``tl.dot(p.to(v.dtype), v)``."""
    if "fn" not in _NARROWED:
        src = inspect.getsource(_flash_attn_fwd.fn)
        old = "        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))\n"
        assert src.count(old) == 1
        path = root / "narrowed_probability.py"
        path.write_text(
            "import triton\nimport triton.language as tl\n\n\n"
            + src.replace(old, "        acc += tl.dot(p.to(v.dtype), v)\n")
        )
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        _NARROWED["fn"] = getattr(module, _flash_attn_fwd.fn.__name__)
    return _NARROWED["fn"]


@pytest.fixture(scope="module")
def narrowed_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("narrowed_probability")


# Every tile the float16 standard forward computes with, and the shader each one takes.
TILES = [
    (32, 32, 64, MATRIX),
    (32, 64, 64, MATRIX),
    (64, 32, 64, TILED),
    (64, 64, 64, TILED),
    (32, 32, 128, MATRIX),
    (32, 32, 32, None),
]


@pytest.mark.parametrize("bm,bn,dim,marker", [t for t in TILES if t[3] == MATRIX])
def test_the_float16_matrix_shader_holds_the_probability_tile_in_float32(bm, bn, dim, marker):
    msl = _lowered(_flash_attn_fwd, "fp16", _constants(bm, bn, dim))
    assert marker in msl
    assert "threadgroup float tgP[32 * 64];" in msl and "threadgroup half tgP" not in msl
    assert "tgP[pr*BN+cj] = p;" in msl and "half(p)" not in msl
    assert "simdgroup_float8x8 pf; simdgroup_half8x8 vf, vfs[64/8];" in msl


def test_the_float32_matrix_shader_is_as_it_was():
    msl = _lowered(_flash_attn_fwd, "fp32", _constants(32, 64, 64))
    assert MATRIX in msl and "tgP" not in msl and "simdgroup_float8x8 pf, vf, vfs[64/8];" in msl


def _inputs(dtype, n, dim):
    q = torch.zeros((1, 1, n, dim), dtype=dtype)
    q[..., 0] = 8
    k = torch.zeros_like(q)
    k[..., 0] = -100
    k[:, :, 0, 0] = 0
    k[:, :, 1, 0] = -(2.0**-13)
    v = torch.zeros_like(q)
    v[:, :, 0, :] = 65504
    v[:, :, 1, :] = -65504
    return q, k, v


def _run(kernel, dtype, bm, bn, dim, n, executed):
    q, k, v = _inputs(dtype, n, dim)
    before = [x.clone() for x in (q, k, v)]
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full_like(q, float("nan"))
    compiled = kernel[(triton.cdiv(n, bm), 1)](
        q, k, v, out, *q.stride(), *k.stride(), *v.stride(), *out.stride(), 1, 1, n,
        BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=dim, IS_CAUSAL=False,
    )  # fmt: skip
    torch.mps.synchronize()
    for x, was in zip((q, k, v), before):
        assert torch.equal(x.cpu(), was)
    return out.cpu(), _executed_source(executed, compiled)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("bm,bn,dim,marker", TILES)
def test_two_probabilities_that_round_to_one_float16_value_do_not_cancel(bm, bn, dim, marker, n, executed, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    got, source = _run(_flash_attn_fwd, torch.float16, bm, bn, dim, n, executed)
    if marker:
        assert marker in source
    # One float16 unit in the last place is 0.002 below 4 and 0.004 above; the rounded P gives 0.
    want = torch.tensor(_want(dim, n)).to(torch.float16)
    assert got.unique().numel() == 1 and abs(float(got.unique()[0]) - float(want)) <= 0.004, got.unique()
    if dim == 64:
        assert float(want) == 3.998046875 and torch.equal(got, torch.full_like(got, 3.998046875))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("bm,bn", [(32, 32), (32, 64)])
def test_the_float32_kernel_computes_the_same_case(bm, bn, executed, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    got, source = _run(_flash_attn_fwd, torch.float32, bm, bn, 64, 64, executed)
    assert MATRIX in source
    torch.testing.assert_close(got, torch.full_like(got, _want(64)), rtol=2e-3, atol=0)


@pytest.mark.parametrize("bm,bn", [(32, 32), (32, 64)])
def test_a_source_that_narrows_the_probability_itself_is_lowered_or_refused(bm, bn, narrowed_dir):
    """Whichever it is, the float32 tile is not what such a source asks for."""
    try:
        msl = _lowered(_narrowed(narrowed_dir), "fp16", _constants(bm, bn, 64))
    except MetalNonRecoverableError:
        return
    assert MATRIX in msl
    if bn == 64:
        assert "threadgroup half tgP[32 * 64];" in msl and "tgP[pr*BN+cj] = half(p);" in msl
    else:
        # Two 32-key source blocks to the tile: P is narrowed against its own block's maximum
        # and rescaled to the tile's in float32 (test_attention_forwards_follow_the_source_recurrence).
        assert "threadgroup float tgP[32 * 64];" in msl
        assert "tgP[pr*BN+cj] = float(half(exp(tg_S[pr*BN+cj] - ms))) * exp(ms - mr);" in msl


def test_half_accumulators_leave_a_per_block_narrowing_to_the_tiled_shader(narrowed_dir, monkeypatch):
    """The opt-in half accumulators have no float P tile to rescale a block's narrowed P in."""
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "1")
    assert TILED in _lowered(_narrowed(narrowed_dir), "fp16", _constants(32, 32, 64))
    msl = _lowered(_narrowed(narrowed_dir), "fp16", _constants(32, 64, 64))
    assert MATRIX in msl and "tgP[pr*BN+cj] = half(p);" in msl


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("bm,bn", [(32, 32), (32, 64), (64, 32), (64, 64)])
def test_a_source_that_narrows_the_probability_itself_gets_its_own_rounding(
    bm, bn, narrowed_dir, executed, monkeypatch
):
    """Rounded to float16, the two probabilities are both 1 and the source's own product is 0."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    try:
        got, _ = _run(_narrowed(narrowed_dir), torch.float16, bm, bn, 64, 64, executed)
    except MetalNonRecoverableError:
        pytest.skip("the narrowed source is refused")
    print("NARROWED-ROUTE", MATRIX in _ and "matrix", TILED in _ and "tiled", "half tgP" in _)
    assert torch.equal(got, torch.zeros_like(got)), got.unique()


# The maker itself, at each place it can hold P: a float32 tile of its own (QK head dimension up to
# 170), the score tile (above that, where the float32 tile no longer fits in 32 KB), and the half
# tile it keeps for a source that narrows P.
@pytest.mark.parametrize("dim,holder", [(64, "tgP"), (128, "tgP"), (168, "tgP"), (176, "tg_S"), (192, "tg_S")])
def test_where_the_float16_maker_holds_a_float32_probability(dim, holder):
    from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup as make

    msl = make(dim, 32, 64, False, "fp16", kernel_name="k", v_head_dim=64)
    assert "half(p)" not in msl and "threadgroup half tgP" not in msl
    assert ("threadgroup float tgP[32 * 64];" in msl) == (holder == "tgP")
    assert f"simdgroup_load(pf, {holder} + 0u*BN + kk, BN);" in msl
    assert "simdgroup_float8x8 pf; simdgroup_half8x8 vf, vfs[64/8];" in msl
    # The accumulators are rescaled in the score tile only while P is not in it.
    assert ("tg_S + sgitg*256u" in msl) == (holder == "tgP")
    narrowed = make(dim, 32, 64, False, "fp16", kernel_name="k", v_head_dim=64, round_p="half")
    assert "threadgroup half tgP[32 * 64];" in narrowed and "tgP[pr*BN+cj] = half(p);" in narrowed
    assert "simdgroup_half8x8 pf, vf, vfs[64/8];" in narrowed


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim,round_p", [(64, None), (128, None), (168, None), (176, None), (192, None), (64, "half")])
def test_the_maker_multiplies_the_probability_it_holds(dim, round_p, causal):
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup as make

    rt = CompileShaderRuntime()
    if not rt.available():
        pytest.skip("compile_shader is not available")
    n = 64
    lib = rt.get_library(make(dim, 32, 64, causal, "fp16", kernel_name="pv", v_head_dim=64, round_p=round_p))
    q, k, _ = _inputs(torch.float16, n, dim)
    _, _, v = _inputs(torch.float16, n, 64)
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full_like(v, float("nan"))
    args = [q, k, v, out] + [x.stride(i) for x in (q, k, v, out) for i in range(4)] + [1, 1, n]
    rt.dispatch(lib, "pv", args, threads=(2 * 256, 1, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()
    got = out.cpu()
    if causal:
        # Row 0 sees key 0 alone and stores its value; every later row sees both.
        assert torch.equal(got[0, 0, 0], torch.full((64,), 65504.0, dtype=torch.float16))
        got = got[:, :, 1:]
    want = 0.0 if round_p else float(torch.tensor(_want(dim, n)).to(torch.float16))
    if causal and not round_p:
        # Fewer keys at -100 are visible; their weight is below float16 resolution at these sizes.
        want = float(torch.tensor(_want(dim, 2)).to(torch.float16))
    assert float((got.float() - want).abs().max()) <= 0.004, got.unique()


# The variable-length forward has its own matrix shader, with 16-bit tiles for float16 and for
# bfloat16. bfloat16 holds 65536 and not 65504, and the result, 4.0002, as 4.
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype,big", [(torch.float16, 65504.0), (torch.bfloat16, 65536.0), (torch.float32, 65504.0)])
def test_the_variable_length_matrix_shader_multiplies_a_float32_probability(dtype, big, dim, causal, executed):
    from tests.test_varlen_fa_routing import _varlen_causal_fwd, _varlen_fwd

    if dtype == torch.float32 and dim == 128:
        pytest.skip("the float32 tiles of head dimension 128 take the scalar template")
    n, heads = 64, 2
    q = torch.zeros((n, heads, dim), dtype=dtype)
    q[..., 0] = 8
    k = torch.zeros_like(q)
    k[..., 0] = -100
    k[0, :, 0] = 0
    k[1, :, 0] = -(2.0**-13)
    v = torch.zeros_like(q)
    v[0], v[1] = big, -big
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full_like(q, float("nan"))
    cu = torch.tensor([0, n], device="mps", dtype=torch.int32)
    kernel = _varlen_causal_fwd if causal else _varlen_fwd
    compiled = kernel[(2, heads)](
        q, k, v, out, cu, cu, *q.stride(), *k.stride(), *v.stride(), *out.stride(), heads, n, dim**-0.5, 32, 32, dim
    )
    torch.mps.synchronize()
    source = _executed_source(executed, compiled)
    assert "simdgroup_multiply_accumulate" in source
    got = out.cpu().float()
    if causal:
        assert torch.equal(got[0], torch.full((heads, dim), big))
        got = got[1:]
    scale = float(torch.tensor(dim**-0.5, dtype=torch.float32))
    p = math.exp(-(2.0**-13) * 8 * scale)
    want = float(torch.tensor(big * (1.0 - p) / (1.0 + p)).to(dtype))
    assert float((got - want).abs().max()) <= (0.04 if dtype == torch.bfloat16 else 0.004), got.unique()
    if dtype != torch.float32:
        assert "tg_P" not in source and "threadgroup float tg_R[4 * 64];" in source


def _narrowed_varlen(narrowed_dir):
    import tests.test_varlen_fa_routing as routing

    name = "narrowed_varlen"
    if name not in _NARROWED:
        src = inspect.getsource(routing._varlen_fwd.fn)
        old = "        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))\n"
        assert src.count(old) == 1
        path = narrowed_dir / f"{name}.py"
        path.write_text(
            "import triton\nimport triton.language as tl\n\n\n"
            + src.replace(old, "        acc += tl.dot(p.to(v.dtype), v)\n")
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _NARROWED[name] = module._varlen_fwd
    return _NARROWED[name]


# A float32 Out resolves what the narrowing costs: with keys at scores 0 and -0.5 and every value 3,
# the source computes 3 * (1 + round(p)) / (1 + p). Rounding the denominator too would give 3.
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,grid",
    [
        (torch.float32, torch.float16, 2048),
        (torch.float16, torch.float16, 2048),
        (torch.float32, torch.bfloat16, 256),
    ],
)
def test_a_narrowed_probability_beside_a_float32_out_leaves_the_denominator_wide(
    q_dtype, kv_dtype, grid, narrowed_dir, executed
):
    kernel = _narrowed_varlen(narrowed_dir)
    n, heads, dim = 64, 2, 64
    q = torch.zeros((n, heads, dim), dtype=q_dtype)
    q[..., 0] = 8
    k = torch.zeros((n, heads, dim), dtype=kv_dtype)
    k[..., 0] = -100
    k[0, :, 0] = 0
    k[1, :, 0] = -0.5
    v = torch.full((n, heads, dim), 3, dtype=kv_dtype)
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full((n, heads, dim), float("nan"), dtype=torch.float32, device="mps")
    cu = torch.tensor([0, n], device="mps", dtype=torch.int32)
    compiled = kernel[(2, heads)](
        q, k, v, out, cu, cu, *q.stride(), *k.stride(), *v.stride(), *out.stride(), heads, n, dim**-0.5, 32, 32, dim
    )
    torch.mps.synchronize()
    source = _executed_source(executed, compiled)
    tile = "half" if kv_dtype == torch.float16 else "bfloat"
    assert f"float p_v = float({tile}(p));" in source
    p = math.exp(-0.5)
    rounded = round(p * grid) / grid
    want = 3 * (1 + rounded) / (1 + p)
    assert 1e-4 < 3 - want < 2.1e-3
    got = out.cpu().double()
    assert float((got - want).abs().max()) <= 2e-5, got.unique()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("q_dtype", [None, torch.float32])
@pytest.mark.parametrize("dtype,big,tile", [(torch.float16, 65504.0, "half"), (torch.bfloat16, 65536.0, "bfloat")])
def test_a_variable_length_source_that_narrows_the_probability_gets_its_own_rounding(
    dtype, big, tile, q_dtype, narrowed_dir, executed
):
    # A float32 Q beside 16-bit K, V and Out takes the one-key-at-a-time shader, which narrows the
    # probability for the value product as the matrix shader does and sums the float32 one.
    kernel = _narrowed_varlen(narrowed_dir)
    n, heads, dim = 64, 2, 64
    q = torch.zeros((n, heads, dim), dtype=dtype)
    q[..., 0] = 8
    k = torch.zeros_like(q)
    k[..., 0] = -100
    k[0, :, 0] = 0
    k[1, :, 0] = -(2.0**-13)
    v = torch.zeros_like(q)
    v[0], v[1] = big, -big
    q, k, v = [x.to("mps") for x in (q.to(q_dtype or dtype), k, v)]
    out = torch.full_like(v, float("nan"))
    cu = torch.tensor([0, n], device="mps", dtype=torch.int32)
    try:
        compiled = kernel[(2, heads)](
            q, k, v, out, cu, cu, *q.stride(), *k.stride(), *v.stride(), *out.stride(), heads, n, dim**-0.5, 32, 32, dim
        )
    except MetalNonRecoverableError:
        pytest.skip("the narrowed source is refused")
    torch.mps.synchronize()
    source = _executed_source(executed, compiled)
    got = out.cpu().float()
    assert torch.equal(got, torch.zeros_like(got)), got.unique()
    if q_dtype is None:
        assert "simdgroup_multiply_accumulate" in source
        assert f"threadgroup {tile} tg_P[BM*BN];" in source and "tg_R" not in source
    else:
        assert "simdgroup_multiply_accumulate" not in source
        assert "l_i = l_i * alpha + p_sum;" in source
        assert f"float p_v = float({tile}(p));" in source


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("dtype,big", [(torch.float16, 65504.0), (torch.bfloat16, 65536.0)])
def test_the_one_key_variable_length_shader_multiplies_a_float32_probability(dtype, big, executed):
    # The same inputs and types as the narrowed source above, from the source that keeps P wide.
    from tests.test_varlen_fa_routing import _varlen_fwd

    n, heads, dim = 64, 2, 64
    q = torch.zeros((n, heads, dim), dtype=torch.float32)
    q[..., 0] = 8
    k = torch.zeros((n, heads, dim), dtype=dtype)
    k[..., 0] = -100
    k[0, :, 0] = 0
    k[1, :, 0] = -(2.0**-13)
    v = torch.zeros_like(k)
    v[0], v[1] = big, -big
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full_like(v, float("nan"))
    cu = torch.tensor([0, n], device="mps", dtype=torch.int32)
    compiled = _varlen_fwd[(2, heads)](
        q, k, v, out, cu, cu, *q.stride(), *k.stride(), *v.stride(), *out.stride(), heads, n, dim**-0.5, 32, 32, dim
    )
    torch.mps.synchronize()
    source = _executed_source(executed, compiled)
    assert "simdgroup_multiply_accumulate" not in source
    assert "acc[d] = acc[d] * alpha + p * tg_V[kk * D + d];" in source
    p = math.exp(-(2.0**-13) * 8 * float(torch.tensor(dim**-0.5, dtype=torch.float32)))
    want = float(torch.tensor(big * (1.0 - p) / (1.0 + p)).to(dtype))
    got = out.cpu().float()
    assert want > 3.9
    assert float((got - want).abs().max()) <= (0.04 if dtype == torch.bfloat16 else 0.004), got.unique()
