"""A dense attention source with a 32 by 64 tile takes the matrix shader, whose own tile is 32 by 64.

The matrix (simdgroup) attention shader has always computed 64 keys a tile, and was chosen for a
source tile of 32 by 32 alone: a source that asked for 64 keys a block got the head-dim-tiled
shader, about six times slower at sequence 1024 and head dimension 64. The matrix maker replays
the source's key loop from the proved block width (the keys past the length a zero-filled load
still scores, the keys an unmasked load reads, where a causal loop stops), so a 64-wide source
block is the case where one tile is one source iteration.

Admitted: head dimension 64 for Q.K and for V, tile 32 by 64, float16 or float32 with K, V and Out
of one type, the innermost stride of all four folded to one, and no separate query length. Every
other source keeps the shader it had; so does one whose Q constant the half tiles cannot stage.

The references are float64 numpy, computed a query block at a time over the keys the source's
loop visits; they share nothing with the lowering.
"""

import importlib.util
import inspect
import sys

import numpy as np
import pytest
import torch
import triton

from tests.test_fa_tail_semantics import _executed_source, _tail_attention, executed  # noqa: F401
from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.errors import MetalNonRecoverableError

MATRIX = "simdgroup-MMA FlashAttention-2"
TILED = "Head-dim-tiled FlashAttention-2"
TORCH = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
SENTINEL = 7.0


def _lowered(kernel, types, constants):
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
        n: ("*" + types[n] if n in ("Q", "K", "V", "Out") else "i32") for n in kernel.arg_names if n not in constants
    }
    source = ASTSource(kernel, signature, constexprs=constants)
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(target, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), context)
    meta = {}
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    lower = GenericLowerer(walk_ttgir(mod, opts), opts)
    return lower, lower.lower()


def _tail_constants(dim=64, bm=32, bn=64, mask=0, plain="", contiguous=True, **more):
    constants = dict(
        Z=1,
        BLOCK_M=bm,
        BLOCK_N=bn,
        HEAD_DIM=dim,
        MASK=mask,
        START=0,
        STEP=bn,
        FILL_K=0.0,
        FILL_V=0.0,
        PLAIN_K="K" in plain,
        PLAIN_V="V" in plain,
    )
    if contiguous:
        constants.update(sqk=1, skk=1, svk=1, sok=1)
    constants.update(more)
    return constants


def _uniform(dtype):
    return dict.fromkeys(("Q", "K", "V", "Out"), dtype)


@pytest.mark.parametrize("plain", ["", "K", "V", "KV"])
@pytest.mark.parametrize("mask", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_a_32_by_64_source_is_lowered_to_the_matrix_shader(dtype, mask, plain):
    lower, msl = _lowered(_tail_attention, _uniform(dtype), _tail_constants(mask=mask, plain=plain))
    assert MATRIX in msl and TILED not in msl
    assert lower.effective_block_size == 256
    assert lower._flash_attention[0] == "flash_attention"
    # The keys past the length that the source's last 64-wide iteration still scores.
    assert ("source KV tail: 64" in msl) == (mask in (0, 2))
    assert "source KV tail: 32" not in msl
    assert ("source KV visit: 64" in msl) == (bool(plain) or mask == 3)
    assert ("// source KV loop stop" in msl) == (mask == 3)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_the_standard_forward_at_32_by_64_is_lowered_to_the_matrix_shader(dtype, causal):
    constants = dict(
        Z=1, stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1, BLOCK_M=32, BLOCK_N=64, HEAD_DIM=64, IS_CAUSAL=causal
    )
    lower, msl = _lowered(_flash_attn_fwd, _uniform(dtype), constants)
    assert MATRIX in msl and TILED not in msl
    assert lower.effective_block_size == 256


@pytest.mark.parametrize(
    "what,types,constants",
    [
        ("a 64 by 32 tile", _uniform("fp16"), _tail_constants(bm=64, bn=32)),
        ("a 64 by 64 tile", _uniform("fp16"), _tail_constants(bm=64, bn=64)),
        ("a Q whose innermost stride is an argument", _uniform("fp16"), _tail_constants(contiguous=False)),
        ("a float32 Q beside float16 K, V and Out", dict(_uniform("fp16"), Q="fp32"), _tail_constants()),
        ("a float32 K beside float16 Q, V and Out", dict(_uniform("fp16"), K="fp32"), _tail_constants()),
        ("a float32 V beside float16 Q, K and Out", dict(_uniform("fp16"), V="fp32"), _tail_constants()),
        ("a float16 Out beside float32 Q, K and V", dict(_uniform("fp32"), Out="fp16"), _tail_constants()),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_other_32_by_64_source_keeps_the_tiled_shader(what, types, constants):
    lower, msl = _lowered(_tail_attention, types, constants)
    assert TILED in msl and MATRIX not in msl
    assert lower.effective_block_size == 1024


def test_the_32_by_32_source_keeps_the_matrix_shader_and_its_own_tail():
    _, msl = _lowered(_tail_attention, _uniform("fp16"), _tail_constants(bn=32))
    assert MATRIX in msl and "source KV tail: 32" in msl and "source KV tail: 64" not in msl


@pytest.mark.parametrize("q_dtype", ["fp16", "bf16"])
def test_a_narrower_q_beside_float32_tiles_is_lowered_to_the_matrix_shader(q_dtype):
    """The matrix shader stages Q through a scalar cast, as it does for a 32 by 32 source."""
    _, msl = _lowered(_tail_attention, dict(_uniform("fp32"), Q=q_dtype), _tail_constants())
    assert MATRIX in msl and TILED not in msl


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_head_dimension_128_at_32_by_64_is_still_refused(dtype):
    with pytest.raises(MetalNonRecoverableError, match="head_dim > 64"):
        _lowered(_tail_attention, _uniform(dtype), _tail_constants(dim=128))


def test_uniform_bfloat16_at_32_by_64_is_still_refused():
    with pytest.raises(MetalNonRecoverableError):
        _lowered(_tail_attention, _uniform("bf16"), _tail_constants())


def _reference(q, k, v, n, mask, plain, bn=64, bm=32):
    """Out in float64: each query block over the keys its source loop visits, padding as the source loads it."""
    dim = q.shape[-1]
    out = np.empty(q.shape[:2] + (n, dim))
    for block in range(-(-n // bm)):
        rows = np.arange(block * bm, min((block + 1) * bm, n))
        stop = min((block + 1) * bm, n) if mask == 3 else n
        visited = -(-stop // bn) * bn
        keys, values = k[:, :, :visited].copy(), v[:, :, :visited].copy()
        if "K" not in plain:
            keys[:, :, n:] = 0
        if "V" not in plain:
            values[:, :, n:] = 0
        scores = (q[:, :, rows] * dim**-0.5) @ keys.swapaxes(-1, -2)
        cols = np.arange(visited)
        if mask == 1:
            scores[..., cols >= n] = -np.inf
        if mask in (3, 6):
            scores[..., rows[:, None] < cols[None, :]] = -np.inf
        with np.errstate(invalid="ignore"):
            p = np.exp(scores - scores.max(-1, keepdims=True))
            out[:, :, rows] = (p @ values) / p.sum(-1, keepdims=True)
    return out


def _tensors(dtype, n, heads, seed, scale=1.0, q_dtype=None):
    """Q of n rows; K and V of whole 64-key blocks with values in the padding; Out with guard rows."""
    rng = np.random.default_rng(seed)
    alloc = -(-n // 64) * 64
    td = TORCH[dtype]
    q = torch.tensor(rng.standard_normal((1, heads, n, 64)) * scale).to(TORCH[q_dtype or dtype])
    k = torch.tensor(rng.standard_normal((1, heads, alloc, 64)) * scale).to(td)
    v = torch.tensor(rng.standard_normal((1, heads, alloc, 64)) + 1.0).to(td)
    out = torch.full((1, heads, n + 32, 64), SENTINEL, dtype=td)
    out[:, :, :n] = float("nan")
    return q, k, v, out


def _launch_tail(q, k, v, out, n, mask, plain, bn=64):
    heads = q.shape[1]
    return _tail_attention[(triton.cdiv(n, 32), heads)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1,
        heads,
        n,
        BLOCK_M=32,
        BLOCK_N=bn,
        HEAD_DIM=64,
        MASK=mask,
        START=0,
        STEP=bn,
        FILL_K=0.0,
        FILL_V=0.0,
        PLAIN_K="K" in plain,
        PLAIN_V="V" in plain,
    )


def _check(got, out_before, inputs, inputs_before, want, n, tolerance=0.002):
    for after, before in zip(inputs, inputs_before):
        assert torch.equal(after.cpu(), before), "an input changed"
    got = got.cpu().double().numpy()
    assert np.array_equal(got[:, :, n:], out_before[:, :, n:].double().numpy()), "a row past the length was written"
    want_nan = np.isnan(want)
    assert np.array_equal(np.isnan(got[:, :, :n]), want_nan)
    np.testing.assert_allclose(got[:, :, :n][~want_nan], want[~want_nan], atol=tolerance, rtol=tolerance)


CASES = [
    # mask, n, plain. 0: zero-filled keys past the length are scored; 1: they are masked;
    # 2: a row mask; 3: causal, the loop stopping at the query block's end; 6: causal, the whole loop.
    (0, 64, ""),
    (0, 128, ""),
    (0, 33, ""),
    (0, 63, ""),
    (0, 65, ""),
    (0, 96, ""),
    (0, 127, ""),
    (0, 129, ""),
    (0, 200, ""),
    (1, 65, ""),
    (1, 96, ""),
    (1, 127, ""),
    (2, 65, ""),
    (2, 96, ""),
    (3, 64, ""),
    (3, 65, ""),
    (3, 96, ""),
    (3, 200, ""),
    (6, 65, ""),
    (6, 96, ""),
    (6, 128, ""),
    (0, 65, "K"),
    (0, 96, "V"),
    (0, 127, "KV"),
    (1, 96, "KV"),
    (3, 96, "KV"),
    (6, 96, "K"),
    (6, 65, "V"),
]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("mask,n,plain", CASES)
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_the_matrix_shader_computes_a_32_by_64_source(dtype, mask, n, plain, executed, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    q, k, v, out = _tensors(dtype, n, 2, 1000 * mask + n)
    want = _reference(q.double().numpy(), k.double().numpy(), v.double().numpy(), n, mask, plain)
    before = [x.clone() for x in (q, k, v)]
    out_before = out.clone()
    q, k, v, out = [x.to("mps") for x in (q, k, v, out)]
    compiled = _launch_tail(q, k, v, out, n, mask, plain)
    torch.mps.synchronize()
    assert MATRIX in _executed_source(executed, compiled)
    _check(out, out_before, (q, k, v), before, want, n)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("row", [40, 70, 90, 100])
@pytest.mark.parametrize("role", ["K", "V"])
@pytest.mark.parametrize("mask", [3, 6])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_a_nan_reaches_exactly_the_query_blocks_whose_loop_visits_its_key(
    dtype, mask, role, row, executed, monkeypatch
):
    """A masked score still multiplies what was loaded: 0 * NaN is NaN.

    With 64 keys a block, the query block of rows 0 to 31 visits keys 0 to 63: a NaN in V at key
    40 reaches it, where a 32-wide source block would not have loaded that key. A causal loop
    that stops at the query block's end does not visit key 70 until the block of rows 64 to 95.
    Key 100 is past the length of 96 and is not loaded at all.
    """
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    n = 96
    q, k, v, out = _tensors(dtype, n, 2, 5000 + row)
    (k if role == "K" else v)[:, :, row] = float("nan")
    want = _reference(q.double().numpy(), k.double().numpy(), v.double().numpy(), n, mask, "")
    # Written out here, apart from the reference: a key at or past the length is never loaded; a
    # V key reaches every row of a block that visits it; a K key reaches the rows whose causal
    # mask lets its score through, the others replacing that score with -inf.
    rows = np.arange(n)
    visited = np.array([-(-(min((r // 32 + 1) * 32, n) if mask == 3 else n) // 64) * 64 for r in rows])
    reached = (row < n) & (row < visited) & ((rows >= row) | (role == "V"))
    assert np.array_equal(np.isnan(want).all(-1), np.broadcast_to(reached, (1, 2, n)))
    assert np.array_equal(np.isnan(want).any(-1), np.broadcast_to(reached, (1, 2, n)))
    before = [x.clone() for x in (q, k, v)]
    out_before = out.clone()
    q, k, v, out = [x.to("mps") for x in (q, k, v, out)]
    compiled = _launch_tail(q, k, v, out, n, mask, "")
    torch.mps.synchronize()
    assert MATRIX in _executed_source(executed, compiled)
    got = out.cpu().double().numpy()
    assert np.array_equal(got[:, :, n:], out_before[:, :, n:].double().numpy())
    assert np.array_equal(np.isnan(got[:, :, :n]), np.isnan(want))
    keep = ~np.isnan(want)
    np.testing.assert_allclose(got[:, :, :n][keep], want[keep], atol=0.002, rtol=0.002)
    for after, original in zip((q, k, v), before):
        assert torch.equal(torch.nan_to_num(after.cpu(), nan=123.0), torch.nan_to_num(original, nan=123.0))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("mask,n", [(0, 96), (1, 127), (3, 96)])
def test_scores_that_no_float16_holds_are_computed_in_float32(mask, n, executed, monkeypatch):
    """Float16 Q and K of magnitude near 150: a score passes 65504, and the softmax is one key's."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    q, k, v, out = _tensors("fp16", n, 2, 77 + n, scale=150.0)
    want = _reference(q.double().numpy(), k.double().numpy(), v.double().numpy(), n, mask, "")
    scores = (q.double().numpy() * 0.125) @ np.where(
        np.arange(k.shape[2])[:, None] < n, k.double().numpy(), 0
    ).swapaxes(-1, -2)
    assert np.abs(scores).max() > 65504 and np.isfinite(want).all()
    before = [x.clone() for x in (q, k, v)]
    out_before = out.clone()
    q, k, v, out = [x.to("mps") for x in (q, k, v, out)]
    compiled = _launch_tail(q, k, v, out, n, mask, "")
    torch.mps.synchronize()
    assert MATRIX in _executed_source(executed, compiled)
    _check(out, out_before, (q, k, v), before, want, n)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("bn", [64, 32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_the_standard_forward_at_sequence_1024_and_eight_heads(dtype, causal, bn, executed, monkeypatch):
    """The timed call: batch 1, 8 heads, 1024 keys, head dimension 64; 32 by 32 beside it as a control."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    n, heads = 1024, 8
    q, k, v, _ = _tensors(dtype, n, heads, 9)
    want = _reference(q.double().numpy(), k.double().numpy(), v.double().numpy(), n, 6 if causal else 0, "", bn=bn)
    before = [x.clone() for x in (q, k, v)]
    q, k, v = [x.to("mps") for x in (q, k, v)]
    out = torch.full_like(q, float("nan"))
    compiled = _flash_attn_fwd[(n // 32, heads)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1,
        heads,
        n,
        BLOCK_M=32,
        BLOCK_N=bn,
        HEAD_DIM=64,
        IS_CAUSAL=causal,
    )
    torch.mps.synchronize()
    assert MATRIX in _executed_source(executed, compiled)
    for after, original in zip((q, k, v), before):
        assert torch.equal(after.cpu(), original)
    np.testing.assert_allclose(out.cpu().double().numpy(), want, atol=0.002, rtol=0.002)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize(
    "types,contiguous",
    [(dict(_uniform("fp16"), Q="fp32"), True), (dict(_uniform("fp16"), K="fp32"), True), (_uniform("fp16"), False)],
    ids=["float32 Q", "float32 K", "strided Q"],
)
def test_a_source_left_on_the_tiled_shader_still_computes(types, contiguous, executed, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    n = 96
    q, k, v, out = _tensors("fp16", n, 2, 31)
    q, k = q.to(TORCH[types["Q"]]), k.to(TORCH[types["K"]])
    want = _reference(q.double().numpy(), k.double().numpy(), v.double().numpy(), n, 0, "")
    out_before = out.clone()
    q, k, v, out = [x.to("mps") for x in (q, k, v, out)]
    if not contiguous:
        wide = torch.zeros(1, 2, n, 128, dtype=q.dtype, device="mps")
        wide[..., ::2] = q
        q = wide[..., ::2]
    compiled = _launch_tail(q, k, v, out, n, 0, "")
    torch.mps.synchronize()
    assert TILED in _executed_source(executed, compiled)
    got = out.cpu().double().numpy()
    assert np.array_equal(got[:, :, n:], out_before[:, :, n:].double().numpy())
    np.testing.assert_allclose(got[:, :, :n], want, atol=0.002, rtol=0.002)


_WIDE = {}


def _wide_product(constant, root):
    """The standard forward with Q's product formed in float32: ``q.to(tl.float32) * constant``."""
    if constant not in _WIDE:
        src = inspect.getsource(_flash_attn_fwd.fn)
        old = "    q = q * qk_scale\n"
        assert src.count(old) == 1
        path = root / f"wide_product_{len(_WIDE)}.py"
        path.write_text(
            "import triton\nimport triton.language as tl\n\n\n"
            + src.replace(old, f"    q = q.to(tl.float32) * {constant!r}\n")
        )
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        _WIDE[constant] = getattr(module, _flash_attn_fwd.fn.__name__)
    return _WIDE[constant]


@pytest.fixture(scope="module")
def wide_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("wide_product_64")


# 0.25 is the score multiply itself; 0.3 is staged as a float32 product in the matrix shader; with
# 3e38 or 1e-30 the product leaves float32, which the half tiles do not follow: the tiled shader
# stages Q in float32 and computed that source before.
@pytest.mark.parametrize("constant,marker", [(0.25, MATRIX), (0.3, MATRIX), (3e38, TILED), (1e-30, TILED)])
def test_a_float32_product_the_half_tiles_cannot_stage_keeps_the_tiled_shader(constant, marker, wide_dir):
    constants = dict(
        Z=1, stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1, BLOCK_M=32, BLOCK_N=64, HEAD_DIM=64, IS_CAUSAL=False
    )
    _, msl = _lowered(_wide_product(constant, wide_dir), _uniform("fp16"), constants)
    assert marker in msl and (TILED if marker == MATRIX else MATRIX) not in msl


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("constant,marker", [(0.25, MATRIX), (0.3, MATRIX), (1e-30, TILED)])
def test_a_float32_product_of_float16_values_is_computed_on_either_shader(
    constant, marker, wide_dir, executed, monkeypatch
):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    n, heads = 96, 2
    q, k, v, out = _tensors("fp16", n, heads, 53)
    keys = np.where(np.arange(k.shape[2])[:, None] < n, k.double().numpy(), 0.0)
    values = np.where(np.arange(v.shape[2])[:, None] < n, v.double().numpy(), 0.0)
    scores = (q.double().numpy() * float(np.float32(constant))) @ keys.swapaxes(-1, -2)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    want = (p @ values) / p.sum(-1, keepdims=True)
    before = [x.clone() for x in (q, k, v)]
    out_before = out.clone()
    q, k, v, out = [x.to("mps") for x in (q, k, v, out)]
    compiled = _wide_product(constant, wide_dir)[(triton.cdiv(n, 32), heads)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1,
        heads,
        n,
        BLOCK_M=32,
        BLOCK_N=64,
        HEAD_DIM=64,
        IS_CAUSAL=False,
    )
    torch.mps.synchronize()
    assert marker in _executed_source(executed, compiled)
    _check(out, out_before, (q, k, v), before, want, n)


def _strided(q):
    """The same Q values at an innermost stride of two: the source the tiled shader keeps."""
    wide = torch.zeros(q.shape[:3] + (128,), dtype=q.dtype, device=q.device)
    wide[..., ::2] = q
    return wide[..., ::2]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize(
    "mask,n,plain,scale,nan",
    [
        (0, 96, "", 1.0, None),
        (0, 127, "KV", 1.0, None),
        (1, 65, "", 1.0, None),
        (3, 200, "", 1.0, None),
        (6, 96, "K", 1.0, None),
        (0, 96, "", 150.0, None),
        (3, 96, "", 150.0, None),
        (3, 96, "", 1.0, ("V", 40)),
        (3, 96, "", 1.0, ("K", 70)),
        (6, 96, "", 1.0, ("K", 40)),
    ],
)
@pytest.mark.parametrize("dtype,q_dtype", [("fp16", "fp16"), ("fp32", "fp32"), ("fp32", "fp16"), ("fp32", "bf16")])
def test_the_matrix_shader_and_the_tiled_shader_agree_on_one_source(
    dtype, q_dtype, mask, n, plain, scale, nan, monkeypatch
):
    """One source and one set of values: contiguous it takes the matrix shader, with Q strided the tiled one."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    q, k, v, out = _tensors(dtype, n, 2, 400 + n + mask, scale=scale, q_dtype=q_dtype)
    if nan:
        (k if nan[0] == "K" else v)[:, :, nan[1]] = float("nan")
    want = _reference(q.double().numpy(), k.double().numpy(), v.double().numpy(), n, mask, plain)
    results = {}
    for marker in (MATRIX, TILED):
        qd, kd, vd, od = [x.to("mps") for x in (q, k, v, out)]
        compiled = _launch_tail(_strided(qd) if marker == TILED else qd, kd, vd, od, n, mask, plain)
        torch.mps.synchronize()
        assert marker in compiled.asm["msl"]
        results[marker] = od.cpu().double().numpy()
        assert np.array_equal(results[marker][:, :, n:], out[:, :, n:].double().numpy())
    matrix, tiled = results[MATRIX][:, :, :n], results[TILED][:, :, :n]
    assert np.array_equal(np.isnan(matrix), np.isnan(want)) and np.array_equal(np.isnan(tiled), np.isnan(want))
    keep = ~np.isnan(want)
    close = 0.002 if dtype == "fp16" else 0.0002
    np.testing.assert_allclose(matrix[keep], tiled[keep], atol=close, rtol=close)
    np.testing.assert_allclose(matrix[keep], want[keep], atol=0.002, rtol=0.002)


def _score_scaled(root):
    """The standard forward with its constant on the finished score, so that Q is used as loaded."""
    if "post" not in _WIDE:
        src = inspect.getsource(_flash_attn_fwd.fn)
        dot = "        qk = tl.dot(q, tl.trans(k).to(q.dtype))\n"
        assert src.count("    q = q * qk_scale\n") == 1 and src.count(dot) == 1
        src = src.replace("    q = q * qk_scale\n", "").replace(dot, dot[:-1] + " * qk_scale\n")
        path = root / "score_scaled.py"
        path.write_text("import triton\nimport triton.language as tl\n\n\n" + src)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        _WIDE["post"] = getattr(module, _flash_attn_fwd.fn.__name__)
    return _WIDE["post"]


def test_a_float32_q_with_no_constant_of_its_own_keeps_the_tiled_shader(wide_dir):
    """The half tiles would round such a Q to float16; no Q constant is there to refuse the staging."""
    constants = dict(
        Z=1, stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1, BLOCK_M=32, BLOCK_N=64, HEAD_DIM=64, IS_CAUSAL=False
    )
    kernel = _score_scaled(wide_dir)
    _, msl = _lowered(kernel, dict(_uniform("fp16"), Q="fp32"), constants)
    assert TILED in msl and MATRIX not in msl
    _, msl = _lowered(kernel, _uniform("fp16"), constants)
    assert MATRIX in msl and TILED not in msl
