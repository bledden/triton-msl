"""A load mask written along the other axis of a square tile leaves out the cells the source leaves out.

The biased attention kernels mask their Q, K and bias loads with the sequence boundary:
`tl.load(q_ptrs, mask_m[:, None])` leaves out the query rows past the length. On a 32 by 32
tile `mask_m[None, :]` has the same shape and leaves out other cells: the head columns `d`
with `start_m + d >= N`, in every row. Those cells are zero in the source (a masked load with
no `other` supplies zero, like one with `other=0.0`), so in the last, partial tile the scores of
live rows change.

The template used to run its own row and key boundaries whatever the source wrote, and then
refused these masks. It now applies the boundary along the axis the source wrote it on, for
the Q, K and bias loads of the biased kernels. The V load, the selects and the stores keep
their one axis, and so do the other attention routes.

Every expectation here is the source run tile by tile in float64.
"""

import math

import pytest
import torch
import triton

from tests.test_fa_biased_routing import _biased_fa, _biased_tri_fa
from tests.test_fa_launch_tile_contracts import _variant
from tests.test_template_scalar_abi import _lower
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0
BM = 32
AXIS = "does not vary along the template's axis for this role"
OTHER = "the load's 'other' value is not zero"
BOUNDARY = "the mask is not a sequence-boundary comparison"

# The source lines, per kernel: Q load, K load, bias load, V load.
LINES = {
    "2d": dict(
        q="q_block = tl.load(q_ptrs, mask_m[:, None])",
        k="kt_block = tl.load(kt_ptrs, mask_n[None, :])",
        b="b_block = tl.load(b_ptrs, mask_m[:, None] & mask_n[None, :])",
        v="v_block = tl.load(v_ptrs, mask_n[:, None])",
        sel="scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, neg_inf)",
        rows="mask_m",
        keys="mask_n",
        row_at="m_idxs",
        key_at="(n_idxs + start_n)",
    ),
    "tri": dict(
        q="q_block = tl.load(q_ptrs, mask_j[:, None])",
        k="kt_block = tl.load(kt_ptrs, mask_k[None, :])",
        b="b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :])",
        v="v_block = tl.load(v_ptrs, mask_k[:, None])",
        sel="scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)",
        rows="mask_j",
        keys="mask_k",
        row_at="j_idxs",
        key_at="(k_idxs + start_k)",
    ),
}

# name -> (the loads rewritten, the masks the reference turns). "bq" is the query boundary of
# the bias load, "bk" its key boundary.
TURNED = {
    "q": ("q", {"q"}),
    "k": ("k", {"k"}),
    "bias": ("b", {"bq", "bk"}),
    "bias_rows_only": ("b", {"bq"}),
    "bias_keys_only": ("b", {"bk"}),
    "q_and_k": ("qk", {"q", "k"}),
    "q_and_bias": ("qb", {"q", "bq", "bk"}),
    "all_three": ("qkb", {"q", "k", "bq", "bk"}),
}


def _edits(mode, loads, turned, other=""):
    """The source edits that turn the named masks; ``other`` is appended to each rewritten load."""
    line, rows, keys = LINES[mode], LINES[mode]["rows"], LINES[mode]["keys"]
    edits = []
    if "q" in loads:
        edits.append((line["q"], f"q_block = tl.load(q_ptrs, {rows}[None, :]{other})"))
    if "k" in loads:
        edits.append((line["k"], f"kt_block = tl.load(kt_ptrs, {keys}[:, None]{other})"))
    if "b" in loads:
        bq = f"{rows}[None, :]" if "bq" in turned else f"{rows}[:, None]"
        bk = f"{keys}[:, None]" if "bk" in turned else f"{keys}[None, :]"
        edits.append((line["b"], f"b_block = tl.load(b_ptrs, {bq} & {bk}{other})"))
    return edits


def _source(q, k, v, b, mask, n, dim, sentinel, turned, block_n, elem):
    """One (batch, head) slice of the source in float64: returns Out [rows, dim] and Lse [rows].

    ``q``, ``k``, ``v``: [rows, dim]; ``b``: [rows, keys]; ``mask``: [keys]. A masked-off cell of
    a load is zero. Rows at or past ``n`` come back as POISON, as the masked stores leave them.
    ``turned`` names the boundaries written along the other axis; ``<name>_too`` says the
    load carries that boundary along its own axis as well.
    """
    rows = q.shape[0]
    out = torch.full((rows, dim), POISON, dtype=torch.float64)
    lse = torch.full((rows,), POISON, dtype=torch.float64)
    d = torch.arange(dim)
    scale = float(torch.tensor(dim**-0.5, dtype=elem))
    inv_ln2, ln2 = 1.4426950408889634, 0.6931471824645996
    zero = torch.zeros((), dtype=torch.float64)
    for start_m in range(0, rows, BM):
        m = start_m + torch.arange(BM)
        mask_m = m < n
        q_mask = (start_m + d < n)[None, :].expand(BM, dim) if "q" in turned else mask_m[:, None].expand(BM, dim)
        if "q_too" in turned:
            q_mask = q_mask & mask_m[:, None]
        q_block = torch.where(q_mask, q[m].double(), zero).to(elem).double() * scale
        q_block = q_block.to(elem).double()
        top = torch.full((BM,), -math.inf, dtype=torch.float64)
        den = torch.zeros(BM, dtype=torch.float64)
        acc = torch.zeros(BM, dim, dtype=torch.float64)
        for start_n in range(0, n, block_n):
            keys = start_n + torch.arange(block_n)
            mask_n = keys < n
            k_mask = (
                (start_n + d < n)[:, None].expand(dim, block_n)
                if "k" in turned
                else mask_n[None, :].expand(dim, block_n)
            )
            if "k_too" in turned:
                k_mask = k_mask & mask_n[None, :]
            kt = torch.where(k_mask, k[keys].double().T, zero)
            b_rows = (start_m + torch.arange(block_n) < n)[None, :] if "bq" in turned else mask_m[:, None]
            b_keys = (start_n + torch.arange(BM) < n)[:, None] if "bk" in turned else mask_n[None, :]
            if "bq_too" in turned:
                b_rows = b_rows & mask_m[:, None]
            if "bk_too" in turned:
                b_keys = b_keys & mask_n[None, :]
            bias = torch.where(b_rows & b_keys, b[m][:, keys].double(), zero)
            masked = torch.where(mask_n, mask[keys], torch.zeros((), dtype=mask.dtype)) != 0
            scores = (bias.float().double() + q_block @ kt) * inv_ln2
            scores = torch.where(masked[None, :], torch.tensor(sentinel, dtype=torch.float64), scores)
            scores = torch.where(mask_m[:, None] & mask_n[None, :], scores, torch.tensor(sentinel, dtype=torch.float64))
            block_top = torch.maximum(top, scores.max(1).values)
            weights = torch.exp2(scores - block_top[:, None])
            shrink = torch.exp2(top - block_top)
            den = den * shrink + weights.sum(1)
            values = torch.where(mask_n[:, None], v[keys].double(), zero)
            acc = acc * shrink[:, None] + weights.to(elem).double() @ values
            top = block_top
        live = m[mask_m]
        out[live] = (acc / den[:, None])[mask_m]
        lse[live] = (top * ln2 + torch.log(den))[mask_m]
    return out, lse


def _tensors(lead, rows, keys, dim, elem, seed, bias_lead=None, mask_lead=None):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(*lead, rows, dim, generator=gen).to(elem)
    k, v = [torch.randn(*lead, keys, dim, generator=gen).to(elem) for _ in range(2)]
    b = torch.randn(*(lead if bias_lead is None else bias_lead), rows, keys, generator=gen).to(elem)
    mask = (torch.rand(*(lead if mask_lead is None else mask_lead), keys, generator=gen) < 0.25).to(torch.uint8)
    mask[..., 0] = 0  # no query row is left without a key
    return q, k, v, b, mask


def _launch(mode, fn, n, dim, elem, turned, sentinel=-1e9, block_n=32, seed=2101):
    """Launch one kernel on tiles that hold ``n`` rows; returns (Out, Lse, expected Out, expected Lse)."""
    tiles = triton.cdiv(n, BM)
    rows, keys = tiles * BM, triton.cdiv(n, block_n) * block_n
    st = lambda t: tuple(t.stride())  # noqa: E731
    if mode == "2d":
        z, h = 1, 2
        q, k, v, b, mask = _tensors((z, h), rows, keys, dim, elem, seed)
        slices = [((a, c), (a, c), (a, c)) for a in range(z) for c in range(h)]
    else:
        hc, hh, inst = 4, 2, 2
        q, k, v, b, mask = _tensors(
            (hc, inst), rows, keys, dim, elem, seed, bias_lead=(hc,), mask_lead=(hc // hh, inst)
        )
        slices = [((a, c), (a,), (a // hh, c)) for a in range(hc) for c in range(inst)]
    o = torch.full(q.shape, POISON, dtype=elem, device="mps")
    lse = torch.full(q.shape[:-1], POISON, device="mps")
    dev = [x.to("mps") for x in (q, k, v, b, mask)]
    args = [o, *st(o), lse, *st(lse)]
    for t in dev:
        args += [t, *st(t)]
    if mode == "2d":
        fn[(tiles, z * h)](*args, dim**-0.5, sentinel, z, h, n, dim, BM, block_n)
    else:
        fn[(tiles, inst, hc)](*args, dim**-0.5, sentinel, n, hh, dim, BM, block_n)
    torch.mps.synchronize()
    o_exp = torch.full(o.shape, POISON, dtype=torch.float64)
    lse_exp = torch.full(lse.shape, POISON, dtype=torch.float64)
    for at, bias_at, mask_at in slices:
        o_exp[at], lse_exp[at] = _source(
            q[at], k[at], v[at], b[bias_at], mask[mask_at], n, dim, sentinel, turned, block_n, elem
        )
    return o, lse, o_exp, lse_exp


def _spelled(mode, new):
    line = LINES[mode]
    return new.format(**{name: line[name] for name in ("rows", "keys", "row_at", "key_at")})


def _rewritten(mode, mode_line, new):
    return _variant(KERNEL[mode], (LINES[mode][mode_line], _spelled(mode, new)))


def _assert_source(result, tol):
    """The written cells are the source's, and the cells the source does not store stay poison."""
    for got, expected in zip(result[:2], result[2:]):
        got = got.cpu().double()
        written = expected != POISON
        assert torch.equal(got != POISON, written)
        torch.testing.assert_close(got[written], expected[written], atol=tol, rtol=tol)


TOL = {torch.float32: 2e-3, torch.float16: 2.5e-2, torch.bfloat16: 8e-2}
KERNEL = {"2d": _biased_fa, "tri": _biased_tri_fa}


@pytest.fixture
def fa_hits(monkeypatch):
    """Record whether the attention replacement dispatch ran (the driver imports it per call)."""
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits, real = [], fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


@requires_mps
@pytest.mark.parametrize("other", ["", ", other=0.0"], ids=["no other", "other zero"])
@pytest.mark.parametrize("n", [48, 33, 63, 17, 64, 100])
@pytest.mark.parametrize("name", sorted(TURNED))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_mask_along_the_other_axis_leaves_out_the_source_cells(mode, name, n, other, fa_hits):
    loads, turned = TURNED[name]
    fn = _variant(KERNEL[mode], *_edits(mode, loads, turned, other))
    _assert_source(_launch(mode, fn, n, 32, torch.float32, turned), TOL[torch.float32])
    assert fa_hits == [True]


@requires_mps
@pytest.mark.parametrize("elem", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
@pytest.mark.parametrize("n", [48, 33, 100])
@pytest.mark.parametrize("name", sorted(TURNED))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_mask_along_the_other_axis_on_half_types(mode, name, n, elem, fa_hits):
    loads, turned = TURNED[name]
    fn = _variant(KERNEL[mode], *_edits(mode, loads, turned))
    _assert_source(_launch(mode, fn, n, 32, elem, turned), TOL[elem])
    assert fa_hits == [True]


@requires_mps
@pytest.mark.parametrize("elem", [torch.float32, torch.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("n", [48, 40, 100])
@pytest.mark.parametrize("name", ["bias", "bias_rows_only", "bias_keys_only"])
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_bias_mask_along_the_other_axis_at_head_dimension_64(mode, name, n, elem, fa_hits):
    # The bias tile is 32 by 32 whatever the head dimension; the Q and K tiles are not square here.
    loads, turned = TURNED[name]
    fn = _variant(KERNEL[mode], *_edits(mode, loads, turned))
    _assert_source(_launch(mode, fn, n, 64, elem, turned), TOL[elem])
    assert fa_hits == [True]


@requires_mps
@pytest.mark.parametrize("elem", [torch.float32, torch.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("n", [48, 100, 130])
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_key_mask_along_the_head_axis_of_a_64_key_block(mode, n, elem, fa_hits):
    # The K tile is [64 head columns, 64 keys]: square, so its key boundary fits the head axis.
    fn = _variant(KERNEL[mode], *_edits(mode, "k", {"k"}))
    _assert_source(_launch(mode, fn, n, 64, elem, {"k"}, block_n=64), TOL[elem])
    assert fa_hits == [True]


@requires_mps
@pytest.mark.parametrize("sentinel", [float("-inf"), 0.0, -3.5])
@pytest.mark.parametrize("n", [48, 33])
@pytest.mark.parametrize("name", ["q", "k", "bias", "all_three"])
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_mask_along_the_other_axis_with_another_sentinel(mode, name, n, sentinel, fa_hits):
    # A finite sentinel gives the keys past the length a weight; their V rows are zero.
    loads, turned = TURNED[name]
    fn = _variant(KERNEL[mode], *_edits(mode, loads, turned))
    _assert_source(_launch(mode, fn, n, 32, torch.float32, turned, sentinel=sentinel), TOL[torch.float32])
    assert fa_hits == [True]


@requires_mps
@pytest.mark.parametrize("mode", ["2d", "tri"])
@pytest.mark.parametrize("n", [48, 33])
def test_the_bound_written_first_is_the_same_mask(mode, n, fa_hits):
    line = LINES[mode]
    rows = "m_idxs" if mode == "2d" else "j_idxs"
    fn = _variant(KERNEL[mode], (line["q"], f"q_block = tl.load(q_ptrs, (N > {rows})[None, :])"))
    _assert_source(_launch(mode, fn, n, 32, torch.float32, {"q"}), TOL[torch.float32])
    assert fa_hits == [True]


# One boundary along both axes of a Q or K load: (which load, the mask, what the reference applies).
BOTH_AXES = {
    "q": ("q", "q_block = tl.load(q_ptrs, {rows}[:, None] & {rows}[None, :])", {"q", "q_too"}),
    "k": ("k", "kt_block = tl.load(kt_ptrs, {keys}[None, :] & {keys}[:, None])", {"k", "k_too"}),
}


@requires_mps
@pytest.mark.parametrize("sentinel", [-1e9, 0.0])
@pytest.mark.parametrize("n", [48, 33, 100])
@pytest.mark.parametrize("name", sorted(BOTH_AXES))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_boundary_along_both_axes_of_a_load(mode, name, n, sentinel, fa_hits):
    which, new, turned = BOTH_AXES[name]
    fn = _rewritten(mode, which, new)
    _assert_source(_launch(mode, fn, n, 32, torch.float32, turned, sentinel=sentinel), TOL[torch.float32])
    assert fa_hits == [True]


@requires_mps
@pytest.mark.parametrize("n", [48, 33, 64])
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_the_masks_as_written_are_computed_as_before(mode, n, fa_hits):
    _assert_source(_launch(mode, _variant(KERNEL[mode]), n, 32, torch.float32, set()), TOL[torch.float32])
    assert fa_hits == [True]


# A vector mask is broadcast like any other operand: along the last axis of its tile.
VECTORS = {
    "q": ("q", "q_block = tl.load(q_ptrs, {rows})", {"q"}),
    "k": ("k", "kt_block = tl.load(kt_ptrs, {keys})", set()),
    "bias": ("b", "b_block = tl.load(b_ptrs, {rows} & {keys})", {"bq"}),
}


@requires_mps
@pytest.mark.parametrize("n", [48, 33, 100])
@pytest.mark.parametrize("name", sorted(VECTORS))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_vector_mask_lies_along_the_last_axis_of_its_tile(mode, name, n, fa_hits):
    which, new, turned = VECTORS[name]
    _assert_source(_launch(mode, _rewritten(mode, which, new), n, 32, torch.float32, turned), TOL[torch.float32])
    assert fa_hits == [True]


REFUSED = {
    "v": ("v", "v_block = tl.load(v_ptrs, {keys}[None, :])", AXIS),
    "v_other_zero": ("v", "v_block = tl.load(v_ptrs, {keys}[None, :], other=0.0)", AXIS),
    "q_other_one": ("q", "q_block = tl.load(q_ptrs, {rows}[None, :], other=1.0)", OTHER),
    "k_other_one": ("k", "kt_block = tl.load(kt_ptrs, {keys}[:, None], other=1.0)", OTHER),
    "bias_other_one": (
        "b",
        "b_block = tl.load(b_ptrs, {rows}[None, :] & {keys}[:, None], other=1.0)",
        OTHER,
    ),
    "select": ("sel", "scores = tl.where({rows}[None, :] & {keys}[:, None], scores, neg_inf)", AXIS),
    # A bias mask is one query boundary and one key boundary, each along one axis.
    "bias_rows_along_both_axes": (
        "b",
        "b_block = tl.load(b_ptrs, {rows}[:, None] & {rows}[None, :] & {keys}[None, :])",
        BOUNDARY,
    ),
    "bias_keys_along_both_axes": (
        "b",
        "b_block = tl.load(b_ptrs, {rows}[:, None] & {keys}[None, :] & {keys}[:, None])",
        BOUNDARY,
    ),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(REFUSED))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_what_the_template_cannot_leave_out_is_refused(mode, name, fa_hits):
    which, new, fragment = REFUSED[name]
    with pytest.raises(MetalNonRecoverableError) as info:
        _launch(mode, _rewritten(mode, which, new), 48, 32, torch.float32, set())
    assert fragment in str(info.value)
    assert fa_hits == []


# ---------------------------------------------------------------------------------------
# No device: which template the lowering asks for, and what the template writes
# ---------------------------------------------------------------------------------------

GUARD = {
    "Q": "(((q_start + dc + c) < N_CTX) ? float(Q[q_base + q_row * q_sm + (dc + c) * q_sk]) : 0.0f)",
    "K": "] = (kv_row < N_CTX && (kv_start + dc + c) < N_CTX)\n",
    "bq": "s += ((q_start + cj) < N_CTX) ? float(Bias[bias_base + q_row * b_sm + kv_row * b_sn]) : 0.0f;",
    "bk": "s += ((kv_start + r) < N_CTX) ? float(Bias[bias_base + q_row * b_sm + kv_row * b_sn]) : 0.0f;",
    "bqbk": (
        "s += ((kv_start + r) < N_CTX && (q_start + cj) < N_CTX)"
        " ? float(Bias[bias_base + q_row * b_sm + kv_row * b_sn]) : 0.0f;"
    ),
}
AS_WRITTEN = {
    "K": "] = (kv_row < N_CTX)\n",
    "Bias": "s += float(Bias[bias_base + q_row * b_sm + kv_row * b_sn]);",
}


@pytest.fixture
def makers(monkeypatch):
    """Record each attention template the biased lowering asks for, with its arguments."""
    import triton_msl.codegen._msl_templates as templates

    calls = []
    for name in ("make_flash_attention_kernel_simdgroup", "make_flash_attention_kernel_tiled"):
        real = getattr(templates, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append((_name.rsplit("_", 1)[1], _real, args, kwargs))
            return _real(*args, **kwargs)

        monkeypatch.setattr(templates, name, spy)
    return calls


def _lowered(mode, edits, float_type="fp32", dim=32, block_n=32):
    """Lower one kernel as a launch on contiguous tensors specializes it; returns the shader text."""
    fn = _variant(KERNEL[mode], *edits)
    if mode == "2d":
        const = dict(DIM=dim, BLOCK_M=BM, BLOCK_N=block_n, BAD_TEMP=False, I3D=False, o_sk=1, q_sk=1, k_sk=1, v_sk=1)
    else:
        const = dict(DIM=dim, BLOCK_J=BM, BLOCK_K=block_n, o_sd=1, q_sd=1, k_sd=1, v_sd=1)
    sig = {}
    for name in fn.arg_names:
        if name in const:
            continue
        if name.endswith("_ptr"):
            sig[name] = "*i8" if name == "mask_ptr" else "*fp32" if name == "lse_ptr" else f"*{float_type}"
        else:
            sig[name] = "fp32" if name in ("sm_scale", "neg_inf") else "i32"
    return _lower(fn, sig, const).lower()


@pytest.mark.parametrize("float_type", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("name", sorted(TURNED))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_mask_along_the_other_axis_takes_the_template_that_can_leave_cells_out(mode, name, float_type, makers):
    loads, turned = TURNED[name]
    text = _lowered(mode, _edits(mode, loads, turned), float_type)
    ((which, _real, _args, kwargs),) = makers
    assert which == "tiled"
    want = {}
    if "q" in turned:
        want["Q"] = ("Q",)
    if "k" in turned:
        want["K"] = ("K",)
    if turned & {"bq", "bk"}:
        want["Bias"] = tuple(leaf for leaf, key in (("K", "bk"), ("Q", "bq")) if key in turned)
    assert kwargs["other_axis_masks"] == want
    assert (GUARD["Q"] in text) == ("q" in turned)
    assert (GUARD["K"] in text) == ("k" in turned)
    # The values are staged into the same array behind the key boundary alone, turned or not.
    assert text.count(AS_WRITTEN["K"]) == (1 if "k" in turned else 2)
    bias = "".join(sorted(turned & {"bq", "bk"}, reverse=True))
    for key in ("bq", "bk", "bqbk"):
        assert (GUARD[key] in text) == (key == bias)
    assert (AS_WRITTEN["Bias"] in text) == (not bias)


@pytest.mark.parametrize("float_type,which", [("fp32", "simdgroup"), ("fp16", "tiled"), ("bf16", "tiled")])
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_the_masks_as_written_keep_their_template_and_its_text(mode, float_type, which, makers):
    text = _lowered(mode, [], float_type)
    ((got, real, args, kwargs),) = makers
    assert got == which
    if which == "tiled":
        assert kwargs["other_axis_masks"] == {}
        # No boundary along the other axis is the text the template wrote before it had any.
        plain = {key: value for key, value in kwargs.items() if key != "other_axis_masks"}
        assert real(*args, **plain) == text
        assert real(*args, **{**plain, "other_axis_masks": None}) == text
        assert real(*args, **{**plain, "other_axis_masks": {"Q": (), "Bias": ()}}) == text
        assert all(guard not in text for guard in GUARD.values())


def _tiled_call(makers, mode="2d", dim=32, block_n=32):
    _lowered(mode, [], "fp16", dim, block_n)
    ((which, real, args, kwargs),) = makers
    assert which == "tiled"
    return real, list(args), kwargs


@pytest.mark.parametrize(
    "masks,fragment",
    [
        ({"V": ("K",)}, "no boundary along the other axis"),
        ({"Q": ("K",)}, "no boundary along the other axis"),
        ({"K": ("Q",)}, "no boundary along the other axis"),
        ({"Bias": ("Q", "V")}, "no boundary along the other axis"),
        ({"Out": ("Q",)}, "no boundary along the other axis"),
    ],
    ids=repr,
)
def test_the_template_refuses_a_boundary_it_has_no_text_for(masks, fragment, makers):
    real, args, kwargs = _tiled_call(makers)
    with pytest.raises(ValueError, match=fragment):
        real(*args, **{**kwargs, "other_axis_masks": masks})


@pytest.mark.parametrize(
    "masks,dim,block_n,changed",
    [
        ({"Q": ("Q",)}, 64, 32, {}),
        ({"K": ("K",)}, 64, 32, {}),
        ({"K": ("K",)}, 32, 64, {}),
        ({"Bias": ("Q",)}, 32, 64, {}),
        ({"Bias": ("K",)}, 32, 64, {}),
        ({"Bias": ("Q", "K")}, 32, 32, {"bias": False}),
    ],
    ids=repr,
)
def test_the_template_refuses_a_boundary_along_an_axis_of_another_length(masks, dim, block_n, changed, makers):
    real, args, kwargs = _tiled_call(makers)
    assert args[:3] == [32, BM, 32]
    args[0], args[2] = dim, block_n
    real(*args, **{**kwargs, **changed})
    with pytest.raises(ValueError, match="needs a square tile"):
        real(*args, **{**kwargs, **changed, "other_axis_masks": masks})


def test_the_template_refuses_a_boundary_with_the_dense_source_replays(makers):
    real, args, kwargs = _tiled_call(makers)
    for extra in (dict(query_tile=32), dict(query_length="N_Q"), dict(source_kv_block=32), dict(unmasked_kv=("K",))):
        with pytest.raises(ValueError, match="not combined with the dense source replays"):
            real(*args, **{**kwargs, **extra, "other_axis_masks": {"Q": ("Q",)}})


# ---------------------------------------------------------------------------------------
# No device: the same masks spelled another way, and what the proof does not read
# ---------------------------------------------------------------------------------------

Q_TURNED = "q_block = tl.load(q_ptrs, {rows}[None, :])"
K_TURNED = "kt_block = tl.load(kt_ptrs, {keys}[:, None])"
# name -> (the line rewritten, the spelling, the line it is the mask of: None for the kernel's own)
SPELLED = {
    "q_expand_dims_0": ("q", "q_block = tl.load(q_ptrs, tl.expand_dims({rows}, 0))", Q_TURNED),
    "q_expand_dims_minus_2": ("q", "q_block = tl.load(q_ptrs, tl.expand_dims({rows}, -2))", Q_TURNED),
    "q_expand_dims_1": ("q", "q_block = tl.load(q_ptrs, tl.expand_dims({rows}, 1))", None),
    "q_expand_dims_minus_1": ("q", "q_block = tl.load(q_ptrs, tl.expand_dims({rows}, -1))", None),
    "q_broadcast_turned": ("q", "q_block = tl.load(q_ptrs, tl.broadcast_to({rows}[None, :], (32, 32)))", Q_TURNED),
    "q_broadcast": ("q", "q_block = tl.load(q_ptrs, tl.broadcast_to({rows}[:, None], (32, 32)))", None),
    "q_index_turned": ("q", "q_block = tl.load(q_ptrs, {row_at}[None, :] < N)", Q_TURNED),
    "q_index": ("q", "q_block = tl.load(q_ptrs, {row_at}[:, None] < N)", None),
    "q_bound_first_turned": ("q", "q_block = tl.load(q_ptrs, N > {row_at}[None, :])", Q_TURNED),
    "q_vector": ("q", "q_block = tl.load(q_ptrs, {rows})", Q_TURNED),
    "k_expand_dims_1": ("k", "kt_block = tl.load(kt_ptrs, tl.expand_dims({keys}, 1))", K_TURNED),
    "k_index_turned": ("k", "kt_block = tl.load(kt_ptrs, {key_at}[:, None] < N)", K_TURNED),
    "k_vector": ("k", "kt_block = tl.load(kt_ptrs, {keys})", None),
    "bias_vectors": (
        "b",
        "b_block = tl.load(b_ptrs, {rows} & {keys})",
        "b_block = tl.load(b_ptrs, {rows}[None, :] & {keys}[None, :])",
    ),
    "bias_expanded_after_the_and": (
        "b",
        "b_block = tl.load(b_ptrs, ({rows} & {keys})[:, None])",
        "b_block = tl.load(b_ptrs, {rows}[:, None] & {keys}[:, None])",
    ),
}


@pytest.mark.parametrize("float_type", ["fp32", "fp16"])
@pytest.mark.parametrize("name", sorted(SPELLED))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_another_spelling_of_a_mask_is_the_same_shader(mode, name, float_type):
    which, new, same = SPELLED[name]
    line = LINES[mode][which]
    want = _lowered(mode, [] if same is None else [(line, _spelled(mode, same))], float_type)
    assert _lowered(mode, [(line, _spelled(mode, new))], float_type) == want
    if same is not None:
        assert want != _lowered(mode, [], float_type)


NOT_THE_BOUNDARY = "the mask is not exactly the template's boundary for this role"
# A reshape gives a coordinate an axis the proof does not read, so none of these is a boundary.
RESHAPED = {
    "q_index_as_a_column": ("q", "q_block = tl.load(q_ptrs, tl.reshape({row_at}, (32, 1)) < N)", NOT_THE_BOUNDARY),
    "q_index_as_a_row": ("q", "q_block = tl.load(q_ptrs, tl.reshape({row_at}, (1, 32)) < N)", NOT_THE_BOUNDARY),
    "q_mask_as_a_column": ("q", "q_block = tl.load(q_ptrs, tl.reshape({rows}, (32, 1)))", BOUNDARY),
    "q_mask_as_a_row": ("q", "q_block = tl.load(q_ptrs, tl.reshape({rows}, (1, 32)))", BOUNDARY),
    "k_index_as_a_row": ("k", "kt_block = tl.load(kt_ptrs, tl.reshape({key_at}, (1, 32)) < N)", NOT_THE_BOUNDARY),
    "k_index_as_a_column": ("k", "kt_block = tl.load(kt_ptrs, tl.reshape({key_at}, (32, 1)) < N)", NOT_THE_BOUNDARY),
    "bias_rows_as_a_column": (
        "b",
        "b_block = tl.load(b_ptrs, (tl.reshape({row_at}, (32, 1)) < N) & {keys}[None, :])",
        NOT_THE_BOUNDARY,
    ),
    "bias_keys_as_a_column": (
        "b",
        "b_block = tl.load(b_ptrs, {rows}[:, None] & (tl.reshape({key_at}, (32, 1)) < N))",
        NOT_THE_BOUNDARY,
    ),
}


@pytest.mark.parametrize("name", sorted(RESHAPED))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_a_reshaped_mask_is_refused(mode, name):
    which, new, fragment = RESHAPED[name]
    with pytest.raises(MetalNonRecoverableError) as info:
        _lowered(mode, [(LINES[mode][which], _spelled(mode, new))])
    assert fragment in str(info.value)


@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_two_bias_loads_with_different_masks_are_refused(mode):
    line = LINES[mode]
    new = (
        line["b"]
        + _spelled(mode, "\n        b_two = tl.load(b_ptrs, {rows}[None, :] & {keys}[:, None])")
        + "\n        sm_denom = sm_denom + 0.0 * tl.sum(b_two, 1)"
    )
    with pytest.raises(MetalNonRecoverableError) as info:
        _lowered(mode, [(line["b"], new)])
    assert "two loads of this tensor carry different masks" in str(info.value)


# A boundary along the other axis has the length of its own tile side. The compiler front end
# does not broadcast it to an axis of another length, so the lowering never sees one.
NOT_SQUARE = {
    "q_at_head_dim_64": ("q", Q_TURNED, dict(dim=64)),
    "q_vector_at_head_dim_64": ("q", "q_block = tl.load(q_ptrs, {rows})", dict(dim=64)),
    "k_at_head_dim_64": ("k", K_TURNED, dict(dim=64)),
    "k_at_a_64_key_block": ("k", K_TURNED, dict(block_n=64)),
    "bias_rows_at_a_64_key_block": (
        "b",
        "b_block = tl.load(b_ptrs, {rows}[None, :] & {keys}[None, :])",
        dict(block_n=64),
    ),
    "bias_keys_at_a_64_key_block": (
        "b",
        "b_block = tl.load(b_ptrs, {rows}[:, None] & {keys}[:, None])",
        dict(block_n=64),
    ),
}


@pytest.mark.parametrize("name", sorted(NOT_SQUARE))
@pytest.mark.parametrize("mode", ["2d", "tri"])
def test_the_front_end_does_not_broadcast_a_boundary_to_an_axis_of_another_length(mode, name):
    which, new, shape = NOT_SQUARE[name]
    with pytest.raises(triton.CompilationError):
        _lowered(mode, [(LINES[mode][which], _spelled(mode, new))], **shape)
