"""What the biased attention backward kernels compute past the end of the sequence.

A masked ``tl.load`` supplies its ``other`` where the mask drops the element, and the score
of every cell past ``N`` is the kernel's sentinel. In a partial tile those cells are terms of
the stored sums: the dK / dV kernel sums each key column over the query rows of the tile, the
dQ kernel sums each query row over its keys. The templates used to count zero for all of
them, which is the source only when every ``other`` is zero and every such term is zero
times a finite number: not with a sentinel whose ``exp2(sentinel - lse)`` is infinite or NaN.

Here every load of trifast's ``_bwd_kv`` and ``_bwd_q`` (copied in
tests/test_fa_bwd_routing.py) is given an explicit ``other``, the sentinel is a launch
argument, and the result is compared with a tile-by-tile evaluation of the source that
rounds where the source rounds. Tensors are longer than ``N`` and hold another value there,
which no masked load may read.
"""

import math

import pytest
import torch
import triton

import tests.test_fa_bwd_address_witness as witness
import triton_msl.codegen._msl_templates as templates
import triton_msl.codegen.generic_lowerer as lowering
from tests.test_fa_bwd_routing import _bwd_b, _bwd_kv, _bwd_q
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = 77.0
BEYOND = 3.0  # what the tensors hold past N
BLOCK = 32
INV_LN2 = 1.4426950408889634
NAN, INF = float("nan"), float("inf")
KERNELS = {"kv": _bwd_kv, "q": _bwd_q, "b": _bwd_b}
ORDER = {
    "kv": ("delta", "q", "k", "v", "b", "lse", "mask", "do", "dk", "dv"),
    "q": ("delta", "q", "k", "v", "b", "lse", "mask", "o", "do", "dq"),
}
OUTPUTS = {"kv": ("dk", "dv"), "q": ("dq", "delta")}

# Each load of the two kernels, with a place for its ``other``.
LOADS = {
    "kv": {
        "v": ("vt_block = tl.load(vt_ptrs, mask_k[None, :])", "vt_block = tl.load(vt_ptrs, mask_k[None, :], other={})"),
        "k": ("kt_block = tl.load(kt_ptrs, mask_k[None, :])", "kt_block = tl.load(kt_ptrs, mask_k[None, :], other={})"),
        "mask": ("m_block = tl.load(mask_ptrs, mask_k, ", "m_block = tl.load(mask_ptrs, mask_k, other={}, "),
        "q": ("q_block = tl.load(q_ptrs, mask_j[:, None])", "q_block = tl.load(q_ptrs, mask_j[:, None], other={})"),
        "b": (
            "b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :])",
            "b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :], other={})",
        ),
        "lse": ("row_max = tl.load(l_ptrs, mask=mask_j)", "row_max = tl.load(l_ptrs, mask=mask_j, other={})"),
        "do": ("do = tl.load(do_ptrs, mask_j[:, None])", "do = tl.load(do_ptrs, mask_j[:, None], other={})"),
        "delta": ("delta = tl.load(d_ptrs, mask_j)", "delta = tl.load(d_ptrs, mask_j, other={})"),
    },
    "q": {
        "q": ("q_block = tl.load(q_ptrs, mask_j[:, None])", "q_block = tl.load(q_ptrs, mask_j[:, None], other={})"),
        "lse": ("sm_denom = tl.load(l_ptrs, mask_j)", "sm_denom = tl.load(l_ptrs, mask_j, other={})"),
        "do": (
            "do_block = tl.load(do_ptrs, mask_j[:, None])",
            "do_block = tl.load(do_ptrs, mask_j[:, None], other={})",
        ),
        "o": ("o_block = tl.load(o_ptrs, mask_j[:, None])", "o_block = tl.load(o_ptrs, mask_j[:, None], other={})"),
        "b": (
            "b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :])",
            "b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :], other={})",
        ),
        "mask": ("m_block = tl.load(mask_ptrs, mask_k, ", "m_block = tl.load(mask_ptrs, mask_k, other={}, "),
        "k": ("k_block = tl.load(k_ptrs, mask_k[:, None])", "k_block = tl.load(k_ptrs, mask_k[:, None], other={})"),
        "v": ("vt_block = tl.load(vt_ptrs, mask_k[None, :])", "vt_block = tl.load(vt_ptrs, mask_k[None, :], other={})"),
    },
}
CG = ', cache_modifier=".cg")'
LOADS["b"] = {
    role: (line + CG, line + ", other={}" + CG)
    for role, line in {
        "q": "q_block = tl.load(q_ptrs, mask_j[:, None]",
        "k": "k_block = tl.load(k_ptrs, mask_k[:, None]",
        "b": "b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]",
        "mask": "m_block = tl.load(mask_ptrs, mask_k",
        "lse": "sm_denom = tl.load(l_ptrs, mask=mask_j",
        "do": "do = tl.load(do_ptrs, mask_j[:, None]",
        "delta": "delta = tl.load(d_ptrs, mask_j",
        "v": "v_block = tl.load(v_ptrs, mask_k[:, None]",
    }.items()
}
# The loads whose dropped elements are terms of a stored sum. Every dbias cell is its own sum
# over the instances, of elements below N only.
SUMMED = {"kv": ("q", "do", "lse", "delta"), "q": ("k", "v")}


def _spell(value):
    if isinstance(value, str):
        return value
    if value != value:
        return "float('nan')"
    if value in (INF, -INF):
        return "float('inf')" if value > 0 else "float('-inf')"
    return repr(value)


def _kernel(kind, others, *edits):
    """The kernel with ``other`` on every load: zero where ``others`` names no value."""
    fn = triton.JITFunction(KERNELS[kind].fn)
    src = fn.src
    for role, (old, new) in LOADS[kind].items():
        assert src.count(old) == 1, old
        zero = 0 if role == "mask" else 0.0
        src = src.replace(old, new.format(_spell(others.get(role, zero))))
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    fn._unsafe_update_src(src)
    return fn


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    return tmp_path_factory.mktemp("backward_others")


# The maker call of each kernel this module compiled. A later launch of the same kernel comes
# from the module's cache and calls no maker.
ROUTES = {}


@pytest.fixture
def makers(monkeypatch, cache_root):
    """The backward makers called, with the ``others`` each was given. The caches start empty
    with the module, so the first launch of every kernel here compiles it."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(cache_root / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root / "triton"))
    calls = []
    for name in ("kv_kernel", "kv_kernel_simd", "q_kernel", "q_kernel_simd", "b_kernel"):
        real = getattr(templates, "make_flash_attention_bwd_" + name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append((_name, dict(kwargs.get("others") or {})))
            return _real(*args, **kwargs)

        monkeypatch.setattr(templates, "make_flash_attention_bwd_" + name, spy)
    return calls


def _inputs(kind, n, dim, dtype, seed, masked_keys=True):
    gen = torch.Generator().manual_seed(seed)
    phys = n + 16
    sm = 1.0 / math.sqrt(dim)
    t = {}
    for name in ("q", "k", "v", "do", "o"):
        t[name] = torch.full((1, 1, phys, dim), BEYOND)
        t[name][0, 0, :n] = torch.randn(n, dim, generator=gen) * 0.5
    t["b"] = torch.full((1, phys, phys), BEYOND)
    t["b"][0, :n, :n] = torch.randn(n, n, generator=gen) * 0.5
    t["mask"] = torch.ones(1, 1, phys, dtype=torch.uint8)
    t["mask"][0, 0, :n] = (torch.rand(n, generator=gen) < 0.15) if masked_keys else 0
    t = {name: x.to(dtype) if x.is_floating_point() else x for name, x in t.items()}
    # lse as the forward stores it for these scores (fp32 on every route).
    raw = t["q"][0, 0, :n].double() @ (t["k"][0, 0, :n].double() * sm).T + t["b"][0, :n, :n].double()
    raw = raw.masked_fill(t["mask"][0, 0, :n].bool()[None, :], -INF)
    t["lse"] = torch.full((1, 1, phys), BEYOND)
    t["lse"][0, 0, :n] = torch.logsumexp(raw, -1).float()
    t["delta"] = torch.full((1, 1, phys), BEYOND)
    t["delta"][0, 0, :n] = torch.randn(n, generator=gen) * 0.5
    t["delta"] = t["delta"].to(dtype)
    for name in ("dq", "dk", "dv"):
        t[name] = torch.full((1, 1, phys, dim), POISON, dtype=dtype)
    if kind == "q":
        t["delta"] = torch.full((1, 1, phys), POISON, dtype=dtype)  # an output of the dQ kernel
    return t


def _launch(kind, fn, t, n, dim, sentinel, tile=(BLOCK, BLOCK)):
    """One program per tile of keys (dK / dV) or of query rows (dQ); ``tile`` is the query
    rows and the keys of one."""
    dev = {name: x.to("mps") for name, x in t.items()}
    args = []
    for name in ORDER[kind]:
        args += [dev[name], *dev[name].stride()]
    block_j, block_k = tile
    programs = triton.cdiv(n, block_k if kind == "kv" else block_j)
    fn[(programs, 1, 1)](*args, 1.0 / math.sqrt(dim), sentinel, n, 1, dim, n, block_j, block_k)
    torch.mps.synchronize()
    return [dev[name].cpu() for name in OUTPUTS[kind]]


def _mm(a, b):
    """A product that multiplies every pair, so 0 * inf is the NaN it is in the kernel."""
    return (a[:, :, None] * b[None, :, :]).sum(1)


def _tile(x, start, n, other, block=BLOCK):
    """One tile of ``block`` rows of ``x`` from ``start`` as a masked load reads it."""
    idx = torch.arange(block) + start
    keep = idx < n
    rows = x[idx.clamp(max=n - 1)].double()
    keep_ = keep.reshape(-1, *([1] * (rows.dim() - 1)))
    return torch.where(keep_, rows, torch.tensor(float(other), dtype=torch.float64)), keep


def _row_sums(x, dim, dtype):
    """The sum along each row of ``x`` in the tensors' type. A 16-bit sum of 32 columns rounds
    after every addition, in the runtime's order: a binary tree over each eight columns, then
    the four partial sums two apart, then the last two (tests/test_fa_bwd_narrow_delta.py)."""
    r = lambda v: v.to(dtype).double()
    if dtype == torch.float32:
        return r(x.sum(1))
    assert dim == 32, "the 16-bit dQ kernel is routed at head dimension 32 only"
    c = x.reshape(-1, 4, 8)
    for _ in range(3):
        c = r(c[..., 0::2] + c[..., 1::2])
    c = c[..., 0]
    return r(r(c[:, 0] + c[:, 2]) + r(c[:, 1] + c[:, 3]))


def _source(kind, t, others, sentinel, n, dim, dtype, tile=(BLOCK, BLOCK)):
    """The source on one head and instance, tile by tile, rounding where it rounds."""
    bj, bk = tile
    r = lambda x: x.to(dtype).double()  # a value of the tensors' type
    f = lambda x: x.float().double()  # an fp32 value
    o = lambda role: others.get(role, 0.0)
    sm = f(torch.tensor(1.0 / math.sqrt(dim), dtype=torch.float64))
    snt = f(torch.tensor(float(sentinel), dtype=torch.float64))
    q, k, v, do, lse = (t[name][0, 0] for name in ("q", "k", "v", "do", "lse"))
    out = {name: torch.full(t[name].shape, POISON, dtype=torch.float64) for name in OUTPUTS[kind]}

    def probabilities(qb, kb, j0, k0, keep_j, keep_k):
        jj = (torch.arange(bj) + j0).clamp(max=n - 1)
        kk = (torch.arange(bk) + k0).clamp(max=n - 1)
        inside = keep_j[:, None] & keep_k[None, :]
        bias = torch.where(inside, t["b"][0][jj][:, kk].double(), torch.tensor(float(o("b")), dtype=torch.float64))
        masked = torch.where(keep_k, t["mask"][0, 0][kk].double(), torch.tensor(float(o("mask")))) != 0
        s = f(_mm(qb, kb.T) + bias)
        s = torch.where(inside, s, snt)
        s = torch.where(masked[None, :], snt, s)
        row_lse, _ = _tile(lse, j0, n, o("lse"), bj)
        return f(torch.exp2(f(f(s - row_lse[:, None]) * INV_LN2)))

    if kind == "kv":
        for k0 in range(0, n, bk):
            vb, keep_k = _tile(v, k0, n, o("v"), bk)
            kb, _ = _tile(k, k0, n, o("k"), bk)
            kb = r(kb * r(sm))
            dk = torch.zeros(bk, dim, dtype=torch.float64)
            dv = torch.zeros(bk, dim, dtype=torch.float64)
            for j0 in range(0, n, bj):
                qb, keep_j = _tile(q, j0, n, o("q"), bj)
                p = probabilities(qb, kb, j0, k0, keep_j, keep_k)
                dob, _ = _tile(do, j0, n, o("do"), bj)
                dv = dv + _mm(r(p).T, dob)
                delta, _ = _tile(t["delta"][0, 0], j0, n, o("delta"), bj)
                ds = r(f(p * (f(_mm(dob, vb.T)) - delta[:, None])))
                dk = dk + _mm(ds.T, qb)
            rows = int(keep_k.sum())
            out["dk"][0, 0, k0 : k0 + rows] = r(f(f(dk) * sm))[:rows]
            out["dv"][0, 0, k0 : k0 + rows] = r(f(dv))[:rows]
    else:
        for j0 in range(0, n, bj):
            qb, keep_j = _tile(q, j0, n, o("q"), bj)
            dob, _ = _tile(do, j0, n, o("do"), bj)
            ob, _ = _tile(t["o"][0, 0], j0, n, o("o"), bj)
            delta = _row_sums(r(ob * dob), dim, dtype)
            dq = torch.zeros(bj, dim, dtype=torch.float64)
            for k0 in range(0, n, bk):
                kb, keep_k = _tile(k, k0, n, o("k"), bk)
                kb = r(kb * r(sm))
                p = probabilities(qb, kb, j0, k0, keep_j, keep_k)
                vb, _ = _tile(v, k0, n, o("v"), bk)
                ds = r(f(p * (f(_mm(dob, vb.T)) - delta[:, None])))
                dq = dq + _mm(ds, kb)
            rows = int(keep_j.sum())
            out["dq"][0, 0, j0 : j0 + rows] = r(f(dq))[:rows]
            out["delta"][0, 0, j0 : j0 + rows] = delta[:rows]
    return [out[name] for name in OUTPUTS[kind]]


def _same(outputs, expected, tol):
    for got, want in zip(outputs, expected):
        got = got.double()
        assert torch.equal(got.isnan(), want.isnan()), "NaN where the source has none, or none where it has one"
        infinite = want.isinf()
        assert torch.equal(got.isinf(), infinite) and torch.equal(got[infinite], want[infinite])
        finite = want.isfinite()
        written = want[finite & (want != POISON)]
        scale = max(1.0, float(written.abs().max())) if written.numel() else 1.0
        torch.testing.assert_close(got[finite], want[finite], atol=tol * scale, rtol=tol)


def _run(
    kind, others, sentinel, n, dim, dtype, maker, makers, seed, masked_keys=True, edit_inputs=None, tile=(BLOCK, BLOCK)
):
    t = _inputs(kind, n, dim, dtype, seed, masked_keys)
    if edit_inputs:
        edit_inputs(t)
    outputs = _launch(kind, _kernel(kind, others), t, n, dim, sentinel, tile)
    summed = {role: others[role] for role in SUMMED[kind] if others.get(role, 0.0) != 0.0}
    key = (kind, dim, dtype, tile, tuple(sorted((role, _spell(value)) for role, value in others.items())))
    if makers or key not in ROUTES:
        assert [name for name, _ in makers] == [maker]
        ROUTES[key] = set(makers[0][1])
    assert ROUTES[key] == set(summed)
    _same(outputs, _source(kind, t, others, sentinel, n, dim, dtype, tile), TOLERANCE[dtype])
    return outputs, t


TYPE_ID = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
TOLERANCE = {torch.float32: 3e-4, torch.float16: 1e-2, torch.bfloat16: 3e-2}
SHAPES = {
    "kv": [
        (32, torch.float32, "kv_kernel_simd"),
        (32, torch.float16, "kv_kernel_simd"),
        (32, torch.bfloat16, "kv_kernel_simd"),
        (64, torch.float32, "kv_kernel"),
        (64, torch.float16, "kv_kernel"),
        (64, torch.bfloat16, "kv_kernel"),
    ],
    # At head dimension 64 the 16-bit dQ kernels are refused, with or without an `other`.
    "q": [
        (32, torch.float32, "q_kernel_simd"),
        (32, torch.float16, "q_kernel_simd"),
        (32, torch.bfloat16, "q_kernel_simd"),
        (64, torch.float32, "q_kernel"),
    ],
}
SHAPE_IDS = {kind: [f"{d}-{TYPE_ID[t]}" for d, t, _ in rows] for kind, rows in SHAPES.items()}

# ---------------------------------------------------------------------------------------
# The sentinel past N, every `other` zero
# ---------------------------------------------------------------------------------------

SENTINELS = [-1e9, -INF, 0.0, 5.0, 80.0, 100.0, 1e9, INF, NAN]


@requires_mps
@pytest.mark.parametrize("sentinel", SENTINELS, ids=[_spell(s) for s in SENTINELS])
@pytest.mark.parametrize("n", [48, 64])
@pytest.mark.parametrize("dim,dtype,maker", SHAPES["kv"], ids=SHAPE_IDS["kv"])
def test_dk_dv_cells_past_n_hold_the_sentinel(dim, dtype, maker, n, sentinel, makers):
    _run("kv", {}, sentinel, n, dim, dtype, maker, makers, seed=2001)


@requires_mps
@pytest.mark.parametrize("sentinel", SENTINELS, ids=[_spell(s) for s in SENTINELS])
@pytest.mark.parametrize("n", [48, 64])
@pytest.mark.parametrize("dim,dtype,maker", SHAPES["q"], ids=SHAPE_IDS["q"])
def test_dq_cells_past_n_hold_the_sentinel(dim, dtype, maker, n, sentinel, makers):
    _run("q", {}, sentinel, n, dim, dtype, maker, makers, seed=2002)


@requires_mps
@pytest.mark.parametrize("n,nan_rows", [(48, True), (64, False)])
@pytest.mark.parametrize("dim,dtype,maker", SHAPES["q"], ids=SHAPE_IDS["q"])
def test_dq_nan_sentinel_reaches_a_row_only_through_a_key_past_n(dim, dtype, maker, n, nan_rows, makers):
    """No key is masked, so only a key past N holds the NaN sentinel: every row of a partial
    tile is NaN and no row of whole tiles is."""
    (dq, _delta), _t = _run("q", {}, NAN, n, dim, dtype, maker, makers, seed=2003, masked_keys=False)
    assert bool(dq[0, 0, :n].isnan().all()) is nan_rows
    assert bool(dq[0, 0, :n].isnan().any()) is nan_rows


# ---------------------------------------------------------------------------------------
# A load's `other`
# ---------------------------------------------------------------------------------------

# Values every type here holds exactly. The second set of each kernel is on loads whose
# dropped elements reach no stored value.
FINITE = {
    "kv": {"q": 0.5, "do": -0.25, "lse": 1.5, "delta": 0.75},
    "q": {"k": 0.5, "v": -0.25},
}
ELSEWHERE = {
    "kv": {"k": NAN, "v": NAN, "b": NAN, "mask": 1},
    "q": {"q": NAN, "do": NAN, "o": NAN, "lse": -INF, "b": NAN, "mask": 1},
}
ONE_OTHER = {
    "kv": {
        "q_nan": {"q": NAN},
        "q_one": {"q": 1.0},
        "q_inf": {"q": INF},
        "do_nan": {"do": NAN},
        "do_one": {"do": 1.0},
        "do_minus_inf": {"do": -INF},
        "lse_minus_inf": {"lse": -INF},
        "lse_inf": {"lse": INF},
        "lse_nan": {"lse": NAN},
        "lse_two": {"lse": 2.0},
        "delta_nan": {"delta": NAN},
        "delta_one": {"delta": 1.0},
        "delta_inf": {"delta": INF},
    },
    "q": {
        "k_nan": {"k": NAN},
        "k_one": {"k": 1.0},
        "k_minus_one": {"k": -1.0},
        "k_inf": {"k": INF},
        "v_nan": {"v": NAN},
        "v_one": {"v": 1.0},
        "v_minus_inf": {"v": -INF},
    },
}
OTHER_SENTINELS = [-1e9, 0.0, 5.0, 80.0]


def _others_cases(kind):
    cases = [(name, others) for name, others in ONE_OTHER[kind].items()]
    cases.append(("all_finite", FINITE[kind]))
    cases.append(("all_finite_and_elsewhere", {**FINITE[kind], **ELSEWHERE[kind]}))
    cases.append(("elsewhere", ELSEWHERE[kind]))
    return cases


@requires_mps
@pytest.mark.parametrize("sentinel", OTHER_SENTINELS, ids=[_spell(s) for s in OTHER_SENTINELS])
@pytest.mark.parametrize("name,others", _others_cases("kv"), ids=[c[0] for c in _others_cases("kv")])
@pytest.mark.parametrize("n", [48, 64])
@pytest.mark.parametrize("dim,dtype,maker", SHAPES["kv"], ids=SHAPE_IDS["kv"])
def test_dk_dv_rows_past_n_hold_each_load_s_other(dim, dtype, maker, n, name, others, sentinel, makers):
    _run("kv", others, sentinel, n, dim, dtype, maker, makers, seed=2004)


@requires_mps
@pytest.mark.parametrize("sentinel", OTHER_SENTINELS, ids=[_spell(s) for s in OTHER_SENTINELS])
@pytest.mark.parametrize("name,others", _others_cases("q"), ids=[c[0] for c in _others_cases("q")])
@pytest.mark.parametrize("n", [48, 64])
@pytest.mark.parametrize("dim,dtype,maker", SHAPES["q"], ids=SHAPE_IDS["q"])
def test_dq_keys_past_n_hold_each_load_s_other(dim, dtype, maker, n, name, others, sentinel, makers):
    _run("q", others, sentinel, n, dim, dtype, maker, makers, seed=2005)


# Query rows and keys of one tile. The kernels run on the wide templates when the head
# dimension and both sides are at most 32.
TILES = [(16, 16), (8, 16), (32, 16), (16, 32), (64, 32), (32, 64), (64, 64)]


def _tile_cases(kind):
    """Every shape at every tile its route takes. The 16-bit dQ kernel is routed with 32 query
    rows and at most 32 keys to a tile, and the dQ kernel at head dimension 64 with at most 32
    keys; the others are refused."""
    cases = []
    for dim, dtype, maker in SHAPES[kind]:
        for bj, bk in TILES:
            if kind == "q" and dtype != torch.float32 and (bj != 32 or bk > 32):
                continue
            if kind == "q" and dim == 64 and bk > 32:
                continue
            wide = max(dim, bj, bk) <= 32
            name = maker if wide else maker.removesuffix("_simd")
            cases.append(pytest.param(dim, dtype, name, (bj, bk), id=f"{dim}-{TYPE_ID[dtype]}-{bj}x{bk}"))
    return cases


@requires_mps
@pytest.mark.parametrize("sentinel", [-1e9, 5.0], ids=["-1e9", "5.0"])
@pytest.mark.parametrize("others", ["none", "all_finite"])
@pytest.mark.parametrize("n", [44, 100])
@pytest.mark.parametrize("dim,dtype,maker,tile", _tile_cases("kv"))
def test_dk_dv_past_n_at_other_tile_sizes(dim, dtype, maker, tile, n, others, sentinel, makers):
    """44 and 100 are a whole number of no tile here, along either side."""
    chosen = FINITE["kv"] if others == "all_finite" else {}
    _run("kv", chosen, sentinel, n, dim, dtype, maker, makers, seed=2010, tile=tile)


@requires_mps
@pytest.mark.parametrize("sentinel", [-1e9, 5.0], ids=["-1e9", "5.0"])
@pytest.mark.parametrize("others", ["none", "all_finite"])
@pytest.mark.parametrize("n", [44, 100])
@pytest.mark.parametrize("dim,dtype,maker,tile", _tile_cases("q"))
def test_dq_past_n_at_other_tile_sizes(dim, dtype, maker, tile, n, others, sentinel, makers):
    chosen = FINITE["q"] if others == "all_finite" else {}
    _run("q", chosen, sentinel, n, dim, dtype, maker, makers, seed=2011, tile=tile)


@requires_mps
@pytest.mark.parametrize("kind", ["kv", "q"])
@pytest.mark.parametrize("sentinel", [-1e9, 0.0], ids=["-1e9", "0.0"])
def test_an_other_that_reaches_no_stored_value_changes_nothing(kind, sentinel, makers):
    """The same bits as with zero there, and no value handed to the maker."""
    dim, dtype, maker = SHAPES[kind][0]
    plain, _t = _run(kind, {}, sentinel, 48, dim, dtype, maker, makers, seed=2006)
    makers.clear()
    elsewhere, _t = _run(kind, ELSEWHERE[kind], sentinel, 48, dim, dtype, maker, makers, seed=2006)
    for a, b in zip(plain, elsewhere):
        assert torch.equal(a, b)


@requires_mps
@pytest.mark.parametrize("kind", ["kv", "q"])
def test_the_usual_sentinels_and_no_other_keep_their_values(kind, makers):
    """With -1e9 or -inf and finite tensors every cell past N counts exactly zero, so a
    partial tile's sums are the sums over the rows and keys below N."""
    dim, dtype, maker = SHAPES[kind][0]
    a, t = _run(kind, {}, -1e9, 48, dim, dtype, maker, makers, seed=2007)
    makers.clear()
    b, _t = _run(kind, {}, -INF, 48, dim, dtype, maker, makers, seed=2007)
    n, sm = 48, 1.0 / math.sqrt(dim)
    q, k, v, do = (t[name][0, 0, :n].double() for name in ("q", "k", "v", "do"))
    s = q @ (k * sm).T + t["b"][0, :n, :n].double()
    p = torch.exp(s - t["lse"][0, 0, :n].double()[:, None]).masked_fill(t["mask"][0, 0, :n].bool()[None, :], 0.0)
    if kind == "kv":
        ds = p * (do @ v.T - t["delta"][0, 0, :n].double()[:, None])
        closed = [sm * ds.T @ q, p.T @ do]
    else:
        delta = (t["o"][0, 0, :n].double() * do).sum(-1)
        closed = [(p * (do @ v.T - delta[:, None])) @ (k * sm), delta]
    for x, y, want in zip(a, b, closed):
        assert torch.equal(x, y)
        got = x[0, 0, :n].double()
        torch.testing.assert_close(got, want, atol=3e-4 * max(1.0, float(want.abs().max())), rtol=3e-4)


@requires_mps
@pytest.mark.parametrize("value", [INF, NAN], ids=["inf", "nan"])
@pytest.mark.parametrize("dim,dtype,maker", SHAPES["kv"], ids=SHAPE_IDS["kv"])
def test_dk_of_a_key_with_a_value_that_is_not_finite_next_to_rows_past_n(dim, dtype, maker, value, makers):
    """dP of a row past N is its dO, zero, times V: NaN where V is not finite. With the usual
    sentinel P is zero there and 0 * NaN is the key's whole dK row, as in the source."""

    def edit(t):
        t["v"][0, 0, 5, 3] = value

    (dk, dv), _t = _run("kv", {}, -1e9, 48, dim, dtype, maker, makers, seed=2008, edit_inputs=edit)
    assert bool(dk[0, 0, 5].isnan().all())
    assert not bool(dk[0, 0, :5].isnan().any()) and not bool(dk[0, 0, 6:48].isnan().any())
    assert not bool(dv[0, 0, :48].isnan().any())


# ---------------------------------------------------------------------------------------
# Refused: an `other` the templates cannot hold
# ---------------------------------------------------------------------------------------

CONSTANT = "every masked load's `other` being absent or a constant"
REFUSED = {
    # A value of the launch is not a constant of the kernel.
    "kv_q_other_is_an_argument": ("kv", torch.float32, {"q": "sm_scale"}, (), CONSTANT),
    "kv_lse_other_is_an_argument": ("kv", torch.float32, {"lse": "neg_inf"}, (), CONSTANT),
    "q_k_other_is_an_argument": ("q", torch.float32, {"k": "sm_scale"}, (), CONSTANT),
    "q_o_other_is_an_argument": ("q", torch.float32, {"o": "sm_scale"}, (), CONSTANT),
    # Rounded to float16 inside the kernel: the constant the load holds is not in the program text.
    "kv_q_other_rounded_in_the_kernel": ("kv", torch.float16, {"q": 0.1}, (), CONSTANT),
    "q_v_other_rounded_in_the_kernel": ("q", torch.float16, {"v": -1e9}, (), CONSTANT),
    # Two loads of one tensor that disagree past N. The cache modifier keeps the second one a
    # load of its own when both supply one value (the compiler merges two that are the same).
    "kv_two_do_loads": (
        "kv",
        torch.float32,
        {"do": 1.0},
        (
            (
                'dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee")',
                'do2 = tl.load(do_ptrs, mask_j[:, None], other=2.0, cache_modifier=".cg")\n'
                '        dsm_value = tl.dot(do2, vt_block, dsm_value, input_precision="ieee")',
            ),
        ),
        "every do load supplying the same `other`",
    ),
    # A load the proof gives no role. (Without this requirement the kernel is refused one
    # step later, for the dS it computes.)
    "kv_other_on_one_more_load": (
        "kv",
        torch.float32,
        {},
        (
            (
                "dscores = sm_value * (dsm_value - delta[:, None])",
                "extra = tl.load(d_ptrs + 1, mask_j, other=3.0)\n"
                "        dscores = sm_value * (dsm_value - delta[:, None] - 0 * extra[:, None])",
            ),
        ),
        "a constant `other` only on a load the proof recognises",
    ),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(REFUSED))
def test_an_other_the_templates_cannot_hold_refuses(name, makers):
    kind, dtype, others, edits, fragment = REFUSED[name]
    t = _inputs(kind, 48, 32, dtype, seed=2009)
    with pytest.raises(MetalNonRecoverableError) as info:
        _launch(kind, _kernel(kind, others, *edits), t, 48, 32, -1e9)
    assert fragment in str(info.value)
    assert makers == []


@requires_mps
@pytest.mark.parametrize("value,spelled", [(1.0, "(1.0f)"), (NAN, "NAN")], ids=["one", "nan"])
def test_two_loads_of_one_tensor_with_one_other_compute(value, spelled, makers):
    """The control of the refusal above: the second dO load with the first one's value."""
    old, new = REFUSED["kv_two_do_loads"][3][0]
    t = _inputs("kv", 48, 32, torch.float32, seed=2010)
    fn = _kernel("kv", {"do": value}, (old, new.replace("other=2.0", "other=" + _spell(value))))
    outputs = _launch("kv", fn, t, 48, 32, 5.0)
    assert makers == [("kv_kernel_simd", {"do": spelled})]
    _same(outputs, _source("kv", t, {"do": value}, 5.0, 48, 32, torch.float32), 3e-4)


# The value of each type next to one tenth. MLIR prints a constant with the digits that tell
# it from its neighbours in its own type, which are fewer than the value has.
NEXT_TO_A_TENTH = {torch.float32: 0.10000000149011612, torch.float16: 0.0999755859375, torch.bfloat16: 0.10009765625}


@requires_mps
@pytest.mark.parametrize(
    "kind,role", [("kv", "q"), ("kv", "do"), ("kv", "delta"), ("kv", "lse"), ("q", "k"), ("q", "v")]
)
@pytest.mark.parametrize("dtype", list(TYPE_ID), ids=list(TYPE_ID.values()))
def test_the_template_holds_the_constant_of_the_load_s_own_type(dtype, kind, role, makers):
    value = NEXT_TO_A_TENTH[torch.float32 if role == "lse" else dtype]  # lse is fp32 beside any tensor type
    t = _inputs(kind, 48, 32, dtype, seed=2013)
    outputs = _launch(kind, _kernel(kind, {role: value}), t, 48, 32, 5.0)
    assert makers == [(kind + "_kernel_simd", {role: f"({value!r}f)"})]
    _same(outputs, _source(kind, t, {role: value}, 5.0, 48, 32, dtype), TOLERANCE[dtype])


@requires_mps
def test_an_other_whose_value_is_not_read_refuses(makers, monkeypatch):
    """A constant is staged as the value it was read as, and as nothing else: with the reading
    of a float16 decimal failing, the kernel is refused."""

    def unread(value, elem_type):
        raise OverflowError

    monkeypatch.setattr(lowering, "narrow_constant", unread)
    t = _inputs("kv", 48, 32, torch.float16, seed=2016)
    with pytest.raises(MetalNonRecoverableError) as info:
        _launch("kv", _kernel("kv", {"q": 0.4375}), t, 48, 32, -1e9)
    assert "the q load's `other` being a decodable fp16 / bf16 / fp32 constant" in str(info.value)
    assert makers == []


# A value of each 16-bit type for which both roundings of the source's `K * sm_scale` matter:
# its product with the rounded scale is not a value of the type, and the product with the
# scale as passed rounds to another one.
K_ROUNDED_TWICE = {torch.float16: 1.005859375, torch.bfloat16: 1.2734375}


@requires_mps
@pytest.mark.parametrize("sentinel", [0.0, 5.0], ids=["0.0", "5.0"])
@pytest.mark.parametrize("dtype", list(K_ROUNDED_TWICE), ids=[TYPE_ID[d] for d in K_ROUNDED_TWICE])
def test_dq_key_past_n_is_its_other_times_the_scale_rounded_where_the_source_rounds(dtype, sentinel, makers):
    """Sixteen keys past N weigh on every row, so a key staged one unit away moves hundreds of
    the 1536 cells (599 to 1432 measured, with either rounding left out); staged as the source
    rounds it, at most 4 differ from the source computed on the host."""
    value = K_ROUNDED_TWICE[dtype]
    r = lambda x: x.to(dtype).double()
    f = lambda x: x.float().double()
    sm = f(torch.tensor(1.0 / math.sqrt(32), dtype=torch.float64))
    other = torch.tensor(value, dtype=torch.float64)
    staged = r(f(other * r(sm)))
    assert r(other) == other
    assert staged != f(other * r(sm)), "the product is a value of the type: its rounding is not tested"
    assert staged != r(f(other * sm)), "the scale's rounding is not tested"
    (dq, _delta), t = _run("q", {"k": value}, sentinel, 48, 32, dtype, "q_kernel_simd", makers, seed=2014)
    want, _ = _source("q", t, {"k": value}, sentinel, 48, 32, dtype)
    differ = int((dq.double()[0, 0, :48] != want[0, 0, :48]).sum())
    assert differ <= 32, f"{differ} of 1536 cells differ from the source"


def _bits(x):
    return x.contiguous().view(torch.int32)


@requires_mps
def test_the_bfloat16_rounding_of_a_staged_key_is_to_the_nearest_even():
    """The staged expression itself, on products the kernels' own scales do not reach: every
    tie and its two neighbours for some 4000 values of the type, products of two factors, the
    largest float32 (which rounds to infinity), the infinities, zeros and NaNs."""
    lib = torch.mps.compile_shader(f"""#include <metal_stdlib>
using namespace metal;
kernel void staged(device float* out, device const float* a, device const float* b, uint i [[thread_position_in_grid]]) {{
    out[i] = {templates._bwd_k_other("a[i]", "bfloat", "b[i]")};
}}
kernel void of_one(device float* out, device const float* b, uint i [[thread_position_in_grid]]) {{
    out[i] = {templates._bwd_k_other("(1.0f)", "bfloat", "b[i]")};
}}
""")
    g = torch.Generator().manual_seed(2015)
    high = torch.randint(0, 1 << 16, (4096,), generator=g, dtype=torch.int64)
    high = high[((high >> 7) & 0xFF != 0) & ((high >> 7) & 0xFF != 0xFF)]  # finite and not subnormal
    low = torch.tensor([0x7FFF, 0x8000, 0x8001, 0x0001, 0xFFFF], dtype=torch.int64)
    near = ((high[:, None] << 16) | low[None, :]).reshape(-1)
    special = torch.tensor(
        [0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0, 0x80000000, 0x7FC00000, 0x7FFFFFFF, 0xFFFFFFFF]
    )
    one_factor = torch.cat([near, special])
    one_factor = (one_factor - ((one_factor >> 31) << 32)).to(torch.int32).view(torch.float32)
    factors = torch.rand(2, 8192, generator=g) * 1.5 + 0.5
    a = torch.cat([one_factor, factors[0]])
    b = torch.cat([torch.ones_like(one_factor), factors[1]])
    out = torch.full_like(a, POISON).to("mps")
    lib.staged(out, a.to("mps"), b.to("mps"))
    got = out.cpu()
    product = a * b
    want = product.to(torch.bfloat16).float()
    ties = (_bits(product) & 0xFFFF) == 0x8000
    assert int(ties.sum()) >= 4000 and int((want != product).sum()) >= 20000
    assert torch.equal(got.isnan(), want.isnan()) and int(want.isnan().sum()) == 3
    assert torch.equal(_bits(got)[~want.isnan()], _bits(want)[~want.isnan()])
    # An `other` of one: Metal compiles the product to the other factor itself, and a NaN
    # that was only loaded keeps its bits, which the rounding would carry into the sign.
    out = torch.full_like(one_factor, POISON).to("mps")
    lib.of_one(out, one_factor.to("mps"))
    got = out.cpu()
    want = one_factor.to(torch.bfloat16).float()
    assert torch.equal(got.isnan(), want.isnan()) and int(want.isnan().sum()) == 3
    assert torch.equal(_bits(got)[~want.isnan()], _bits(want)[~want.isnan()])


# ---------------------------------------------------------------------------------------
# dbias: no stored cell holds a dropped element
# ---------------------------------------------------------------------------------------

DBIAS_OTHERS = {"q": NAN, "k": NAN, "b": NAN, "mask": 1, "lse": NAN, "do": NAN, "delta": NAN, "v": NAN}


def _dbias_inputs(dtype, seed):
    t = witness._inputs("b", seed)
    return {name: x.to(dtype) if x.is_floating_point() and name != "lse" else x for name, x in t.items()}


@requires_mps
@pytest.mark.parametrize("dtype", list(TYPE_ID), ids=list(TYPE_ID.values()))
def test_dbias_is_the_same_whatever_its_loads_supply(dtype, makers):
    t = _dbias_inputs(dtype, seed=2011)
    (plain,) = witness._launch("b", _kernel("b", {}), t)
    (supplied,) = witness._launch("b", _kernel("b", DBIAS_OTHERS), t)
    assert makers == [("b_kernel", {}), ("b_kernel", {})]
    assert torch.equal(plain.cpu(), supplied.cpu())
    (want,) = witness._expected("b", {}, t)
    got = supplied.cpu().double()
    written = want != POISON
    assert bool((got[~written] == POISON).all())
    tol = 4e-3 if dtype == torch.bfloat16 else 2e-3
    scale = max(1.0, float(want[written].abs().max()))
    torch.testing.assert_close(got[written], want[written], atol=tol * scale, rtol=tol)


@requires_mps
def test_dbias_other_that_is_not_a_constant_refuses(makers):
    with pytest.raises(MetalNonRecoverableError) as info:
        witness._launch("b", _kernel("b", {"q": "sm_scale"}), _dbias_inputs(torch.float32, seed=2012))
    assert CONSTANT in str(info.value)
    assert makers == []
