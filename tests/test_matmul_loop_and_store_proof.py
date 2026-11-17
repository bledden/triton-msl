"""A matmul template runs only the reduction and the store clip the source itself spells.

The K-loop templates reduce elements [0, extent) in BLOCK_K blocks and clip the store once per
axis at ``row < ARG``. At 34378c9 the extent was whatever the loop's upper bound traced to, else
an argument merely NAMED ``K``, whatever the loop said: a later start, ``K - BK``, ``K // 2``, a
floor count, a count argument and a constant loop beside a stride called ``K`` all ran as
``range(0, K, BK)``; a non-looped dot reduced over ``K`` instead of its own tile; and a second
clip on an axis, ``<=`` and a flat-grid constant clip were dropped (GPU: silent, both matmul
routes, every device case below). The source-replaying epilogue route executes the loop and the
mask as written and keeps them. Every device case runs at the default and with TRITON_MSL_FAST_MATMUL=0.
"""

import pytest
import torch
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tests.test_matmul_source_extents import _kernel
from tests.test_matmul_subtiled_twin import routes  # noqa: F401
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError

# {{name}} survives for tests.test_matmul_source_extents._kernel; every other slot is per variant.
SOURCE = """
@triton.jit
def {{name}}(a, b, c, {sig}, X, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    {pid}
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
{body}
    tl.store(c + rm[:, None] * {N} + rn[None, :], {value}, mask={smask})
"""
LOOP = """    acc = tl.zeros((BM, BN), tl.float32)
    for k in {loop}:
        av = tl.load(a + rm[:, None] * {K} + ({ko} + rk)[None, :]{amask})
        bv = tl.load(b + ({ko} + rk)[:, None] * {N} + rn[None, :]{bmask})
        acc += tl.dot(av, bv)"""
SINGLE = """    av = tl.load(a + rm[:, None] * {K} + rk[None, :])
    bv = tl.load(b + rk[:, None] * {N} + rn[None, :])
    acc = tl.dot(av, bv)"""
GRID = "pm = tl.program_id(0)\n    pn = tl.program_id(1)"
FLAT = "pid = tl.program_id(0)\n    npm = tl.cdiv({M}, BM)\n    pm = pid % npm\n    pn = pid // npm"
AMASK = ", mask=(rm[:, None] < {M}) & (({ko} + rk)[None, :] < {K}), other=0."
BMASK = ", mask=(({ko} + rk)[:, None] < {K}) & (rn[None, :] < {N}), other=0."
SMASK = "(rm[:, None] < {M}) & (rn[None, :] < {N})"
DEFAULT = dict(
    sig="{M}, {N}, {K}",
    pid=GRID,
    body=LOOP,
    loop="range(0, {K}, BK)",
    ko="k",
    amask=AMASK,
    bmask=BMASK,
    value="acc",
    smask=SMASK,
    M="M",
    N="N",
    K="K",
)
COUNTED = dict(ko="k * BK")
UNMASKED = dict(amask="", bmask="")
POISON = -98765.0


def _format(text, **slots):
    slots = {**DEFAULT, **slots}
    for _ in range(4):  # slots nest: body -> masks -> names
        text = text.format(name="{name}", **slots)
    return text


def _variant(tmp_path, tag, **slots):
    return _kernel(tmp_path, "loop_store_" + tag, _format(SOURCE, **slots))


def _lower(fn, block=32):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    constants = dict(BM=block, BN=block, BK=32)
    signature = {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(fn.arg_names)}
    signature.update({name: "constexpr" for name in constants})
    module = ASTSource(fn=fn, signature=signature, constexprs=constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    return lowerer, lowerer.lower()


OTHER_REDUCTIONS = {
    "later_start": (dict(loop="range(BK, {K}, BK)"), "does not start at literal 0"),
    "one_block_short": (dict(loop="range(0, {K} - BK, BK)"), "neither a runtime argument nor a constant"),
    "half": (dict(loop="range(0, {K} // 2, BK)"), "neither a runtime argument nor a constant"),
    "minimum": (dict(loop="range(0, tl.minimum({K}, X), BK)"), "neither a runtime argument nor a constant"),
    "floor_count": (dict(loop="range(0, {K} // BK)", **COUNTED), "floor count"),
    "count_argument": (dict(loop="range(0, X)", **COUNTED, **UNMASKED), "separate count argument"),
    "ceiling_of_another": (dict(loop="range(0, tl.cdiv(X, BK))", **COUNTED), "K bound is not the loop's K extent"),
    "another_extent": (dict(loop="range(0, X, BK)"), "K bound is not the loop's K extent"),
    "constant_beside_k_masks": (dict(loop="range(0, 64, BK)"), "K bound is not the loop's K extent"),
}


@pytest.mark.parametrize("tag", sorted(OTHER_REDUCTIONS))
def test_a_loop_that_is_not_the_template_reduction_refuses(tmp_path, tag):
    slots, reason = OTHER_REDUCTIONS[tag]
    # 34378c9 lowered every one of these as range(0, K, BK) over the argument named K.
    with pytest.raises(MetalNonRecoverableError, match=reason):
        _lower(_variant(tmp_path, tag, **slots))


@pytest.mark.parametrize(
    "tag,slots,line",
    [
        ("element_stepped", {}, "long _K = (long)_mmarg5;"),
        ("ceiling_count", dict(loop="range(0, tl.cdiv({K}, BK))", **COUNTED), "long _K = (long)_mmarg5;"),
        (
            "renamed_ceiling",
            dict(loop="range(0, tl.cdiv({K}, BK))", **COUNTED, M="R", N="S", K="T"),
            "long _K = (long)_mmarg5;",
        ),
        ("another_extent_unmasked", dict(loop="range(0, X, BK)", **UNMASKED), "long _K = (long)_mmarg6;"),
        ("constant", dict(loop="range(0, 64, BK)", **UNMASKED), "long _K = (long)(64);"),
        ("constant_count", dict(loop="range(0, 3)", **COUNTED, **UNMASKED), "long _K = (long)(96);"),
        ("single_tile", dict(body=SINGLE), "long _K = 32;"),
    ],
)
def test_the_template_extent_is_what_the_loop_bounds_say(tmp_path, tag, slots, line):
    _lowerer, msl = _lower(_variant(tmp_path, tag, **slots))
    # Argument 5 is K (or T), 6 is X. ``constant``, ``constant_count`` and ``single_tile`` read argument 5
    # at 34378c9, where it is only A's row stride.
    assert line in msl


@pytest.mark.parametrize("tag", ["constant", "constant_count", "another_extent_unmasked"])
def test_no_direct_twin_without_a_proved_extent_argument(tmp_path, tag):
    slots = {
        "constant": dict(loop="range(0, 64, BK)", **UNMASKED),
        "constant_count": dict(loop="range(0, 3)", **COUNTED, **UNMASKED),
        "another_extent_unmasked": dict(loop="range(0, X, BK)", **UNMASKED),
    }[tag]
    lowerer, msl = _lower(_variant(tmp_path, tag, **slots))
    # The launcher hands the twin one runtime argument as the depth; a constant has none (34378c9: K).
    assert ("__mmdirect" in msl) is (tag == "another_extent_unmasked")
    if lowerer._fast_matmul is not None:
        assert lowerer.graph.args[lowerer._fast_matmul[3]].name == "X"


STORE_CLIPS = {
    "second_row_clip": "(rm[:, None] < {M}) & (rm[:, None] < X) & (rn[None, :] < {N})",
    "second_col_clip": "(rm[:, None] < {M}) & (rn[None, :] < {N}) & (rn[None, :] < X)",
    "non_strict_rows": "(rm[:, None] <= {M}) & (rn[None, :] < {N})",
    "non_strict_cols": "(rm[:, None] < {M}) & (rn[None, :] <= {N})",
}


@pytest.mark.parametrize("tag", sorted(STORE_CLIPS))
def test_a_store_clip_the_template_cannot_spell_refuses(tmp_path, tag):
    # 34378c9 kept one ``<`` clip per axis: min(M, X) became one of the two, ``<= M`` became ``< M``.
    with pytest.raises(MetalNonRecoverableError, match="non-tile-boundary output store mask"):
        _lower(_variant(tmp_path, tag, smask=STORE_CLIPS[tag]))


@pytest.mark.parametrize(
    "smask", ["(rm[:, None] < 48) & (rn[None, :] < {N})", "(rm[:, None] <= 47) & (rn[None, :] < {N})"]
)
def test_a_flat_grid_constant_store_clip_refuses(tmp_path, smask):
    # The flat grid reaches the program id through ``pid % n``; 34378c9 judged 48 against one tile.
    with pytest.raises(MetalNonRecoverableError, match="non-tile-boundary output store mask"):
        _lower(_variant(tmp_path, "flat_constant", pid=FLAT, smask=smask))


OTHER_INDEX_CLIPS = {
    "shifted_rows": "((rm + 1)[:, None] < {M}) & (rn[None, :] < {N})",
    "lowered_rows": "((rm - 16)[:, None] < {M}) & (rn[None, :] < {N})",
    "scaled_rows": "((rm * 2)[:, None] < {M}) & (rn[None, :] < {N})",
    "shifted_cols": "(rm[:, None] < {M}) & ((rn + 1)[None, :] < {N})",
    "columns_as_rows": "(rn[:, None] < {M}) & (rn[None, :] < {N})",
    "rows_as_columns": "(rm[:, None] < {M}) & (rm[None, :] < {N})",
    "shifted_constant": "((rk + 16)[:, None] < 32) & (rn[None, :] < {N})",
}


@pytest.mark.parametrize("body", ["loop", "single"])
@pytest.mark.parametrize("tag", sorted(OTHER_INDEX_CLIPS))
def test_a_store_clip_on_another_index_than_the_address_refuses(tmp_path, tag, body):
    # The address is ``rm`` by ``rn``. 34378c9 and 8f2c58f followed add, subtract and multiply to any
    # arange on the axis, so each of these ran as ``(rm < M) & (rn < N)`` (GPU: 64 to 2048 cells wrong).
    slots = dict(smask=OTHER_INDEX_CLIPS[tag], **({"body": SINGLE} if body == "single" else {}))
    with pytest.raises(MetalNonRecoverableError, match="non-tile-boundary output store mask"):
        _lower(_variant(tmp_path, f"{tag}_{body}", **slots))


@pytest.mark.parametrize(
    "tag,smask",
    [
        ("canonical", SMASK),
        ("bound_on_the_left", "({M} > rm[:, None]) & ({N} > rn[None, :])"),
        ("columns_first", "(rn[None, :] < {N}) & (rm[:, None] < {M})"),
        ("rows_only", "rm[:, None] < {M}"),
        ("compared_before_lifting", "(rm < {M})[:, None] & (rn < {N})[None, :]"),
    ],
    ids=["canonical", "bound_on_the_left", "columns_first", "rows_only", "compared_before_lifting"],
)
def test_a_store_clip_on_the_address_index_keeps_the_template(tmp_path, tag, smask):
    lowerer, msl = _lower(_variant(tmp_path, tag, smask=smask))
    assert not getattr(lowerer, "_store_mask_replay_only", False)
    assert "simdgroup_store" in msl and not msl.startswith("// Source-replayed K-loop epilogue")


ONE_TILE = "pm = 0\n    pn = 0"


@pytest.mark.parametrize("bound,kept", [("32", True), ("33", True), ("31", False), ("16", False)])
def test_a_constant_store_clip_is_judged_against_the_bare_tile(tmp_path, bound, kept):
    # One program at the origin: the address index IS arange(0, 32), so only a bound of at least 32 is whole.
    fn = _variant(tmp_path, "one_tile_" + bound, pid=ONE_TILE, smask=f"(rm[:, None] < {bound}) & (rn[None, :] < 32)")
    if kept:
        _lower(fn)
    else:
        with pytest.raises(MetalNonRecoverableError, match="non-tile-boundary output store mask"):
            _lower(fn)


def test_another_route_may_not_claim_a_replay_only_store_mask(tmp_path, monkeypatch):
    fn = _variant(tmp_path, "replay_only", value="acc * 2.0", smask=STORE_CLIPS["second_row_clip"])
    lowerer, msl = _lower(fn)
    assert lowerer._store_mask_replay_only and msl.startswith("// Source-replayed K-loop epilogue")
    monkeypatch.setattr(
        GenericLowerer, "_lower_routes", lambda self: setattr(self, "_store_mask_replay_only", True) or ""
    )
    with pytest.raises(MetalNonRecoverableError, match="another route claimed"):
        _lower(fn)


def _reference(a, b, shape, stride_k, programs, block, offsets, load_rows, load_k, load_cols, store, scale):
    """The source, program by program: masked loads read zero, the store keeps ``store(rm, rn)``."""
    rows, cols = shape
    out = torch.full((rows * cols,), POISON, dtype=torch.float64)
    rk = torch.arange(32)
    for pm, pn in programs:
        rm, rn = pm * block + torch.arange(block), pn * block + torch.arange(block)
        acc = torch.zeros((block, block), dtype=torch.float64)
        for offset in offsets:
            kk = offset + rk
            a_keep = torch.ones((block, 32), dtype=torch.bool)
            b_keep = torch.ones((32, block), dtype=torch.bool)
            if load_rows is not None:
                a_keep = (rm[:, None] < load_rows) & (kk[None, :] < load_k)
                b_keep = (kk[:, None] < load_k) & (rn[None, :] < load_cols)
            a_at, b_at = rm[:, None] * stride_k + kk[None, :], kk[:, None] * cols + rn[None, :]
            assert int(a_at[a_keep].max()) < a.numel() and int(b_at[b_keep].max()) < b.numel()
            av = torch.where(a_keep, a.double()[a_at.clamp(max=a.numel() - 1)], 0.0)
            bv = torch.where(b_keep, b.double()[b_at.clamp(max=b.numel() - 1)], 0.0)
            acc += av @ bv
        keep = store(rm[:, None], rn[None, :])
        at = (rm[:, None] * cols + rn[None, :])[keep]
        assert at.numel() == 0 or int(at.max()) < out.numel()
        out[at] = scale * acc[keep]
    return out.float()


CANONICAL_STORE = (64, 64)  # the store keeps rows < 64 and columns < 64
DEVICE = [
    # tag, slots, K (A's row stride and B's row count), X, block-K offsets, loads masked, store, scale, replayed
    (
        "constant_beside_stride_k",
        dict(loop="range(0, 64, BK)", **UNMASKED),
        128,
        7,
        (0, 32),
        False,
        CANONICAL_STORE,
        1,
        False,
    ),
    (
        "constant_count_beside_k",
        dict(loop="range(0, 3)", **COUNTED, **UNMASKED),
        128,
        7,
        (0, 32, 64),
        False,
        CANONICAL_STORE,
        1,
        False,
    ),
    (
        "count_argument_of_one",
        dict(loop="range(0, X)", **COUNTED, **UNMASKED),
        96,
        1,
        (0,),
        False,
        CANONICAL_STORE,
        1,
        False,
    ),
    (
        "another_extent_unmasked",
        dict(loop="range(0, X, BK)", **UNMASKED),
        128,
        64,
        (0, 32),
        False,
        CANONICAL_STORE,
        1,
        False,
    ),
    ("single_tile_beside_k", dict(body=SINGLE), 96, 7, (0,), False, CANONICAL_STORE, 1, False),
    (
        "renamed_ceiling",
        dict(loop="range(0, tl.cdiv({K}, BK))", **COUNTED, M="R", N="S", K="T"),
        48,
        7,
        (0, 32),
        True,
        CANONICAL_STORE,
        1,
        False,
    ),
    (
        # 8f2c58f read a clip compared before its ``[None, :]`` as a ROW clip: rows < 40, every column.
        "column_clip_compared_before_lifting",
        dict(smask="(rn < X)[None, :]"),
        64,
        40,
        (0, 32),
        True,
        (64, 40),
        1,
        False,
    ),
    (
        "epilogue_later_start",
        dict(loop="range(BK, {K}, BK)", value="acc * 2.0"),
        96,
        7,
        (32, 64),
        True,
        CANONICAL_STORE,
        2,
        True,
    ),
    (
        "epilogue_one_block_short",
        dict(loop="range(0, {K} - BK, BK)", value="acc * 2.0"),
        96,
        7,
        (0, 32),
        True,
        CANONICAL_STORE,
        2,
        True,
    ),
    (
        "epilogue_floor_count",
        dict(loop="range(0, {K} // BK)", value="acc * 2.0", **COUNTED),
        80,
        7,
        (0, 32),
        True,
        CANONICAL_STORE,
        2,
        True,
    ),
    (
        "epilogue_second_row_clip",
        dict(value="acc * 2.0", smask=STORE_CLIPS["second_row_clip"]),
        64,
        40,
        (0, 32),
        True,
        (40, 64),
        2,
        True,
    ),
    (
        "epilogue_flat_constant_clip",
        dict(value="acc * 2.0", pid=FLAT, smask="(rm[:, None] < 48) & (rn[None, :] < {N})"),
        64,
        7,
        (0, 32),
        True,
        (48, 64),
        2,
        True,
    ),
    (
        "epilogue_shifted_row_clip",
        dict(value="acc * 2.0", smask=OTHER_INDEX_CLIPS["shifted_rows"]),
        64,
        7,
        (0, 32),
        True,
        lambda rm, rn: (rm + 1 < 64) & (rn < 64),
        2,
        True,
    ),
    (
        "epilogue_scaled_row_clip",
        dict(value="acc * 2.0", smask=OTHER_INDEX_CLIPS["scaled_rows"]),
        64,
        7,
        (0, 32),
        True,
        lambda rm, rn: (rm * 2 < 64) & (rn < 64),
        2,
        True,
    ),
    (
        "epilogue_rows_as_columns",
        dict(value="acc * 2.0", smask="(rm[:, None] < {M}) & (rm[None, :] < X)"),
        64,
        40,
        (0, 32),
        True,
        lambda rm, rn: (rm < 64) & (rm.reshape(1, -1) < 40),
        2,
        True,
    ),
]


@pytest.mark.parametrize("fast", ["default", "0"])
@pytest.mark.parametrize("tag,slots,k,x,offsets,masked,store,scale,replayed", DEVICE, ids=[c[0] for c in DEVICE])
def test_both_matmul_routes_run_the_source_loop_and_store(
    tmp_path, monkeypatch, routes, tag, slots, k, x, offsets, masked, store, scale, replayed, fast
):
    if fast == "default":
        monkeypatch.delenv("TRITON_MSL_FAST_MATMUL", raising=False)
    else:
        monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", fast)
    fn = _variant(tmp_path, f"{tag}_fast_{fast}", **slots)
    m = n = 64
    generator = torch.Generator().manual_seed(34378)
    a_cpu = torch.randint(-3, 4, (m * k,), generator=generator).float()
    b_cpu = torch.randint(-3, 4, (k * n,), generator=generator).float()
    a, b = a_cpu.to("mps"), b_cpu.to("mps")
    backing = torch.full((m * n + 64,), POISON, device="mps")
    c = backing[32:-32]
    flat = slots.get("pid") == FLAT
    grid = (4,) if flat else (2, 2)
    programs = [(p % 2, p // 2) for p in range(4)] if flat else [(pm, pn) for pm in range(2) for pn in range(2)]
    fn[grid](a, b, c, m, n, k, x, 32, 32, 32)
    torch.mps.synchronize()
    expected = _reference(
        a_cpu,
        b_cpu,
        (m, n),
        k,
        programs,
        32,
        offsets,
        m if masked else None,
        k,
        n,
        store if callable(store) else lambda rm, rn: (rm < store[0]) & (rn < store[1]),
        scale,
    )
    # Small-integer data: every sum is exact in fp32, so the default route must reproduce it exactly.
    assert torch.equal(c.cpu(), expected)
    assert torch.all(backing[:32].cpu() == POISON) and torch.all(backing[-32:].cpu() == POISON)
    assert torch.equal(a.cpu(), a_cpu) and torch.equal(b.cpu(), b_cpu)
    assert routes
    assert all(route[1] != "simdgroup_matmul_fast" for route in routes) or not (replayed or fast == "0")
