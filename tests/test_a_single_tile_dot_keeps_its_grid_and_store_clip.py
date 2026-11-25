"""A loop-free dot keeps the source's program_id tiling, store clip and in-tile transposes.

The dense single-tile template computes tile (0, 0) of baked extents in one threadgroup. A
loop-free dot whose row strides are the literal tile sizes took it even when the source tiled
an axis by ``program_id`` or clipped its store at a runtime extent: a second row tile was never
computed, and a store clipped at ``M = 20`` wrote all 32 rows (384 cells the source does not
store; past the end of an output of exactly ``M`` rows). Such a kernel now takes the
stride-aware scalar template, which replays the proved grid mapping and the store clip, and the
dense template only ever sees an untiled, unclipped tile. The scalar template itself addressed
an in-tile transposed operand as if it were not transposed; it now walks the transposed index.

Every launch here is a defined program: a numpy model executes the source, program by program,
on the backing arrays and asserts that each address lies inside the tensor that is passed. The
tensors are views inside larger storages, and the whole output storage is compared, so a store
the source does not make is seen wherever it lands.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.msl_emitter import emit_msl
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")

BLOCK = 32
FILL = -777.0
OFFSET, SIZE = 1500, 16384
DENSE_MARK = "tg_A[32 * 8]"  # the dense template's staging array


@triton.jit
def _tile(
    A,
    B,
    C,
    M,
    N,
    SA,
    SB,
    SC,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    AXES: tl.constexpr,
    CLIP: tl.constexpr,
    TRANS: tl.constexpr,
    RUNTIME: tl.constexpr,
    LC: tl.constexpr,
    SOFTMAX: tl.constexpr,
):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    if AXES & 1:
        rm = tl.program_id(0) * BM + rm
    if AXES & 2:
        rn = tl.program_id(1) * BN + rn
    if RUNTIME:
        a = tl.load(A + rm[:, None] * SA + rk[None, :])
        b = tl.load(B + rk[:, None] * SB + rn[None, :])
        cp = C + rm[:, None] * SC + rn[None, :]
    else:
        a = tl.load(A + rm[:, None] * BK + rk[None, :])
        b = tl.load(B + rk[:, None] * BN + rn[None, :])
        cp = C + rm[:, None] * LC + rn[None, :]
    if TRANS == 1:
        a = tl.trans(a)
    if TRANS == 2:
        b = tl.trans(b)
    c = tl.dot(a, b)
    if SOFTMAX:
        c = c - tl.max(c, 1)[:, None]
        c = tl.exp(c)
        c = c / tl.sum(c, 1)[:, None]
    if CLIP == 3:
        tl.store(cp, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
    elif CLIP == 1:
        tl.store(cp, c, mask=rm[:, None] < M)
    elif CLIP == 2:
        tl.store(cp, c, mask=rn[None, :] < N)
    else:
        tl.store(cp, c)


def _backing(seed):
    i = np.arange(SIZE, dtype=np.int64)
    # Small integers: every float32 product and sum below is exact.
    return (((i * 2654435761 + seed * 40503) >> 7) % 7 - 3).astype(np.float32)


def _strides(case):
    """(A row, B row, C row) strides the source uses."""
    if case["runtime"]:
        return case["strides"]
    return (BLOCK, BLOCK, case.get("lc", BLOCK))


def _source(case, a_host, b_host, lengths):
    """The source program on the backing arrays -> the expected output storage."""
    sa, sb, sc = _strides(case)
    span = np.arange(BLOCK)
    expected = np.full(SIZE, FILL, np.float64)
    writers = {}
    for px in range(case["grid"][0]):
        for py in range(case["grid"][1]):
            rm = (px * BLOCK if case["axes"] & 1 else 0) + span
            rn = (py * BLOCK if case["axes"] & 2 else 0) + span
            ap = rm[:, None] * sa + span[None, :]
            bp = span[:, None] * sb + rn[None, :]
            assert 0 <= ap.min() and ap.max() < lengths[0], "the source would read outside A"
            assert 0 <= bp.min() and bp.max() < lengths[1], "the source would read outside B"
            a = a_host[OFFSET + ap].astype(np.float64)
            b = b_host[OFFSET + bp].astype(np.float64)
            if case["trans"] == 1:
                a = a.T
            if case["trans"] == 2:
                b = b.T
            c = a @ b
            if case["softmax"]:
                c = c.astype(np.float32)
                c = np.exp(c - c.max(axis=1, keepdims=True))
                c = (c / c.sum(axis=1, keepdims=True)).astype(np.float64)
            mask = np.ones((BLOCK, BLOCK), bool)
            if case["clip"] & 1:
                mask &= rm[:, None] < case["M"]
            if case["clip"] & 2:
                mask &= rn[None, :] < case["N"]
            cp = rm[:, None] * sc + rn[None, :] + 0 * span[None, :]
            if mask.any():
                assert 0 <= cp[mask].min() and cp[mask].max() < lengths[2], "the source would store outside C"
            for address, value in zip(cp[mask].tolist(), c[mask].tolist()):
                assert writers.setdefault(address, (px, py)) == (px, py), "two programs store one cell"
                expected[OFFSET + address] = value
    return expected


def _case(
    grid,
    m,
    n,
    axes,
    clip,
    trans=0,
    runtime=False,
    strides=(80, 72, 96),
    lc=BLOCK,
    lengths=(8192, 4096, 8192),
    softmax=False,
):
    return dict(
        grid=grid,
        M=m,
        N=n,
        axes=axes,
        clip=clip,
        trans=trans,
        runtime=runtime,
        strides=strides,
        lc=lc,
        lengths=lengths,
        softmax=softmax,
    )


CASES = {
    # the dense template's own domain: one whole unclipped tile
    "whole_tile": _case((1, 1), 32, 32, 0, 0),
    "whole_tile_trans_a": _case((1, 1), 32, 32, 0, 0, trans=1),
    "whole_tile_trans_b": _case((1, 1), 32, 32, 0, 0, trans=2),
    # C = A(64x32) @ B(32x32), rows tiled by program_id(0)
    "two_row_tiles": _case((2, 1), 64, 32, 1, 3),
    "two_row_tiles_second_partial": _case((2, 1), 52, 32, 1, 3),
    "two_row_tiles_row_clip_only": _case((2, 1), 64, 32, 1, 1),
    # one program whose store is clipped inside the tile
    "row_clipped": _case((1, 1), 20, 32, 1, 3),
    "column_clipped": _case((1, 1), 32, 20, 1, 3),
    "both_clipped": _case((1, 1), 20, 12, 1, 3),
    "no_program_id_row_clipped": _case((1, 1), 20, 32, 0, 3),
    "no_program_id_column_clipped": _case((1, 1), 32, 20, 0, 2),
    "clip_at_the_tile": _case((1, 1), 32, 32, 0, 3),
    "clip_beyond_the_tile": _case((1, 1), 1000, 1000, 0, 3),
    # in-tile transposes with a clip or a tiled row axis
    "trans_a_row_clipped": _case((1, 1), 20, 32, 0, 3, trans=1),
    "trans_b_column_clipped": _case((1, 1), 32, 20, 0, 3, trans=2),
    "trans_a_two_row_tiles": _case((2, 1), 52, 32, 1, 3, trans=1),
    "trans_b_two_row_tiles": _case((2, 1), 52, 32, 1, 3, trans=2),
    # Two tiled axes. With every stride the literal tile size the output rows of
    # neighbouring column tiles are the same cells (C row stride 32, columns to 63), so
    # that source is a race and is left out. An output row stride that holds both column
    # tiles makes it a defined program.
    "two_axes_wide_output_rows": _case((2, 2), 52, 52, 3, 3, lc=96),
    # runtime strides: the scalar template's existing domain, and its transposes
    "runtime_strides_two_axes": _case((2, 2), 52, 52, 3, 3, runtime=True),
    "runtime_strides_trans_a": _case((1, 1), 32, 32, 0, 0, trans=1, runtime=True),
    "runtime_strides_trans_b": _case((1, 1), 32, 32, 0, 0, trans=2, runtime=True),
    "runtime_strides_trans_a_two_row_tiles": _case((2, 1), 52, 32, 1, 3, trans=1, runtime=True),
    "runtime_strides_trans_b_two_column_tiles": _case((1, 2), 32, 52, 2, 3, trans=2, runtime=True),
    # the fused matmul + row softmax template: one untiled tile, its store clipped
    "softmax_whole": _case((1, 1), 32, 32, 0, 0, softmax=True),
    "softmax_row_clipped": _case((1, 1), 20, 32, 0, 3, softmax=True),
    "softmax_column_clipped": _case((1, 1), 32, 20, 0, 2, softmax=True),
    "softmax_both_clipped": _case((1, 1), 20, 12, 0, 3, softmax=True),
    "softmax_clip_at_the_tile": _case((1, 1), 32, 32, 0, 3, softmax=True),
    "softmax_runtime_strides_row_clipped": _case((1, 1), 20, 32, 0, 1, runtime=True, softmax=True),
}


def _keywords(case):
    return dict(
        BM=BLOCK,
        BN=BLOCK,
        BK=BLOCK,
        AXES=case["axes"],
        CLIP=case["clip"],
        TRANS=case["trans"],
        RUNTIME=case["runtime"],
        LC=case["lc"],
        SOFTMAX=case["softmax"],
    )


def _launch(case, c_view, a_view, b_view):
    return _tile[case["grid"]](a_view, b_view, c_view, case["M"], case["N"], *case["strides"], **_keywords(case))


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_single_tile_dot_equals_the_source(name, compile_shader, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    case = CASES[name]
    a_host, b_host, c_host = _backing(1), _backing(2), np.full(SIZE, FILL, np.float32)
    expected = _source(case, a_host, b_host, case["lengths"])
    a, b, c = (torch.from_numpy(x).to("mps") for x in (a_host, b_host, c_host))
    views = [t[OFFSET : OFFSET + n] for t, n in zip((a, b, c), case["lengths"])]
    handle = _launch(case, views[2], views[0], views[1])
    torch.mps.synchronize()
    got = c.cpu().numpy().astype(np.float64)
    # The softmax is float arithmetic on the GPU; every other case is exact in float32.
    close = np.isclose(got, expected, rtol=2e-5, atol=2e-6) if case["softmax"] else got == expected
    wrong = np.flatnonzero(~close)
    assert len(wrong) == 0, (
        f"{len(wrong)} cells of the output storage differ from the source: "
        f"{int(np.sum((got == FILL) & (expected != FILL)))} the source stores were not written, "
        f"{int(np.sum((got != FILL) & (expected == FILL)))} it does not store were written; first at "
        f"{int(wrong[0]) - OFFSET} (got {got[wrong[0]]}, source {expected[wrong[0]]})"
    )
    assert np.array_equal(a.cpu().numpy(), a_host) and np.array_equal(b.cpu().numpy(), b_host)
    if not case["softmax"]:
        dense = DENSE_MARK in handle.asm["msl"]
        assert dense == (name.startswith("whole_tile")), "only an untiled, unclipped literal-stride tile is dense"


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("m,n", [(20, 32), (32, 20), (20, 12)])
def test_clipped_store_fits_an_output_of_exactly_its_extent(m, n, compile_shader, monkeypatch):
    """The clipped store is all the source writes: an output that ends at its last cell is enough."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    a_host, b_host = _backing(1)[:1024], _backing(2)[:1024]
    cells = (m - 1) * BLOCK + n  # row stride 32: the last stored cell is (m - 1, n - 1)
    a, b = torch.from_numpy(a_host).to("mps"), torch.from_numpy(b_host).to("mps")
    c = torch.full((cells,), FILL, device="mps")
    _tile[(1, 1)](
        a,
        b,
        c,
        m,
        n,
        1,
        1,
        1,
        BM=BLOCK,
        BN=BLOCK,
        BK=BLOCK,
        AXES=1,
        CLIP=3,
        TRANS=0,
        RUNTIME=False,
        LC=BLOCK,
        SOFTMAX=False,
    )
    torch.mps.synchronize()
    product = a_host.reshape(32, 32).astype(np.float64) @ b_host.reshape(32, 32).astype(np.float64)
    expected = np.full(m * BLOCK, FILL)
    expected.reshape(m, BLOCK)[:, :n] = product[:m, :n]
    assert np.array_equal(c.cpu().numpy().astype(np.float64), expected[:cells])


def _lower(axes, clip, trans=0, runtime=False, lc=BLOCK, pointer="*fp32", softmax=False):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    signature = {name: pointer for name in ("A", "B")}
    signature["C"] = "*fp32"
    signature.update({name: "i32" for name in ("M", "N", "SA", "SB", "SC")})
    constants = dict(
        BM=BLOCK, BN=BLOCK, BK=BLOCK, AXES=axes, CLIP=clip, TRANS=trans, RUNTIME=runtime, LC=lc, SOFTMAX=softmax
    )
    signature.update({name: "constexpr" for name in constants})
    context = ir.context()
    ir.load_dialects(context)
    module = ASTSource(_tile, signature, constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    return emit_msl(module, {}, options)


@pytest.mark.parametrize("pointer", ["*fp32", "*fp16"])
@pytest.mark.parametrize("trans", [0, 1, 2])
@pytest.mark.parametrize("clip", [0, 1, 2, 3])
@pytest.mark.parametrize("axes", [0, 1, 2, 3])
def test_dense_template_only_for_an_untiled_unclipped_tile(axes, clip, trans, pointer):
    """Literal tile strides: dense only without a program id and without a runtime store clip."""
    try:
        msl = _lower(axes, clip, trans, pointer=pointer)
    except MetalNonRecoverableError:
        # A refusal is allowed for a tiled kernel only; an untiled one has to compile.
        assert axes, "an untiled single-tile dot must compile"
        return
    if axes == 0 and clip == 0:
        assert DENSE_MARK in msl
        return
    assert DENSE_MARK not in msl
    assert "single, non-looped dot tile" in msl
    # Each tiled axis takes its tile from the program id; an untiled axis is tile 0.
    assert ("pid_m = pid3.x;" in msl) == bool(axes & 1)
    assert ("pid_n = pid3.y;" in msl) == bool(axes & 2)
    # The clipped extent is read from its scalar argument (M is argument 3, N argument 4).
    assert ("_mmarg3 = _mmarg3_buf[0]" in msl) and ("_mmarg4 = _mmarg4_buf[0]" in msl)
    if clip & 1:
        assert "long _M = (long)(_mmarg3);" in msl
    if clip & 2:
        assert "long _N = (long)(_mmarg4);" in msl


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("trans", [1, 2])
def test_scalar_template_walks_the_transposed_index(trans, runtime):
    """``trans(load(A + rm * s + rk))`` reduces over the row term, with the row tile's offset."""
    msl = _lower(1, 3, trans, runtime=runtime)
    assert DENSE_MARK not in msl
    if trans == 1:
        assert "(uint)(row_base + k) * " in msl and "(uint)_lm * " in msl
    else:
        assert "(uint)_ln * " in msl and "(uint)(col_base + k) * " in msl


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("clip", [0, 1, 2, 3])
def test_fused_softmax_template_clips_its_store(clip, runtime):
    """The fused matmul + softmax template stores only below the store mask's runtime extents."""
    msl = _lower(0, clip, runtime=runtime, softmax=True)
    assert "tg_C[" in msl, "the fused template stages its output"
    rows, cols = "(long)(mstrip + row) < (long)(_mmarg3)", "(long)(col) < (long)(_mmarg4)"
    assert (rows in msl) == bool(clip & 1) and (cols in msl) == bool(clip & 2)
    if clip == 3:
        assert f"if ({rows} && {cols}) " in msl
