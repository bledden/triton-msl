"""The fast matmul replacement runs only where it is the source's own store/load/grid domain.

The replacement shader computes rows [0, args[3]) by columns [0, args[4]) and nothing else.
At 34378c9 it was offered to any proved K-loop matmul, so a store clip, load clip or 1-D grid
divisor that was a SEPARATE argument, and an unmasked store over a partial tile, were replaced
by a different result on the default route (GPU: silent, every case below). The suites that
cover these domains set TRITON_MSL_FAST_MATMUL=0; every device case here leaves it at its default.
"""

import pytest
import torch
import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tests.test_fast_matmul_tail_contract import _Runtime
from tests.test_matmul_source_extents import _kernel
from tests.test_matmul_subtiled_twin import routes  # noqa: F401
from triton_msl.autotuning._fast_matmul_dispatch import dispatch_fast_matmul
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError

# {{name}} survives for tests.test_matmul_source_extents._kernel; the roles are spelled per variant.
SOURCE = """
@triton.jit
def {{name}}(a, b, c, M, N, K, MA, NB, MS, NS, GD,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FLAT: tl.constexpr, STORE: tl.constexpr):
    if FLAT:
        pid = tl.program_id(0)
        nm = tl.cdiv({gd}, BM)
        pm = pid % nm
        pn = pid // nm
    else:
        pm = tl.program_id(0)
        pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        av = tl.load(a + rm[:, None] * K + (k + rk)[None, :],
                     mask=(rm[:, None] < {la}) & ((k + rk)[None, :] < K), other=0.)
        bv = tl.load(b + (k + rk)[:, None] * N + rn[None, :],
                     mask=((k + rk)[:, None] < K) & (rn[None, :] < {lb}), other=0.)
        acc += tl.dot(av, bv)
    if STORE:
        tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < {sr}) & (rn[None, :] < {sc}))
    else:
        tl.store(c + rm[:, None] * N + rn[None, :], acc)
"""
CANONICAL = dict(la="M", lb="N", sr="M", sc="N", gd="M")
SEPARATE = {
    "store_rows": dict(sr="MS"),
    "store_cols": dict(sc="NS"),
    "load_rows": dict(la="MA"),
    "load_cols": dict(lb="NB"),
    "grid_divisor": dict(gd="GD"),
    "constant_load_rows": dict(la="64"),
    "constant_load_cols": dict(lb="64"),
}
POISON = -98765.0


def _variant(tmp_path, tag, **roles):
    return _kernel(tmp_path, "fast_domain_" + tag, SOURCE.format(**{**CANONICAL, **roles}))


def _lower(fn, block=32, flat=0, store=1):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    constants = dict(BM=block, BN=block, BK=32, FLAT=flat, STORE=store)
    signature = {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(fn.arg_names)}
    signature.update({name: "constexpr" for name in constants})
    module = ASTSource(fn=fn, signature=signature, constexprs=constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    return lowerer, lowerer.lower()


@pytest.mark.parametrize("flat", [0, 1])
@pytest.mark.parametrize("store", [0, 1])
def test_canonical_domain_keeps_the_replacement(tmp_path, flat, store):
    lowerer, _msl = _lower(_variant(tmp_path, "canonical"), flat=flat, store=store)
    assert lowerer._fast_matmul is not None
    assert lowerer._fast_matmul[12] == (not store, not store)


@pytest.mark.parametrize("role", sorted(SEPARATE))
@pytest.mark.parametrize("flat", [0, 1])
def test_a_separate_domain_argument_is_never_replaced(tmp_path, role, flat):
    # This fails on 34378c9 for every role: the descriptor was built from args 3/4/5 alone.
    lowerer, _msl = _lower(_variant(tmp_path, role, **SEPARATE[role]), flat=flat)
    # A 2-D grid never reads the divisor, so that kernel IS the canonical one.
    assert (lowerer._fast_matmul is None) is (flat == 1 or role != "grid_divisor")


@pytest.mark.parametrize("rows,cols,accepted", [(128, 128, True), (96, 128, False), (128, 96, False), (32, 64, False)])
def test_unmasked_store_is_replaced_only_on_whole_source_tiles(tmp_path, rows, cols, accepted):
    lowerer, _msl = _lower(_variant(tmp_path, "canonical"), block=64, store=0)
    rt = _Runtime()
    grid = (triton.cdiv(rows, 64), triton.cdiv(cols, 64), 1)
    # 34378c9 accepted all four: the replacement stops at M x N, the source stores whole tiles.
    assert dispatch_fast_matmul(rt, lowerer._fast_matmul, [object()] * 3 + [rows, cols, 64], grid=grid) is accepted
    assert bool(rt.calls) is accepted


def test_masked_store_keeps_the_replacement_on_partial_tiles(tmp_path):
    lowerer, _msl = _lower(_variant(tmp_path, "canonical"), block=64, store=1)
    rt = _Runtime()
    assert dispatch_fast_matmul(rt, lowerer._fast_matmul, [object()] * 3 + [96, 128, 64], grid=(2, 2, 1)) is True


@pytest.mark.parametrize(
    "role,line",
    [("constant_load_rows", "long _A_M = (long)(64L);"), ("constant_load_cols", "long _B_N = (long)(64L);")],
)
def test_constant_load_clip_is_replayed_and_proved(tmp_path, role, line):
    lowerer, msl = _lower(_variant(tmp_path, role, **SEPARATE[role]))
    # This fails on v0.3.0..34378c9: the constant was accepted, then dropped for the whole tile.
    assert line in msl
    plan = lowerer._tail_access_bounds
    assert plan[0] == "matmul_grid_storage_v1"
    assert ("literal", 64, "i64") in plan[-1][5]


def test_constant_load_clip_below_the_tile_keeps_its_refusal(tmp_path):
    with pytest.raises(MetalNonRecoverableError, match="constant"):
        _lower(_variant(tmp_path, "constant_small", la="17"))


def _oracle(a, b, shape, programs, block, la, lb, sr, sc):
    """The source, program by program: zero-masked loads, then each program's clipped tile."""
    ea, eb = a.clone(), b.clone()
    ea[la:] = 0
    eb[:, lb:] = 0
    full = (ea.double() @ eb.double()).float()
    out = torch.full(shape, POISON)
    for pm, pn in programs:
        r0, c0 = pm * block, pn * block
        r1 = min(r0 + block, shape[0] if sr is None else sr)
        c1 = min(c0 + block, shape[1] if sc is None else sc)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = full[r0:r1, c0:c1]
    return out


DEVICE = [
    # tag, roles, (M, N, K), scalars MA NB MS NS GD, block, flat, store, grid, replaced
    ("canonical", {}, (64, 64, 64), (64, 64, 64, 64, 64), 32, 0, 1, (2, 2), True),
    ("canonical_flat", {}, (64, 64, 64), (64, 64, 64, 64, 64), 32, 1, 1, (4,), True),
    ("canonical_whole", {}, (128, 64, 64), (0, 0, 0, 0, 0), 64, 0, 0, (2, 1), True),
    ("canonical_partial", {}, (96, 64, 64), (0, 0, 0, 0, 0), 64, 0, 0, (2, 1), False),
    ("canonical_partial_flat", {}, (96, 64, 64), (0, 0, 0, 0, 0), 64, 1, 0, (2,), False),
    ("store_rows", SEPARATE["store_rows"], (64, 64, 64), (64, 64, 32, 64, 64), 32, 0, 1, (2, 2), False),
    ("store_cols", SEPARATE["store_cols"], (64, 64, 64), (64, 64, 64, 32, 64), 32, 1, 1, (4,), False),
    ("load_rows", SEPARATE["load_rows"], (64, 64, 64), (32, 64, 64, 64, 64), 32, 0, 1, (2, 2), False),
    ("load_cols", SEPARATE["load_cols"], (64, 64, 64), (64, 32, 64, 64, 64), 32, 1, 1, (4,), False),
    ("grid_divisor", SEPARATE["grid_divisor"], (64, 64, 64), (64, 64, 64, 64, 32), 32, 1, 1, (4,), False),
    ("constant_load_rows", SEPARATE["constant_load_rows"], (128, 64, 64), (0, 0, 0, 0, 0), 32, 0, 1, (4, 2), False),
    ("constant_load_cols", SEPARATE["constant_load_cols"], (64, 128, 64), (0, 0, 0, 0, 0), 32, 0, 1, (2, 4), False),
    ("constant_rows_flat", SEPARATE["constant_load_rows"], (128, 64, 64), (0, 0, 0, 0, 0), 32, 1, 1, (8,), False),
    ("constant_cols_flat", SEPARATE["constant_load_cols"], (64, 128, 64), (0, 0, 0, 0, 0), 32, 1, 1, (8,), False),
]


@pytest.mark.parametrize("tag,roles,shape,scalars,block,flat,store,grid,replaced", DEVICE, ids=[c[0] for c in DEVICE])
def test_default_route_is_the_source_domain(
    tmp_path, monkeypatch, routes, tag, roles, shape, scalars, block, flat, store, grid, replaced
):
    monkeypatch.delenv("TRITON_MSL_FAST_MATMUL", raising=False)
    fn = _variant(tmp_path, tag, **roles)
    m, n, k = shape
    spelled = dict(zip(("MA", "NB", "MS", "NS", "GD"), scalars), M=m, N=n)
    la, lb, sr, sc, gd = (
        int(v) if v.isdigit() else spelled[v]
        for v in ({**CANONICAL, **roles}[r] for r in ("la", "lb", "sr", "sc", "gd"))
    )
    if flat:
        nm = triton.cdiv(gd, block)
        programs = [(pid % nm, pid // nm) for pid in range(grid[0])]
    else:
        programs = [(pm, pn) for pm in range(grid[0]) for pn in range(grid[1])]
    rows = triton.cdiv(m, block) * block
    generator = torch.Generator().manual_seed(34378)
    a_cpu = torch.randn(rows, k, generator=generator)
    b_cpu = torch.randn(k, n, generator=generator)
    a, b = a_cpu.to("mps"), b_cpu.to("mps")
    backing = torch.full((rows * n + 64,), POISON, device="mps")
    c = backing[32:-32].view(rows, n)
    fn[grid](a, b, c, m, n, k, *scalars, block, block, 32, flat, store)
    expected = _oracle(a_cpu, b_cpu, (rows, n), programs, block, la, lb, sr if store else None, sc if store else None)
    assert [r[1] == "simdgroup_matmul_fast" for r in routes] == [replaced]
    actual = c.cpu()
    written = expected != POISON
    torch.testing.assert_close(actual[written], expected[written], atol=3e-3, rtol=3e-3)
    assert torch.all(actual[~written] == POISON)
    assert torch.all(backing[:32].cpu() == POISON) and torch.all(backing[-32:].cpu() == POISON)
    torch.testing.assert_close(a.cpu(), a_cpu, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), b_cpu, rtol=0, atol=0)
