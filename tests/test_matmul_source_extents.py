"""Source-proved matmul clips and grid domains, independent of scalar spelling.

GPU cases are proposed regressions until run under the shared device lease.
"""

import importlib.util
import sys

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tests.test_matmul_subtiled_twin import routes, _assert_no_twin  # noqa: F401
from triton_msl.backend import driver, _generic_launch
from triton_msl.backend._matmul_domain import matmul_grid_domain
from triton_msl.backend.compiler import MetalBackend


SOURCE = """
@triton.jit
def {name}(a, b, c, rows, cols, depth, M, N, K, sak, sbn, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           STORE: tl.constexpr, LOAD: tl.constexpr, MAP: tl.constexpr):
    pid = tl.program_id(0)
    if MAP == 0:
        pm = pid
        pn = tl.program_id(1)
    elif MAP == 1:
        nm = (rows + BM - 1) // BM
        pm = pid % nm
        pn = pid // nm
    else:
        nn = (cols + BN - 1) // BN
        pm = pid // nn
        pn = pid % nn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, depth, BK):
        if LOAD:
            av = tl.load(a + rm[:, None] * M + (k + rk)[None, :] * sak,
                         mask=(rm[:, None] < rows) & ((k + rk)[None, :] < depth), other=0.)
            bv = tl.load(b + (k + rk)[:, None] * N + rn[None, :] * sbn,
                         mask=((k + rk)[:, None] < depth) & (rn[None, :] < cols), other=0.)
        else:
            av = tl.load(a + rm[:, None] * M + (k + rk)[None, :] * sak)
            bv = tl.load(b + (k + rk)[:, None] * N + rn[None, :] * sbn)
        acc += tl.dot(av, bv)
    if STORE == 0:
        tl.store(c + rm[:, None] * K + rn[None, :] * scn, acc)
    elif STORE == 1:
        tl.store(c + rm[:, None] * K + rn[None, :] * scn, acc,
                 mask=(rm[:, None] < rows) & (rn[None, :] < cols))
    else:
        tl.store(c + rm[:, None] * K + rn[None, :] * scn, acc,
                 mask=(rm[:, None] < rows))
"""


def _kernel(tmp_path, name="source_extent_mm", source=SOURCE):
    path = tmp_path / (name + ".py")
    path.write_text("import triton\nimport triton.language as tl\n" + source.format(name=name))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def _compile(fn, bm=128, bn=64, store=0, load=False, mapping=0, layout="contig"):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    constants = dict(
        BM=bm, BN=bn, BK=32, STORE=store, LOAD=load, MAP=mapping, sak=1, sbn=1 if layout == "contig" else 128, scn=1
    )
    if layout == "bt":
        constants["N"] = 1
        constants.pop("sbn")
    signature = {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(fn.arg_names)}
    signature.update({name: "constexpr" for name in constants})
    source = ASTSource(fn=fn, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options), metadata


@pytest.mark.parametrize("mapping", [0, 1, 2])
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("bm,bn", [(32, 32), (128, 64), (128, 128)])
def test_unmasked_decoy_extents_are_entire_source_tiles(tmp_path, mapping, layout, bm, bn):
    msl, metadata = _compile(_kernel(tmp_path), bm, bn, mapping=mapping, layout=layout)
    # This fails on 6088d8e: _M comes from the unrelated row stride M.
    assert f"long _M = (long)((long)row_base + {bm}L);" in msl
    assert f"long _N = (long)((long)col_base + {bn}L);" in msl
    assert metadata["batched_dot_bounds"][0] == "matmul_grid_storage_v1"
    assert metadata["batched_dot_bounds"][3][:2] == (None, None)


@pytest.mark.parametrize("store", [0, 1, 2])
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_store_and_load_masks_have_independent_domains(tmp_path, store, layout):
    msl, metadata = _compile(_kernel(tmp_path), store=store, load=True, layout=layout)
    # rows and cols precede the stride decoys M/N/K in the unchanged ABI.
    assert "long _A_M = (long)(_mmarg3);" in msl
    assert "long _B_N = (long)(_mmarg4);" in msl
    assert "m < _A_M ?" in msl and "n < _B_N ?" in msl
    assert ("long _M = (long)(_mmarg3);" in msl) == (store != 0)
    assert ("long _N = (long)(_mmarg4);" in msl) == (store == 1)
    if store != 1:
        assert metadata["batched_dot_bounds"][0] == "matmul_grid_storage_v1"


def _plan(mapping="2d", axes=(True, True), loads=(None, None), full=True):
    return (32, 32, mapping, axes, (("literal", 64, "i32"), ("literal", 96, "i32")), loads, full)


@pytest.mark.parametrize(
    "mapping,grid,expected",
    [
        ("2d", (4, 3, 1), (128, 96)),
        ("1d", (7, 1, 1), (64, 128)),
        (("flat_n", 32), (7, 1, 1), (96, 96)),
        (("grouped", 4, 32), (6, 1, 1), (64, 96)),
    ],
)
def test_program_grid_domain_is_source_mapping(mapping, grid, expected):
    assert matmul_grid_domain(_plan(mapping), grid, lambda ref: ref[1]) == expected


def _bounds(store=(None, None), loads=(None, None), full=True):
    lit = lambda x: ("literal", x, "i64")
    return (
        "matmul_grid_storage_v1",
        32,
        (0, 1, 2),
        (*store, ("arg", 3, "i32")),
        (lit(32), lit(1), lit(64), lit(1), lit(64), lit(1)),
        (("i32", "i32", False),) * 3,
        _plan(loads=loads, full=full),
    )


def test_aligned_k_grid_plan_proves_full_unmasked_storage():
    good = [torch.empty(64 * 32), torch.empty(32 * 64), torch.empty(64 * 64), 32]
    assert driver._matmul_tail_bounds_reason(_bounds(), good, grid=(2, 2, 1)) is None
    for index in range(3):
        bad = good.copy()
        bad[index] = torch.empty(1)
        assert "outside" in driver._matmul_tail_bounds_reason(_bounds(), bad, grid=(2, 2, 1))


def test_source_masks_do_not_hide_unmasked_operand_storage():
    clip = ("literal", 1, "i32")
    args = [torch.empty(32), torch.empty(32 * 64), torch.empty(1), 32]
    # The output stores one cell but the unmasked source loads every tile row.
    assert "outside" in driver._matmul_tail_bounds_reason(_bounds((clip, clip)), args, grid=(2, 2, 1))
    assert driver._matmul_tail_bounds_reason(_bounds((clip, clip), (clip, clip)), args, grid=(2, 2, 1)) is None


@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_ordinary_and_layout_fallback_finish_reads_before_stores(tmp_path, layout):
    msl, _ = _compile(_kernel(tmp_path), layout=layout)
    ordinary = msl.split("__mmdirect(")[0]
    if layout == "contig":
        fast = ordinary.split("if (row_base +", 1)[1].split("return;", 1)[0]
        assert (
            fast.rfind("simdgroup_load")
            < fast.index("threadgroup_barrier(mem_flags::mem_device)")
            < fast.index("simdgroup_store")
        )
        assert "_fallback_cells" in ordinary
    else:
        assert "_cell[" in ordinary
    assert "threadgroup_barrier(mem_flags::mem_device)" in ordinary


requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


@requires_gpu
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("mapping", [0, 1, 2])
@pytest.mark.parametrize("store,load", [(0, False), (0, True), (1, True), (2, True)])
@pytest.mark.parametrize("reduction", [33, 128])
@pytest.mark.parametrize("ordinary_only", [False, True])
def test_source_extent_device_poison_and_guard(
    tmp_path, monkeypatch, routes, layout, mapping, store, load, reduction, ordinary_only
):
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    if ordinary_only:
        monkeypatch.setattr(_generic_launch, "direct_twin", lambda *args: None)
    fn = _kernel(tmp_path)
    rows, cols, depth = (129, 65, reduction) if load else (256, 128, 128)
    rm, rn = triton.cdiv(rows, 128) * 128, triton.cdiv(cols, 64) * 64
    generator = torch.Generator().manual_seed(17391)
    a_cpu = torch.randn(rm, depth, generator=generator)
    b_cpu = torch.randn(depth, rn, generator=generator)
    a = a_cpu.to("mps")
    b = (b_cpu if layout == "contig" else b_cpu.t().contiguous().t()).to("mps")
    backing = torch.full((rm * rn + 64,), -98765.0, device="mps")
    c = backing[32:-32].view(rm, rn)
    grid = (
        (triton.cdiv(rows, 128), triton.cdiv(cols, 64))
        if mapping == 0
        else (triton.cdiv(rows, 128) * triton.cdiv(cols, 64),)
    )
    handle = fn[grid](
        a,
        b,
        c,
        rows,
        cols,
        depth,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        a.stride(1),
        b.stride(1),
        c.stride(1),
        128,
        64,
        32,
        store,
        load,
        mapping,
    )
    assert len(routes) == 1
    if ordinary_only:
        _assert_no_twin(handle, routes)
    expected_a, expected_b = a_cpu.clone(), b_cpu.clone()
    if load:
        expected_a[rows:] = 0
        expected_b[:, cols:] = 0
    expected = (expected_a.double() @ expected_b.double()).float()
    mask = torch.ones((rm, rn), dtype=torch.bool)
    if store:
        mask[rows:] = False
    if store == 1:
        mask[:, cols:] = False
    torch.testing.assert_close(a.cpu(), a_cpu, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), b_cpu, rtol=0, atol=0)
    actual = c.cpu()
    torch.testing.assert_close(actual[mask], expected[mask], atol=3e-3, rtol=3e-3)
    assert torch.all(actual[~mask] == -98765.0)
    assert torch.all(backing[:32].cpu() == -98765.0) and torch.all(backing[-32:].cpu() == -98765.0)


def test_missing_m_n_scalar_names_still_emit_source_tiles(tmp_path):
    import re

    source = SOURCE
    for old, new in (("M", "lda"), ("N", "ldb"), ("K", "ldc")):
        source = re.sub(r"\b" + old + r"\b", new, source)
    msl, metadata = _compile(_kernel(tmp_path, source=source))
    assert "long _M = (long)((long)row_base + 128L);" in msl
    assert metadata["batched_dot_bounds"][0] == "matmul_grid_storage_v1"


@pytest.mark.parametrize("bound", [17, 128, 256])
def test_literal_multi_program_store_masks_remain_explicit_refusals(tmp_path, bound):
    from triton_msl.errors import MetalNonRecoverableError

    source = SOURCE.replace("rm[:, None] < rows", "rm[:, None] < " + str(bound))
    with pytest.raises(MetalNonRecoverableError):
        _compile(_kernel(tmp_path, source=source), store=1)


@requires_gpu
@pytest.mark.parametrize("alias", ["a", "b"])
@pytest.mark.parametrize("forced_scalar", [False, True])
@pytest.mark.parametrize("block", [64, 128])
def test_single_program_ordinary_alias_finishes_all_reads(tmp_path, monkeypatch, routes, alias, forced_scalar, block):
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")

    if forced_scalar:
        original = GenericLowerer._simd_layout_guard_lines

        def false_layout(self, *args, **kwargs):
            return [*original(self, *args, **kwargs), "    _simd_ok = false;"]

        monkeypatch.setattr(GenericLowerer, "_simd_layout_guard_lines", false_layout)
    fn = _kernel(tmp_path, name=f"alias_{alias}_{forced_scalar}_{block}")
    generator = torch.Generator().manual_seed(29011)
    initial_a = torch.randn(block, block, generator=generator)
    initial_b = torch.randn(block, block, generator=generator)
    expected = (initial_a.double() @ initial_b.double()).float()
    for repeat in range(12):
        ab = torch.full((block * block + 64,), -98765.0, device="mps")
        bb = torch.full_like(ab, -98765.0)
        a, b = ab[32:-32].view(block, block), bb[32:-32].view(block, block)
        a.copy_(initial_a)
        b.copy_(initial_b)
        c = a if alias == "a" else b
        handle = fn[(1, 1)](a, b, c, block, block, block, block, block, block, 1, 1, 1, block, block, 32, 0, False, 0)
        # Alias dispatch must use the ordinary entry, whose load/store boundary
        # is independently checked in CPU emission tests above.
        assert handle.metadata.mm_two_kernel is not None
        assert len(routes) == repeat + 1
        _assert_no_twin(handle, routes)
        if forced_scalar:
            assert "_simd_ok = false;" in handle.asm["msl"]
        torch.testing.assert_close(c.cpu(), expected, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(
            (b if alias == "a" else a).cpu(), initial_b if alias == "a" else initial_a, atol=0, rtol=0
        )
        for backing in (ab, bb):
            assert torch.all(backing[:32].cpu() == -98765.0) and torch.all(backing[-32:].cpu() == -98765.0)


@requires_gpu
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("mapping", [0, 1, 2])
def test_roomy_unmasked_source_stores_surplus_tiles(tmp_path, monkeypatch, routes, layout, mapping):
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    # Exercise the repaired ordinary entry even when a direct twin is eligible.
    monkeypatch.setattr(_generic_launch, "direct_twin", lambda *args: None)
    fn = _kernel(tmp_path, name="roomy_source_mm")
    generator = torch.Generator().manual_seed(84711)
    ah = torch.randn(96, 64, generator=generator)
    bh = torch.randn(64, 64, generator=generator)
    a = ah.to("mps")
    b = (bh if layout == "contig" else bh.t().contiguous().t()).to("mps")
    owner = torch.full((96 * 64 + 64,), -98765.0, device="mps")
    c = owner[32:-32].view(96, 64)
    # rows/cols describe the nominal 64x64 problem, but this unmasked source
    # explicitly launches and stores a sixth tile. All A/B/C backing covers it.
    grid = (3, 2) if mapping == 0 else (6,)
    # M-fastest splits by rows=64 -> two row tiles and three column tiles.
    # Supply the actual row divisor 96 so all three mappings cover 96x64.
    source_rows = 96 if mapping == 1 else 64
    handle = fn[grid](
        a,
        b,
        c,
        source_rows,
        64,
        64,
        *[a.stride(0), b.stride(0), c.stride(0)],
        a.stride(1),
        b.stride(1),
        c.stride(1),
        32,
        32,
        32,
        0,
        False,
        mapping,
    )
    assert len(routes) == 1
    _assert_no_twin(handle, routes)
    expected = (ah.double() @ bh.double()).float()
    torch.testing.assert_close(c.cpu(), expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(a.cpu(), ah, atol=0, rtol=0)
    torch.testing.assert_close(b.cpu(), bh, atol=0, rtol=0)
    assert torch.all(owner[:32].cpu() == -98765.0) and torch.all(owner[-32:].cpu() == -98765.0)


def test_independent_grid_divisors_get_a_checked_plan_even_with_matching_masks(tmp_path):
    source = SOURCE.replace("c, rows, cols, depth,", "c, grid_rows, grid_cols, rows, cols, depth,")
    source = source.replace("(rows + BM - 1)", "(grid_rows + BM - 1)")
    source = source.replace("(cols + BN - 1)", "(grid_cols + BN - 1)")
    for mapping in (1, 2):
        _, metadata = _compile(
            _kernel(tmp_path, name=f"independent_divisor_{mapping}", source=source), store=1, load=True, mapping=mapping
        )
        desc = metadata["batched_dot_bounds"]
        assert desc[0] == "matmul_grid_storage_v1"
        assert desc[6][4][mapping - 1] == ("arg", 3 if mapping == 1 else 4, "i32")


@pytest.mark.parametrize("mapping", ["1d", ("flat_n", 32), ("grouped", 4, 32)])
@pytest.mark.parametrize("bad_extent", [-64, 0, 0x7FFFFFFF])
def test_unproved_split_arithmetic_refuses_before_tensor_introspection(mapping, bad_extent):
    from triton_msl.backend._matmul_domain import matmul_grid_rectangles

    plan = list(_plan(mapping))
    plan[4] = (("literal", bad_extent, "i32"),) * 2
    with pytest.raises(ValueError):
        matmul_grid_rectangles(tuple(plan), (2, 1, 1), lambda ref: ref[1])


@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_final_flat_grid_uses_exact_c_address_union(reverse):
    from triton_msl.backend._matmul_domain import matmul_grid_rectangles

    desc = list(_bounds())
    desc[6] = _plan("1d")
    lit = lambda x: ("literal", x, "i64")
    desc[4] = (*desc[4][:4], lit(-128 if reverse else 128), lit(-1 if reverse else 1))
    owner = torch.empty(8160)
    c = owner[8159:] if reverse else owner
    args = [torch.empty(2048), torch.empty(2112), c, 32]
    # Seven tiles omit the bottom-right 32x32 corner: bbox max8191,
    # actual max8159. This exact-sized valid source must remain usable.
    assert matmul_grid_rectangles(desc[6], (7, 1, 1), lambda ref: ref[1]) == ((0, 64, 0, 96), (0, 32, 96, 128))
    assert driver._matmul_tail_bounds_reason(tuple(desc), args, grid=(7, 1, 1)) is None
    smaller = torch.empty(8159)
    args[2] = smaller[8158:] if reverse else smaller
    assert "outside" in driver._matmul_tail_bounds_reason(tuple(desc), args, grid=(7, 1, 1))


def test_modulo_loads_retain_their_separate_store_bound_proof(tmp_path):
    source = SOURCE.replace("a + rm[:, None]", "a + (rm % rows)[:, None]")
    source = source.replace("rn[None, :] * sbn", "(rn % cols)[None, :] * sbn")
    msl, metadata = _compile(_kernel(tmp_path, name="modulo_extent_mm", source=source), store=1)
    # The row/column source moduli match the output clips, so modulo is the
    # identity on every stored cell; out-of-output staging may be zero-filled.
    assert "long _A_M = (long)(_mmarg3);" in msl
    assert "long _B_N = (long)(_mmarg4);" in msl
    assert metadata["batched_dot_bounds"][0] == "matmul_full_k_storage_v1"


def test_modulo_load_with_an_unproved_separate_mask_keeps_its_refusal(tmp_path):
    from triton_msl.errors import MetalNonRecoverableError

    source = SOURCE.replace("c, rows, cols, depth,", "c, load_rows, rows, cols, depth,")
    source = source.replace("    acc =", "    arm = rm % rows\n    acc =")
    source = source.replace("a + rm[:, None]", "a + arm[:, None]")
    source = source.replace("mask=(rm[:, None] < rows) & ((k", "mask=(arm[:, None] < load_rows) & ((k")
    with pytest.raises(MetalNonRecoverableError):
        _compile(_kernel(tmp_path, name="unproved_modulo_mask", source=source), store=1, load=True)
