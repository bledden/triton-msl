"""The sub-tiled ``__mmdirect`` twin for both B layouts: text, descriptor, launch route and device results.

A tile that splits into 32x32 sub-tiles gets a twin whose 32-lane simdgroups each
compute whole sub-tiles; the twin runs at its own threadgroup width (at most four
simdgroups), carried in the descriptor. ``contig`` is row-major B (the K-loop
template); ``bt`` is B stored transposed, ``w.t()`` (the strided template).
"""

import importlib.util
import sys

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from tests.cache_helpers import patch_live_singleton_method
from triton_msl.backend import _generic_launch
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")

SOURCE = """
@triton.jit
def {name}(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MASKED: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = (N + BN - 1) // BN
    offs_m = (pid // n_tiles) * BM + tl.arange(0, BM)
    offs_n = (pid % n_tiles) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * sam + (k + offs_k)[None, :] * sak)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * sbk + offs_n[None, :] * sbn)
        acc += tl.dot(a, b)
    if MASKED:
        tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    else:
        tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc)
"""

# The tutorial's grid: one program id per axis, and masked loads.
SOURCE_2D = """
@triton.jit
def {name}(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MASKED: tl.constexpr):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * sam + (k + offs_k)[None, :] * sak,
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * sbk + offs_n[None, :] * sbn,
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
"""


def _kernel(directory, name, source=SOURCE):
    path = directory / (name + ".py")
    path.write_text("import triton\nimport triton.language as tl\n" + source.format(name=name))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


@pytest.fixture(scope="module")
def mm(tmp_path_factory):
    return _kernel(tmp_path_factory.mktemp("subtiled"), "subtiled_mm")


@pytest.fixture(scope="module")
def mm_2d(tmp_path_factory):
    return _kernel(tmp_path_factory.mktemp("subtiled_2d"), "subtiled_mm_2d", SOURCE_2D)


NAMES = ("a_ptr", "b_ptr", "c_ptr", "M", "N", "K", "sam", "sak", "sbk", "sbn", "scm", "scn")
# Triton specialises an integer argument equal to 1, which is how a call's layout
# reaches the lowerer; the emission tests state the same specialisation directly.
UNIT = {"contig": ("sak", "sbn", "scn"), "bt": ("sak", "sbk", "scn"), "general": ("scn",)}


def _compile(fn, layout, bm, bn, bk=32, masked=False, dtype="fp32"):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    signature = {name: "*" + dtype if i < 3 else "i32" for i, name in enumerate(NAMES)}
    constants = {"BM": bm, "BN": bn, "BK": bk, "MASKED": masked, **{name: 1 for name in UNIT[layout]}}
    for name in constants:
        signature[name] = "constexpr"
    source = ASTSource(fn=fn, signature=signature, constexprs=constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return backend.make_msl(module, metadata, options), metadata


def _twin(msl):
    return msl.split("__mmdirect(", 1)[1]


# Triton's ``arange`` needs a power-of-two range, so 96 is not a reachable width.
TILES = [(32, 32, 32), (32, 64, 64), (64, 32, 64), (64, 64, 128), (128, 64, 128), (128, 128, 128)]


@pytest.mark.parametrize(
    "block_m,block_n,width",
    TILES + [(96, 32, 96), (32, 96, 96), (256, 32, 128), (16, 32, None), (32, 48, None), (0, 32, None), (40, 40, None)],
)
def test_twin_width_is_at_most_four_whole_simdgroups(block_m, block_n, width):
    assert GenericLowerer._direct_twin_width(block_m, block_n) == width


@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("block_m,block_n,width", TILES)
def test_descriptor_carries_the_twin_width_and_roles(mm, layout, masked, block_m, block_n, width):
    msl, metadata = _compile(mm, layout, block_m, block_n, masked=masked)
    descriptor = metadata["mm_two_kernel"]
    assert descriptor == {
        "direct_name": descriptor["direct_name"],
        "direct_block": width,
        "block_m": block_m,
        "block_n": block_n,
        "m_idx": 3,
        "n_idx": 4,
        "k_idx": 5,
        "c_idx": 2,
    }
    assert descriptor["direct_name"].endswith("__mmdirect")
    assert msl.count("kernel void " + descriptor["direct_name"] + "(") == 1
    aligned = [torch.zeros(1), torch.zeros(1), torch.zeros(2 * block_m, 3 * block_n), 2 * block_m, 3 * block_n, 40]
    assert _generic_launch.direct_twin(descriptor, aligned, 6) == (descriptor["direct_name"], width)
    for index, value in ((3, 2 * block_m + 1), (4, 3 * block_n - 32 + 8), (5, 44)):
        assert _generic_launch.direct_twin(descriptor, [*aligned[:index], value, *aligned[index + 1 :]], 6) is None


@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("block_m,block_n,width", TILES)
def test_twin_text_is_subtiled_guarded_and_free_of_threadgroup_memory(mm, layout, block_m, block_n, width):
    twin = _twin(_compile(mm, layout, block_m, block_n)[0])
    subtiles, groups = (block_m // 32) * (block_n // 32), width // 32
    assert "uint _sgw [[threads_per_simdgroup]]" in twin
    assert f"for (uint _st = sgitg; _st < {subtiles}u; _st += {groups}u) {{" in twin
    assert f"uint _sr = row_base + (_st / {block_n // 32}u) * 32u;" in twin
    assert f"uint _sc = col_base + (_st % {block_n // 32}u) * 32u;" in twin
    # one uniform verdict: the layout proof and the 32-lane simdgroup the layout assumes
    assert twin.count("bool _simd_ok =") >= 1 and "_simd_ok = _simd_ok && _sgw == 32u;" in twin
    guard = twin.index("if (!_simd_ok) {")
    fallback = twin[guard : twin.index("return;", guard)]
    assert f"for (uint _fe = tiitg; _fe < {block_m * block_n}u; _fe += {width}u) {{" in fallback
    assert guard < twin.index("for (uint _st = sgitg;")
    code = [line.strip() for line in twin.splitlines() if not line.strip().startswith("//")]
    assert not any(line.startswith("threadgroup ") or "threadgroup_barrier" in line for line in code)
    assert sum("simdgroup_multiply_accumulate(" in line for line in code) == 16
    assert sum(line.startswith("simdgroup_store(") for line in code) == 16


def _descriptor(**changes):
    descriptor = {"direct_name": "mm__mmdirect", "direct_block": 32, "block_m": 32, "block_n": 32}
    return {**descriptor, "m_idx": 3, "n_idx": 4, "k_idx": 5, "c_idx": 2, **changes}


def test_twin_needs_an_output_whose_storage_no_other_pointer_shares():
    """The twin stores sub-tiles while other simdgroups still load, so an output
    that aliases an operand keeps the ordinary entry; operands may alias each other."""

    class Opaque:
        def data_ptr(self):
            return 1

    a, b, c = torch.zeros(4), torch.zeros(4), torch.zeros(64, 64)
    twin = ("mm__mmdirect", 32)
    assert _generic_launch.direct_twin(_descriptor(), [a, b, c, 64, 64, 64], 4) == twin
    assert _generic_launch.direct_twin(_descriptor(), [a, a.view(2, 2).t(), c, 64, 64, 64], 4) == twin
    assert _generic_launch.direct_twin(_descriptor(c_idx=0), [c, a, a, 64, 64, 64], 4) == twin
    for arguments in (
        [a, c, c, 64, 64, 64],
        [c, b, c, 64, 64, 64],
        [a, c[:32], c[32:], 64, 64, 64],  # disjoint views of one storage: not proven apart
        [a, b, None, 64, 64, 64],
        [a, Opaque(), c, 64, 64, 64],
        [a, b, Opaque(), 64, 64, 64],
    ):
        assert _generic_launch.direct_twin(_descriptor(), arguments, 1) is None


def test_twin_needs_an_output_that_holds_a_whole_tile_for_every_program():
    """On an axis the source stores without a mask the twin stores every launched
    tile whole, so a grid with more tiles than the output holds keeps the ordinary entry."""
    a, b = torch.zeros(4), torch.zeros(4)
    backing = torch.zeros(6 * 32 * 32 + 5)
    for cells, programs, expected in (
        (4 * 1024, 4, ("mm__mmdirect", 32)),
        (4 * 1024, 5, None),
        (4 * 1024, 2, ("mm__mmdirect", 32)),
        (5 * 1024, 5, ("mm__mmdirect", 32)),
        (5 * 1024 - 1, 5, None),
        (0, 1, None),
    ):
        c = backing[5 : 5 + cells]  # the view's own extent counts, not its storage
        assert _generic_launch.direct_twin(_descriptor(), [a, b, c, 64, 64, 64], programs) == expected


@pytest.mark.parametrize(
    "descriptor",
    [
        {k: v for k, v in _descriptor().items() if k != "c_idx"},  # the earlier seven-key form
        _descriptor(c_idx=6),
        _descriptor(c_idx=-1),
        _descriptor(c_idx=True),
        _descriptor(direct_block=64),  # a 32x32 tile is one sub-tile: 32 threads
        _descriptor(block_m=64, block_n=64, direct_block=64),  # four sub-tiles: 128
        _descriptor(block_m=16),  # four-strip layout: 128
        _descriptor(block_m=96, direct_block=128),  # three sub-tiles: 96
    ],
)
def test_twin_descriptor_with_a_missing_role_or_a_width_its_tile_does_not_imply_is_not_understood(descriptor):
    arguments = [torch.zeros(1), torch.zeros(1), torch.zeros(192, 192), 192, 192, 64]
    assert _generic_launch.direct_twin(descriptor, arguments, 1) is False


@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("block_m,block_n", [(bm, bn) for bm, bn, _ in TILES] + [(16, 32)])
def test_twin_stores_nothing_for_a_program_outside_the_store_mask_extents(mm, layout, block_m, block_n):
    """The twin stores whole tiles and the launcher proves aligned extents, not an
    exact grid: where the source masks its store, a surplus program must return
    before any store. Without a mask there is no extent, only arguments named M
    and N, and the twin stores every tile as the source does."""
    try:
        msl, metadata = _compile(mm, layout, block_m, block_n, masked=True)
        unmasked = _twin(_compile(mm, layout, block_m, block_n)[0])
    except Exception:
        pytest.skip("kernel refused: no twin")
    if metadata.get("mm_two_kernel") is None:
        pytest.skip("no twin for this layout and tile")
    assert "(long)row_base >= _M" not in unmasked and "(long)col_base >= _N" not in unmasked
    twin = _twin(msl)
    code = [line.strip() for line in twin.splitlines()]
    surplus = code.index("if ((long)row_base >= _M || (long)col_base >= _N) return;")
    assert code.count(code[surplus]) == 1
    assert surplus > max(i for i, line in enumerate(code) if line.startswith(("uint row_base =", "uint col_base =")))
    assert surplus < min(i for i, line in enumerate(code) if "_simd_ok" in line or "simdgroup_store(" in line)
    assert surplus < min(i for i, line in enumerate(code) if line.startswith("for (") or "] = " in line)


@pytest.mark.parametrize("block_m,block_n,cells", [(64, 64, 16), (128, 128, 64), (32, 64, 8), (16, 64, 4)])
def test_unstaged_strided_entry_loads_its_whole_tile_before_it_stores(mm, block_m, block_n, cells):
    msl, _ = _compile(mm, "bt", block_m, block_n, masked=True)
    entry = msl[msl.index("kernel void subtiled_mm(") :].split("kernel void subtiled_mm__mmdirect(")[0]
    code = [line.strip() for line in entry.splitlines()]
    barrier = code.index("threadgroup_barrier(mem_flags::mem_none);")
    assert f"float _cell[{cells}];" in code[:barrier]
    assert code.index("_cell[_e / 256u] = _sum;") < barrier
    assert not any("_mmarg2[" in line for line in code[:barrier])
    (store,) = [line for line in code[barrier:] if line.startswith("_mmarg2[")]
    assert store.endswith("] = (float)_cell[_e / 256u];")
    assert not any("_mmarg0[" in line or "_mmarg1[" in line for line in code[barrier:])


def test_transposed_twin_loads_b_in_place_with_the_transposing_load(mm):
    msl, _ = _compile(mm, "bt", 32, 32)
    twin = _twin(msl)
    for column in range(4):
        assert (
            f"simdgroup_load(b_frag{column}, _mmarg1 + (ulong)(_sc + {8 * column}u) * _ldb + k, _ldb, ulong2(0), true);"
            in twin
        )
    assert "simdgroup_load(a_frag, _mmarg0 + (ulong)(_sr + 0u) * _lda + k, _lda);" in twin
    assert "simdgroup_store(acc_3_3, _mmarg2 + (ulong)(_sr + 24u) * _ldc + _sc + 24u, _ldc);" in twin
    # B's maximal offset is taken along its own layout: (N-1) * ldb + (K-1)
    assert "(_N - 1L) * (long)_ldb + (_K - 1L) < " in twin
    assert "(_K - 1L) * (long)_ldb + (_N - 1L) < " not in twin


def test_contiguous_twin_keeps_the_row_major_load_and_bound(mm):
    twin = _twin(_compile(mm, "contig", 32, 32)[0])
    assert "simdgroup_load(b_frag0, _mmarg1 + (ulong)k * _ldb + _sc + 0u, _ldb);" in twin
    assert "ulong2(0), true" not in twin
    assert "(_K - 1L) * (long)_ldb + (_N - 1L) < " in twin


@pytest.mark.parametrize(
    "layout,kwargs",
    [
        ("general", {}),  # B's K stride is a runtime value: not provably the transposed layout
        ("bt", {"dtype": "fp16"}),  # the transposed twin is fp32 only
        ("bt", {"bm": 16}),  # tiles that do not split into 32x32 sub-tiles
        ("bt", {"bn": 48}),
        ("bt", {"bk": 4}),  # the twin steps K by 8
    ],
)
def test_no_transposed_twin_outside_its_proven_shape(mm, layout, kwargs):
    arguments = {"bm": 32, "bn": 32, "bk": 32, **kwargs}
    try:
        msl, metadata = _compile(
            mm, layout, arguments["bm"], arguments["bn"], arguments["bk"], dtype=arguments.get("dtype", "fp32")
        )
    except Exception:
        return  # a refused kernel has no twin either
    assert metadata.get("mm_two_kernel") is None
    assert "__mmdirect" not in msl


def test_contiguous_tile_that_does_not_split_keeps_the_four_strip_twin(mm):
    msl, metadata = _compile(mm, "contig", 16, 32)
    assert metadata["mm_two_kernel"]["direct_block"] == 128
    twin = _twin(msl)
    assert "_sgw" not in twin and "uint _st" not in twin
    assert "+ col_base + sgitg * 8u + 0u" in twin
    assert "if (!_simd_ok) {" in twin


def _operands(layout, m, n, k, bm, bn, bk, device, integers=True, dtype=torch.float32, slack=0):
    """A, B, a poisoned C with guard bands, and the float64 oracle.

    ``slack`` rows follow the M output rows inside C itself: room for the whole
    tiles of surplus programs, none of which a masked store may touch.

    The source's loads are unmasked, so it reads whole tiles and whole K blocks;
    the backing covers those reads and the oracle sums the K blocks it executes.
    """
    rows, cols, depth = -(-m // bm) * bm, -(-n // bn) * bn, -(-k // bk) * bk
    gen = torch.Generator().manual_seed(977 + m + 3 * n + 5 * k + 7 * bm + 11 * bn)
    if integers:
        a = torch.randint(-3, 4, (rows, depth), generator=gen).float()
        w = torch.randint(-3, 4, (cols, depth), generator=gen).float()
    else:
        a, w = torch.randn((rows, depth), generator=gen), torch.randn((cols, depth), generator=gen)
    a, w = a.to(dtype), w.to(dtype)
    oracle = (a[:m].double() @ w[:n].double().t()).float()
    aa, ww = a.to(device), w.to(device)
    bb = ww.t() if layout == "bt" else ww.t().contiguous()
    owner = torch.full(((m + slack) * n + 8192,), -16384.0, device=device)
    cc = owner[4096:-4096].view(m + slack, n)
    return (a, w), (aa, ww, bb), owner, cc, oracle


@pytest.fixture
def routes(monkeypatch):
    """Every submission this test makes: the zero-copy entry name or the host pipeline, and its width."""
    from triton_msl.backend.driver import _get_compile_shader_runtime, _get_utils

    seen = []

    def zero_copy(instance, original):
        def dispatch(lib, name, args, **kwargs):
            group = kwargs["group_size"]
            seen.append(("zero-copy", name, group if isinstance(group, int) else group[0]))
            return original(lib, name, args, **kwargs)

        return dispatch

    def host(instance, original):
        def launch(pipeline, grid, group, buffers):
            seen.append(("host", pipeline, group[0]))
            return original(pipeline, grid, group, buffers)

        return launch

    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime, "dispatch", zero_copy)
    patch_live_singleton_method(monkeypatch, _get_utils, "launch", host)
    return seen


def _launch(
    fn, layout, m, n, k, bm, bn, bk, masked, device, routes, integers=True, dtype=torch.float32, grid=None, slack=0
):
    (a, w), (aa, ww, bb), owner, cc, oracle = _operands(layout, m, n, k, bm, bn, bk, device, integers, dtype, slack)
    grid = grid or (-(-m // bm) * -(-n // bn),)
    handle = None
    for poison in (-8192.0, 8192.0):
        cc.fill_(poison)
        handle = fn[grid](
            aa, bb, cc, m, n, k, *aa.stride(), *bb.stride(), *cc.stride(), BM=bm, BN=bn, BK=bk, MASKED=masked
        )
        if device == "mps":
            torch.mps.synchronize()
        got, beyond = cc.cpu()[:m], cc.cpu()[m:]
        assert bool((beyond == poison).all())
        if integers:
            assert torch.equal(got, oracle)
        else:
            assert torch.allclose(got, oracle, rtol=1e-4, atol=1e-3)
        after = owner.cpu()
        assert bool((after[:4096] == -16384).all() & (after[-4096:] == -16384).all())
        assert torch.equal(aa.cpu(), a) and torch.equal(ww.cpu(), w)
    return handle


def _assert_route(handle, routes, twin_expected, width):
    from triton_msl.backend.driver import _PIPELINE_BUCKETS

    descriptor = handle.metadata.mm_two_kernel
    assert descriptor is not None and descriptor["direct_block"] == width
    assert len(routes) == 2
    for kind, target, group in routes:
        if kind == "zero-copy":
            is_twin = target == descriptor["direct_name"]
            assert is_twin or not target.endswith("__mmdirect")
        else:
            entry = _PIPELINE_BUCKETS[id(handle.function)].latest("direct")
            is_twin = entry is not None and target is entry.direct
        assert is_twin == twin_expected
        if is_twin:
            assert group == width


def _assert_no_twin(handle, routes):
    from triton_msl.backend.driver import _PIPELINE_BUCKETS

    for kind, target, _ in routes:
        if kind == "zero-copy":
            assert not target.endswith("__mmdirect")
        else:
            entry = _PIPELINE_BUCKETS[id(handle.function)].latest("direct")
            assert entry is None or target is not entry.direct


SHAPES = [
    # (rows, cols, depth) in tiles or elements, masked, twin expected
    (lambda bm, bn: (2 * bm, 3 * bn, 64), False, True),
    (lambda bm, bn: (bm, 2 * bn, 40), False, True),  # K is a multiple of 8, not of BLOCK_K
    (lambda bm, bn: (2 * bm, 2 * bn, 96), True, True),
    (lambda bm, bn: (2 * bm - 5, 2 * bn, 64), True, False),  # partial row tile: the ordinary entry
    (lambda bm, bn: (bm, 2 * bn - 9, 64), True, False),
    (lambda bm, bn: (bm, bn, 36), True, False),  # K is not a multiple of 8
]


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("shape", range(len(SHAPES)))
@pytest.mark.parametrize("block_m,block_n,width", TILES)
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_device_result_and_route(mm, routes, layout, block_m, block_n, width, shape, device):
    extents, masked, twin_expected = SHAPES[shape]
    m, n, k = extents(block_m, block_n)
    handle = _launch(mm, layout, m, n, k, block_m, block_n, 32, masked, device, routes)
    _assert_route(handle, routes, twin_expected, width)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("block_m,block_n,width", [(32, 32, 32), (64, 64, 128), (128, 128, 128)])
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_device_result_with_real_valued_operands(mm, routes, layout, block_m, block_n, width, device):
    handle = _launch(
        mm, layout, 2 * block_m, 2 * block_n, 256, block_m, block_n, 32, False, device, routes, integers=False
    )
    _assert_route(handle, routes, True, width)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("block_m,block_n,width", [(32, 32, 32), (64, 64, 128), (128, 128, 128)])
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_device_result_with_half_precision_operands(mm, routes, layout, block_m, block_n, width, device):
    """fp16 A and B, fp32 C: the contiguous twin runs on half fragments; a transposed B has no twin."""
    handle = _launch(
        mm, layout, 2 * block_m, 2 * block_n, 64, block_m, block_n, 32, False, device, routes, dtype=torch.float16
    )
    if layout == "contig":
        assert "simdgroup_half8x8" in _twin(handle.asm["msl"])
        _assert_route(handle, routes, True, width)
    else:
        assert handle.metadata.mm_two_kernel is None and "__mmdirect" not in handle.asm["msl"]
        assert len(routes) == 2


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("block_m,block_n,width", [(32, 32, 32), (32, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_twin_scalar_fallback_runs_and_matches(tmp_path, monkeypatch, routes, layout, block_m, block_n, width, device):
    """A twin whose proof is false takes its scalar fallback at the twin's own width."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    proof = GenericLowerer._simd_layout_guard_lines

    def refuted(self, *args, **kwargs):
        return [*proof(self, *args, **kwargs), "    _simd_ok = false;"]

    monkeypatch.setattr(GenericLowerer, "_simd_layout_guard_lines", refuted)
    fn = _kernel(tmp_path, f"refuted_{layout}_{block_m}_{block_n}_{device}")
    handle = _launch(fn, layout, 2 * block_m, 2 * block_n, 64, block_m, block_n, 32, False, device, routes)
    twin = _twin(handle.asm["msl"])
    assert "_simd_ok = false;" in twin
    _assert_route(handle, routes, True, width)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("surplus", [1, 5])
@pytest.mark.parametrize("block_m,block_n,width", [(32, 32, 32), (32, 64, 64), (64, 64, 128), (128, 128, 128)])
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("roomy", [False, True])
def test_surplus_programs_of_a_flat_grid_store_nothing(
    mm, routes, layout, block_m, block_n, width, surplus, roomy, device
):
    """A flat grid with more programs than tiles. The output is exact and every cell
    after it is intact (``_launch`` checks both). An output with room for the surplus
    tiles runs the twin, whose masked-extent test stops them; one without keeps the
    ordinary entry."""
    handle = _launch(
        mm, layout, 2 * block_m, 2 * block_n, 64, block_m, block_n, 32, True, device, routes,
        grid=(4 + surplus,), slack=3 * block_m * roomy,
    )  # fmt: skip
    _assert_route(handle, routes, roomy, width)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_surplus_programs_of_an_unmasked_store_keep_the_ordinary_entry(mm, routes, layout, device):
    """Unmasked, the twin would store the surplus tiles whole, past the output."""
    handle = _launch(mm, layout, 64, 64, 64, 32, 32, 32, False, device, routes, grid=(5,))
    _assert_route(handle, routes, False, 32)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("grid", [(2, 2), (2, 3), (3, 2), (4, 5)])
@pytest.mark.parametrize("block_m,block_n,width", [(32, 32, 32), (64, 64, 128), (128, 64, 128)])
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("roomy", [False, True])
def test_two_axis_grid_exact_and_oversized(mm_2d, routes, layout, block_m, block_n, width, grid, roomy, device):
    """One program id per axis. A surplus column of programs would land on the next
    output row, a surplus row past the output's end; neither may store anything."""
    handle = _launch(
        mm_2d, layout, 2 * block_m, 2 * block_n, 64, block_m, block_n, 32, True, device, routes,
        grid=grid, slack=8 * block_m * roomy,
    )  # fmt: skip
    if layout == "bt":  # the contiguous two-axis form is taken by the fast-matmul route
        _assert_route(handle, routes, roomy or grid == (2, 2), width)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("aliased", ["a", "b"])
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("tile", [32, 64, 128])
def test_single_program_output_aliasing_an_operand_keeps_the_ordinary_entry(mm, routes, layout, tile, aliased, device):
    """One program loads its whole tile before it stores, so ``C is A`` or
    ``C is B`` is well defined; the twin does not keep that order and is not chosen.
    The ordinary entries do: staged, or each thread holding its cells until the
    threadgroup has loaded."""
    gen = torch.Generator().manual_seed(31 + tile)
    a = torch.randint(-3, 4, (tile, tile), generator=gen).float()
    w = torch.randint(-3, 4, (tile, tile), generator=gen).float()
    oracle = (a.double() @ w.double().t()).float()
    for _ in range(3):
        aa, ww = a.clone().to(device), w.clone().to(device)
        bb = ww.t() if layout == "bt" else ww.t().contiguous()
        # C addresses its storage row-major, so the transposed layout's output is B's owner
        cc = aa if aliased == "a" else (ww if layout == "bt" else bb)
        handle = mm[(1,)](
            aa, bb, cc, tile, tile, tile, *aa.stride(), *bb.stride(), *cc.stride(), BM=tile, BN=tile, BK=32, MASKED=True
        )
        if device == "mps":
            torch.mps.synchronize()
        assert torch.equal(cc.cpu(), oracle)
    assert handle.metadata.mm_two_kernel is not None and len(routes) == 3
    _assert_no_twin(handle, routes)


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_operands_sharing_storage_with_each_other_still_take_the_twin(mm, routes, device):
    a = torch.randint(-3, 4, (64, 64), generator=torch.Generator().manual_seed(5)).float()
    aa = a.to(device)
    cc = torch.full((64, 64), -8192.0, device=device)
    for _ in range(2):
        handle = mm[(4,)](aa, aa.t(), cc, 64, 64, 64, 64, 1, 1, 64, 64, 1, BM=32, BN=32, BK=32, MASKED=False)
        if device == "mps":
            torch.mps.synchronize()
        assert torch.equal(cc.cpu(), (a.double() @ a.double().t()).float())
    _assert_route(handle, routes, True, 32)


ROLE_SOURCE = """
@triton.jit
def {name}(A, B, C, {rows}, {cols}, {depth}):
    pid = tl.program_id(0)
    ncols = tl.cdiv({cols}, 32)
    rm = (pid // ncols) * 32 + tl.arange(0, 32)
    rn = (pid % ncols) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for kk in range(0, {depth}, 32):
        x = tl.load(A + rm[:, None] * {depth} + (kk + rk)[None, :])
        y = tl.load(B + (kk + rk)[:, None] * {cols} + rn[None, :])
        acc += tl.dot(x, y)
    tl.store(C + rm[:, None] * {cols} + rn[None, :], acc)
"""


@requires_gpu
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("shape", [(64, 64, 64), (64, 32, 32), (64, 64, 32), (96, 64, 32), (64, 96, 128)])
@pytest.mark.parametrize("names", [("M", "N", "K"), ("K", "N", "M")])
def test_unmasked_store_is_not_clipped_at_an_argument_named_for_another_role(tmp_path, names, shape, device):
    """Nothing in an unmasked source says which argument is the row extent. With the
    names M and K exchanged, a twin clipping rows at ``M`` would stop at the depth."""
    rows, cols, depth = shape
    name = "roles_" + "".join(names).lower()
    fn = _kernel(
        tmp_path, name, ROLE_SOURCE.replace("{name}", "{{name}}").format(**dict(zip(("rows", "cols", "depth"), names)))
    )
    gen = torch.Generator().manual_seed(rows + 3 * cols + 5 * depth)
    a = torch.randint(-3, 4, (rows, depth), generator=gen).float()
    b = torch.randint(-3, 4, (depth, cols), generator=gen).float()
    owner = torch.full((rows * cols + 8192,), -16384.0, device=device)
    cc = owner[4096:-4096].view(rows, cols)
    fn[((rows // 32) * (cols // 32),)](a.to(device), b.to(device), cc, rows, cols, depth)
    if device == "mps":
        torch.mps.synchronize()
    assert torch.equal(cc.cpu(), (a.double() @ b.double()).float())
    after = owner.cpu()
    assert bool((after[:4096] == -16384).all() & (after[-4096:] == -16384).all())
