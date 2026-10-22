"""Operation-owned `ttg.memdesc_trans` permutations are exact-or-refuse."""

from tests.source_identity import assert_candidate_source

from pathlib import Path

import pytest
import triton
import triton.language as tl
import triton_msl

from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import SSAValue
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _dot_with_transpose(A, B, C, N: tl.constexpr):
    row = tl.arange(0, N)
    k = tl.arange(0, N)
    a = tl.load(A + row[:, None] * N + k[None, :])
    b = tl.load(B + row[:, None] * N + k[None, :])
    tl.store(C + row[:, None] * N + k[None, :], tl.dot(a, tl.trans(b)))


def _ttgir():
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend

    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(
        _dot_with_transpose,
        {"A": "*fp32", "B": "*fp32", "C": "*fp32", "N": "constexpr"},
        {"N": 32},
    )
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    return str(backend.make_ttgir(module, {}, options))


def _emit_direct_ir(text, path):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import IRSource
    from triton_msl.backend.compiler import MetalBackend

    path.write_text(text)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    context = ir.context()
    parsed = IRSource(str(path), context, backend)
    assert parsed.module.verify()
    return backend.make_msl(parsed.module, {}, backend.parse_options({"num_warps": 4}))


def _direct_lowerer(text, path):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import IRSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir

    path.write_text(text)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    parsed = IRSource(str(path), context, backend)
    assert parsed.module.verify()
    return GenericLowerer(walk_ttgir(parsed.module, options), options)


def _lowerer(shape=(32, 32), transposed=False):
    root = Path(__file__).resolve().parents[1]
    assert_candidate_source(root, triton_msl)
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.env = {1: "shared_a"}
    lowerer.env_types = {1: "float"}
    lowerer.env_shapes = {1: shape}
    lowerer.env_is_mask = set()
    lowerer.env_is_ptr = {}
    lowerer.env_ptr_array = {}
    lowerer._bcast_layout = {}
    lowerer._is_splat = set()
    lowerer.env_array = {}
    lowerer.env_n_elems = {}
    lowerer._shared_mem_descs = {1: ("shared_a", shape, "float")}
    lowerer._shared_mem_orientations = {1: transposed}
    return lowerer


def _op(order, result=2, source=1):
    return SSAValue(result, f"v{result}", "ttg.memdesc_trans", [source], {"order": order}, "memdesc", "f32", False)


def _descriptor_chain(canonical, orders, sibling_order=None):
    first = next(line for line in canonical.splitlines() if "ttg.memdesc_trans" in line)
    load = next(line for line in canonical.splitlines() if "ttg.local_load %1" in line)
    lines = []
    source, layout = "%0", "#shared"
    for index, order in enumerate(orders):
        result = f"%chain{index}"
        swapped = list(order) == [1, 0]
        next_layout = "#shared1" if (layout == "#shared") == swapped else "#shared"
        line = first.replace("%1 =", f"{result} =").replace("%0 ", f"{source} ")
        line = line.replace(
            "{order = array<i32: 1, 0>}",
            f"{{order = array<i32: {order[0]}, {order[1]}>}}",
        )
        line = line.replace(
            "#shared, #smem> -> !ttg.memdesc<32x32xf32, #shared1,",
            f"{layout}, #smem> -> !ttg.memdesc<32x32xf32, {next_layout},",
        )
        lines.append(line)
        source, layout = result, next_layout
    if sibling_order is not None:
        sibling_layout = "#shared1" if list(sibling_order) == [1, 0] else "#shared"
        sibling = (
            first.replace("%1 =", "%sibling =")
            .replace(
                "{order = array<i32: 1, 0>}",
                f"{{order = array<i32: {sibling_order[0]}, {sibling_order[1]}>}}",
            )
            .replace("#shared1,", f"{sibling_layout},")
        )
        lines.append(sibling)
    text = canonical.replace(first, "\n".join(lines))
    return text.replace(load, load.replace("%1", source).replace("#shared1", layout))


def test_identity_preserves_descriptor_alias_shape_and_existing_view_state():
    lowerer = _lowerer(transposed=True)
    lowerer._lower_memdesc_trans(_op([0, 1]))
    assert lowerer.env[2] == "shared_a"
    assert lowerer.env_shapes[2] == (32, 32)
    assert lowerer._shared_mem_descs[2] is lowerer._shared_mem_descs[1]
    assert lowerer._shared_mem_orientations[2] is True


def test_inner_swap_retains_existing_descriptor_behavior():
    lowerer = _lowerer((2, 4, 8))
    lowerer._lower_memdesc_trans(_op([0, 2, 1]))
    assert lowerer.env_shapes[2] == (2, 8, 4)
    assert lowerer._shared_mem_descs[2] == ("shared_a", (2, 8, 4), "float")
    assert lowerer._shared_mem_orientations[2] is True


def test_descriptor_orientations_compose_per_ssa_and_allow_sibling_views():
    lowerer = _lowerer()
    lowerer._lower_memdesc_trans(_op([1, 0], 2, 1))
    lowerer._lower_memdesc_trans(_op([0, 1], 3, 2))
    lowerer._lower_memdesc_trans(_op([1, 0], 4, 2))
    assert lowerer._shared_mem_orientations == {1: False, 2: True, 3: True, 4: False}
    assert lowerer.env[3] == lowerer.env[4] == "shared_a"


def test_descriptor_with_missing_incoming_orientation_refuses_before_mutation():
    lowerer = _lowerer()
    lowerer._shared_mem_orientations.clear()
    with pytest.raises(MetalNonRecoverableError, match="no proved descriptor orientation"):
        lowerer._lower_memdesc_trans(_op([0, 1]))
    assert 2 not in lowerer.env


@pytest.mark.parametrize("order", [None, [0, 0], [1, 0, 2], [1, 2, 0]])
def test_unknown_malformed_or_noninner_permutation_refuses(order):
    lowerer = _lowerer((2, 4, 8) if isinstance(order, list) and len(order) == 3 else (32, 32))
    with pytest.raises(MetalNonRecoverableError, match="operation-owned|only identity"):
        lowerer._lower_memdesc_trans(_op(order))


def test_full_backend_identity_and_swap_select_distinct_addressing(tmp_path, monkeypatch):
    canonical = _ttgir()
    assert canonical.count("ttg.memdesc_trans") == 1
    identity = canonical.replace("{order = array<i32: 1, 0>}", "{order = array<i32: 0, 1>}", 1)
    line = next(line for line in identity.splitlines() if "ttg.memdesc_trans" in line)
    identity = identity.replace(line, line.replace("#shared1", "#shared"))
    identity = identity.replace(
        "ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared1,",
        "ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared,",
    )

    # Separate caches are essential: this test is about the actual backend
    # route, not reuse of a prior variant in the same persistent cache.
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "swap-cache"))
    swapped_msl = _emit_direct_ir(canonical, tmp_path / "swap.ttgir")
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "identity-cache"))
    identity_msl = _emit_direct_ir(identity, tmp_path / "identity.ttgir")
    assert "gc * K + gr" in swapped_msl
    assert "gr * N + gc" in identity_msl
    assert swapped_msl != identity_msl


@pytest.mark.parametrize(
    ("orders", "sibling", "expected"),
    [
        (([1, 0], [0, 1]), None, "gc * K + gr"),
        (([1, 0], [1, 0]), None, "gr * N + gc"),
        (([1, 0], [0, 1], [1, 0], [1, 0]), None, "gc * K + gr"),
        (([1, 0],), [0, 1], "gc * K + gr"),
        (([0, 1],), [1, 0], "gr * N + gc"),
    ],
)
def test_full_backend_composes_deep_and_sibling_descriptor_views(tmp_path, monkeypatch, orders, sibling, expected):
    text = _descriptor_chain(_ttgir(), orders, sibling)
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "cache"))
    msl = _emit_direct_ir(text, tmp_path / "chain.ttgir")
    assert expected in msl


def test_full_backend_refuses_descriptor_chain_beyond_proof_bound(tmp_path, monkeypatch):
    text = _descriptor_chain(_ttgir(), [[0, 1]] * 65)
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(MetalNonRecoverableError, match="64-operation proof bound"):
        _emit_direct_ir(text, tmp_path / "too-deep.ttgir")


def test_cycle_in_descriptor_metadata_refuses_without_recursion(tmp_path):
    lowerer = _direct_lowerer(_ttgir(), tmp_path / "cycle.ttgir")
    trans = next(op for op in lowerer.graph.ops if op.op == "ttg.memdesc_trans")
    trans.operand_ids[:] = [trans.id]
    with pytest.raises(MetalNonRecoverableError, match="contains a cycle"):
        lowerer._detect_simple_dot()
