"""No-wrap launch proof for source load moduli on unmasked matmul output axes."""

from pathlib import Path

import pytest
import torch
import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget

from tests.test_matmul_source_extents import SOURCE, _kernel, _compile
from tests.test_matmul_subtiled_twin import routes, _assert_no_twin  # noqa: F401
from triton_msl.backend import driver, _generic_launch, _launch_signature
from triton_msl.backend._matmul_domain import matmul_grid_rectangles, validate_matmul_no_wrap_moduli
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError

FIXTURE = Path(__file__).parent / "fixtures/upstream_tl_range_num_stages_unmasked_modulo.ttgir"
MODULO_SOURCE = SOURCE.replace("a + rm[:, None]", "a + (rm % rows)[:, None]")
MODULO_SOURCE = MODULO_SOURCE.replace("rn[None, :] * sbn", "(rn % cols)[None, :] * sbn")


def _retained_lowerer():
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(FIXTURE), context)
    module._retained_context = context
    return GenericLowerer(walk_ttgir(module, options), options)


def test_actual_upstream_range_num_stages_ir_has_an_explicit_no_wrap_plan():
    lowerer = _retained_lowerer()
    msl = lowerer.lower()
    desc = lowerer._tail_access_bounds
    assert desc[0] == "matmul_grid_storage_v2"
    assert desc[3] == (None, None, ("arg", 5, "i32"))
    assert desc[6] == (
        64,
        64,
        "1d",
        (True, True),
        (("arg", 3, "i32"), None),
        (None, None),
        True,
        (("arg", 3, "i32"), ("arg", 4, "i32")),
    )
    # No store/load clip is invented from M or N. The runtime witness proves
    # that the source remainder is identity on every coordinate we store.
    assert "long _M = (long)((long)row_base + 64L);" in msl
    assert "long _N = (long)((long)col_base + 64L);" in msl
    assert "long _A_M = (long)((long)row_base + 64L);" in msl
    assert "long _B_N = (long)((long)col_base + 64L);" in msl
    args = [
        torch.empty(64, 512, dtype=torch.float16),
        torch.empty(512, 64, dtype=torch.float16),
        torch.empty(64, 64),
        64,
        64,
        512,
        512,
        64,
        64,
    ]
    assert driver._matmul_tail_bounds_reason(desc, args, grid=(1, 1, 1)) is None


@pytest.mark.parametrize("store", [0, 1, 2])
@pytest.mark.parametrize("mapping", [0, 1, 2])
@pytest.mark.parametrize("layout", ["contig", "bt"])
def test_source_masks_and_no_wrap_obligations_have_separate_provenance(tmp_path, store, mapping, layout):
    msl, metadata = _compile(_kernel(tmp_path, source=MODULO_SOURCE), store=store, mapping=mapping, layout=layout)
    desc = metadata["batched_dot_bounds"]
    if store == 1:
        assert desc[0] == "matmul_full_k_storage_v1"
        assert "long _A_M = (long)(_mmarg3);" in msl
        assert "long _B_N = (long)(_mmarg4);" in msl
    else:
        assert desc[0] == "matmul_grid_storage_v2"
        assert desc[6][7] == ((("arg", 3, "i32") if store == 0 else None), ("arg", 4, "i32"))
        assert "long _B_N = (long)((long)col_base + 64L);" in msl
        if store == 0:
            assert "long _A_M = (long)((long)row_base + 128L);" in msl


def _value(ref):
    return ref[1]


@pytest.mark.parametrize(
    "mapping,grid,moduli,store",
    [
        ("2d", (2, 3, 1), (64, 96), (64, 96)),
        ("1d", (7, 1, 1), (64, 128), (64, 128)),
        (("flat_n", 32), (7, 1, 1), (96, 96), (96, 96)),
        (("grouped", 4, 32), (7, 1, 1), (64, 128), (64, 128)),
        # Oversized launch with one masked output axis: the unmasked axis still
        # requires its own no-wrap witness on every actual stored coordinate.
        ("2d", (3, 4, 1), (64, 128), (64, 128)),
        ("2d", (3, 4, 1), (96, 96), (96, 96)),
    ],
)
def test_no_wrap_witness_matches_exact_store_rectangle_domains(mapping, grid, moduli, store):
    lit = lambda v: ("literal", v, "i32")
    plan = (32, 32, mapping, (True, True), (lit(64), lit(96)), (None, None), True)
    rectangles = matmul_grid_rectangles(plan, grid, _value)
    clipped = tuple((m0, min(m1, store[0]), n0, min(n1, store[1])) for m0, m1, n0, n1 in rectangles)
    validate_matmul_no_wrap_moduli(tuple(map(lit, moduli)), clipped, _value)
    for axis in (0, 1):
        smaller = list(moduli)
        smaller[axis] = min(moduli[axis], max((r[1] if axis == 0 else r[3]) for r in clipped)) - 1
        with pytest.raises(ValueError, match="wrap"):
            validate_matmul_no_wrap_moduli(tuple(map(lit, smaller)), clipped, _value)


@pytest.mark.parametrize("modulus", [0, -1, -64])
def test_no_wrap_requires_positive_modulus_even_for_empty_output(modulus):
    refs = (("literal", modulus, "i32"), None)
    with pytest.raises(ValueError, match="positive"):
        validate_matmul_no_wrap_moduli(refs, ((64, 32, 0, 0),), _value)
    validate_matmul_no_wrap_moduli((("literal", 1, "i32"), None), ((64, 32, 0, 0),), _value)


def test_no_wrap_captured_payload_and_full_storage_remain_required():
    lowerer = _retained_lowerer()
    lowerer.lower()
    desc = lowerer._tail_access_bounds
    args = [
        torch.empty(64, 512, dtype=torch.float16),
        torch.empty(512, 64, dtype=torch.float16),
        torch.empty(64, 64),
        64,
        64,
        512,
        512,
        64,
        64,
    ]
    sigs = ["*fp16", "*fp16", "*fp32"] + ["i32"] * 6
    payloads = [None] * 3 + [_launch_signature.scalar_bytes(value, "i32") for value in args[3:]]
    args[4] = 65
    assert "changed" in driver._matmul_tail_bounds_reason(desc, args, sigs, payloads, grid=(1, 1, 1))
    args[4] = 64
    args[0] = torch.empty(1, dtype=torch.float16)
    assert "outside" in driver._matmul_tail_bounds_reason(desc, args, sigs, payloads, grid=(1, 1, 1))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("modulus", [0, -1, 32, 63, 1 << 31])
def test_no_wrap_identity_cannot_be_inferred_from_roomy_backing(axis, modulus):
    lowerer = _retained_lowerer()
    lowerer.lower()
    desc = lowerer._tail_access_bounds
    args = [
        torch.empty(64, 512, dtype=torch.float16),
        torch.empty(512, 128, dtype=torch.float16),
        torch.empty(128, 128),
        64,
        64,
        512,
        512,
        128,
        128,
    ]
    args[3 + axis] = modulus
    assert driver._matmul_tail_bounds_reason(desc, args, grid=(1, 1, 1)) is not None


requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


@requires_gpu
@pytest.mark.parametrize("layout", ["contig", "bt"])
@pytest.mark.parametrize("mapping", [0, 1, 2])
@pytest.mark.parametrize("store", [0, 2])
@pytest.mark.parametrize("ordinary_only", [False, True])
def test_unmasked_modulo_no_wrap_device_matches_source(
    tmp_path, monkeypatch, routes, layout, mapping, store, ordinary_only
):
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    if ordinary_only:
        monkeypatch.setattr(_generic_launch, "direct_twin", lambda *args: None)
    fn = _kernel(tmp_path, name="no_wrap_modulo_mm", source=MODULO_SOURCE)
    generator = torch.Generator().manual_seed(6157001)
    ah, bh = torch.randn(128, 64, generator=generator), torch.randn(64, 128, generator=generator)
    a = ah.to("mps")
    b = (bh if layout == "contig" else bh.t().contiguous().t()).to("mps")
    owner = torch.full((128 * 128 + 64,), -17391.0, device="mps")
    c = owner[32:-32].view(128, 128)
    grid = (2, 2) if mapping == 0 else (4,)
    handle = fn[grid](
        a,
        b,
        c,
        128,
        128,
        64,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        a.stride(1),
        b.stride(1),
        c.stride(1),
        64,
        64,
        32,
        store,
        False,
        mapping,
    )
    assert handle.metadata.batched_dot_bounds[0] == "matmul_grid_storage_v2"
    assert len(routes) == 1
    if ordinary_only:
        _assert_no_twin(handle, routes)
    expected = (ah.double() @ bh.double()).float()
    torch.testing.assert_close(c.cpu(), expected, atol=3e-3, rtol=3e-3)
    assert torch.equal(a.cpu(), ah) and torch.equal(b.cpu(), bh)
    assert torch.all(owner[:32].cpu() == -17391.0) and torch.all(owner[-32:].cpu() == -17391.0)


@requires_gpu
@pytest.mark.parametrize("rows", [0, -1, 32, 63])
def test_wrapping_unmasked_source_declines_before_any_submission(tmp_path, monkeypatch, rows):
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.backend.driver import MetalUtils, _get_utils, _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    fn = _kernel(tmp_path, name="wrapping_modulo_mm", source=MODULO_SOURCE)
    a, b, c = (
        torch.ones(128, 64, device="mps"),
        torch.ones(64, 64, device="mps"),
        torch.full((128, 64), -17391.0, device="mps"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unproved source modulo reached submission")

    monkeypatch.setattr(CompileShaderRuntime, "dispatch", forbidden)
    monkeypatch.setattr(MetalUtils, "launch", forbidden)
    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime, "dispatch", lambda *args: forbidden)
    patch_live_singleton_method(monkeypatch, _get_utils, "launch", lambda *args: forbidden)
    with pytest.raises(MetalNonRecoverableError):
        fn[(2, 1)](
            a,
            b,
            c,
            rows,
            64,
            64,
            *[a.stride(0), b.stride(0), c.stride(0)],
            a.stride(1),
            b.stride(1),
            c.stride(1),
            64,
            64,
            32,
            0,
            False,
            0,
        )
    assert torch.all(c.cpu() == -17391.0)


@pytest.mark.parametrize(
    "index,cast",
    [
        ("rm", "rows.to(tl.int64)"),
        ("rm.to(tl.uint64)", "rows.to(tl.uint64)"),
        ("rm", "rows.to(tl.int16).to(tl.int32)"),
        ("rm.to(tl.uint32)", "rows.to(tl.int16).to(tl.uint32)"),
    ],
)
def test_no_wrap_does_not_strip_a_source_modulus_cast(tmp_path, index, cast):
    source = MODULO_SOURCE.replace("rm % rows", index + " % " + cast)
    with pytest.raises(MetalNonRecoverableError):
        _compile(_kernel(tmp_path, name="casted_modulus_mm", source=source))


def test_actual_i64_to_i32_modulus_truncation_is_not_a_runtime_i64_bound(tmp_path):
    # Exercise the actual native IR cast while retaining all source grid/index
    # operands from the upstream fixture; no Python-to-IR scalar specialization.
    text = FIXTURE.read_text()
    text = text.replace("%M: i32 {tt.divisibility = 16 : i32}", "%M: i64 {tt.divisibility = 16 : i32}")
    # Replace all uses with the actual truncated value, then keep the sole
    # original argument as the input of that truncation.
    import re

    head, body = text.split("attributes {noinline = false} {", 1)
    body = re.sub(r"%M\b", "%M_trunc", body)
    text = head + "attributes {noinline = false} {\n    %M_trunc = arith.trunci %M : i64 to i32\n" + body
    path = tmp_path / "truncated_modulus.ttgir"
    path.write_text(text)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    with pytest.raises(MetalNonRecoverableError):
        lowerer.lower()


@pytest.mark.parametrize(
    "payload", [None, (), (None,), (None, None), (("literal", 64, "i32"),), "not-an-obligation-pair"]
)
def test_malformed_v2_no_wrap_payload_fails_closed(payload):
    lowerer = _retained_lowerer()
    lowerer.lower()
    desc = lowerer._tail_access_bounds
    desc = (*desc[:6], (*desc[6][:7], payload))
    args = [
        torch.empty(64, 512, dtype=torch.float16),
        torch.empty(512, 64, dtype=torch.float16),
        torch.empty(64, 64),
        64,
        64,
        512,
        512,
        64,
        64,
    ]
    assert driver._matmul_tail_bounds_reason(desc, args, grid=(1, 1, 1)) is not None


@pytest.mark.parametrize("captured", [False, True])
def test_no_wrap_uses_packed_scalar_bytes_instead_of_overridden_int(captured):
    class DifferentInt(int):
        def __int__(self):
            return 128

    lowerer = _retained_lowerer()
    lowerer.lower()
    desc = lowerer._tail_access_bounds
    # M64 keeps the grid valid; the source B modulus is32, although __int__
    # would claim128. Roomy allocations cannot turn that source wrap into identity.
    args = [
        torch.empty(64, 512, dtype=torch.float16),
        torch.empty(512, 128, dtype=torch.float16),
        torch.empty(64, 128),
        64,
        DifferentInt(32),
        512,
        512,
        128,
        128,
    ]
    if captured:
        sigs = ["*fp16", "*fp16", "*fp32"] + ["i32"] * 6
        payloads = [None] * 3 + [_launch_signature.scalar_bytes(v, "i32") for v in args[3:]]
        reason = driver._matmul_tail_bounds_reason(desc, args, sigs, payloads, grid=(1, 1, 1))
    else:
        reason = driver._matmul_tail_bounds_reason(desc, args, grid=(1, 1, 1))
    assert "wrap" in reason
