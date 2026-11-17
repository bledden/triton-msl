"""CPU proofs for the unmasked, full-source-block K-tail launch boundary."""

import pytest
import torch

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir
import triton
import triton.language as tl

from triton_msl.backend import driver, _cache_contract, _launch_contract, _launch_signature
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.codegen.msl_emitter import emit_msl
from triton_msl.errors import MetalNonRecoverableError


def _desc(strides=None):
    refs = (("arg", 3, "i32"), ("arg", 4, "i32"), ("arg", 5, "i32"))
    strides = strides or (
        ("arg", 5, "i32"),
        ("literal", 1, "i64"),
        ("arg", 4, "i32"),
        ("literal", 1, "i64"),
        ("arg", 4, "i32"),
        ("literal", 1, "i64"),
    )
    arithmetic = (("i32", "i32", False),) * 3
    return ("matmul_full_k_storage_v1", 32, (0, 1, 2), refs, strides, arithmetic)


def test_exact_k_storage_refuses_but_padded_backing_accepts():
    exact = [torch.empty(66), torch.empty(66), torch.empty(4), 2, 2, 33]
    reason = driver._matmul_tail_bounds_reason(_desc(), exact)
    assert "outside" in reason and "mask the K tail" in reason
    padded = [torch.empty(128), torch.empty(128), torch.empty(4), 2, 2, 33]
    assert driver._matmul_tail_bounds_reason(_desc(), padded) is None


def test_negative_stride_and_storage_offset_are_proved_by_interval():
    a_base = torch.empty(128)
    a = a_base[64:]
    strides = (
        ("literal", -64, "i64"),
        ("literal", 1, "i64"),
        ("arg", 4, "i32"),
        ("literal", 1, "i64"),
        ("arg", 4, "i32"),
        ("literal", 1, "i64"),
    )
    args = [a, torch.empty(128), torch.empty(4), 2, 2, 33]
    assert driver._matmul_tail_bounds_reason(_desc(strides), args) is None


def test_pointer_roles_need_not_match_argument_order():
    # Runtime order is C,A,B while the immutable role map remains A,B,C.
    desc = ("matmul_full_k_storage_v1", 32, (1, 2, 0), _desc()[3], _desc()[4], _desc()[5])
    args = [torch.empty(4), torch.empty(128), torch.empty(128), 2, 2, 33]
    assert driver._matmul_tail_bounds_reason(desc, args) is None


def test_aligned_k_keeps_ordinary_launch_without_storage_introspection():
    assert driver._matmul_tail_bounds_reason(_desc(), [object(), object(), object(), 2, 2, 64]) is None


def test_aligned_k_does_not_observe_m_n_or_stride_scalars():
    class NoInt:
        def __int__(self):
            raise AssertionError("aligned tail guard re-observed an unrelated scalar")

    refs = (("arg", 3, "i32"), ("arg", 4, "i32"), ("arg", 5, "i32"))
    strides = (("arg", 6, "i32"),) * 6
    desc = ("matmul_full_k_storage_v1", 32, (0, 1, 2), refs, strides, _desc()[5])
    assert (
        driver._matmul_tail_bounds_reason(desc, [object(), object(), object(), NoInt(), NoInt(), 64] + [NoInt()] * 6)
        is None
    )


def test_source_width_and_address_overflow_refuse():
    args = [torch.empty(128), torch.empty(128), torch.empty(4), 2, 2, 1 << 31]
    assert "could not be proven" in driver._matmul_tail_bounds_reason(_desc(), args)
    huge = (("literal", 1 << 62, "i64"), ("literal", 1, "i64")) * 3
    args[3], args[-1] = 3, 33
    assert "wrap" in driver._matmul_tail_bounds_reason(_desc(huge), args)


def test_guard_scalars_must_match_captured_launch_payload():
    args = [torch.empty(128), torch.empty(128), torch.empty(4), 2, 2, 33]
    sigs = ["*fp32", "*fp32", "*fp32", "i32", "i32", "i32"]
    payloads = [None, None, None] + [_launch_signature.scalar_bytes(value, "i32") for value in args[3:]]
    args[5] = 34  # models a mutable scalar provider changed by the enter hook
    assert "could not be proven" in driver._matmul_tail_bounds_reason(_desc(), args, sigs, payloads)


@triton.jit
def _unmasked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap, bp = A + rm[:, None] * K + rk[None, :], B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK
        bp += BK * N
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _masked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        av = tl.load(ap, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0.0)
        bv = tl.load(bp, mask=(k + rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
        ap += BK
        bp += BK * N
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


def test_native_ir_producer_binds_roles_extents_strides_and_block():
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    ctx = ir.context()
    ir.load_dialects(ctx)
    sig = {
        "A": "*fp32",
        "B": "*fp32",
        "C": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "BM": "constexpr",
        "BN": "constexpr",
        "BK": "constexpr",
    }
    module = ASTSource(_unmasked, sig, {"BM": 32, "BN": 32, "BK": 32}).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    lowerer.lower()
    # Store clips do not mask the source's complete operand tiles.
    expected = ("matmul_grid_storage_v1", *_desc()[1:], (32, 32, "2d", (True, True), (None, None), (None, None), True))
    assert lowerer._tail_access_bounds == expected
    metadata = {"name": "tail_contract"}
    emit_msl(module, metadata, options)
    assert metadata["batched_dot_bounds"] == expected

    masked = ASTSource(_masked, sig, {"BM": 32, "BN": 32, "BK": 32}).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx
    )
    masked = backend.make_ttir(masked, {}, options)
    masked = backend.make_ttgir(masked, {}, options)
    masked_metadata = {"name": "masked_tail"}
    emit_msl(masked, masked_metadata, options)
    assert masked_metadata["batched_dot_bounds"] is None


def test_enter_hook_mutation_is_checked_before_runtime(monkeypatch):
    events = []
    flat = [torch.empty(128), torch.empty(128), torch.empty(4), 2, 2, 33]
    packed = (4, 1, 0, 128, None, False, None, None, None, None, _desc(), None)
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher._execution_contract = "checked"
    launcher._packed_contract = "checked"
    launcher._binding_plan = None
    launcher.arg_names = []
    launcher.signature = {}
    sigs = ["*fp32", "*fp32", "*fp32", "i32", "i32", "i32"]
    payloads = [None, None, None] + [_launch_signature.scalar_bytes(value, "i32") for value in flat[3:]]
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda value: value)
    monkeypatch.setattr(_launch_contract, "validate_packed_launch", lambda live, expected: tuple(live))
    monkeypatch.setattr(_launch_signature, "bind_arguments", lambda *a: (flat, sigs, [], payloads))
    monkeypatch.setattr(driver, "_get_utils", lambda: events.append("runtime"))

    def hook(_):
        events.append("hook")
        flat[0] = torch.empty(66)

    with pytest.raises(MetalNonRecoverableError, match="mask the K tail"):
        launcher(1, 1, 1, None, None, packed, None, hook, None)
    assert events == ["hook"]


@pytest.mark.parametrize(
    "address_bounds",
    [
        ("unknown_address_contract_v1",),
        ("matmul_full_k_storage_v1",),
        [],
        {"tag": "matmul_full_k_storage_v1"},
    ],
)
def test_sealed_unknown_or_malformed_address_contract_refuses_before_runtime(address_bounds, monkeypatch):
    metadata = dict(
        name="sealed_bad_bounds",
        execution_contract="checked",
        num_warps=4,
        num_ctas=1,
        shared=0,
        block_size=128,
        output_arg_indices=None,
        needs_2d_grid=False,
        mm_two_kernel=None,
        fast_matmul=None,
        quant_matmul=None,
        flash_attention=None,
        batched_dot_bounds=address_bounds,
        device_assert=None,
    )
    _launch_contract.seal_launch_metadata(metadata)
    packed = _launch_contract.pack_launch_metadata(metadata)
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher._execution_contract = "checked"
    launcher._packed_contract = _launch_contract.validate_launch_metadata(metadata)
    launcher._binding_plan = None
    launcher.arg_names, launcher.signature = [], {}
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda value: value)
    monkeypatch.setattr(_launch_signature, "bind_arguments", lambda *a: ([], [], [], []))
    events = []
    monkeypatch.setattr(driver, "_get_utils", lambda: events.append("runtime"))
    with pytest.raises(MetalNonRecoverableError, match="Refusing"):
        launcher(1, 1, 1, None, None, packed, None, None, None)
    assert events == []


def test_both_valid_address_contract_tags_reach_their_proofs(monkeypatch):
    tail_args = [torch.empty(128), torch.empty(128), torch.empty(4), 2, 2, 33]
    assert driver._matmul_tail_bounds_reason(_desc(), tail_args) is None
    batched = (
        "batched_dot_host_bounds_v1",
        1,
        2,
        2,
        2,
        False,
        0,
        1,
        2,
        (("literal", 4), ("literal", 2), ("literal", 1)),
        (("literal", 4), ("literal", 2), ("literal", 1)),
        (("literal", 4), ("literal", 2), ("literal", 1)),
    )
    assert (
        driver._batched_dot_host_bounds_reason(batched, [torch.empty(4), torch.empty(4), torch.empty(4)], (1, 1, 1))
        is None
    )

    def launch_to_proof(address_bounds, args, sigs, payloads, expected, rejected, monkeypatch):
        metadata = dict(
            name="sealed_valid_bounds",
            execution_contract="checked",
            num_warps=4,
            num_ctas=1,
            shared=0,
            block_size=128,
            output_arg_indices=None,
            needs_2d_grid=False,
            mm_two_kernel=None,
            fast_matmul=None,
            quant_matmul=None,
            flash_attention=None,
            batched_dot_bounds=address_bounds,
            device_assert=None,
        )
        _launch_contract.seal_launch_metadata(metadata)
        packed = _launch_contract.pack_launch_metadata(metadata)
        launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
        launcher._execution_contract = "checked"
        launcher._packed_contract = _launch_contract.validate_launch_metadata(metadata)
        launcher._binding_plan = None
        launcher.arg_names, launcher.signature = [], {}
        monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda value: value)
        monkeypatch.setattr(_launch_signature, "bind_arguments", lambda *a: (args, sigs, [], payloads))
        calls = []
        monkeypatch.setattr(driver, expected, lambda *a: calls.append(expected) or "planned refusal")
        monkeypatch.setattr(driver, rejected, lambda *a: (_ for _ in ()).throw(AssertionError(rejected)))
        monkeypatch.setattr(driver, "_get_utils", lambda: (_ for _ in ()).throw(AssertionError("runtime")))
        with pytest.raises(MetalNonRecoverableError, match="planned refusal"):
            launcher(1, 1, 1, None, None, packed, None, None, None)
        assert calls == [expected]

    tail_sigs = ["*fp32", "*fp32", "*fp32", "i32", "i32", "i32"]
    tail_payloads = [None, None, None] + [_launch_signature.scalar_bytes(value, "i32") for value in tail_args[3:]]
    launch_to_proof(
        _desc(),
        tail_args,
        tail_sigs,
        tail_payloads,
        "_matmul_tail_bounds_reason",
        "_batched_dot_host_bounds_reason",
        monkeypatch,
    )
    launch_to_proof(
        batched,
        [torch.empty(4)] * 3,
        ["*fp32"] * 3,
        [None] * 3,
        "_batched_dot_host_bounds_reason",
        "_matmul_tail_bounds_reason",
        monkeypatch,
    )
