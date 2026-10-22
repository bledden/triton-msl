"""Source/binary binding for host packed arguments. CPU only, no shader execution."""

import copy
import hashlib
import json
import struct
import sys
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl

from triton_msl.backend import _argument_abi as abi, _launch_contract as contract
from triton_msl.codegen.msl_emitter import KernelBuilder
from triton_msl.errors import MetalNonRecoverableError


def builder(count=32, scalar="i32"):
    kb = KernelBuilder("abi_test", 128)
    for i in range(count):
        if i in (0, 2, 5, 9, 13, 17, 21):
            kb.add_ptr_arg(f"a{i}", "fp32")
        else:
            kb.add_scalar_arg(f"a{i}", scalar)
    return kb


def complete(count=32):
    kb = builder(count)
    source = kb.build()
    metadata = {"argument_abi": kb._argument_abi, "binary_route": "msl"}
    abi.finish_record(metadata, source, b"compiled-library")
    return metadata["argument_abi"]


def metadata(record=None):
    data = dict(name="abi_test", execution_contract="controlled", num_warps=4, num_ctas=1)
    if record is not None:
        data["argument_abi"] = record
    contract.seal_launch_metadata(data)
    return data


@pytest.mark.parametrize("count", [32, 36])
@pytest.mark.parametrize("scalar,msl", [("i32", "int"), ("u32", "uint"), ("fp32", "float"), ("i1", "bool")])
def test_generic_emission_and_record_have_identical_physical_mapping(count, scalar, msl):
    kb = builder(count, scalar)
    source = kb.build()
    record = kb._argument_abi
    layout = abi.validate_record(record, complete=False)
    assert layout.physical_count == 8
    assert f"constant uint* _packed_scalars [[buffer(7)]]" in source
    for slot, index in enumerate(layout.pointer_indices):
        assert f"float* a{index} [[buffer({slot})]]" in source
    for word, index in enumerate(layout.scalar_indices):
        rhs = f"_packed_scalars[{word}]"
        if scalar == "i1":
            rhs += " != 0u"
        elif scalar != "u32":
            rhs = f"as_type<{msl}>({rhs})"
        assert f"{msl} a{index} = {rhs};" in source
    assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert record["binary_sha256"] is record["binary_route"] is None


def test_positional_and_prebuilt_emitters_do_not_gain_a_count_based_certificate():
    direct = builder(31)
    assert "_packed_scalars" not in direct.build()
    assert direct._argument_abi is None
    prebuilt = builder()
    prebuilt.build()
    assert prebuilt._argument_abi is not None
    prebuilt.set_prebuilt_msl("kernel void old_name() {}")
    assert prebuilt.build() == "kernel void abi_test() {}"
    assert prebuilt._argument_abi is None


@pytest.mark.parametrize("scalar", ["fp64", "i64", "u1", "fp16"])
def test_unproved_source_width_has_no_packed_host_grant(scalar):
    kb = builder(scalar=scalar)
    assert abi.producer_record(kb.args, "generic-msl-v1", "source") is None


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_triangle_producer_binds_exact_emission(contiguous, enabled):
    from tests.test_tiled_k_layout_routing import triangle

    lowerer = triangle(contiguous=contiguous, enabled=enabled)
    source = lowerer.lower()
    record = lowerer._argument_abi
    layout = abi.validate_record(record, complete=False)
    assert record["producer"] == "biased-forward-v1"
    assert len(layout.declarations) == (32 if contiguous else 36)
    assert len(layout.pointer_indices) == 7
    assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    # MLIR erases the mask pointer's unsigned spelling. Public JIT retains it.
    live = tuple("*u8" if ty == "*i8" else ty for ty in layout.declarations)
    payloads = tuple(None if ty.startswith("*") else bytes(4) for ty in live)
    assert len(abi.prepare_host_payload(layout, live, payloads, (object(),) * len(live))) == (
        4 * len(layout.scalar_indices)
    )


def test_binary_binding_requires_fresh_msl_source_and_does_not_grant_air_abi():
    kb = builder()
    source = kb.build()
    pending = kb._argument_abi
    with pytest.raises(abi.LayoutError, match="not bound"):
        abi.validate_record(pending)
    data = {"argument_abi": pending, "binary_route": "msl"}
    with pytest.raises(abi.LayoutError, match="source changed"):
        abi.finish_record(data, source + "\n", b"binary")
    abi.finish_record(data, source, b"binary")
    assert data["argument_abi"]["binary_sha256"] == hashlib.sha256(b"binary").hexdigest()
    assert pending["binary_sha256"] is None
    with pytest.raises(abi.LayoutError, match="already finalized"):
        abi.finish_record(data, source, b"binary")
    data = {"argument_abi": pending, "binary_route": "cpp"}
    abi.finish_record(data, source, b"AIR")
    assert data["argument_abi"] is None


@pytest.mark.parametrize("damage", ["missing", "extra", "map", "wide", "route", "hash", "schema"])
def test_restored_record_rederives_mapping_and_checks_binary(damage):
    record = complete()
    if damage == "missing":
        record.pop("source_sha256")
    elif damage == "extra":
        record["future"] = True
    elif damage == "map":
        record["pointer_indices"].reverse()
    elif damage == "wide":
        record["declarations"][1] = "i64"
    elif damage == "route":
        record["binary_route"] = "cpp"
    elif damage == "hash":
        record["binary_sha256"] = "missing"
    else:
        record["schema"] = True
    with pytest.raises(abi.LayoutError):
        abi.validate_record(record)


def test_sidecar_seal_keeps_direct_hot_descriptor_identical_and_owns_snapshot():
    direct = metadata()
    packed = metadata(complete())
    assert json.loads(direct["launch_contract"])["schema"] == 2
    assert json.loads(packed["launch_contract"])["schema"] == 3
    hot, snapshot = contract.validate_launch_products(SimpleNamespace(**packed))
    assert hot == contract.validate_launch_metadata(direct)
    assert len(json.loads(hot)) == 12
    layout, digest = abi.capture_host_layout(snapshot)
    packed["argument_abi"]["declarations"][1] = "i64"
    assert layout.declarations[1] == "i32"
    assert digest == hashlib.sha256(b"compiled-library").hexdigest()
    with pytest.raises(abi.LayoutError):
        contract.validate_launch_products(packed)
    assert abi.capture_host_layout(None) == (None, None)


@pytest.mark.parametrize("damage", ["drop", "valid_replacement"])
def test_sidecar_cannot_be_changed_without_invalidating_the_seal(damage):
    data = metadata(complete())
    if damage == "drop":
        data.pop("argument_abi")
    else:
        data["argument_abi"] = complete(36)
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        contract.validate_launch_products(data)


@pytest.mark.parametrize("field", ["mm_two_kernel", "fast_matmul", "quant_matmul"])
def test_replacement_abi_cannot_borrow_generic_packing(field):
    data = metadata(complete())
    data[field] = {"unproved": True}
    with pytest.raises(MetalNonRecoverableError, match="replacement"):
        contract.seal_launch_metadata(data)


def test_payload_uses_captured_bytes_and_checks_declared_width_not_python_value():
    declarations = ("*fp32", "fp32", "i1", "u32") + ("i32",) * 28
    layout = abi.make_layout(declarations, kind="uint32-packed-v1")
    payloads = (None, struct.pack("<f", -0.0), b"\x01", b"\xff" * 4) + (b"\0" * 4,) * 28
    args = (object(),) + (object(),) * 31  # Never call these objects.
    assert abi.prepare_host_payload(layout, declarations, payloads, args) == (
        b"\0\0\0\x80\x01\0\0\0" + b"\xff" * 4 + b"\0" * 112
    )
    signed = list(declarations)
    signed[3] = "i32"  # Signless MLIR preserves source-checked bits.
    assert abi.prepare_host_payload(layout, signed, payloads, args) == abi.assemble_scalar_bytes(layout, payloads)
    bad = list(payloads)
    bad[1] = b"\0" * 8
    with pytest.raises(abi.LayoutError, match="width"):
        abi.prepare_host_payload(layout, declarations, bad, args)
    forged = copy.deepcopy(complete())
    forged["scalar_indices"].reverse()
    with pytest.raises(abi.LayoutError, match="mapping"):
        abi.validate_record(forged)


@pytest.mark.parametrize(
    "actual,expected,match",
    [
        ("*u8", "*i8", True),
        ("*i16", "*u16", True),
        ("*u32", "*i32", True),
        ("*i64", "*u64", True),
        ("*i8", "*i16", False),
        ("*fp64", "*i64", False),
        ("*fp32", "*u32", False),
        ("**i8", "**u8", False),
        ("u64", "i64", False),
        ("u1", "i1", False),
    ],
)
def test_signless_pointer_equivalence_does_not_equate_widths_or_conversions(actual, expected, match):
    assert abi._compatible(actual, expected) is match


def test_loader_binds_actual_binary_and_entry_without_querying_a_device(monkeypatch, tmp_path):
    from triton_msl.backend import driver
    from tests.test_runtime_storage_pipeline_scalar import _Device

    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: p)))
    monkeypatch.setattr(driver.tempfile, "tempdir", str(tmp_path))
    loader = driver.MetalUtils.__new__(driver.MetalUtils)
    loader._device = _Device("recording", "absent")
    binary = b"fake-library-never-executed"
    _, pipeline, *_ = loader.load_binary("abi_test", binary, 0)
    driver._validate_packed_pipeline(pipeline, "abi_test", hashlib.sha256(binary).hexdigest(), owner=loader)
    with pytest.raises(abi.LayoutError, match="loaded pipeline"):
        driver._validate_packed_pipeline(pipeline, "other_entry", hashlib.sha256(binary).hexdigest(), owner=loader)


def test_compiler_stage_finalizes_selected_binary_before_sealing(monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend import _cache_contract, compiler

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: "controlled")
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda _: None)
    monkeypatch.setattr(compiler.MetalBackend, "_initialize_compilation_runtime", staticmethod(lambda: None))
    monkeypatch.setattr(compiler.MetalBackend, "make_metallib", staticmethod(lambda *a: b"selected-library"))
    backend = compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32))
    stages = {}
    backend.add_stages(stages, compiler.MetalOptions())
    kb = builder()
    source = kb.build()
    data = dict(name="abi_test", num_warps=4, num_ctas=1, argument_abi=kb._argument_abi)
    assert stages["metallib"](source, data) == b"selected-library"
    _, snapshot = contract.validate_launch_products(data)
    assert abi.capture_host_layout(snapshot)[1] == hashlib.sha256(b"selected-library").hexdigest()


@triton.jit
def _generic_overflow(
    X,
    Y,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    s16,
    s17,
    s18,
    s19,
    s20,
    s21,
    s22,
    s23,
    s24,
    s25,
    s26,
    s27,
    s28,
    s29,
):
    i = tl.arange(0, 128)
    tl.store(Y + i, tl.load(X + i) + s0)


def test_generic_final_source_is_bound_after_late_alias_pass(monkeypatch):
    from tests.test_template_scalar_abi import _lower
    from triton_msl.codegen import generic_lowerer

    original = generic_lowerer._alias_shared_memory
    calls = []

    def rewritten(source, **kwargs):
        calls.append(source)
        return original(source, **kwargs) + "\n// controlled late rewrite\n"

    monkeypatch.setattr(generic_lowerer, "_alias_shared_memory", rewritten)
    signature = {n: "*fp32" if n in ("X", "Y") else "i32" for n in _generic_overflow.arg_names}
    lowerer = _lower(_generic_overflow, signature, {})
    source = lowerer.lower()
    assert len(calls) == 1 and source != calls[0]
    assert lowerer._argument_abi["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert lowerer._argument_abi["source_sha256"] != lowerer.kb._argument_abi["source_sha256"]
    data = {"argument_abi": lowerer._argument_abi, "binary_route": "msl"}
    abi.finish_record(data, source, b"controlled-library")
    abi.validate_record(data["argument_abi"])
