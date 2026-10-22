"""Native switch, pre-adapter decisions, and selected-stage semantic checks (CPU)."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton  # noqa: F401

from triton_msl.backend.compiler import MetalBackend
from triton_msl.backend import _cache_contract, device_detect
from test_cpp_memory_semantics import CORPUS, check_case, function_body

cpp = pytest.importorskip("triton_msl._triton_msl_cpp")
FIXTURES = Path(__file__).with_name("fixtures") / "cpp_cfg"
CHAINED = CORPUS["LANE_INDEXED_F32"].replace(
    "%pp = tt.addptr %p, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>",
    "%one = arith.constant dense<1> : tensor<128xi32>\n"
    "    %pa = tt.addptr %p, %i : tensor<128x!tt.ptr<f32>>, tensor<128xi32>\n"
    "    %pp = tt.addptr %pa, %one : tensor<128x!tt.ptr<f32>>, tensor<128xi32>",
)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_integrated_selection_semantics_and_metadata(monkeypatch, record_property, name):
    monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", "1")
    monkeypatch.setattr(device_detect, "air_version_components", lambda: (2, 8, 4, 0))
    selected = []
    adapter = MetalBackend._opaque_to_typed_ptrs

    def capture(text, **kwargs):
        selected.append(text)
        return adapter(text, **kwargs)

    monkeypatch.setattr(MetalBackend, "_opaque_to_typed_ptrs", staticmethod(capture))
    metadata = {}
    typed = MetalBackend.make_llir(CORPUS[name], metadata, SimpleNamespace(num_warps=4))
    assert len(selected) == 1
    assert not check_case(name, function_body(selected[0]))
    expected = "fallback" if name == "TWO_PTRS_SAME_MASK" else "optimized"
    assert metadata["cpp_scalar_cfg_route"] == expected, metadata
    assert metadata["cpp_scalar_cfg_selected_opaque_sha256"] == hashlib.sha256(selected[0].encode()).hexdigest()
    assert metadata["cpp_scalar_cfg_typed_ir_sha256"] == hashlib.sha256(typed.encode()).hexdigest()
    accepted = cpp.run_to_llvm(CORPUS[name], 2, 8, 4, 0)
    accepted_typed = adapter(accepted)
    assert [x for x in typed.splitlines() if x.startswith("!")] == [
        x for x in accepted_typed.splitlines() if x.startswith("!")
    ]
    if expected == "fallback":
        assert selected[0] == accepted
        assert typed == accepted_typed
    record_property("route", expected)
    record_property("typed_ir_sha256", metadata["cpp_scalar_cfg_typed_ir_sha256"])


@pytest.mark.parametrize("block", [128, 2048])
@pytest.mark.parametrize("name", ["TWO_PTRS_SAME_MASK", "CHAINED_ADDPTR_F32"])
def test_same_process_fallback_retains_capability(monkeypatch, name, block):
    source = (CHAINED if name == "CHAINED_ADDPTR_F32" else CORPUS[name]).replace("128", str(block))
    monkeypatch.setattr(device_detect, "air_version_components", lambda: (2, 8, 4, 0))
    monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", "0")
    before = {}
    accepted = MetalBackend.make_llir(source, before, SimpleNamespace(num_warps=4))
    monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", "1")
    after = {}
    gated = MetalBackend.make_llir(source, after, SimpleNamespace(num_warps=4))
    assert before["cpp_scalar_cfg_route"] == "disabled"
    assert after["cpp_scalar_cfg_route"] == "fallback"
    assert after["cpp_scalar_cfg_reason"]
    assert gated == accepted


@pytest.mark.parametrize(
    "name",
    ["TWO_PTRS_SAME_MASK", "LANE_INDEXED_F32", "LANE_INDEXED_I32", "STRIDED_2_F32", "NEGATED_MASK", "VOLATILE_LOAD"],
)
def test_runtime_switch_reproduces_reviewed_helpers(name):
    for optimize, arm in [(False, "cpp-volatile-build"), (True, "cpp-cfg-native-build")]:
        expected = (FIXTURES / arm / name / "opaque.ll").read_text()
        assert cpp.run_to_llvm(CORPUS[name], 2, 8, 4, 0, optimize_scalar_cfg=optimize) == expected
    assert cpp.run_to_llvm(CORPUS[name], 2, 8, 4, 0) == cpp.run_to_llvm(
        CORPUS[name], 2, 8, 4, 0, optimize_scalar_cfg=False
    )


def test_scalar_cfg_is_opt_in_and_cache_bound(monkeypatch):
    monkeypatch.delenv("TRITON_MSL_CPP_SCALAR_CFG", raising=False)
    disabled = _cache_contract.effective_policy()
    assert disabled["CPP_SCALAR_CFG"] is False
    monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", "1")
    enabled = _cache_contract.effective_policy()
    assert enabled["CPP_SCALAR_CFG"] is True
    assert _cache_contract._digest(disabled) != _cache_contract._digest(enabled)


@pytest.mark.parametrize("error", [RuntimeError("pass verification failed"), TypeError("old helper")])
def test_optional_native_failure_uses_original_before_adapter(monkeypatch, error):
    monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", "1")
    monkeypatch.setattr(device_detect, "air_version_components", lambda: (2, 8, 4, 0))
    native = cpp.run_to_llvm
    source = CORPUS["TWO_PTRS_SAME_MASK"]
    original = native(source, 2, 8, 4, 0)

    def refuse(*args, **kwargs):
        if kwargs.get("optimize_scalar_cfg"):
            raise error
        return native(*args, **kwargs)

    monkeypatch.setattr(cpp, "run_to_llvm", refuse)
    metadata = {}
    typed = MetalBackend.make_llir(source, metadata, SimpleNamespace(num_warps=4))
    assert typed == MetalBackend._strip_unsupported_llvm_attrs(MetalBackend._opaque_to_typed_ptrs(original))
    assert metadata["cpp_scalar_cfg_route"] == "fallback"
    assert str(error) in metadata["cpp_scalar_cfg_reason"]
    assert metadata["cpp_scalar_cfg_optimized_opaque_sha256"] is None
