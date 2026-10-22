"""Persistent Triton key boundary; not resident-JIT or toolchain completeness."""

import triton  # noqa: F401 -- discover backends before importing ours
import pytest
from triton.backends.compiler import GPUTarget

from triton_msl.backend import _cache_contract as contract
from triton_msl.backend.compiler import MetalBackend


@pytest.fixture
def backend(monkeypatch):
    # Deterministic dependency witness. Actual discovery has separate tests.
    monkeypatch.setattr("triton_msl.backend.compiler.subprocess.check_output", lambda *a, **k: b"26.0\n")
    monkeypatch.setattr(contract, "toolchain_identity", lambda: "controlled-toolchain", raising=False)
    # This file isolates policy/target/schema changes, not the loader lifecycle.
    # USE_CPP can load a native provider that remains resident after the flag is
    # restored, so an uncontrolled framework identity need not return to its old
    # value. Real late-provider rekeying is required separately by
    # test_native_lazy_contract::test_late_provider_rekeys_every_product_boundary.
    monkeypatch.setattr(contract, "framework_identity", lambda: "controlled-framework")
    return MetalBackend(GPUTarget("metal", "apple-m4", 32))


@pytest.mark.parametrize(
    "name,a,b",
    [
        ("MEPT", "1", "0"),
        ("QUANT_MATMUL", "1", "0"),
        ("FAST_MATMUL", "1", "0"),
        ("COMPILE_SHADER", "1", "0"),
        ("FA_FAST", "1", "0"),
        ("INFER_LAYOUT", "0", "1"),
        ("LEGACY", "0", "1"),
        ("USE_CPP", "0", "1"),
        ("CPP_SCALAR_CFG", "0", "1"),
        ("FORCE_PYTHON", "0", "1"),
        ("FA_HALF_ACCUM", "0", "1"),
        ("CPP_SKIP", "", "reduce"),
    ],
)
def test_same_backend_instance_rekeys_current_policy(backend, monkeypatch, name, a, b):
    monkeypatch.setenv("TRITON_MSL_" + name, a)
    first = backend.hash()
    monkeypatch.setenv("TRITON_MSL_" + name, b)
    assert backend.hash() != first
    monkeypatch.setenv("TRITON_MSL_" + name, a)
    assert backend.hash() == first


def test_implementation_changes_without_human_label_bump(backend, monkeypatch):
    monkeypatch.setattr(contract, "implementation_identity", lambda: "a" * 64)
    first = backend.hash()
    monkeypatch.setattr(contract, "implementation_identity", lambda: "b" * 64)
    assert backend.hash() != first


def test_source_schema_rekeys_backend(backend, monkeypatch):
    first = backend.hash()
    monkeypatch.setattr(contract, "SOURCE_SCHEMA", contract.SOURCE_SCHEMA + 1)
    assert backend.hash() != first


@pytest.mark.parametrize("field,value", [("backend", "mps"), ("arch", "apple-m3"), ("warp_size", 64)])
def test_complete_target_participates_in_key(backend, field, value):
    fields = {"backend": "metal", "arch": "apple-m4", "warp_size": 32}
    fields[field] = value
    assert MetalBackend(GPUTarget(**fields)).hash() != backend.hash()


def test_equivalent_policy_spellings_share_key(backend, monkeypatch):
    monkeypatch.delenv("TRITON_MSL_MEPT", raising=False)
    first = backend.hash()
    monkeypatch.setenv("TRITON_MSL_MEPT", "1")
    assert backend.hash() == first
    monkeypatch.setenv("TRITON_MSL_MEPT", "True")
    assert backend.hash() == first


def test_diagnostics_do_not_change_semantic_key(backend, monkeypatch):
    first = backend.hash()
    monkeypatch.setenv("TRITON_MSL_DEBUG", "1")
    monkeypatch.setenv("TRITON_MSL_DUMP_DIR", "/a/different/diagnostic/directory")
    assert backend.hash() == first


def test_current_source_contract_is_not_a_resident_handle_certificate(backend):
    # Scope/control: repeated lookups are stable. This does NOT exercise the
    # earlier external JIT lookup, so none of these rows claim it is protected.
    assert backend.hash() == backend.hash()
