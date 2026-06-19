"""Tests for the data-driven C++ family allowlist (Phase 1, T1)."""
import json
import os

import triton  # noqa: F401


def test_families_cover_legacy_allowlist():
    from triton_msl.backend.cpp_families import FAMILIES, enabled_ops
    assert "elementwise" in FAMILIES and "tt.load" in enabled_ops()


def test_router_uses_table():
    from triton_msl.backend.compiler import MetalBackend
    assert MetalBackend._has_complex_ops("  %0 = tt.fancy_unknown %a") is True
    assert MetalBackend._has_complex_ops("  %0 = tt.splat %a") is False


def test_dot_not_default_on(monkeypatch):
    """tt.dot must NOT route C++ by default (multi-tile 2D grids are
    wrong via the C++ dot path: pid_n tiles never written — regression
    surfaced by test_integration.py::test_triton_jit_matmul when
    elementwise went default-on)."""
    from triton_msl.backend.compiler import MetalBackend
    monkeypatch.delenv("TRITON_MSL_USE_CPP", raising=False)
    assert MetalBackend._has_complex_ops("  %0 = tt.dot %a, %b") is True
    assert MetalBackend._has_complex_ops("  %0 = tt.reduce %a") is True


def test_coverage_report_matches_enabled():
    """reports/cpp_coverage.json must mirror the in-code ENABLED set."""
    from triton_msl.backend.cpp_families import ENABLED
    report = os.path.join(os.path.dirname(__file__), os.pardir,
                          "reports", "cpp_coverage.json")
    with open(report) as f:
        assert sorted(json.load(f)["enabled"]) == sorted(ENABLED)


def test_use_cpp_optin_keeps_legacy_surface(monkeypatch):
    """TRITON_MSL_USE_CPP=1 preserves the pre-Phase-1 opt-in surface
    (reduce/dot/shared-mem) that test_cpp_backend.py exercises."""
    from triton_msl.backend.cpp_families import FAMILIES, enabled_ops
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    legacy = set().union(*FAMILIES.values())
    assert enabled_ops() == legacy
    assert {"tt.dot", "tt.reduce", "ttg.local_alloc"} <= enabled_ops()
