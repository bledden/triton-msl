"""Tests for the data-driven C++ family allowlist (Phase 1, T1)."""
import triton  # noqa: F401


def test_families_cover_legacy_allowlist():
    from triton_metal.backend.cpp_families import FAMILIES, enabled_ops
    assert "elementwise" in FAMILIES and "tt.load" in enabled_ops()


def test_router_uses_table():
    from triton_metal.backend.compiler import MetalBackend
    assert MetalBackend._has_complex_ops("  %0 = tt.fancy_unknown %a") is True
    assert MetalBackend._has_complex_ops("  %0 = tt.splat %a") is False
