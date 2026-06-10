"""Legacy text parser is opt-in only (Phase 0 T4): the heuristic fallback can
emit silently-wrong kernels, so by default an unlowerable graph refuses."""
import os
import pytest

import triton  # noqa: F401


def test_fallback_refuses_by_default(monkeypatch):
    from triton_metal.codegen import msl_emitter
    from triton_metal.errors import MetalNonRecoverableError
    monkeypatch.delenv("TRITON_METAL_LEGACY", raising=False)
    with pytest.raises(MetalNonRecoverableError):
        msl_emitter._legacy_fallback("module {}", {}, None, "lowerer failed")


def test_fallback_allowed_when_opted_in(monkeypatch):
    from triton_metal.codegen import msl_emitter
    monkeypatch.setenv("TRITON_METAL_LEGACY", "1")
    try:
        msl_emitter._legacy_fallback("module {}", {}, None, "lowerer failed")
    except Exception as e:
        assert "NonRecoverable" not in type(e).__name__
