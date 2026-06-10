"""Cache keys must include codegen version + MEPT flag (Phase 0, audit debt #1)."""
import hashlib

import triton  # noqa: F401  (backend discovery must precede compiler import)


def test_msl_cache_key_includes_codegen_version():
    from triton_metal import CODEGEN_VERSION
    from triton_metal.backend.compiler import _msl_cache_key
    unversioned = hashlib.sha256("modtext_optshash".encode()).hexdigest()[:16]
    key = _msl_cache_key("modtext_", "optshash")
    assert key != unversioned
    assert _msl_cache_key("modtext_", "optshash") == key  # deterministic
    assert CODEGEN_VERSION  # non-empty


def test_msl_cache_key_changes_with_mept(monkeypatch):
    from triton_metal.backend.compiler import _msl_cache_key
    monkeypatch.delenv("TRITON_METAL_MEPT", raising=False)
    off = _msl_cache_key("m", "o")
    monkeypatch.setenv("TRITON_METAL_MEPT", "1")
    assert _msl_cache_key("m", "o") != off
