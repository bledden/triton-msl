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
    # Since M5 the default is ON, so no-env and "1" share a key; the escape
    # hatch "0" must produce a DISTINCT key (else a scalar-path metallib could
    # be served to a default/array-path compile, or vice versa).
    from triton_metal.backend.compiler import _msl_cache_key
    monkeypatch.delenv("TRITON_METAL_MEPT", raising=False)  # default ON
    default_on = _msl_cache_key("m", "o")
    monkeypatch.setenv("TRITON_METAL_MEPT", "0")            # escape hatch
    assert _msl_cache_key("m", "o") != default_on
    monkeypatch.setenv("TRITON_METAL_MEPT", "1")            # explicit ON == default
    assert _msl_cache_key("m", "o") == default_on
