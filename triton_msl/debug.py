"""Debug utilities for triton-msl.

Provides multi-level debug output controlled by environment variables:
  TRITON_MSL_DEBUG   - Debug verbosity level (0=off, 1=dump IR/MSL, 2=+timing)
  TRITON_MSL_DUMP_DIR - Directory for debug dumps (default: /tmp/triton_msl_debug)
"""

import os

# Sentinel to distinguish "not yet cached" from "cached value is 0".
_UNSET = object()

_cached_debug_level = _UNSET
_cached_dump_dir = _UNSET


def _debug_level() -> int:
    """Return the current debug level from TRITON_MSL_DEBUG env var.

    Returns 0 if not set. Caches the result to avoid repeated os.environ lookups.
    Call _reset_debug_cache() in tests to clear the cached value.
    """
    global _cached_debug_level
    if _cached_debug_level is _UNSET:
        raw = os.environ.get("TRITON_MSL_DEBUG", "")
        try:
            _cached_debug_level = int(raw) if raw else 0
        except ValueError:
            _cached_debug_level = 0
    return _cached_debug_level


def _dump_dir() -> str:
    """Return the debug dump directory from TRITON_MSL_DUMP_DIR env var.

    Defaults to /tmp/triton_msl_debug if not set.
    Caches the result to avoid repeated os.environ lookups.
    """
    global _cached_dump_dir
    if _cached_dump_dir is _UNSET:
        _cached_dump_dir = os.environ.get(
            "TRITON_MSL_DUMP_DIR", "/tmp/triton_msl_debug"
        )
    return _cached_dump_dir


_cached_fallback_mode = _UNSET


def _fallback_mode() -> str:
    """Return the fallback mode from TRITON_MSL_FALLBACK env var.

    Controls behavior when MSL codegen or Metal compilation fails:
      "warn"   - emit a warning, then re-raise so Triton/torch.compile falls back (default)
      "silent" - re-raise without a warning
      "error"  - re-raise with the original exception (no fallback hint in message)

    Caches the result to avoid repeated os.environ lookups.
    """
    global _cached_fallback_mode
    if _cached_fallback_mode is _UNSET:
        raw = os.environ.get("TRITON_MSL_FALLBACK", "warn").lower().strip()
        if raw not in ("warn", "silent", "error"):
            raw = "warn"
        _cached_fallback_mode = raw
    return _cached_fallback_mode


def _reset_debug_cache():
    """Reset cached debug settings. Intended for use in tests."""
    global _cached_debug_level, _cached_dump_dir, _cached_fallback_mode
    _cached_debug_level = _UNSET
    _cached_dump_dir = _UNSET
    _cached_fallback_mode = _UNSET
