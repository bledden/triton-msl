"""triton-msl: Metal (Apple Silicon) backend for OpenAI Triton."""

# Bump on ANY emitter/lowerer change: persistent caches at ~/.cache/triton_msl
# are keyed by TTGIR + options only; without this, codegen fixes silently
# replay stale compiled kernels after upgrade (Phase 0, audit debt #1).
# RELEASE CHECKLIST: this MUST be bumped before every PyPI release so an
# in-place upgrade can never hit a warm cache entry from a pre-fix codegen.
CODEGEN_VERSION = "2026.06.24.1"

# Distribution version, discoverable as ``triton_msl.__version__`` (falls back
# gracefully when the package metadata is unavailable, e.g. a source checkout).
try:  # pragma: no cover - trivial metadata lookup
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    try:
        __version__ = _pkg_version("triton-msl")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"
