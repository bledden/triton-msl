"""triton-msl: Metal (Apple Silicon) backend for OpenAI Triton."""

# An explicit version marker supplements the content-bound cache identity.
# Source/native implementation bytes, framework/toolchain selection and relevant
# policy also enter the keys; correctness no longer relies on manually bumping
# this constant after every codegen edit. Installed-code changes require restart.
CODEGEN_VERSION = "2026.09.28.1"

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

# NOTE: tl.extra.libdevice is filled in from the Metal backend compiler
# (MetalBackend.__init__), NOT here — at package-import time Triton is only
# partially initialized (this module is imported via the backend entry-point
# during `import triton`), so the libdevice import hits a circular ImportError.
# See triton_msl/_libdevice.py and backend/compiler.py.
