"""Shared toolchain diagnostics without compiler/backend discovery imports."""

_METAL_TOOLCHAIN_MISSING_HINT = (
    "The Metal Toolchain is not installed. macOS 26 / Xcode 26 ship the Metal "
    "compiler as a separate on-demand component, so a fresh setup or a recent "
    "Xcode update can leave it absent. Install it with:\n"
    "    sudo xcodebuild -downloadComponent MetalToolchain\n"
    "(or Xcode > Settings > Components > Metal Toolchain), then verify with "
    "`xcrun metal --version`."
)


def _is_metal_toolchain_missing(stderr: str) -> bool:
    """Recognize the existing missing-component diagnostic, not a retry hint."""
    return "missing Metal Toolchain" in stderr or "cannot execute tool 'metal'" in stderr
