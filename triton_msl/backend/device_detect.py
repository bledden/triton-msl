"""Metal device detection for triton-msl.

Detects Apple Silicon chip generation, Metal version, GPU capabilities,
and Neural Accelerator availability.  Results are cached per-process
(the Metal device is a system singleton on macOS).

Usage:
    from triton_msl.backend.device_detect import get_device_info
    info = get_device_info()
    print(info.chip_family, info.metal_version, info.supports_metal4)
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# DeviceInfo dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    """Cached snapshot of the Metal device capabilities."""

    chip_family: str  # "M1", "M2", "M3", "M4", "M5", "unknown"
    chip_variant: str  # "base", "Pro", "Max", "Ultra"
    gpu_core_count: int
    max_threads_per_threadgroup: int
    metal_version: str  # "3.0", "3.1", "3.2", "4.0", "4.1"
    has_neural_accelerator: bool  # True for M5+
    has_bfloat16: bool  # True for Metal 3.1+
    supports_metal4: bool
    supports_tensor_ops: bool  # M5 Neural Accelerators in GPU pipeline

    @property
    def metal_std_flag(self) -> str:
        """Return the ``-std=`` flag for xcrun metal compilation."""
        return f"-std=metal{self.metal_version}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Chip family pattern: "Apple M4 Max" -> ("M4", "Max")
_CHIP_RE = re.compile(
    r"Apple\s+(M[1-9]\d*)\s*(Pro|Max|Ultra)?",
    re.IGNORECASE,
)

# Maps chip family to a rough default GPU core count per variant.
# Real core counts vary by binning; these are canonical full-config values.
_CORE_COUNTS: dict[str, dict[str, int]] = {
    "M1": {"base": 8, "Pro": 16, "Max": 32, "Ultra": 64},
    "M2": {"base": 10, "Pro": 19, "Max": 38, "Ultra": 76},
    "M3": {"base": 10, "Pro": 18, "Max": 40, "Ultra": 80},
    "M4": {"base": 10, "Pro": 20, "Max": 40, "Ultra": 80},
    "M5": {"base": 12, "Pro": 24, "Max": 48, "Ultra": 96},
}


def _parse_chip(device_name: str) -> tuple[str, str]:
    """Extract (chip_family, chip_variant) from a Metal device name string.

    Returns ("unknown", "base") if the name doesn't match.
    """
    if not device_name:
        return ("unknown", "base")
    m = _CHIP_RE.search(device_name)
    if m is None:
        return ("unknown", "base")
    family = m.group(1).upper()  # "m4" -> "M4"
    variant = m.group(2) or "base"
    if variant != "base":
        # Normalize case: "pro" -> "Pro"
        variant = variant.capitalize()
    return (family, variant)


def _estimate_core_count(family: str, variant: str) -> int:
    """Return the expected GPU core count for a chip family/variant."""
    family_map = _CORE_COUNTS.get(family)
    if family_map is None:
        # Unknown future chip — guess based on variant
        return {"base": 10, "Pro": 20, "Max": 40, "Ultra": 80}.get(variant, 10)
    return family_map.get(variant, family_map.get("base", 10))


def _detect_sdk_version() -> Optional[str]:
    """Return the macOS SDK version string via ``xcrun --show-sdk-version``.

    Returns None if xcrun is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["xcrun", "--show-sdk-version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _probe_metal_compiler() -> Optional[str]:
    """Probe xcrun metal to find the highest *usable* -std=metalX.Y version.

    Tries versions from newest to oldest.  For each candidate the probe:
      1. compiles a trivial kernel to .air,
      2. links it into a .metallib,
      3. attempts to load the .metallib via the Metal runtime.

    Step 3 is critical because the compiler may accept a version that the
    current OS runtime cannot load (e.g. metal4.0 compiles on SDK 26.2
    but MTLDevice refuses to load the library until macOS 26.4+).

    This is more reliable than guessing from SDK version numbers (which
    jumped from 15.x to 26.x with macOS Tahoe).
    """
    import tempfile, os
    candidates = ["4.1", "4.0", "3.2", "3.1", "3.0"]
    test_src = "kernel void _probe() {}\n"
    for ver in candidates:
        air_path = None
        lib_path = None
        metal_path = None
        try:
            # Write source
            with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
                f.write(test_src)
                metal_path = f.name
            air_path = metal_path.replace(".metal", ".air")
            lib_path = metal_path.replace(".metal", ".metallib")

            # Step 1: compile to .air
            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metal", "-std=metal" + ver,
                 "-c", metal_path, "-o", air_path],
                capture_output=True, timeout=10, check=True,
            )
            # Step 2: link to .metallib
            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", lib_path],
                capture_output=True, timeout=10, check=True,
            )
            # Step 3: verify the OS runtime can load it
            try:
                import Metal, Foundation
                device = Metal.MTLCreateSystemDefaultDevice()
                if device is not None:
                    url = Foundation.NSURL.fileURLWithPath_(lib_path)
                    lib, error = device.newLibraryWithURL_error_(url, None)
                    if error is not None:
                        continue  # Runtime rejected this version
            except ImportError:
                pass  # No Metal/Foundation module — trust the compiler result

            return ver
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired, OSError):
            continue
        finally:
            for p in (metal_path, air_path, lib_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
    return None


def _infer_metal_version(sdk_version: Optional[str], chip_family: str) -> str:
    """Infer the highest supported Metal Shading Language version.

    First tries to probe the compiler directly (most reliable).  Falls
    back to SDK-version heuristics if probing fails.  Metal 4.x is
    clamped to 3.2 on pre-M4 hardware.
    """
    # Prefer runtime probe — works regardless of SDK version numbering.
    probed = _probe_metal_compiler()
    if probed is not None:
        # Clamp Metal 4.x to 3.2 on pre-M4 chips.
        if probed.startswith("4"):
            chip_gen = _chip_generation(chip_family)
            if chip_gen < 4:
                return "3.2"
        return probed

    # Fallback: guess from SDK version.
    if sdk_version is None:
        return "3.0"

    try:
        parts = sdk_version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return "3.0"

    # macOS Tahoe uses SDK version 26.x.  Older macOS used 14.x / 15.x.
    if major >= 26:
        sdk_metal = "4.0"
    elif major == 15 and minor >= 4:
        sdk_metal = "4.0"
    elif major >= 15:
        sdk_metal = "3.2"
    elif major >= 14:
        sdk_metal = "3.1"
    else:
        sdk_metal = "3.0"

    # Metal 4.x requires M4+ hardware.
    if sdk_metal.startswith("4"):
        chip_gen = _chip_generation(chip_family)
        if chip_gen < 4:
            return "3.2" if major >= 15 or major >= 26 else ("3.1" if major >= 14 else "3.0")

    return sdk_metal


def _chip_generation(family: str) -> int:
    """Return the numeric generation from a chip family string (e.g. "M4" -> 4)."""
    m = re.match(r"M(\d+)", family, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cached_info: Optional[DeviceInfo] = None
# Serializes first-time detection. Beyond preventing redundant detection, the
# lock is required for correctness under concurrent first use: _detect_device_info
# does `import Metal; Metal.MTLCreateSystemDefaultDevice()`, and PyObjC populates
# that symbol lazily on first attribute access. Two threads accessing it at once
# can race the lazy loader and raise KeyError('MTLCreateSystemDefaultDevice').
# Serializing detection keeps that import+access single-threaded.
_cache_lock = threading.Lock()


def get_device_info() -> DeviceInfo:
    """Return cached device info.  Detection runs at most once per process."""
    global _cached_info
    if _cached_info is not None:
        return _cached_info
    with _cache_lock:
        # Double-checked: another thread may have detected while we waited.
        if _cached_info is None:
            _cached_info = _detect_device_info()
        return _cached_info


def _detect_device_info() -> DeviceInfo:
    """Run full device detection (called once, then cached)."""
    # Lazy-import Metal to avoid crash on non-macOS or missing PyObjC.
    try:
        import Metal  # type: ignore[import-untyped]
        device = Metal.MTLCreateSystemDefaultDevice()
    except ImportError:
        device = None

    if device is None:
        # Fallback: no Metal device available.
        return DeviceInfo(
            chip_family="unknown",
            chip_variant="base",
            gpu_core_count=0,
            max_threads_per_threadgroup=0,
            metal_version="3.0",
            has_neural_accelerator=False,
            has_bfloat16=False,
            supports_metal4=False,
            supports_tensor_ops=False,
        )

    device_name = device.name() or ""
    family, variant = _parse_chip(device_name)
    core_count = _estimate_core_count(family, variant)
    max_threads = device.maxThreadsPerThreadgroup().width

    sdk_version = _detect_sdk_version()
    metal_version = _infer_metal_version(sdk_version, family)

    gen = _chip_generation(family)

    # Metal 3.1+ supports bfloat16 in shaders.
    metal_major, metal_minor = (int(x) for x in metal_version.split("."))
    has_bfloat16 = (metal_major, metal_minor) >= (3, 1)

    supports_metal4 = metal_major >= 4

    # Neural Accelerators (GPU tensor ops) are available on M5+ with Metal 4.1+.
    has_neural_accel = gen >= 5
    supports_tensor_ops = has_neural_accel and (metal_major, metal_minor) >= (4, 1)

    return DeviceInfo(
        chip_family=family,
        chip_variant=variant,
        gpu_core_count=core_count,
        max_threads_per_threadgroup=max_threads,
        metal_version=metal_version,
        has_neural_accelerator=has_neural_accel,
        has_bfloat16=has_bfloat16,
        supports_metal4=supports_metal4,
        supports_tensor_ops=supports_tensor_ops,
    )


def reset_device_cache() -> None:
    """Clear cached device info.  Intended for testing."""
    global _cached_info
    _cached_info = None
