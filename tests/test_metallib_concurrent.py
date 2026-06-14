"""Regression test: concurrent make_metallib calls on the same kernel must not race.

Root cause: make_metallib / make_metallib_from_llir used FIXED staging paths
({base}.metal / {base}.air / {base}.metallib.tmp) keyed only on content hash.
When two threads compiled the same kernel concurrently, they shared these paths;
one thread's os.replace({base}.tmp -> {base}.metallib) consumed the temp file
before the other thread could rename it → FileNotFoundError.

Fix: per-call tempfile.mkdtemp() under the cache dir so every call has unique
intermediates; the final os.replace onto the content-addressed .metallib is
still atomic and idempotent (last-writer-wins with identical content).
"""
import os
import platform
import subprocess
import threading

import pytest
import triton  # noqa: F401  (backend discovery must precede compiler import)

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)


def _has_metal_compiler():
    try:
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_metal_compiler = pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="xcrun metal not available",
)

# Minimal valid MSL kernel that xcrun metal is happy to compile.
_MINIMAL_MSL = """\
#include <metal_stdlib>
using namespace metal;

kernel void test_concurrent_k(
    device float* out [[ buffer(0) ]],
    uint tid [[ thread_position_in_grid ]])
{
    out[tid] = float(tid);
}
"""


@requires_metal_compiler
def test_make_metallib_concurrent_no_race(tmp_path, monkeypatch):
    """10 threads × 10 iterations of make_metallib on the same kernel must all succeed."""
    # Redirect the cache to a fresh scratch dir so every run of this test starts cold.
    monkeypatch.setenv("TRITON_METAL_CACHE_DIR", str(tmp_path))

    # Import after monkeypatching so _get_cache_dir() sees the overridden env var.
    from triton_metal.backend.compiler import MetalBackend, MetalOptions

    options = MetalOptions()
    metadata = {"name": "test_concurrent_k"}

    errors: list[Exception] = []
    lock = threading.Lock()

    def worker():
        for _ in range(10):
            try:
                data = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
                assert data and len(data) > 0, "make_metallib returned empty bytes"
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], (
        f"{len(errors)} thread(s) raised exceptions during concurrent make_metallib:\n"
        + "\n".join(f"  {type(e).__name__}: {e}" for e in errors[:5])
    )
