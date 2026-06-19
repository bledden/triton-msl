"""Tests for bounded retry in the Metal compilation pipeline.

Covers:
1. Happy path: compile + cache still works after the retry refactor.
2. Real compile error raises promptly (no retry storm).
3. Simulated transient: metal -c exits 0 but writes no .air on the first 2
   calls, succeeds on the 3rd — compile ultimately succeeds via retry.
"""
import os
import platform
import subprocess
import time

import pytest

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

# Minimal valid MSL kernel used for happy-path and error tests.
_MINIMAL_MSL = """\
#include <metal_stdlib>
using namespace metal;

kernel void test_retry_k(
    device float* out [[ buffer(0) ]],
    uint tid [[ thread_position_in_grid ]])
{
    out[tid] = float(tid) + 1.0f;
}
"""

# Deliberately invalid MSL — the metal compiler must reject this.
_INVALID_MSL = """\
this is not valid metal {
"""


@requires_metal_compiler
def test_make_metallib_happy_path_and_cache(tmp_path, monkeypatch):
    """Compile a valid MSL kernel twice: first call compiles, second hits cache.

    Guards that the retry-loop refactor did not break normal compilation or
    the content-addressed cache hit path.
    """
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))

    import triton  # noqa: F401 — backend discovery must precede compiler import
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    options = MetalOptions()
    metadata = {"name": "test_retry_k"}

    # First call — must compile from scratch and return non-empty bytes.
    data1 = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
    assert data1 and len(data1) > 0, "make_metallib returned empty bytes on first call"

    # Second call with the same source — must hit cache and return the same bytes.
    data2 = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
    assert data2 == data1, "Second make_metallib call returned different bytes (cache miss or corruption)"


@requires_metal_compiler
def test_make_metallib_real_error_raises_promptly(tmp_path, monkeypatch):
    """Invalid MSL raises MetalCompilationError for SHADER COMPILATION, not linking.

    A CalledProcessError from the metal -c step is a real deterministic error.
    The retry loop must raise immediately (on attempt 0), not after 3 attempts.
    The error message must identify it as a shader compilation failure, NOT a
    library-linking failure.
    """
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))

    import triton  # noqa: F401
    from triton_msl.backend.compiler import MetalBackend, MetalOptions
    from triton_msl.errors import MetalCompilationError

    options = MetalOptions()
    metadata = {"name": "test_bad_msl_k"}

    t0 = time.perf_counter()
    with pytest.raises(MetalCompilationError) as exc_info:
        MetalBackend.make_metallib(_INVALID_MSL, dict(metadata), options)
    elapsed = time.perf_counter() - t0

    err_msg = str(exc_info.value)

    # Must identify as a SHADER COMPILATION error, not a linking error.
    assert "shader compilation" in err_msg.lower() or "compilation failed" in err_msg.lower(), (
        f"Expected 'shader compilation' or 'compilation failed' in error, got: {err_msg!r}"
    )
    assert "linking" not in err_msg.lower(), (
        f"Error should not mention 'linking' for an MSL syntax error, got: {err_msg!r}"
    )

    # Must be prompt — not after retrying 3 times with sleeps (max ~0.3 s + compiler time).
    # Allow 30 s for the compiler itself; if we're retrying 3× it would be much longer.
    assert elapsed < 30.0, (
        f"make_metallib took {elapsed:.1f}s for an invalid MSL — suggests unwanted retrying"
    )


@requires_metal_compiler
def test_make_metallib_retries_on_missing_air(tmp_path, monkeypatch):
    """Simulate the transient flake: metal -c exits 0 but .air is missing on first 2 calls.

    Uses monkeypatch to intercept subprocess.run.  On the first 2 calls to
    xcrun metal -c, the stub returns exit 0 WITHOUT writing the .air file.
    On the 3rd call the stub delegates to the real subprocess.run.  The
    compile must ultimately succeed via the retry mechanism.
    """
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))

    import triton  # noqa: F401
    import triton_msl.backend.compiler as compiler_mod
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    options = MetalOptions()
    metadata = {"name": "test_transient_k"}

    real_run = subprocess.run
    call_count = {"metal_c": 0}

    def fake_run(cmd, **kwargs):
        # Only intercept "xcrun ... metal -c ..." (not metallib).
        if (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and "metal" in cmd
            and "-c" in cmd
            and "metallib" not in cmd[2]  # exclude xcrun metallib
        ):
            call_count["metal_c"] += 1
            if call_count["metal_c"] <= 2:
                # Simulate: exits 0 but writes no .air (transient flake).
                class _FakeResult:
                    returncode = 0
                    stdout = b""
                    stderr = b""
                    args = cmd
                return _FakeResult()
            # 3rd call: let the real compiler run.
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(compiler_mod.subprocess, "run", fake_run)

    data = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)

    assert data and len(data) > 0, "make_metallib returned empty bytes after retry"
    assert call_count["metal_c"] == 3, (
        f"Expected 3 metal -c calls (2 transient + 1 success), got {call_count['metal_c']}"
    )


@requires_metal_compiler
def test_make_metallib_retries_on_transient_compile_exit(tmp_path, monkeypatch):
    """A nonzero `metal -c` exit WITHOUT a source-location diagnostic is a
    TRANSIENT toolchain flake (spawn failure / signal / SDK race under heavy
    parallel load), not a real MSL error — it must be RETRIED, not raised on
    attempt 0.

    Regression for a NaN-divergence in torch.compile training: under full-suite
    load `xcrun metal -c` occasionally exits 1 on otherwise-valid MSL. The old
    code treated EVERY nonzero `metal -c` exit as a deterministic syntax error
    and raised immediately (no retry); the raised compile failure then surfaced
    as a non-converging (NaN) kernel rather than recovering.
    """
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))

    import triton  # noqa: F401
    import triton_msl.backend.compiler as compiler_mod
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    options = MetalOptions()
    metadata = {"name": "test_transient_compile_k"}

    real_run = subprocess.run
    call_count = {"metal_c": 0}

    def fake_run(cmd, **kwargs):
        if (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and "metal" in cmd
            and "-c" in cmd
            and "metallib" not in cmd[2]
        ):
            call_count["metal_c"] += 1
            if call_count["metal_c"] <= 2:
                # Transient: nonzero exit, NO `file:line:col: error:` diagnostic
                # (a toolchain-level failure — note " error:" present but not a
                # source location, so it must NOT be classified as a real error).
                raise subprocess.CalledProcessError(
                    returncode=1, cmd=cmd, output=b"",
                    stderr=b"clang: error: unable to execute command: Segmentation fault: 11\n",
                )
            # 3rd call: let the real compiler run.
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(compiler_mod.subprocess, "run", fake_run)

    data = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
    assert data and len(data) > 0, "make_metallib returned empty bytes after transient-exit retry"
    assert call_count["metal_c"] == 3, (
        f"Expected 3 metal -c calls (2 transient + 1 success), got {call_count['metal_c']}"
    )


@requires_metal_compiler
def test_make_metallib_real_compile_error_no_retry(tmp_path, monkeypatch):
    """A nonzero `metal -c` exit WITH a `file:line:col: error:` diagnostic is a
    REAL deterministic MSL error — it must raise on the FIRST attempt (retrying
    a genuine syntax error wastes time). Locks the discriminator so the
    transient-retry change above does not turn real errors into retry storms.
    """
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))

    import triton  # noqa: F401
    import triton_msl.backend.compiler as compiler_mod
    from triton_msl.backend.compiler import MetalBackend, MetalOptions
    from triton_msl.errors import MetalCompilationError

    options = MetalOptions()
    metadata = {"name": "test_real_err_k"}

    real_run = subprocess.run
    call_count = {"metal_c": 0}

    def fake_run(cmd, **kwargs):
        if (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and "metal" in cmd
            and "-c" in cmd
            and "metallib" not in cmd[2]
        ):
            call_count["metal_c"] += 1
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output=b"",
                stderr=b"program_source.metal:3:95: error: use of undeclared identifier 'foo'\n",
            )
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(compiler_mod.subprocess, "run", fake_run)

    with pytest.raises(MetalCompilationError):
        MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
    assert call_count["metal_c"] == 1, (
        f"Real compile error must raise on attempt 1 (no retry), got {call_count['metal_c']} calls"
    )
