"""Regression tests: metallib reads must tolerate a vanished cache file.

Root cause: make_metallib / make_metallib_from_llir read the metallib in two
places that sit OUTSIDE the 3-attempt compile retry:

1. The cache-hit read (TOCTOU): the file can be deleted between os.path.exists
   and open().
2. The post-retry-loop final read: the file is os.replace'd inside the loop but
   can be deleted before the read below the loop.

A concurrent cache clear (another process running `rm -rf ~/.cache/triton_metal`,
or any external deletion) hitting either window raised a bare FileNotFoundError
that escaped — a loud CPU-fallback flake that passes on rerun (never
silent-wrong).

Fix: treat a vanished metallib as a cache miss / retriable transient.  The
cache-hit read recompiles on FileNotFoundError; the final read is moved inside
the retry loop.  Real metal -c / metallib compile errors still raise.
"""
import os
import platform
import subprocess
import threading
import time

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

kernel void test_vanish_k(
    device float* out [[ buffer(0) ]],
    uint tid [[ thread_position_in_grid ]])
{
    out[tid] = float(tid) + 2.0f;
}
"""

# Deliberately invalid MSL — the metal compiler must reject this.
_INVALID_MSL = """\
this is not valid metal {
"""


@requires_metal_compiler
def test_cache_hit_vanish_recovers(tmp_path, monkeypatch):
    """If the cached metallib vanishes mid-read, make_metallib recompiles (no raise).

    Deterministic simulation: compile once to populate the cache, then
    monkeypatch the module's `open` so the FIRST call (the cache-hit read)
    raises FileNotFoundError and subsequent calls delegate to the real builtin.
    The cache-hit path must swallow the FileNotFoundError, fall through to
    recompile, and return valid (non-empty) bytes.
    """
    monkeypatch.setenv("TRITON_METAL_CACHE_DIR", str(tmp_path))

    import builtins
    import triton_metal.backend.compiler as compiler_mod
    from triton_metal.backend.compiler import MetalBackend, MetalOptions

    options = MetalOptions()
    metadata = {"name": "test_vanish_k"}

    # 1) Populate the cache.
    data1 = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
    assert data1 and len(data1) > 0, "make_metallib returned empty bytes on first call"

    # 2) Make the next metallib read raise FileNotFoundError exactly once
    #    (simulating the file vanishing between os.path.exists and open()).
    real_open = builtins.open
    state = {"raised": False}

    def flaky_open(path, *args, **kwargs):
        if (
            not state["raised"]
            and isinstance(path, (str, bytes, os.PathLike))
            and str(path).endswith(".metallib")
        ):
            state["raised"] = True
            raise FileNotFoundError(f"simulated concurrent delete of {path}")
        return real_open(path, *args, **kwargs)

    # compiler.py calls the builtin `open` with no module-level alias.  Injecting
    # `open` into the module's namespace shadows the builtin for code in that
    # module only (module globals are searched before builtins), so other code is
    # unaffected.  monkeypatch.setattr with raising=False adds the name, then
    # removes it on teardown.
    monkeypatch.setattr(compiler_mod, "open", flaky_open, raising=False)

    # 3) Second call hits the cache → first read raises FNF → must recompile and succeed.
    data2 = MetalBackend.make_metallib(_MINIMAL_MSL, dict(metadata), options)
    assert state["raised"], "test harness never triggered the simulated vanish — patch ineffective"
    assert data2 and len(data2) > 0, (
        "make_metallib raised/returned empty after the cached metallib vanished mid-read"
    )
    assert data2 == data1, "recompiled metallib differs from the original (cache key drift?)"


@requires_metal_compiler
def test_concurrent_compile_and_clear(tmp_path):
    """Stress: many compilers + a concurrent cache-clearer must never leak FileNotFoundError.

    8 worker threads compile the SAME kernel ~10x each against a fresh cache dir,
    while one clearer thread continuously deletes the cached .metallib artifacts
    (simulating a concurrent `rm -rf ~/.cache/triton_metal`).  This races the
    clearer directly against BOTH metallib reads — the cache-hit read (TOCTOU
    after os.path.exists) and the post-replace final read.  No FileNotFoundError
    may escape make_metallib, and (because the read-hardening recompiles on a
    vanished file) no other exception may escape either.  Run a few rounds since
    it is a race.

    Scope note: the clearer deletes the cache ARTIFACTS (the content-addressed
    .metallib files) rather than rmtree-ing the whole cache dir.  Deleting the
    dir out from under an in-flight tempfile.mkdtemp() work dir is a *different*
    vector (a write-side race where `metal -c` is handed a source file that was
    deleted before it ran → a genuine CalledProcessError that, per design, must
    still surface as a real compile error).  Hardening that is out of scope for
    this read-only fix; this test targets exactly the read flake being closed.
    """
    cache_dir = str(tmp_path / "stress_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_METAL_CACHE_DIR"] = cache_dir
    try:
        from triton_metal.backend.compiler import MetalBackend, MetalOptions

        options = MetalOptions()
        metadata = {"name": "test_vanish_k"}

        for _round in range(3):
            errors: list[Exception] = []
            lock = threading.Lock()
            stop = threading.Event()

            def worker():
                for _ in range(10):
                    try:
                        data = MetalBackend.make_metallib(
                            _MINIMAL_MSL, dict(metadata), options
                        )
                        assert data and len(data) > 0, "empty metallib bytes"
                    except Exception as exc:  # noqa: BLE001
                        with lock:
                            errors.append(exc)

            def clearer():
                # Continuously delete the cached .metallib artifacts (what a real
                # cache clear targets), racing both metallib reads.  Keep the dir
                # itself so in-flight mkdtemp work dirs survive (the write-side
                # work-dir race is a separate, out-of-scope vector).
                while not stop.is_set():
                    try:
                        for name in os.listdir(cache_dir):
                            if name.endswith(".metallib"):
                                try:
                                    os.unlink(os.path.join(cache_dir, name))
                                except OSError:
                                    pass
                    except OSError:
                        pass
                    # Brief pause so a compile can complete a read between
                    # deletions — aggressive enough to hit the race, not so
                    # relentless that every one of the 3 retry attempts fails.
                    time.sleep(0.001)

            workers = [threading.Thread(target=worker) for _ in range(8)]
            clr = threading.Thread(target=clearer, daemon=True)
            clr.start()
            for t in workers:
                t.start()
            for t in workers:
                t.join()
            stop.set()
            clr.join(timeout=2.0)

            fnf = [e for e in errors if isinstance(e, FileNotFoundError)]
            assert not fnf, (
                f"round {_round}: {len(fnf)} FileNotFoundError escaped make_metallib "
                f"under concurrent cache clear:\n"
                + "\n".join(f"  {type(e).__name__}: {e}" for e in fnf[:5])
            )
            assert errors == [], (
                f"round {_round}: {len(errors)} non-FNF exception(s) during stress:\n"
                + "\n".join(f"  {type(e).__name__}: {e}" for e in errors[:5])
            )
    finally:
        os.environ.pop("TRITON_METAL_CACHE_DIR", None)


@requires_metal_compiler
def test_real_compile_error_still_raises(tmp_path, monkeypatch):
    """Invalid MSL must still raise MetalCompilationError for shader compilation.

    The read-hardening must not have masked real metal -c errors.
    """
    monkeypatch.setenv("TRITON_METAL_CACHE_DIR", str(tmp_path))

    from triton_metal.backend.compiler import MetalBackend, MetalOptions
    from triton_metal.errors import MetalCompilationError

    options = MetalOptions()
    metadata = {"name": "test_bad_msl_vanish_k"}

    with pytest.raises(MetalCompilationError) as exc_info:
        MetalBackend.make_metallib(_INVALID_MSL, dict(metadata), options)

    err_msg = str(exc_info.value).lower()
    assert "shader compilation" in err_msg or "compilation failed" in err_msg, (
        f"Expected a shader-compilation error, got: {err_msg!r}"
    )
    assert "linking" not in err_msg, (
        f"Error should not mention 'linking' for an MSL syntax error, got: {err_msg!r}"
    )
