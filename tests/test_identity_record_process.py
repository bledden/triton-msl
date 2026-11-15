"""Identity records across real processes: strict, record miss, record hit, strict again.

Every step is a fresh interpreter with a private TRITON_MSL_IDENTITY_DIR, so a
record published by one step is what the next step finds, exactly as for two
application processes. CPU only: the children compute identities without
importing torch, Metal or objc and create no GPU device; each child asserts
that about itself.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
_CHILD = r"""
import hashlib
import json
import os
import sys

kind, how = sys.argv[1], sys.argv[2]
if how == "backend":
    from triton_msl.backend import _cache_contract

    digest = getattr(_cache_contract, kind + "_identity")()
else:
    # A direct call, before anything imported the cache contract.
    from triton_msl.backend import _framework_contract

    digest = _framework_contract.framework_identity()
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_record as rec

gpu = sorted(name for name in ("torch", "Metal", "objc", "mlx", "triton_msl.backend.driver") if name in sys.modules)
assert not gpu, f"identity computation imported GPU modules: {gpu}"
loaded = [os.fsdecode(raw) for raw in fw._loaded_native_images()]
guard = None if fw._native_guard is None else sorted(str(path) for path in fw._native_guard.providers)
outcome = rec.last_outcome(kind)
print(
    json.dumps(
        {
            "digest": digest,
            "outcome": None if outcome is None else outcome[0],
            "loaded": loaded,
            "guard": guard,
            "cache_contract": "triton_msl.backend._cache_contract" in sys.modules,
        }
    )
)
"""


def _child(kind, how, switch, records, cwd):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TRITON_MSL_IDENTITY_", "DYLD_", "PYTHON"))
    }
    environment.update(
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        TRITON_MSL_IDENTITY_DIR=str(records),
        TRITON_MSL_IDENTITY_RECORD=switch,
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, kind, how],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _sequence(kind, how, tmp_path):
    records = tmp_path / "records"
    return records, [_child(kind, how, switch, records, tmp_path) for switch in ("0", "1", "1", "0")]


def _assert_parity(steps, *, framework):
    strict, miss, hit, again = steps
    assert [step["outcome"] for step in steps] == [None, "published", "hit", None]
    assert strict["digest"] == miss["digest"] == hit["digest"] == again["digest"]
    if framework:
        # The image list is part of the framework identity, and the guard is what the
        # later per-compilation checks enforce: a hit must leave both exactly as strict.
        assert strict["loaded"] == miss["loaded"] == hit["loaded"] == again["loaded"]
        assert strict["guard"] is not None
        assert strict["guard"] == miss["guard"] == hit["guard"] == again["guard"]


def test_framework_identity_on_the_backend_path(tmp_path):
    records, steps = _sequence("framework", "backend", tmp_path)
    _assert_parity(steps, framework=True)
    assert len(list(records.glob("framework-*.json"))) == 1


def test_direct_framework_call_before_the_cache_contract_is_strict(tmp_path):
    records = tmp_path / "records"
    strict = _child("framework", "direct", "0", records, tmp_path)
    used = _child("framework", "direct", "1", records, tmp_path)
    assert used["digest"] == strict["digest"] and used["loaded"] == strict["loaded"]
    assert used["outcome"] == "unavailable" and not used["cache_contract"]
    assert not records.exists()


def _metal_available():
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@pytest.mark.slow
def test_toolchain_identity_on_the_backend_path(tmp_path):
    if not _metal_available():
        pytest.skip("xcrun -sdk macosx metal --version is unavailable")
    records, steps = _sequence("toolchain", "backend", tmp_path)
    _assert_parity(steps, framework=False)
    assert len(list(records.glob("toolchain-*.json"))) == 1
