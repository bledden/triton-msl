"""Controlled binary-cache witnesses, not GPU numerical/compiler emulation claims."""

from pathlib import Path
import importlib
import json
import subprocess
from types import SimpleNamespace

import pytest
import triton  # noqa: F401

from triton_msl.backend import compiler

try:
    contract = importlib.import_module("triton_msl.backend._cache_contract")
except ModuleNotFoundError as exc:
    if exc.name != "triton_msl.backend._cache_contract":
        raise
    # Clean cbff49c has no contract module. Only the controlled toolchain hook
    # needs a namespace there; the old compiler ignores it. All binary calls
    # still execute the baseline compiler, and every assertion is unchanged.
    contract = SimpleNamespace()


@pytest.fixture
def build(monkeypatch, tmp_path):
    from triton_msl.backend import _toolchain_contract as tc

    monkeypatch.setattr(tc, "compiler_process_context", lambda: ("xcrun", str(tmp_path)))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))
    state = {"toolchain": "toolchain-one"}
    calls = []
    # The old code ignores this input; the candidate must consult it before
    # looking up a binary. Real dependency discovery has separate tests.
    monkeypatch.setattr(contract, "toolchain_identity", lambda: state["toolchain"], raising=False)

    def run(cmd, **kwargs):
        assert kwargs["cwd"] == str(tmp_path)
        output = Path(cmd[cmd.index("-o") + 1])
        if "metal" in cmd:
            calls.append(tuple(cmd))
            language = next((v for v in cmd if v.startswith("-std=")), "ir")
            output.write_bytes((state["toolchain"] + ":" + language).encode())
        else:
            assert "metallib" in cmd
            output.write_bytes(b"LIB:" + Path(cmd[cmd.index("metallib") + 1]).read_bytes())
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", run)

    def compile(route="msl", standard="3.1"):
        fn = compiler.MetalBackend.make_metallib if route == "msl" else compiler.MetalBackend.make_metallib_from_llir
        return fn(
            "same controlled source", {"name": "binary_contract"}, compiler.MetalOptions(target_metal_version=standard)
        )

    return compile, state, calls, tmp_path


def test_msl_binary_key_includes_effective_standard(build):
    compile, _, calls, _ = build
    assert compile(standard="3.1") != compile(standard="3.2")
    assert len(calls) == 2


@pytest.mark.parametrize("route", ["msl", "llir"])
@pytest.mark.parametrize("stage", ["metal", "metallib"])
def test_failed_toolchain_stage_is_not_retried_by_diagnostic_spelling(build, monkeypatch, route, stage):
    """Unknown compiler/linker failure is not a proved missing-artifact race.

    Controlled subprocess failures/products only; this does not qualify C++
    computation. A would-succeed second attempt makes a swallowed failure bite.
    """
    from triton_msl.errors import MetalCompilationError, MetalResourceError

    compile, _, _, _ = build
    run_ok = compiler.subprocess.run
    for diagnostic in (b"internal compiler error: verifier failed", b"LLVM ERROR: Broken module found", b""):
        attempts = []

        def run(cmd, **kwargs):
            if stage in cmd:
                attempts.append(tuple(cmd))
                if len(attempts) == 1:
                    raise subprocess.CalledProcessError(1, cmd, b"", diagnostic)
            return run_ok(cmd, **kwargs)

        monkeypatch.setattr(compiler.subprocess, "run", run)
        with pytest.raises(MetalCompilationError) as caught:
            compile(route)
        assert len(attempts) == 1, "a failed toolchain stage was silently retried"
        assert not isinstance(caught.value, MetalResourceError), "unknown failure became prunable"
        if diagnostic:
            assert diagnostic.decode() in str(caught.value), "compiler diagnostic was discarded"


@pytest.mark.parametrize("route", ["msl", "llir"])
def test_binary_key_tracks_toolchain_before_lookup(build, route):
    compile, state, calls, _ = build
    first = compile(route)
    state["toolchain"] = "toolchain-two"
    assert compile(route) != first
    assert len(calls) == 2


def test_msl_and_llvm_are_distinct_compiler_inputs(build):
    compile, _, calls, _ = build
    assert compile("msl") != compile("llir")
    assert len(calls) == 2


@pytest.mark.parametrize("route", ["msl", "llir"])
def test_corrupted_binary_is_not_returned(build, route):
    compile, _, calls, root = build
    first = compile(route)
    (path,) = root.glob("*.metallib")
    path.write_bytes(b"corrupted binary")
    assert compile(route) == first
    assert len(calls) == 2


@pytest.mark.parametrize("route", ["msl", "llir"])
def test_unbound_binary_is_not_a_cache_hit(build, route):
    compile, _, calls, root = build
    first = compile(route)
    # Retain any commit record; old unbound cache has none, which is the bite.
    for path in root.glob("*.meta.json"):
        path.rename(path.with_suffix(".json.retained"))
    assert compile(route) == first
    assert len(calls) == 2


@pytest.mark.parametrize("route", ["msl", "llir"])
def test_valid_binary_product_remains_a_cache_hit(build, route):
    compile, _, calls, _ = build
    first = compile(route)
    assert compile(route) == first
    assert len(calls) == 1


@pytest.mark.parametrize("route", ["msl", "llir"])
@pytest.mark.parametrize("damage", ["schema", "kind", "digest"])
def test_binary_record_must_describe_this_product(build, route, damage):
    compile, _, calls, root = build
    first = compile(route)
    (path,) = root.glob("*.meta.json")
    record = json.loads(path.read_text())
    if damage == "schema":
        record["schema"] = -1
    elif damage == "kind":
        record["kind"] = "msl-source-product"
    else:
        record["metadata"]["binary_sha256"] = "wrong"
    path.write_text(json.dumps(record))
    assert compile(route) == first
    assert len(calls) == 2


@pytest.mark.parametrize("route", ["msl", "llir"])
def test_public_cache_replacement_cannot_be_certified_as_private_output(build, route, monkeypatch):
    compile, _, calls, root = build
    real = compiler.os.replace

    def replacing(source, destination):
        real(source, destination)
        if str(destination).endswith(".metallib"):
            Path(destination).write_bytes(b"foreign concurrent replacement")

    monkeypatch.setattr(compiler.os, "replace", replacing)
    expected = b"LIB:toolchain-one:" + (b"-std=metal3.1" if route == "msl" else b"ir")
    assert compile(route) == expected
    assert len(calls) == 1
    assert next(root.glob("*.metallib")).read_bytes() == b"foreign concurrent replacement"
    # The record binds OUR linker result; the replaced pair cannot hit next time.
    assert compile(route) == expected
    assert len(calls) == 2
