"""Cache ownership, standalone entrypoints and original-input fuzzer failures."""

from pathlib import Path

import pytest
import torch
import triton

from tests.cache_helpers import fresh_compiler_caches


def _benchmark_module(monkeypatch, tmp_path):
    import importlib.util
    import subprocess
    import tempfile

    path = Path(__file__).parents[1] / "benchmarks/bench_compile_shader.py"
    spec = importlib.util.spec_from_file_location("benchmark_cache_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = subprocess.run
    deletion_attempts = []

    def safe_run(command, *args, **kwargs):
        # Baseline bites must NEVER execute the old shared-home deletion.
        if command[0] == "rm":
            deletion_attempts.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", safe_run)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return module, deletion_attempts


def test_benchmark_owns_distinct_cache_children(monkeypatch, tmp_path):
    import json
    import os

    module, deletion_attempts = _benchmark_module(monkeypatch, tmp_path)
    keys = (
        "TRITON_CACHE_DIR",
        "TRITON_MSL_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CLANG_MODULE_CACHE_PATH",
    )
    parents = {}
    for key in keys:
        parent = tmp_path / key
        parent.mkdir()
        (parent / "canary").write_text("retain")
        monkeypatch.setenv(key, str(parent))
        parents[key] = str(parent)
    marker = tmp_path / "child.json"
    monkeypatch.setenv("CACHE_RECEIPT", str(marker))
    module._INNER = (
        "import inspect,json,os,triton_msl\nfrom pathlib import Path\n"
        f"assert Path(triton_msl.__file__).resolve() == Path({str(Path(__file__).resolve().parents[1] / 'triton_msl/__init__.py')!r})\n"
        "def source_witness(): return 7\n"
        "assert 'return 7' in inspect.getsource(source_witness)\n"
        f"parents={parents!r}\n"
        "roots={k:Path(os.environ[k]) for k in parents}\n"
        "assert all(p.parent==Path(parents[k]) and p!=Path(parents[k]) for k,p in roots.items())\n"
        "assert all(not list(p.iterdir()) for p in roots.values())\n"
        "for p in roots.values(): (p/'session-marker').write_text('retain')\n"
        "Path(os.environ['CACHE_RECEIPT']).write_text(json.dumps({k:str(p) for k,p in roots.items()}))\n"
        "print(json.dumps({k:{'ms':1.0,'gbps':2.0} for k in ['vector_add','elementwise','softmax','reduction']}))\n"
    )
    # Exercise an entry point launched outside the repository.
    monkeypatch.chdir(tmp_path)
    sessions = []
    for flag in ("1", "0"):
        assert module.run_one_flag(flag)["softmax"]["ms"] == 1
        sessions.append(json.loads(marker.read_text()))
    assert not deletion_attempts
    assert all(sessions[0][k] != sessions[1][k] for k in keys)
    assert all((Path(root) / "session-marker").read_text() == "retain" for s in sessions for root in s.values())
    assert all((Path(p) / "canary").read_text() == "retain" and os.environ[k] == p for k, p in parents.items())


def test_benchmark_retains_success_failure_and_timeout_output(monkeypatch, tmp_path, capsys):
    import json
    import subprocess

    module, deletion_attempts = _benchmark_module(monkeypatch, tmp_path)
    module._INNER = (
        "import json,sys\nprint('visible-stdout')\nprint('visible-stderr',file=sys.stderr)\n"
        "print(json.dumps({k:{'ms':1.0,'gbps':2.0} for k in ['vector_add','elementwise','softmax','reduction']}))\n"
    )
    module.run_one_flag("1")
    output = capsys.readouterr()
    assert "visible-stdout" in output.out and "visible-stderr" in output.err
    assert not deletion_attempts
    module._INNER = "import sys;print('failure-stdout');print('failure-stderr',file=sys.stderr);sys.exit(7)"
    with pytest.raises(RuntimeError, match="7"):
        module.run_one_flag("0")
    output = capsys.readouterr()
    assert "failure-stdout" in output.out and "failure-stderr" in output.err
    module._INNER = (
        "import json,sys\nprint('invalid-stdout')\nprint('invalid-stderr',file=sys.stderr)\n"
        "print(json.dumps(['vector_add','elementwise','softmax','reduction']))\n"
    )
    with pytest.raises(ValueError, match="benchmark rows"):
        module.run_one_flag("0")
    output = capsys.readouterr()
    assert "invalid-stdout" in output.out and "invalid-stderr" in output.err

    def timeout(command, **kwargs):
        kwargs["stdout"].write("timeout-stdout\n")
        kwargs["stderr"].write("timeout-stderr\n")
        raise subprocess.TimeoutExpired(command, 300)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_one_flag("1")
    output = capsys.readouterr()
    assert "timeout-stdout" in output.out and "timeout-stderr" in output.err
    receipts = [json.loads(p.read_text()) for p in tmp_path.glob("compile-shader-*/exit.json")]
    assert {r["classification"] for r in receipts} == {
        "UNQUALIFIED_TIMING_WITH_STDERR",
        "PROCESS_FAILED",
        "INVALID_RESULT",
        "TIMEOUT",
    }
    assert all(r["release_claim_qualified"] is False for r in receipts)


def test_shared_cache_allocator_preserves_input_and_refuses_bad_parent(tmp_path):
    from triton_msl.profiling.cache_session import fresh_cache_environment

    parent = tmp_path / "caller"
    parent.mkdir()
    (parent / "canary").write_text("retain")
    env = {"TORCHINDUCTOR_CACHE_DIR": str(parent), "KEEP": "unchanged"}
    a = fresh_cache_environment(env, keys=("TORCHINDUCTOR_CACHE_DIR",))
    b = fresh_cache_environment(env, keys=("TORCHINDUCTOR_CACHE_DIR",))
    assert a != b and a["KEEP"] == b["KEEP"] == "unchanged"
    assert all(Path(x["TORCHINDUCTOR_CACHE_DIR"]).parent == parent for x in (a, b))
    assert env == {"TORCHINDUCTOR_CACHE_DIR": str(parent), "KEEP": "unchanged"}
    assert (parent / "canary").read_text() == "retain"
    with pytest.raises(FileExistsError):
        fresh_cache_environment({"TRITON_CACHE_DIR": str(parent / "canary")}, keys=("TRITON_CACHE_DIR",))


@triton.jit
def _unused_kernel():
    pass


def test_cold_cache_rotation_preserves_existing_directories(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys
    import tempfile

    original = {}
    for key in ("TRITON_MSL_CACHE_DIR", "TRITON_CACHE_DIR"):
        directory = tmp_path / key
        directory.mkdir()
        (directory / "canary").write_text("owned by someone else")
        monkeypatch.setenv(key, str(directory))
        original[key] = directory
    mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        "triton_msl.profiling.cache_session.tempfile.mkdtemp", lambda **kwargs: mkdtemp(dir=tmp_path, **kwargs)
    )
    roots = []
    for _ in range(2):
        _unused_kernel.device_caches[999] = "stale executable"
        root = fresh_compiler_caches({"kernel": _unused_kernel})
        roots.append(root)
        assert not _unused_kernel.device_caches
        for key, suffix in (("TRITON_MSL_CACHE_DIR", "msl"), ("TRITON_CACHE_DIR", "triton")):
            assert Path(os.environ[key]) == root / suffix
            assert list((root / suffix).iterdir()) == []
    assert roots[0] != roots[1] and all(p.is_dir() for p in roots)
    assert all((p / "canary").read_text() == "owned by someone else" for p in original.values())
    # The shared helper must also load from the documented direct-script entry.
    # Zero seeds exercise imports/CLI only, not an additional GPU fuzz sweep.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("test_fuzz_reduce.py")), "0"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "reduce fuzz: {} over 0 seed(s)" in run.stdout


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS tensor setup needed")
def test_fuzzer_missing_file_keeps_original_seed_and_failure(monkeypatch):
    from tests import test_fuzz_matmul as mm
    from tests import test_fuzz_reduce as red

    seed_calls = []
    manual_seed = torch.manual_seed

    def seed(value):
        seed_calls.append(value)
        return manual_seed(value)

    monkeypatch.setattr(torch, "manual_seed", seed)
    cases = [
        (mm, "_k_simple", lambda: mm._run_cell("simple", 16, 16, 16, torch.float32, 277)),
        (
            mm,
            "_k_single_strided",
            lambda: mm._run_strided_cell("single", 16, 16, 16, torch.float32, "rowmaj", "rowmaj", "rowmaj", 277),
        ),
        (mm, "_k_batched", lambda: mm._run_batched_cell(2, 16, 16, 16, torch.float32, 277)),
        (red, "_r1d_sum", lambda: red._run_cell("1d_sum", (16,), torch.float32, 277)),
    ]
    failures = []
    for module, name, run in cases:
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("unexpected retry")
            raise FileNotFoundError("original missing cache artifact")

        with monkeypatch.context() as patch:
            # On a pre-fix tree this prevents the old handler deleting real caches.
            patch.setattr(module, "_clear_cache", lambda: None, raising=False)
            patch.setattr(getattr(module, name), "run", fail)
            seed_calls.clear()
            result = run()
            if (
                result != ("crash:FileNotFoundError", "original missing cache artifact")
                or calls != [1]
                or seed_calls != [277]
            ):
                failures.append((name, result, list(calls), list(seed_calls)))
    assert not failures, failures


def test_pytest_sessions_own_cold_inductor_caches(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    import sys

    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(Path(__file__).with_name("conftest.py"), project / "conftest.py")
    # The conftest registers this plugin by module name; it travels with it.
    shutil.copyfile(Path(__file__).with_name("identity_record_plugin.py"), project / "identity_record_plugin.py")
    # Capture cache state DURING collection, then assert it survives session setup.
    # All deletion targets on the unpatched side are disposable owned canaries.
    (project / "test_probe.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from triton_msl.profiling import cache_session\n"
        "assert Path(cache_session.__file__).resolve() == Path(os.environ['EXPECTED_HELPER'])\n"
        "root = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])\n"
        "cold_at_collection = not list(root.iterdir())\n"
        "def test_owned():\n"
        "    from torch._inductor.runtime.cache_dir_utils import cache_dir\n"
        "    assert Path(cache_dir()) == root\n"
        "    parent = os.environ.get('EXPECTED_PARENT')\n"
        "    if parent:\n"
        "        assert root.parent == Path(parent) and root != Path(parent)\n"
        "        assert (Path(parent) / 'canary').read_text() == 'retain'\n"
        "    assert cold_at_collection\n"
        "    (root / 'session-marker').write_text('retain')\n"
        "    Path(os.environ['RECEIPT']).write_text(json.dumps(str(root)))\n"
    )
    parent = tmp_path / "caller-cache"
    parent.mkdir()
    (parent / "canary").write_text("retain")
    roots = []
    for index, inherited in enumerate((False, True, True)):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("TORCHINDUCTOR_CACHE_DIR", "PYTHONPATH", "PYTEST_ADDOPTS", "EXPECTED_PARENT")
        }
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        # The copied conftest's shared helper must come from this exact candidate,
        # not an unrelated globally installed package. Assert it inside collection.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        env["EXPECTED_HELPER"] = str(Path(__file__).resolve().parents[1] / "triton_msl/profiling/cache_session.py")
        env["TMPDIR"] = str(tmp_path)
        receipt = tmp_path / f"session-{index}.json"
        env["RECEIPT"] = str(receipt)
        if inherited:
            env["TORCHINDUCTOR_CACHE_DIR"] = str(parent)
            env["EXPECTED_PARENT"] = str(parent)
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--confcutdir", str(project), str(project / "test_probe.py")],
            cwd=project,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        roots.append(Path(json.loads(receipt.read_text())))
    assert len(set(roots)) == 3
    assert all((p / "session-marker").read_text() == "retain" for p in roots)
    assert (parent / "canary").read_text() == "retain"
