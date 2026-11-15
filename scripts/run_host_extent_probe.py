"""One installed-package allocator scenario; retains an early probe and owned worker."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys

NODE = "tests/test_host_extent_gpu.py::test_delegating_custom_handler_gpu_tail_and_lifetime"
LOCK = Path("/private/tmp/triton-msl-project-gpu.lock")


def terminate_owned(child):
    """Signal only the owned session group; boundedly reap the direct child."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Also kill descendants if the direct child exited before them.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def package_inventory(package_root):
    result = {}
    for name, module in list(sys.modules.items()):
        if name == "triton_msl" or name.startswith("triton_msl."):
            origin = Path(module.__file__).resolve()
            assert origin.is_relative_to(package_root), (name, str(origin))
            result[name] = dict(path=str(origin), sha256=hashlib.sha256(origin.read_bytes()).hexdigest())
    assert "triton_msl" in result
    return result


def worker(args):
    work = args.output / "work"
    os.chdir(work)
    # -I excludes PYTHONPATH/user-site; add only the copied tests, never checkout.
    sys.path.insert(0, str(work))
    assert not any(Path(p or '.').resolve().is_relative_to(args.checkout) for p in sys.path)
    support = runpy.run_path(str(work / "tests/host_extent_probe_support.py"))
    probe_dir = args.output / "allocator-probe"
    support["load_probe"](work / "tests/host_extent_allocator_probe.c", probe_dir, early=True)
    os.environ["TRITON_MSL_EXTENT_PROBE_EVIDENCE"] = str(probe_dir)
    # No package import or identity computation was allowed before probe load.
    from triton_msl.profiling.cache_session import CACHE_KEYS, fresh_cache_environment
    os.environ.update(fresh_cache_environment(dict(os.environ, **{
        k: str(args.output / k) for k in CACHE_KEYS})))
    from triton_msl.backend import _cache_contract as contract
    initial = package_inventory(args.package_root)
    helper = contract._validation_native
    assert helper is not None and contract._HOST_EXTENT_READY
    helper_path = Path(helper.__file__).resolve()
    assert helper_path.is_relative_to(args.package_root)
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == args.helper_sha256
    (args.output / "installed-before.json").write_text(json.dumps(initial, indent=2) + "\n")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    class ExactNode:
        def __init__(self):
            self.statuses = []

        def pytest_collection_finish(self, session):
            assert [i.nodeid for i in session.items] == [NODE]
            package_inventory(args.package_root)

        def pytest_runtest_logreport(self, report):
            self.statuses.append((report.when, report.outcome))

    observer = ExactNode()
    command = [NODE, "-c", str(work / "pytest.ini"), "--confcutdir=" + str(work),
               "--import-mode=importlib", "-p", "no:randomly", "--color=no", "--tb=short", "-q", "-ra",
               "--junitxml=" + str(args.output / "probe.xml"),
               "-o", "junit_logging=all", "-o", "junit_log_passing_tests=true"]
    rc = int(pytest.main(command, plugins=[observer]))
    final = package_inventory(args.package_root)
    assert final["triton_msl.backend._validation_native"]["sha256"] == args.helper_sha256
    (args.output / "installed-after.json").write_text(json.dumps(final, indent=2) + "\n")
    (args.output / "node-statuses.json").write_text(json.dumps(observer.statuses) + "\n")
    assert observer.statuses == [("setup", "passed"), ("call", "passed"), ("teardown", "passed")]
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--package-root", required=True, type=Path, help="Exact installed triton_msl directory")
    p.add_argument("--helper-sha256", required=True, help="Expected installed validation helper hash")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.package_root = args.package_root.resolve()
    args.checkout = (args.checkout or Path(__file__).resolve().parents[1]).resolve()
    assert not args.package_root.is_relative_to(args.checkout), "installed package must be outside checkout"
    assert not args.output.is_relative_to(args.checkout), "neutral output must be outside checkout"
    assert len(args.helper_sha256) == 64 and all(c in '0123456789abcdef' for c in args.helper_sha256)
    if os.environ.get("PYTEST_ADDOPTS"):
        p.error("clear PYTEST_ADDOPTS for the fixed selection")
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.output / "work"
    tests = work / "tests"
    tests.mkdir(parents=True)
    copied = {}
    for name in ("__init__.py", "conftest.py", "identity_record_plugin.py", "test_host_extent_gpu.py",
                 "host_extent_probe_support.py", "host_extent_allocator_probe.c"):
        source = args.checkout / "tests" / name
        target = tests / name
        shutil.copyfile(source, target)
        copied["tests/" + name] = hashlib.sha256(target.read_bytes()).hexdigest()
    shutil.copyfile(__file__, work / "runner.py")
    (work / "pytest.ini").write_text("[pytest]\n")
    (args.output / "test-inputs.json").write_text(json.dumps(copied, indent=2) + "\n")
    command = [sys.executable, "-I", "-B", str(work / "runner.py"), "--worker", "--checkout", str(args.checkout),
               "--output", str(args.output), "--package-root", str(args.package_root),
               "--helper-sha256", args.helper_sha256]
    (args.output / "worker-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.output / "raw.txt").open("x") as raw:
            child = subprocess.Popen(command, cwd=work, stdout=raw, stderr=raw, start_new_session=True)
            try:
                rc = child.wait(timeout=args.timeout)
                result = dict(returncode=rc, release_qualified=False)
            except subprocess.TimeoutExpired:
                terminate_owned(child)
                rc = 124
                result = dict(returncode=rc, error="TIMEOUT", release_qualified=False)
            except BaseException as error:
                terminate_owned(child)
                (args.output / "result.json").write_text(json.dumps(dict(
                    error=type(error).__name__, release_qualified=False)) + "\n")
                raise
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
