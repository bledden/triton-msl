"""Retained, authenticated allocator probe; no framework imports at module scope."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

NAME = "_public_extent_probe"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend_modules():
    return sorted(n for n in sys.modules if n == "triton_msl" or n.startswith("triton_msl."))


def load_probe(source, evidence, early=False):
    source, evidence = Path(source).resolve(), Path(evidence).resolve()
    manifest = evidence / "probe.json"
    if NAME in sys.modules:
        record = json.loads(manifest.read_text())
        probe = sys.modules[NAME]
        origin = Path(probe.__file__).resolve()
        assert origin == Path(record["binary"]).resolve()
        assert digest(origin) == record["binary_sha256"]
        assert digest(source) == record["source_sha256"]
        assert digest(evidence / "host_extent_allocator_probe.c") == record["source_sha256"]
        if early:
            assert not backend_modules(), "backend already imported"
        return probe
    if early:
        assert not backend_modules(), "probe must precede every backend import"
    evidence.mkdir(parents=True, exist_ok=False)
    retained = evidence / "host_extent_allocator_probe.c"
    retained.write_bytes(source.read_bytes())
    import numpy as np

    if early:
        assert not backend_modules(), "NumPy import loaded backend"
    binary = evidence / (NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    command = shlex.split(sysconfig.get_config_var("LDSHARED")) + [
        "-O2",
        "-I" + sysconfig.get_path("include"),
        "-I" + np.get_include(),
        str(retained),
        "-o",
        str(binary),
    ]
    (evidence / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (evidence / "compiler.stdout").write_text(run.stdout)
    (evidence / "compiler.stderr").write_text(run.stderr)
    (evidence / "compiler-result.json").write_text(json.dumps({"returncode": run.returncode}))
    assert run.returncode == 0, run.stderr
    before = backend_modules()
    if early:
        assert not before, "compiler phase loaded backend"
    spec = importlib.util.spec_from_file_location(NAME, binary)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    sys.modules[NAME] = probe
    after = backend_modules()
    if early:
        assert not after, "probe load imported backend"
    manifest.write_text(
        json.dumps(
            dict(
                binary=str(binary),
                binary_sha256=digest(binary),
                source_sha256=digest(retained),
                python=sys.executable,
                backend_modules_before=before,
                backend_modules_after=after,
                loaded_before_backend=early,
            ),
            indent=2,
        )
        + "\n"
    )
    return probe
