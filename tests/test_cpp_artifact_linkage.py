"""Device-free controls for the explicit installed-extension binary check."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "check_cpp_artifact", Path(__file__).resolve().parents[1] / "scripts" / "check_cpp_artifact.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
PROVIDER = "\t@rpath/libtriton.so (compatibility version 0.0.0, current version 0.0.0)\n"


def test_module_dependencies_admit():
    assert CHECK.check_load_commands("/tmp/image.so:\n" + PROVIDER) == ["@rpath/libtriton.so"]


@pytest.mark.parametrize(
    "dependency",
    [
        "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/Python",
        "@rpath/libpython3.14.dylib",
        "/some directory/libpython3.14.dylib",
    ],
)
def test_python_dependencies_refuse(dependency):
    row = f"\t{dependency} (compatibility version 3.14.0, current version 3.14.0)\n"
    with pytest.raises(ValueError, match="embeds a Python runtime"):
        CHECK.check_load_commands("/tmp/image.so:\n" + PROVIDER + row)


@pytest.mark.parametrize("listing", ["", "image.so:\n", "image.so:\nmalformed\n", PROVIDER])
def test_unproved_listing_refuses(listing):
    with pytest.raises(ValueError):
        CHECK.check_load_commands(listing)


def test_hash_mismatch_never_runs_otool(tmp_path, monkeypatch):
    image = tmp_path / "image.so"
    image.write_bytes(b"wrong artifact")

    def forbidden(*args, **kwargs):
        pytest.fail("otool was reached for the wrong image")

    monkeypatch.setattr(CHECK.subprocess, "check_output", forbidden)
    with pytest.raises(ValueError, match="hash mismatch"):
        CHECK.inspect(image, "0" * 64)
