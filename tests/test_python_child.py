"""CPU controls for cold child selection; no framework or device creation."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
from tests.python_child import package_child_command


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "selected"
    path = root / "triton_msl" / "__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text("MARKER = 'selected'\n")
    module = ModuleType("triton_msl")
    module.__file__ = str(path)
    module.__path__ = [str(path.parent)]
    module.__spec__ = importlib.util.spec_from_file_location("triton_msl", path)
    monkeypatch.setitem(sys.modules, "triton_msl", module)
    return root, module


@pytest.mark.parametrize("shadow", ("cwd", "pythonpath", "both"))
def test_cold_child_ignores_unselected_roots(package, tmp_path, monkeypatch, shadow):
    root, _ = package
    hostile = tmp_path / "hostile"
    (hostile / "triton_msl").mkdir(parents=True)
    (hostile / "triton_msl/__init__.py").write_text("raise AssertionError('wrong package')\n")
    if shadow in ("pythonpath", "both"):
        monkeypatch.setenv("PYTHONPATH", str(hostile))
    program = (
        "assert 'triton_msl' not in sys.modules\n"
        "import triton_msl\n"
        "assert triton_msl.MARKER == 'selected'\n"
        "assert sys.argv == ['-c', 'one', 'two words']\n"
        "assert sys.flags.isolated and sys.dont_write_bytecode\n"
        "print('COLD_SELECTED')\n"
    )
    command = package_child_command(program, "one", "two words")
    done = subprocess.run(
        command, cwd=hostile if shadow != "pythonpath" else tmp_path, text=True, capture_output=True, timeout=20
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == "COLD_SELECTED\n"
    assert command[4].endswith(program)
    assert not list(root.rglob("*.pyc"))


@pytest.mark.parametrize("fault", ("program", "argument", "layout", "spec", "spec_name", "origin", "path"))
def test_unproved_parent_or_command_refuses(package, fault):
    root, module = package
    program, args = "pass", ()
    if fault == "program":
        program = b"pass"
    elif fault == "argument":
        args = (1,)
    elif fault == "layout":
        module.__file__ = str(root / "other.py")
    elif fault == "spec":
        module.__spec__ = None
    elif fault == "spec_name":
        module.__spec__.name = "other"
    elif fault == "origin":
        module.__spec__.origin = str(root / "other.py")
    elif fault == "path":
        module.__path__.append("/unbound")
    with pytest.raises(AssertionError):
        package_child_command(program, *args)
