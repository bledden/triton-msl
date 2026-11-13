"""Run a cold test program against the parent's selected package, not its cwd.

This selects an import root; it does not certify arbitrary child mutations or
replace the regression controller's installed-artifact binding.
"""

from pathlib import Path
import sys
from types import ModuleType


def package_child_command(program, *arguments):
    import triton_msl

    assert type(program) is str, "child program must be text"
    assert all(type(argument) is str for argument in arguments), "child arguments must be text"
    assert isinstance(triton_msl, ModuleType), "parent package must be a module"
    path = Path(triton_msl.__file__).resolve()
    spec = triton_msl.__spec__
    assert path.name == "__init__.py" and path.parent.name == "triton_msl", "parent package layout"
    assert spec is not None and spec.name == "triton_msl", "parent package spec"
    assert Path(spec.origin).resolve() == path, "parent package origin"
    assert list(triton_msl.__path__) == [str(path.parent)], "parent package search path"
    # Do not import triton_msl here: several programs test its first import.
    prefix = "import sys\nsys.path.insert(0, " + repr(str(path.parent.parent)) + ")\n"
    return [sys.executable, "-I", "-B", "-c", prefix + program, *arguments]
