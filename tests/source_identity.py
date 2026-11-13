"""Authenticate checkout or installed candidate sources used by regression tests.

This is a source-file check, not a runtime mutation or binary identity proof.
Installed runs still need their runner's full artifact and helper binding.
"""

import hashlib
from pathlib import Path
import sysconfig
from types import ModuleType


def assert_candidate_source(reference_root, *modules):
    reference_root = Path(reference_root).resolve()
    assert modules and modules[0].__name__ == "triton_msl", "package must be first"
    package_root = Path(modules[0].__file__).resolve().parent
    allowed = {(reference_root / "triton_msl").resolve()}
    allowed.update((Path(sysconfig.get_path(kind)) / "triton_msl").resolve() for kind in ("purelib", "platlib"))
    assert package_root in allowed, "unapproved candidate package root"
    receipts = []
    for module in modules:
        assert isinstance(module, ModuleType), "expected module"
        name = module.__name__
        parts = name.split(".")
        assert parts[0] == "triton_msl" and all(p.isidentifier() for p in parts), "module name"
        spec = module.__spec__
        assert spec is not None and spec.name == name, "module spec name"
        relative = (
            Path(*parts) / "__init__.py"
            if spec.submodule_search_locations is not None
            else Path(*parts).with_suffix(".py")
        )
        actual = Path(module.__file__).resolve()
        expected = package_root.parent / relative
        assert actual == expected.resolve(), "mixed candidate module roots"
        assert Path(spec.origin).resolve() == actual, "module spec origin"
        reference = reference_root / relative
        actual_bytes = actual.read_bytes()
        assert actual_bytes == reference.read_bytes(), "candidate source differs from checkout"
        receipts.append(
            dict(
                module=name, path=str(actual), reference=str(reference), sha256=hashlib.sha256(actual_bytes).hexdigest()
            )
        )
    return receipts
