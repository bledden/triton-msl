"""Python 3.10 compatibility of the declared ``requires-python = ">=3.10"``.

CPU only. No Metal device, MPS tensor, Triton import or kernel launch.
``driver.py`` imports Triton at module level, so its cleanup helper is loaded
from the source text with ``ast`` and executed in an isolated namespace.
``_pipeline_ownership`` imports only the standard library and is imported directly.
"""

import ast
from pathlib import Path

import pytest

from triton_msl.backend import _pipeline_ownership as ownership

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "triton_msl"
DRIVER = PACKAGE / "backend" / "driver.py"


def _load_release_host_pool_buffers():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"), filename=str(DRIVER))
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_release_host_pool_buffers"]
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DRIVER), "exec"), namespace)
    return namespace["_release_host_pool_buffers"]


release = _load_release_host_pool_buffers()


class NoNoteError(Exception):
    """Stands in for a Python 3.10 exception, which has no ``add_note``."""

    add_note = None


class Pool:
    def __init__(self, failures):
        self.failures = list(failures)
        self.visited = []

    def release(self, *entry):
        self.visited.append(entry)
        failure = self.failures.pop(0)
        if failure is not None:
            raise failure

    def release_scalar(self, buf, size):
        self.release("scalar", buf, size)


def _entries():
    return [(object(), object(), 4), ("scalar", object(), 8), (object(), object(), 16)]


def test_active_error_with_add_note_records_every_cleanup_failure():
    primary = ValueError("primary")
    pool = Pool([RuntimeError("first"), None, RuntimeError("third")])
    entries = _entries()
    with pytest.raises(ValueError) as raised:
        try:
            raise primary
        finally:
            release(pool, entries)
    assert raised.value is primary
    assert primary.__notes__ == [
        "Host buffer cleanup also failed: first",
        "Host buffer cleanup also failed: third",
    ]
    assert pool.visited == entries


def test_active_error_without_add_note_still_propagates_unchanged():
    primary = NoNoteError("primary")
    pool = Pool([RuntimeError("first"), RuntimeError("second"), None])
    entries = _entries()
    with pytest.raises(NoNoteError) as raised:
        try:
            raise primary
        finally:
            release(pool, entries)
    assert raised.value is primary
    assert not hasattr(primary, "__notes__")
    assert raised.value.__context__ is None
    assert pool.visited == entries


def test_active_error_is_seen_inside_an_except_block():
    primary = NoNoteError("primary")
    pool = Pool([RuntimeError("cleanup"), None, None])
    with pytest.raises(NoNoteError) as raised:
        try:
            raise primary
        except NoNoteError:
            release(pool, _entries())
            raise
    assert raised.value is primary
    assert len(pool.visited) == 3


def test_without_active_error_the_first_cleanup_failure_raises_after_all_entries():
    first = RuntimeError("first")
    pool = Pool([None, first, RuntimeError("third")])
    entries = _entries()
    with pytest.raises(RuntimeError) as raised:
        release(pool, entries)
    assert raised.value is first
    assert not hasattr(first, "__notes__")
    assert pool.visited == entries


def test_successful_cleanup_raises_nothing():
    pool = Pool([None, None, None])
    release(pool, _entries())
    assert len(pool.visited) == 3


class _FailingModules(dict):
    """Fails publication after the record exists, reaching the rollback handler."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def get(self, *args):
        raise self.error


class _FailingRollback(ownership.Registry):
    def _undo(self, index, module_key, record):
        raise RuntimeError("rollback")


class _CleanRollback(ownership.Registry):
    def _undo(self, index, module_key, record):
        return None


def _bind(registry, primary):
    registry.modules = _FailingModules(primary)
    index = {}
    with pytest.raises(type(primary)) as raised:
        registry.bind(index, 7, object(), object(), ("proof",), None)
    assert raised.value is primary
    assert index == {}


def test_pipeline_rollback_failure_is_noted_when_add_note_exists():
    primary = LookupError("publication")
    _bind(_FailingRollback(), primary)
    assert primary.__notes__ == ["pipeline publication rollback failed: RuntimeError('rollback')"]


def test_pipeline_rollback_failure_without_add_note_keeps_the_original_error():
    primary = NoNoteError("publication")
    _bind(_FailingRollback(), primary)
    assert not hasattr(primary, "__notes__")


def test_pipeline_clean_rollback_adds_no_note():
    primary = LookupError("publication")
    _bind(_CleanRollback(), primary)
    assert not hasattr(primary, "__notes__")


# Python 3.11+ (and later) standard-library names, by module. Each is flagged as
# an attribute access ``module.name`` or as ``from module import name``.
_NEWER_NAMES = {
    "sys": {"exception", "monitoring", "last_exc"},
    "typing": {
        "Self",
        "Never",
        "LiteralString",
        "reveal_type",
        "assert_never",
        "assert_type",
        "override",
        "TypeVarTuple",
        "Unpack",
        "Required",
        "NotRequired",
        "dataclass_transform",
        "get_overloads",
        "clear_overloads",
    },
    "enum": {"StrEnum", "ReprEnum", "EnumCheck", "verify", "member", "nonmember", "global_enum"},
    "datetime": {"UTC"},
    "asyncio": {"TaskGroup", "timeout", "timeout_at", "Runner", "Barrier"},
    "contextlib": {"chdir"},
    "hashlib": {"file_digest"},
    "operator": {"call"},
    "math": {"cbrt", "exp2", "sumprod", "fma"},
    "logging": {"getLevelNamesMapping", "getHandlerByName", "getHandlerNames"},
    "inspect": {"getmembers_static", "markcoroutinefunction"},
    "re": {"NOFLAG"},
    "itertools": {"batched"},
    "locale": {"getencoding"},
    "copy": {"replace"},
    "warnings": {"deprecated"},
    "os": {"process_cpu_count"},
}
_NEWER_MODULES = {"tomllib", "annotationlib"}
_NEWER_BUILTINS = {"ExceptionGroup", "BaseExceptionGroup", "PythonFinalizationError"}


def _sources():
    files = sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)
    return [*files, ROOT / "setup.py"]


def _findings(path, root=ROOT):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    found = []

    def add(node, what):
        found.append(f"{path.relative_to(root)}:{getattr(node, 'lineno', 0)}: {what}")

    aliases = {}  # local name -> module, from ``import module as local``
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                aliases[alias.asname or top] = alias.name if alias.asname else top
                if alias.name in _NEWER_MODULES:
                    add(node, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in _NEWER_MODULES:
                add(node, f"from {node.module} import")
            for alias in node.names:
                if alias.name in _NEWER_NAMES.get(node.module, ()):
                    add(node, f"from {node.module} import {alias.name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id)
            if module is not None and node.attr in _NEWER_NAMES.get(module, ()):
                add(node, f"{module}.{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_note":
            add(node, "unguarded .add_note(...) call; use getattr(error, 'add_note', None)")
        if isinstance(node, ast.Name) and node.id in _NEWER_BUILTINS:
            add(node, node.id)
        if type(node).__name__ in ("TryStar", "TypeAlias"):
            add(node, type(node).__name__)
        if getattr(node, "type_params", None):
            add(node, "PEP 695 type parameters")
    # Best-effort grammar check: the parser rejects some newer syntax under
    # feature_version, but not every newer construct (Ruff's py310 target covers more).
    try:
        ast.parse(text, filename=str(path), feature_version=(3, 10))
    except SyntaxError as error:
        found.append(f"{path.relative_to(root)}:{error.lineno}: Python 3.10 grammar: {error.msg}")
    return found


def test_package_source_avoids_python_311_only_names_and_syntax():
    found = [item for path in _sources() for item in _findings(path)]
    assert found == []


def test_scanner_detects_the_forms_it_guards(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        "import tomllib\n"
        "from typing import Self\n"
        "import itertools as it\n"
        "error = sys.exception()\n"
        "error.add_note('x')\n"
        "it.batched([], 2)\n"
        "raise ExceptionGroup('g', [])\n"
        "try:\n    pass\nexcept* ValueError:\n    pass\n",
        encoding="utf-8",
    )
    found = _findings(probe, root=tmp_path)
    text = "\n".join(found)
    for expected in (
        "import tomllib",
        "from typing import Self",
        "sys.exception",
        "unguarded .add_note",
        "itertools.batched",
        "ExceptionGroup",
        "TryStar",
        "Python 3.10 grammar",
    ):
        assert expected in text, (expected, found)
