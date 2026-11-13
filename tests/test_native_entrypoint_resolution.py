"""Only non-entry regular children may avoid resolving their known parent again."""

import os
from pathlib import Path

import pytest

from triton_msl.backend import _framework_contract as fc
from test_native_dependency_contract import _image


@pytest.fixture
def layout(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    executable = tmp_path / "Python"
    executable.write_bytes(_image())
    roots = {"package": (root,), "python-executable": (executable,)}
    return root, executable, roots


def test_only_ordinary_files_avoid_full_resolution(layout, monkeypatch):
    root, executable, roots = layout
    folder = root / "sub"
    folder.mkdir()
    ordinary, extension = folder / "source.py", folder / "extension.so"
    ordinary.write_bytes(b"source remains a tree input")
    extension.write_bytes(_image())
    real = Path.resolve
    calls = []

    def resolve(path, *args, **kw):
        calls.append((path, kw.get("strict")))
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert fc._native_entrypoints(roots) == [extension, executable]
    assert (ordinary, True) not in calls
    for path in (root, folder, extension, executable):
        assert (path, True) in calls


@pytest.mark.parametrize(
    "alias_suffix,target_suffix,selected",
    [(".py", ".so", True), (".so", ".py", False), ("", ".so", True), (".so", ".so", True)],
)
def test_symlink_selection_uses_resolved_target(layout, tmp_path, alias_suffix, target_suffix, selected):
    root, executable, roots = layout
    target = tmp_path / ("external" + target_suffix)
    target.write_bytes(_image())
    (root / ("alias" + alias_suffix)).symlink_to(target)
    assert fc._native_entrypoints(roots) == ([target] if selected else []) + [executable]


def test_directory_aliases_cycles_and_seed_order(layout):
    root, executable, roots = layout
    folder = root / "z"
    folder.mkdir()
    a, b = folder / "a.so", folder / "b.so"
    a.write_bytes(_image())
    b.write_bytes(_image())
    (folder / "back").symlink_to(root, target_is_directory=True)
    (root / "first").symlink_to(folder, target_is_directory=True)
    assert fc._native_entrypoints(roots) == [a, b, executable]


def test_hard_link_entrypoints_keep_distinct_paths(layout):
    root, executable, roots = layout
    a, b = root / "a.so", root / "b.so"
    a.write_bytes(_image())
    os.link(a, b)
    assert fc._native_entrypoints(roots) == [a, b, executable]


@pytest.mark.parametrize("relative", [False, True])
def test_root_alias_always_resolves(layout, tmp_path, monkeypatch, relative):
    root, executable, roots = layout
    extension = root / "entry.so"
    extension.write_bytes(_image())
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    roots["package"] = (Path("alias") if relative else alias,)
    assert fc._native_entrypoints(roots) == [extension, executable]


def test_explicit_torch_library_still_resolves(layout, monkeypatch):
    root, executable, _ = layout
    library = root / "lib" / "libtorch_global_deps.dylib"
    library.parent.mkdir()
    library.write_bytes(_image())
    real = Path.resolve
    calls = []

    def resolve(path, *args, **kw):
        calls.append(path)
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert fc._native_entrypoints({"python-executable": (executable,), "torch": (root,)}) == [executable, library]
    assert library in calls


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_fresh_child_lstat_failure_is_not_ignored(layout, monkeypatch, failure):
    root, _, roots = layout
    ordinary = root / "source.py"
    ordinary.write_bytes(b"ordinary")
    real = Path.lstat

    def lstat(path, *args, **kw):
        if path == ordinary:
            raise failure("child unavailable")
        return real(path, *args, **kw)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(failure, match="child unavailable"):
        fc._native_entrypoints(roots)


def test_child_deleted_after_listing_still_refuses(layout, monkeypatch):
    root, _, roots = layout
    ordinary = root / "source.py"
    ordinary.write_bytes(b"ordinary")
    real = Path.iterdir

    def listing(path):
        entries = list(real(path))
        if path == root:
            ordinary.unlink()
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", listing)
    with pytest.raises(FileNotFoundError):
        fc._native_entrypoints(roots)


def test_child_replaced_by_symlink_after_listing_is_resolved(layout, tmp_path, monkeypatch):
    root, executable, roots = layout
    ordinary = root / "source.py"
    ordinary.write_bytes(b"ordinary")
    target = tmp_path / "external.so"
    target.write_bytes(_image())
    real = Path.iterdir

    def listing(path):
        entries = list(real(path))
        if path == root:
            ordinary.unlink()
            ordinary.symlink_to(target)
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", listing)
    assert fc._native_entrypoints(roots) == [target, executable]


@pytest.mark.parametrize("kind", ["dangling", "loop", "bad_native"])
def test_existing_refusals_remain(layout, kind):
    root, _, roots = layout
    if kind == "dangling":
        (root / "ordinary.py").symlink_to("missing")
    elif kind == "loop":
        (root / "ordinary.py").symlink_to("ordinary.py")
    else:
        (root / "entry.so").write_bytes(b"unproved native contents")
    with pytest.raises((OSError, RuntimeError)):
        fc._native_entrypoints(roots)


def test_excluded_names_keep_original_behavior(layout):
    root, executable, roots = layout
    for directory in ("site-packages", "dist-packages", "__pycache__"):
        folder = root / directory
        folder.mkdir()
        (folder / "invalid.so").write_bytes(b"unproved")
    (root / "bad.pyc").symlink_to("missing")
    (root / "bad.pyo").symlink_to("missing")
    assert fc._native_entrypoints({"python-stdlib": (root,), "python-executable": (executable,)}) == [executable]
