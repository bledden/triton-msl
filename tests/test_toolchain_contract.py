"""Dependency identity tests; no GPU and no real SDK needed by these fixtures."""

import os
from pathlib import Path

import pytest

from triton_msl.backend import _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    for name in tc._EXTERNAL_SEARCH:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tc, "_snapshot", None)
    monkeypatch.setattr(tc, "_compiler_context", None)
    toolchain, sdk = tmp_path / "Toolchain.xctoolchain", tmp_path / "SDK.sdk"
    toolchain.mkdir()
    sdk.mkdir()
    (toolchain / "selector").write_bytes(b"small selector")
    (toolchain / "actual-compiler").write_bytes(b"actual compiler implementation")
    (toolchain / "runtime.dylib").write_bytes(b"native runtime dependency")
    (sdk / "header.h").write_text("typedef float scalar;\n")
    metadata = {"schema": 1, "sdk_build": "A", "compiler_version": "B", "os_build": "C"}
    roots = {"metal": toolchain, "metallib": toolchain, "sdk": sdk}
    monkeypatch.setattr(tc, "_resolve_toolchain", lambda **kwargs: (roots, metadata))
    return roots, metadata


@pytest.mark.parametrize("name", ["actual-compiler", "runtime.dylib", "selector"])
def test_actual_dependency_bytes_rekey_after_restart(inputs, name, monkeypatch):
    roots, _ = inputs
    first = tc.toolchain_identity()
    (roots["metal"] / name).write_bytes(b"replacement implementation, same label")
    # Installation edits are explicitly a RESTART boundary, not hot-reload.
    monkeypatch.setattr(tc, "_snapshot", None)
    assert tc.toolchain_identity() != first


def test_sdk_header_bytes_are_not_replaced_by_a_version_label(inputs, monkeypatch):
    roots, _ = inputs
    first = tc.toolchain_identity()
    (roots["sdk"] / "header.h").write_text("typedef int scalar;\n")
    monkeypatch.setattr(tc, "_snapshot", None)
    assert tc.toolchain_identity() != first


@pytest.mark.parametrize("key", ["schema", "sdk_build", "compiler_version", "os_build"])
def test_build_contract_fields_participate(inputs, monkeypatch, key):
    _, metadata = inputs
    first = tc.toolchain_identity()
    metadata[key] = "different"
    monkeypatch.setattr(tc, "_snapshot", None)
    assert tc.toolchain_identity() != first


@pytest.mark.parametrize("key", tc._SELECTION)
def test_live_selection_change_refuses_before_reuse(inputs, monkeypatch, key):
    tc.toolchain_identity()
    monkeypatch.setenv(key, "a different selected toolchain")
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        tc.toolchain_identity()


@pytest.mark.parametrize("key", tc._EXTERNAL_SEARCH)
def test_untracked_search_paths_refuse_even_after_a_cached_snapshot(inputs, monkeypatch, key):
    tc.toolchain_identity()
    monkeypatch.setenv(key, "/untracked/inputs")
    with pytest.raises(MetalNonRecoverableError, match="untracked compiler search"):
        tc.toolchain_identity()


def test_repeated_snapshot_does_not_rehash_gigabytes(inputs, monkeypatch):
    calls = []
    real = tc._tree_manifest

    def read(root):
        calls.append(root)
        return real(root)

    monkeypatch.setattr(tc, "_tree_manifest", read)
    first = tc.toolchain_identity()
    assert tc.toolchain_identity() == first
    assert len(calls) == 2  # metal and metallib share the one complete bundle


def test_source_policy_change_is_not_a_toolchain_change(inputs, monkeypatch):
    first = tc.toolchain_identity()
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    assert tc.toolchain_identity() == first  # policy is separately in source_contract


def test_logical_manifest_is_relocatable_and_tracks_cycles(tmp_path):
    roots = [tmp_path / "one", tmp_path / "relocated"]
    for root in roots:
        root.mkdir()
        (root / "header.h").write_text("header contents")
        (root / "recursive").symlink_to(".", target_is_directory=True)
    first = tc._tree_manifest(roots[0])
    assert first == tc._tree_manifest(roots[1])
    assert ("recursive", 0, "", "directory-back-edge") in first
    (roots[1] / "recursive").unlink()
    assert tc._tree_manifest(roots[1]) != first


def test_external_directory_symlink_contents_are_included(tmp_path):
    root, external = tmp_path / "root", tmp_path / "external"
    root.mkdir()
    external.mkdir()
    (external / "dependency").write_bytes(b"first")
    (root / "linked").symlink_to(external, target_is_directory=True)
    first = tc._tree_manifest(root)
    (external / "dependency").write_bytes(b"second")
    assert tc._tree_manifest(root) != first


@pytest.mark.parametrize("damage", ["missing", "empty", "dangling", "special"])
def test_incomplete_inventory_cannot_become_unknown_identity(inputs, monkeypatch, damage, tmp_path):
    roots, _ = inputs
    empty = tmp_path / "incomplete"
    empty.mkdir()
    if damage == "missing":
        empty.rmdir()
    elif damage == "dangling":
        (empty / "broken").symlink_to("absent")
    elif damage == "special":
        os.mkfifo(empty / "pipe")
    roots["sdk"] = empty
    with pytest.raises(MetalNonRecoverableError, match="cannot establish"):
        tc.toolchain_identity()
    assert tc._snapshot is None


def test_selection_change_during_hashing_does_not_publish_a_snapshot(inputs, monkeypatch):
    real = tc._tree_manifest

    def changing(root):
        monkeypatch.setenv("DEVELOPER_DIR", "/changed/during/hash")
        return real(root)

    monkeypatch.setattr(tc, "_tree_manifest", changing)
    with pytest.raises(MetalNonRecoverableError, match="selection changed during"):
        tc.toolchain_identity()
    assert tc._snapshot is None


def test_unavailable_resolver_has_no_shared_unknown_fallback(monkeypatch):
    monkeypatch.setattr(tc, "_compiler_context", None)
    monkeypatch.setattr(tc.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="xcrun is unavailable"):
        tc._resolve_toolchain()


@pytest.mark.parametrize("spelling", ["valid", "no_installed_dir", "external_compiler"])
def test_resolver_proves_actual_compiler_is_in_the_inventory(tmp_path, monkeypatch, spelling):
    monkeypatch.setattr(tc, "_compiler_context", None)
    # This fixture checks selector/InstalledDir containment using text stand-ins.
    # Actual Mach-O external-provider binding has its own pre-fix-biting fixture.
    monkeypatch.setattr(tc, "_toolchain_native_dependencies", lambda roots, entries: {})
    bundle = tmp_path / "Selected.xctoolchain"
    selectors = bundle / "usr/bin"
    actual = bundle / "usr/metal/current/bin"
    sdk = tmp_path / "Selected.sdk"
    for directory in (selectors, actual, sdk):
        directory.mkdir(parents=True)
    for name in ("metal", "metallib"):
        (selectors / name).write_bytes(b"tiny selector")
    (actual / "metal").write_bytes(b"actual large compiler")
    resolver = tmp_path / "xcrun"
    resolver.write_bytes(b"resolver")
    monkeypatch.setattr(tc.shutil, "which", lambda name: str(resolver))
    if spelling == "external_compiler":
        actual = tmp_path / "external"
        actual.mkdir()
        (actual / "metal").write_bytes(b"unbound compiler")
    version = "Metal build1\n" + ("" if spelling == "no_installed_dir" else f"InstalledDir: {actual}\n")
    results = {
        ("--find", "metal"): str(selectors / "metal"),
        ("--find", "metallib"): str(selectors / "metallib"),
        ("metal", "--version"): version,
        ("--show-sdk-path",): str(sdk),
        ("--show-sdk-build-version",): "SDK-build1",
    }

    def query(command, **kwargs):
        if command == ["/usr/sbin/sysctl", "-n", "kern.osversion"]:
            return "OS-build1\n"
        assert command[:3] == [str(resolver), "-sdk", "macosx"]
        return results[tuple(command[3:])]

    monkeypatch.setattr(tc.subprocess, "check_output", query)
    if spelling != "valid":
        with pytest.raises(RuntimeError, match="InstalledDir|outside"):
            tc._resolve_toolchain()
    else:
        roots, metadata = tc._resolve_toolchain()
        assert roots == {"metal": bundle, "metallib": bundle, "sdk": sdk}
        assert metadata["compiler_version"] == "Metal build1"
        assert "InstalledDir" not in str(metadata)


def test_relative_versioned_lookup_uses_installation_not_application_directory(inputs, monkeypatch, tmp_path):
    """No per-reuse filesystem work: compiler CWD is explicitly the inventoried root."""
    roots, metadata = inputs
    entry = (1, "@rpath/comparison.dylib", ("absent-relative", "@rpath/comparison.dylib"))
    row = ("digest", (("DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../F", ("versioned-overrides", (entry,))),))
    metadata["native_providers"] = {"metal-helper:usr/bin/tool": (row,)}
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    first = tc.toolchain_identity()
    assert tc._working_directory_lookups(metadata) == ("@rpath/comparison.dylib",)
    assert tc.toolchain_identity() == first
    shadow = work / "@rpath" / "comparison.dylib"
    shadow.parent.mkdir()
    shadow.write_bytes(b"a newer comparison library dyld would now open")
    assert tc.toolchain_identity() == first
    shadow.unlink()
    shadow.symlink_to(tmp_path / "gone.dylib")  # a dangling link is still an open() input
    assert tc.toolchain_identity() == first
    shadow.unlink()
    shadow.parent.rmdir()
    assert tc.toolchain_identity() == first
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)  # a clean different directory keeps the same proved identity
    assert tc.toolchain_identity() == first
    assert tc._working_directory_lookups({}) == ()
    monkeypatch.setattr(tc.os.path, "lexists", lambda *args: pytest.fail("warm identity did filesystem work"))
    assert tc.toolchain_identity() == first
