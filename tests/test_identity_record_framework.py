"""Persisted framework identity records, the documented limitation and the CLI.

CPU only: the loaded-image list and the native guard are modeled, as in the
ambient native contract tests; no GPU device is created.
"""

import mmap
import os
import sys

import pytest

from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_record as rec
from triton_msl.backend import _toolchain_contract as tc
from triton_msl.tools import identity as cli
from identity_record_fixture import (  # noqa: F401
    framework,
    load_body,
    opened_inside,
    outcome,
    record_file,
    records,
    restart_framework,
    restart_toolchain,
    strict_framework,
    toolchain,
    track_opens,
    write_signed,
)
from test_identity_record import TREE_MUTATIONS, _tree_mutation
from test_native_dependency_contract import LC_DYLD_ENVIRONMENT, MH_EXECUTE, _command, _framework, _image


def published(framework, records, monkeypatch):
    strict = strict_framework(monkeypatch)
    assert fw.framework_identity() == strict
    assert outcome("framework") == "published"
    restart_framework()
    return strict


def test_record_digest_equals_strict_and_hit_reads_no_tree_contents(framework, records, monkeypatch):
    strict = published(framework, records, monkeypatch)
    evidence = load_body(record_file(records, "framework"))["evidence"]
    assert sorted(evidence["guard"]) == sorted(
        str(path) for path in (framework.executable, framework.extension, framework.dependency, framework.external)
    )
    assert ["absent", str(framework.absent), None] in evidence["probes"]
    assert ["resolve", str(framework.torch / "lib"), str(framework.torch / "lib")] in evidence["probes"]
    opened = track_opens(monkeypatch)
    assert fw.framework_identity() == strict
    assert outcome("framework") == "hit"
    assert opened_inside(opened, framework.root) == []
    assert opened_inside(opened, framework.external.parent) == []


def test_module_state_after_a_hit_equals_strict_and_later_calls_use_the_memo(framework, records, monkeypatch):
    strict = published(framework, records, monkeypatch)
    monkeypatch.setenv(rec.SWITCH, "0")
    fw.framework_identity()
    strict_snapshot, strict_providers = fw._snapshot, fw._native_guard.providers
    strict_checked = fw._native_guard.checked_names
    monkeypatch.delenv(rec.SWITCH)
    restart_framework()
    assert fw.framework_identity() == strict and outcome("framework") == "hit"
    assert fw._snapshot == strict_snapshot
    assert fw._native_guard.providers == strict_providers
    assert type(fw._native_guard.providers) is frozenset
    assert fw._native_guard.checked_names == strict_checked

    def forbidden(*args, **kwargs):
        pytest.fail("a warm call must take the process memo")

    monkeypatch.setattr(rec, "framework_attempt", forbidden)
    monkeypatch.setattr(fw, "_tree_manifest", forbidden)
    assert fw.framework_identity() == strict


@pytest.mark.parametrize("value", ["0", "off", ""])
def test_strict_and_unknown_switch_values_neither_read_nor_write(framework, records, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("strict mode touched the identity record")

    for name in ("_open_directory", "_read_record", "_write_record", "framework_attempt"):
        monkeypatch.setattr(rec, name, forbidden)
    monkeypatch.setenv(rec.SWITCH, value)
    first = fw.framework_identity()
    restart_framework()
    assert fw.framework_identity() == first
    assert not records.exists()


def test_refresh_overwrites_a_stale_digest(framework, records, monkeypatch):
    strict = published(framework, records, monkeypatch)
    path = record_file(records, "framework")
    write_signed(path, {**load_body(path), "digest": "f" * 64})
    monkeypatch.setenv(rec.SWITCH, "refresh")
    assert fw.framework_identity() == strict and outcome("framework") == "published"
    assert load_body(path)["digest"] == strict


@pytest.mark.parametrize("mutation", TREE_MUTATIONS)
def test_writable_root_changes_are_misses_with_the_strict_digest(framework, records, monkeypatch, mutation):
    published(framework, records, monkeypatch)
    _tree_mutation(mutation, framework.shader)
    expected = strict_framework(monkeypatch)
    assert fw.framework_identity() == expected
    assert outcome("framework") == "published"
    restart_framework()
    assert fw.framework_identity() == expected and outcome("framework") == "hit"


def _change(framework, name):
    if name == "provider_state":
        before = framework.external.stat()
        os.utime(framework.external, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    elif name == "provider_bytes":
        framework.external.write_bytes(_image() + b"external v2")
    elif name == "executable":
        framework.executable.write_bytes(framework.executable.read_bytes() + b"rebuilt")
    elif name == "absent_now_present":
        framework.absent.write_bytes(_image() + b"weak provider appeared")
    elif name == "stdlib":
        (framework.roots["python-stdlib"][0] / "os.py").write_text("changed implementation\n")
    elif name == "platstdlib_provider":
        (framework.roots["python-platstdlib"][0] / "provider.py").write_text("new platform implementation\n")
    elif name == "bytecode_only":
        (framework.torch / "__pycache__" / "other.cpython.pyc").write_bytes(b"ignored by both")
    else:
        raise AssertionError(name)


@pytest.mark.parametrize(
    "change",
    ["provider_state", "provider_bytes", "executable", "absent_now_present", "stdlib", "platstdlib_provider"],
)
def test_evidence_changes_are_misses_with_the_strict_digest(framework, records, monkeypatch, change):
    published(framework, records, monkeypatch)
    _change(framework, change)
    expected = strict_framework(monkeypatch)
    assert fw.framework_identity() == expected
    assert outcome("framework") == "published"


def test_excluded_bytecode_is_outside_both_identities(framework, records, monkeypatch):
    strict = published(framework, records, monkeypatch)
    _change(framework, "bytecode_only")
    assert fw.framework_identity() == strict and outcome("framework") == "hit"


def test_loaded_image_list_selects_a_different_record(framework, records, monkeypatch):
    first = published(framework, records, monkeypatch)
    extra = framework.torch / "_extra.so"
    extra.write_bytes(_image() + b"another extension")
    framework.names.append(str(extra).encode())
    expected = strict_framework(monkeypatch)
    assert expected != first
    assert fw.framework_identity() == expected
    assert outcome("framework") == "published"
    assert len(list(records.glob("framework-*.json"))) == 2
    framework.names[1:] = reversed(framework.names[1:])  # same images, other order
    restart_framework()
    assert fw.framework_identity() == strict_framework(monkeypatch)
    assert len(list(records.glob("framework-*.json"))) == 3


def test_os_build_is_framework_evidence(framework, records, monkeypatch):
    strict = published(framework, records, monkeypatch)
    monkeypatch.setattr(rec, "_os_build", lambda: "another-build")
    assert fw.framework_identity() == strict and outcome("framework") == "published"


def test_selection_change_is_a_different_record(framework, records, monkeypatch):
    published(framework, records, monkeypatch)
    framework.metadata["runtime_environment"] = [["PYTORCH_MPS_FAST_MATH", "1"]]
    expected = strict_framework(monkeypatch)
    assert fw.framework_identity() == expected and outcome("framework") == "published"
    assert len(list(records.glob("framework-*.json"))) == 2


def test_mutation_during_the_strict_inventory_is_never_recorded(framework, records, monkeypatch):
    real = fw._tree_manifest

    def mutating(root, **options):
        result = real(root, **options)
        if root == framework.torch:
            framework.shader.write_text("float shader_function(int x);\n")
        return result

    monkeypatch.setattr(fw, "_tree_manifest", mutating)
    fw.framework_identity()
    assert outcome("framework") == "changed"
    assert not list(records.glob("framework-*.json"))


def test_loaded_image_during_the_strict_inventory_is_never_recorded(framework, records, monkeypatch):
    real = fw._tree_manifest
    extra = framework.torch / "_lazy.so"
    extra.write_bytes(_image() + b"lazily loaded")

    def loading(root, **options):
        # The strict inventory itself then sees the new image and binds it;
        # the evidence collected before the inventory no longer describes it.
        if str(extra).encode() not in framework.names:
            framework.names.append(str(extra).encode())
        return real(root, **options)

    monkeypatch.setattr(fw, "_tree_manifest", loading)
    fw.framework_identity()
    assert outcome("framework") != "published"
    assert not list(records.glob("framework-*.json"))


def _strict_calls(monkeypatch):
    calls = []
    real = fw._tree_manifest

    def counting(root, **options):
        calls.append(root)
        return real(root, **options)

    monkeypatch.setattr(fw, "_tree_manifest", counting)
    return calls


@pytest.mark.parametrize("step", ["guard", "loaded", "selection", "signature", "probe", "image", "os_build"])
def test_failed_record_use_leaves_no_partial_state(framework, records, monkeypatch, step):
    strict = published(framework, records, monkeypatch)
    monkeypatch.setenv(rec.SWITCH, "0")
    fw.framework_identity()
    strict_state = fw._snapshot, fw._native_guard.providers
    monkeypatch.delenv(rec.SWITCH)
    restart_framework()
    if step == "guard":
        factory = fw._LoadedNativeGuard
        adopted = len(framework.guards) + 1  # the next guard is the one built from the record

        def refuse():
            raise fw._NativeSelectionChanged("injected")

        def failing_once(providers):
            guard = factory(providers)
            if len(framework.guards) == adopted:
                guard.verify = refuse
            return guard

        monkeypatch.setattr(fw, "_LoadedNativeGuard", failing_once)
    elif step == "loaded":
        real = fw._loaded_native_images

        def other_during_adoption():
            import sys

            if sys._getframe(1).f_code.co_name == "_adopt_record":
                return real() + (b"/injected/image.so",)
            return real()

        monkeypatch.setattr(fw, "_loaded_native_images", other_during_adoption)
    elif step == "selection":
        calls = []

        def discover():
            calls.append(1)
            roots = dict(framework.roots)
            if len(calls) == 2:  # the re-check after adopting the record
                roots["numpy"] = ()
            return roots, framework.metadata

        monkeypatch.setattr(fw, "_discover_selection", discover)
    else:
        target, name = {
            "signature": (tc, "_stat_signature"),
            "probe": (rec, "_probe_value"),
            "image": (rec.os, "stat"),
            "os_build": (rec, "_os_build"),
        }[step]
        real = getattr(target, name)
        armed = [True]

        def failing(*args, **kwargs):
            if armed[0]:
                armed[0] = False
                raise RuntimeError("injected validation failure")
            return real(*args, **kwargs)

        if step == "image":
            # Fail the first provider stat of the lookup only.
            def failing(path, *args, **kwargs):  # noqa: F811
                if armed[0] and str(path) == str(framework.external):
                    armed[0] = False
                    raise FileNotFoundError(path)
                return real(path, *args, **kwargs)

        monkeypatch.setattr(target, name, failing)
    calls = _strict_calls(monkeypatch)
    assert fw.framework_identity() == strict
    assert outcome("framework") != "hit"
    assert calls, "the strict body must run"
    assert (fw._snapshot, fw._native_guard.providers) == strict_state


def test_failed_adoption_restores_the_previous_snapshot_and_guard(framework, records, monkeypatch):
    published(framework, records, monkeypatch)
    sentinel_snapshot, sentinel_guard = ("old-selection", "old-digest"), object()
    fw._snapshot, fw._native_guard = None, sentinel_guard
    seen = []

    def adopt(roots, metadata, record):
        seen.append((fw._snapshot, fw._native_guard))
        fw._snapshot = sentinel_snapshot
        raise RuntimeError("injected adoption failure")

    def stop(root, **options):
        seen.append((fw._snapshot, fw._native_guard))
        raise KeyboardInterrupt

    monkeypatch.setattr(fw, "_adopt_record", adopt)
    monkeypatch.setattr(fw, "_tree_manifest", stop)
    with pytest.raises(KeyboardInterrupt):
        fw.framework_identity()
    assert seen == [(None, sentinel_guard), (None, sentinel_guard)]
    assert (fw._snapshot, fw._native_guard) == (None, sentinel_guard)


def _versioned_directory(framework):
    """Give the executable a DYLD_VERSIONED_FRAMEWORK_PATH, so the graph lists a directory."""
    directory = framework.root / "versioned"
    directory.mkdir()
    environment = _command(LC_DYLD_ENVIRONMENT, f"DYLD_VERSIONED_FRAMEWORK_PATH={directory}")
    framework.executable.write_bytes(_image(environment, file_type=MH_EXECUTE) + b"interpreter")
    return directory


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_listing_probe_binds_each_childs_kind(framework, records, monkeypatch, kind):
    """A child that becomes a real framework under the same name changes the graph, so the record misses."""
    directory = _versioned_directory(framework)
    child = directory / "Foo.framework"
    if kind == "file":
        child.write_bytes(b"not a bundle")
    else:
        elsewhere = framework.root / "elsewhere"
        elsewhere.mkdir()
        _framework(elsewhere, "Foo")
        child.symlink_to(elsewhere / "Foo.framework", target_is_directory=True)
    original = published(framework, records, monkeypatch)
    probes = load_body(record_file(records, "framework"))["evidence"]["probes"]
    listing = [result for probe, path, result in probes if probe == "listing" and path == str(directory)]
    assert listing == [rec._canonical([["Foo.framework", kind == "symlink", kind == "symlink"]])]
    assert fw.framework_identity() == original and outcome("framework") == "hit"
    restart_framework()
    child.unlink()
    _framework(directory, "Foo")  # the same name, now a real candidate
    expected = strict_framework(monkeypatch)
    assert expected != original
    assert fw.framework_identity() == expected and outcome("framework") == "published"
    restart_framework()
    assert fw.framework_identity() == expected and outcome("framework") == "hit"


def test_provider_off_apfs_is_never_trusted(framework, records, monkeypatch):
    """A native provider whose filesystem is not APFS: a recorded state is not compared, none is published."""
    strict = published(framework, records, monkeypatch)
    real = rec._require_apfs
    # Query every path (not once per device: every fixture path shares one device).
    monkeypatch.setattr(rec, "_require_apfs", lambda entries, filesystems: real(entries, {}))
    monkeypatch.setattr(rec, "_filesystem_type", lambda path: "hfs" if str(path) == str(framework.external) else "apfs")
    path = record_file(records, "framework")
    before = path.read_bytes()
    assert fw.framework_identity() == strict and outcome("framework") == "unavailable"
    assert path.read_bytes() == before  # not replaced either
    restart_framework()
    path.unlink()
    assert fw.framework_identity() == strict and outcome("framework") == "unavailable"
    assert not list(records.glob("framework-*.json"))


@pytest.mark.parametrize("published_first", [False, True])
def test_non_apfs_root_runs_strict_without_reading_or_writing(framework, records, monkeypatch, published_first):
    strict = published(framework, records, monkeypatch) if published_first else strict_framework(monkeypatch)
    queried = []

    def filesystem_type(path):
        # One device holds every fixture root, so the answer is per volume.
        queried.append(str(path))
        return "hfs"

    def forbidden(*args, **kwargs):
        pytest.fail("the identity record directory was touched")

    monkeypatch.setattr(rec, "_filesystem_type", filesystem_type)
    for name in ("_open_directory", "_read_record", "_write_record"):
        monkeypatch.setattr(rec, name, forbidden)
    assert fw.framework_identity() == strict
    assert rec.last_outcome("framework") == ("unavailable", None) and queried
    assert len(list(records.glob("framework-*.json"))) == int(published_first)


def test_direct_call_before_the_cache_contract_is_imported_is_strict(framework, records, monkeypatch):
    strict = strict_framework(monkeypatch)
    monkeypatch.delitem(sys.modules, rec._CACHE_CONTRACT)
    assert fw.framework_identity() == strict
    assert rec.last_outcome("framework") == ("unavailable", None)
    assert rec._CACHE_CONTRACT not in sys.modules
    assert not records.exists()


def test_failed_adoption_is_never_reported_as_a_hit(framework, records, monkeypatch, capsys):
    strict = published(framework, records, monkeypatch)

    def adopt(roots, metadata, record):
        raise RuntimeError("injected adoption failure")

    monkeypatch.setattr(fw, "_adopt_record", adopt)
    monkeypatch.setattr(rec._Attempt, "publish", lambda self, digest, guard=None: None)
    assert fw.framework_identity() == strict and outcome("framework") == "rejected"
    restart_framework()
    monkeypatch.setattr(cli, "_components", lambda: (("framework", fw.framework_identity, restart_framework),))
    assert cli.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "ok (hit)" not in out and "no earlier record was compared: rejected" in out
    assert record_file(records, "framework")  # nothing was deleted on that basis


def test_held_shared_mapping_write_is_missed_by_records_and_caught_by_strict(framework, records, monkeypatch):
    """Decision 13 limitation pin: documented, deliberately not closed.

    A write through a shared mapping that stays open can leave size and times
    unchanged on a writable root. The record path then returns the old digest;
    strict mode and ``verify`` re-read the bytes. Strict-mode coverage of the
    same hazard: test_readonly_toolchain_manifest.py::
    test_open_shared_mapping_is_reread_on_writable_mount.
    """
    original = published(framework, records, monkeypatch)
    target = framework.shader
    before = target.stat()
    with target.open("r+b") as writer, mmap.mmap(writer.fileno(), 0, access=mmap.ACCESS_WRITE) as mapping:
        mapping[:] = bytes(reversed(mapping[:]))
        after = target.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            pytest.skip("this filesystem updated the file's times for a mapped write")
        assert fw.framework_identity() == original and outcome("framework") == "hit"
        assert strict_framework(monkeypatch) != original
        restart_framework()
        monkeypatch.setattr(cli, "_components", lambda: (("framework", fw.framework_identity, restart_framework),))
        assert cli.main(["verify"]) == 1
        assert not list(records.glob("framework-*.json"))


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def both(toolchain, framework, records):
    restart_toolchain()
    restart_framework()
    return toolchain, framework


def test_cli_status_refresh_clear_verify(both, records, monkeypatch, capsys):
    assert cli.main(["status"]) == 0
    assert "mode: record" in capsys.readouterr().out
    assert cli.main(["refresh"]) == 0
    out = capsys.readouterr().out
    assert "toolchain:" in out and "framework:" in out and out.count("published") == 2
    assert os.environ.get(rec.SWITCH) is None  # the command restores the switch
    assert len(list(records.glob("*.json"))) == 2
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert out.count("decodes=True current=True") == 2
    assert cli.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "toolchain: ok (hit)" in out and "framework: ok (hit)" in out
    assert cli.main(["clear"]) == 0
    assert "removed 2 record(s)" in capsys.readouterr().out
    assert not list(records.glob("*.json"))


def test_cli_verify_removes_a_differing_record_and_exits_1(both, records, monkeypatch, capsys):
    assert cli.main(["refresh"]) == 0
    path = record_file(records, "toolchain")
    write_signed(path, {**load_body(path), "digest": "e" * 64})
    capsys.readouterr()
    assert cli.main(["verify"]) == 1
    out = capsys.readouterr().out
    assert "toolchain: DIFFERS" in out and f"removed {path.name}" in out and "framework: ok (hit)" in out
    assert not path.exists()
    assert record_file(records, "framework")
    assert cli.main(["verify"]) == 0  # its record-mode run republishes; strict agrees
    assert "toolchain: ok (no earlier record was compared; published a new record)" in capsys.readouterr().out


def test_cli_clear_on_a_missing_directory_removes_nothing(both, records, capsys):
    assert not records.exists()
    assert cli.main(["clear"]) == 0
    assert "removed 0 record(s)" in capsys.readouterr().out
    assert not records.exists()


def test_cli_verify_without_earlier_records_says_nothing_was_compared(both, records, capsys):
    assert cli.main(["verify"]) == 0
    out = capsys.readouterr().out
    for kind in ("toolchain", "framework"):
        assert f"{kind}: ok (no earlier record was compared; published a new record)" in out
    assert "ok (hit)" not in out
    assert len(list(records.glob("*.json"))) == 2
    assert cli.main(["verify"]) == 0
    assert capsys.readouterr().out.count("ok (hit)") == 2


def test_cli_reports_an_untrusted_directory(both, records, monkeypatch, capsys):
    records.mkdir(mode=0o755)
    os.chmod(records, 0o755)
    assert cli.main(["status"]) == 0
    assert "directory unusable" in capsys.readouterr().out
    assert cli.main(["clear"]) == 2
    assert "untrusted" in capsys.readouterr().err
