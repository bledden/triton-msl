"""Persisted toolchain identity records and the record format; CPU only, no GPU.

Every record use must return exactly the digest strict mode returns, and every
doubt must be a silent miss followed by the unchanged strict inventory.
"""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from triton_msl.backend import _identity_record as rec
from triton_msl.backend import _toolchain_contract as tc
from identity_record_fixture import (  # noqa: F401
    load_body,
    opened_inside,
    outcome,
    readonly_bundle,
    record_file,
    records,
    restart_toolchain,
    strict_toolchain,
    toolchain,
    track_opens,
    write_signed,
)
from test_native_dependency_contract import MH_EXECUTE, _image


def published(toolchain, records, monkeypatch):
    strict = strict_toolchain(monkeypatch)
    assert tc.toolchain_identity() == strict
    assert outcome("toolchain") == "published"
    restart_toolchain()
    return strict


def test_record_digest_equals_strict_and_hit_reads_no_tree_contents(toolchain, records, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    path = record_file(records, "toolchain")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(records.stat().st_mode) == 0o700
    body = load_body(path)
    assert body["digest"] == strict and body["kind"] == "toolchain"
    opened = track_opens(monkeypatch)
    assert tc.toolchain_identity() == strict
    assert outcome("toolchain") == "hit"
    anchors = [
        toolchain.bundle / "ToolchainInfo.plist",
        toolchain.bundle / "usr/metal/32023/ToolchainInfo.json",
        toolchain.bundle / "usr/metal/32023/ToolchainInfo.plist",
        toolchain.bundle / "usr/metal/current/ToolchainInfo.json",
        toolchain.bundle / "usr/metal/current/ToolchainInfo.plist",
        toolchain.sdk / "SDKSettings.json",
        toolchain.sdk / "System/Library/CoreServices/SystemVersion.plist",
        toolchain.contents / "version.plist",
    ]
    # Live selection reads the resolver in full, as strict mode does.
    assert [path for _how, _fd, path in opened if path == str(toolchain.resolver)]
    assert opened_inside(opened, toolchain.bundle, anchors) == []
    assert opened_inside(opened, toolchain.sdk, anchors) == []
    assert opened_inside(opened, toolchain.external.parent) == []


def test_module_state_after_a_hit_equals_strict_and_later_calls_use_the_memo(toolchain, records, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    monkeypatch.setenv(rec.SWITCH, "0")
    tc.toolchain_identity()
    strict_state = tc._snapshot, tc._compiler_context
    monkeypatch.delenv(rec.SWITCH)
    restart_toolchain()
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "hit"
    assert (tc._snapshot, tc._compiler_context) == strict_state

    def forbidden(*args, **kwargs):
        pytest.fail("a warm call must take the process memo")

    monkeypatch.setattr(rec, "toolchain_attempt", forbidden)
    monkeypatch.setattr(tc, "_select_toolchain", forbidden)
    assert tc.toolchain_identity() == strict


@pytest.mark.parametrize("value", ["0", "yes", "", "true"])
def test_strict_and_unknown_switch_values_neither_read_nor_write(toolchain, records, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("strict mode touched the identity record")

    for name in ("_open_directory", "_read_record", "_write_record", "toolchain_attempt", "_locator"):
        monkeypatch.setattr(rec, name, forbidden)
    monkeypatch.setenv(rec.SWITCH, value)
    first = tc.toolchain_identity()
    restart_toolchain()
    assert tc.toolchain_identity() == first
    assert not records.exists()


def test_refresh_overwrites_and_a_same_user_forgery_is_trusted_until_then(toolchain, records, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    path = record_file(records, "toolchain")
    body = load_body(path)
    body["digest"] = "0" * 64
    write_signed(path, body)
    # The documented limit: a record forged by the same user is not detected.
    assert tc.toolchain_identity() == "0" * 64 and outcome("toolchain") == "hit"
    restart_toolchain()
    monkeypatch.setenv(rec.SWITCH, "refresh")
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "published"
    restart_toolchain()
    monkeypatch.delenv(rec.SWITCH)
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "hit"
    assert load_body(path)["digest"] == strict


def _rewrite(path, data, *, keep_mtime=False):
    before = path.stat()
    path.write_bytes(data)
    if keep_mtime:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _tree_mutation(name, header):
    old = header.read_bytes()
    same_size = bytes(reversed(old))
    if name == "content_size":
        header.write_bytes(old + b"// more\n")
    elif name == "same_size":
        _rewrite(header, same_size)
    elif name == "same_size_restored_mtime":
        _rewrite(header, same_size, keep_mtime=True)
    elif name == "atomic_replace":
        temporary = header.with_name(".replacement")
        temporary.write_bytes(same_size)
        os.replace(temporary, header)
    elif name == "chmod":
        os.chmod(header, 0o600)
    elif name == "add_file":
        (header.parent / "added.h").write_text("int added;\n")
    elif name == "delete_file":
        (header.parent / "other.h").unlink()
    elif name == "rename_file":
        (header.parent / "other.h").rename(header.parent / "renamed.h")
    elif name == "empty_directory":
        (header.parent / "empty").mkdir()
    elif name == "symlink_retarget":
        alias = header.parent / "alias.h"
        alias.unlink()
        alias.symlink_to("other.h")
    elif name == "directory_back_edge":
        (header.parent / "loop").symlink_to("..", target_is_directory=True)
    else:
        raise AssertionError(name)


TREE_MUTATIONS = [
    "content_size",
    "same_size",
    "same_size_restored_mtime",
    "atomic_replace",
    "chmod",
    "add_file",
    "delete_file",
    "rename_file",
    "empty_directory",
    "symlink_retarget",
    "directory_back_edge",
]


@pytest.mark.parametrize("mutation", TREE_MUTATIONS)
def test_writable_bundle_changes_are_misses_with_the_strict_digest(toolchain, records, monkeypatch, mutation):
    published(toolchain, records, monkeypatch)
    _tree_mutation(mutation, toolchain.header)
    expected = strict_toolchain(monkeypatch)
    assert tc.toolchain_identity() == expected
    assert outcome("toolchain") == "published"  # a miss, then a new record
    restart_toolchain()
    assert tc.toolchain_identity() == expected and outcome("toolchain") == "hit"


def _change(toolchain, name):
    if name == "version_file":
        (toolchain.version / "ToolchainInfo.json").write_text('{"Identifier": "32024"}\n')
    elif name == "version_file_added":
        (toolchain.version / "ToolchainInfo.plist").write_text("<plist>new</plist>\n")
    elif name == "provider_state":
        before = toolchain.external.stat()
        os.utime(toolchain.external, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    elif name == "sdk_anchor":
        (toolchain.sdk / "SDKSettings.json").write_text('{"Version": "27.3"}\n')
    elif name == "sdk_system_version":
        (toolchain.sdk / "System/Library/CoreServices/SystemVersion.plist").write_text("<plist>26C1</plist>\n")
    elif name == "xcode_version":
        (toolchain.contents / "version.plist").write_text("<plist>27.3 27C1</plist>\n")
    elif name == "xcode_seal":
        (toolchain.contents / "_CodeSignature" / "CodeResources").write_text("<plist>new seal</plist>\n")
    elif name == "compiler_version":
        toolchain.replies[("metal", "--version")] = toolchain.replies[("metal", "--version")].replace(
            "32023.922", "32023.923"
        )
    elif name == "sdk_build":
        toolchain.replies[("--show-sdk-build-version",)] = "27C1"
    elif name == "os_build":
        toolchain.os_build = "26C1"
    elif name == "xcrun_bytes":
        toolchain.resolver.write_bytes(_image(file_type=MH_EXECUTE) + b"resolver v2")
    elif name == "absent_now_present":
        toolchain.absent.write_bytes(_image() + b"weak provider appeared")
    else:
        raise AssertionError(name)


@pytest.mark.parametrize(
    "change",
    [
        "version_file",
        "version_file_added",
        "provider_state",
        "sdk_anchor",
        "sdk_system_version",
        "xcode_version",
        "xcode_seal",
        "compiler_version",
        "sdk_build",
        "os_build",
        "xcrun_bytes",
        "absent_now_present",
    ],
)
@pytest.mark.parametrize("readonly", [False, True])
def test_evidence_changes_are_misses_with_the_strict_digest(toolchain, records, monkeypatch, change, readonly):
    if readonly:
        monkeypatch.setattr(tc, "_read_only_mount", lambda value: True)
        monkeypatch.setattr(rec, "_mount_identity", lambda path: [7, True, 100, 200])
    published(toolchain, records, monkeypatch)
    _change(toolchain, change)
    expected = strict_toolchain(monkeypatch)
    assert tc.toolchain_identity() == expected
    assert outcome("toolchain") == "published"


def test_selection_environment_is_a_different_record(toolchain, records, monkeypatch):
    published(toolchain, records, monkeypatch)
    monkeypatch.setenv("TOOLCHAINS", "com.example.other")
    expected = strict_toolchain(monkeypatch)
    assert tc.toolchain_identity() == expected
    assert outcome("toolchain") == "published"
    assert len(list(records.glob("toolchain-*.json"))) == 2


def test_readonly_bundle_uses_mount_root_version_files_and_tree(toolchain, records, readonly_bundle, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    bundle = next(item for item in load_body(record_file(records, "toolchain"))["evidence"]["base"]["toolchain"])
    # The mount identity is kept, and the whole tree's stat signature is bound as for a writable bundle.
    assert bundle["read_only"] is True and bundle["mount"] == [7, True, 100, 200]
    assert bundle["tree"] == tc._stat_signature(toolchain.bundle)
    assert [name for name, digest in bundle["version_files"] if digest is not None] == [
        "ToolchainInfo.plist",
        "usr/metal/32023/ToolchainInfo.json",
        "usr/metal/current/ToolchainInfo.json",
    ]
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "hit"
    restart_toolchain()
    readonly_bundle[0] = [8, True, 100, 200]  # a remount or a different volume
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "published"


def test_sdk_anchor_policy_is_what_records_trust(toolchain, records, readonly_bundle, monkeypatch):
    """Pins the accepted limit: an edit of a non-anchor SDK file is not seen by a record."""
    original = published(toolchain, records, monkeypatch)
    (toolchain.sdk / "usr" / "include" / "stdio.h").write_text("int printf(const char *restrict, ...);\n")
    changed = strict_toolchain(monkeypatch)
    assert changed != original
    assert tc.toolchain_identity() == original and outcome("toolchain") == "hit"


@pytest.mark.parametrize("mutation", ["content_size", "same_size", "add_file", "symlink_retarget"])
def test_readonly_bundle_tree_changes_are_misses_with_the_strict_digest(
    toolchain, records, readonly_bundle, monkeypatch, mutation
):
    # The mount identity is modeled as unchanged; the tree's stat signature still sees the change.
    original = published(toolchain, records, monkeypatch)
    _tree_mutation(mutation, toolchain.header)
    expected = strict_toolchain(monkeypatch)
    assert expected != original
    assert tc.toolchain_identity() == expected and outcome("toolchain") == "published"


def test_readonly_bundle_symlink_to_a_writable_target_is_bound(toolchain, records, readonly_bundle, monkeypatch):
    """A symlink inside the (read-only) bundle resolving outside it: the mount does not cover the target."""
    target = toolchain.root / "writable" / "outside.h"
    target.parent.mkdir()
    target.write_text("float outside(float x);\n")
    (toolchain.version / "lib" / "outside.h").symlink_to(target)
    original = published(toolchain, records, monkeypatch)
    assert tc.toolchain_identity() == original and outcome("toolchain") == "hit"
    restart_toolchain()
    target.write_text("float outside(float y);\n")  # same size, new bytes and times
    expected = strict_toolchain(monkeypatch)
    assert expected != original
    assert tc.toolchain_identity() == expected and outcome("toolchain") == "published"


def test_mutation_during_the_strict_inventory_is_never_recorded(toolchain, records, monkeypatch):
    real = tc._tree_manifest

    def mutating(root, **options):
        result = real(root, **options)
        if Path(root) == toolchain.bundle:
            toolchain.header.write_text("float metal_header(int x);\n")
        return result

    monkeypatch.setattr(tc, "_tree_manifest", mutating)
    tc.toolchain_identity()
    assert outcome("toolchain") == "changed"
    assert not list(records.glob("toolchain-*.json"))


def test_provider_change_after_hashing_is_never_recorded(toolchain, records, monkeypatch):
    real = tc._tree_manifest
    seen = []

    def mutating(root, **options):
        if not seen:
            seen.append(root)
            before = toolchain.external.stat()
            os.utime(toolchain.external, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return real(root, **options)

    monkeypatch.setattr(tc, "_tree_manifest", mutating)
    tc.toolchain_identity()
    assert outcome("toolchain") == "changed"
    assert not list(records.glob("toolchain-*.json"))


@pytest.mark.parametrize(
    "target,name",
    [
        (tc, "_stat_signature"),
        (rec, "_version_files"),
        (rec, "_small_file_digest"),
        (rec, "_optional_stat"),
        (rec, "_probe_value"),
        (rec, "_stat_row"),
        (rec, "decode"),
    ],
)
def test_failing_validation_step_runs_the_strict_body(toolchain, records, monkeypatch, target, name):
    strict = published(toolchain, records, monkeypatch)
    calls = []
    real = tc._tree_manifest

    def counting(root, **options):
        calls.append(root)
        return real(root, **options)

    def failing(*args, **kwargs):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(tc, "_tree_manifest", counting)
    monkeypatch.setattr(target, name, failing)
    assert tc.toolchain_identity() == strict
    assert outcome("toolchain") != "hit" and calls
    assert tc._snapshot == (tuple(os.environ.get(key) for key in tc._SELECTION), strict)


# ------------------------------------------------------------- record hardening


def _damage(path, records, name, toolchain, monkeypatch):
    body = load_body(path)
    data = path.read_bytes()
    if name == "truncated":
        path.write_bytes(data[: len(data) // 2])
    elif name == "empty":
        path.write_bytes(b"")
    elif name == "oversize":
        path.write_bytes(data[:-1] + b" " * (rec._MAX_BYTES + 1) + b"}")
    elif name == "invalid_json":
        path.write_bytes(b"{not json")
    elif name == "nan":
        path.write_bytes(data.replace(b'"created":', b'"created":NaN,"x":', 1))
    elif name == "duplicate_key":
        path.write_bytes(data[:-1] + b',"digest":"' + body["digest"].encode() + b'"}')
    elif name == "extra_key":
        write_signed(path, {**body, "extra": 1})
    elif name == "missing_key":
        write_signed(path, {key: value for key, value in body.items() if key not in ("created", "record_sha256")})
    elif name == "bool_for_int":
        write_signed(path, {**body, "schema": True})
    elif name == "int_for_str":
        write_signed(path, {**body, "kind": 1})
    elif name == "list_for_dict":
        write_signed(path, {**body, "producer": []})
    elif name == "float_for_int":
        write_signed(path, {**body, "created": 1.5})
    elif name == "nested_wrong_type":
        evidence = json.loads(json.dumps(body["evidence"]))
        evidence["base"]["sdk"]["root"][0] = str(evidence["base"]["sdk"]["root"][0])
        write_signed(path, {**body, "evidence": evidence})
    elif name == "wrong_checksum":
        body["digest"] = "1" * 64
        path.write_bytes(rec._canonical(body).encode())
    elif name == "wrong_schema":
        write_signed(path, {**body, "schema": rec.SCHEMA + 1})
    elif name == "wrong_kind":
        write_signed(path, {**body, "kind": "framework"})
    elif name == "wrong_producer":
        write_signed(path, {**body, "producer": {**body["producer"], "version": "0.0.0+other"}})
    elif name == "uppercase_hex":
        write_signed(path, {**body, "digest": body["digest"].upper()})
    elif name == "short_hex":
        write_signed(path, {**body, "digest": body["digest"][:-1]})
    elif name == "symlink":
        target = records.parent / "elsewhere.json"
        target.write_bytes(data)
        os.chmod(target, 0o600)
        path.unlink()
        path.symlink_to(target)
    elif name == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif name == "hard_link":
        os.link(path, records.parent / "second-link.json")
    elif name == "mode_0644":
        os.chmod(path, 0o644)
    elif name == "directory_0755":
        os.chmod(records, 0o755)
    elif name == "directory_symlink":
        moved = records.parent / "real-records"
        records.rename(moved)
        records.symlink_to(moved, target_is_directory=True)
    elif name == "directory_owner":
        uid = os.geteuid()
        monkeypatch.setattr(rec.os, "geteuid", lambda: uid + 1)
    elif name == "other_locator":
        monkeypatch.setenv("TOOLCHAINS", "com.example.other")
        restart_toolchain()
        tc.toolchain_identity()
        monkeypatch.delenv("TOOLCHAINS")
        other = next(item for item in records.glob("toolchain-*.json") if item != path)
        os.replace(other, path)
    else:
        raise AssertionError(name)


HARDENING = [
    "truncated",
    "empty",
    "oversize",
    "invalid_json",
    "nan",
    "duplicate_key",
    "extra_key",
    "missing_key",
    "bool_for_int",
    "int_for_str",
    "list_for_dict",
    "float_for_int",
    "nested_wrong_type",
    "wrong_checksum",
    "wrong_schema",
    "wrong_kind",
    "wrong_producer",
    "uppercase_hex",
    "short_hex",
    "symlink",
    "fifo",
    "hard_link",
    "mode_0644",
    "directory_0755",
    "directory_symlink",
    "directory_owner",
    "other_locator",
]


@pytest.mark.parametrize("damage", HARDENING)
def test_damaged_or_untrusted_record_is_a_silent_miss(toolchain, records, monkeypatch, damage):
    strict = published(toolchain, records, monkeypatch)
    path = record_file(records, "toolchain")
    _damage(path, records, damage, toolchain, monkeypatch)
    restart_toolchain()
    reads = []
    real_read = rec._read_record

    def reading(dir_fd, name):
        result = real_read(dir_fd, name)
        reads.append(result)
        return result

    monkeypatch.setattr(rec, "_read_record", reading)
    assert tc.toolchain_identity() == strict
    assert outcome("toolchain") != "hit"
    if damage in ("directory_0755", "directory_symlink", "directory_owner"):
        assert outcome("toolchain") == "unavailable" and reads == []
    if damage in ("symlink", "fifo", "hard_link", "mode_0644", "empty", "oversize"):
        assert reads in ([], [None])


def test_decode_requires_the_exact_template_at_every_level():
    assert rec._conforms(1, int) and not rec._conforms(True, int) and not rec._conforms(1.0, int)
    assert not rec._conforms("A" * 64, rec._HEX) and rec._conforms("a" * 64, rec._HEX)
    assert not rec._conforms({"a": 1, "b": 2}, {"a": int})
    assert not rec._conforms([1, 2], rec._Row(int))
    assert rec._conforms(None, rec._Optional(int)) and not rec._conforms("x", rec._Optional(int))
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"a":1,"a":2}', object_pairs_hook=rec._unique_keys)


# ----------------------------------------------------------- concurrency, prune


def test_concurrent_publishers_and_readers_never_see_a_torn_record(toolchain, records, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    path = record_file(records, "toolchain")
    body = load_body(path)
    kind, locator = "toolchain", body["locator"]
    variants = [rec.encode(kind, locator, body["evidence"], strict) for _ in range(8)]
    failures, results = [], []
    barrier = threading.Barrier(16)

    def writer(data):
        try:
            barrier.wait()
            for _ in range(25):
                fd = rec._open_directory(str(records), create=True)
                try:
                    rec._write_record(fd, path.name, data)
                finally:
                    os.close(fd)
        except Exception as exc:
            failures.append(exc)

    def reader():
        try:
            barrier.wait()
            for _ in range(50):
                fd = rec._open_directory(str(records))
                try:
                    data = rec._read_record(fd, path.name)
                finally:
                    os.close(fd)
                # None is a miss: a reader whose file was replaced meanwhile sees
                # st_nlink == 0. Bytes that are returned must always decode.
                if data is not None:
                    results.append(rec.decode(data, kind, locator))
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(data,)) for data in variants]
    threads += [threading.Thread(target=reader) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert results and all(item is not None and item["digest"] == strict for item in results)
    assert [item.name for item in records.iterdir()] == [path.name]  # no temporary file left
    restart_toolchain()
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "hit"


def test_concurrent_attempts_publish_one_valid_record(toolchain, records, monkeypatch):
    strict = strict_toolchain(monkeypatch)
    selected = tc._select_toolchain()
    selection = tuple(os.environ.get(key) for key in tc._SELECTION)
    attempts = [rec.toolchain_attempt(os.environ, "record", selection, selected) for _ in range(8)]
    for attempt in attempts:
        assert attempt.lookup() is None
    barrier = threading.Barrier(len(attempts))

    def publish(attempt):
        barrier.wait()
        attempt.publish(strict)

    threads = [threading.Thread(target=publish, args=(attempt,)) for attempt in attempts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert rec.decode(record_file(records, "toolchain").read_bytes(), "toolchain", attempts[0].locator)
    restart_toolchain()
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "hit"


def test_pruning_keeps_the_newest_64_records_per_kind_and_drops_stale_temporaries(tmp_path):
    directory = tmp_path / "records"
    directory.mkdir(mode=0o700)
    # 70 toolchain and 66 framework records, interleaved in age.
    names = [f"{'toolchain' if index % 2 or index >= 132 else 'framework'}-{index:064x}.json" for index in range(136)]
    for index, name in enumerate(names):
        (directory / name).write_text("{}")
        os.utime(directory / name, ns=(index * 10**9, index * 10**9))
    stale = f".toolchain-{1:064x}.json.123.456.{'ab' * 6}.tmp"  # the writer's own pattern
    fresh = f".framework-{2:064x}.json.123.456.{'cd' * 6}.tmp"
    keep = ["notes.json", "toolchain-short.json", f".toolchain-{0:064x}.json.1.2.abc.tmp", "readme.txt", ".x.tmp"]
    for name in [*keep, stale, fresh]:
        (directory / name).write_text("{}")
        os.utime(directory / name, ns=(0, 0))
    now = time.time()
    os.utime(directory / fresh, (now, now))
    (directory / f"framework-{999:064x}.json").symlink_to("notes.json")
    fd = rec._open_directory(str(directory))
    try:
        rec._prune(fd)
    finally:
        os.close(fd)
    remaining = sorted(item.name for item in directory.iterdir())
    toolchain = [name for name in names if name.startswith("toolchain-")]
    framework = [name for name in names if name.startswith("framework-")]
    assert (len(toolchain), len(framework)) == (70, 66)
    expected = toolchain[-64:] + framework[-64:] + keep + [fresh, f"framework-{999:064x}.json"]
    assert sorted(expected) == remaining


def test_status_and_clear_only_touch_record_files(toolchain, records, monkeypatch):
    published(toolchain, records, monkeypatch)
    (records / "notes.json").write_text("{}")
    (records / "toolchain-sub.json").mkdir()
    report = rec.status()
    assert report["mode"] == "record" and report["trusted"] and report["directory"] == str(records)
    [entry] = report["records"]
    assert entry["kind"] == "toolchain" and entry["decodes"] and entry["current"]
    assert entry["producer"] == rec.producer() and type(entry["created"]) is int
    assert rec.clear() == 1
    assert sorted(item.name for item in records.iterdir()) == ["notes.json", "toolchain-sub.json"]


def test_directory_defaults_to_a_per_user_cache_path(monkeypatch, tmp_path):
    monkeypatch.delenv(rec.DIRECTORY, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / ".cache" / "triton_msl" / f"identity-v1-uid{os.geteuid()}"
    assert rec.directory(os.environ) == str(expected)
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "shared"))
    assert rec.directory(os.environ) == str(expected)
    assert [rec.mode({rec.SWITCH: value}) for value in (None, "1", "refresh", "0", "no")] == [
        "record",
        "record",
        "refresh",
        "strict",
        "strict",
    ]
    assert rec.mode({}) == "record"


# ------------------------------------------- filesystem, ACL, directory, import


def _forbid_record_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the identity record directory was touched")

    for name in ("_open_directory", "_read_record", "_write_record"):
        monkeypatch.setattr(rec, name, forbidden)


def test_non_apfs_toolchain_runs_strict_without_reading_or_writing(toolchain, records, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    before = record_file(records, "toolchain").read_bytes()
    queried = []

    def filesystem_type(path):
        queried.append(str(path))
        return "hfs"

    monkeypatch.setattr(rec, "_filesystem_type", filesystem_type)
    _forbid_record_io(monkeypatch)
    assert tc.toolchain_identity() == strict
    assert rec.last_outcome("toolchain") == ("unavailable", None)
    assert queried  # the stub was asked, so this is the filesystem rule and nothing else
    assert record_file(records, "toolchain").read_bytes() == before


def test_non_apfs_toolchain_never_creates_a_record(toolchain, records, monkeypatch):
    monkeypatch.setattr(rec, "_filesystem_type", lambda path: "msdos")
    strict = strict_toolchain(monkeypatch)
    _forbid_record_io(monkeypatch)
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "unavailable"
    assert not records.exists()


def test_filesystem_type_of_this_temporary_directory_is_reported(tmp_path):
    # The real statfs(2) answer: macOS temporary directories live on APFS.
    assert rec._filesystem_type(str(tmp_path)) == "apfs"
    assert rec._filesystem_type("/dev") == "devfs"
    with pytest.raises(OSError):
        rec._filesystem_type(str(tmp_path / "missing"))


def test_stat_signature_reports_every_device_it_relies_on(tmp_path, monkeypatch):
    """A symlink target on another device is reported, so the APFS rule covers it too."""
    root = tmp_path.resolve() / "root"
    root.mkdir()
    (root / "file.h").write_text("int file;\n")
    outside = tmp_path / "outside.h"
    outside.write_text("int outside;\n")
    link = root / "link.h"
    link.symlink_to(outside)
    real = Path.stat

    def stat_with_foreign_target(self, *args, **kwargs):
        status = real(self, *args, **kwargs)
        if self == link:
            fields = ("st_mode", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            return SimpleNamespace(st_dev=status.st_dev + 12345, **{name: getattr(status, name) for name in fields})
        return status

    monkeypatch.setattr(Path, "stat", stat_with_foreign_target)
    devices = {}
    tc._stat_signature(root, devices=devices)
    home = os.stat(root).st_dev
    # The link's own state is on the root's device; what it resolves to is reported by the link's path.
    assert devices == {home: root, home + 12345: link}
    monkeypatch.setattr(rec, "_filesystem_type", lambda path: "apfs" if Path(path) == root else "nfs")
    with pytest.raises(RuntimeError, match="APFS"):
        rec._require_apfs(devices.items(), {})


@pytest.fixture
def acl_parent(tmp_path):
    """A directory carrying an inheritable ACL entry; removed again so cleanup works."""
    parent = tmp_path / "acl"
    parent.mkdir()
    chmod = "/bin/chmod"  # not shutil.which: the toolchain fixture models it
    try:
        result = subprocess.run(
            [chmod, "+a", "everyone allow read,write,file_inherit,directory_inherit", str(parent)],
            capture_output=True,
            timeout=30,
        )
    except OSError:
        pytest.skip("chmod +a is unavailable")
    if result.returncode != 0:
        pytest.skip(f"chmod +a is unavailable: {result.stderr.decode(errors='replace').strip()}")
    try:
        yield parent
    finally:
        subprocess.run([chmod, "-R", "-N", str(parent)], capture_output=True, timeout=30)


def _add_acl(path):
    subprocess.run(["/bin/chmod", "+a", "everyone allow read", str(path)], check=True, capture_output=True, timeout=30)


def test_inherited_acl_makes_the_leaf_untrusted(toolchain, acl_parent, monkeypatch):
    leaf = acl_parent / "records"
    monkeypatch.setenv(rec.DIRECTORY, str(leaf))
    strict = strict_toolchain(monkeypatch)
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "unavailable"
    # The leaf was created, inherited the entry and was refused before any record was written.
    assert leaf.is_dir() and stat.S_IMODE(leaf.stat().st_mode) == 0o700
    fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert rec._has_extended_acl(fd)
    finally:
        os.close(fd)
    with pytest.raises(PermissionError, match="untrusted"):
        os.close(rec._open_directory(str(leaf)))
    assert list(leaf.iterdir()) == []
    restart_toolchain()
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "unavailable"
    assert list(leaf.iterdir()) == []


def test_acl_on_an_existing_directory_or_record_is_a_miss(toolchain, records, monkeypatch):
    strict = published(toolchain, records, monkeypatch)
    path = record_file(records, "toolchain")
    fd = os.open(path, os.O_RDONLY)
    try:
        assert not rec._has_extended_acl(fd)
    finally:
        os.close(fd)
    _add_acl(path)
    reads = []
    real_read = rec._read_record
    monkeypatch.setattr(rec, "_read_record", lambda dir_fd, name: reads.append(real_read(dir_fd, name)) or reads[-1])
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "published"
    assert reads == [None]  # refused, then replaced by a file without the entry
    restart_toolchain()
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "hit"
    restart_toolchain()
    _add_acl(records)
    try:
        reads.clear()
        assert tc.toolchain_identity() == strict and outcome("toolchain") == "unavailable"
        assert reads == []
    finally:
        subprocess.run(["/bin/chmod", "-N", str(records)], check=True, capture_output=True, timeout=30)


def test_relative_record_directory_means_strict(toolchain, records, tmp_path, monkeypatch):
    strict = strict_toolchain(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(rec.DIRECTORY, "relative/records")
    with pytest.raises(RuntimeError, match="absolute"):
        rec.directory(os.environ)
    _forbid_record_io(monkeypatch)
    assert tc.toolchain_identity() == strict and outcome("toolchain") == "unavailable"
    restart_toolchain()
    assert tc.toolchain_identity() == strict
    assert not (tmp_path / "relative").exists()


def test_no_attempt_before_the_cache_contract_is_imported(toolchain, records, monkeypatch):
    strict = strict_toolchain(monkeypatch)
    monkeypatch.delitem(sys.modules, rec._CACHE_CONTRACT)
    selected = tc._select_toolchain()
    selection = tuple(os.environ.get(key) for key in tc._SELECTION)
    assert rec.toolchain_attempt(os.environ, "record", selection, selected) is None
    assert rec.last_outcome("toolchain") == ("unavailable", None)
    assert rec._CACHE_CONTRACT not in sys.modules  # the record module did not import it
    with pytest.raises(RuntimeError, match="imported"):
        rec.producer()
    _forbid_record_io(monkeypatch)
    restart_toolchain()
    # A real call now makes no attempt either: nothing is read or written and the digest is strict.
    assert tc.toolchain_identity() == strict


def test_small_file_digest_reads_only_the_regular_file_it_checked(tmp_path, monkeypatch):
    regular = tmp_path / "anchor.json"
    regular.write_text('{"Version": "1"}\n')
    alias = tmp_path / "alias.json"
    alias.symlink_to(regular)
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo, 0o600)
    (tmp_path / "directory.json").mkdir()
    expected = rec.hashlib.sha256(regular.read_bytes()).hexdigest()
    assert rec._small_file_digest(regular) == expected == rec._small_file_digest(alias)
    assert rec._small_file_digest(tmp_path / "missing.json") is None
    assert rec._small_file_digest(regular / "below") is None
    opened = track_opens(monkeypatch)
    for path in (fifo, tmp_path / "directory.json"):
        with pytest.raises(ValueError, match="unexpected"):
            rec._small_file_digest(path)
    assert opened == []  # neither was opened
    real_open = os.open

    def swapping_open(path, *args, **kwargs):
        if str(path) == str(regular):
            os.replace(alias, regular)  # a different file appears between stat and open
            alias.symlink_to(regular)
            other = tmp_path / "other.json"
            other.write_text('{"Version": "2"}\n')
            os.replace(other, regular)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(ValueError, match="changed"):
        rec._small_file_digest(regular)
