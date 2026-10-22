"""Independent failure-order and truly pooled-alias regression controls."""

import hashlib
from pathlib import Path
import threading

import pytest
from tests.test_parallel_inventory import baseline, candidate


@pytest.mark.parametrize("later", ["small", "unsupported"])
def test_first_failure_matches_serial_across_worker_and_traversal(tmp_path, monkeypatch, later):
    (tmp_path / "a-large").write_bytes(b"a" * 262144)
    if later == "small":
        (tmp_path / "z-small").write_bytes(b"z")
    else:
        import os

        os.mkfifo(tmp_path / "z-fifo")
    first = PermissionError("first large file cannot be opened")
    second = OSError("later small file cannot be opened")
    original = Path.open

    def opening(path, *args, **kwargs):
        if path.name == "a-large":
            raise first
        if path.name == "z-small":
            raise second
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(PermissionError) as serial:
        baseline._tree_manifest(tmp_path)
    assert serial.value is first
    try:
        candidate._tree_manifest(tmp_path, workers=2)
    except BaseException as observed:
        assert {t.ident for t in threading.enumerate()} == before
        assert observed is first, f"candidate raised {type(observed).__name__}: {observed}"
    else:
        pytest.fail("candidate did not refuse")


def test_real_pooled_aliases_wait_for_one_digest(tmp_path, monkeypatch):
    payload = b"a" * 300000
    target = tmp_path / "a-large"
    target.write_bytes(payload)
    for name in ("b-alias", "c-alias", "d-alias"):
        (tmp_path / name).symlink_to(target)
    monkeypatch.setattr(candidate, "_read_only_mount", lambda value: True)
    monkeypatch.setattr(baseline, "_read_only_mount", lambda value: True)
    expected = baseline._tree_manifest(tmp_path, reuse_readonly_aliases=True)
    real_hash = hashlib.sha256
    real_open = Path.open
    opened = []
    hashes = []
    other_open = threading.Event()

    def opening(path, *args, **kwargs):
        opened.append((path.name, threading.current_thread().name))
        if path.name != "a-large":
            other_open.set()
        return real_open(path, *args, **kwargs)

    class Hash:
        def __init__(self):
            hashes.append(threading.current_thread().name)
            self.inner = real_hash()

        def update(self, chunk):
            assert other_open.wait(2), "second pooled alias never opened"
            self.inner.update(chunk)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(hashlib, "sha256", Hash)
    got = candidate._tree_manifest(tmp_path, workers=2, reuse_readonly_aliases=True)
    assert got == expected
    assert len(hashes) == 1
    assert len(opened) == 4
    assert all(name.startswith("triton-msl-inventory") for _, name in opened)
