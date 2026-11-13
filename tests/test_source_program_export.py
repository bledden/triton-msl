"""CPU export controls; no Metal compiler or device is invoked."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
from triton_msl.backend.compiler import MetalBackend

ROOT = Path(__file__).resolve().parents[1]
from triton_msl.tools import export_source_program as tool


@pytest.mark.parametrize("cooperative", [False, True])
def test_source_export_uses_real_proof_and_metadata(tmp_path, monkeypatch, cooperative):
    monkeypatch.setattr(MetalBackend, "make_metallib", lambda *a: pytest.fail("compiler reached"))
    target = tmp_path / "export"
    manifest = tool.export(
        ROOT / "tests/fixtures/source_program/masked.ttgir", target, cooperative=cooperative, source_only=True
    )
    assert manifest == json.loads((target / "manifest.json").read_text())
    assert manifest["metadata"]["block_size"] == (128 if cooperative else 32)
    assert manifest["public_dispatch_enabled"] is False
    assert manifest["files"] == {"kernel.metal": hashlib.sha256((target / "kernel.metal").read_bytes()).hexdigest()}
    assert not (target / "kernel.metallib").exists()
    with pytest.raises(FileExistsError):
        tool.export("does-not-exist", target)


def test_non_source_program_and_failed_proof_never_export(tmp_path, monkeypatch):
    target = tmp_path / "export"
    monkeypatch.setattr(MetalBackend, "make_msl", lambda *a: "unproved")
    monkeypatch.setattr(MetalBackend, "make_metallib", lambda *a: pytest.fail("compiler reached"))
    with pytest.raises(ValueError, match="not a proved"):
        tool.export(ROOT / "tests/fixtures/source_program/masked.ttgir", target)
    assert not target.exists()
