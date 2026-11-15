"""Run directly: exact hosted bundle promotion gates, no package or device imports."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hosted_artifacts", ROOT / "scripts/hosted_artifacts.py")
hosted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosted)


class PromotionControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dist = Path(self.temp.name)
        for name in hosted.expected_files("0.4.0"):
            (self.dist / name).write_bytes(b"original artifact " + name.encode())
        self.record = dict(
            schema=1,
            repository="bledden/triton-msl",
            run_id=123,
            artifact_id=456,
            artifact_name="dist-" + "a" * 40,
            source_sha="a" * 40,
            source_digest="b" * 64,
            version="0.4.0",
            files=hosted.files(self.dist, "0.4.0"),
            qualification=dict(outcome="pass", scope="hosted-native-pure-installed", report_sha256="c" * 64),
        )
        self.run = dict(
            id=123,
            head_sha="a" * 40,
            repository=dict(full_name="bledden/triton-msl"),
            head_repository=dict(full_name="bledden/triton-msl"),
            path=hosted.WORKFLOW,
            head_branch="qualify/0.4.0-test",
            event="push",
            status="completed",
            conclusion="success",
        )
        self.artifact = dict(
            id=456, name=self.record["artifact_name"], expired=False, workflow_run=dict(id=123, head_sha="a" * 40)
        )

    def test_exact_record_and_run(self):
        self.assertIs(hosted.validate_record(self.record, "v0.4.0"), self.record)
        hosted.verify_run(self.record, self.run, self.artifact)
        self.assertEqual(hosted.files(self.dist, "0.4.0"), self.record["files"])

    def test_stale_unqualified_malformed_records_refuse(self):
        mutations = [
            ("schema", True),
            ("run_id", True),
            ("artifact_id", -1),
            ("repository", "foreign/repo"),
            ("source_sha", "main"),
            ("source_digest", "x" * 64),
            ("artifact_name", "dist-latest"),
            ("version", "../evil"),
            ("unexpected", "field"),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.record)
                bad[key] = value
                with self.assertRaises(ValueError):
                    hosted.validate_record(bad, "v0.4.0")
        for key, value in [("outcome", "pending"), ("scope", "smoke-only"), ("report_sha256", "")]:
            with self.subTest(qualification=key):
                bad = copy.deepcopy(self.record)
                bad["qualification"][key] = value
                with self.assertRaises(ValueError):
                    hosted.validate_record(bad, "v0.4.0")
        with self.assertRaises(ValueError):
            hosted.validate_record(self.record, "v0.4.1")

    def test_wrong_run_or_artifact_refuses(self):
        for key, value in [
            ("id", 999),
            ("head_sha", "d" * 40),
            ("conclusion", "failure"),
            ("status", "in_progress"),
            ("event", "pull_request"),
            ("path", ".github/workflows/other.yml"),
            ("head_branch", "main"),
            ("head_branch", None),
            ("head_repository", dict(full_name="fork/repo")),
        ]:
            with self.subTest(run=key):
                bad = copy.deepcopy(self.run)
                bad[key] = value
                with self.assertRaises(ValueError):
                    hosted.verify_run(self.record, bad, self.artifact)
        for key, value in [
            ("id", 999),
            ("name", "dist-latest"),
            ("expired", True),
            ("workflow_run", dict(id=999, head_sha="a" * 40)),
            ("workflow_run", dict(id=123, head_sha="d" * 40)),
        ]:
            with self.subTest(artifact=key, value=value):
                bad = copy.deepcopy(self.artifact)
                bad[key] = value
                with self.assertRaises(ValueError):
                    hosted.verify_run(self.record, self.run, bad)

    def test_changed_or_extra_bytes_cannot_be_promoted(self):
        file = next(self.dist.glob("*.whl"))
        file.write_bytes(file.read_bytes() + b"changed")
        self.assertNotEqual(hosted.files(self.dist, "0.4.0"), self.record["files"])
        (self.dist / "extra.whl").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            hosted.files(self.dist, "0.4.0")

    def test_symlink_artifact_refuses(self):
        file = next(self.dist.glob("*.whl"))
        file.unlink()
        file.symlink_to("/dev/null")
        with self.assertRaises(ValueError):
            hosted.files(self.dist, "0.4.0")

    def test_exact_files_required_in_record(self):
        for mutation in ("missing", "extra", "bad_hash", "bool_size"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.record)
                name = next(iter(bad["files"]))
                if mutation == "missing":
                    del bad["files"][name]
                elif mutation == "extra":
                    bad["files"]["extra.whl"] = bad["files"][name]
                elif mutation == "bad_hash":
                    bad["files"][name]["sha256"] = "latest"
                else:
                    bad["files"][name]["size"] = True
                with self.assertRaises(ValueError):
                    hosted.validate_record(bad, "v0.4.0")

    def test_real_bundle_inspection_and_promotion_gates(self):
        # Reuse the archive fixture, not its checker: promotion loads the real
        # release inspector and reads all three synthetic archives from disk.
        spec = importlib.util.spec_from_file_location("bundle_fixture", ROOT / "tests/test_release_artifacts.py")
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        fixture = fixture_module.ReleaseArtifacts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source, dist = fixture.source, fixture.dist
        (source / "scripts").mkdir()
        (source / "scripts/release_artifacts.py").write_bytes((ROOT / "scripts/release_artifacts.py").read_bytes())
        report = source / "release/HOSTED-QUALIFICATION.txt"
        report.parent.mkdir()
        report.write_text("Synthetic qualification control; no real artifact is qualified.\n")
        record = copy.deepcopy(self.record)
        record["version"] = "0.3.0rc4"
        record["files"] = hosted.files(dist, record["version"])
        record["qualification"]["report_sha256"] = hosted.digest(report)
        hosted.validate_record(record, "v0.3.0rc4")
        # Source-digest calculation has its own checker suite. Keep this control
        # independent of unrelated ledger fixtures while exercising its equality gate.
        with patch.object(hosted, "source_digest", return_value=record["source_digest"]):
            hosted.verify_bundle(record, source, dist)
            for field, value, reason in [
                ("version", "0.4.0", "current version"),
                ("source_digest", "d" * 64, "source changed"),
            ]:
                bad = copy.deepcopy(record)
                bad[field] = value
                with self.assertRaisesRegex(ValueError, reason):
                    hosted.verify_bundle(bad, source, dist)
            report.write_text("changed")
            with self.assertRaisesRegex(ValueError, "report changed"):
                hosted.verify_bundle(record, source, dist)
            record["qualification"]["report_sha256"] = hosted.digest(report)
            wheel = next(dist.glob("*.whl"))
            wheel.write_bytes(wheel.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "qualified bytes changed"):
                hosted.verify_bundle(record, source, dist)
            # Even a refreshed file hash cannot admit a malformed wheel payload.
            fixture.wheels["native"][1]["triton_msl/__init__.py"] = b"wrong implementation"
            fixture.write()
            record["files"] = hosted.files(dist, record["version"])
            with self.assertRaisesRegex(RuntimeError, "payload differs"):
                hosted.verify_bundle(record, source, dist)


if __name__ == "__main__":
    unittest.main()
