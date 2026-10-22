"""Run directly with python -I; no pytest hooks, frameworks, or GPU imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engineering_checker", ROOT / "scripts/check_engineering.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class IntegrityControls(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((ROOT / "engineering/ledger.json").read_text())
        # Synthetic ledger fixture, independent of future qualification state.
        for row in self.ledger["obligations"] + self.ledger["workloads"]:
            row["state"] = "open"
            row["next_experiment"] = "fixture"
            row.pop("receipt", None)

    def source_tree(self, root):
        (root / "triton_msl").mkdir()
        (root / "tests/fixtures").mkdir(parents=True)
        for name in (
            "setup.py", "pyproject.toml", "MANIFEST.in", "scripts/check_engineering.py",
            "engineering/test_checker.py", "engineering/CONTRACT.md", "AGENTS.md",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture " + name)
        (root / "engineering/ledger.json").write_text(json.dumps(self.ledger))

    def test_open_is_valid_but_not_ready(self):
        errors, pending = checker.check(self.ledger, ROOT, "current")
        self.assertEqual(errors, [])
        self.assertTrue(pending)

    def test_cannot_drop_dimension_or_existing_case(self):
        for key in ("obligations", "workloads"):
            ledger = copy.deepcopy(self.ledger)
            ledger[key].pop(0)
            self.assertTrue(checker.check(ledger, ROOT, "current")[0])

    def test_empty_duplicate_and_hardware_waiver_refused(self):
        for mutation in ("empty", "duplicate", "hardware"):
            ledger = copy.deepcopy(self.ledger)
            if mutation == "empty":
                ledger["workloads"] = []
            elif mutation == "duplicate":
                ledger["workloads"].append(copy.deepcopy(ledger["workloads"][0]))
            else:
                ledger["workloads"][0]["state"] = "hardware_exempt"
            self.assertTrue(checker.check(ledger, ROOT, "current")[0])

    def test_missing_receipt_does_not_pass(self):
        self.ledger["obligations"][0]["state"] = "evidence_ready"
        self.assertTrue(checker.check(self.ledger, ROOT, "current")[0])

    def test_source_and_test_mutations_change_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            before = checker.source_digest(root)
            (root / "triton_msl/bridge.mm").write_text("native source")
            after = checker.source_digest(root)
            self.assertNotEqual(before, after)
            (root / "tests/test_oracle.py").write_text("oracle")
            self.assertNotEqual(after, checker.source_digest(root))

    def test_gate_code_and_contract_mutations_change_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            for name in ("scripts/check_engineering.py", "engineering/test_checker.py",
                         "engineering/CONTRACT.md", "AGENTS.md"):
                with self.subTest(file=name):
                    before = checker.source_digest(root)
                    path = root / name
                    path.write_text(path.read_text() + " changed")
                    self.assertNotEqual(before, checker.source_digest(root))

    def test_ledger_definition_is_bound_progress_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            path = root / "engineering/ledger.json"
            original = copy.deepcopy(self.ledger)
            before = checker.source_digest(root)
            progress = copy.deepcopy(original)
            progress["active_investigation"] = "next task"
            progress["workloads"][0].update(state="evidence_ready", receipt={"path": "receipt", "sha256": "hash"}, next_experiment="done")
            path.write_text(json.dumps(progress, indent=4, sort_keys=True))
            self.assertEqual(before, checker.source_digest(root))
            for change in ("schema", "objective", "id", "baseline", "criteria", "drop"):
                with self.subTest(change=change):
                    value = copy.deepcopy(original)
                    if change in ("schema", "objective"):
                        value[change] = "different"
                    elif change == "drop":
                        value["workloads"].pop()
                    else:
                        value["workloads"][0][change] = "different"
                    path.write_text(json.dumps(value))
                    self.assertNotEqual(before, checker.source_digest(root))

    def test_all_fixture_bytes_and_removal_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            for name in ("opaque.ll", "unknown.extension", "raw"):
                with self.subTest(file=name):
                    path = root / "tests/fixtures" / name
                    before = checker.source_digest(root)
                    path.write_bytes(b"input bytes")
                    added = checker.source_digest(root)
                    self.assertNotEqual(before, added)
                    path.write_bytes(b"different bytes")
                    self.assertNotEqual(added, checker.source_digest(root))
                    path.unlink()
                    self.assertEqual(before, checker.source_digest(root))

    def test_gate_and_fixture_changes_stale_real_bound_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_tree(root)
            fixture = root / "tests/fixtures/opaque.ll"
            fixture.write_text("original")
            binding = lambda p: {"path": str(p), "sha256": checker.digest(p)}
            evidence = {}
            for key in ("protocol", "results", "review"):
                path = root / key
                path.write_text("synthetic " + key)
                evidence[key] = binding(path)
            current = checker.source_digest(root)
            row = self.ledger["obligations"][0]
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(dict(subject=row["id"], outcome="pass", source_digest=current, **evidence)))
            row.update(state="evidence_ready", receipt=binding(receipt))
            (root / "engineering/ledger.json").write_text(json.dumps(self.ledger))
            self.assertEqual(current, checker.source_digest(root))  # no receipt-hash cycle
            self.assertEqual(checker.check(self.ledger, root, current)[0], [])
            for path in (fixture, root / "scripts/check_engineering.py"):
                with self.subTest(file=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original + b" changed")
                    errors, _ = checker.check(self.ledger, root, checker.source_digest(root))
                    self.assertTrue(any("stale" in error for error in errors))
                    path.write_bytes(original)

    def test_receipts_reject_stale_source_tampering_and_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = lambda p: {"path": str(p), "sha256": checker.digest(p)}
            evidence = {}
            for key in ("protocol", "results", "review"):
                path = root / key
                path.write_text("synthetic test fixture: " + key)
                evidence[key] = binding(path)
            row = self.ledger["obligations"][0]
            receipt = dict(subject=row["id"], outcome="pass", source_digest="current", **evidence)
            path = root / "receipt.json"
            def publish(value):
                path.write_text(json.dumps(value))
                row.update(state="evidence_ready", receipt=binding(path))
            publish(receipt)
            errors, pending = checker.check(self.ledger, ROOT, "current")
            self.assertEqual(errors, [])
            self.assertNotIn(row["id"], pending)
            self.assertTrue(checker.check(self.ledger, ROOT, "changed")[0])
            for outcome in ("refused", "skipped", "unexecuted", "fail"):
                publish(dict(receipt, outcome=outcome))
                self.assertTrue(checker.check(self.ledger, ROOT, "current")[0])
            publish(receipt)
            (root / "results").write_text("changed results")
            self.assertTrue(checker.check(self.ledger, ROOT, "current")[0])


if __name__ == "__main__":
    unittest.main()
