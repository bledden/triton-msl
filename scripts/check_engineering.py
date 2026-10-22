"""Read-only engineering ledger integrity checks. Never imports the backend."""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = {
    "correctness", "safety", "capability", "performance", "resources",
    "compatibility", "distribution", "maintainability",
}
STATES = {"open", "blocked", "evidence_ready"}
WARM_CASES = (
    "add", "softmax", "matmul_ieee", "varlen_48_32", "layernorm_1024x1024",
    "quant_decode_512x1024", "fa512_f4_causal0", "fa512_f4_causal1",
    "fa512_f2_causal0", "fa512_f2_causal1", "mm2048_f4", "mm2048_f2",
    "bandwidth_16M", "assert_valid", "mla_canonical", "compiled_mlp",
    "gpt2_small", "attention_forward_backward",
)
REQUIRED_CASES = {
    f"warm/{mode}/{name}" for mode in ("native", "pure") for name in WARM_CASES
} | {
    f"cold/{mode}/{name}" for mode in ("native", "pure")
    for name in ("add", "compiled_mlp", "gpt2_small")
} | {"external/triangle_attention", "external/metal_pipeline"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root):
    files = []
    for directory in ("triton_msl", "tests"):
        for p in (root / directory).rglob("*"):
            fixture = p.is_relative_to(root / "tests/fixtures")
            if p.is_file() and (fixture or p.suffix in {".py", ".c", ".cpp", ".h", ".m", ".mm", ".metal", ".td", ".json", ".ll"} or p.name == "CMakeLists.txt"):
                files.append(p)
    files.extend(root / n for n in (
        "setup.py", "pyproject.toml", "MANIFEST.in", "scripts/check_engineering.py",
        "engineering/test_checker.py", "engineering/CONTRACT.md", "AGENTS.md",
    ))
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(digest(p).encode() + b"\n")
    # Bind definitions, not the mutable evidence index. Including receipt hashes
    # would make each receipt depend on its own hash. These exclusions are status
    # and work tracking only; identifiers, scope, criteria and baselines remain bound.
    ledger = json.loads((root / "engineering/ledger.json").read_text())
    definition = {k: v for k, v in ledger.items() if k != "active_investigation"}
    for key in ("obligations", "workloads"):
        definition[key] = [
            {k: v for k, v in row.items() if k not in {"state", "receipt", "next_experiment"}}
            for row in ledger[key]
        ]
    h.update(b"engineering/ledger-definition-v1\0")
    h.update(json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())
    return h.hexdigest()


def bound_file(record, root):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"]:
        raise ValueError("missing file binding")
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty evidence: {path}")
    if digest(path) != record.get("sha256"):
        raise ValueError(f"evidence hash mismatch: {path}")
    return path.resolve()


def check(ledger, root, current_digest):
    errors, pending = [], []
    if ledger.get("schema") != 1:
        errors.append("unknown ledger schema")
    obligations = ledger.get("obligations", [])
    workloads = ledger.get("workloads", [])
    if not isinstance(obligations, list) or not isinstance(workloads, list):
        return ["obligations and workloads must be lists"], []
    if not all(isinstance(row, dict) for row in obligations + workloads):
        return ["rows must be objects"], []
    if {r.get("id") for r in obligations} != DIMENSIONS:
        errors.append("all eight engineering obligations are required")
    if not workloads:
        errors.append("workload ledger must not be empty")
    missing = REQUIRED_CASES - {r.get("id") for r in workloads}
    if missing:
        errors.append("baseline cases removed: " + ", ".join(sorted(missing)))
    if not ledger.get("active_investigation"):
        errors.append("missing active investigation")
    seen = set()
    for row in obligations + workloads:
        name = row.get("id")
        if not isinstance(name, str) or not name or name in seen:
            errors.append(f"missing or duplicate row id: {name}")
            continue
        seen.add(name)
        state = row.get("state")
        if state not in STATES:
            errors.append(f"{name}: invalid state {state}")
        elif state != "evidence_ready":
            pending.append(name)
            if not row.get("next_experiment"):
                errors.append(f"{name}: missing next experiment")
        else:
            try:
                receipt = json.loads(bound_file(row.get("receipt"), root).read_text())
                if receipt.get("subject") != name or receipt.get("outcome") != "pass":
                    raise ValueError("receipt does not establish this row's positive result")
                if receipt.get("source_digest") != current_digest:
                    raise ValueError("receipt is stale for current source/test/gate definition")
                paths = [bound_file(receipt.get(k), root) for k in ("protocol", "results", "review")]
                if len(set(paths)) != 3:
                    raise ValueError("protocol, results and review must be separate files")
            except (ValueError, OSError, AttributeError, TypeError) as exc:
                errors.append(f"{name}: {exc}")
    return errors, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        ledger = json.loads((root / "engineering/ledger.json").read_text())
        current = source_digest(root)
        errors, pending = check(ledger, root, current)
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        print(f"INVALID: {exc}")
        return 1
    print(f"Source/test/gate digest: {current}")
    for row in ledger.get("obligations", []):
        print(f"{row.get('id')}: {row.get('state')}")
    print(f"Workload rows: {len(ledger.get('workloads', []))}; open/blocked obligations and rows: {len(pending)}")
    print("Active investigation:", ledger.get("active_investigation"))
    for error in errors:
        print("INVALID:", error)
    print("RELEASE EVIDENCE: " + ("HOLD" if errors or pending else "READY FOR HUMAN REVIEW; NOT RELEASE APPROVAL"))
    return 1 if errors else 2 if args.release_check and pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
