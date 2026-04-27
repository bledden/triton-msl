#!/usr/bin/env python3
"""Run Triton's upstream test suite against the Metal backend.

Captures pass/fail/skip/error results per test, categorizes failures,
and writes both a JSON report and a human-readable summary.

Usage:
    python scripts/run_upstream_tests.py [--test-file test_core.py] [--report-dir reports/]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def find_triton_test_dir():
    """Find the Triton test directory."""
    candidates = [
        Path(os.environ.get("TRITON_TEST_DIR", "")),
        Path.home() / "Documents/triton/python/test",
        Path("/usr/local/lib/python3.12/site-packages/triton/test"),
    ]
    for p in candidates:
        if p.is_dir() and (p / "conftest.py").exists():
            return p
    return None


def run_tests(test_dir, test_file, timeout=900):
    """Run pytest and capture structured output."""
    test_path = test_dir / "unit" / "language" / test_file

    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return None

    env = os.environ.copy()
    env["TRITON_DEFAULT_BACKEND"] = "metal"
    # Pin imports to this checkout so worktree changes are exercised instead
    # of the editable install at ~/Documents/triton-metal. Mirrors
    # scripts/run_upstream_test.sh.
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = (
        repo_root + (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH")
        else repo_root
    )

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "--device", "cpu",
        "--tb=line",
        "-v",
        "--no-header",
    ]

    print(f"Running: {' '.join(cmd[:6])}...")
    print(f"Test file: {test_path}")
    print(f"Backend: metal")
    print()

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        env=env, cwd=str(test_dir),
    )

    return result


def parse_results(output):
    """Parse pytest -v output into structured results."""
    results = {"passed": [], "failed": [], "skipped": [], "error": []}
    failure_reasons = Counter()

    for line in output.split("\n"):
        # Match pytest -v format: test_path::test_name STATUS [pct%]
        m = re.match(r"^(.*?)\s+(PASSED|FAILED|SKIPPED|ERROR)\s*(\[.*\])?\s*$", line)
        if m:
            test_name = m.group(1).strip()
            status = m.group(2).lower()
            results[status].append(test_name)
            continue

        # Match failure reason lines from --tb=line
        if line.startswith("FAILED "):
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                reason = parts[1].strip()
                # Categorize
                if "float64" in reason or "fp64" in reason.lower():
                    failure_reasons["No FP64 support"] += 1
                elif "float8" in reason or "fp8" in reason.lower():
                    failure_reasons["No FP8 support"] += 1
                elif "UNSUPPORTED" in reason:
                    failure_reasons["Unsupported TTGIR op"] += 1
                elif "Metal shader compilation" in reason:
                    failure_reasons["MSL compilation error"] += 1
                elif "AssertionError" in reason or "assert" in reason.lower():
                    failure_reasons["Numerical mismatch"] += 1
                elif "TypeError" in reason:
                    failure_reasons["Type error"] += 1
                elif "RuntimeError" in reason:
                    failure_reasons["Runtime error"] += 1
                elif "triton.comp" in reason:
                    failure_reasons["Triton compilation error"] += 1
                else:
                    failure_reasons["Other"] += 1

    # Parse pytest's own summary line as cross-check
    pytest_summary = {}
    for m in re.finditer(r"(\d+) (passed|failed|skipped|error|warnings?|deselected)", output):
        pytest_summary[m.group(2)] = int(m.group(1))

    return results, failure_reasons, pytest_summary


def write_report(results, failure_reasons, pytest_summary, report_dir, test_file):
    """Write JSON and text reports."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in results.values())
    passed = len(results["passed"])
    failed = len(results["failed"])
    skipped = len(results["skipped"])
    errors = len(results["error"])

    pass_rate = (passed / total * 100) if total > 0 else 0

    # JSON report
    report = {
        "test_file": test_file,
        "backend": "metal",
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "pass_rate": round(pass_rate, 1),
        "failure_categories": dict(failure_reasons.most_common()),
        "pytest_summary": pytest_summary,
        "passed_tests": results["passed"][:50],
        "failed_tests": results["failed"][:100],
    }

    json_path = report_dir / f"upstream_{test_file.replace('.py', '')}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Text summary
    txt_path = report_dir / f"upstream_{test_file.replace('.py', '')}.txt"
    lines = [
        f"Triton Upstream Test Results: {test_file}",
        f"Backend: metal",
        f"{'=' * 50}",
        f"",
        f"Total:   {total}",
        f"Passed:  {passed} ({pass_rate:.1f}%)",
        f"Failed:  {failed}",
        f"Skipped: {skipped}",
        f"Errors:  {errors}",
        f"",
        f"Failure Categories:",
    ]
    for category, count in failure_reasons.most_common():
        lines.append(f"  {category}: {count}")

    if pytest_summary:
        lines.extend([
            f"",
            f"Pytest summary (cross-check):",
        ])
        for k, v in sorted(pytest_summary.items()):
            lines.append(f"  {k}: {v}")

    lines.extend([
        f"",
        f"Sample passing tests:",
    ])
    for t in results["passed"][:20]:
        lines.append(f"  {t}")

    lines.extend([
        f"",
        f"Sample failing tests:",
    ])
    for t in results["failed"][:20]:
        lines.append(f"  {t}")

    summary = "\n".join(lines)

    with open(txt_path, "w") as f:
        f.write(summary)

    # Print summary
    print(summary)
    print(f"\nReports written to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Triton upstream tests on Metal")
    parser.add_argument("--test-file", default="test_core.py",
                       help="Test file to run (default: test_core.py)")
    parser.add_argument("--report-dir", default="reports",
                       help="Directory for reports (default: reports/)")
    parser.add_argument("--timeout", type=int, default=900,
                       help="Test timeout in seconds (default: 900)")
    args = parser.parse_args()

    test_dir = find_triton_test_dir()
    if test_dir is None:
        print("ERROR: Could not find Triton test directory.")
        print("Set TRITON_TEST_DIR environment variable.")
        sys.exit(1)

    print(f"Triton test dir: {test_dir}")

    result = run_tests(test_dir, args.test_file, timeout=args.timeout)
    if result is None:
        sys.exit(1)

    # Save raw output for debugging
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_path = report_dir / f"upstream_{args.test_file.replace('.py', '')}_raw.txt"
    full_output = result.stdout + "\n" + result.stderr
    with open(raw_path, "w") as f:
        f.write(full_output)
    print(f"Raw output saved to: {raw_path}")

    results, failure_reasons, pytest_summary = parse_results(full_output)

    report = write_report(results, failure_reasons, pytest_summary, args.report_dir, args.test_file)
    sys.exit(0 if report["pass_rate"] > 0 else 1)


if __name__ == "__main__":
    main()
