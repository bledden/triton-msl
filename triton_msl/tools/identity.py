"""Inspect, refresh, clear or verify the per-user identity records.

status   print the mode in force, the record directory and its records
refresh  run the strict toolchain and framework inventories and publish records
clear    delete record files (toolchain-*.json, framework-*.json) in the directory;
         a directory that does not exist holds none ("removed 0 record(s)")
verify   compare each record this process would use with a strict inventory;
         a differing record is deleted and the exit status is 1. When no
         earlier record was used (none existed, or it no longer matched),
         the record-mode run is itself a strict inventory, so the line says
         that nothing earlier was compared

Exit status: 0 on success (for verify: every digest agreed), 1 when verify
found and removed a differing record, 2 on an error (for example an untrusted
record directory).

The framework record is specific to the ordered list of loaded native images,
so ``refresh`` and ``verify`` cover only the framework record that matches this
command's own process shape. For an application's framework record, run
``clear``, or run the application once with TRITON_MSL_IDENTITY_RECORD=refresh.
No GPU device is created.
"""

import argparse
import os
import sys


def _components():
    # Identity records are only used once the cache contract is imported, as on
    # the backend path; main() imports it before anything is computed.
    from triton_msl.backend import _framework_contract as framework
    from triton_msl.backend import _toolchain_contract as toolchain

    def reset_toolchain():
        toolchain._snapshot = None

    def reset_framework():
        framework._snapshot = None
        framework._native_guard = None

    # Either order is correct. The framework record that is used and published
    # is the one matching the images this process has loaded by then.
    return (
        ("toolchain", toolchain.toolchain_identity, reset_toolchain),
        ("framework", framework.framework_identity, reset_framework),
    )


def _run(kind, compute, reset, policy):
    from triton_msl.backend import _identity_record as record

    os.environ[record.SWITCH] = policy
    reset()
    digest = compute()
    outcome = record.last_outcome(kind) if policy != "0" else None
    return digest, outcome


def _status(_args):
    from triton_msl.backend import _identity_record as record

    report = record.status()
    print(f"mode: {report['mode']}")
    print(f"directory: {report['directory']}")
    if report["error"] is not None:
        print(f"directory unusable: {report['error']}")
    for entry in report["records"]:
        producer = entry["producer"] or {}
        print(
            f"{entry['kind']:9} {entry['name']} decodes={entry['decodes']} current={entry['current']}"
            f" created={entry['created']} version={producer.get('version')}"
        )
    if report["trusted"] and not report["records"]:
        print("no records")
    return 0


def _refresh(_args):
    for kind, compute, reset in _components():
        digest, outcome = _run(kind, compute, reset, "refresh")
        print(f"{kind}: {digest} {outcome[0] if outcome else 'not published'}")
    return 0


def _clear(_args):
    from triton_msl.backend import _identity_record as record

    print(f"removed {record.clear()} record(s)")
    return 0


def _verify(_args):
    from triton_msl.backend import _identity_record as record

    differences = 0
    for kind, compute, reset in _components():
        recorded, outcome = _run(kind, compute, reset, "1")
        strict, _ = _run(kind, compute, reset, "0")
        state, locator = outcome if outcome else ("unavailable", None)
        if recorded != strict:
            differences += 1
            print(f"{kind}: DIFFERS record={recorded} strict={strict}")
            if state in ("hit", "published") and locator is not None:
                record.remove(os.environ, kind, locator)
                print(f"{kind}: removed {kind}-{locator}.json")
        elif state == "hit":
            print(f"{kind}: ok (hit) {strict}")
        elif state == "published":
            print(f"{kind}: ok (no earlier record was compared; published a new record) {strict}")
        else:
            print(f"{kind}: ok (no earlier record was compared: {state}) {strict}")
    return 1 if differences else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m triton_msl.tools.identity", description=__doc__)
    parser.add_argument("command", choices=("status", "refresh", "clear", "verify"))
    args = parser.parse_args(argv)
    from triton_msl.backend._identity_record import SWITCH

    saved = os.environ.get(SWITCH)
    try:
        from triton_msl.backend import _cache_contract  # noqa: F401

        return {"status": _status, "refresh": _refresh, "clear": _clear, "verify": _verify}[args.command](args)
    except Exception as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if saved is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = saved


if __name__ == "__main__":
    sys.exit(main())
