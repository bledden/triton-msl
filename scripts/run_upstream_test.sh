#!/usr/bin/env bash
# Run upstream Triton pytest targets against this checkout's triton_metal.
#
# Why this exists: pytest from upstream's ~/Documents/triton/python/test will
# import `triton_metal` from whichever copy is registered as the editable
# install (usually ~/Documents/triton-metal), not from a worktree you're
# actively developing in. Setting PYTHONPATH to the current repo root pins
# imports to this checkout so worktree changes are exercised.
#
# Usage: scripts/run_upstream_test.sh <pytest-args...>
#   e.g. scripts/run_upstream_test.sh unit/language/test_core.py::test_trans_reshape -v
#
# Resolves the upstream test directory from $TRITON_TEST_DIR or
# ~/Documents/triton/python/test (matching scripts/run_upstream_tests.py).

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"

test_dir="${TRITON_TEST_DIR:-$HOME/Documents/triton/python/test}"
if [[ ! -d "$test_dir" ]]; then
    echo "Upstream test dir not found: $test_dir" >&2
    echo "Set TRITON_TEST_DIR to override." >&2
    exit 1
fi

# Add the repo root and ``scripts/`` to PYTHONPATH so pytest can both
# (1) import ``triton_metal`` from this checkout and (2) load
# ``conftest_metal`` as a plugin via ``-p`` (the skip rules for unsupported
# dtypes / precisions live there; upstream's own conftest doesn't know
# about them).
export PYTHONPATH="$repo_root:$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_DEFAULT_BACKEND="${TRITON_DEFAULT_BACKEND:-metal}"

cd "$test_dir"
exec python3 -m pytest -p conftest_metal --device cpu "$@"
