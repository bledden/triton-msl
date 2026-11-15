"""Pytest plugin: identity records stay private to a session and off in contract suites.

Identity records persist across processes. A session therefore writes them
only into a fresh private directory, never into the user's own, unless the
caller already chose a directory. The existing contract suites below replace
toolchain/framework internals and assert strict-mode inventories, reads and
module state, so they keep strict mode. Other suites run with records on, as
users do. Register with ``-p identity_record_plugin`` under ``--noconftest``.

Strict pinning is a function-scoped autouse fixture: it is in force only while
a test function runs. A module- or session-scoped fixture in one of the strict
suites must therefore not compute an identity, or it would do so with records
on. Running with ``--noconftest`` without this plugin (and without
``TRITON_MSL_IDENTITY_RECORD=0``) uses the user's own record directory, so
records would be read from and written to ``~/.cache/triton_msl``.
"""

import os
import shutil
import tempfile

import pytest


_DIRECTORY = "TRITON_MSL_IDENTITY_DIR"
_SWITCH = "TRITON_MSL_IDENTITY_RECORD"
_created = None
STRICT_MODULES = frozenset(
    (
        "test_backend_cache_contract",
        "test_binder_cold_initialization",
        "test_cache_contract_primitives",
        "test_compiler_dependency_lifetime",
        "test_compiler_working_directory",
        "test_device_probe_contract",
        "test_discovery_memo_retention",
        "test_driver_activation_context",
        "test_environment_dependency_checks",
        "test_environment_snapshot",
        "test_execution_contract_compile",
        "test_execution_evaluation",
        "test_framework_contract",
        "test_host_extent",
        "test_identity_callback_boundaries",
        "test_identity_colliding_keys",
        "test_identity_dict_epoch",
        "test_identity_fast_path",
        "test_identity_live_owners",
        "test_identity_namespace",
        "test_identity_namespace_certificate",
        "test_identity_option_a",
        "test_identity_probe_program",
        "test_identity_state_batch",
        "test_identity_type_versions",
        "test_launch_source_signature",
        "test_manifest_streaming",
        "test_metallib_cache_contract",
        "test_native_ambient_contract",
        "test_native_dependency_contract",
        "test_native_entrypoint_resolution",
        "test_native_generation_guard",
        "test_native_lazy_contract",
        "test_packed_launch_contract",
        "test_parallel_inventory",
        "test_parallel_inventory_review",
        "test_readonly_toolchain_manifest",
        "test_resident_execution_contract",
        "test_source_identity",
        "test_stash_contract",
        "test_test_cache_contract",
        "test_toolchain_context_errors",
        "test_toolchain_contract",
        "test_toolchain_cross_stage_digests",
        "test_toolchain_discovery_reads",
        "test_toolchain_native_dependencies",
        "test_validation_native",
    )
)


def pytest_configure(config):
    global _created
    if not os.environ.get(_DIRECTORY):
        _created = tempfile.mkdtemp(prefix="triton_msl_identity_")
        os.environ[_DIRECTORY] = _created


def pytest_unconfigure(config):
    global _created
    if _created is not None and os.environ.get(_DIRECTORY) == _created:
        del os.environ[_DIRECTORY]
        shutil.rmtree(_created, ignore_errors=True)
    _created = None


@pytest.fixture(autouse=True)
def _strict_identity_in_contract_suites(request, monkeypatch):
    module = getattr(request, "module", None)
    if module is not None and module.__name__.rpartition(".")[2] in STRICT_MODULES:
        monkeypatch.setenv(_SWITCH, "0")
