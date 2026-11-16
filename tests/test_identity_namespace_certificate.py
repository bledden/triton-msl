"""Ordinary namespace certificates: CPU primitive checks, no device initialization."""

from importlib._bootstrap_external import _NamespacePath
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tests.native_helper_policy import require_selected_native

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _identity_fast_path as fast
from triton_msl.errors import MetalNonRecoverableError


def _capture():
    require_selected_native(cache._validation_native, "validation")
    native = cache._validation_native
    assert native is not None, "this selection requires the built native helper"
    namespace = _NamespacePath("example", ["/first", "/second"], lambda *args: None)
    functions = []
    probes = fast._namespace_probes(native, namespace, "example", functions)
    assert probes is not None
    assert native.identity_probes(probes)
    assert len(functions) == 5
    return native, namespace, probes, functions


def _hit(native, probes, functions):
    return native.identity_probes(probes) and all(native.function_state_is(fn, state) for fn, state in functions)


def test_settled_namespace_is_not_iterated_or_replaced():
    native, namespace, probes, functions = _capture()
    state, path = namespace.__dict__, namespace._path
    for _ in range(10):
        assert _hit(native, probes, functions)
    assert namespace.__dict__ is state and namespace._path is path
    assert list(namespace) == ["/first", "/second"]


@pytest.mark.parametrize(
    "change",
    ["parent", "parent-replace", "path", "path-replace", "dictionary", "epoch", "finder", "instance-method", "name"],
)
def test_live_namespace_change_invalidates(monkeypatch, change):
    native, namespace, probes, functions = _capture()
    with monkeypatch.context() as m:
        if change == "parent":
            m.setattr(sys, "path", [*sys.path, "/new-parent"])
        elif change == "parent-replace":
            m.setattr(sys, "path", list(sys.path))
        elif change == "path":
            namespace._path[0] = "/changed"
        elif change == "path-replace":
            namespace._path = list(namespace._path)
        elif change == "dictionary":
            namespace.__dict__ = dict(namespace.__dict__)
        elif change == "epoch":
            importlib.invalidate_caches()
        elif change == "finder":
            namespace._path_finder = lambda *args: None
        elif change == "instance-method":
            namespace._get_parent_path = lambda: ()
        else:
            namespace._name = "different"
        assert not _hit(native, probes, functions), change


@pytest.mark.parametrize("name", ["__iter__", "__len__", "_recalculate", "_get_parent_path", "_find_parent_path_names"])
def test_changed_method_code_misses_without_calling_it(monkeypatch, name):
    native, namespace, probes, functions = _capture()
    calls = []
    fn = _NamespacePath.__dict__[name]

    def replacement(self):
        raise AssertionError("must not execute during certification")

    with monkeypatch.context() as m:
        m.setattr(fn, "__code__", replacement.__code__)
        assert not _hit(native, probes, functions)
        assert fast._namespace_probes(native, namespace, "example", []) is None
    assert calls == []


def test_custom_iterator_declines_without_observation(monkeypatch):
    native, namespace, _, _ = _capture()
    calls = []

    def observe(self):
        calls.append("iter")
        return iter(self._path)

    with monkeypatch.context() as m:
        m.setattr(_NamespacePath, "__iter__", observe)
        assert fast._namespace_probes(native, namespace, "example", []) is None
    assert calls == []


def test_unsettled_namespace_declines_without_finder_call(monkeypatch):
    native, namespace, _, _ = _capture()
    calls = []
    namespace._path_finder = lambda *args: calls.append(args)
    with monkeypatch.context() as m:
        m.setattr(sys, "path", [*sys.path, "/new-parent"])
        assert fast._namespace_probes(native, namespace, "example", []) is None
    assert calls == []


def test_foreign_key_declines_without_equality():
    native, namespace, _, _ = _capture()
    calls = []

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append(other)
            return super().__eq__(other)

    state = dict(namespace.__dict__)
    old = state.pop("_path")
    state[Key("_path")] = old
    namespace.__dict__ = state
    assert fast._namespace_probes(native, namespace, "example", []) is None
    assert calls == []


def test_nested_and_custom_namespaces_keep_fallback():
    native, namespace, _, _ = _capture()
    assert fast._namespace_probes(native, namespace, "example.child", []) is None

    class Custom(_NamespacePath):
        pass

    custom = Custom("example", ["/path"], lambda *args: None)
    assert fast._namespace_probes(native, custom, "example", []) is None


def test_real_loaded_namespace_gets_hit_without_normalization(monkeypatch):
    """No ordinary_framework fixture; keep other loaded namespaces real too."""
    import numpy

    require_selected_native(cache._validation_native, "validation")
    original = numpy.__spec__.submodule_search_locations
    namespace = _NamespacePath("numpy", list(original), lambda *args: None)
    try:
        with monkeypatch.context() as m:
            m.setattr(numpy.__spec__, "submodule_search_locations", namespace)
            fast._record = None
            stamp = cache.execution_contract()
            hits = fast._stats["hits"]
            assert cache.execution_contract() == stamp
            assert fast._stats["hits"] == hits + 1
            assert numpy.__spec__.submodule_search_locations is namespace
            m.setenv("TRITON_MSL_IDENTITY_FAST_PATH", "0")
            hits = fast._stats["hits"]
            assert cache.execution_contract() == stamp
            assert fast._stats["hits"] == hits
    finally:
        fast._record = None


def test_namespace_cache_invalidation_runs_finder_once(monkeypatch):
    import numpy

    original = numpy.__spec__.submodule_search_locations
    calls = []

    def finder(*args):
        calls.append(args)
        return None

    namespace = _NamespacePath("numpy", list(original), finder)
    try:
        with monkeypatch.context() as m:
            m.setattr(numpy.__spec__, "submodule_search_locations", namespace)
            fast._record = None
            stamp = cache.execution_contract()
            assert cache.execution_contract() == stamp
            assert calls == []
            m.setattr(_NamespacePath, "_epoch", _NamespacePath._epoch + 1)
            hits = fast._stats["hits"]
            assert cache.execution_contract() == stamp
            assert fast._stats["hits"] == hits
            assert len(calls) == 1
    finally:
        fast._record = None


def test_namespace_live_path_change_refuses_old_contract(monkeypatch, tmp_path):
    import numpy

    namespace = _NamespacePath("numpy", list(numpy.__spec__.submodule_search_locations), lambda *args: None)
    try:
        with monkeypatch.context() as m:
            m.setattr(numpy.__spec__, "submodule_search_locations", namespace)
            fast._record = None
            stamp = cache.execution_contract()
            assert cache.execution_contract() == stamp
            namespace._path.append(str(tmp_path))
            hits = fast._stats["hits"]
            with pytest.raises(MetalNonRecoverableError):
                cache.validate_execution_contract(stamp)
            assert fast._stats["hits"] == hits
    finally:
        fast._record = None


@pytest.mark.parametrize("mode", ["iterator-code", "recalculate-code", "provider", "missing-layout"])
def test_preimport_namespace_replacements_keep_fallback(mode):
    """Exercise the actual import, before the fast module captures providers."""
    script = r"""
import _imp
import importlib._bootstrap_external as external
import sys
from pathlib import Path

mode, root = sys.argv[1:]
calls = []
namespace = external._NamespacePath('example', ['/one'], lambda *args: None)

def replacement(self):
    calls.append('unexpected observer')
    return iter(self._path)

if mode == 'iterator-code':
    external._NamespacePath.__iter__.__code__ = replacement.__code__
elif mode == 'recalculate-code':
    external._NamespacePath._recalculate.__code__ = replacement.__code__
elif mode == 'provider':
    original_provider = _imp.get_frozen_object
    def foreign_provider(name, *args):
        if name == '_frozen_importlib_external':
            calls.append('unexpected provider call')
            raise AssertionError('our certificate must not call this provider')
        # Keep unrelated Python imports working so the test reaches our code.
        return original_provider(name, *args)
    _imp.get_frozen_object = foreign_provider
elif mode == 'missing-layout':
    del external._NamespacePath

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _identity_fast_path as fast
assert Path(fast.__file__).resolve() == Path(root) / 'triton_msl/backend/_identity_fast_path.py'
assert fast._NS_METHODS == ()
assert fast._namespace_probes(cache._validation_native, namespace, 'example', []) is None
assert calls == [], calls
print('PREIMPORT_FALLBACK_PASS')
"""
    root = str(Path(fast.__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=root, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", script, mode, root],
        env=env,
        cwd="/private/tmp",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "PREIMPORT_FALLBACK_PASS" in result.stdout


def test_canonical_method_comparison_checks_constants_without_callbacks():
    original = _NamespacePath._find_parent_path_names.__code__
    changed = original.replace(
        co_consts=tuple("/different" if type(item) is str else item for item in original.co_consts)
    )
    assert changed.co_code == original.co_code
    assert not fast._namespace_code_equal(changed, original)
    assert not fast._namespace_code_equal(original.replace(co_names=(*original.co_names, "new_name")), original)
    calls = []

    class Observed:
        def __eq__(self, other):
            calls.append(other)
            return True

    poisoned = original.replace(co_consts=tuple(Observed() for _ in original.co_consts))
    assert not fast._namespace_code_equal(poisoned, original)
    assert calls == []
