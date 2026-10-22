"""Import-machinery memo ownership and observation order, CPU-only fixtures."""

import gc
from types import SimpleNamespace
import weakref

import pytest

from triton_msl.backend import _framework_contract as module


class Finder:
    def __init__(self, root):
        self.root = root


@pytest.fixture
def machinery(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fake = SimpleNamespace(
        modules={},
        path=[],
        meta_path=[Finder(a)],
        path_hooks=[],
        version=module.sys.version,
        implementation=module.sys.implementation,
        byteorder=module.sys.byteorder,
    )
    calls = []

    def resolve(name):
        calls.append(name)
        root = fake.meta_path[0].root
        return SimpleNamespace(origin=str(root / "__init__.py"), submodule_search_locations=[str(root)])

    util = SimpleNamespace(find_spec=resolve)
    monkeypatch.setattr(module, "sys", fake)
    monkeypatch.setattr(module, "importlib", SimpleNamespace(util=util))
    monkeypatch.setattr(module, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(module, "_REQUIRED_NAMES", ())
    monkeypatch.setattr(module, "_SELECTION_NAMES", ("pkg",))
    monkeypatch.setattr(module, "_RUNTIME_SELECTION_ENV", ())
    monkeypatch.setattr(module, "_python_roots", lambda: {})
    monkeypatch.setattr(module, "_discovery_snapshot", None)
    monkeypatch.setattr(module, "_owned_selection", None)
    state = SimpleNamespace(a=a.resolve(), b=b.resolve(), sys=fake, util=util, calls=calls, active=True)
    yield state
    state.active = False
    module._discovery_snapshot = None
    module._owned_selection = None


def test_memo_owns_observed_finder_until_replaced(machinery):
    h = machinery
    assert module._discover_selection()[0]["pkg"] == (h.a,)
    old = weakref.ref(h.sys.meta_path[0])
    h.sys.meta_path[:] = [Finder(h.b)]
    gc.collect()
    assert old() is not None, "An id-only memo must not outlive its objects"
    assert module._discover_selection()[0]["pkg"] == (h.b,)
    gc.collect()
    assert old() is None


def test_change_during_resolution_does_not_publish_stale_memo(machinery):
    h = machinery
    original = h.util.find_spec

    def resolve(name):
        result = original(name)
        h.sys.meta_path[:] = [Finder(h.b)]
        return result

    h.util.find_spec = resolve
    assert module._discover_selection()[0]["pkg"] == (h.a,)
    assert module._discovery_snapshot is None
    h.util.find_spec = original
    assert module._discover_selection()[0]["pkg"] == (h.b,)


@pytest.mark.parametrize("live_list", [False, True])
def test_foreign_capture_constructor_never_certifies_ownership(machinery, monkeypatch, live_list):
    h = machinery

    def constructor(value=()):
        if live_list and value is h.sys.meta_path:
            return value
        return tuple(value)

    monkeypatch.setattr(module, "tuple", constructor, raising=False)
    assert module._discover_selection()[0]["pkg"] == (h.a,)
    assert module._discovery_snapshot is None
    h.sys.meta_path[:] = [Finder(h.b)]
    before = len(h.calls)
    assert module._discover_selection()[0]["pkg"] == (h.b,)
    assert len(h.calls) == before + 1
    assert module._discovery_snapshot is None


@pytest.mark.parametrize("field", ["meta_path", "path_hooks", "find_spec"])
def test_original_callable_selected_before_owner_callback(machinery, monkeypatch, field):
    h = machinery
    events = []
    active = False
    failure = RuntimeError("selected replacement")

    def poison(*args):
        events.append("replacement")
        raise failure

    class Sys:
        def __getattr__(self, name):
            return getattr(h.sys, name)

        @property
        def meta_path(self):
            events.append("meta_path")
            if active and field == "meta_path":
                monkeypatch.setattr(module, "map", poison, raising=False)
            return h.sys.meta_path

        @property
        def path_hooks(self):
            events.append("path_hooks")
            if active and field == "path_hooks":
                monkeypatch.setattr(module, "map", poison, raising=False)
            return h.sys.path_hooks

    class Util:
        @property
        def find_spec(self):
            events.append("find_spec")
            if active and field == "find_spec":
                monkeypatch.setattr(module, "id", poison, raising=False)
            return h.util.find_spec

    monkeypatch.setattr(module, "sys", Sys())
    monkeypatch.setattr(module, "importlib", SimpleNamespace(util=Util()))
    first = module._discover_selection()
    active = True
    events.clear()
    if field == "meta_path":
        with pytest.raises(RuntimeError) as exc:
            module._discover_selection()
        assert exc.value is failure
        assert events == ["meta_path", "path_hooks", "replacement"]
    else:
        result = module._discover_selection()
        assert result[0] is first[0]
        assert events == ["meta_path", "path_hooks", "find_spec"]


def test_partial_failure_preserves_memo_and_exact_exception(machinery):
    h = machinery
    module._discover_selection()
    before = module._discovery_snapshot
    error = RuntimeError("resolution failed")

    def fail(name):
        raise error

    h.util.find_spec = fail
    with pytest.raises(RuntimeError) as exc:
        module._discover_selection()
    assert exc.value is error
    assert module._discovery_snapshot is before


@pytest.mark.parametrize("reenter", [False, True])
def test_retirement_mutation_and_reentry_preserve_next_selection(machinery, reenter):
    h = machinery
    events = []

    class Hook:
        def __del__(self):
            if not h.active:
                return
            h.sys.meta_path[:] = [Finder(h.b)]
            events.append("replace")
            if reenter:
                events.append(module._discover_selection()[0]["pkg"])

    h.sys.path_hooks.append(Hook())
    assert module._discover_selection()[0]["pkg"] == (h.a,)
    h.sys.path_hooks.clear()
    assert events == []
    assert module._discover_selection()[0]["pkg"] == (h.a,)
    assert events == (["replace", (h.b,)] if reenter else ["replace"])
    before = len(h.calls)
    assert module._discover_selection()[0]["pkg"] == (h.b,)
    assert (len(h.calls) == before) is reenter
