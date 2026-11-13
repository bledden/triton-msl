"""Pure dependency checks preserve live lookup and consumer semantics."""

import builtins
from types import FunctionType, SimpleNamespace

import pytest

from triton_msl.backend import _environment_snapshot as environment


@pytest.fixture(autouse=True)
def pure_checks(monkeypatch):
    monkeypatch.setattr(environment, "_validation_native", None)
    monkeypatch.setattr(environment, "_native_all", builtins.all)
    monkeypatch.setattr(environment, "all", builtins.all, raising=False)


def guarded_function(namespace, dependencies):
    fn = FunctionType((lambda: None).__code__, namespace)
    return fn, fn.__code__, fn.__defaults__, dependencies


def test_builtin_default_lookup_precedes_live_global_lookup():
    events = []
    wanted = object()

    class Globals(dict):
        def get(self, name, default=None):
            events.append(("global", name))
            return super().get(name, default)

    namespace = Globals()

    class Builtins(dict):
        def get(self, name, default=None):
            events.append(("builtin", name))
            # The default expression runs even when a global binding exists.
            namespace[name] = wanted
            return super().get(name, default)

    namespace["__builtins__"] = Builtins()
    guard = guarded_function(namespace, (("first", wanted), ("second", wanted)))
    assert environment._implementation_unchanged(guard)
    assert events == [
        ("builtin", "first"), ("global", "first"),
        ("builtin", "second"), ("global", "second"),
    ]


def test_mismatch_does_not_observe_later_dependency():
    events = []

    class Namespace(dict):
        def get(self, name, default=None):
            events.append(name)
            if name == "later":
                raise AssertionError("observed beyond first mismatch")
            return super().get(name, default)

    namespace = Namespace(__builtins__={})
    guard = guarded_function(namespace, (("first", object()), ("later", object())))
    assert environment._implementation_unchanged(guard) is False
    assert events == ["first"]


def test_mutation_of_later_binding_is_observed():
    first, old, new = object(), object(), object()

    class Namespace(dict):
        def get(self, name, default=None):
            if name == "first":
                self["later"] = new
            return super().get(name, default)

    namespace = Namespace(__builtins__={}, first=first, later=old)
    guard = guarded_function(namespace, (("first", first), ("later", old)))
    assert environment._implementation_unchanged(guard) is False


@pytest.mark.parametrize("consume", [False, True])
def test_custom_all_keeps_lazy_generator_and_return_value(monkeypatch, consume):
    events = []
    sentinel = object()

    class Namespace(dict):
        def get(self, name, default=None):
            events.append(name)
            return super().get(name, default)

    namespace = Namespace(__builtins__={}, first=sentinel)
    guard = guarded_function(namespace, (("first", sentinel), ("later", sentinel)))

    def custom_all(iterator):
        events.append("consumer")
        assert iter(iterator) is iterator
        if consume:
            assert next(iterator) is True
        return sentinel

    monkeypatch.setattr(environment, "all", custom_all)
    assert environment._implementation_unchanged(guard) is sentinel
    assert events == (["consumer", "first"] if consume else ["consumer"])


def test_dependency_iterator_error_propagates_once():
    error = RuntimeError("dependency iterator")
    events = []
    wanted = object()

    def dependencies():
        events.append("first")
        yield "first", wanted
        events.append("error")
        raise error

    guard = guarded_function({"__builtins__": {}, "first": wanted}, dependencies())
    with pytest.raises(RuntimeError) as caught:
        environment._implementation_unchanged(guard)
    assert caught.value is error
    assert events == ["first", "error"]


@pytest.mark.parametrize("damage", ["code", "defaults", "kwdefaults"])
def test_changed_function_header_stops_before_consumer(monkeypatch, damage):
    guard = guarded_function({"__builtins__": {}}, ())
    fn = guard[0]
    if damage == "code":
        fn.__code__ = (lambda: 1).__code__
    elif damage == "defaults":
        fn.__defaults__ = (None,)
    else:
        fn.__kwdefaults__ = {}

    def forbidden(iterator):
        raise AssertionError("consumer reached after changed header")

    monkeypatch.setattr(environment, "all", forbidden)
    assert environment._implementation_unchanged(guard) is False


def test_native_delegation_is_unchanged(monkeypatch):
    sentinel = object()
    calls = []

    def native(namespace, guard):
        calls.append((namespace, guard))
        return sentinel

    monkeypatch.setattr(environment, "_validation_native", SimpleNamespace(implementation_unchanged=native))
    guard = object()
    assert environment._implementation_unchanged(guard) is sentinel
    assert calls == [(vars(environment), guard)]


def test_dependency_iterator_stops_at_first_mismatch():
    def dependencies():
        yield "missing", object()
        raise AssertionError("eager dependency consumption")

    guard = guarded_function({"__builtins__": {}}, dependencies())
    assert environment._implementation_unchanged(guard) is False


def test_get_method_is_resolved_again_for_each_dependency():
    wanted = object()
    events = []

    class Namespace(dict):
        pass

    def changed(self, name, default=None):
        events.append(("changed", name))
        return object()

    def initial(self, name, default=None):
        events.append(("initial", name))
        Namespace.get = changed
        return wanted

    Namespace.get = initial
    guard = guarded_function(Namespace(__builtins__={}), (("a", wanted), ("b", wanted)))
    assert environment._implementation_unchanged(guard) is False
    assert events == [("initial", "a"), ("changed", "b")]


def test_function_like_object_keeps_live_attribute_reads():
    wanted = object()
    events = []

    class FunctionLike:
        __code__ = object()
        __defaults__ = None
        __kwdefaults__ = None
        __builtins__ = {}

        @property
        def __globals__(self):
            events.append("globals")
            return {"a": wanted, "b": wanted if len(events) == 1 else object()}

    fn = FunctionLike()
    guard = fn, fn.__code__, fn.__defaults__, (("a", wanted), ("b", wanted))
    assert environment._implementation_unchanged(guard) is False
    assert events == ["globals", "globals"]


def test_standard_function_avoids_dependency_generator_frame():
    # Discriminates the intended allocation removal from the old generator path.
    import sys

    calls = []
    filename = environment._implementation_unchanged.__code__.co_filename

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == filename:
            calls.append(frame.f_code.co_name)

    wanted = object()
    guard = guarded_function({"__builtins__": {}, "a": wanted}, (("a", wanted),))
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = environment._implementation_unchanged(guard)
    finally:
        sys.setprofile(previous)
    assert result is True
    assert "_implementation_unchanged" in calls
    assert "<genexpr>" not in calls
