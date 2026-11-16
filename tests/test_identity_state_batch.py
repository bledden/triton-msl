"""Native batching preserves comparisons and exact observer fallback."""

import builtins
import types

import pytest
from tests.native_helper_policy import require_selected_native

from triton_msl.backend import _cache_contract as cache, _identity_fast_path as fast


@pytest.fixture
def sample():
    require_selected_native(cache._validation_native, "validation")
    native = cache._validation_native
    assert native is not None and fast._STATE_BATCH is not None

    def make(value):
        def fn(arg=1):
            return value + arg

        return fn

    functions = [make(index) for index in range(3)]
    states = tuple((fn, native.function_state(fn)) for fn in functions)
    assert native.function_states_are(native, states, any) is True
    assert fast._function_states_hold(native, states) is True
    return native, functions, states


@pytest.mark.parametrize("field", ["__code__", "__defaults__", "__kwdefaults__", "cell"])
def test_state_mutation_is_not_hidden(sample, monkeypatch, field):
    native, functions, states = sample
    fn = functions[1]
    if field == "cell":
        fn.__closure__[0].cell_contents = 93
    elif field == "__code__":

        def replacement(value):
            def inner(arg=1):
                return value - arg

            return inner

        monkeypatch.setattr(fn, field, replacement(0).__code__)
    else:
        monkeypatch.setattr(fn, field, (9,) if field == "__defaults__" else {"extra": 4})
    assert native.function_states_are(native, states, any) is False
    assert fast._function_states_hold(native, states) is False


@pytest.mark.parametrize("failure", ["none", "runtime", "stop"])
def test_replaced_observer_runs_original_lazy_loop(sample, monkeypatch, failure):
    native, _, states = sample
    original = native.function_state_is
    calls = []
    marker = RuntimeError("observer marker") if failure == "runtime" else StopIteration("observer marker")

    def observed(fn, state):
        calls.append(fn)
        if failure != "none":
            raise marker
        return original(fn, state)

    monkeypatch.setattr(native, "function_state_is", observed)
    assert native.function_states_are(native, states, any) is NotImplemented
    assert calls == []
    if failure == "none":
        assert fast._function_states_hold(native, states)
        assert calls == [pair[0] for pair in states]
    else:
        with pytest.raises(RuntimeError) as caught:
            fast._function_states_hold(native, states)
        assert caught.value is marker if failure == "runtime" else caught.value.__cause__ is marker
        assert calls == [states[0][0]]


def test_custom_any_receives_generator_without_speculative_comparisons(sample, monkeypatch):
    native, _, states = sample
    calls = []

    def custom(values):
        calls.append(type(values))
        return False

    monkeypatch.setattr(fast, "any", custom, raising=False)
    assert native.function_states_are(native, states, custom) is NotImplemented
    assert fast._function_states_hold(native, states)
    assert calls == [types.GeneratorType]


def test_changed_module_lookup_uses_original_observations(sample, monkeypatch):
    native, _, states = sample
    calls = []

    class Observed(types.ModuleType):
        def __getattribute__(self, name):
            if name == "function_state_is":
                calls.append(name)
            return super().__getattribute__(name)

    monkeypatch.setattr(native, "__class__", Observed)
    assert fast._function_states_hold(native, states)
    assert calls == ["function_state_is"] * len(states)


def test_missing_provider_keeps_module_getattr(sample, monkeypatch):
    native, _, states = sample
    original = native.function_state_is
    calls = []

    def missing(name):
        calls.append(name)
        if name == "function_state_is":
            return original
        raise AttributeError(name)

    monkeypatch.delattr(native, "function_state_is")
    monkeypatch.setattr(native, "__getattr__", missing, raising=False)
    assert fast._function_states_hold(native, states)
    assert calls == ["function_state_is"] * len(states)


def test_foreign_registry_key_declines_without_equality(sample):
    native, _, states = sample
    original = native.__dict__.pop("function_state_is")
    calls = []

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append(other)
            return super().__eq__(other)

    key = Key("function_state_is")
    native.__dict__[key] = original
    try:
        assert native.function_states_are(native, states, any) is NotImplemented
        assert calls == []
        assert fast._function_states_hold(native, states)
        assert calls == ["function_state_is"] * len(states)
    finally:
        del native.__dict__[key]
        native.__dict__["function_state_is"] = original


def test_replaced_batch_entry_falls_back_without_calling_replacement(sample, monkeypatch):
    native, _, states = sample
    calls = []
    monkeypatch.setattr(native, "function_states_are", lambda *args: calls.append(args))
    assert fast._function_states_hold(native, states)
    assert calls == []


def test_old_image_and_empty_states_keep_python_fallback(sample, monkeypatch):
    native, _, states = sample
    monkeypatch.setattr(fast, "_STATE_BATCH", None)
    assert fast._function_states_hold(native, states)
    assert fast._function_states_hold(native, ())


def test_malformed_owned_entry_preserves_unpacking_error(sample):
    native, _, _ = sample
    with pytest.raises(ValueError, match="not enough values"):
        fast._function_states_hold(native, ((1,),))


def test_unproved_builtins_owner_declines_and_fallback_still_works(sample):
    native, _, states = sample
    copied_builtins = dict(vars(builtins))
    namespace = {"__builtins__": copied_builtins, "native": native, "states": states}
    exec("def probe():\n    return native.function_states_are(native, states, any)", namespace)
    assert namespace["probe"]() is NotImplemented
    original = fast._function_states_hold
    copied = types.FunctionType(original.__code__, dict(original.__globals__, __builtins__=copied_builtins))
    assert copied(native, states) is True
