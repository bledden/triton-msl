"""Cold key certification never invokes user hash/equality/iteration callbacks."""

from types import SimpleNamespace
import importlib.util
import pytest
from tests.native_helper_policy import require_selected_native
from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _identity_fast_path as fast


@pytest.fixture
def native():
    require_selected_native(cache._validation_native, "validation")
    assert cache._validation_native is not None
    return cache._validation_native


def test_all_live_native_contexts_are_invalidated(native):
    spec = importlib.util.spec_from_file_location("_validation_native", native.__file__)
    second = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(second)
    assert second is not native
    missing, value = object(), object()
    mapping = {"wanted": value, "spare": object()}
    probes = ((4, mapping, "wanted", missing, value),)
    assert native.identity_probes(probes) is True
    assert second.identity_probes(probes) is True

    class Foreign(str):
        pass

    del mapping["spare"]
    mapping[Foreign("foreign")] = object()
    assert native.identity_probes(probes) is False
    assert second.identity_probes(probes) is False


def test_key_cache_reset_unwatches_before_reusing_id(native):
    value, missing = object(), object()
    mapping = {"wanted": value}
    probes = ((4, mapping, "wanted", missing, value),)
    assert native.identity_probes(probes) is True
    for _ in range(8):
        native.key_cache_reset()
        assert native.key_cache_info()["active"] == 1
        assert native.identity_probes(probes) is True
        mapping["unrelated"] = object()
        assert native.key_cache_info()["watch_events"] == 1
        del mapping["unrelated"]


def test_key_cache_capacity_falls_back_without_eviction(native):
    native.key_cache_reset()
    missing = object()
    mappings = [{str(i): object()} for i in range(129)]
    for i, mapping in enumerate(mappings):
        value = mapping[str(i)]
        assert native.identity_probes(((4, mapping, str(i), missing, value),)) is True
    info = native.key_cache_info()
    assert info["entries"] == info["capacity"] == 128
    assert info["table_full"] == 1


@pytest.mark.parametrize(
    "value", [{}, {"a": 1}, {"a": object(), "β": None, "\ud800": 3}, {str(i): i for i in range(4096)}]
)
def test_exact_string_dicts(native, value):
    assert native.dict_keys_exact_str(value) is True


@pytest.mark.parametrize("value", [None, [], (), [("a", 1)], {"a"}, {"a": 1, 3: 2}, {b"a": 1}, {None: 1}])
def test_foreign_or_nonstring_inputs(native, value):
    assert native.dict_keys_exact_str(value) is False


def test_dict_subclass_is_rejected_without_protocol_callbacks(native):
    seen = []

    class Foreign(dict):
        def __iter__(self):
            seen.append("iter")
            raise AssertionError

        def __len__(self):
            seen.append("len")
            raise AssertionError

        def __getitem__(self, key):
            seen.append("getitem")
            raise AssertionError

        def keys(self):
            seen.append("keys")
            raise AssertionError

        def items(self):
            seen.append("items")
            raise AssertionError

    value = Foreign(a=1)
    assert native.dict_keys_exact_str(value) is False
    assert seen == []


@pytest.mark.parametrize("string_subclass", [False, True])
def test_key_hash_and_equality_are_not_replayed(native, string_subclass):
    seen = []
    armed = False
    base = str if string_subclass else object

    class Key(base):
        def __hash__(self):
            seen.append("hash")
            if armed:
                raise AssertionError("hash callback")
            return hash("collision")

        def __eq__(self, other):
            seen.append("eq")
            if armed:
                raise AssertionError("equality callback")
            return False

    key = Key("different") if string_subclass else Key()
    value = {"collision": object(), key: object()}
    seen.clear()
    armed = True
    assert native.dict_keys_exact_str(value) is False
    assert seen == []


def test_values_are_not_observed_and_key_mutations_are_live(native):
    class Value:
        def __bool__(self):
            raise AssertionError("value bool")

        def __iter__(self):
            raise AssertionError("value iter")

        def __eq__(self, other):
            raise AssertionError("value eq")

    value = {"a": Value()}
    assert native.dict_keys_exact_str(value) is True
    value[7] = Value()
    assert native.dict_keys_exact_str(value) is False
    del value[7]
    assert native.dict_keys_exact_str(value) is True
    value.clear()
    assert native.dict_keys_exact_str(value) is True


@pytest.mark.parametrize("new_entry", [None, 7])
def test_present_old_or_broken_image_fails_loud(new_entry):
    methods = {
        name: lambda *args: True
        for name in ("implementation_unchanged", "same_items", "identity_probes", "type_version", "mro_absent")
    }
    if new_entry is not None:
        methods["dict_keys_exact_str"] = new_entry
    with pytest.raises(ImportError, match="dict_keys_exact_str"):
        cache._select_validation_native(lambda name: SimpleNamespace(**methods), lambda name: object(), {})


def test_absence_remains_pure():
    def forbidden(name):
        raise AssertionError("absent helper imported")

    assert cache._select_validation_native(forbidden, lambda name: None, {}) is None


@pytest.mark.parametrize("dependency", ["type", "str", "any", "dict"])
def test_replaced_python_dependency_keeps_original_scan(monkeypatch, dependency):
    seen = []

    def forbidden(mapping):
        raise AssertionError("native bypassed replaced dependency")

    native = SimpleNamespace(dict_keys_exact_str=forbidden)
    if dependency == "type":

        def replacement(value):
            seen.append(value)
            return type(value)

        expected = True
    elif dependency == "str":
        replacement = object()
        expected = False
    elif dependency == "dict":
        replacement = type("ForeignDict", (dict,), {})
        expected = True
    else:

        def replacement(values):
            seen.append("any")
            return any(values)

        expected = True
    monkeypatch.setattr(fast, dependency, replacement, raising=False)
    assert fast._exact_str_keys(native, {"a": 1, "b": 2}) is expected
    if dependency == "type":
        assert seen == ["a", "b"]
    elif dependency == "any":
        assert seen == ["any"]


@pytest.mark.parametrize("raises", [False, True])
def test_shadowed_dict_dependency_preserves_integrated_iteration(native, monkeypatch, raises):
    import sys
    from types import ModuleType

    seen = []
    error = RuntimeError("dict dependency observer")

    class Dict(dict):
        def __iter__(self):
            seen.append("iter")
            if raises:
                raise error
            return dict.__iter__(self)

    fake = ModuleType("sys")
    vars(fake).update(vars(sys))
    fake.modules = Dict()
    monkeypatch.setattr(fast, "sys", fake)
    monkeypatch.setattr(fast, "dict", Dict, raising=False)
    if raises:
        with pytest.raises(RuntimeError) as caught:
            fast._framework_probes(native)
        assert caught.value is error
    else:
        fast._framework_probes(native)
    assert seen == ["iter"]
