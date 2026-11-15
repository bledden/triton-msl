"""Preflight specialization preserves ordered checks and callback behavior."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native():
    from triton_msl.backend import _cache_contract as cache

    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    assert cache._validation_native is not None, "this suite requires the selected native helper"
    image = Path(cache._validation_native.__file__).resolve()
    assert image.is_file()
    spec = importlib.util.spec_from_file_location("_validation_native", image)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [0, 1, 31, 32, 33, 95, 96, 97, 127, 128, 129, 260])
def test_dictionary_dedup_boundaries(native, count):
    value, absent = object(), object()
    mappings = [{"key": value} for _ in range(count)]
    rows = tuple((4, mapping, "key", absent, value) for mapping in mappings)
    assert native.identity_probes(rows + tuple(reversed(rows))) is True
    if count:
        mappings[-1]["key"] = object()
        assert native.identity_probes(rows + tuple(reversed(rows))) is False


@pytest.mark.parametrize("count", [1, 127, 128, 129, 260])
def test_schema_preflight_still_precedes_all_observers(native, count):
    calls = []

    class Owner:
        @property
        def value(self):
            calls.append("read")
            return None

    rows = ((1, Owner(), "value", None, None),) * count
    with pytest.raises(TypeError, match="malformed identity probe"):
        native.identity_probes(rows + ((0, None),))
    assert calls == []
    with pytest.raises(OverflowError):
        native.identity_probes(rows + ((1 << 200, None, None, None, None),))
    assert calls == []


@pytest.mark.parametrize("count", [1, 128, 129])
def test_early_false_still_suppresses_later_unknown_opcode(native, count):
    false = (0, object(), None, None, object())
    unknown = (999, None, None, None, None)
    assert native.identity_probes((false,) * count + (unknown,)) is False
    with pytest.raises(TypeError, match="unknown identity probe kind"):
        native.identity_probes(((0, None, None, None, None),) * count + (unknown,))


def test_callback_changes_later_live_value_without_replaying(native):
    original, changed, absent = object(), object(), object()
    mapping = {"value": original}
    calls = []

    class Owner:
        @property
        def value(self):
            calls.append("read")
            mapping["value"] = changed
            return None

    rows = ((1, Owner(), "value", None, None), (4, mapping, "value", absent, original))
    assert native.identity_probes(rows) is False
    assert calls == ["read"]
    rows = rows[:-1] + ((4, mapping, "value", absent, changed),)
    assert native.identity_probes(rows) is True
    assert calls == ["read", "read"]


def test_foreign_key_preflight_remains_callback_free(native):
    calls = []

    class Foreign(str):
        def __eq__(self, other):
            calls.append(other)
            return super().__eq__(other)

        __hash__ = str.__hash__

    value, absent = object(), object()
    mapping = {Foreign("key"): value}
    assert native.identity_probes(((4, mapping, "key", absent, value),)) is False
    assert calls == []
