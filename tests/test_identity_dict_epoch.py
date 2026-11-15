"""Dictionary certificates never cover mutable children or observer effects."""

import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

import pytest


@pytest.fixture(scope="module")
def image():
    from triton_msl.backend import _cache_contract as cache

    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    assert cache._validation_native is not None, "this suite requires the selected native helper"
    path = Path(cache._validation_native.__file__).resolve()
    spec = importlib.util.spec_from_file_location("_validation_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(image):
    image.key_cache_reset()
    yield image
    image.key_cache_reset()


def warm(native, probes):
    assert native.identity_probes(probes) is True
    before = native.probe_cache_info()["hits"]
    assert native.identity_probes(probes) is True
    assert native.probe_cache_info()["hits"] == before + 1


@pytest.mark.parametrize("change", ["replace", "delete", "clear", "update", "insert"])
def test_mutation_invalidates_dictionary_value_and_key_certificate(native, change):
    original, absent = object(), object()
    mapping = {"value": original}
    probes = ((4, mapping, "value", absent, original), (4, mapping, "new", absent, absent))
    warm(native, probes)
    if change == "replace":
        mapping["value"] = object()
    elif change == "delete":
        del mapping["value"]
    elif change == "clear":
        mapping.clear()
    elif change == "update":
        mapping.update(value=object())
    else:
        mapping["new"] = object()
    assert native.identity_probes(probes) is False
    mapping.clear()
    mapping["value"] = original
    warm(native, probes)


def test_empty_clone_and_restoration_cannot_bypass_lookup(native):
    absent = object()
    mapping = {}
    probes = ((4, mapping, "value", absent, absent),)
    warm(native, probes)
    mapping.update({"value": object(), "other": object()})
    assert native.identity_probes(probes) is False
    mapping.clear()
    warm(native, probes)


def test_dependency_fallback_then_primary_stays_live(native):
    value = object()
    primary, fallback = {}, {"name": value}
    probes = ((5, primary, fallback, "name", value),)
    warm(native, probes)
    fallback["name"] = object()
    assert native.identity_probes(probes) is False
    primary["name"] = value
    warm(native, probes)


def test_foreign_colliding_key_still_declines_without_equality(native):
    calls = []

    class Foreign(str):
        def __eq__(self, other):
            calls.append(other)
            return super().__eq__(other)

        __hash__ = str.__hash__

    value, absent = object(), object()
    mapping = {"name": value}
    probes = ((4, mapping, "name", absent, value),)
    warm(native, probes)
    mapping.clear()
    mapping[Foreign("name")] = value
    assert native.identity_probes(probes) is False
    assert calls == []


@pytest.mark.parametrize("reset", [False, True])
def test_observer_changes_later_dict_during_hot_execution(native, reset):
    value, absent = object(), object()
    mapping = {"name": value}
    calls = []
    change = [False]

    class Owner:
        @property
        def value(self):
            calls.append("read")
            if change[0]:
                if reset:
                    native.key_cache_reset()
                mapping["name"] = object()
            return None

    probes = ((1, Owner(), "value", None, None), (4, mapping, "name", absent, value))
    warm(native, probes)
    change[0] = True
    assert native.identity_probes(probes) is False
    assert calls == ["read", "read", "read"]


def test_mutable_expected_lists_are_never_certified(native):
    value = object()
    mapping = {"name": value}
    keys, values = ["name"], [value]
    probes = ((6, mapping, keys, values, None),)
    warm(native, probes)
    values[0] = object()
    assert native.identity_probes(probes) is False


def test_in_place_list_and_closure_mutation_stays_live(native):
    def make(value):
        return lambda: value

    fn = make(object())
    values = ["path"]
    probes = ((8, values, ("path",), None, None), (9, fn, "__closure__", None, fn()))
    warm(native, probes)
    values.append("new")
    assert native.identity_probes(probes) is False
    values.pop()
    warm(native, probes)
    fn.__closure__[0].cell_contents = object()
    assert native.identity_probes(probes) is False


def test_type_and_metaclass_owner_identity_stays_live(native):
    class Meta(type):
        pass

    class Owner(metaclass=Meta):
        value = None

    tag = native.type_version(Owner, ((12, Owner, "value", object(), None),))
    assert tag
    probes = ((11, Owner, None, None, Meta), (13, Owner, None, None, tag))
    warm(native, probes)
    Owner.changed = True
    assert native.identity_probes(probes) is False


def test_disabled_watcher_cache_retains_normal_checks(native):
    value, absent = object(), object()
    mapping = {"name": value}
    probes = ((4, mapping, "name", absent, value),)
    warm(native, probes)
    native.key_cache_enabled(False)
    hits = native.probe_cache_info()["hits"]
    assert native.identity_probes(probes) is True
    assert native.probe_cache_info()["hits"] == hits
    mapping["name"] = object()
    assert native.identity_probes(probes) is False


def test_full_program_cache_does_not_evict_or_silently_skip(native):
    absent = object()
    held = []
    capacity = native.probe_cache_info()["capacity"]
    for _ in range(capacity + 3):
        value = object()
        mapping = {"name": value}
        probes = ((4, mapping, "name", absent, value),)
        held.append(probes)
        assert native.identity_probes(probes) is True
    assert native.probe_cache_info()["entries"] == capacity
    native.probe_cache_collect()
    assert native.probe_cache_info()["entries"] == capacity, "held programs must survive collection"
    probes = held[-1]
    hits = native.probe_cache_info()["hits"]
    assert native.identity_probes(probes) is True
    assert native.probe_cache_info()["hits"] == hits
    probes[0][1]["name"] = object()
    assert native.identity_probes(probes) is False
    held.clear()
    native.probe_cache_collect()
    assert native.probe_cache_info()["entries"] == 0, "orphaned programs must free their slots"
    probes[0][1]["name"] = probes[0][4]
    warm(native, probes)


def test_no_release_during_identity_probes_and_cold_collection_reenters_safely(native):
    calls = []
    held = []

    class OnDelete:
        def __del__(self):
            calls.append("released")
            native.probe_cache_collect()  # no recursive drain
            native.key_cache_reset()
            replacement = ((0, object(), None, None, None),)
            # Make a valid probe; keep it live after the finalizer returns.
            replacement = ((0, replacement, None, None, replacement),)
            warm(native, replacement)
            held.append(replacement)

    for _ in range(20):
        owner = OnDelete()
        probes = ((0, owner, None, None, owner),)
        warm(native, probes)
        del owner, probes
        before = len(calls)
        # Hot probing cannot collect a dead program or run its finalizer.
        live = ((0, None, None, None, None),)
        assert native.identity_probes(live)
        assert len(calls) == before
        native.probe_cache_collect()
        assert len(calls) == before + 1
        assert native.key_cache_info()["enabled"] == 1
        warm(native, held[-1])  # outer cleanup must not overwrite this slot
        held.clear()


def test_collect_keeps_the_program_active_inside_its_own_observer(native):
    calls = []

    class Owner:
        @property
        def value(self):
            native.probe_cache_collect()
            calls.append("read")
            return None

    probes = ((1, Owner(), "value", None, None),)
    warm(native, probes)
    assert calls == ["read", "read"]
    assert native.probe_cache_info()["entries"] == 1


def test_unwatched_dictionary_on_full_registry_never_certifies(native):
    hold = [{"n": i} for i in range(native.key_cache_info()["capacity"])]
    for mapping in hold:
        assert native.dict_keys_exact_str(mapping)
    value, absent = object(), object()
    mapping = {"name": value}
    probes = ((4, mapping, "name", absent, value),)
    builds = native.probe_cache_info()["builds"]
    assert native.identity_probes(probes) is True
    assert native.probe_cache_info()["builds"] == builds
    mapping["name"] = object()
    assert native.identity_probes(probes) is False


def test_finalizer_reset_does_not_orphan_watcher_contexts(native):
    calls = []

    class ResetOnDelete:
        def __del__(self):
            native.key_cache_reset()
            calls.append("reset")

    for _ in range(20):
        owner = ResetOnDelete()
        probes = ((0, owner, None, None, owner),)
        warm(native, probes)
        del owner, probes
        native.key_cache_reset()
        assert native.key_cache_info()["enabled"] == 1
    assert calls == ["reset"] * 20


@pytest.fixture(scope="module")
def notification_helper(tmp_path_factory):
    directory = tmp_path_factory.mktemp("notification-control")
    source = Path(__file__).with_name("identity_notification_error.c")
    output = directory / ("notification_error" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = shlex.split(sysconfig.get_config_var("LDSHARED")) + [
        "-O2",
        "-I" + sysconfig.get_path("include"),
        str(source),
        "-o",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    return output


@pytest.mark.parametrize("mode", ["dictget", "dep_primary", "dep_fallback", "new_foreign_key"])
def test_notification_error_hook_cannot_certify_pending_write(image, notification_helper, mode):
    script = Path(__file__).with_name("identity_notification_control.py")
    command = [sys.executable, str(script), image.__file__, str(notification_helper), mode]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
