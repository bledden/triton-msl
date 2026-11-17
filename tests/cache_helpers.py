"""Shared helpers for stateful compiler/runtime tests."""

import os
from pathlib import Path
import sys

# Direct fuzzer scripts import this as `cache_helpers`, outside the tests package.
# Resolve their backend from the same checkout before importing the shared helper.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton.runtime.jit import JITFunction
from triton_msl.profiling.cache_session import private_directory


def patch_live_singleton_method(monkeypatch, getter, method_name, wrapper_factory):
    """Patch the method on the exact singleton consumed by production.

    Patching a singleton's class is not sufficient: another test can install an
    instance attribute with the same name, which shadows the class-level spy.
    ``wrapper_factory`` receives the live instance and its currently bound
    method, and must return the call-through replacement.
    """
    instance = getter()
    original = getattr(instance, method_name)
    replacement = wrapper_factory(instance, original)
    if not callable(original) or not callable(replacement):
        raise TypeError(f"{method_name} observer must wrap callable methods")
    if method_name in vars(instance):
        monkeypatch.setattr(instance, method_name, replacement)
    else:
        # Undo must REMOVE the instance attribute. ``setattr`` would restore the
        # looked-up value as an instance attribute, and when the class is patched
        # too that value is the class-level replacement, which then outlives the test.
        monkeypatch.setitem(vars(instance), method_name, replacement)
    if getter() is not instance or getattr(instance, method_name) is not replacement:
        raise AssertionError(f"{method_name} was not patched on the live singleton")
    return instance


def fresh_compiler_caches(namespace):
    """Rotate to newly owned disk caches and invalidate this module's JIT entries.

    Retain previous directories: they may still back live executable descriptors.
    This intentionally does not promise a cold Metal driver or Inductor cache.
    The caller must not invoke this concurrently with launches from this module.
    """
    root = private_directory(prefix="triton-msl-test-cold-")
    for key, child in (("TRITON_MSL_CACHE_DIR", "msl"), ("TRITON_CACHE_DIR", "triton")):
        directory = root / child
        directory.mkdir()
        os.environ[key] = str(directory)
    for value in tuple(namespace.values()):
        if isinstance(value, JITFunction):
            value.device_caches.clear()
    return root
