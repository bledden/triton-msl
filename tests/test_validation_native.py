"""Native observations supplement the unchanged full contract, never mask failure."""

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import pytest
from tests.python_child import package_child_command
import triton

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as env
from triton_msl.backend import _framework_contract as framework


@pytest.fixture
def native():
    if cache._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    assert cache._validation_native is not None, "this suite requires the tested native binary"
    return cache._validation_native


@pytest.mark.parametrize("field", ["meta_path", "path_hooks", "path", "version", "byteorder"])
def test_remainder_of_token_stays_live(monkeypatch, field):
    # Derive the prefix from the ACTUAL integrated function, stopping before
    # filesystem discovery. No replacement implementation of the token fields.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(framework._discover_selection))
    fn = tree.body[0]
    body = []
    for node in fn.body:
        body.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "token" for t in node.targets):
            break
    assert isinstance(body[-1], ast.Assign)
    fn.body = body + [ast.Return(ast.Name("token", ast.Load()))]
    ns = dict(vars(framework))
    fake = SimpleNamespace(
        **{
            k: getattr(sys, k)
            for k in (
                "platform",
                "modules",
                "path",
                "meta_path",
                "path_hooks",
                "version",
                "implementation",
                "byteorder",
            )
        }
    )
    ns["sys"] = fake
    exec(compile(ast.fix_missing_locations(tree), str(framework.__file__), "exec"), ns)
    before = ns["_discover_selection"]()
    setattr(
        fake,
        field,
        [*getattr(fake, field), object()]
        if field != "path" and isinstance(getattr(fake, field), list)
        else [*fake.path, "additional"]
        if field == "path"
        else "changed",
    )
    after = ns["_discover_selection"]()
    assert after != before


def test_same_function_decoder_code_mutation_declines_owned_snapshot(native, monkeypatch):
    # Reuse the established private standard-codec fixture, which proves initial
    # admission before mutating the same function object; not a fabricated cache.
    from test_environment_snapshot import _isolated_snapshot, _changed_codec

    with monkeypatch.context() as patch:
        module, private, functions = _isolated_snapshot(patch)
        assert module._validation_native is native
        first = module.environment_snapshot()
        assert module.is_snapshot(first)
        decoder = functions["decode"]
        original = decoder.__code__
        decoder.__code__ = _changed_codec(decoder.__closure__[0].cell_contents).__code__
        assert module.environment_snapshot() is private
        assert private.get("TRITON_MSL_FA_HALF_ACCUM") == "1"
        decoder.__code__ = original
        assert module.environment_snapshot() is first


def test_nonstandard_all_receives_lazy_dependency_observations(native, monkeypatch):
    guard = env._implementations[0]
    assert native.implementation_unchanged(vars(env), guard) is True
    seen = []

    def ignore(values):
        seen.append(type(values).__name__)
        return "custom-result"

    with monkeypatch.context() as patch:
        patch.setattr(env, "all", ignore, raising=False)
        assert native.implementation_unchanged(vars(env), guard) == "custom-result"
    assert seen == ["generator"]


def test_function_code_audit_veto_is_not_bypassed(native):
    # Hooks cannot be unregistered: isolate this observer in a fresh CPU process.
    code = """
import sys
from triton_msl.backend import _environment_snapshot as e
n = e._validation_native
g = e._implementations[0]
events = []
error = RuntimeError('code observation veto')
def audit(name, args):
    if name == 'object.__getattr__' and args[0] is g[0] and args[1] == '__code__':
        events.append(name)
        raise error
sys.addaudithook(audit)
try:
    n.implementation_unchanged(vars(e), g)
except RuntimeError as caught:
    assert caught is error
else:
    raise AssertionError('audit veto bypassed')
assert events == ['object.__getattr__']
print('AUDIT_VETO_PRESERVED')
"""
    completed = subprocess.run(package_child_command(code), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "AUDIT_VETO_PRESERVED"


def test_same_items_rejects_foreign_mapping_without_observing_it(native):
    class Foreign(dict):
        def __iter__(self):
            raise AssertionError("foreign mapping observed")

    assert native.same_items(Foreign(), {}) is False
    value = object()
    raw = {b"a": value}
    assert native.same_items(raw, raw.copy()) is True
    assert native.same_items(raw, {b"a": object()}) is False


@pytest.mark.parametrize("key", ["all", "_native_all"])
def test_global_lookup_error_is_not_missing_builtin_fallback(native, key):
    error = RuntimeError("namespace lookup failed")

    class Collision:
        def __hash__(self):
            return hash(key)

        def __eq__(self, other):
            raise error

    ns = dict(vars(env))
    ns.pop(key, None)
    ns[Collision()] = object()
    with pytest.raises(RuntimeError) as caught:
        native.implementation_unchanged(ns, env._implementations[0])
    assert caught.value is error


def test_native_loader_absence_is_distinct_from_failure():
    def forbidden(*args):
        raise AssertionError("proved absence should not import")

    assert cache._select_validation_native(forbidden, lambda name: None, {}) is None
    error = RuntimeError("loader failure")

    def failed(name):
        raise error

    with pytest.raises(RuntimeError) as caught:
        cache._select_validation_native(failed, lambda name: object(), {})
    assert caught.value is error
    with pytest.raises(ImportError, match="implementation_unchanged"):
        cache._select_validation_native(lambda name: SimpleNamespace(), lambda name: object(), {})
