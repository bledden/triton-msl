"""Saved-identity fast path for the execution contract.

After one recognized complete evaluation, the native helper can validate its
recorded inputs using raw reads and reuse the whole stamp. An admitted hit does
not invoke user key equality, descriptors, iterators, or metadata-observation
audit callbacks, and therefore does not reproduce their side effects. Set
``TRITON_MSL_IDENTITY_FAST_PATH=0`` when those complete-evaluator observations
are required on every validation. Unknown/custom state declines before an
exploratory callback and runs the complete evaluator once. This optimization is
not a Python security boundary and does not claim to suppress tracing, profiling,
signals, or arbitrary same-process code. Both public validations and every
launcher/native guard remain unchanged.
"""

import builtins
import _imp
import ctypes
import importlib.util
from importlib import _bootstrap_external as _namespace_importlib
import os
import sys
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MethodType,
    ModuleType,
    SimpleNamespace,
)
from importlib.machinery import ModuleSpec

import triton_msl
from . import _cache_contract as cache
from . import _environment_snapshot as env
from . import _framework_contract as fw
from . import _toolchain_contract as tc
from . import _native_generation as ng
from triton_msl.errors import MetalNonRecoverableError

(
    IDENT,
    ATTR,
    ATTRDEF,
    EQATTRDEF,
    DICTGET,
    DEPGET,
    DICTITEMS,
    IDLIST,
    EQLIST,
    CLOSURE0,
    EQTUPLEATTRDEF,
    TYPEOF,
    CLASSGET,
    TYPEVERSION,
) = range(14)
_ABSENT = object()
_CODEC_NAMES = ("encodekey", "decodekey", "encodevalue", "decodevalue")
# Standard providers, captured once at import: a replaced provider is a miss and
# the full evaluation (which honours overrides, vetoes and callbacks) runs.
_STANDARD = {
    "cache.framework_identity": cache.framework_identity,
    "cache.toolchain_identity": cache.toolchain_identity,
    "cache.implementation_identity": cache.implementation_identity,
    "fw.framework_identity": fw.framework_identity,
    "fw._discover_selection": fw._discover_selection,
}
_record = None
_stats = {"hits": 0, "misses": 0, "guard_rebuilds": 0, "builds": 0}
_GUARD_PRIMITIVES = None
if type(cache._validation_native) is ModuleType:
    _owner = cache._validation_native
    _mapping = _owner.__dict__
    if all(type(key) is str for key in _mapping):
        _values = tuple(_mapping.get(name) for name in ("identity_probes", "function_code_is", "dict_keys_exact_str"))
        if all(
            type(value) is BuiltinFunctionType and value.__self__ is _owner and value.__name__ == name
            for name, value in zip(("identity_probes", "function_code_is", "dict_keys_exact_str"), _values)
        ):
            _GUARD_PRIMITIVES = (
                _owner,
                _values[0],
                (
                    (TYPEOF, _owner, None, None, ModuleType),
                    (DICTGET, cache.__dict__, "_validation_native", None, _owner),
                    *(
                        (DICTGET, _mapping, name, None, value)
                        for name, value in zip(("identity_probes", "function_code_is", "dict_keys_exact_str"), _values)
                    ),
                ),
            )


def _guard_batch_available(native):
    selected = _GUARD_PRIMITIVES
    # The captured callback-free primitive checks the live owner and observers.
    # Its exact-key preflight declines before equality on a foreign dictionary
    # key. A custom observer is never called merely to choose a bypass.
    return selected is not None and native is selected[0] and selected[1](selected[2]) is True


_STATE_BATCH = None
if type(cache._validation_native) is ModuleType:
    _native_state = cache._validation_native.__dict__
    if all(type(key) is str for key in _native_state):
        _batch = _native_state.get("function_states_are")
        if (
            type(_batch) is BuiltinFunctionType
            and _batch.__self__ is cache._validation_native
            and _batch.__name__ == "function_states_are"
        ):
            _STATE_BATCH = _batch
_GUARD_TYPE = fw._LoadedNativeGuard
_GUARD_VERIFY = _GUARD_TYPE.verify
_GUARD_CODE = _GUARD_VERIFY.__code__
_READER_TYPE = ng._GenerationReader
_READER_CALL = _READER_TYPE.__call__
_READER_CODE = _READER_CALL.__code__
# _environment_snapshot may be the module initiating this circular import.
# The recognized os._Environ uses object's lookup; do not read env._type here.
_ENV_GETATTRIBUTE = object.__getattribute__
_ENV_TYPE = type(os.environ)
_dict_descriptor = _ENV_TYPE.__dict__.get("__dict__")
_ENV_INSTANCE_DICT = (
    _dict_descriptor
    if type(_dict_descriptor) is GetSetDescriptorType
    and _dict_descriptor.__objclass__ is _ENV_TYPE
    and _dict_descriptor.__name__ == "__dict__"
    else None
)
_INFO_TYPE = ng._ImageInfosPrefix
_INFO_GETATTRIBUTE = _INFO_TYPE.__getattribute__
_INFO_FIELDS = tuple((name, getattr(_INFO_TYPE, name)) for name, *_ in _INFO_TYPE._fields_)
_GUARD_FIELDS = (
    "count",
    "name",
    "providers",
    "checked_count",
    "approved_names",
    "checked_names",
    "_generation_reader",
    "_generation_checked",
)
_READER_FIELDS = ("info", "address", "version")
_INSTANCE_DICTS = {cls: cls.__dict__["__dict__"] for cls in (_GUARD_TYPE, _READER_TYPE)}
_SPEC_DICT = ModuleSpec.__dict__["__dict__"]
_RUNTIME_ENV_NAMES = fw._RUNTIME_SELECTION_ENV
_STANDARD_LOCK = fw._lock
_LOCK_TYPE = type(_STANDARD_LOCK)
_DISCOVER_CODE = _STANDARD["fw._discover_selection"].__code__
_NS_TYPE = getattr(_namespace_importlib, "_NamespacePath", None)
_NS_DICT = _NS_TYPE.__dict__.get("__dict__") if type(_NS_TYPE) is type else None
_NS_FIELDS = ("_name", "_path", "_last_parent_path", "_last_epoch", "_path_finder")
_NS_METHOD_NAMES = ("__iter__", "__len__", "_recalculate", "_get_parent_path", "_find_parent_path_names")


def _standard_namespace_methods():
    """Recognize frozen CPython methods, not pre-import replacements.

    This is a cold optional certificate. Unknown/private runtime layouts keep
    the complete evaluator; obtaining a certificate must not call a substituted
    Python provider or make the backend unimportable.
    """
    if type(_NS_TYPE) is not type or type(_NS_DICT) is not GetSetDescriptorType:
        return ()
    provider = _imp.__dict__.get("get_frozen_object")
    if (
        type(provider) is not BuiltinFunctionType
        or provider.__name__ != "get_frozen_object"
        or provider.__module__ != "_imp"
        or provider.__self__ is not _imp
    ):
        return ()
    try:
        module = provider("_frozen_importlib_external")
    except (ImportError, RuntimeError, TypeError, ValueError):
        return ()
    if type(module) is not CodeType:
        return ()
    classes = [value for value in module.co_consts if type(value) is CodeType and value.co_name == "_NamespacePath"]
    if len(classes) != 1:
        return ()
    methods = []
    for name in _NS_METHOD_NAMES:
        codes = [value for value in classes[0].co_consts if type(value) is CodeType and value.co_name == name]
        fn = _NS_TYPE.__dict__.get(name)
        if len(codes) != 1 or type(fn) is not FunctionType:
            return ()
        code = fn.__code__
        if not _namespace_code_equal(code, codes[0]):
            return ()
        methods.append((name, fn, code))
    return tuple(methods)


def _namespace_code_equal(actual, canonical):
    # Compare only exact builtin values, not CodeType's version-specific rich
    # comparison. These five frozen methods need only None/string/tuple constants.
    def ordinary_constants(value):
        if type(value) in (type(None), str):
            return True
        if type(value) is tuple:
            return all(ordinary_constants(item) for item in value)
        return False

    if type(actual) is not CodeType or type(canonical) is not CodeType:
        return False
    fields = [
        ("co_argcount", int),
        ("co_posonlyargcount", int),
        ("co_kwonlyargcount", int),
        ("co_nlocals", int),
        ("co_stacksize", int),
        ("co_flags", int),
        ("co_firstlineno", int),
        ("co_code", bytes),
        ("co_name", str),
        ("co_filename", str),
    ]
    fields.extend(
        (name, kind)
        for name, kind in (("co_qualname", str), ("co_linetable", bytes), ("co_exceptiontable", bytes))
        if hasattr(canonical, name)
    )
    if not hasattr(canonical, "co_linetable"):
        fields.append(("co_lnotab", bytes))
    for name, kind in fields:
        left, right = getattr(actual, name), getattr(canonical, name)
        if type(left) is not kind or type(right) is not kind or left != right:
            return False
    for name in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
        left, right = getattr(actual, name), getattr(canonical, name)
        if (
            type(left) is not tuple
            or type(right) is not tuple
            or any(type(item) is not str for values in (left, right) for item in values)
            or left != right
        ):
            return False
    try:
        return (
            ordinary_constants(actual.co_consts)
            and ordinary_constants(canonical.co_consts)
            and actual.co_consts == canonical.co_consts
        )
    except RecursionError:
        return False


_NS_METHODS = _standard_namespace_methods()
_NS_DEPENDENCIES = (("sys", sys), ("getattr", getattr), ("tuple", tuple), ("iter", iter), ("len", len))


def _ordinary_guard(guard, native):
    """Do not speculate through a user verifier, then replay it on a miss.

    Such verifiers remain supported by the complete evaluator at its original
    observation point. The saved-identity shortcut is only for the ordinary
    native guard. Check live owners, not a saved bound method.
    """
    if guard is None:
        return True
    if (
        type(guard) is not _GUARD_TYPE
        or _GUARD_TYPE.verify is not _GUARD_VERIFY
        or not native.function_code_is(_GUARD_VERIFY, _GUARD_CODE)
    ):
        return False
    state = guard.__dict__
    if type(state) is not dict or not native.dict_keys_exact_str(state):
        return False
    override = state.get("verify", _ABSENT)
    if override is not _ABSENT and (
        type(override) is not MethodType or override.__func__ is not _GUARD_VERIFY or override.__self__ is not guard
    ):
        return False
    reader = getattr(guard, "_generation_reader", None)
    return (
        type(reader) is _READER_TYPE
        and type(reader.__dict__) is dict
        and native.dict_keys_exact_str(reader.__dict__)
        and _READER_TYPE.__call__ is _READER_CALL
        and native.function_code_is(_READER_CALL, _READER_CODE)
    )


def _guard_probes(guard, native):
    """Certify only the callback-free generation branch, never the image scan.

    Nonstandard readers/structures keep the complete evaluator. The guard's
    owned inputs must remain the same objects; ctypes hooks are observed too.
    A changed generation subsequently declines *before* count/name are called.
    """
    if guard is None:
        return []
    # Decline custom metaclass lookups before Python reads any class attribute.
    # The native cold check inspects tp_getattro without invoking the override.
    if any(not native.type_version(cls, ()) for cls in (_GUARD_TYPE, _READER_TYPE, _INFO_TYPE)):
        return None
    # Check inherited descriptors before touching instance-owned fields. Looking
    # only in the own dictionary speculatively executes a base-class observer.
    if not native.mro_absent(_GUARD_TYPE, _GUARD_FIELDS) or not native.mro_absent(_READER_TYPE, _READER_FIELDS):
        return None
    if (
        type(guard) is not _GUARD_TYPE
        or _GUARD_TYPE.__getattribute__ is not object.__getattribute__
        or getattr(_GUARD_TYPE, "__getattr__", None) is not None
        or _GUARD_TYPE.__dict__.get("__dict__") is not _INSTANCE_DICTS[_GUARD_TYPE]
        or any(name in _GUARD_TYPE.__dict__ for name in _GUARD_FIELDS)
        or not _ordinary_guard(guard, native)
    ):
        return None
    reader = guard._generation_reader
    if (
        _READER_TYPE.__getattribute__ is not object.__getattribute__
        or getattr(_READER_TYPE, "__getattr__", None) is not None
        or _READER_TYPE.__dict__.get("__dict__") is not _INSTANCE_DICTS[_READER_TYPE]
        or any(name in _READER_TYPE.__dict__ for name in _READER_FIELDS)
        or type(reader.info) is not _INFO_TYPE
        or _INFO_TYPE.__getattribute__ is not _INFO_GETATTRIBUTE
        or getattr(_INFO_TYPE, "__getattr__", None) is not None
        or reader.info.__dict__
        or type(reader.address) is not int
        or type(reader.version) is not int
    ):
        return None
    generation = guard._generation_checked
    if type(generation) is not tuple or len(generation) != 3 or any(type(value) is not int for value in generation):
        return None
    probes = []
    for cls, fields in ((_GUARD_TYPE, _GUARD_FIELDS), (_READER_TYPE, _READER_FIELDS)):
        probes.append((CLASSGET, cls, "__dict__", _ABSENT, _INSTANCE_DICTS[cls]))
        for name in fields:
            probes.append((CLASSGET, cls, name, _ABSENT, _ABSENT))
    for obj, cls, lookup in (
        (guard, _GUARD_TYPE, object.__getattribute__),
        (reader, _READER_TYPE, object.__getattribute__),
        (reader.info, _INFO_TYPE, _INFO_GETATTRIBUTE),
    ):
        probes += [
            (TYPEOF, obj, None, None, cls),
            (ATTR, cls, "__getattribute__", None, lookup),
            (ATTRDEF, cls, "__getattr__", None, None),
            (ATTR, obj, "__dict__", None, obj.__dict__),
        ]
    # verify's original bound-method override is separately certified by
    # _ordinary_guard. Unused extra attributes cannot affect this branch.
    state = guard.__dict__
    if any(name not in state for name in _GUARD_FIELDS):
        return None
    for name in _GUARD_FIELDS:
        probes.append((ATTR, guard, name, None, state[name]))
    for obj in (guard, reader, reader.info):
        state = obj.__dict__
        probes.append((DICTITEMS, state, list(state.keys()), list(state.values()), None))
        probes.append((DICTGET, state, "__triton_msl_identity_absent__", _ABSENT, _ABSENT))
    for name, field in _INFO_FIELDS:
        if getattr(_INFO_TYPE, name) is not field:
            return None
        probes.append((ATTR, _INFO_TYPE, name, None, field))
    for function in (guard.count, guard.name):
        cls = type(function)
        if (
            not native.type_version(cls, ())
            or type(cls) is not type(ctypes._CFuncPtr)
            or ctypes._CFuncPtr not in cls.__mro__
            or cls.__getattribute__ is not ctypes._CFuncPtr.__getattribute__
            or getattr(cls, "__getattr__", None) is not None
            or any(name in cls.__dict__ for name in ("restype", "argtypes", "errcheck"))
        ):
            return None
        probes += [
            (TYPEOF, function, None, None, cls),
            (ATTR, cls, "__getattribute__", None, ctypes._CFuncPtr.__getattribute__),
            (ATTRDEF, cls, "__getattr__", None, None),
        ]
        # These hooks cannot be reached by allow_scan=False, but mutations are
        # still misses rather than silently approving a changed scanner.
        for name in ("restype", "argtypes", "errcheck"):
            probes.append((CLASSGET, cls, name, _ABSENT, _ABSENT))
            value = getattr(function, name, None)
            probes.append((ATTRDEF, function, name, None, value))
            if name == "argtypes" and type(value) is list:
                probes.append((IDLIST, value, tuple(value), None, None))
    return probes


def _framework_provider_probes():
    g, f = cache.__dict__, fw.__dict__
    return [
        (DICTGET, g, "framework_identity", _ABSENT, _STANDARD["cache.framework_identity"]),
        (DICTGET, f, "framework_identity", _ABSENT, _STANDARD["fw.framework_identity"]),
        (DICTGET, f, "_discover_selection", _ABSENT, _STANDARD["fw._discover_selection"]),
        (DICTGET, f, "_discovery_snapshot", _ABSENT, fw._discovery_snapshot),
        (DICTGET, f, "_owned_selection", _ABSENT, fw._owned_selection),
        (DICTGET, f, "_REQUIRED_NAMES", _ABSENT, fw._REQUIRED_NAMES),
        (DICTGET, f, "_SELECTION_NAMES", _ABSENT, fw._SELECTION_NAMES),
        (DICTGET, f, "_RUNTIME_SELECTION_ENV", _ABSENT, _RUNTIME_ENV_NAMES),
    ]


def _whole_provider_probes(schema, label):
    g, e, f, t, b, top = (
        cache.__dict__,
        env.__dict__,
        fw.__dict__,
        tc.__dict__,
        builtins.__dict__,
        triton_msl.__dict__,
    )
    probes = [
        (DICTGET, g, "source_contract", _ABSENT, cache._standard_source_contract),
        (DICTGET, g, "effective_policy", _ABSENT, cache._standard_effective_policy),
        (DICTGET, g, "_policy_for_environment", _ABSENT, cache._standard_policy_for_environment),
        (DICTGET, g, "toolchain_identity", _ABSENT, cache._standard_toolchain_identity),
        (DICTGET, g, "framework_identity", _ABSENT, _STANDARD["cache.framework_identity"]),
        (DICTGET, g, "implementation_identity", _ABSENT, _STANDARD["cache.implementation_identity"]),
        (DICTGET, g, "SOURCE_SCHEMA", _ABSENT, schema),
        (DICTGET, f, "framework_identity", _ABSENT, _STANDARD["fw.framework_identity"]),
        (DICTGET, t, "toolchain_identity", _ABSENT, tc._standard_toolchain_identity),
        (DICTGET, t, "_identity_for_environment", _ABSENT, tc._standard_identity_for_environment),
        (DICTGET, e, "environment_snapshot", _ABSENT, env._standard_environment_snapshot),
        (DEPGET, e, b, "all", builtins.all),
        (DEPGET, e, b, "map", builtins.map),
        (DICTGET, e, "is_", _ABSENT, env._builtin_identity),
        (DICTGET, e, "_standard", _ABSENT, True),
        (DICTGET, top, "CODEGEN_VERSION", _ABSENT, label),
    ]
    # These are the exact LOAD_GLOBAL fallbacks accepted by the full
    # environment implementation recognizer. Native DEPGET preflights both
    # dictionaries before lookup, so foreign/colliding keys decline without
    # consuming equality and an ordinary rebinding becomes a clean miss.
    for fn, _code, _defaults, dependencies in env._implementations:
        for name, expected in dependencies:
            probes.append((DEPGET, fn.__globals__, fn.__builtins__, name, expected))
    return probes


_KEY_SCAN_TYPE, _KEY_SCAN_STR, _KEY_SCAN_ANY, _KEY_SCAN_DICT = type, str, any, dict


def _exact_str_keys(native, mapping):
    # Preserve the original Python observation path if its builtin dependencies
    # are replaced. With ordinary builtins the native predicate invokes no
    # Python callbacks; dictionary exactness is still checked by the caller.
    if type is _KEY_SCAN_TYPE and str is _KEY_SCAN_STR and any is _KEY_SCAN_ANY and dict is _KEY_SCAN_DICT:
        return native.dict_keys_exact_str(mapping)
    return not any(type(key) is not str for key in mapping)


def _namespace_probes(native, locations, name, function_states):
    """Certify an already-settled ordinary top-level importlib namespace.

    Never iterate or normalize it here. The full evaluation settles the path;
    changes to parent paths, invalidate_caches' epoch, the live path, owners or
    importlib implementations invalidate the record before any shortcut hit.
    Nested namespaces and custom implementations keep the complete path.
    """
    if not _NS_METHODS or type(locations) is not _NS_TYPE or type(name) is not str or "." in name:
        return None
    checks = (
        (ATTR, _NS_TYPE, "__getattribute__", None, object.__getattribute__),
        (ATTRDEF, _NS_TYPE, "__getattr__", None, None),
        (CLASSGET, _NS_TYPE, "__dict__", _ABSENT, _NS_DICT),
        *((CLASSGET, _NS_TYPE, key, _ABSENT, fn) for key, fn, _ in _NS_METHODS),
    )
    if not native.type_version(_NS_TYPE, checks) or not native.mro_absent(_NS_TYPE, _NS_FIELDS):
        return None
    state = locations.__dict__
    if (
        type(state) is not dict
        or not _exact_str_keys(native, state)
        or len(state) != len(_NS_FIELDS)
        or any(key not in state for key in _NS_FIELDS)
    ):
        return None
    path, previous, epoch = state["_path"], state["_last_parent_path"], state["_last_epoch"]
    sys_state = sys.__dict__
    if not _exact_str_keys(native, sys_state):
        return None
    modules, parent_path = sys_state.get("modules"), sys_state.get("path")
    if (
        type(modules) is not dict
        or not _exact_str_keys(native, modules)
        or modules.get("sys") is not sys
        or type(parent_path) is not list
        or type(state["_name"]) is not str
        or state["_name"] != name
        or type(path) not in (list, tuple)
        or type(previous) is not tuple
        or type(epoch) is not int
        or any(type(value) is not str for seq in (parent_path, path, previous) for value in seq)
    ):
        return None
    # Access only after the exact ordinary metaclass/lookup has been certified.
    current_epoch = _NS_TYPE.__dict__.get("_epoch")
    if type(current_epoch) is not int or current_epoch != epoch or tuple(parent_path) != previous:
        return None
    probes = [
        *checks,
        (TYPEOF, locations, None, None, _NS_TYPE),
        (ATTR, locations, "__dict__", None, state),
        (DICTITEMS, state, list(state.keys()), list(state.values()), None),
        (CLASSGET, _NS_TYPE, "_epoch", _ABSENT, current_epoch),
        (DICTGET, modules, "sys", _ABSENT, sys),
        (DICTGET, sys_state, "path", _ABSENT, parent_path),
        (EQLIST, parent_path, tuple(parent_path), None, None),
    ]
    if type(path) is list:
        probes.append((EQLIST, path, tuple(path), None, None))
    states = []
    for _key, fn, code in _NS_METHODS:
        saved = native.function_state(fn)
        if (
            saved is None
            or saved[0] is not code
            or saved[1] is not None
            or saved[2] is not None
            or saved[3] is not None
            or saved[4] != ()
        ):
            return None
        states.append((fn, saved))
        probes.extend((DEPGET, fn.__globals__, fn.__builtins__, key, value) for key, value in _NS_DEPENDENCIES)
    probes = _type_versions(native, probes)
    if probes is None or not native.identity_probes(probes):
        return None
    function_states.extend(states)
    return probes


def _framework_probes(native, namespace_states=None):
    """Record only callback-free import metadata; custom owners stay full-path.

    Settled ordinary top-level namespaces can use a raw state certificate;
    other observable namespace objects still decline without a second iteration.
    """
    if type(sys) is not ModuleType or type(sys.modules) is not dict or not _exact_str_keys(native, sys.modules):
        return None
    spec_checks = (
        (ATTR, ModuleSpec, "__getattribute__", None, object.__getattribute__),
        (ATTRDEF, ModuleSpec, "__getattr__", None, None),
        (CLASSGET, ModuleSpec, "__dict__", _ABSENT, _SPEC_DICT),
    )
    # Native cold lookup rejects custom metaclasses before reading attributes.
    if not native.type_version(ModuleSpec, spec_checks) or not native.mro_absent(
        ModuleSpec, ("origin", "submodule_search_locations")
    ):
        return None
    sys_state = sys.__dict__
    util_state = importlib.util.__dict__
    implementation = sys.implementation
    if (
        type(sys_state) is not dict
        or not _exact_str_keys(native, sys_state)
        or type(util_state) is not dict
        or not _exact_str_keys(native, util_state)
        or type(implementation) is not SimpleNamespace
        or type(implementation.__dict__) is not dict
        or not _exact_str_keys(native, implementation.__dict__)
    ):
        return None
    probes = [
        (TYPEOF, sys, None, None, ModuleType),
        (DICTGET, sys_state, "modules", _ABSENT, sys.modules),
        *spec_checks,
        (TYPEOF, implementation, None, None, SimpleNamespace),
        (ATTR, implementation, "__dict__", None, implementation.__dict__),
    ]
    modules = sys.modules
    for name in fw._SELECTION_NAMES:
        module = modules.get(name, _ABSENT)
        probes.append((DICTGET, modules, name, _ABSENT, module))
        if module is _ABSENT:
            continue
        if type(module) is not ModuleType:
            return None
        state = module.__dict__
        # An exact dict may still contain a str subclass whose equality runs on
        # lookup. Certify keys by iteration before any get/in operation. These
        # cold checks never run on an ordinary recorded hit.
        if type(state) is not dict or not _exact_str_keys(native, state):
            return None
        if "__spec__" not in state and "__getattr__" in state:
            return None
        spec = state.get("__spec__")
        probes += [
            (TYPEOF, module, None, None, ModuleType),
            (DICTGET, state, "__spec__", _ABSENT, state.get("__spec__", _ABSENT)),
        ]
        if "__spec__" not in state:
            probes.append((DICTGET, state, "__getattr__", _ABSENT, _ABSENT))
        if spec is not None:
            if type(spec) is not ModuleSpec:
                return None
            fields = spec.__dict__
            if type(fields) is not dict or not _exact_str_keys(native, fields):
                return None
            origin = fields.get("origin")
            locations = fields.get("submodule_search_locations")
            if origin is not None and type(origin) is not str:
                return None
            namespace = locations is not None and type(locations) not in (list, tuple)
            if namespace:
                if namespace_states is None:
                    return None
                extra = _namespace_probes(native, locations, name, namespace_states)
                if extra is None:
                    return None
                probes.extend(extra)
            elif locations is not None and any(type(value) is not str for value in locations):
                return None
            probes += [
                (TYPEOF, spec, None, None, ModuleSpec),
                (ATTR, spec, "__dict__", None, fields),
                (DICTGET, fields, "origin", _ABSENT, fields.get("origin", _ABSENT)),
                (
                    DICTGET,
                    fields,
                    "submodule_search_locations",
                    _ABSENT,
                    fields.get("submodule_search_locations", _ABSENT),
                ),
            ]
            if locations is not None and not namespace:
                probes.append((EQLIST, locations, tuple(locations), None, None))
    if (
        type(sys.path) not in (list, tuple)
        or any(type(p) is not str for p in sys.path)
        or type(sys.meta_path) not in (list, tuple)
        or type(sys.path_hooks) not in (list, tuple)
    ):
        return None
    probes += [
        (DICTGET, sys_state, "path", _ABSENT, sys.path),
        (EQLIST, sys.path, tuple(sys.path), None, None),
        (DICTGET, sys_state, "meta_path", _ABSENT, sys.meta_path),
        (IDLIST, sys.meta_path, tuple(sys.meta_path), None, None),
        (DICTGET, sys_state, "path_hooks", _ABSENT, sys.path_hooks),
        (IDLIST, sys.path_hooks, tuple(sys.path_hooks), None, None),
        (DICTGET, util_state, "find_spec", _ABSENT, importlib.util.find_spec),
        (DICTGET, sys_state, "version", _ABSENT, sys.version),
        (DICTGET, sys_state, "implementation", _ABSENT, implementation),
        (DICTGET, implementation.__dict__, "cache_tag", _ABSENT, implementation.cache_tag),
        (DICTGET, sys_state, "byteorder", _ABSENT, sys.byteorder),
        # The stamp was derived from these exact snapshots; a rebuilt inventory
        # or a new native guard is a new derivation.
        (DICTGET, fw.__dict__, "_snapshot", _ABSENT, fw._snapshot),
        (DICTGET, fw.__dict__, "_native_guard", _ABSENT, fw._native_guard),
    ]
    return _type_versions(native, probes)


def _framework_environment_probes(native):
    os_state = os.__dict__
    env_state = env.__dict__
    if (
        type(os_state) is not dict
        or not _exact_str_keys(native, os_state)
        or type(env_state) is not dict
        or not _exact_str_keys(native, env_state)
    ):
        return None
    mapping = os_state.get("environ", _ABSENT)
    dict_owner = ((CLASSGET, env._type, "__dict__", _ABSENT, _ENV_INSTANCE_DICT),)
    if (
        _ENV_INSTANCE_DICT is None
        or type(mapping) is not env._type
        or env._type is not _ENV_TYPE
        or not native.type_version(env._type, dict_owner)
        or not native.mro_absent(env._type, ("_data", *_CODEC_NAMES))
        or env._type.__getattribute__ is not _ENV_GETATTRIBUTE
        or getattr(env._type, "__getattr__", None) is not None
        or "_data" in env._type.__dict__
    ):
        return None
    instance = mapping.__dict__
    data = object.__getattribute__(mapping, "_data")
    if (
        type(instance) is not dict
        or not _exact_str_keys(native, instance)
        or type(data) is not dict
        or not native.dict_keys_exact_bytes(data)
    ):
        return None
    # Recognize the frozen ordinary implementations established by the full
    # environment evaluator; never bless whichever code happens to be current
    # when this optional record is built.
    guards = (
        env._implementations[0],
        env._implementations[1],
        env._implementations[2],
        env._implementations[3],
        env._implementations[4],
        env._implementations[5],
        env._implementations[4],
        env._implementations[5],
    )
    functions = (env._get, env._getitem, env._iter, env._copy, *env._codecs)
    states = tuple((fn, native.function_state(fn)) for fn in functions)
    for index, ((fn, state), guard) in enumerate(zip(states, guards)):
        if (
            state is None
            or fn is not guard[0]
            or state[0] is not guard[1]
            or state[1] is not guard[2]
            or state[2] is not None
        ):
            return None
        closure, contents = state[3], state[4]
        if index < 4:
            if closure is not None or contents != ():
                return None
        elif (
            type(closure) is not tuple
            or len(closure) != 1
            or type(contents) is not tuple
            or len(contents) != 1
            or type(contents[0]) is not str
            or contents[0] != env._encoding
        ):
            return None
    probes = _type_versions(
        native,
        (
            (DICTGET, os_state, "environ", _ABSENT, mapping),
            (TYPEOF, mapping, None, None, env._type),
            (CLASSGET, env._type, "__dict__", _ABSENT, _ENV_INSTANCE_DICT),
            (ATTR, mapping, "__dict__", None, instance),
            (ATTR, env._type, "__getattribute__", None, _ENV_GETATTRIBUTE),
            (ATTRDEF, env._type, "__getattr__", None, None),
            (ATTR, env._type, "get", None, env._get),
            (ATTR, env._type, "__getitem__", None, env._getitem),
            (ATTR, env._type, "__iter__", None, env._iter),
            (ATTR, env._type, "copy", None, env._copy),
            (DICTGET, instance, "get", _ABSENT, _ABSENT),
            (DICTGET, instance, "__getitem__", _ABSENT, _ABSENT),
            (DICTGET, instance, "__iter__", _ABSENT, _ABSENT),
            (DICTGET, instance, "copy", _ABSENT, _ABSENT),
            (DICTGET, instance, "_data", _ABSENT, data),
            *((ATTR, mapping, name, None, fn) for name, fn in zip(_CODEC_NAMES, env._codecs)),
        ),
    )
    if probes is None:
        return None
    raw = []
    for name in _RUNTIME_ENV_NAMES:
        key = name.encode()
        value = data.get(key)
        if value is not None and type(value) is not bytes:
            return None
        raw.append((key, value))
    return probes, data, tuple(raw), states


def _environment_probes(native):
    mapping = os.environ
    if type(mapping) is not env._type or env._cached is None or env._cached[0] is not mapping:
        return None
    if not native.type_version(env._type, ()):
        return None
    # A descriptor or custom lookup may itself be an observer. Decline without
    # invoking it speculatively; the complete evaluator owns those callbacks.
    if (
        not native.mro_absent(env._type, ("_data", *_CODEC_NAMES))
        or env._type.__getattribute__ is not _ENV_GETATTRIBUTE
        or getattr(env._type, "__getattr__", None) is not None
        or any(name in env._type.__dict__ for name in ("_data", *_CODEC_NAMES))
    ):
        return None
    instance = mapping.__dict__
    data = getattr(mapping, "_data", None)
    if type(data) is not dict:
        return None
    raw = env._cached[1]  # the private copy: exact bytes, insertion order
    probes = [
        (ATTR, os, "environ", None, mapping),
        (TYPEOF, mapping, None, None, env._type),
        (ATTR, env._type, "__getattribute__", None, _ENV_GETATTRIBUTE),
        (ATTRDEF, env._type, "__getattr__", None, None),
        (ATTR, mapping, "__dict__", None, instance),
        (ATTR, os, "_Environ", None, env._type),
        (ATTR, env._type, "__getitem__", None, env._getitem),
        (ATTR, env._type, "__iter__", None, env._iter),
        (ATTR, env._type, "copy", None, env._copy),
        (ATTR, env._type, "get", None, env._get),
        (DICTGET, instance, "get", _ABSENT, _ABSENT),  # no instance-level override
        (DICTGET, instance, "copy", _ABSENT, _ABSENT),
    ]
    for name in ("_data", *_CODEC_NAMES):
        probes.append((CLASSGET, env._type, name, _ABSENT, _ABSENT))
    for name, codec in zip(_CODEC_NAMES, env._codecs):
        probes.append((ATTR, mapping, name, None, codec))
    for codec in env._codecs[:2]:
        probes.append((CLOSURE0, codec, "__closure__", None, codec.__closure__[0].cell_contents))
    for guard in env._implementations:
        fn, code, defaults, dependencies = guard
        probes += [
            (ATTR, fn, "__code__", None, code),
            (ATTR, fn, "__defaults__", None, defaults),
            (ATTR, fn, "__kwdefaults__", None, None),
        ]
        for name, expected in dependencies:
            probes.append((DEPGET, fn.__globals__, fn.__builtins__, name, expected))
    probes.append((ATTR, mapping, "_data", None, data))
    probes.append((DICTITEMS, data, list(raw.keys()), list(raw.values()), None))
    probes.append((DICTGET, tc.__dict__, "_snapshot", _ABSENT, tc._snapshot))
    probes.append((ATTR, triton_msl, "CODEGEN_VERSION", None, triton_msl.CODEGEN_VERSION))
    return probes


def _type_versions(native, probes):
    """Collapse verified class predicates, not live instance/function checks.

    CPython invalidates a subtype's nonzero tag when any base changes. Capture
    tags only after callback-free verification of every replaced predicate;
    uncacheable types keep the complete evaluator. Put type guards first so a
    newly installed descriptor cannot execute before its invalidation is seen.
    Metaclass tags are retained too. The tuple owns strong type references.
    """
    classes = {}
    remaining = []
    for probe in probes:
        kind, owner, *_ = probe
        if kind in (ATTR, ATTRDEF, CLASSGET) and isinstance(owner, type):
            classes.setdefault(owner, []).append(probe)
        else:
            remaining.append(probe)
    versions = []
    for cls, checks in tuple(classes.items()):
        classes.setdefault(type(cls), [])
    for cls, checks in classes.items():
        tag = native.type_version(cls, tuple(checks))
        if not tag:
            return None
        # A metaclass replacement can leave the class's own tag unchanged.
        versions.append((TYPEOF, cls, None, None, type(cls)))
        versions.append((TYPEVERSION, cls, None, None, tag))
    return tuple(versions + remaining)


def build(stamp):
    """Record the identities behind ``stamp`` after a complete owned evaluation."""
    global _record
    native = cache._validation_native
    if native is None or native.identity_fast_path_disabled(os) is not False:
        _record = None
        return
    if native is None or not env._standard or env._builtin_identity is None or fw._snapshot is None:
        _record = None
        return
    # Full framework discovery serializes selection and guard observation under
    # this lock.  Only the original callback-free RLock is eligible: a replaced
    # lock remains the complete evaluator's responsibility and is acquired once.
    if (
        type(fw.__dict__) is not dict
        or not native.dict_keys_exact_str(fw.__dict__)
        or fw.__dict__.get("_lock", _ABSENT) is not _STANDARD_LOCK
        or type(_STANDARD_LOCK) is not _LOCK_TYPE
    ):
        _record = None
        return
    with _STANDARD_LOCK:
        _build_locked(stamp, native)


def _build_locked(stamp, native):
    global _record
    namespace_states = []
    framework = _framework_probes(native, namespace_states)
    runtime = _framework_environment_probes(native)
    if framework is None or runtime is None:
        _record = None
        return
    guard_probes = _guard_probes(fw._native_guard, native)
    if guard_probes is None:
        _record = None
        return
    probes = (
        (DICTGET, fw.__dict__, "_lock", _ABSENT, _STANDARD_LOCK),
        *tuple(_framework_provider_probes()),
        *framework,
    )
    guard_probes = _type_versions(native, guard_probes)
    if guard_probes is None:
        _record = None
        return
    _record = (
        tuple(probes),
        fw._snapshot[1],
        fw._native_guard,
        guard_probes,
        runtime[0],
        runtime[1],
        runtime[2],
        _STANDARD_LOCK,
        runtime[3] + tuple(namespace_states),
    )
    _stats["builds"] += 1


def finalize(stamp, phase):
    """Attach only the later inputs actually selected by this full evaluation."""
    global _record
    record = _record
    native = cache._validation_native
    if native is None or native.identity_fast_path_disabled(os) is not False:
        _record = None
        return
    if record is None or len(record) != 9 or native is None:
        _record = None
        return
    (
        schema,
        implementation,
        frameworks,
        policy_provider,
        capture,
        policy_helper,
        environment,
        compiler_provider,
        compiler,
        label,
    ) = phase
    cached = env._cached
    if (
        frameworks is not record[1]
        and frameworks != record[1]
        or policy_provider is not cache._standard_effective_policy
        or capture is not env._standard_environment_snapshot
        or policy_helper is not cache._standard_policy_for_environment
        or compiler_provider is not cache._standard_toolchain_identity
        or type(cached) is not tuple
        or len(cached) != 3
        or cached[0] is not os.environ
        or cached[2] is not environment
        or type(cached[1]) is not dict
        or not native.dict_keys_exact_bytes(cached[1])
        or not native.same_items(record[5], cached[1])
    ):
        _record = None
        return
    probes = _type_versions(
        native,
        tuple(_whole_provider_probes(schema, label))
        + (
            (DICTGET, env.__dict__, "_cached", _ABSENT, cached),
            (DICTGET, tc.__dict__, "_snapshot", _ABSENT, tc._snapshot),
        ),
    )
    if probes is None:
        _record = None
        return
    # Values, not merely provider objects, must still be the phase values used
    # to derive this stamp before a complete record can be admitted.
    snapshot = cache._execution_snapshot
    if (
        type(snapshot) is not tuple
        or len(snapshot) != 7
        or snapshot[:6] != (environment, schema, implementation, frameworks, label, compiler)
        or snapshot[6] != stamp
    ):
        _record = None
        return
    _record = record + (probes, cached[1], stamp)


def check():
    """Return a whole saved stamp after raw proof, omitting full observation callbacks."""
    global _record
    native = cache._validation_native
    disabled = None if native is None else native.identity_fast_path_disabled(os)
    if disabled is not False:
        _stats["misses"] += 1
        return None
    record = _record
    if record is None or len(record) != 12:
        if record is not None:
            _record = None
        _stats["misses"] += 1
        return None
    lock = record[7]
    if native is None or type(lock) is not _LOCK_TYPE:
        _record = None
        _stats["misses"] += 1
        return None
    # Acquire before any framework observation, matching framework_identity.
    # If the live lock changed, the raw identity probe declines while holding
    # only the old ordinary RLock; fallback then acquires the replacement once.
    with lock:
        return _check_locked(record, native)


def _function_states_hold(native, states):
    if _STATE_BATCH is not None:
        result = _STATE_BATCH(native, states, any)
        if result is not NotImplemented:
            return result
    # Keep the original lazy generator and observer calls for replaced helpers,
    # custom aggregators, old native images and unsupported owned structures.
    return not any(not native.function_state_is(fn, state) for fn, state in states)


def _check_locked(record, native):
    global _record
    guard = record[2]
    if (
        not native.function_code_is(_STANDARD["fw._discover_selection"], _DISCOVER_CODE)
        or not native.identity_probes(record[0])
        or not (guard_probe := native.identity_probes)(record[3])
        or not (
            # The successful ordinary guard program has already checked exact
            # owner types, complete instance dictionaries, method overrides and
            # class/MRO version tags. Cold admission checked method identities.
            # Keep both live code checks HERE, at their original position.
            (
                native.function_code_is(_GUARD_VERIFY, _GUARD_CODE)
                and native.function_code_is(_READER_CALL, _READER_CODE)
            )
            if (
                guard is not None
                and _GUARD_PRIMITIVES is not None
                and guard_probe is _GUARD_PRIMITIVES[1]
                and _guard_batch_available(native)
            )
            else _ordinary_guard(guard, native)
        )
        or not native.identity_probes(record[4])
        or not native.dict_keys_exact_bytes(record[5])
        or not native.bytes_dict_items_are(record[5], record[6])
        or not _function_states_hold(native, record[8])
    ):
        # At least one recorded identity changed: the record is stale by
        # definition. Drop it so the next complete evaluation records afresh.
        _record = None
        _stats["misses"] += 1
        return None
    if guard is not None:
        try:
            guard.verify(allow_scan=False)
        except fw._NativeSelectionChanged:
            _record = None
            _stats["guard_rebuilds"] += 1
            return None
        except BaseException as exc:
            _record = None
            if isinstance(exc, (OSError, ValueError, RuntimeError, TypeError)) and not isinstance(
                exc, MetalNonRecoverableError
            ):
                raise MetalNonRecoverableError(f"cannot establish framework package identity: {exc}") from exc
            raise
    if not native.identity_probes(record[9]) or not native.same_items(record[5], record[10]):
        _record = None
        _stats["misses"] += 1
        return None
    _stats["hits"] += 1
    return record[11]
