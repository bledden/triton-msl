"""Exact, read-only snapshots of the ordinary Python environment mapping.

The fast path compares a COPY of every encoded key/value, not just the known
policy keys or the mapping's identity/size. New injection variables therefore
invalidate it too. Nonstandard environment mappings retain their live behavior.
No watcher, private dict version counter, or mutation hook is installed.
"""

import os
import builtins
import dis
import _collections_abc
import _operator
import sys
import threading
from collections.abc import Mapping
from importlib.machinery import FrozenImporter
from itertools import repeat
from operator import is_
from types import BuiltinFunctionType, CodeType, FunctionType, MappingProxyType, MethodType
from triton_msl.backend._cache_contract import _validation_native

_native_all = all if type(all) is BuiltinFunctionType and all.__self__ is builtins and all.__name__ == "all" else None
_native_map = _validation_native.map_type if _validation_native is not None else None
_native_builtins = __builtins__


def _dependency_checks(fn, iterator):
    # The nonstandard all() fallback receives lazy observations, not an eagerly
    # evaluated list. A custom consumer may stop or raise before any lookup.
    for name, expected in iterator:
        yield fn.__globals__.get(name, fn.__builtins__.get(name)) is expected


def _frozen_codes(module="os"):
    """Recognize the interpreter's implementation, not a preinstalled override.

    On interpreters without frozen os, use the unoptimized live mapping. Do not
    read arbitrary files, execute another module, or guess from function names.
    """
    try:
        root = FrozenImporter.get_code(module)
    except ImportError:
        return {}
    codes = {}

    def visit(code):
        for value in code.co_consts:
            if isinstance(value, CodeType):
                if hasattr(value, "co_qualname"):
                    codes[value.co_qualname] = value
                visit(value)

    if isinstance(root, CodeType):
        visit(root)
    return codes


_type = type(os.environ)
_get = Mapping.get
_getitem = _type.__getitem__
_iter = _type.__iter__
_copy = getattr(_type, "copy", None)
_codecs = tuple(getattr(os.environ, name, None) for name in ("encodekey", "decodekey", "encodevalue", "decodevalue"))
_codes = _frozen_codes()
_encoding = sys.getfilesystemencoding()


def _implementation(fn, code, namespace, *, getter=False):
    """Admit frozen code only with its ordinary globals and call signature.

    Equal code can be borrowed by a function with different globals/builtins.
    Resolve only the frozen code's global loads; attribute names are not globals.
    Unsupported global dependencies decline this optional optimization.
    """
    if (
        type(fn) is not FunctionType
        or code is None
        or fn.__code__ != code
        or fn.__globals__ is not namespace
        or fn.__builtins__ is not vars(builtins)
        or fn.__kwdefaults__ is not None
    ):
        return None
    defaults = fn.__defaults__
    if (
        getter
        and not (type(defaults) is tuple and len(defaults) == 1 and defaults[0] is None)
        or not getter
        and defaults is not None
    ):
        return None
    known = {
        "KeyError": KeyError,
        "list": list,
        "dict": dict,
        "isinstance": isinstance,
        "str": str,
        "TypeError": TypeError,
        "type": type,
    }
    names = {op.argval for op in dis.get_instructions(code) if op.opname == "LOAD_GLOBAL"}
    if not names <= known.keys():
        return None
    dependencies = tuple((name, known[name]) for name in sorted(names))
    guard = fn, fn.__code__, defaults, dependencies
    return guard if _implementation_unchanged(guard) else None


def _implementation_unchanged(guard):
    if _validation_native is not None and _native_all is not None:
        return _validation_native.implementation_unchanged(globals(), guard)
    fn, code, defaults, dependencies = guard
    if fn.__code__ is not code or fn.__defaults__ is not defaults or fn.__kwdefaults__ is not None:
        return False
    if _native_all is not None and all is _native_all and type(fn) is FunctionType:
        # These two function attributes are read-only dictionary references.
        # Preserve live lookups, eager default evaluation and short-circuiting;
        # only the standard all() path avoids the per-check generator.
        namespace, builtins_namespace = fn.__globals__, fn.__builtins__
        for name, expected in dependencies:
            if namespace.get(name, builtins_namespace.get(name)) is not expected:
                return False
        return True
    return all(fn.__globals__.get(name, fn.__builtins__.get(name)) is expected for name, expected in dependencies)


def _standard_encoding(fn):
    closure = getattr(fn, "__closure__", None)
    return (
        closure is not None
        and len(closure) == 1
        and type(closure[0].cell_contents) is str
        and closure[0].cell_contents == _encoding
    )


_standard = (
    os.name == "posix"  # Other environment normalization rules use live fallback.
    and _codecs[0] is _codecs[2]
    and _codecs[1] is _codecs[3]
    and _standard_encoding(_codecs[0])
    and _standard_encoding(_codecs[1])
    and getattr(_get, "__code__", None) is not None
    and _get.__code__ == _frozen_codes("_collections_abc").get("Mapping.get")
    and all(
        getattr(fn, "__code__", None) is not None and fn.__code__ == _codes.get(name)
        for fn, name in ((_getitem, "_Environ.__getitem__"), (_iter, "_Environ.__iter__"), (_copy, "_Environ.copy"))
    )
    and all(
        getattr(fn, "__code__", None) is not None
        and fn.__code__ == _codes.get(getattr(fn.__code__, "co_qualname", None))
        and getattr(fn.__code__, "co_qualname", "").startswith("_create_environ_mapping.<locals>.")
        for fn in _codecs
    )
)
_implementations = (
    (
        _implementation(
            _get, _frozen_codes("_collections_abc").get("Mapping.get"), vars(_collections_abc), getter=True
        ),
        *(
            _implementation(fn, _codes.get(name), vars(os))
            for fn, name in ((_getitem, "_Environ.__getitem__"), (_iter, "_Environ.__iter__"), (_copy, "_Environ.copy"))
        ),
        *(
            _implementation(fn, _codes.get(getattr(getattr(fn, "__code__", None), "co_qualname", None)), vars(os))
            for fn in _codecs[:2]
        ),
    )
    if _standard
    else ()
)
_standard = _standard and all(guard is not None for guard in _implementations)
_lock = threading.RLock()
_cached = None
# Do not bless an arbitrary pre-import override as an identity predicate.
# Builtin name/self identify the interpreter's _operator.is_; Python wrappers
# and later rebinding retain the original type/content fallback.
_builtin_identity = (
    is_ if type(is_) is BuiltinFunctionType and is_.__self__ is _operator and is_.__name__ == "is_" else None
)


def environment_snapshot():
    """Recheck all bytes each time; decode only after a change.

    CPython's os._Environ exposes its encoded backing dict. This is an optional
    optimization only: a foreign mapping/codec/getter uses the original mapping.
    The result is immutable only on the recognized path.
    """
    global _cached
    env = os.environ
    getter, copier = getattr(env, "get", None), getattr(env, "copy", None)
    if (
        not _standard
        or type(env) is not _type
        or _type is not getattr(os, "_Environ", None)
        or not all(_implementation_unchanged(guard) for guard in _implementations)
        or _type.__getitem__ is not _getitem
        or _type.__iter__ is not _iter
        or type(getter) is not MethodType
        or getter.__func__ is not _get
        or getter.__self__ is not env
        or type(copier) is not MethodType
        or copier.__func__ is not _copy
        or copier.__self__ is not env
        or any(
            getattr(env, name, None) is not fn
            for name, fn in zip(("encodekey", "decodekey", "encodevalue", "decodevalue"), _codecs)
        )
        or not _standard_encoding(_codecs[0])
        or not _standard_encoding(_codecs[1])
        or type(getattr(env, "_data", None)) is not dict
    ):
        return env
    with _lock:
        raw = env._data.copy()
        cached = _cached
        # The cached private copy contains ONLY exact bytes. Matching every
        # key/value by identity therefore proves their types and contents too,
        # without calling user equality or decoding. Keep strong references and
        # check length: zip/map alone would accept a truncated mapping. Changed
        # identities/order take the existing exact-type/content path below.
        if (
            _builtin_identity is not None
            and is_ is _builtin_identity
            and cached is not None
            and cached[0] is env
            and len(cached[1]) == len(raw)
            and (
                _validation_native.same_items(raw, cached[1])
                if _validation_native is not None and all is _native_all and map is _native_map
                else (all(map(is_, raw, cached[1])) and all(map(is_, raw.values(), cached[1].values())))
            )
        ):
            return cached[2]
        # environb accepts bytes subclasses whose decode/equality can depend on
        # mutable state. Equal byte payloads alone do not prove those objects
        # immutable, so check exact types BEFORE the cached equality fast path.
        if not all(map(is_, map(type, raw), repeat(bytes))) or not all(
            map(is_, map(type, raw.values()), repeat(bytes))
        ):
            return env
        if _cached is not None and _cached[0] is env and _cached[1] == raw:
            return _cached[2]
        # Decode the private copy, not the live mapping: A->B->A edits during
        # env.copy() could otherwise cache B under A's unchanged raw token.
        decodekey, decodevalue = _codecs[1], _codecs[3]
        values = {decodekey(key): decodevalue(value) for key, value in raw.items()}
        if env._data != raw:
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                "environment changed while capturing cache policy; retry with stable settings"
            )
        snapshot = MappingProxyType(values)
        _cached = env, raw, snapshot
        return snapshot


def is_snapshot(value):
    # A foreign MappingProxyType can wrap somebody else's mutable dictionary.
    # Only our own private copy proves immutability, never the wrapper type.
    cached = _cached
    return cached is not None and value is cached[2]


_standard_environment_snapshot = environment_snapshot
