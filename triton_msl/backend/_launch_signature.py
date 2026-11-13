"""Declared source positions and scalar byte widths; no Python-value width guess.

This is not yet the full dynamic argument/pipeline certificate. Pointer storage,
tuple-copyback mapping and pre-hook validation are separate consumers to audit.
"""

import struct
from collections import namedtuple
from types import MappingProxyType

from triton_msl.errors import MetalNonRecoverableError


def _refuse(reason):
    raise MetalNonRecoverableError("source signature: " + reason)


def _canonical_type(ty):
    if isinstance(ty, tuple):
        return tuple(_canonical_type(t) for t in ty)
    if not isinstance(ty, str):
        _refuse("unknown declared type")
    if ty.startswith("*"):
        return "*" + _canonical_type(ty[1:])
    return {"f16": "fp16", "f32": "fp32", "f64": "fp64"}.get(ty, ty)


def ordered_source_signature(src):
    signature = getattr(src, "signature", None)
    if not isinstance(signature, dict):
        _refuse("missing signature map")
    fn = getattr(src, "fn", None)
    if fn is not None and hasattr(fn, "arg_names"):
        original = list(fn.arg_names)
        if len(set(original)) != len(original) or any(type(k) is not str or k not in original for k in signature):
            _refuse("AST signature does not match the declared parameter names")
        # JIT signatures include constexpr entries; direct ASTSource signatures
        # may omit them. Do not insert an omitted parameter into runtime positions.
        names = [name for name in original if name in signature]
    else:
        if any(type(k) is not int for k in signature) or sorted(signature) != list(range(len(signature))):
            _refuse("IRSource positions must be dense integers starting at zero")
        names = list(range(len(signature)))
    return names, {name: _canonical_type(signature[name]) for name in names}


_INTEGER = {
    "i1": "?",
    "u1": "?",
    "i8": "b",
    "u8": "B",
    "i16": "h",
    "u16": "H",
    "i32": "i",
    "u32": "I",
    "i64": "q",
    "u64": "Q",
}


def scalar_bytes(value, declared_type):
    """Pack exactly the declared scalar width, or refuse an unproved conversion."""
    ty = _canonical_type(declared_type)
    try:
        if ty in _INTEGER:
            if not isinstance(value, (int, bool)) or (ty in ("i1", "u1") and value not in (0, 1)):
                _refuse(f"{ty} requires a representable integer value")
            return struct.pack("<" + _INTEGER[ty], value)
        if ty in ("fp16", "fp32", "bf16"):
            if not isinstance(value, (float, int, bool)):
                _refuse(f"{ty} requires a numeric scalar")
            if ty == "bf16":
                # Preserve the existing host bf16 rounding/NaN behavior.
                import torch

                bits = torch.tensor([value], dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item()
                return struct.pack("<h", bits)
            return struct.pack("<" + ("e" if ty == "fp16" else "f"), value)
    except (struct.error, OverflowError, ValueError) as exc:
        _refuse(f"value cannot be represented as {ty}: {exc}")
    _refuse(f"unsupported or missing scalar declaration {ty!r}; no width is inferred from the Python value")


def bind_arguments(args, names, signature, *, _plan=None):
    """Exact tuple shape and constexpr-free TTGIR positions, before hooks.

    Return source-ordered leaves and prepacked scalar payloads. Tensor storage
    validation remains the driver's responsibility; this does not guess types
    from tensors or claim a foreign pipeline's identity.
    """
    if len(args) != len(names):
        _refuse(f"expected {len(names)} runtime parameter positions, got {len(args)}")
    leaves, types, origins, payloads = [], [], [], []
    packers = _plan.packers if type(_plan) is _BindingPlan else None
    # Flat declarations use the inline path. Build the recursive helper only
    # when a live declaration first needs it, without inspecting later names.
    bind = None
    for index, (value, name) in enumerate(zip(args, names)):
        # Keep each declaration read at its original position, after all prior
        # callbacks. Exact string leaves can run inline; tuple/custom declarations
        # retain the original recursive path without replaying an observed prefix.
        ty = signature[name]
        if type(ty) is not str:
            if bind is None:

                def bind(value, ty, origin):
                    if isinstance(ty, tuple):
                        if not isinstance(value, tuple) or len(value) != len(ty):
                            _refuse("tuple argument structure does not match its declaration")
                        for child, child_ty in zip(value, ty):
                            bind(child, child_ty, origin)
                        return
                    if ty == "constexpr":
                        return
                    if isinstance(value, tuple):
                        _refuse("tuple value in a scalar/pointer parameter")
                    if ty.startswith("*"):
                        if value is not None and not callable(getattr(value, "data_ptr", None)):
                            _refuse("pointer parameter requires a tensor/pointer wrapper or None")
                        payload = None
                    else:
                        packing = packers.get(ty) if packers is not None and type(ty) is str else None
                        value_type = type(value)
                        if packing is not None and (
                            value_type is int or value_type is bool or (packing[0] and value_type is float)
                        ):
                            if ty in ("i1", "u1") and value not in (0, 1):
                                _refuse(f"{ty} requires a representable integer value")
                            try:
                                payload = packing[1](value)
                            except (struct.error, OverflowError, ValueError) as exc:
                                _refuse(f"value cannot be represented as {ty}: {exc}")
                        else:
                            # Per-node fallback preserves callback ordering; never restart
                            # the already-observed prefix of this invocation.
                            payload = scalar_bytes(value, ty)
                    leaves.append(value)
                    types.append(ty)
                    origins.append(origin)
                    payloads.append(payload)

            bind(value, ty, index)
            continue
        if ty == "constexpr":
            continue
        if isinstance(value, tuple):
            _refuse("tuple value in a scalar/pointer parameter")
        if ty.startswith("*"):
            if value is not None and not callable(getattr(value, "data_ptr", None)):
                _refuse("pointer parameter requires a tensor/pointer wrapper or None")
            payload = None
        else:
            packing = packers.get(ty) if packers is not None and type(ty) is str else None
            value_type = type(value)
            if packing is not None and (
                value_type is int or value_type is bool or (packing[0] and value_type is float)
            ):
                if ty in ("i1", "u1") and value not in (0, 1):
                    _refuse(f"{ty} requires a representable integer value")
                try:
                    payload = packing[1](value)
                except (struct.error, OverflowError, ValueError) as exc:
                    _refuse(f"value cannot be represented as {ty}: {exc}")
            else:
                # Per-node fallback preserves callback ordering; never restart
                # the already-observed prefix of this invocation.
                payload = scalar_bytes(value, ty)
        leaves.append(value)
        types.append(ty)
        origins.append(index)
        payloads.append(payload)
    return leaves, types, origins, payloads


_BindingPlan = namedtuple("_BindingPlan", "packers native", defaults=(None,))


def make_binding_plan(names, signature):
    """Cache canonical scalar packers, never live declaration traversal.

    A pointer getter or custom scalar can change a later public declaration.
    bind_arguments must retain its original sequential reads across callbacks.
    This private immutable map contains only type-specific struct packers;
    current names, positions, tuple shapes and values are still read per call.
    """
    if type(names) is not list or type(signature) is not dict:
        return None
    packers = {}
    for ty in signature.values():
        if type(ty) is not str:
            continue
        if ty in _INTEGER:
            packers[ty] = (False, struct.Struct("<" + _INTEGER[ty]).pack)
        elif ty in ("fp16", "fp32"):
            packers[ty] = (True, struct.Struct("<" + ("e" if ty == "fp16" else "f")).pack)
    if not packers:
        return None
    if _binder_native is not None:
        parts = _binder_native.make_parts(packers)
        if parts is not None:
            return _BindingPlan(*parts)
    return _BindingPlan(MappingProxyType(packers))


def bind_arguments_with_plan(args, names, signature, plan):
    return bind_arguments(args, names, signature, _plan=plan)


def _select_binder_native():
    # Genuine absence alone selects Python; broken native imports remain errors.
    from importlib import import_module
    from importlib.util import find_spec
    import sys

    name = __package__ + "._binder_native"
    if name not in sys.modules and find_spec(name) is None:
        return None
    native = import_module(name)
    validation_name = __package__ + "._validation_native"
    census = None
    if validation_name in sys.modules or find_spec(validation_name) is not None:
        validation = import_module(validation_name)
        census = getattr(validation, "dict_keys_exact_str", None)
    if not native.configure(bind_arguments, _BindingPlan, bind_arguments_with_plan, census):
        return None
    return native


_binder_native = _select_binder_native()
if _binder_native is not None:
    bind_arguments_with_plan = _binder_native.bind_with_plan
