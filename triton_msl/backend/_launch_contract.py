"""Versioned producer-to-consumer packed descriptor, with no restore defaults.

This binds descriptor contents, not the dynamic tensor argument ABI or a foreign
pipeline's binary. Those boundaries must not infer protection from this record.
"""

import json
import math
from functools import lru_cache

from triton_msl.errors import MetalNonRecoverableError


FIELDS = (
    "num_warps",
    "num_ctas",
    "shared",
    "block_size",
    "output_arg_indices",
    "needs_2d_grid",
    "mm_two_kernel",
    "fast_matmul",
    "quant_matmul",
    "flash_attention",
    "batched_dot_bounds",
    "device_assert",
)


def _refuse(reason):
    raise MetalNonRecoverableError("packed launch contract: " + reason + "; recompile the kernel")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        _refuse(f"unrepresentable descriptor ({exc})")


def _record(metadata):
    required = ("name", "execution_contract", *FIELDS)
    if any(name not in metadata for name in required):
        _refuse("missing producer field")
    if type(metadata["name"]) is not str or not metadata["name"]:
        _refuse("invalid kernel name")
    for name in ("num_warps", "num_ctas", "block_size"):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            _refuse(f"invalid {name}")
    if type(metadata["shared"]) is not int or metadata["shared"] < 0:
        _refuse("invalid shared-memory declaration")
    if type(metadata["needs_2d_grid"]) is not bool:
        _refuse("invalid grid mode")
    outputs = metadata["output_arg_indices"]
    attention = metadata["flash_attention"]
    if type(attention) is dict and attention.get("kind") == "cooperative-row-v1":
        from ._attention_variant import validate_descriptor

        validate_descriptor(attention, primary=metadata["name"], scalar_threads=metadata["block_size"])
        if metadata["shared"] not in (0, 1) or metadata["needs_2d_grid"]:
            _refuse("cooperative attention primary must keep scalar32/0 geometry")
    assertion = metadata["device_assert"]
    if assertion is not None:
        if (
            type(assertion) is not dict
            or set(assertion) != {"schema", "messages", "buffer_index"}
            or type(assertion["schema"]) is not int
            or assertion["schema"] != 1
            or type(assertion["buffer_index"]) is not int
            or not 0 <= assertion["buffer_index"] < 31
            or type(assertion["messages"]) is not list
            or not assertion["messages"]
            or any(type(m) is not str for m in assertion["messages"])
        ):
            _refuse("invalid device assertion descriptor")
        if any(metadata[name] is not None for name in FIELDS[6:11]):
            _refuse("device assertions require the generic launch ABI")
    if outputs is not None and (
        type(outputs) not in (list, tuple)
        or any(type(i) is not int or i < 0 for i in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        _refuse("invalid output indices")
    record = {
        "schema": 2,
        "kind": "metal-packed-launch",
        "name": metadata["name"],
        "execution_contract": metadata["execution_contract"],
        "fields": {name: metadata[name] for name in FIELDS},
    }
    abi = metadata.get("argument_abi")
    if abi is not None:
        from ._argument_abi import validate_record

        validate_record(abi)
        if assertion is not None or any(metadata[name] is not None for name in FIELDS[6:9]):
            _refuse("packed argument ABI cannot certify assertion or replacement bindings")
        record["schema"] = 3
        record["argument_abi"] = abi
    return record


def seal_launch_metadata(metadata):
    """Called ONLY after successful fresh compilation, never during restore.

    Explicit producer defaults retain existing fresh IRSource semantics. Missing
    fields in a restored record are never interpreted through these defaults.
    """
    if "num_warps" not in metadata or "num_ctas" not in metadata:
        _refuse("compiler omitted launch options")
    defaults = {
        "shared": 0,
        "block_size": metadata["num_warps"] * 32,
        "output_arg_indices": None,
        "needs_2d_grid": False,
        **{name: None for name in FIELDS[6:]},
    }
    for name, value in defaults.items():
        metadata.setdefault(name, value)
    metadata["launch_contract"] = _canonical(_record(metadata))


def validate_launch_metadata(metadata):
    """Validate a full producer record; return an immutable packed-value snapshot."""
    return validate_launch_products(metadata)[0]


def validate_launch_products(metadata):
    """Capture ABI sidecar and hot descriptor from one checked producer record.

    Schema2 grants no new packed host route. The optional schema3 ABI is
    captured immutably at construction, never restored from inferred defaults.
    The per-call twelve-field descriptor remains unchanged.
    """
    if isinstance(metadata, dict):
        values = metadata
    elif hasattr(metadata, "__dict__"):
        values = vars(metadata)
    elif hasattr(metadata, "_asdict"):
        values = metadata._asdict()
    else:
        _refuse("unrecognized metadata container")
    record = _canonical(_record(values))
    if type(values.get("launch_contract")) is not str or values["launch_contract"] != record:
        _refuse("missing, altered or incompatible producer record")
    # The serialized copy cannot alias a mutable list/dict in restored metadata.
    abi = json.loads(record).get("argument_abi")
    return _canonical([values[name] for name in FIELDS]), (_canonical(abi) if abi is not None else None)


def validate_packed_launch(packed, expected):
    if type(packed) not in (tuple, list) or len(packed) != len(FIELDS):
        _refuse("wrong packed descriptor arity")
    # Template descriptors carry whole immutable MSL strings. Re-encoding and
    # decoding those strings on every call costs more than checking the small
    # mutable descriptor around them. Small records retain the cheaper JSON path.
    if type(expected) is str and len(expected) >= 1024:
        plan = _packed_snapshot_plan(expected)
        if plan is not None:
            try:
                return tuple(_checked_packed_copy(packed, plan))
            except _UsePackedJSON:
                # JSON supports some custom containers/scalars and non-string
                # dictionary keys. Preserve its existing behavior for those.
                pass
    if type(expected) is not str or _canonical(packed) != expected:
        _refuse("descriptor changed after launcher construction")
    # Hooks can hold references to the caller's tuple and nested lists. Execute
    # from an unaliased checked snapshot, not mutable data checked before a hook.
    return tuple(json.loads(expected))


class _UsePackedJSON(Exception):
    pass


@lru_cache(maxsize=64)
def _packed_snapshot_plan(expected):
    """Only canonical JSON supplies a reusable, recursively immutable plan."""
    try:
        decoded = json.loads(expected)
        if _canonical(decoded) != expected:
            return None
    except (ValueError, TypeError, OverflowError, RecursionError, MetalNonRecoverableError):
        return None

    has_large_string = False

    def freeze(value, depth=0):
        nonlocal has_large_string
        if depth > 32:
            raise _UsePackedJSON
        kind = type(value)
        if kind is list:
            return kind, tuple(freeze(child, depth + 1) for child in value)
        if kind is dict:
            return kind, tuple((key, freeze(child, depth + 1)) for key, child in value.items())
        if kind is str and len(value) >= 512:
            has_large_string = True
        return kind, value

    try:
        plan = freeze(decoded)
        # Large scalar-heavy arrays do not have the measured shader-string
        # cost. Keep those on their original C JSON encoder/decoder path.
        return plan if has_large_string else None
    except (RecursionError, _UsePackedJSON):
        return None


def _checked_packed_copy(value, plan):
    """Check every live value and allocate fresh containers for this invocation.

    Builtin leaf type equality preserves JSON's bool/int/float distinctions;
    signed floating zero also has distinct canonical spelling. Tuple/list input
    equivalence is preserved, and both produce new lists as json.loads does.
    No mutable object from either the caller or an earlier return is reused.
    """
    kind, wanted = plan
    if kind is list:
        if type(value) is not list and type(value) is not tuple:
            raise _UsePackedJSON
        observed = tuple(value)
        if len(observed) != len(wanted):
            _refuse("descriptor changed after launcher construction")
        return [_checked_packed_copy(child, child_plan) for child, child_plan in zip(observed, wanted)]
    if kind is dict:
        if type(value) is not dict:
            raise _UsePackedJSON
        observed = value.copy()
        if any(type(key) is not str for key in observed):
            raise _UsePackedJSON
        if len(observed) != len(wanted) or any(key not in observed for key, _ in wanted):
            # An explicit Unicode surrogate pair and its decoded character
            # can spell the same JSON object key. Let the encoder decide.
            raise _UsePackedJSON
        return {key: _checked_packed_copy(observed[key], child_plan) for key, child_plan in wanted}
    if type(value) is not kind:
        raise _UsePackedJSON
    if kind is str and value != wanted:
        # JSON's ASCII escaping can equate a surrogate pair with one Unicode
        # character even though the Python strings compare unequal.
        raise _UsePackedJSON
    if value != wanted or (kind is float and value == 0.0 and math.copysign(1.0, value) != math.copysign(1.0, wanted)):
        _refuse("descriptor changed after launcher construction")
    return wanted


def pack_launch_metadata(metadata):
    expected = validate_launch_metadata(metadata)
    # Version 2 appends the assertion descriptor; nested JSON arrays remain lists,
    # as they already are after Triton's persistent metadata restoration.
    return tuple(json.loads(expected))


# The optional host-only helper performs the same checked copy, not codegen or
# device execution. A genuinely absent extension retains the original Python
# implementation. Broken imports/initializers are not classified as absence.
def _select_checked_copy(import_module, find_spec, modules):
    name = __package__ + "._packed_native"
    # The loader's exception name is not evidence that it never ran. An
    # installed module can raise ModuleNotFoundError naming itself. Establish
    # absence before invoking any loader, then propagate every import failure.
    # Preserve sys.modules selection, including its explicit None/block sentinel.
    if name not in modules and find_spec(name) is None:
        return None
    native = import_module(name)
    if not callable(getattr(native, "copy", None)):
        raise ImportError("packed-copy extension has no callable copy entry point")
    return native


from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
import sys as _sys

_checked_packed_copy_python = _checked_packed_copy
_packed_native = _select_checked_copy(_import_module, _find_spec, _sys.modules)
PACKED_COPY_IMPLEMENTATION = "python" if _packed_native is None else "native"
if _packed_native is not None:

    def _checked_packed_copy(value, plan):
        return _packed_native.copy(value, plan, globals())
