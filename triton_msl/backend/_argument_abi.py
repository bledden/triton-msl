"""Shared declared argument layout; admission requires a sealed producer record.

Consumes canonical constexpr-free flat declarations and previously captured binder
bytes. No live argument objects, imports of torch/Triton, conversion callbacks or
device allocations. A future consumer must first validate the source-bound plan.
"""

from dataclasses import dataclass
import hashlib
import json

from triton_msl.errors import MetalNonRecoverableError


class LayoutError(MetalNonRecoverableError):
    pass


@dataclass(frozen=True, slots=True)
class Layout:
    declarations: tuple
    kind: str
    pointer_indices: tuple
    scalar_indices: tuple
    physical_slots: tuple
    scalar_words: tuple
    packed_slot: int | None
    status_slot: int | None
    physical_count: int


def make_layout(declarations, *, kind, status_slots=0):
    """Model an explicitly named producer policy; never infer it from live values."""
    if type(declarations) is not tuple:
        raise LayoutError("requires an owned exact tuple of flat canonical declarations")
    if type(kind) is not str or kind not in ("direct-v1", "uint32-packed-v1"):
        raise LayoutError("unknown producer ABI")
    if type(status_slots) is not int or status_slots not in (0, 1):
        raise LayoutError("unproved status reservation")
    # Only a role here: pointee address, dtype, extent, alias and ownership proofs
    # remain required in the launcher. Nested/constexpr declarations are NOT leaves.
    for ty in declarations:
        if type(ty) is not str or not ty or ty == "constexpr" or ty == "*":
            raise LayoutError("requires flat canonical declarations")
    pointers = tuple(i for i, ty in enumerate(declarations) if ty.startswith("*"))
    scalars = tuple(i for i, ty in enumerate(declarations) if not ty.startswith("*"))
    if kind == "direct-v1":
        count = len(declarations) + status_slots
        if count > 31:
            raise LayoutError("direct bindings exceed31 slots")
        return Layout(
            declarations,
            kind,
            pointers,
            scalars,
            tuple(range(len(declarations))),
            (None,) * len(declarations),
            None,
            len(declarations) if status_slots else None,
            count,
        )
    if len(declarations) <= 31:
        raise LayoutError("packed policy requires an overflow source signature")
    if status_slots:
        raise LayoutError("packed assertion producer is not qualified")
    if not scalars or len(pointers) + 1 > 31:
        raise LayoutError("packed pointer and scalar-buffer count exceeds31 slots")
    for i in scalars:
        if declarations[i] not in ("i32", "u32", "fp32", "i1"):
            raise LayoutError("unproved packed scalar width: " + declarations[i])
    slots = [None] * len(declarations)
    words = [None] * len(declarations)
    for slot, i in enumerate(pointers):
        slots[i] = slot
    for word, i in enumerate(scalars):
        slots[i] = len(pointers)
        words[i] = word
    return Layout(
        declarations, kind, pointers, scalars, tuple(slots), tuple(words), len(pointers), None, len(pointers) + 1
    )


def assemble_scalar_bytes(layout, payloads):
    """Concatenate already checked bytes; do not observe original scalar objects.

    Only for internal layouts returned by make_layout. Type identity is not a
    certificate against forged Layout objects; restored records must be rederived
    and compared with the producer before this is used by a production consumer.
    """
    if type(layout) is not Layout or layout.kind != "uint32-packed-v1":
        raise LayoutError("requires a validated packed layout")
    if type(payloads) not in (tuple, list) or len(payloads) != len(layout.declarations):
        raise LayoutError("captured payload arity/container mismatch")
    words = []
    for ty, payload in zip(layout.declarations, payloads):
        if ty.startswith("*"):
            if payload is not None:
                raise LayoutError("pointer payload must remain absent")
            continue
        if type(payload) is not bytes:
            raise LayoutError("requires captured exact immutable scalar bytes")
        if ty == "i1":
            if payload not in (b"\x00", b"\x01"):
                raise LayoutError("boolean requires one captured zero/one byte")
            words.append(payload + b"\x00\x00\x00")
        elif ty in ("i32", "u32", "fp32") and len(payload) == 4:
            words.append(payload)
        else:
            raise LayoutError("captured width disagrees with packed declaration")
    return b"".join(words)


def canonical_type(ty):
    """Canonical source spelling; i32/u32 share a packed bit representation."""
    if type(ty) is not str:
        raise LayoutError("argument ABI requires exact declared type strings")
    if ty.startswith("*"):
        return "*" + canonical_type(ty[1:])
    return {"f16": "fp16", "f32": "fp32", "float": "fp32", "f64": "fp64", "bool": "i1"}.get(ty, ty)


def _compatible(actual, expected):
    if type(actual) is not str:
        return False
    actual = canonical_type(actual)
    if actual == expected:
        return True
    if actual.startswith("*") and expected.startswith("*"):
        # Signless MLIR also erases integer pointee signedness. The host binds
        # the same storage bits, not converted scalar values. Do not equate
        # floats, widths, or nested pointer types here.
        return any({actual[1:], expected[1:]} == {"i" + width, "u" + width} for width in ("8", "16", "32", "64"))
    # MLIR integer arguments may erase signedness, but captured source bytes
    # retain source range checking. No conversion or width change takes place.
    return actual in ("i32", "u32") and expected in ("i32", "u32")


def producer_record(args, producer, source):
    """Certify only the three audited MSL packing producers, never count alone.

    Unqualified declarations retain their prior lowering but grant no new host
    route. This is not a repair of legacy compile_shader scalar materialization.
    """
    if len(args) <= 31:
        return None
    if producer not in ("generic-msl-v1", "biased-forward-v1", "biased-backward-v1"):
        raise LayoutError("unknown argument ABI producer")
    declarations = tuple(
        ("*" if a.is_ptr else "") + canonical_type(a.dtype if hasattr(a, "dtype") else a.elem_type) for a in args
    )
    try:
        layout = make_layout(declarations, kind="uint32-packed-v1")
    except LayoutError:
        return None
    return {
        "schema": 1,
        "producer": producer,
        "kind": layout.kind,
        "declarations": list(layout.declarations),
        "pointer_indices": list(layout.pointer_indices),
        "scalar_indices": list(layout.scalar_indices),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "binary_route": None,
        "binary_sha256": None,
    }


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_record(record, *, complete=True):
    """Rederive every mapping; never trust a restored Layout or supplied indices."""
    if type(record) is not dict or set(record) != {
        "schema",
        "producer",
        "kind",
        "declarations",
        "pointer_indices",
        "scalar_indices",
        "source_sha256",
        "binary_route",
        "binary_sha256",
    }:
        raise LayoutError("missing or unknown argument ABI fields; recompile")
    if type(record["schema"]) is not int or record["schema"] != 1:
        raise LayoutError("unknown argument ABI schema; recompile")
    if type(record["producer"]) is not str or record["producer"] not in (
        "generic-msl-v1",
        "biased-forward-v1",
        "biased-backward-v1",
    ):
        raise LayoutError("unknown argument ABI producer")
    if type(record["declarations"]) is not list:
        raise LayoutError("invalid argument ABI declarations")
    layout = make_layout(tuple(record["declarations"]), kind=record["kind"])
    if layout.kind != "uint32-packed-v1":
        raise LayoutError("unsupported argument ABI certificate")
    for field, expected in (("pointer_indices", layout.pointer_indices), ("scalar_indices", layout.scalar_indices)):
        value = record[field]
        if type(value) is not list or any(type(i) is not int for i in value) or tuple(value) != expected:
            raise LayoutError("argument ABI mapping differs from declared source")
    if not _digest(record["source_sha256"]):
        raise LayoutError("invalid argument ABI source binding")
    if complete:
        if record["binary_route"] != "msl" or not _digest(record["binary_sha256"]):
            raise LayoutError("argument ABI is not bound to an MSL binary")
    elif record["binary_route"] is not None or record["binary_sha256"] is not None:
        raise LayoutError("argument ABI producer record was already finalized")
    return layout


def finish_record(metadata, source, binary):
    """Only a fresh selected MSL compilation can grant the host packed route."""
    record = metadata.get("argument_abi")
    if record is None:
        return
    validate_record(record, complete=False)
    if record["source_sha256"] != hashlib.sha256(source.encode("utf-8")).hexdigest():
        raise LayoutError("argument ABI source changed before binary compilation")
    if metadata.get("binary_route") != "msl":
        # AIR has its own positional ABI, even when the MSL stash is packed.
        metadata["argument_abi"] = None
        return
    if type(binary) is not bytes:
        raise LayoutError("argument ABI requires exact compiled binary bytes")
    record = dict(record)
    record["binary_route"] = "msl"
    record["binary_sha256"] = hashlib.sha256(binary).hexdigest()
    validate_record(record)
    metadata["argument_abi"] = record


def capture_host_layout(serialized):
    if serialized is None:
        return None, None
    record = json.loads(serialized)
    return validate_record(record), record["binary_sha256"]


def prepare_host_payload(layout, declarations, payloads, args):
    """Check this binder result against the sealed source, before allocation."""
    if type(layout) is not Layout:
        raise LayoutError("host packed argument ABI is missing; recompile")
    if len(declarations) != len(layout.declarations) or any(
        not _compatible(actual, expected) for actual, expected in zip(declarations, layout.declarations)
    ):
        raise LayoutError("live argument declarations differ from the sealed packed ABI")
    if len(args) != len(layout.declarations) or any(args[i] is None for i in layout.pointer_indices):
        raise LayoutError("nullable packed pointer representation is not qualified")
    return assemble_scalar_bytes(layout, payloads)


def pointee_storage_width(declaration):
    """Host storage width, not a numerical dtype or access-reach certificate."""
    widths = {
        "i1": 1,
        "i8": 1,
        "u8": 1,
        "i16": 2,
        "u16": 2,
        "i32": 4,
        "u32": 4,
        "i64": 8,
        "u64": 8,
        "fp16": 2,
        "bf16": 2,
        "fp32": 4,
        "fp64": 8,
        "fp8e4nv": 1,
        "fp8e5": 1,
        "fp8e4b15": 1,
        "fp8e5b16": 1,
    }
    if type(declaration) is not str or not declaration.startswith("*") or declaration[1:] not in widths:
        raise LayoutError("host packed pointee storage width is not qualified")
    return widths[declaration[1:]]
