"""CPU-only selection contract for the measured attention sibling.

Selection is a load-time operation; callers
must supply facts from the actual compiled pipeline, not a device-family guess.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineFacts:
    execution_width: int
    max_threads: int
    static_shared: int


@dataclass(frozen=True)
class VariantSelection:
    cooperative: bool
    threads: int
    shared: int
    reason: str


def select_variant(facts, *, device_max_threads, device_max_shared, route_enabled):
    """Keep scalar execution unless every cooperative prerequisite is proved.

    route_enabled is a caller policy, not numerical evidence. The only current
    caller is the default-off experimental route awaiting per-route GPU review.
    This function creates no pipeline, submits nothing, and retains no resources.
    """

    def scalar(reason):
        return VariantSelection(False, 32, 0, reason)

    if route_enabled is not True:
        return scalar("cooperative route not enabled")
    if type(facts) is not PipelineFacts:
        return scalar("pipeline facts unavailable")
    values = (facts.execution_width, facts.max_threads, facts.static_shared, device_max_threads, device_max_shared)
    if any(type(value) is not int or value < 0 for value in values):
        return scalar("invalid pipeline or device facts")
    if facts.execution_width != 32:
        return scalar("cooperative pipeline execution width is not 32")
    if min(facts.max_threads, device_max_threads) < 512:
        return scalar("cooperative threadgroup exceeds pipeline or device limit")
    if facts.static_shared != 16384 or device_max_shared < facts.static_shared:
        return scalar("cooperative shared-memory facts do not match emission")
    return VariantSelection(True, 512, 16384, "qualified cooperative pipeline")


def validate_variant_geometry(selection, *, threads, shared):
    """Refuse an old scalar geometry attached to a selected cooperative entry.

    Wiring must call this before dispatch, not after clamping a stale value.
    The immutable selection is owned by the loaded pipeline, not a source stash.
    """
    from triton_msl.errors import MetalNonRecoverableError

    if (
        type(selection) is not VariantSelection
        or type(selection.cooperative) is not bool
        or type(selection.threads) is not int
        or type(selection.shared) is not int
        or type(threads) is not int
        or type(shared) is not int
        or (threads, shared) != (selection.threads, selection.shared)
        or (selection.threads, selection.shared) != ((512, 16384) if selection.cooperative else (32, 0))
    ):
        raise MetalNonRecoverableError("attention variant geometry differs from the selected pipeline; recompile")


def validate_descriptor(descriptor, *, primary, scalar_source=None, scalar_threads=32):
    """Validate the sealed attention-union member and scalar stash association.

    Validation happens at construction, before any cooperative compilation.
    This is not an admission proof for arbitrary shader text.
    """
    import hashlib
    import re

    from triton_msl.errors import MetalNonRecoverableError

    keys = {
        "kind",
        "primary",
        "entry",
        "source",
        "source_sha256",
        "scalar_sha256",
        "threads",
        "shared",
        "execution_width",
        "scalar_threads",
        "scalar_shared",
        "arg_count",
        "roles",
    }
    valid = type(descriptor) is dict and set(descriptor) == keys
    if valid:
        d = descriptor
        valid = (
            d["kind"] == "cooperative-row-v1"
            and type(d["primary"]) is str
            and d["primary"] == primary
            and type(d["entry"]) is str
            and d["entry"] == primary + "__attncoop16"
            and type(d["source"]) is str
            and bool(d["source"])
            and type(d["source_sha256"]) is str
            and hashlib.sha256(d["source"].encode()).hexdigest() == d["source_sha256"]
            and type(d["scalar_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", d["scalar_sha256"]) is not None
            and all(
                type(d[k]) is int
                for k in ("threads", "shared", "execution_width", "scalar_threads", "scalar_shared", "arg_count")
            )
            and (d["threads"], d["shared"], d["execution_width"], d["scalar_threads"], d["scalar_shared"])
            == (512, 16384, 32, 32, 0)
            and d["arg_count"] == 7
            and type(d["roles"]) is list
            and len(d["roles"]) == 7
            and all(type(i) is int and 0 <= i < d["arg_count"] for i in d["roles"])
            and len(set(d["roles"])) == 7
            and type(scalar_threads) is int
            and scalar_threads == 32
        )
        if valid and scalar_source is not None:
            valid = (
                type(scalar_source) is str and hashlib.sha256(scalar_source.encode()).hexdigest() == d["scalar_sha256"]
            )
    if not valid:
        raise MetalNonRecoverableError("cooperative attention source/geometry binding differs; recompile")
