"""Shared host grid mapping and conservative generic 2D zero-copy eligibility."""

import json
from functools import lru_cache


def host_launch_geometry(grid, block_size, needs_2d_grid):
    """Existing host mapping: threadgroup counts and one-dimensional groups.

    Keep this arithmetic unchanged for existing host callers. Eligibility
    checks for a new zero-copy route belong to generic_2d_geometry below.
    """
    gx, gy, gz = grid
    groups = (gx, gy, gz) if needs_2d_grid else (gx * gy * gz, 1, 1)
    return groups, (min(block_size, 1024), 1, 1)


@lru_cache(maxsize=16)
def _msl_only_policy(execution_contract):
    # Caller has already unconditionally validated this exact live stamp.
    # Reuse only parsing of an immutable value, not execution validation.
    try:
        return json.loads(execution_contract)["source"]["policy"]["USE_CPP"] is False
    except (ValueError, TypeError, KeyError):
        return False


_TWIN_KEYS = frozenset(("direct_name", "direct_block", "m_idx", "n_idx", "k_idx", "c_idx", "block_m", "block_n"))


def _twin_fields(descriptor, arguments):
    """``(name, width, m, n, k, c_idx)`` of a well-formed twin descriptor, else ``None``.

    The width must be the one the tile implies: ``32 * min(sub-tiles, 4)`` for a
    tile of whole 32x32 sub-tiles, 128 for the four-strip layout of any other.
    """
    if type(descriptor) is not dict or descriptor.keys() != _TWIN_KEYS:
        return None
    name = descriptor["direct_name"]
    if type(name) is not str or not name:
        return None
    fields = []
    for key in ("direct_block", "block_m", "block_n", "m_idx", "n_idx", "k_idx", "c_idx"):
        value = descriptor[key]
        if type(value) is not int or value < (0 if key.endswith("_idx") else 1):
            return None
        if key.endswith("_idx") and value >= len(arguments):
            return None
        fields.append(value)
    width, block_m, block_n, m_idx, n_idx, k_idx, c_idx = fields
    subtiled = block_m % 32 == 0 and block_n % 32 == 0
    if width != (32 * min((block_m // 32) * (block_n // 32), 4) if subtiled else 128):
        return None
    return name, width, arguments[m_idx], arguments[n_idx], arguments[k_idx], c_idx


def _storage_address(value):
    owner = value if hasattr(value, "untyped_storage") else getattr(value, "base", None)
    try:
        return owner.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError, TypeError):
        return None


def direct_twin(descriptor, arguments, programs):
    """Entry and threadgroup width the matmul twin descriptor selects for this call.

    Returns ``(name, width)`` when the twin may run, ``None`` when the ordinary
    entry must run, and ``False`` for a descriptor this module does not
    understand (the caller then keeps the host route). The twin shares the
    ordinary entry's grid and buffer ABI; only the dispatched function of the
    same library and its threadgroup width change.

    The twin needs tile-aligned extents, and an output whose storage no other
    pointer argument shares: its simdgroups store finished sub-tiles while
    others still load, so it does not keep the ordinary entry's load-then-store
    order for an output that aliases an operand. Operands may share storage
    with each other.

    It also needs an output that holds a whole tile for each of the ``programs``
    launched. The twin clips a surplus program only on an axis whose extent the
    source's store mask proves; on any other it stores every tile whole, where
    the ordinary entry follows the same source store domains. The checked grid
    address plan proves the actual operand loads and output stores before either
    entry is selected.
    """
    fields = _twin_fields(descriptor, arguments)
    if fields is None:
        return False
    name, width, m, n, k, c_idx = fields
    if type(m) is not int or type(n) is not int or type(k) is not int:
        return None
    if m <= 0 or n <= 0 or k <= 0 or m % descriptor["block_m"] or n % descriptor["block_n"] or k % 8:
        return None
    output = _storage_address(arguments[c_idx])
    if output is None:
        return None
    try:
        if programs * descriptor["block_m"] * descriptor["block_n"] > arguments[c_idx].numel():
            return None
    except (AttributeError, RuntimeError, TypeError):
        return None
    for index, value in enumerate(arguments):
        if index != c_idx and hasattr(value, "data_ptr") and _storage_address(value) in (output, None):
            return None
    return name, width


def generic_2d_geometry(launcher, arguments, signatures, metadata, grid, *, validated_tail_bounds=None):
    """Match existing sealed generic host geometry and direct buffer ABI.

    No source/name recognition. The launcher resolves a policy-checked source
    stash, validates packed metadata and binds live scalars before calling us.
    Unknown templates, packing, geometry or representations keep the host path.
    Tensor aliases are passed as the original views of the same Metal storage;
    no mirror, base-view substitution or deduplication is needed.

    A full-K descriptor is admissible only when the caller supplies that exact
    checked snapshot after validating it against this call's bound arguments.
    This is invocation-local evidence, never a cached launcher property. Other
    address contracts retain their existing route restrictions.
    """
    import torch

    if (
        type(launcher._msl) is not str
        or not launcher._msl
        or not _msl_only_policy(launcher._execution_contract)
        or len(metadata) != 12
        or metadata[5] is not True
        or (metadata[6] is not None and _twin_fields(metadata[6], arguments) is None)
        or any(v is not None for v in metadata[7:10])
        or (
            metadata[10] is not None
            and (
                metadata[10] is not validated_tail_bounds
                or type(validated_tail_bounds) not in (tuple, list)
                or len(validated_tail_bounds) not in (6, 7)
                or (len(validated_tail_bounds), validated_tail_bounds[0])
                not in ((6, "matmul_full_k_storage_v1"), (7, "matmul_grid_storage_v1"), (7, "matmul_grid_storage_v2"))
            )
        )
        or type(launcher._msl_block_size) is not int
        or not 1 <= launcher._msl_block_size <= 1024
        or launcher._msl_block_size != metadata[3]
        or len(grid) != 3
        or any(type(v) is not int or not 0 < v <= 0x7FFFFFFF for v in grid)
        or grid[2] != 1
        or len(arguments) != len(signatures)
        or not arguments
        or len(arguments) + (metadata[11] is not None) > 31
    ):
        return None
    dtypes = {"*fp16": torch.float16, "*fp32": torch.float32, "*i32": torch.int32, "*i64": torch.int64}
    pointers = 0
    for value, declared in zip(arguments, signatures):
        if type(declared) is not str:
            return None
        if declared.startswith("*"):
            if (
                type(value) is not torch.Tensor
                or value.device.type != "mps"
                or value.dtype != dtypes.get(declared)
                or value.layout != torch.strided
                or value.is_conj()
                or value.is_neg()
                or value.numel() <= 0
            ):
                return None
            storage = value.untyped_storage()
            offset = value.storage_offset() * value.element_size()
            size = storage.nbytes()
            if (
                storage.data_ptr() <= 0
                or offset < 0
                or offset + value.numel() * value.element_size() > size
                or value.data_ptr() != storage.data_ptr() + offset
            ):
                return None
            if not value.is_contiguous():
                # A full-storage matrix transpose is physically dense: for
                # shape (r, c), stride (1, r), the last element is r*c - 1.
                # Thus the existing numel-byte extent proves the entire span.
                # Keep the original view for raw storage binding and source
                # stride arithmetic; do not gather, rebase or substitute it.
                if (
                    declared not in {"*fp16", "*fp32"}
                    or value.dim() != 2
                    or any(d <= 1 for d in value.shape)
                    or tuple(value.stride()) != (1, value.shape[0])
                    or offset != 0
                    or value.numel() * value.element_size() != size
                ):
                    return None
            pointers += 1
        else:
            # Existing scalar representability/declared-width checks have run.
            # PyTorch's raw scalar bridge accepts float32 or signed int64;
            # lower-width integer declarations read the checked low bytes.
            if declared == "fp32":
                if type(value) is not float:
                    return None
            elif declared in {"i1", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"}:
                if type(value) not in (int, bool) or not -(1 << 63) <= value < (1 << 63):
                    return None
            else:
                return None
    if not pointers:
        return None
    groups, group_size = host_launch_geometry(grid, metadata[3], metadata[5])
    threads = tuple(count * width for count, width in zip(groups, group_size))
    if any(v > 0x7FFFFFFF for v in threads):
        return None
    return threads, group_size
