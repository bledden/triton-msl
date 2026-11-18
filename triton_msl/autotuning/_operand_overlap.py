"""Does a replacement launch's output overlap an operand it reads?

A source program loads every operand before it stores, so a launch of one program whose
output shares memory with an operand is defined: the result is computed from the operand's
old contents. A replacement template spreads that program over several threadgroups, which
store finished tiles while others still load, so it does not keep that order. A dispatcher
therefore declines the replacement when the output may receive bytes an operand is read
from.

A kernel addresses a tensor through its own stride and length arguments, so the view a
caller passes does not bound the cells the kernel reaches through it. Where a route knows
the element count it addresses from a pointer, the byte ranges are compared and views of
one storage that stay apart remain eligible. For every other tensor the comparison is of
whole storages: an output that shares a storage with it is reported, and so is an output
beside an argument whose storage cannot be read.

No Triton import: the dispatchers that use this stay importable without backend discovery.
"""


def _base(value):
    # triton.reinterpret wraps a tensor and addresses the same bytes.
    return value if hasattr(value, "untyped_storage") else value.base


def _span(value):
    """Elements from the first to the last one a view addresses."""
    span = 1
    for size, stride in zip(value.shape, value.stride()):
        if size == 0:
            return 0
        span += (size - 1) * abs(stride)
    return span


def _storage(value):
    """(device type, first byte, one past the last byte) of the storage behind an argument.

    A host address range on the CPU. On the device the first entry identifies the
    allocation and the range carries no address meaning.
    """
    value = _base(value)
    storage = value.untyped_storage()
    start = storage.data_ptr()
    return value.device.type, start, start + storage.nbytes()


def _share(one, other):
    """Do two storages hold a common byte? An empty storage holds none."""
    if one[0] != other[0] or one[1] == one[2] or other[1] == other[2]:
        return False
    if one[0] == "cpu":
        # Two host storages can cover the same memory without being one object.
        return one[1] < other[2] and other[1] < one[2]
    return one[1] == other[1]


def _addressed(value, elements):
    """(storage, first byte, one past the last byte) of ``elements`` from a view's pointer.

    Host addresses on the CPU, offsets into the allocation on the device. The range ends
    with the storage: a count that runs past it reaches no byte of this tensor, and on the
    host those addresses may belong to an unrelated allocation.
    """
    value = _base(value)
    storage = _storage(value)
    start = value.storage_offset() * value.element_size()
    end = storage[2] - storage[1]
    if storage[0] == "cpu":
        start += storage[1]
        end = storage[2]
    return storage, start, min(start + elements * value.element_size(), end)


def _meet(one, other):
    """Do two addressed ranges hold a common byte?"""
    (kind, allocation, _), first, last = one
    (other_kind, other_allocation, _), start, stop = other
    if kind != other_kind or (kind != "cpu" and allocation != other_allocation):
        return False
    return start < last and first < stop


def output_overlaps_operand(arguments, output, written, reads, *, exact=False):
    """True when the output may receive bytes an operand is read from, or that cannot be decided.

    ``arguments`` is the launch's argument list, ``output`` the output's index and
    ``written`` the elements stored from its pointer. ``reads`` maps an operand's index to
    the elements read from its pointer; a listed operand is compared by the bytes addressed.
    With ``exact`` the counts are the whole contract and only the listed operands are
    compared. Otherwise each range also covers the view the caller passed, and a tensor
    argument that is not listed counts whenever it shares the output's storage, wherever
    its view lies: nothing bounds what the kernel addresses through it.

    An argument without tensor storage (a scalar, or a placeholder in a dispatcher unit
    test) cannot overlap. A tensor whose storage cannot be read is reported as overlapping.
    """
    target = arguments[output]
    if not hasattr(target, "data_ptr"):
        return False
    try:
        mine = _addressed(target, written if exact else max(written, _span(_base(target))))
        for index, value in enumerate(arguments):
            if index == output or not hasattr(value, "data_ptr"):
                continue
            if index in reads:
                elements = reads[index] if exact else max(reads[index], _span(_base(value)))
                if _meet(mine, _addressed(value, elements)):
                    return True
            elif not exact and _share(mine[0], _storage(value)):
                return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return True
    return False


def overlapping_output(arguments, outputs):
    """(output index, other index) of the first output not known to be apart from another tensor.

    An output is apart from another tensor argument only when the two have different
    storages. Views of one storage are reported even when their cells do not intersect:
    the kernel addresses each through its own stride and length arguments, so a view
    shorter than the rows addressed through it, or slot views given contiguous strides,
    reach cells the views do not cover. An argument whose storage cannot be read is
    reported with every output, because nothing shows them apart. None when every
    output is apart from every other tensor argument.
    """
    storages = {}
    for index, value in enumerate(arguments):
        if hasattr(value, "data_ptr"):
            try:
                storages[index] = _storage(value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                storages[index] = None
    for output in outputs:
        if output not in storages:
            continue
        mine = storages[output]
        for index, other in storages.items():
            if index != output and (mine is None or other is None or _share(mine, other)):
                return output, index
    return None
