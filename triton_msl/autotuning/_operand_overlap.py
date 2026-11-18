"""Does a replacement launch's output overlap an operand it reads?

A source program loads every operand before it stores, so a launch of one program whose
output shares memory with an operand is defined: the result is computed from the operand's
old contents. A replacement template spreads that program over several threadgroups, which
store finished tiles while others still load, so it does not keep that order. A dispatcher
therefore declines the replacement when the bytes the output receives intersect the bytes
an operand is read from. Views of one storage whose byte ranges do not intersect stay
eligible.

No Triton import: the dispatchers that use this stay importable without backend discovery.
"""

import math


def _span(value):
    """Elements from the first to the last one a view addresses."""
    span = 1
    for size, stride in zip(value.shape, value.stride()):
        if size == 0:
            return 0
        span += (size - 1) * abs(stride)
    return span


def _byte_range(value, elements):
    start = value.storage_offset() * value.element_size()
    return start, start + elements * value.element_size()


def output_overlaps_operand(arguments, output, written, reads, *, exact=False):
    """True when the output's bytes intersect an operand's, or when that cannot be decided.

    ``arguments`` is the launch's argument list, ``output`` the output's index and
    ``written`` the elements stored from its pointer. ``reads`` maps an operand's index to
    the elements read from its pointer. With ``exact`` those counts are the whole contract
    and only the listed operands are compared; otherwise every other tensor argument is
    compared, and each range covers at least the view the caller passed.

    An argument without tensor storage (a scalar, or a placeholder in a dispatcher unit
    test) cannot overlap. A tensor whose storage cannot be read is reported as overlapping.
    """
    target = arguments[output]
    if not hasattr(target, "data_ptr"):
        return False
    try:
        storage = target.untyped_storage().data_ptr()
        first, last = _byte_range(target, written if exact else max(written, _span(target)))
        for index, value in enumerate(arguments):
            if index == output or not hasattr(value, "data_ptr") or (exact and index not in reads):
                continue
            if value.untyped_storage().data_ptr() != storage:
                continue
            elements = reads.get(index, 0)
            start, stop = _byte_range(value, elements if exact else max(elements, _span(value)))
            if start < last and first < stop:
                return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _view(value):
    """(storage, first byte, one past the last byte, tensor) of the view an argument passes."""
    if not hasattr(value, "untyped_storage"):
        # triton.reinterpret wraps a tensor and addresses the same bytes.
        value = value.base
    start, stop = _byte_range(value, _span(value))
    return value.untyped_storage().data_ptr(), start, stop, value


_RESIDUE_CAP = 1 << 12


def _residues(value, modulus):
    """The elements a view addresses, modulo ``modulus``; None above the cap."""
    found = {value.storage_offset() % modulus}
    for size, stride in zip(value.shape, value.stride()):
        step = stride % modulus
        if size < 2 or step == 0:
            continue
        size = min(size, modulus // math.gcd(step, modulus))  # the residues repeat after that
        if len(found) * size > _RESIDUE_CAP:
            return None
        found = {(base + index * step) % modulus for base in found for index in range(size)}
    return found


def _never_share_a_cell(a, b):
    """True when two views of one storage whose ranges intersect are proved apart.

    The slots of a packed buffer (query, key, value and output as ``buf[..., i, :]``)
    and a column next to its matrix (``buf[..., :d]`` and ``buf[..., d]``) interleave
    without sharing a cell. Along a stride of either view the two then address
    different residues. Any modulus that separates them is a proof; False when none
    does or when the views are not the same element width.
    """
    if a.element_size() != b.element_size():
        return False
    for modulus in sorted({abs(s) for value in (a, b) for s in value.stride() if abs(s) > 1}):
        first = _residues(a, modulus)
        second = _residues(b, modulus) if first is not None else None
        if second is not None and first.isdisjoint(second):
            return True
    return False


def overlapping_output(arguments, outputs):
    """(output index, other index) of the first output that overlaps another tensor.

    Compares the views the caller passed within one storage: their byte ranges, and
    for ranges that intersect the cells the views address. None when no output
    overlaps. An argument whose storage cannot be read is skipped: this is a refusal
    for launches that are otherwise accepted, so what cannot be decided is left alone.
    A kernel that addresses rows its view does not cover (a view shorter than the
    stride and length arguments describe) is compared by the view all the same.
    """
    views = {}
    for index, value in enumerate(arguments):
        if hasattr(value, "data_ptr"):
            try:
                views[index] = _view(value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
    for output in outputs:
        if output not in views:
            continue
        storage, first, last, target = views[output]
        for index, (other, start, stop, value) in views.items():
            if index == output or other != storage or not (start < last and first < stop):
                continue
            if not _never_share_a_cell(target, value):
                return output, index
    return None
