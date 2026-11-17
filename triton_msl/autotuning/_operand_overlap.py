"""Does a replacement launch's output overlap an operand it reads?

A source program loads every operand before it stores, so a launch of one program whose
output shares memory with an operand is defined: the result is computed from the operand's
old contents. A replacement template spreads that program over several threadgroups, which
store finished tiles while others still load, so it does not keep that order. A dispatcher
therefore declines the replacement when the bytes the output receives intersect the bytes
an operand is read from. Views of one storage that do not intersect stay eligible.

No Triton import: the dispatchers that use this stay importable without backend discovery.
"""


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


def _view_bytes(value):
    """(storage, first byte, one past the last byte) of the view an argument passes."""
    if not hasattr(value, "untyped_storage"):
        # triton.reinterpret wraps a tensor and addresses the same bytes.
        value = value.base
    start, stop = _byte_range(value, _span(value))
    return value.untyped_storage().data_ptr(), start, stop


def overlapping_output(arguments, outputs):
    """(output index, other index) of the first output that overlaps another tensor.

    Compares the byte range of each view within one storage; None when no output
    overlaps. An argument whose storage cannot be read is skipped: this is a refusal
    for launches that are otherwise accepted, so what cannot be decided is left alone.
    """
    views = {}
    for index, value in enumerate(arguments):
        if hasattr(value, "data_ptr"):
            try:
                views[index] = _view_bytes(value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
    for output in outputs:
        if output not in views:
            continue
        storage, first, last = views[output]
        for index, (other, start, stop) in views.items():
            if index != output and other == storage and start < last and first < stop:
                return output, index
    return None
