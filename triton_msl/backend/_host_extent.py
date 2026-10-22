"""Opt-in CPU storage borrowing; conformance assumptions are documented in USAGE.

The trace records preparation decisions, not successful GPU execution. It owns no
storage or Metal buffer and is local to the current execution context.
"""
from contextvars import ContextVar

_last_launch = ContextVar("triton_msl_host_extent", default=None)


def clear_launch():
    _last_launch.set(None)


def last_launch():
    trace = _last_launch.get()
    return None if trace is None else [dict(row) for row in trace]


def record(row, launch_metadata=None):
    trace = (*(_last_launch.get() or ()), dict(row))
    _last_launch.set(trace)
    if launch_metadata is not None:
        if isinstance(launch_metadata, dict):
            launch_metadata["host_extent"] = [dict(item) for item in trace]
        elif hasattr(launch_metadata, "add"):
            launch_metadata.add(lambda rows: {"host_extent": [dict(item) for item in rows]}, (trace,))


def select(storage, ptr, length, state, query, page=16384, *, owner=None):
    """Return (route, span, reason); a stale owner must REFUSE, never copy stale bytes."""
    if ptr > 0 and ptr % page == 0 and length >= page and length % page == 0:
        return "existing", length, "existing_exact_page_path"
    if state is False:
        return "mirror", length, "disabled"
    if ptr <= 0 or ptr % page or length < page:
        return "mirror", length, "geometry"
    rounded = ((length + page - 1) // page) * page
    if length > (1 << 64) - 1 or ptr > (1 << 64) - 1 - rounded:
        return "mirror", length, "overflow"
    if state is not True or query is None:
        return "mirror", length, "unavailable"
    def agrees():
        try:
            current = storage if owner is None else owner.untyped_storage()
            return (storage.data_ptr() == ptr and storage.nbytes() == length
                    and current.data_ptr() == ptr and current.nbytes() == length)
        except Exception:
            return False

    if not agrees():
        return "refuse", 0, "storage_changed_before_query"
    span, reason = query(ptr, length)
    if not agrees():
        return "refuse", 0, "storage_changed_during_query"
    if type(span) is int and span == rounded:
        return "borrow", span, "admitted"
    return "mirror", length, reason
