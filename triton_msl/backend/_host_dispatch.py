"""Own host launch backing storage across an uncertain submission failure.

Records are installed before entering the runtime. Only the matching command's
terminal status may retire a failed submission; no replay or delayed copy-back.
"""

from contextlib import contextmanager
from threading import Lock

from triton_msl.errors import PostSubmitError

_lock = Lock()
_retained = {}


class RetainedBuffers(list):
    """List-compatible bindings that carry the launch's ownership witness."""


class _Lease:
    def __init__(self, buffers, owners):
        self.buffers = RetainedBuffers(buffers)
        self.buffers.lease = self
        self.owners = owners
        self.command = None
        self.stage = "unbound"
        self.failed = False


def reap_completed():
    """Retire provably safe failed records; return the still-quarantined count.

    This never waits, replays, copies results or returns buffers to the pool.
    A missing witness, nonterminal state or failed status query retains ownership.
    """
    with _lock:
        candidates = [lease for lease in _retained.values() if lease.failed]
    for lease in candidates:
        safe = lease.stage == "encoding"  # Failure before any commit attempt.
        if lease.stage == "submission_attempted" and lease.command is not None:
            try:
                import Metal

                safe = lease.command.status() in (
                    Metal.MTLCommandBufferStatusCompleted,
                    Metal.MTLCommandBufferStatusError,
                )
            except Exception:
                safe = False
        if safe:
            with _lock:
                _retained.pop(id(lease), None)
    with _lock:
        return sum(lease.failed for lease in _retained.values())


def check_pending():
    """Fail before preparing another launch if earlier completion is unknown."""
    if reap_completed():
        raise PostSubmitError(
            "A previous host dispatch has unconfirmed completion; its buffers and owners "
            "remain retained. Refusing another host launch. Retry only after completion "
            "can be established, or restart the worker."
        )


@contextmanager
def retain(buffers, *, owners):
    lease = _Lease(buffers, owners)
    with _lock:
        # Recheck after preparation in case another thread failed in the meantime.
        if any(item.failed for item in _retained.values()):
            raise PostSubmitError("A concurrent host dispatch failed; refusing submission.")
        _retained[id(lease)] = lease
    returned = False
    try:
        yield lease.buffers
        returned = True  # The host runtime's normal return is synchronous completion.
    finally:
        with _lock:
            if returned:
                _retained.pop(id(lease), None)
            else:
                lease.failed = True
