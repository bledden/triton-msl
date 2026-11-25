"""Remember a launch check that passed, for the same inputs, so the next launch skips it.

The launcher proves several facts about a launch from its arguments before anything is
submitted (the lane order, the matmul grid storage proof, the fast matmul output
overlap). Each is a pure function of a few values: the grid, the scalar arguments as the
GPU reads them, and per tensor the storage behind it, its extent, the view's offset and
element width and the live data pointer. None reads tensor contents or the view's
shape. A launch of the same kernel with the same values has the same verdict, so a
verdict that passed is remembered under those values and the check is skipped on the
next launch that has them. A refusal is never remembered: it is computed again, and
raised, every time.

The key is read before the check, and the check reads the arguments again. A storage
rebound between the two reads (by another thread, or by code that runs inside a tensor
accessor) would leave a pass that belongs to the new binding stored under the old one,
which the check may refuse. So after a check passes the key is read once more and the
pass is remembered only when both readings are equal; otherwise the launch has taken
the check and nothing is kept. A launch that is already remembered pays for one reading
only. Two changes inside the one check that restore the first binding exactly are not
seen by this comparison.

Only a passing verdict is remembered, and only when every input of the check can be
read and is of a type the check was reviewed for (an exact ``torch.Tensor``, an exact
``int``, ``bool`` or ``float``). Anything else, or a failure reading a value, bypasses
the memo and the launch takes the check itself. The memo lives on one launcher, whose
packed descriptors ``validate_packed_launch`` keeps immutable, so the descriptor is not
part of the key; the entries are bounded and cleared when the bound is reached.

Environment:

``TRITON_MSL_LAUNCH_MEMO=0`` turns the memo off: every launch takes every check.
``TRITON_MSL_LAUNCH_MEMO_VERIFY=1`` is the test mode: on every hit the original check
runs at the same point and a disagreement raises ``LaunchMemoDisagreement``.
``TRITON_MSL_LAUNCH_MEMO_DIAG=1`` counts hits, misses, bypasses and evictions per kernel
and check (``diagnostics()``); off by default so that counting does not cost time.
All three are read when this module is imported.
"""

import os

ENABLED = os.environ.get("TRITON_MSL_LAUNCH_MEMO", "1") != "0"
VERIFY = os.environ.get("TRITON_MSL_LAUNCH_MEMO_VERIFY", "0") == "1"
DIAG = os.environ.get("TRITON_MSL_LAUNCH_MEMO_DIAG", "0") == "1"
LIMIT = 256  # remembered launches per launcher and check; reaching it clears them all

_SCALARS = (int, bool, float)
_TENSORS = None
_LANE_TENSORS = None
_DIAGNOSTICS = {}


class LaunchMemoDisagreement(RuntimeError):
    """Test mode: a remembered passing verdict and the check run again disagree."""


def _tensor_types():
    global _TENSORS
    if _TENSORS is None:
        import torch

        _TENSORS = (torch.Tensor,)
    return _TENSORS


def _lane_tensor_types():
    """Exact ``torch.Tensor``, and ``torch.nn.Parameter`` while it leaves the tensor methods alone.

    A compiled model passes its weights as Parameters to every kernel that reads them.
    Parameter is torch's own subclass with ``__torch_function__`` disabled, so the storage,
    offset, width and pointer the lane-order check reads of one are the plain tensor's.
    Any other subclass may answer those questions as it likes and takes the check.
    """
    global _LANE_TENSORS
    if _LANE_TENSORS is None:
        import torch

        found = [torch.Tensor]
        parameter = torch.nn.Parameter
        disabled = getattr(torch._C, "_disabled_torch_function_impl", None)
        if disabled is not None and getattr(parameter, "__torch_function__", None) is disabled:
            found.append(parameter)
        _LANE_TENSORS = tuple(found)
    return _LANE_TENSORS


def tensor_facts(value, types=None):
    """What a launch check reads of one tensor argument, or None when its type is not in ``types``.

    (device type, storage data pointer, storage bytes, view offset in elements, element
    width, live data pointer). Read at the point the check would read them. ``types`` is
    the exact ``torch.Tensor`` alone unless given.
    """
    if type(value) not in (_tensor_types() if types is None else types):
        return None
    storage = value.untyped_storage()
    return (
        value.device.type,
        storage.data_ptr(),
        storage.nbytes(),
        value.storage_offset(),
        value.element_size(),
        value.data_ptr(),
    )


def _grid(grid):
    if len(grid) != 3 or any(type(extent) is not int for extent in grid):
        return None
    return tuple(grid)


def matmul_proof_key(descriptor, arguments, signatures, payloads, grid):
    """The inputs of ``_matmul_tail_bounds_reason`` for this launch, or None to take the check.

    The descriptor is the launcher's immutable one. The proof reads the grid, the
    declared type and captured payload of each scalar it references (re-packing the live
    value, which for an exact built-in scalar reproduces the payload), and the storage
    facts of the three role tensors. Every scalar must be an exact built-in with a packed
    payload and every role tensor an exact MPS tensor.
    """
    if not ENABLED:
        return None
    try:
        indices = descriptor[2]
        extents = _grid(grid)
        if extents is None or type(indices) not in (tuple, list) or len(indices) != 3:
            return None
        facts = []
        for index in indices:
            if type(index) is not int or not 0 <= index < len(arguments):
                return None
            found = tensor_facts(arguments[index])
            if found is None or found[0] != "mps":
                return None
            facts.append(found)
        if len(payloads) != len(arguments) or len(signatures) != len(arguments):
            return None
        for value, payload in zip(arguments, payloads):
            if payload is None:
                continue
            if type(value) not in _SCALARS or type(payload) is not bytes:
                return None
        return (extents, tuple(signatures), tuple(payloads), facts[0], facts[1], facts[2])
    except Exception:
        return None


def overlap_key(arguments, scalar_indices):
    """The inputs of the fast matmul's exact output-overlap check, or None to take the check.

    The check compares the bytes M*K, K*N and M*N elements from the A, B and C pointers
    (``arguments[0:3]``, exact MPS tensors) with M, N and K read from the exact built-in
    scalars at ``scalar_indices``.
    """
    if not ENABLED or len(arguments) < 3:
        return None
    try:
        scalars = tuple(arguments[index] for index in scalar_indices)
        if any(type(value) not in (int, bool) for value in scalars):
            return None
        facts = tuple(tensor_facts(arguments[index]) for index in range(3))
    except Exception:
        return None
    if any(found is None or found[0] != "mps" for found in facts):
        return None
    return scalars + facts


def lane_order_key(arguments, grid):
    """Every value the lane-order check can read of this launch, or None to take the check.

    The check reads, of each pointer argument, the storage behind it (grouping the
    arguments that share one), the view's offset and element width and, on the host, the
    data pointer; of each scalar argument its integer value where an address, a loop bound
    or a branch uses it; and the grid. The key holds all of them, whichever the plan uses:
    every argument is an exact tensor (or a ``torch.nn.Parameter``, see
    ``_lane_tensor_types``) or an exact int, bool or float, else the check runs. A float
    is refused wherever the check reads it, so its value cannot turn a verdict into a
    proof; it is kept in the key all the same, apart from the integers.
    """
    if not ENABLED:
        return None
    extents = _grid(grid)
    if extents is None:
        return None
    parts = [extents]
    try:
        tensors = _lane_tensor_types()
        for value in arguments:
            kind = type(value)
            if kind is int or kind is bool:
                parts.append(value)
            elif kind is float:
                parts.append(("float", value))
            else:
                found = tensor_facts(value, tensors)
                if found is None:
                    return None
                parts.append(found)
    except Exception:
        return None
    return tuple(parts)


class LaunchMemo:
    """The passing verdicts of one check on one launcher."""

    __slots__ = ("kernel", "kind", "entries")

    def __init__(self, kernel, kind):
        self.kernel = kernel
        self.kind = kind
        self.entries = set()

    def recall(self, key, check, passing, read, inputs):
        """``check(*inputs)``'s result, or ``passing`` without running it when ``key`` is remembered.

        A key of None bypasses: the check runs and nothing is remembered. A result that
        is ``passing`` (identity: ``None`` or ``False``) is remembered under the key when
        ``read(*inputs)``, the key read again after the check, still equals it. A key that
        changed while the check ran is counted as a bypass: the result is returned and
        nothing is remembered. Any other result is returned and forgotten. A hit never
        calls ``read``. In test mode a hit runs the check too and a result other than
        ``passing`` raises.
        """
        if key is None:
            if DIAG:
                _count(self, "bypasses")
            return check(*inputs)
        if key in self.entries:
            if DIAG:
                _count(self, "hits")
            if VERIFY:
                found = check(*inputs)
                if found is not passing:
                    raise LaunchMemoDisagreement(
                        f"{self.kind} memo of kernel {self.kernel!r}: a remembered passing launch is now "
                        f"answered {found!r} by the check"
                    )
            return passing
        found = check(*inputs)
        if found is not passing:
            if DIAG:
                _count(self, "misses")
            return found
        try:
            unchanged = read(*inputs) == key
        except Exception:
            unchanged = False
        if not unchanged:
            if DIAG:
                _count(self, "bypasses")
            return found
        if DIAG:
            _count(self, "misses")
        if len(self.entries) >= LIMIT:
            self.entries.clear()
            if DIAG:
                _count(self, "evictions")
        self.entries.add(key)
        return found


def _count(memo, event):
    counts = _DIAGNOSTICS.setdefault(memo.kernel, {}).setdefault(memo.kind, {})
    counts[event] = counts.get(event, 0) + 1


def diagnostics():
    """{kernel name: {check: {"hits"|"misses"|"bypasses"|"evictions": count}}} since the last reset."""
    return {kernel: {kind: dict(counts) for kind, counts in kinds.items()} for kernel, kinds in _DIAGNOSTICS.items()}


def reset_diagnostics():
    _DIAGNOSTICS.clear()
