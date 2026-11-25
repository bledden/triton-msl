"""Does a launch keep the source's order where lanes of one program meet at a memory cell?

A source program finishes each operation over every lane before it starts the next. The
ordinary lowering keeps each lane's own order and nothing more (see
``triton_msl.codegen._access_plan``). So a launch computes the source's result when, in
every program, no memory cell that is written is touched by two different lanes, or when
the shader text shows every read of that memory done before a barrier and every store
after it.

``LaunchCheck.verdicts`` evaluates a kernel's access plan with one launch's arguments. It
answers per group of arguments that share a storage. An answer is a proof or a reason
there is none; the absence of a proof is never reported as safety. Two programs of one
launch touching the same cell is a race in the source and is not looked at here.
"""

import numpy as np

from triton_msl.autotuning._operand_overlap import _base, _share, _storage

PID, LANE, LOOP, OPAQUE = 1, 2, 4, 8
BUDGET = 1 << 22  # addresses compared at a time
STORAGE_BUDGET = 1 << 18  # address instances one access is read at, for the storage proof
SLICES = 16  # and how many times that for one launch, a run of programs at a time

# Atomic updates that leave the same cell whatever order they land in. A floating-point add
# is not one: it commutes but does not associate, so the order of the terms is the rounding.
_COMMUTING = ("add", "and", "or", "xor", "max", "min", "umax", "umin")
SAFE = ("no-write", "one-store", "atomic-only", "one-lane", "text", "text+lanes", "lanes")
# The findings that count an access in although a mask or a branch may leave it out.
_HEDGED = {"lanes-meet": "may-meet", "lanes-meet-narrow": "may-meet", "order-meet": "may-meet"}

_CHILDREN = {
    "b": (3, 4),
    "x": (4,),
    "cmp": (3, 4),
    "sel": (1, 2, 3),
    "bc": (2,),
    "xd": (2,),
    "rs": (3,),
    "ptr+": (1, 2),
}
_M32 = (1 << 32) - 1


# What each answer that is not a proof means, for the refusal text.
_REASONS = {
    "storage": "an argument's storage could not be read, so it may share memory with any other",
    "unknown-base": "an access could not be traced to the argument it points into",
    "unbounded": (
        "an access sits in a loop whose trips are not known before the launch, and its lanes are not shown "
        "to keep to elements of their own"
    ),
    "untraced": "the shader does not perform these accesses one by one in the source's order",
    "opaque-address": "an address depends on a value the check cannot compute before the launch",
    "lanes-meet": "two lanes of one program reach the same element and one of them stores it",
    "lanes-meet-narrow": "an access narrower than the block reaches an element another lane also reaches",
    "order-meet": "accesses the shader performs in another order than the source reach the same element",
    "may-meet": (
        "two lanes of one program reach the same element, one of them storing it, unless a branch, or a mask "
        "computed from values the check cannot know before the launch, keeps them apart"
    ),
    "element-size": (
        "arguments of different element sizes reach the same bytes of the memory they share, or one of them "
        "is not aligned to its element size"
    ),
    "loop-bounds": "a loop around the accesses has bounds that are not fixed before the launch",
    "budget": "there are too many addresses to compare",
    "eval": "an address uses an operation or an argument the check does not evaluate",
}


def launch_plan(metadata):
    """The access plan a launcher may rely on, or None when the kernel was given none.

    The plan is built beside the MSL. A kernel compiled on the other binary route may
    run that binary instead, so what the plan says about the MSL text is dropped for it.
    """
    plan = getattr(metadata, "access_plan", None)
    if plan is not None and getattr(metadata, "binary_route", None) == "cpp":
        from triton_msl.codegen._access_plan import source_only

        plan = source_only(plan)
    return plan


def unproved(check, arguments, grid):
    """(reason, argument indexes) for the first group of this launch with no proof, else None.

    A failure inside the check is an answer too: it is never read as a proof.
    """
    try:
        found = check.verdicts(arguments, grid)
    except Exception as error:
        return f"the check itself failed ({type(error).__name__})", ()
    for answer, members in found:
        if answer in SAFE:
            continue
        if answer.startswith("plan:"):
            return "the kernel's accesses could not be described (" + answer[5:] + ")", tuple(members)
        return _REASONS.get(answer, answer), tuple(members)
    return None


def remembered_unproved(check, arguments, grid):
    """``unproved``, skipped where ``check.memo`` remembers a proof for these very arguments and grid.

    The key is every value the check reads (``_launch_memo.lane_order_key``: each
    argument's storage, extent, offset, width and pointer, or its exact scalar value, and
    the grid). An argument the key cannot hold, or a check without a memo, runs the check;
    only a proof is remembered, and only when the key read again after the check is the
    key read before it. In test mode a remembered launch runs the check too.
    """
    memo = getattr(check, "memo", None)
    if memo is None:
        return unproved(check, arguments, grid)
    from triton_msl.backend._launch_memo import lane_order_key

    return memo.recall(lane_order_key(arguments, grid), unproved, None, _key_again, (check, arguments, grid))


def _key_again(check, arguments, grid):
    """The memo's key for these arguments now (read after ``unproved`` ran, with its inputs)."""
    from triton_msl.backend._launch_memo import lane_order_key

    return lane_order_key(arguments, grid)


def outside_storage(check, arguments, grid):
    """What a launch's source addresses outside a storage, or None.

    ``(kind, argument index, lowest, highest, before, after)``: an access of ``kind``
    ("L", "S" or "A") through that pointer argument reaches elements ``lowest`` to
    ``highest`` from the pointer, where the storage holds ``before`` elements before the
    pointer's first one and ``after`` from it on. Only an access the check can compute
    exactly is reported; anything it cannot read is left to run as before.
    """
    try:
        return check.outside(arguments, grid)
    except Exception:
        return None


def remembered_outside_storage(memo, check, arguments, grid, hooked):
    """``outside_storage``, skipped where ``memo`` remembers these arguments and grid inside their storages.

    The key is the lane-order key: every argument's storage, extent, offset, width and
    pointer, or its exact scalar value, and the grid. A launch whose hook ran takes the
    check itself, as does one the key cannot hold.
    """
    if memo is None:
        return outside_storage(check, arguments, grid)
    from triton_msl.backend._launch_memo import lane_order_key

    key = None if hooked else lane_order_key(arguments, grid)
    return memo.recall(key, outside_storage, None, _key_again, (check, arguments, grid))


class Unproved(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class _NotSeparable(Exception):
    pass


def _wrap(value, width):
    """Two's complement at ``width`` bits; a one-bit value stays 0 or 1."""
    if width >= 64:
        return value
    if width == 1:
        return value & 1
    half = 1 << (width - 1)
    return ((value + half) & ((1 << width) - 1)) - half


def _unsigned(value, width):
    if width >= 64:
        return value.astype(np.uint64)
    return value & ((1 << width) - 1)


def _binary(name, width, left, right):
    if name in ("sdiv", "udiv", "srem", "urem", "fdiv", "cdiv") and not np.all(right):
        raise Unproved("eval", "division by zero")
    if name in ("sdiv", "srem", "fdiv", "cdiv") and 1 < width <= 64:
        # The smallest value over -1 is outside the type; the source does not say what it is.
        if np.any((np.asarray(right) == -1) & (np.asarray(left) == -(1 << (width - 1)))):
            raise Unproved("eval", "quotient outside the type")
    if name == "add":
        out = left + right
    elif name == "sub":
        out = left - right
    elif name == "mul":
        out = left * right
    elif name == "and":
        out = left & right
    elif name == "or":
        out = left | right
    elif name == "xor":
        out = left ^ right
    elif name in ("sdiv", "srem"):
        # No absolute value and no negation: the smallest 64-bit value has neither.  The floor
        # quotient is one below the truncated one where the signs differ and a remainder is left.
        floor = left // right
        quotient = floor + (((left < 0) != (right < 0)) & (floor * right != left))
        out = quotient if name == "sdiv" else left - quotient * right
    elif name == "fdiv":
        out = left // right
    elif name == "cdiv":
        floor = left // right
        out = floor + (floor * right != left)
    elif name in ("udiv", "urem", "umin", "umax", "lshr"):
        a, b = _unsigned(left, width), _unsigned(right, width)
        if name == "udiv":
            out = a // b
        elif name == "urem":
            out = a % b
        elif name == "umin":
            out = np.minimum(a, b)
        elif name == "umax":
            out = np.maximum(a, b)
        else:
            if np.any(right < 0) or np.any(right >= width):
                raise Unproved("eval", "shift amount")
            out = a >> b
        out = out.astype(np.int64)
    elif name == "smin":
        out = np.minimum(left, right)
    elif name == "smax":
        out = np.maximum(left, right)
    elif name in ("shl", "ashr"):
        if np.any(right < 0) or np.any(right >= width):
            raise Unproved("eval", "shift amount")
        out = left << right if name == "shl" else left >> right
    else:
        raise Unproved("eval", name)
    return _wrap(out, width)


def _compare(name, width, left, right):
    if name in ("ult", "ule", "ugt", "uge"):
        left, right = _unsigned(left, width), _unsigned(right, width)
    try:
        out = {
            "eq": np.equal,
            "ne": np.not_equal,
            "slt": np.less,
            "ult": np.less,
            "sle": np.less_equal,
            "ule": np.less_equal,
            "sgt": np.greater,
            "ugt": np.greater,
            "sge": np.greater_equal,
            "uge": np.greater_equal,
        }[name](left, right)
    except KeyError:
        raise Unproved("eval", name) from None
    return out.astype(np.int64)


class _Env:
    def __init__(self, lead, programs, pid, trips, axis, origins, args, grid):
        self.lead = lead
        self.programs = programs
        self.pid = pid
        self.trips = trips
        self.axis = axis
        self.origins = origins
        self.args = args
        self.grid = grid
        self.memo = {}
        self.spaces = False  # True: pointers are read as which allocation they are in
        self.picks = {}  # loop: the trips to read instead of all of them (the storage proof)


class LaunchCheck:
    """One kernel's access plan, prepared for evaluation."""

    def __init__(self, plan):
        self.nodes = plan["nodes"]
        self.loops = plan["loops"]
        self.accesses = plan["accesses"]
        self.block = int(plan["block"])
        self.why = plan.get("why")
        self.pointers = {int(index): entry for index, entry in plan["pointers"].items()}
        self.barriers = plan.get("barriers") or []
        self.text = {int(index): entry for index, entry in (plan.get("text") or {}).items()}
        self.flags, self.follows = [], []
        for node in self.nodes:
            kind = node[0]
            flag, follows = 0, frozenset()
            if kind == "pid":
                flag = PID
            elif kind == "r":
                flag = LANE
            elif kind == "it":
                flag, follows = LOOP, frozenset((node[1],))
            elif kind == "trip":
                flag = OPAQUE if any(self.flags[bound] for bound in self.loops[node[1]][:3]) else 0
            elif kind in ("o", "pu"):
                flag = OPAQUE
            for position in _CHILDREN.get(kind, ()):
                flag |= self.flags[node[position]]
                follows |= self.follows[node[position]]
            self.flags.append(flag)
            self.follows.append(follows)
        self.bases = [self._bases(access[1]) for access in self.accesses]
        self.by_pointer = {index: [] for index in self.pointers}
        self.unknown_base = []
        self.joined = []  # buffers one access can point into: one group, whatever they share
        for ordinal, bases in enumerate(self.bases):
            if not bases:
                self.unknown_base.append(ordinal)
            elif len(bases) > 1:
                self.joined.append(sorted(bases))
            for index in bases or ():
                self.by_pointer.setdefault(index, []).append(ordinal)
        self.untraced = set(plan["untraced"]) if "untraced" in plan else set(range(len(self.accesses)))
        # A buffer the shader names outside its recorded memory operations.
        loose = set(plan["loose"]) if "loose" in plan else set(self.pointers)
        for ordinal, bases in enumerate(self.bases):
            if bases is None or bases & loose:
                self.untraced.add(ordinal)
        self.order = [tuple(pair) for pair in plan.get("order") or ()]
        # Accesses only one lane of a program performs, and whether there is one lane at all.
        self.single = set(plan.get("single") or ()) | set(plan.get("quiet") or ())
        # Accesses of one element per row that the row's first lane performs alone.
        self.rows = set(plan.get("rows") or ()) - self.single
        self.one_lane = plan.get("threads") == [1, False] and self.block == 1
        # Accesses under a branch: the comparison counts them in whichever way the branch goes.
        self.branched = set(plan.get("branched") or ())
        # The branches around each of them, [condition, the way that reaches the access]. A
        # condition that is one number before the launch is read: the launch performs the
        # access or it does not, and the comparison no longer has to count it in both ways.
        self.guards = {
            int(ordinal): [(int(node), bool(way)) for node, way in guards]
            for ordinal, guards in plan.get("guards") or ()
        }
        self.atomic = {int(ordinal): entry for ordinal, entry in (plan.get("atomic") or {}).items()}
        self.pairs = {tuple(pair) for pair in plan.get("pairs") or ()}
        # A source program of one lane is sequential. The shader is too when one thread
        # runs it, or when every thread runs it, they share nothing but the buffers, and
        # only thread 0 writes: what the other threads read reaches no memory.
        threads = plan.get("threads")
        alone = isinstance(threads, list) and len(threads) == 2 and not threads[1] and threads[0] == 1
        guarded = set(plan.get("single") or ())
        only_first = bool(plan.get("solo")) and all(
            ordinal in guarded for ordinal, access in enumerate(self.accesses) if access[0] != "L"
        )
        self.sequential = (
            self.block == 1
            and isinstance(threads, list)
            and len(threads) == 2
            and not threads[1]
            and (alone or only_first)
        )
        self.cache = {}
        # The elements each access reaches, per grid and scalar arguments (``outside``).
        self.spans = {}
        self.scalars = None
        # Proofs remembered for whole launches (a _launch_memo.LaunchMemo the launcher
        # sets), this check's alone: a replaced check starts with none.
        self.memo = None

    def _bases(self, node):
        entry = self.nodes[node]
        kind = entry[0]
        if kind == "p":
            return frozenset((entry[1],))
        if kind == "pu":
            return frozenset(entry[1])
        if kind == "ptr+":
            return self._bases(entry[1])
        if kind in ("bc", "xd"):
            return self._bases(entry[2])
        if kind == "rs":
            return self._bases(entry[3])
        if kind == "sel":
            one, other = self._bases(entry[2]), self._bases(entry[3])
            return None if one is None or other is None else one | other
        return None

    # -- one launch ----------------------------------------------------------------------

    def verdicts(self, arguments, grid):
        """[(answer, argument indexes)] for each storage group that is written."""
        if self.why:
            return [("plan:" + self.why, ())]
        storages, unread = {}, []
        for index in self.pointers:
            value = arguments[index] if index < len(arguments) else None
            try:
                storages[index] = _storage(value)
            except Exception:
                unread.append(index)
        groups = []
        for index, storage in storages.items():
            joined = [group for group in groups if any(_share(storage, storages[other]) for other in group)]
            merged = [index]
            for group in joined:
                merged.extend(group)
                groups.remove(group)
            groups.append(merged)
        for together in self.joined:
            joined = [group for group in groups if any(index in group for index in together)]
            if len(joined) > 1:
                for group in joined:
                    groups.remove(group)
                groups.append([index for group in joined for index in group])
        # An argument whose storage cannot be read may share memory with anything.
        if unread:
            groups = [sorted(unread + [index for group in groups for index in group])]
        left_out, reached = self._branches(arguments, grid)
        out = []
        for members in groups:
            members = sorted(members)
            ordinals = sorted(
                ({ordinal for index in members for ordinal in self.by_pointer.get(index, ())} | set(self.unknown_base))
                - left_out
            )
            answer = self._group(members, ordinals, bool(unread), storages, arguments, grid, reached)
            out.append((answer, tuple(members)))
        return out

    # -- storage -------------------------------------------------------------------------

    def outside(self, arguments, grid):
        """The first access of this launch's source that leaves its argument's storage, or None.

        Every program of a launch performs every load, store and atomic of the source,
        on the lanes its mask keeps and once per trip of the loops around it. An access
        is looked at when all of that is one computation before the launch: one pointer
        argument, an address and a mask that follow nothing but the grid, the lane, the
        loop trips and the integer arguments, counted loops, and no branch that may leave
        it out. Its lowest and highest element are then exact, and the launch is outside
        when either lies outside the storage behind the pointer. Any other access is not
        judged here. A launch of more address instances than ``STORAGE_BUDGET`` is read
        at the first two and last two programs of each grid axis only, and past that at
        the first two and last two trips of each loop: what is found there is still an
        access of the source, what lies between is not looked at.

        The elements reached do not depend on the storages, so they are kept per grid and
        scalar arguments: a launch with other tensors compares its storages and no more.
        """
        if self.why:
            return None
        key = (tuple(int(extent) for extent in grid), tuple(self._plain(arguments, index) for index in self._scalars()))
        spans = self.spans.get(key)
        if spans is None:
            spans = self._spans(arguments, grid)
            if len(self.spans) > 256:
                self.spans.clear()
            self.spans[key] = spans
        for kind, index, low, high in spans:
            try:
                room = self._room(arguments[index], self.pointers[index][1])
            except Exception:
                continue
            if room is not None and (low < -room[0] or high >= room[1]):
                return (kind, index, low, high, room[0], room[1])
        return None

    def _scalars(self):
        """The scalar arguments any address, mask, loop bound or branch of the plan reads."""
        if self.scalars is None:
            self.scalars = sorted({node[1] for node in self.nodes if node[0] == "a"})
        return self.scalars

    @staticmethod
    def _plain(arguments, index):
        value = arguments[index] if index < len(arguments) else None
        if type(value) in (int, bool):
            return value
        # Anything else is read by the evaluation itself, which refuses what it cannot
        # read; the key only has to tell two launches apart.
        return (type(value).__name__, repr(value))

    def _spans(self, arguments, grid):
        """[(kind, pointer argument, lowest, highest)] for every access this launch performs and the check can read."""
        left_out, reached = self._branches(arguments, grid)
        found = []
        for ordinal, access in enumerate(self.accesses):
            if ordinal in left_out or (ordinal in self.branched and ordinal not in reached):
                continue
            bases = self.bases[ordinal]
            if access[5] or not bases or len(bases) != 1:
                continue
            (index,) = bases
            try:
                span = self._span(ordinal, index, arguments, grid)
            except Exception:
                continue
            if span is not None:
                found.append((access[0], index, span[0], span[1]))
        return found

    @staticmethod
    def _room(value, width):
        """(elements before the pointer's first one, elements from it on) in its storage, or None."""
        value = _base(value)
        size = int(value.element_size())
        storage = value.untyped_storage()
        before = int(value.storage_offset())
        extent = int(storage.nbytes())
        if size <= 0 or size != width or before < 0:
            return None
        if extent and int(value.data_ptr()) != int(storage.data_ptr()) + before * size:
            return None
        return before, extent // size - before

    def _span(self, ordinal, index, arguments, grid):
        """(lowest, highest) element this access reaches from its pointer over the launch, or None for no element."""
        _, pointer, mask, shape, inside, _ = self.accesses[ordinal]
        shape = tuple(shape)
        size = 1
        for extent in shape:
            size *= extent
        if self.flags[pointer] & OPAQUE or self.follows[pointer] - set(inside):
            raise Unproved("opaque-address")
        if mask >= 0 and (self.flags[mask] & OPAQUE or self.follows[mask] - set(inside)):
            raise Unproved("eval", "a mask the check cannot compute")
        trips, axis = {}, {}
        for loop in sorted(inside):
            lower, upper, step = (self._scalar(bound, arguments, grid) for bound in self.loops[loop][:3])
            if step <= 0:
                raise Unproved("loop-bounds")
            trips[loop] = max(0, -((lower - upper) // step))
            axis[loop] = 1 + len(axis)
        if any(count == 0 for count in trips.values()):
            return None
        follows = self.follows[pointer] | (self.follows[mask] if mask >= 0 else frozenset())
        each = size
        for loop in follows:
            each *= trips[loop]
        extents = [int(extent) for extent in grid]
        programs = extents[0] * extents[1] * extents[2]
        if programs <= 0 or each <= 0:
            return None

        def ends(extent):
            return sorted({at for at in (0, 1, extent - 2, extent - 1) if 0 <= at < extent})

        picks = {}
        if not (self.flags[pointer] | (self.flags[mask] if mask >= 0 else 0)) & PID:
            # Every program reaches the same elements.
            order = np.zeros(1, dtype=np.int64)
        elif programs * each <= STORAGE_BUDGET:
            order = np.arange(programs, dtype=np.int64)
        else:
            corners = [ends(extent) for extent in extents]
            order = np.array(
                [x + extents[0] * (y + extents[1] * z) for z in corners[2] for y in corners[1] for x in corners[0]],
                dtype=np.int64,
            )
        if order.size * each > STORAGE_BUDGET:
            # The first two and last two trips of each loop the access follows.
            each = size
            for loop in follows:
                picks[loop] = np.array(ends(trips[loop]), dtype=np.int64)
                each *= picks[loop].size
        if order.size * each > STORAGE_BUDGET:
            raise Unproved("budget")
        lead = 1 + len(trips)
        count = int(order.size)
        order = order.reshape((count,) + (1,) * (lead - 1))
        pid = [order % extents[0], (order // extents[0]) % extents[1], order // (extents[0] * extents[1])]
        env = _Env(lead, count, pid, dict(trips), axis, {index: 0}, arguments, grid)
        env.picks = picks
        with np.errstate(over="ignore"):
            value, rank = self._eval(pointer, env)
            if rank == 0:
                value = value.reshape(value.shape + (1,) * len(shape))
            elif rank != len(shape):
                raise Unproved("eval", "rank")
            wide = [1] * lead
            wide[0] = count
            for loop in follows:
                wide[axis[loop]] = picks[loop].size if loop in picks else trips[loop]
            full = tuple(wide) + shape
            if mask >= 0:
                keep, mask_rank = self._eval(mask, env)
                if mask_rank == 0:
                    keep = keep.reshape(keep.shape + (1,) * len(shape))
                elif mask_rank != len(shape):
                    raise Unproved("eval", "rank")
                value = np.broadcast_to(value, full)[np.broadcast_to(keep, full) != 0]
        if not value.size:
            return None
        return int(value.min()), int(value.max())

    def _branches(self, arguments, grid):
        """(the accesses this launch does not perform, those it is known to perform), of the ones under a branch.

        A branch condition that follows nothing but the launch's scalar arguments is one
        number for every program, lane and trip, so the source performs what is under the
        other way of it nowhere in this launch. Any other condition stays unread, and the
        accesses under it are counted in as before.
        """
        known, left_out, reached = {}, set(), set()
        for ordinal, guards in self.guards.items():
            ways = []
            for node, way in guards:
                if node not in known:
                    try:
                        known[node] = bool(self._scalar(node, arguments, grid))
                    except Exception:
                        known[node] = None
                ways.append(None if known[node] is None else known[node] == way)
            if False in ways:
                left_out.add(ordinal)
            elif None not in ways:
                reached.add(ordinal)
        return left_out, reached

    def _group(self, members, ordinals, unread, storages, arguments, grid, reached=frozenset()):
        accesses = [self.accesses[ordinal] for ordinal in ordinals]
        writes = [ordinal for ordinal in ordinals if self.accesses[ordinal][0] != "L"]
        if not writes:
            return "no-write"
        if len(ordinals) == 1 and not accesses[0][4] and not accesses[0][5]:
            return "one-store"
        # The lanes of one atomic operation are not ordered by the source. Its trips are,
        # in a counted loop and in a ``while`` loop alike: inside a loop only an update that
        # commutes with itself is free of that order.
        if len(ordinals) == 1 and accesses[0][0] == "A" and not accesses[0][4] and not accesses[0][5]:
            return "atomic-only"
        if self._commuting(ordinals):
            return "atomic-only"
        # One thread does everything that reaches memory, in the order it is written.
        if (
            self.sequential
            and not self._swapped(ordinals)
            and not any(ordinal in self.untraced for ordinal in ordinals)
        ):
            return "one-lane"
        if unread:
            return "storage"
        if any(ordinal in self.unknown_base for ordinal in ordinals):
            return "unknown-base"
        loads_first = self._text_proof(members, 0, 1) and self._loads_first(ordinals, writes)
        proved = "lanes"
        # The other way round: every store has landed before any thread reads.
        if loads_first or (self._text_proof(members, 2, 3) and self._stores_first(ordinals, writes)):
            if len(writes) == 1 and not self.accesses[writes[0]][4] and not self.accesses[writes[0]][5]:
                return "text"
            # A store in a ``while`` loop comes round again after the loads of the loop's
            # next trip, whatever the text says of one trip: the loads stay in the comparison.
            if not any(self.accesses[ordinal][5] for ordinal in writes):
                ordinals, proved = writes, "text+lanes"
        # A ``while`` loop's trips are not known before the launch. What the comparison
        # reads of an access in one is the same on every trip (an address that changes
        # from trip to trip is one the check cannot compute, and a mask that does is
        # left out, which only adds cells), so "apart" holds for any number of trips:
        # each cell is one lane's alone, and a lane keeps the source's order. Any other
        # answer there is not a finding, since a cell may be met on some trips only.
        unbounded = any(self.accesses[ordinal][5] for ordinal in ordinals)
        # Lane order is the source's only for accesses written one by one, in order.
        if any(ordinal in self.untraced for ordinal in ordinals):
            return "unbounded" if unbounded else "untraced"
        swapped = self._swapped(ordinals)
        try:
            origins, spaces, scales = self._origins(members, storages, arguments)
            used = self._used_arguments(ordinals)
            key = (
                tuple(ordinals),
                tuple(sorted(origins.items())),
                tuple(self._argument(arguments, index) for index in used),
            )
            key += (tuple(sorted(spaces.items())),)
            key += (tuple(int(extent) for extent in grid),)
            if len(self.cache) > 512:
                self.cache.clear()
            answer = self.cache.get(key)
            if answer is None:
                answer = self._lanes(ordinals, origins, arguments, grid, None, swapped, spaces, scales)
                if answer == "per-program":
                    answer = self._per_program(ordinals, origins, arguments, grid, swapped, spaces, scales)
                self.cache[key] = answer
        except Unproved as unproved:
            return "unbounded" if unbounded else unproved.reason
        if answer == "apart":
            return proved
        if unbounded:
            return "unbounded"
        if answer in _HEDGED and any(self._unread(ordinal, reached) for ordinal in ordinals):
            return _HEDGED[answer]
        return answer

    def _unread(self, ordinal, reached=frozenset()):
        """Does the comparison count this access in although a mask or a branch may leave it out?"""
        mask = self.accesses[ordinal][2]
        branch = ordinal in self.branched and ordinal not in reached
        return branch or (mask >= 0 and bool(self.flags[mask] & OPAQUE))

    def _per_program(self, ordinals, origins, arguments, grid, swapped, spaces, scales):
        """The lanes of every program compared, a run of programs at a time.

        A cell is compared within its own program only, so the grid can be cut anywhere.
        """
        count = int(grid[0]) * int(grid[1]) * int(grid[2])
        first, step, done = 0, count, 0
        while first < count:
            try:
                answer = self._lanes(
                    ordinals, origins, arguments, grid, (first, min(count, first + step)), swapped, spaces, scales
                )
            except Unproved as unproved:
                if unproved.reason != "budget" or step == 1:
                    raise
                step = (step + 1) // 2
                continue
            if answer != "apart":
                return answer
            first += step
            done += 1
            if done + -(-(count - first) // step) > SLICES:
                raise Unproved("budget")
        return "apart"

    def _swapped(self, ordinals):
        """The accesses among these that the shader writes in another order than the source, a store among them."""
        among = set(ordinals)
        return frozenset(
            ordinal
            for pair in self.order
            if pair[0] in among
            and pair[1] in among
            and (self.accesses[pair[0]][0] != "L" or self.accesses[pair[1]][0] != "L")
            for ordinal in pair
        )

    def _commuting(self, ordinals):
        """Are these all atomic updates that leave the same cells whatever order they land in?

        One floating-point minimum or maximum (its two updates), or one associative and
        commutative operation repeated with no returned value read: a returned value
        would say which updates had landed before it.
        """
        if tuple(ordinals) in self.pairs:
            return True
        found = [self.atomic.get(ordinal) for ordinal in ordinals]
        if any(entry is None for entry in found):
            return False
        return (
            len({entry[0] for entry in found}) == 1
            and found[0][0] in _COMMUTING
            and not any(entry[1] for entry in found)
        )

    def _loads_first(self, ordinals, writes):
        """Does the source load everything of this memory before it stores any of it?"""
        loads = [ordinal for ordinal in ordinals if self.accesses[ordinal][0] == "L"]
        if any(self.accesses[ordinal][0] == "A" for ordinal in writes):
            return False
        if loads and max(loads) > min(writes):
            return False
        looped = {loop for ordinal in writes for loop in self.accesses[ordinal][4]}
        return not any(loop in looped for ordinal in loads for loop in self.accesses[ordinal][4])

    def _stores_first(self, ordinals, writes):
        """Does the source store everything of this memory before it loads any of it?"""
        loads = [ordinal for ordinal in ordinals if self.accesses[ordinal][0] == "L"]
        if any(self.accesses[ordinal][0] == "A" for ordinal in writes):
            return False
        if loads and max(writes) > min(loads):
            return False
        looped = {loop for ordinal in writes for loop in self.accesses[ordinal][4]}
        return not any(loop in looped for ordinal in loads for loop in self.accesses[ordinal][4])

    def _text_proof(self, members, above, below):
        """Is everything of one kind above one barrier and everything of the other below it?

        ``above`` and ``below`` index a buffer's text positions: 0 and 1 for every read
        above and every store below, 2 and 3 for every store above and every read below.
        """
        for position, end in self.barriers:
            for index in members:
                entry = self.text.get(index)
                if entry is None or len(entry) < 4 or not entry[above] < position:
                    break
                if not entry[below] > end:
                    break
            else:
                return True
        return False

    def _origins(self, members, storages, arguments):
        """Each argument's first element, counted in its own elements from the group's lowest pointer.

        With them, when the arguments differ in element size, how many of the smallest
        size one element of each argument is: the comparison then counts in that size.
        """
        sizes = {index: self.pointers[index][1] for index in members}
        unit, widest = min(sizes.values()), max(sizes.values())
        if not unit or any(size % unit or widest % size for size in sizes.values()):
            raise Unproved("element-size")
        mixed = unit != widest
        # Host addresses are one space. A device allocation is a space of its own: its
        # arguments are counted from its lowest pointer and it is told apart by number.
        first, space = {}, {}
        for index in members:
            value = _base(arguments[index])
            if storages[index][0] == "cpu":
                first[index], space[index] = value.data_ptr(), ("cpu",)
            else:
                first[index], space[index] = value.storage_offset() * value.element_size(), storages[index][:2]
        number = {one: position for position, one in enumerate(sorted(set(space.values()), key=repr))}
        origins = {}
        for index, byte in first.items():
            low = min(other for at, other in first.items() if space[at] == space[index])
            if mixed:
                # From a byte every size is aligned at, so that the lowest pointer being
                # a narrow view at an odd byte does not put the wider views out of step.
                low -= low % widest
            if (byte - low) % sizes[index]:
                raise Unproved("element-size")
            origins[index] = (byte - low) // sizes[index]
        scales = {index: size // unit for index, size in sizes.items()} if mixed else {}
        return origins, ({index: number[space[index]] for index in members} if len(number) > 1 else {}), scales

    def _reach(self, node, seen):
        if node in seen:
            return
        seen.add(node)
        entry = self.nodes[node]
        if entry[0] in ("it", "trip"):
            for bound in self.loops[entry[1]][:3]:
                self._reach(bound, seen)
        for position in _CHILDREN.get(entry[0], ()):
            self._reach(entry[position], seen)

    def _used_arguments(self, ordinals):
        seen = set()
        for ordinal in ordinals:
            access = self.accesses[ordinal]
            self._reach(access[1], seen)
            if access[2] >= 0:
                self._reach(access[2], seen)
            for loop in access[4]:
                for bound in self.loops[loop][:3]:
                    self._reach(bound, seen)
        return sorted({self.nodes[node][1] for node in seen if self.nodes[node][0] == "a"})

    @staticmethod
    def _argument(arguments, index):
        value = arguments[index]
        if isinstance(value, (bool, int, np.integer)):
            return int(value)
        raise Unproved("eval", "a scalar argument that is not an integer")

    # -- lanes ---------------------------------------------------------------------------

    def _scalar(self, node, arguments, grid):
        if self.flags[node]:
            raise Unproved("loop-bounds")
        value, _ = self._eval(node, _Env(0, 1, None, {}, {}, {}, arguments, grid))
        value = np.asarray(value).reshape(-1)
        if value.size == 0 or (value != value[0]).any():
            raise Unproved("eval", "a factor that is not one number")
        return int(value[0])

    def _pid_part(self, node, arguments, grid):
        """The part of a value that follows the program id: {node: coefficient mod 2**32}.

        Exact modulo 2**32: sums and products at 32 bits or more, pointer additions and
        extensions from 32 bits keep a value's residue. A part that also follows the lane
        or the loop trip cannot be separated.
        """
        if not self.flags[node] & PID:
            return {}
        entry = self.nodes[node]
        kind = entry[0]

        def merged(one, other, sign):
            out = dict(one)
            for atom, coefficient in other.items():
                out[atom] = (out.get(atom, 0) + sign * coefficient) & _M32
            return {atom: coefficient for atom, coefficient in out.items() if coefficient}

        def scaled(part, factor):
            out = {atom: (coefficient * factor) & _M32 for atom, coefficient in part.items()}
            return {atom: coefficient for atom, coefficient in out.items() if coefficient}

        part = self._pid_part
        if kind == "ptr+":
            return merged(part(entry[1], arguments, grid), part(entry[2], arguments, grid), 1)
        if kind == "b" and entry[2] >= 32:
            if entry[1] in ("add", "sub"):
                sign = 1 if entry[1] == "add" else -1
                return merged(part(entry[3], arguments, grid), part(entry[4], arguments, grid), sign)
            if entry[1] == "mul":
                for fixed, other in ((entry[3], entry[4]), (entry[4], entry[3])):
                    if not self.flags[fixed]:
                        return scaled(part(other, arguments, grid), self._scalar(fixed, arguments, grid))
            if entry[1] == "shl" and not self.flags[entry[4]]:
                amount = self._scalar(entry[4], arguments, grid)
                if 0 <= amount < 32:
                    return scaled(part(entry[3], arguments, grid), 1 << amount)
        if kind == "x" and (entry[2] >= 32 if entry[1] in ("sext", "zext") else entry[3] >= 32):
            return part(entry[4], arguments, grid)
        if kind in ("bc", "xd"):
            return part(entry[2], arguments, grid)
        if kind == "rs":
            return part(entry[3], arguments, grid)
        if self.flags[node] & (LANE | LOOP | OPAQUE):
            raise _NotSeparable
        return {node: 1}

    def _scale(self, ordinal, scales):
        """How many units of the comparison one element of this access is."""
        if not scales:
            return 1
        found = {scales.get(index) for index in self.bases[ordinal] or (None,)}
        if len(found) != 1 or None in found:
            raise Unproved("element-size")
        return found.pop()

    def _lanes(self, ordinals, origins, arguments, grid, programs, swapped=frozenset(), spaces=None, scales=None):
        """ "apart", "per-program" (ask again with the grid), or the kind of meeting found.

        ``scales`` is given when the arguments differ in element size. Each element is
        then every unit of the smallest size it covers, and a unit that elements of two
        sizes reach, one of them storing it, is not compared by lane: it is refused.
        """
        separable = programs is None
        if separable:
            try:
                parts = [
                    {
                        atom: coefficient * self._scale(ordinal, scales) & _M32
                        for atom, coefficient in self._pid_part(self.accesses[ordinal][1], arguments, grid).items()
                    }
                    for ordinal in ordinals
                ]
                parts = [{atom: coefficient for atom, coefficient in part.items() if coefficient} for part in parts]
            except _NotSeparable:
                return "per-program"
            if any(part != parts[0] for part in parts):
                return "per-program"
            count = 1
        else:
            count = programs[1] - programs[0]
        loops = sorted({loop for ordinal in ordinals for loop in self.accesses[ordinal][4]})
        trips, axis = {}, {}
        for loop in loops:
            lower, upper, step = (self._scalar(bound, arguments, grid) for bound in self.loops[loop][:3])
            if step <= 0:
                raise Unproved("loop-bounds")
            trips[loop] = max(0, -((lower - upper) // step))
            axis[loop] = 1 + len(axis)
        lead = 1 + len(loops)
        if separable:
            pid = [np.zeros((1,) * lead, dtype=np.int64)] * 3
        else:
            index = np.arange(programs[0], programs[1], dtype=np.int64).reshape((count,) + (1,) * (lead - 1))
            pid = [index % grid[0], (index // grid[0]) % grid[1], index // (grid[0] * grid[1])]
        env = _Env(lead, count, pid, trips, axis, origins, arguments, grid)
        # Which allocation each address is in, when the group has several: the same
        # expression with every pointer argument read as its allocation's number and
        # every offset as nothing.
        where_env = None
        if spaces:
            where_env = _Env(lead, count, pid, trips, axis, spaces, arguments, grid)
            where_env.spaces = True
        shapes = {}
        cells, lanes, written, wheres, widths = [], [], [], [], []
        total = 0
        unmasked = False  # a mask left out because it differs from program to program
        for ordinal in ordinals:
            kind, pointer, mask, shape, inside, _ = self.accesses[ordinal]
            shape = tuple(shape)
            size = 1
            for extent in shape:
                size *= extent
            if self.flags[pointer] & OPAQUE:
                raise Unproved("opaque-address")
            if self.follows[pointer] - set(inside):
                raise Unproved("eval", "an address that follows a loop it is not in")
            use_mask = mask >= 0 and not self.flags[mask] & OPAQUE and not (separable and self.flags[mask] & PID)
            use_mask = use_mask and not self.follows[mask] - set(inside)
            unmasked = unmasked or (separable and mask >= 0 and not use_mask and not self.flags[mask] & OPAQUE)
            follows = self.follows[pointer] | (self.follows[mask] if use_mask else frozenset())
            scale = self._scale(ordinal, scales)
            instances = size * count * scale
            for loop in follows:
                instances *= trips[loop]
            total += instances
            if total > BUDGET:
                raise Unproved("budget")
            with np.errstate(over="ignore"):
                value, rank = self._eval(pointer, env)
                if rank == 0:
                    value = value.reshape(value.shape + (1,) * len(shape))
                elif rank != len(shape):
                    raise Unproved("eval", "rank")
                wide = [1] * lead
                wide[0] = 1 if separable else count
                for loop in follows:
                    wide[axis[loop]] = trips[loop]
                full = tuple(wide) + shape
                value = np.broadcast_to(value, full)
                where = np.zeros((1,) * len(full), dtype=np.int64)
                if where_env is not None:
                    where, where_rank = self._eval(pointer, where_env)
                    if where_rank == 0:
                        where = where.reshape(where.shape + (1,) * len(shape))
                where = np.broadcast_to(where, full)
                if ordinal in swapped:
                    # No claim about which lane performs an element, only that two
                    # elements are two lanes.
                    lane = -(1 << 60) - (ordinal << 24) - np.arange(size, dtype=np.int64).reshape(shape)
                elif self.one_lane or ordinal in self.single:
                    # No claim about which lane of a wider access that one lane is.
                    lane = np.full(shape, -(1 << 59), dtype=np.int64)
                elif ordinal in self.rows:
                    # Row by row the same lane, and no claim about which lane of a
                    # wider access it is.
                    lane = -(1 << 58) + np.arange(size, dtype=np.int64).reshape(shape)
                elif size == self.block:
                    lane = np.arange(size, dtype=np.int64).reshape(shape) + (
                        shapes.setdefault(shape, len(shapes)) << 40
                    )
                else:
                    # One lane for each element, and no claim about which: two elements
                    # that meet in a cell, in one trip or in two, are two lanes.
                    lane = -1 - (ordinal << 24) - np.arange(size, dtype=np.int64).reshape(shape)
                lane = np.broadcast_to(lane, full)
                program = np.broadcast_to(
                    np.arange(full[0], dtype=np.int64).reshape((full[0],) + (1,) * (len(full) - 1)), full
                )
                if use_mask:
                    keep, mask_rank = self._eval(mask, env)
                    if mask_rank == 0:
                        keep = keep.reshape(keep.shape + (1,) * len(shape))
                    elif mask_rank != len(shape):
                        raise Unproved("eval", "rank")
                    keep = np.broadcast_to(keep, full) != 0
                    value, lane, program, where = value[keep], lane[keep], program[keep], where[keep]
            value, program, where, lane = (one.reshape(-1) for one in (value, program, where, lane))
            if scales:
                # Every unit of the element, in the unit the group is counted in.
                value = (value[:, None] * scale + np.arange(scale, dtype=np.int64)).reshape(-1)
                program, where, lane = (np.repeat(one, scale) for one in (program, where, lane))
            cells.append((value, program))
            wheres.append(where)
            lanes.append(lane)
            written.append(np.full(lane.shape, kind != "L"))
            widths.append(np.full(lane.shape, scale, dtype=np.int64))
        cell = np.concatenate([one for one, _ in cells])
        if not cell.size:
            return "apart"
        where = np.concatenate(wheres)
        several = 1 + max(spaces.values()) if spaces else 1
        if separable:
            # Programs differ by one translation of every address, exact modulo 2**32.
            cell = (cell & _M32) + (where << 32)
        else:
            program = np.concatenate([other for _, other in cells])
            low = int(cell.min())
            span = int(cell.max()) - low + 1
            if span * count * several >= 1 << 62:
                raise Unproved("budget")
            cell = (cell - low) + (program * several + where) * span
        lane = np.concatenate(lanes)
        write = np.concatenate(written)
        order = np.argsort(cell, kind="stable")
        cell, lane, write = cell[order], lane[order], write[order]
        starts = np.concatenate(([0], np.flatnonzero(cell[1:] != cell[:-1]) + 1))
        stored = np.logical_or.reduceat(write, starts)
        if scales:
            width = np.concatenate(widths)[order]
            if (stored & (np.minimum.reduceat(width, starts) != np.maximum.reduceat(width, starts))).any():
                if unmasked:
                    return "per-program"
                raise Unproved("element-size")
        lowest = np.minimum.reduceat(lane, starts)
        highest = np.maximum.reduceat(lane, starts)
        meeting = stored & (lowest != highest)
        if not meeting.any():
            return "apart"
        if unmasked:
            # The lanes a mask switches off may be the ones that met.
            return "per-program"
        if (lowest[meeting] <= -(1 << 60)).any():
            return "order-meet"
        return "lanes-meet-narrow" if (lowest[meeting] < 0).any() else "lanes-meet"

    def _eval(self, node, env):
        found = env.memo.get(node)
        if found is not None:
            return found
        entry = self.nodes[node]
        kind = entry[0]
        lead = env.lead

        def scalar(number):
            return np.full((1,) * lead, number, dtype=np.int64), 0

        def aligned(*nodes):
            values = [self._eval(child, env) for child in nodes]
            rank = max(value[1] for value in values)
            out = []
            for value, own in values:
                if own == 0:
                    value = value.reshape(value.shape + (1,) * rank)
                elif own != rank:
                    raise Unproved("eval", "rank")
                out.append(value)
            return out, rank

        if kind == "c":
            out = scalar(int(entry[1]))
        elif kind == "a":
            number = self._argument(env.args, entry[1])
            width = entry[2]
            number &= (1 << width) - 1
            if width > 1 and number >> (width - 1):
                number -= 1 << width
            out = scalar(number)
        elif kind == "np":
            out = scalar(int(env.grid[entry[1]]))
        elif kind == "p":
            if entry[1] not in env.origins:
                raise Unproved("unknown-base")
            out = scalar(env.origins[entry[1]])
        elif kind == "pid":
            out = env.pid[entry[1]], 0
        elif kind == "r":
            out = np.arange(entry[1], entry[2], dtype=np.int64).reshape((1,) * lead + (-1,)), 1
        elif kind == "it":
            shape = [1] * lead
            picked = env.picks.get(entry[1])
            if picked is None:
                picked = np.arange(env.trips[entry[1]], dtype=np.int64)
            shape[env.axis[entry[1]]] = picked.size
            out = picked.reshape(shape), 0
        elif kind == "trip":
            if entry[1] not in env.trips:
                lower, upper, step = (self._scalar(bound, env.args, env.grid) for bound in self.loops[entry[1]][:3])
                if step <= 0:
                    raise Unproved("loop-bounds")
                env.trips[entry[1]] = max(0, -((lower - upper) // step))
            out = scalar(env.trips[entry[1]])
        elif kind == "b":
            (left, right), rank = aligned(entry[3], entry[4])
            out = _binary(entry[1], entry[2], left, right), rank
        elif kind == "cmp":
            (left, right), rank = aligned(entry[3], entry[4])
            out = _compare(entry[1], entry[2], left, right), rank
        elif kind == "sel":
            (chosen, one, other), rank = aligned(entry[1], entry[2], entry[3])
            out = np.where(chosen != 0, one, other), rank
        elif kind == "x":
            value, rank = self._eval(entry[4], env)
            if entry[1] == "sext":
                value = -(value & 1) if entry[2] == 1 else value
            elif entry[1] == "zext":
                value = value & ((1 << entry[2]) - 1) if entry[2] < 64 else value
            else:
                value = _wrap(value, entry[3])
            out = value, rank
        elif kind == "bc":
            value, rank = self._eval(entry[2], env)
            shape = tuple(entry[1])
            if rank == 0:
                value = value.reshape(value.shape + (1,) * len(shape))
            elif rank != len(shape):
                raise Unproved("eval", "rank")
            out = np.broadcast_to(value, value.shape[:lead] + shape), len(shape)
        elif kind == "xd":
            value, rank = self._eval(entry[2], env)
            position = entry[1] if entry[1] >= 0 else entry[1] + rank + 1
            if rank == 0 or not 0 <= position <= rank:
                raise Unproved("eval", "rank")
            out = np.expand_dims(value, lead + position), rank + 1
        elif kind == "rs":
            value, rank = self._eval(entry[3], env)
            source, shape = tuple(entry[1]), tuple(entry[2])
            if rank != len(source):
                raise Unproved("eval", "rank")
            value = np.broadcast_to(value, value.shape[:lead] + source)
            out = value.reshape(value.shape[:lead] + shape), len(shape)
        elif kind == "ptr+":
            (pointer, offset), rank = aligned(entry[1], entry[2])
            out = pointer + (0 * offset if env.spaces else offset), rank
        else:
            raise Unproved("opaque-address")
        env.memo[node] = out
        return out
