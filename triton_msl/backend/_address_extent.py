"""Which cells does a launch reach through each pointer argument?

An attention launch is refused when an output is a view of the storage of another tensor
argument, because a view's shape does not bound what the kernel addresses through it: the
kernel forms its addresses from its own stride and length arguments. The source's access
plan (``triton_msl.codegen._access_plan``) describes those addresses as expressions over
the launch's values, so a launch can bound them.

``reached`` evaluates every load, store and atomic of the plan in interval arithmetic over
all programs, lanes and loop trips of one launch, and returns for each pointer argument the
first and last element offset any of them can reach. It is a hull: every address of the
source lies inside it, and cells inside it need not be addressed. A mask of the form
``value < bound`` narrows the value it compares where the address uses that same value at
the same lane; any other mask is ignored, which only widens the hull. A value the plan does
not describe, a loop whose trips are not bounded before the launch, and arithmetic that
could wrap its width all end in ``NoExtent``: there is then no bound, never a guessed one.

``written_apart`` is the use made of it: are the cells a launch stores through one argument
apart, byte for byte, from the cells it reaches through every other tensor argument of the
same storage, with every hull inside its storage?

The bound is of the source. The shader that runs may be a template written for the kernel
and not its lowering, and the answer holds for it on two conditions the routes are tested
for on storages of their own: it stores only in cells the source stores, and what it
computes does not depend on a cell the source does not load. A template that reads past
the source's cells and drops what it read (a whole tile at the end of a tensor) is within
them.
"""

import numpy as np

from triton_msl.autotuning._operand_overlap import _base, _share, _storage

PLAN_VERSION = 1
_ELEMENTWISE_WRAPPERS = ("bc",)


class NoExtent(Exception):
    pass


def _fits(low, high, width):
    return -(1 << (width - 1)) <= low and high < (1 << (width - 1))


def _divided(name, left, right):
    (low, high), (least, most) = left, right
    if least <= 0:
        raise NoExtent("a divisor that may not be positive")
    if name == "cdiv":
        if least != most:
            raise NoExtent("a rounded-up division by a value that varies")
        return -((-low) // least), -((-high) // least)
    if name == "fdiv":
        if least != most:
            if low < 0:
                raise NoExtent("a division of a value that may be negative")
            return low // most, high // least
        return low // least, high // least
    # sdiv truncates toward zero, udiv divides the unsigned value.
    if low >= 0:
        return low // most, high // least
    if name == "udiv" or least != most:
        raise NoExtent("a division of a value that may be negative")

    def truncated(value):
        return -((-value) // least) if value < 0 else value // least

    return truncated(low), truncated(high)


def _remainder(name, left, right):
    (low, high), (least, most) = left, right
    if least <= 0:
        raise NoExtent("a divisor that may not be positive")
    if low >= 0:
        return 0, min(high, most - 1)
    if name == "urem":
        raise NoExtent("an unsigned remainder of a value that may be negative")
    return -(most - 1), (most - 1 if high > 0 else 0)


def _binary(name, width, left, right):
    (low, high), (least, most) = left, right
    if width == 1:
        return 0, 1
    if name == "add":
        out = low + least, high + most
    elif name == "sub":
        out = low - most, high - least
    elif name == "mul":
        products = (low * least, low * most, high * least, high * most)
        out = min(products), max(products)
    elif name in ("sdiv", "udiv", "fdiv", "cdiv"):
        out = _divided(name, left, right)
    elif name in ("srem", "urem"):
        out = _remainder(name, left, right)
    elif name == "and":
        if low >= 0 and least >= 0:
            out = 0, min(high, most)
        elif low >= 0:
            out = 0, high
        elif least >= 0:
            out = 0, most
        else:
            raise NoExtent("a bitwise and of values that may be negative")
    elif name in ("or", "xor"):
        if low < 0 or least < 0:
            raise NoExtent("a bitwise or of values that may be negative")
        out = 0, (1 << max(high, most).bit_length()) - 1
    elif name in ("shl", "ashr", "lshr"):
        if least != most or not 0 <= least < width or (name == "lshr" and low < 0):
            raise NoExtent("a shift this bound does not follow")
        out = (low << least, high << least) if name == "shl" else (low >> least, high >> least)
    elif name in ("smin", "umin", "smax", "umax"):
        if name[0] == "u" and (low < 0 or least < 0):
            raise NoExtent("an unsigned comparison of values that may be negative")
        pick = min if name.endswith("min") else max
        out = pick(low, least), pick(high, most)
    else:
        raise NoExtent("an operation this bound does not follow")
    if not _fits(out[0], out[1], width):
        raise NoExtent("arithmetic that may wrap its width")
    return out


class _Hull:
    """One launch's bounds on the plan's values."""

    def __init__(self, plan, arguments, grid):
        self.nodes = plan["nodes"]
        self.loops = plan["loops"]
        self.arguments = arguments
        self.grid = [int(extent) for extent in grid]
        self.free = {}  # node -> bounds with no mask in force
        self.trips = {}

    def _scalar_argument(self, index, width):
        value = self.arguments[index] if index < len(self.arguments) else None
        if not isinstance(value, (bool, int, np.integer)):
            raise NoExtent("a scalar argument that is not an integer")
        number = int(value) & ((1 << width) - 1)
        if width > 1 and number >> (width - 1):
            number -= 1 << width
        return number

    def _most_trips(self, loop):
        found = self.trips.get(loop)
        if found is None:
            lower, upper, step = (self.value(bound, None) for bound in self.loops[loop][:3])
            if step[0] != step[1] or step[0] <= 0:
                raise NoExtent("a loop whose step is not one positive number")
            found = self.trips[loop] = max(0, -((lower[0] - upper[1]) // step[0]))
        return found

    def value(self, node, narrowed):
        """(least, most) of an integer node. ``narrowed`` maps a node to tighter bounds
        that hold at the lane being addressed, or is None."""
        if narrowed and node in narrowed:
            return narrowed[node]
        if not narrowed:
            found = self.free.get(node)
            if found is not None:
                return found
        entry = self.nodes[node]
        kind = entry[0]
        if kind == "c":
            out = int(entry[1]), int(entry[1])
        elif kind == "a":
            number = self._scalar_argument(entry[1], entry[2])
            out = number, number
        elif kind == "np":
            out = self.grid[entry[1]], self.grid[entry[1]]
        elif kind == "pid":
            out = 0, self.grid[entry[1]] - 1
        elif kind == "r":
            out = int(entry[1]), int(entry[2]) - 1
        elif kind == "it":
            out = 0, max(0, self._most_trips(entry[1]) - 1)
        elif kind == "trip":
            out = 0, self._most_trips(entry[1])
        elif kind == "b":
            out = _binary(entry[1], entry[2], self.value(entry[3], narrowed), self.value(entry[4], narrowed))
        elif kind == "cmp":
            out = 0, 1
        elif kind == "sel":
            one, other = self.value(entry[2], narrowed), self.value(entry[3], narrowed)
            out = min(one[0], other[0]), max(one[1], other[1])
        elif kind == "x":
            low, high = self.value(entry[4], narrowed)
            if entry[2] == 1 and not (0 <= low and high <= 1):
                low, high = 0, 1  # one bit, whatever it was written as
            if entry[1] == "sext":
                out = (-high, -low) if entry[2] == 1 else (low, high)
            elif entry[1] == "zext":
                if low < 0:
                    raise NoExtent("a zero extension of a value that may be negative")
                out = low, high
            elif entry[3] == 1:
                out = 0, 1
            else:
                if not _fits(low, high, entry[3]):
                    raise NoExtent("a truncation that may drop bits")
                out = low, high
        elif kind == "bc":
            out = self.value(entry[2], narrowed)
        elif kind == "xd":
            # Another lane's element from here down: what a mask says of this lane's is not known of it.
            out = self.value(entry[2], None)
        elif kind == "rs":
            out = self.value(entry[3], None)
        else:
            raise NoExtent("a value the plan does not describe")
        if not narrowed:
            self.free[node] = out
        return out

    def pointer(self, node, narrowed):
        """(argument index, least, most element offset) of a pointer node."""
        entry = self.nodes[node]
        kind = entry[0]
        if kind == "p":
            return entry[1], 0, 0
        if kind == "ptr+":
            index, low, high = self.pointer(entry[1], narrowed)
            least, most = self.value(entry[2], narrowed)
            return index, low + least, high + most
        if kind == "bc":
            return self.pointer(entry[2], narrowed)
        if kind == "xd":
            return self.pointer(entry[2], None)
        if kind == "rs":
            return self.pointer(entry[3], None)
        if kind == "sel":
            one, other = self.pointer(entry[2], narrowed), self.pointer(entry[3], narrowed)
            if one[0] != other[0]:
                raise NoExtent("a pointer chosen between two arguments")
            return one[0], min(one[1], other[1]), max(one[2], other[2])
        raise NoExtent("a pointer the plan does not trace to an argument")

    def narrowing(self, mask):
        """Tighter bounds a true mask gives the values it compares, or None for a mask no lane passes.

        Only what the mask says of the lane itself is used: a comparison reached through
        elementwise operations and broadcasts. Everything else is left out, which keeps
        the wider bound.
        """
        out = {}
        pending = [mask]
        while pending:
            node = pending.pop()
            entry = self.nodes[node]
            kind = entry[0]
            if kind in _ELEMENTWISE_WRAPPERS:
                pending.append(entry[2])
            elif kind == "b" and entry[1] == "and" and entry[2] == 1:
                pending.extend((entry[3], entry[4]))
            elif kind == "cmp":
                self._compared(entry, out)
        for low, high in out.values():
            if low > high:
                return None
        return out

    def _compared(self, entry, out):
        predicate, left, right = str(entry[1]), entry[3], entry[4]
        try:
            one, other = self.value(left, None), self.value(right, None)
        except NoExtent:
            return
        if predicate.startswith("u"):
            if one[0] < 0 or other[0] < 0:
                return
            predicate = "s" + predicate[1:]
        if predicate in ("sgt", "sge"):
            predicate = {"sgt": "slt", "sge": "sle"}[predicate]
            left, right, one, other = right, left, other, one
        if predicate not in ("slt", "sle"):
            return
        strict = predicate == "slt"

        def target(node):
            # The value itself under its broadcasts: the same element at this lane.
            while self.nodes[node][0] in _ELEMENTWISE_WRAPPERS:
                node = self.nodes[node][2]
            return node

        def tighten(node, low, high):
            before = out.get(node) or self.value(node, None)
            out[node] = max(before[0], low), min(before[1], high)

        # left < right (or <=): left is at most the most right can be, right at least the least left can be.
        for node in {left, target(left)}:
            tighten(node, one[0], other[1] - strict)
        for node in {right, target(right)}:
            tighten(node, one[0] + strict, other[1])


def reached(plan, arguments, grid):
    """{pointer argument: [least, most element offset reached, whether stored]} for one launch.

    Raises NoExtent when the plan gives no bound.
    """
    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION or plan.get("why"):
        raise NoExtent("the kernel has no access plan")
    if any(int(extent) < 1 for extent in grid):
        return {}
    hull = _Hull(plan, arguments, grid)
    out = {}
    for kind, pointer, mask, _shape, _loops, unbounded in plan["accesses"]:
        if unbounded:
            raise NoExtent("an access in a loop whose trips are not known before the launch")
        narrowed = None
        if mask != -1:
            narrowed = hull.narrowing(mask)
            if narrowed is None:
                continue
        index, low, high = hull.pointer(pointer, narrowed)
        entry = out.setdefault(index, [low, high, False])
        entry[0], entry[1] = min(entry[0], low), max(entry[1], high)
        entry[2] = entry[2] or kind != "L"
    return out


def _bytes_reached(value, low, high, size):
    """(storage, first byte, one past the last byte) of a hull, or None when it leaves the storage."""
    value = _base(value)
    storage = _storage(value)
    if not size or value.element_size() != size:
        return None
    first = (int(value.storage_offset()) + low) * size
    last = (int(value.storage_offset()) + high + 1) * size
    if first < 0 or last > storage[2] - storage[1]:
        return None
    if storage[0] == "cpu":
        first += storage[1]
        last += storage[1]
    return storage, first, last


def written_apart(plan, arguments, grid, outputs=()):
    """True when every cell the launch stores is shown apart from every other tensor's cells.

    For each tensor argument the launch stores through, the bytes its accesses can reach
    share nothing with the bytes reached through any other tensor argument of the same
    storage, and every hull lies inside its storage. ``outputs`` are argument indexes that
    count as stored whatever the plan says of them; one the plan gives no extent is not
    shown apart. False when that is not shown, for whatever reason.
    """
    try:
        extents = reached(plan, arguments, grid)
        pointers = {int(index): entry for index, entry in plan["pointers"].items()}
        spans = {}
        for index, (low, high, stored) in extents.items():
            value = arguments[index] if index < len(arguments) else None
            if not hasattr(value, "data_ptr"):
                return False
            span = _bytes_reached(value, low, high, pointers[index][1])
            if span is None:
                return False
            spans[index] = span, stored or index in outputs
        if any(index not in spans for index in outputs):
            return False
        for index, value in enumerate(arguments):
            if not hasattr(value, "data_ptr"):
                continue
            # A tensor the plan does not name may be addressed by something it does not describe.
            if index not in pointers:
                return False
            # One whose storage cannot be read may share memory with anything.
            _storage(value)
        for index, ((storage, first, last), stored) in spans.items():
            if not stored:
                continue
            for other, ((other_storage, start, stop), _) in spans.items():
                if other == index or not _share(storage, other_storage):
                    continue
                if first < stop and start < last:
                    return False
        return True
    except Exception:
        return False
