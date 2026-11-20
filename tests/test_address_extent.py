"""The bound on the cells a launch reaches through each pointer argument.

``triton_msl.backend._address_extent.reached`` bounds, from a kernel's access plan and one
launch's arguments and grid, the first and the last element any load or store of the source
reaches through each pointer argument. An attention launch whose output is a view of the
storage of another tensor argument is admitted on that bound alone, so the bound has to
hold every address there is: one cell short and an output is written over an input.

The tests here need no device. The exact addresses come from the evaluator the lane-order
check uses (``triton_msl.backend._lane_order``), lane by lane and program by program:

* each arithmetic rule holds every result of the exact operation over the values of its
  operand intervals, at eight bits (where a sum wraps) and at thirty-two, or gives no bound;
* plans drawn at random (typed expressions over constants, scalar arguments, the program
  id, lane ranges, extensions, selects and masks that compare values the address uses) are
  bounded at least as wide as their exact addresses, or not at all;
* a mask narrows only the value it compares at the same lane; an opaque value, a loop whose
  trips are not known, a pointer chosen between two arguments and arithmetic that may wrap
  give no bound;
* ``written_apart`` on host tensors: blocks of one storage that touch are apart, blocks
  that share one cell are not, nor is a bound that leaves the storage, an element size the
  plan does not name, or a tensor the plan does not name.

tests/test_attention_shared_storage_layouts.py holds the launches.
"""

import itertools
import random

import numpy as np
import pytest
import torch

from triton_msl.backend import _address_extent as extent
from triton_msl.backend import _lane_order as exact
from triton_msl.backend._address_extent import NoExtent, reached, written_apart

BINARY = (
    "add",
    "sub",
    "mul",
    "sdiv",
    "udiv",
    "fdiv",
    "cdiv",
    "srem",
    "urem",
    "and",
    "or",
    "xor",
    "shl",
    "ashr",
    "lshr",
    "smin",
    "smax",
    "umin",
    "umax",
)


class Plan:
    """An access plan written by hand: nodes, then accesses."""

    def __init__(self, pointers=2, size=4):
        self.nodes, self.loops, self.accesses = [], [], []
        self.pointers = {index: [f"P{index}", size] for index in range(pointers)}

    def node(self, *fields):
        self.nodes.append(list(fields))
        return len(self.nodes) - 1

    def const(self, value):
        return self.node("c", value)

    def binary(self, name, width, left, right):
        return self.node("b", name, width, left, right)

    def at(self, pointer, offset):
        return self.node("ptr+", self.node("p", pointer), offset)

    def access(self, kind, pointer, mask=-1, shape=(), loops=(), unbounded=False):
        self.accesses.append([kind, pointer, mask, list(shape), list(loops), unbounded])

    def plan(self, **changes):
        block = 1
        for access in self.accesses:
            block = max(block, int(np.prod(access[3] or [1])))
        plan = {
            "version": extent.PLAN_VERSION,
            "why": None,
            "block": block,
            "nodes": self.nodes,
            "loops": self.loops,
            "accesses": self.accesses,
            "pointers": self.pointers,
        }
        plan.update(changes)
        return plan


def addresses(plan, arguments, grid):
    """{pointer argument: [least, most element offset, whether stored]} over every lane the
    plan's masks keep, of every program and loop trip of one launch, evaluated exactly.

    A pointer argument no lane reaches is absent. Raises ``exact.Unproved`` for a plan the
    exact evaluator does not follow.
    """
    check = exact.LaunchCheck(plan)
    origins = {index: 0 for index in check.pointers}
    found = {}
    for z, y, x in itertools.product(*(range(int(count)) for count in reversed(grid))):
        for ordinal, (kind, pointer, mask, shape, loops, _) in enumerate(check.accesses):
            (base,) = check.bases[ordinal]
            program = [np.full((1,), number, dtype=np.int64) for number in (x, y, z)]
            outside = exact._Env(1, 1, program, {}, {}, origins, arguments, grid)
            trips, axis = {}, {}
            for loop in loops:
                lower, upper, step = (
                    int(check._eval(bound, outside)[0].reshape(-1)[0]) for bound in check.loops[loop][:3]
                )
                assert step > 0
                trips[loop] = max(0, -((lower - upper) // step))
                axis[loop] = 1 + len(axis)
            if any(count == 0 for count in trips.values()):
                continue
            lead = 1 + len(loops)
            program = [one.reshape((1,) * lead) for one in program]
            env = exact._Env(lead, 1, program, trips, axis, origins, arguments, grid)
            with np.errstate(over="ignore"):
                value, rank = check._eval(pointer, env)
                if rank == 0:
                    value = value.reshape(value.shape + (1,) * len(shape))
                keep = np.ones((1,) * value.ndim, dtype=bool)
                if mask >= 0:
                    keep, mask_rank = check._eval(mask, env)
                    if mask_rank == 0:
                        keep = keep.reshape(keep.shape + (1,) * len(shape))
                    keep = keep != 0
            value, keep = np.broadcast_arrays(value, keep)
            value = value[keep]
            if not value.size:
                continue
            entry = found.setdefault(base, [int(value.min()), int(value.max()), False])
            entry[0], entry[1] = min(entry[0], int(value.min())), max(entry[1], int(value.max()))
            entry[2] = entry[2] or kind != "L"
    return found


def assert_holds(plan, arguments, grid):
    """The bound holds every exact address. Returns False when there is no bound to check."""
    try:
        bound = reached(plan, arguments, grid)
    except NoExtent:
        return False
    for index, (low, high, stored) in addresses(plan, arguments, grid).items():
        assert index in bound, (index, bound)
        assert bound[index][0] <= low and high <= bound[index][1], (index, bound[index], low, high)
        assert bound[index][2] or not stored
    return True


# -- the arithmetic rules ---------------------------------------------------------------------


def _intervals(rng, width, count):
    half = 1 << (width - 1)
    edges = [-half, -half + 1, -3, -1, 0, 1, 2, 7, half - 2, half - 1]
    for _ in range(count):
        low = rng.choice(edges) if rng.random() < 0.4 else rng.randint(-40, 40)
        high = min(half - 1, low + rng.choice((0, 0, 1, 3, 9)))
        yield max(low, -half), high


@pytest.mark.parametrize("width", [8, 32])
@pytest.mark.parametrize("name", BINARY)
def test_a_rule_holds_every_result_of_the_exact_operation_or_gives_no_bound(name, width):
    rng = random.Random(f"{name}-{width}")
    bounded = 0
    for (low, high), (least, most) in zip(_intervals(rng, width, 600), _intervals(rng, width, 600)):
        left = np.arange(low, high + 1, dtype=np.int64).reshape(-1, 1)
        right = np.arange(least, most + 1, dtype=np.int64).reshape(1, -1)
        try:
            out = extent._binary(name, width, (low, high), (least, most))
        except NoExtent:
            continue
        with np.errstate(over="ignore"):
            got = exact._binary(name, width, *np.broadcast_arrays(left, right))
        assert out[0] <= int(got.min()) and int(got.max()) <= out[1], (name, width, low, high, least, most, out)
        bounded += 1
    assert bounded >= (15 if name in ("shl", "ashr", "lshr") else 60), bounded


@pytest.mark.parametrize("name", ["sdiv", "udiv", "fdiv", "cdiv", "srem", "urem"])
def test_a_divisor_that_may_be_zero_gives_no_bound(name):
    for divisor in ((0, 0), (-2, 3), (0, 5), (-4, -1)):
        with pytest.raises(NoExtent):
            extent._binary(name, 32, (0, 9), divisor)


def test_a_sum_that_may_wrap_its_width_gives_no_bound():
    assert extent._binary("add", 32, (0, 2**31 - 2), (0, 1)) == (0, 2**31 - 1)
    with pytest.raises(NoExtent, match="wrap"):
        extent._binary("add", 32, (0, 2**31 - 1), (0, 1))
    with pytest.raises(NoExtent, match="wrap"):
        extent._binary("mul", 32, (0, 65536), (0, 32768))
    assert extent._binary("mul", 64, (0, 65536), (0, 32768)) == (0, 65536 * 32768)


# -- plans drawn at random --------------------------------------------------------------------

LANES = 6


class _Drawn(Plan):
    """A typed expression graph: every node has a width, and an operation's operands share one."""

    def __init__(self, rng):
        super().__init__(pointers=2)
        self.rng = rng
        self.pool = {}  # (width, rank) -> nodes, reused so that masks compare what addresses use

    def leaf(self, width, rank):
        rng = self.rng
        if rank:
            if width == 32 and rng.random() < 0.7:
                start = rng.choice((0, 0, 3))
                return self.node("r", start, start + LANES)
            return self.node("bc", (LANES,), self.leaf(width, 0))
        pick = rng.random()
        if width == 1:
            return self.const(rng.choice((0, 1)))
        if width == 32 and pick < 0.2:
            return self.node("pid", rng.choice((0, 1)))
        if width == 32 and pick < 0.3:
            return self.node("np", rng.choice((0, 1)))
        if pick < 0.6:
            return self.node("a", 2 + rng.randrange(3) + {8: 0, 32: 3, 64: 6}[width], width)
        # Not the least 64-bit number: the exact evaluator's own int64 arithmetic wraps on it.
        half = 1 << (min(width, 63) - 1)
        return self.const(rng.choice((0, 1, 2, 3, 5, 8, -1, -4, half - 1, -half)) if pick < 0.75 else rng.randint(0, 9))

    def value(self, width, rank, depth):
        rng = self.rng
        known = self.pool.setdefault((width, rank), [])
        if known and rng.random() < 0.35:
            return rng.choice(known)
        if depth == 0 or rng.random() < 0.2:
            node = self.leaf(width, rank)
        elif width == 1:
            other = rng.choice((8, 32, 64))
            sides = [self.value(other, rank if rng.random() < 0.7 else 0, depth - 1) for _ in range(2)]
            node = self.node(
                "cmp", rng.choice(("slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge", "eq", "ne")), other, *sides
            )
            if rank and not self._rank(node):
                node = self.node("bc", (LANES,), node)
        else:
            pick = rng.random()
            ranks = [rank, rank if rng.random() < 0.6 else 0]
            rng.shuffle(ranks)
            if pick < 0.6:
                node = self.binary(rng.choice(BINARY), width, *(self.value(width, one, depth - 1) for one in ranks))
            elif pick < 0.8:
                source = rng.choice([other for other in (1, 8, 32, 64) if other != width])
                kind = "trunc" if source > width else rng.choice(("sext", "zext"))
                node = self.node("x", kind, source, width, self.value(source, rank, depth - 1))
            else:
                node = self.node(
                    "sel",
                    self.value(1, ranks[0], depth - 1),
                    self.value(width, rank, depth - 1),
                    self.value(width, ranks[1], depth - 1),
                )
            if rank and not self._rank(node):
                node = self.node("bc", (LANES,), node)
        known.append(node)
        return node

    def _rank(self, node):
        entry = self.nodes[node]
        if entry[0] in ("r", "bc"):
            return 1
        children = {"b": (3, 4), "x": (4,), "cmp": (3, 4), "sel": (1, 2, 3)}.get(entry[0], ())
        return max((self._rank(entry[position]) for position in children), default=0)

    def mask(self):
        rng = self.rng
        terms = []
        for _ in range(rng.choice((1, 1, 2))):
            width = rng.choice((32, 64))
            sides = [self.value(width, 1, 1), self.value(width, rng.choice((0, 1)), 1)]
            rng.shuffle(sides)
            terms.append(
                self.node("cmp", rng.choice(("slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "ne")), width, *sides)
            )
        mask = terms[0]
        for term in terms[1:]:
            mask = self.binary(rng.choice(("and", "and", "or")), 1, mask, term)
        return mask if self._rank(mask) else self.node("bc", (LANES,), mask)


def _drawn(seed):
    rng = random.Random(seed)
    drawn = _Drawn(rng)
    for _ in range(rng.choice((1, 2, 3))):
        mask = drawn.mask() if rng.random() < 0.6 else -1
        offset = drawn.value(rng.choice((32, 64)), 1, rng.choice((1, 2, 3)))
        if not drawn._rank(offset):
            offset = drawn.node("bc", (LANES,), offset)
        drawn.access(rng.choice("LS"), drawn.at(rng.choice((0, 1)), offset), mask, (LANES,))
    scalars = [rng.choice((0, 1, 2, 3, 5, 17, -1, -6, 100)) for _ in range(9)]
    return drawn.plan(), [None, None, *scalars], (rng.choice((1, 3)), rng.choice((1, 2)), 1)


def test_plans_drawn_at_random_are_bounded_at_least_as_wide_as_their_exact_addresses():
    bounded = evaluated = 0
    for seed in range(6000):
        plan, arguments, grid = _drawn(seed)
        try:
            addresses(plan, arguments, grid)
        except exact.Unproved:
            continue  # a division by zero or a shift past the width: not a program
        evaluated += 1
        try:
            bounded += assert_holds(plan, arguments, grid)
        except AssertionError as error:
            raise AssertionError(f"seed {seed}: {error}") from None
    assert evaluated >= 2000 and bounded >= 600, (evaluated, bounded)


# -- what a mask narrows, and what gives no bound ---------------------------------------------


def _rows(n_argument=2, lanes=64, masked=True, same=True):
    """``store(P0 + arange(lanes), mask=arange(lanes) < n)`` and the load of ``P1`` the same way."""
    plan = Plan()
    lane = plan.node("r", 0, lanes)
    other = lane if same else plan.node("r", 0, lanes)
    count = plan.node("bc", (lanes,), plan.node("a", n_argument, 32))
    mask = plan.node("cmp", "slt", 32, lane, count) if masked else -1
    plan.access("L", plan.at(1, other), mask, (lanes,))
    plan.access("S", plan.at(0, other), mask, (lanes,))
    return plan


def test_a_mask_narrows_the_value_it_compares():
    plan = _rows().plan()
    assert reached(plan, [None, None, 40], (1, 1, 1)) == {1: [0, 39, False], 0: [0, 39, True]}
    assert reached(_rows(masked=False).plan(), [None, None, 40], (1, 1, 1)) == {1: [0, 63, False], 0: [0, 63, True]}
    assert assert_holds(plan, [None, None, 40], (1, 1, 1))
    # A count past the lanes narrows nothing, and one that admits no lane leaves no access.
    assert reached(plan, [None, None, 400], (1, 1, 1)) == {1: [0, 63, False], 0: [0, 63, True]}
    assert reached(plan, [None, None, 0], (1, 1, 1)) == {}
    assert reached(plan, [None, None, -5], (1, 1, 1)) == {}


def test_a_mask_on_another_range_narrows_nothing():
    """The same numbers, another value: nothing says the two ranges are compared lane for lane."""
    assert reached(_rows(same=False).plan(), [None, None, 40], (1, 1, 1))[0] == [0, 63, True]


@pytest.mark.parametrize("predicate, bound, last", [("slt", 40, 39), ("sle", 40, 40), ("ult", 40, 39), ("ule", 40, 40)])
def test_each_comparison_narrows_to_its_own_last_lane(predicate, bound, last):
    plan = Plan()
    lane = plan.node("r", 0, 64)
    mask = plan.node("cmp", predicate, 32, lane, plan.node("bc", (64,), plan.const(bound)))
    plan.access("S", plan.at(0, lane), mask, (64,))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [0, last, True]}
    assert assert_holds(plan.plan(), [None, None], (1, 1, 1))
    # The comparison written the other way round.
    swapped = Plan()
    lane = swapped.node("r", 0, 64)
    flipped = {"slt": "sgt", "sle": "sge", "ult": "ugt", "ule": "uge"}[predicate]
    mask = swapped.node("cmp", flipped, 32, swapped.node("bc", (64,), swapped.const(bound)), lane)
    swapped.access("S", swapped.at(0, lane), mask, (64,))
    assert reached(swapped.plan(), [None, None], (1, 1, 1)) == {0: [0, last, True]}


def test_a_lower_bound_narrows_from_below():
    plan = Plan()
    lane = plan.node("r", 0, 64)
    mask = plan.node("cmp", "sge", 32, lane, plan.node("bc", (64,), plan.const(10)))
    plan.access("S", plan.at(0, lane), mask, (64,))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [10, 63, True]}
    assert assert_holds(plan.plan(), [None, None], (1, 1, 1))


@pytest.mark.parametrize("joined", ["or", "xor"])
def test_a_mask_that_either_of_two_comparisons_passes_narrows_nothing(joined):
    plan = Plan()
    lane = plan.node("r", 0, 64)
    low = plan.node("cmp", "slt", 32, lane, plan.node("bc", (64,), plan.const(3)))
    high = plan.node("cmp", "sge", 32, lane, plan.node("bc", (64,), plan.const(60)))
    plan.access("S", plan.at(0, lane), plan.binary(joined, 1, low, high), (64,))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [0, 63, True]}
    assert assert_holds(plan.plan(), [None, None], (1, 1, 1))


@pytest.mark.parametrize("predicate", ["eq", "ne"])
def test_a_comparison_for_equality_narrows_nothing(predicate):
    plan = Plan()
    lane = plan.node("r", 0, 64)
    mask = plan.node("cmp", predicate, 32, lane, plan.node("bc", (64,), plan.const(3)))
    plan.access("S", plan.at(0, lane), mask, (64,))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [0, 63, True]}
    assert assert_holds(plan.plan(), [None, None], (1, 1, 1))


def test_an_unsigned_comparison_of_a_value_that_may_be_negative_narrows_nothing():
    plan = Plan()
    lane = plan.binary("sub", 32, plan.node("r", 0, 64), plan.node("bc", (64,), plan.const(8)))
    mask = plan.node("cmp", "ult", 32, lane, plan.node("bc", (64,), plan.const(40)))
    plan.access("S", plan.at(0, lane), mask, (64,))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [-8, 55, True]}
    assert assert_holds(plan.plan(), [None, None], (1, 1, 1))


def _tile(rows, columns, row_bound, column_bound, through_one_node=True):
    """A rows x columns tile at ``P0 + row[:, None] * stride + column[None, :]`` under
    ``(row[:, None] < row_bound) & (column[None, :] < column_bound)``, the stride argument 3."""
    plan = Plan()
    shape = (rows, columns)
    row = plan.node("bc", shape, plan.node("xd", 1, plan.node("r", 0, rows)))
    column = plan.node("bc", shape, plan.node("xd", 0, plan.node("r", 0, columns)))
    stride = plan.node("bc", shape, plan.node("a", 3, 32))
    offset = plan.binary("add", 32, plan.binary("mul", 32, row, stride), column)
    if not through_one_node:
        # The mask compares its own expansion of the same range.
        row = plan.node("bc", shape, plan.node("xd", 1, plan.node("r", 0, rows)))
    mask = plan.binary(
        "and",
        1,
        plan.node("cmp", "slt", 32, row, plan.node("bc", shape, plan.node("a", row_bound, 32))),
        plan.node("cmp", "slt", 32, column, plan.node("bc", shape, plan.node("a", column_bound, 32))),
    )
    plan.access("S", plan.at(0, offset), mask, shape)
    return plan.plan()


def test_a_tile_mask_narrows_rows_and_columns():
    arguments = [None, None, 0, 100, 5, 20]
    plan = _tile(8, 32, 4, 5)
    assert reached(plan, arguments, (1, 1, 1)) == {0: [0, 4 * 100 + 19, True]}
    assert assert_holds(plan, arguments, (1, 1, 1))
    # Without the node in common the rows keep their whole range; the columns still narrow.
    apart = _tile(8, 32, 4, 5, through_one_node=False)
    assert reached(apart, arguments, (1, 1, 1)) == {0: [0, 7 * 100 + 19, True]}
    assert assert_holds(apart, arguments, (1, 1, 1))


def test_a_value_seen_through_another_axis_is_not_narrowed():
    """``mask = row < bound`` says nothing of ``row`` laid along another axis."""
    plan = Plan()
    row = plan.node("r", 0, 8)
    across = plan.node("rs", (8,), (8,), row)
    mask = plan.node("cmp", "slt", 32, row, plan.node("bc", (8,), plan.const(3)))
    plan.access("S", plan.at(0, across), mask, (8,))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [0, 7, True]}


def test_rows_and_columns_laid_from_one_range_are_narrowed_apart():
    """``row < 3`` at a lane says nothing of the column there, though one range is laid out as both."""
    plan = Plan()
    one = plan.node("r", 0, 8)
    row = plan.node("bc", (8, 8), plan.node("xd", 1, one))
    column = plan.node("bc", (8, 8), plan.node("xd", 0, one))
    mask = plan.node("cmp", "slt", 32, row, plan.node("bc", (8, 8), plan.const(3)))
    plan.access("S", plan.at(0, column), mask, (8, 8))
    plan.access("S", plan.at(1, row), mask, (8, 8))
    assert reached(plan.plan(), [None, None], (1, 1, 1)) == {0: [0, 7, True], 1: [0, 2, True]}
    assert assert_holds(plan.plan(), [None, None], (1, 1, 1))


def test_a_counted_loop_is_bounded_over_its_trips_and_the_grid():
    plan = Plan()
    plan.loops.append([plan.const(0), plan.node("a", 2, 32), plan.const(32), 32])
    start = plan.binary("add", 32, plan.const(0), plan.binary("mul", 32, plan.node("it", 0), plan.const(32)))
    row = plan.binary("mul", 32, plan.node("pid", 0), plan.node("a", 3, 32))
    lane = plan.binary("add", 32, plan.node("bc", (32,), plan.binary("add", 32, row, start)), plan.node("r", 0, 32))
    mask = plan.node("cmp", "slt", 32, lane, plan.node("bc", (32,), plan.binary("add", 32, row, plan.node("a", 2, 32))))
    plan.access("L", plan.at(1, lane), mask, (32,), loops=(0,))
    built = plan.plan()
    # 70 keys a row, 3 programs, rows 100 apart: the last key of the last row.
    arguments = [None, None, 70, 100]
    assert reached(built, arguments, (3, 1, 1)) == {1: [0, 2 * 100 + 69, False]}
    assert assert_holds(built, arguments, (3, 1, 1))
    # No key at all: the bound does not know the loop never runs, and still holds.
    assert assert_holds(built, [None, None, 0, 100], (3, 1, 1))
    assert reached(built, arguments, (0, 1, 1)) == {}


def _one_store(offset_of):
    plan = Plan()
    plan.access("S", plan.at(0, offset_of(plan)), -1, ())
    return plan.plan()


@pytest.mark.parametrize(
    "offset_of, reason",
    [
        (lambda plan: plan.node("o", 1), "does not describe"),
        (lambda plan: plan.binary("add", 32, plan.node("a", 2, 32), plan.const(2**31 - 1)), "wrap"),
        (
            lambda plan: plan.node("x", "zext", 32, 64, plan.binary("sub", 32, plan.node("a", 2, 32), plan.const(9))),
            "zero extension",
        ),
        (lambda plan: plan.node("x", "trunc", 64, 32, plan.const(2**31)), "truncation"),
        (lambda plan: plan.node("a", 3, 32), "not an integer"),
        (lambda plan: plan.binary("shl", 32, plan.const(1), plan.const(40)), "shift"),
        (lambda plan: plan.binary("shl", 32, plan.const(1), plan.node("pid", 0)), "shift"),
        (lambda plan: plan.binary("frob", 32, plan.const(1), plan.const(1)), "does not follow"),
    ],
    ids=[
        "opaque",
        "wrap",
        "zero-extension",
        "truncation",
        "float-argument",
        "shift-past-width",
        "shift-that-varies",
        "unknown-operation",
    ],
)
def test_what_the_bound_cannot_follow_gives_no_bound(offset_of, reason):
    with pytest.raises(NoExtent, match=reason):
        reached(_one_store(offset_of), [None, None, 5, 0.5], (4, 1, 1))


def test_a_bit_written_as_minus_one_extends_as_a_bit():
    plan = _one_store(lambda plan: plan.node("x", "sext", 1, 32, plan.node("x", "trunc", 32, 1, plan.const(-1))))
    assert reached(plan, [None, None], (1, 1, 1)) == {0: [-1, 0, True]}
    assert assert_holds(plan, [None, None], (1, 1, 1))


def test_an_access_in_a_loop_of_unknown_trips_gives_no_bound():
    plan = Plan()
    plan.access("S", plan.at(0, plan.const(0)), -1, (), unbounded=True)
    with pytest.raises(NoExtent, match="trips are not known"):
        reached(plan.plan(), [None, None], (1, 1, 1))


def test_a_pointer_chosen_between_two_arguments_gives_no_bound():
    plan = Plan()
    chosen = plan.node(
        "sel", plan.node("cmp", "slt", 32, plan.node("a", 2, 32), plan.const(4)), plan.node("p", 0), plan.node("p", 1)
    )
    plan.access("S", plan.node("ptr+", chosen, plan.const(0)), -1, ())
    with pytest.raises(NoExtent, match="between two arguments"):
        reached(plan.plan(), [None, None, 1], (1, 1, 1))
    carried = Plan()
    carried.access("S", carried.node("ptr+", carried.node("pu", (0,), 1), carried.const(0)), -1, ())
    with pytest.raises(NoExtent, match="does not trace"):
        reached(carried.plan(), [None, None], (1, 1, 1))


@pytest.mark.parametrize("change", [{"why": "a call this plan cannot follow"}, {"version": extent.PLAN_VERSION + 1}])
def test_a_plan_that_is_not_one_gives_no_bound(change):
    with pytest.raises(NoExtent):
        reached(_rows().plan(**change), [None, None, 40], (1, 1, 1))
    with pytest.raises(NoExtent):
        reached(None, [None, None, 40], (1, 1, 1))


# -- written_apart, on host tensors -----------------------------------------------------------


def _blocks(first, second, cells=256, dtype=torch.float32):
    storage = torch.zeros(cells, dtype=dtype)
    return storage[first[0] : first[1]], storage[second[0] : second[1]]


def test_blocks_of_one_storage_that_touch_are_apart_and_blocks_that_share_a_cell_are_not():
    plan = _rows().plan()
    out, source = _blocks((0, 40), (40, 80))
    assert written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    out, source = _blocks((40, 80), (0, 40))
    assert written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    out, source = _blocks((0, 40), (39, 79))
    assert not written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    out, source = _blocks((39, 79), (0, 40))
    assert not written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    # The views are not the measure: 40 rows are reached through a view of 8.
    out, source = _blocks((0, 8), (8, 48))
    assert not written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    # Nor are views that overlap, when the cells reached through them do not.
    out, source = _blocks((0, 64), (40, 104))
    assert written_apart(plan, [out, source, 40], (1, 1, 1), (0,))


def test_the_same_view_twice_is_not_apart():
    out, _ = _blocks((0, 40), (40, 80))
    assert not written_apart(_rows().plan(), [out, out, 40], (1, 1, 1), (0,))


def test_two_host_storages_over_the_same_memory_are_compared_by_address():
    memory = np.zeros(128, dtype=np.float32)
    plan = _rows().plan()
    assert written_apart(plan, [torch.from_numpy(memory[:40]), torch.from_numpy(memory[40:80]), 40], (1, 1, 1), (0,))
    assert not written_apart(
        plan, [torch.from_numpy(memory[:40]), torch.from_numpy(memory[39:79]), 40], (1, 1, 1), (0,)
    )


def test_a_bound_that_leaves_its_storage_is_not_apart():
    plan = _rows().plan()
    out, source = _blocks((0, 40), (236, 256))  # 40 rows from cell 236 of 256
    assert not written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    out, source = _blocks((236, 256), (0, 40))
    assert not written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    # A bound that starts before its storage.
    before = Plan()
    lane = before.binary("sub", 32, before.node("r", 0, 8), before.node("bc", (8,), before.const(1)))
    before.access("S", before.at(0, lane), -1, (8,))
    before.access("L", before.at(1, before.node("r", 0, 8)), -1, (8,))
    out, source = _blocks((0, 8), (100, 108))
    assert not written_apart(before.plan(), [out, source], (1, 1, 1), (0,))
    out, source = _blocks((1, 9), (100, 108))
    assert written_apart(before.plan(), [out, source], (1, 1, 1), (0,))


def test_an_element_size_the_plan_does_not_name_is_not_apart():
    plan = _rows().plan()
    out, source = _blocks((0, 40), (40, 80), dtype=torch.float16)
    assert not written_apart(plan, [out, source, 40], (1, 1, 1), (0,))
    half = _rows()
    half.pointers = {0: ["P0", 2], 1: ["P1", 2]}
    assert written_apart(half.plan(), [out, source, 40], (1, 1, 1), (0,))


def test_a_tensor_the_plan_does_not_name_is_not_apart():
    plan = _rows(n_argument=3).plan()
    out, source = _blocks((0, 40), (40, 80))
    assert written_apart(plan, [out, source, 7, 40], (1, 1, 1), (0,))
    assert not written_apart(plan, [out, source, torch.zeros(4), 40], (1, 1, 1), (0,))


def test_an_output_the_plan_does_not_store_counts_as_stored():
    """Two arguments the source only loads may share cells; a published output may not."""
    plan = Plan()
    lane = plan.node("r", 0, 16)
    plan.access("L", plan.at(0, lane), -1, (16,))
    plan.access("L", plan.at(1, lane), -1, (16,))
    one, other = _blocks((0, 16), (8, 24))
    assert written_apart(plan.plan(), [one, other], (1, 1, 1))
    assert not written_apart(plan.plan(), [one, other], (1, 1, 1), (0,))
    # An output the plan reaches nowhere has no bound.
    unreached = Plan()
    unreached.access("L", unreached.at(1, unreached.node("r", 0, 16)), -1, (16,))
    assert not written_apart(unreached.plan(), [one, other], (1, 1, 1), (0,))


def test_no_plan_no_bound_and_a_scalar_that_is_not_an_integer_are_not_apart():
    out, source = _blocks((0, 40), (40, 80))
    assert not written_apart(None, [out, source, 40], (1, 1, 1), (0,))
    assert not written_apart(_rows().plan(why="a call this plan cannot follow"), [out, source, 40], (1, 1, 1), (0,))
    assert not written_apart(_rows().plan(), [out, source, 40.0], (1, 1, 1), (0,))
    assert not written_apart(_rows().plan(), [out, source], (1, 1, 1), (0,))
    assert not written_apart(_rows().plan(), [out, source, 40], (0, 1, 1), (0,))
