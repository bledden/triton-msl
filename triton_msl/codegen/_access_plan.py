"""The memory accesses of a kernel, as expressions a launch can evaluate.

A source program runs each operation over every lane before the next operation starts, so
a store sees every earlier load of the same program complete. The ordinary lowering does
not keep that order between lanes: it carries one lane through the kernel body before the
same thread starts another, and threads do not wait for one another unless the shader says
so. What it keeps is each lane's own order. A launch whose result depends on the order of
two different lanes at one memory cell can therefore differ from the source.

Whether two lanes meet at a cell depends on the launch: which arguments share memory, the
scalar arguments, the grid. This module records, at compile time, where each load, store
and atomic points, as a small expression graph over the launch's values, and what the
emitted shader text shows about the order of its reads and stores. The launcher evaluates
it (``triton_msl.backend._lane_order``).

Everything here is a description. A value this module cannot describe becomes an opaque
node, which the launcher treats as unknown, never as absent.
"""

import re

from .mlir_walker import _extract_shape

PLAN_VERSION = 1

_BINARY = {
    "arith.addi": "add",
    "arith.subi": "sub",
    "arith.muli": "mul",
    "arith.divsi": "sdiv",
    "arith.divui": "udiv",
    "arith.remsi": "srem",
    "arith.remui": "urem",
    "arith.floordivsi": "fdiv",
    "arith.ceildivsi": "cdiv",
    "arith.andi": "and",
    "arith.ori": "or",
    "arith.xori": "xor",
    "arith.shli": "shl",
    "arith.shrsi": "ashr",
    "arith.shrui": "lshr",
    "arith.minsi": "smin",
    "arith.maxsi": "smax",
    "arith.minui": "umin",
    "arith.maxui": "umax",
}
_EXTENSION = {"arith.extsi": "sext", "arith.extui": "zext", "arith.trunci": "trunc"}
# The value of the result is the value of the first operand, lane for lane.
_TRANSPARENT = {"ttg.convert_layout"}
_MEMORY = {"tt.load": "L", "tt.store": "S", "tt.atomic_rmw": "A", "tt.atomic_cas": "A"}


def _walk(ops):
    for op in ops:
        yield op
        yield from _walk(op.region_ops or ())
        yield from _walk(op.else_ops or ())


class _Scope:
    """The values of one function body: its operations, loop arguments and known nodes."""

    def __init__(self, ops, values):
        self.values = dict(values)
        self.defs = {}
        self.loop_args = {}
        self.while_args = {}
        self.loops = {}
        self.used = set()
        for op in _walk(ops):
            self.used.update(op.operand_ids or ())
            self.defs.setdefault(op.id, (op, 0))
            for position, result in enumerate(op.result_ids or ()):
                self.defs.setdefault(result, (op, position))
            if op.op == "scf.for":
                for position, argument in enumerate(op.attrs.get("block_arg_ids") or ()):
                    self.loop_args[argument] = (op, position)
            if op.op == "scf.while":
                for position, argument in enumerate(op.attrs.get("block_arg_ids") or ()):
                    self.while_args[argument] = (op, False, position)
                for position, argument in enumerate(op.attrs.get("else_block_arg_ids") or ()):
                    self.while_args[argument] = (op, True, position)


class _Builder:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = []
        self.index = {}
        self.depends = []  # per node: the loops whose trip it follows, None when opaque
        self.loops = []
        self.accesses = []
        self.opaque_count = 0
        self.functions = {function.name: function for function in graph.called_funcs or ()}
        self.types = {}
        for holder in (graph, *self.functions.values()):
            for value_id, meta in (getattr(holder, "result_meta", None) or {}).items():
                self.types[value_id] = meta.type.raw
            for argument in holder.args:
                self.types.setdefault(argument.id, argument.type_str)
        self.depth = 0
        self.sites = []
        self.site = None
        self.atomic = {}  # access: [operation, is its result used]
        self.pairs = []  # the two updates one floating-point atomic minimum or maximum is made of
        self.last_atomic = None
        self.under_branch = 0
        self.branched = []  # the accesses a branch may leave out

    # -- types ---------------------------------------------------------------------------

    def _type(self, scope, value_id):
        text = self.types.get(value_id)
        if text is None and value_id in scope.defs:
            op, position = scope.defs[value_id]
            text = op.type_str if position == 0 else None
        return text or ""

    def _width(self, scope, value_id):
        text = self._type(scope, value_id)
        if "tt.ptr" in text:
            return None
        found = re.search(r"(?:^|[<x])i(\d+)\b", text)
        return int(found.group(1)) if found else None

    # -- nodes ---------------------------------------------------------------------------

    def node(self, *fields, loops=frozenset(), opaque=False):
        found = self.index.get(fields)
        if found is None:
            found = len(self.nodes)
            self.index[fields] = found
            self.nodes.append(list(fields))
            self.depends.append(None if opaque else frozenset(loops))
        return found

    def opaque(self):
        self.opaque_count += 1
        return self.node("o", self.opaque_count, opaque=True)

    def apply(self, *fields):
        """A node over other nodes; the trailing integer fields named in ``children``."""
        children = [field for field in fields[1:] if isinstance(field, _Child)]
        plain = tuple(field.index if isinstance(field, _Child) else field for field in fields)
        loops = frozenset()
        for child in children:
            if self.depends[child.index] is None:
                return self.node(*plain, opaque=True)
            loops |= self.depends[child.index]
        return self.node(*plain, loops=loops)

    def binary(self, name, width, left, right):
        return self.apply("b", name, width, _Child(left), _Child(right))

    # -- values --------------------------------------------------------------------------

    def value(self, scope, value_id):
        found = scope.values.get(value_id)
        if found is None:
            found = scope.values[value_id] = self._value(scope, value_id)
        return found

    def _value(self, scope, value_id):
        if value_id in scope.loop_args:
            op, position = scope.loop_args[value_id]
            loop = self.loop(scope, op)
            if position == 0:
                lower, _, step, width = self.loops[loop]
                count = self.node("it", loop, loops=frozenset((loop,)))
                return self.binary("add", width, lower, self.binary("mul", width, count, step))
            return self.carried(scope, op, position - 1, final=False)
        if value_id in scope.while_args:
            op, after, position = scope.while_args[value_id]
            return self.forwarded(scope, op, position) if after else self.entering(scope, op, position)
        if value_id not in scope.defs:
            return self.opaque()
        op, position = scope.defs[value_id]
        if op.op == "scf.for":
            return self.carried(scope, op, position, final=True)
        if op.op == "scf.while":
            return self.forwarded(scope, op, position)
        if op.op == "scf.if":
            return self.branch(scope, op, position)
        if position:
            return self.opaque()
        name = op.op
        operands = op.operand_ids
        if name == "arith.constant":
            constant = op.attrs.get("value")
            if not isinstance(constant, (bool, int)) or self._width(scope, value_id) is None:
                return self.opaque()
            shape = _extract_shape(self._type(scope, value_id))
            scalar = self.node("c", int(constant))
            return self.apply("bc", tuple(shape), _Child(scalar)) if shape else scalar
        if name in _TRANSPARENT and operands:
            return self.value(scope, operands[0])
        if name == "tt.bitcast" and len(operands) == 1:
            # A pointer seen as a pointer to elements of the same size is the same address
            # counted in the same elements (a boolean buffer stored as bytes).
            pointees = [re.search(r"tt\.ptr<(\w+)", self._type(scope, one)) for one in (operands[0], value_id)]
            if all(pointees) and _bytes(pointees[0].group(1)) == _bytes(pointees[1].group(1)) != 0:
                return self.value(scope, operands[0])
            return self.opaque()
        if name == "tt.get_program_id":
            return self.node("pid", int(op.attrs.get("axis", 0)))
        if name == "tt.get_num_programs":
            return self.node("np", int(op.attrs.get("axis", 0)))
        if name == "tt.make_range":
            return self.node("r", int(op.attrs["start"]), int(op.attrs["end"]))
        if name in _BINARY and len(operands) == 2:
            width = self._width(scope, value_id)
            if width is None:
                return self.opaque()
            return self.binary(_BINARY[name], width, self.value(scope, operands[0]), self.value(scope, operands[1]))
        if name in _EXTENSION and len(operands) == 1:
            source, target = self._width(scope, operands[0]), self._width(scope, value_id)
            if source is None or target is None:
                return self.opaque()
            return self.apply("x", _EXTENSION[name], source, target, _Child(self.value(scope, operands[0])))
        if name == "arith.cmpi" and len(operands) == 2:
            width = self._width(scope, operands[0])
            predicate = op.attrs.get("predicate_name")
            if width is None or not predicate:
                return self.opaque()
            left, right = (self.value(scope, operand) for operand in operands)
            return self.apply("cmp", str(predicate), width, _Child(left), _Child(right))
        if name == "arith.select" and len(operands) == 3:
            return self.apply("sel", *(_Child(self.value(scope, operand)) for operand in operands))
        if name in ("tt.broadcast", "tt.splat") and len(operands) == 1:
            shape = _extract_shape(self._type(scope, value_id))
            return self.apply("bc", tuple(shape), _Child(self.value(scope, operands[0])))
        if name == "tt.reshape" and len(operands) == 1:
            source = _extract_shape(self._type(scope, operands[0]))
            shape = _extract_shape(self._type(scope, value_id))
            return self.apply("rs", tuple(source), tuple(shape), _Child(self.value(scope, operands[0])))
        if name == "tt.expand_dims" and len(operands) == 1:
            return self.apply("xd", int(op.attrs.get("axis", 0)), _Child(self.value(scope, operands[0])))
        if name == "tt.addptr" and len(operands) == 2:
            if self._width(scope, operands[1]) is None:
                return self.opaque()
            return self.apply("ptr+", _Child(self.value(scope, operands[0])), _Child(self.value(scope, operands[1])))
        return self.opaque()

    def loop(self, scope, op):
        found = scope.loops.get(op.id)
        if found is None:
            lower, upper, step = (self.value(scope, operand) for operand in op.operand_ids[:3])
            width = self._width(scope, op.operand_ids[0]) or 32
            found = scope.loops[op.id] = len(self.loops)
            self.loops.append([lower, upper, step, width])
        return found

    def bases(self, node):
        """The pointer arguments a pointer node points into, or None when that is unknown."""
        entry = self.nodes[node]
        kind = entry[0]
        if kind == "p":
            return frozenset((entry[1],))
        if kind == "pu":
            return frozenset(entry[1])
        if kind == "ptr+":
            return self.bases(entry[1])
        if kind in ("bc", "xd"):
            return self.bases(entry[2])
        if kind == "rs":
            return self.bases(entry[3])
        if kind == "sel":
            one, other = self.bases(entry[2]), self.bases(entry[3])
            return None if one is None or other is None else one | other
        return None

    def flowing(self, scope, argument, sources):
        """A pointer carried round a loop: somewhere in the buffers its sources point into.

        ``sources`` are the values the loop argument takes (its start and what each trip
        hands on). Which element it points at is not described; which buffers it can
        point into is, so a launch still knows the memory an access through it can touch.
        """
        if not sources or "tt.ptr" not in self._type(scope, sources[0]):
            return self.opaque()
        known = set(scope.values)
        # The argument adds nothing to its own sources. Values worked out while it stands
        # for nothing are dropped again below.
        self.opaque_count += 1
        scope.values[argument] = self.node("pu", (), self.opaque_count, opaque=True)
        found = set()
        try:
            for source in sources:
                bases = self.bases(self.value(scope, source))
                if bases is None:
                    return self.opaque()
                found |= bases
        finally:
            for value_id in set(scope.values) - known:
                del scope.values[value_id]
        if not found:
            return self.opaque()
        self.opaque_count += 1
        return self.node("pu", tuple(sorted(found)), self.opaque_count, opaque=True)

    @staticmethod
    def _while_parts(op):
        conditions = [inner for inner in op.region_ops or () if inner.op == "scf.condition"]
        yields = [inner for inner in op.else_ops or () if inner.op == "scf.yield"]
        if len(conditions) != 1 or len(yields) != 1 or "else_block_arg_ids" not in (op.attrs or {}):
            return None
        return conditions[0], yields[0]

    def entering(self, scope, op, position):
        """An argument of a while loop's condition: its start, or what the last trip handed on."""
        parts = self._while_parts(op)
        arguments = op.attrs.get("block_arg_ids") or ()
        if parts is None or position >= len(arguments) or position >= len(op.operand_ids or ()):
            return self.opaque()
        if position >= len(parts[1].operand_ids or ()):
            return self.opaque()
        return self.flowing(scope, arguments[position], (op.operand_ids[position], parts[1].operand_ids[position]))

    def forwarded(self, scope, op, position):
        """What a while loop's condition hands to the body and, at the end, to the results."""
        parts = self._while_parts(op)
        if parts is None or 1 + position >= len(parts[0].operand_ids or ()):
            return self.opaque()
        return self.value(scope, parts[0].operand_ids[1 + position])

    def branch(self, scope, op, position):
        """A result of an if: one branch's value or the other's, by the condition."""
        thens = [inner for inner in op.region_ops or () if inner.op == "scf.yield"]
        elses = [inner for inner in op.else_ops or () if inner.op == "scf.yield"]
        if len(thens) != 1 or len(elses) != 1 or not op.operand_ids:
            return self.opaque()
        if position >= len(thens[0].operand_ids or ()) or position >= len(elses[0].operand_ids or ()):
            return self.opaque()
        return self.apply(
            "sel",
            _Child(self.value(scope, op.operand_ids[0])),
            _Child(self.value(scope, thens[0].operand_ids[position])),
            _Child(self.value(scope, elses[0].operand_ids[position])),
        )

    def carried(self, scope, op, position, final):
        """A loop-carried value: its start plus one fixed amount each trip, or a pointer
        somewhere in known buffers, or opaque."""
        found = self._advancing(scope, op, position, final)
        if found is not None:
            return found
        arguments = op.attrs.get("block_arg_ids") or ()
        yields = [inner for inner in op.region_ops or () if inner.op == "scf.yield"]
        if len(yields) != 1 or position + 1 >= len(arguments) or position >= len(yields[0].operand_ids or ()):
            return self.opaque()
        if 3 + position >= len(op.operand_ids):
            return self.opaque()
        return self.flowing(
            scope, arguments[position + 1], (op.operand_ids[3 + position], yields[0].operand_ids[position])
        )

    def _advancing(self, scope, op, position, final):
        """A loop-carried value that advances by one fixed amount each trip, or None."""
        loop = self.loop(scope, op)
        arguments = op.attrs.get("block_arg_ids") or ()
        yields = [inner for inner in op.region_ops or () if inner.op == "scf.yield"]
        if len(yields) != 1 or position + 1 >= len(arguments) or position >= len(yields[0].operand_ids):
            return None
        if 3 + position >= len(op.operand_ids):
            return None
        argument = arguments[position + 1]
        start = self.value(scope, op.operand_ids[3 + position])
        update = scope.defs.get(yields[0].operand_ids[position])
        if update is None or update[1] != 0:
            return None
        update = update[0]
        if update.op == "arith.addi" and argument in update.operand_ids and len(update.operand_ids) == 2:
            other = update.operand_ids[1] if update.operand_ids[0] == argument else update.operand_ids[0]
            pointer = False
        elif update.op == "tt.addptr" and len(update.operand_ids) == 2 and update.operand_ids[0] == argument:
            other = update.operand_ids[1]
            pointer = True
        else:
            return None
        if other == argument:
            return None
        amount = self.value(scope, other)
        width = self._width(scope, other)
        follows = self.depends[amount]
        if follows is None or loop in follows or width is None:
            return None
        count = self.node("trip", loop) if final else self.node("it", loop, loops=frozenset((loop,)))
        if final:
            # The trip count follows whatever the loop's bounds follow.
            bounds = frozenset()
            for bound in self.loops[loop][:3]:
                if self.depends[bound] is None:
                    return None
                bounds |= self.depends[bound]
            self.depends[count] = bounds
        if pointer:
            if width < 64:
                amount = self.apply("x", "sext", width, 64, _Child(amount))
            return self.apply("ptr+", _Child(start), _Child(self.binary("mul", 64, count, amount)))
        return self.binary("add", width, start, self.binary("mul", width, count, amount))

    # -- accesses ------------------------------------------------------------------------

    def _constant(self, scope, value_id):
        op = scope.defs.get(value_id, (None,))[0]
        return op.attrs.get("value") if op is not None and op.op == "arith.constant" else None

    def _float_pair(self, scope, first, second):
        """Are these the two updates of one floating-point atomic minimum or maximum?

        The frontend writes it over the value's bits: a signed update for the lanes whose
        value has a clear sign bit and an unsigned one, the other way round, for the
        rest. On values of those signs the two give the same cell in either order, so
        the pair is one update that commutes with itself.
        """
        signed, unsigned = (first, second) if first.attrs.get("rmw_op") in _SIGNED_WITH else (second, first)
        if _SIGNED_WITH.get(signed.attrs.get("rmw_op")) != unsigned.attrs.get("rmw_op"):
            return False
        if len(signed.operand_ids) != 3 or len(unsigned.operand_ids) != 3:
            return False
        if signed.operand_ids[:2] != unsigned.operand_ids[:2]:
            return False

        def same(value_id):
            """The value a change of layout was made from: the same elements."""
            for _ in range(8):
                op = scope.defs.get(value_id, (None,))[0]
                if op is None or op.op != "ttg.convert_layout" or not op.operand_ids:
                    break
                value_id = op.operand_ids[0]
            return value_id

        value = same(signed.operand_ids[1])
        if not re.search(r"(?:^|[x<])i32\b", self._type(scope, value)):
            return False

        def op_of(value_id):
            return scope.defs.get(same(value_id), (None,))[0]

        def sign_set(value_id):
            test = op_of(value_id)
            if test is None or test.op != "arith.cmpi" or test.attrs.get("predicate") != 1:
                return False
            shifted = op_of(test.operand_ids[0])
            if shifted is None or shifted.op != "arith.shrui" or self._constant(scope, test.operand_ids[1]) != 0:
                return False
            return same(shifted.operand_ids[0]) == value and self._constant(scope, shifted.operand_ids[1]) == 31

        def sign_clear(value_id):
            flipped = op_of(value_id)
            if flipped is None or flipped.op != "arith.xori":
                return False
            if not re.search(r"(?:^|[x<])i1\b", self._type(scope, flipped.id)):
                return False
            return self._constant(scope, flipped.operand_ids[1]) in (-1, 1, True) and sign_set(flipped.operand_ids[0])

        def implies(value_id, test):
            if test(value_id):
                return True
            both = op_of(value_id)
            return both is not None and both.op == "arith.andi" and any(test(operand) for operand in both.operand_ids)

        return implies(signed.operand_ids[2], sign_clear) and implies(unsigned.operand_ids[2], sign_set)

    def collect(self, scope, ops, loops, unbounded):
        for op in ops:
            kind = _MEMORY.get(op.op)
            if kind and op.operand_ids:
                mask = None
                if op.op == "tt.load" and len(op.operand_ids) > 1:
                    mask = op.operand_ids[1]
                if op.op in ("tt.store", "tt.atomic_rmw") and len(op.operand_ids) > 2:
                    mask = op.operand_ids[2]
                self.accesses.append(
                    [
                        kind,
                        self.value(scope, op.operand_ids[0]),
                        -1 if mask is None else self.value(scope, mask),
                        list(_extract_shape(self._type(scope, op.operand_ids[0]))),
                        list(loops),
                        bool(unbounded),
                    ]
                )
                # A called function is written once, in its own order, where it is called.
                self.sites.append(id(op) if self.site is None else self.site)
                if self.under_branch:
                    self.branched.append(len(self.accesses) - 1)
                if op.op == "tt.atomic_rmw":
                    ordinal = len(self.accesses) - 1
                    self.atomic[ordinal] = [str(op.attrs.get("rmw_op")), op.id in scope.used]
                    before = self.last_atomic
                    if before is not None and before[1] is scope and self._float_pair(scope, before[2], op):
                        self.pairs.append([before[0], ordinal])
                    self.last_atomic = (ordinal, scope, op)
            if op.op == "scf.for":
                self.collect(scope, op.region_ops or (), (*loops, self.loop(scope, op)), unbounded)
            elif op.op == "scf.if":
                self.under_branch += 1
                self.collect(scope, op.region_ops or (), loops, unbounded)
                self.collect(scope, op.else_ops or (), loops, unbounded)
                self.under_branch -= 1
            elif op.op == "tt.call":
                function = self.functions.get(op.attrs.get("callee"))
                if function is None or self.depth > 8 or len(function.args) != len(op.operand_ids):
                    raise _Unknown("a call this plan cannot follow")
                values = {
                    argument.id: self.value(scope, operand) for argument, operand in zip(function.args, op.operand_ids)
                }
                outer = self.site
                if outer is None:
                    self.site = id(op)
                self.depth += 1
                self.collect(_Scope(function.ops, values), function.ops, loops, unbounded)
                self.depth -= 1
                self.site = outer
            else:
                # Any other region (a while loop, a reduction's combiner) repeats an
                # unknown number of times.
                self.collect(scope, op.region_ops or (), loops, True)
                self.collect(scope, op.else_ops or (), loops, True)


_SIGNED_WITH = {"max": "umin", "min": "umax"}


def _bytes(elem_type):
    """Bytes of one element of a pointee type, 0 when not known."""
    found = re.fullmatch(r"(?:bf|[fiu])(\d+)", str(elem_type or ""))
    return 0 if not found else max(1, int(found.group(1)) // 8)


class _Child:
    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


class _Unknown(Exception):
    pass


def text_order(pointer_names, msl):
    """What the shader text shows about when each argument buffer is read and stored.

    Returns the positions of the barriers that every thread of a threadgroup passes
    together, each with the position where its enclosing block ends, and per buffer the
    positions of its last read, first store, last store and first read. A buffer maps to None when the text
    does not let that be told: it is used other than by indexing its own name, it is not
    declared volatile (a volatile read above a barrier is not moved below it), or it is
    updated in place by one statement.

    The text is matched, not parsed. A thread that returns early would not reach a
    barrier, so a shader with a return has no usable barrier at all.
    """
    msl = re.sub(r"//[^\n]*|/\*.*?\*/", lambda found: " " * len(found.group(0)), msl, flags=re.S)
    wanted = {found.start() for found in re.finditer(r"\bthreadgroup_barrier\(", msl)}
    stack, closes, inside = [], {}, {}
    for position, character in enumerate(msl):
        if character == "{":
            stack.append(position)
        elif character == "}":
            if not stack:
                return {"barriers": [], "buffers": {name: None for name in pointer_names}}
            closes[stack.pop()] = position
        elif position in wanted:
            inside[position] = tuple(stack)
    barriers = []
    if not stack and not re.search(r"\breturn\b", msl) and len(re.findall(r"\bkernel void\b", msl)) == 1:
        for position in sorted(inside):
            enclosing = inside[position]
            if len(enclosing) == 1:
                barriers.append([position, position])
            elif len(enclosing) == 2:
                # Inside one loop of the kernel body that every thread runs the same,
                # positive number of times.
                opened = enclosing[1]
                header = msl[max(0, opened - 160) : opened].rstrip()
                loop = re.search(r"for \(uint (\w+) = 0u; \1 < (\d+)u; \1 \+= (\d+)u\)$", header)
                if loop and int(loop.group(2)) > 0 and int(loop.group(3)) > 0:
                    body = msl[opened : closes[opened]]
                    changed = re.search(rf"\b{loop.group(1)}\s*(?:=(?!=)|\+=|-=|\+\+|--)", body)
                    if not changed and not re.search(r"\b(?:break|continue)\b", body):
                        barriers.append([position, closes[opened]])
    buffers = {}
    for name in pointer_names:
        escaped = re.escape(name)
        declared = [found.end() for found in re.finditer(rf"\bvolatile device \w+\* {escaped}(?= \[\[buffer\()", msl)]
        uses = [found.start() for found in re.finditer(rf"\b{escaped}\b", msl)]
        if len(declared) != 1 or not uses or uses[0] + len(name) != declared[0]:
            buffers[name] = None
            continue
        last_read, first_store, known = -1, len(msl), True
        last_store, first_read = -1, len(msl)
        for position in uses[1:]:
            before = msl[max(0, position - 16) : position].rstrip()
            opened = position + len(name)
            while opened < len(msl) and msl[opened] in " \t\n":
                opened += 1
            if before.endswith(("&", ".", "->")) or opened >= len(msl) or msl[opened] != "[":
                known = False
                break
            level, end = 0, None
            for cursor in range(opened, min(len(msl), opened + 4096)):
                if msl[cursor] == "[":
                    level += 1
                elif msl[cursor] == "]":
                    level -= 1
                    if level == 0:
                        end = cursor
                        break
            if end is None:
                known = False
                break
            after = msl[end + 1 : end + 33].lstrip()
            if re.match(r"(?:\+\+|--|[-+*/%&|^]=|<<=|>>=|\[|\.)", after) or before.endswith(("++", "--")):
                known = False
                break
            if after.startswith("=") and not after.startswith("=="):
                first_store = min(first_store, position)
                last_store = max(last_store, position)
            else:
                last_read = max(last_read, position)
                first_read = min(first_read, position)
        buffers[name] = [last_read, first_store, last_store, first_read] if known else None
    return {"barriers": barriers, "buffers": buffers}


def _emission_order(sites, emitted, names, body):
    """What the lowering's own record shows about the order accesses were written in.

    ``emitted`` lists (operation, first body line, one past its last body line) for each
    time the lowering wrote a memory operation; one operation can be written again where
    a later pass needs its value. ``body`` is the kernel body those line numbers index.

    Returns the accesses never written that way, the pairs written out of order, and the
    pointer arguments named by a body line no memory operation wrote. A pair is out of
    order when some writing of the earlier access comes after the first writing of the
    later one: a load written again below a store reads what the store left. A buffer
    named outside every memory operation is read or stored by something this record
    does not describe.
    """
    if emitted is None or body is None:
        return list(range(len(sites))), [], list(range(len(names)))
    emitted = sorted(emitted, key=lambda entry: entry[1])
    covered = [False] * len(body)
    where = {}
    for position, (site, start, end) in enumerate(emitted):
        if not 0 <= start < end <= len(body):
            return list(range(len(sites))), [], list(range(len(names)))
        for line in range(start, end):
            covered[line] = True
        found = where.get(site)
        where[site] = (position, position) if found is None else (found[0], position)
    untraced = [ordinal for ordinal, site in enumerate(sites) if site not in where]
    order = []
    for later, site in enumerate(sites):
        if site not in where:
            continue
        first = where[site][0]
        for earlier in range(later):
            other = sites[earlier]
            if other in where and other != site and where[other][1] > first:
                order.append([earlier, later])
    loose_text = "\n".join(line for line, inside in zip(body, covered) if not inside)
    loose = [position for position, name in enumerate(names) if re.search(rf"\b{re.escape(name)}\b", loose_text)]
    return untraced, order, loose


# Operations whose result is computed from their operands and nothing else.
_PURE = {
    "tt.splat",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.reshape",
    "tt.fp_to_fp",
    "tt.bitcast",
    "tt.extern_elementwise",
    "tt.clampf",
    "ttg.convert_layout",
}


def _single_lane(graph, sites, accesses, single, whole=True):
    """The accesses that only one lane of a program performs, as the lowering wrote them.

    ``single`` names the stores the lowering gave to one lane (a store of one element).
    A load of one element is written on every lane; when its value reaches nothing but
    such stores, through operations that only compute, the copies on the other lanes are
    dropped unseen and the load is that one lane's too.

    With ``whole`` false the stores are those of one value per row of a two-dimensional
    block, each given to its row's first lane. A load of the same shape is written on
    every lane of its row, and is the first lane's alone under the same condition.
    """
    single = set(single or ())
    # ``single`` also names the atomics over one element: the first lane alone performs them.
    stores = [ordinal for ordinal, site in enumerate(sites) if accesses[ordinal][0] in "SA" and site in single]
    if not stores:
        return [], []
    shapes = {tuple(accesses[ordinal][3]) for ordinal in stores}
    if not whole and len(shapes) != 1:
        # Rows of different blocks: which lane is a row's first is not one rule.
        return [], []
    uses, owner = {}, {}
    for op in _walk(graph.ops):
        owner[id(op)] = op
        for position, operand in enumerate(op.operand_ids or ()):
            uses.setdefault(operand, []).append((op, position))

    def dropped(value_id, seen):
        for op, position in uses.get(value_id, ()):
            if op.op == "tt.store":
                if id(op) not in single or position == 0:
                    return False
                continue
            if op.region_ops or op.else_ops or len(op.result_ids or ()) > 1:
                return False
            if not (op.op.startswith(("arith.", "math.")) or op.op in _PURE) or op.op == "arith.constant":
                return False
            if op.id not in seen:
                seen.add(op.id)
                if not dropped(op.id, seen):
                    return False
        return True

    quiet = []
    for ordinal, site in enumerate(sites):
        op = owner.get(site)
        if op is None or op.op != "tt.load" or accesses[ordinal][0] != "L":
            continue
        if whole and any(extent != 1 for extent in accesses[ordinal][3]):
            continue
        if not whole and tuple(accesses[ordinal][3]) not in shapes:
            continue
        if dropped(op.id, {op.id}):
            quiet.append(ordinal)
    return stores, quiet


# What a plan says about one emitted shader rather than about the source program.
_SHADER_FACTS = ("untraced", "order", "loose", "single", "quiet", "rows", "threads", "solo", "barriers", "text")


def source_only(plan):
    """``plan`` without what it says about the shader it was built beside.

    For a launch that may run another binary of the same source (the opt-in C++ route):
    the launcher then reads every access as untraced, so only answers that hold for any
    lowering remain.
    """
    if not isinstance(plan, dict):
        return plan
    return {key: value for key, value in plan.items() if key not in _SHADER_FACTS}


def build_access_plan(graph, msl=None, emitted=None, body=None, single=None, threads=None, rows=None):
    """The access plan of ``graph``, with the text order of ``msl`` when given.

    ``emitted`` is the lowering's record of where each memory operation landed in the
    kernel body ``body``; ``single`` the stores it guarded to one lane; ``rows`` the
    stores it guarded to the first lane of each row; ``threads`` the threadgroup width
    the shader is written for and whether a thread carries several lanes.
    """
    builder = _Builder(graph)
    scope = _Scope(graph.ops, {})
    for argument in graph.args:
        if argument.is_ptr:
            scope.values[argument.id] = builder.node("p", argument.index)
        else:
            width = builder._width(scope, argument.id)
            scope.values[argument.id] = builder.opaque() if width is None else builder.node("a", argument.index, width)
    plan = {"version": PLAN_VERSION, "why": None}
    try:
        builder.collect(scope, graph.ops, (), False)
    except _Unknown as unknown:
        plan["why"] = str(unknown)
    block = 1
    for holder in (graph, *builder.functions.values()):
        for op in _walk(holder.ops):
            size = 1
            for extent in _extract_shape(op.type_str or ""):
                size *= extent
            block = max(block, size)
    plan["block"] = block
    plan["nodes"] = builder.nodes
    plan["loops"] = builder.loops
    plan["accesses"] = builder.accesses
    plan["atomic"] = {str(ordinal): entry for ordinal, entry in builder.atomic.items()}
    plan["pairs"] = builder.pairs
    plan["branched"] = builder.branched
    pointers = [argument for argument in graph.args if argument.is_ptr]
    plan["untraced"], plan["order"], loose = _emission_order(
        builder.sites, emitted, [argument.name for argument in pointers], body
    )
    plan["loose"] = [pointers[position].index for position in loose]
    plan["single"], plan["quiet"] = _single_lane(graph, builder.sites, builder.accesses, single)
    # The accesses each row's first lane performs alone: its stores, then the loads.
    plan["rows"] = [
        ordinal
        for found in _single_lane(graph, builder.sites, builder.accesses, rows, whole=False)
        for ordinal in found
    ]
    # The threads of a threadgroup the shader is written for, and whether one thread
    # carries several lanes in turn.
    plan["threads"] = threads
    if len(plan["order"]) > 4096 and not plan["why"]:
        plan["why"] = "too many accesses written out of order"
    plan["pointers"] = {
        argument.index: [argument.name, _bytes(argument.elem_type)] for argument in graph.args if argument.is_ptr
    }
    if msl is not None:
        # Nothing but the buffers passes between the threads of a threadgroup.
        plan["solo"] = not re.search(r"threadgroup_barrier|\bthreadgroup\s", msl)
        order = text_order([argument.name for argument in graph.args if argument.is_ptr], msl)
        plan["barriers"] = order["barriers"]
        plan["text"] = {
            argument.index: order["buffers"].get(argument.name) for argument in graph.args if argument.is_ptr
        }
    return plan
