"""Address witnesses for template replacements.

A template replaces a recognised source kernel wholesale, so every address it computes must be
the address the source computes. The detectors recover each pointer's strides by their POSITION
in the address chain, which says nothing about the index each stride multiplies: a row index
shifted by one, a doubled loop advance, swapped batch coordinates and an equivalent respelling
all yield a stride list, and the template then runs its own canonical formula on it.

This module evaluates the source's own load/store addresses exactly, on concrete integers, and
compares them with the template's formula. Every address in the admitted vocabulary is a
polynomial in a handful of atoms (the integer arguments, the program ids, a program id divided
by a batch divisor, the loop iteration) with the tile's own ``arange`` enumerated element by
element, so agreement on independent random atoms is a polynomial identity test: two different
polynomials of this degree agree on a random point with probability below 2**-15, and the test
uses three. Anything outside that vocabulary (a compare, a select, a min/max, a loaded index, a
conditional access, a narrowing cast) is refused by name rather than evaluated.

The witness also BINDS: each template stride is chosen by value from the source's own strides,
so an equivalent spelling the positional extraction misreads is bound to the stride it uses
rather than refused or, worse, run on the wrong one.
"""

import random

from triton_msl.codegen.mlir_walker import _extract_shape

_TRANSPARENT = frozenset(
    {
        "arith.index_cast",
        "arith.index_castui",
        "arith.extsi",
        "arith.extui",
        "builtin.unrealized_conversion_cast",
        "ttg.convert_layout",
    }
)
_PID_OPS = frozenset({"tt.get_program_id", "tt.program_id"})
_INT_TYPES = frozenset({"i32", "i64"})
_LOW, _HIGH = 1 << 20, 1 << 40
_SEED = 0x7A17F0A4


class WitnessRefusal(Exception):
    """The source address is outside the admitted vocabulary or differs from the template's."""


class _Tensor:
    __slots__ = ("shape", "data")

    def __init__(self, shape, data):
        self.shape, self.data = tuple(shape), data


class _Ptr:
    __slots__ = ("base", "off")

    def __init__(self, base, off):
        self.base, self.off = base, off


def _size(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _zip(f, a, b):
    ta, tb = isinstance(a, _Tensor), isinstance(b, _Tensor)
    if ta and tb:
        if a.shape != b.shape:
            raise WitnessRefusal("an address combines tensors of different shapes")
        return _Tensor(a.shape, [f(x, y) for x, y in zip(a.data, b.data)])
    if ta:
        return _Tensor(a.shape, [f(x, b) for x in a.data])
    if tb:
        return _Tensor(b.shape, [f(a, y) for y in b.data])
    return f(a, b)


def _add(a, b):
    return _zip(lambda x, y: x + y, a, b)


def _mul(a, b):
    return _zip(lambda x, y: x * y, a, b)


def _broadcast(a, shape):
    shape = tuple(shape)
    if not isinstance(a, _Tensor):
        return _Tensor(shape, [a] * _size(shape))
    if a.shape == shape:
        return a
    if len(a.shape) != len(shape) or any(s not in (1, t) for s, t in zip(a.shape, shape)):
        raise WitnessRefusal("an address broadcast that is not a plain size-1 expansion")
    if len(shape) == 2 and a.shape == (shape[0], 1):
        return _Tensor(shape, [x for x in a.data for _ in range(shape[1])])
    if len(shape) == 2 and a.shape == (1, shape[1]):
        return _Tensor(shape, a.data * shape[0])
    out, index = [], [0] * len(shape)
    for _ in range(_size(shape)):
        src = 0
        for s, i in zip(a.shape, index):
            src = src * s + (i if s != 1 else 0)
        out.append(a.data[src])
        for axis in range(len(shape) - 1, -1, -1):
            index[axis] += 1
            if index[axis] < shape[axis]:
                break
            index[axis] = 0
    return _Tensor(shape, out)


class _Env:
    """One assignment of the atoms. ``zero`` pins every coordinate atom to 0 (``ones`` to 1),
    which isolates one template stride at a time; the scalars stay random and are shared."""

    def __init__(self, rng, scalars, has_loop, zero=False, ones=()):
        self.memo = {}
        self._rng, self._scalars, self._has_loop = rng, scalars, has_loop
        self._zero, self._ones, self._atoms = zero, frozenset(ones), {}

    def _draw(self, key):
        if key not in self._scalars:
            self._scalars[key] = self._rng.randrange(_LOW, _HIGH)
        return self._scalars[key]

    def scalar(self, index):
        return self._draw(("arg", index))

    def atom(self, key):
        if key == ("t",) and not self._has_loop:
            return 0  # a loop specialised away runs its single iteration at offset zero
        if key == ("trip",):
            return self._draw(key)
        if key not in self._atoms:
            if key in self._ones:
                self._atoms[key] = 1
            elif self._zero:
                self._atoms[key] = 0
            else:
                self._atoms[key] = self._rng.randrange(_LOW, _HIGH)
        return self._atoms[key]


def stride_value(expr, env):
    """The value of a detector stride expression (argument index, c1/c0, literal, product)."""
    if expr == "c1":
        return 1
    if expr == "c0":
        return 0
    if type(expr) is int:
        return env.scalar(expr)
    if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "literal" and type(expr[1]) is int:
        return expr[1]
    if isinstance(expr, tuple) and len(expr) == 3 and expr[0] == "mul":
        return stride_value(expr[1], env) * stride_value(expr[2], env)
    raise WitnessRefusal("a stride expression the template cannot bind")


class AddressWitness:
    """Exact evaluation of every load/store address of a walked kernel graph."""

    def __init__(self, graph, tables=()):
        self.args = {a.id: a for a in graph.args}
        self.tables = frozenset(tables)
        self.ops, self.loops, self.accesses = {}, [], []
        self._collect(graph.ops, False)
        self.iv, self.carried, self.results, self.malformed = {}, {}, {}, None
        for loop in self.loops:
            ids = list(loop.operand_ids or [])
            bids = list((loop.attrs or {}).get("block_arg_ids") or [])
            inits = ids[3:]
            results = list(loop.result_ids or ([loop.id] if inits else []))
            if len(ids) < 3 or len(bids) != len(inits) + 1 or len(results) != len(inits):
                self.malformed = "a loop whose carried values could not be matched to its results"
                continue
            self.iv[bids[0]] = loop
            for i, bid in enumerate(bids[1:]):
                self.carried[bid] = (loop, i)
            for i, rid in enumerate(results):
                self.results[rid] = (loop, i)

    def _collect(self, ops, guarded):
        for op in ops:
            self.ops[op.id] = op
            if op.op in ("tt.load", "tt.store"):
                self.accesses.append((op, "load" if op.op == "tt.load" else "store", guarded))
            if op.op == "scf.for":
                self.loops.append(op)
                self._collect(op.region_ops or [], guarded)
            else:
                self._collect(op.region_ops or [], True)
                self._collect(op.else_ops or [], True)

    def env(self, rng, scalars, **kw):
        return _Env(rng, scalars, bool(self.loops), **kw)

    def _strip(self, vid):
        seen = set()
        while vid not in seen and vid not in self.results:
            seen.add(vid)
            op = self.ops.get(vid)
            if op is None or op.op not in _TRANSPARENT or len(op.operand_ids or []) != 1:
                break
            vid = op.operand_ids[0]
        return vid

    def bounds(self, loop, env):
        values = [self.value(v, env) for v in loop.operand_ids[:3]]
        if any(type(v) is not int for v in values):
            raise WitnessRefusal("a loop bound that is not an integer scalar")
        return values

    def _loop_dependent(self, vid, loop, seen):
        if vid in seen:
            return False
        seen.add(vid)
        if self.iv.get(vid) is loop or (vid in self.carried and self.carried[vid][0] is loop):
            return True
        op = self.ops.get(vid)
        if op is None or vid in self.results:
            return False
        return any(self._loop_dependent(o, loop, seen) for o in (op.operand_ids or []))

    def _increment(self, loop, slot, env):
        bid = loop.attrs["block_arg_ids"][slot + 1]
        yields = [o for o in (loop.region_ops or []) if o.op == "scf.yield"]
        if len(yields) != 1 or len(yields[0].operand_ids or []) <= slot:
            raise WitnessRefusal("a loop without a single terminating yield")
        yid = self._strip(yields[0].operand_ids[slot])
        if yid == bid:
            return 0
        step = None
        op = self.ops.get(yid)
        if op is not None and len(op.operand_ids or []) == 2:
            first, second = (self._strip(v) for v in op.operand_ids)
            if op.op == "tt.addptr" and first == bid:
                step = op.operand_ids[1]
            elif op.op == "arith.addi" and bid in (first, second):
                step = op.operand_ids[0] if second == bid else op.operand_ids[1]
        if step is None or self._loop_dependent(step, loop, set()):
            raise WitnessRefusal("a loop-carried address that is not advanced by one fixed step per iteration")
        return self.value(step, env)

    def _carried(self, loop, slot, env, iteration):
        init = self.value(loop.operand_ids[3 + slot], env)
        delta = _mul(self._increment(loop, slot, env), iteration)
        if isinstance(delta, _Ptr):
            raise WitnessRefusal("a pointer used as a loop step")
        if isinstance(init, _Ptr):
            return _Ptr(init.base, _add(init.off, delta))
        return _add(init, delta)

    def value(self, vid, env):
        if vid in env.memo:
            value = env.memo[vid]
            if value is None:
                raise WitnessRefusal("a cyclic address")
            return value
        env.memo[vid] = None
        value = env.memo[vid] = self._value(vid, env)
        return value

    def _value(self, vid, env):
        arg = self.args.get(vid)
        if arg is not None:
            if arg.is_ptr:
                return _Ptr(arg.index, 0)
            if str(arg.type_str) not in _INT_TYPES:
                raise WitnessRefusal(f"the non-integer argument {arg.name!r} in an address")
            return env.scalar(arg.index)
        if vid in self.iv:
            lower, _upper, step = self.bounds(self.iv[vid], env)
            return lower + env.atom(("t",)) * step
        if vid in self.carried:
            return self._carried(*self.carried[vid], env, env.atom(("t",)))
        if vid in self.results:
            return self._carried(*self.results[vid], env, env.atom(("trip",)))
        op = self.ops.get(vid)
        if op is None:
            raise WitnessRefusal("an address operand with no defining operation")
        name, ids = op.op, list(op.operand_ids or [])
        if name == "arith.constant":
            constant = (op.attrs or {}).get("value")
            if type(constant) is not int:
                raise WitnessRefusal("a non-integer constant in an address")
            shape = _extract_shape(op.type_str or "")
            return _broadcast(constant, shape) if shape else constant
        if name == "tt.make_range":
            start, end = (op.attrs or {}).get("start"), (op.attrs or {}).get("end")
            if type(start) is not int or type(end) is not int or end <= start:
                raise WitnessRefusal("an arange without literal bounds")
            return _Tensor((end - start,), list(range(start, end)))
        if name in _PID_OPS:
            axis = (op.attrs or {}).get("axis")
            if axis not in (0, 1, 2):
                raise WitnessRefusal("a program id without a grid axis")
            return env.atom(("pid", axis))
        if name in _TRANSPARENT and len(ids) == 1:
            return self.value(ids[0], env)
        if name == "tt.bitcast" and len(ids) == 1:
            inner = self.value(ids[0], env)
            if not isinstance(inner, _Ptr):
                raise WitnessRefusal("a bitcast of an index in an address")
            return inner
        if name in ("tt.splat", "tt.broadcast", "tt.expand_dims") and len(ids) == 1:
            inner = self.value(ids[0], env)
            off = inner.off if isinstance(inner, _Ptr) else inner
            if name == "tt.expand_dims":
                axis = (op.attrs or {}).get("axis")
                if not isinstance(off, _Tensor) or type(axis) is not int or not 0 <= axis <= len(off.shape):
                    raise WitnessRefusal("an expand_dims the address evaluation cannot follow")
                off = _Tensor(off.shape[:axis] + (1,) + off.shape[axis:], off.data)
            else:
                shape = _extract_shape(op.type_str or "")
                if not shape or (name == "tt.splat" and isinstance(off, _Tensor)):
                    raise WitnessRefusal(f"a {name} without a tensor result in an address")
                off = _broadcast(off, shape)
            return _Ptr(inner.base, off) if isinstance(inner, _Ptr) else off
        if name == "tt.load":
            # An index table (cu_seqlens): one unmasked scalar read. Its value is an atom named
            # by the table and the index read, so the template has to read the same entry.
            pointer = self.value(ids[0], env) if len(ids) == 1 else None
            if not isinstance(pointer, _Ptr) or pointer.base not in self.tables or type(pointer.off) is not int:
                raise WitnessRefusal("a loaded value in an address that is not one unmasked read of an index table")
            return env.atom(("load", pointer.base, pointer.off))
        if name in ("arith.addi", "arith.subi", "arith.muli") and len(ids) == 2:
            left, right = self.value(ids[0], env), self.value(ids[1], env)
            if isinstance(left, _Ptr) or isinstance(right, _Ptr):
                raise WitnessRefusal("integer arithmetic on a pointer")
            if name == "arith.muli":
                return _mul(left, right)
            return _add(left, right if name == "arith.addi" else _mul(right, -1))
        if name == "tt.addptr" and len(ids) == 2:
            pointer, offset = self.value(ids[0], env), self.value(ids[1], env)
            if not isinstance(pointer, _Ptr) or isinstance(offset, _Ptr):
                raise WitnessRefusal("a pointer advance that is not pointer plus index")
            return _Ptr(pointer.base, _add(pointer.off, offset))
        if name in ("arith.divsi", "arith.divui", "arith.remsi", "arith.remui") and len(ids) == 2:
            # Division is not polynomial. The one admitted form is a program id over a scalar
            # (the batch decode), which becomes an atom of its own that the template's formula
            # has to name as well.
            numerator = self.ops.get(self._strip(ids[0]))
            did = self._strip(ids[1])
            divisor, darg = self.ops.get(did), self.args.get(did)
            if numerator is None or numerator.op not in _PID_OPS:
                raise WitnessRefusal("a division in an address that is not a program id over a scalar")
            axis = (numerator.attrs or {}).get("axis")
            kind = "div" if name in ("arith.divsi", "arith.divui") else "rem"
            if darg is not None and not darg.is_ptr and str(darg.type_str) in _INT_TYPES:
                return env.atom((kind, axis, ("arg", darg.index)))
            if divisor is not None and divisor.op == "arith.constant":
                constant = (divisor.attrs or {}).get("value")
                if type(constant) is int and constant >= 1:
                    if constant == 1:
                        return env.atom(("pid", axis)) if kind == "div" else 0
                    return env.atom((kind, axis, ("const", constant)))
            raise WitnessRefusal("a program id divided by something other than an integer argument or constant")
        raise WitnessRefusal(f"{name} in an address")

    def offsets(self, env):
        """[(kind, base argument index, offset)] for every load and store, in graph order."""
        out = []
        for op, kind, guarded in self.accesses:
            if guarded:
                raise WitnessRefusal("a conditional load or store")
            pointer = self.value(op.operand_ids[0], env)
            if not isinstance(pointer, _Ptr):
                raise WitnessRefusal("a load or store whose address is not rooted at a pointer argument")
            out.append((kind, pointer.base, pointer.off))
        return out


def prove_template_addresses(graph, roles, loop):
    """Prove every source access is its role's template address; return the bound strides.

    ``roles`` is a list of dicts: ``name``, ``arg`` (pointer argument index), ``kind`` (``load``
    or ``store``), ``strides`` (the detector's positional list), ``shape`` (the access tile),
    ``scalars`` ([(slot, atom)]: template coordinates that are one integer per program; an
    atom may be ``fn(env) -> atom`` when it names a table entry) and ``axes`` ([(slot, tensor
    axis, fn(env) -> coordinates along that axis)]). The template address is the sum over both
    lists of coordinate times ``strides[slot]``. A role with ``table`` (``fn(env) -> the
    entries the template reads``) is an index table instead: every source read of it must be
    one unmasked scalar read of one of those entries.

    ``loop`` is ``None`` when the template's loop was specialised away, else ``(bound, step)``:
    the source must have exactly one loop, over ``range(0, bound, step)``.

    Returns ``{name: strides}`` with every used slot bound by value. Raises ``WitnessRefusal``.
    """
    tables = [role for role in roles if "table" in role]
    roles = [role for role in roles if "table" not in role]
    witness = AddressWitness(graph, [role["arg"] for role in tables])
    if witness.malformed:
        raise WitnessRefusal(witness.malformed)
    if len(witness.loops) != (0 if loop is None else 1):
        raise WitnessRefusal(
            f"{len(witness.loops)} source loops where the template runs {'none' if loop is None else 'exactly one'}"
        )
    by_arg = {role["arg"]: role for role in roles + tables}
    if len(by_arg) != len(roles) + len(tables):
        raise WitnessRefusal("two template roles on one pointer argument")
    for role in roles:
        if any(n < 2 for n in role["shape"]):
            raise WitnessRefusal(f"{role['name']}: a tile axis shorter than two elements")
    rng = random.Random(_SEED)
    pool = sorted(a.index for a in graph.args if not a.is_ptr and str(a.type_str) == "i32")

    def _grouped(env):
        grouped = {}
        for kind, base, off in witness.offsets(env):
            role = by_arg.get(base)
            if role is None:
                raise WitnessRefusal("a load or store on a pointer the template does not replay")
            if "table" in role:
                if kind != "load" or type(off) is not int or off not in role["table"](env):
                    raise WitnessRefusal(f"{role['name']}: a read of an entry the template does not read")
                continue
            if kind != role["kind"]:
                raise WitnessRefusal(f"{role['name']}: a {kind} where the template performs a {role['kind']}")
            if not isinstance(off, _Tensor) or off.shape != tuple(role["shape"]):
                raise WitnessRefusal(f"{role['name']}: an access that is not one {role['shape']} tile")
            grouped.setdefault(base, []).append(off)
        for role in roles:
            if role["arg"] not in grouped:
                raise WitnessRefusal(f"{role['name']}: no source {role['kind']}")
        return grouped

    # Isolate each stride: with every coordinate atom at zero the tile's first element is the
    # constant term, one step along an axis is that axis's stride, and raising one atom to 1
    # moves the first element by that atom's stride.
    scalars = {}
    zero = witness.env(rng, scalars, zero=True)
    base = _grouped(zero)

    def _key(atom, env):
        return atom(env) if callable(atom) else atom

    atoms = sorted({_key(atom, zero) for role in roles for _slot, atom in role["scalars"]}, key=repr)
    raised = {atom: _grouped(witness.env(rng, scalars, zero=True, ones=(atom,))) for atom in atoms}
    bound = {}
    for role in roles:
        strides = list(role["strides"])
        reference = base[role["arg"]][0]
        wanted = {}
        for slot, atom in role["scalars"]:
            wanted[slot] = raised[_key(atom, zero)][role["arg"]][0].data[0] - reference.data[0]
        for slot, axis, coords in role["axes"]:
            line = coords(zero)
            if len(line) != role["shape"][axis] or line[1] - line[0] != 1:
                raise WitnessRefusal(f"{role['name']}: a template coordinate that is not a unit-step tile")
            unit = _size(role["shape"][axis + 1 :])
            wanted[slot] = reference.data[unit] - reference.data[0]
        for slot, value in wanted.items():
            candidates = [strides[slot]] + [s for s in role["strides"]] + pool + ["c1"]
            chosen = next((c for c in candidates if stride_value(c, zero) == value), None)
            if chosen is None:
                raise WitnessRefusal(
                    f"{role['name']}: the source address has no stride argument where the template "
                    f"needs its slot-{slot} stride (an index is scaled, or its stride is not an argument)"
                )
            strides[slot] = chosen
        bound[role["name"]] = strides

    for _ in range(3):
        env = witness.env(rng, {})
        if loop is not None:
            lower, upper, step = witness.bounds(witness.loops[0], env)
            if (lower, upper, step) != (0, stride_value(loop[0], env), loop[1]):
                raise WitnessRefusal(f"a source loop that is not range(0, sequence length, {loop[1]})")
        grouped = _grouped(env)
        for role in roles:
            strides = bound[role["name"]]
            origin = sum(env.atom(_key(atom, env)) * stride_value(strides[slot], env) for slot, atom in role["scalars"])
            expected = _Tensor((), [origin])
            for slot, axis, coords in sorted(role["axes"], key=lambda term: term[1]):
                step = stride_value(strides[slot], env)
                shape = [1] * len(role["shape"])
                shape[axis] = role["shape"][axis]
                line = _Tensor(shape, [c * step for c in coords(env)])
                expected = (
                    _add(_broadcast(line, role["shape"]), expected.data[0])
                    if not expected.shape
                    else _add(expected, _broadcast(line, role["shape"]))
                )
            for off in grouped[role["arg"]]:
                if off.data != expected.data:
                    raise WitnessRefusal(
                        f"{role['name']}: the source {role['kind']} address is not the template's "
                        "(an index, a batch offset or a loop advance differs)"
                    )
    return bound
