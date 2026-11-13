"""Isolated i32 uniform-expression proof. Does not enable any compiler route."""

from dataclasses import dataclass


class Refusal(ValueError):
    pass


@dataclass(frozen=True)
class Expr:
    op: str
    value: int | None = None
    children: tuple = ()
    width: int = 32

    def __post_init__(self):
        if self.width != 32 or type(self.children) is not tuple:
            raise Refusal("invalid expression representation")
        if self.op in ("arg", "const"):
            if (
                type(self.value) is not int
                or self.value < 0
                or self.children
                or (self.op == "const" and self.value >= 1 << 32)
            ):
                raise Refusal("invalid leaf")
        elif self.op in ("add", "mul"):
            if self.value is not None or len(self.children) != 2 or any(type(c) is not Expr for c in self.children):
                raise Refusal("invalid arithmetic node")
        else:
            raise Refusal("unknown expression operation")

    def bits(self, arguments):
        if self.op == "arg":
            value = arguments[self.value]
            if type(value) is not int or not -(1 << 31) <= value < (1 << 32):
                raise Refusal("argument outside i32 bit representation")
            return value & 0xFFFFFFFF
        if self.op == "const":
            return self.value
        a, b = (c.bits(arguments) for c in self.children)
        if self.op == "mul":
            return (a * b) & 0xFFFFFFFF
        if self.op == "add":
            return (a + b) & 0xFFFFFFFF
        raise Refusal("unknown expression operation")

    def msl(self, names):
        if self.op == "arg":
            name = names[self.value]
            if not name.isidentifier():
                raise Refusal("unsafe scalar name")
            return "(uint)" + name
        if self.op == "const":
            return str(self.value) + "u"
        operator = {"mul": "*", "add": "+"}.get(self.op)
        if operator is None:
            raise Refusal("unknown expression operation")
        return "(" + self.children[0].msl(names) + " " + operator + " " + self.children[1].msl(names) + ")"


def walk(ops):
    for op in ops:
        yield op
        yield from walk(op.region_ops or [])
        yield from walk(op.else_ops or [])


class UniformProof:
    """Use the installed ownership checker and a deliberately closed grammar.

    No cache across calls: a changed graph must be proved again. The returned tree
    is immutable and contains argument slots, never the current argument values.
    This proves uniform value semantics only, not coordinate/axis/extent validity.
    """

    def __init__(self, lowerer):
        self.lowerer = lowerer
        self.ops = {}
        for op in walk(lowerer.graph.ops):
            for value in op.result_ids or [op.id]:
                if value in self.ops:
                    raise Refusal("duplicate result owner")
                self.ops[value] = op
        self.args = {a.id: a for a in lowerer.graph.args}
        if len(self.args) != len(lowerer.graph.args) or self.args.keys() & self.ops.keys():
            raise Refusal("ambiguous argument owner")

    def prove(self, value):
        # Do not reuse either our graph index or GenericLowerer's ownership cache
        # across requests; callers can replace graph nodes, not just their fields.
        current = type(self.lowerer)(self.lowerer.graph, self.lowerer.options)
        return UniformProof(current)._prove(value)

    def _prove(self, value):
        budget = [128]
        active = set()

        def visit(vid, depth):
            budget[0] -= 1
            if depth > 32 or budget[0] < 0 or vid in active:
                raise Refusal("bounded proof exhausted or cyclic")
            facts = self.lowerer._native_value_facts(vid, op_name="uniform stride proof")
            if facts.kind != "integer" or facts.elem != "i32" or facts.width != 32:
                raise Refusal("exact i32 required")
            if type(facts.shape) is not tuple or any(type(d) is not int or d <= 0 for d in facts.shape):
                raise Refusal("concrete shape required")
            active.add(vid)
            try:
                arg = self.args.get(vid)
                if arg is not None:
                    if arg.is_ptr or arg.elem_type != "i32" or facts.is_tensor or facts.shape != ():
                        raise Refusal("exact scalar i32 argument required")
                    return Expr("arg", arg.index)
                op = self.ops.get(vid)
                if op is None or op.id != vid or (op.result_ids and op.result_ids != [vid]):
                    raise Refusal("one owned result required")
                operands = op.operand_ids or []
                attrs = dict(op.attrs or {})
                # Walker bookkeeping is not an arithmetic flag. Bind its owner
                # before excluding only these two known structural fields.
                if "_block_id" in attrs and attrs["_block_id"] != self.lowerer.graph.result_meta[vid].owner_id:
                    raise Refusal("walker block contradicts native owner")
                if "_walk_order" in attrs and (type(attrs["_walk_order"]) is not int or attrs["_walk_order"] < 0):
                    raise Refusal("invalid walker order")
                attrs.pop("_block_id", None)
                attrs.pop("_walk_order", None)
                if op.op == "arith.constant":
                    value = attrs.get("value")
                    if (
                        operands
                        or set(attrs) != {"value"}
                        or type(value) is not int
                        or not -(1 << 31) <= value < (1 << 32)
                    ):
                        raise Refusal("exact i32 uniform constant required")
                    return Expr("const", value & 0xFFFFFFFF)
                if op.op in ("arith.addi", "arith.muli"):
                    if len(operands) != 2 or attrs:
                        raise Refusal("plain binary wrapping arithmetic required")
                    for operand in operands:
                        other = self.lowerer._native_value_facts(operand, op_name="uniform stride proof")
                        if other.shape != facts.shape or other.is_tensor != facts.is_tensor:
                            raise Refusal("arithmetic shape mismatch")
                    return Expr(
                        "add" if op.op == "arith.addi" else "mul", children=tuple(visit(v, depth + 1) for v in operands)
                    )
                if op.op in ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout"):
                    if len(operands) != 1:
                        raise Refusal("unary uniform transport required")
                    source = self.lowerer._native_value_facts(operands[0], op_name="uniform stride proof")
                    if source.kind != "integer" or source.elem != "i32" or source.width != 32:
                        raise Refusal("transport element changed")
                    a, b = source.shape, facts.shape
                    if op.op == "tt.splat":
                        valid = a == () and not source.is_tensor and facts.is_tensor and not attrs
                    elif op.op == "ttg.convert_layout":
                        valid = a == b and source.is_tensor == facts.is_tensor and not attrs
                    elif op.op == "tt.broadcast":
                        valid = (
                            source.is_tensor
                            and facts.is_tensor
                            and len(a) == len(b)
                            and all(x == y or x == 1 for x, y in zip(a, b))
                            and not attrs
                        )
                    else:
                        axis = attrs.get("axis")
                        valid = (
                            type(axis) is int
                            and source.is_tensor
                            and facts.is_tensor
                            and 0 <= axis <= len(a)
                            and b == a[:axis] + (1,) + a[axis:]
                            and set(attrs) == {"axis"}
                        )
                    if not valid:
                        raise Refusal("unproved uniform transport")
                    return visit(operands[0], depth + 1)
                raise Refusal("unsupported uniform operation: " + op.op)
            finally:
                active.remove(vid)

        return visit(value, 0)


def address_bits(links, arguments):
    """Arithmetic model only: independent i32 addptr links, pointer modulus64.

    Each link is the COMPLETE source offset expression, not an already widened
    coefficient or an arbitrarily factored sum. No allocation/bounds claim.
    """
    address = 0
    for link in links:
        bits = link.bits(arguments)
        signed = bits if bits < 0x80000000 else bits - (1 << 32)
        address = (address + signed) & 0xFFFFFFFFFFFFFFFF
    return address


def address_msl(links, names):
    if not links:
        raise Refusal("empty address path")
    return "as_type<long>(" + " + ".join("(ulong)as_type<int>(" + e.msl(names) + ")" for e in links) + ")"
