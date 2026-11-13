"""Finite row-local source-DAG renderer, not the production affine template.

Entry is only through native capture. Arrays represent one output row; K and V
remain indexed loads in the dot. No threadgroup storage or cross-lane reduction.
"""

import hashlib
import json
from dataclasses import asdict
from . import precision


def need(ok, msg):
    precision.need(ok, msg)


def integer(e, coords):
    op = e[0]
    if op == "arg":
        need(e[1] in (9, 10), "integer arg")
        return f"as_type<uint>(a{e[1]})"
    if op == "const":
        return f"{e[1] & 0xFFFFFFFF}u"
    if op == "lane":
        return coords[e[1]]
    if op == "pid":
        return ("pid.x", "pid.y", "pid.z")[e[1]]
    if op == "iv":
        return "iv"
    if op in ("add", "mul"):
        return "(" + integer(e[1], coords) + (" + " if op == "add" else " * ") + integer(e[2], coords) + ")"
    if op == "divsi":
        return (
            "(as_type<uint>(as_type<int>(" + integer(e[1], coords) + ") / as_type<int>(" + integer(e[2], coords) + ")))"
        )
    raise precision.Refusal("unknown integer expression")


def offset(record, coords):
    return "(" + " + ".join("(ulong)((long)as_type<int>(" + integer(e, coords) + "))" for e in record.links) + ")"


def admit(n, h, batch, grid, threads, capacities):
    need(all(type(x) is int for x in (n, h, batch)), "exact integer domain")
    need(n in (32, 64) and h in (1, 2) and batch in (1, 2), "prototype full-tile domain")
    need(grid == (n // 32, n, batch * h) and threads == (32, 1, 1), "exact finite launch geometry")
    nn = n * n
    needed = (
        batch * h * nn * 32,
        batch * h * nn,
        batch * h * nn * 32,
        batch * h * nn * 32,
        batch * h * nn * 32,
        batch * h * nn,
        batch * nn,
    )
    need(
        type(capacities) is tuple
        and len(capacities) == 7
        and all(type(x) is int and x >= v for x, v in zip(capacities, needed)),
        "buffer element capacities",
    )
    return needed


class Row:
    def __init__(self, program):
        self.p = program
        self.addresses = {x["record"].root: x["record"] for x in program["addresses"]}
        self.code = []
        self.cache = {}
        self.instructions = []
        self.count = 0
        self.storage = 0

    def emit(self, s):
        self.code.append(s)

    def prepare(self, e):
        if e in self.cache:
            return
        op, shape, *a = e
        if op in ("const", "arg", "load", "carry", "final", "byte-zero", "slt"):
            return
        if op in ("broadcast", "byte-ne", "and"):
            for x in a:
                if isinstance(x, tuple) and x and isinstance(x[0], str):
                    self.prepare(x)
            return
        need(
            op
            in (
                "arith.addf",
                "arith.subf",
                "arith.mulf",
                "arith.divf",
                "arith.maxnumf",
                "math.exp2",
                "math.log",
                "select",
                "dot",
                "reduce-arith.addf",
                "reduce-arith.maxnumf",
            ),
            "render expression grammar",
        )
        for x in a:
            if isinstance(x, tuple) and x and isinstance(x[0], str):
                self.prepare(x)
        need(shape in ((), (32,), (32, 32)), "row expression shape")
        name = f"t{self.count}"
        self.count += 1
        vector = shape == (32, 32)
        self.cache[e] = (name, vector)
        self.instructions.append({"name": name, "expression": e, "vector": vector})
        self.storage += 128 if vector else 4
        self.emit("float " + name + ("[32];" if vector else ";"))
        if vector:
            self.emit("for (uint c = 0u; c < 32u; ++c) {")
        coords = ("row", "c") if vector else ("row",)
        target = name + ("[c]" if vector else "")
        if op == "dot":
            need(a[3:] == ["ieee-source", 0], "proved dot policy")
            need(a[1][0] == "load" and a[1][2] in (3, 4), "dot RHS must be original K or V")
            self.emit(target + " = " + self.ref(a[2], coords) + ";")
            self.emit("for (uint k = 0u; k < 32u; ++k) {")
            self.emit("float product = " + self.ref(a[0], ("row", "k")) + " * " + self.ref(a[1], ("k", "c")) + ";")
            self.emit(target + " = " + target + " + product;")
            self.emit("}")
        elif op.startswith("reduce-"):
            self.emit(target + " = " + self.ref(a[0], ("row", "0u")) + ";")
            self.emit("for (uint k = 1u; k < 32u; ++k) {")
            x = self.ref(a[0], ("row", "k"))
            self.emit(
                target
                + " = "
                + (target + " + " + x if op == "reduce-arith.addf" else "fmax(" + target + ", " + x + ")")
                + ";"
            )
            self.emit("}")
        else:
            self.emit(target + " = " + self.operation(e, coords) + ";")
        if vector:
            self.emit("}")

    def ref(self, e, coords):
        if e in self.cache:
            name, vector = self.cache[e]
            return name + ("[" + coords[1] + "]" if vector else "")
        op, shape, *a = e
        if op == "const":
            return f"as_type<float>(0x{a[0]:08x}u)"
        if op == "arg":
            return f"a{a[0]}"
        if op in ("carry", "final"):
            return ("m", "l", "acc")[a[0]] + ("[" + coords[1] + "]" if a[0] == 2 else "")
        if op == "load":
            return f"b{a[0]}[" + offset(self.addresses[a[0]], coords) + "]"
        if op == "byte-zero":
            return "uchar(0)"
        if op == "broadcast":
            child, axes = a
            return self.ref(child, tuple("0u" if child[1][i] == 1 else coords[axis] for i, axis in enumerate(axes)))
        if op == "slt":
            return "(as_type<int>(" + integer(a[0], coords) + ") < as_type<int>(" + integer(a[1], coords) + "))"
        if op == "byte-ne":
            return "(" + self.ref(a[0], coords) + " != " + self.ref(a[1], coords) + ")"
        if op == "and":
            return "(" + self.ref(a[0], coords) + " && " + self.ref(a[1], coords) + ")"
        raise precision.Refusal("unmaterialized expression " + op)

    def operation(self, e, coords):
        op, shape, *a = e
        x = [self.ref(v, coords) for v in a]
        if op in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf"):
            return (
                x[0] + {"arith.addf": " + ", "arith.subf": " - ", "arith.mulf": " * ", "arith.divf": " / "}[op] + x[1]
            )
        if op == "arith.maxnumf":
            return "fmax(" + ", ".join(x) + ")"
        if op == "select":
            return "(" + x[0] + " ? " + x[1] + " : " + x[2] + ")"
        if op in ("math.exp2", "math.log"):
            return "precise::" + op.split(".")[1] + "(" + x[0] + ")"
        raise precision.Refusal("unknown floating operation")

    def generate(self, entry=None):
        import re

        if entry is None:
            entry = "trifast_expression_" + ("padded" if self.p["padded"] else "masked")
        need(
            type(entry) is str and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", entry) is not None,
            "invalid source-program entry symbol",
        )
        signature = []
        for i in range(7):
            type_ = "uchar" if i == 6 else "float"
            const = "" if i in (0, 1) else "const "
            signature.append(f"device {const}{type_}* b{i} [[buffer({i})]]")
        signature += [f"constant {'float' if i < 9 else 'int'}& a{i} [[buffer({i})]]" for i in range(7, 11)]
        signature += ["uint3 pid [[threadgroup_position_in_grid]]", "uint row [[thread_index_in_threadgroup]]"]
        self.emit("#include <metal_stdlib>\nusing namespace metal;")
        self.emit("// CPU prototype. Host must enforce the bound full-tile admission. Not device qualified.")
        self.emit("#pragma clang fp contract(off) reassociate(off)")
        self.emit("kernel void " + entry + "(" + ",\n".join(signature) + ") {")
        self.emit("if (row >= 32u) return;")
        self.emit("float m = as_type<float>(0xff800000u); float l = 0.0f; float acc[32];")
        self.emit("for (uint c=0u; c<32u; ++c) acc[c]=0.0f;")

        def dots(e):
            if not isinstance(e, tuple) or not e or not isinstance(e[0], str):
                return []
            return ([e] if e[0] == "dot" else []) + [d for a in e[2:] for d in dots(a)]

        score_dots = {e for root in self.p["recurrence"] for e in dots(root) if e[3][0] == "load" and e[3][2] == 3}
        need(len(score_dots) == 1, "single source score dot")
        self.prepare(next(iter(score_dots))[2])
        self.emit("for (uint iv=0u; iv<as_type<uint>(a9); iv+=32u) {")
        for e in self.p["recurrence"]:
            self.prepare(e)
        self.emit("m = " + self.ref(self.p["recurrence"][0], ("row",)) + ";")
        self.emit("l = " + self.ref(self.p["recurrence"][1], ("row",)) + ";")
        self.emit("for (uint c=0u; c<32u; ++c) acc[c] = " + self.ref(self.p["recurrence"][2], ("row", "c")) + ";")
        self.emit("}")
        self.cache = {}
        self.prepare(self.p["out"])
        self.prepare(self.p["lse"])
        if not self.p["padded"]:
            self.emit("if (as_type<int>(row + pid.x*32u) < a9) {")
        self.emit(
            "for (uint c=0u; c<32u; ++c) b0["
            + offset(self.addresses[0], ("row", "c"))
            + "] = "
            + self.ref(self.p["out"], ("row", "c"))
            + ";"
        )
        coords = ("row", "0u") if self.p["padded"] else ("row",)
        self.emit("b1[" + offset(self.addresses[1], coords) + "] = " + self.ref(self.p["lse"], coords) + ";")
        if not self.p["padded"]:
            self.emit("}")
        self.emit("}")
        code = "\n".join(self.code) + "\n"
        return code, {
            "instructions": self.instructions,
            "declared_private_bytes_upper_bound": self.storage + 144,
            "resource_scope": "sum of declared float arrays and scalars, including both dot products; excludes integer controls, ABI, compiler registers and spills; not measured peak",
            "threadgroup_bytes": 0,
            "threads": 32,
            "policy": "separate f32 mul/add, increasing-k reductions, precise namespace requested, full tiles only; no compiler or hardware qualification",
        }


def render(module, ir, walker, options, semantic, Lowerer):
    p, receipt = precision.capture(module, ir, walker, options, semantic, Lowerer)
    text, manifest = Row(p).generate()
    manifest["msl_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    manifest["precision"] = receipt
    return text, manifest, p
