"""Capture precision from the same native module that produces the proof graph."""

import hashlib


class Refusal(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise Refusal(message)


def capture(module, ir, walk_ttgir, options, semantic, Lowerer):
    before = str(module)
    graph = walk_ttgir(module, options)
    native = []

    def visit(op):
        if op.get_name() == "tt.dot":
            need(op.get_num_results() == 1 and op.get_num_operands() == 3, "native dot arity")
            native.append(
                {
                    "id": op.get_result(0).id(),
                    "operands": [op.get_operand(i).id() for i in range(3)],
                    "precision": op.get_int_attr("inputPrecision"),
                    "max_imprecise": op.get_int_attr("maxNumImpreciseAcc"),
                }
            )

    module.walk(visit)
    enum = int(ir.INPUT_PRECISION.IEEE)
    need(type(enum) is int, "native IEEE enum missing")
    dots = [o for o in semantic.address.uniform.walk(graph.ops) if o.op == "tt.dot"]
    need(len(native) == len(dots) == 2 and len({r["id"] for r in native}) == 2, "native dot population")
    for r, op in zip(native, dots):
        need(r["id"] == op.id and r["operands"] == op.operand_ids, "native graph correspondence")
        need(type(r["precision"]) is int and r["precision"] == enum, "native precision is not IEEE")
        need(type(r["max_imprecise"]) is int and r["max_imprecise"] == 0, "native imprecise accumulation")
    _, program = semantic.prove(Lowerer(graph, options))
    need(str(module) == before, "native module changed during capture")
    return program, {
        "native": native,
        "ieee_enum": enum,
        "module_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "source": "native get_int_attr on module used by walk_ttgir",
    }
