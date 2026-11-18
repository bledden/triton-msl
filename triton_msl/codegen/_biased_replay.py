"""Bounded generic replay for biased FA without a saved logsumexp output.

This is not a replacement attention template and does not invent a single N_CTX.
Every source pointer, mask, loop bound and value operation goes through generic
lowering. Query tiles of 8/16/32 rows, 32-key score tiles, f16/bf16/f32
storage, and explicit f16 probability operands are covered here. Wider score
tiles, bf16 dot operands, dot-result scales, nested control and extra effects
remain outside this recovery.
"""


def eligible(lowerer):
    graph = lowerer.graph
    if graph.called_funcs:
        return False
    loops = [o for o in graph.ops if o.op == "scf.for"]
    if len(loops) != 1:
        return False
    loop = loops[0]
    if len(loop.operand_ids) != 6 or len(loop.result_ids or []) != 3:
        return False
    ops = list(graph.ops) + list(loop.region_ops or [])
    dots = [o for o in ops if o.op == "tt.dot"]
    if len(dots) != 2:
        return False
    qshape = lowerer._native_shape(dots[0].operand_ids[0], op_name="tt.dot")
    kshape = lowerer._native_shape(dots[0].operand_ids[1], op_name="tt.dot")
    if len(qshape) != 2 or len(kshape) != 2:
        return False
    bm, d = qshape
    dk, bn = kshape
    if bm not in (8, 16, 32) or bn != 32 or d not in (32, 64) or dk != d:
        return False
    # Reductions are the sole nested regions. Their generic classifier independently
    # checks the returned value; do not let a proof projection erase their bodies.
    reductions = [o for o in ops if o.op == "tt.reduce"]
    if len(reductions) != 2:
        return False
    for red in reductions:
        if (
            red.attrs.get("axis") != 1
            or len(red.operand_ids) != 1
            or lowerer._native_shape(red.operand_ids[0], op_name=red.op) != (bm, bn)
            or lowerer._native_shapes_for_op(red) != ((bm,),)
        ):
            return False
    if any(o.else_ops or (o.region_ops and o is not loop and o not in reductions) for o in ops):
        return False
    ops += [o for red in reductions for o in (red.region_ops or [])]
    allowed = {
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.remsi",
        "arith.divui",
        "arith.remui",
        "arith.minsi",
        "arith.minui",
        "arith.cmpi",
        "arith.andi",
        "arith.ori",
        "arith.select",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.maxnumf",
        "arith.maximumf",
        "arith.extf",
        "arith.truncf",
        "math.exp",
        "math.exp2",
        "tt.get_program_id",
        "tt.make_range",
        "tt.splat",
        "tt.expand_dims",
        "tt.broadcast",
        "tt.addptr",
        "tt.load",
        "tt.store",
        "tt.dot",
        "tt.trans",
        "tt.reduce",
        "tt.reduce.return",
        "ttg.local_alloc",
        "ttg.local_load",
        "ttg.memdesc_trans",
        "ttg.convert_layout",
        "scf.for",
        "scf.yield",
        "tt.return",
    }
    if any(o.op not in allowed for o in ops):
        return False
    if any(a.elem_type not in ("bf16", "f16", "f32") if a.is_ptr else a.elem_type != "i32" for a in graph.args):
        return False
    shapes = {(), (bm, bn), (bm, d), (bn, d), (d, bn)}
    for size in (bm, bn, d):
        shapes.update({(size,), (size, 1), (1, size)})
    for op in ops:
        if any(s not in shapes for s in lowerer._native_shapes_for_op(op)):
            return False
        if op.op == "arith.constant" and not isinstance(op.attrs.get("value"), (int, float)):
            return False
        if op.op == "arith.select":
            facts = graph.result_meta[op.id].type
            if facts.kind not in ("integer", "float") or facts.elem not in ("i1", "i32", "bf16", "f16", "f32"):
                return False
    dots = [o for o in ops if o.op == "tt.dot"]
    loads = [o for o in ops if o.op == "tt.load"]
    stores = [o for o in ops if o.op == "tt.store"]
    if len(dots) != 2 or len(loads) != 4 or len(stores) != 1:
        return False
    if any(d not in loop.region_ops or len(d.operand_ids) != 3 or d.elem_type != "f32" for d in dots):
        return False
    qk, pv = dots
    shape = lambda vid: lowerer._native_shape(vid, op_name="tt.dot")
    if lowerer._native_shapes_for_op(qk) != ((bm, bn),):
        return False
    d = shape(qk.operand_ids[0])[-1]
    if d not in (32, 64):
        return False
    if (
        shape(qk.operand_ids[0]) != (bm, d)
        or shape(qk.operand_ids[1]) != (d, bn)
        or shape(pv.operand_ids[0]) != (bm, bn)
        or shape(pv.operand_ids[1]) != (bn, d)
        or lowerer._native_shapes_for_op(pv) != ((bm, d),)
    ):
        return False
    if any(graph.result_meta[v].type.elem not in ("f16", "f32") for dot in dots for v in dot.operand_ids):
        return False
    by = {o.id: o for o in ops}

    def peel(vid, wrappers=("ttg.convert_layout", "arith.extf")):
        seen = set()
        while vid not in seen:
            seen.add(vid)
            op = by.get(vid)
            if op is None or op.op not in wrappers:
                return vid
            if len(op.operand_ids) != 1:
                return None
            vid = op.operand_ids[0]
        return None

    bias = by.get(peel(qk.operand_ids[2]))
    if bias not in loads or lowerer._native_shapes_for_op(bias) != ((bm, bn),):
        return False
    # In particular, a scale after a causal select is still a score-result scale.
    # Inspect the whole score-to-first-reduction segment, not only direct users
    # of the dot (a select would otherwise hide that same protected epilogue).
    region = loop.region_ops
    qi = region.index(qk)
    ri = next((i for i, o in enumerate(region) if o.op == "tt.reduce"), -1)
    if ri <= qi or any(o.op in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf") for o in region[qi + 1 : ri]):
        return False
    # Dot-result arithmetic is not part of this recovery: the existing integrity
    # backstop still owns those sources. The bias is the actual dot accumulator.
    for op in ops:
        if op.op in ("arith.mulf", "arith.divf", "arith.addf", "arith.subf"):
            for vid in op.operand_ids:
                origin = peel(vid, ("ttg.convert_layout", "arith.extf", "arith.truncf"))
                if origin in (qk.id, pv.id):
                    # Score minus a broadcast row maximum is the softmax input.
                    # A same-shape arbitrary tensor isn't enough to certify it.
                    if op.op != "arith.subf" or vid != op.operand_ids[0] or origin != qk.id:
                        return False
                    rhs = by.get(op.operand_ids[1])
                    if rhs is None or rhs.op != "tt.broadcast" or shape(rhs.operand_ids[0]) != (bm, 1):
                        return False
    return True


def emit_loaded_transpose(lowerer, op):
    """Materialize a large, directly loaded transpose with source-exact masking.

    This narrow producer reloads every logical element, rather than repeating the
    one scalar held by a physical lane. Computed operands keep the old refusal.
    """

    def walk(ops):
        for item in ops:
            yield item
            yield from walk(item.region_ops or [])
            yield from walk(item.else_ops or [])

    by = {item.id: item for item in walk(lowerer.graph.ops)}
    src = by.get(op.operand_ids[0])
    if src is None or src.op != "tt.load" or not 1 <= len(src.operand_ids) <= 3:
        return False
    shape = lowerer._native_shape(src.id, op_name=op.op)
    if len(shape) != 2 or lowerer._parse_trans_order(op, 2) != [1, 0]:
        return False
    m, n = shape
    if lowerer._native_shapes_for_op(op) != ((n, m),):
        return False
    load = lowerer._rebuild_staged_fill_load(src.operand_ids[0], by, m, n)
    if len(src.operand_ids) > 1:
        mask = lowerer._rebuild_staged_fill_mask(src.operand_ids[1], by, m, n, kind="load")
        other = "0.0f"
        if len(src.operand_ids) == 3:
            leaf = by.get(src.operand_ids[2])
            while leaf is not None and leaf.op in ("tt.splat", "tt.broadcast"):
                leaf = by.get(leaf.operand_ids[0])
            if leaf is None or leaf.op != "arith.constant" or not isinstance(leaf.attrs.get("value"), (int, float)):
                return False
            other = lowerer._lookup(src.operand_ids[2])
        load = f"({mask} ? ({load}) : ({other}))"
    name = f"trans_shared_{lowerer._shared_counter}"
    lowerer._shared_counter += 1
    lowerer.kb.declare_threadgroup_array(name, dtype="fp32", size=m * n)
    width = getattr(lowerer, "_actual_dispatch_threads", lowerer.effective_block_size)
    lowerer.kb.raw_line(f"    for (uint _tx = lid; _tx < {m * n}u; _tx += {width}u) {{")
    lowerer.kb.raw_line(f"        uint _fill_row = _tx % {m}u;")
    lowerer.kb.raw_line(f"        uint _fill_col = _tx / {m}u;")
    lowerer.kb.raw_line(f"        {name}[_tx] = {load};")
    lowerer.kb.raw_line("    }")
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    value = lowerer._next_var("trans")
    lowerer.kb.raw_line(f"    float {value} = {name}[lid];")
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lowerer.env[op.id] = value
    lowerer.env_types[op.id] = lowerer.env_types.get(src.id, "fp32")
    lowerer.env_shapes[op.id] = (n, m)
    if not hasattr(lowerer, "_shared_mem_descs"):
        lowerer._shared_mem_descs = {}
    lowerer._shared_mem_descs[op.id] = (name, (n, m), "fp32")
    return True


def emit_row_broadcast(lowerer, op):
    """Rebind row scalars from the score's width to a smaller output tile."""
    src = op.operand_ids[0]
    shape = lowerer._native_shape(src, op_name=op.op)
    target = lowerer._native_shape(op.id, op_name=op.op)
    facts = lowerer.graph.result_meta[src].type
    width = lowerer._actual_dispatch_threads
    if len(shape) != 2 or shape[1] != 1 or facts.elem != "f32" or len(target) != 2:
        return False
    m, n = target
    if m * n > width:
        return False
    loop = next(o for o in lowerer.graph.ops if o.op == "scf.for")
    qk = next(o for o in loop.region_ops if o.op == "tt.dot")
    bm, bn = lowerer._native_shape(qk.id, op_name=qk.op)
    if m != bm or n == bn:
        return False
    name = f"row_rebind_{lowerer._shared_counter}"
    lowerer._shared_counter += 1
    lowerer.kb.declare_threadgroup_array(name, dtype="fp32", size=m)
    lowerer.kb.raw_line(f"    if (lid < {bm * bn}u && lid % {bn}u == 0u) {name}[lid / {bn}u] = {lowerer._lookup(src)};")
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    value = lowerer._next_var("row_rebind")
    lowerer.kb.raw_line(f"    float {value} = (lid < {m * n}u) ? {name}[lid / {n}u] : 0.0f;")
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lowerer.env[op.id] = value
    lowerer.env_types[op.id] = "fp32"
    lowerer.env_shapes[op.id] = target
    return True


def emit_small_store(lowerer, op):
    """Store in the output matrix's coordinates, not the score matrix's."""
    from .msl_types import triton_type_to_msl

    shape = lowerer._native_shape(op.operand_ids[0], op_name=op.op)
    if len(shape) != 2 or shape[0] * shape[1] > lowerer._actual_dispatch_threads:
        return False
    by = {o.id: o for o in lowerer.graph.ops}
    for item in lowerer.graph.ops:
        by.update({o.id: o for o in item.region_ops or []})
    m, n = shape
    base, _ = lowerer.env_is_ptr[op.operand_ids[0]]
    offset = lowerer._rebuild_staged_fill_offset(op.operand_ids[0], by, base, m, n)
    mask = lowerer._rebuild_staged_fill_mask(op.operand_ids[2], by, m, n) if len(op.operand_ids) > 2 else "true"
    dtype = triton_type_to_msl(lowerer._trace_ptr_dtype(op.operand_ids[0]))
    # Every load is emitted above this line and the program is one threadgroup, so behind
    # this barrier the store cannot reach a cell another thread still reads. The loop's
    # own barriers do not cover a loop that runs no iteration. store_follows_every_load
    # checks the emitted order before the launcher's overlap refusal is lifted.
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lowerer.kb.raw_line(f"    if (lid < {m * n}u) {{")
    lowerer.kb.raw_line(f"        uint _fill_row = lid / {n}u;")
    lowerer.kb.raw_line(f"        uint _fill_col = lid % {n}u;")
    lowerer.kb.raw_line(
        f"        if ({mask}) {base}[{offset}] = static_cast<{dtype}>({lowerer._lookup(op.operand_ids[1])});"
    )
    lowerer.kb.raw_line("    }")
    return True


def store_follows_every_load(lowerer, msl):
    """Does one barrier separate every load of an argument buffer from the one store?

    Reads the emitted shader: its last barrier must stand in the kernel body, outside
    every loop and branch, so that each thread of the program's one threadgroup passes
    it once, with the store the only use of an argument buffer below it. An output that
    shares memory with an input then keeps the source's result. False when the text does
    not show that, for example a load rendered inside the stored value.

    The text is matched, not parsed, so it has to be of the plain kind the replay emits:
    no thread returns early, and an argument buffer is only ever declared or indexed by
    its own name. A buffer passed to a call, copied to another pointer or dereferenced
    could be read below the barrier under a name this does not look for. Each buffer
    must be declared volatile: every thread waits at the barrier whatever its memory
    flags, and a volatile load above it is not moved below it.
    """
    import re

    pointers = [re.escape(arg.name) for arg in lowerer.graph.args if arg.is_ptr]
    names = "|".join(pointers)
    msl = re.sub(r"//[^\n]*|/\*.*?\*/", "", msl, flags=re.S)
    head, barrier, tail = msl.rpartition("threadgroup_barrier(")
    if not names or not barrier or head.count("{") - head.count("}") != 1:
        return False
    if re.search(r"\breturn\b", msl) or re.search(rf"\b(?:{names})\b(?!\s*\[)", msl):
        return False
    if any(len(re.findall(rf"\bvolatile device \w+\* {name} \[\[buffer\(", msl)) != 1 for name in pointers):
        return False
    uses = re.findall(rf"\b(?:{names})\s*\[", tail)
    return len(uses) == 1 and re.search(rf"\b(?:{names})\s*\[[^;]*\]\s*=(?!=)[^;]*;", tail) is not None


_WIDE_ROW_CASES = {(32, 64, 64, False), (64, 32, 64, True)}
_WIDE_REPRESENTATION = {
    "arith.extf",
    "ttg.convert_layout",
    "ttg.local_alloc",
    "ttg.local_load",
}
_WIDE_LAYOUT = {"tt.broadcast", "tt.expand_dims", "tt.splat", "ttg.convert_layout"}


def _wide_walk(ops):
    for op in ops:
        yield op
        yield from _wide_walk(op.region_ops or [])
        yield from _wide_walk(op.else_ops or [])


def _wide_by_id(ops):
    result = {}
    for op in ops:
        result[op.id] = op
        for result_id in op.result_ids or []:
            result[result_id] = op
    return result


def _wide_shape(lowerer, value_id):
    try:
        return lowerer._native_shape(value_id, op_name="tt.dot")
    except Exception:
        return ()


def _wide_native_type(lowerer, value_id, shape, elem):
    """Match one ownership-checked native slot, including pointer pointees."""
    facts = lowerer._native_value_facts(value_id, op_name="wide biased attention")
    if facts.shape != shape or facts.is_tensor != bool(shape):
        return False
    if elem.startswith("ptr<"):
        pointee = facts.pointee
        return (
            facts.kind == "pointer" and pointee is not None and pointee.kind == "float" and pointee.elem == elem[4:-1]
        )
    kind = "float" if elem.startswith(("f", "bf")) else "integer"
    return facts.kind == kind and facts.elem == elem


def _wide_peel(by_id, value_id, allowed):
    seen = set()
    while value_id not in seen:
        seen.add(value_id)
        op = by_id.get(value_id)
        if op is None or op.op not in allowed or len(op.operand_ids or []) != 1:
            return value_id
        value_id = op.operand_ids[0]
    return None


def _wide_constant(by_id, value_id):
    core = _wide_peel(by_id, value_id, _WIDE_LAYOUT)
    op = by_id.get(core)
    if op is None or op.op != "arith.constant":
        return None
    value = (op.attrs or {}).get("value")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _wide_scalar_arg(by_id, args_by_id, value_id):
    core = _wide_peel(by_id, value_id, _WIDE_LAYOUT)
    arg = args_by_id.get(core)
    return arg if arg is not None and not arg.is_ptr else None


def _wide_cone_loads(by_id, value_id):
    loads = set()
    seen = set()
    stack = [value_id]
    while stack and len(seen) < 256:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        op = by_id.get(current)
        if op is None:
            continue
        if op.op == "tt.load":
            loads.add(op.id)
            continue
        stack.extend(op.operand_ids or [])
    return loads


def _wide_pointer_address(by_id, args_by_id, value_id):
    """Return one pointer root and every addptr offset, without erasing axes."""
    offsets = []
    seen = set()
    while value_id not in seen and len(seen) < 64:
        seen.add(value_id)
        arg = args_by_id.get(value_id)
        if arg is not None:
            return (arg, offsets) if arg.is_ptr else None
        op = by_id.get(value_id)
        if op is None:
            return None
        if op.op == "tt.addptr" and len(op.operand_ids or []) == 2:
            value_id, offset = op.operand_ids
            offsets.append(offset)
            continue
        if (
            op.op in ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout")
            and len(op.operand_ids or []) == 1
        ):
            value_id = op.operand_ids[0]
            continue
        return None
    return None


def _wide_axis_vector(lowerer, by_id, value_id, axis, rows, cols):
    """Prove matrix placement, then return the varying one-dimensional value."""
    value_id = _wide_peel(by_id, value_id, {"ttg.convert_layout"})
    if value_id is None:
        return None
    full = (rows, cols)
    singleton = (rows, 1) if axis == 0 else (1, cols)
    vector = (rows,) if axis == 0 else (cols,)
    if _wide_shape(lowerer, value_id) == full:
        op = by_id.get(value_id)
        if op is None or op.op != "tt.broadcast" or len(op.operand_ids or []) != 1:
            return None
        value_id = _wide_peel(by_id, op.operand_ids[0], {"ttg.convert_layout"})
    if value_id is None or _wide_shape(lowerer, value_id) != singleton:
        return None
    op = by_id.get(value_id)
    expected_expand_axis = 1 if axis == 0 else 0
    if (
        op is None
        or op.op != "tt.expand_dims"
        or len(op.operand_ids or []) != 1
        or (op.attrs or {}).get("axis") != expected_expand_axis
        or _wide_shape(lowerer, op.operand_ids[0]) != vector
    ):
        return None
    return op.operand_ids[0]


def _wide_coord(
    lowerer,
    by_id,
    args_by_id,
    value_id,
    kind,
    axis,
    loop_iv,
    matrix_rows,
    matrix_cols,
    rows,
    cols,
    depth,
):
    vector = _wide_axis_vector(lowerer, by_id, value_id, axis, matrix_rows, matrix_cols)
    if vector is None:
        return False
    shape = lowerer._index_shape(vector, by_id, args_by_id, loop_iv, 1)
    if (
        shape["bad"] is not None
        or shape["mod"] is not None
        or shape["pid_div"] is not None
        or shape["index_casts"]
        or shape["range"] != 1
    ):
        return False
    if kind == "row":
        return shape["bounds"] == (0, rows) and shape["pid"] == 0 and shape["coef"] == rows and shape["iv"] == 0
    if kind == "key":
        return shape["bounds"] == (0, cols) and shape["pid"] is None and shape["iv"] == 1
    return shape["bounds"] == (0, depth) and shape["pid"] is None and shape["iv"] == 0


def _wide_address_term(
    lowerer,
    by_id,
    args_by_id,
    value_id,
    *,
    kind,
    axis,
    matrix_rows,
    matrix_cols,
    rows,
    cols,
    loop_iv,
    depth,
    multiplier,
):
    value_id = _wide_peel(by_id, value_id, {"ttg.convert_layout"})
    if value_id is None:
        return False
    if multiplier is None:
        return _wide_coord(
            lowerer,
            by_id,
            args_by_id,
            value_id,
            kind,
            axis,
            loop_iv,
            matrix_rows,
            matrix_cols,
            rows,
            cols,
            depth,
        )
    op = by_id.get(value_id)
    if op is None or op.op != "arith.muli" or len(op.operand_ids or []) != 2:
        return False
    for coordinate, factor in (op.operand_ids, op.operand_ids[::-1]):
        multiplier_kind, multiplier_value = multiplier
        if multiplier_kind == "constant":
            factor_ok = _wide_constant(by_id, factor) == multiplier_value
        elif multiplier_kind == "argument":
            scalar = _wide_scalar_arg(by_id, args_by_id, factor)
            factor_ok = scalar is not None and scalar.index == multiplier_value
        else:
            factor_ok = False
        if factor_ok and _wide_coord(
            lowerer,
            by_id,
            args_by_id,
            coordinate,
            kind,
            axis,
            loop_iv,
            matrix_rows,
            matrix_cols,
            rows,
            cols,
            depth,
        ):
            return True
    return False


def _wide_address_matches(
    lowerer,
    by_id,
    args_by_id,
    op,
    root_index,
    matrix_shape,
    specs,
    loop_iv,
    rows,
    cols,
    depth,
):
    if not op.operand_ids:
        return False
    address = _wide_pointer_address(by_id, args_by_id, op.operand_ids[0])
    if address is None:
        return False
    root, offsets = address
    if root.index != root_index or len(offsets) != 2 or len(specs) != 2:
        return False
    first, second = specs

    def matches(value_id, spec):
        kind, axis, multiplier = spec
        return _wide_address_term(
            lowerer,
            by_id,
            args_by_id,
            value_id,
            kind=kind,
            axis=axis,
            matrix_rows=matrix_shape[0],
            matrix_cols=matrix_shape[1],
            rows=rows,
            cols=cols,
            loop_iv=loop_iv,
            depth=depth,
            multiplier=multiplier,
        )

    return (matches(offsets[0], first) and matches(offsets[1], second)) or (
        matches(offsets[0], second) and matches(offsets[1], first)
    )


def _wide_negative_infinity(lowerer, by_id, value_id):
    """Recognize only a native f32 negative-infinity constant.

    The walker exposes a floating APFloat either as a Python float or its raw
    integer bit word.  Interpret an integer only after the ownership-checked
    native type fixes the width: Python numeric equality must never make a
    finite float such as 64512.0 equal the f16 bit pattern 0xFC00.
    """
    core = _wide_peel(by_id, value_id, _WIDE_LAYOUT)
    op = by_id.get(core)
    if op is None or op.op != "arith.constant":
        return False
    facts = lowerer._native_value_facts(core, op_name="wide biased attention")
    if facts.kind != "float" or facts.elem != "f32" or facts.width != 32:
        return False
    value = (op.attrs or {}).get("value")
    if isinstance(value, float):
        return value == float("-inf")
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 0xFF800000
    return False


def _wide_score_policy(lowerer, by_id, args_by_id, qk, reduction, loop_iv, rows, cols, depth, nk_index):
    """Prove the live tail and optional causal selects by numeric coordinate graph."""
    current = _wide_peel(by_id, reduction.operand_ids[0], _WIDE_REPRESENTATION)
    qk_ids = {qk.id, *(qk.result_ids or [])}
    saw_tail = False
    saw_causal = False
    while current not in qk_ids:
        op = by_id.get(current)
        if op is None or op.op != "arith.select" or len(op.operand_ids or []) != 3:
            return None
        condition, yes, no = op.operand_ids
        if not _wide_negative_infinity(lowerer, by_id, no):
            return None
        cmp_id = _wide_peel(by_id, condition, _WIDE_LAYOUT)
        cmp_op = by_id.get(cmp_id)
        if cmp_op is None or cmp_op.op != "arith.cmpi" or len(cmp_op.operand_ids or []) != 2:
            return None
        lhs, rhs = cmp_op.operand_ids
        predicate = (cmp_op.attrs or {}).get("predicate_name")
        bound = _wide_scalar_arg(by_id, args_by_id, rhs)
        if (
            predicate in ("slt", "ult")
            and bound is not None
            and bound.index == nk_index
            and _wide_coord(lowerer, by_id, args_by_id, lhs, "key", 1, loop_iv, rows, cols, rows, cols, depth)
        ):
            if saw_tail:
                return None
            saw_tail = True
        elif (
            predicate in ("sge", "uge")
            and _wide_coord(lowerer, by_id, args_by_id, lhs, "row", 0, loop_iv, rows, cols, rows, cols, depth)
            and _wide_coord(lowerer, by_id, args_by_id, rhs, "key", 1, loop_iv, rows, cols, rows, cols, depth)
        ):
            if saw_causal:
                return None
            saw_causal = True
        else:
            return None
        current = _wide_peel(by_id, yes, _WIDE_REPRESENTATION)
    return saw_causal if saw_tail else None


def wide_row_plan(lowerer):
    """Return an exact row-replay plan for the two >1024 score-tile sources."""
    graph = lowerer.graph
    if graph.called_funcs or len(graph.args) != 7:
        return None
    ops = list(_wide_walk(graph.ops))
    by_id = _wide_by_id(ops)
    args_by_id = {arg.id: arg for arg in graph.args}
    loops = [op for op in ops if op.op == "scf.for"]
    dots = [op for op in ops if op.op == "tt.dot"]
    loads = [op for op in ops if op.op == "tt.load"]
    stores = [op for op in ops if op.op == "tt.store"]
    reductions = [op for op in ops if op.op == "tt.reduce"]
    if len(loops) != 1 or len(dots) != 2 or len(loads) != 4 or len(stores) != 1 or len(reductions) != 2:
        return None
    loop = loops[0]
    if len(loop.operand_ids or []) != 6 or len(loop.result_ids or []) != 3:
        return None
    block_args = (loop.attrs or {}).get("block_arg_ids") or []
    if len(block_args) != 4:
        return None
    loop_iv = block_args[0]
    lower = _wide_constant(by_id, loop.operand_ids[0])
    step = _wide_constant(by_id, loop.operand_ids[2])
    nk_arg = _wide_scalar_arg(by_id, args_by_id, loop.operand_ids[1])
    scalar_args = [arg for arg in graph.args if not arg.is_ptr]
    if lower != 0 or nk_arg is None or len(scalar_args) != 2 or any(arg.elem_type != "i32" for arg in scalar_args):
        return None
    nq_args = [arg for arg in scalar_args if arg.index != nk_arg.index]
    if len(nq_args) != 1:
        return None
    nq_arg = nq_args[0]

    qk, pv = dots
    q_shape = _wide_shape(lowerer, qk.operand_ids[0])
    kt_shape = _wide_shape(lowerer, qk.operand_ids[1])
    p_shape = _wide_shape(lowerer, pv.operand_ids[0])
    v_shape = _wide_shape(lowerer, pv.operand_ids[1])
    if len(q_shape) != 2 or len(kt_shape) != 2:
        return None
    rows, depth = q_shape
    depth_k, cols = kt_shape
    if (
        depth != 64
        or depth_k != depth
        or step != cols
        or p_shape != (rows, cols)
        or v_shape != (cols, depth)
        or _wide_shape(lowerer, qk.id) != (rows, cols)
        or _wide_shape(lowerer, pv.id) != (rows, depth)
        or qk not in (loop.region_ops or [])
        or pv not in (loop.region_ops or [])
    ):
        return None
    if any(
        red.attrs.get("axis") != 1
        or _wide_shape(lowerer, red.operand_ids[0]) != (rows, cols)
        or lowerer._native_shapes_for_op(red) != ((rows,),)
        for red in reductions
    ):
        return None
    # MSL fmax implements maxNum. NaN-propagating maximum has different
    # semantics and must remain outside this replay rather than being erased.
    if sum(op.op == "arith.maxnumf" for op in ops) != 2 or any(op.op == "arith.maximumf" for op in ops):
        return None

    q_load_ids = _wide_cone_loads(by_id, qk.operand_ids[0])
    k_load_ids = _wide_cone_loads(by_id, qk.operand_ids[1])
    v_load_ids = _wide_cone_loads(by_id, pv.operand_ids[1])
    bias_id = _wide_peel(by_id, qk.operand_ids[2], {"ttg.convert_layout"})
    bias_load = by_id.get(bias_id)
    if not all(len(ids) == 1 for ids in (q_load_ids, k_load_ids, v_load_ids)):
        return None
    q_load, k_load, v_load = (by_id[next(iter(ids))] for ids in (q_load_ids, k_load_ids, v_load_ids))
    if bias_load is None or bias_load.op != "tt.load" or bias_load.id not in {op.id for op in loads}:
        return None

    role_ops = (q_load, k_load, v_load, bias_load, stores[0])
    addresses = [_wide_pointer_address(by_id, args_by_id, op.operand_ids[0]) for op in role_ops]
    if any(address is None for address in addresses):
        return None
    role_args = [address[0] for address in addresses]
    if len({arg.index for arg in role_args}) != 5:
        return None
    q_arg, k_arg, v_arg, bias_arg, out_arg = role_args
    if (
        [q_arg.elem_type, k_arg.elem_type, v_arg.elem_type, bias_arg.elem_type, out_arg.elem_type]
        != ["f16", "f16", "f16", "f32", "f32"]
        or len([arg for arg in graph.args if arg.is_ptr]) != 5
        or _wide_shape(lowerer, q_load.id) != (rows, depth)
        or _wide_shape(lowerer, k_load.id) != (cols, depth)
        or _wide_shape(lowerer, v_load.id) != (cols, depth)
        or _wide_shape(lowerer, bias_load.id) != (rows, cols)
    ):
        return None

    yields = [op for op in (loop.region_ops or []) if op.op == "scf.yield"]
    native_contract = [
        (q_load.id, (rows, depth), "f16"),
        (k_load.id, (cols, depth), "f16"),
        (v_load.id, (cols, depth), "f16"),
        (bias_load.id, (rows, cols), "f32"),
        (qk.operand_ids[0], (rows, depth), "f32"),
        (qk.operand_ids[1], (depth, cols), "f32"),
        (qk.operand_ids[2], (rows, cols), "f32"),
        (qk.id, (rows, cols), "f32"),
        (pv.operand_ids[0], (rows, cols), "f32"),
        (pv.operand_ids[1], (cols, depth), "f32"),
        (pv.operand_ids[2], (rows, depth), "f32"),
        (pv.id, (rows, depth), "f32"),
        (loop.operand_ids[0], (), "i32"),
        (loop.operand_ids[1], (), "i32"),
        (loop.operand_ids[2], (), "i32"),
        (loop.operand_ids[3], (rows,), "f32"),
        (loop.operand_ids[4], (rows,), "f32"),
        (loop.operand_ids[5], (rows, depth), "f32"),
        (block_args[0], (), "i32"),
        (block_args[1], (rows,), "f32"),
        (block_args[2], (rows,), "f32"),
        (block_args[3], (rows, depth), "f32"),
        (loop.result_ids[0], (rows,), "f32"),
        (loop.result_ids[1], (rows,), "f32"),
        (loop.result_ids[2], (rows, depth), "f32"),
        (stores[0].operand_ids[0], (rows, depth), "ptr<f32>"),
        (stores[0].operand_ids[1], (rows, depth), "f32"),
        (stores[0].operand_ids[2], (rows, depth), "i1"),
    ]
    if len(yields) != 1 or len(yields[0].operand_ids or []) != 3:
        return None
    native_contract.extend(
        (
            (yields[0].operand_ids[0], (rows,), "f32"),
            (yields[0].operand_ids[1], (rows,), "f32"),
            (yields[0].operand_ids[2], (rows, depth), "f32"),
        )
    )
    if not all(_wide_native_type(lowerer, value_id, shape, elem) for value_id, shape, elem in native_contract):
        return None

    address_specs = (
        (
            q_load,
            q_arg.index,
            (rows, depth),
            (("row", 0, ("constant", depth)), ("head", 1, None)),
        ),
        (
            k_load,
            k_arg.index,
            (cols, depth),
            (("key", 0, ("constant", depth)), ("head", 1, None)),
        ),
        (
            v_load,
            v_arg.index,
            (cols, depth),
            (("key", 0, ("constant", depth)), ("head", 1, None)),
        ),
        (
            bias_load,
            bias_arg.index,
            (rows, cols),
            (("row", 0, ("argument", nk_arg.index)), ("key", 1, None)),
        ),
        (
            stores[0],
            out_arg.index,
            (rows, depth),
            (("row", 0, ("constant", depth)), ("head", 1, None)),
        ),
    )
    if not all(
        _wide_address_matches(
            lowerer,
            by_id,
            args_by_id,
            op,
            root,
            matrix_shape,
            specs,
            loop_iv,
            rows,
            cols,
            depth,
        )
        for op, root, matrix_shape, specs in address_specs
    ):
        return None

    first_reduce = min(reductions, key=lambda op: (loop.region_ops or []).index(op))
    causal = _wide_score_policy(
        lowerer,
        by_id,
        args_by_id,
        qk,
        first_reduce,
        loop_iv,
        rows,
        cols,
        depth,
        nk_arg.index,
    )
    if causal is None or (rows, cols, depth, causal) not in _WIDE_ROW_CASES:
        return None

    q_muls = [
        op
        for op in ops
        if op.op == "arith.mulf" and op.id in {value_id for value_id in _wide_value_cone_ids(by_id, qk.operand_ids[0])}
    ]
    if len(q_muls) != 1 or not any(
        _wide_constant(by_id, value_id) == depth**-0.5 for value_id in q_muls[0].operand_ids
    ):
        return None

    multiplier = lowerer._fa_verify_value_paths(
        detected_q_index=q_arg.index,
        detected_k_index=k_arg.index,
        detected_v_index=v_arg.index,
        detected_out_index=out_arg.index,
        q_scale_required=True,
        scalar_rounding_replay=True,
        k_transposes=1,
        verify_score_path=True,
        expected_score_scale={"factor": 1.0, "op_ids": ()},
        detected_n_ctx_index=nk_arg.index,
        detected_query_index=nq_arg.index,
        load_mask_policy="boundary",
        q_rows=rows,
        kv_cols=cols,
        extra_loads=[(bias_arg.index, "Bias", ("Q", "K"))],
        extra_stores=[(out_arg.index, "Out", ("Q",), True)],
    )
    rounding = getattr(lowerer, "_fa_rounding", {})
    if multiplier != 1.0 or rounding.get("round_q") != "half" or rounding.get("round_p") is not None:
        return None
    return {
        "rows": rows,
        "cols": cols,
        "depth": depth,
        "causal": causal,
        "q": q_arg.name,
        "k": k_arg.name,
        "v": v_arg.name,
        "bias": bias_arg.name,
        "out": out_arg.name,
        "nq": nq_arg.name,
        "nk": nk_arg.name,
    }


def _wide_value_cone_ids(by_id, value_id):
    seen = set()
    stack = [value_id]
    while stack and len(seen) < 256:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        op = by_id.get(current)
        if op is not None and op.op != "tt.load":
            stack.extend(op.operand_ids or [])
    return seen


def lower_wide_rows(lowerer, plan):
    """Replay one independent query row per lane with deferred source-order stores."""
    from ._lowerer_helpers import _check_replay_shared_memory_budget
    from .msl_emitter import KernelBuilder

    rows, cols, depth = plan["rows"], plan["cols"], plan["depth"]
    q, k, v = plan["q"], plan["k"], plan["v"]
    bias, out, nq, nk = plan["bias"], plan["out"], plan["nq"], plan["nk"]
    lowerer.effective_block_size = rows
    lowerer._actual_dispatch_threads = rows
    lowerer.kb = KernelBuilder(lowerer.graph.func_name, block_size=rows)
    lowerer._prescan_stores()
    lowerer._register_args()
    kb = lowerer.kb
    kb.raw_line("    // Source-replayed wide biased attention: one complete query row per lane.")
    kb.raw_line(f"    uint _p515_row = pid * {rows}u + lid;")
    kb.raw_line(f"    bool _p523_active = lid < {rows}u && int(_p515_row) < {nq};")
    kb.raw_line(f"    float _p523_output[{depth}];")
    kb.raw_line("    if (_p523_active) {")

    def emit_score(tag, indent, key_name):
        score = f"_p515_score_{tag}"
        kd = f"_p515_kd_{tag}"
        qv = f"_p515_q_{tag}"
        kb.raw_line(f"{indent}float {score} = {bias}[_p515_row * uint({nk}) + uint({key_name})];")
        kb.raw_line(f"{indent}for (uint {kd} = 0u; {kd} < {depth}u; ++{kd}) {{")
        kb.raw_line(f"{indent}    float {qv} = static_cast<float>({q}[_p515_row * {depth}u + {kd}]);")
        kb.raw_line(f"{indent}    {qv} = static_cast<float>(static_cast<half>({qv} * {depth**-0.5!r}f));")
        kb.raw_line(f"{indent}    {score} += {qv} * static_cast<float>({k}[uint({key_name}) * {depth}u + {kd}]);")
        kb.raw_line(f"{indent}}}")
        if plan["causal"]:
            kb.raw_line(f"{indent}if (int(_p515_row) < {key_name}) {score} = -INFINITY;")
        return score

    # Scores depend only on Q/K/Bias, and no output is stored until every
    # lane has completed all source reads. Retain one block privately, then
    # overwrite its scores with probabilities after the ordered maximum pass.
    # The existing output row holds accumulators until final normalization.
    kb.raw_line(f"        float _p553_block_values[{cols}];")
    kb.raw_line(f"        for (uint _p515_od = 0u; _p515_od < {depth}u; ++_p515_od) {{")
    kb.raw_line("            _p523_output[_p515_od] = 0.0f;")
    kb.raw_line("        }")
    kb.raw_line("        float _p515_denom = 0.0f;")
    kb.raw_line("        float _p515_maximum_d = -INFINITY;")
    kb.raw_line(f"        for (int _p515_start_d = 0; _p515_start_d < {nk}; _p515_start_d += {cols}) {{")
    kb.raw_line("            float _p515_block_max_d = -INFINITY;")
    kb.raw_line(f"            for (uint _p515_j_d0 = 0u; _p515_j_d0 < {cols}u; ++_p515_j_d0) {{")
    kb.raw_line("                int _p515_key_d0 = _p515_start_d + int(_p515_j_d0);")
    kb.raw_line("                float _p515_live_d0 = -INFINITY;")
    kb.raw_line(f"                if (_p515_key_d0 < {nk}) {{")
    score = emit_score("d0", "                    ", "_p515_key_d0")
    kb.raw_line(f"                    _p515_live_d0 = {score};")
    kb.raw_line("                }")
    kb.raw_line("                _p553_block_values[_p515_j_d0] = _p515_live_d0;")
    kb.raw_line("                _p515_block_max_d = fmax(_p515_block_max_d, _p515_live_d0);")
    kb.raw_line("            }")
    kb.raw_line("            float _p515_newmax_d = fmax(_p515_maximum_d, _p515_block_max_d);")
    kb.raw_line("            float _p515_alpha_d = exp(_p515_maximum_d - _p515_newmax_d);")
    kb.raw_line("            float _p515_p_sum_d = 0.0f;")
    kb.raw_line(f"            for (uint _p515_j_d1 = 0u; _p515_j_d1 < {cols}u; ++_p515_j_d1) {{")
    kb.raw_line("                float _p553_p = exp(_p553_block_values[_p515_j_d1] - _p515_newmax_d);")
    kb.raw_line("                _p553_block_values[_p515_j_d1] = _p553_p;")
    kb.raw_line("                _p515_p_sum_d += _p553_p;")
    kb.raw_line("            }")
    kb.raw_line("            _p515_denom = _p515_denom * _p515_alpha_d + _p515_p_sum_d;")
    # Preserve each output's block-rescale and ascending valid-key accumulation
    # order. Padded keys still participate in the denominator (including NaNs),
    # but must not introduce a new zero-times-nonfinite numerator operation.
    kb.raw_line(f"            for (uint _p515_od = 0u; _p515_od < {depth}u; ++_p515_od) {{")
    kb.raw_line("                float _p515_acc = _p523_output[_p515_od];")
    kb.raw_line("                _p515_acc *= _p515_alpha_d;")
    kb.raw_line(f"                for (uint _p515_j_a1 = 0u; _p515_j_a1 < {cols}u; ++_p515_j_a1) {{")
    kb.raw_line("                    int _p515_key_a1 = _p515_start_d + int(_p515_j_a1);")
    kb.raw_line(f"                    if (_p515_key_a1 < {nk}) {{")
    kb.raw_line(
        f"                        _p515_acc += _p553_block_values[_p515_j_a1] * "
        f"static_cast<float>({v}[uint(_p515_key_a1) * {depth}u + _p515_od]);"
    )
    kb.raw_line("                    }")
    kb.raw_line("                }")
    kb.raw_line("                _p523_output[_p515_od] = _p515_acc;")
    kb.raw_line("            }")
    kb.raw_line("            _p515_maximum_d = _p515_newmax_d;")
    kb.raw_line("        }")
    kb.raw_line(f"        for (uint _p515_od = 0u; _p515_od < {depth}u; ++_p515_od) {{")
    kb.raw_line("            float _p515_acc = _p523_output[_p515_od];")
    kb.raw_line("            _p523_output[_p515_od] = _p515_acc / _p515_denom;")
    kb.raw_line("        }")
    kb.raw_line("    }")
    # The source performs its only store after the attention loop has completed
    # every Q/K/V/Bias load for the whole program.  Preserve that ordering for
    # legal runtime aliases: inactive lanes must also reach this uniform barrier.
    kb.raw_line("    threadgroup_barrier(mem_flags::mem_device);")
    kb.raw_line("    if (_p523_active) {")
    kb.raw_line(f"        for (uint _p523_od = 0u; _p523_od < {depth}u; ++_p523_od) {{")
    kb.raw_line(f"            {out}[_p515_row * {depth}u + _p523_od] = _p523_output[_p523_od];")
    kb.raw_line("        }")
    kb.raw_line("    }")
    if kb._threadgroup_arrays:
        raise AssertionError("wide row replay must not allocate threadgroup scratch")
    lowerer._replay_shared_bytes = _check_replay_shared_memory_budget(kb._threadgroup_arrays, {})
    return "// Source-replayed wide biased attention\n" + kb.build()
