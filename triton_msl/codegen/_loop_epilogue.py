"""Admission of source-replayed floating-point K-loop epilogues.

The proof-only projection below checks the underlying bare dot's existing address,
carry and operand contracts. The actual graph is never edited: generic lowering
replays its original loop, masks, scalar operations and bias load.
"""

from dataclasses import replace


def eligible(lowerer):
    graph = lowerer.graph
    loops = [o for o in graph.ops if o.op == "scf.for"]
    if len(loops) != 1 or graph.called_funcs:
        return False
    loop = loops[0]
    if len(loop.operand_ids) != 4 or len(loop.attrs.get("block_arg_ids", [])) != 2:
        return False
    allops = list(graph.ops) + list(loop.region_ops or [])
    if any(o.region_ops or o.else_ops for o in allops if o is not loop):
        return False
    allowed = {
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.remsi",
        "arith.cmpi",
        "arith.andi",
        "arith.addf",
        "arith.mulf",
        "arith.extf",
        "arith.truncf",
        "tt.get_program_id",
        "tt.make_range",
        "tt.splat",
        "tt.expand_dims",
        "tt.broadcast",
        "tt.addptr",
        "tt.load",
        "tt.store",
        "tt.dot",
        "ttg.local_alloc",
        "ttg.local_load",
        "ttg.convert_layout",
        "scf.for",
        "scf.yield",
        "tt.return",
    }
    if any(o.op not in allowed for o in allops):
        return False
    floats = ("f16", "bf16", "f32")
    if any(a.elem_type not in floats if a.is_ptr else a.elem_type not in ("f32", "i32") for a in graph.args):
        return False
    dots = [o for o in allops if o.op == "tt.dot"]
    if len(dots) != 1:
        return False
    dot = dots[0]
    a_shape = lowerer._native_shape(dot.operand_ids[0], op_name="tt.dot")
    b_shape = lowerer._native_shape(dot.operand_ids[1], op_name="tt.dot")
    if len(a_shape) != 2 or len(b_shape) != 2:
        return False
    bm, bk = a_shape
    k2, bn = b_shape
    if k2 != bk or any(d not in (16, 32, 64) for d in (bm, bn, bk)):
        return False
    shapes = {(), (1,), (bm, bk), (bk, bn), (bm, bn)}
    for d in (bm, bn, bk):
        shapes.update({(d,), (d, 1), (1, d)})
    for o in allops:
        if any(tuple(s) not in shapes for s in lowerer._native_shapes_for_op(o)):
            return False
        if o.op == "arith.constant" and not isinstance(o.attrs.get("value"), (int, float)):
            return False
    dots = [o for o in allops if o.op == "tt.dot"]
    stores = [o for o in allops if o.op == "tt.store"]
    if len(dots) != 1 or len(stores) != 1 or dots[0] not in loop.region_ops:
        return False
    dot, store = dots[0], stores[0]
    if dot.elem_type != "f32" or lowerer._native_shapes_for_op(dot) != ((bm, bn),):
        return False
    if any(o.op in ("arith.addf", "arith.mulf") for o in loop.region_ops):
        return False
    by = {o.id: o for o in allops}
    args = {a.id: a for a in graph.args}
    root = (loop.result_ids or [loop.id])[0]
    epi, bias_loads = set(), []
    reached = False
    pending = [store.operand_ids[1]]
    while pending:
        vid = pending.pop()
        if vid == root:
            reached = True
            continue
        if vid in epi:
            continue
        if vid in args:
            if args[vid].is_ptr or args[vid].elem_type != "f32":
                return False
            continue
        op = by.get(vid)
        if op is None or op not in graph.ops or op.elem_type not in floats:
            return False
        if op.op in ("arith.addf", "arith.mulf") and op.elem_type != "f32":
            # Narrow storage and explicit narrow/extend casts are replayed; this
            # path does not silently widen a source's narrow arithmetic operation.
            return False
        if op.op not in (
            "arith.addf",
            "arith.mulf",
            "arith.extf",
            "arith.truncf",
            "arith.constant",
            "tt.splat",
            "tt.expand_dims",
            "tt.broadcast",
            "ttg.convert_layout",
            "tt.load",
        ):
            return False
        epi.add(vid)
        if op.op == "tt.load":
            bias_loads.append(op)
        else:
            pending.extend(op.operand_ids)
    if not reached or not any(by[i].op in ("arith.addf", "arith.mulf") for i in epi):
        return False
    if len(bias_loads) > 1:
        return False
    # The only additional memory input is a source-masked column bias. The
    # generic per-thread path reads it at exactly the output's column coordinate.
    if bias_loads:
        load = bias_loads[0]
        if lowerer._native_shapes_for_op(load) != ((bn,),):
            return False
        ptr = by.get(load.operand_ids[0])
        if ptr is None or ptr.op != "tt.addptr" or len(ptr.operand_ids) != 2:
            return False
        base = by.get(ptr.operand_ids[0])
        if base is None or base.op != "tt.splat" or len(base.operand_ids) != 1:
            return False
        arg = args.get(base.operand_ids[0])
        if arg is None or not arg.is_ptr or arg.elem_type not in floats:
            return False
        traced = lowerer.infer_dot_strides(with_index=True)
        if traced is None:
            return False
        indices = traced[1]
        bshape = lowerer._index_shape(ptr.operand_ids[1], by, args, None)
        cshape = lowerer._index_shape(indices["C"][1], by, args, None)
        if bshape is None or bshape["bad"] or bshape != cshape:
            return False
    # Keep just the dependencies of a hypothetical raw-result store, including
    # every loop body op. This isolates the already-reviewed matmul proof from the
    # separately checked epilogue without changing the actual IR or its metadata.
    # Preserve the real final store conversion in the proof projection. Dropping
    # it would make a narrow C falsely look like an unconverted f32 store.
    # All intermediate epilogue casts are replayed by the ORIGINAL generic graph.
    terminal = store.operand_ids[1]
    while by.get(terminal) is not None and by[terminal].op == "ttg.convert_layout":
        terminal = by[terminal].operand_ids[0]
    cast = by.get(terminal)
    replacements = {}
    projected_value = root
    if cast is not None and cast.op == "arith.truncf" and cast.elem_type in ("f16", "bf16"):
        replacements[cast.id] = replace(cast, operand_ids=[root])
        projected_value = cast.id
    projected_store = replace(store, operand_ids=[store.operand_ids[0], projected_value, *store.operand_ids[2:]])
    keep = {store.id, loop.id}
    pending = list(projected_store.operand_ids) + list(loop.operand_ids)
    for op in loop.region_ops:
        pending.extend(op.operand_ids)
    while pending:
        vid = pending.pop()
        if vid in keep:
            continue
        keep.add(vid)
        op = replacements.get(vid, by.get(vid))
        if op is not None:
            pending.extend(op.operand_ids)
    projected = replace(
        graph, ops=[projected_store if o is store else replacements.get(o.id, o) for o in graph.ops if o.id in keep]
    )
    prover = type(lowerer)(projected, lowerer.options)
    # The loop bounds are replayed, not reduced to a template extent: any spelling is the source's.
    prover._source_loop_replayed = True
    return prover._dot_template_value_paths()[0] is None


def deferred_wide_ops(lowerer):
    """Value-only epilogue ops to emit once, inside the wide store loop.

    The generic wide binary/cast emitters update shared storage IN PLACE. They
    must not run before the per-element replay, which needs the raw loop result.
    Pointer/mask computation and the source bias load still emit normally.
    """
    by = {o.id: o for o in lowerer.graph.ops}
    store = next(o for o in lowerer.graph.ops if o.op == "tt.store")
    shape = lowerer._native_shape(store.operand_ids[0], op_name="tt.store")
    if shape[0] * shape[1] <= 1024:
        return set()
    loop = next(o for o in lowerer.graph.ops if o.op == "scf.for")
    root = (loop.result_ids or [loop.id])[0]
    pending, deferred = [store.operand_ids[1]], set()
    while pending:
        vid = pending.pop()
        op = by.get(vid)
        if vid == root or vid in deferred or op is None or op.op in ("tt.load", "arith.constant"):
            continue
        deferred.add(vid)
        pending.extend(op.operand_ids)
    return deferred


def emit_store(lowerer, store):
    """Replay a proved column epilogue per logical output, not once per thread.

    The K-loop already carries wide accumulators in shared memory. The ordinary
    scalar epilogue only observes its first thread-width slice. Re-evaluate that
    epilogue's actual SSA DAG for each carried value; do not algebraically fuse
    alpha/beta or discard intermediate precision conversions. For a tile fitting
    the threadgroup the generic value is already correct, but its STORE address
    still needs the output's own coordinates rather than a wider operand's map.
    """
    from triton_msl.errors import MetalNonRecoverableError
    from .msl_types import triton_type_to_msl

    shape = lowerer._native_shape(store.operand_ids[0], op_name="tt.store")
    width = lowerer._actual_dispatch_threads
    if len(shape) != 2:
        raise MetalNonRecoverableError("epilogue output is not a native matrix")
    bm, bn = shape
    wide = bm * bn > width
    if width % bn:
        raise MetalNonRecoverableError("epilogue column broadcast does not repeat at the physical stride")
    loop = next(o for o in lowerer.graph.ops if o.op == "scf.for")
    root = (loop.result_ids or [loop.id])[0]
    desc = getattr(lowerer, "_shared_mem_descs", {}).get(root)
    if wide and (desc is None or desc[1] != shape or desc[2] != "fp32"):
        raise MetalNonRecoverableError("wide epilogue lacks its exact carried accumulator storage")
    by = {o.id: o for o in lowerer.graph.ops}
    all_by = dict(by)
    all_by.update({o.id: o for o in loop.region_ops})
    memo = {root: f"{desc[0]}[_st]"} if wide else {}
    declarations = []

    def value(vid):
        if vid in memo:
            return memo[vid]
        op = by.get(vid)
        if op is None or op.op in ("arith.constant", "tt.load"):
            # Only proven uniform scalars and column-bias loads reach this leaf.
            # The physical stride is a whole number of columns, so each thread's
            # column value is identical for every _st it owns.
            return lowerer._lookup(vid)
        if op.op in ("tt.splat", "tt.expand_dims", "tt.broadcast", "ttg.convert_layout"):
            return value(op.operand_ids[0])
        if op.op in ("arith.extf", "arith.truncf"):
            dtype = {"f16": "half", "bf16": "bfloat", "f32": "float"}[op.elem_type]
            expression = f"float(static_cast<{dtype}>({value(op.operand_ids[0])}))"
        elif op.op in ("arith.addf", "arith.mulf"):
            symbol = "+" if op.op == "arith.addf" else "*"
            expression = f"({value(op.operand_ids[0])} {symbol} {value(op.operand_ids[1])})"
        else:
            raise MetalNonRecoverableError(f"wide epilogue cannot replay {op.op}")
        name = lowerer._next_var("epilogue")
        declarations.append(f"        float {name} = {expression};")
        memo[vid] = name
        return name

    result = value(store.operand_ids[1]) if wide else lowerer._lookup(store.operand_ids[1])
    base, _ = lowerer.env_is_ptr[store.operand_ids[0]]
    offset = lowerer._rebuild_staged_fill_offset(store.operand_ids[0], all_by, base, bm, bn)

    def replay_mask(vid):
        op = all_by.get(vid)
        if op is not None and op.op in ("tt.broadcast", "ttg.convert_layout"):
            return replay_mask(op.operand_ids[0])
        if op is not None and op.op == "arith.andi" and len(op.operand_ids) == 2:
            return f"({replay_mask(op.operand_ids[0])} && {replay_mask(op.operand_ids[1])})"
        return lowerer._rebuild_staged_fill_mask(vid, all_by, bm, bn)

    mask = replay_mask(store.operand_ids[2]) if len(store.operand_ids) > 2 else None
    storage = triton_type_to_msl(lowerer._trace_ptr_dtype(store.operand_ids[0]))
    lowerer.kb.raw_line(f"    for (uint _st = lid; _st < {bm * bn}u; _st += {width}u) {{")
    lowerer.kb.raw_line(f"        uint _fill_row = _st / {bn}u;")
    lowerer.kb.raw_line(f"        uint _fill_col = _st % {bn}u;")
    for line in declarations:
        lowerer.kb.raw_line(line)
    condition = f"if ({mask}) " if mask else ""
    lowerer.kb.raw_line(f"        {condition}{base}[{offset}] = static_cast<{storage}>({result});")
    lowerer.kb.raw_line("    }")
    return True
