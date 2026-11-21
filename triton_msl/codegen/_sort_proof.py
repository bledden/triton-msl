"""Proof that a kernel is a row-wise ``tl.sort`` and nothing else.

The row-wise sort template replaces the whole kernel: each thread reads one row of N
cells from ``X + row * stride + column``, sorts it in registers and writes it to the same
place under Z. It reads nothing of the source but the facts it is handed, so it may stand
only for a source that is exactly that: one load of an (M, N) tile from a base argument at
those addresses, the bitonic network ``triton.language.sort`` writes, and one store of the
result. Four things more are facts the template can be handed, and are proved here:

- the load takes column ``c`` of every row from a constant place ``columns[c]`` of the row
  (the columns in reverse, say);
- a mask on the load or on the store that is true on every cell of the tile, which is no
  mask;
- the network run again on its own result (a sort of a sort), in either direction each time;
- one conversion of the sorted values to the stored type.

Anything else the source says (a program term in an address, an offset in the store's, a
mask that leaves a cell out, arithmetic on the values, another rearrangement, a
``tl.flip``, whose reshape and exclusive-or reductions resemble a sort's) is lost by the
template.

``prove_row_wise_sort`` checks the source operation by operation and evaluates its index
and mask tensors on the host. It answers with the facts the template needs, or None, which
leaves the kernel to the ordinary lowering.
"""

import numpy as np

from .mlir_walker import _extract_shape

# Every operation a source that is only a sort of one loaded tile is made of.
_VOCABULARY = frozenset(
    {
        "arith.addi",
        "arith.cmpf",
        "arith.cmpi",
        "arith.constant",
        "arith.extsi",
        "arith.extui",
        "arith.extf",
        "arith.truncf",
        "arith.muli",
        "arith.select",
        "arith.sitofp",
        "arith.subi",
        "arith.trunci",
        "arith.uitofp",
        "arith.xori",
        "tt.addptr",
        "tt.bitcast",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.load",
        "tt.make_range",
        "tt.reduce",
        "tt.reshape",
        "tt.return",
        "tt.splat",
        "tt.store",
        "ttg.convert_layout",
    }
)

_GREATER = {"arith.cmpf": ("ogt",), "arith.cmpi": ("sgt", "ugt")}

# A conversion the stored value may be, of the sorted values. ``arith.extui`` is also how
# every step of the network widens its comparison, so it is known by where it stands; each
# of the others may stand nowhere else in the kernel.
_CONVERSIONS = frozenset(
    {"arith.extsi", "arith.extui", "arith.extf", "arith.truncf", "arith.sitofp", "arith.trunci", "arith.uitofp"}
)
_STORED_ONLY = _CONVERSIONS - {"arith.extui"}

# The comparisons a mask is read through. Index tensors are exact integers that fit 32
# bits, so the signed orders and equality are the source's.
_MASK_COMPARISONS = {
    "eq": np.equal,
    "ne": np.not_equal,
    "slt": np.less,
    "sle": np.less_equal,
    "sgt": np.greater,
    "sge": np.greater_equal,
}

# The most networks one kernel may run on a tile; the shader writes each of them out.
MOST_SORTS = 4


# The arithmetic an index tensor is built with. Its operands fit 32 bits, so every result
# is exact in 64, and one that leaves 32 bits is not followed.
_INDEX_ARITHMETIC = {
    "arith.addi": np.add,
    "arith.muli": np.multiply,
    "arith.subi": np.subtract,
    "arith.xori": np.bitwise_xor,
}


class _Values(dict):
    """Operations by the id of their result; a layout conversion stands for the value it converts."""

    def __init__(self, ops):
        super().__init__()
        converted = []

        def collect(ops):
            for op in ops:
                self[op.id] = op
                if op.op == "ttg.convert_layout" and len(op.operand_ids or ()) == 1:
                    converted.append(op)
                for region in (op.region_ops, op.else_ops):
                    if region:
                        collect(region)

        collect(ops)
        for op in converted:  # in program order, so a chain of conversions resolves to its start
            source = self.get(op.operand_ids[0])
            if source is not None:
                self[op.id] = source

    def same(self, left, right):
        """Whether two ids name one value."""
        left, right = self.get(left), self.get(right)
        return left is not None and left is right


def _index_tensor(by_id, value_id, depth=0):
    """The integer tensor ``value_id`` holds when it is built from ranges and constants alone, else None."""
    op = by_id.get(value_id)
    if op is None or depth > 64:
        return None
    shape = _extract_shape(op.type_str or "")
    shape = tuple(shape) if shape else ()
    operands = [_index_tensor(by_id, operand, depth + 1) for operand in op.operand_ids or ()]
    if any(operand is None for operand in operands):
        return None
    attrs = op.attrs or {}
    if op.op == "arith.constant":
        value = attrs.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        result = np.full(shape, value, dtype=np.int64)
    elif op.op == "tt.make_range":
        start, end = attrs.get("start"), attrs.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        result = np.arange(start, end, dtype=np.int64)
    elif op.op == "tt.expand_dims" and len(operands) == 1:
        axis = attrs.get("axis")
        if not isinstance(axis, int) or not 0 <= axis <= operands[0].ndim:
            return None
        result = np.expand_dims(operands[0], axis)
    elif op.op == "tt.broadcast" and len(operands) == 1:
        if operands[0].ndim != len(shape) or any(have not in (1, want) for have, want in zip(operands[0].shape, shape)):
            return None
        result = np.broadcast_to(operands[0], shape)
    elif op.op == "tt.reshape" and len(operands) == 1:
        if operands[0].size != int(np.prod(shape, dtype=np.int64)):
            return None
        result = operands[0].reshape(shape)
    elif op.op in _INDEX_ARITHMETIC and len(operands) == 2:
        if operands[0].shape != operands[1].shape:
            return None
        result = _INDEX_ARITHMETIC[op.op](operands[0], operands[1])
    else:
        return None
    if result.shape != shape or (result.size and (result.min() < -(2**31) or result.max() >= 2**31)):
        return None  # nothing here follows a 32-bit wrap, or needs a wider index
    return result


def _pointer_offsets(by_id, arguments, pointer_id, depth=0):
    """(argument, offsets) when ``pointer_id`` is one pointer argument plus constant offsets, else None.

    ``X + rows * S + columns`` is two pointer additions, the first often on a column of
    pointers that is then broadcast. Every offset is a constant of the source that fits 32
    bits, so their 64-bit sum is the address exactly.
    """
    op = by_id.get(pointer_id)
    if op is None or depth > 8:
        return None
    shape = tuple(_extract_shape(op.type_str or "") or ())
    if op.op == "tt.splat" and len(op.operand_ids or ()) == 1:
        argument = arguments.get(op.operand_ids[0])
        if argument is None or not argument.is_ptr:
            return None
        return argument, np.zeros(shape, dtype=np.int64)
    if op.op == "tt.addptr" and len(op.operand_ids or ()) == 2:
        base = _pointer_offsets(by_id, arguments, op.operand_ids[0], depth + 1)
        step = _index_tensor(by_id, op.operand_ids[1])
        if base is None or step is None or base[1].shape != shape or step.shape != shape:
            return None
        return base[0], base[1] + step
    if op.op == "tt.broadcast" and len(op.operand_ids or ()) == 1:
        base = _pointer_offsets(by_id, arguments, op.operand_ids[0], depth + 1)
        if base is None or base[1].ndim != len(shape):
            return None
        if any(have not in (1, want) for have, want in zip(base[1].shape, shape)):
            return None
        return base[0], np.broadcast_to(base[1], shape)
    return None


def _mask_tensor(by_id, value_id, depth=0):
    """The boolean tensor ``value_id`` holds when it compares tensors of ranges and constants, else None."""
    op = by_id.get(value_id)
    if op is None or depth > 8:
        return None
    shape = tuple(_extract_shape(op.type_str or "") or ())
    attrs = op.attrs or {}
    if op.op == "arith.cmpi" and len(op.operand_ids or ()) == 2:
        compare = _MASK_COMPARISONS.get(attrs.get("predicate_name"))
        left, right = (_index_tensor(by_id, operand) for operand in op.operand_ids)
        if compare is None or left is None or right is None or left.shape != right.shape:
            return None
        result = compare(left, right)
    elif op.op in ("tt.expand_dims", "tt.broadcast") and len(op.operand_ids or ()) == 1:
        inner = _mask_tensor(by_id, op.operand_ids[0], depth + 1)
        if inner is None:
            return None
        if op.op == "tt.expand_dims":
            axis = attrs.get("axis")
            if not isinstance(axis, int) or not 0 <= axis <= inner.ndim:
                return None
            result = np.expand_dims(inner, axis)
        else:
            if inner.ndim != len(shape) or any(have not in (1, want) for have, want in zip(inner.shape, shape)):
                return None
            result = np.broadcast_to(inner, shape)
    else:
        return None
    return result if result.shape == shape else None


def _every_cell(by_id, mask_id, rows, columns):
    """Whether the mask ``mask_id`` is proved true on every cell of the (rows, columns) tile."""
    mask = _mask_tensor(by_id, mask_id)
    return mask is not None and mask.shape == (rows, columns) and bool(mask.all())


def _row_stride(by_id, arguments, pointer_id, rows, columns, placed=False):
    """(argument, S, places) when ``pointer_id`` is ``argument + row * S + places[column]`` over the tile.

    ``places`` is ``0 .. columns - 1`` unless ``placed``, which lets it be any constants
    that are the same on every row and fit 32 bits.
    """
    traced = _pointer_offsets(by_id, arguments, pointer_id)
    if traced is None or traced[1].shape != (rows, columns):
        return None
    argument, offsets = traced
    stride = int(offsets[1, 0] - offsets[0, 0]) if rows > 1 else columns
    places = offsets[0] if placed else np.arange(columns, dtype=np.int64)
    row = np.arange(rows, dtype=np.int64)[:, None]
    if not columns <= stride < 2**31 or not np.array_equal(offsets, row * stride + places[None, :]):
        return None
    if places.min() < -(2**31) or places.max() >= 2**31:
        return None
    return argument, stride, tuple(int(place) for place in places)


def _expected_networks(dims, row_bits):
    """The (axis, swap mask) steps ``triton.language.sort`` takes over a hypercube, for each direction.

    A step compares every element with its partner along ``axis`` and takes the partner
    where ``(x > partner) != mask``. The mask is the element's coordinate along the axis
    exclusive-or the direction of its subsequence: alternating with the coordinate one
    stage up while runs are being built, the requested direction in the last stage.
    """
    coordinate = np.indices((2,) * dims, dtype=np.int64)
    networks = {}
    for descending in (False, True):
        steps = []
        for stage in range(1, row_bits + 1):
            direction = coordinate[dims - stage - 1] if stage < row_bits else int(descending)
            for swap in range(stage - 1, -1, -1):
                axis = dims - 1 - swap
                steps.append((axis, coordinate[axis] ^ direction))
        networks[descending] = steps
    return networks


def _xor_reduction(op, axis):
    attrs = op.attrs or {}
    arguments = list(attrs.get("block_arg_ids") or ())
    returned = list(attrs.get("return_ids") or ())
    body = list(op.region_ops or ())
    if attrs.get("axis") != axis or len(arguments) != 2 or len(returned) != 1 or len(body) != 1:
        return False
    combine = body[0]
    return (
        combine.op == "arith.xori"
        and combine.id == returned[0]
        and sorted(combine.operand_ids or ()) == sorted(arguments)
    )


def _swap_step(by_id, select, hypercube):
    """(input value, axis, swap mask, comparison) of the compare-and-swap that ``select`` ends, or None."""

    def producer(value_id, name, operands):
        op = by_id.get(value_id)
        if op is None or op.op != name or len(op.operand_ids or ()) != operands:
            return None
        return op

    if select.op != "arith.select" or len(select.operand_ids or ()) != 3:
        return None
    condition_id, partner_id, value_id = select.operand_ids
    # The partner: the value with its bits exchanged along one axis, written as an
    # exclusive-or with the exclusive-or of the pair.
    partner_bits = producer(partner_id, "tt.bitcast", 1)
    exchanged = producer(partner_bits.operand_ids[0] if partner_bits else partner_id, "arith.xori", 2)
    if exchanged is None:
        return None
    bits_id, spread_id = exchanged.operand_ids
    value_bits = producer(bits_id, "tt.bitcast", 1)
    if bool(value_bits) != bool(partner_bits):
        return None
    if not by_id.same(value_bits.operand_ids[0] if value_bits else bits_id, value_id):
        return None
    spread = producer(spread_id, "tt.broadcast", 1)
    kept = producer(spread.operand_ids[0], "tt.expand_dims", 1) if spread else None
    reduction = producer(kept.operand_ids[0], "tt.reduce", 1) if kept else None
    if reduction is None or not by_id.same(reduction.operand_ids[0], bits_id):
        return None
    axis = (kept.attrs or {}).get("axis")
    if not isinstance(axis, int) or not _xor_reduction(reduction, axis):
        return None
    for op in (exchanged, spread):
        if tuple(_extract_shape(op.type_str or "") or ()) != hypercube:
            return None
    # The condition: (value > partner) != mask, the mask built from ranges and constants.
    differs = producer(condition_id, "arith.cmpi", 2)
    if differs is None or (differs.attrs or {}).get("predicate_name") != "ne":
        return None
    widened = producer(differs.operand_ids[0], "arith.extui", 1)
    greater = by_id.get(widened.operand_ids[0]) if widened else None
    if (
        greater is None
        or (greater.attrs or {}).get("predicate_name") not in _GREATER.get(greater.op, ())
        or len(greater.operand_ids or ()) != 2
        or not by_id.same(greater.operand_ids[0], value_id)
        or not by_id.same(greater.operand_ids[1], partner_id)
    ):
        return None
    mask = _index_tensor(by_id, differs.operand_ids[1])
    if mask is None or mask.shape != hypercube:
        return None
    return value_id, axis, mask, greater.attrs["predicate_name"]


def prove_row_wise_sort(graph, load, store, rows, columns):
    """The facts of a kernel that is exactly a sort of each row of one loaded tile, or None.

    Returns ``{"networks", "compare", "input", "output", "input_stride", "output_stride",
    "columns", "convert", "masked"}``: the direction of each network in the order they run
    (True: descending), the comparison every step makes (``ogt``, ``sgt`` or ``ugt``), the
    two pointer arguments and the constant row strides of their addresses, the place in
    its row each column is loaded from (None: its own), the conversion the stored value
    is of the sorted one (None: none), and whether an access carries a mask, which is
    then true on every cell.
    """
    by_id = _Values(graph.ops)
    if any(op.op not in _VOCABULARY for op in graph.ops):
        return None
    if any(not argument.is_ptr for argument in graph.args):
        return None  # the template's strides are constants of the source
    if sum(op.op == "tt.load" for op in graph.ops) != 1 or sum(op.op == "tt.store" for op in graph.ops) != 1:
        return None
    if not 1 <= len(load.operand_ids or ()) <= 3 or not 2 <= len(store.operand_ids or ()) <= 3:
        return None
    if (load.attrs or {}).get("isVolatile", False):
        return None
    total = rows * columns
    if rows < 1 or columns < 2 or total & (total - 1) or columns & (columns - 1):
        return None
    dims, row_bits = total.bit_length() - 1, columns.bit_length() - 1
    hypercube = (2,) * dims

    arguments = {argument.id: argument for argument in graph.args}
    source = _row_stride(by_id, arguments, load.operand_ids[0], rows, columns, placed=True)
    target = _row_stride(by_id, arguments, store.operand_ids[0], rows, columns)
    if source is None or target is None or source[0].id == target[0].id:
        return None
    # A mask that is true on every cell leaves nothing out, and the value a masked load
    # gives the cells it leaves out is then given to none.
    masks = [load.operand_ids[1]] if len(load.operand_ids) > 1 else []
    masks += [store.operand_ids[2]] if len(store.operand_ids) > 2 else []
    if not all(_every_cell(by_id, mask, rows, columns) for mask in masks):
        return None

    # The stored value, read back to the load one compare-and-swap at a time. One
    # conversion to the stored type may stand before the store; nothing else may stand in
    # between.
    folded, convert = by_id.get(store.operand_ids[1]), None
    if folded is not None and folded.op in _CONVERSIONS and len(folded.operand_ids or ()) == 1:
        if tuple(_extract_shape(folded.type_str or "") or ()) != (rows, columns):
            return None
        convert, folded = folded.op, by_id.get(folded.operand_ids[0])
    if sum(op.op in _STORED_ONLY for op in graph.ops) != (convert in _STORED_ONLY):
        return None
    if folded is None or folded.op != "tt.reshape" or len(folded.operand_ids or ()) != 1:
        return None
    if tuple(_extract_shape(folded.type_str or "") or ()) != (rows, columns):
        return None
    networks = _expected_networks(dims, row_bits)
    count = len(networks[False])
    # A sort of a sort is the second network's steps on the first one's result.
    current, steps = folded.operand_ids[0], []
    while len(steps) < MOST_SORTS * count:
        select = by_id.get(current)
        if select is None or select.op != "arith.select":
            break
        if tuple(_extract_shape(select.type_str or "") or ()) != hypercube:
            return None
        step = _swap_step(by_id, select, hypercube)
        if step is None:
            return None
        current = step[0]
        steps.append(step[1:])
    if not steps or len(steps) % count:
        return None
    steps.reverse()
    comparisons = {comparison for _, _, comparison in steps}
    if len(comparisons) != 1:
        return None
    unfolded = by_id.get(current)
    if (
        unfolded is None
        or unfolded.op != "tt.reshape"
        or len(unfolded.operand_ids or ()) != 1
        or by_id.get(unfolded.operand_ids[0]) is not load
        or tuple(_extract_shape(unfolded.type_str or "") or ()) != hypercube
        or tuple(_extract_shape(load.type_str or "") or ()) != (rows, columns)
    ):
        return None
    directions = []
    for start in range(0, len(steps), count):
        ran = steps[start : start + count]
        found = [
            descending
            for descending, expected in networks.items()
            if all(axis == want and np.array_equal(mask, swap) for (axis, mask, _), (want, swap) in zip(ran, expected))
        ]
        if len(found) != 1:
            return None
        directions.append(found[0])
    return {
        "networks": directions,
        "compare": comparisons.pop(),
        "input": source[0],
        "output": target[0],
        "input_stride": source[1],
        "output_stride": target[1],
        "columns": None if source[2] == tuple(range(columns)) else source[2],
        "convert": convert,
        "masked": bool(masks),
    }
