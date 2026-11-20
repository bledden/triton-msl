"""Free helpers shared by ``generic_lowerer.py`` and ``_device_func_lowerer.py``.

Extracted from the original ``generic_lowerer.py`` (which used to inline these
in a 9.9 kLOC file) so multiple lowerer entry points can share them without
import cycles.

Contents:
- Comparison predicate maps (``CMPI_*`` / ``CMPF_*``)
- MLIR-to-Triton dtype mapping (``_mlir_to_triton_dtype``)
- MSL integer type helper (``_msl_int_type``)
- Shape arithmetic (``_shape_numel``)
- Layout signature extraction (``_extract_layout_signature``)
- Threadgroup-array aliasing post-pass (``_alias_shared_memory``)
"""

import re

# ---------------------------------------------------------------------------
# Comparison predicate maps
# ---------------------------------------------------------------------------

# MLIR integer comparison predicates (arith.cmpi)
CMPI_PREDICATES = {
    0: "==",  # eq
    1: "!=",  # ne
    2: "<",  # slt
    3: "<=",  # sle
    4: ">",  # sgt
    5: ">=",  # sge
    6: "<",  # ult (unsigned, same op in MSL)
    7: "<=",  # ule
    8: ">",  # ugt
    9: ">=",  # uge
}

# Named predicate map
CMPI_NAMED = {
    "eq": "==",
    "ne": "!=",
    "slt": "<",
    "sle": "<=",
    "sgt": ">",
    "sge": ">=",
    "ult": "<",
    "ule": "<=",
    "ugt": ">",
    "uge": ">=",
}

# MLIR float comparison predicates (arith.cmpf)
CMPF_PREDICATES = {
    0: "false",  # false (always false)
    1: "==",  # oeq
    2: ">",  # ogt
    3: ">=",  # oge
    4: "<",  # olt
    5: "<=",  # ole
    6: "!=",  # one
    # 7-15: unordered variants
}

CMPF_NAMED = {
    "oeq": "==",
    "ogt": ">",
    "oge": ">=",
    "olt": "<",
    "ole": "<=",
    "one": "!=",
    "ueq": "==",
    "ugt": ">",
    "uge": ">=",
    "ult": "<",
    "ule": "<=",
    "une": "!=",
}


# ---------------------------------------------------------------------------
# MLIR type → Triton dtype mapping
# ---------------------------------------------------------------------------


def _mlir_to_triton_dtype(mlir_type: str) -> str:
    """Map MLIR element type to Triton dtype string."""
    _map = {
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
        "f64": "fp64",
        "i1": "i1",
        "i8": "i8",
        "i16": "i16",
        "i32": "i32",
        "i64": "i64",
        # ui64 → u64 so the reshape-drops-type fallback (which can surface a
        # bare "ui64" base type) doesn't misclassify uint64 as fp32.
        "ui64": "u64",
    }
    result = _map.get(mlir_type)
    if result:
        return result
    # FP8 MLIR types: f8E4M3FN, f8E5M2, f8E4M3B11FNUZ, f8E4M3FNUZ, f8E5M2FNUZ
    _fp8_map = {
        "f8E4M3FN": "fp8e4nv",
        "f8E4M3FNUZ": "fp8e4nv",
        "f8E5M2": "fp8e5",
        "f8E5M2FNUZ": "fp8e5",
        "f8E4M3B11FNUZ": "fp8e4b15",
        "f8E4M3": "fp8e4nv",
    }
    fp8_result = _fp8_map.get(mlir_type)
    if fp8_result:
        return fp8_result
    # Handle multi-dim type strings like "4xi32" → extract base type
    m = re.search(r"([a-z]\w*)$", mlir_type)
    if m:
        base = m.group(1)
        return _map.get(base, "fp32")
    # Also check for FP8 in multi-dim strings like "256xf8E4M3FN"
    m = re.search(r"(f8E\w+)$", mlir_type)
    if m:
        return _fp8_map.get(m.group(1), "fp32")
    return "fp32"


# Map MLIR integer elem_type to (MSL type, internal dtype)
_INT_TYPE_MAP = {
    "i1": ("bool", "i1"),
    "i8": ("char", "i8"),
    "i16": ("short", "i16"),
    "i32": ("int", "i32"),
    "i64": ("long", "i64"),
}

# Unsigned variants
_UINT_TYPE_MAP = {
    "i8": ("uchar", "u8"),
    "i16": ("ushort", "u16"),
    "i32": ("uint", "u32"),
    "i64": ("ulong", "u64"),
}


def _msl_int_type(elem_type: str, unsigned: bool = False) -> tuple:
    """Return (msl_type, triton_dtype) for an integer elem_type.

    Args:
        elem_type: MLIR type like "i8", "i16", "i32", "i64"
        unsigned: If True, use unsigned MSL types

    Returns:
        Tuple of (MSL type string, internal dtype string)
    """
    if unsigned and elem_type in _UINT_TYPE_MAP:
        return _UINT_TYPE_MAP[elem_type]
    return _INT_TYPE_MAP.get(elem_type, ("int", "i32"))


def _shape_numel(shape: tuple) -> int:
    """Return the total number of elements in a shape tuple.

    Examples:
        () -> 1 (scalar)
        (32,) -> 32
        (32, 64) -> 2048
    """
    result = 1
    for d in shape:
        result *= d
    return result


def _extract_layout_signature(type_str):
    """Extract the layout portion of an MLIR tensor type string.

    Returns the substring after the first ``,`` (skipping shape/element type)
    up to the closing ``>`` of the outermost ``tensor<`` — i.e., the layout
    attribute.  Returns None if the type has no layout (e.g. bare tensor).

    Examples:
        tensor<256xf32, #blocked> → "#blocked"
        tensor<1x1x2xi32, #ttg.slice<{dim = 5, parent = #blocked}>>
            → "#ttg.slice<{dim = 5, parent = #blocked}>"
    """
    if not type_str or "tensor<" not in type_str:
        return None
    # Find the outermost tensor< ... > and extract everything after the first
    # comma inside it (which separates shape/element from layout).
    start = type_str.find("tensor<")
    if start < 0:
        return None
    # Skip past "tensor<"
    i = start + len("tensor<")
    depth = 1
    comma_pos = -1
    while i < len(type_str) and depth > 0:
        c = type_str[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                break
        elif c == "," and depth == 1 and comma_pos < 0:
            comma_pos = i
        i += 1
    if depth != 0 or comma_pos < 0:
        return None
    layout = type_str[comma_pos + 1 : i].strip()
    return layout if layout else None


# ---------------------------------------------------------------------------
# Shared memory aliasing post-pass
# ---------------------------------------------------------------------------


def _shared_memory_phases(msl: str):
    """Conservative synchronization epochs for generated MSL, not a CFG proof.

    Record standalone threadgroup-memory barriers and their lexical scopes.
    Callers must check both the forward handoff and every common loop backedge.
    Conditional/unknown scopes cannot certify synchronization. We don't insert
    barriers; unrecognized spellings merely prevent reuse.
    """
    # Strip comments and literals without moving line boundaries. In particular,
    # comments/strings containing braces or a barrier cannot certify anything.
    lexical = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
    code = lexical.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), msl)
    # A preprocessor branch or line splice needs preprocessing, not text proof.
    if (
        re.search(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef)\b", code, re.M)
        or "\\\n" in msl
        or re.search(r"\bgoto\b", code)
    ):
        return None
    barrier = re.compile(r"\s*threadgroup_barrier\(\s*([\w:|\s]+)\s*\)\s*;\s*")
    stack, blocks, barriers = [], {}, []
    transfers = set()
    previous = ""
    contexts = []
    code_lines = code.split("\n")
    arrays = set(re.findall(r"^\s*threadgroup\s+\w+\s+(\w+)\s*\[", code, re.M))
    uses_array = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(arrays))) + r")\b") if arrays else None

    def leaves_at_the_head(i):
        # `for (;;) {` whose only way out is `if (!(c)) break;` with nothing but
        # plain statements before it is a loop that tests `c` at its head: the
        # same control flow as `for (...; c; ...)`, which the callers already
        # reason about (zero trips included). The statements before the exit run
        # once more than the body, so they may hold no barrier and touch no
        # threadgroup array. Any other exit keeps the whole function unshared.
        if not re.fullmatch(r"\s*if \(!\(\w+\)\) break;\s*", code_lines[i]):
            return False
        opened = stack[-1]
        if blocks[opened][0] != "for" or not re.fullmatch(r"\s*for \(;;\) \{\s*", code_lines[opened]):
            return False
        for line in code_lines[opened + 1 : i]:
            if (
                "{" in line
                or "}" in line
                or "threadgroup" in line
                or re.search(r"\b(?:break|continue|return|goto)\b", line)
                or (uses_array is not None and uses_array.search(line))
            ):
                return False
        return True

    for i, line in enumerate(code_lines):
        contexts.append(tuple(stack))
        if stack and re.search(r"\b(?:break|continue|return)\b", line) and not leaves_at_the_head(i):
            # An early exit may bypass a lexically unconditional barrier. Until
            # this post-pass has a CFG, do not reuse arrays in that function.
            transfers.add(stack[0])
        match = barrier.fullmatch(line)
        # Reject an unbraced control header on the preceding nonempty line.
        # Known generated statements/blocks end with these delimiters.
        if (
            stack
            and match
            and previous.endswith((";", "{", "}"))
            and all(blocks[b][0] in ("function", "for") for b in stack)
        ):
            flags = {f.strip() for f in match.group(1).split("|")}
            if "mem_flags::mem_threadgroup" in flags and flags <= {
                "mem_flags::mem_threadgroup",
                "mem_flags::mem_device",
            }:
                barriers.append(i)
        for char in line:
            if char == "{":
                # Multiple opens on a line are unknown structured control, so
                # never credit a barrier in those bodies.
                kind = "function" if not stack else "for" if re.match(r"\s*for\s*\(", line) else "unknown"
                if i in blocks:
                    return None
                blocks[i] = [kind, None]
                stack.append(i)
            elif char == "}":
                if not stack:
                    return None
                blocks[stack.pop()][1] = i
        if line.strip():
            previous = line.rstrip()
    return (contexts, blocks, barriers, transfers) if not stack else None


def _alias_shared_memory(msl: str, *, allocation_aliases=None) -> str:
    """Rewrite threadgroup array declarations to reuse memory.

    After the lowerer generates MSL with one threadgroup array per allocation,
    this pass finds arrays with non-overlapping, synchronization-separated
    lifetimes and aliases them to the same physical array. Textual last-use is
    NOT read completion across threads (reduce/scan/split/atomic consumers).

    The algorithm:
    1. Parse all `threadgroup float NAME[SIZE];` declarations.
    2. For each, scan the MSL body for first and last LINE where NAME appears
       (excluding the declaration itself).
    3. Build a conflict graph: overlapping lifetimes OR absence of a proven
       scope-correct threadgroup-memory barrier makes two arrays conflict.
    4. Greedily assign arrays to "physical" slots. For each array (sorted by
       size descending), try to reuse a physical slot whose current occupants
       don't conflict. If none, create a new slot.
    5. Rename arrays in each group to the physical name (the first/largest
       member), remove duplicate declarations, and adjust the declaration
       size to the group maximum.
    """

    if allocation_aliases is not None:
        allocation_aliases.clear()
    lines = msl.split("\n")
    synchronization = _shared_memory_phases(msl)
    if synchronization is None:
        return msl
    contexts, blocks, barriers, transfers = synchronization

    # 1. Parse declarations: name -> (size, decl_line_idx, dtype)
    # Handle common threadgroup types: float, int, uint, half, etc.
    decl_re = re.compile(r"^\s*threadgroup\s+(float|int|uint|half|short|ushort|char|uchar|bool)\s+(\w+)\[(\d+)\];")
    decls = {}  # name -> (size, line_idx, dtype)
    for i, line in enumerate(lines):
        m = decl_re.match(line)
        if m:
            dtype, name, size = m.group(1), m.group(2), int(m.group(3))
            if name in decls:
                return msl  # repeated local names need scope-aware rewriting
            decls[name] = (size, i, dtype)

    if len(decls) < 2:
        return msl  # nothing to alias

    # 2a. Detect loop boundaries: for (...) { ... }
    # Any array used inside a loop has its live range expanded to cover
    # the entire loop, because all iterations share the same memory.
    loop_ranges = []  # list of (loop_start_line, loop_end_line)
    brace_stack = []
    for_line_re = re.compile(r"^\s*for\s*\(")
    for i, line in enumerate(lines):
        if for_line_re.match(line):
            brace_stack.append(i)
        # Track braces — simplistic but works for generated MSL
        opens = line.count("{")
        closes = line.count("}")
        for _ in range(opens):
            if not brace_stack or brace_stack[-1] != i:
                brace_stack.append(i)
        for _ in range(closes):
            if brace_stack:
                start = brace_stack.pop()
                # Only record loops (for statements), not bare blocks
                if for_line_re.match(lines[start]):
                    loop_ranges.append((start, i))

    # 2b. Compute live ranges: name -> (first_use_line, last_use_line)
    live = {}
    raw_live = {}
    for name in decls:
        decl_line = decls[name][1]
        first_use = None
        last_use = None
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        for i, line in enumerate(lines):
            if i == decl_line:
                continue
            if pattern.search(line):
                if first_use is None:
                    first_use = i
                last_use = i
        if first_use is not None:
            raw_live[name] = (first_use, last_use)
            # Expand range to cover enclosing loop ONLY if the array is
            # used both outside and inside the loop (persistent across
            # iterations, like Q). Arrays first allocated inside the loop
            # are loop-local and don't need expansion — they're rewritten
            # each iteration so their within-iteration range is sufficient.
            for loop_start, loop_end in loop_ranges:
                used_inside = first_use >= loop_start and last_use <= loop_end
                used_before = first_use < loop_start and last_use >= loop_start
                used_after = last_use > loop_end and first_use <= loop_end
                if used_before or used_after:
                    # Array spans into/out of the loop → expand to full loop
                    first_use = min(first_use, loop_start)
                    last_use = max(last_use, loop_end)
                # If used_inside only: keep the within-loop range as-is
            live[name] = (first_use, last_use)
        else:
            live[name] = (decl_line, decl_line)
            raw_live[name] = (decl_line, decl_line)

    def separates(first, last, scope):
        # A barrier inside a nested loop might execute zero times. Only one in
        # the same unconditional scope (or an enclosing scope) can certify this
        # transition. Conditional scopes were excluded when collecting barriers.
        return any(first < i < last and scope[: len(contexts[i])] == contexts[i] for i in barriers)

    # 3. Textual separation alone is not cross-thread read completion.
    def overlaps(a, b):
        a0, a1 = live[a]
        b0, b1 = live[b]
        if a0 <= b1 and b0 <= a1:
            return True
        if a0 > b0:
            a, b = b, a
        a0, a1 = raw_live[a]
        b0, b1 = raw_live[b]
        common = []
        for left, right in zip(contexts[a1], contexts[b0]):
            if left != right:
                break
            common.append(left)
        if not common or common[0] in transfers:
            return True
        forward = separates(a1, b0, tuple(common))
        if not forward:
            # An array used ONLY inside a loop can finish its last reads at an
            # unconditional barrier in that loop before exiting. Zero iterations
            # access no such array. Likewise a barrier at the beginning of B's
            # loop protects its first write. Do not credit an unrelated zero-trip
            # loop, or an array whose lifetime extends outside that loop.
            for i in barriers:
                scope = contexts[i]
                if not a1 < i < b0 or scope[: len(common)] != tuple(common):
                    continue
                extra = scope[len(common) :]
                if extra and all(
                    blocks[block][0] == "for"
                    and (block < a0 <= a1 < blocks[block][1] or block < b0 <= b1 < blocks[block][1])
                    for block in extra
                ):
                    forward = True
                    break
        if not forward:
            return True
        if not all(contexts[i] and contexts[i][0] == common[0] for i in (a0, a1, b0, b1)):
            return True
        # A->B in iteration n is insufficient: B(n)->A(n+1) must also be safe.
        # For every loop enclosing both whole lifetimes, require a barrier at
        # that loop's scope before A or after B. Inner zero-trip loops don't count.
        for level, block in enumerate(common):
            kind, end = blocks[block]
            if kind == "for" and block < a0 <= a1 < b0 <= b1 < end:
                scope = tuple(common[: level + 1])
                if not (separates(block, a0, scope) or separates(b1, end, scope)):
                    return True
        return False

    # 4. Greedy coloring: assign arrays to physical slots.
    # Sort by size descending so large arrays get first pick.
    names_sorted = sorted(decls.keys(), key=lambda n: decls[n][0], reverse=True)

    # Each slot: (physical_name, max_size, [member_names])
    slots = []
    assignment = {}  # name -> physical_name

    for name in names_sorted:
        size = decls[name][0]
        assigned = False
        for slot in slots:
            phys_name, slot_size, members = slot
            # Check if this array conflicts with any member
            conflicts = any(overlaps(name, m) for m in members)
            if not conflicts:
                # Assign to this slot
                members.append(name)
                if size > slot_size:
                    slot[1] = size  # update max size
                assignment[name] = phys_name
                assigned = True
                break
        if not assigned:
            # New slot
            slots.append([name, size, [name]])
            assignment[name] = name  # self-assigned

    # 5. Rename in MSL
    # Build replacement map: old_name -> new_name
    renames = {}
    for name in decls:
        if assignment[name] != name:
            renames[name] = assignment[name]

    if not renames:
        return msl  # no aliasing needed

    # Update declaration sizes to group maximum
    slot_max_size = {}
    for slot in slots:
        phys_name, max_size, members = slot
        slot_max_size[phys_name] = max_size

    # Process lines: rename, update sizes, remove duplicate declarations
    # Only alias arrays of the same dtype within a slot. Pre-filter slots so
    # each slot contains only same-dtype members (re-greedy over dtype groups
    # would be more optimal; this is a conservative correctness fix).
    # Here, enforce dtype homogeneity by splitting mixed slots before renaming.
    dtype_by_name = {n: decls[n][2] for n in decls}
    homogeneous_slots = []
    for slot in slots:
        phys_name, max_size, members = slot
        # Group members by dtype
        by_dtype = {}
        for m in members:
            by_dtype.setdefault(dtype_by_name[m], []).append(m)
        for dt, ms in by_dtype.items():
            size_max = max(decls[m][0] for m in ms)
            homogeneous_slots.append([ms[0], size_max, ms, dt])
    # Rebuild assignment from homogeneous slots
    assignment = {}
    slot_max_size = {}
    slot_dtype = {}
    for slot in homogeneous_slots:
        phys_name, max_size, members, dt = slot
        slot_max_size[phys_name] = max_size
        slot_dtype[phys_name] = dt
        for m in members:
            assignment[m] = phys_name
    renames = {n: assignment[n] for n in decls if assignment.get(n) != n}
    if allocation_aliases is not None:
        allocation_aliases.update(assignment)
    if not renames:
        return msl

    seen_decls = set()
    new_lines = []
    for i, line in enumerate(lines):
        m = decl_re.match(line)
        if m:
            dtype, name, _size = m.group(1), m.group(2), m.group(3)
            phys_name = assignment.get(name, name)
            if phys_name in seen_decls:
                continue  # remove duplicate declaration
            seen_decls.add(phys_name)
            # Update size and dtype to the slot's
            max_size = slot_max_size.get(phys_name, int(_size))
            phys_dtype = slot_dtype.get(phys_name, dtype)
            new_lines.append(f"    threadgroup {phys_dtype} {phys_name}[{max_size}];")
        else:
            new_line = line
            for old, new in renames.items():
                if old in new_line:
                    new_line = re.sub(r"\b" + re.escape(old) + r"\b", new, new_line)
            new_lines.append(new_line)

    return "\n".join(new_lines)


def _check_generic_shared_memory_budget(msl: str, *, limit=32768):
    """Refuse a generic shader whose static threadgroup arrays cannot fit, before a pipeline is asked for.

    The count is taken from the declarations of the FINAL text, after reuse. A
    type this does not know counts as nothing and padding is not counted, so the
    sum is a lower bound on what the pipeline will ask for: a lower bound above
    the limit is a proof of excess, and nothing below it is refused here.
    """
    from triton_msl.errors import MetalResourceError

    widths = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2, "half": 2, "int": 4, "uint": 4, "float": 4}
    lexical = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
    code = lexical.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), msl)
    required = sum(
        widths.get(dtype, 0) * int(count)
        for dtype, count in re.findall(r"^\s*threadgroup\s+(\w+)\s+\w+\s*\[(\d+)\]\s*;", code, re.M)
    )
    if required > limit:
        raise MetalResourceError(required, limit, "threadgroup memory bytes")
    return required


def _check_replay_shared_memory_budget(allocations, aliases, *, extra_bytes=0, limit=32768):
    """Check typed allocations AFTER the emitter's actual reuse plan.

    Do not infer capacity from a compiler diagnostic or count pre-alias logical
    arrays: the former can hide defects, the latter rejects legal shared reuse.
    Unknown facts fail as integrity errors, never prunable resource failures.
    This covers builder-owned arrays in the bounded source replay paths, not
    arbitrary hand-written MSL or every template's memory footprint.
    """
    from .msl_emitter import _msl_compute_type
    from triton_msl.errors import MetalNonRecoverableError, MetalResourceError

    widths = {
        "bool": 1,
        "char": 1,
        "uchar": 1,
        "short": 2,
        "ushort": 2,
        "half": 2,
        "bfloat": 2,
        "int": 4,
        "uint": 4,
        "float": 4,
        "long": 8,
        "ulong": 8,
    }
    logical = {}
    for name, dtype, count in allocations:
        ty = _msl_compute_type(dtype)
        if name in logical or ty not in widths or type(count) is not int or count <= 0:
            raise MetalNonRecoverableError(
                "source replay has an unproved threadgroup allocation", op_name="threadgroup"
            )
        logical[name] = (ty, count)
    if any(name not in logical or target not in logical for name, target in aliases.items()):
        raise MetalNonRecoverableError("source replay scratch alias has no typed allocation", op_name="threadgroup")
    physical = {}
    for name, (ty, count) in logical.items():
        target = aliases.get(name, name)
        if aliases.get(target, target) != target or logical[target][0] != ty:
            raise MetalNonRecoverableError(
                "source replay scratch alias has incompatible storage", op_name="threadgroup"
            )
        previous = physical.get(target, (ty, 0))
        physical[target] = (ty, max(previous[1], count))
    if type(extra_bytes) is not int or extra_bytes < 0:
        raise MetalNonRecoverableError("source replay has unproved extra scratch storage", op_name="threadgroup")
    required = extra_bytes + sum(widths[ty] * count for ty, count in physical.values())
    if required > limit:
        raise MetalResourceError(required, limit, "threadgroup memory bytes")
    return required
