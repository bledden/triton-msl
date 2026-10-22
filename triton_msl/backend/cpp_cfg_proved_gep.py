"""Narrow, proved GEP normalization for the opt-in scalar cleanup.

One-index [N x i8] and scalar GEP have identical offsets when allocation sizes
match. Prove scalar type from ALL result uses and a homogeneous root's uses.
Unknown/escaping pointers, chains, mixed types, flags, indices or address spaces
refuse rather than relying on a guessed element type. Scalar operations and
index SSA/width are never rewritten. Production adapter runs afterwards.
"""

import re


class UnprovedGEP(ValueError):
    pass


SIZE = {"i8": 1, "i16": 2, "half": 2, "i32": 4, "float": 4, "i64": 8}
AIR_LAYOUT = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
SSA = r"%[\w.]+"
ARRAY = re.compile(rf"\s*({SSA}) = getelementptr \[(\d+) x i8\], ptr ({SSA}), (i32|i64) ({SSA}|-?\d+)\s*$")
SCALAR = re.compile(rf"\s*({SSA}) = getelementptr (\w+), ptr ({SSA}), (i32|i64) ({SSA}|-?\d+)\s*$")
CAST = re.compile(rf"\s*({SSA}) = addrspacecast ptr addrspace\(1\) ({SSA}) to ptr\s*$")
LOAD = re.compile(rf"\s*{SSA} = load (?:volatile )?(\w+), ptr ({SSA})(?:, align \d+)?\s*$")
STORE = re.compile(rf"\s*store (\w+) \S+, ptr ({SSA})(?:, align \d+)?\s*$")


def normalize_array_geps(ir):
    lines = ir.splitlines(keepends=True)
    clean = [line.split(";", 1)[0].strip() for line in lines]
    arrays = {i: m for i, line in enumerate(clean) if (m := ARRAY.fullmatch(line))}
    # Malformed / more general device array GEPs must not silently pass through.
    for i, line in enumerate(clean):
        if re.search(r"getelementptr\s+(?:\w+\s+)*\[", line) and "addrspace(3)" not in line and i not in arrays:
            raise UnprovedGEP(f"unsupported array GEP: {line}")
    if not arrays:
        return ir
    if re.findall(r'^target datalayout = "([^"]+)"$', ir, re.M) != [AIR_LAYOUT]:
        raise UnprovedGEP("allocation-size proof requires the native emitter's pinned AIR layout")
    casts = {m[1]: m[2] for line in clean if (m := CAST.fullmatch(line))}
    definitions = [line for line in clean if line.startswith("define ")]
    if len(definitions) != 1:
        raise UnprovedGEP("requires exactly one kernel")
    roots = set(re.findall(rf"ptr addrspace\(1\) ({SSA})(?=,|\))", definitions[0]))
    geps = {i: m for i, line in enumerate(clean) if (m := ARRAY.fullmatch(line) or SCALAR.fullmatch(line))}
    refs = [{*re.findall(SSA, line)} for line in clean]

    def root_of(base):
        root = casts.get(base, base)
        if root not in roots:
            raise UnprovedGEP(f"unproved base {base}")
        return root

    def access_type(pointer, definition):
        types = set()
        for i, line in enumerate(clean):
            if pointer not in refs[i] or i == definition:
                continue
            m = LOAD.fullmatch(line) or STORE.fullmatch(line)
            if not m or m[2] != pointer or m[1] not in SIZE:
                raise UnprovedGEP(f"unproved result use {pointer}: {line}")
            types.add(m[1])
        if len(types) != 1:
            raise UnprovedGEP(f"missing or mixed access type for {pointer}: {types}")
        return next(iter(types))

    affected = {root_of(m[3]) for m in arrays.values()}
    root_types = {r: set() for r in affected}
    choices = {}
    for i, m in geps.items():
        # Leave unrelated scalar GEPs to the existing adapter.
        root = casts.get(m[3], m[3])
        if root not in affected:
            continue
        ty = access_type(m[1], i)
        if i in arrays:
            if int(m[2]) != SIZE[ty]:
                raise UnprovedGEP(f"array stride and access allocation size differ: {clean[i]}")
            choices[i] = ty
        elif m[2] != ty:
            raise UnprovedGEP(f"scalar GEP/access type mismatch: {clean[i]}")
        root_types[root].add(ty)
    for root in affected:
        aliases = {root} | {name for name, base in casts.items() if base == root}
        for i, line in enumerate(clean):
            if not refs[i].intersection(aliases):
                continue
            if line.startswith("define ") or CAST.fullmatch(line):
                continue
            if i in geps and geps[i][3] in aliases:
                continue
            m = LOAD.fullmatch(line) or STORE.fullmatch(line)
            if m and m[2] in aliases and m[1] in SIZE:
                root_types[root].add(m[1])
                continue
            raise UnprovedGEP(f"unproved root use {root}: {line}")
        if len(root_types[root]) != 1:
            raise UnprovedGEP(f"mixed root access types: {root_types[root]}")
    for i, ty in choices.items():
        # Change only the allocation type, never flags, pointer, index or width.
        lines[i] = lines[i].replace(f"[{arrays[i][2]} x i8]", ty, 1)
    return "".join(lines)


def lower_disjoint_or(ir):
    # Removing a poison-generating promise is a refinement: ordinary OR agrees
    # whenever disjoint OR is defined. Keep the instruction, operands and width.
    return re.sub(r"(?m)^(\s*%[\w.]+ = or) disjoint (i\d+ )", r"\1 \2", ir)
