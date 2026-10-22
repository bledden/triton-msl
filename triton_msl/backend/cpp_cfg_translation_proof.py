"""Positive translation proof for optimized opaque IR, run BEFORE the production typed-pointer adapter.

Conservative pre-adapter selection for the opt-in scalar cleanup:

    decision = select_lowering(optimized_opaque_ir, unoptimized_opaque_ir)
    decision.ir      -> the IR to hand to MetalBackend._opaque_to_typed_ptrs
    decision.route   -> "optimized" or "fallback"
    decision.reason  -> why the optimized IR was refused (None when accepted)

The proof is syntactic and closed: every device pointer in the kernel must be typable by the
production adapter's rules without guessing.  Anything else refuses, and the unoptimized
lowering is used.  A Metal compile failure is never the eligibility gate.

Proved subset (per addrspace(1) kernel argument, the "root"):
  * uses of the root or its `addrspacecast ... to ptr` alias are only: the cast itself,
    `getelementptr <scalar>, ptr <alias>, (i32|i64) <idx>`, `load <scalar>` / `store <scalar>`
    with the pointer operand;
  * every GEP element type and every access type on the root and on its GEP results is ONE
    scalar type (float/half/i8/i16/i32/i64), and each GEP result is used only by loads/stores
    of that type;
  * no `[N x i8]` GEP (must have been normalized), no `getelementptr i8` unless the root's
    proved type is i8, a GEP of a GEP result only with the same element type (the adapter types
    every derived pointer from the root), no pointer `select`/`phi`, no pointer passed to a
    `call`, no pointer stored as a value, no `inbounds`/multi-index forms.
addrspace(2) arguments (scalar buffers) may only be loaded as one scalar type.  Threadgroup
(addrspace(3)) forms are out of this proof's scope and refuse.
"""

from dataclasses import dataclass
import re

from .cpp_cfg_proved_gep import UnprovedGEP, lower_disjoint_or, normalize_array_geps

SSA = r"%[\w.]+"
SCALARS = ("float", "half", "i8", "i16", "i32", "i64")
_CAST = re.compile(rf"^({SSA}) = addrspacecast ptr addrspace\(1\) ({SSA}) to ptr$")
_GEP = re.compile(rf"^({SSA}) = getelementptr (\w+), ptr ({SSA}), (i32|i64) ({SSA}|-?\d+)$")
_LOAD = re.compile(rf"^({SSA}) = load (?:volatile )?(\w+), ptr ({SSA})(?:, align \d+)?$")
_CLOAD = re.compile(rf"^({SSA}) = load (\w+), ptr addrspace\(2\) ({SSA})(?:, align \d+)?$")
_STORE = re.compile(rf"^store (?:volatile )?(\w+) (\S+), ptr ({SSA})(?:, align \d+)?$")


class UnprovedTranslation(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    route: str
    ir: str
    reason: str | None


def _body_lines(ir):
    if len(re.findall(r"^define ", ir, re.M)) != 1 or "\n}" not in ir:
        raise UnprovedTranslation("exactly one complete kernel required")
    start, end = ir.index("define "), ir.index("\n}", ir.index("define "))
    header = ir[start : ir.index("{", start) + 1]
    lines = [line.split(";", 1)[0].strip() for line in ir[start:end].splitlines()[1:]]
    return header, [line for line in lines if line and not re.match(r"^[\w.]+:$", line)]


def prove_translatable(ir):
    """Raise UnprovedTranslation unless every device pointer is typable without guessing."""
    if re.search(r"getelementptr[^\n]*\[", ir):
        raise UnprovedTranslation("array GEP survived normalization")
    if "addrspace(3)" in ir:
        raise UnprovedTranslation("threadgroup memory is outside this proof")
    header, lines = _body_lines(ir)
    roots = re.findall(rf"ptr addrspace\(1\) ({SSA})", header)
    consts = re.findall(rf"ptr addrspace\(2\) ({SSA})", header)
    if len(re.findall(r"\bptr\b", header)) != len(roots) + len(consts):
        raise UnprovedTranslation("unproved pointer in kernel signature")
    alias_of = {}
    for line in lines:
        m = _CAST.match(line)
        if m:
            if m[2] not in roots:
                raise UnprovedTranslation(f"cast of a non-root pointer: {line}")
            alias_of[m[1]] = m[2]
    pointer_root, root_types = {r: r for r in roots}, {r: set() for r in roots}
    pointer_root.update(alias_of)
    for line in lines:
        m = _GEP.match(line)
        if not m:
            if line.startswith("getelementptr") or " = getelementptr" in line:
                raise UnprovedTranslation(f"unproved GEP form: {line}")
            continue
        result, elem, base = m[1], m[2], m[3]
        if elem not in SCALARS:
            raise UnprovedTranslation(f"non-scalar GEP element type: {line}")
        if base not in pointer_root:
            raise UnprovedTranslation(f"GEP base is not a kernel buffer: {line}")
        root = pointer_root[base]
        root_types[root].add(elem)
        pointer_root[result] = root
    const_types = {c: set() for c in consts}
    for line in lines:
        if _CAST.match(line) or _GEP.match(line):
            continue
        m = _CLOAD.match(line)
        if m:
            if m[3] not in const_types or m[2] not in SCALARS:
                raise UnprovedTranslation(f"unproved constant-buffer load: {line}")
            const_types[m[3]].add(m[2])
            continue
        if line.startswith("store ") and any(tok in pointer_root for tok in re.findall(SSA, line.split(", ptr")[0])):
            raise UnprovedTranslation(f"pointer stored as a value: {line}")
        m = _LOAD.match(line) or _STORE.match(line)
        if m:
            ty, pointer = (m[2], m[3]) if line.startswith("%") else (m[1], m[3])
            if pointer not in pointer_root:
                raise UnprovedTranslation(f"access through an unproved pointer: {line}")
            if ty not in SCALARS:
                raise UnprovedTranslation(f"non-scalar access type: {line}")
            if line.startswith("store") and m[2] in pointer_root:
                raise UnprovedTranslation(f"pointer stored as a value: {line}")
            root_types[pointer_root[pointer]].add(ty)
            continue
        used = set(re.findall(SSA, line)) & (set(pointer_root) | set(const_types))
        if used or re.search(r"\bptr\b", line):
            raise UnprovedTranslation(f"pointer escapes into an unproved use: {line}")
    for line in lines:
        m = _GEP.match(line)
        if m and m[2] == "i8" and root_types[pointer_root[m[3]]] != {"i8"}:
            raise UnprovedTranslation(f"byte-offset GEP on a non-byte buffer (InstCombine form): {line}")
    for root, types in root_types.items():
        if len(types) > 1:
            raise UnprovedTranslation(f"mixed access types on {root}: {sorted(types)}")
    for const, types in const_types.items():
        if len(types) > 1:
            raise UnprovedTranslation(f"mixed constant-buffer types on {const}: {sorted(types)}")
    return {root: next(iter(types), None) for root, types in root_types.items()}


def select_lowering(optimized_ir, unoptimized_ir):
    """Optimized IR only when normalized AND positively proved; otherwise the unoptimized lowering."""
    try:
        candidate = lower_disjoint_or(normalize_array_geps(optimized_ir))
        prove_translatable(candidate)
    except (UnprovedGEP, UnprovedTranslation) as error:
        return Decision("fallback", unoptimized_ir, f"{type(error).__name__}: {error}")
    return Decision("optimized", candidate, None)
