"""Walk TTGIR MLIR modules using Triton's Python bindings.

Replaces the text-based regex parsing in ttgir_parser.py with proper
MLIR programmatic access via module.walk().

Produces an IRGraph — a flat list of SSAValue nodes in topological order
with their operand IDs, attributes, and type info — suitable for
op-by-op lowering to MSL.

Hybrid approach: uses module.walk() for structure (ops, operands, types)
and str(module) for supplementary info (constant values, predicate names,
argument names) that the bindings don't expose.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SSAValue:
    """A single SSA value produced by an MLIR operation."""
    id: int                       # Unique ID from value.id() (first result)
    name: str                     # Generated MSL name: "v0", "v1", ...
    op: str                       # MLIR op name: "tt.load", "arith.addf"
    operand_ids: List[int]        # IDs of input SSA values
    attrs: Dict[str, Any]         # Attributes: {"start": 0, "end": 256, ...}
    type_str: str                 # Full type string: "tensor<256xf32>"
    elem_type: str                # Scalar element type: "f32", "i32"
    is_tensor: bool               # True for tensor values
    region_ops: Optional[List['SSAValue']] = None  # Body ops for reduce/for (or "then" for scf.if)
    else_ops: Optional[List['SSAValue']] = None  # "else" body ops for scf.if
    result_ids: Optional[List[int]] = None  # IDs of ALL results (multi-result ops)


@dataclass
class FuncArg:
    """A function argument (pointer or scalar)."""
    id: int           # SSA value ID
    name: str         # Argument name from MLIR (e.g., "x_ptr")
    type_str: str     # Full type string
    elem_type: str    # Element type: "f32", "f16", "i32"
    is_ptr: bool      # True for !tt.ptr types
    index: int        # Position in argument list


@dataclass
class CalledFunc:
    """A non-entry (noinline) function called via tt.call."""
    name: str                     # Mangled function name (e.g., "__main__.add_fn__fp32_fp32")
    args: List[FuncArg]           # Function arguments
    ops: List[SSAValue]           # Body ops in topological order
    return_types: List[str]       # Return types (e.g., ["f32"] or ["f32", "f32"])


@dataclass
class IRGraph:
    """Structured representation of a TTGIR kernel."""
    func_name: str
    args: List[FuncArg]
    ops: List[SSAValue]           # Top-level ops in topological order
    block_size: int = 256         # From tt.make_range end attribute
    num_warps: int = 4            # From module attribute
    called_funcs: Optional[List[CalledFunc]] = None  # Noinline function defs
    size_per_thread: Optional[List[int]] = None  # From #ttg.blocked layout
    # Module text — kept around so the lowerer can parse layout attributes
    # (#ttg.blocked, #ttg.linear, #ttg.slice) on demand. Populated by the
    # walker; defaults to empty for tests that hand-construct an IRGraph.
    mod_text: str = ""


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _extract_elem_type(type_str: str) -> str:
    """Extract scalar element type from an MLIR type string.

    Examples:
        "tensor<256xf32>" -> "f32"
        "tensor<256x!tt.ptr<f16>>" -> "f16"
        "!tt.ptr<f32>" -> "f32"
        "f32" -> "f32"
        "i32" -> "i32"
    """
    # Pointer type: !tt.ptr<f32>
    m = re.search(r"!tt\.ptr<(\w+)>", type_str)
    if m:
        return m.group(1)
    # Tensor of pointers: tensor<256x!tt.ptr<f32>>
    m = re.search(r"tensor<[^>]*x!tt\.ptr<(\w+)>", type_str)
    if m:
        return m.group(1)
    # Tensor type: tensor<256xf32> or tensor<128x64xf16>
    # Element types start with a letter (f32, i32, bf16), dimensions start with
    # a digit (2x4x...), so [a-z] distinguishes the element type from shape dims.
    m = re.search(r"x([a-z]\w*)(?:,\s*#[^>]*)?>", type_str)
    if m:
        return m.group(1)
    # Scalar type
    m = re.match(r"^(\w+)$", type_str.strip())
    if m:
        return m.group(1)
    return "f32"


def _is_tensor_type(type_str: str) -> bool:
    """Check if a type string represents a tensor."""
    return "tensor<" in type_str


def _is_ptr_type(type_str: str) -> bool:
    """Check if a type string is a pointer or tensor of pointers."""
    return "!tt.ptr" in type_str


def _extract_shape(type_str: str) -> tuple:
    """Extract tensor shape from an MLIR type string.

    Examples:
        "tensor<256xf32>" -> (256,)
        "tensor<32x64xf16>" -> (32, 64)
        "tensor<4x4x!tt.ptr<f32>>" -> (4, 4)
        "tensor<256xf32, #ttg.blocked<{...}>>" -> (256,)
        "f32" -> ()
        "!tt.ptr<f32>" -> ()
    """
    if "tensor<" not in type_str:
        return ()
    # Match one or more dimension groups: digit sequences followed by 'x'
    # Pattern: tensor<D1xD2x...xTYPE> — dimensions are always \d+x sequences
    m = re.search(r"tensor<((?:\d+x)+)", type_str)
    if m:
        dims_str = m.group(1).rstrip("x")  # e.g., "32x64" or "256"
        dims = tuple(int(d) for d in dims_str.split("x") if d)
        return dims
    return ()


# ---------------------------------------------------------------------------
# Module text parsing (for info not exposed by bindings)
# ---------------------------------------------------------------------------

def _parse_blocked_layout(mod_text: str) -> Optional[Dict[str, List[int]]]:
    """Extract sizePerThread, threadsPerWarp, warpsPerCTA from TTGIR text.

    Parses layout attributes like:
        #blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32],
                                 warpsPerCTA = [4], order = [0]}>

    Returns a dict with keys 'size_per_thread', 'threads_per_warp',
    'warps_per_cta' (lists of ints), or None if no blocked layout found.
    """
    m = re.search(
        r"#ttg\.blocked<\{[^}]*"
        r"sizePerThread\s*=\s*\[([^\]]+)\]"
        r"[^}]*threadsPerWarp\s*=\s*\[([^\]]+)\]"
        r"[^}]*warpsPerCTA\s*=\s*\[([^\]]+)\]",
        mod_text,
    )
    if not m:
        return None
    return {
        "size_per_thread": [int(x.strip()) for x in m.group(1).split(",")],
        "threads_per_warp": [int(x.strip()) for x in m.group(2).split(",")],
        "warps_per_cta": [int(x.strip()) for x in m.group(3).split(",")],
    }


class _ModuleTextIndex:
    """Pre-parsed index of module text for fast constant/predicate lookup.

    Since str(op) returns Python repr (not MLIR text), we parse the
    module text once and index constant values and predicate names
    by their SSA name for later lookup.
    """

    def __init__(self, module_text: str):
        self.text = module_text
        self.func_name = self._parse_func_name()
        self.arg_names = self._parse_arg_names()
        self.constants = self._parse_constants()
        self.constants_by_position = self._parse_constants_by_position()
        self.predicates = self._parse_predicates()
        self.atomic_ops = self._parse_atomic_ops()
        self.call_targets = self._parse_call_targets()
        self.func_defs = self._parse_func_defs()
        self.cond_br_ops = self._parse_cond_br_ops()
        self.extern_elementwise_ops = self._parse_extern_elementwise_ops()

    def _parse_func_name(self) -> str:
        m = re.search(
            r"(?:tt\.func|func\.func)\s+(?:public\s+)?@(\w+)",
            self.text
        )
        return m.group(1) if m else "kernel"

    def _parse_arg_names(self, func_name: str = None) -> List[str]:
        # The function signature contains loc(...) with nested parens,
        # so we can't use a simple [^)]* regex. Instead, find the
        # function keyword and then balanced-paren match the args.
        if func_name:
            pattern = rf"(?:tt\.func|func\.func)\s+(?:public\s+|private\s+)?@{re.escape(func_name)}\s*\("
        else:
            pattern = r"(?:tt\.func|func\.func)\s+(?:public\s+)?@\w+\s*\("
        m = re.search(pattern, self.text)
        if not m:
            return []
        start = m.end()  # Position right after the opening '('
        depth = 1
        i = start
        while i < len(self.text) and depth > 0:
            if self.text[i] == '(':
                depth += 1
            elif self.text[i] == ')':
                depth -= 1
            i += 1
        args_text = self.text[start:i - 1]  # Exclude the closing ')'
        # Arg names may contain '.' (e.g. tuple-flattened names like "Ptrs.0").
        return [am.group(1) for am in re.finditer(r"%([\w.]+)\s*:", args_text)]

    def _parse_call_targets(self) -> Dict[str, str]:
        """Parse tt.call ops and map SSA name -> callee function name.

        Matches patterns like:
            %0 = tt.call @callee_name(...) : ...
            tt.call @callee_name(...) : ...   (void return)
        """
        result = {}
        # With result(s):
        for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*tt\.call\s+@([^\s(]+)",
            self.text
        ):
            result[m.group(1)] = m.group(2)
        # Void calls (no result):
        for m in re.finditer(
            r"^\s*tt\.call\s+@([^\s(]+)",
            self.text, re.MULTILINE
        ):
            # Use function name as key for void calls
            result[f"_void_call_{m.group(1)}"] = m.group(1)
        return result

    def _parse_func_defs(self) -> Dict[str, Dict]:
        """Parse all tt.func definitions (public and private).

        Returns a dict mapping function name -> {
            'is_public': bool,
            'arg_names': list of arg names,
            'arg_types': list of arg type strings,
            'return_types': list of return type strings,
        }
        """
        result = {}
        for m in re.finditer(
            r"tt\.func\s+(public|private)\s+@(\S+)\s*\(",
            self.text
        ):
            visibility = m.group(1)
            func_name = m.group(2)

            # Parse args (balanced paren matching)
            start = m.end()
            depth = 1
            i = start
            while i < len(self.text) and depth > 0:
                if self.text[i] == '(':
                    depth += 1
                elif self.text[i] == ')':
                    depth -= 1
                i += 1
            args_text = self.text[start:i - 1]

            # Extract arg names and types. Arg names may contain '.'
            # (e.g. tuple-flattened names like "Ptrs.0").
            arg_names = [am.group(1) for am in re.finditer(r"%([\w.]+)\s*:", args_text)]

            # Extract arg types
            arg_types = []
            for am in re.finditer(r"%[\w.]+\s*:\s*([^\s{,]+(?:<[^>]+>)?)", args_text):
                arg_types.append(am.group(1))

            # Parse return types: ) -> TYPE or ) -> (TYPE, TYPE)
            after_args = self.text[i:i+200]  # look at text after closing paren
            return_types = []
            rm = re.match(r"\s*->\s*\(([^)]+)\)", after_args)
            if rm:
                # Multiple return types: -> (f32, f32)
                for rt in rm.group(1).split(","):
                    return_types.append(rt.strip())
            else:
                rm = re.match(r"\s*->\s*(\S+)", after_args)
                if rm:
                    return_types.append(rm.group(1))

            result[func_name] = {
                'is_public': visibility == 'public',
                'arg_names': arg_names,
                'arg_types': arg_types,
                'return_types': return_types,
            }

        return result

    def _parse_constants(self) -> Dict[str, Any]:
        """Parse all arith.constant ops and map SSA name -> value.

        Note: When multiple functions define constants with the same SSA
        name (e.g., %cst in function A and %cst in function B), the dict
        will only contain the last value. Use constants_by_position for
        position-based lookup in multi-function modules.
        """
        result = {}
        # Match: %name = arith.constant VALUE : TYPE
        # or: %name = arith.constant dense<VALUE> : tensor<...>
        # SSA names may contain `-`, `.`, and `$` in addition to \w — MLIR
        # constants with negative values often get printed as e.g. `%c-123_i32`.
        for m in re.finditer(
            r"%([\w.$-]+(?::\d+)?)\s*=\s*arith\.constant\s+(.+?)(?:\s+loc\(|$)",
            self.text, re.MULTILINE
        ):
            ssa_name = m.group(1)
            rest = m.group(2).strip()

            # Dense constant: dense<VALUE> : tensor<...>
            dm = re.match(r"dense<([^>]+)>", rest)
            if dm:
                val_str = dm.group(1).strip()
                result[ssa_name] = _try_parse_number(val_str)
                continue

            # Scalar constant: VALUE : TYPE
            sm = re.match(r"(.+?)\s*:\s*\S+", rest)
            if sm:
                val_str = sm.group(1).strip()
                if val_str == "true":
                    result[ssa_name] = True
                elif val_str == "false":
                    result[ssa_name] = False
                else:
                    result[ssa_name] = _try_parse_number(val_str)

        return result

    def _parse_constants_by_position(self) -> List[Any]:
        """Parse all arith.constant ops and return values in text order.

        Unlike _parse_constants() which maps by SSA name (lossy for
        duplicate names across functions), this returns a list indexed
        by text position, suitable for walk-order matching.
        """
        result = []
        # SSA names may contain `-`, `.`, and `$` in addition to \w — MLIR
        # constants with negative values often get printed as e.g. `%c-123_i32`.
        for m in re.finditer(
            r"%([\w.$-]+(?::\d+)?)\s*=\s*arith\.constant\s+(.+?)(?:\s+loc\(|$)",
            self.text, re.MULTILINE
        ):
            rest = m.group(2).strip()

            dm = re.match(r"dense<([^>]+)>", rest)
            if dm:
                result.append(_try_parse_number(dm.group(1).strip()))
                continue

            sm = re.match(r"(.+?)\s*:\s*\S+", rest)
            if sm:
                val_str = sm.group(1).strip()
                if val_str == "true":
                    result.append(True)
                elif val_str == "false":
                    result.append(False)
                else:
                    result.append(_try_parse_number(val_str))
            else:
                result.append(0)

        return result

    def _parse_predicates(self) -> Dict[str, str]:
        """Parse comparison predicate names: SSA name -> predicate."""
        result = {}
        for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*arith\.cmp[if]\s+(\w+)",
            self.text
        ):
            result[m.group(1)] = m.group(2)
        return result

    def _parse_atomic_ops(self) -> Dict[str, Dict[str, str]]:
        """Parse tt.atomic_rmw and tt.atomic_cas ops: SSA name -> attributes.

        The MLIR text for atomics looks like:
            %0 = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val, %mask : ...
            %0 = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val : ...

        Returns dict mapping SSA name to {"rmw_op": ..., "sem": ...} for
        atomic_rmw, or {"sem": ...} for atomic_cas.
        """
        result = {}
        # tt.atomic_rmw: %name = tt.atomic_rmw OP, SEM, SCOPE, ...
        for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*tt\.atomic_rmw\s+(\w+),\s*(\w+),\s*(\w+)",
            self.text
        ):
            result[m.group(1)] = {
                "rmw_op": m.group(2),
                "sem": m.group(3),
                "scope": m.group(4),
            }
        # tt.atomic_cas: %name = tt.atomic_cas SEM, SCOPE, ...
        for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*tt\.atomic_cas\s+(\w+),\s*(\w+)",
            self.text
        ):
            result[m.group(1)] = {
                "sem": m.group(2),
                "scope": m.group(3),
            }
        return result

    def _parse_extern_elementwise_ops(self) -> List[Dict[str, Any]]:
        """Parse tt.extern_elementwise ops and return attrs in text order.

        The TTGIR text typically looks like:
            %0 = tt.extern_elementwise %arg {symbol = "func", libname = "lib", pure = true} : ...

        Returns list of dicts with 'symbol' and 'libname' keys in text order,
        suitable for walk-order matching.
        """
        results = []
        for m in re.finditer(
            r'tt\.extern_elementwise\b[^{]*\{([^}]*)\}',
            self.text
        ):
            attrs_text = m.group(1)
            info = {}
            # Extract symbol = "..."
            sm = re.search(r'symbol\s*=\s*"([^"]*)"', attrs_text)
            if sm:
                info["symbol"] = sm.group(1)
            # Extract libname = "..."
            lm = re.search(r'libname\s*=\s*"([^"]*)"', attrs_text)
            if lm:
                info["libname"] = lm.group(1)
            # Extract pure = true/false
            pm = re.search(r'pure\s*=\s*(true|false)', attrs_text)
            if pm:
                info["pure"] = pm.group(1) == "true"
            results.append(info)
        return results

    def _parse_cond_br_ops(self) -> list:
        """Parse cf.cond_br ops to extract branch arg counts.

        Returns list of (n_true_args, n_false_args) in text order.
        Used to split cf.cond_br operand_ids into condition, true args, false args.
        """
        results = []
        pattern = r'cf\.cond_br\s+%[^\s,]+\s*,\s*\^\w+(\([^)]*\))?\s*,\s*\^\w+(\([^)]*\))?'
        for m in re.finditer(pattern, self.text):
            true_args_str = m.group(1) or ""
            false_args_str = m.group(2) or ""
            n_true = true_args_str.count('%')
            n_false = false_args_str.count('%')
            results.append((n_true, n_false))
        return results


def _try_parse_number(s: str) -> Any:
    """Try to parse a string as int or float.

    Handles MLIR hex float bit patterns (e.g., 0x7F800000 for +inf).
    """
    try:
        return int(s)
    except ValueError:
        pass
    # Hex integer that may be an IEEE 754 float bit pattern
    if s.startswith("0x") or s.startswith("0X"):
        try:
            return int(s, 16)
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# SSA name extraction from module text
# ---------------------------------------------------------------------------

def _build_result_id_to_ssa_name(module_text: str) -> Dict[str, str]:
    """Map result SSA names to their position for cross-referencing.

    We can't directly map value.id() to SSA name from bindings alone,
    but we can match by operation order since walk() and text are both
    in forward order within the entry block.
    """
    # This is used for constant lookups — we match by walk order position
    return {}


# ---------------------------------------------------------------------------
# MLIR Walker
# ---------------------------------------------------------------------------

class MLIRWalker:
    """Walk a TTGIR MLIR module and produce an IRGraph."""

    def __init__(self, module, options=None):
        self.module = module
        self.options = options
        self._var_counter = 0
        self._value_map = {}      # value.id() -> SSAValue
        self._func_args = []      # FuncArg list
        self._block_size = 256
        self._num_warps = getattr(options, 'num_warps', 4) if options else 4

        # Parse module text once for supplementary info
        self._mod_text = str(module)
        self._text_index = _ModuleTextIndex(self._mod_text)
        self._layout = _parse_blocked_layout(self._mod_text)

        # Constant matching: we match constants by walk order
        # Walk visits entry block ops in forward order, same as text
        self._constant_names_in_order = self._get_constant_ssa_names_in_order()
        self._constant_walk_index = 0

        # Predicate matching: same approach
        self._predicate_names_in_order = self._get_predicate_ssa_names_in_order()
        self._predicates_in_order = self._get_predicates_in_order()
        self._predicate_walk_index = 0

        # Atomic op matching: same approach
        self._atomic_rmw_names_in_order = self._get_atomic_rmw_ssa_names_in_order()
        self._atomic_rmw_walk_index = 0
        self._atomic_cas_names_in_order = self._get_atomic_cas_ssa_names_in_order()
        self._atomic_cas_walk_index = 0

        # Call target matching: map walk order -> callee name
        self._call_targets_in_order = self._get_call_targets_in_order()
        self._call_walk_index = 0

        # cf.cond_br matching: branch arg counts by text order
        self._cond_br_walk_index = 0

        # tt.extern_elementwise matching: symbol/libname by text order
        self._extern_elementwise_walk_index = 0

    def _get_constant_ssa_names_in_order(self) -> List[str]:
        r"""Get arith.constant SSA names in text order.

        SSA names may contain `-`, `.`, and `$` in addition to \w — MLIR
        constants with negative values often get printed as `%c-123_i32`.
        """
        return [m.group(1) for m in re.finditer(
            r"%([\w.$-]+(?::\d+)?)\s*=\s*arith\.constant",
            self._mod_text
        )]

    def _get_predicate_ssa_names_in_order(self) -> List[str]:
        """Get arith.cmp* SSA names in text order."""
        return [m.group(1) for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*arith\.cmp[if]",
            self._mod_text
        )]

    def _get_predicates_in_order(self) -> List[str]:
        """Get arith.cmp* PREDICATES in text order, parallel to the SSA-name
        list above. Used positionally (by walk order) instead of keying by SSA
        name: scf.while/scf.if regions reuse local SSA names (%0 in both before
        and after), so a name->predicate dict collides and the later region
        overwrites the earlier one — corrupting the loop condition. Same anchored
        regex as the name list so the two stay index-aligned."""
        return [m.group(2) for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*arith\.cmp[if]\s+(\w+)",
            self._mod_text
        )]

    def _get_atomic_rmw_ssa_names_in_order(self) -> List[str]:
        """Get tt.atomic_rmw SSA names in text order."""
        return [m.group(1) for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*tt\.atomic_rmw",
            self._mod_text
        )]

    def _get_call_targets_in_order(self) -> List[str]:
        """Get tt.call callee names in text order.

        Matches both void and non-void calls:
            tt.call @callee(...)     -> void
            %x = tt.call @callee(...) -> non-void
        """
        targets = []
        for m in re.finditer(r"tt\.call\s+@(\S+)\(", self._mod_text):
            targets.append(m.group(1))
        return targets

    def _get_atomic_cas_ssa_names_in_order(self) -> List[str]:
        """Get tt.atomic_cas SSA names in text order."""
        return [m.group(1) for m in re.finditer(
            r"%(\w+(?::\d+)?)\s*=\s*tt\.atomic_cas",
            self._mod_text
        )]

    def _next_var(self) -> str:
        name = f"v{self._var_counter}"
        self._var_counter += 1
        return name

    def walk(self) -> IRGraph:
        """Walk the module and return an IRGraph."""
        # Collect ops via walk
        entry_block_id, top_ops, nested_ops, block_args_map, callee_funcs_raw = self._collect_ops()

        # Extract function args from entry block
        self._extract_func_args(entry_block_id)

        # Attach nested ops to parent ops and extract block arguments
        self._attach_nested_ops(top_ops, nested_ops, block_args_map)

        # Build CalledFunc objects for callee functions
        called_funcs = self._build_called_funcs(callee_funcs_raw, nested_ops, block_args_map)

        return IRGraph(
            func_name=self._text_index.func_name,
            args=self._func_args,
            ops=top_ops,
            block_size=self._block_size,
            num_warps=self._num_warps,
            called_funcs=called_funcs if called_funcs else None,
            size_per_thread=self._layout["size_per_thread"] if self._layout else None,
            mod_text=self._mod_text,
        )

    def _attach_nested_ops(self, ops, nested_ops, block_args_map):
        """Recursively attach nested ops to parent ops and extract block args."""
        for ssa in ops:
            pid = ssa.attrs.get("_parent_id")
            if pid is not None and pid in nested_ops:
                all_ops = nested_ops[pid]

                # Collect distinct block IDs from children
                block_ids_seen = []
                for child in all_ops:
                    bid = child.attrs.get("_block_id")
                    if bid is not None and bid not in block_ids_seen:
                        block_ids_seen.append(bid)

                # Attach block arguments from the first block that has them
                if ssa.op in ("scf.for", "tt.reduce", "tt.scan", "tt.map_elementwise"):
                    for bid in block_ids_seen:
                        args = block_args_map.get(bid, [])
                        if args:
                            ssa.attrs["block_arg_ids"] = args
                            break

                # For scf.if: separate ops into then_ops and else_ops
                if ssa.op == "scf.if":
                    if len(block_ids_seen) >= 2:
                        then_bid = block_ids_seen[0]
                        else_bid = block_ids_seen[1]
                        then_ops = [c for c in all_ops if c.attrs.get("_block_id") == then_bid]
                        else_ops = [c for c in all_ops if c.attrs.get("_block_id") == else_bid]
                        ssa.region_ops = then_ops
                        ssa.else_ops = else_ops
                    else:
                        ssa.region_ops = all_ops
                elif ssa.op == "scf.while":
                    # scf.while has two regions: "before" (condition) and "after" (body)
                    # Separate them by block_id ordering — first block = before, second = after
                    if len(block_ids_seen) >= 2:
                        before_bid = block_ids_seen[0]
                        after_bid = block_ids_seen[1]
                        before_ops = [c for c in all_ops if c.attrs.get("_block_id") == before_bid]
                        after_ops = [c for c in all_ops if c.attrs.get("_block_id") == after_bid]
                        ssa.region_ops = before_ops   # "before" region (condition)
                        ssa.else_ops = after_ops      # "after" region (body)
                        # Block args for "before" region
                        before_args = block_args_map.get(before_bid, [])
                        if before_args:
                            ssa.attrs["block_arg_ids"] = before_args
                        # Block args for "after" region
                        after_args = block_args_map.get(after_bid, [])
                        if after_args:
                            ssa.attrs["else_block_arg_ids"] = after_args
                    else:
                        ssa.region_ops = all_ops
                        # Single block — attach its args
                        for bid in block_ids_seen:
                            args = block_args_map.get(bid, [])
                            if args:
                                ssa.attrs["block_arg_ids"] = args
                                break
                else:
                    ssa.region_ops = all_ops

                # Recurse into nested ops that themselves have children
                self._attach_nested_ops(all_ops, nested_ops, block_args_map)

    def _collect_ops(self):
        """Walk all ops and categorize them.

        Strategy: the walk is post-order, so nested body ops (reduce, for)
        appear before their parent op. We track block IDs to distinguish
        entry-block ops from nested ops. For each block, we record its
        parent region ID (via block.get_parent()) so that nested
        region-owning ops (scf.for inside scf.for) can correctly claim
        their body ops via region containment.

        Child scoping for nested region-containing ops: when a nested scf.if/for
        pops its children, it uses block_first_seen ordering to only claim ops
        in blocks that were first visited AFTER its own block was first visited.
        This prevents sibling blocks (e.g., outer then block) from being grabbed
        by a nested region-containing op (e.g., inner scf.if in the else block).

        Block argument extraction: when we encounter the first op in a new
        nested block, we call block.get_num_arguments() / get_argument(i) to
        capture the block's arguments (e.g., induction variable and iter_args
        for scf.for, lhs/rhs for tt.reduce). These are stored in block_args_map
        and later attached to the parent op.

        Multi-function support: when the module contains multiple tt.func
        definitions (noinline functions), the post-order walk visits each
        function's body ops before the tt.func op itself. We use tt.func
        boundaries to separate entry function ops from callee function ops.
        """
        import os as _os
        top_level = []
        pending_stack = [[]]
        nested = {}
        parent_counter = [0]
        entry_block_id = [None]
        entry_block_ref = [None]
        block_args_map = {}
        # Track when each block was first seen (walk order)
        block_first_seen = {}
        # Map block_id -> parent region id (for nested-region containment checks)
        block_parent_region = {}
        walk_counter = [0]

        # Multi-function tracking: collect callee function data
        # Each callee is stored as (func_name, is_public, block_ref, ops_list)
        callee_funcs_raw = []
        # Track all block IDs that belong to the entry function
        entry_func_block_ids = set()
        # Flag: have we finished processing the entry function?
        entry_func_done = [False]
        # Current callee function's block IDs and ops
        current_callee_block_ids = [set()]
        current_callee_ops = [[]]
        # Per-callee nesting state (mirrors entry function nesting logic)
        callee_pending_stack = [[[]]]  # stack of pending op lists
        callee_nested = [{}]           # parent_id -> children ops
        callee_entry_block_id = [None] # entry block of current callee
        callee_block_first_seen = [{}] # block_id -> walk order
        callee_block_parent_region = [{}] # block_id -> parent region id

        def walk_fn(op):
            name = op.get_name()

            # Handle tt.func/func.func: marks the end of a function's body
            # (post-order: body ops come before the func op itself)
            if name in ("tt.func", "func.func"):
                if not entry_func_done[0]:
                    # This is the entry function (public kernel)
                    entry_func_done[0] = True
                else:
                    # This is a callee function — collect its accumulated ops
                    # The ops in current_callee_ops belong to this callee
                    callee_ops = list(current_callee_ops[0])
                    callee_block_ids = set(current_callee_block_ids[0])
                    callee_nested_copy = dict(callee_nested[0])

                    callee_funcs_raw.append({
                        'ops': callee_ops,
                        'block_ids': callee_block_ids,
                        'nested': callee_nested_copy,
                    })

                    # Reset for next callee
                    current_callee_ops[0] = []
                    current_callee_block_ids[0] = set()
                    callee_pending_stack[0] = [[]]
                    callee_nested[0] = {}
                    callee_entry_block_id[0] = None
                    callee_block_first_seen[0] = {}
                    callee_block_parent_region[0] = {}
                return

            # Skip other structural ops
            if name in ("builtin.module", "module"):
                return

            block = op.get_block()
            block_id = block.id() if block is not None else None

            # Detect entry block from first op
            if entry_block_id[0] is None and block is not None:
                entry_block_id[0] = block_id
                entry_block_ref[0] = block
                entry_func_block_ids.add(block_id)

            # Determine if this op belongs to the entry function or a callee
            if entry_func_done[0]:
                # We're past the entry function — this op belongs to a callee
                if block_id is not None:
                    current_callee_block_ids[0].add(block_id)

                    # Detect callee's entry block from first op
                    if callee_entry_block_id[0] is None:
                        callee_entry_block_id[0] = block_id

                    # Track block first-seen order and parent region for callee
                    if block_id not in callee_block_first_seen[0]:
                        callee_block_first_seen[0][block_id] = walk_counter[0]
                        try:
                            parent_region = block.get_parent()
                            callee_block_parent_region[0][block_id] = parent_region.id()
                        except Exception:
                            callee_block_parent_region[0][block_id] = None

                    # Extract block arguments for callee blocks
                    if block_id not in block_args_map and block is not None:
                        try:
                            n_args = block.get_num_arguments()
                            arg_ids = []
                            for i in range(n_args):
                                arg = block.get_argument(i)
                                arg_ids.append(arg.id())
                            block_args_map[block_id] = arg_ids
                        except Exception:
                            block_args_map[block_id] = []

                # Build SSA value for callee ops
                if name in ("tt.reduce.return",):
                    walk_counter[0] += 1
                    return

                callee_is_nested = (block_id is not None and
                                    block_id != callee_entry_block_id[0])

                if name in ("scf.yield", "scf.condition", "tt.scan.return"):
                    ssa = self._make_ssa_value(op, name)
                    ssa.attrs["_block_id"] = block_id
                    ssa.attrs["_walk_order"] = walk_counter[0]
                    walk_counter[0] += 1
                    if callee_is_nested:
                        callee_pending_stack[0][-1].append(ssa)
                    else:
                        current_callee_ops[0].append(ssa)
                    return

                ssa = self._make_ssa_value(op, name)
                ssa.attrs["_block_id"] = block_id
                ssa.attrs["_walk_order"] = walk_counter[0]
                walk_counter[0] += 1

                if callee_is_nested:
                    if name in ("tt.reduce", "tt.scan", "scf.for", "scf.if", "scf.while", "tt.map_elementwise"):
                        pid = parent_counter[0]
                        parent_counter[0] += 1
                        ssa.attrs["_parent_id"] = pid

                        # Collect own region IDs for containment check
                        own_region_ids = set()
                        try:
                            for ri in range(op.get_num_regions()):
                                own_region_ids.add(op.get_region(ri).id())
                        except Exception:
                            pass

                        # Scope children via region parent (robust for nested
                        # scf.for / scf.if)
                        current_pending = callee_pending_stack[0][-1]
                        children = []
                        remaining = []
                        for pending_op in current_pending:
                            pb = pending_op.attrs.get("_block_id")
                            pr = callee_block_parent_region[0].get(pb)
                            if pb != block_id and pr is not None and pr in own_region_ids:
                                children.append(pending_op)
                            else:
                                remaining.append(pending_op)

                        callee_pending_stack[0][-1] = remaining
                        if children:
                            callee_nested[0][pid] = children
                        callee_pending_stack[0][-1].append(ssa)
                    else:
                        callee_pending_stack[0][-1].append(ssa)
                else:
                    # Callee entry-block op
                    if name in ("tt.reduce", "tt.scan", "scf.for", "scf.if", "scf.while", "tt.map_elementwise"):
                        pid = parent_counter[0]
                        parent_counter[0] += 1
                        ssa.attrs["_parent_id"] = pid
                        current_pending = callee_pending_stack[0].pop()
                        if current_pending:
                            callee_nested[0][pid] = current_pending
                        callee_pending_stack[0].append([])
                    current_callee_ops[0].append(ssa)
                return

            # Entry function op processing (original logic)
            is_nested = (block_id is not None and block_id != entry_block_id[0])
            if is_nested:
                entry_func_block_ids.add(block_id)

            # Track block first-seen order and parent region id
            if block_id is not None and block_id not in block_first_seen:
                block_first_seen[block_id] = walk_counter[0]
                try:
                    parent_region = block.get_parent()
                    block_parent_region[block_id] = parent_region.id()
                except Exception:
                    block_parent_region[block_id] = None

            # Extract block arguments the first time we see a nested block
            if is_nested and block_id not in block_args_map and block is not None:
                try:
                    n_args = block.get_num_arguments()
                    arg_ids = []
                    for i in range(n_args):
                        arg = block.get_argument(i)
                        arg_ids.append(arg.id())
                    block_args_map[block_id] = arg_ids
                except Exception:
                    block_args_map[block_id] = []

            # Skip reduce return (terminator in reduce body)
            if name in ("tt.reduce.return",):
                walk_counter[0] += 1
                return

            if name in ("scf.yield", "scf.condition", "tt.scan.return"):
                ssa = self._make_ssa_value(op, name)
                ssa.attrs["_block_id"] = block_id
                ssa.attrs["_walk_order"] = walk_counter[0]
                walk_counter[0] += 1
                if is_nested:
                    pending_stack[-1].append(ssa)
                else:
                    top_level.append(ssa)
                return

            ssa = self._make_ssa_value(op, name)
            ssa.attrs["_block_id"] = block_id
            ssa.attrs["_walk_order"] = walk_counter[0]
            walk_counter[0] += 1

            if is_nested:
                if name in ("tt.reduce", "tt.scan", "scf.for", "scf.if", "scf.while", "tt.map_elementwise"):
                    pid = parent_counter[0]
                    parent_counter[0] += 1
                    ssa.attrs["_parent_id"] = pid

                    # Collect this op's own region IDs (the regions it contains)
                    own_region_ids = set()
                    try:
                        for ri in range(op.get_num_regions()):
                            own_region_ids.add(op.get_region(ri).id())
                    except Exception:
                        pass

                    # Scope children: claim pending ops whose block's parent
                    # region is one of our regions. This is robust for deeply
                    # nested scf.for / scf.if, because nested region-owning
                    # ops are themselves claimed as direct children (their
                    # own bodies have already been claimed recursively before
                    # this op is post-order visited).
                    current_pending = pending_stack[-1]
                    children = []
                    remaining = []
                    for pending_op in current_pending:
                        pb = pending_op.attrs.get("_block_id")
                        pr = block_parent_region.get(pb)
                        if pb != block_id and pr is not None and pr in own_region_ids:
                            children.append(pending_op)
                        else:
                            remaining.append(pending_op)

                    pending_stack[-1] = remaining
                    if children:
                        nested[pid] = children
                    pending_stack[-1].append(ssa)
                else:
                    pending_stack[-1].append(ssa)
            else:
                # Entry-block op
                if name in ("tt.reduce", "tt.scan", "scf.for", "scf.if", "scf.while", "tt.map_elementwise"):
                    pid = parent_counter[0]
                    parent_counter[0] += 1
                    ssa.attrs["_parent_id"] = pid
                    current_pending = pending_stack.pop()
                    if current_pending:
                        nested[pid] = current_pending
                    pending_stack.append([])
                top_level.append(ssa)

        self.module.walk(walk_fn)

        return entry_block_ref[0], top_level, nested, block_args_map, callee_funcs_raw

    def _build_called_funcs(self, callee_funcs_raw, nested_ops, block_args_map):
        """Build CalledFunc objects from raw callee function data.

        Uses the text-parsed function definitions to get function names,
        argument info, and return types. Matches callee ops to their function
        definitions by order (walk order matches text order).
        """
        if not callee_funcs_raw:
            return []

        # Get non-public function names from text parsing
        func_defs = self._text_index.func_defs
        private_funcs = [
            (name, info) for name, info in func_defs.items()
            if not info['is_public']
        ]

        called_funcs = []
        for i, raw in enumerate(callee_funcs_raw):
            if i >= len(private_funcs):
                break

            func_name, func_info = private_funcs[i]
            ops = raw['ops']
            callee_block_ids = raw['block_ids']

            # Build function arguments from block_args_map
            # The callee's entry block is the first block in callee_block_ids
            args = []
            arg_names = func_info.get('arg_names', [])
            arg_types = func_info.get('arg_types', [])

            # Find the callee's entry block (the one with block args)
            for bid in callee_block_ids:
                if bid in block_args_map and block_args_map[bid]:
                    arg_ids = block_args_map[bid]
                    for j, arg_id in enumerate(arg_ids):
                        a_name = arg_names[j] if j < len(arg_names) else f"arg{j}"
                        a_type = arg_types[j] if j < len(arg_types) else "f32"
                        elem_type = _extract_elem_type(a_type)
                        is_ptr = _is_ptr_type(a_type)
                        fa = FuncArg(
                            id=arg_id,
                            name=a_name,
                            type_str=a_type,
                            elem_type=elem_type,
                            is_ptr=is_ptr,
                            index=j,
                        )
                        args.append(fa)
                        # Register in value map so callee ops can resolve operands
                        self._value_map[arg_id] = SSAValue(
                            id=arg_id,
                            name=a_name,
                            op="func_arg",
                            operand_ids=[],
                            attrs={"index": j, "is_ptr": is_ptr},
                            type_str=a_type,
                            elem_type=elem_type,
                            is_tensor=_is_tensor_type(a_type),
                        )
                    break

            # Attach nested ops to parent ops within callee (for scf.if/for inside callees)
            # Use the callee's own nested dict, not the entry function's
            callee_nested_ops = raw.get('nested', {})
            self._attach_nested_ops(ops, callee_nested_ops, block_args_map)

            return_types = func_info.get('return_types', [])

            called_funcs.append(CalledFunc(
                name=func_name,
                args=args,
                ops=ops,
                return_types=return_types,
            ))

        return called_funcs

    def _extract_func_args(self, entry_block):
        """Extract function arguments from the entry block."""
        if entry_block is None:
            return

        n_args = entry_block.get_num_arguments()
        arg_names = self._text_index.arg_names

        for i in range(n_args):
            arg = entry_block.get_argument(i)
            arg_id = arg.id()
            type_str = str(arg.get_type())
            elem_type = _extract_elem_type(type_str)
            is_ptr = _is_ptr_type(type_str)
            name = arg_names[i] if i < len(arg_names) else f"arg{i}"
            # Triton frontend flattens tuple args into dot-indexed names
            # (e.g. `Ptrs.0`). Dots are not valid in C identifiers, so
            # rewrite to underscores for the MSL emission.
            if "." in name:
                name = name.replace(".", "_")

            func_arg = FuncArg(
                id=arg_id,
                name=name,
                type_str=type_str,
                elem_type=elem_type,
                is_ptr=is_ptr,
                index=i,
            )
            self._func_args.append(func_arg)

            # Register in value map for operand resolution
            self._value_map[arg_id] = SSAValue(
                id=arg_id,
                name=name,
                op="func_arg",
                operand_ids=[],
                attrs={"index": i, "is_ptr": is_ptr},
                type_str=type_str,
                elem_type=elem_type,
                is_tensor=_is_tensor_type(type_str),
            )

    def _make_ssa_value(self, op, name: str) -> SSAValue:
        """Create an SSAValue from an MLIR operation."""
        # Collect operand IDs
        operand_ids = []
        for i in range(op.get_num_operands()):
            operand_ids.append(op.get_operand(i).id())

        # Get ALL result IDs (multi-result ops like scf.for)
        result_ids = []
        result_id = None
        type_str = ""
        n_results = op.get_num_results()
        for r in range(n_results):
            rid = op.get_result(r).id()
            result_ids.append(rid)
            if r == 0:
                result_id = rid
                type_str = str(op.get_result(r).get_type())

        # Extract attributes
        attrs = self._extract_attrs(op, name)

        # Block arg extraction is done in _collect_ops via block.get_argument()
        # and attached to parent ops in _attach_nested_ops.

        elem_type = _extract_elem_type(type_str) if type_str else "f32"
        is_tensor = _is_tensor_type(type_str) if type_str else False
        var_name = self._next_var()

        ssa = SSAValue(
            id=result_id if result_id is not None else -self._var_counter,
            name=var_name,
            op=name,
            operand_ids=operand_ids,
            attrs=attrs,
            type_str=type_str,
            elem_type=elem_type,
            is_tensor=is_tensor,
            result_ids=result_ids if len(result_ids) > 1 else None,
        )

        # Register ALL results in value map
        if result_id is not None:
            self._value_map[result_id] = ssa
        for rid in result_ids[1:]:
            self._value_map[rid] = ssa

        # Extract block_size from tt.make_range
        if name == "tt.make_range":
            end = attrs.get("end")
            if end is not None:
                self._block_size = end

        return ssa

    def _extract_attrs(self, op, name: str) -> Dict[str, Any]:
        """Extract relevant attributes for an operation."""
        attrs = {}

        if name == "tt.make_range":
            start = op.get_int_attr("start")
            end = op.get_int_attr("end")
            if start is not None:
                attrs["start"] = start
            if end is not None:
                attrs["end"] = end

        elif name == "tt.get_program_id":
            axis = op.get_int_attr("axis")
            attrs["axis"] = axis if axis is not None else 0

        elif name == "tt.get_num_programs":
            axis = op.get_int_attr("axis")
            attrs["axis"] = axis if axis is not None else 0

        elif name == "arith.constant":
            # `get_constant_value` (added in upstream triton 6cfdc3c37) handles
            # both scalar IntegerAttr and splat DenseIntElementsAttr — strictly
            # more than `get_int_attr("value")` which only handles the scalar
            # case. Try it first to avoid the text-fallback for splat int
            # constants like `dense<0> : tensor<32xi32>`.
            int_val = op.get_constant_value() if hasattr(op, "get_constant_value") else None
            if int_val is None:
                int_val = op.get_int_attr("value")
            if int_val is not None:
                attrs["value"] = int_val
            else:
                # Text fallback for floats and non-splat dense constants.
                # Use position-based list (not SSA-name dict) to handle
                # duplicate SSA names across functions (e.g., %cst in
                # multiple noinline functions).
                idx = self._constant_walk_index
                if idx < len(self._text_index.constants_by_position):
                    attrs["value"] = self._text_index.constants_by_position[idx]
                elif idx < len(self._constant_names_in_order):
                    ssa_name = self._constant_names_in_order[idx]
                    if ssa_name in self._text_index.constants:
                        attrs["value"] = self._text_index.constants[ssa_name]
            self._constant_walk_index += 1

        elif name in ("arith.cmpi", "arith.cmpf"):
            pred = op.get_int_attr("predicate")
            if pred is not None:
                attrs["predicate"] = pred
            # Look up predicate name POSITIONALLY by walk order (not by SSA
            # name — region-local names collide across scf.while/scf.if regions
            # and a name-keyed lookup returns the wrong region's predicate).
            idx = self._predicate_walk_index
            if idx < len(self._predicates_in_order):
                attrs["predicate_name"] = self._predicates_in_order[idx]
            self._predicate_walk_index += 1

        elif name == "cf.cond_br":
            idx = self._cond_br_walk_index
            if idx < len(self._text_index.cond_br_ops):
                n_true, n_false = self._text_index.cond_br_ops[idx]
                attrs["n_true_operands"] = n_true
                attrs["n_false_operands"] = n_false
            self._cond_br_walk_index += 1

        elif name == "tt.expand_dims":
            axis = op.get_int_attr("axis")
            if axis is not None:
                attrs["axis"] = axis
            else:
                attrs["axis"] = 0

        elif name == "tt.trans":
            # Extract permutation order from the op
            # The order attribute is an array<i32: 1, 0> for 2D transpose
            # Try to extract via text parsing since bindings may not expose array attrs
            attrs["order"] = None  # Will be populated from text if available

        elif name == "tt.reduce":
            axis = op.get_int_attr("axis")
            if axis is not None:
                attrs["axis"] = axis
            else:
                attrs["axis"] = 0

        elif name == "tt.scan":
            axis = op.get_int_attr("axis")
            if axis is not None:
                attrs["axis"] = axis
            else:
                attrs["axis"] = 0
            reverse = op.get_bool_attr("reverse")
            attrs["reverse"] = bool(reverse) if reverse is not None else False

        elif name == "tt.dot":
            allow_tf32 = op.get_bool_attr("allowTF32")
            if allow_tf32 is not None:
                attrs["allowTF32"] = allow_tf32
            max_imprecise = op.get_int_attr("maxNumImpreciseAcc")
            if max_imprecise is not None:
                attrs["maxNumImpreciseAcc"] = max_imprecise

        elif name == "tt.clampf":
            propagate_nan = op.get_int_attr("propagateNan")
            # 0 = none (NaN-quiet), 0xffff = all (NaN-propagating)
            attrs["propagateNan"] = "all" if propagate_nan else "none"

        elif name == "tt.fp_to_fp":
            # Optional rounding mode: 0 = RTZ, 1 = RTNE
            rounding = op.get_int_attr("rounding")
            if rounding is not None:
                attrs["rounding"] = "rtz" if rounding == 0 else "rtne"

        elif name == "tt.atomic_rmw":
            # Look up rmw_op and sem from pre-parsed module text by walk order
            idx = self._atomic_rmw_walk_index
            if idx < len(self._atomic_rmw_names_in_order):
                ssa_name = self._atomic_rmw_names_in_order[idx]
                info = self._text_index.atomic_ops.get(ssa_name, {})
                if "rmw_op" in info:
                    attrs["rmw_op"] = info["rmw_op"]
                if "sem" in info:
                    attrs["sem"] = info["sem"]
            self._atomic_rmw_walk_index += 1

        elif name == "tt.atomic_cas":
            # Look up sem from pre-parsed module text by walk order
            idx = self._atomic_cas_walk_index
            if idx < len(self._atomic_cas_names_in_order):
                ssa_name = self._atomic_cas_names_in_order[idx]
                info = self._text_index.atomic_ops.get(ssa_name, {})
                if "sem" in info:
                    attrs["sem"] = info["sem"]
            self._atomic_cas_walk_index += 1

        elif name == "tt.call":
            # Look up callee name from pre-parsed module text by walk order
            idx = self._call_walk_index
            if idx < len(self._call_targets_in_order):
                attrs["callee"] = self._call_targets_in_order[idx]
            self._call_walk_index += 1

        elif name == "tt.extern_elementwise":
            # Look up symbol/libname from pre-parsed module text by walk order
            idx = self._extern_elementwise_walk_index
            if idx < len(self._text_index.extern_elementwise_ops):
                info = self._text_index.extern_elementwise_ops[idx]
                if "symbol" in info:
                    attrs["symbol"] = info["symbol"]
                if "libname" in info:
                    attrs["libname"] = info["libname"]
                if "pure" in info:
                    attrs["pure"] = info["pure"]
            self._extern_elementwise_walk_index += 1

        return attrs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def walk_ttgir(module, options=None) -> IRGraph:
    """Walk a TTGIR MLIR module and return an IRGraph.

    Args:
        module: The MLIR module after TTGIR passes.
        options: MetalOptions instance (optional).

    Returns:
        IRGraph with structured operation data.
    """
    walker = MLIRWalker(module, options)
    return walker.walk()
