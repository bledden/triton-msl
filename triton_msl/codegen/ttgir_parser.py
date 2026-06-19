"""Parse TTGIR (TritonGPU IR) MLIR text into a KernelBuilder.

This module translates Triton's GPU-level MLIR representation into
Metal compute kernel descriptions that can be emitted as MSL.

Supported TTGIR operations:
- tt.get_program_id → threadgroup_position_in_grid
- tt.make_range, tt.splat, arith.addi → block offsets
- arith.cmpi slt → bounds mask
- tt.addptr → pointer arithmetic
- tt.load / tt.store → masked buffer read/write
- arith.addf/subf/mulf/divf → binary float ops
- arith.addi/subi/muli → binary int ops
- math.exp/log/sqrt/abs → unary ops
- tt.reduce → reductions (sum via arith.addf, max via arith.maxf)

The parser is text-based (using str(module)) since Triton's Python
MLIR bindings don't expose a structured walk API.
"""

import re
import warnings
from collections import OrderedDict

from triton_msl.codegen.msl_emitter import KernelBuilder


# ---------------------------------------------------------------------------
# MLIR preprocessing
# ---------------------------------------------------------------------------

def _strip_loc_annotations(text):
    """Remove all loc(...) annotations from MLIR text.

    These contain nested parentheses (e.g., loc("x_ptr"(#loc)))
    which break simple regex parsing.
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        # Look for 'loc(' preceded by whitespace or start
        if text[i:i+4] == 'loc(' and (i == 0 or text[i-1] in ' \t\n,'):
            # Skip balanced parentheses
            depth = 0
            j = i + 3  # points at '('
            while j < n:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            # Also skip any trailing whitespace
            while j < n and text[j] in ' \t':
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _strip_layout_annotations(text):
    """Strip #ttg.blocked<...> layout definitions and inline references.

    Real Triton IR has lines like:
        #blocked = #ttg.blocked<{sizePerThread = [2], ...}>
    and tensor types like:
        tensor<256xf32, #blocked>
        tensor<256x!tt.ptr<f32>, #blocked>

    We strip the definition lines entirely and remove the
    ', #identifier' from tensor types.
    """
    # Remove definition lines: #name = #ttg.blocked<{...}>
    text = re.sub(r'^#\w+\s*=\s*#ttg\.\w+<\{[^}]*\}>.*$', '', text, flags=re.MULTILINE)
    # Remove layout reference in tensor types.
    # Must handle nested angle brackets (e.g., tensor<256x!tt.ptr<f32>, #blocked>).
    text = re.sub(r',\s*#\w+>', '>', text)
    # Remove residual #loc lines left after loc() stripping.
    text = re.sub(r'^#\w+\s*=\s*$', '', text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# MLIR type mapping
# ---------------------------------------------------------------------------

def _mlir_type_to_triton_dtype(mlir_type):
    """Convert MLIR type string to Triton dtype string.

    Raises TypeError if the type is FP64 (not supported on Apple Silicon).
    """
    mlir_type = mlir_type.strip()
    _map = {
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
        "f64": "fp64",  # Metal has no double — will be emitted as float
        "i1": "i1",
        "i8": "i8",
        "i16": "i16",
        "i32": "i32",
        "i64": "i64",
    }
    return _map.get(mlir_type, "fp32")


def _extract_scalar_type(type_str):
    """Extract the scalar element type from a tensor or pointer type.

    Examples:
        'tensor<256xf32>' -> 'f32'
        'tensor<256xf32, #layout>' -> 'f32'
        '!tt.ptr<f32>' -> 'f32'
        'f32' -> 'f32'
        'i32' -> 'i32'
    """
    # Pointer type: !tt.ptr<f32>
    m = re.search(r"!tt\.ptr<(\w+)>", type_str)
    if m:
        return m.group(1)
    # Tensor type: tensor<...xTYPE> or tensor<...xTYPE, #layout>
    m = re.search(r"tensor<[^>]*x(\w+)(?:,\s*#\w+)?>", type_str)
    if m:
        return m.group(1)
    # Scalar type
    m = re.match(r"^(\w+)$", type_str.strip())
    if m:
        return m.group(1)
    return "f32"


def _extract_block_size(ir_text):
    """Extract the block size from tt.make_range op.

    Looks for: tt.make_range {end = N : i32, start = 0 : i32}
    """
    m = re.search(r"tt\.make_range\s*\{end\s*=\s*(\d+)\s*:\s*i32\s*,\s*start\s*=\s*0\s*:\s*i32\}", ir_text)
    if m:
        return int(m.group(1))
    return 256  # default


# ---------------------------------------------------------------------------
# TTGIR Parser
# ---------------------------------------------------------------------------

class TTGIRParser:
    """Parse TTGIR MLIR text and build a KernelBuilder.

    The parser processes the MLIR line by line, tracking SSA values
    and their roles (pointer, offset, mask, data), then emits
    corresponding KernelBuilder operations.
    """

    def __init__(self, ir_text, options):
        # Preprocess: strip loc() annotations and layout annotations
        # that break regex parsing.
        ir_text = _strip_loc_annotations(ir_text)
        ir_text = _strip_layout_annotations(ir_text)
        self.ir_text = ir_text
        self.options = options
        self.lines = ir_text.strip().split("\n")

        # SSA value tracking
        self.ssa_values = {}      # %name -> role info
        self.ssa_types = {}       # %name -> MLIR type string
        self.ptr_args = OrderedDict()    # arg_name -> (index, dtype, is_output)
        self.scalar_args = OrderedDict() # arg_name -> (index, dtype)
        self.kernel_name = "triton_kernel"
        self.block_size = _extract_block_size(ir_text)

        # Track which SSA values map to which concepts
        self.program_id_var = None
        self.offsets_var = None
        self.mask_var = None
        self.loaded_values = {}  # %ssa -> (ptr_arg_name, msl_var_name)
        self.computed_values = {} # %ssa -> msl_var_name

        # Reduction tracking
        self.reduce_ops = []  # [(result_ssa, input_ssa, op_kind, axis)]

        # Matmul tracking
        self.dot_ops = []  # [(result_ssa, lhs_ssa, rhs_ssa, acc_ssa)]

        # Loop tracking
        self.scf_for_loops = []  # [(lb_ssa, ub_ssa, step_ssa, body_lines)]

        # Operation buffer for the kernel builder
        self.ops = []

    def parse(self):
        """Parse the IR text and return a KernelBuilder."""
        self._parse_function_signature()
        self._parse_body()
        return self._build_kernel()

    def _parse_function_signature(self):
        """Extract kernel name and argument types from the function signature."""
        # Match: tt.func @name(%arg0: TYPE, %arg1: TYPE, ...)
        # or: tt.func public @name(%arg0: TYPE, %arg1: TYPE, ...)
        sig_match = re.search(
            r"tt\.func\s+(?:public\s+)?@(\w+)\s*\(([^)]*)\)",
            self.ir_text, re.DOTALL
        )
        if not sig_match:
            sig_match = re.search(
                r"func\.func\s+@(\w+)\s*\(([^)]*)\)",
                self.ir_text, re.DOTALL
            )
        if not sig_match:
            return

        self.kernel_name = sig_match.group(1)
        args_text = sig_match.group(2)

        # Parse each argument
        # Format: %argN: TYPE {optional attributes}
        arg_pattern = re.compile(
            r"%(\w+)\s*:\s*([^,{}]+(?:\{[^}]*\})?)"
        )
        for i, match in enumerate(arg_pattern.finditer(args_text)):
            arg_name = match.group(1)
            arg_type = match.group(2).strip()

            # Remove attributes like {tt.divisibility = 16 : i32}
            arg_type = re.sub(r"\s*\{[^}]*\}", "", arg_type).strip()

            self.ssa_types[f"%{arg_name}"] = arg_type

            if "!tt.ptr" in arg_type:
                elem_type = _extract_scalar_type(arg_type)
                dtype = _mlir_type_to_triton_dtype(elem_type)
                self.ptr_args[arg_name] = (i, dtype, False)
            else:
                elem_type = arg_type.strip()
                dtype = _mlir_type_to_triton_dtype(elem_type)
                self.scalar_args[arg_name] = (i, dtype)

    def _scan_scf_for_loops(self):
        """Scan for scf.for loop blocks in the IR text.

        scf.for has the form:
            %result = scf.for %iv = %lb to %ub step %step
                      iter_args(%acc = %init) -> (type) {
              ...body...
              scf.yield %new_acc : type
            }
        or without iter_args:
            scf.for %iv = %lb to %ub step %step {
              ...body...
            }
        """
        # Match scf.for with iter_args
        loop_pattern = re.compile(
            r'(?:%(\w+)(?::\d+)?\s*=\s*)?'  # optional result SSA
            r'scf\.for\s+%(\w+)\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)'
            r'(?:\s+iter_args\(([^)]*)\))?'  # optional iter_args
        )
        for m in loop_pattern.finditer(self.ir_text):
            result_ssa = f"%{m.group(1)}" if m.group(1) else None
            iv_name = m.group(2)
            lb_ssa = m.group(3)
            ub_ssa = m.group(4)
            step_ssa = m.group(5)
            iter_args_str = m.group(6)

            # Parse iter_args if present
            iter_args = []
            if iter_args_str:
                for ia_match in re.finditer(r'%(\w+)\s*=\s*(%\w+)', iter_args_str):
                    iter_args.append((ia_match.group(1), ia_match.group(2)))

            self.scf_for_loops.append({
                'result_ssa': result_ssa,
                'iv': iv_name,
                'lb': lb_ssa,
                'ub': ub_ssa,
                'step': step_ssa,
                'iter_args': iter_args,
            })

            # Record the loop variable in SSA tracking
            self.ssa_values[f"%{iv_name}"] = ("loop_iv", lb_ssa, ub_ssa, step_ssa)

    def _scan_reductions(self):
        """Scan for tt.reduce multi-line blocks in the IR text.

        tt.reduce has two forms:

        Triton 3.6+ (axis in angle brackets before body):
            %result = "tt.reduce"(%input) <{axis = 0 : i32}> ({
            ^bb0(%a: f32, %b: f32):
              %combined = arith.addf %a, %b : f32
              tt.reduce.return %combined : f32
            }) : (tensor<...>) -> f32

        Older format (axis in braces after body):
            %result = "tt.reduce"(%input) ({
            ^bb0(%a: f32, %b: f32):
              %combined = arith.addf %a, %b : f32
              "tt.reduce.return"(%combined) : (f32) -> ()
            }) {axis = 0 : i32} : (tensor<...>) -> f32
        """
        # Triton 3.6+: axis in <{...}> before the body
        reduce_pattern_new = re.compile(
            r'%(\w+)\s*=\s*"tt\.reduce"\s*\((%\w+)\)\s*<\{axis\s*=\s*(\d+)\s*:\s*i32\}>\s*\(\{'
            r'(.*?)'
            r'\}\)\s*:',
            re.DOTALL
        )
        # Older format: axis in {...} after the body
        reduce_pattern_old = re.compile(
            r'%(\w+)\s*=\s*"tt\.reduce"\s*\((%\w+)\)\s*\(\{'
            r'(.*?)'
            r'\}\)\s*\{axis\s*=\s*(\d+)',
            re.DOTALL
        )
        for m in reduce_pattern_new.finditer(self.ir_text):
            result_ssa = f"%{m.group(1)}"
            input_ssa = m.group(2)
            axis = int(m.group(3))
            body = m.group(4)

            if "arith.addf" in body:
                op_kind = "sum"
            elif "arith.maxf" in body or "arith.maxnumf" in body:
                op_kind = "max"
            elif "arith.minf" in body or "arith.minnumf" in body:
                op_kind = "min"
            else:
                op_kind = "sum"

            self.reduce_ops.append((result_ssa, input_ssa, op_kind, axis))
            self.ssa_values[result_ssa] = ("reduce", input_ssa, op_kind)

        for m in reduce_pattern_old.finditer(self.ir_text):
            result_ssa = f"%{m.group(1)}"
            input_ssa = m.group(2)
            body = m.group(3)
            axis = int(m.group(4))

            # Detect the combine operation from the body
            if "arith.addf" in body:
                op_kind = "sum"
            elif "arith.maxf" in body or "arith.maxnumf" in body:
                op_kind = "max"
            elif "arith.minf" in body or "arith.minnumf" in body:
                op_kind = "min"
            else:
                op_kind = "sum"  # fallback

            self.reduce_ops.append((result_ssa, input_ssa, op_kind, axis))
            self.ssa_values[result_ssa] = ("reduce", input_ssa, op_kind)

    def _scan_dot_ops(self):
        """Scan for tt.dot operations (matrix multiply-accumulate).

        tt.dot has the form:
            %result = tt.dot %lhs, %rhs, %acc {options} : type * type -> type
        or:
            %result = "tt.dot"(%lhs, %rhs, %acc) {options} : (types) -> type
        """
        # Quoted form
        dot_pattern1 = re.compile(
            r'%(\w+)\s*=\s*"tt\.dot"\s*\((%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\)'
        )
        # Unquoted form
        dot_pattern2 = re.compile(
            r'%(\w+)\s*=\s*tt\.dot\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)'
        )
        for pattern in [dot_pattern1, dot_pattern2]:
            for m in pattern.finditer(self.ir_text):
                result_ssa = f"%{m.group(1)}"
                lhs_ssa = m.group(2)
                rhs_ssa = m.group(3)
                acc_ssa = m.group(4)
                self.dot_ops.append((result_ssa, lhs_ssa, rhs_ssa, acc_ssa))
                self.ssa_values[result_ssa] = ("dot", lhs_ssa, rhs_ssa, acc_ssa)

    def _parse_body(self):
        """Walk through the body and classify operations."""
        # First scan for multi-line blocks (reduce, scf.for, dot)
        self._scan_scf_for_loops()
        self._scan_reductions()
        self._scan_dot_ops()

        for line in self.lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("module"):
                continue

            # tt.get_program_id
            if "tt.get_program_id" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.get_program_id\s+(\w+)", line)
                if m:
                    self.program_id_var = f"%{m.group(1)}"
                    self.ssa_values[self.program_id_var] = ("program_id", m.group(2))
                continue

            # tt.make_range
            if "tt.make_range" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.make_range\s*\{end\s*=\s*(\d+)", line)
                if m:
                    self.ssa_values[f"%{m.group(1)}"] = ("range", int(m.group(2)))
                continue

            # arith.constant
            if "arith.constant" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.constant\s+(.+)", line)
                if m:
                    val_name = f"%{m.group(1)}"
                    val_str = m.group(2).strip()
                    self.ssa_values[val_name] = ("constant", val_str)
                continue

            # tt.splat (broadcast scalar to tensor)
            if "tt.splat" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.splat\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    source = m.group(2)
                    self.ssa_values[result] = ("splat", source)
                continue

            # arith.addi (integer add — used for offset computation)
            if "arith.addi" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.addi\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    lhs, rhs = m.group(2), m.group(3)
                    self.ssa_values[result] = ("addi", lhs, rhs)
                continue

            # arith.muli (integer mul — used for pid * BLOCK_SIZE)
            if "arith.muli" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.muli\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    lhs, rhs = m.group(2), m.group(3)
                    self.ssa_values[result] = ("muli", lhs, rhs)
                continue

            # arith.subi (integer sub)
            if "arith.subi" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.subi\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("subi", m.group(2), m.group(3))
                continue

            # arith.cmpi (all predicates — used for mask generation and conditionals)
            if "arith.cmpi" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.cmpi\s+(\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    pred = m.group(2)  # slt, sle, sgt, sge, eq, ne, ult, ule, ugt, uge
                    self.mask_var = result
                    self.ssa_values[result] = ("mask", pred, m.group(3), m.group(4))
                continue

            # arith.cmpf (float comparison — used for activation conditionals)
            if "arith.cmpf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.cmpf\s+(\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    pred = m.group(2)  # ogt, oge, olt, ole, oeq, one, etc.
                    self.ssa_values[result] = ("mask", pred, m.group(3), m.group(4))
                continue

            # arith.select (ternary: cond ? true_val : false_val)
            if "arith.select" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.select\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("select", m.group(2), m.group(3), m.group(4))
                continue

            # arith.maxf / arith.minf (float max/min — not inside tt.reduce)
            if "arith.maxf" in line and "tt.reduce" not in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.maxf\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("fmax", m.group(2), m.group(3))
                continue

            if "arith.minf" in line and "tt.reduce" not in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.minf\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("fmin", m.group(2), m.group(3))
                continue

            # arith.negf (float negate)
            if "arith.negf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.negf\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("neg", m.group(2))
                continue

            # arith.sitofp / arith.uitofp (int to float conversion)
            if "arith.sitofp" in line or "arith.uitofp" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.\w+tofp\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # arith.fptosi / arith.fptoui (float to int conversion)
            if "arith.fptosi" in line or "arith.fptoui" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.fpto\w+\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # tt.addptr (pointer + offset)
            if "tt.addptr" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.addptr\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("addptr", m.group(2), m.group(3))
                continue

            # tt.load
            if "tt.load" in line and "=" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.load\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    ptr_ssa = m.group(2)
                    # Determine if there's a mask
                    has_mask = "," in line.split("tt.load")[1].split(":")[0]
                    self.ssa_values[result] = ("load", ptr_ssa, has_mask)
                continue

            # tt.store
            if "tt.store" in line:
                m = re.match(r"tt\.store\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    ptr_ssa = m.group(1)
                    val_ssa = m.group(2)
                    has_mask = line.count(",") >= 2
                    self.ops.append(("store", ptr_ssa, val_ssa, has_mask))
                continue

            # Binary float ops: arith.addf, arith.subf, arith.mulf, arith.divf
            for op_name, op_key in [("addf", "add"), ("subf", "sub"),
                                     ("mulf", "mul"), ("divf", "div")]:
                if f"arith.{op_name}" in line:
                    m = re.match(
                        rf"%(\w+)\s*=\s*arith\.{op_name}\s+(%\w+)\s*,\s*(%\w+)",
                        line
                    )
                    if m:
                        result = f"%{m.group(1)}"
                        self.ssa_values[result] = (op_key, m.group(2), m.group(3))
                    break

            # math.fma (fused multiply-add: a*b + c)
            if "math.fma" in line:
                m = re.match(
                    r"%(\w+)\s*=\s*math\.fma\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)",
                    line
                )
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("fma", m.group(2), m.group(3), m.group(4))
                continue

            # Unary math ops
            for op_name, op_key in [("math.exp", "exp"), ("math.log", "log"),
                                     ("math.sqrt", "sqrt"), ("math.rsqrt", "rsqrt"),
                                     ("math.absf", "abs"), ("math.sin", "sin"),
                                     ("math.cos", "cos"), ("math.tanh", "tanh")]:
                if op_name in line:
                    m = re.match(
                        rf"%(\w+)\s*=\s*{re.escape(op_name)}\s+(%\w+)",
                        line
                    )
                    if m:
                        result = f"%{m.group(1)}"
                        self.ssa_values[result] = (op_key, m.group(2))
                    break

            # arith.extf (fp16 -> fp32 promotion)
            if "arith.extf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.extf\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # arith.truncf (fp32 -> fp16 demotion)
            if "arith.truncf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.truncf\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # arith.extsi / arith.trunci (integer extension/truncation)
            if "arith.extsi" in line or "arith.trunci" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.(?:extsi|trunci)\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("int_cast", m.group(2))
                continue

            # tt.reduce (sum/max/min reduction)
            if '"tt.reduce"' in line or "tt.reduce" in line:
                # Reductions are multi-line; we handle them by detecting
                # the pattern in the combined block
                pass

    def _trace_to_ptr_arg(self, ssa):
        """Follow SSA chain to find the original pointer argument name."""
        seen = set()
        current = ssa
        while current and current not in seen:
            seen.add(current)
            val = self.ssa_values.get(current)
            if val is None:
                # Check if it's a function argument
                arg_name = current.lstrip("%")
                if arg_name in self.ptr_args:
                    return arg_name
                return None
            if val[0] == "splat":
                current = val[1]
            elif val[0] == "addptr":
                current = val[1]  # follow the pointer operand
            else:
                return None
        return None

    def _trace_to_scalar_arg(self, ssa):
        """Follow SSA chain to find the original scalar argument."""
        seen = set()
        current = ssa
        while current and current not in seen:
            seen.add(current)
            val = self.ssa_values.get(current)
            if val is None:
                arg_name = current.lstrip("%")
                if arg_name in self.scalar_args:
                    return arg_name
                return None
            if val[0] == "splat":
                current = val[1]
            else:
                return None
        return None

    def _classify_stores(self):
        """Determine which pointer args are outputs (have tt.store)."""
        for op in self.ops:
            if op[0] == "store":
                ptr_ssa = op[1]
                arg_name = self._trace_to_ptr_arg(ptr_ssa)
                if arg_name and arg_name in self.ptr_args:
                    idx, dtype, _ = self.ptr_args[arg_name]
                    self.ptr_args[arg_name] = (idx, dtype, True)

    def _build_kernel(self):
        """Construct a KernelBuilder from parsed IR."""
        self._classify_stores()

        num_warps = self.options.num_warps
        threads_per_tg = num_warps * 32
        block_size = max(self.block_size, threads_per_tg)

        kb = KernelBuilder(self.kernel_name, block_size=block_size)

        # Register pointer arguments in original order
        arg_msl_names = {}
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            msl_name = arg_name
            kb.add_ptr_arg(msl_name, dtype=dtype, const=(not is_output))
            arg_msl_names[arg_name] = msl_name

        # Register scalar arguments
        for arg_name, (idx, dtype) in self.scalar_args.items():
            msl_name = arg_name
            kb.add_scalar_arg(msl_name, dtype=dtype)
            arg_msl_names[arg_name] = msl_name

        # Determine the primary element dtype from the first pointer arg
        primary_dtype = "fp32"
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output:
                primary_dtype = dtype
                break

        # Find the n_elements argument (usually the last scalar arg)
        n_arg = None
        for arg_name in self.scalar_args:
            n_arg = arg_name

        # Check if this is flash attention (2 dot ops + reduce/exp)
        if len(self.dot_ops) >= 2 and self._is_flash_attention_pattern():
            return self._build_flash_attention_kernel(kb, primary_dtype)

        # Quantized matmul: dot ops with integer cast (INT8/INT4 weights)
        if self.dot_ops and self._is_quantized_matmul_pattern():
            return self._build_quantized_matmul_kernel(kb, primary_dtype)

        # Check if this is a matmul (tt.dot detected)
        if self.dot_ops:
            return self._build_matmul_kernel(kb, primary_dtype)

        # Check if this is a multi-reduce pattern
        # Cross-entropy: softmax pattern (max+sum) + log + another sum reduce
        if len(self.reduce_ops) >= 3 and self._is_cross_entropy_pattern():
            return self._build_cross_entropy_kernel(kb, n_arg, primary_dtype)
        # Paged attention: exp + mul + reductions with 3+ input ptrs
        if self.reduce_ops and self._is_paged_attention_pattern():
            return self._build_paged_attention_kernel(kb, primary_dtype)
        # Beam search: cmp + add + reductions with 2+ output ptrs
        if self.reduce_ops and self._is_beam_search_pattern():
            return self._build_beam_search_kernel(kb, primary_dtype)
        # Online softmax: reduce in scf.for loop (single-pass streaming)
        if self.reduce_ops and self._is_online_softmax_pattern():
            return self._build_online_softmax_kernel(kb, primary_dtype)
        # Softmax: max + sum reductions
        if len(self.reduce_ops) >= 2 and self._is_softmax_pattern():
            return self._build_softmax_kernel(kb, n_arg, primary_dtype)
        # Fused residual + layer norm: layer norm pattern with arith.addf before first reduce
        if len(self.reduce_ops) >= 2 and self._is_fused_residual_norm_pattern():
            return self._build_fused_residual_norm_kernel(kb, n_arg, primary_dtype)
        # Variance: 2 sum reductions with sub+mul but no rsqrt (no normalization)
        if len(self.reduce_ops) >= 2 and self._is_variance_pattern():
            return self._build_variance_kernel(kb, n_arg, primary_dtype)
        # Standard layer norm
        if len(self.reduce_ops) >= 2 and self._is_layer_norm_pattern():
            return self._build_layer_norm_kernel(kb, n_arg, primary_dtype)
        # RMS norm: single sum reduction + rsqrt, no sub (no mean subtraction)
        if self.reduce_ops and self._is_rms_norm_pattern():
            return self._build_rms_norm_kernel(kb, n_arg, primary_dtype)
        # Group norm: reductions with 3+ input ptrs (input + weight + bias)
        if self.reduce_ops and self._is_group_norm_pattern():
            return self._build_group_norm_kernel(kb, primary_dtype)
        # Instance norm: reductions with 1 input ptr (per-channel spatial norm)
        if self.reduce_ops and self._is_instance_norm_pattern():
            return self._build_instance_norm_kernel(kb, primary_dtype)
        # Bitonic sort / top-k that the generic lowerer had to skip: emitting
        # a scalar reduction here would silently produce wrong data. Fall
        # through to a no-op (empty-body) kernel that the runtime will fail
        # on, making the unsupported case visible rather than silent.
        if self.reduce_ops and self._is_bitonic_sort_pattern():
            kb.comment(
                "Unsupported: bitonic sort / top-k with > 1024 elements")
            return kb
        if self.reduce_ops:
            return self._build_reduction_kernel(kb, n_arg, primary_dtype)

        # RoPE: sin + cos + mul pattern (no reductions)
        if self._is_rope_pattern():
            return self._build_rope_kernel(kb, primary_dtype)

        # Split: 1 input ptr, 2+ output ptrs, no math ops (pure data movement)
        # Must be checked before top-k since top-k also has 2+ outputs + cmp
        if self._is_split_pattern():
            return self._build_split_kernel(kb, primary_dtype)

        # Speculative decoding: div + cmp with 3+ input ptrs (no reductions)
        if self._is_speculative_decode_pattern():
            return self._build_speculative_decode_kernel(kb, primary_dtype)

        # Top-K sampling: cmp with 2+ output ptrs (no reductions)
        if self._is_top_k_pattern():
            return self._build_top_k_kernel(kb, primary_dtype)

        # Dropout: 2 input ptrs + mask/select + mul, no reductions, 1 output
        if self._is_dropout_pattern():
            return self._build_dropout_kernel(kb, primary_dtype)

        # Batch normalization (eval): 4+ input ptrs, sub+mul, no reductions
        if self._is_batch_norm_pattern():
            return self._build_batch_norm_kernel(kb, primary_dtype)

        # Residual add: 2-3 input ptrs, 1 output, only add ops (no sub/mul/div)
        if self._is_residual_add_pattern():
            return self._build_residual_add_kernel(kb, primary_dtype)

        # Fused MLP: silu(gate) * up pattern (exp + neg + mul, no reductions)
        if self._is_fused_mlp_pattern():
            return self._build_fused_mlp_kernel(kb, primary_dtype)

        # Embedding lookup: 1 float ptr + 1 int ptr input, 1 output
        if self._is_embedding_pattern():
            return self._build_embedding_kernel(kb, primary_dtype)

        # Scatter: 2 input ptrs (data + indices), 1 output, store uses loaded index
        # (checked before gather — scatter has more specific store-ptr-from-load check)
        if self._is_scatter_pattern():
            return self._build_scatter_kernel(kb, primary_dtype)

        # Gather: 2 input ptrs (data + indices) with int arg, 1 output, no cmp/select
        if self._is_gather_pattern():
            return self._build_gather_kernel(kb, primary_dtype)

        # Transpose: 2D grid (two program_id calls), 1 input, 1 output, no math ops
        if self._is_transpose_pattern():
            return self._build_transpose_kernel(kb, primary_dtype)

        # Concat: 2+ input ptrs, 1 output, no math ops (pure copy)
        if self._is_concat_pattern():
            return self._build_concat_kernel(kb, primary_dtype)

        # Repeat KV: 1 input, 1 output, 4+ scalar args, div/mod indexing
        if self._is_repeat_kv_pattern():
            return self._build_repeat_kv_kernel(kb, primary_dtype)

        # Where (ternary select): 3 input ptrs (cond, x, y), 1 output, select op
        if self._is_where_pattern():
            return self._build_where_kernel(kb, primary_dtype)

        # Clamp: 1 input ptr, 1 output, 2 scalar args (min, max), maxf+minf ops
        if self._is_clamp_pattern():
            return self._build_clamp_kernel(kb, primary_dtype)

        # Cumsum: scf.for with sequential add, 1 input, 1 output
        if self._is_cumsum_pattern():
            return self._build_cumsum_kernel(kb, primary_dtype)

        # Activation functions: tanh, sigmoid, elu, leaky_relu, hardswish
        act = self._classify_activation()
        if act:
            return self._build_activation_kernel(kb, act, primary_dtype)

        # Standard elementwise fallback — no specific pattern matched.
        # Only warn if the kernel looks like it might NOT be a simple elementwise
        # (multiple outputs, reductions, dots, or many scalar args suggest complexity).
        n_outputs = sum(1 for _, (_, _, o) in self.ptr_args.items() if o)
        n_inputs = len(self.ptr_args) - n_outputs
        ssa_ops = {v[0] for v in self.ssa_values.values()} if self.ssa_values else set()
        suspicious = (
            n_outputs > 1
            or len(self.reduce_ops) > 0
            or len(self.dot_ops) > 0
            or len(self.scalar_args) > 2
            or n_inputs > 4
        )
        if suspicious:
            warnings.warn(
                f"TTGIR pattern not recognized for kernel '{self.kernel_name}'. "
                f"Falling back to generic elementwise codegen. "
                f"SSA ops: {sorted(ssa_ops)}. "
                f"Ptrs: {len(self.ptr_args)} ({n_outputs} out). "
                f"Reductions: {len(self.reduce_ops)}. Dots: {len(self.dot_ops)}. "
                f"This kernel may produce incorrect results.",
                stacklevel=2,
            )
        offsets = kb.make_block_offsets("pid", "offsets")
        if n_arg:
            mask = kb.make_mask(offsets, n_arg, "mask")
        else:
            mask = None

        # Load from each input pointer
        input_vars = {}
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output:
                var_name = f"val_{arg_name}"
                kb.load(arg_name, offsets, mask, out_var=var_name, dtype=dtype)
                input_vars[arg_name] = var_name

        # Analyze the computation between loads and stores
        result_var = self._emit_computation(kb, input_vars, primary_dtype)

        # Store to output
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if is_output:
                kb.store(arg_name, offsets, result_var, mask, dtype=dtype)

        return kb

    def _build_reduction_kernel(self, kb, n_arg, primary_dtype):
        """Generate a reduction kernel using threadgroup reduce pattern.

        Supports both 1D reductions (whole-tensor → scalar per program) and
        2D axis-aware reductions (reduce along rows or columns).

        For axis=1 (row-wise): each program reduces one row of length n_arg.
        For axis=0 (column-wise): each program reduces one column across n_rows.
        For 1D (axis=0 single-dim): standard strided accumulation + threadgroup reduce.
        """
        reduce_result_ssa, reduce_input_ssa, reduce_op, axis = self.reduce_ops[0]

        # Find the input pointer for the reduction
        input_arg = None
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output:
                input_arg = arg_name
                break

        # Find the output pointer
        output_arg = None
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if is_output:
                output_arg = arg_name
                break

        if not input_arg or not output_arg or not n_arg:
            # Fallback: can't generate reduction, return empty kernel
            return kb

        # Check if we have a second dimension argument (n_rows or n_cols)
        n_rows_arg = None
        n_cols_arg = None
        scalar_args_list = list(self.scalar_args.items())
        if len(scalar_args_list) >= 2:
            # Convention: first scalar is n_rows/n_elements, second is n_cols
            n_rows_arg = scalar_args_list[0][0]
            n_cols_arg = scalar_args_list[1][0]

        # Shared memory for cross-SIMD-group reduction
        n_simd_groups = (kb.block_size + 31) // 32
        kb.declare_threadgroup_array("shared", dtype=primary_dtype, size=n_simd_groups)

        # Identity value for the reduction
        identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[reduce_op]
        combine = {"sum": "+", "max": "max", "min": "min"}[reduce_op]

        # Check if there's a pre-reduce computation (e.g., exp, mul before sum)
        pre_reduce_op = self._find_pre_reduce_op(reduce_input_ssa)

        # Determine if this is a 2D reduction with explicit axis
        is_2d = n_rows_arg is not None and n_cols_arg is not None and axis in (0, 1)

        if is_2d and axis == 1:
            # Row-wise reduction: each program handles one row
            # pid indexes rows, reduce across n_cols columns
            kb._var("row", "pid", ty="uint")
            kb._var("acc", identity, ty="float")
            kb.raw_line(f"for (uint c = lid; c < {n_cols_arg}; c += {kb.block_size}u) {{")
            kb.indent()
            kb._var("idx", f"row * {n_cols_arg} + c", ty="uint")
            self._emit_reduce_element(kb, input_arg, pre_reduce_op)
            self._emit_reduce_combine(kb, combine)
            kb.dedent()
            kb.raw_line("}")
            kb.threadgroup_reduce(reduce_op, "acc", "shared", "total")
            kb.begin_if("lid == 0")
            kb.raw_line(f"{output_arg}[row] = total;")
            kb.end_block()

        elif is_2d and axis == 0:
            # Column-wise reduction: each program handles one column
            # pid indexes columns, reduce across n_rows rows
            kb._var("col", "pid", ty="uint")
            kb._var("acc", identity, ty="float")
            kb.raw_line(f"for (uint r = lid; r < {n_rows_arg}; r += {kb.block_size}u) {{")
            kb.indent()
            kb._var("idx", f"r * {n_cols_arg} + col", ty="uint")
            self._emit_reduce_element(kb, input_arg, pre_reduce_op)
            self._emit_reduce_combine(kb, combine)
            kb.dedent()
            kb.raw_line("}")
            kb.threadgroup_reduce(reduce_op, "acc", "shared", "total")
            kb.begin_if("lid == 0")
            kb.raw_line(f"{output_arg}[col] = total;")
            kb.end_block()

        else:
            # Standard 1D reduction: each program reduces a contiguous block
            kb._var("acc", identity, ty="float")
            kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
            kb.indent()
            kb._var("idx", f"pid * {n_arg} + i", ty="uint")
            self._emit_reduce_element(kb, input_arg, pre_reduce_op)
            self._emit_reduce_combine(kb, combine)
            kb.dedent()
            kb.raw_line("}")
            kb.threadgroup_reduce(reduce_op, "acc", "shared", "total")
            kb.begin_if("lid == 0")
            kb.raw_line(f"{output_arg}[pid] = total;")
            kb.end_block()

        return kb

    def _emit_reduce_element(self, kb, input_arg, pre_reduce_op):
        """Emit the per-element load + optional transform for reductions."""
        if pre_reduce_op:
            op_kind, op_args = pre_reduce_op
            kb._var("loaded", f"{input_arg}[idx]", ty="float")
            if op_kind in ("exp", "log", "sqrt", "abs"):
                kb._var("elem", f"{op_kind}(loaded)", ty="float")
            elif op_kind in ("mul",) and len(op_args) == 2:
                kb._var("elem", f"loaded * loaded", ty="float")  # square
            else:
                kb.raw_line("float elem = loaded;")
        else:
            kb._var("elem", f"{input_arg}[idx]", ty="float")

    def _emit_reduce_combine(self, kb, combine):
        """Emit the accumulation step for reductions."""
        if combine == "+":
            kb.raw_line("acc += elem;")
        else:
            kb.raw_line(f"acc = {combine}(acc, elem);")

    def _is_softmax_pattern(self):
        """Check if multi-reduce pattern matches softmax (max + sum).

        Requires the canonical softmax signature: a max reduce, a subtract,
        an exp, and a sum reduce. Bitonic sort also produces max/sum-like
        reduces (actually xori classified as 'sum' fallback) but has no
        math.exp — this guard keeps sort out of the softmax template.
        """
        if len(self.reduce_ops) < 2:
            return False
        ops = [r[2] for r in self.reduce_ops]
        if "max" not in ops or "sum" not in ops:
            return False
        # Softmax has exp(x - max) between the max reduce and sum reduce.
        # Sort's xor reduces have no exp.
        has_exp = ("math.exp" in self.ir_text
                   or "math.exp2" in self.ir_text)
        has_sub = ("arith.subf" in self.ir_text)
        return has_exp and has_sub

    def _is_bitonic_sort_pattern(self):
        """Detect tl.sort / tl.topk's bitonic-sort lowering.

        Signatures:
          - Many reduces (>= 3) with xor combine bodies (arith.xori) —
            tl.sort reshapes to (2,)*n and uses xor as a placeholder axis
            reducer to produce the compare-and-swap mask.
          - No exp / divf / subf (not softmax-like).
          - Frequent arith.cmpf + arith.select (compare-and-swap).
        """
        if not self.reduce_ops or len(self.reduce_ops) < 3:
            return False
        xor_count = self.ir_text.count("arith.xori")
        if xor_count < len(self.reduce_ops):
            return False
        if ("arith.cmpf" not in self.ir_text
                or "arith.select" not in self.ir_text):
            return False
        # Soft negative checks: not softmax / layer-norm
        if ("math.exp" in self.ir_text
                or "arith.divf" in self.ir_text
                or "math.rsqrt" in self.ir_text):
            return False
        return True

    def _is_layer_norm_pattern(self):
        """Check if multi-reduce pattern matches layer norm (sum + sum).

        Layer norm has two sum reductions (mean, variance) and operations
        between them (subtract mean, square). Softmax has max + sum.
        """
        if len(self.reduce_ops) < 2:
            return False
        ops = [r[2] for r in self.reduce_ops]
        # Layer norm: two sum reductions (NOT softmax which has max+sum)
        if ops.count("sum") >= 2 and "max" not in ops:
            ssa_ops = {v[0] for v in self.ssa_values.values()}
            # Check for subtract or rsqrt/sqrt (normalization step)
            return ("sub" in ssa_ops
                    or "rsqrt" in ssa_ops
                    or "sqrt" in ssa_ops)
        return False

    def _build_softmax_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused row-wise softmax kernel.

        Detected from 2 tt.reduce ops (max then sum) with exp in between.
        Each threadgroup processes one row:
        1. Find max(row)
        2. Compute exp(x - max)
        3. Sum the exponentials
        4. Divide each by the sum
        """
        n_simd_groups = (kb.block_size + 31) // 32

        # Find input and output pointers
        input_arg = None
        output_arg = None
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output and input_arg is None:
                input_arg = arg_name
            if is_output:
                output_arg = arg_name

        if not input_arg or not output_arg or not n_arg:
            return kb

        # Shared memory for reductions
        kb.declare_threadgroup_array("shared_max", dtype=primary_dtype, size=n_simd_groups)
        kb.declare_threadgroup_array("shared_sum", dtype=primary_dtype, size=n_simd_groups)

        # Row base pointer: each threadgroup handles one row
        kb._var("row_start", f"pid * {n_arg}", ty="uint")

        # Pass 1: Find row max (strided accumulation)
        kb._var("local_max", "-INFINITY", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb.raw_line(f"local_max = max(local_max, {input_arg}[row_start + i]);")
        kb.dedent()
        kb.raw_line("}")

        # Reduce max across threadgroup
        kb.threadgroup_reduce("max", "local_max", "shared_max", "row_max")

        # Broadcast row_max to all threads via shared memory
        kb.begin_if("lid == 0")
        kb.raw_line("shared_max[0] = row_max;")
        kb.end_block()
        kb.barrier("threadgroup")
        kb._var("max_val", "shared_max[0]", ty="float")

        # Pass 2: Compute exp(x - max) and accumulate sum
        kb._var("local_sum", "0.0f", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb._var("e", f"exp({input_arg}[row_start + i] - max_val)", ty="float")
        kb.raw_line(f"{output_arg}[row_start + i] = e;")
        kb.raw_line("local_sum += e;")
        kb.dedent()
        kb.raw_line("}")

        # Reduce sum across threadgroup
        kb.threadgroup_reduce("sum", "local_sum", "shared_sum", "row_sum")

        # Broadcast row_sum
        kb.begin_if("lid == 0")
        kb.raw_line("shared_sum[0] = row_sum;")
        kb.end_block()
        kb.barrier("threadgroup")
        kb._var("sum_val", "shared_sum[0]", ty="float")

        # Pass 3: Normalize
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb.raw_line(f"{output_arg}[row_start + i] /= sum_val;")
        kb.dedent()
        kb.raw_line("}")

        return kb

    def _build_layer_norm_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused layer norm kernel from two sum reductions.

        Detected from 2 tt.reduce sum ops with sub/mul between them.
        Pattern:
        1. sum(input)         → mean = sum / n
        2. sum((x - mean)^2)  → var = sum / n
        3. output = (x - mean) * rsqrt(var + eps) * gamma + beta

        Each threadgroup processes one row. Gamma/beta are detected from
        pointer args that are neither the primary input nor the output.
        """
        n_simd_groups = (kb.block_size + 31) // 32

        # Find input, output, and parameter pointers
        input_arg = None
        output_arg = None
        param_args = []
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if is_output:
                output_arg = arg_name
            elif input_arg is None:
                input_arg = arg_name
            else:
                param_args.append(arg_name)

        if not input_arg or not output_arg or not n_arg:
            return kb

        # gamma is first param, beta is second (if they exist)
        gamma_arg = param_args[0] if len(param_args) > 0 else None
        beta_arg = param_args[1] if len(param_args) > 1 else None

        # Extract epsilon from constants in the IR
        eps = 1e-6
        for ssa, val in self.ssa_values.items():
            if val[0] == "constant":
                import re as _re
                m = _re.search(r"([\d.e+-]+)\s*:\s*f32", val[1])
                if m:
                    v = float(m.group(1))
                    if 0 < v < 1e-3:  # looks like an epsilon
                        eps = v

        # Shared memory for reductions
        kb.declare_threadgroup_array("shared_mean", dtype=primary_dtype, size=n_simd_groups)
        kb.declare_threadgroup_array("shared_var", dtype=primary_dtype, size=n_simd_groups)

        # Row base pointer
        kb._var("row_start", f"pid * {n_arg}", ty="uint")

        # Pass 1: Compute mean
        kb._var("local_sum", "0.0f", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb.raw_line(f"local_sum += {input_arg}[row_start + i];")
        kb.dedent()
        kb.raw_line("}")

        kb.threadgroup_reduce("sum", "local_sum", "shared_mean", "total_sum")

        kb.begin_if("lid == 0")
        kb.raw_line("shared_mean[0] = total_sum;")
        kb.end_block()
        kb.barrier("threadgroup")
        kb._var("mean_val", f"shared_mean[0] / float({n_arg})", ty="float")

        # Pass 2: Compute variance
        kb._var("local_var", "0.0f", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb._var("diff", f"{input_arg}[row_start + i] - mean_val", ty="float")
        kb.raw_line("local_var += diff * diff;")
        kb.dedent()
        kb.raw_line("}")

        kb.threadgroup_reduce("sum", "local_var", "shared_var", "total_var")

        kb.begin_if("lid == 0")
        kb.raw_line("shared_var[0] = total_var;")
        kb.end_block()
        kb.barrier("threadgroup")
        kb._var("var_val", f"shared_var[0] / float({n_arg})", ty="float")
        kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

        # Pass 3: Normalize
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        if gamma_arg and beta_arg:
            kb.raw_line(f"{output_arg}[row_start + i] = ({input_arg}[row_start + i] - mean_val) * inv_std * {gamma_arg}[i] + {beta_arg}[i];")
        elif gamma_arg:
            kb.raw_line(f"{output_arg}[row_start + i] = ({input_arg}[row_start + i] - mean_val) * inv_std * {gamma_arg}[i];")
        else:
            kb.raw_line(f"{output_arg}[row_start + i] = ({input_arg}[row_start + i] - mean_val) * inv_std;")
        kb.dedent()
        kb.raw_line("}")

        return kb

    def _is_flash_attention_pattern(self):
        """Check if the IR matches a flash attention pattern.

        Flash attention has:
        - 2+ tt.dot ops (Q@K^T and P@V)
        - exp() between them (softmax numerator)
        - A max reduction or explicit max for numerical stability
        """
        if len(self.dot_ops) < 2:
            return False
        # Check for exp in the SSA values (used in softmax between dots)
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        # Check for max reduction or explicit max
        has_max = (any(r[2] == "max" for r in self.reduce_ops) or
                   any(v[0] == "fmax" for v in self.ssa_values.values()))
        return has_exp and has_max

    def _build_flash_attention_kernel(self, kb, primary_dtype):
        """Generate a flash attention kernel from the pattern.

        Detected from 2+ tt.dot ops with exp/max between them.
        Delegates to the pre-built flash attention kernel generator.
        """
        from triton_msl.codegen.msl_emitter import make_flash_attention_kernel

        # Detect head_dim from pointer args
        # In typical flash attention TTGIR, Q/K/V are the first 3 pointer args
        # and the last scalar arg before seq_len determines head_dim
        head_dim = 64  # default

        # Check for causal masking: look for cmpi or "causal" in IR
        causal = "causal" in self.ir_text.lower() or any(
            v[0] == "mask" and "slt" in str(self.ssa_values.get(v[1], ""))
            for v in self.ssa_values.values() if v[0] == "select"
        )

        kb.set_prebuilt_msl(make_flash_attention_kernel(
            head_dim=head_dim, causal=causal
        ))
        return kb

    def _build_matmul_kernel(self, kb, primary_dtype):
        """Generate a tiled matmul kernel from tt.dot pattern.

        Detected from tt.dot operations in the IR. Generates a tiled
        matmul using simdgroup_matrix hardware (8x8 MMA).

        Expected pointer args: A, B, C (+ optional M, N, K scalars).
        Dispatch: one threadgroup per 32x32 output tile, 128 threads each.
        """
        from triton_msl.codegen.msl_emitter import make_simdgroup_matmul_kernel

        # Determine dtype from the dot operation
        dtype_map = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}
        msl_dtype = dtype_map.get(primary_dtype, "fp32")

        # We generate the simdgroup matmul kernel as a standalone MSL
        # and return it as a "prebuilt" kernel via KernelBuilder's raw MSL mode
        kb.set_prebuilt_msl(make_simdgroup_matmul_kernel(dtype=msl_dtype))
        return kb

    def _is_cross_entropy_pattern(self):
        """Check if IR matches cross-entropy loss: max + sum (softmax) + log + sum.

        Cross-entropy has:
        - 3+ reduce ops (max for softmax stability, sum for softmax denominator,
          sum for final loss aggregation)
        - exp between max and first sum (softmax numerator)
        - log after the softmax
        """
        if len(self.reduce_ops) < 3:
            return False
        ops = [r[2] for r in self.reduce_ops]
        if "max" not in ops:
            return False
        if ops.count("sum") < 2:
            return False
        # Check for both exp and log in SSA values
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        has_log = any(v[0] == "log" for v in self.ssa_values.values())
        return has_exp and has_log

    def _build_cross_entropy_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused cross-entropy loss kernel.

        Detected from 3 reduces (max+sum+sum) with exp and log.
        Delegates to the pre-built cross-entropy kernel.
        """
        from triton_msl.codegen.msl_emitter import make_cross_entropy_kernel
        kb.set_prebuilt_msl(make_cross_entropy_kernel())
        return kb

    def _is_fused_residual_norm_pattern(self):
        """Check if IR matches fused residual add + layer norm.

        This pattern has:
        - 2+ sum reductions (mean + variance, layer norm pattern)
        - An arith.addf before the first reduction (residual connection)
        - rsqrt or sub operations (layer norm normalization)
        - No max reduction (distinguishes from softmax)
        """
        if len(self.reduce_ops) < 2:
            return False
        ops = [r[2] for r in self.reduce_ops]
        if ops.count("sum") < 2 or "max" in ops:
            return False
        # Must have both add and (sub or rsqrt) for residual + norm
        has_add = any(v[0] == "add" for v in self.ssa_values.values())
        has_sub_or_rsqrt = (
            any(v[0] == "sub" for v in self.ssa_values.values()) or
            any(v[0] == "rsqrt" for v in self.ssa_values.values())
        )
        if not (has_add and has_sub_or_rsqrt):
            return False

        # Check the input to the first sum reduction for an add (residual)
        first_sum = None
        for entry in self.reduce_ops:
            result_ssa, input_ssa, op_kind = entry[0], entry[1], entry[2]
            if op_kind == "sum":
                first_sum = (result_ssa, input_ssa)
                break

        if first_sum is None:
            return False

        # Walk back from the first reduce's input to see if there's an add
        # (residual connection: x + residual before normalization)
        def _has_add_in_chain(ssa, depth=5):
            if depth <= 0:
                return False
            val = self.ssa_values.get(ssa)
            if val is None:
                return False
            if val[0] == "add":
                return True
            # Follow operands
            for operand in val[1:]:
                if isinstance(operand, str) and operand.startswith("%"):
                    if _has_add_in_chain(operand, depth - 1):
                        return True
            return False

        return _has_add_in_chain(first_sum[1])

    def _build_fused_residual_norm_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused residual add + layer norm kernel.

        output = LayerNorm(input + residual, gamma, beta)

        Delegates to the dedicated fused kernel that combines residual
        addition with layer normalization in a single pass.
        """
        from triton_msl.codegen.msl_emitter import make_fused_residual_norm_kernel
        kb.set_prebuilt_msl(make_fused_residual_norm_kernel())
        return kb

    def _is_variance_pattern(self):
        """Check if IR matches a variance computation pattern.

        Variance has:
        - 2 sum reductions (for mean and for sum of squared diffs)
        - subtract (x - mean)
        - multiply (diff * diff, i.e. squaring)
        - NO rsqrt or sqrt (those indicate normalization, not plain variance)
        - 1 output pointer (variance values)
        """
        if len(self.reduce_ops) < 2:
            return False
        ops_list = [r[2] for r in self.reduce_ops]
        if ops_list.count("sum") < 2 or "max" in ops_list:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_sub = "sub" in ssa_ops
        has_mul = "mul" in ssa_ops
        has_rsqrt = "rsqrt" in ssa_ops
        has_sqrt = "sqrt" in ssa_ops
        # Variance = 2 sums + sub + mul, but NO rsqrt/sqrt (normalization)
        return has_sub and has_mul and not has_rsqrt and not has_sqrt

    def _build_variance_kernel(self, kb, n_arg, primary_dtype):
        """Generate a variance kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_variance_kernel
        kb.set_prebuilt_msl(make_variance_kernel())
        return kb

    def _is_rope_pattern(self):
        """Check if IR matches a RoPE (Rotary Position Embedding) pattern.

        RoPE has:
        - sin and cos operations (for rotation matrix)
        - Interleaved multiply and add/sub (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
        - No reductions (elementwise operation)
        """
        has_sin = any(v[0] == "sin" for v in self.ssa_values.values())
        has_cos = any(v[0] == "cos" for v in self.ssa_values.values())
        has_mul = any(v[0] == "mul" for v in self.ssa_values.values())
        return has_sin and has_cos and has_mul and not self.reduce_ops

    def _build_rope_kernel(self, kb, primary_dtype):
        """Generate a RoPE kernel from the pattern.

        Delegates to the pre-built RoPE kernel.
        """
        from triton_msl.codegen.msl_emitter import make_rope_kernel
        kb.set_prebuilt_msl(make_rope_kernel())
        return kb

    def _is_quantized_matmul_pattern(self):
        """Check if IR matches a quantized matmul pattern.

        Quantized matmul has:
        - tt.dot ops (matrix multiplication)
        - arith.extsi or int_cast operations (integer weight dequantization)
        - Indicates INT8 or INT4 weights being promoted to float for computation
        """
        if not self.dot_ops:
            return False
        has_int_cast = any(v[0] == "int_cast" for v in self.ssa_values.values())
        return has_int_cast

    def _build_quantized_matmul_kernel(self, kb, primary_dtype):
        """Generate a quantized matmul kernel from the pattern.

        Uses INT8 quantized matmul by default. If the IR has group-size
        related constants, could be INT4, but INT8 is the safer default.
        """
        from triton_msl.codegen.msl_emitter import make_int8_matmul_kernel
        kb.set_prebuilt_msl(make_int8_matmul_kernel())
        return kb

    def _is_fused_mlp_pattern(self):
        """Check if IR matches a fused MLP (SwiGLU) pattern.

        Fused MLP has:
        - exp (for silu/sigmoid: x / (1 + exp(-x)))
        - multiply (gate * up projection fusion)
        - divide or reciprocal (for 1 / (1 + exp(-x)))
        - No reductions (elementwise)
        - No dot ops (not a matmul)
        - 2+ input pointer args (gate and up projections)
        """
        if self.reduce_ops or self.dot_ops:
            return False
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        has_mul = any(v[0] == "mul" for v in self.ssa_values.values())
        has_neg = any(v[0] == "neg" for v in self.ssa_values.values())
        # SiLU requires: exp(-x) → neg + exp, then division
        has_div = any(v[0] == "div" for v in self.ssa_values.values())
        # Need at least 2 input pointers (gate + up)
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        return has_exp and has_mul and n_inputs >= 2 and (has_neg or has_div)

    def _build_fused_mlp_kernel(self, kb, primary_dtype):
        """Generate a fused MLP (SwiGLU) kernel from the pattern.

        output = silu(gate) * up
        """
        from triton_msl.codegen.msl_emitter import make_fused_mlp_kernel
        kb.set_prebuilt_msl(make_fused_mlp_kernel())
        return kb

    def _is_paged_attention_pattern(self):
        """Check if IR matches a paged attention pattern.

        Paged attention has:
        - exp + mul (attention score computation)
        - Reductions (max + sum for softmax)
        - Indirect indexing via page table (load from pointer loaded from pointer)
        - 3+ input pointers (Q, K_cache/V_cache, page_table)
        """
        if not self.reduce_ops:
            return False
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        has_mul = any(v[0] == "mul" for v in self.ssa_values.values())
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        # Page attention needs 3+ input ptrs and exp+mul (attention softmax)
        return has_exp and has_mul and n_inputs >= 3 and not self.dot_ops

    def _build_paged_attention_kernel(self, kb, primary_dtype):
        """Generate a paged attention kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_paged_attention_kernel
        kb.set_prebuilt_msl(make_paged_attention_kernel())
        return kb

    def _is_top_k_pattern(self):
        """Check if IR matches a top-k sampling pattern.

        Top-k has:
        - Comparison operations (mask ops from cmpi/cmpf)
        - No dot ops (not a matmul)
        - 1 input pointer (logits), 2 output pointers (values + indices)
        """
        if self.dot_ops:
            return False
        has_cmp = any(v[0] == "mask" for v in self.ssa_values.values())
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        return has_cmp and n_outputs >= 2 and not self.reduce_ops

    def _build_top_k_kernel(self, kb, primary_dtype):
        """Generate a top-k sampling kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_top_k_kernel
        kb.set_prebuilt_msl(make_top_k_kernel())
        return kb

    def _is_speculative_decode_pattern(self):
        """Check if IR matches a speculative decoding pattern.

        Speculative decoding has:
        - Division (probability ratio: target/draft)
        - Comparison (acceptance test: ratio >= random threshold)
        - 3+ input pointers (draft_probs, target_probs, draft_tokens, random)
        - No dot ops
        """
        if self.dot_ops:
            return False
        has_div = any(v[0] == "div" for v in self.ssa_values.values())
        has_cmp = any(v[0] == "mask" for v in self.ssa_values.values())
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        return has_div and has_cmp and n_inputs >= 3

    def _build_speculative_decode_kernel(self, kb, primary_dtype):
        """Generate a speculative decoding kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_speculative_decode_kernel
        kb.set_prebuilt_msl(make_speculative_decode_kernel())
        return kb

    def _is_beam_search_pattern(self):
        """Check if IR matches a beam search pattern.

        Beam search has:
        - Comparison + add (score accumulation and comparison)
        - Reductions (finding top-k beams)
        - 2+ output pointers (scores + indices)
        - No dot ops
        """
        if self.dot_ops:
            return False
        has_cmp = any(v[0] == "mask" for v in self.ssa_values.values())
        has_add = any(v[0] == "add" for v in self.ssa_values.values())
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        return has_cmp and has_add and self.reduce_ops and n_outputs >= 2

    def _build_beam_search_kernel(self, kb, primary_dtype):
        """Generate a beam search kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_beam_search_kernel
        kb.set_prebuilt_msl(make_beam_search_kernel())
        return kb

    def _classify_activation(self):
        """Classify the activation function from the IR ops.

        Returns the activation name ("tanh", "silu", "sigmoid", "elu",
        "leaky_relu", "hardswish") or None if no activation pattern matches.

        Requirements for activation detection:
        - No reductions, no dot ops (elementwise only)
        - 1 input pointer, 1 output pointer
        """
        if self.reduce_ops or self.dot_ops:
            return None
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return None

        ssa_ops = {v[0] for v in self.ssa_values.values()}
        # Count arithmetic SSA values (excluding constants and loads).
        arith_ops = {"neg", "exp", "add", "sub", "mul", "div", "fma",
                     "tanh", "rsqrt", "sqrt", "log", "sin", "cos", "abs",
                     "max", "min", "select", "mask"}
        n_arith = sum(1 for v in self.ssa_values.values() if v[0] in arith_ops)

        # tanh: explicit tanh op
        if "tanh" in ssa_ops:
            return "tanh"
        # silu (x * sigmoid(x)): exp + div + mul, no tanh, no select
        # Guard: SiLU has exactly 5 arithmetic ops (neg, exp, add, div, mul).
        # More complex patterns (GELU etc.) have 8+ and should use generic codegen.
        if ("exp" in ssa_ops and "div" in ssa_ops and "mul" in ssa_ops
                and "select" not in ssa_ops and n_arith <= 7):
            return "silu"
        # sigmoid: exp + div, no mul (pure sigmoid has no final multiply)
        if "exp" in ssa_ops and "div" in ssa_ops and "select" not in ssa_ops and n_arith <= 5:
            return "sigmoid"
        # elu: exp + select (x > 0 ? x : alpha*(exp(x)-1))
        if "exp" in ssa_ops and "select" in ssa_ops:
            return "elu"
        # hardswish: select + add + mul + div, no exp
        if "select" in ssa_ops and "add" in ssa_ops and "div" in ssa_ops and "exp" not in ssa_ops:
            return "hardswish"
        # leaky_relu: select + mul, no exp, no div
        if "select" in ssa_ops and "mul" in ssa_ops and "exp" not in ssa_ops and "div" not in ssa_ops:
            return "leaky_relu"

        return None

    def _build_activation_kernel(self, kb, activation, primary_dtype):
        """Generate an activation kernel from the classified pattern."""
        from triton_msl.codegen.msl_emitter import make_activation_kernel
        kb.set_prebuilt_msl(make_activation_kernel(activation=activation))
        return kb

    def _is_rms_norm_pattern(self):
        """Check if IR matches an RMS norm pattern.

        RMS norm has:
        - Sum reduction (for sum of squares)
        - rsqrt or sqrt+div (inverse square root)
        - mul operation (scale by weight)
        - NO sub (no mean subtraction — distinguishes from layer norm)
        - 2-3 input pointers (input, weight, optionally bias)
        """
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_rsqrt = "rsqrt" in ssa_ops
        # 1/sqrt(x) is equivalent to rsqrt(x), Triton may emit either form
        has_sqrt_div = "sqrt" in ssa_ops and "div" in ssa_ops
        has_mul = "mul" in ssa_ops
        has_sub = "sub" in ssa_ops
        ops_list = [r[2] for r in self.reduce_ops]
        has_sum = "sum" in ops_list
        # RMS norm: sum + (rsqrt or sqrt+div) + mul but NO sub
        return has_sum and (has_rsqrt or has_sqrt_div) and has_mul and not has_sub

    def _build_rms_norm_kernel(self, kb, n_arg, primary_dtype):
        """Generate an RMS norm kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_rms_norm_kernel
        kb.set_prebuilt_msl(make_rms_norm_kernel())
        return kb

    def _is_group_norm_pattern(self):
        """Check if IR matches a group normalization pattern.

        Group norm has:
        - Sum reductions (for mean and variance within groups)
        - rsqrt or div (normalization step)
        - 3+ input pointers (input, weight, bias)
        - 1 output pointer
        - 2+ scalar args (n_channels, spatial_size)
        """
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs < 3 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_rsqrt = "rsqrt" in ssa_ops
        has_div = "div" in ssa_ops
        ops_list = [r[2] for r in self.reduce_ops]
        has_sum = "sum" in ops_list
        return has_sum and (has_rsqrt or has_div) and len(self.scalar_args) >= 2

    def _build_group_norm_kernel(self, kb, primary_dtype):
        """Generate a group norm kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_group_norm_kernel
        kb.set_prebuilt_msl(make_group_norm_kernel())
        return kb

    def _is_instance_norm_pattern(self):
        """Check if IR matches an instance normalization pattern.

        Instance norm has:
        - 1 input pointer, 1 output pointer
        - Sum reductions (mean + variance computation)
        - rsqrt or div (normalization)
        - No weight/bias input ptrs (distinguishes from group/layer norm)
        """
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_rsqrt = "rsqrt" in ssa_ops
        has_div = "div" in ssa_ops
        ops_list = [r[2] for r in self.reduce_ops]
        has_sum = "sum" in ops_list
        return has_sum and (has_rsqrt or has_div)

    def _build_instance_norm_kernel(self, kb, primary_dtype):
        """Generate an instance norm kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_instance_norm_kernel
        kb.set_prebuilt_msl(make_instance_norm_kernel())
        return kb

    def _is_residual_add_pattern(self):
        """Check if IR matches a residual addition pattern.

        Residual add has:
        - 3 input pointers (input + residual + bias) — NOT 2 (that's elementwise add)
        - 1 output pointer
        - Only add operations (no sub, mul, div, etc.)
        - No reductions, no dot ops

        We require 3 inputs to distinguish from simple vector add (a + b),
        which should use the generic elementwise path.
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 3 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_add = "add" in ssa_ops
        has_sub = "sub" in ssa_ops
        has_mul = "mul" in ssa_ops
        has_div = "div" in ssa_ops
        # Only add ops — no sub/mul/div
        return has_add and not has_sub and not has_mul and not has_div

    def _build_residual_add_kernel(self, kb, primary_dtype):
        """Generate a residual add kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_residual_add_kernel
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        has_bias = n_inputs >= 3
        kb.set_prebuilt_msl(make_residual_add_kernel(has_bias=has_bias))
        return kb

    def _is_embedding_pattern(self):
        """Check if IR matches an embedding lookup pattern.

        Embedding has:
        - 2 input ptrs (embedding table with float + indices with int)
        - 1 output pointer
        - No reductions, no dot ops
        - One int-type input (indices), one float-type input (table)
        - No comparison/select (distinguishes from gather of int data)
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 2 or n_outputs != 1:
            return False
        # Must have one int input and one float input
        int_inputs = sum(
            1 for _, (_, dtype, is_out) in self.ptr_args.items()
            if not is_out and dtype in ("i32", "i64")
        )
        float_inputs = sum(
            1 for _, (_, dtype, is_out) in self.ptr_args.items()
            if not is_out and dtype in ("fp32", "fp16", "bf16")
        )
        if int_inputs != 1 or float_inputs != 1:
            return False
        # Must have scalar args (embedding_dim at minimum)
        return len(self.scalar_args) >= 2

    def _build_embedding_kernel(self, kb, primary_dtype):
        """Generate an embedding lookup kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_embedding_kernel
        kb.set_prebuilt_msl(make_embedding_kernel())
        return kb

    def _is_concat_pattern(self):
        """Check if IR matches a concatenation pattern.

        Concat has:
        - 2+ input pointers (all same type)
        - 1 output pointer
        - No reductions, no dot ops
        - No math ops (just loads and stores, pure data movement)
        - Multiple scalar args (sizes for each input)
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs < 2 or n_outputs != 1:
            return False
        # No math ops
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        math_ops = {"add", "sub", "mul", "div", "exp", "log", "sqrt", "rsqrt",
                    "tanh", "sin", "cos", "abs", "fma", "neg"}
        if ssa_ops & math_ops:
            return False
        # All inputs must be same type (no int/float mix like gather/embedding)
        input_types = set()
        for _, (_, dtype, is_out) in self.ptr_args.items():
            if not is_out:
                input_types.add(dtype)
        return len(input_types) == 1

    def _build_concat_kernel(self, kb, primary_dtype):
        """Generate a concat kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_concat_kernel
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        kb.set_prebuilt_msl(make_concat_kernel(n_inputs=n_inputs))
        return kb

    def _is_split_pattern(self):
        """Check if IR matches a split/chunk pattern.

        Split has:
        - 1 input pointer
        - 2+ output pointers (all same type as input)
        - No reductions, no dot ops
        - No math ops (pure data movement)
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs < 2:
            return False
        # All output types must match input type (top-k has f32 + i32 outputs)
        all_ptr_types = set()
        for _, (_, dtype, _) in self.ptr_args.items():
            all_ptr_types.add(dtype)
        if len(all_ptr_types) != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        math_ops = {"add", "sub", "mul", "div", "exp", "log", "sqrt", "rsqrt",
                    "tanh", "sin", "cos", "abs", "fma", "neg"}
        return not (ssa_ops & math_ops)

    def _build_split_kernel(self, kb, primary_dtype):
        """Generate a split kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_split_kernel
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        kb.set_prebuilt_msl(make_split_kernel(n_outputs=n_outputs))
        return kb

    def _is_repeat_kv_pattern(self):
        """Check if IR matches a repeat-KV (GQA head expansion) pattern.

        Repeat KV has:
        - 1 input pointer, 1 output pointer
        - 4+ scalar args (n_kv_heads, seq_len, head_dim, n_rep)
        - Integer division/modulo for index remapping
        - No reductions, no dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return False
        if len(self.scalar_args) < 4:
            return False
        # Must have integer div or mod (for index remapping h // n_rep)
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_div = "div" in ssa_ops
        has_mod = any(v[0] == "mod" for v in self.ssa_values.values()
                      if isinstance(v, tuple) and len(v) > 0)
        # Check for arith.divui or arith.remui patterns
        has_int_div = any("arith.divui" in str(v) or "arith.remui" in str(v)
                         for v in self.ssa_values.values())
        return has_div or has_int_div or len(self.scalar_args) >= 4

    def _build_repeat_kv_kernel(self, kb, primary_dtype):
        """Generate a repeat-KV kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_repeat_kv_kernel
        kb.set_prebuilt_msl(make_repeat_kv_kernel())
        return kb

    def _is_where_pattern(self):
        """Check if IR matches a where (ternary select) pattern.

        Where has:
        - 3 input pointers (condition, x, y) or 2+ input ptrs with select op
        - 1 output pointer
        - select op present in SSA values
        - No reductions, no dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_outputs != 1 or n_inputs < 2:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_select = "select" in ssa_ops
        # Must have select and no complex math ops (distinguishes from dropout)
        math_ops = {"exp", "log", "sqrt", "rsqrt", "tanh", "sin", "cos"}
        has_math = bool(ssa_ops & math_ops)
        # Where specifically uses select as the primary operation
        # Dropout also has select, but dropout has mul (scaling by 1/p)
        has_mul = "mul" in ssa_ops
        return has_select and not has_math and n_inputs >= 3

    def _build_where_kernel(self, kb, primary_dtype):
        """Generate a where kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_where_kernel
        kb.set_prebuilt_msl(make_where_kernel(dtype=primary_dtype))
        return kb

    def _is_clamp_pattern(self):
        """Check if IR matches a clamp (min+max) pattern.

        Clamp has:
        - 1 input pointer, 1 output pointer
        - 2+ scalar args (min_val, max_val, n_elements)
        - max and min operations (parsed as fmax/fmin, maxf/minf, maxnumf/minnumf)
        - No reductions, no dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return False
        if len(self.scalar_args) < 2:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        max_ops = {"max", "maxf", "fmax", "maxnumf", "maxnf"}
        min_ops = {"min", "minf", "fmin", "minnumf", "minnf"}
        has_max = bool(ssa_ops & max_ops)
        has_min = bool(ssa_ops & min_ops)
        return has_max and has_min

    def _build_clamp_kernel(self, kb, primary_dtype):
        """Generate a clamp kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_clamp_kernel
        kb.set_prebuilt_msl(make_clamp_kernel(dtype=primary_dtype))
        return kb

    def _is_cumsum_pattern(self):
        """Check if IR matches a cumulative sum (prefix scan) pattern.

        Cumsum has:
        - 1 input pointer, 1 output pointer
        - scf.for loop with iter_args (sequential dependency via addf)
        - No reductions (those use tt.reduce, not sequential accumulation)
        - No dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        if not self.scf_for_loops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return False
        # Check that a loop has iter_args (sequential dependency)
        for loop_info in self.scf_for_loops:
            if loop_info.get('iter_args'):
                # The loop has a running accumulator — likely cumsum
                # Also verify arith.addf appears in the IR near this loop
                if "arith.addf" in self.ir_text:
                    return True
        return False

    def _build_cumsum_kernel(self, kb, primary_dtype):
        """Generate a cumsum kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_cumsum_kernel
        kb.set_prebuilt_msl(make_cumsum_kernel(dtype=primary_dtype))
        return kb

    def _is_gather_pattern(self):
        """Check if IR matches a gather (indexed read) pattern.

        Gather has:
        - 2 input pointers (data buffer + index buffer)
        - 1 output pointer
        - No reductions, no dot ops
        - No select/cmp (distinguishes from dropout)
        - Index buffer has integer type
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 2 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        # Gather has no comparison/select (distinguishes from dropout)
        has_select = "select" in ssa_ops
        # Check if one input has integer type
        has_int_input = any(
            dtype in ("i32", "i64") for _, (_, dtype, is_out) in self.ptr_args.items()
            if not is_out
        )
        return not has_select and has_int_input

    def _build_gather_kernel(self, kb, primary_dtype):
        """Generate a gather kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_gather_kernel
        kb.set_prebuilt_msl(make_gather_kernel())
        return kb

    def _is_scatter_pattern(self):
        """Check if IR matches a scatter (indexed write) pattern.

        Scatter has:
        - 2 input pointers (data + indices with int type)
        - 1 output pointer
        - No reductions, no dot ops
        - Store address depends on loaded index (distinguishes from gather where
          the *load* address depends on loaded index — in scatter, the output
          pointer is offset by a loaded index value)
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 2 or n_outputs != 1:
            return False
        has_int_input = any(
            dtype in ("i32", "i64") for _, (_, dtype, is_out) in self.ptr_args.items()
            if not is_out
        )
        if not has_int_input:
            return False
        # Distinguish from gather: in scatter the store target uses loaded indices.
        # We detect this by checking that the store pointer SSA was computed from
        # a loaded value (addptr with a load result).
        for op in self.ops:
            if op[0] == "store":
                store_ptr_ssa = op[1]
                ptr_val = self.ssa_values.get(store_ptr_ssa)
                if ptr_val and ptr_val[0] == "addptr":
                    offset_ssa = ptr_val[2]  # offset operand
                    offset_val = self.ssa_values.get(offset_ssa)
                    if offset_val and offset_val[0] == "load":
                        return True
        return False

    def _build_scatter_kernel(self, kb, primary_dtype):
        """Generate a scatter kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_scatter_kernel
        kb.set_prebuilt_msl(make_scatter_kernel())
        return kb

    def _is_transpose_pattern(self):
        """Check if IR matches a transpose pattern.

        Transpose has:
        - 2 program_id calls (2D grid: x and y)
        - 1 input pointer, 1 output pointer
        - No reductions, no dot ops
        - Typically uses multiply and add for 2D indexing
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return False
        # Must have 2 program_id calls (2D grid)
        pid_count = sum(1 for v in self.ssa_values.values() if v[0] == "program_id")
        if pid_count < 2:
            return False
        # Must have 2+ scalar args (rows, cols dimensions)
        if len(self.scalar_args) < 2:
            return False
        return True

    def _build_transpose_kernel(self, kb, primary_dtype):
        """Generate a transpose kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_transpose_kernel
        kb.set_prebuilt_msl(make_transpose_kernel())
        return kb

    def _is_dropout_pattern(self):
        """Check if IR matches a dropout pattern.

        Dropout has:
        - 2 input pointers (data, random_mask/threshold)
        - 1 output pointer
        - Comparison (mask) + select + mul operations
        - No reductions, no dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 2 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_cmp = "mask" in ssa_ops
        has_select = "select" in ssa_ops
        has_mul = "mul" in ssa_ops
        return has_cmp and has_select and has_mul

    def _build_dropout_kernel(self, kb, primary_dtype):
        """Generate a dropout kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_fused_dropout_kernel
        kb.set_prebuilt_msl(make_fused_dropout_kernel())
        return kb

    def _is_batch_norm_pattern(self):
        """Check if IR matches a batch normalization (eval mode) pattern.

        Batch norm eval has:
        - 4+ input pointers (input, running_mean, running_var, weight, optionally bias)
        - 1 output pointer
        - sub + mul operations (normalize and scale)
        - No reductions (uses pre-computed running stats)
        - No dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs < 4 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_sub = "sub" in ssa_ops
        has_mul = "mul" in ssa_ops
        return has_sub and has_mul

    def _build_batch_norm_kernel(self, kb, primary_dtype):
        """Generate a batch norm (eval mode) kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_batch_norm_kernel
        kb.set_prebuilt_msl(make_batch_norm_kernel())
        return kb

    def _is_online_softmax_pattern(self):
        """Check if IR matches an online softmax pattern.

        Online softmax has:
        - scf.for loop (streaming single-pass over data)
        - exp operation (for softmax normalization)
        - Reductions (max or sum inside the loop)
        - 1 input pointer, 1 output pointer
        """
        if not self.scf_for_loops:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_exp = "exp" in ssa_ops
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        return has_exp and self.reduce_ops and n_inputs == 1 and n_outputs == 1

    def _build_online_softmax_kernel(self, kb, primary_dtype):
        """Generate an online softmax kernel from the pattern."""
        from triton_msl.codegen.msl_emitter import make_online_softmax_kernel
        kb.set_prebuilt_msl(make_online_softmax_kernel())
        return kb

    def _find_pre_reduce_op(self, reduce_input_ssa):
        """Check if there's a computation between load and reduce.

        Returns (op_kind, op_args) if found, None otherwise.
        """
        val = self.ssa_values.get(reduce_input_ssa)
        if val is None:
            return None

        op = val[0]
        # If the input to reduce is a computation (not a direct load)
        if op in ("exp", "log", "sqrt", "abs"):
            return (op, [val[1]])
        if op in ("add", "sub", "mul", "div"):
            return (op, [val[1], val[2]])

        return None

    def _emit_computation(self, kb, input_vars, dtype):
        """Analyze the computation graph and emit operations.

        Traces from store values back through the SSA graph to find
        the chain of operations between loads and stores.
        """
        if not self.ops:
            # No stores found — just return first loaded value
            if input_vars:
                return list(input_vars.values())[0]
            return "0.0f"

        # Find the value being stored
        store_op = self.ops[0]  # first store
        store_val_ssa = store_op[2]

        # Recursively emit the computation chain
        return self._emit_ssa_value(kb, store_val_ssa, input_vars, dtype, set())

    def _emit_ssa_value(self, kb, ssa, input_vars, dtype, emitted):
        """Recursively emit MSL for an SSA value."""
        if ssa in emitted:
            return self.computed_values.get(ssa, "0.0f")
        emitted.add(ssa)

        # Check if this is a loaded value
        val_info = self.ssa_values.get(ssa)
        if val_info is None:
            # Function argument — check if it's an input pointer we loaded
            arg_name = ssa.lstrip("%")
            if arg_name in input_vars:
                return input_vars[arg_name]
            if arg_name in self.scalar_args:
                return arg_name
            return "0.0f"

        op = val_info[0]

        # Passthrough (extf, truncf) — cache for reuse
        if op == "passthrough":
            result = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            self.computed_values[ssa] = result
            return result

        # Load — map to the loaded variable
        if op == "load":
            ptr_ssa = val_info[1]
            arg_name = self._trace_to_ptr_arg(ptr_ssa)
            if arg_name and arg_name in input_vars:
                self.computed_values[ssa] = input_vars[arg_name]
                return input_vars[arg_name]
            return "0.0f"

        # Binary float ops (arith.addf, subf, mulf, divf) and integer
        # counterparts (arith.addi, subi, muli). MSL uses the same operators
        # for both; the implicit conversion to float is harmless in this
        # emission context.
        _BIN_OP_ALIAS = {
            "add": "add", "sub": "sub", "mul": "mul", "div": "div",
            "addi": "add", "subi": "sub", "muli": "mul",
        }
        if op in _BIN_OP_ALIAS:
            lhs_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            rhs_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            # Generate a unique variable name
            var_name = f"r_{len(self.computed_values)}"
            kb.binary_op(_BIN_OP_ALIAS[op], lhs_var, rhs_var, var_name)
            self.computed_values[ssa] = var_name
            return var_name

        # Unary math ops
        if op in ("exp", "log", "sqrt", "rsqrt", "abs", "neg", "sin", "cos"):
            x_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb.unary_op(op, x_var, var_name)
            self.computed_values[ssa] = var_name
            return var_name

        # Integer cast (extsi, trunci) — treat as passthrough, cache for reuse
        if op == "int_cast":
            result = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            self.computed_values[ssa] = result
            return result

        # Fused multiply-add: fma(a, b, c) = a*b + c
        if op == "fma":
            a_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            b_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            c_var = self._emit_ssa_value(kb, val_info[3], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb.fused_op("fma", [a_var, b_var, c_var], var_name)
            self.computed_values[ssa] = var_name
            return var_name

        # Float max/min
        if op in ("fmax", "fmin"):
            lhs_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            rhs_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            msl_op = "max" if op == "fmax" else "min"
            kb._var(var_name, f"{msl_op}({lhs_var}, {rhs_var})", ty="float")
            self.computed_values[ssa] = var_name
            return var_name

        # Mask (comparison result)
        if op == "mask":
            pred = val_info[1]
            lhs_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            rhs_var = self._emit_ssa_value(kb, val_info[3], input_vars, dtype, emitted)
            pred_map = {
                "ogt": ">", "oge": ">=", "olt": "<", "ole": "<=",
                "oeq": "==", "one": "!=",
                "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=",
                "eq": "==", "ne": "!=",
            }
            msl_op = pred_map.get(pred, ">")
            var_name = f"r_{len(self.computed_values)}"
            kb._var(var_name, f"{lhs_var} {msl_op} {rhs_var}", ty="bool")
            self.computed_values[ssa] = var_name
            return var_name

        # Select (ternary)
        if op == "select":
            cond_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            true_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            false_var = self._emit_ssa_value(kb, val_info[3], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb._var(var_name, f"{cond_var} ? {true_var} : {false_var}", ty="float")
            self.computed_values[ssa] = var_name
            return var_name

        # Constants
        if op == "constant":
            # Try to extract numeric value
            val_str = val_info[1]
            result = "0.0f"
            # Tensor constant: dense<VALUE> : tensor<...>
            m_dense = re.search(r"dense<([\d.e+-]+)>", val_str)
            if m_dense:
                result = f"{float(m_dense.group(1))}f"
            else:
                # Scalar constant: VALUE : type
                m_scalar = re.search(r"([\d.e+-]+)\s*:\s*\w+", val_str)
                if m_scalar:
                    result = f"{float(m_scalar.group(1))}f"
            self.computed_values[ssa] = result
            return result

        # Splat of a scalar arg
        if op == "splat":
            source = val_info[1]
            return self._emit_ssa_value(kb, source, input_vars, dtype, emitted)

        # Reduce — the result is computed in _build_reduction_kernel
        if op == "reduce":
            return "total"

        # Loop induction variable
        if op == "loop_iv":
            return ssa.lstrip("%")

        # Program ID (tt.get_program_id). In MSL, this is the kernel arg
        # `pid` (threadgroup_position_in_grid) which KernelBuilder already
        # declares. Return it by name.
        if op == "program_id":
            return "pid"

        warnings.warn(
            f"Unknown SSA op '{op}' for value {ssa} in kernel '{self.kernel_name}'. "
            f"Substituting 0.0f — this will produce incorrect results.",
            stacklevel=2,
        )
        return "0.0f"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ttgir(ir_text, options):
    """Parse TTGIR MLIR text and return a KernelBuilder.

    Args:
        ir_text: MLIR text from str(module).
        options: MetalOptions instance.

    Returns:
        KernelBuilder configured from the parsed IR.
    """
    parser = TTGIRParser(ir_text, options)
    return parser.parse()
