"""Lower the body of a device function (noinline callee) to MSL lines.

This is a lightweight version of ``GenericLowerer`` that operates on a
``CalledFunc`` instead of an ``IRGraph``. It reuses the same op dispatch but
produces raw MSL lines instead of using ``KernelBuilder``.

Lives in its own file (extracted from the 9.9 kLOC ``generic_lowerer.py``)
because it is logically separate from the kernel lowerer — it only handles
device-function callees.
"""

from typing import List

from triton_msl.codegen.mlir_walker import CalledFunc, SSAValue
from triton_msl.codegen.msl_emitter import _msl_compute_type
from triton_msl.codegen.msl_types import triton_type_to_msl

from triton_msl.codegen._lowerer_helpers import (
    _mlir_to_triton_dtype,
    CMPI_NAMED,
    CMPF_NAMED,
)


class _DeviceFuncLowerer:
    """Lower the body of a device function (noinline callee) to MSL lines.

    This is a lightweight version of GenericLowerer that operates on
    a CalledFunc instead of an IRGraph. It reuses the same op dispatch
    but produces raw MSL lines instead of using KernelBuilder.
    """

    def __init__(self, cfunc: CalledFunc, options=None):
        self.cfunc = cfunc
        self.options = options
        self.env = {}
        self.env_types = {}
        self.env_is_mask = {}
        self.env_is_ptr = {}
        self._var_counter = 0
        self._lines = []

    def _next_var(self, prefix="r") -> str:
        name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        return name

    def _lookup(self, ssa_id: int) -> str:
        if ssa_id in self.env:
            return self.env[ssa_id]
        return f"UNKNOWN_{ssa_id}"

    def _emit(self, line: str):
        self._lines.append(f"    {line}")

    def lower_body(self) -> List[str]:
        """Lower all ops and return lines of MSL body code."""
        # Register function arguments
        for arg in self.cfunc.args:
            self.env[arg.id] = arg.name
            self.env_types[arg.id] = _mlir_to_triton_dtype(arg.elem_type)
            if arg.is_ptr:
                self.env_is_ptr[arg.id] = (arg.name, None)

        # Lower each op
        for ssa in self.cfunc.ops:
            self._lower_op(ssa)

        return self._lines

    def _lower_op(self, ssa: SSAValue):
        """Lower a single op in a device function."""
        op = ssa.op

        if op == "tt.return":
            self._lower_return(ssa)
        elif op == "tt.call":
            self._lower_call(ssa)
        elif op == "tt.load":
            self._lower_load(ssa)
        elif op == "tt.store":
            self._lower_store(ssa)
        elif op == "tt.addptr":
            self._lower_addptr(ssa)
        elif op == "tt.splat":
            self._emit_passthrough(ssa)
        elif op == "tt.broadcast":
            self._emit_passthrough(ssa)
        elif op == "arith.constant":
            self._lower_constant(ssa)
        elif op.startswith("arith."):
            self._lower_arith(ssa)
        elif op.startswith("math."):
            self._lower_math(ssa)
        elif op == "scf.if":
            self._lower_scf_if(ssa)
        elif op in ("scf.yield", "scf.condition"):
            pass
        elif op == "scf.for":
            self._lower_scf_for(ssa)
        elif op == "tt.extern_elementwise":
            self._lower_extern_elementwise(ssa)
        elif op == "tt.fp_to_fp":
            # FP8 cast in device function — passthrough (compute in float)
            self._emit_passthrough(ssa)
            if ssa.elem_type:
                self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type)
        elif op.startswith("ttg.") or op in ("tt.reshape", "tt.expand_dims",
                                               "tt.unsplat", "tt.make_range"):
            self._emit_passthrough(ssa)
        else:
            self._emit(f"// UNSUPPORTED in device func: {op}")

    def _lower_return(self, ssa: SSAValue):
        """tt.return with value(s) in a device function."""
        if not ssa.operand_ids:
            self._emit("return;")
        elif len(ssa.operand_ids) == 1:
            val = self._lookup(ssa.operand_ids[0])
            self._emit(f"return {val};")
        else:
            # Multi-value return: construct struct
            vals = [self._lookup(oid) for oid in ssa.operand_ids]
            ret_struct = f"_ret_{self.cfunc.name.replace('.', '_')}"
            fields = ", ".join(vals)
            self._emit(f"return {ret_struct}{{{fields}}};")

    def _lower_call(self, ssa: SSAValue):
        """tt.call in a device function (nested calls)."""
        callee = ssa.attrs.get("callee", "unknown_fn")
        safe_callee = callee.replace(".", "_")
        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        n_results = len(ssa.result_ids) if ssa.result_ids else (1 if ssa.type_str else 0)

        if n_results == 0 or not ssa.type_str:
            self._emit(f"{safe_callee}({args_str});")
        elif n_results == 1 or not ssa.result_ids:
            msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(ssa.elem_type or "f32"))
            var = self._next_var("r")
            self._emit(f"{msl_ty} {var} = {safe_callee}({args_str});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type or "f32")
        else:
            # Multi-return
            ret_struct = f"_ret_{safe_callee}"
            var = self._next_var("rv")
            self._emit(f"{ret_struct} {var} = {safe_callee}({args_str});")
            for i, rid in enumerate(ssa.result_ids):
                self.env[rid] = f"{var}.v{i}"
                self.env_types[rid] = _mlir_to_triton_dtype(ssa.elem_type or "f32")

    def _lower_load(self, ssa: SSAValue):
        """tt.load — scalar load in device function."""
        if not ssa.operand_ids:
            return
        ptr_id = ssa.operand_ids[0]
        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)
        msl_ty = _msl_compute_type(triton_dtype)
        var = self._next_var("ld")

        if ptr_id in self.env_is_ptr:
            base, offs = self.env_is_ptr[ptr_id]
            if offs:
                self._emit(f"{msl_ty} {var} = static_cast<{msl_ty}>({base}[{offs}]);")
            else:
                self._emit(f"{msl_ty} {var} = static_cast<{msl_ty}>({base}[0]);")
        else:
            ptr = self._lookup(ptr_id)
            self._emit(f"{msl_ty} {var} = static_cast<{msl_ty}>({ptr}[0]);")

        self.env[ssa.id] = var
        self.env_types[ssa.id] = triton_dtype

    def _lower_store(self, ssa: SSAValue):
        """tt.store — scalar store in device function."""
        if len(ssa.operand_ids) < 2:
            return
        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]
        val = self._lookup(val_id)

        if ptr_id in self.env_is_ptr:
            base, offs = self.env_is_ptr[ptr_id]
            if offs:
                self._emit(f"{base}[{offs}] = {val};")
            else:
                self._emit(f"{base}[0] = {val};")
        else:
            ptr = self._lookup(ptr_id)
            self._emit(f"{ptr}[0] = {val};")

    def _lower_addptr(self, ssa: SSAValue):
        """tt.addptr — pointer arithmetic."""
        if len(ssa.operand_ids) < 2:
            return
        ptr_id = ssa.operand_ids[0]
        off_id = ssa.operand_ids[1]
        off = self._lookup(off_id)

        if ptr_id in self.env_is_ptr:
            base, existing_off = self.env_is_ptr[ptr_id]
            if existing_off:
                combined = f"({existing_off} + {off})"
            else:
                combined = off
            self.env_is_ptr[ssa.id] = (base, combined)
        else:
            base = self._lookup(ptr_id)
            self.env_is_ptr[ssa.id] = (base, off)

        self.env[ssa.id] = self._lookup(ptr_id)
        if ptr_id in self.env_types:
            self.env_types[ssa.id] = self.env_types[ptr_id]

    def _lower_constant(self, ssa: SSAValue):
        """arith.constant — emit a constant value."""
        val = ssa.attrs.get("value")
        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)

        if val is None:
            val = 0

        if isinstance(val, bool):
            msl_val = "true" if val else "false"
            msl_ty = "bool"
        elif isinstance(val, float) or (isinstance(val, int) and elem.startswith("f")):
            msl_val = f"{float(val)}f"
            msl_ty = "float"
        else:
            msl_val = str(val)
            msl_ty = triton_type_to_msl(triton_dtype)

        var = self._next_var("c")
        self._emit(f"{msl_ty} {var} = {msl_val};")
        self.env[ssa.id] = var
        self.env_types[ssa.id] = triton_dtype

    def _lower_arith(self, ssa: SSAValue):
        """Lower arith.* ops."""
        op = ssa.op
        ids = ssa.operand_ids

        arith_map = {
            "arith.addf": "+", "arith.subf": "-",
            "arith.mulf": "*", "arith.divf": "/",
            "arith.addi": "+", "arith.subi": "-",
            "arith.muli": "*", "arith.divsi": "/", "arith.divui": "/",
            "arith.remsi": "%", "arith.remui": "%",
            "arith.andi": "&", "arith.ori": "|", "arith.xori": "^",
            "arith.maxnumf": "max", "arith.minnumf": "min",
            "arith.maximumf": "max", "arith.minimumf": "min",
            "arith.maxsi": "max", "arith.minsi": "min",
            "arith.maxui": "max", "arith.minui": "min",
            "arith.shrsi": ">>", "arith.shrui": ">>", "arith.shli": "<<",
        }

        if op in arith_map and len(ids) >= 2:
            a = self._lookup(ids[0])
            b = self._lookup(ids[1])
            symbol = arith_map[op]
            var = self._next_var("r")
            elem = ssa.elem_type or "f32"
            triton_dtype = _mlir_to_triton_dtype(elem)
            msl_ty = triton_type_to_msl(triton_dtype)

            if symbol in ("+", "-", "*", "/", "%", "&", "|", "^", ">>", "<<"):
                self._emit(f"{msl_ty} {var} = {a} {symbol} {b};")
            else:
                self._emit(f"{msl_ty} {var} = {symbol}({a}, {b});")

            self.env[ssa.id] = var
            self.env_types[ssa.id] = triton_dtype
            return

        if op in ("arith.cmpi", "arith.cmpf"):
            pred = ssa.attrs.get("predicate_name", "")
            pred_op = CMPF_NAMED.get(pred, CMPI_NAMED.get(pred, "=="))
            if len(ids) >= 2:
                a = self._lookup(ids[0])
                b = self._lookup(ids[1])
                var = self._next_var("cmp")
                self._emit(f"bool {var} = ({a} {pred_op} {b});")
                self.env[ssa.id] = var
                self.env_types[ssa.id] = "i1"
                self.env_is_mask[ssa.id] = True
            return

        if op == "arith.negf" and len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = -{a};")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
            return

        if op == "arith.select" and len(ids) >= 3:
            cond = self._lookup(ids[0])
            a = self._lookup(ids[1])
            b = self._lookup(ids[2])
            var = self._next_var("sel")
            self._emit(f"float {var} = {cond} ? {a} : {b};")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
            return

        if op in ("arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi",
                   "arith.extsi", "arith.extui", "arith.trunci", "arith.uitofp",
                   "arith.fptoui", "arith.index_cast", "arith.index_castui",
                   "arith.bitcast"):
            self._emit_passthrough(ssa)
            if ssa.elem_type:
                self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type)
            return

        self._emit(f"// UNSUPPORTED arith in device func: {op}")

    def _lower_math(self, ssa: SSAValue):
        """Lower math.* ops."""
        op = ssa.op
        ids = ssa.operand_ids

        math_map = {
            "math.exp": "exp", "math.exp2": "exp2",
            "math.log": "log", "math.log2": "log2",
            "math.sqrt": "sqrt",
            "math.rsqrt": "rsqrt", "math.abs": "abs", "math.absf": "abs",
            "math.ceil": "ceil", "math.floor": "floor",
            "math.sin": "sin", "math.cos": "cos", "math.tanh": "tanh",
            "math.round": "round", "math.roundeven": "rint", "math.trunc": "trunc",
            "math.fma": "fma",
        }

        short = op.split(".")[-1] if "." in op else op
        func = math_map.get(op, short)

        if op == "math.fma" and len(ids) >= 3:
            a, b, c = [self._lookup(i) for i in ids[:3]]
            var = self._next_var("r")
            self._emit(f"float {var} = fma({a}, {b}, {c});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
        elif op == "math.log1p" and len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = log(1.0f + {a});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
        elif op == "math.expm1" and len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = (exp({a}) - 1.0f);")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
        elif len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = {func}({a});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"

    def _lower_extern_elementwise(self, ssa: SSAValue):
        """tt.extern_elementwise → direct MSL function call in device func."""
        func_name = ssa.attrs.get("symbol", "")
        if not func_name:
            func_name = ssa.attrs.get("libname", "")
        if not func_name:
            self._emit(f"// UNSUPPORTED: tt.extern_elementwise (no symbol)")
            return

        # Sanitize __nv_* CUDA libdevice names to Metal equivalents
        safe_name = func_name
        if safe_name.startswith("__nv_"):
            stripped = safe_name[5:]
            if stripped.endswith("f") and len(stripped) > 1:
                stripped = stripped[:-1]
            safe_name = stripped

        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)
        if triton_dtype.startswith("fp") or triton_dtype.startswith("bf"):
            msl_ty = "float"
        elif triton_dtype.startswith("u"):
            msl_ty = "uint"
        elif triton_dtype == "i64":
            msl_ty = "long"
        else:
            msl_ty = triton_type_to_msl(triton_dtype)

        var = self._next_var("r")
        self._emit(f"{msl_ty} {var} = {safe_name}({args_str});")
        self.env[ssa.id] = var
        self.env_types[ssa.id] = triton_dtype

    def _lower_scf_for(self, ssa: SSAValue):
        """scf.for → MSL for loop with iter_args in device function.

        Reuses the same logic as GenericLowerer._lower_scf_for():
        scf.for has operands: [start, end, step, init_0, init_1, ...]
        Results: [result_0, result_1, ...] (same count as iter_args)
        Body block args: [induction_var, iter_arg_0, iter_arg_1, ...]
        """
        if len(ssa.operand_ids) < 3:
            return

        start_var = self._lookup(ssa.operand_ids[0])
        end_var = self._lookup(ssa.operand_ids[1])
        step_var = self._lookup(ssa.operand_ids[2])

        # iter_args initial values: operands[3:]
        init_ids = ssa.operand_ids[3:]
        n_iter_args = len(init_ids)

        # Declare and initialize iter_arg variables
        iter_vars = []
        iter_dtypes = []
        result_elem = ssa.elem_type or "f32"
        for i, init_id in enumerate(init_ids):
            var_name = self._next_var("iter")
            init_val = self._lookup(init_id)
            init_type = self.env_types.get(init_id, "fp32")
            if result_elem in ("i64",) and init_type in ("i32", "fp32"):
                init_type = result_elem
            if init_type.startswith("f") or init_type.startswith("bf"):
                msl_type = "float"
            elif init_type in ("i64",):
                msl_type = "long"
            elif init_type.startswith("u"):
                msl_type = "uint"
            else:
                msl_type = "int"
            self._emit(f"{msl_type} {var_name} = {init_val};")
            iter_vars.append(var_name)
            iter_dtypes.append(init_type)

        # Emit for loop
        start_type = self.env_types.get(ssa.operand_ids[0], "i32")
        is_i64 = start_type == "i64" or "i64" in (ssa.type_str or "")
        loop_type = "long" if is_i64 else "int"
        loop_var = self._next_var("k")

        self._emit(
            f"for ({loop_type} {loop_var} = {start_var}; "
            f"{loop_var} < {end_var}; {loop_var} += {step_var}) {{"
        )

        # Map block args to MSL variables
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids:
            # First block arg is induction variable
            self.env[block_arg_ids[0]] = loop_var
            self.env_types[block_arg_ids[0]] = start_type
            # Remaining block args are iter_args
            for i, var in enumerate(iter_vars):
                if i + 1 < len(block_arg_ids):
                    self.env[block_arg_ids[i + 1]] = var
                    self.env_types[block_arg_ids[i + 1]] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"

        # Process body ops
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
                    # Update iter_arg variables from yield operands
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(iter_vars):
                            yield_val = self._lookup(yield_id)
                            self._emit(f"    {iter_vars[i]} = {yield_val};")
                else:
                    self._lower_op(body_op)

        self._emit("}")

        # Map scf.for results to iter_arg variables
        if ssa.result_ids:
            for i, var in enumerate(iter_vars):
                if i < len(ssa.result_ids):
                    self.env[ssa.result_ids[i]] = var
                    self.env_types[ssa.result_ids[i]] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"
        elif iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"

    def _lower_scf_if(self, ssa: SSAValue):
        """scf.if in a device function.

        For scf.if with results, we declare result variables before the if
        statement and assign them in each branch via scf.yield. This ensures
        the variables are in scope after the if block.
        """
        if not ssa.operand_ids:
            return
        cond = self._lookup(ssa.operand_ids[0])

        # Pre-declare result variables if the scf.if produces results
        # result_ids may be None for single-result scf.if (walker stores in ssa.id)
        rids = ssa.result_ids if ssa.result_ids else ([ssa.id] if ssa.type_str else [])
        result_vars = []
        if rids:
            for i, rid in enumerate(rids):
                var = self._next_var("if_res")
                # Determine type from elem_type or default to float
                msl_ty = "float"
                if ssa.elem_type and ssa.elem_type.startswith("i"):
                    msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(ssa.elem_type))
                self._emit(f"{msl_ty} {var};")
                result_vars.append(var)
                self.env[rid] = var
                self.env_types[rid] = _mlir_to_triton_dtype(ssa.elem_type or "f32")

        self._emit(f"if ({cond}) {{")

        if ssa.region_ops:
            for sub_op in ssa.region_ops:
                if sub_op.op == "scf.yield":
                    # Assign yielded values to pre-declared result variables
                    if result_vars and sub_op.operand_ids:
                        for var, yid in zip(result_vars, sub_op.operand_ids):
                            val = self._lookup(yid)
                            self._emit(f"    {var} = {val};")
                else:
                    self._lower_op(sub_op)

        if ssa.else_ops:
            self._emit("} else {")
            for sub_op in ssa.else_ops:
                if sub_op.op == "scf.yield":
                    # Assign yielded values to pre-declared result variables
                    if result_vars and sub_op.operand_ids:
                        for var, yid in zip(result_vars, sub_op.operand_ids):
                            val = self._lookup(yid)
                            self._emit(f"    {var} = {val};")
                else:
                    self._lower_op(sub_op)

        self._emit("}")

    def _emit_passthrough(self, ssa: SSAValue):
        """Emit a passthrough (type conversion that's a no-op)."""
        if ssa.operand_ids:
            src_id = ssa.operand_ids[0]
            self.env[ssa.id] = self._lookup(src_id)
            if src_id in self.env_types:
                self.env_types[ssa.id] = self.env_types[src_id]
            if src_id in self.env_is_mask:
                self.env_is_mask[ssa.id] = True
            if src_id in self.env_is_ptr:
                self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            # Propagate shape through passthrough
            if src_id in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[src_id]

