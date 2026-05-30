"""Structured error types for triton-metal.

Provides clear, actionable error messages for common failure modes:
- Shader compilation failures
- Codegen lowering failures
- Pre-lowering validation failures
- Hardware-unsupported operations
- Not-yet-implemented operations
- Kernel dispatch/launch failures
"""


class MetalCompilationError(RuntimeError):
    """MSL shader compilation failed (xcrun metal returned an error)."""

    def __init__(self, message, msl_source=None, stderr=None):
        self.msl_source = msl_source
        self.stderr = stderr
        parts = [message]
        if stderr:
            parts.append(f"\nCompiler output:\n{stderr}")
        super().__init__("\n".join(parts))


class MetalCodegenError(RuntimeError):
    """TTGIR → MSL lowering failed for a specific operation."""

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        self.op_name = op_name
        self.ssa_id = ssa_id
        self.type_str = type_str
        parts = [message]
        if op_name:
            parts.append(f"  op: {op_name}")
        if ssa_id:
            parts.append(f"  ssa: {ssa_id}")
        if type_str:
            parts.append(f"  type: {type_str}")
        super().__init__("\n".join(parts))


class MetalUnsupportedError(MetalCodegenError):
    """Operation requires hardware features not available on Apple GPUs.

    Examples: FP64 arithmetic, FP8 types, TF32 tensor cores.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        full_msg = f"Hardware unsupported: {message}"
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalValidationError(MetalCodegenError):
    """Pre-lowering validation failed.

    Raised when IR validation detects issues before codegen begins,
    e.g. unsupported tensor ranks, invalid block sizes, or type mismatches
    that can be caught statically.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None, constraint=None):
        self.constraint = constraint
        full_msg = f"Validation failed: {message}"
        if constraint:
            full_msg += f"\n  constraint: {constraint}"
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalNotImplementedError(MetalCodegenError):
    """Operation is not yet implemented in triton-metal but could be.

    This indicates a gap in the compiler, not a hardware limitation.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        full_msg = (
            f"Not yet implemented: {message}\n"
            f"  If you need this, please file an issue at "
            f"https://github.com/bledden/triton-metal/issues"
        )
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalNonRecoverableError(MetalCodegenError):
    """The lowerer recognized a kernel it cannot lower CORRECTLY, and knows
    the legacy fallback parser cannot either, so it refuses rather than emit
    silently-wrong output.

    Unlike ``MetalNotImplementedError`` (which signals "I can't, but a
    fallback path might"), this is raised when falling back would only
    substitute one wrong result for another — e.g. a pid-tiled matmul whose
    full dimensions are baked in as ``tl.constexpr`` (no runtime M/N/K), so
    no template can derive the true output strides. ``emit_msl`` re-raises
    this instead of falling back, turning a silent-wrong into a clear error.
    Integrity guarantee: the backend never returns numbers it can't vouch for.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        full_msg = (
            f"Refusing to emit silently-wrong output: {message}\n"
            f"  This kernel pattern is not supported and cannot be safely "
            f"approximated. File an issue at "
            f"https://github.com/bledden/triton-metal/issues"
        )
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalLaunchError(RuntimeError):
    """Kernel dispatch failed at launch time.

    Raised when a compiled kernel cannot be dispatched to the GPU,
    e.g. buffer allocation failures, invalid grid dimensions, or
    Metal command buffer errors.
    """

    def __init__(self, message, kernel_name=None, grid=None, reason=None):
        self.kernel_name = kernel_name
        self.grid = grid
        self.reason = reason
        parts = [message]
        if kernel_name:
            parts.append(f"  kernel: {kernel_name}")
        if grid:
            parts.append(f"  grid: {grid}")
        if reason:
            parts.append(f"  reason: {reason}")
        super().__init__("\n".join(parts))
