"""Triton → MLX backend: zero-copy Metal dispatch via mx.fast.metal_kernel().

Usage:
    import triton
    import triton.language as tl
    import triton_msl.mlx as tmlx
    import mlx.core as mx

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    x = mx.random.normal((1024,))
    y = mx.random.normal((1024,))
    out = mx.zeros((1024,))
    (result,) = tmlx.triton_call(
        add_kernel, x, y, out, 1024,
        grid=(4,), BLOCK=256,
    )
"""

from triton_msl.mlx.msl_extractor import extract_msl_for_mlx, MSLExtraction
from triton_msl.mlx.mlx_launcher import MLXLauncher

__all__ = ["triton_call", "mlx_available"]

# Cache: (fn_hash, sig_hash, constexpr_hash) → (MSLExtraction, metadata)
_compile_cache = {}


def mlx_available():
    """Check if MLX is available for metal_kernel dispatch."""
    try:
        import mlx.core as mx
        mx.fast.metal_kernel  # verify API exists
        return True
    except (ImportError, AttributeError):
        return False


def _mlx_dtype_to_triton_sig(dtype):
    """Map MLX dtype to Triton signature string."""
    import mlx.core as mx
    mapping = {
        mx.float32: "*fp32",
        mx.float16: "*fp16",
        mx.bfloat16: "*bf16",
        mx.int32: "*i32",
        mx.uint32: "*u32",
        mx.int16: "*i16",
        mx.uint16: "*u16",
        mx.int8: "*i8",
        mx.uint8: "*u8",
        mx.bool_: "*i1",
    }
    return mapping.get(dtype, "*fp32")


def _scalar_to_triton_sig(val):
    """Map a Python scalar to Triton signature string."""
    if isinstance(val, bool):
        return "i1"
    elif isinstance(val, int):
        return "i32"
    elif isinstance(val, float):
        return "fp32"
    return "i32"


def _build_signature(jit_fn, args, constexpr_kwargs):
    """Build Triton signature dict from @triton.jit fn and runtime args."""
    import mlx.core as mx

    arg_names = jit_fn.arg_names
    signature = {}
    constexprs = {}

    runtime_idx = 0
    for name in arg_names:
        if name in constexpr_kwargs:
            signature[name] = "constexpr"
            constexprs[name] = constexpr_kwargs[name]
        else:
            if runtime_idx >= len(args):
                raise ValueError(
                    f"Not enough args: expected arg for '{name}' at position {runtime_idx}"
                )
            arg = args[runtime_idx]
            if isinstance(arg, mx.array):
                signature[name] = _mlx_dtype_to_triton_sig(arg.dtype)
            elif arg is None:
                # Output placeholder — assume fp32 pointer
                signature[name] = "*fp32"
            else:
                signature[name] = _scalar_to_triton_sig(arg)
            runtime_idx += 1

    return signature, constexprs


def _compile_kernel(jit_fn, signature, constexprs):
    """Compile a @triton.jit function to MSL via Triton's pipeline."""
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend

    target = GPUTarget("metal", "apple-m4", 32)

    src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
    compiled = triton_compile(src, target=target, options={})

    msl_source = compiled.asm["msl"]
    metadata = compiled.metadata

    return msl_source, metadata, compiled


def _cache_key(jit_fn, signature, constexprs):
    """Build a hashable cache key."""
    sig_key = tuple(sorted(signature.items()))
    const_key = tuple(sorted((k, v) for k, v in constexprs.items()))
    fn_key = id(jit_fn)  # Use JITFunction identity
    return (fn_key, sig_key, const_key)


def triton_call(kernel_fn, *args, grid, num_warps=4, **constexpr_kwargs):
    """Call a @triton.jit kernel with MLX arrays via zero-copy Metal dispatch.

    Compiles the kernel to MSL, extracts the body for mx.fast.metal_kernel(),
    and dispatches with zero buffer copies.

    Args:
        kernel_fn: @triton.jit decorated function.
        *args: Kernel arguments in signature order (excluding constexpr).
            MLX arrays for pointer args, Python int/float for scalars.
            Output args should be MLX arrays (shape/dtype used for allocation).
        grid: Tuple of threadgroup counts, e.g. (4,) or (4, 2) or (4, 2, 1).
        num_warps: Warps (SIMD groups) per threadgroup. Default 4.
        **constexpr_kwargs: Compile-time constants (e.g. BLOCK_SIZE=256).

    Returns:
        List of output MLX arrays.

    Example:
        x = mx.random.normal((1024,))
        y = mx.random.normal((1024,))
        out = mx.zeros((1024,))
        (result,) = triton_call(
            add_kernel, x, y, out, 1024,
            grid=(4,), BLOCK=256,
        )
    """
    # Keep JITFunction for arg_names; use .fn for compilation
    jit_fn = kernel_fn

    signature, constexprs = _build_signature(jit_fn, args, constexpr_kwargs)
    key = _cache_key(jit_fn, signature, constexprs)

    if key in _compile_cache:
        extraction, block_size, needs_2d_grid = _compile_cache[key]
    else:
        msl_source, metadata, compiled = _compile_kernel(jit_fn, signature, constexprs)

        block_size = getattr(metadata, "block_size", num_warps * 32)
        output_arg_indices = getattr(metadata, "output_arg_indices", None)
        needs_2d_grid = getattr(metadata, "needs_2d_grid", False)

        extraction = extract_msl_for_mlx(msl_source, output_arg_indices)
        _compile_cache[key] = (extraction, block_size, needs_2d_grid)

    launcher = MLXLauncher(extraction, block_size=block_size,
                           needs_2d_grid=needs_2d_grid)
    return launcher(grid, *args)
