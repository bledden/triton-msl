"""Triton type to MSL type mappings."""

import warnings

# FP64 types that Metal cannot represent — downcast to float32 with a warning.
_FP64_TYPES = {"fp64", "f64"}

# Triton dtype string -> MSL type string
_TYPE_MAP = {
    "fp16": "half",
    "f16": "half",
    "bf16": "bfloat",  # Metal 3.1+ supports bfloat
    "fp32": "float",
    "f32": "float",
    "fp64": "float",  # Metal has no double — downcast to float32
    "f64": "float",   # Metal has no double — downcast to float32
    "i1": "bool",
    "i8": "char",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "u8": "uchar",
    "u16": "ushort",
    "u32": "uint",
    "u64": "ulong",
    # FP8 types — no hardware support on Metal, stored as uchar (uint8_t).
    # Software emulation converts to/from float for computation.
    "fp8e4nv": "uchar",    # e4m3 (4 exponent, 3 mantissa, bias 7)
    "fp8e5": "uchar",      # e5m2 (5 exponent, 2 mantissa, bias 15)
    "fp8e4b15": "uchar",   # e4m3 with bias 15
    "fp8e4b8": "uchar",    # e4m3 with bias 8
    "fp8e5b16": "uchar",   # e5m2 with bias 16
    "fp8_e4m3": "uchar",   # alias
    "fp8_e5m2": "uchar",   # alias
}

# Pointer types map to device pointers.
_PTR_QUALIFIER = "device"


def _warn_fp64_downcast(triton_type: str):
    """Warn when FP64 types are downcast to float32 on Metal.

    Apple Silicon GPUs have no FP64 hardware, so double-precision types are
    silently mapped to float32.  This warning lets users know about the
    precision loss so they can decide whether the downcast is acceptable.
    """
    base = triton_type.lstrip("*").strip()
    if base in _FP64_TYPES:
        warnings.warn(
            f"FP64 (double) is not supported on Apple Silicon GPUs. "
            f"Type '{triton_type}' will be downcast to float32. "
            f"Cast to float32 explicitly to silence this warning.",
            UserWarning,
            stacklevel=3,
        )


def triton_type_to_msl(triton_type: str) -> str:
    """Convert a Triton type string to its MSL equivalent.

    Args:
        triton_type: e.g. "fp32", "*fp16", "i32"

    Returns:
        MSL type string, e.g. "float", "device half*", "int"

    Warns:
        UserWarning: If the type is FP64, which is downcast to float32 on Metal.
    """
    _warn_fp64_downcast(triton_type)
    if triton_type.startswith("*"):
        inner = triton_type[1:]
        msl_inner = _TYPE_MAP.get(inner, inner)
        return f"{_PTR_QUALIFIER} {msl_inner}*"
    return _TYPE_MAP.get(triton_type, triton_type)


def triton_type_to_msl_const_ref(triton_type: str) -> str:
    """Convert a scalar Triton type to a constant reference MSL type.

    Used for kernel arguments passed as constant buffers.
    """
    msl_type = triton_type_to_msl(triton_type)
    return f"constant {msl_type}&"
