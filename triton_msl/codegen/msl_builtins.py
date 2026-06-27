"""Metal built-in function and qualifier mappings for Triton operations."""

# Triton's tt.get_program_id(axis) maps to Metal's threadgroup position.
# These are injected as kernel parameters with Metal attribute qualifiers.
PROGRAM_ID_QUALIFIERS = {
    0: ("uint", "tgid_x", "threadgroup_position_in_grid"),
    1: ("uint", "tgid_y", "threadgroup_position_in_grid"),
    2: ("uint", "tgid_z", "threadgroup_position_in_grid"),
}

# Thread-level position qualifiers.
THREAD_QUALIFIERS = {
    "thread_id": ("uint", "tid", "thread_position_in_grid"),
    "local_id": ("uint", "lid", "thread_position_in_threadgroup"),
    "simd_lane": ("uint", "simd_lane", "thread_index_in_simdgroup"),
    "simd_group": ("uint", "simd_group", "simdgroup_index_in_threadgroup"),
    "tg_size": ("uint", "tg_size", "threads_per_threadgroup"),
}

# SIMD-group reduction intrinsics (Metal built-ins).
SIMD_REDUCTIONS = {
    "sum": "simd_sum",
    "prod": "simd_product",
    "max": "simd_max",
    "min": "simd_min",
    "and": "simd_and",
    "or": "simd_or",
    "xor": "simd_xor",
    # NaN-PROPAGATING max/min (inductor triton_helpers.maximum/minimum). The simd
    # intrinsic itself is NaN-quiet; threadgroup_reduce adds an any-NaN side-channel
    # so the result is NaN when any element is NaN.
    "nanmax": "simd_max",
    "nanmin": "simd_min",
}

# Metal memory barrier functions.
BARRIERS = {
    "threadgroup": "threadgroup_barrier(mem_flags::mem_threadgroup)",
    "device": "threadgroup_barrier(mem_flags::mem_device)",
    "all": "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)",
}

# Metal address space qualifiers.
ADDRESS_SPACES = {
    "global": "device",
    "shared": "threadgroup",
    "local": "thread",
    "constant": "constant",
}

# ---------------------------------------------------------------------------
# FP8 software emulation — device functions for uchar ↔ float conversion
# ---------------------------------------------------------------------------
# Metal GPUs have no FP8 hardware. FP8 values are stored as uchar (uint8_t)
# and converted to/from float for computation. These MSL device functions
# are injected into the kernel source when FP8 types are detected.

# FP8 E4M3 format (fp8e4nv): 1 sign + 4 exponent + 3 mantissa, bias 7
# Range: ±448, smallest subnormal: 2^-9, NaN: exp=15 mant=7
FP8_E4M3_TO_FLOAT = """\
inline float fp8e4m3_to_float(uchar x) {
    uint sign = (x >> 7) & 1u;
    uint exp = (x >> 3) & 0xFu;
    uint mant = x & 0x7u;
    if (exp == 0u) {
        float val = float(mant) / 8.0f * exp2(-6.0f);
        return sign ? -val : val;
    }
    if (exp == 15u && mant == 7u) return NAN;
    float val = (1.0f + float(mant) / 8.0f) * exp2(float(exp) - 7.0f);
    return sign ? -val : val;
}"""

FP8_FLOAT_TO_E4M3 = """\
inline uchar float_to_fp8e4m3(float x) {
    if (isnan(x)) return 0x7Fu;
    uint sign = 0u;
    if (x < 0.0f) { sign = 1u; x = -x; }
    if (x > 448.0f) x = 448.0f;
    if (x < exp2(-9.0f)) return uchar(sign << 7);
    // Subnormal range: x < 2^-6
    if (x < exp2(-6.0f)) {
        uint mant = uint(x * exp2(6.0f) * 8.0f + 0.5f);
        if (mant > 7u) mant = 7u;
        return uchar((sign << 7) | mant);
    }
    float e_f = floor(log2(x));
    int e = int(e_f);
    float frac = x / exp2(float(e)) - 1.0f;
    uint mant = uint(frac * 8.0f + 0.5f);
    if (mant > 7u) { mant = 0u; e += 1; }
    // Clamp: biased exponent must be in [1, 14] (15 with mant=7 is NaN)
    int biased = e + 7;
    if (biased > 15) { biased = 15; mant = 6u; }  // clamp to max normal (not NaN)
    if (biased < 1) { biased = 0; }  // underflow to subnormal/zero
    // exp=15 mant=7 is NaN, cap at mant=6
    if (biased == 15 && mant >= 7u) mant = 6u;
    return uchar((sign << 7) | (uint(biased) << 3) | mant);
}"""

# FP8 E5M2 format (fp8e5): 1 sign + 5 exponent + 2 mantissa, bias 15
# Range: ±57344, smallest subnormal: 2^-16, has infinity and NaN
FP8_E5M2_TO_FLOAT = """\
inline float fp8e5m2_to_float(uchar x) {
    uint sign = (x >> 7) & 1u;
    uint exp = (x >> 2) & 0x1Fu;
    uint mant = x & 0x3u;
    if (exp == 0u) {
        float val = float(mant) / 4.0f * exp2(-14.0f);
        return sign ? -val : val;
    }
    if (exp == 31u) {
        if (mant == 0u) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    float val = (1.0f + float(mant) / 4.0f) * exp2(float(exp) - 15.0f);
    return sign ? -val : val;
}"""

FP8_FLOAT_TO_E5M2 = """\
inline uchar float_to_fp8e5m2(float x) {
    if (isnan(x)) return 0x7Fu;
    uint sign = 0u;
    if (x < 0.0f) { sign = 1u; x = -x; }
    // Triton\'s reference downcast clamps inf and overflowing finite
    // values to ``finfo(fp8e5).max`` (= 57344.0); preserving inf as
    // the e5m2 exponent-all-ones encoding (0x7C) would fail
    // ``test_typeconvert_downcast_clamping``.
    if (isinf(x) || x > 57344.0f) return uchar((sign << 7) | 0x7Bu);
    // Subnormal range: x < 2^-14. The subnormal mantissa formula
    // ``round(x * 2^16)`` already collapses values below ~2^-17 to zero
    // via RTNE; the previous early ``x < 2^-16 → 0`` cutoff was too
    // aggressive and missed the (2^-17, 2^-16) bucket that should round
    // to the smallest subnormal.
    if (x < exp2(-14.0f)) {
        uint mant = uint(x * exp2(14.0f) * 4.0f + 0.5f);
        if (mant > 3u) mant = 3u;
        return uchar((sign << 7) | mant);
    }
    float e_f = floor(log2(x));
    int e = int(e_f);
    float frac = x / exp2(float(e)) - 1.0f;
    uint mant = uint(frac * 4.0f + 0.5f);
    if (mant > 3u) { mant = 0u; e += 1; }
    int biased = e + 15;
    // exp=31 is inf/NaN, cap at exp=30 mant=3 (max finite)
    if (biased >= 31) { biased = 30; mant = 3u; }
    if (biased < 1) biased = 0;
    return uchar((sign << 7) | (uint(biased) << 2) | mant);
}"""

# FP8 E4M3 with bias 15 (fp8e4b15): same mantissa as e4m3 but bias=15 like fp16
FP8_E4M3B15_TO_FLOAT = """\
inline float fp8e4m3b15_to_float(uchar x) {
    uint sign = (x >> 7) & 1u;
    uint exp = (x >> 3) & 0xFu;
    uint mant = x & 0x7u;
    if (exp == 0u) {
        float val = float(mant) / 8.0f * exp2(-14.0f);
        return sign ? -val : val;
    }
    if (exp == 15u && mant == 7u) return NAN;
    float val = (1.0f + float(mant) / 8.0f) * exp2(float(exp) - 15.0f);
    return sign ? -val : val;
}"""

FP8_FLOAT_TO_E4M3B15 = """\
inline uchar float_to_fp8e4m3b15(float x) {
    if (isnan(x)) return 0x7Fu;
    uint sign = 0u;
    if (x < 0.0f) { sign = 1u; x = -x; }
    float max_val = (1.0f + 6.0f/8.0f) * exp2(-8.0f);
    if (x > max_val) x = max_val;
    if (x < exp2(-17.0f)) return uchar(sign << 7);
    if (x < exp2(-14.0f)) {
        uint mant = uint(x * exp2(14.0f) * 8.0f + 0.5f);
        if (mant > 7u) mant = 7u;
        return uchar((sign << 7) | mant);
    }
    float e_f = floor(log2(x));
    int e = int(e_f);
    float frac = x / exp2(float(e)) - 1.0f;
    uint mant = uint(frac * 8.0f + 0.5f);
    if (mant > 7u) { mant = 0u; e += 1; }
    int biased = e + 15;
    if (biased > 15) { biased = 15; mant = 6u; }
    if (biased < 1) biased = 0;
    if (biased == 15 && mant >= 7u) mant = 6u;
    return uchar((sign << 7) | (uint(biased) << 3) | mant);
}"""

# FP8 E4M3 with bias 8 (fp8e4b8)
FP8_E4M3B8_TO_FLOAT = """\
inline float fp8e4m3b8_to_float(uchar x) {
    uint sign = (x >> 7) & 1u;
    uint exp = (x >> 3) & 0xFu;
    uint mant = x & 0x7u;
    if (exp == 0u) {
        float val = float(mant) / 8.0f * exp2(-7.0f);
        return sign ? -val : val;
    }
    if (exp == 15u && mant == 7u) return NAN;
    float val = (1.0f + float(mant) / 8.0f) * exp2(float(exp) - 8.0f);
    return sign ? -val : val;
}"""

FP8_FLOAT_TO_E4M3B8 = """\
inline uchar float_to_fp8e4m3b8(float x) {
    if (isnan(x)) return 0x7Fu;
    uint sign = 0u;
    if (x < 0.0f) { sign = 1u; x = -x; }
    float max_val = (1.0f + 6.0f/8.0f) * exp2(6.0f);
    if (x > max_val) x = max_val;
    if (x < exp2(-10.0f)) return uchar(sign << 7);
    if (x < exp2(-7.0f)) {
        uint mant = uint(x * exp2(7.0f) * 8.0f + 0.5f);
        if (mant > 7u) mant = 7u;
        return uchar((sign << 7) | mant);
    }
    float e_f = floor(log2(x));
    int e = int(e_f);
    float frac = x / exp2(float(e)) - 1.0f;
    uint mant = uint(frac * 8.0f + 0.5f);
    if (mant > 7u) { mant = 0u; e += 1; }
    int biased = e + 8;
    if (biased > 15) { biased = 15; mant = 6u; }
    if (biased < 1) biased = 0;
    if (biased == 15 && mant >= 7u) mant = 6u;
    return uchar((sign << 7) | (uint(biased) << 3) | mant);
}"""

# FP8 E5M2 with bias 16 (fp8e5b16)
FP8_E5M2B16_TO_FLOAT = """\
inline float fp8e5m2b16_to_float(uchar x) {
    uint sign = (x >> 7) & 1u;
    uint exp = (x >> 2) & 0x1Fu;
    uint mant = x & 0x3u;
    if (exp == 0u) {
        float val = float(mant) / 4.0f * exp2(-15.0f);
        return sign ? -val : val;
    }
    if (exp == 31u) {
        if (mant == 0u) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    float val = (1.0f + float(mant) / 4.0f) * exp2(float(exp) - 16.0f);
    return sign ? -val : val;
}"""

FP8_FLOAT_TO_E5M2B16 = """\
inline uchar float_to_fp8e5m2b16(float x) {
    if (isnan(x)) return 0x7Fu;
    uint sign = 0u;
    if (x < 0.0f) { sign = 1u; x = -x; }
    if (isinf(x)) return uchar((sign << 7) | 0x7Cu);
    float max_val = (1.0f + 3.0f/4.0f) * exp2(14.0f);
    if (x > max_val) return uchar((sign << 7) | 0x7Bu);
    if (x < exp2(-17.0f)) return uchar(sign << 7);
    if (x < exp2(-15.0f)) {
        uint mant = uint(x * exp2(15.0f) * 4.0f + 0.5f);
        if (mant > 3u) mant = 3u;
        return uchar((sign << 7) | mant);
    }
    float e_f = floor(log2(x));
    int e = int(e_f);
    float frac = x / exp2(float(e)) - 1.0f;
    uint mant = uint(frac * 4.0f + 0.5f);
    if (mant > 3u) { mant = 0u; e += 1; }
    int biased = e + 16;
    if (biased >= 31) { biased = 30; mant = 3u; }
    if (biased < 1) biased = 0;
    return uchar((sign << 7) | (uint(biased) << 2) | mant);
}"""

# Mapping from Triton dtype string to (to_float_func, from_float_func, to_float_src, from_float_src)
FP8_CONVERSION_MAP = {
    "fp8e4nv":  ("fp8e4m3_to_float",   "float_to_fp8e4m3",   FP8_E4M3_TO_FLOAT,   FP8_FLOAT_TO_E4M3),
    "fp8_e4m3": ("fp8e4m3_to_float",   "float_to_fp8e4m3",   FP8_E4M3_TO_FLOAT,   FP8_FLOAT_TO_E4M3),
    "fp8e5":    ("fp8e5m2_to_float",   "float_to_fp8e5m2",   FP8_E5M2_TO_FLOAT,   FP8_FLOAT_TO_E5M2),
    "fp8_e5m2": ("fp8e5m2_to_float",   "float_to_fp8e5m2",   FP8_E5M2_TO_FLOAT,   FP8_FLOAT_TO_E5M2),
    "fp8e4b15": ("fp8e4m3b15_to_float","float_to_fp8e4m3b15", FP8_E4M3B15_TO_FLOAT, FP8_FLOAT_TO_E4M3B15),
    "fp8e4b8":  ("fp8e4m3b8_to_float", "float_to_fp8e4m3b8",  FP8_E4M3B8_TO_FLOAT,  FP8_FLOAT_TO_E4M3B8),
    "fp8e5b16": ("fp8e5m2b16_to_float","float_to_fp8e5m2b16", FP8_E5M2B16_TO_FLOAT, FP8_FLOAT_TO_E5M2B16),
}

# All FP8 dtype strings (Triton convention)
FP8_TYPES = frozenset(FP8_CONVERSION_MAP.keys())


def is_fp8_type(dtype: str) -> bool:
    """Check if a Triton dtype string is an FP8 type."""
    return dtype in FP8_TYPES


def fp8_to_float_func(dtype: str) -> str:
    """Return the MSL function name for FP8→float conversion."""
    info = FP8_CONVERSION_MAP.get(dtype)
    return info[0] if info else "fp8e4m3_to_float"


def fp8_from_float_func(dtype: str) -> str:
    """Return the MSL function name for float→FP8 conversion."""
    info = FP8_CONVERSION_MAP.get(dtype)
    return info[1] if info else "float_to_fp8e4m3"


def fp8_device_functions(dtype: str) -> list:
    """Return MSL device function source strings needed for the given FP8 dtype.

    Returns both to_float and from_float functions as a list of strings.
    """
    info = FP8_CONVERSION_MAP.get(dtype)
    if not info:
        return []
    return [info[2], info[3]]


def all_fp8_device_functions() -> list:
    """Return all FP8 device function sources (deduplicated).

    Used when multiple FP8 types are present in the same kernel.
    """
    seen = set()
    funcs = []
    for dtype, (to_name, from_name, to_src, from_src) in FP8_CONVERSION_MAP.items():
        if to_name not in seen:
            seen.add(to_name)
            funcs.append(to_src)
        if from_name not in seen:
            seen.add(from_name)
            funcs.append(from_src)
    return funcs
