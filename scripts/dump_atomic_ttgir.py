"""Dump TTIR and TTGIR for atomic Triton kernels to understand the IR structure."""
import triton
import triton.language as tl
import triton._C.libtriton as _triton
ir = _triton.ir


# Scalar atomic add (reduce then atomic)
@triton.jit
def atomic_add_scalar_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    total = tl.sum(x, axis=0)
    tl.atomic_add(Y, total)


# Scalar atomic max
@triton.jit
def atomic_max_scalar_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=float('-inf'))
    local_max = tl.max(x, axis=0)
    tl.atomic_max(Y, local_max)


# Atomic CAS (scalar)
@triton.jit
def atomic_cas_kernel(Ptr, Cmp, Val):
    cmp = tl.load(Cmp)
    val = tl.load(Val)
    tl.atomic_cas(Ptr, cmp, val)


# Vector atomic add (each element to different address)
@triton.jit
def atomic_add_vector_kernel(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    tl.atomic_add(Y + offsets, x)


# Masked vector atomic add
@triton.jit
def masked_atomic_add_kernel(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    tl.atomic_add(Y + offsets, x, mask=mask)


# Scalar atomic min (i32)
@triton.jit
def atomic_min_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=2147483647)
    local_min = tl.min(x, axis=0)
    tl.atomic_min(Y, local_min)


# Vector atomic or (bitwise)
@triton.jit
def atomic_or_kernel(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    tl.atomic_or(Y + offsets, x)


# Vector atomic xchg (returns old value)
@triton.jit
def atomic_xchg_kernel(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    old = tl.atomic_xchg(Y + offsets, x)
    tl.store(X + offsets, old)


# Atomic and (bitwise)
@triton.jit
def atomic_and_kernel(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    tl.atomic_and(Y + offsets, x)


# Atomic xor (bitwise)
@triton.jit
def atomic_xor_kernel(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    tl.atomic_xor(Y + offsets, x)


def compile_to_ttgir(kernel_fn, sig, constexprs=None):
    """Compile a @triton.jit kernel through TTIR and TTGIR stages."""
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})

    src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs or {})
    context = ir.context()
    ir.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    mod = src.make_ir(target, options, codegen_fns, module_map, context)

    ttir_str = str(mod)

    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    ttir_post_str = str(mod)

    mod = backend.make_ttgir(mod, metadata, options)
    ttgir_str = str(mod)

    return ttir_str, ttir_post_str, ttgir_str


def dump_kernel(name, kernel_fn, sig, constexprs=None):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    try:
        ttir, ttir_post, ttgir = compile_to_ttgir(kernel_fn, sig, constexprs)
        print("\n--- TTIR (raw) ---")
        print(ttir)
        print("\n--- TTGIR ---")
        print(ttgir)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    os.system("rm -rf ~/.triton/cache/")

    dump_kernel(
        "atomic_add_scalar (fp32) - reduce then scalar atomic",
        atomic_add_scalar_kernel,
        {"X": "*fp32", "Y": "*fp32", "N": "i32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_max_scalar (fp32)",
        atomic_max_scalar_kernel,
        {"X": "*fp32", "Y": "*fp32", "N": "i32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_cas (fp32, scalar)",
        atomic_cas_kernel,
        {"Ptr": "*fp32", "Cmp": "*fp32", "Val": "*fp32"},
    )

    dump_kernel(
        "atomic_add_vector (fp32) - element-wise",
        atomic_add_vector_kernel,
        {"X": "*fp32", "Y": "*fp32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "masked_atomic_add (fp32)",
        masked_atomic_add_kernel,
        {"X": "*fp32", "Y": "*fp32", "N": "i32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_min (i32)",
        atomic_min_kernel,
        {"X": "*i32", "Y": "*i32", "N": "i32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_or (i32, vector)",
        atomic_or_kernel,
        {"X": "*i32", "Y": "*i32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_xchg (fp32, returns old)",
        atomic_xchg_kernel,
        {"X": "*fp32", "Y": "*fp32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_and (i32, vector)",
        atomic_and_kernel,
        {"X": "*i32", "Y": "*i32"},
        {"BLOCK": 64},
    )

    dump_kernel(
        "atomic_xor (i32, vector)",
        atomic_xor_kernel,
        {"X": "*i32", "Y": "*i32"},
        {"BLOCK": 64},
    )
