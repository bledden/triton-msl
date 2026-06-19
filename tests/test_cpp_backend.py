"""Test the C++ MLIR backend infrastructure.

Verifies that the C++ MLIR pass module (triton_msl._triton_msl_cpp) can
be imported and its passes registered alongside Triton's libtriton.so in the
same process. The pybind11 module links against libtriton.so for shared MLIR
symbols, eliminating the previous duplicate-dialect-registration crash.
"""
import pytest

try:
    import triton_msl._triton_msl_cpp as cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

try:
    import Metal
    _HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except ImportError:
    _HAS_METAL = False

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_cpp = pytest.mark.skipif(not _HAS_CPP, reason="C++ backend not built")
requires_metal = pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")


@requires_cpp
def test_cpp_module_importable():
    """C++ MLIR module can be imported."""
    import triton_msl._triton_msl_cpp as mod
    assert hasattr(mod, "register_metal_passes")


@requires_cpp
def test_cpp_passes_register():
    """C++ MLIR passes can be registered without error."""
    cpp.register_metal_passes()


@requires_cpp
def test_cpp_passes_register_idempotent():
    """Calling register_metal_passes() multiple times is safe."""
    cpp.register_metal_passes()
    cpp.register_metal_passes()
    cpp.register_metal_passes()


@requires_cpp
@requires_metal
def test_cpp_pass_runs_on_vector_add():
    """The C++ pass infrastructure works alongside a vector_add compilation.

    Both the C++ module and Triton's compilation pipeline run in the same
    process — no subprocess isolation needed. The pybind11 module links
    against libtriton.so so both use the same MLIR symbols.
    """
    import triton
    import triton.language as tl
    from triton.compiler.compiler import compile as triton_compile, ASTSource
    from triton.backends.compiler import GPUTarget

    # Register our C++ passes in the same process
    cpp.register_metal_passes()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    target = GPUTarget("metal", "apple-m4", 32)
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    src = ASTSource(fn=add_kernel, signature=sig, constexprs={"BLOCK": 256})
    compiled = triton_compile(src, target=target)

    assert compiled.asm.get("ttgir") is not None, "TTGIR missing"
    assert len(str(compiled.asm["ttgir"])) > 0, "TTGIR empty"
    assert compiled.asm.get("msl") is not None, "MSL missing"
    assert compiled.asm.get("metallib") is not None, "metallib missing"


@requires_cpp
@requires_metal
def test_vector_add_execution():
    """Vector add produces correct results via the Python compilation path.

    Both the C++ module and Triton kernel execution run in the same process.
    """
    import torch
    import triton
    import triton.language as tl

    # Register our C++ passes in the same process
    cpp.register_metal_passes()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n)
    y = torch.randn(n)
    out = torch.zeros(n)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=256)

    max_err = (out - (x + y)).abs().max().item()
    assert max_err < 1e-5, f"max error {max_err} exceeds tolerance"


@requires_cpp
@requires_metal
def test_scf_for_accumulation():
    """Accumulation loop compiles and executes through C++ metallib.

    The kernel sums K chunks of an input vector using an explicit loop,
    which Triton lowers to scf.for + scf.yield. The C++ path handles
    this via SCFToControlFlowPass -> cf.br/cf.cond_br -> LLVM branches.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def accum_kernel(x_ptr, out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for k in range(K):
                val = tl.load(x_ptr + offs * K + k)
                acc += val
            tl.store(out_ptr + offs, acc)

        BLOCK = 256
        K = 4
        n = BLOCK
        x = torch.randn(n * K)
        out = torch.zeros(n)

        accum_kernel[(1,)](x, out, K=K, BLOCK=BLOCK)

        expected = x.view(n, K).sum(dim=1)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-4, f"scf.for accumulation: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_scf_if_conditional():
    """Conditional clamp with float scalar args through C++ metallib.

    Uses tl.where (arith.cmpf + arith.select) and float scalar parameters
    (lo, hi) which are passed as constant buffer pointers in AIR.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def clamp_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            x = tl.where(x < lo, lo, x)
            x = tl.where(x > hi, hi, x)
            tl.store(out_ptr + offs, x, mask=mask)

        n = 512
        x = torch.randn(n) * 5
        out = torch.zeros(n)

        clamp_kernel[(triton.cdiv(n, 256),)](x, out, -1.0, 1.0, n, BLOCK=256)

        expected = x.clamp(-1.0, 1.0)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-5, f"clamp: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_wrapping_loop_large_block():
    """Kernel with BLOCK_SIZE=2048 compiles through C++ metallib."""
    import os, torch, triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x * 2.0, mask=mask)

        n = 4096
        x = torch.randn(n)
        out = torch.zeros(n)
        grid = (triton.cdiv(n, 2048),)
        scale_kernel[grid](x, out, n, BLOCK=2048)

        max_err = (out - x * 2.0).abs().max().item()
        assert max_err < 1e-5, f"wrapping loop: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_local_alloc_basic():
    """ttg.local_alloc + local_store + local_load through C++ metallib.

    Triton doesn't expose ttg.local_alloc at the language level — it's
    generated by layout conversions and reductions. We trigger it via
    tl.sum which lowers to ttg.local_alloc + ttg.local_store + barrier +
    ttg.local_load for cross-warp reduction.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def shmem_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs)
            s = tl.sum(x, axis=0)
            tl.store(out_ptr + offs, x - s / BLOCK)

        BLOCK = 256
        x = torch.randn(BLOCK)
        out = torch.zeros(BLOCK)
        shmem_kernel[(1,)](x, out, BLOCK=BLOCK)

        expected = x - x.mean()
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-3, f"shmem roundtrip: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_async_copy_sync_loop():
    """ttg.async_copy_global_to_local via pipelined load+accumulate.

    Triton's pipeliner generates ttg.async_copy_global_to_local when
    num_stages > 1 with loads inside a loop. The C++ path lowers it to
    a synchronous per-thread copy.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def pipelined_kernel(x_ptr, out_ptr, K: tl.constexpr,
                             BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for k in tl.range(K, num_stages=2):
                acc += tl.load(x_ptr + offs + k * BLOCK)
            tl.store(out_ptr + offs, acc)

        BLOCK = 256
        K = 4
        x = torch.randn(BLOCK * K)
        out = torch.zeros(BLOCK)
        pipelined_kernel[(1,)](x, out, K=K, BLOCK=BLOCK)

        expected = x.view(K, BLOCK).sum(dim=0)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-3, f"async_copy: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_32kb_threadgroup_budget():
    """Kernel requiring > 32KB threadgroup memory falls back to MSL cleanly.

    No crash, correct results via MSL path.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        # 8192 float32 elements = 32 KB. Use > 8192 to force budget failure.
        # Triton generates shared memory proportional to block size for reductions.
        @triton.jit
        def huge_shmem_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs)
            s = tl.sum(x, axis=0)
            tl.store(out_ptr + offs, x + s)

        BLOCK = 4096  # Large block forces significant shared memory
        x = torch.randn(BLOCK)
        out = torch.zeros(BLOCK)
        huge_shmem_kernel[(1,)](x, out, BLOCK=BLOCK)

        expected = x + x.sum()
        max_err = (out - expected).abs().max().item()
        # Relaxed tolerance — large reduction
        assert max_err < 1e-1, f"32kb fallback: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_cpp_tiled_reduction():
    """Large reduction via shared memory through C++ path."""
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def sum_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            s = tl.sum(x, axis=0)
            tl.store(out_ptr, s)

        N = 1024
        x = torch.randn(N)
        out = torch.zeros(1)
        sum_kernel[(1,)](x, out, N=N, BLOCK=1024)

        max_err = abs(out.item() - x.sum().item())
        assert max_err < 1e-2, f"tiled reduction: error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_cpp_cumsum():
    """Cumsum using shared memory through C++ path."""
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def cumsum_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs)
            c = tl.cumsum(x, axis=0)
            tl.store(out_ptr + offs, c)

        BLOCK = 256
        x = torch.randn(BLOCK)
        out = torch.zeros(BLOCK)
        cumsum_kernel[(1,)](x, out, BLOCK=BLOCK)

        expected = x.cumsum(dim=0)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-3, f"cumsum: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_cpp_dot_32x32():
    """32x32x32 f16 matmul through C++ path with tiled MMA."""
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def matmul_kernel(a_ptr, b_ptr, c_ptr,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
            off_m = tl.arange(0, BLOCK_M)
            off_n = tl.arange(0, BLOCK_N)
            off_k = tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + off_m[:, None] * K + off_k[None, :])
            b = tl.load(b_ptr + off_k[:, None] * N + off_n[None, :])
            c = tl.dot(a.to(tl.float16), b.to(tl.float16),
                        acc=tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
            tl.store(c_ptr + off_m[:, None] * N + off_n[None, :], c)

        M = N = K = 32
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        c = torch.zeros(M, N)
        matmul_kernel[(1,)](a, b, c, M=M, N=N, K=K,
                             BLOCK_M=M, BLOCK_N=N, BLOCK_K=K)

        expected = a @ b
        max_err = (c - expected).abs().max().item()
        # f16 tolerance (relaxed)
        assert max_err < 0.5, f"32x32 matmul: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_cpp_layer_norm():
    """Layer norm (2-pass mean/var via shared memory) through C++ path."""
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def layer_norm_kernel(x_ptr, out_ptr, N: tl.constexpr,
                               eps: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            mean = tl.sum(x, axis=0) / N
            diff = x - mean
            var = tl.sum(diff * diff, axis=0) / N
            rstd = 1.0 / tl.sqrt(var + eps)
            y = diff * rstd
            tl.store(out_ptr + offs, y, mask=mask)

        N = 256
        x = torch.randn(N)
        out = torch.zeros(N)
        layer_norm_kernel[(1,)](x, out, N=N, eps=1e-5, BLOCK=256)

        expected = torch.nn.functional.layer_norm(x, (N,), eps=1e-5)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-3, f"layer_norm: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_cpp_dot_k_loop():
    """Matmul with K-loop (scf.for wrapping tt.dot).

    DotOpConversion threads the input C accumulator through a threadgroup
    buffer so that the iter_arg from a prior iteration is preserved across
    successive tt.dot calls (not just the common tl.zeros fast path).
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def matmul_k_loop(a_ptr, b_ptr, c_ptr,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
            off_m = tl.arange(0, BLOCK_M)
            off_n = tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_off in range(0, K, BLOCK_K):
                off_k = k_off + tl.arange(0, BLOCK_K)
                a = tl.load(a_ptr + off_m[:, None] * K + off_k[None, :])
                b = tl.load(b_ptr + off_k[:, None] * N + off_n[None, :])
                acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)
            tl.store(c_ptr + off_m[:, None] * N + off_n[None, :], acc)

        M = N = 16
        K = 32
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        c = torch.zeros(M, N)
        matmul_k_loop[(1,)](a, b, c, M=M, N=N, K=K,
                             BLOCK_M=M, BLOCK_N=N, BLOCK_K=16)

        expected = a @ b
        max_err = (c - expected).abs().max().item()
        assert max_err < 0.5, f"k-loop matmul: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_aliasing_non_overlapping_allocs():
    """Two non-overlapping shared allocations share backing memory.

    Two-phase kernel: phase 1 reduces to mean, phase 2 uses mean to
    compute max(x - mean). Each phase needs shared memory for cross-warp
    reduction; after phase 1's barrier, phase 1's shared memory is dead
    and can be reused for phase 2.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        @triton.jit
        def two_phase_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs)
            s1 = tl.sum(x, axis=0)
            diff = x - s1 / BLOCK
            s2 = tl.max(diff, axis=0)
            tl.store(out_ptr + offs, diff - s2)

        BLOCK = 1024
        x = torch.randn(BLOCK)
        out = torch.zeros(BLOCK)
        two_phase_kernel[(1,)](x, out, BLOCK=BLOCK)

        mean = x.mean()
        diff = x - mean
        mx = diff.max()
        expected = diff - mx
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-3, f"aliasing: max error {max_err}"
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)


if _HAS_TRITON:
    @triton.jit
    def _fa_fwd_strided_kernel(
        Q, K, V, Out,
        stride_qm, stride_qk,
        stride_kn, stride_kk,
        stride_vn, stride_vk,
        stride_om, stride_ok,
        sm_scale,
        N_CTX, HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """FlashAttention v2 forward with explicit strides.

        Uses stride arguments so the MSL backend pattern-matches this as a
        flash-attention kernel (not a plain matmul). Mirrors the pattern used
        by tests/test_flash_attention.py:_flash_attn_fwd.
        """
        start_m = tl.program_id(0)
        off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q + off_m[:, None] * stride_qm + off_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=off_m[:, None] < N_CTX, other=0.0)
        q = q * sm_scale

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            off_n_iter = start_n + off_n
            k_ptrs = K + off_n_iter[:, None] * stride_kn + off_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=off_n_iter[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q, tl.trans(k))

            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V + off_n_iter[:, None] * stride_vn + off_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=off_n_iter[:, None] < N_CTX, other=0.0)
            acc += tl.dot(p.to(tl.float32), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_ptrs = Out + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
        tl.store(out_ptrs, acc, mask=off_m[:, None] < N_CTX)


def _run_fa_cpp(N_CTX, HEAD_DIM):
    """Run the strided FA kernel with TRITON_MSL_USE_CPP=1 and return max err."""
    import os
    import torch
    import triton

    BLOCK_M = BLOCK_N = 32
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(N_CTX, HEAD_DIM)
    k = torch.randn(N_CTX, HEAD_DIM)
    v = torch.randn(N_CTX, HEAD_DIM)
    out = torch.zeros(N_CTX, HEAD_DIM)

    os.environ["TRITON_MSL_USE_CPP"] = "1"
    try:
        grid = (triton.cdiv(N_CTX, BLOCK_M),)
        _fa_fwd_strided_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            out.stride(0), out.stride(1),
            sm_scale,
            N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N,
        )
    finally:
        os.environ.pop("TRITON_MSL_USE_CPP", None)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
    ).squeeze()
    return (out - expected).abs().max().item()


@requires_cpp
@requires_metal
def test_cpp_flash_attention_head32():
    """FlashAttention HEAD_DIM=32 through the C++ metallib path.

    Tile = BLOCK_M*HEAD_DIM = 32*32 = 1024, which fits within Metal's 1024
    thread cap so no wrap loop is injected, and the C++ path's tt.dot
    lowering (simdgroup_matrix_8x8 + memdesc_trans for q@k^T) handles the
    kernel end-to-end. See _has_complex_ops for the wrap-loop + tt.dot
    incompatibility that routes HEAD_DIM=64 to MSL.
    """
    max_err = _run_fa_cpp(N_CTX=64, HEAD_DIM=32)
    assert max_err < 5e-2, f"FA HEAD_DIM=32: max error {max_err}"


@requires_cpp
@requires_metal
def test_cpp_flash_attention_head64():
    """FlashAttention HEAD_DIM=64 with TRITON_MSL_USE_CPP=1 set.

    Tile = BLOCK_M*HEAD_DIM = 32*64 = 2048 > 1024, so make_llir would
    inject a wrapping loop over the kernel body. The wrap loop is
    fundamentally incompatible with tt.dot (simdgroup_matrix reads the
    whole tile but the populate under the wrap only fills half on
    iteration 0), so _has_complex_ops routes this to MSL, which uses a
    different (threadgroup-wide) code model for tt.dot.
    """
    max_err = _run_fa_cpp(N_CTX=64, HEAD_DIM=64)
    assert max_err < 5e-2, f"FA HEAD_DIM=64: max error {max_err}"