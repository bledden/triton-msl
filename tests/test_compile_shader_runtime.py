"""CompileShaderRuntime: compile MSL via torch.mps.compile_shader + dispatch
zero-copy against MPS tensors. Serial GPU."""
import pytest
try:
    import torch
    HAS = hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
except Exception:
    HAS = False
requires_cs = pytest.mark.skipif(not HAS, reason="torch.mps.compile_shader needed")

_VADD = '''#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                 device float* OUT [[buffer(2)]], constant int& N [[buffer(3)]],
                 uint pid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
    uint i = pid*256u + lid; if (i < (uint)N) OUT[i] = A[i] + B[i];
}'''

@requires_cs
def test_available():
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    assert CompileShaderRuntime().available() is True

@requires_cs
def test_dispatch_vadd_zero_copy():
    import torch
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    rt = CompileShaderRuntime()
    N = 4096
    A = torch.randn(N, device="mps"); B = torch.randn(N, device="mps"); OUT = torch.empty(N, device="mps")
    lib = rt.get_library(_VADD)
    assert rt.get_library(_VADD) is lib   # cached (same object)
    rt.dispatch(lib, "vadd", [A, B, OUT, N], threads=N, group_size=256)
    torch.mps.synchronize()
    torch.testing.assert_close(OUT, A + B, rtol=1e-4, atol=1e-4)

_BLOCKSUM = '''#include <metal_stdlib>
using namespace metal;
kernel void blocksum(device const float* X [[buffer(0)]], device float* OUT [[buffer(1)]],
                     constant int& N [[buffer(2)]],
                     uint pid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                     uint sgitg [[simdgroup_index_in_threadgroup]], uint tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float shared[8];
    uint i = pid*256u + lid;
    float v = (i < (uint)N) ? X[i] : 0.0f;
    float s = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shared[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = (tiisg < 8u) ? shared[tiisg] : 0.0f;
    total = simd_sum(total);
    if (lid == 0) OUT[pid] = total;
}'''

@requires_cs
def test_dispatch_reduction_shared_mem():
    import torch
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    rt = CompileShaderRuntime()
    N = 256 * 64
    X = torch.randn(N, device="mps")
    OUT = torch.empty(N // 256, device="mps")
    lib = rt.get_library(_BLOCKSUM)
    rt.dispatch(lib, "blocksum", [X, OUT, N], threads=N, group_size=256)
    torch.mps.synchronize()
    expected = X.view(N // 256, 256).sum(1)
    torch.testing.assert_close(OUT, expected, rtol=1e-3, atol=1e-3)

_K2D = '''#include <metal_stdlib>
using namespace metal;
kernel void k2(device float* o [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    o[gid.y*4+gid.x] = float(gid.x+gid.y);
}'''

@requires_cs
def test_dispatch_2d_grid():
    import torch
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    rt = CompileShaderRuntime()
    O = torch.empty(16, device="mps")
    lib = rt.get_library(_K2D)
    rt.dispatch(lib, "k2", [O], threads=(4, 4), group_size=(4, 4))
    torch.mps.synchronize()
    expected = torch.tensor([0,1,2,3, 1,2,3,4, 2,3,4,5, 3,4,5,6], dtype=torch.float32, device="mps")
    torch.testing.assert_close(O, expected, rtol=0, atol=0)
