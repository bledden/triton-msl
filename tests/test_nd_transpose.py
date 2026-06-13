"""Generic N-D transpose (tt.trans rank>=3) via the closed-form direct-copy
template. Mirrors upstream test_trans_4d. Serial GPU."""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _trans4d(In, Out,
                 s1: tl.constexpr, s2: tl.constexpr,
                 s3: tl.constexpr, s4: tl.constexpr,
                 o1: tl.constexpr, o2: tl.constexpr,
                 o3: tl.constexpr, o4: tl.constexpr,
                 t1: tl.constexpr, t2: tl.constexpr,
                 t3: tl.constexpr, t4: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            base=In, shape=[s1, s2, s3, s4],
            strides=[s4 * s3 * s2, s4 * s3, s4, 1],
            block_shape=[s1, s2, s3, s4])
        out_desc = tl.make_tensor_descriptor(
            base=Out, shape=[o1 * o2 * o3 * o4], strides=[1],
            block_shape=[o1 * o2 * o3 * o4])
        val = in_desc.load([0, 0, 0, 0]).permute((t1, t2, t3, t4))
        out_desc.store([0], val.reshape(out_desc.block_shape))

    def _alloc(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cpu")


@requires_metal
@pytest.mark.parametrize("shape,perm", [
    ((4, 4, 4, 16), (3, 1, 0, 2)),     # 1024, non-trivial perm
    ((4, 4, 4, 16), (0, 2, 1, 3)),     # 1024, mid-axis swap
    ((2, 2, 8, 64), (1, 0, 3, 2)),     # 2048 (>1024) — exercises the strided loop
    ((2, 2, 8, 64), (3, 2, 1, 0)),     # 2048, full reverse
])
@pytest.mark.parametrize("dt", [torch.int32, torch.int8])
def test_nd_transpose(shape, perm, dt):
    triton.set_allocator(_alloc)
    total = shape[0] * shape[1] * shape[2] * shape[3]
    hi = 127 if dt == torch.int8 else 100000
    In = torch.randint(-hi, hi, shape, dtype=dt)
    Out = torch.zeros(total, dtype=dt)
    out_shape = [shape[i] for i in perm]
    _trans4d[(1,)](In, Out, *shape, *out_shape, *perm, num_warps=8)
    want = In.permute(perm).reshape(-1)
    assert torch.equal(Out, want), (
        f"shape={shape} perm={perm} dt={dt}: mismatch")
