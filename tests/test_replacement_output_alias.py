"""A replacement template must not run when the output overlaps an operand it reads.

One source program loads every operand before it stores, so a single-program launch whose
output shares memory with an operand is defined. The matmul replacement templates spread
that program over several threadgroups that store finished tiles while others still load.
Measured on the device before the gate: a 256x256 in-place matmul was wrong in 4 to 5 of 300
launches, an in-place quantized GEMV in 1 to 11 of 200 depending on the run. The fast matmul
route, split-K included, now keeps the compiled entry for such a launch and the quantized
routes refuse it; views of one storage whose byte ranges do not intersect keep the replacement.
"""

import platform

import pytest
import torch

from triton_msl.autotuning._operand_overlap import output_overlaps_operand

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


# ---------------------------------------------------------------------------- the predicate
def test_same_tensor_overlaps():
    a, b = torch.zeros(8, 8), torch.zeros(8, 8)
    assert output_overlaps_operand([a, b, a, 8, 8, 8], 2, 64, {0: 64, 1: 64}, exact=True)
    assert output_overlaps_operand([a, b, b, 8, 8, 8], 2, 64, {0: 64, 1: 64}, exact=True)
    assert output_overlaps_operand([a, b, a], 2, 64, {})


def test_separate_storages_do_not_overlap():
    a, b, c = torch.zeros(8, 8), torch.zeros(8, 8), torch.zeros(8, 8)
    assert not output_overlaps_operand([a, b, c, 8, 8, 8], 2, 64, {0: 64, 1: 64}, exact=True)
    assert not output_overlaps_operand([a, b, c], 2, 64, {0: 64})


@pytest.mark.parametrize("exact", [True, False])
def test_disjoint_views_of_one_storage_do_not_overlap(exact):
    arena = torch.zeros(3 * 64)
    a, b, c = (arena[i * 64 : (i + 1) * 64].view(8, 8) for i in range(3))
    for order in ((a, b, c), (c, a, b), (b, c, a)):
        assert not output_overlaps_operand([*order, 8, 8, 8], 2, 64, {0: 64, 1: 64}, exact=exact)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("shift", [1, 63])
def test_partly_overlapping_views_overlap(exact, shift):
    arena = torch.zeros(4 * 64)
    a = arena[0:64].view(8, 8)
    b = arena[128:192].view(8, 8)
    before = arena[64 - shift : 128 - shift].view(8, 8)  # begins on A's last elements
    after = arena[192 - shift : 256 - shift].view(8, 8)  # begins on B's last elements
    assert output_overlaps_operand([a, b, before, 8, 8, 8], 2, 64, {0: 64, 1: 64}, exact=exact)
    assert output_overlaps_operand([a, b, after, 8, 8, 8], 2, 64, {0: 64, 1: 64}, exact=exact)


def test_adjacent_views_do_not_overlap():
    arena = torch.zeros(128)
    a, c = arena[:64].view(8, 8), arena[64:].view(8, 8)
    assert not output_overlaps_operand([a, a, c], 2, 64, {0: 64, 1: 64}, exact=True)
    assert not output_overlaps_operand([c, c, a], 2, 64, {0: 64, 1: 64}, exact=True)


def test_exact_counts_are_the_whole_contract():
    """The fast matmul reads M*K elements from A's pointer whatever the view's own size."""
    arena = torch.zeros(256)
    whole = arena.view(16, 16)  # passed as A with M = 4 rows of K = 16: 64 elements are read
    inside = arena[128:192].view(8, 8)  # inside the view, after the elements read
    args = [whole, torch.zeros(16, 8), inside, 4, 8, 16]
    assert not output_overlaps_operand(args, 2, 32, {0: 64, 1: 128}, exact=True)
    assert output_overlaps_operand(args, 2, 32, {0: 64, 1: 128})
    # ... and a pointer view shorter than the elements read from it still counts in full.
    row = arena[:16]
    after_row = arena[32:64].view(4, 8)
    assert output_overlaps_operand([row, torch.zeros(16, 8), after_row], 2, 32, {0: 64, 1: 128}, exact=True)
    assert output_overlaps_operand([row, torch.zeros(16, 8), after_row], 2, 32, {0: 64})
    assert not output_overlaps_operand([row, torch.zeros(16, 8), after_row], 2, 32, {})


def test_strided_view_range_covers_its_last_element():
    arena = torch.zeros(64)
    every_other = arena[::2]  # elements 0, 2, ..., 62
    tail = arena[62:64]
    gap = arena[63:64]
    assert output_overlaps_operand([every_other, tail], 1, 2, {})
    assert not output_overlaps_operand([every_other, gap], 1, 1, {})


def test_element_sizes_are_compared_in_bytes():
    arena = torch.zeros(64, dtype=torch.float32)
    as_bytes = arena.view(torch.int8)  # 256 bytes over the same storage
    out = arena[32:]
    assert output_overlaps_operand([as_bytes[100:200], out], 1, 32, {})
    assert not output_overlaps_operand([as_bytes[0:128], out], 1, 32, {})


def test_arguments_without_storage():
    class Opaque:
        def data_ptr(self):
            return 0

    t = torch.zeros(4)
    assert not output_overlaps_operand([None, None, None, 4, 4, 4], 2, 16, {0: 16, 1: 16}, exact=True)
    assert not output_overlaps_operand([t, 3, 2.5, None, torch.zeros(4)], 4, 4, {})
    # A tensor-like value whose storage cannot be read is not proven separate.
    assert output_overlaps_operand([Opaque(), t], 1, 4, {})
    assert output_overlaps_operand([t, Opaque()], 1, 4, {})


# ---------------------------------------------------------------------------- the dispatchers
class _Runtime:
    def __init__(self):
        self.calls = []
        self.unsupported = []

    def is_unsupported(self, msl):
        return False

    def mark_unsupported(self, msl):
        self.unsupported.append(msl)

    def get_library(self, msl):
        return msl

    def dispatch(self, lib, name, args, *, threads, group_size):
        self.calls.append(name)


def _arena(count, elements, dtype=torch.float32):
    arena = torch.zeros(count * elements, dtype=dtype)
    return [arena[i * elements : (i + 1) * elements] for i in range(count)]


def _fast(args, grid=(2, 2, 1)):
    from triton_msl.autotuning._fast_matmul_dispatch import dispatch_fast_matmul

    rt = _Runtime()
    descriptor = ("MSL", 3, 4, 5, 32, 32, None, None, (), ("2d", 32, 32, True, True))
    return dispatch_fast_matmul(rt, descriptor, args, grid=grid), rt


def test_fast_matmul_declines_an_output_that_is_an_operand():
    a, b, c = (torch.zeros(64, 64) for _ in range(3))
    for args in ([a, b, a, 64, 64, 64], [a, b, b, 64, 64, 64]):
        taken, rt = _fast(args)
        # A miss before submission: nothing ran and the shader is not marked unsupported.
        assert taken is False and rt.calls == [] and rt.unsupported == []
    taken, rt = _fast([a, b, c, 64, 64, 64])
    assert taken is True and rt.calls == ["simdgroup_matmul_fast"]
    taken, rt = _fast([a, a, c, 64, 64, 64])  # operands may share memory with each other
    assert taken is True and rt.calls == ["simdgroup_matmul_fast"]


def test_fast_matmul_keeps_disjoint_views_of_one_storage():
    a, b, c = (v.view(64, 64) for v in _arena(3, 64 * 64))
    taken, rt = _fast([a, b, c, 64, 64, 64])
    assert taken is True and rt.calls == ["simdgroup_matmul_fast"]
    inside = a.reshape(-1)[32 : 32 + 32 * 64].view(32, 64)  # starts inside the rows read from A
    taken, rt = _fast([a, b, inside, 32, 64, 64], grid=(1, 2, 1))
    assert taken is False and rt.calls == []
    past = a.reshape(-1)[32 * 64 :].view(32, 64)  # A's storage, after the 32 rows read
    taken, rt = _fast([a, b, past, 32, 64, 64], grid=(1, 2, 1))
    assert taken is True and rt.calls == ["simdgroup_matmul_fast"]


def _quant(kind, args):
    from triton_msl.autotuning._quant_matmul_dispatch import dispatch_quant_matmul

    rt = _Runtime()
    if kind == "gemm":  # input, weight, output, scale, zero, M, N, K
        return dispatch_quant_matmul(rt, ("MSL", 5, 6, 7, 32, 16, (), (32, 32)), args, grid=(2, 2, 1)), rt
    if kind == "gemv":  # input, weight, output, scale, zero, N, K
        return dispatch_quant_matmul(rt, ("gemv", "MSL", 0, 1, 2, 3, 4, 5, 6, (), 32), args, grid=(2, 1, 1)), rt
    if kind == "gemv_int4":
        descriptor = ("gemv_int4", "MSL", 0, 1, 2, 3, 4, 5, 6, None, None, None, 32, 32)
        return dispatch_quant_matmul(rt, descriptor, args, grid=(2, 1, 1)), rt
    if kind == "sym":  # input, weight, output, scale, M, N, K, then strides
        descriptor = ("sym_int8", "MSL", 4, 5, 6, (7, 8, 9, 10, 11, 12, -1), (32, 32))
        return dispatch_quant_matmul(rt, descriptor, [*args, 64, 1, 64, 1, 64, 1], grid=(2, 2, 1)), rt
    descriptor = ("pergroup_int8", "MSL", 5, 6, 7, tuple(range(8, 18)), None, 4, 4, 32, 32, (32, 32))
    return dispatch_quant_matmul(rt, descriptor, [*args, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1], grid=(2, 2, 1)), rt


def _quant_args(kind, x, out):
    w = torch.zeros(64, 64, dtype=torch.int8)
    if kind in ("gemv", "gemv_int4"):
        return [x, w, out, torch.zeros(64, 2), torch.zeros(64, 2), 64, 64]
    if kind == "sym":
        return [x, w, out, torch.zeros(64), 64, 64, 64]
    return [x, w, out, torch.zeros(2, 64), torch.zeros(2, 64), 64, 64, 64]


@pytest.mark.parametrize("kind", ["gemm", "gemv", "gemv_int4", "sym", "pergroup"])
def test_quant_routes_decline_an_in_place_output(kind):
    elements = 64 if kind.startswith("gemv") else 64 * 64
    shape = (64,) if kind.startswith("gemv") else (64, 64)
    x, out = torch.zeros(shape), torch.zeros(shape)
    taken, rt = _quant(kind, _quant_args(kind, x, out))
    assert taken is True and len(rt.calls) == 1, "the control launch must reach the template"

    taken, rt = _quant(kind, _quant_args(kind, x, x))
    assert taken is False and rt.calls == [] and rt.unsupported == []

    xa, oa = (v.view(shape) for v in _arena(2, elements))
    taken, rt = _quant(kind, _quant_args(kind, xa, oa))
    assert taken is True and len(rt.calls) == 1, "disjoint views of one storage stay eligible"

    # The output may not overlap the scale tensor either (any operand, not only the input).
    args = _quant_args(kind, x, out)
    args[3] = out.reshape(-1)[-1:]
    taken, rt = _quant(kind, args)
    assert taken is False and rt.calls == []


# ---------------------------------------------------------------------------- on the device
@pytest.fixture
def fast_matmul_routes(monkeypatch):
    import triton  # noqa: F401  (backend discovery before the driver's lazy imports)

    import triton_msl.autotuning._fast_matmul_dispatch as fast

    taken = []
    real = fast.dispatch_fast_matmul

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        taken.append(bool(result))
        return result

    monkeypatch.setattr(fast, "dispatch_fast_matmul", spy)
    return taken


def _matmul_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def in_place_mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return in_place_mm


def _operands(n, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-3, 4, (n, n), generator=g).float()
    b = torch.randint(-3, 4, (n, n), generator=g).float()
    return a, b, (a.double() @ b.double()).float()  # small integers: exact in fp32


@requires_mps
@pytest.mark.parametrize("which", ["A", "B"])
def test_in_place_matmul_keeps_the_source_order(fast_matmul_routes, which):
    kernel, n = _matmul_kernel(), 256
    for seed in range(12):
        a, b, want = _operands(n, seed)
        A, Bm = a.to("mps"), b.to("mps")
        C = A if which == "A" else Bm
        fast_matmul_routes.clear()
        kernel[(1, 1)](A, Bm, C, n, n, n, BM=n, BN=n, BK=32)
        torch.mps.synchronize()
        assert fast_matmul_routes == [False], "the replacement must decline an in-place output"
        assert torch.equal(C.cpu(), want)


@requires_mps
@pytest.mark.parametrize("which", ["A", "B", "apart"])
def test_in_place_deep_matmul_declines_the_split_k_replacement_too(fast_matmul_routes, monkeypatch, which):
    # Few output tiles and a deep K: with a separate output this is the split-K launch.
    import triton_msl.autotuning._fast_matmul_dispatch as fast

    split, real = [], fast._maybe_splitk_dispatch

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        split.append(bool(result))
        return result

    monkeypatch.setattr(fast, "_maybe_splitk_dispatch", spy)
    kernel, n, depth = _matmul_kernel(), 64, 2048
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        a = torch.randint(-3, 4, (n, depth), generator=g).float()
        b = torch.randint(-3, 4, (depth, n), generator=g).float()
        want = (a.double() @ b.double()).float()
        A, Bm = a.to("mps"), b.to("mps")
        C = (
            {"A": A, "B": Bm}[which].reshape(-1)[: n * n].view(n, n)
            if which != "apart"
            else torch.empty(n, n, device="mps")
        )
        fast_matmul_routes.clear()
        split.clear()
        kernel[(1, 1)](A, Bm, C, n, n, depth, BM=n, BN=n, BK=32)
        torch.mps.synchronize()
        assert (fast_matmul_routes, split) == (([True], [True]) if which == "apart" else ([False], []))
        assert torch.equal(C.cpu(), want)


@requires_mps
def test_matmul_with_a_separate_output_keeps_the_replacement(fast_matmul_routes):
    kernel, n = _matmul_kernel(), 256
    a, b, want = _operands(n, 100)
    A, Bm, C = a.to("mps"), b.to("mps"), torch.empty(n, n, device="mps")
    kernel[(1, 1)](A, Bm, C, n, n, n, BM=n, BN=n, BK=32)
    torch.mps.synchronize()
    assert fast_matmul_routes == [True]
    assert torch.equal(C.cpu(), want)

    # Three views of one storage that do not intersect: still the replacement.
    arena = torch.empty(3 * n * n, device="mps")
    Av, Bv, Cv = (arena[i * n * n : (i + 1) * n * n].view(n, n) for i in range(3))
    Av.copy_(a)
    Bv.copy_(b)
    fast_matmul_routes.clear()
    kernel[(1, 1)](Av, Bv, Cv, n, n, n, BM=n, BN=n, BK=32)
    torch.mps.synchronize()
    assert fast_matmul_routes == [True]
    assert torch.equal(Cv.cpu(), want)
    assert torch.equal(Av.cpu(), a) and torch.equal(Bv.cpu(), b)


def _gemv_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def in_place_gemv(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        scale = tl.load(scale_ptr + offs_n)
        zero = tl.load(zero_ptr + offs_n)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(x_ptr + offs_k + k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk).to(tl.float32)
            acc += tl.sum(x[None, :] * (w - zero[:, None]), axis=1)
        tl.store(o_ptr + offs_n, acc * scale)

    return in_place_gemv


@requires_mps
def test_in_place_quantized_gemv_refuses_and_disjoint_views_compute():
    from triton_msl.errors import MetalNonRecoverableError

    kernel, n = _gemv_kernel(), 128
    torch.manual_seed(0)
    w = torch.randint(-127, 127, (n, n), device="mps", dtype=torch.int8)
    scale = torch.rand(n, device="mps") * 0.02 + 0.005
    zero = torch.randint(-8, 8, (n,), device="mps").float()
    arena = torch.randn(2 * n, device="mps")
    x, out = arena[:n], arena[n:]
    kept = x.clone()
    ref = ((w.float() - zero[:, None]) * scale[:, None]) @ x

    kernel[(1,)](x, w, out, scale, zero, n, n, w.stride(0), w.stride(1), BN=n, BK=32)
    torch.mps.synchronize()
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    assert torch.equal(x, kept)

    with pytest.raises(MetalNonRecoverableError, match="overlaps one of its operands"):
        kernel[(1,)](x, w, x, scale, zero, n, n, w.stride(0), w.stride(1), BN=n, BK=32)
    torch.mps.synchronize()
    assert torch.equal(x, kept), "a refused launch must not write"
