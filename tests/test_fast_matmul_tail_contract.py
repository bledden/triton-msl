import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.autotuning._fast_matmul_dispatch import dispatch_fast_matmul
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir


@triton.jit
def _unmasked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK
        bp += BK * N
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _masked(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        av = tl.load(ap, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0.0)
        bv = tl.load(bp, mask=(k + rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
        ap += BK
        bp += BK * N
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


def _descriptor(kernel):
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    ctx = ir.context()
    ir.load_dialects(ctx)
    sig = {
        "A": "*fp32",
        "B": "*fp32",
        "C": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "BM": "constexpr",
        "BN": "constexpr",
        "BK": "constexpr",
    }
    module = ASTSource(kernel, sig, {"BM": 32, "BN": 32, "BK": 32}).make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    lowerer.lower()
    return lowerer._fast_matmul


class _Runtime:
    def __init__(self):
        self.calls = []

    def is_unsupported(self, _source):
        return False

    def get_library(self, _source):
        self.calls.append("library")
        return object()

    def dispatch(self, _lib, name, args, **kwargs):
        self.calls.append((name, int(args[5]), kwargs))

    def mark_unsupported(self, _source):
        raise AssertionError("a contract miss must not blacklist the shader")


def _dispatch(desc, k):
    rt = _Runtime()
    accepted = dispatch_fast_matmul(
        rt,
        desc,
        [object(), object(), object(), 32, 32, k],
        grid=(1, 1, 1),
    )
    return accepted, rt.calls


def test_unmasked_nondisible_tail_stays_on_generic_before_runtime_work():
    desc = _descriptor(_unmasked)
    assert desc[10:12] == ("full_blocks", 32)
    accepted, calls = _dispatch(desc, 40)
    assert accepted is False
    assert calls == []


def test_unmasked_divisible_tail_retains_fast_replacement():
    desc = _descriptor(_unmasked)
    accepted, calls = _dispatch(desc, 64)
    assert accepted is True
    assert calls[0] == "library"
    assert calls[1][0:2] == ("simdgroup_matmul_fast", 64)


def test_both_zero_masked_tail_retains_logical_k_fast_replacement():
    desc = _descriptor(_masked)
    assert desc[10:12] == ("masked_zero", 32)
    accepted, calls = _dispatch(desc, 40)
    assert accepted is True
    assert calls[0] == "library"
    assert calls[1][0:2] == ("simdgroup_matmul_fast", 40)
