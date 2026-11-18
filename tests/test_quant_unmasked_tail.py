"""Unmasked quantized kernels compute WHOLE source blocks; the template routes must too.

Every kernel the quantized descriptors accept is unmasked: a program loads whole ``BK``
blocks along K and stores its whole ``BM x BN`` (``BN`` for a GEMV) tile, through flat
addresses formed from each pointer and the kernel's own strides. ``M`` / ``N`` are dead
arguments and ``K`` only bounds ``range(0, K, BK)``. With ``K % BK != 0`` or a size that
is not whole tiles, the kernel as written reads and stores past M / N / K. When those
cells lie inside the tensors' backing storage the program is defined, and its result is
NOT the template's result clipped at the arguments M / N / K (the released behaviour:
tail inputs dropped from every sum, tail outputs never stored).

Reference: a per-program simulation of the kernel as written. Integer data and
power-of-two scales, so float32 arithmetic is exact in any order and the comparison is
exact. Every output is NaN-filled and larger than the store domain, so a cell stored
outside the kernel's domain and a cell of the domain left unstored both fail.

Each test also asserts the ROUTE (the template shader that ran): whole-block launches
keep their fast template, and so do launches whose whole blocks leave a storage (an
undefined program, which keeps the clipped template).
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


# ------------------------------------------------------------------------------- kernels
@triton.jit
def _gemv8(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
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


@triton.jit
def _gemv8_inreduce(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    scale = tl.load(scale_ptr + offs_n)
    zero = tl.load(zero_ptr + offs_n)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        x = tl.load(x_ptr + offs_k + k)
        w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk).to(tl.float32)
        acc += tl.sum(x[None, :] * ((w - zero[:, None]) * scale[:, None]), axis=1)
    tl.store(o_ptr + offs_n, acc)


@triton.jit
def _gemv8_masked(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mn = offs_n < N
    scale = tl.load(scale_ptr + offs_n, mask=mn, other=0.0)
    zero = tl.load(zero_ptr + offs_n, mask=mn, other=0.0)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        mk = (offs_k + k) < K
        x = tl.load(x_ptr + offs_k + k, mask=mk, other=0.0)
        w = tl.load(
            w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk, mask=mn[:, None] & mk[None, :], other=0
        ).to(tl.float32)
        acc += tl.sum(x[None, :] * (w - zero[:, None]), axis=1)
    tl.store(o_ptr + offs_n, acc * scale, mask=mn)


@triton.jit
def _gemv8_store_masked(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
    # whole-block loads, store masked at N
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
    tl.store(o_ptr + offs_n, acc * scale, mask=offs_n < N)


@triton.jit
def _gemv4(x_ptr, w_ptr, o_ptr, s_ptr, z_ptr, N, K, ng, swn, ssn, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :])
        w4 = (packed >> ((kk % 2) * 4)[None, :]) & 0xF
        g = kk // G
        s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
        z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
        acc += tl.sum(tl.load(x_ptr + kk)[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
    tl.store(o_ptr + on, acc)


@triton.jit
def _gemv4_masked(
    x_ptr, w_ptr, o_ptr, s_ptr, z_ptr, N, K, ng, swn, ssn, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr
):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    mn = on < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        mk = kk < K
        m2 = mn[:, None] & mk[None, :]
        packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :], mask=m2, other=0)
        w4 = (packed >> ((kk % 2) * 4)[None, :]) & 0xF
        g = kk // G
        s = tl.load(s_ptr + on[:, None] * ssn + g[None, :], mask=m2, other=0.0)
        z = tl.load(z_ptr + on[:, None] * ssn + g[None, :], mask=m2, other=0.0)
        acc += tl.sum(tl.load(x_ptr + kk, mask=mk, other=0.0)[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
    tl.store(o_ptr + on, acc, mask=mn)


@triton.jit
def _gemm8(
    a_ptr,
    w_ptr,
    c_ptr,
    scale_ptr,
    zero_ptr,
    M,
    N,
    K,
    sam,
    sak,
    swk,
    swn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    w_ptrs = w_ptr + offs_k[:, None] * swk + offs_n[None, :] * swn
    scale = tl.load(scale_ptr + offs_n)
    zero = tl.load(zero_ptr + offs_n)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs)
        w = (tl.load(w_ptrs).to(tl.float32) - zero[None, :]) * scale[None, :]
        acc += tl.dot(a, w)
        a_ptrs += BK * sak
        w_ptrs += BK * swk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc)


@triton.jit
def _gemm8_nk(
    a_ptr, w_ptr, c_ptr, s_ptr, z_ptr, M, N, K, sam, swn, scm, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :]
    s = tl.load(s_ptr + offs_n)
    z = tl.load(z_ptr + offs_n)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs)
        w = (tl.load(w_ptrs).to(tl.float32) - z[:, None]) * s[:, None]
        acc += tl.dot(a, tl.trans(w))
        a_ptrs += BK
        w_ptrs += BK
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :], acc)


@triton.jit
def _pergroup8(
    a_ptr,
    w_ptr,
    c_ptr,
    scale_ptr,
    zero_ptr,
    M,
    N,
    K,
    sam,
    sak,
    swk,
    swn,
    ssg,
    ssn,
    zsg,
    zsn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
    wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        g = k // G
        s = tl.load(scale_ptr + g * ssg + on * ssn)
        z = tl.load(zero_ptr + g * zsg + on * zsn)
        w = (tl.load(wp).to(tl.float32) - z[None, :]) * s[None, :]
        acc += tl.dot(tl.load(ap), w)
        ap += BK * sak
        wp += BK * swk
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)


@triton.jit
def _sym8(
    a_ptr,
    w_ptr,
    c_ptr,
    s_ptr,
    M,
    N,
    K,
    sam,
    sak,
    swk,
    swn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
    wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
    scale = tl.load(s_ptr + on)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        w = tl.load(wp).to(tl.float32) * scale[None, :]
        acc += tl.dot(tl.load(ap), w)
        ap += BK * sak
        wp += BK * swk
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)


@triton.jit
def _pergroup4(
    a_ptr,
    w_ptr,
    c_ptr,
    s_ptr,
    z_ptr,
    M,
    N,
    K,
    sam,
    sak,
    wbk,
    wsn,
    scm,
    scn,
    ssg,
    ssn,
    zsg,
    zsn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak)
        packed = tl.load(w_ptr + (kk // 2)[:, None] * wbk + rn[None, :] * wsn)
        w4 = (packed >> ((kk % 2) * 4)[:, None]) & 0xF
        g = k // G
        s = tl.load(s_ptr + g * ssg + rn * ssn)
        z = tl.load(z_ptr + g * zsg + rn * zsn)
        acc += tl.dot(a, (w4.to(tl.float32) - z[None, :]) * s[None, :])
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)


# ---------------------------------------------------------------- the kernels as written
def _cdiv(a, b):
    return -(-a // b)


def _take(arr, idx):
    """Flat gather. An address outside the buffer (only ever a masked-off lane, or an
    exact-storage launch that is compared with the clipped math) reads 0."""
    idx = np.asarray(idx)
    inside = (idx >= 0) & (idx < arr.size)
    return np.where(inside, arr[np.clip(idx, 0, arr.size - 1)], 0)


class _Stored:
    """The output as the programs store it: NaN where no program stores. A cell that two
    programs store different values to is a race in the source; every value stored to it
    is kept, and any of them is the kernel's result."""

    def __init__(self, length):
        self.ref = np.full(length, np.nan)
        self.alternatives = {}

    def put(self, idx, val, keep):
        for i, v, k in zip(np.asarray(idx).ravel(), np.asarray(val).ravel(), np.asarray(keep).ravel()):
            if not k or i < 0 or i >= self.ref.size:
                continue
            if not np.isnan(self.ref[i]) and self.ref[i] != v:
                self.alternatives.setdefault(int(i), {float(self.ref[i])}).add(float(v))
            self.ref[i] = v

    def assert_equals(self, out):
        got = out.detach().cpu().numpy().astype(np.float64).reshape(-1)
        assert got.shape == self.ref.shape
        raced = np.zeros(got.size, bool)
        raced[list(self.alternatives)] = True
        want = np.where(raced, 0.0, self.ref)
        np.testing.assert_array_equal(np.where(raced, 0.0, got), want)
        for i, values in self.alternatives.items():
            assert got[i] in values, (i, got[i], values)


def _sim_gemv8(x, w, s, z, o_len, *, N, K, BN, BK, masked=False, store_masked=False):
    """Weight flat [n*K + k]. masked: loads zero-filled outside N / K, store clipped at N.
    store_masked: whole-block loads, store clipped at N."""
    out = _Stored(o_len)
    for p in range(_cdiv(N, BN)):
        rn = p * BN + np.arange(BN)
        zr, sr = _take(z, rn).astype(np.float64), _take(s, rn).astype(np.float64)
        acc = np.zeros(BN)
        for kb in range(0, K, BK):
            kk = kb + np.arange(BK)
            xb = _take(x, kk).astype(np.float64)
            wb = _take(w, rn[:, None] * K + kk[None, :]).astype(np.float64)
            if masked:
                xb = np.where(kk < K, xb, 0.0)
                wb = np.where((rn < N)[:, None] & (kk < K)[None, :], wb, 0.0)
                zr = np.where(rn < N, zr, 0.0)
            acc += (xb[None, :] * (wb - zr[:, None])).sum(1)
        out.put(rn, acc * sr, (rn < N) if (masked or store_masked) else np.ones(BN, bool))
    return out


def _sim_gemv4(x, packed, s, z, o_len, *, N, K, BN, BK, G, masked=False):
    """Packed weight flat [n*(K/2) + k/2], scale / zero flat [n*(K/G) + k/G]."""
    out = _Stored(o_len)
    swn, ssn = K // 2, K // G
    for p in range(_cdiv(N, BN)):
        rn = p * BN + np.arange(BN)
        acc = np.zeros(BN)
        for kb in range(0, K, BK):
            kk = kb + np.arange(BK)
            xb = _take(x, kk).astype(np.float64)
            byte = _take(packed, rn[:, None] * swn + (kk // 2)[None, :]).astype(np.int64)
            w4 = ((byte >> ((kk % 2) * 4)[None, :]) & 0xF).astype(np.float64)
            g = kk // G
            sb = _take(s, rn[:, None] * ssn + g[None, :]).astype(np.float64)
            zb = _take(z, rn[:, None] * ssn + g[None, :]).astype(np.float64)
            if masked:
                m2 = (rn < N)[:, None] & (kk < K)[None, :]
                xb = np.where(kk < K, xb, 0.0)
                w4, sb, zb = np.where(m2, w4, 0.0), np.where(m2, sb, 0.0), np.where(m2, zb, 0.0)
            acc += (xb[None, :] * ((w4 - zb) * sb)).sum(1)
        out.put(rn, acc, (rn < N) if masked else np.ones(BN, bool))
    return out


def _sim_gemm(a, weight, s, z, c_len, *, M, N, K, BM, BN, BK, G, ssg, scm, clipped=False):
    """Input flat [m*K + k], output flat [m*scm + n], scale / zero flat [g*ssg + n] with g
    the group of the block's FIRST k (``k // G``; 0 without groups). ``weight(kk, rn)``
    gathers the (BK, BN) integer weight block. clipped: the math over m < M, n < N, k < K
    only, with the group of every element (what the templates compute at M / N / K)."""
    out = _Stored(c_len)
    for pm in range(_cdiv(M, BM)):
        for pn in range(_cdiv(N, BN)):
            rm, rn = pm * BM + np.arange(BM), pn * BN + np.arange(BN)
            acc = np.zeros((BM, BN))
            for kb in range(0, K, BK):
                kk = kb + np.arange(BK)
                ab = _take(a, rm[:, None] * K + kk[None, :]).astype(np.float64)
                wb = weight(kk, rn)
                if G:
                    g = (kk // G) if clipped else np.full(BK, kb // G)
                else:
                    g = np.zeros(BK, np.int64)
                sb = _take(s, g[:, None] * ssg + rn[None, :]).astype(np.float64)
                zb = _take(z, g[:, None] * ssg + rn[None, :]).astype(np.float64) if z is not None else 0.0
                if clipped:
                    ab = np.where((rm < M)[:, None] & (kk < K)[None, :], ab, 0.0)
                    wb = np.where((kk < K)[:, None] & (rn < N)[None, :], wb, 0.0)
                    zb = np.where((kk < K)[:, None] & (rn < N)[None, :], zb, 0.0)
                acc += ab @ ((wb - zb) * sb)
            keep = (rm < M)[:, None] & (rn < N)[None, :] if clipped else np.ones((BM, BN), bool)
            out.put(rm[:, None] * scm + rn[None, :], acc, keep)
    return out


# ------------------------------------------------------------------------------- harness
@pytest.fixture
def route(monkeypatch):
    """What ran: ``route.quant`` = dispatch_quant_matmul's results, ``route.shaders`` = the
    compile_shader entry points launched."""
    import triton_msl.autotuning._quant_matmul_dispatch as quant_dispatch
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_compile_shader_runtime

    class Route:
        quant = []
        shaders = []

    seen = Route()
    seen.quant, seen.shaders = [], []
    real_quant = quant_dispatch.dispatch_quant_matmul

    def spy_quant(*args, **kwargs):
        result = real_quant(*args, **kwargs)
        seen.quant.append(bool(result))
        return result

    def observed(_runtime, launch):
        def spy_dispatch(lib, kernel_name, args, **kwargs):
            seen.shaders.append(kernel_name)
            return launch(lib, kernel_name, args, **kwargs)

        return spy_dispatch

    monkeypatch.setattr(quant_dispatch, "dispatch_quant_matmul", spy_quant)
    # The runtime is one object for the process, and a test that ran earlier may have left
    # its own ``dispatch`` on it, which a spy on the class would never see.
    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime, "dispatch", observed)
    return seen


def _dev(array):
    return torch.from_numpy(array).to("mps")


def _nan(*shape):
    return torch.full(shape, float("nan"), device="mps")


def _gemv8_case(N, K, BN, BK, *, slack, seed=1):
    """Buffers of an int8 GEMV launch. slack: every buffer holds the kernel's whole blocks
    (nonzero data past N / K) and the output is longer than the store domain."""
    g = np.random.default_rng(seed)
    Nr, Kr = _cdiv(N, BN) * BN, _cdiv(K, BK) * BK
    rows, xlen, vlen, olen = (Nr + 2, Kr + 8, Nr + 8, Nr + 8) if slack else (N, K, N, N)
    x = g.integers(-4, 5, size=xlen).astype(np.float32)
    w = g.integers(-32, 32, size=(rows, K)).astype(np.int8)
    s = (2.0 ** g.integers(-1, 3, size=vlen)).astype(np.float32)
    z = g.integers(-4, 5, size=vlen).astype(np.float32)
    return x, w, s, z, olen


def _launch_gemv8(kernel, N, K, BN, BK, x, w, s, z, olen):
    gw, out = _dev(w), _nan(olen)
    kernel[(_cdiv(N, BN),)](_dev(x), gw, out, _dev(s), _dev(z), N, K, gw.stride(0), gw.stride(1), BN=BN, BK=BK)
    torch.mps.synchronize()
    return out


def _gemv4_case(N, K, BN, BK, G, *, slack, seed=2):
    g = np.random.default_rng(seed)
    Nr, Kr = _cdiv(N, BN) * BN, _cdiv(K, BK) * BK
    rows, xlen, olen = (Nr + 3, Kr + 8, Nr + 8) if slack else (N, K, N)
    x = g.integers(-4, 5, size=xlen).astype(np.float32)
    packed = g.integers(0, 256, size=(rows, K // 2)).astype(np.uint8)
    s = (2.0 ** g.integers(-1, 3, size=(rows, K // G))).astype(np.float32)
    z = g.integers(0, 16, size=(rows, K // G)).astype(np.float32)
    return x, packed, s, z, olen


def _launch_gemv4(kernel, N, K, BN, BK, G, x, packed, s, z, olen):
    gp, gs, out = _dev(packed), _dev(s), _nan(olen)
    kernel[(_cdiv(N, BN),)](_dev(x), gp, out, gs, _dev(z), N, K, K // G, gp.stride(0), gs.stride(0), BN=BN, BK=BK, G=G)
    torch.mps.synchronize()
    return out


def _gemm_case(kind, M, N, K, BM, BN, BK, G, *, storage, seed=3):
    """Buffers and the two references of a GEMM launch. kind: 'kn' / 'nk' canonical,
    'pergroup8', 'sym8', 'pergroup4'. storage: 'exact' (the logical extents), 'slack'
    (whole blocks fit, output rows of N cells) or 'wide' (as slack, and an output row
    holds the rounded N, so whole-tile stores of one row do not reach the next)."""
    g = np.random.default_rng(seed)
    Mr, Nr, Kr = _cdiv(M, BM) * BM, _cdiv(N, BN) * BN, _cdiv(K, BK) * BK
    exact = storage == "exact"
    a = g.integers(-4, 5, size=(M if exact else Mr + 2, K)).astype(np.float32)
    ccols = Nr + 3 if storage == "wide" else N
    crows = M if exact else Mr + 2
    grouped = kind in ("pergroup8", "pergroup4")
    if kind == "pergroup4":
        w = g.integers(0, 256, size=((K // 2) if exact else Kr // 2 + 3, N)).astype(np.uint8)
    elif kind == "nk":
        w = g.integers(-32, 32, size=(N if exact else Nr + 3, K)).astype(np.int8)
    else:
        w = g.integers(-32, 32, size=(K if exact else Kr + 3, N)).astype(np.int8)
    if grouped:
        groups = _cdiv(K, G) if exact else _cdiv(Kr, G) + 3
        s = (2.0 ** g.integers(-1, 3, size=(groups, N))).astype(np.float32)
        z = g.integers(0, 16, size=(groups, N)) if kind == "pergroup4" else g.integers(-4, 5, size=(groups, N))
        z = z.astype(np.float32)
    else:
        s = (2.0 ** g.integers(-1, 3, size=N if exact else Nr + 8)).astype(np.float32)
        z = g.integers(-4, 5, size=N if exact else Nr + 8).astype(np.float32)
    wf = w.reshape(-1)
    if kind == "pergroup4":

        def weight(kk, rn):
            byte = _take(wf, (kk // 2)[:, None] * N + rn[None, :]).astype(np.int64)
            return ((byte >> ((kk % 2) * 4)[:, None]) & 0xF).astype(np.float64)

    elif kind == "nk":

        def weight(kk, rn):
            return _take(wf, kk[:, None] + rn[None, :] * K).astype(np.float64)

    else:

        def weight(kk, rn):
            return _take(wf, kk[:, None] * N + rn[None, :]).astype(np.float64)

    sim = dict(M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, G=G if grouped else 0, ssg=N if grouped else 0, scm=ccols)
    zf = None if kind == "sym8" else z.reshape(-1)
    written = _sim_gemm(a.reshape(-1), weight, s.reshape(-1), zf, crows * ccols, **sim)
    clipped = _sim_gemm(a.reshape(-1), weight, s.reshape(-1), zf, crows * ccols, clipped=True, **sim)
    return (a, w, s, z, (crows, ccols)), written, clipped


def _launch_gemm(kind, M, N, K, BM, BN, BK, G, buffers):
    a, w, s, z, cshape = buffers
    ga, gw, gs, gz, out = _dev(a), _dev(w), _dev(s), _dev(z), _nan(*cshape)
    grid = (_cdiv(M, BM), _cdiv(N, BN))
    blocks = dict(BM=BM, BN=BN, BK=BK)
    if kind == "kn":
        _gemm8[grid](ga, gw, out, gs, gz, M, N, K, *ga.stride(), *gw.stride(), *out.stride(), **blocks)
    elif kind == "nk":
        _gemm8_nk[grid](ga, gw, out, gs, gz, M, N, K, ga.stride(0), gw.stride(0), out.stride(0), **blocks)
    elif kind == "pergroup8":
        _pergroup8[grid](
            ga,
            gw,
            out,
            gs,
            gz,
            M,
            N,
            K,
            *ga.stride(),
            *gw.stride(),
            *gs.stride(),
            *gz.stride(),
            *out.stride(),
            G=G,
            **blocks,
        )
    elif kind == "sym8":
        _sym8[grid](ga, gw, out, gs, M, N, K, *ga.stride(), *gw.stride(), *out.stride(), **blocks)
    else:
        _pergroup4[grid](
            ga,
            gw,
            out,
            gs,
            gz,
            M,
            N,
            K,
            *ga.stride(),
            *gw.stride(),
            *out.stride(),
            *gs.stride(),
            *gz.stride(),
            G=G,
            **blocks,
        )
    torch.mps.synchronize()
    return out


# ------------------------------------------------------------------------------ int8 GEMV
# (N, K, BN, BK, the template that computes the kernel as written). K is the weight row
# stride of int8_gemv as well as its extent, so a K tail needs the stride-generic template.
_GEMV8_TAILS = [
    (64, 100, 32, 32, "int8_matmul_pergroup"),  # K tail
    (64, 96, 16, 64, "int8_matmul_pergroup"),  # K tail, BK wider than the template's
    (40, 128, 32, 32, "int8_gemv"),  # N tail
    (40, 100, 32, 32, "int8_matmul_pergroup"),  # both
    (24, 96, 16, 64, "int8_matmul_pergroup"),  # both
]


@requires_mps
@pytest.mark.parametrize("kernel", [_gemv8, _gemv8_inreduce], ids=["epilogue-scale", "in-reduce-scale"])
@pytest.mark.parametrize("N,K,BN,BK,shader", _GEMV8_TAILS)
def test_int8_gemv_unmasked_tail_is_the_kernel_as_written(route, kernel, N, K, BN, BK, shader):
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=True)
    out = _launch_gemv8(kernel, N, K, BN, BK, x, w, s, z, olen)
    _sim_gemv8(x, w.reshape(-1), s, z, olen, N=N, K=K, BN=BN, BK=BK).assert_equals(out)
    assert route.quant == [True] and route.shaders == [shader]


@requires_mps
@pytest.mark.parametrize("slack", [True, False], ids=["slack", "exact"])
@pytest.mark.parametrize("kernel", [_gemv8, _gemv8_inreduce], ids=["epilogue-scale", "in-reduce-scale"])
def test_int8_gemv_whole_blocks_keep_the_fast_template(route, kernel, slack):
    N, K, BN, BK = 64, 128, 32, 32
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=slack)
    out = _launch_gemv8(kernel, N, K, BN, BK, x, w, s, z, olen)
    _sim_gemv8(x, w.reshape(-1), s, z, olen, N=N, K=K, BN=BN, BK=BK).assert_equals(out)
    assert route.quant == [True] and route.shaders == ["int8_gemv"]


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK", [(64, 100, 32, 32), (40, 128, 32, 32), (24, 96, 16, 64)])
def test_int8_gemv_tail_outside_the_storage_keeps_the_clipped_template(route, N, K, BN, BK):
    """Buffers of exactly N / K elements: the whole blocks leave every storage, so the
    program is undefined as written. The launch keeps the fast template clipped at N / K,
    which stays inside the tensors."""
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=False)
    out = _launch_gemv8(_gemv8, N, K, BN, BK, x, w, s, z, olen)
    _sim_gemv8(x, w.reshape(-1), s, z, olen, N=N, K=K, BN=BN, BK=BK, masked=True).assert_equals(out)
    assert route.quant == [True] and route.shaders == ["int8_gemv"]


@requires_mps
def test_int8_gemv_tail_with_one_short_operand_keeps_the_clipped_template(route):
    """Whole blocks fit every buffer but the input vector: still undefined as written."""
    N, K, BN, BK = 64, 100, 32, 32
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=True)
    x = x[:K].copy()
    out = _launch_gemv8(_gemv8, N, K, BN, BK, x, w, s, z, olen)
    _sim_gemv8(x, w.reshape(-1), s, z, olen, N=N, K=K, BN=BN, BK=BK, masked=True).assert_equals(out)
    assert route.quant == [True] and route.shaders == ["int8_gemv"]


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK", [(64, 128, 32, 32), (64, 100, 32, 32), (40, 128, 32, 32)])
@pytest.mark.parametrize("short", ["input", "weight", "output", "scale", "zero"])
def test_int8_gemv_with_a_tensor_shorter_than_its_sizes_is_refused_by_name(route, N, K, BN, BK, short):
    """A tensor that does not hold the cells addressed at N / K: no template is launched
    on it, whole blocks or not."""
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=False)
    x = x[: K - 1].copy() if short == "input" else x
    w = w.reshape(-1)[: N * K - 1].copy() if short == "weight" else w
    s = s[: N - 1].copy() if short == "scale" else s
    z = z[: N - 1].copy() if short == "zero" else z
    out = _nan(olen - 1 if short == "output" else olen)
    with pytest.raises(MetalNonRecoverableError, match="quantized.*storage is smaller than the cells"):
        _gemv8[(_cdiv(N, BN),)](_dev(x), _dev(w), out, _dev(s), _dev(z), N, K, K, 1, BN=BN, BK=BK)
    assert route.quant == [False] and route.shaders == []
    assert torch.isnan(out).all()


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK", [(64, 128, 32, 32), (64, 100, 32, 32), (40, 128, 32, 32), (24, 96, 16, 64)])
def test_int8_gemv_masked_spelling_is_clipped_by_its_masks(route, N, K, BN, BK):
    """Control: with masks the source itself clips at N / K. Not a quantized-template
    launch (the descriptor accepts only the unmasked form)."""
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=True)
    out = _launch_gemv8(_gemv8_masked, N, K, BN, BK, x, w, s, z, olen)
    _sim_gemv8(x, w.reshape(-1), s, z, olen, N=N, K=K, BN=BN, BK=BK, masked=True).assert_equals(out)
    assert route.quant == [] and "int8_gemv" not in route.shaders


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK", [(64, 100, 32, 32), (40, 128, 32, 32), (24, 96, 16, 64)])
def test_int8_gemv_store_masked_spelling_reads_whole_blocks(route, N, K, BN, BK):
    """Control: whole-block loads and a store masked at N — the K tail is read, rows past
    N are not stored. Not a quantized-template launch."""
    x, w, s, z, olen = _gemv8_case(N, K, BN, BK, slack=True)
    out = _launch_gemv8(_gemv8_store_masked, N, K, BN, BK, x, w, s, z, olen)
    _sim_gemv8(x, w.reshape(-1), s, z, olen, N=N, K=K, BN=BN, BK=BK, store_masked=True).assert_equals(out)
    assert route.quant == [] and "int8_gemv" not in route.shaders


# ------------------------------------------------------------------------------ int4 GEMV
_GEMV4_TAILS = [
    (64, 96, 32, 64, 32, "int4_matmul_pergroup"),  # K tail
    (40, 128, 32, 32, 32, "int4_gemv"),  # N tail
    (40, 96, 32, 64, 32, "int4_matmul_pergroup"),  # both
    (64, 100, 32, 32, 4, "int4_matmul_pergroup"),  # K tail, small groups
]


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK,G,shader", _GEMV4_TAILS)
def test_int4_gemv_unmasked_tail_is_the_kernel_as_written(route, N, K, BN, BK, G, shader):
    x, packed, s, z, olen = _gemv4_case(N, K, BN, BK, G, slack=True)
    out = _launch_gemv4(_gemv4, N, K, BN, BK, G, x, packed, s, z, olen)
    ref = _sim_gemv4(x, packed.reshape(-1), s.reshape(-1), z.reshape(-1), olen, N=N, K=K, BN=BN, BK=BK, G=G)
    ref.assert_equals(out)
    assert route.quant == [True] and route.shaders == [shader]


@requires_mps
@pytest.mark.parametrize("slack", [True, False], ids=["slack", "exact"])
@pytest.mark.parametrize("G", [32, 64])
def test_int4_gemv_whole_blocks_keep_the_fast_template(route, G, slack):
    N, K, BN, BK = 64, 128, 32, 32
    x, packed, s, z, olen = _gemv4_case(N, K, BN, BK, G, slack=slack)
    out = _launch_gemv4(_gemv4, N, K, BN, BK, G, x, packed, s, z, olen)
    ref = _sim_gemv4(x, packed.reshape(-1), s.reshape(-1), z.reshape(-1), olen, N=N, K=K, BN=BN, BK=BK, G=G)
    ref.assert_equals(out)
    assert route.quant == [True] and route.shaders == ["int4_gemv"]


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK,G", [(64, 96, 32, 64, 32), (40, 128, 32, 32, 32)])
def test_int4_gemv_tail_outside_the_storage_keeps_the_clipped_template(route, N, K, BN, BK, G):
    x, packed, s, z, olen = _gemv4_case(N, K, BN, BK, G, slack=False)
    out = _launch_gemv4(_gemv4, N, K, BN, BK, G, x, packed, s, z, olen)
    ref = _sim_gemv4(
        x, packed.reshape(-1), s.reshape(-1), z.reshape(-1), olen, N=N, K=K, BN=BN, BK=BK, G=G, masked=True
    )
    ref.assert_equals(out)
    assert route.quant == [True] and route.shaders == ["int4_gemv"]


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK,G", [(64, 128, 32, 32, 32), (40, 128, 32, 32, 32)])
def test_int4_gemv_masked_spelling_is_clipped_by_its_masks(route, N, K, BN, BK, G):
    x, packed, s, z, olen = _gemv4_case(N, K, BN, BK, G, slack=True)
    out = _launch_gemv4(_gemv4_masked, N, K, BN, BK, G, x, packed, s, z, olen)
    ref = _sim_gemv4(
        x, packed.reshape(-1), s.reshape(-1), z.reshape(-1), olen, N=N, K=K, BN=BN, BK=BK, G=G, masked=True
    )
    ref.assert_equals(out)
    assert route.quant == [] and "int4_gemv" not in route.shaders


# ----------------------------------------------------------------------------------- GEMM
# (kind, M, N, K, BM, BN, BK, G, storage, the template that computes the kernel as written).
# M is an extent only of the fast templates; N and K are strides as well, so a rounded N or
# K needs the stride-generic scalar template.
_GEMM_TAILS = [
    ("kn", 96, 64, 64, 64, 32, 32, 0, "slack", "int8_matmul_fast"),  # M tail
    ("kn", 64, 64, 96, 32, 32, 64, 0, "slack", "int8_matmul_pergroup"),  # K tail
    ("kn", 64, 48, 64, 32, 32, 32, 0, "slack", "int8_matmul_pergroup"),  # N tail (rows collide)
    ("nk", 96, 64, 64, 64, 32, 32, 0, "slack", "int8_matmul_fast"),
    ("nk", 64, 64, 96, 32, 32, 64, 0, "slack", "int8_matmul_pergroup"),
    ("nk", 64, 48, 64, 32, 32, 32, 0, "slack", "int8_matmul_pergroup"),
    ("pergroup8", 96, 64, 64, 64, 32, 32, 32, "slack", "int8_matmul_pergroup_fast"),
    ("pergroup8", 48, 64, 64, 32, 32, 32, 32, "slack", "int8_matmul_pergroup_fast"),
    ("pergroup8", 64, 64, 80, 32, 32, 32, 32, "slack", "int8_matmul_pergroup"),
    ("pergroup8", 64, 64, 96, 32, 32, 64, 64, "slack", "int8_matmul_pergroup"),
    ("pergroup8", 64, 40, 64, 32, 32, 32, 32, "wide", "int8_matmul_pergroup"),
    ("pergroup8", 48, 40, 80, 32, 32, 32, 32, "wide", "int8_matmul_pergroup"),
    ("pergroup8", 64, 48, 64, 32, 32, 32, 32, "slack", "int8_matmul_pergroup"),  # rows collide
    ("sym8", 96, 64, 64, 64, 32, 32, 0, "slack", "int8_matmul_pergroup"),
    ("sym8", 64, 64, 80, 32, 32, 32, 0, "slack", "int8_matmul_pergroup"),
    ("sym8", 48, 40, 80, 32, 32, 32, 0, "wide", "int8_matmul_pergroup"),
    ("sym8", 64, 40, 64, 32, 32, 32, 0, "slack", "int8_matmul_pergroup"),  # rows collide
    ("pergroup4", 96, 64, 64, 64, 32, 32, 32, "slack", "int4_matmul_pergroup_fast"),
    ("pergroup4", 48, 64, 64, 32, 32, 32, 32, "slack", "int4_matmul_pergroup_fast"),
    ("pergroup4", 64, 64, 80, 32, 32, 32, 32, "slack", "int4_matmul_pergroup"),
    ("pergroup4", 64, 64, 96, 32, 32, 64, 64, "slack", "int4_matmul_pergroup"),
    ("pergroup4", 48, 40, 80, 32, 32, 32, 32, "wide", "int4_matmul_pergroup"),
    ("pergroup4", 64, 48, 64, 32, 32, 32, 32, "slack", "int4_matmul_pergroup"),  # rows collide
]


def _gemm_id(case):
    kind, M, N, K, BM, BN, BK, G, storage = case[:9]
    return f"{kind}-M{M}-N{N}-K{K}-BM{BM}-BN{BN}-BK{BK}-G{G}-{storage}"


@requires_mps
@pytest.mark.parametrize("case", _GEMM_TAILS, ids=_gemm_id)
def test_gemm_unmasked_tail_is_the_kernel_as_written(route, case):
    kind, M, N, K, BM, BN, BK, G, storage, shader = case
    buffers, written, clipped = _gemm_case(kind, M, N, K, BM, BN, BK, G, storage=storage)
    # The case is a tail: the kernel as written is not the clipped math.
    assert not np.array_equal(written.ref, clipped.ref, equal_nan=True)
    out = _launch_gemm(kind, M, N, K, BM, BN, BK, G, buffers)
    written.assert_equals(out)
    assert route.quant == [True] and route.shaders == [shader]


_GEMM_WHOLE = [
    ("kn", 64, 64, 64, 32, 32, 32, 0, "int8_matmul_fast"),
    ("kn", 64, 64, 128, 32, 32, 64, 0, "int8_matmul_fast"),
    ("nk", 64, 64, 64, 32, 32, 32, 0, "int8_matmul_fast"),
    ("pergroup8", 64, 64, 64, 32, 32, 32, 32, "int8_matmul_pergroup_fast"),
    ("pergroup8", 64, 64, 128, 32, 32, 64, 64, "int8_matmul_pergroup_fast"),
    ("sym8", 64, 64, 128, 32, 32, 64, 0, "int8_matmul_pergroup"),
    ("pergroup4", 64, 64, 64, 32, 32, 32, 32, "int4_matmul_pergroup_fast"),
    ("pergroup4", 64, 64, 128, 32, 32, 64, 64, "int4_matmul_pergroup_fast"),
]


@requires_mps
@pytest.mark.parametrize("storage", ["slack", "exact"])
@pytest.mark.parametrize("case", _GEMM_WHOLE, ids=lambda c: _gemm_id((*c[:8], "")))
def test_gemm_whole_blocks_keep_their_template(route, case, storage):
    kind, M, N, K, BM, BN, BK, G, shader = case
    buffers, written, clipped = _gemm_case(kind, M, N, K, BM, BN, BK, G, storage=storage)
    assert np.array_equal(written.ref, clipped.ref, equal_nan=True) and not written.alternatives
    out = _launch_gemm(kind, M, N, K, BM, BN, BK, G, buffers)
    written.assert_equals(out)
    assert route.quant == [True] and route.shaders == [shader]


# (kind, M, N, K, BM, BN, BK, G, the template each launch has always used)
_GEMM_EXACT_TAILS = [
    ("kn", 96, 64, 64, 64, 32, 32, 0, "int8_matmul_fast"),
    ("kn", 64, 64, 96, 32, 32, 64, 0, "int8_matmul_fast"),
    ("nk", 64, 48, 64, 32, 32, 32, 0, "int8_matmul_fast"),
    ("pergroup8", 48, 64, 64, 32, 32, 32, 32, "int8_matmul_pergroup"),
    ("pergroup8", 64, 40, 64, 32, 32, 32, 32, "int8_matmul_pergroup"),
    ("sym8", 48, 40, 80, 32, 32, 32, 0, "int8_matmul_pergroup"),
    ("pergroup4", 96, 64, 64, 64, 32, 32, 32, "int4_matmul_pergroup_fast"),
]


@requires_mps
@pytest.mark.parametrize("case", _GEMM_EXACT_TAILS, ids=lambda c: _gemm_id((*c[:8], "exact")))
def test_gemm_tail_outside_the_storage_keeps_the_clipped_template(route, case):
    """Tensors of exactly M / N / K: the whole blocks leave the storages (undefined as
    written), and the launch keeps the template clipped at M / N / K."""
    kind, M, N, K, BM, BN, BK, G, shader = case
    buffers, _written, clipped = _gemm_case(kind, M, N, K, BM, BN, BK, G, storage="exact")
    out = _launch_gemm(kind, M, N, K, BM, BN, BK, G, buffers)
    clipped.assert_equals(out)
    assert route.quant == [True] and route.shaders == [shader]


@requires_mps
@pytest.mark.parametrize("kind", ["kn", "nk"])
@pytest.mark.parametrize("M,N,K,BM,BN,BK", [(64, 64, 64, 32, 32, 32), (96, 64, 64, 64, 32, 32)])
def test_canonical_gemm_with_an_input_shorter_than_its_sizes_is_refused_by_name(route, kind, M, N, K, BM, BN, BK):
    (a, w, s, z, cshape), _written, _clipped = _gemm_case(kind, M, N, K, BM, BN, BK, 0, storage="exact")
    with pytest.raises(MetalNonRecoverableError, match="quantized.*storage is smaller than the cells"):
        _launch_gemm(kind, M, N, K, BM, BN, BK, 0, (a[:-1].copy(), w, s, z, cshape))
    assert route.quant == [False] and route.shaders == []


@requires_mps
@pytest.mark.parametrize("N,K,BN,BK,G", [(64, 128, 32, 32, 32), (40, 128, 32, 32, 32)])
def test_int4_gemv_with_zeros_shorter_than_its_sizes_is_refused_by_name(route, N, K, BN, BK, G):
    x, packed, s, z, olen = _gemv4_case(N, K, BN, BK, G, slack=False)
    with pytest.raises(MetalNonRecoverableError, match="quantized.*storage is smaller than the cells"):
        _launch_gemv4(_gemv4, N, K, BN, BK, G, x, packed, s, z.reshape(-1)[:-1].copy(), olen)
    assert route.quant == [False] and route.shaders == []


# A per-group kernel loads ONE scale / zero per K block, at the group of the block's first
# k. Both per-group templates take the group of every element: a different result whenever
# a block straddles a group boundary — at whole-block sizes too.
@requires_mps
@pytest.mark.parametrize("kind", ["pergroup8", "pergroup4"])
@pytest.mark.parametrize(
    "M,N,K,BM,BN,BK,G",
    [(64, 64, 64, 32, 32, 32, 16), (64, 64, 128, 32, 32, 64, 32), (64, 64, 96, 32, 32, 64, 32)],
)
@pytest.mark.parametrize("storage", ["slack", "exact"])
def test_pergroup_block_straddling_groups_is_refused_by_name(route, kind, M, N, K, BM, BN, BK, G, storage):
    buffers, written, clipped = _gemm_case(kind, M, N, K, BM, BN, BK, G, storage=storage)
    # What the templates would compute is not the kernel as written.
    assert not np.array_equal(written.ref, clipped.ref, equal_nan=True)
    with pytest.raises(MetalNonRecoverableError, match=rf"quantized.*K block of {BK}.*straddles groups of {G}"):
        _launch_gemm(kind, M, N, K, BM, BN, BK, G, buffers)
    assert route.quant == [False] and route.shaders == []


@requires_mps
@pytest.mark.parametrize("kind", ["pergroup8", "pergroup4"])
def test_pergroup_block_inside_one_group_still_runs(route, kind):
    """G not a multiple of BK, but every block of the launch lies in group 0."""
    M, N, K, BM, BN, BK, G = 64, 64, 32, 32, 32, 32, 48
    buffers, written, clipped = _gemm_case(kind, M, N, K, BM, BN, BK, G, storage="slack")
    assert np.array_equal(written.ref, clipped.ref, equal_nan=True)
    out = _launch_gemm(kind, M, N, K, BM, BN, BK, G, buffers)
    written.assert_equals(out)
    assert route.quant == [True] and len(route.shaders) == 1


# ------------------------------------------------------------- tensors in one storage
_POISON = -16384.0


def _slices(order, sizes, *, room=0):
    """One device storage of poison and a view of it per name, five cells apart, in ``order``."""
    arena = torch.full((sum(sizes[name] + 5 for name in order) + room,), _POISON, device="mps")
    views, spans, at = {}, {}, 0
    for name in order:
        views[name], spans[name] = arena[at : at + sizes[name]], (at, at + sizes[name])
        at += sizes[name] + 5
    return arena, views, spans


def _outside(arena, before, span):
    """Cells of the storage outside ``span`` that a launch changed."""
    changed = arena != before
    changed[span[0] : span[1]] = False
    return int(changed.sum())


_GEMV_ORDERS = [("s", "o"), ("o", "s"), ("z", "o"), ("o", "z"), ("x", "o"), ("x", "s", "z", "o"), ("o", "z", "s", "x")]


@requires_mps
@pytest.mark.parametrize("order", _GEMV_ORDERS, ids=["-".join(order) for order in _GEMV_ORDERS])
def test_int8_gemv_computes_with_operands_in_the_output_storage_and_no_shared_cell(route, order):
    """The stride contract pins every cell the launch reaches, so tensors that share a
    storage without sharing a cell are apart: scale and zero count like the input."""
    N, K, BN, BK = 64, 128, 32, 32
    x, w, s, z, _ = _gemv8_case(N, K, BN, BK, slack=False)
    arena, views, spans = _slices(order, {"x": K, "o": N, "s": N, "z": N})
    tensors = {"x": _dev(x), "s": _dev(s), "z": _dev(z), "o": _nan(N)}
    for name in order:
        if name != "o":
            views[name].copy_(tensors[name])
        tensors[name] = views[name]
    before, gw = arena.clone(), _dev(w)
    _gemv8[(N // BN,)](tensors["x"], gw, tensors["o"], tensors["s"], tensors["z"], N, K, K, 1, BN=BN, BK=BK)
    torch.mps.synchronize()
    _sim_gemv8(x, w.reshape(-1), s, z, N, N=N, K=K, BN=BN, BK=BK).assert_equals(tensors["o"])
    assert _outside(arena, before, spans["o"]) == 0
    assert route.quant == [True] and route.shaders == ["int8_gemv"]


@requires_mps
@pytest.mark.parametrize("operand", ["x", "s", "z"])
@pytest.mark.parametrize("side", ["before", "after"])
def test_int8_gemv_refuses_an_operand_that_shares_one_cell_with_the_output(route, operand, side):
    N, K, BN, BK = 64, 128, 32, 32
    x, w, s, z, _ = _gemv8_case(N, K, BN, BK, slack=False)
    count = K if operand == "x" else N
    arena = torch.full((count + N - 1,), _POISON, device="mps")
    out, view = (arena[count - 1 :], arena[:count]) if side == "before" else (arena[:N], arena[N - 1 :])
    tensors = {"x": _dev(x), "s": _dev(s), "z": _dev(z)}
    view.copy_(tensors[operand])
    tensors[operand] = view
    before = arena.clone()
    with pytest.raises(MetalNonRecoverableError, match="quantized.*overlaps one of its operands"):
        _gemv8[(N // BN,)](tensors["x"], _dev(w), out, tensors["s"], tensors["z"], N, K, K, 1, BN=BN, BK=BK)
    assert torch.equal(arena, before)
    assert route.quant == [False] and route.shaders == []


@requires_mps
def test_int8_gemv_tail_is_compared_at_the_whole_blocks_it_stores(route):
    """N = 40 in blocks of 32: as written the launch stores 64 cells. With the scale 45
    cells above the output those stores land on scale cells another program loads, so
    the launch is refused; with the scale below the output it computes all 64."""
    N, K, BN, BK = 40, 128, 32, 32
    x, w, s, z, _ = _gemv8_case(N, K, BN, BK, slack=True)
    gx, gw, gz = _dev(x), _dev(w), _dev(z)
    arena, views, _ = _slices(("o", "s"), {"o": N, "s": 64}, room=64)
    views["s"].copy_(_dev(s[:64]))
    before = arena.clone()
    with pytest.raises(MetalNonRecoverableError, match="quantized.*overlaps one of its operands"):
        _gemv8[(2,)](gx, gw, views["o"], views["s"], gz, N, K, K, 1, BN=BN, BK=BK)
    assert torch.equal(arena, before)
    assert route.quant == [False] and route.shaders == []

    arena, views, spans = _slices(("s", "o"), {"o": N, "s": 64}, room=64)
    views["s"].copy_(_dev(s[:64]))
    before = arena.clone()
    _gemv8[(2,)](gx, gw, views["o"], views["s"], gz, N, K, K, 1, BN=BN, BK=BK)
    torch.mps.synchronize()
    start = spans["o"][0]
    _sim_gemv8(x, w.reshape(-1), s, z, 64, N=N, K=K, BN=BN, BK=BK).assert_equals(arena[start : start + 64])
    assert _outside(arena, before, (start, start + 64)) == 0
    assert route.quant == [False, True] and route.shaders == ["int8_gemv"]


_GEMM_ORDERS = [("s", "c"), ("c", "s"), ("z", "c"), ("c", "z"), ("a", "c"), ("a", "s", "z", "c")]


@requires_mps
@pytest.mark.parametrize("widen", [None, "a", "s", "z"])
@pytest.mark.parametrize("order", _GEMM_ORDERS, ids=["-".join(order) for order in _GEMM_ORDERS])
def test_canonical_gemm_computes_with_operands_in_the_output_storage_and_no_shared_cell(route, order, widen):
    """``widen`` passes that operand as a view that runs on over what follows it in the
    storage: the kernel reads only the cells its sizes address, so nothing changes."""
    if widen is not None and widen not in order:
        pytest.skip("the operand is not in the shared storage")
    M, N, K, BM, BN, BK = 64, 48, 96, 32, 16, 32
    g = np.random.default_rng(11)
    a = g.integers(-4, 5, size=(M, K)).astype(np.float32)
    w = g.integers(-32, 32, size=(K, N)).astype(np.int8)
    s = (2.0 ** g.integers(-1, 3, size=N)).astype(np.float32)
    z = g.integers(-4, 5, size=N).astype(np.float32)
    arena, views, spans = _slices(order, {"a": M * K, "c": M * N, "s": N, "z": N})
    tensors = {"a": _dev(a).reshape(-1), "s": _dev(s), "z": _dev(z), "c": _nan(M * N)}
    for name in order:
        if name != "c":
            views[name].copy_(tensors[name])
        tensors[name] = arena[spans[name][0] :] if name == widen else views[name]
    before = arena.clone()
    _gemm8[(M // BM, N // BN)](
        tensors["a"], _dev(w), tensors["c"], tensors["s"], tensors["z"], M, N, K, K, 1, N, 1, N, 1, BM=BM, BN=BN, BK=BK
    )
    torch.mps.synchronize()
    want = a.astype(np.float64) @ ((w.astype(np.float64) - z[None, :]) * s[None, :])
    got = views["c"].cpu().numpy().astype(np.float64).reshape(M, N)
    np.testing.assert_array_equal(got, want)
    assert _outside(arena, before, spans["c"]) == 0
    assert route.quant == [True] and len(route.shaders) == 1


@requires_mps
@pytest.mark.parametrize("operand", ["a", "s", "z"])
def test_canonical_gemm_refuses_an_operand_that_shares_one_cell_with_the_output(route, operand):
    M, N, K, BM, BN, BK = 64, 48, 96, 32, 16, 32
    g = np.random.default_rng(12)
    tensors = {
        "a": _dev(g.integers(-4, 5, size=M * K).astype(np.float32)),
        "s": _dev((2.0 ** g.integers(-1, 3, size=N)).astype(np.float32)),
        "z": _dev(g.integers(-4, 5, size=N).astype(np.float32)),
    }
    w = _dev(g.integers(-32, 32, size=(K, N)).astype(np.int8))
    count = tensors[operand].numel()
    arena = torch.full((count + M * N - 1,), _POISON, device="mps")
    arena[:count].copy_(tensors[operand])
    tensors[operand], out = arena[:count], arena[count - 1 :]
    before = arena.clone()
    with pytest.raises(MetalNonRecoverableError, match="quantized.*overlaps one of its operands"):
        _gemm8[(M // BM, N // BN)](
            tensors["a"], w, out, tensors["s"], tensors["z"], M, N, K, K, 1, N, 1, N, 1, BM=BM, BN=BN, BK=BK
        )
    assert torch.equal(arena, before)
    assert route.quant == [False] and route.shaders == []


# ---------------------------------------------------------------------- dispatcher (unit)
class _FakeRuntime:
    def __init__(self):
        self.calls = []
        self.unsupported = set()

    def is_unsupported(self, msl):
        return msl in self.unsupported

    def mark_unsupported(self, msl):
        self.unsupported.add(msl)

    def get_library(self, msl):
        return msl

    def dispatch(self, lib, kernel_name, args, *, threads, group_size):
        self.calls.append((lib, kernel_name, [a for a in args if isinstance(a, int)], threads, group_size))


def _host_gemv(N, K, *, rows, xlen, vlen):
    return [
        torch.zeros(xlen),
        torch.zeros(rows, K, dtype=torch.int8),
        torch.zeros(vlen),
        torch.ones(vlen),
        torch.zeros(vlen),
        N,
        K,
        K,
    ]


def test_dispatcher_rounds_a_gemv_to_whole_blocks_only_inside_the_storages():
    from triton_msl.autotuning._quant_matmul_dispatch import dispatch_quant_matmul

    descriptor = ("gemv", "FAST", 0, 1, 2, 3, 4, 5, 6, ((7, 6),), 32, 32, "SCALAR")
    # N tail, whole K, slack: the fast template with N rounded to the source's blocks.
    rt = _FakeRuntime()
    assert dispatch_quant_matmul(rt, descriptor, _host_gemv(40, 128, rows=64, xlen=128, vlen=64), grid=(2, 1, 1))
    assert rt.calls == [("FAST", "int8_gemv", [128, 64], 2048, 256)]
    # K tail, slack: the scalar template as a one-row GEMM over the rounded N x K.
    rt = _FakeRuntime()
    assert dispatch_quant_matmul(rt, descriptor, _host_gemv(40, 100, rows=65, xlen=128, vlen=64), grid=(2, 1, 1))
    assert rt.calls == [("SCALAR", "int8_matmul_pergroup", [1, 64, 128, 0, 1, 1, 100, 0, 1, 0, 1, 0, 1], 256, 256)]
    # The last weight row ends before the rounded K (64 rows of 100): outside -> clipped.
    rt = _FakeRuntime()
    assert dispatch_quant_matmul(rt, descriptor, _host_gemv(40, 100, rows=64, xlen=128, vlen=64), grid=(2, 1, 1))
    assert rt.calls == [("FAST", "int8_gemv", [100, 40], 1280, 256)]
    # A descriptor without the source K block / scalar template (an older cache entry):
    # K is taken as whole, so only N is rounded.
    rt = _FakeRuntime()
    assert dispatch_quant_matmul(rt, descriptor[:11], _host_gemv(40, 100, rows=65, xlen=128, vlen=64), grid=(2, 1, 1))
    assert rt.calls == [("FAST", "int8_gemv", [100, 64], 2048, 256)]


def test_dispatcher_declines_a_tensor_shorter_than_the_sizes_passed():
    from triton_msl.autotuning._quant_matmul_dispatch import dispatch_quant_matmul

    descriptor = ("gemv", "FAST", 0, 1, 2, 3, 4, 5, 6, ((7, 6),), 32, 32, "SCALAR")
    # Whole blocks, an input of 100 cells for K = 128; a tail whose clipped cells do not fit.
    for kargs in (_host_gemv(64, 128, rows=64, xlen=100, vlen=64), _host_gemv(40, 100, rows=40, xlen=100, vlen=39)):
        rt, reasons = _FakeRuntime(), []
        assert not dispatch_quant_matmul(rt, descriptor, kargs, grid=(2, 1, 1), decline=reasons)
        assert rt.calls == [] and len(reasons) == 1 and "storage is smaller" in reasons[0]
    # A view that is shorter than N inside a storage that holds the cells is addressed
    # through its pointer, as the kernel addresses it: it runs.
    kargs = _host_gemv(64, 128, rows=64, xlen=128, vlen=64)
    kargs[2] = torch.zeros(96)[8:40]
    rt = _FakeRuntime()
    assert dispatch_quant_matmul(rt, descriptor, kargs, grid=(2, 1, 1))
    assert rt.calls == [("FAST", "int8_gemv", [128, 64], 2048, 256)]


def test_dispatcher_names_a_missing_scalar_template_and_keeps_the_fast_one_usable():
    from triton_msl.autotuning._quant_matmul_dispatch import dispatch_quant_matmul

    descriptor = ("gemv", "FAST", 0, 1, 2, 3, 4, 5, 6, ((7, 6),), 32, 32, None)
    rt, reasons = _FakeRuntime(), []
    kargs = _host_gemv(40, 100, rows=65, xlen=128, vlen=64)
    assert not dispatch_quant_matmul(rt, descriptor, kargs, grid=(2, 1, 1), decline=reasons)
    assert rt.calls == [] and len(reasons) == 1 and "scalar template" in reasons[0]
    assert not rt.is_unsupported("FAST")
