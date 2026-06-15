"""Compile-time detector: eligible matmuls emit a fast_matmul descriptor in
cached metadata; ineligible ones do not. Inspects ~/.cache/triton_metal/*.meta.json
(the descriptor round-trips through the JSON cache as a list of str+ints).
Serial GPU.

NOTE: The test kernels use stride_* named args so that _detect_simple_dot()
rejects them (has_strides=True) and the kernel routes through
_lower_dot_via_prebuilt_template -> _lower_dot_simple_template, which is the
path where the fast_matmul descriptor is recorded.
"""
import os, glob, json, shutil, pytest
try:
    import torch, triton, triton.language as tl
    HAS = torch.backends.mps.is_available()
except Exception:
    HAS = False
requires = pytest.mark.skipif(not HAS, reason="MPS needed")

CACHE = os.path.expanduser("~/.cache/triton_metal")


@triton.jit
def _mm_fp32(a_ptr, b_ptr, c_ptr, M, N, K,
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """fp32 in -> fp32 out (eligible for fast template).

    Uses stride_* arg names so that _detect_simple_dot rejects it (has_strides=True)
    and it routes through _lower_dot_via_prebuilt_template -> _lower_dot_simple_template.
    """
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * stride_am + offk[None, :] * stride_ak)
    b_ptrs = b_ptr + (offk[:, None] * stride_bk + offn[None, :] * stride_bn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * stride_ak; b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + (offm[:, None] * stride_cm + offn[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


@triton.jit
def _mm_fp16_out(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """fp16 in -> fp16 out (NOT eligible: fast template is float* output only)."""
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * stride_am + offk[None, :] * stride_ak)
    b_ptrs = b_ptr + (offk[:, None] * stride_bk + offn[None, :] * stride_bn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * stride_ak; b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + (offm[:, None] * stride_cm + offn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16))


def _descriptors():
    out = []
    for p in glob.glob(os.path.join(CACHE, "*.meta.json")):
        with open(p) as f:
            m = json.load(f)
        if m.get("fast_matmul"):
            out.append(m["fast_matmul"])
    return out


def _run(kernel, A, B, C, M, N, K):
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    kernel[grid](A, B, C, M, N, K,
                 A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                 BM=64, BN=64, BK=32)
    torch.mps.synchronize()


@requires
def test_eligible_fp32_emits_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_mm_fp32, A, B, C, M, N, K)            # fp32 in, fp32 out (no cast)
    descs = _descriptors()
    assert descs, "expected a fast_matmul descriptor for an eligible fp32 matmul"
    msl, m_idx, n_idx, k_idx, tile_m, tile_n = descs[0]
    assert (m_idx, n_idx, k_idx, tile_m, tile_n) == (3, 4, 5, 32, 128)
    assert "simdgroup_matmul_fast" in msl


@requires
def test_fp16_output_no_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)   # fp16 OUTPUT -> must NOT use float* template
    _run(_mm_fp16_out, A, B, C, M, N, K)
    assert not _descriptors(), "fp16-output matmul must not emit a fast_matmul descriptor"


@requires
def test_flag_off_no_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_METAL_FAST_MATMUL", "0")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_mm_fp32, A, B, C, M, N, K)
    assert not _descriptors(), "flag off must emit no descriptor"
