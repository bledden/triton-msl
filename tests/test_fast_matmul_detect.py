"""Compile-time detector: eligible matmuls emit a fast_matmul descriptor in
cached metadata; ineligible ones do not. Inspects ~/.cache/triton_msl/*.meta.json
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

CACHE = os.path.expanduser("~/.cache/triton_msl")


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
    """fp16 in -> fp16 out (fast template now supports half* C via cast epilogue)."""
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


@triton.jit
def _mm_bf16_out(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """bf16 in -> bf16 out (eligible: simdgroup_bfloat8x8 input + bf16 cast-epilogue out)."""
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * stride_am + offk[None, :] * stride_ak)
    b_ptrs = b_ptr + (offk[:, None] * stride_bk + offn[None, :] * stride_bn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * stride_ak; b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + (offm[:, None] * stride_cm + offn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16))


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
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_mm_fp32, A, B, C, M, N, K)            # fp32 in, fp32 out (no cast)
    descs = _descriptors()
    assert descs, "expected a fast_matmul descriptor for an eligible fp32 matmul"
    desc = descs[0]
    msl, m_idx, n_idx, k_idx, tile_m, tile_n = desc[0], desc[1], desc[2], desc[3], desc[4], desc[5]
    assert (m_idx, n_idx, k_idx, tile_m, tile_n) == (3, 4, 5, 32, 128)
    assert "simdgroup_matmul_fast" in msl
    # Finding 1 coverage: descriptor must be 8 elements with msl_dtype / msl_out fields
    assert len(desc) == 8, (
        f"descriptor must be 8 elements (msl,m_idx,n_idx,k_idx,tile_m,tile_n,msl_dtype,msl_out); "
        f"got {len(desc)}"
    )
    assert desc[6] == "fp32", f"desc[6] (msl_dtype) must be 'fp32' for fp32 input, got {desc[6]!r}"
    assert desc[7] == "fp32", f"desc[7] (msl_out) must be 'fp32' for fp32 output, got {desc[7]!r}"


@requires
def test_fp16_output_emits_half_variant_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    B = torch.randn(K, N, device="mps", dtype=torch.float16)
    C = torch.empty(M, N, device="mps", dtype=torch.float16)   # fp16 OUTPUT
    _run(_mm_fp16_out, A, B, C, M, N, K)
    descs = _descriptors()
    assert descs, "fp16-output matmul must now emit a fast_matmul descriptor"
    desc = descs[0]
    msl, m_idx, n_idx, k_idx, tile_m, tile_n = desc[0], desc[1], desc[2], desc[3], desc[4], desc[5]
    assert (m_idx, n_idx, k_idx, tile_m, tile_n) == (3, 4, 5, 32, 128)
    assert "device half* C [[buffer(2)]]" in msl          # the fp16-output variant
    assert "half(scratch[sgitg*64u + i])" in msl
    # Finding 1 coverage: descriptor must be 8 elements with msl_dtype / msl_out fields
    assert len(desc) == 8, (
        f"descriptor must be 8 elements; got {len(desc)}"
    )
    assert desc[6] in ("fp16", "f16"), (
        f"desc[6] (msl_dtype) must be a fp16 dtype string for fp16 input, got {desc[6]!r}"
    )
    assert desc[7] in ("fp16", "f16"), (
        f"desc[7] (msl_out) must be a fp16 dtype string for fp16 output, got {desc[7]!r}"
    )


@requires
def test_bf16_emits_descriptor_and_computes(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="mps", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="mps", dtype=torch.bfloat16)
    # bf16 is now a fast-matmul input (M-series simdgroup_bfloat8x8): bf16 in / bf16
    # out must emit a descriptor AND compute correctly (within bf16 tolerance).
    _run(_mm_bf16_out, A, B, C, M, N, K)
    assert _descriptors(), "bf16 matmul must emit a fast_matmul descriptor"
    torch.testing.assert_close(C.float(), A.float() @ B.float(), rtol=3e-2, atol=3e-2)


@requires
def test_flag_off_no_descriptor(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_mm_fp32, A, B, C, M, N, K)
    assert not _descriptors(), "flag off must emit no descriptor"


@triton.jit
def _mm_abbrev(a_ptr, b_ptr, c_ptr, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """fp32 in -> fp32 out with ABBREVIATED stride-arg names (sam, sak, ...).

    These names do NOT contain 'stride', so _detect_simple_dot does NOT reject
    them on the has_strides heuristic. The kernel therefore routes through
    _lower_simple_dot_inline (the _detect_simple_dot path), which is the second
    bare-matmul lowering path. This test verifies that the new call site at the
    top of _lower_simple_dot_inline fires the detector and emits the descriptor.
    """
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN); offk = tl.arange(0, BK)
    a_ptrs = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b_ptrs = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = c_ptr + (offm[:, None] * scm + offn[None, :] * scn)
    tl.store(c_ptrs, acc)


@requires
def test_abbreviated_name_emits_descriptor(monkeypatch):
    """Abbreviated stride-arg names route through _lower_simple_dot_inline.

    The new call site at the top of that method must emit the fast_matmul
    descriptor with indices (3, 4, 5, 32, 128) — same contract as the
    stride_* path through _lower_dot_simple_template.
    """
    shutil.rmtree(CACHE, ignore_errors=True)
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "1")
    M = N = K = 256
    A = torch.randn(M, K, device="mps"); B = torch.randn(K, N, device="mps"); C = torch.empty(M, N, device="mps")
    _run(_mm_abbrev, A, B, C, M, N, K)
    descs = _descriptors()
    assert descs, (
        "expected a fast_matmul descriptor for an abbreviated-name fp32 matmul "
        "(routes through _lower_simple_dot_inline, not _lower_dot_simple_template)"
    )
    desc = descs[0]
    msl, m_idx, n_idx, k_idx, tile_m, tile_n = desc[0], desc[1], desc[2], desc[3], desc[4], desc[5]
    assert (m_idx, n_idx, k_idx, tile_m, tile_n) == (3, 4, 5, 32, 128)
    assert "simdgroup_matmul_fast" in msl
