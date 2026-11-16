"""Round-3 pins for the rank-3 / batched ``tt.dot`` recovery.

Triton keeps the leading dimension in the dot/result types but folds away a
size-one batch address term.  For B>1 it emits that term explicitly.  The
lowering must prove the batch/row/K/column address grammar and the complete
load -> dot -> store value path; it may not treat every rank-3 dot as a stack
of independent 2-D matmuls.
"""

import os
import re

import numpy as np
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False


requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")


@pytest.fixture()
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    os.makedirs(tmp_path / "msl", exist_ok=True)
    if HAS:
        for fn in (
            _dot3d_b1,
            _dot3d_b1_plus_one,
            _dot3d_b1_shifted_a,
            _dot3d_batched,
            _dot3d_batched_nonzero_acc,
            _dot3d_output_first,
            _dot_multidim_flat,
            _dot_multidim_flat_plus_one,
            _dot_multidim_flat_shifted_x,
        ):
            cache = getattr(fn, "device_caches", None)
            assert cache is not None
            cache.clear()
    return tmp_path


if HAS:

    @triton.jit
    def _dot3d_b1(
        A,
        B,
        C,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ib = tl.arange(0, 1)
        im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        ap = A + ib[:, None, None] * stride_ab + im[None, :, None] * stride_am + kk[None, None, :] * stride_ak
        bp = B + ib[:, None, None] * stride_bb + kk[None, :, None] * stride_bk + jn[None, None, :] * stride_bn
        cp = C + ib[:, None, None] * stride_cb + im[None, :, None] * stride_cm + jn[None, None, :] * stride_cn
        tl.store(cp, tl.dot(tl.load(ap), tl.load(bp)))

    @triton.jit
    def _dot3d_b1_plus_one(
        A,
        B,
        C,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ib = tl.arange(0, 1)
        im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        ap = A + ib[:, None, None] * stride_ab + im[None, :, None] * stride_am + kk[None, None, :] * stride_ak
        bp = B + ib[:, None, None] * stride_bb + kk[None, :, None] * stride_bk + jn[None, None, :] * stride_bn
        cp = C + ib[:, None, None] * stride_cb + im[None, :, None] * stride_cm + jn[None, None, :] * stride_cn
        tl.store(cp, tl.dot(tl.load(ap), tl.load(bp)) + 1.0)

    @triton.jit
    def _dot3d_b1_shifted_a(
        A,
        B,
        C,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ib = tl.arange(0, 1)
        im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        ap = A + 1 + ib[:, None, None] * stride_ab + im[None, :, None] * stride_am + kk[None, None, :] * stride_ak
        bp = B + ib[:, None, None] * stride_bb + kk[None, :, None] * stride_bk + jn[None, None, :] * stride_bn
        cp = C + ib[:, None, None] * stride_cb + im[None, :, None] * stride_cm + jn[None, None, :] * stride_cn
        tl.store(cp, tl.dot(tl.load(ap), tl.load(bp)))

    @triton.jit
    def _dot3d_batched(
        A,
        B,
        C,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BATCH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ib = tl.arange(0, BATCH)
        im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        ap = A + ib[:, None, None] * stride_ab + im[None, :, None] * stride_am + kk[None, None, :] * stride_ak
        bp = B + ib[:, None, None] * stride_bb + kk[None, :, None] * stride_bk + jn[None, None, :] * stride_bn
        cp = C + ib[:, None, None] * stride_cb + im[None, :, None] * stride_cm + jn[None, None, :] * stride_cn
        tl.store(cp, tl.dot(tl.load(ap), tl.load(bp)))

    @triton.jit
    def _dot3d_batched_nonzero_acc(
        A,
        B,
        C,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BATCH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ib = tl.arange(0, BATCH)
        im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        ap = A + ib[:, None, None] * stride_ab + im[None, :, None] * stride_am + kk[None, None, :] * stride_ak
        bp = B + ib[:, None, None] * stride_bb + kk[None, :, None] * stride_bk + jn[None, None, :] * stride_bn
        cp = C + ib[:, None, None] * stride_cb + im[None, :, None] * stride_cm + jn[None, None, :] * stride_cn
        acc = tl.full((BATCH, BLOCK_M, BLOCK_N), 1.0, tl.float32)
        tl.store(cp, tl.dot(tl.load(ap), tl.load(bp), acc))

    @triton.jit
    def _dot3d_output_first(
        C,
        A,
        B,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BATCH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ib = tl.arange(0, BATCH)
        im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        ap = A + ib[:, None, None] * stride_ab + im[None, :, None] * stride_am + kk[None, None, :] * stride_ak
        bp = B + ib[:, None, None] * stride_bb + kk[None, :, None] * stride_bk + jn[None, None, :] * stride_bn
        cp = C + ib[:, None, None] * stride_cb + im[None, :, None] * stride_cm + jn[None, None, :] * stride_cn
        tl.store(cp, tl.dot(tl.load(ap), tl.load(bp)))

    @triton.jit
    def _dot_multidim_flat(X, Y, Z, RANK: tl.constexpr, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
        shape: tl.constexpr = [2] * (RANK - 2) + [32, 32]
        x = tl.load(X + tl.arange(0, 256 << RANK)).reshape(shape)
        y = tl.load(Y + tl.arange(0, 256 << RANK)).reshape(shape)
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(Z + tl.arange(0, 256 << RANK), z.reshape([256 << RANK]))

    @triton.jit
    def _dot_multidim_flat_plus_one(
        X,
        Y,
        Z,
        RANK: tl.constexpr,
        TRANS_A: tl.constexpr,
        TRANS_B: tl.constexpr,
    ):
        shape: tl.constexpr = [2] * (RANK - 2) + [32, 32]
        x = tl.load(X + tl.arange(0, 256 << RANK)).reshape(shape)
        y = tl.load(Y + tl.arange(0, 256 << RANK)).reshape(shape)
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y) + 1.0
        tl.store(Z + tl.arange(0, 256 << RANK), z.reshape([256 << RANK]))

    @triton.jit
    def _dot_multidim_flat_shifted_x(
        X,
        Y,
        Z,
        RANK: tl.constexpr,
        TRANS_A: tl.constexpr,
        TRANS_B: tl.constexpr,
    ):
        shape: tl.constexpr = [2] * (RANK - 2) + [32, 32]
        x = tl.load(X + 1 + tl.arange(0, 256 << RANK)).reshape(shape)
        y = tl.load(Y + tl.arange(0, 256 << RANK)).reshape(shape)
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(Z + tl.arange(0, 256 << RANK), z.reshape([256 << RANK]))


def _lowerer(
    fn=_dot3d_b1,
    *,
    batch=1,
    k=32,
    input_type="fp32",
    output_type="fp32",
    stride_type="i32",
):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    signature = {
        "A": f"*{input_type}",
        "B": f"*{input_type}",
        "C": f"*{output_type}",
        "stride_ab": stride_type,
        "stride_am": stride_type,
        "stride_ak": stride_type,
        "stride_bb": stride_type,
        "stride_bk": stride_type,
        "stride_bn": stride_type,
        "stride_cb": stride_type,
        "stride_cm": stride_type,
        "stride_cn": stride_type,
    }
    constexprs = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": k}
    if fn in (_dot3d_batched, _dot3d_batched_nonzero_acc, _dot3d_output_first):
        constexprs["BATCH"] = batch
    src = ASTSource(
        fn=fn,
        signature=signature,
        constexprs=constexprs,
    )
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


def _flat_lowerer(*, rank, ta=False, tb=False, fn=_dot_multidim_flat):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    src = ASTSource(
        fn=fn,
        signature={"X": "*bf16", "Y": "*bf16", "Z": "*fp32"},
        constexprs={"RANK": rank, "TRANS_A": int(ta), "TRANS_B": int(tb)},
    )
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


@requires
def test_b1_lowering_replays_rank3_address_contract():
    lowerer = _lowerer()
    msl = lowerer.lower()
    assert "round3_batched_dot" in msl
    assert "for (uint batch = 0u; batch < 1u; batch++)" in msl
    assert "pid3.x * 32u" in msl
    assert "pid3.y * 32u" in msl
    assert lowerer.effective_block_size == 128
    assert lowerer._used_pid_axes == {0, 1}


@requires
@pytest.mark.parametrize("flattened", [False, True])
def test_batched_shape_proof_uses_native_metadata(flattened):
    """Printed operation shapes cannot change the proved batched plan."""
    lowerer = _flat_lowerer(rank=6, ta=True, tb=True) if flattened else _lowerer(_dot3d_batched, batch=2)
    expected = lowerer._detect_batched_dot()
    assert expected is not None
    expected_msl = lowerer._lower_batched_dot_template(expected)
    changed = 0
    for op in lowerer.graph.ops:
        if op.op in {"tt.dot", "tt.reshape", "tt.trans", "tt.make_range", "tt.load", "tt.expand_dims"}:
            op.type_str = "tensor<1x999xf32>"
            changed += 1
    assert changed >= 4
    actual = lowerer._detect_batched_dot()
    assert actual == expected
    assert lowerer._lower_batched_dot_template(actual) == expected_msl


@requires
@pytest.mark.parametrize("flattened", [False, True])
def test_batched_shape_proof_refuses_missing_or_dynamic_native_metadata(flattened):
    """The recognizer itself must reject damaged facts before template emission."""
    from dataclasses import replace

    for damage in ("missing", "dynamic", "false_scalar"):
        lowerer = _flat_lowerer(rank=6) if flattened else _lowerer(_dot3d_batched, batch=2)
        assert lowerer._detect_batched_dot() is not None
        dot = next(op for op in lowerer.graph.ops if op.op == "tt.dot")
        meta = lowerer.graph.result_meta[dot.id]
        if damage == "missing":
            lowerer.graph.result_meta.pop(dot.id)
        elif damage == "dynamic":
            lowerer.graph.result_meta[dot.id] = replace(meta, type=replace(meta.type, shape=(None, 32, 32)))
        else:
            lowerer.graph.result_meta[dot.id] = replace(meta, type=replace(meta.type, is_tensor=False, shape=()))
        with pytest.raises(MetalNonRecoverableError, match="native.*(missing|shape|tensor-kind)"):
            lowerer._detect_batched_dot()


@requires
@pytest.mark.parametrize("flattened", [False, True])
def test_batched_dtype_proof_and_emitter_ignore_legacy_fields(flattened):
    """Admission and emission must use the same native dtype authority."""
    lowerer = _flat_lowerer(rank=6, ta=True, tb=True) if flattened else _lowerer(_dot3d_batched, batch=2)
    expected = lowerer._detect_batched_dot()
    assert expected is not None
    expected_msl = lowerer._lower_batched_dot_template(expected)
    for value in [*lowerer.graph.args, *lowerer.graph.ops]:
        value.elem_type = "i64"
    actual = lowerer._detect_batched_dot()
    assert actual == expected
    assert lowerer._lower_batched_dot_template(actual) == expected_msl


@requires
@pytest.mark.parametrize("flattened", [False, True])
def test_batched_dtype_proof_rejects_contradictory_native_facts(flattened):
    """Pointer pointees and result widths cannot inherit legacy dtype defaults."""
    from dataclasses import replace

    for target in ("input", "output", "dot", "load"):
        lowerer = _flat_lowerer(rank=6) if flattened else _lowerer(_dot3d_batched, batch=2)
        plan = lowerer._detect_batched_dot()
        assert plan is not None
        if target in {"input", "output"}:
            value = plan["a_arg" if target == "input" else "c_arg"]
        else:
            value = next(op for op in lowerer.graph.ops if op.op == ("tt.dot" if target == "dot" else "tt.load"))
        meta = lowerer.graph.result_meta[value.id]
        facts = meta.type
        if facts.kind == "pointer":
            broken = replace(facts, pointee=replace(facts.pointee, width=7))
        else:
            broken = replace(facts, width=7)
        lowerer.graph.result_meta[value.id] = replace(meta, type=broken)
        with pytest.raises(MetalNonRecoverableError, match="batched.dot.*dtype"):
            lowerer._detect_batched_dot()


@requires
def test_b2_lowering_replays_batch_stride_and_uniform_barriers():
    lowerer = _lowerer(_dot3d_batched, batch=2)
    msl = lowerer.lower()
    assert "for (uint batch = 0u; batch < 2u; batch++)" in msl
    # Preserve the source's signed stride semantics while widening the
    # coordinate product.  Casting a negative stride to uint silently wraps.
    assert "batch * (long)_mmarg3" in msl
    assert "batch * (long)_mmarg6" in msl
    assert "batch * (long)_mmarg9" in msl
    assert not re.search(r"\(uint\)_mmarg(?:[3-9]|10|11)\b", msl)
    assert lowerer._batched_dot_bounds == (
        "batched_dot_host_bounds_v1",
        2,
        32,
        32,
        32,
        True,
        0,
        1,
        2,
        (("arg", 3), ("arg", 4), ("arg", 5)),
        (("arg", 6), ("arg", 7), ("arg", 8)),
        (("arg", 9), ("arg", 10), ("arg", 11)),
    )
    # Two barriers delimit the reusable K-stage and each of four accumulator
    # tiles has a store/copy barrier pair.  All ten statements are outside
    # conditionals and execute uniformly for every batch.
    assert msl.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 10


@requires
def test_i64_runtime_strides_stay_fail_closed():
    # The template ABI currently materializes runtime scalars as MSL int.
    # Admitting an i64 stride would truncate before address replay.
    with pytest.raises(MetalNonRecoverableError):
        _lowerer(_dot3d_batched, batch=2, stride_type="i64").lower()


@requires
@pytest.mark.parametrize(("role", "stride_index"), [("A", 3), ("B", 6), ("C", 9)])
def test_host_bounds_descriptor_checks_every_pointer_role(role, stride_index):
    from triton_msl.backend.driver import _batched_dot_host_bounds_reason

    lowerer = _lowerer(_dot3d_batched, batch=2)
    lowerer.lower()
    a = torch.empty((2, 32, 32))
    b = torch.empty((2, 32, 32))
    c = torch.empty((2, 32, 32))
    kargs = [a, b, c, *a.stride(), *b.stride(), *c.stride()]
    kargs[stride_index] = -abs(int(kargs[stride_index]))
    reason = _batched_dot_host_bounds_reason(lowerer._batched_dot_bounds, kargs, (1, 1, 1))
    assert reason is not None
    assert f"batched-dot {role} forms byte offset" in reason
    assert "before its backing storage" in reason


@requires
def test_int8_lowering_uses_exact_float_accumulation_envelope():
    lowerer = _lowerer(
        _dot3d_batched,
        batch=8,
        k=64,
        input_type="i8",
        output_type="i32",
    )
    msl = lowerer.lower()
    assert "round3_batched_dot" in msl
    assert "for (uint batch = 0u; batch < 8u; batch++)" in msl
    assert "threadgroup float tg_A[256]" in msl
    assert "threadgroup float tg_B[256]" in msl
    assert "device const char* _mmarg0" in msl
    assert "device int* _mmarg2" in msl
    assert "= int(tg_store[" in msl


@requires
def test_flattened_rank6_lowering_replays_inner_transposes_and_batch_loop():
    lowerer = _flat_lowerer(rank=6, ta=True, tb=True)
    msl = lowerer.lower()
    assert "round3_batched_dot" in msl
    assert "for (uint batch = 0u; batch < 16u; batch++)" in msl
    # One batch stride at each A/B staging site, plus one in each of the four
    # statically emitted output-tile copy loops.
    assert msl.count("batch * 1024l") == 6
    assert "device const bfloat* _mmarg0" in msl
    assert "device const bfloat* _mmarg1" in msl
    assert "device float* _mmarg2" in msl
    assert "pid3.x * 32u" not in msl
    assert "pid3.y * 32u" not in msl
    assert lowerer._used_pid_axes == set()
    assert lowerer._batched_dot_bounds == (
        "batched_dot_host_bounds_v1",
        16,
        32,
        32,
        32,
        False,
        0,
        1,
        2,
        (("literal", 1024), ("literal", 1), ("literal", 32)),
        (("literal", 1024), ("literal", 1), ("literal", 32)),
        (("literal", 1024), ("literal", 32), ("literal", 1)),
    )


@requires
@pytest.mark.parametrize("fn", [_dot_multidim_flat_plus_one, _dot_multidim_flat_shifted_x])
def test_flattened_multidim_near_misses_stay_fail_closed(fn):
    with pytest.raises(MetalNonRecoverableError):
        _flat_lowerer(rank=3, fn=fn).lower()


@requires
@pytest.mark.parametrize("fn", [_dot3d_b1_plus_one, _dot3d_b1_shifted_a])
def test_b1_near_misses_stay_fail_closed(fn):
    with pytest.raises(MetalNonRecoverableError):
        _lowerer(fn).lower()


@requires
def test_nonzero_rank3_dot_accumulator_stays_fail_closed():
    with pytest.raises(MetalNonRecoverableError):
        _lowerer(_dot3d_batched_nonzero_acc, batch=2).lower()


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_b1_gpu_matches_distinct_reference(monkeypatch, cold_gpu_caches, dtype):
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    hits = []
    original = getattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", None)
    assert original is not None

    def spy(self, info):
        hits.append(tuple(info["batch_dims"]))
        return original(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", spy)
    torch.manual_seed(913)
    a = torch.randn((1, 32, 32), device="mps", dtype=dtype)
    b = torch.randn((1, 32, 32), device="mps", dtype=dtype)
    c = torch.full((1, 32, 32), float("nan"), device="mps", dtype=torch.float32)
    _dot3d_b1[(1, 1)](
        a,
        b,
        c,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    ref = a.float() @ b.float()
    assert hits == [(1,)]
    assert not torch.isnan(c).any()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.parametrize(
    ("batch", "dtype"),
    [(2, torch.float32), (2, torch.float16), (4, torch.float16), (8, torch.float16)],
)
def test_batched_gpu_isolates_batches_and_preserves_canaries(monkeypatch, cold_gpu_caches, batch, dtype):
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    hits = []
    original = getattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", None)
    assert original is not None

    def spy(self, info):
        hits.append(tuple(info["batch_dims"]))
        return original(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", spy)
    torch.manual_seed(917 + batch)
    a = torch.randn((batch, 32, 32), device="mps", dtype=dtype)
    b = torch.randn((batch, 32, 32), device="mps", dtype=dtype)
    # Make accidental cross-batch aliasing obvious rather than relying on chance.
    for i in range(batch):
        a[i].mul_(i + 1)
        b[i].add_(3 * i)
    sentinel = 12345.0
    storage = torch.full((batch, 32, 40), sentinel, device="mps", dtype=torch.float32)
    c = storage[:, :, :32]
    _dot3d_batched[(1, 1)](
        a,
        b,
        c,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    ref = a.float() @ b.float()
    assert hits == [(batch,)]
    assert not torch.isnan(c).any()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    assert torch.all(storage[:, :, 32:] == sentinel)


@requires_gpu
@pytest.mark.parametrize("broadcast_a", [False, True])
def test_batched_gpu_replays_nontrivial_runtime_strides_and_broadcast(
    monkeypatch,
    cold_gpu_caches,
    broadcast_a,
):
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    hits = []
    original = generic_lowerer.GenericLowerer._lower_batched_dot_template

    def spy(self, info):
        hits.append((tuple(info["batch_dims"]), tuple(info["a_strides"]), tuple(info["b_strides"])))
        return original(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", spy)
    batch = 4
    torch.manual_seed(922)
    if broadcast_a:
        # Pass a zero batch stride explicitly while keeping the tensor itself
        # non-overlapping; the host driver's conservative input round-trip
        # cannot copy back into an expanded (overlapping) view.
        a = torch.randn((batch, 32, 32), device="mps")
        a_strides = (0, a.stride(1), a.stride(2))
        ref_a = a[:1].expand(batch, -1, -1)
    else:
        a_storage = torch.randn((batch, 35, 37), device="mps")
        a = a_storage[:, :32, :32]
        a_strides = a.stride()
        ref_a = a
    b_storage = torch.randn((batch, 36, 41), device="mps")
    b = b_storage[:, :32, :32]
    sentinel = 12345.0
    c_storage = torch.full((batch, 35, 40), sentinel, device="mps")
    c = c_storage[:, :32, :32]
    _dot3d_batched[(1, 1)](
        a,
        b,
        c,
        *a_strides,
        *b.stride(),
        *c.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert hits == [((batch,), ("stride_ab", "stride_am", "1"), ("stride_bb", "stride_bk", "1"))]
    torch.testing.assert_close(c, ref_a @ b, rtol=1e-3, atol=1e-3)
    assert torch.all(c_storage[:, :32, 32:] == sentinel)
    assert torch.all(c_storage[:, 32:, :] == sentinel)


@requires_gpu
@pytest.mark.parametrize("address_case", ["negative-batch", "negative-row", "past-batch"])
def test_batched_host_roundtrip_replays_offsets_outside_view_inside_storage(
    monkeypatch,
    cold_gpu_caches,
    address_case,
):
    """The 2-D host launcher preserves valid addresses outside a logical view."""
    import triton_msl.backend.driver as driver
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    route_hits = []
    original_lower = generic_lowerer.GenericLowerer._lower_batched_dot_template

    def lower_spy(self, info):
        route_hits.append(tuple(info["batch_dims"]))
        return original_lower(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", lower_spy)
    launches = []
    utils = driver._get_utils()
    original_launch = utils.launch

    def launch_spy(*args, **kwargs):
        launches.append(True)
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", launch_spy)

    batch = 2
    torch.manual_seed({"negative-batch": 925, "negative-row": 926, "past-batch": 928}[address_case])
    storage_batch = batch + 1 if address_case == "past-batch" else batch
    a_storage = torch.randn((storage_batch, 32, 32), device="mps")
    if address_case == "negative-batch":
        a = a_storage[1:]
        a_strides = (-a_storage.stride(0), a_storage.stride(1), a_storage.stride(2))
        ref_a = torch.stack((a_storage[1], a_storage[0]))
    elif address_case == "negative-row":
        a = a_storage[:, 31:, :]
        a_strides = (a_storage.stride(0), -a_storage.stride(1), a_storage.stride(2))
        ref_a = torch.flip(a_storage, (1,))
    else:
        # The second logical batch reads storage batch 2: valid in the
        # underlying allocation, but outside the logical [:2] view.
        a = a_storage[:batch]
        a_strides = (2 * a_storage.stride(0), a_storage.stride(1), a_storage.stride(2))
        ref_a = torch.stack((a_storage[0], a_storage[2]))
    b = torch.randn((batch, 32, 32), device="mps")
    sentinel = 12345.0
    c = torch.full((batch, 32, 32), sentinel, device="mps")

    _dot3d_batched[(1, 1)](
        a,
        b,
        c,
        *a_strides,
        *b.stride(),
        *c.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert route_hits == [(batch,)]
    assert launches == [True]
    torch.testing.assert_close(c, ref_a @ b, rtol=1e-3, atol=1e-3)


@requires_gpu
def test_batched_host_roundtrip_accepts_positive_offset_view(
    monkeypatch,
    cold_gpu_caches,
):
    """An offset view is safe when every replayed offset stays in its mirror."""
    import triton_msl.backend.driver as driver
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    route_hits = []
    original_lower = generic_lowerer.GenericLowerer._lower_batched_dot_template

    def lower_spy(self, info):
        route_hits.append(tuple(info["batch_dims"]))
        return original_lower(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", lower_spy)
    launches = []
    utils = driver._get_utils()
    original_launch = utils.launch

    def launch_spy(*args, **kwargs):
        launches.append(True)
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", launch_spy)

    batch = 2
    torch.manual_seed(927)
    a = torch.randn((batch + 1, 32, 32), device="mps")[1:]
    b = torch.randn((batch, 32, 32), device="mps")
    c = torch.full((batch, 32, 32), float("nan"), device="mps")
    _dot3d_batched[(1, 1)](
        a,
        b,
        c,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert route_hits == [(batch,)]
    assert launches == [True]
    torch.testing.assert_close(c, a @ b, rtol=1e-3, atol=1e-3)


@requires_gpu
def test_batched_host_bounds_rechecks_runtime_strides_after_warm_compile(
    monkeypatch,
    cold_gpu_caches,
):
    """A cached launcher must validate each call's runtime stride values."""
    import triton_msl.backend.driver as driver

    monkeypatch.delenv("TRITON_ALWAYS_COMPILE", raising=False)
    launches = []
    utils = driver._get_utils()
    original_launch = utils.launch

    def launch_spy(*args, **kwargs):
        launches.append(True)
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", launch_spy)

    batch = 2
    torch.manual_seed(930)
    a_safe = torch.randn((batch, 32, 32), device="mps")
    b = torch.randn((batch, 32, 32), device="mps")
    c_safe = torch.full((batch, 32, 32), float("nan"), device="mps")
    _dot3d_batched[(1, 1)](
        a_safe,
        b,
        c_safe,
        *a_safe.stride(),
        *b.stride(),
        *c_safe.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    torch.testing.assert_close(c_safe, a_safe @ b, rtol=1e-3, atol=1e-3)
    assert launches == [True]

    # Same specialization and cached launcher, but this call's stride reaches
    # before the logical view base while remaining in the backing allocation.
    # The descriptor must retain runtime stride references rather than freezing
    # the safe first call's values, and the marshaller must bind the offset base.
    a_storage = torch.randn((batch + 1, 32, 32), device="mps")
    a_unsafe = a_storage[1:]
    a_strides = (-a_storage.stride(0), a_storage.stride(1), a_storage.stride(2))
    sentinel = 12345.0
    c_unsafe = torch.full((batch, 32, 32), sentinel, device="mps")
    _dot3d_batched[(1, 1)](
        a_unsafe,
        b,
        c_unsafe,
        *a_strides,
        *b.stride(),
        *c_unsafe.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert launches == [True, True]
    ref_a = torch.stack((a_storage[1], a_storage[0]))
    torch.testing.assert_close(c_unsafe, ref_a @ b, rtol=1e-3, atol=1e-3)


@requires_gpu
def test_flattened_multidim_host_roundtrip_bounds_control(
    monkeypatch,
    cold_gpu_caches,
):
    """The fixed-stride descriptor also permits a safe host-path fallback."""
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    rank = 3
    shape = (2, 32, 32)
    torch.manual_seed(929)
    a = torch.randint(-4, 5, shape, dtype=torch.bfloat16, device="mps")
    b = torch.randint(-4, 5, shape, dtype=torch.bfloat16, device="mps")
    c = torch.full(shape, float("nan"), dtype=torch.float32, device="mps")
    _dot_multidim_flat[(1,)](a, b, c, rank, False, False)
    torch.mps.synchronize()
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-3, atol=1e-2)


@requires_gpu
def test_batched_gpu_resolves_output_first_pointer_roles(monkeypatch, cold_gpu_caches):
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    roles = []
    original = generic_lowerer.GenericLowerer._lower_batched_dot_template

    def spy(self, info):
        roles.append((info["a_arg"].name, info["b_arg"].name, info["c_arg"].name))
        return original(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", spy)
    torch.manual_seed(923)
    a = torch.randn((2, 32, 32), device="mps")
    b = torch.randn((2, 32, 32), device="mps")
    c = torch.full((2, 32, 32), float("nan"), device="mps")
    _dot3d_output_first[(1, 1)](
        c,
        a,
        b,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BATCH=2,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert roles == [("A", "B", "C")]
    torch.testing.assert_close(c, a @ b, rtol=1e-3, atol=1e-3)


@requires_gpu
@pytest.mark.parametrize(("batch", "k"), [(1, 64), (8, 32)])
def test_int8_batched_gpu_is_bit_exact_at_worst_products(monkeypatch, cold_gpu_caches, batch, k):
    """The float MMA path is exact here: each int8 product is exactly representable,
    and K<=64 keeps every partial sum below 2**24.  Exercise both signs and demand
    bit equality, rather than borrowing the floating-point test tolerance.
    """
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    hits = []
    original = getattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", None)
    assert original is not None

    def spy(self, info):
        hits.append((tuple(info["batch_dims"]), info["K"], info["a_arg"].elem_type))
        return original(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", spy)
    a = torch.empty((batch, 32, k), device="mps", dtype=torch.int8)
    b = torch.empty((batch, k, 32), device="mps", dtype=torch.int8)
    for i in range(batch):
        a[i].fill_(-128 if i % 2 == 0 else 127)
        b[i].fill_(-128 if i % 3 == 0 else 127)
    sentinel = -(2**30)
    storage = torch.full((batch, 32, 40), sentinel, device="mps", dtype=torch.int32)
    c = storage[:, :, :32]
    _dot3d_batched[(1, 1)](
        a,
        b,
        c,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BATCH=batch,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=k,
        num_warps=4,
    )
    torch.mps.synchronize()
    ref_np = np.matmul(a.cpu().numpy().astype(np.int32), b.cpu().numpy().astype(np.int32))
    assert hits == [((batch,), k, "i8")]
    assert np.array_equal(c.cpu().numpy(), ref_np)
    assert torch.all(storage[:, :, 32:] == sentinel)


@requires_gpu
@pytest.mark.parametrize("rank", [3, 4, 5, 6])
@pytest.mark.parametrize("ta", [False, True])
@pytest.mark.parametrize("tb", [False, True])
def test_flattened_multidim_gpu_matches_every_upstream_spelling(
    monkeypatch,
    cold_gpu_caches,
    rank,
    ta,
    tb,
):
    import triton_msl.codegen.generic_lowerer as generic_lowerer

    hits = []
    original = getattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", None)
    assert original is not None

    def spy(self, info):
        hits.append((tuple(info["batch_dims"]), info["a_arg"].elem_type))
        return original(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_batched_dot_template", spy)
    shape = (2,) * (rank - 2) + (32, 32)
    batch = 2 ** (rank - 2)
    torch.manual_seed(3000 + 100 * rank + 10 * int(ta) + int(tb))
    a = torch.randint(-4, 5, shape, dtype=torch.bfloat16, device="mps")
    b = torch.randint(-4, 5, shape, dtype=torch.bfloat16, device="mps")
    a_flat = a.reshape(batch, 32, 32)
    b_flat = b.reshape(batch, 32, 32)
    for i in range(batch):
        a_flat[i].mul_(i % 3 + 1)
        b_flat[i].add_(i % 5)
    sentinel = 12345.0
    storage = torch.full((batch * 1024 + 32,), sentinel, dtype=torch.float32, device="mps")
    c = storage[: batch * 1024].view(shape)
    _dot_multidim_flat[(1,)](a, b, c, rank, ta, tb)
    torch.mps.synchronize()
    aa = a.transpose(-1, -2) if ta else a
    bb = b.transpose(-1, -2) if tb else b
    assert hits == [((batch,), "bf16")]
    assert not torch.isnan(c).any()
    torch.testing.assert_close(c, aa.float() @ bb.float(), rtol=1e-3, atol=1e-2)
    assert torch.all(storage[batch * 1024 :] == sentinel)
