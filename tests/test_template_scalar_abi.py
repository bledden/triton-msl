"""Template signatures must preserve native scalar storage types, even unused ones."""

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir


@triton.jit
def _matrix(A, B, C, M, N, K, S, SA, SB, SC, MODE: tl.constexpr):
    mm = tl.program_id(0) * 32 + tl.arange(0, 32)
    nn = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    ap = A + mm[:, None] * SA + kk[None, :]
    bp = B + kk[:, None] * SB + nn[None, :] * (2 if MODE == 1 else 1)
    acc = tl.zeros((32, 32), tl.float32)
    for block in range(0, 1 if MODE == 2 else tl.cdiv(K, 32)):
        a = tl.load(ap, kk[None, :] < K - block * 32, other=0.0)
        b = tl.load(bp, kk[:, None] < K - block * 32, other=0.0)
        acc = tl.dot(a, b, acc)
        ap += 32
        bp += 32 * SB
    if MODE == 3:
        acc = acc * S
    tl.store(C + mm[:, None] * SC + nn[None, :], acc, (mm[:, None] < M) & (nn[None, :] < N))


@triton.jit
def _simple(A, B, C, S):
    i = tl.arange(0, 32)
    a = tl.load(A + i[:, None] * 32 + i[None, :])
    b = tl.load(B + i[:, None] * 32 + i[None, :])
    tl.store(C + i[:, None] * 32 + i[None, :], tl.dot(a, b))


@triton.jit
def _shape(X, Y, S, MODE: tl.constexpr):
    i = tl.arange(0, 4)
    j = tl.arange(0, 4)
    k = tl.arange(0, 4)
    off = i[:, None, None] * 16 + j[None, :, None] * 4 + k[None, None, :]
    x = tl.load(X + off)
    if MODE == 0:
        tl.store(Y + off, tl.flip(x, 1))
    elif MODE == 1:
        tl.store(Y + i[:, None] * 4 + j[None, :], tl.sum(x, 2))
    else:
        tl.store(Y + i[:, None] * 4 + j[None, :], tl.argmax(x, 2))


def _lower(fn, signature, constexprs):
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


@pytest.mark.parametrize("kind", ["loop", "strided", "simple", "flip", "reduce", "argmax"])
def test_native_template_scalar_buffer_type(kind):
    if kind in ("loop", "strided", "simple"):
        sig = {
            n: "*fp32" if n in ("A", "B", "C") else "fp32" if n == "S" else "i32"
            for n in _matrix.arg_names
            if n != "MODE"
        }
        const = {"MODE": {"loop": 0, "strided": 1, "simple": 2}[kind]}
        obj = _lower(_matrix, sig, const)
        if kind == "simple":
            obj = _lower(_simple, {"A": "*fp32", "B": "*fp32", "C": "*fp32", "S": "fp32"}, {})
        maker = {
            "loop": "_lower_k_loop_dot_inline",
            "strided": "_lower_strided_scalar_matmul",
            "simple": "_lower_simple_dot_inline",
        }[kind]
    else:
        mode = {"flip": 0, "reduce": 1, "argmax": 2}[kind]
        obj = _lower(_shape, {"X": "*fp32", "Y": "*i32" if mode == 2 else "*fp32", "S": "fp32"}, {"MODE": mode})
        maker = {
            "flip": "_lower_flip_template",
            "reduce": "_lower_3d_reduce_template",
            "argmax": "_lower_3d_argminmax_template",
        }[kind]
    original = getattr(obj, maker)
    hits = []

    def witness(*args, **kwargs):
        hits.append(maker)
        return original(*args, **kwargs)

    setattr(obj, maker, witness)
    msl = obj.lower()
    assert hits == [maker], "must exercise the named template, not generic fallback"
    # The matmul templates emit every argument positionally (``_mmargN``); S is
    # argument 6 of ``_matrix`` and argument 3 of ``_simple``; shape templates
    # preserve S as argument 2.
    s = {"loop": "_mmarg6", "strided": "_mmarg6", "simple": "_mmarg3"}.get(kind, "_mmarg2")
    assert f"device float* {s}_buf" in msl
    assert f"device int* {s}_buf" not in msl
    if kind in ("loop", "strided"):
        assert f"float {s} = {s}_buf[0];" in msl
        assert f"int {s} = {s}_buf[0];" not in msl


def test_native_scalar_widths_and_integer_control():
    # One compact ABI matrix, not a dtype x every-template cross-product.
    for dtype, msltype in [
        ("i1", "bool"),
        ("i8", "char"),
        ("i16", "short"),
        ("i32", "int"),
        ("i64", "long"),
        ("u64", "long"),
        ("fp16", "half"),
        ("bf16", "bfloat"),
    ]:
        sig = {
            n: "*fp32" if n in ("A", "B", "C") else dtype if n == "S" else "i32"
            for n in _matrix.arg_names
            if n != "MODE"
        }
        obj = _lower(_matrix, sig, {"MODE": 0})
        msl = obj.lower()
        # Native Triton IR integers are signless. u64 has an i64 argument;
        # operations carry signedness. The storage load must still read 8 bytes.
        if dtype == "u64":
            assert next(a for a in obj.graph.args if a.name == "S").elem_type == "i64"
        assert f"device {msltype}* _mmarg6_buf [[buffer(6)]]" in msl, dtype
        assert f"{msltype} _mmarg6 = _mmarg6_buf[0];" in msl, dtype
        assert "device int* _mmarg3_buf" in msl and "int _mmarg3 = _mmarg3_buf[0];" in msl

    from triton_msl.backend._launch_signature import scalar_bytes
    import struct

    assert scalar_bytes(1.25, "fp32") == struct.pack("<f", 1.25)
    assert scalar_bytes(2**63 + 3, "u64") == struct.pack("<Q", 2**63 + 3)
    assert len(scalar_bytes(1.25, "fp16")) == 2
    assert len(scalar_bytes(True, "i1")) == 1


def test_scalar_abi_repair_does_not_admit_unreplayed_loop_epilogue():
    from triton_msl.errors import MetalNonRecoverableError

    sig = {
        n: "*fp32" if n in ("A", "B", "C") else "fp32" if n == "S" else "i32" for n in _matrix.arg_names if n != "MODE"
    }
    with pytest.raises(MetalNonRecoverableError, match="trailing compute epilogue"):
        _lower(_matrix, sig, {"MODE": 3}).lower()


@pytest.mark.parametrize("mode", [0, 1])
def test_scalar_signature_templates_execute_without_changing_matmul(monkeypatch, mode):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    calls = []

    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kwargs):
            calls.append((pipeline, grid, group))
            return real(pipeline, grid, group, buffers, **kwargs)

        return wrapper

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    g = torch.Generator().manual_seed(489)
    ac = torch.randint(-2, 3, (64, 32), generator=g).float()
    bc = torch.randint(-2, 3, (32, 128 if mode else 64), generator=g).float()
    a, b = ac.to("mps"), bc.to("mps")
    out = torch.full((64, 64), -8192.0, device="mps")
    handle = _matrix[(2, 2)](a, b, out, 64, 64, 32, 1.25, 32, bc.shape[1], 64, MODE=mode)
    torch.mps.synchronize()
    if mode == 0:
        # Aligned K-loop dispatch intentionally selects the sibling pure-direct
        # pipeline from the SAME library. Require that exact registered object;
        # compiled.function identifies the staged entry, not this execution.
        from triton_msl.backend.driver import _PIPELINE_BUCKETS

        record = _PIPELINE_BUCKETS[id(handle.function)].latest("direct")
        assert record.primary is handle.function
        expected_pipeline = record.direct
    else:
        expected_pipeline = handle.function
    assert len(calls) == 1 and calls[0][0] is expected_pipeline
    assert tuple(calls[0][1]) == (2, 2, 1)
    msl = handle.asm.get("msl", handle.asm.get("metal", ""))
    assert "device float* _mmarg6_buf" in msl and "float _mmarg6 = _mmarg6_buf[0];" in msl
    assert ("float _sum" in msl) if mode else ("simdgroup_multiply_accumulate" in msl)
    expected = ac.double() @ (bc[:, ::2] if mode else bc).double()
    assert torch.equal(out.cpu(), expected.float())
