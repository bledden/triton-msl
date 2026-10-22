"""Focused CPU compiler pins for cmp/select reduction exceptional semantics."""

from tests.source_identity import assert_candidate_source

from pathlib import Path
import itertools
import struct

import pytest

from tests.test_fa_tail_semantics import _executed_source, executed  # noqa: F401

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    import triton_msl
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False


requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")


if HAS:

    @triton.jit
    def _quietmax(a, b):
        return tl.where((a > b) | (b != b), a, b)

    @triton.jit
    def _propmax(a, b):
        return tl.where((a > b) | (a != a), a, b)

    @triton.jit
    def _quietmin(a, b):
        return tl.where((a < b) | (b != b), a, b)

    @triton.jit
    def _propmin(a, b):
        return tl.where((a < b) | (a != a), a, b)

    @triton.jit
    def _plain_ge_max(a, b):
        return tl.where(a >= b, a, b)

    @triton.jit
    def _plain_le_min(a, b):
        return tl.where(a <= b, a, b)

    @triton.jit
    def _reduce_quiet(x, z, n: tl.constexpr):
        i = tl.arange(0, n)
        tl.store(z, tl.reduce(tl.load(x + i), 0, _quietmax))

    @triton.jit
    def _reduce_prop(x, z, n: tl.constexpr):
        i = tl.arange(0, n)
        tl.store(z, tl.reduce(tl.load(x + i), 0, _propmax))

    @triton.jit
    def _reduce_quiet_min(x, z, n: tl.constexpr):
        i = tl.arange(0, n)
        tl.store(z, tl.reduce(tl.load(x + i), 0, _quietmin))

    @triton.jit
    def _reduce_prop_min(x, z, n: tl.constexpr):
        i = tl.arange(0, n)
        tl.store(z, tl.reduce(tl.load(x + i), 0, _propmin))

    @triton.jit
    def _reduce_plain_ge(x, z, n: tl.constexpr):
        i = tl.arange(0, n)
        tl.store(z, tl.reduce(tl.load(x + i), 0, _plain_ge_max))

    @triton.jit
    def _reduce_plain_le(x, z, n: tl.constexpr):
        i = tl.arange(0, n)
        tl.store(z, tl.reduce(tl.load(x + i), 0, _plain_le_min))

    @triton.jit
    def _reduce_quiet_2d(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, m), tl.reduce(tl.load(x + i * n + j), 1, _quietmax))

    @triton.jit
    def _reduce_prop_2d(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, m), tl.reduce(tl.load(x + i * n + j), 1, _propmax))

    @triton.jit
    def _reduce_quiet_min_2d(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, m), tl.reduce(tl.load(x + i * n + j), 1, _quietmin))

    @triton.jit
    def _reduce_prop_min_2d(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, m), tl.reduce(tl.load(x + i * n + j), 1, _propmin))

    @triton.jit
    def _reduce_quiet_2d_axis0(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, n), tl.reduce(tl.load(x + i * n + j), 0, _quietmax))

    @triton.jit
    def _reduce_prop_2d_axis0(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, n), tl.reduce(tl.load(x + i * n + j), 0, _propmax))

    @triton.jit
    def _reduce_quiet_min_2d_axis0(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, n), tl.reduce(tl.load(x + i * n + j), 0, _quietmin))

    @triton.jit
    def _reduce_prop_min_2d_axis0(x, z, m: tl.constexpr, n: tl.constexpr):
        i = tl.arange(0, m)[:, None]
        j = tl.arange(0, n)[None, :]
        tl.store(z + tl.arange(0, n), tl.reduce(tl.load(x + i * n + j), 0, _propmin))

    @triton.jit
    def _reduce_quiet_nested(x, z, n: tl.constexpr):
        if tl.program_id(0) == 0:
            i = tl.arange(0, n)
            tl.store(z, tl.reduce(tl.load(x + i), 0, _quietmax))


def _compile_msl(fn, signature, constexprs, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    source = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return backend.make_msl(mod, metadata, options)


def _full_msl(fn, tmp_path, monkeypatch, n=64):
    return _compile_msl(
        fn,
        {"x": "*fp32", "z": "*fp32", "n": "constexpr"},
        {"n": n},
        tmp_path,
        monkeypatch,
    )


@requires
def test_package_identity_is_the_checkout_under_test():
    root = Path(__file__).resolve().parents[1]
    assert_candidate_source(root, triton_msl, generic_lowerer)


@requires
@pytest.mark.parametrize("kernel", [_reduce_quiet, _reduce_quiet_min] if HAS else [])
def test_quiet_isnan_false_slot_uses_ordered_replay(kernel, tmp_path, monkeypatch):
    msl = _full_msl(kernel, tmp_path, monkeypatch)
    assert msl.count("simd_shuffle(ordered_") == 5
    assert "ordered_right_" in msl
    assert "simd_shuffle" in msl
    assert "simd_max" not in msl and "simd_min" not in msl


@requires
@pytest.mark.parametrize(
    "kernel",
    [_reduce_prop, _reduce_prop_min] if HAS else [],
)
def test_propagating_isnan_true_slot_uses_operand_preserving_replay(kernel, tmp_path, monkeypatch):
    msl = _full_msl(kernel, tmp_path, monkeypatch)
    assert msl.count("simd_shuffle(ordered_") == 5
    assert "? (float)NAN : simd_" not in msl
    assert "!= ordered_" in msl


@requires
def test_plain_ge_finite_control_is_not_blanket_refused(tmp_path, monkeypatch):
    msl = _full_msl(_reduce_plain_ge, tmp_path, monkeypatch)
    assert msl.count("simd_shuffle(ordered_") == 5
    assert "simd_max" not in msl


def test_cmp_select_truth_table_includes_nan_and_signed_zero():
    nan = float("nan")

    def quiet(a, b):
        return a if (a > b) or (b != b) else b

    def prop(a, b):
        return a if (a > b) or (a != a) else b

    assert quiet(nan, 3.0) == 3.0
    assert quiet(3.0, nan) == 3.0
    assert quiet(nan, nan) != quiet(nan, nan)
    assert prop(nan, 3.0) != prop(nan, 3.0)
    assert prop(3.0, nan) != prop(3.0, nan)
    assert prop(nan, nan) != prop(nan, nan)
    assert str(quiet(0.0, -0.0)).startswith("-")
    assert not str(quiet(-0.0, 0.0)).startswith("-")


def test_supported_ordered_combiners_are_bitwise_associative_on_edge_domain():
    bits = (0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00001, 0x7FC00002)
    vals = [struct.unpack("!f", struct.pack("!I", x))[0] for x in bits]
    as_bits = lambda x: struct.unpack("!I", struct.pack("!f", x))[0]
    ops = (
        lambda a, b: a if (a > b) or (b != b) else b,
        lambda a, b: a if (a > b) or (a != a) else b,
        lambda a, b: a if (a < b) or (b != b) else b,
        lambda a, b: a if (a < b) or (a != a) else b,
    )
    for op in ops:
        for a, b, c in itertools.product(vals, repeat=3):
            assert as_bits(op(op(a, b), c)) == as_bits(op(a, op(b, c)))


def test_plain_ge_nan_domain_is_not_claimed_associative():
    nan = struct.unpack("!f", struct.pack("!I", 0x7FC00001))[0]
    op = lambda a, b: a if a >= b else b
    left = op(op(0.0, nan), -0.0)
    right = op(0.0, op(nan, -0.0))
    assert struct.pack("!f", left) != struct.pack("!f", right)


def test_ordered_two_level_fold_preserves_within_and_cross_simd_order():
    unpack = lambda x: struct.unpack("!f", struct.pack("!I", x))[0]
    pack = lambda x: struct.unpack("!I", struct.pack("!f", x))[0]
    values = [unpack(0x80000000 if i % 2 else 0x00000000) for i in range(64)]
    values[0] = unpack(0x7FC00001)
    values[31] = unpack(0x7FC00002)
    values[32] = unpack(0x7FC00003)
    values[63] = unpack(0x7FC00004)
    ops = (
        lambda a, b: a if (a > b) or (b != b) else b,
        lambda a, b: a if (a > b) or (a != a) else b,
    )

    def fold(seq, op):
        result = seq[0]
        for value in seq[1:]:
            result = op(result, value)
        return result

    for op in ops:
        sequential = fold(values, op)
        group_partials = [fold(values[i : i + 32], op) for i in (0, 32)]
        two_level = fold(group_partials, op)
        assert pack(two_level) == pack(sequential)


def test_adjacent_balanced_fold_is_bitwise_equal_to_sequential_reference():
    def balanced(seq, op):
        values = list(seq)
        while len(values) > 1:
            values = [op(values[i], values[i + 1]) for i in range(0, len(values), 2)]
        return values[0]

    bits = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00001, 0x7FC00002]
    words = [bits[(i * 5 + i // 7) % len(bits)] for i in range(128)]
    for kind in ("quiet_max", "prop_max", "quiet_min", "prop_min"):
        sequential = _fold_bits(words, kind)
        op = lambda a, b, _kind=kind: _fold_bits([a, b], _kind)
        assert balanced(words, op) == sequential
    finite_zero = [0x80000000 if i % 2 else 0x00000000 for i in range(128)]
    for kind in ("plain_ge", "plain_le"):
        sequential = _fold_bits(finite_zero, kind)
        op = lambda a, b, _kind=kind: _fold_bits([a, b], _kind)
        assert balanced(finite_zero, op) == sequential


@requires
def test_ordered_reduce_partial_simd_n16_has_exact_four_rounds(tmp_path, monkeypatch):
    msl = _full_msl(_reduce_quiet, tmp_path, monkeypatch, n=16)
    assert msl.count("simd_shuffle(ordered_") == 4
    assert "ordered_right_" in msl
    assert "+ 16u" not in msl


@requires
def test_ordered_reduce_sub16_stays_refused(tmp_path, monkeypatch):
    with pytest.raises(MetalNonRecoverableError, match="16-or-more-value"):
        _full_msl(_reduce_quiet, tmp_path, monkeypatch, n=8)


@requires
@pytest.mark.parametrize(
    "kernel,axis_loop,seed",
    [
        (_reduce_quiet_2d, "for (uint j = 1u; j < 32u", "[lid * 32u]"),
        (_reduce_quiet_2d_axis0, "for (uint i = 1u; i < 2u", "[lid]"),
    ]
    if HAS
    else [],
)
def test_ordered_reduce_rank2_seeds_first_value_and_folds_in_axis_order(kernel, axis_loop, seed, tmp_path, monkeypatch):
    msl = _compile_msl(
        kernel,
        {"x": "*fp32", "z": "*fp32", "m": "constexpr", "n": "constexpr"},
        {"m": 2, "n": 32},
        tmp_path,
        monkeypatch,
    )
    assert axis_loop in msl
    assert seed in msl
    assert "acc = (float)0" not in msl


@requires
def test_ordered_reduce_nested_refuses(tmp_path, monkeypatch):
    with pytest.raises(MetalNonRecoverableError, match="fully staged rank-2 axis tile"):
        _full_msl(_reduce_quiet_nested, tmp_path, monkeypatch)


@requires
def test_ordered_reduce_multipass_recovers_source_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_MEPT", "0")
    msl = _full_msl(_reduce_quiet, tmp_path, monkeypatch, n=2048)
    assert "shared_ordered_multipass_0[2048]" in msl
    assert "shared_ordered_multipass_0[_loop_e]" in msl
    assert msl.count("for (uint _ord_mp = lid;") == 11
    assert "simd_max" not in msl and "simd_min" not in msl


@requires
def test_ordered_reduce_mept_request_uses_proved_multipass(tmp_path, monkeypatch):
    # The real compiler does not admit this custom recurrence into MEPT; at
    # N=4096 it uses the proved multipass tree even with MEPT requested. The
    # separate register-array helper refusal remains pinned below.
    monkeypatch.setenv("TRITON_MSL_MEPT", "1")
    msl = _full_msl(_reduce_quiet, tmp_path, monkeypatch, n=4096)
    assert "shared_ordered_multipass_0[4096]" in msl
    assert "shared_ordered_multipass_0[_loop_e]" in msl
    assert msl.count("for (uint _ord_mp = lid;") == 12
    assert "simd_max" not in msl and "simd_min" not in msl


@requires
def test_ordered_reduce_mept_fold_has_no_default():
    from triton_msl.codegen.mlir_walker import IRGraph
    from triton_msl.codegen.msl_emitter import KernelBuilder

    class Options:
        num_warps = 4

    lowerer = generic_lowerer.GenericLowerer(IRGraph(func_name="mept_refusal", args=[], ops=[]), Options())
    lowerer.kb = KernelBuilder("mept_refusal", block_size=128)
    with pytest.raises(MetalNonRecoverableError, match="unsupported combine op 'ordered_ogt_a_none'"):
        lowerer._mept_reduce_fold("arr", 4, "ordered_ogt_a_none", "float")


def _fold_bits(bits, kind):
    def value(word):
        return struct.unpack("!f", struct.pack("!I", word))[0]

    def combine(a, b):
        av, bv = value(a), value(b)
        if kind == "quiet_max":
            take_a = av > bv or bv != bv
        elif kind == "prop_max":
            take_a = av > bv or av != av
        elif kind == "quiet_min":
            take_a = av < bv or bv != bv
        elif kind == "prop_min":
            take_a = av < bv or av != av
        elif kind == "plain_ge":
            take_a = av >= bv
        elif kind == "plain_le":
            take_a = av <= bv
        else:
            raise AssertionError(kind)
        return a if take_a else b

    result = bits[0]
    for word in bits[1:]:
        result = combine(result, word)
    return result


_GPU_CASES = (
    (
        ("quiet_max", _reduce_quiet),
        ("prop_max", _reduce_prop),
        ("quiet_min", _reduce_quiet_min),
        ("prop_min", _reduce_prop_min),
        ("plain_ge", _reduce_plain_ge),
        ("plain_le", _reduce_plain_le),
    )
    if HAS
    else ()
)


@requires_gpu
@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("kind,kernel", _GPU_CASES)
def test_ordered_reduce_gpu_bits_canaries_and_live_source(kind, kernel, n, executed):
    # N=32: all-NaN payload selection. N=64: mixed NaN and both zero signs,
    # including the SIMD boundary. N=128: infinities, repeated extrema, and
    # cross-SIMD ties. Plain predicates remain in their finite zero domain.
    if kind.startswith("plain"):
        words = [0x80000000 if i % 2 else 0x00000000 for i in range(n)]
    elif n == 32:
        words = [0x7FC00001 + i for i in range(n)]
    elif n == 64:
        words = [0x80000000 if i % 2 else 0x00000000 for i in range(n)]
        for i, word in (
            (0, 0x7FC00011),
            (15, 0x7FC00012),
            (31, 0x7FC00013),
            (32, 0x7FC00014),
            (47, 0x7FC00015),
            (63, 0x7FC00016),
        ):
            words[i] = word
    else:
        words = [0x3F000000 + (i % 7) for i in range(n)]
        for i, word in (
            (0, 0x7FC00021),
            (31, 0xFF800000),
            (32, 0x7F800000),
            (63, 0x7FC00022),
            (64, 0x7F800000),
            (95, 0xFF800000),
            (96, 0x7FC00023),
            (127, 0x7FC00024),
        ):
            words[i] = word
    input_canary = 0x4A123456
    output_canary = 0x4B654321
    signed = lambda word: word if word < 0x80000000 else word - 0x100000000
    host_storage = torch.tensor(
        [input_canary, input_canary, *map(signed, words), input_canary, input_canary], dtype=torch.int32
    ).view(torch.float32)
    x_storage = host_storage.to("mps")
    x = x_storage[2 : 2 + n]
    out_storage = torch.tensor([output_canary] * 5, dtype=torch.int32).view(torch.float32).to("mps")
    out = out_storage[2:3]
    before = x_storage.cpu().view(torch.int32).clone()
    executed.clear()
    handle = kernel[(1,)](x, out, n=n)
    torch.mps.synchronize()
    after = x_storage.cpu().view(torch.int32)
    out_bits = out_storage.cpu().view(torch.int32)
    assert torch.equal(after, before), "input or input canary mutated"
    assert torch.equal(out_bits[[0, 1, 3, 4]], torch.tensor([output_canary] * 4, dtype=torch.int32))
    # Retained TTGIR is still the literal cmp/or/select recurrence.  Reduction
    # may reassociate this associative operator, but arith.select returns the
    # selected operand; replacing that operand with a canonical NAN is not the
    # bit contract this test is meant to verify.
    expected = _fold_bits(words, kind)
    assert int(out_bits[2].item()) & 0xFFFFFFFF == expected
    msl = _executed_source(executed, handle)
    assert msl == (getattr(handle, "asm", {}) or {}).get("msl", "")
    assert msl.count("simd_shuffle(ordered_") == 5
    assert "for (uint _ord_group = 1u" in msl
    root = Path(__file__).resolve().parents[1]
    assert handle is not None and msl
    assert_candidate_source(root, triton_msl, generic_lowerer)


@requires_gpu
@pytest.mark.parametrize("kind,kernel", _GPU_CASES)
def test_ordered_reduce_gpu_partial_simd_n16(kind, kernel, executed):
    words = [0x80000000 if i % 2 else 0x00000000 for i in range(16)]
    if not kind.startswith("plain"):
        words = [0x7FC00100 + i for i in range(16)]
    canary = 0x4A345678
    signed = lambda word: word if word < 0x80000000 else word - 0x100000000
    x_storage = torch.tensor([canary, *map(signed, words), canary], dtype=torch.int32).view(torch.float32).to("mps")
    out_storage = torch.tensor([canary, canary, canary], dtype=torch.int32).view(torch.float32).to("mps")
    before = x_storage.cpu().view(torch.int32).clone()
    executed.clear()
    compiled = kernel[(1,)](x_storage[1:17], out_storage[1:2], n=16)
    torch.mps.synchronize()
    assert torch.equal(x_storage.cpu().view(torch.int32), before)
    out_bits = out_storage.cpu().view(torch.int32)
    assert int(out_bits[0]) == canary and int(out_bits[2]) == canary
    assert int(out_bits[1]) & 0xFFFFFFFF == _fold_bits(words, kind)
    source = _executed_source(executed, compiled)
    assert source == compiled.asm["msl"]
    assert source.count("simd_shuffle(ordered_") == 4


_RANK2_GPU_CASES = (
    (
        ("quiet_max", _reduce_quiet_2d, _reduce_quiet_2d_axis0),
        ("prop_max", _reduce_prop_2d, _reduce_prop_2d_axis0),
        ("quiet_min", _reduce_quiet_min_2d, _reduce_quiet_min_2d_axis0),
        ("prop_min", _reduce_prop_min_2d, _reduce_prop_min_2d_axis0),
    )
    if HAS
    else ()
)


@requires_gpu
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("kind,axis1_kernel,axis0_kernel", _RANK2_GPU_CASES)
def test_ordered_reduce_gpu_rank2_axis_source_order(kind, axis1_kernel, axis0_kernel, axis, executed):
    m, n = 2, 16
    words = [0x80000000 if i % 2 else 0x00000000 for i in range(m * n)]
    for i, word in (
        (0, 0x7FC00201),
        (7, 0x7FC00202),
        (15, 0x7FC00203),
        (16, 0x7FC00211),
        (23, 0x7FC00212),
        (31, 0x7FC00213),
    ):
        words[i] = word
    canary = 0x4A56789A
    signed = lambda word: word if word < 0x80000000 else word - 0x100000000
    x_storage = torch.tensor([canary, *map(signed, words), canary], dtype=torch.int32).view(torch.float32).to("mps")
    outputs = m if axis == 1 else n
    out_storage = torch.tensor([canary] * (outputs + 2), dtype=torch.int32).view(torch.float32).to("mps")
    before = x_storage.cpu().view(torch.int32).clone()
    executed.clear()
    kernel = axis1_kernel if axis == 1 else axis0_kernel
    compiled = kernel[(1,)](x_storage[1:-1], out_storage[1:-1], m=m, n=n)
    torch.mps.synchronize()
    assert torch.equal(x_storage.cpu().view(torch.int32), before)
    actual = out_storage.cpu().view(torch.int32)
    assert int(actual[0]) == canary and int(actual[-1]) == canary
    if axis == 1:
        expected = [_fold_bits(words[row * n : (row + 1) * n], kind) for row in range(m)]
    else:
        expected = [_fold_bits(words[col::n], kind) for col in range(n)]
    assert [int(x) & 0xFFFFFFFF for x in actual[1:-1]] == expected
    source = _executed_source(executed, compiled)
    assert source == compiled.asm["msl"]
    assert ("for (uint j = 1u" if axis == 1 else "for (uint i = 1u") in source
