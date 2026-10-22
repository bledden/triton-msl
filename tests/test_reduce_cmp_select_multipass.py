"""Compiler-only pins for faithful ordered cmp/select multipass reductions."""

from tests.source_identity import assert_candidate_source

from pathlib import Path

import pytest

from tests import test_reduce_cmp_select_nan as base
from tests.test_fa_tail_semantics import executed  # noqa: F401


requires = pytest.mark.skipif(not base.HAS, reason="Triton compiler needed")


if base.HAS:

    @base.triton.jit
    def _inductor_maximum(a, b):
        return base.tl.where((a > b) | (a != a), a, b)

    @base.triton.jit
    def _plain_maximum(a, b):
        return base.tl.where(a > b, a, b)

    @base.triton.jit
    def _quiet_maximum(a, b):
        return base.tl.where((a > b) | (b != b), a, b)

    @base.triton.jit
    def _quiet_minimum(a, b):
        return base.tl.where((a < b) | (b != b), a, b)

    @base.triton.jit
    def _prop_minimum(a, b):
        return base.tl.where((a < b) | (a != a), a, b)

    @base.triton.jit
    def _plain_minimum(a, b):
        return base.tl.where(a <= b, a, b)

    @base.triton.jit
    def _softmax_ordered(x, z, n, block: base.tl.constexpr, mode: base.tl.constexpr):
        cols = base.tl.arange(0, block)
        values = base.tl.load(x + cols, mask=cols < n, other=-float("inf"))
        if mode == 0:
            maximum = base.tl.reduce(values, 0, _inductor_maximum)
        elif mode == 1:
            maximum = base.tl.reduce(values, 0, _plain_maximum)
        elif mode == 2:
            maximum = base.tl.reduce(values, 0, _quiet_maximum)
        elif mode == 3:
            maximum = base.tl.reduce(values, 0, _quiet_minimum)
        elif mode == 4:
            maximum = base.tl.reduce(values, 0, _prop_minimum)
        else:
            maximum = base.tl.reduce(values, 0, _plain_minimum)
        numer = base.tl.exp(values - maximum)
        denom = base.tl.sum(numer, 0)
        base.tl.store(z + cols, numer / denom, mask=cols < n)


def _softmax_msl(tmp_path, monkeypatch, n, mode):
    return base._compile_msl(
        _softmax_ordered,
        {"x": "*fp32", "z": "*fp32", "n": "i32", "block": "constexpr", "mode": "constexpr"},
        {"block": n, "mode": mode},
        tmp_path,
        monkeypatch,
    )


@requires
def test_package_identity_is_the_861_checkout():
    root = Path(__file__).resolve().parents[1]
    assert_candidate_source(root, base.triton_msl, base.generic_lowerer)


@requires
@pytest.mark.parametrize(
    "mode,nan_side,comparison",
    [
        (0, "left", " > "),
        (1, None, " > "),
        (2, "right", " > "),
        (3, "right", " < "),
        (4, "left", " < "),
        (5, None, " <= "),
    ],
)
@pytest.mark.parametrize("n", [4096])
def test_ordered_multipass_stages_full_adjacent_source_tree(mode, nan_side, comparison, n, tmp_path, monkeypatch):
    msl = _softmax_msl(tmp_path, monkeypatch, n, mode)
    stage_decl = f"threadgroup float shared_ordered_multipass_0[{n}];"
    assert stage_decl in msl
    assert "shared_ordered_multipass_0[_loop_e]" in msl
    assert msl.count("for (uint _ord_mp = lid;") == n.bit_length() - 1
    assert f"_ord_mp < {n // 2}u" in msl
    assert "_ord_mp < 1u" in msl
    assert comparison in msl
    if nan_side == "left":
        assert "[_ord_left] != shared_ordered_multipass_0[_ord_left]" in msl
    elif nan_side == "right":
        assert "[_ord_left + 1u] != shared_ordered_multipass_0[_ord_left + 1u]" in msl
    # The normal publication path consumes the already-complete result. It must
    # retain the ordered combiner rather than silently substituting simd min/max.
    assert msl.count("simd_shuffle(ordered_") == 5
    assert "simd_max" not in msl and "simd_min" not in msl


@requires
def test_ordered_multipass_has_barrier_between_every_adjacent_round(tmp_path, monkeypatch):
    msl = _softmax_msl(tmp_path, monkeypatch, 4096, 0)
    first = msl.index("for (uint _ord_mp = lid;")
    last = msl.index("float _local_acc_", first)
    tree = msl[first:last]
    # One publication barrier immediately before the tree plus one after every round.
    assert msl[:first].rstrip().endswith("threadgroup_barrier(mem_flags::mem_threadgroup);")
    assert tree.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 12


@requires
@pytest.mark.parametrize("mode", range(6))
def test_block1024_preserves_existing_single_pass_ordered_route(mode, tmp_path, monkeypatch):
    msl = _softmax_msl(tmp_path, monkeypatch, 1024, mode)
    assert "shared_ordered_multipass_" not in msl
    assert msl.count("simd_shuffle(ordered_") > 0
    assert "simd_max" not in msl and "simd_min" not in msl


@requires
def test_ordered_multipass_register_array_helper_still_refuses():
    # This recovery belongs to the full logical multipass staging route. It must
    # not make the separate MEPT/register-array helper default to another op.
    base.test_ordered_reduce_mept_fold_has_no_default()


@requires
def test_ordered_multipass_singleton_rank2_is_a_full_logical_reduce(tmp_path, monkeypatch):
    msl = base._compile_msl(
        base._reduce_prop_2d,
        {"x": "*fp32", "z": "*fp32", "m": "constexpr", "n": "constexpr"},
        {"m": 1, "n": 4096},
        tmp_path,
        monkeypatch,
    )
    assert "threadgroup float shared_ordered_multipass_0[4096];" in msl
    assert "shared_ordered_multipass_0[_loop_e]" in msl


@requires
def test_ordered_multipass_never_flattens_multivalue_rank2_axis_reduce(tmp_path, monkeypatch):
    with pytest.raises(base.MetalNonRecoverableError):
        base._compile_msl(
            base._reduce_prop_2d,
            {"x": "*fp32", "z": "*fp32", "m": "constexpr", "n": "constexpr"},
            {"m": 2, "n": 2048},
            tmp_path,
            monkeypatch,
        )


@requires
@pytest.mark.parametrize(
    "shape,axis,total,expected",
    [
        ((4096,), 0, 4096, True),
        ((1, 4096), 1, 4096, True),
        ((4096, 1), 0, 4096, True),
        ((2, 2048), 1, 4096, False),
        ((2, 2048), 1, 2048, False),
        ((1, 2048), 1, 4096, False),
        ((4096,), 1, 4096, False),
    ],
)
def test_ordered_multipass_native_shape_gate(shape, axis, total, expected):
    assert base.generic_lowerer.GenericLowerer._ordered_multipass_shape_is_full(shape, axis, total) is expected


def _emitted_tree_bits(words, kind, multipass):
    combine = lambda a, b: base._fold_bits([a, b], kind)

    def balanced(values):
        values = list(values)
        while len(values) > 1:
            values = [combine(values[i], values[i + 1]) for i in range(0, len(values), 2)]
        return values[0]

    if multipass:
        return balanced(words)
    groups = [balanced(words[i : i + 32]) for i in range(0, len(words), 32)]
    result = groups[0]
    for value in groups[1:]:
        result = combine(result, value)
    return result


@base.requires_gpu
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("kind,kernel", base._GPU_CASES)
@pytest.mark.parametrize("pattern", ["mixed", "finite", "signed_zero", "all_nan"])
def test_ordered_multipass_gpu_bits_canaries_and_live_route(kind, kernel, n, pattern, executed):
    words = [0x80000000 if i % 2 else 0x00000000 for i in range(n)]
    for i, word in (
        (0, 0x7FC01001),
        (31, 0x7FC01002),
        (32, 0x7F800000),
        (127, 0xFF800000),
        (128, 0x7FC01003),
        (n // 2, 0x7FC01004),
        (n - 1, 0x7FC01005),
    ):
        words[i] = word
    if pattern == "finite":
        pool = [0xC0600000, 0x40000000, 0x00000000, 0x80000000, 0xBF800000]
        words = [pool[i % len(pool)] for i in range(n)]
    elif pattern == "signed_zero":
        words = [0x80000000 if i % 2 else 0x00000000 for i in range(n)]
    elif pattern == "all_nan":
        words = [0x7FC01000 + (i % 13) for i in range(n)]
    canary = 0x4A617283
    signed = lambda word: word if word < 0x80000000 else word - 0x100000000
    x_storage = (
        base.torch.tensor([canary, *map(signed, words), canary], dtype=base.torch.int32)
        .view(base.torch.float32)
        .to("mps")
    )
    out_storage = base.torch.tensor([canary, canary, canary], dtype=base.torch.int32).view(base.torch.float32).to("mps")
    before = x_storage.cpu().view(base.torch.int32).clone()
    executed.clear()
    handle = kernel[(1,)](x_storage[1:-1], out_storage[1:2], n=n)
    base.torch.mps.synchronize()
    assert base.torch.equal(x_storage.cpu().view(base.torch.int32), before)
    actual = out_storage.cpu().view(base.torch.int32)
    assert int(actual[0]) == canary and int(actual[2]) == canary
    assert int(actual[1]) & 0xFFFFFFFF == _emitted_tree_bits(words, kind, multipass=n == 4096)
    source = base._executed_source(executed, handle)
    assert source == handle.asm["msl"]
    assert ("shared_ordered_multipass_" in source) is (n == 4096)
    assert "simd_max" not in source and "simd_min" not in source
