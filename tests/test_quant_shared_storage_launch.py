"""Quantized weight-only matmul launches with an operand in the output's storage.

A quantized route knows every cell it addresses through each pointer: the kernel it accepts
forms flat addresses from its sizes and strides. So the launch compares the cells the output
is stored at with the cells each operand is read at, and not whole storages. An operand that
is a block of the output's storage computes as long as no cell is shared, down to an operand
that ends in the cell before the output's first. An operand read at a cell the output is
stored at is refused before anything is launched, because the template stores finished tiles
while other tiles still load.

Routes: the symmetric int8 GEMM, the per-group int8 GEMM on its fast template (zeros laid out
like the scales) and on its stride-generic template (zeros transposed), and the int8 GEMV.
tests/test_quant_unmasked_tail.py holds the same layouts for the float operands of the GEMV
and of the canonical GEMM; here every operand takes part, the int8 weight as a byte view of
the float storage.

Integer data and power-of-two scales, so float32 arithmetic is exact in any order and every
comparison is exact. A storage is compared by its bits: weight bytes read as floats can be
NaN, which compares unequal to itself.
"""

import dataclasses
import platform
import types
from collections.abc import Callable

import numpy as np
import pytest
import torch

from tests.test_quant_unmasked_tail import _gemv8, _pergroup8, _sym8
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0  # fills an output before its launch and the cells between two blocks
GAP = 5  # cells between two blocks that lie apart
REFUSAL = "quantized.*overlaps one of its operands"


@dataclasses.dataclass(frozen=True)
class Route:
    """One quantized kernel at one size, every tensor flat as the kernel addresses it."""

    label: str
    descriptor: str  # tag of the descriptor the kernel publishes
    shader: str  # entry point of the template that computes the launch
    output: str
    cells: int  # cells the launch stores
    host: dict  # operand name -> host array, in argument order
    want: np.ndarray  # the float64 result
    launch: Callable  # device tensors by name -> None


def _symmetric():
    M, N, K, BM, BN, BK = 64, 48, 96, 32, 16, 32
    g = np.random.default_rng(21)
    a = g.integers(-4, 5, size=(M, K)).astype(np.float32)
    w = g.integers(-32, 32, size=(K, N)).astype(np.int8)
    s = (2.0 ** g.integers(-1, 3, size=N)).astype(np.float32)

    def launch(t):
        _sym8[(M // BM, N // BN)](t["a"], t["w"], t["c"], t["s"], M, N, K, K, 1, N, 1, N, 1, BM=BM, BN=BN, BK=BK)

    want = a.astype(np.float64) @ (w.astype(np.float64) * s[None, :])
    host = {"a": a.reshape(-1), "w": w.reshape(-1), "s": s}
    return Route("symmetric", "sym_int8", "int8_matmul_pergroup", "c", M * N, host, want.reshape(-1), launch)


def _pergroup(zeros, shader):
    """``zeros``: 'like-scales' stores the zero points [group, column] as the scales are,
    'transposed' stores them [column, group], which the fast template does not address."""
    M, N, K, BM, BN, BK, G = 64, 64, 64, 32, 32, 32, 32
    g = np.random.default_rng(22)
    a = g.integers(-4, 5, size=(M, K)).astype(np.float32)
    w = g.integers(-32, 32, size=(K, N)).astype(np.int8)
    s = (2.0 ** g.integers(-1, 3, size=(K // G, N))).astype(np.float32)
    z = g.integers(-4, 5, size=(K // G, N)).astype(np.float32)
    zero_strides = (N, 1) if zeros == "like-scales" else (1, K // G)

    def launch(t):
        _pergroup8[(M // BM, N // BN)](
            *(t[name] for name in "awcsz"), M, N, K, K, 1, N, 1, N, 1, *zero_strides, N, 1, G=G, BM=BM, BN=BN, BK=BK
        )

    group = np.arange(K) // G
    want = a.astype(np.float64) @ ((w.astype(np.float64) - z[group]) * s[group])
    stored = z if zeros == "like-scales" else np.ascontiguousarray(z.T)
    host = {"a": a.reshape(-1), "w": w.reshape(-1), "s": s.reshape(-1), "z": stored.reshape(-1)}
    return Route(f"pergroup-zeros-{zeros}", "pergroup_int8", shader, "c", M * N, host, want.reshape(-1), launch)


def _gemv():
    N, K, BN, BK = 64, 128, 32, 32
    g = np.random.default_rng(23)
    x = g.integers(-4, 5, size=K).astype(np.float32)
    w = g.integers(-32, 32, size=(N, K)).astype(np.int8)
    s = (2.0 ** g.integers(-1, 3, size=N)).astype(np.float32)
    z = g.integers(-4, 5, size=N).astype(np.float32)

    def launch(t):
        _gemv8[(N // BN,)](t["x"], t["w"], t["o"], t["s"], t["z"], N, K, K, 1, BN=BN, BK=BK)

    want = ((w.astype(np.float64) - z[:, None]) @ x.astype(np.float64)) * s
    return Route("gemv", "gemv", "int8_gemv", "o", N, {"x": x, "w": w.reshape(-1), "s": s, "z": z}, want, launch)


ROUTES = [
    _symmetric(),
    _pergroup("like-scales", "int8_matmul_pergroup_fast"),
    _pergroup("transposed", "int8_matmul_pergroup"),
    _gemv(),
]
OPERANDS = [(route, operand) for route in ROUTES for operand in route.host]
SIDES = ["operand-then-output", "output-then-operand"]


def _orders(route):
    for operand in route.host:
        yield operand, route.output
        yield route.output, operand
    yield *route.host, route.output


APART = [(route, order) for route in ROUTES for order in _orders(route)]


def _label(value):
    return value.label if isinstance(value, Route) else value if isinstance(value, str) else "-".join(value)


@pytest.fixture
def ran(monkeypatch):
    """What ran: ``descriptors`` holds the tag of each quantized dispatch, ``quant`` its
    result and ``shaders`` the shader entry points launched."""
    import triton_msl.autotuning._quant_matmul_dispatch as quant_dispatch
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_compile_shader_runtime

    seen = types.SimpleNamespace(descriptors=[], quant=[], shaders=[])
    real_quant = quant_dispatch.dispatch_quant_matmul

    def spy_quant(rt, descriptor, *args, **kwargs):
        seen.descriptors.append(descriptor[0])
        result = real_quant(rt, descriptor, *args, **kwargs)
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


def _separate(route):
    t = {name: torch.from_numpy(x).to("mps") for name, x in route.host.items()}
    t[route.output] = torch.full((route.cells,), POISON, device="mps")
    return t


def _in_one_storage(route, order, gap):
    """The device tensors of a launch with those of ``order`` as consecutive blocks of one
    float storage, ``gap`` cells between two blocks; -1 makes the last cell of a block the
    first cell of the next. The int8 weight is a byte view of its block."""
    t = _separate(route)
    sizes = [route.cells if name == route.output else route.host[name].nbytes // 4 for name in order]
    storage = torch.full((sum(sizes) + gap * (len(order) - 1),), POISON, device="mps")
    spans, at = {}, 0
    for name, size in zip(order, sizes):
        block = storage[at : at + size]
        if name != route.output:
            block = block.view(t[name].dtype)
            block.copy_(t[name])
        t[name], spans[name] = block, (at, at + size)
        at += size + gap
    return t, storage, spans


def _bits(storage):
    return storage.view(torch.int32)


def _changed_outside(storage, before, span):
    """Cells of the storage outside ``span`` that a launch changed."""
    changed = _bits(storage) != before
    changed[span[0] : span[1]] = False
    return int(changed.sum())


def _run(route, t):
    route.launch(t)
    torch.mps.synchronize()
    return t[route.output]


def _assert_computed(route, storage, before, span, out, ran):
    assert ran.descriptors == [route.descriptor] and ran.quant == [True] and ran.shaders == [route.shader]
    np.testing.assert_array_equal(out.cpu().numpy().astype(np.float64), route.want)
    assert _changed_outside(storage, before, span) == 0


@pytest.mark.parametrize("route", ROUTES, ids=_label)
def test_separate_storages_compute_the_reference(route, ran):
    out = _run(route, _separate(route))
    assert ran.descriptors == [route.descriptor] and ran.quant == [True] and ran.shaders == [route.shader]
    np.testing.assert_array_equal(out.cpu().numpy().astype(np.float64), route.want)


@pytest.mark.parametrize("route, order", APART, ids=_label)
def test_operands_apart_from_the_output_in_its_storage_compute_the_separate_result(route, order, ran):
    apart = _run(route, _separate(route))
    for seen in (ran.descriptors, ran.quant, ran.shaders):
        seen.clear()
    t, storage, spans = _in_one_storage(route, order, GAP)
    before = _bits(storage).clone()
    out = _run(route, t)
    _assert_computed(route, storage, before, spans[route.output], out, ran)
    assert torch.equal(out, apart)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("route, operand", OPERANDS, ids=_label)
def test_an_operand_read_at_a_cell_of_the_output_is_refused(route, operand, side, ran):
    """The last cell of the first block is the first cell of the second."""
    order = (operand, route.output) if side == SIDES[0] else (route.output, operand)
    t, storage, _ = _in_one_storage(route, order, -1)
    before = _bits(storage).clone()
    with pytest.raises(MetalNonRecoverableError, match=REFUSAL):
        route.launch(t)
    torch.mps.synchronize()
    assert torch.equal(_bits(storage), before)
    assert ran.descriptors == [route.descriptor] and ran.quant == [False] and ran.shaders == []


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("route, operand", OPERANDS, ids=_label)
def test_an_operand_that_ends_where_the_output_begins_computes(route, operand, side, ran):
    """No cell between the two blocks and none shared, in either order."""
    order = (operand, route.output) if side == SIDES[0] else (route.output, operand)
    t, storage, spans = _in_one_storage(route, order, 0)
    before = _bits(storage).clone()
    out = _run(route, t)
    _assert_computed(route, storage, before, spans[route.output], out, ran)


@pytest.mark.parametrize("route, operand", OPERANDS, ids=_label)
def test_an_operand_view_that_runs_on_over_the_output_computes(route, operand, ran):
    """The operand is passed as a view that reaches the end of the storage, across the
    output that follows it. The kernel reads only the cells its sizes address, which stop
    before the output, so the length of the view decides nothing."""
    t, storage, spans = _in_one_storage(route, (operand, route.output), GAP)
    t[operand] = storage[spans[operand][0] :].view(t[operand].dtype)
    before = _bits(storage).clone()
    out = _run(route, t)
    _assert_computed(route, storage, before, spans[route.output], out, ran)
