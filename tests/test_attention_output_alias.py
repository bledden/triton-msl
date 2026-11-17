"""An attention launch refuses an output that shares memory with another tensor argument.

A source program loads its rows, keys and values before it stores, so a launch whose output
overlaps an input is defined in the source. The attention lowerings do not keep that order.
Measured on the device before the refusal, each in 40 of 40 launches: a dense head-dim-64
launch with a 64-row block and the output over the keys or the values ran as two 32-row
passes, so rows 32 and up were computed from the first pass's stored rows; and the dQ
backward template stores delta before it loads lse, so with delta over lse every dq cell was
wrong. Only the wide biased row replay proves the order (a barrier between its last load and
its stores; tests/test_attention_capability.py pins its in-place launch). Every other forward
attention kernel and every backward template publishes the arguments it stores through, and
the launcher refuses one that overlaps another tensor argument. Inputs that share memory with
each other, such as one tensor as query, key and value, still compute, and so does a two-dot
kernel that is not attention.
"""

import math
import platform

import pytest
import torch
import triton
import triton.language as tl

import tests.test_fa_bwd_address_witness as bwd
from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.autotuning._operand_overlap import overlapping_output
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)
REFUSAL = "attention kernel: output argument {} shares memory with argument {}"


# ---------------------------------------------------------------------------- the lookup
def test_an_output_over_an_input_is_found():
    q, k, v, out = (torch.zeros(2, 4, 8) for _ in range(4))
    assert overlapping_output([q, k, v, out, 4, 8], [3]) is None
    assert overlapping_output([q, k, v, k, 4, 8], [3]) == (3, 1)
    assert overlapping_output([q, k, v, q[1:], 4, 8], [3]) == (3, 0)
    assert overlapping_output([out, q, v, v[..., 0], 4, 8], [0, 3]) == (3, 2)


def test_inputs_that_share_memory_are_not_compared():
    x, out = torch.zeros(2, 4, 8), torch.zeros(2, 4, 8)
    assert overlapping_output([x, x, x, out, 4, 8], [3]) is None
    assert overlapping_output([x, x[:1], x[..., 0], out], [3]) is None


def test_two_outputs_over_each_other_are_found():
    x, out = torch.zeros(2, 4, 8), torch.zeros(2, 4, 8)
    assert overlapping_output([x, out, out], [1, 2]) == (1, 2)


def test_views_of_one_storage_are_compared_by_byte_range():
    arena = torch.zeros(4 * 64)
    q, k, v, out = (arena[i * 64 : (i + 1) * 64].view(8, 8) for i in range(4))
    assert overlapping_output([q, k, v, out], [3]) is None
    assert overlapping_output([q, k, v, arena[191:255].view(8, 8)], [3]) == (3, 2)
    # Interleaved columns never share a cell; the range test still reports them.
    pair = torch.zeros(8, 2)
    assert overlapping_output([pair[:, 0], pair[:, 1]], [1]) == (1, 0)
    assert overlapping_output([torch.zeros(0), arena], [0]) is None


def test_a_reinterpreted_tensor_addresses_its_base():
    x, out = torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)
    assert overlapping_output([x, triton.reinterpret(out, tl.float32)], [1]) is None
    assert overlapping_output([x, triton.reinterpret(x, tl.float32)], [1]) == (1, 0)
    assert overlapping_output([triton.reinterpret(x, tl.float32), x[4:]], [1]) == (1, 0)


def test_arguments_without_readable_storage_are_skipped():
    class Opaque:
        def data_ptr(self):
            return 4096

    x = torch.zeros(8)
    assert overlapping_output([Opaque(), x, 3, 2.5, None], [1]) is None
    assert overlapping_output([x, Opaque()], [1]) is None
    assert overlapping_output([x, x, 3], [2]) is None


# ---------------------------------------------------------------------------- on the device
def _dense(t, heads, n, dim, rows):
    _flash_attn_fwd[(triton.cdiv(n, rows), heads)](
        t["q"], t["k"], t["v"], t["o"],
        *t["q"].stride(), *t["k"].stride(), *t["v"].stride(), *t["o"].stride(),
        1, heads, n,
        BLOCK_M=rows, BLOCK_N=32, HEAD_DIM=dim, IS_CAUSAL=False,
    )  # fmt: skip
    if t["o"].device.type == "mps":
        torch.mps.synchronize()


def _dense_inputs(heads, n, dim, device):
    gen = torch.Generator().manual_seed(11)
    t = {name: torch.randn(1, heads, n, dim, generator=gen).to(device) for name in "qkv"}
    t["o"] = torch.zeros(1, heads, n, dim, device=device)
    return t


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("rows, n", [(32, 32), (32, 80), (64, 64)])
@pytest.mark.parametrize("over", ["q", "k", "v"])
def test_dense_output_over_an_input_refuses(device, rows, n, over):
    t = _dense_inputs(2, n, 64, device)
    before = t[over].clone()
    _dense(t, 2, n, 64, rows)  # the separate output computes
    assert bool(t["o"].isfinite().all()) and float(t["o"].abs().max()) > 0
    t["o"] = t[over]
    with pytest.raises(MetalNonRecoverableError, match=REFUSAL.format("Out", over.upper())):
        _dense(t, 2, n, 64, rows)
    assert torch.equal(t[over], before)


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_one_tensor_as_query_key_and_value_computes(device):
    t = _dense_inputs(2, 80, 64, device)
    x = t["q"]
    copies = {"q": x.clone(), "k": x.clone(), "v": x.clone(), "o": torch.zeros_like(x)}
    _dense(copies, 2, 80, 64, 32)
    shared = {"q": x, "k": x, "v": x, "o": torch.zeros_like(x)}
    _dense(shared, 2, 80, 64, 32)
    assert torch.equal(shared["o"], copies["o"])
    assert float(shared["o"].abs().max()) > 0


@requires_mps
def test_views_of_one_storage_compute_when_the_output_is_apart():
    t = _dense_inputs(2, 32, 64, "mps")
    cells = t["q"].numel()
    arena = torch.zeros(4 * cells, device="mps")
    views = {name: arena[i * cells : (i + 1) * cells].view(1, 2, 32, 64) for i, name in enumerate("qkvo")}
    for name in "qkv":
        views[name].copy_(t[name])
    _dense(t, 2, 32, 64, 32)
    _dense(views, 2, 32, 64, 32)
    assert torch.equal(views["o"], t["o"])


NAMES = {"delta": "d_ptr", "lse": "l_ptr"}  # the other pointers are named <tensor>_ptr


def _backward(kind, t):
    args = []
    for name in bwd.ORDER[kind]:
        args += [t[name], *t[name].stride()]
    tail = (bwd.SM, -1e9, bwd.N, bwd.HH, bwd.DIM, bwd.N, 32, 32)
    bwd.KERNELS[kind][(triton.cdiv(bwd.N, 32), bwd.INST, bwd.HC)](*args, *tail)
    torch.mps.synchronize()


@requires_mps
@pytest.mark.parametrize(
    "kind, output, over",
    [("q", "delta", "lse"), ("q", "dq", "q"), ("q", "dq", "do"), ("kv", "dk", "k"), ("kv", "dv", "k")],
)
def test_backward_output_over_an_input_refuses(kind, output, over):
    t = {name: x.to("mps") for name, x in bwd._inputs(kind, 5).items()}
    _backward(kind, t)  # separate outputs compute
    before = t[over].clone()
    t[output] = t[over]
    with pytest.raises(
        MetalNonRecoverableError,
        match=REFUSAL.format(NAMES.get(output, output + "_ptr"), NAMES.get(over, over + "_ptr")),
    ):
        _backward(kind, t)
    assert torch.equal(t[over], before)


@requires_mps
def test_only_attention_kernels_publish_stored_arguments():
    @triton.jit
    def scaled(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + offs, tl.load(X + offs, mask=offs < N) * 2.0, mask=offs < N)

    x = torch.arange(64, dtype=torch.float32, device="mps")
    scaled[(2,)](x, x, 64, BLOCK=32)  # an elementwise in-place launch is not attention
    torch.mps.synchronize()
    assert torch.equal(x.cpu(), torch.arange(64, dtype=torch.float32) * 2.0)
    (compiled,) = (k for cache in scaled.device_caches.values() for k in cache[0].values())
    assert compiled.metadata.attention_outputs is None

    t = _dense_inputs(1, 32, 64, "mps")
    _dense(t, 1, 32, 64, 32)
    stored = [
        k.metadata.attention_outputs for cache in _flash_attn_fwd.device_caches.values() for k in cache[0].values()
    ]
    assert stored and all(list(s) == [3] for s in stored)
    assert math.isfinite(float(t["o"].sum()))


@triton.jit
def _two_layers(X, W1, W2, Y, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    cell = r[:, None] * BLOCK + r[None, :]
    h = tl.dot(tl.load(X + cell), tl.load(W1 + cell))
    h = h / (1.0 + tl.exp(-h))
    tl.store(Y + cell, tl.dot(h, tl.load(W2 + cell)))


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("over", ["X", "W1", "W2"])
def test_a_two_dot_kernel_that_is_not_attention_computes_in_place(device, over):
    # Two dots and an exponential but no max: the generic lowering, which stages every
    # dot operand behind a barrier, so the last load precedes the first store.
    gen = torch.Generator().manual_seed(3)
    t = {name: (torch.randn(16, 16, generator=gen) * 0.2).to(device) for name in ("X", "W1", "W2")}
    apart = torch.zeros(16, 16, device=device)
    _two_layers[(1,)](t["X"], t["W1"], t["W2"], apart, BLOCK=16)
    _two_layers[(1,)](t["X"], t["W1"], t["W2"], t[over], BLOCK=16)
    if device == "mps":
        torch.mps.synchronize()
    assert torch.equal(t[over], apart) and float(apart.abs().max()) > 0
    assert all(
        k.metadata.attention_outputs is None for cache in _two_layers.device_caches.values() for k in cache[0].values()
    )
