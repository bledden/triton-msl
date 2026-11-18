"""An attention launch refuses an output that shares memory with another tensor argument.

A source program loads its rows, keys and values before it stores, so a launch whose output
overlaps an input is defined in the source. The attention lowerings do not keep that order.
Measured on the device before the refusal, each in 40 of 40 launches: a dense head-dim-64
launch with a 64-row block and the output over the keys or the values ran as two 32-row
passes, so rows 32 and up were computed from the first pass's stored rows; and the dQ
backward template stores delta before it loads lse, so with delta over lse every dq cell was
wrong. The two biased source replays keep the order, one threadgroup per program with a
barrier between its last load and its stores, and compute in place. Every other forward
attention kernel and every backward template publishes the arguments it stores through, and
the launcher refuses one that overlaps another tensor argument. Inputs that share memory with
each other, such as one tensor as query, key and value, still compute, and so does a two-dot
kernel that is not attention.

The comparison is of the views as passed. Views that interleave without sharing a cell (the
slots of one packed buffer, a column beside its matrix) are told apart by the residues they
address along a stride and compute. A view shorter than the rows its stride and length
arguments address is compared by its own extent, so such a launch is not refused.
"""

import math
import platform

import pytest
import torch
import triton
import triton.language as tl

import tests.test_fa_bwd_address_witness as bwd
from tests.test_attention_capability import CASES as NARROW_REPLAY
from tests.test_attention_capability import OPEN_CASES as WIDE_ROWS
from tests.test_attention_capability import _attention as _biased_source
from tests.test_attention_capability import compile_case
from tests.test_fa_biased_routing import _biased_fa
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
    assert overlapping_output([torch.zeros(0), arena], [0]) is None


def test_views_that_interleave_without_sharing_a_cell_are_apart():
    pair = torch.zeros(8, 2)
    assert overlapping_output([pair[:, 0], pair[:, 1]], [1]) is None
    assert overlapping_output([pair[:, 0], pair[3:, 0]], [1]) == (1, 0)
    # Query, key, value and output as the slots of one packed buffer.
    slots = torch.zeros(1, 2, 8, 4, 16)
    q, k, v, out = (slots[:, :, :, i] for i in range(4))
    assert overlapping_output([q, k, v, out], [3]) is None
    assert overlapping_output([q, k, v, slots[:, :, :, 2, 8:]], [3]) == (3, 2)
    # A per-row column beside the matrix it belongs to.
    both = torch.zeros(1, 2, 8, 17)
    assert overlapping_output([both[..., :16], both[..., 16]], [0, 1]) is None
    assert overlapping_output([both[..., :16], both[..., 15]], [0, 1]) == (0, 1)
    # Different element widths over one storage are not compared cell by cell.
    wide = torch.zeros(8, 2, dtype=torch.float32)
    halves = wide.view(torch.float16)
    assert overlapping_output([wide[:, 0], halves[:, 3]], [0]) == (0, 1)


def test_the_cell_comparison_never_misses_a_shared_cell():
    # Every pair of strided views of a small arena, against the cells they address.
    cells = torch.arange(4 * 6 * 5).view(4, 6, 5)
    arena = torch.zeros(4 * 6 * 5).view(4, 6, 5)
    cuts = [
        (slice(None), slice(None), slice(None)),
        (slice(1, None, 2), slice(None), slice(None)),
        (slice(None), slice(0, None, 2), slice(None)),
        (slice(None), slice(1, None, 2), slice(None)),
        (slice(None), slice(None), slice(0, 4)),
        (slice(None), slice(None), 4),
        (slice(None), slice(None), 3),
        (slice(None), 2, slice(1, None, 3)),
        (slice(None), slice(None, None, 3), slice(2, None, 2)),
        (slice(2, 3), slice(1, 5), slice(None, None, 2)),
        (0, slice(None), slice(None)),
        (3, slice(4, None), slice(None, 2)),
        (slice(None), slice(None, None, 4), slice(None, None, 4)),
        (slice(None), slice(2, None, 4), slice(1, None, 4)),
    ]
    apart = 0
    for a in cuts:
        for b in cuts:
            shared = bool(set(cells[a].flatten().tolist()) & set(cells[b].flatten().tolist()))
            found = overlapping_output([arena[a], arena[b]], [0])
            assert not shared or found == (0, 1), (a, b)
            apart += found is None
            moved = overlapping_output([arena[a].transpose(0, -1), arena[b]], [0])
            assert not shared or moved == (0, 1), (a, b)
    assert apart > 40  # the proof is used, not only the refusal


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


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("rows, n", [(32, 32), (32, 80), (64, 64)])
def test_the_slots_of_one_packed_buffer_compute(device, rows, n):
    # The four views interleave, so every byte range covers the others; no cell is shared.
    t = _dense_inputs(2, n, 64, device)
    packed = torch.zeros(1, 2, n, 4, 64, device=device)
    views = {name: packed[:, :, :, i] for i, name in enumerate("qkvo")}
    for name in "qkv":
        views[name].copy_(t[name])
    _dense(t, 2, n, 64, rows)
    _dense(views, 2, n, 64, rows)
    assert torch.equal(views["o"], t["o"]) and float(t["o"].abs().max()) > 0
    assert all(torch.equal(views[name], t[name]) for name in "qkv")


def _biased(t, heads, rows, dim, keys):
    args = []
    for name in ("o", "l", "q", "k", "v", "b", "m"):
        args += [t[name], *t[name].stride()]
    _biased_fa[(triton.cdiv(rows, 32), heads)](*args, dim**-0.5, -1e9, 1, heads, rows, dim, 32, keys)
    torch.mps.synchronize()


@requires_mps
@pytest.mark.parametrize(
    "dim, dtype, keys", [(32, torch.float32, 32), (64, torch.float32, 64), (64, torch.float16, 32)]
)
def test_a_log_sum_exp_column_beside_the_output_computes(dim, dtype, keys):
    heads, rows = 2, 64
    gen = torch.Generator().manual_seed(13)
    t = {name: torch.randn(1, heads, rows, dim, generator=gen).to(dtype).to("mps") for name in "qkv"}
    t["b"] = torch.randn(1, heads, rows, rows, generator=gen).to(dtype).to("mps")
    t["m"] = (torch.rand(1, heads, rows, generator=gen) < 0.25).to(torch.uint8).to("mps")
    apart = dict(t, o=torch.zeros(1, heads, rows, dim, dtype=dtype, device="mps"))
    apart["l"] = torch.zeros(1, heads, rows, dtype=dtype, device="mps")
    _biased(apart, heads, rows, dim, keys)
    both = torch.full((1, heads, rows, dim + 1), -8192.0, dtype=dtype, device="mps")
    beside = dict(t, o=both[..., :dim], l=both[..., dim])
    _biased(beside, heads, rows, dim, keys)
    assert torch.equal(beside["o"], apart["o"]) and torch.equal(beside["l"], apart["l"])
    assert float(apart["o"].abs().max()) > 0 and bool(apart["l"].isfinite().all())
    # The column over the output's own last column is two outputs over one cell.
    with pytest.raises(MetalNonRecoverableError, match=REFUSAL.format("o_ptr", "lse_ptr")):
        _biased(dict(t, o=both[..., :dim], l=both[..., dim - 1]), heads, rows, dim, keys)


def _replay_inputs(case):
    dtype, rows, _keys_block, dim, _nq, keys, _narrow, _causal = case
    kind = torch.float16 if dtype == "fp16" else torch.bfloat16
    gen = torch.Generator().manual_seed(511)
    t = {
        name: torch.randn((n, dim), generator=gen).to(kind).to("mps")
        for name, n in (("q", rows), ("k", keys), ("v", keys))
    }
    t["b"] = (torch.randn((rows, keys), generator=gen) * 0.2).to("mps")
    return t


def _replay(case, t, out, **options):
    _dtype, rows, keys_block, dim, _nq, keys, narrow, causal = case
    handle = _biased_source[(1,)](
        t["q"], t["k"], t["v"], t["b"], out, rows, keys,
        D=dim, BM=rows, BN=keys_block, NARROW=narrow, CAUSAL=causal, **options,
    )  # fmt: skip
    torch.mps.synchronize()
    return handle


def _over(t, where, rows, dim):
    """A float32 [rows, dim] view over an input's storage."""
    if where == "bias-start":
        return t["b"].reshape(-1)[: rows * dim].view(rows, dim)
    if where == "bias-end":
        return t["b"].reshape(-1)[-rows * dim :].view(rows, dim)
    return t[where[0]].view(torch.float32).reshape(-1)[: rows * dim].view(rows, dim)


@pytest.mark.parametrize("case", NARROW_REPLAY)
def test_the_narrow_replay_shows_its_store_behind_every_load(case):
    from triton_msl.codegen._biased_replay import store_follows_every_load

    lowerer = compile_case(case)
    msl = lowerer.lower()
    assert "Source-replayed biased attention" in msl and lowerer._attention_outputs is None
    assert store_follows_every_load(lowerer, msl)
    head, barrier, tail = msl.rpartition("    threadgroup_barrier(mem_flags::mem_threadgroup);\n")
    assert barrier and "Out[" in tail
    # With no barrier below the last load, the last one stands above a load or in the loop.
    last_load = max(msl.rfind(name + "[") for name in ("Q", "K", "V", "Bias"))
    assert 0 < last_load < len(head)
    assert not store_follows_every_load(lowerer, msl[:last_load] + msl[last_load:].replace(barrier, ""))
    # A load below it, in a statement of its own or inside the stored value.
    assert not store_follows_every_load(lowerer, head + barrier + "    float late = Bias[0];\n" + tail)
    stored = tail.replace("] = ", "] = K[0] + ", 1)
    assert stored != tail and not store_follows_every_load(lowerer, head + barrier + stored)
    # A second store, and a barrier that only some threads pass.
    assert not store_follows_every_load(lowerer, head + barrier + "    Out[0] = 0.0f;\n" + tail)
    assert not store_follows_every_load(lowerer, head + "    if (lid < 4u) {\n" + barrier + tail + "}\n")
    assert not store_follows_every_load(lowerer, msl.replace("threadgroup_barrier(", "barrier("))


def test_a_store_that_traces_to_no_argument_publishes_nothing(monkeypatch):
    # With no output to compare, the refusal would have to cover inputs that share
    # memory with each other, which is ordinary attention.
    lowerer = compile_case(WIDE_ROWS[0])
    assert lowerer._attention_store_arguments() == [4]
    monkeypatch.setattr(type(lowerer), "_trace_ptr_source", lambda self, ssa_id: None)
    assert lowerer._attention_store_arguments() is None


@pytest.mark.parametrize("case", NARROW_REPLAY[:2])
def test_the_narrow_replay_publishes_its_output_when_the_order_is_not_shown(case, monkeypatch):
    import triton_msl.codegen._biased_replay as replay

    monkeypatch.setattr(replay, "store_follows_every_load", lambda lowerer, msl: False)
    lowerer = compile_case(case)
    assert "Source-replayed biased attention" in lowerer.lower()
    assert lowerer._attention_outputs == [4]


@requires_mps
@pytest.mark.parametrize("shader", ["zero-copy", "host"])
@pytest.mark.parametrize("where", ["bias-start", "bias-end", "key-bytes", "value-bytes"])
@pytest.mark.parametrize("case", NARROW_REPLAY)
def test_the_narrow_biased_replay_computes_in_place(case, where, shader, monkeypatch):
    # One threadgroup per program and a barrier between its last load and its one store.
    if shader == "host":
        monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    rows, dim = case[1], case[3]
    t = _replay_inputs(case)
    apart = torch.full((rows, dim), -8192.0, device="mps")
    handle = _replay(case, t, apart)
    assert "Source-replayed biased attention" in handle.asm["msl"]
    assert handle.metadata.attention_outputs is None
    tail = handle.asm["msl"].rpartition("threadgroup_barrier(")[2]
    assert tail.count("Out[") == 1 and not any(name + "[" in tail for name in ("Q", "K", "V", "Bias"))
    for _ in range(5):
        t = _replay_inputs(case)
        out = _over(t, where, rows, dim)
        _replay(case, t, out)
        assert torch.equal(out, apart)
    assert bool(apart.isfinite().all()) and float(apart.abs().max()) > 0


@requires_mps
@pytest.mark.parametrize("shader", ["zero-copy", "host", "cooperative"])
@pytest.mark.parametrize("where", ["bias-start", "bias-end"])
@pytest.mark.parametrize("case", WIDE_ROWS)
def test_the_wide_biased_row_replay_computes_in_place(case, where, shader, monkeypatch):
    # tests/test_attention_capability.py pins the host path; this adds the default
    # zero-copy path and the cooperative option, which checks the overlap itself.
    if shader == "host":
        monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    options = {"experimental_cooperative_attention": True} if shader == "cooperative" else {}
    rows, dim = case[1], case[3]
    t = _replay_inputs(case)
    apart = torch.full((rows, dim), -8192.0, device="mps")
    handle = _replay(case, t, apart, **options)
    assert "Source-replayed wide biased attention" in handle.asm["msl"]
    assert handle.metadata.attention_outputs is None
    for _ in range(5):
        t = _replay_inputs(case)
        out = _over(t, where, rows, dim)
        _replay(case, t, out, **options)
        assert torch.allclose(out, apart, atol=3e-5, rtol=3e-5)
    assert bool(apart.isfinite().all()) and float(apart.abs().max()) > 0


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
