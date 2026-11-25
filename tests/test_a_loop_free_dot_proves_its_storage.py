"""A loop-free dot is launched only when every cell its source addresses lies inside its storages.

Triton specialises an integer argument equal to 1 to a constant, and a K loop that then runs
once is folded away before the backend sees it: ``_kloop`` below at ``K = 1`` reaches the
lowerer as the same program as a dot written without a loop. That program loads one whole
tile of each operand per launched program, unmasked. Only the K-loop templates recorded a
storage proof, so a loop-free launch whose operand was shorter than the tile, or empty, was
dispatched, where ``K = 2`` on the same tensors is refused.

Both loop-free templates (the stride-aware scalar one and the dense one) now record the grid
storage plan with the literal tile depth, and the launcher proves it before either dispatch
route, through the same launch memo as the K-loop proof. The proof is the source's own access
set: a storage that ends at the last cell the source addresses is accepted, one element less
is refused. Refusals run with both submission methods replaced by a tripwire, so no launch
whose source leaves a storage is ever submitted.

A single dot that no template with a storage plan lowers (an epilogue on the generic route,
the fused softmax template, masked operand loads at a loop bound of 1, the source-replayed
K-loop epilogue) is proved from the kernel's access plan instead: the launcher computes the
lowest and highest element each load and store of the source reaches and refuses a launch
that leaves a storage. The second half of this file is those routes.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from tests.cache_helpers import patch_live_singleton_method
from triton_msl.backend import _lane_order, _launch_memo, driver
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
pytestmark = requires_gpu

FILL = -777.0
REFUSED = "Refusing single-tile dot launch"
B32 = dict(BM=32, BN=32, BK=32)


@triton.jit
def _kloop(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap, bp = A + rm[:, None] * K + rk[None, :], B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK
        bp += BK * N
    tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


# The same source with the loop bound kept a runtime argument at every value.
_kloop_kept = triton.jit(do_not_specialize=["K"])(_kloop.fn)


@triton.jit
def _loop_free(A, B, C, M, N, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * SA + rk[None, :])
    b = tl.load(B + rk[:, None] * SB + rn[None, :])
    tl.store(C + rm[:, None] * SC + rn[None, :], tl.dot(a, b), (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _dense(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TRANS: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * BK + rk[None, :])
    b = tl.load(B + rk[:, None] * BN + rn[None, :])
    if TRANS == 1:
        a = tl.trans(a)
    if TRANS == 2:
        b = tl.trans(b)
    tl.store(C + rm[:, None] * BN + rn[None, :], tl.dot(a, b))


@triton.jit
def _six(X, sxm, sxk, Y, syk, syn, Z, szm, szn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # The single dot of upstream Triton's test_dot: every stride a launch argument.
    om = tl.arange(0, BM)
    on = tl.arange(0, BN)
    ok = tl.arange(0, BK)
    x = tl.load(X + om[:, None] * sxm + ok[None, :] * sxk)
    y = tl.load(Y + ok[:, None] * syk + on[None, :] * syn)
    tl.store(Z + om[:, None] * szm + on[None, :] * szn, tl.dot(x, y))


def _values(count, seed):
    i = np.arange(count, dtype=np.int64)
    # Small integers: every float32 product and sum below is exact.
    return (((i * 2654435761 + seed * 40503) >> 7) % 7 - 3).astype(np.float32)


class Source:
    """A numpy model of one of the kernels above: what it addresses and what it stores."""

    def __init__(
        self, blocks, strides, grid=(1, 1), tiled=(False, False), clip=None, trans=0, depth=None, epilogue=None
    ):
        # ``depth``: how many of the tile's K positions the operand loads keep (masked loads).
        # ``epilogue``: what the source does to the product before it stores it.
        self.epilogue = epilogue
        self.blocks, self.strides, self.grid, self.tiled, self.clip, self.trans = (
            blocks,
            strides,
            grid,
            tiled,
            clip,
            trans,
        )
        bm, bn, bk = blocks
        sam, sak, sbk, sbn, scm, scn = strides
        self.programs = []
        for px in range(grid[0]):
            for py in range(grid[1]):
                rm = (px * bm if tiled[0] else 0) + np.arange(bm)
                rn = (py * bn if tiled[1] else 0) + np.arange(bn)
                rk = np.arange(bk if depth is None else depth)
                mask = np.ones((bm, bn), bool)
                if clip is not None:
                    mask = (rm[:, None] < clip[0]) & (rn[None, :] < clip[1])
                self.programs.append(
                    (
                        rm[:, None] * sam + rk[None, :] * sak,
                        rk[:, None] * sbk + rn[None, :] * sbn,
                        rm[:, None] * scm + rn[None, :] * scn,
                        mask,
                    )
                )
        stored = [cp[mask] for _, _, cp, mask in self.programs if mask.any()]
        # (lowest, highest) element offset from each pointer; None when nothing is addressed
        self.span = (
            (min(int(p[0].min()) for p in self.programs), max(int(p[0].max()) for p in self.programs)),
            (min(int(p[1].min()) for p in self.programs), max(int(p[1].max()) for p in self.programs)),
            (min(int(s.min()) for s in stored), max(int(s.max()) for s in stored)) if stored else None,
        )

    def run(self, a_host, b_host, c_host, pointers):
        """Store the source's result into ``c_host`` (float64), addressing from ``pointers``."""
        for ap, bp, cp, mask in self.programs:
            a = a_host[pointers[0] + ap].astype(np.float64)
            b = b_host[pointers[1] + bp].astype(np.float64)
            if self.trans == 1:
                a = a.T
            if self.trans == 2:
                b = b.T
            c = a @ b
            if self.epilogue == "affine":
                c = c * 2.0 + 1.0
            if self.epilogue == "softmax":
                c = c.astype(np.float32)
                c = np.exp(c - c.max(axis=1, keepdims=True))
                c = (c / c.sum(axis=1, keepdims=True)).astype(np.float64)
            c_host[pointers[2] + cp[mask]] = c[mask]


def _storages(source, layout, device="mps", short=None, dtype=np.float32):
    """Backing arrays and views for one launch.

    ``exact``: each storage begins at the lowest and ends at the highest cell the source
    addresses through that pointer. ``offset``: seven more elements before and five after.
    ``short=(role, end)`` removes one element from that end of that role's exact storage.
    """
    lead, tail = (7, 5) if layout == "offset" else (0, 0)
    hosts, pointers = [], []
    for role, span in enumerate(source.span):
        low, high = span if span is not None else (0, 0)
        before, after = lead - min(low, 0), tail + max(high, 0) + 1
        if short is not None and short[0] == role:
            before, after = (before - 1, after) if short[1] == "front" else (before, after - 1)
        count = before + after
        hosts.append(_values(count, role + 1).astype(dtype) if role < 2 else np.full(count, FILL, np.float32))
        pointers.append(before)
    tensors = [torch.from_numpy(h.copy()).to(device) for h in hosts]
    views = [t[p:] for t, p in zip(tensors, pointers)]
    return hosts, pointers, tensors, views


def _expected(source, hosts, pointers):
    expected = hosts[2].astype(np.float64)
    source.run(hosts[0], hosts[1], expected, pointers)
    return expected


def _install_tripwire(patch):
    """Replace both submission methods: a launch that passes every check ends here, never on the GPU."""
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.backend.driver import MetalUtils, _get_compile_shader_runtime, _get_utils

    hits = []

    def forbidden(*args, **kwargs):
        hits.append(1)
        raise AssertionError("a source that leaves its storage must refuse before any dispatch")

    patch.setattr(CompileShaderRuntime, "dispatch", forbidden)
    patch.setattr(MetalUtils, "launch", forbidden)
    patch_live_singleton_method(patch, _get_compile_shader_runtime, "dispatch", lambda *a: forbidden)
    patch_live_singleton_method(patch, _get_utils, "launch", lambda *a: forbidden)
    return hits


def _forget_tripped_lease(hits):
    if hits:
        # A tripped submission leaves the host route's retained lease marked failed, which
        # would refuse every later launch of this process. Nothing was submitted.
        from triton_msl.backend import _host_dispatch

        getattr(_host_dispatch, "_retained", {}).clear()


@pytest.fixture
def tripwire(monkeypatch):
    hits = _install_tripwire(monkeypatch)
    yield hits
    _forget_tripped_lease(hits)


@pytest.fixture
def counted(monkeypatch):
    """Count the storage proof and start every kernel here from an empty launch memo."""
    counts = dict(proof=0)
    proof = driver._matmul_tail_bounds_reason

    def count_proof(*args, **kwargs):
        counts["proof"] += 1
        return proof(*args, **kwargs)

    monkeypatch.setattr(driver, "_matmul_tail_bounds_reason", count_proof)
    for kernel in (_kloop, _kloop_kept, _loop_free, _dense, _six):
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                memo = getattr(getattr(compiled, "run", None), "_matmul_memo", None)
                if memo is not None:
                    memo.entries.clear()
    return counts


# name -> (kernel, source model, launch(a, b, c))
def _kloop_one(grid=(1, 1), m=32, n=32, kernel=None):
    source = Source((32, 32, 32), (1, 1, n, 1, n, 1), grid, (True, True), (m, n))
    kernel = _kloop if kernel is None else kernel
    return source, lambda a, b, c: kernel[grid](a, b, c, m, n, 1, **B32)


def _free(grid=(1, 1), m=32, n=32, strides=(40, 48, 56)):
    source = Source((32, 32, 32), (strides[0], 1, strides[1], 1, strides[2], 1), grid, (True, True), (m, n))
    return source, lambda a, b, c: _loop_free[grid](a, b, c, m, n, *strides, **B32)


def _whole(trans=0):
    source = Source((32, 32, 32), (32, 1, 32, 1, 32, 1), trans=trans)
    return source, lambda a, b, c: _dense[(1, 1)](a, b, c, TRANS=trans, **B32)


def _upstream(m, n, k, col_a=False, col_b=False, trans_out=False):
    # Layouts of upstream test_dot: an operand stored transposed has unit row stride.
    strides = ((1, m) if col_a else (k, 1)) + ((1, k) if col_b else (n, 1)) + ((1, m) if trans_out else (n, 1))
    source = Source((m, n, k), strides)
    return source, lambda a, b, c: _six[(1, 1)](a, *strides[0:2], b, *strides[2:4], c, *strides[4:6], BM=m, BN=n, BK=k)


LAUNCHES = {
    "loop_bound_one": _kloop_one,
    "loop_bound_one_partial_tiles": lambda: _kloop_one((2, 2), 40, 40),
    "loop_free": _free,
    "loop_free_partial_tiles": lambda: _free((2, 2), 40, 40, (70, 80, 90)),
    "loop_free_negative_row_stride": lambda: _free((2, 1), 64, 32, (-40, 48, 56)),
    "dense": _whole,
    "dense_trans_a": lambda: _whole(1),
    "dense_trans_b": lambda: _whole(2),
}
for _m, _n, _k in ((16, 16, 16), (64, 64, 64), (128, 256, 32), (64, 16, 32), (32, 64, 16), (128, 128, 64)):
    for _col_a, _col_b, _trans in (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ):
        LAUNCHES[f"upstream_{_m}x{_n}x{_k}_{int(_col_a)}{int(_col_b)}{int(_trans)}"] = (
            lambda m=_m, n=_n, k=_k, a=_col_a, b=_col_b, t=_trans: _upstream(m, n, k, a, b, t)
        )


def _check(name, layout, device="mps"):
    source, launch = LAUNCHES[name]()
    hosts, pointers, tensors, views = _storages(source, layout, device)
    expected = _expected(source, hosts, pointers)
    handle = launch(*views)
    if device == "mps":
        torch.mps.synchronize()
    got = tensors[2].cpu().numpy().astype(np.float64)
    # A softmax is float arithmetic on the GPU; everything else here is exact in float32.
    same = np.isclose(got, expected, rtol=2e-5, atol=2e-6) if source.epilogue == "softmax" else got == expected
    wrong = np.flatnonzero(~same)
    assert len(wrong) == 0, f"{len(wrong)} cells of the output storage differ from the source, first at {wrong[0]}"
    assert np.array_equal(tensors[0].cpu().numpy(), hosts[0]) and np.array_equal(tensors[1].cpu().numpy(), hosts[1])
    return handle


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("layout", ["exact", "offset"])
@pytest.mark.parametrize("name", sorted(LAUNCHES))
def test_storage_that_covers_the_source_computes(name, layout, compile_shader, monkeypatch, counted):
    """Storages that end at the source's last cell, and views inside larger ones, compute the source's result."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    handle = _check(name, layout)
    descriptor = handle.metadata.batched_dot_bounds
    assert descriptor[0] == "matmul_grid_storage_v1" and list(descriptor[3][2]) == ["literal", descriptor[1], "i64"]
    assert counted["proof"] == 1
    if name.startswith("loop_bound_one"):
        assert dict(handle.src.signature)["K"] == "constexpr"


@pytest.mark.parametrize("name", [n for n in sorted(LAUNCHES) if n.startswith("upstream")])
def test_host_tensors_of_exactly_the_logical_size_compute(name, counted):
    """Upstream's test_dot passes host tensors that hold exactly the matrices."""
    _check(name, "exact", device="cpu")
    assert counted["proof"] == 1


def _refusal_cases():
    for name in ("loop_bound_one", "loop_bound_one_partial_tiles", "loop_free", "loop_free_partial_tiles", "dense"):
        for role in range(3):
            yield name, (role, "back")
    yield "loop_free_negative_row_stride", (0, "front")
    yield "loop_free_negative_row_stride", (0, "back")
    yield "dense_trans_a", (0, "back")
    yield "upstream_64x16x32_100", (0, "back")
    yield "upstream_64x16x32_010", (1, "back")
    yield "upstream_64x16x32_111", (2, "back")


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("name,short", list(_refusal_cases()))
def test_one_element_short_is_refused_before_dispatch(name, short, compile_shader, monkeypatch, tripwire, counted):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    source, launch = LAUNCHES[name]()
    hosts, pointers, tensors, views = _storages(source, "exact", short=short)
    with pytest.raises(MetalNonRecoverableError, match=REFUSED) as refusal:
        launch(*views)
    assert "ABC"[short[0]] + " source tile reaches outside" in str(refusal.value)
    assert "mask the K tail" not in str(refusal.value)
    assert not tripwire and counted["proof"] == 1
    assert all(np.array_equal(t.cpu().numpy(), h) for t, h in zip(tensors, hosts))


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("name", ["loop_bound_one", "loop_free", "dense"])
@pytest.mark.parametrize("role", [0, 1, 2])
def test_an_empty_tensor_is_refused_before_dispatch(name, role, compile_shader, monkeypatch, tripwire, counted):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    source, launch = LAUNCHES[name]()
    _, _, _, views = _storages(source, "exact")
    views[role] = torch.ones(0, device="mps")
    with pytest.raises(MetalNonRecoverableError, match=REFUSED):
        launch(*views)
    assert not tripwire and counted["proof"] == 1


@pytest.mark.parametrize("compile_shader", ["1", "0"])
def test_the_reported_launch_is_refused_like_its_neighbour(compile_shader, monkeypatch, tripwire, counted):
    """A of M * K elements at K = 1: the source reads 63 cells of it. K = 2 was always refused."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    b, c = torch.ones(64 * 32, device="mps"), torch.zeros(1024, device="mps")
    with pytest.raises(MetalNonRecoverableError, match=REFUSED + ": A source tile reaches outside its 128-byte"):
        _kloop[(1, 1)](torch.ones(32, device="mps"), b, c, 32, 32, 1, **B32)
    with pytest.raises(
        MetalNonRecoverableError, match="Refusing unmasked partial-K matmul launch: A full-block K tail"
    ):
        _kloop[(1, 1)](torch.ones(64, device="mps"), b, c, 32, 32, 2, **B32)
    assert not tripwire and counted["proof"] == 2


@pytest.mark.parametrize("compile_shader", ["1", "0"])
def test_a_loop_bound_kept_at_runtime_keeps_its_proof_and_decisions(compile_shader, monkeypatch, tripwire, counted):
    """``do_not_specialize=["K"]``: the K-loop template, its own descriptor and its own refusal."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    source, launch = _kloop_one(kernel=_kloop_kept)
    b, c = torch.ones(1024, device="mps"), torch.zeros(1024, device="mps")
    for a in (torch.ones(32, device="mps"), torch.ones(0, device="mps")):
        with pytest.raises(MetalNonRecoverableError, match="Refusing unmasked partial-K matmul launch") as refusal:
            launch(a, b, c)
        assert "mask the K tail" in str(refusal.value)
    assert not tripwire and counted["proof"] == 2


@pytest.mark.parametrize("compile_shader", ["1", "0"])
def test_a_loop_bound_kept_at_runtime_computes_with_padding(compile_shader, monkeypatch, counted):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    source, launch = _kloop_one(kernel=_kloop_kept)
    hosts, pointers, tensors, views = _storages(source, "offset")
    expected = _expected(source, hosts, pointers)
    handle = launch(*views)
    torch.mps.synchronize()
    assert np.array_equal(tensors[2].cpu().numpy().astype(np.float64), expected)
    descriptor = handle.metadata.batched_dot_bounds
    assert descriptor[0] == "matmul_grid_storage_v1" and list(descriptor[3][2]) == ["arg", 5, "i32"]
    assert "unmasked source executes complete BLOCK_K iterations" in handle.asm["msl"]
    assert dict(handle.src.signature)["K"] == "i32"


def _outcomes(monkeypatch, enabled, verify):
    """One passing launch, the same launch again, then the operand and the output rebound.

    A launch that has to be refused is made with the submission methods replaced, so a
    stale remembered pass would end at the tripwire and be recorded, not run.
    """
    monkeypatch.setattr(_launch_memo, "ENABLED", enabled)
    monkeypatch.setattr(_launch_memo, "VERIFY", verify)
    for cache in _kloop.device_caches.values():
        for compiled in cache[0].values():
            compiled.run._matmul_memo.entries.clear()
    source, launch = _kloop_one()
    found = []

    def attempt(a, b, c, defined):
        with monkeypatch.context() as patch:
            hits = [] if defined else _install_tripwire(patch)
            try:
                launch(a, b, c)
                torch.mps.synchronize()
                found.append(("computed", c.cpu().numpy().tobytes()))
            except MetalNonRecoverableError as refusal:
                found.append(("refused", str(refusal).split(". ")[0]))
            except Exception:
                if not hits:
                    raise
                found.append(("reached dispatch", ""))
            finally:
                _forget_tripped_lease(hits)

    padded = torch.from_numpy(_values(95, 1)).to("mps")  # the 63 cells the source reads, and more
    b = torch.from_numpy(_values(1024, 2)).to("mps")
    c = torch.full((1024,), FILL, device="mps")
    a = torch.empty(0, device="mps").set_(padded.untyped_storage(), 0, (32,))
    attempt(a, b, c, True)
    attempt(a, b, c, True)
    # The same tensor object over a storage of exactly its 32 elements: the padding is gone.
    exact = torch.from_numpy(_values(32, 1)).to("mps")
    a.set_(exact.untyped_storage(), 0, (32,))
    attempt(a, b, c, False)
    attempt(a, b, c, False)
    # Back on the padded storage, but so far in that the tile leaves it.
    a.set_(padded.untyped_storage(), 40, (32,))
    attempt(a, b, c, False)
    # And at the last offset the tile still fits.
    a.set_(padded.untyped_storage(), 32, (32,))
    attempt(a, b, c, True)
    # The output rebound to a storage one element short of the stored tile.
    attempt(a, b, torch.full((1023,), FILL, device="mps"), False)
    return found


def test_a_remembered_pass_does_not_outlive_its_storage(monkeypatch, counted):
    """Memo on, memo off and test mode give the same launches and the same refusals."""
    memo = _outcomes(monkeypatch, True, False)
    with_memo = counted["proof"]
    fresh = _outcomes(monkeypatch, False, False)
    without_memo = counted["proof"] - with_memo
    verified = _outcomes(monkeypatch, True, True)  # a disagreement raises LaunchMemoDisagreement
    assert [kind for kind, _ in fresh] == [
        "computed",
        "computed",
        "refused",
        "refused",
        "refused",
        "computed",
        "refused",
    ]
    assert memo == fresh == verified
    assert all("source tile reaches outside" in text for kind, text in fresh if kind == "refused")
    # Every launch takes the proof without the memo; with it, only the repeated passing launch is skipped.
    assert without_memo == 7 and with_memo == 6


# ---- one dot on a route without a template storage plan -------------------------------------

OUTSIDE = "Refusing dot launch: the source"


@triton.jit
def _with_epilogue(
    A, B, C, M, N, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SOFTMAX: tl.constexpr
):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    c = tl.dot(tl.load(A + rm[:, None] * SA + rk[None, :]), tl.load(B + rk[:, None] * SB + rn[None, :]))
    if SOFTMAX:
        c = c - tl.max(c, 1)[:, None]
        c = tl.exp(c)
        c = c / tl.sum(c, 1)[:, None]
    else:
        c = c * 2.0 + 1.0
    tl.store(C + rm[:, None] * SC + rn[None, :], c, (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _masked_kloop(A, B, C, M, N, K, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * SA + (k + rk)[None, :], mask=(k + rk)[None, :] < K, other=0.0)
        b = tl.load(B + (k + rk)[:, None] * SB + rn[None, :], mask=(k + rk)[:, None] < K, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * SC + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


def _epilogue(block, softmax, clip=None):
    strides = (40, 48, 56)
    m, n = clip or (block, block)
    source = Source(
        (block, block, block), (40, 1, 48, 1, 56, 1), clip=(m, n), epilogue="softmax" if softmax else "affine"
    )
    return source, lambda a, b, c: _with_epilogue[(1,)](
        a, b, c, m, n, *strides, BM=block, BN=block, BK=block, SOFTMAX=softmax
    )


def _masked(k, grid=(1,), m=32):
    # Loads masked at K: the source reads K columns of A and K rows of B, whatever the tile holds.
    source = Source((32, 32, 32), (40, 1, 48, 1, 56, 1), (grid[0], 1), (True, False), (m, 32), depth=k)
    return source, lambda a, b, c: _masked_kloop[grid](a, b, c, m, 32, k, 40, 48, 56, **B32)


PLAN_LAUNCHES = {
    "epilogue_16": lambda: _epilogue(16, False),
    "epilogue_32": lambda: _epilogue(32, False),
    "epilogue_32_clipped": lambda: _epilogue(32, False, (20, 12)),
    "softmax_32": lambda: _epilogue(32, True),
    "softmax_32_clipped": lambda: _epilogue(32, True, (20, 32)),
    "softmax_16": lambda: _epilogue(16, True),
    "masked_loads_loop_bound_one": lambda: _masked(1),
    "masked_loads_loop_bound_one_two_tiles": lambda: _masked(1, (2,), 52),
}


@pytest.fixture
def plan_proofs(monkeypatch, counted):
    """Count the access-plan storage proof beside the template one."""
    proof = _lane_order.outside_storage

    def count_proof(*args, **kwargs):
        counted["plan"] = counted.get("plan", 0) + 1
        return proof(*args, **kwargs)

    monkeypatch.setattr(_lane_order, "outside_storage", count_proof)
    for kernel in (_with_epilogue, _masked_kloop):
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                memo = getattr(getattr(compiled, "run", None), "_dot_storage_memo", None)
                if memo is not None:
                    memo.entries.clear()
    return counted


def _check_plan(name, layout):
    LAUNCHES[name] = PLAN_LAUNCHES[name]
    try:
        return _check(name, layout)
    finally:
        del LAUNCHES[name]


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("layout", ["exact", "offset"])
@pytest.mark.parametrize("name", sorted(PLAN_LAUNCHES))
def test_a_dot_without_a_template_plan_computes_inside_its_storage(
    name, layout, compile_shader, monkeypatch, plan_proofs
):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    handle = _check_plan(name, layout)
    assert handle.metadata.batched_dot_bounds is None and handle.metadata.access_plan["storage"] is True
    assert plan_proofs == dict(proof=0, plan=1)


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("name", sorted(PLAN_LAUNCHES))
def test_a_dot_without_a_template_plan_is_refused_one_element_short(
    name, role, compile_shader, monkeypatch, tripwire, plan_proofs
):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    source, launch = PLAN_LAUNCHES[name]()
    hosts, pointers, tensors, views = _storages(source, "exact", short=(role, "back"))
    with pytest.raises(MetalNonRecoverableError, match=OUTSIDE) as refusal:
        launch(*views)
    text = str(refusal.value)
    assert ("stores" if role == 2 else "loads") in text and f"argument '{'ABC'[role]}'" in text
    assert not tripwire and plan_proofs.get("plan") == 1
    assert all(np.array_equal(t.cpu().numpy(), h) for t, h in zip(tensors, hosts))
    # A refusal is never remembered.
    with pytest.raises(MetalNonRecoverableError, match=OUTSIDE):
        launch(*views)
    assert not tripwire and plan_proofs.get("plan") == 2


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("role", [0, 1, 2])
def test_masked_loads_are_refused_at_a_loop_bound_of_one_as_at_two(
    role, compile_shader, monkeypatch, tripwire, plan_proofs
):
    """K = 2 takes the K-loop template and its storage plan; K = 1 the generic route and the access plan."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    for k, text in ((1, OUTSIDE), (2, "Refusing unmasked partial-K matmul launch")):
        source, launch = _masked(k)
        _, _, _, views = _storages(source, "exact", short=(role, "back"))
        with pytest.raises(MetalNonRecoverableError, match=text):
            launch(*views)
    assert not tripwire and plan_proofs == dict(proof=1, plan=1)


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("k", [33, 64])
@pytest.mark.parametrize("role", ["A", "B", "Bias", "C"])
def test_the_replayed_loop_epilogue_is_refused_one_element_short(role, k, compile_shader, monkeypatch, tripwire):
    """The source-replayed K-loop epilogue loads whole K blocks: 64 columns of A at K = 33."""
    from tests.test_loop_matmul_epilogue import _scaled_mm

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    sizes = dict(A=31 * 64 + 64, B=63 * 32 + 32, Bias=32, C=31 * 32 + 32)
    sizes[role] -= 1
    a, b, bias, c = (torch.ones(sizes[r], device="mps") for r in ("A", "B", "Bias", "C"))
    with pytest.raises(MetalNonRecoverableError, match=OUTSIDE) as refusal:
        _scaled_mm[(1,)](a, b, bias, c, 32, 32, k, 64, 32, 32, 1.5, 0.5, MODE=0)
    assert f"argument '{role}'" in str(refusal.value) and not tripwire


def test_the_access_plan_proof_follows_a_rebound_storage(monkeypatch, plan_proofs):
    """A remembered pass does not outlive its storage: memo on, memo off and test mode agree."""

    def outcomes(enabled, verify):
        monkeypatch.setattr(_launch_memo, "ENABLED", enabled)
        monkeypatch.setattr(_launch_memo, "VERIFY", verify)
        for cache in _with_epilogue.device_caches.values():
            for compiled in cache[0].values():
                compiled.run._dot_storage_memo.entries.clear()
        source, launch = _epilogue(16, False)
        hosts, pointers, tensors, views = _storages(source, "exact")
        found = []

        def attempt(a, defined):
            with monkeypatch.context() as patch:
                hits = [] if defined else _install_tripwire(patch)
                try:
                    launch(a, views[1], views[2])
                    torch.mps.synchronize()
                    found.append(("computed", tensors[2].cpu().numpy().tobytes()))
                except MetalNonRecoverableError as refusal:
                    found.append(("refused", str(refusal).split(". ")[0]))
                except Exception:
                    if not hits:
                        raise
                    found.append(("reached dispatch", ""))
                finally:
                    _forget_tripped_lease(hits)

        count = len(hosts[0])
        a = torch.empty(0, device="mps").set_(tensors[0].untyped_storage(), 0, (count,))
        attempt(a, True)
        attempt(a, True)
        short = torch.from_numpy(hosts[0][:-1].copy()).to("mps")
        a.set_(short.untyped_storage(), 0, (count - 1,))
        attempt(a, False)
        attempt(a, False)
        a.set_(tensors[0].untyped_storage(), 0, (count,))
        attempt(a, True)
        return found

    memo = outcomes(True, False)
    with_memo = plan_proofs["plan"]
    fresh = outcomes(False, False)
    without_memo = plan_proofs["plan"] - with_memo
    verified = outcomes(True, True)
    assert [kind for kind, _ in fresh] == ["computed", "computed", "refused", "refused", "computed"]
    assert memo == fresh == verified
    # Without the memo every launch takes the proof; with it the two repeats of the passing launch do not.
    assert without_memo == 5 and with_memo == 3
