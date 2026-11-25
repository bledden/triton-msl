"""The access-plan storage proof is exact on large and over-launched grids, or says it is not a proof.

Past 262,144 address instances per access the proof read only the first two and last two
programs of each grid axis and took what it found for the whole launch. With a grid launched
past the data (the last programs fully masked) or the tutorial's 1-D grid split (``pid % n``,
``pid // n``), the cell that leaves the storage belongs to a program in between, and a launch
with a short operand or a short output was dispatched.

Now every program is read, in slices, up to 4,194,304 address instances per access; an address
that is a lane part plus a program part, under a mask that does not follow the program, is
exact for any grid up to that many programs. What is left over is read at the ends only and
recorded as NOT PROVED (``_lane_order.storage_unproved``): a cell found outside there still
refuses the launch, and a launch with nothing found there is dispatched, as it was before the
proof existed.

The first tests compare the proof with the source's exact access set, enumerated in numpy over
every program, on host tensors and without a device. The last ones are launch decisions on
both dispatch routes with the submission methods replaced: nothing is submitted.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from tests.test_a_loop_free_dot_proves_its_storage import OUTSIDE, _forget_tripped_lease, _install_tripwire
from triton_msl.backend import _lane_order
from triton_msl.backend._lane_order import LaunchCheck, launch_plan, outside_storage
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
BLOCK = 16
B16 = dict(BM=BLOCK, BN=BLOCK, BK=BLOCK)


@triton.jit
def _rows(A, B, C, M, N, K, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Rows tiled by program_id(0); every access masked at the data's extent.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * SA + rk[None, :], mask=rm[:, None] < M, other=0.0)
    b = tl.load(B + rk[:, None] * SB + rn[None, :], mask=rk[:, None] < K, other=0.0)
    tl.store(C + rm[:, None] * SC + rn[None, :], tl.dot(a, b), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _split(A, B, C, M, N, K, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # The tutorial's 1-D grid: rows are the remainder of the program id, columns the quotient.
    pid = tl.program_id(0)
    rows = tl.cdiv(M, BM)
    rm = (pid % rows) * BM + tl.arange(0, BM)
    rn = (pid // rows) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * SA + rk[None, :], mask=rm[:, None] < M, other=0.0)
    b = tl.load(B + rk[:, None] * SB + rn[None, :], mask=rn[None, :] < N, other=0.0)
    tl.store(C + rm[:, None] * SC + rn[None, :], tl.dot(a, b), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _plain(A, B, C, M, N, K, SA, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Unmasked loads (the source reads every launched tile) and an epilogue that reads the output.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    cp = C + rm[:, None] * SC + rn[None, :]
    c = tl.dot(tl.load(A + rm[:, None] * SA + rk[None, :]), tl.load(B + rk[:, None] * SB + rn[None, :])) + tl.load(cp)
    tl.store(cp, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _source(kernel, programs, m, n, k, sa, sb, sc):
    """(lowest, highest) element the source reaches from A, B and C over the whole launch; None for none."""
    pid = np.arange(programs, dtype=np.int64)
    span = np.arange(BLOCK, dtype=np.int64)
    if kernel is _split:
        rows = -(-m // BLOCK)
        rm = (pid % rows)[:, None] * BLOCK + span[None, :]
        rn = (pid // rows)[:, None] * BLOCK + span[None, :]
    else:
        rm = pid[:, None] * BLOCK + span[None, :]
        rn = np.broadcast_to(span[None, :], (programs, BLOCK))
    a = rm[:, :, None] * sa + span[None, None, :]
    b = span[None, :, None] * sb + rn[:, None, :]
    c = rm[:, :, None] * sc + rn[:, None, :]
    store = (rm < m)[:, :, None] & (rn < n)[:, None, :]
    if kernel is _plain:
        keep = (np.ones_like(a, bool), np.ones_like(b, bool), np.ones_like(c, bool))  # C is loaded unmasked
    elif kernel is _rows:
        keep = (
            np.broadcast_to((rm < m)[:, :, None], a.shape),
            np.broadcast_to((span < k)[None, :, None], b.shape),
            store,
        )
    else:
        keep = (np.broadcast_to((rm < m)[:, :, None], a.shape), np.broadcast_to((rn < n)[:, None, :], b.shape), store)
    return [(int(x[w].min()), int(x[w].max())) if w.any() else None for x, w in zip((a, b, c), keep)]


_CHECKS = {}


def _unproved(check):
    # What the proof says it did not prove. A proof that keeps no such record claims everything.
    return getattr(check, "unproved", [])


def _recorded():
    return getattr(_lane_order, "storage_unproved", dict)()


def _check(kernel):
    if kernel not in _CHECKS:
        compiled = triton.JITFunction(kernel.fn).warmup(*[torch.float32] * 3, 160, 16, 16, 16, 16, 16, grid=(4,), **B16)
        assert compiled.metadata.batched_dot_bounds is None and compiled.metadata.access_plan["storage"] is True
        _CHECKS[kernel] = LaunchCheck(launch_plan(compiled.metadata))
    return _CHECKS[kernel]


def _launches():
    rng = np.random.RandomState(20261125)
    for kernel in (_rows, _split, _plain):
        for _ in range(40):
            tiles = int(rng.choice([1, 3, 10, 40, 700, 1100, 1500, 5000]))
            over = int(rng.choice([0, 0, 1, 2, 7, 300]))
            m = tiles * BLOCK - int(rng.choice([0, 0, 3, 15]))
            n = int(rng.choice([16, 9])) if kernel is not _split else int(rng.choice([16, 32, 41]))
            programs = (
                tiles + over if kernel is not _split else -(-m // BLOCK) * -(-n // BLOCK) + int(rng.choice([0, 1, 2]))
            )
            sa, sb, sc = int(rng.choice([16, 20])), int(rng.choice([16, 50])), int(rng.choice([16, 64, 33]))
            yield (
                kernel,
                programs,
                m,
                n,
                16,
                sa,
                sb,
                sc,
                int(rng.choice([0, 0, 1, 2])),
                int(rng.choice([0, 1, 5, 4000])),
            )


@pytest.mark.parametrize("case", list(_launches()), ids=lambda case: f"{case[0].fn.__name__}-{case[1]}-{case[2]}")
def test_the_proof_agrees_with_the_source_on_every_grid(case):
    kernel, programs, m, n, k, sa, sb, sc, role, short = case
    spans = _source(kernel, programs, m, n, k, sa, sb, sc)
    sizes = [0 if span is None else span[1] + 1 for span in spans]
    assert all(span is None or span[0] >= 0 for span in spans)
    check = _check(kernel)
    exact = [torch.zeros(size) for size in sizes]
    scalars = [m, n, k, sa, sb, sc]
    assert outside_storage(check, exact + scalars, (programs, 1, 1)) is None
    proved = not _unproved(check)
    if not short or sizes[role] == 0:
        return
    tensors = list(exact)
    tensors[role] = torch.zeros(max(0, sizes[role] - short))
    found = outside_storage(check, tensors + scalars, (programs, 1, 1))
    if proved:
        assert found is not None and found[1] == role, "a launch whose source leaves a storage was passed as proved"
        assert (found[2], found[3]) == spans[role]
    else:
        # Not proved: a refusal still has to be a true one, and the record has to say so.
        assert found is None or found[1] == role
        assert any(why in ("sampled", "budget") for _, _, why in _unproved(check))


def test_grids_within_the_walk_are_proved_and_larger_ones_say_they_are_not():
    scalars = [160, 16, 16, 16, 16, 16]
    check = _check(_rows)
    for programs, proved in ((10, True), (1100, True), (16000, True), (20000, False)):
        spans = _source(_rows, programs, *scalars)
        exact = [torch.zeros(span[1] + 1) for span in spans]
        assert outside_storage(check, exact + scalars, (programs, 1, 1)) is None
        assert (not _unproved(check)) == proved
        if not proved:
            assert {why for _, _, why in _unproved(check)} == {"sampled"}
            # What the ends show is still refused: A without its second row tile.
            short = [torch.zeros(BLOCK * 16)] + exact[1:]
            assert outside_storage(check, short + scalars, (programs, 1, 1))[1] == 0
    # Unmasked loads follow the program by a plain shift: exact for any grid the walk could not read.
    check = _check(_plain)
    scalars = [16 * 70000, 16, 16, 16, 16, 16]
    spans = _source(_plain, 70000, *scalars)
    exact = [torch.zeros(span[1] + 1) for span in spans]
    assert outside_storage(check, exact + scalars, (70000, 1, 1)) is None
    assert not [entry for entry in _unproved(check) if entry[0] == "L"]
    assert outside_storage(check, [torch.zeros(spans[0][1])] + exact[1:] + scalars, (70000, 1, 1))[:4] == (
        "L",
        0,
        0,
        spans[0][1],
    )


def _decision(monkeypatch, launch):
    with monkeypatch.context() as patch:
        hits = _install_tripwire(patch)
        try:
            launch()
            return "dispatched"
        except MetalNonRecoverableError as refusal:
            return "refused" if OUTSIDE in str(refusal) else str(refusal)
        except Exception:
            if not hits:
                raise
            return "dispatched"
        finally:
            _forget_tripped_lease(hits)


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("kernel,programs,n", [(_rows, 1100, 16), (_split, 1102, 1760)], ids=["over_launched", "split"])
def test_a_large_grid_with_a_short_tensor_is_refused_before_dispatch(kernel, programs, n, compile_shader, monkeypatch):
    """The reviewer's reproducer: 160 rows of data, over a thousand programs, tensors holding 32 rows."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    scalars = [160, n, 16, 16, 16, 16 if kernel is _rows else 1760]
    spans = _source(kernel, programs, *scalars)
    sizes = [span[1] + 1 for span in spans]

    def launch(shrink):
        tensors = [torch.zeros(max(0, size - shrink.get(role, 0)), device="mps") for role, size in enumerate(sizes)]
        return lambda: kernel[(programs,)](*tensors, *scalars, **B16)

    assert _decision(monkeypatch, launch({})) == "dispatched"
    assert kernel.fn.__name__ not in _recorded()
    for role in (0, 2):
        assert _decision(monkeypatch, launch({role: 1})) == "refused"
        assert _decision(monkeypatch, launch({role: sizes[role] - 512})) == "refused"


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["1", "0"])
def test_a_grid_past_the_walk_is_dispatched_and_recorded_as_not_proved(compile_shader, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    scalars = [160, 16, 16, 16, 16, 16]
    sizes = [span[1] + 1 for span in _source(_rows, 20000, *scalars)]
    tensors = [torch.zeros(size, device="mps") for size in sizes]
    assert _decision(monkeypatch, lambda: _rows[(20000,)](*tensors, *scalars, **B16)) == "dispatched"
    assert {why for _, _, why in _recorded()["_rows"]} == {"sampled"}
    # In bounds at a smaller grid again: judged in full, and the record is withdrawn.
    assert _decision(monkeypatch, lambda: _rows[(10,)](*tensors, *scalars, **B16)) == "dispatched"
    assert "_rows" not in _recorded()
