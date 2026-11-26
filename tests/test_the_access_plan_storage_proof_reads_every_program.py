"""The access-plan storage proof agrees with the source wherever it says "proved".

The proof's lowest and highest element per access are compared with the source's exact access
set, enumerated in numpy over every program, on host tensors and without a device: grids from
one program to thousands, grids launched past the data, the 1-D grid split, short tensors.
Where the proof calls an access proved, a launch whose source leaves a storage must be
refused, with the source's own extremes. Where it calls it sampled (past 262,144 address
instances, read at the ends of the grid only), a refusal must still be a true one. The launch
decisions on the GPU routes are in
tests/test_a_storage_verdict_is_proved_sampled_or_not_judged.py.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.backend._lane_order import LaunchCheck, launch_plan, outside_storage

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


def test_a_launch_within_the_budget_is_proved_and_a_larger_one_is_sampled():
    scalars = [160, 16, 16, 16, 16, 16]
    check = _check(_rows)
    for programs, proved in ((10, True), (1024, True), (1100, False), (16000, False), (20000, False)):
        spans = _source(_rows, programs, *scalars)
        exact = [torch.zeros(span[1] + 1) for span in spans]
        assert outside_storage(check, exact + scalars, (programs, 1, 1)) is None
        assert (not _unproved(check)) == proved
        if not proved:
            assert {why for _, _, why in _unproved(check)} == {"sampled"}
            # What the ends show is still refused: A without its second row tile.
            short = [torch.zeros(BLOCK * 16)] + exact[1:]
            assert outside_storage(check, short + scalars, (programs, 1, 1))[1] == 0
    # Unmasked loads follow the program by a plain shift: exact for a grid far past the budget.
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


def test_a_program_part_that_could_wrap_32_bits_is_not_called_proved():
    """A row stride that takes the address past 2**31 elements: the closed form does not vouch for it."""
    check = _check(_plain)
    scalars = [16 * 70000, 16, 16, 4000, 16, 16]  # 70,000 programs x 16 rows x 4,000 elements per row
    tensors = [torch.zeros(8), torch.zeros(8), torch.zeros(8)]
    outside_storage(check, tensors + scalars, (70000, 1, 1))
    assert ("L", 0, "sampled") in [tuple(entry) for entry in _unproved(check)]
