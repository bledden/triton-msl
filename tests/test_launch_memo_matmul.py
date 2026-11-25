"""A remembered matmul proof or overlap check is skipped only where the uncached launch would pass it.

The launcher remembers a matmul grid storage proof and a fast matmul output-overlap check
that passed, under every value the check reads, and skips the check on a launch with the
same values (triton_msl.backend._launch_memo). Every launch here is run against the
checks themselves: the check functions are counted, the dispatched entry is observed,
and the result or refusal is the one the uncached path gives. A change of grid, scalar,
storage, view offset or output binding, or a hook that runs, must take the check again;
an argument the memo was not reviewed for must take it every time. A storage that
changes while a check itself runs is not remembered at all. The last tests pin inputs
whose keys could be confused or go stale: a scalar that compares equal to another, a
view of another shape, empty storages, a Parameter whose data is swapped, a hook that is
not Triton's own chain and a storage that grows or shrinks.

This file imports the memo module, so it is not collected on a tree without it.
"""

import hashlib
import random

import pytest
import torch
import triton

from tests.test_fa_tail_semantics import executed  # noqa: F401
from tests.test_matmul_tail_storage_contract import _unmasked
from triton_msl.autotuning import _fast_matmul_dispatch
from triton_msl.backend import _launch_memo, driver
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
pytestmark = requires_gpu

M = N = 32
CONSTANTS = dict(BM=32, BN=32, BK=32)
FAST = "kernel void simdgroup_matmul_fast("


def _data(size, seed=0):
    return ((torch.arange(size, dtype=torch.int64) * (seed + 3) % 7) - 3).float()


def _whole(tensor):
    """The tensor's whole storage, on the host."""
    storage = tensor.untyped_storage()
    count = storage.nbytes() // tensor.element_size()
    return torch.empty(0, device=tensor.device).set_(storage, 0, (count,)).cpu()


def _oracle(a, b, k):
    """What the unmasked source computes now: whole 32-wide K blocks, addressed through K from each pointer."""
    logical_k = triton.cdiv(k, 32) * 32
    rows = a.storage_offset() + torch.arange(M)[:, None] * k + torch.arange(logical_k)[None, :]
    cols = b.storage_offset() + torch.arange(logical_k)[:, None] * N + torch.arange(N)[None, :]
    return (_whole(a)[rows].double() @ _whole(b)[cols].double()).float()


def _clear(kernel):
    """Forget every launch this kernel's launchers remembered in earlier tests."""
    for cache in kernel.device_caches.values():
        for compiled in cache[0].values():
            for name in ("_matmul_memo", "_overlap_memo"):
                memo = getattr(getattr(compiled, "run", None), name, None)
                if memo is not None:
                    memo.entries.clear()


def _remembered(name):
    """How many launches the kernel's launchers remember for one of the two checks."""
    total = 0
    for cache in _unmasked.device_caches.values():
        for compiled in cache[0].values():
            memo = getattr(getattr(compiled, "run", None), name, None)
            total += len(memo.entries) if memo is not None else 0
    return total


@pytest.fixture
def counted(monkeypatch):
    """Count the two checks; turn the diagnostic counters on; start from empty memos."""
    counts = dict(proof=0, overlap=0)
    proof, overlap = driver._matmul_tail_bounds_reason, _fast_matmul_dispatch.output_overlaps_operand

    def count_proof(*args, **kwargs):
        counts["proof"] += 1
        return proof(*args, **kwargs)

    def count_overlap(*args, **kwargs):
        counts["overlap"] += 1
        return overlap(*args, **kwargs)

    monkeypatch.setattr(driver, "_matmul_tail_bounds_reason", count_proof)
    monkeypatch.setattr(_fast_matmul_dispatch, "output_overlaps_operand", count_overlap)
    monkeypatch.setattr(_launch_memo, "DIAG", True)
    monkeypatch.setattr(_launch_memo, "ENABLED", True)
    monkeypatch.setattr(_launch_memo, "VERIFY", False)
    _launch_memo.reset_diagnostics()
    _clear(_unmasked)
    return counts


@pytest.fixture(params=[False, True], ids=["memo", "test mode"])
def verify(request, counted, monkeypatch):
    """Run the test with the memo alone and again in test mode, where every hit runs the check too."""
    monkeypatch.setattr(_launch_memo, "VERIFY", request.param)
    return request.param


def _diag(kind):
    return _launch_memo.diagnostics().get("_unmasked", {}).get(kind, {})


def _checks(verify):
    """How often each check must have run: on every miss and bypass, and in test mode on every hit."""
    found = {}
    for name, kind in (("proof", "matmul_storage"), ("overlap", "matmul_overlap")):
        counts = _diag(kind)
        found[name] = counts.get("misses", 0) + counts.get("bypasses", 0) + (counts.get("hits", 0) if verify else 0)
    return found


def _aligned(device="mps"):
    """K = 64: whole source blocks, the fast replacement's shape."""
    a = _data(M * 64).to(device)
    b = _data(64 * N, 1).to(device)
    c = torch.full((M * N,), 17391.0, device=device)
    return a, b, c


def _launch(a, b, c, k, grid=(1, 1)):
    _unmasked[grid](a, b, c, M, N, k, **CONSTANTS)
    torch.mps.synchronize()


def test_a_repeated_launch_skips_both_checks_and_computes(counted, executed):
    a, b, c = _aligned()
    for _ in range(3):
        _launch(a, b, c, 64)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, 64))
    assert counted == dict(proof=1, overlap=1)
    assert len(executed) == 3 and all(FAST in hit.get("msl", "") for hit in executed)
    assert _diag("matmul_storage") == dict(misses=1, hits=2)
    assert _diag("matmul_overlap") == dict(misses=1, hits=2)


def test_a_changed_grid_takes_the_proof_and_is_refused(counted, executed):
    a, b, c = _aligned()
    _launch(a, b, c, 64)
    before = [x.cpu().clone() for x in (a, b, c)]
    # Two row blocks of 32 reach 64 rows of A; its storage holds 32.
    with pytest.raises(MetalNonRecoverableError, match="A full-block K tail reaches outside"):
        _launch(a, b, c, 64, grid=(2, 1))
    assert counted["proof"] == 2 and len(executed) == 1
    assert all(torch.equal(x.cpu(), y) for x, y in zip((a, b, c), before))
    assert _diag("matmul_storage") == dict(misses=2)


def test_a_changed_scalar_takes_the_proof_and_is_refused(counted, executed):
    a, b, c = _aligned()
    _launch(a, b, c, 64)
    with pytest.raises(MetalNonRecoverableError, match="A full-block K tail reaches outside"):
        _launch(a, b, c, 96)
    assert counted["proof"] == 2 and len(executed) == 1
    # And a refusal is never remembered: the same refused launch is checked again.
    with pytest.raises(MetalNonRecoverableError, match="A full-block K tail reaches outside"):
        _launch(a, b, c, 96)
    assert counted["proof"] == 3 and _diag("matmul_storage") == dict(misses=3)


def test_a_storage_rebound_shorter_takes_the_proof_and_is_refused(counted, executed):
    # K = 33: the unmasked source reads 64 columns of A and 64 rows of B, so the proof
    # depends on the padding behind the 32 by 33 view. The memo must see the padding go.
    k = 33
    padded_a, b = _data(M * 64).to("mps"), _data(64 * N, 1).to("mps")
    a_storage = padded_a.untyped_storage()
    a = torch.empty(0, device="mps").set_(a_storage, 0, (M * k,))
    c = torch.full((M * N,), 17391.0, device="mps")
    _launch(a, b, c, k)
    assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k))
    _launch(a, b, c, k)
    assert counted["proof"] == 1 and len(executed) == 2
    # The same tensor object, rebound to a storage of exactly the view's size.
    exact = _data(M * k).to("mps")
    a.set_(exact.untyped_storage(), 0, (M * k,))
    with pytest.raises(MetalNonRecoverableError, match="A full-block K tail reaches outside"):
        _launch(a, b, c, k)
    assert counted["proof"] == 2 and len(executed) == 2
    # Rebound to the padded storage at another offset: a different proof, which passes.
    a.set_(a_storage, 8, (M * k,))
    _launch(a, b, c, k)
    assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k))
    assert counted["proof"] == 3 and len(executed) == 3
    assert _diag("matmul_storage") == dict(misses=3, hits=1)


def test_an_output_rebound_onto_an_input_leaves_the_fast_entry(counted, executed):
    a, b, c = _aligned()
    _launch(a, b, c, 64)
    _launch(a, b, c, 64)
    assert counted["overlap"] == 1 and len(executed) == 2 and all(FAST in hit["msl"] for hit in executed)
    # The output now shares A's storage. One program loads every operand before it
    # stores, so the launch is defined and computes from A's old contents; the
    # replacement does not keep that order, so the overlap check must run and decline it.
    expected = _oracle(a, b, 64)
    shared = torch.empty(0, device="mps").set_(a.untyped_storage(), 0, (M * N,))
    _launch(a, b, shared, 64)
    assert counted["overlap"] == 2 and len(executed) == 3 and FAST not in executed[2].get("msl", "")
    assert torch.equal(shared.cpu().view(M, N), expected)
    # An output inside B's storage, past the cells B is read from, is apart: checked anew, fast again.
    wide_b = torch.cat([_data(64 * N, 1), torch.zeros(M * N)]).to("mps")
    apart = torch.empty(0, device="mps").set_(wide_b.untyped_storage(), 64 * N, (M * N,))
    a2, b2, _ = _aligned()
    _launch(a2, wide_b, apart, 64)
    assert counted["overlap"] == 3 and len(executed) == 4 and FAST in executed[3]["msl"]
    assert torch.equal(apart.cpu().view(M, N), _oracle(a2, wide_b, 64))
    assert _diag("matmul_overlap") == dict(misses=3, hits=1)


def test_a_launch_with_a_hook_takes_the_checks(counted, executed):
    # K = 33 over a padded A: the proof passes on the padding (see the rebinding test).
    k = 33
    padded_a, b = _data(M * 64).to("mps"), _data(64 * N, 1).to("mps")
    a = torch.empty(0, device="mps").set_(padded_a.untyped_storage(), 0, (M * k,))
    c = torch.full((M * N,), 17391.0, device="mps")
    _launch(a, b, c, k)
    _launch(a, b, c, k)
    assert counted == dict(proof=1, overlap=0) and len(executed) == 2
    exact = _data(M * k).to("mps")
    seen = []

    def shrink_a(launch_metadata):
        seen.append(launch_metadata)
        if len(seen) == 1:
            a.set_(exact.untyped_storage(), 0, (M * k,))

    hooks = triton.knobs.runtime.launch_enter_hook
    hooks.add(shrink_a)
    try:
        # The hook rebinds A to an unpadded storage after the lane check and before the proof.
        with pytest.raises(MetalNonRecoverableError, match="A full-block K tail reaches outside"):
            _launch(a, b, c, k)
        assert counted["proof"] == 2 and len(executed) == 2
        # A hook that changes nothing still makes a launch take the checks, and nothing is
        # remembered from it: the fast shape is checked twice under the hook.
        a2, b2, c2 = _aligned()
        _launch(a2, b2, c2, 64)
        _launch(a2, b2, c2, 64)
        assert counted == dict(proof=4, overlap=2) and len(executed) == 4
        assert torch.equal(c2.cpu().view(M, N), _oracle(a2, b2, 64))
    finally:
        hooks.remove(shrink_a)
    assert len(seen) == 3
    _launch(a2, b2, c2, 64)
    _launch(a2, b2, c2, 64)
    assert counted == dict(proof=5, overlap=3) and len(executed) == 6
    assert _diag("matmul_storage") == dict(misses=2, hits=2, bypasses=3)
    assert _diag("matmul_overlap") == dict(misses=1, hits=1, bypasses=2)


class _Count(int):
    """An integer that is not an exact int: the memo was not reviewed for it."""


@pytest.mark.parametrize("unsupported", ["cpu", "parameter", "int subclass"])
def test_unsupported_inputs_take_the_checks_every_time(unsupported, counted, executed):
    if unsupported == "cpu":
        a, b, c = _aligned("cpu")
        k = 64
    elif unsupported == "parameter":
        a, b, c = _aligned()
        a = torch.nn.Parameter(a, requires_grad=False)
        k = 64
    else:
        a, b, c = _aligned()
        k = _Count(64)
    for _ in range(3):
        _launch(a, b, c, k)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, 64))
    assert counted["proof"] == 3
    assert _diag("matmul_storage") == dict(bypasses=3)
    if unsupported != "cpu":
        assert counted["overlap"] == 3 and _diag("matmul_overlap") == dict(bypasses=3)


def test_entries_are_bounded_and_cleared(counted, monkeypatch):
    monkeypatch.setattr(_launch_memo, "LIMIT", 3)
    a, b, _ = _aligned()
    outputs = [torch.full((M * N,), 17391.0, device="mps") for _ in range(4)]
    for c in outputs:
        _launch(a, b, c, 64)
    assert counted["proof"] == 4 and _diag("matmul_storage") == dict(misses=4, evictions=1)
    _launch(a, b, outputs[0], 64)
    assert counted["proof"] == 5 and _diag("matmul_storage") == dict(misses=5, evictions=1)
    _launch(a, b, outputs[0], 64)
    assert counted["proof"] == 5 and _diag("matmul_storage") == dict(misses=5, evictions=1, hits=1)


def test_fresh_outputs_under_one_launcher(counted, record_property):
    """New intermediates every call, as a compiled model allocates: every launch is counted."""
    a, b, _ = _aligned()
    calls = 40
    for _ in range(calls):
        c = torch.empty(M * N, device="mps")
        _launch(a, b, c, 64)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, 64))
        del c
    for kind in ("matmul_storage", "matmul_overlap"):
        found = _diag(kind)
        assert found.get("hits", 0) + found.get("misses", 0) == calls and "bypasses" not in found
        record_property(f"{kind}_hits", found.get("hits", 0))
        record_property(f"{kind}_misses", found.get("misses", 0))
    assert counted["proof"] == _diag("matmul_storage").get("misses", 0)


def _outcome(launch, tensors):
    """What a launch did: the refusal's first line, or the output's bytes and the entry."""
    try:
        launch()
    except MetalNonRecoverableError as refusal:
        return ("refused", str(refusal).split("; ")[0])
    return ("computed", hashlib.sha256(tensors[2].cpu().numpy().tobytes()).hexdigest())


def test_verify_mode_agrees_with_the_uncached_path_on_a_random_sequence(counted, monkeypatch, executed):
    rng = random.Random(20261124)
    padded = [_data(M * 64, seed).to("mps") for seed in range(3)]
    exact = [_data(M * 33, seed).to("mps") for seed in range(2)]
    bs = [_data(64 * N, seed).to("mps") for seed in range(2)]
    cs = [torch.empty(M * N + 16, device="mps") for _ in range(3)]
    plan = []
    for _ in range(160):
        if rng.random() < 0.8:
            a_pool, k = padded, rng.choice((33, 64, 96))
            offset = rng.choice((0, 8)) if k == 33 else 0
        else:
            a_pool, k, offset = exact, 33, 0
        plan.append(
            dict(
                a=rng.randrange(len(a_pool)),
                a_pool=a_pool,
                offset=offset,
                b=rng.randrange(len(bs)),
                c=rng.randrange(len(cs)),
                c_offset=rng.choice((0, 16)),
                k=k,
                grid=rng.choice(((1, 1), (1, 1), (1, 1), (2, 1))),
            )
        )

    def run(step):
        base = step["a_pool"][step["a"]]
        # The kernel addresses A through K, not the view: a view no longer than the storage.
        size = min(M * step["k"], base.numel() - step["offset"])
        a = torch.empty(0, device="mps").set_(base.untyped_storage(), step["offset"], (size,))
        b = bs[step["b"]]
        c = torch.empty(0, device="mps").set_(cs[step["c"]].untyped_storage(), step["c_offset"], (M * N,))
        c.fill_(-1.0)
        return _outcome(lambda: _launch(a, b, c, step["k"], step["grid"]), (a, b, c))

    monkeypatch.setattr(_launch_memo, "ENABLED", False)
    uncached = [run(step) for step in plan]
    checks_uncached = dict(counted)
    assert len({outcome[0] for outcome in uncached}) == 2, "the sequence must both compute and refuse"
    executed_uncached = [hit.get("msl", "host") for hit in executed]
    del executed[:]
    _launch_memo.reset_diagnostics()

    monkeypatch.setattr(_launch_memo, "ENABLED", True)
    monkeypatch.setattr(_launch_memo, "VERIFY", True)
    remembered = [run(step) for step in plan]
    assert remembered == uncached
    assert [hit.get("msl", "host") for hit in executed] == executed_uncached
    # Test mode runs the check on every hit, so the counts match the uncached run exactly,
    # while the memo did find launches to remember.
    assert counted == {kind: 2 * count for kind, count in checks_uncached.items()}
    found = _diag("matmul_storage")
    assert found.get("hits", 0) > 0 and "bypasses" not in found
    assert found.get("hits", 0) + found.get("misses", 0) == len(plan)


def test_test_mode_raises_a_disagreement_instead_of_falling_back(counted, monkeypatch, executed):
    """A remembered check that now answers otherwise is an error, not a fallback and not a refusal."""
    from triton_msl.autotuning import _operand_overlap

    a, b, c = _aligned()
    _launch(a, b, c, 64)
    monkeypatch.setattr(_launch_memo, "VERIFY", True)
    # The overlap check disagrees inside dispatch_fast_matmul, whose own except clause
    # would otherwise mark the shader unsupported and fall back to the compiled entry.
    monkeypatch.setattr(_fast_matmul_dispatch, "output_overlaps_operand", lambda *args, **kwargs: True)
    with pytest.raises(_launch_memo.LaunchMemoDisagreement, match="matmul_overlap"):
        _launch(a, b, c, 64)
    assert len(executed) == 1
    monkeypatch.setattr(_fast_matmul_dispatch, "output_overlaps_operand", _operand_overlap.output_overlaps_operand)
    _launch(a, b, c, 64)
    assert len(executed) == 2 and FAST in executed[1]["msl"], "the fast shader was not marked unsupported"
    # The storage proof disagrees before any dispatch route is tried.
    monkeypatch.setattr(driver, "_matmul_tail_bounds_reason", lambda *args, **kwargs: "a reason")
    with pytest.raises(_launch_memo.LaunchMemoDisagreement, match="matmul_storage"):
        _launch(a, b, c, 64)
    assert len(executed) == 2


REFUSED = "A full-block K tail reaches outside"


def _padded(k=33):
    """K = 33 over a 32 by 64 storage: the proof passes on the padding behind the view (see the rebinding test)."""
    storage = _data(M * 64).to("mps")
    a = torch.empty(0, device="mps").set_(storage.untyped_storage(), 0, (M * k,))
    return a, _data(64 * N, 1).to("mps"), torch.full((M * N,), 17391.0, device="mps")


def test_a_storage_rebound_while_the_proof_runs_is_not_remembered(counted, executed, monkeypatch):
    """The key is read, then the proof reads the tensors again: a change in between is not a proof of the key.

    The proof here rebinds A onto a padded storage while it runs, as another thread or
    code inside a tensor accessor could. It passes on what it read and the launch reads
    inside that storage. The key was read from the unpadded binding, which the proof
    refuses; remembered under it, a later launch would read outside A's allocation.
    """
    k = 33
    a, b, c = _padded(k)
    padded = a.untyped_storage()
    exact = _data(M * k).to("mps")
    a.set_(exact.untyped_storage(), 0, (M * k,))
    with pytest.raises(MetalNonRecoverableError, match=REFUSED):
        _launch(a, b, c, k)
    assert counted["proof"] == 1 and len(executed) == 0
    proof, fired = driver._matmul_tail_bounds_reason, []

    def rebinding(*args, **kwargs):
        if not fired:
            fired.append(True)
            a.set_(padded, 0, (M * k,))
        return proof(*args, **kwargs)

    monkeypatch.setattr(driver, "_matmul_tail_bounds_reason", rebinding)
    _launch(a, b, c, k)
    assert fired and counted["proof"] == 2 and len(executed) == 1
    assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k))
    # Asserted before the refused binding is launched again: this is what keeps that launch inside A.
    assert _remembered("_matmul_memo") == 0, "a proof found on other facts was remembered under the refused ones"
    assert _diag("matmul_storage") == dict(misses=1, bypasses=1)
    # The changed facts are a launch of their own: checked once, then remembered.
    _launch(a, b, c, k)
    _launch(a, b, c, k)
    assert counted["proof"] == 3 and _remembered("_matmul_memo") == 1 and len(executed) == 3
    # Back on the unpadded storage, with nothing changing during the launch: refused.
    a.set_(exact.untyped_storage(), 0, (M * k,))
    before = c.cpu().clone()
    with pytest.raises(MetalNonRecoverableError, match=REFUSED):
        _launch(a, b, c, k)
    assert counted["proof"] == 4 and len(executed) == 3 and torch.equal(c.cpu(), before)
    assert _diag("matmul_storage") == dict(misses=3, bypasses=1, hits=1)


def test_an_output_rebound_while_the_overlap_check_runs_is_not_remembered(counted, executed, monkeypatch):
    """The same window for the fast matmul's overlap check.

    The key is read with the output inside A's storage, which the check declines. The
    check rebinds the output apart while it runs and answers for that. Remembered under
    the shared binding, a later launch into A's storage would take the fast entry, whose
    threadgroups store while others still load.
    """
    a, b, apart = _aligned()
    expected = _oracle(a, b, 64)
    c = torch.empty(0, device="mps").set_(a.untyped_storage(), 0, (M * N,))
    overlap, fired = _fast_matmul_dispatch.output_overlaps_operand, []

    def rebinding(*args, **kwargs):
        if not fired:
            fired.append(True)
            c.set_(apart.untyped_storage(), 0, (M * N,))
        return overlap(*args, **kwargs)

    monkeypatch.setattr(_fast_matmul_dispatch, "output_overlaps_operand", rebinding)
    _launch(a, b, c, 64)
    assert fired and counted["overlap"] == 1 and len(executed) == 1 and FAST in executed[0]["msl"]
    assert torch.equal(apart.cpu().view(M, N), expected) and torch.equal(_oracle(a, b, 64), expected)
    assert _remembered("_overlap_memo") == 0, "a pass found on other facts was remembered under the declined ones"
    assert _diag("matmul_overlap") == dict(bypasses=1)
    # The changed facts are a launch of their own: checked once, then remembered.
    _launch(a, b, c, 64)
    _launch(a, b, c, 64)
    assert counted["overlap"] == 2 and _remembered("_overlap_memo") == 1
    assert len(executed) == 3 and all(FAST in hit["msl"] for hit in executed)
    # Back inside A's storage, with nothing changing during the launch: the check runs and declines.
    c.set_(a.untyped_storage(), 0, (M * N,))
    _launch(a, b, c, 64)
    assert counted["overlap"] == 3 and len(executed) == 4 and FAST not in executed[3].get("msl", "")
    assert torch.equal(c.cpu().view(M, N), expected)
    assert _diag("matmul_overlap") == dict(misses=2, bypasses=1, hits=1)


def test_a_scalar_that_compares_equal_to_one_is_launched_as_itself(counted, verify, executed):
    """K of 1 and K of True compare equal; the proof key holds each scalar's declared type and packed bytes.

    Triton compiles a bool K separately from an int K, and that source is refused when
    it is compiled, so nothing remembered for an int K can stand in for it. The
    remembered K = 64 launch is still the remembered launch afterwards.
    """
    a, b, c = _aligned()
    _launch(a, b, c, 64)
    for _ in range(2):
        c.fill_(17391.0)
        _launch(a, b, c, 1)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, 1))
    before = [x.cpu().clone() for x in (a, b, c)]
    ran = len(executed)
    for _ in range(2):
        with pytest.raises(MetalNonRecoverableError, match="matmul operand stride could not be inferred"):
            _launch(a, b, c, True)
    assert len(executed) == ran and all(torch.equal(x.cpu(), y) for x, y in zip((a, b, c), before))
    for k in (1, 64):
        c.fill_(17391.0)
        _launch(a, b, c, k)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k))
    assert _diag("matmul_storage").get("misses") == 1 and _diag("matmul_storage").get("hits", 0) >= 1
    assert counted == _checks(verify)


def test_a_view_of_another_shape_over_the_same_storage_and_offset_is_the_same_launch(counted, verify, executed):
    """Neither check reads a view's shape or strides: the kernel addresses each pointer through M, N and K."""
    a, b, c = _aligned()
    shapes = [((M * 64,), (64 * N,), (M * N,)), ((M, 64), (64, N), (M, N)), ((64, M), (N, 64), (2, 16, N))]
    for a_shape, b_shape, c_shape in shapes:
        c.fill_(17391.0)
        _launch(a.view(a_shape), b.view(b_shape), c.view(c_shape), 64)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, 64))
    assert all(FAST in hit["msl"] for hit in executed) and len(executed) == 3
    assert _diag("matmul_storage") == dict(misses=1, hits=2) and _diag("matmul_overlap") == dict(misses=1, hits=2)
    # The same shapes with the grid's second row block are refused, whatever the shape.
    before = [x.cpu().clone() for x in (a, b, c)]
    for a_shape, b_shape, c_shape in shapes:
        with pytest.raises(MetalNonRecoverableError, match=REFUSED):
            _launch(a.view(a_shape), b.view(b_shape), c.view(c_shape), 64, grid=(2, 1))
    assert len(executed) == 3 and all(torch.equal(x.cpu(), y) for x, y in zip((a, b, c), before))
    assert _diag("matmul_storage") == dict(misses=4, hits=2) and counted == _checks(verify)


def test_an_empty_storage_is_refused_every_time(counted, verify, executed):
    """Two distinct empty storages have the same facts; the proof refuses an empty A, and a refusal is not remembered."""
    a, b, c = _aligned()
    _launch(a, b, c, 64)
    empties = [torch.empty(0, device="mps") for _ in range(2)]
    assert _launch_memo.tensor_facts(empties[0]) == _launch_memo.tensor_facts(empties[1])
    before = [x.cpu().clone() for x in (b, c)]
    for empty in (empties[0], empties[1], empties[0]):
        with pytest.raises(MetalNonRecoverableError, match="Refusing unmasked partial-K matmul launch"):
            _launch(empty, b, c, 64)
    assert len(executed) == 1 and all(torch.equal(x.cpu(), y) for x, y in zip((b, c), before))
    assert _remembered("_matmul_memo") == 1 and _diag("matmul_storage").get("hits", 0) == 0
    assert counted["proof"] == 4 == _checks(verify)["proof"]
    # The real A again is still the remembered launch.
    _launch(a, b, c, 64)
    assert torch.equal(c.cpu().view(M, N), _oracle(a, b, 64)) and _diag("matmul_storage").get("hits", 0) == 1


def test_a_parameter_whose_data_is_swapped_is_refused(counted, executed):
    """A Parameter takes the matmul checks on every launch, so the storage behind it is read every time."""
    k = 33
    a, b, c = _padded(k)
    weight = torch.nn.Parameter(a, requires_grad=False)
    for _ in range(2):
        _launch(weight, b, c, k)
        assert torch.equal(c.cpu().view(M, N), _oracle(weight.data, b, k))
    before = c.cpu().clone()
    weight.data = _data(M * k, 2).to("mps")
    for _ in range(2):
        with pytest.raises(MetalNonRecoverableError, match=REFUSED):
            _launch(weight, b, c, k)
    assert torch.equal(c.cpu(), before) and len(executed) == 2
    weight.data = a
    _launch(weight, b, c, k)
    assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k))
    assert counted["proof"] == 5 and _diag("matmul_storage") == dict(bypasses=5) and _remembered("_matmul_memo") == 0


@pytest.mark.parametrize("shape", ["plain function", "HookChain subclass"])
def test_a_hook_that_is_not_a_plain_chain_takes_the_checks(shape, counted, verify, executed, monkeypatch):
    """A hook Triton did not build (a bare function, a HookChain subclass) is a hook: the launch takes the checks."""
    k = 33
    a, b, c = _padded(k)
    _launch(a, b, c, k)
    _launch(a, b, c, k)
    assert _diag("matmul_storage") == dict(misses=1, hits=1)
    exact = _data(M * k).to("mps")
    seen = []

    def shrink_a_the_second_time(launch_metadata):
        seen.append(launch_metadata)
        if len(seen) == 2:
            a.set_(exact.untyped_storage(), 0, (M * k,))

    if shape == "plain function":
        hook = shrink_a_the_second_time
    else:

        class Chain(type(triton.knobs.runtime.launch_enter_hook)):
            pass

        hook = Chain()
        hook.add(shrink_a_the_second_time)
    monkeypatch.setattr(triton.knobs.runtime, "launch_enter_hook", hook)
    # The hook changes nothing the first time: the launch still takes the proof, and computes.
    c.fill_(17391.0)
    _launch(a, b, c, k)
    assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k)) and len(executed) == 3
    assert _diag("matmul_storage") == dict(misses=1, hits=1, bypasses=1)
    # The second time it rebinds A to an unpadded storage, after the lane check and before the proof.
    before = c.cpu().clone()
    with pytest.raises(MetalNonRecoverableError, match=REFUSED):
        _launch(a, b, c, k)
    assert len(seen) == 2 and len(executed) == 3 and torch.equal(c.cpu(), before)
    assert _diag("matmul_storage") == dict(misses=1, hits=1, bypasses=2)
    # The fast shape under the same hook: both checks run on each launch and nothing is remembered.
    a2, b2, c2 = _aligned()
    for _ in range(2):
        _launch(a2, b2, c2, 64)
        assert torch.equal(c2.cpu().view(M, N), _oracle(a2, b2, 64))
    assert _diag("matmul_storage") == dict(misses=1, hits=1, bypasses=4)
    assert _diag("matmul_overlap") == dict(bypasses=2) and counted == _checks(verify)


def test_a_storage_that_grows_or_shrinks_takes_the_proof(counted, verify, executed):
    """A storage of another size behind the same tensor object. On MPS torch will not resize a storage in place.

    torch 2.12.1 raises "UntypedStorage.resize_: got unexpected device type mps", so the
    shrinking half of this test rebinds the tensor onto a storage of exactly the view's
    size with ``set_``. Should a later torch resize a device storage in place, that
    branch is skipped and the same refusal is required of the resized storage. The
    growing half uses ``Tensor.resize_``, which torch does allow: it gives the tensor a
    new, larger storage.
    """
    k = 33
    b, c = _data(64 * N, 1).to("mps"), torch.full((M * N,), 17391.0, device="mps")
    a = _data(M * k).to("mps")
    for _ in range(2):
        with pytest.raises(MetalNonRecoverableError, match=REFUSED):
            _launch(a, b, c, k)
    assert len(executed) == 0 and _diag("matmul_storage") == dict(misses=2)
    # Grown: the padding the unmasked K tail reads now exists behind the same view length.
    unpadded = _launch_memo.tensor_facts(a)
    a.resize_(M * 64)
    a.copy_(_data(M * 64))
    a.resize_(M * k)
    assert a.untyped_storage().nbytes() == M * 64 * 4 and _launch_memo.tensor_facts(a) != unpadded
    for _ in range(2):
        c.fill_(17391.0)
        _launch(a, b, c, k)
        assert torch.equal(c.cpu().view(M, N), _oracle(a, b, k))
    assert len(executed) == 2 and _diag("matmul_storage") == dict(misses=3, hits=1)
    # Shrunk to exactly the view's size: the padding is gone, and so is the proof.
    padded = _launch_memo.tensor_facts(a)
    try:
        a.untyped_storage().resize_(M * k * 4)
    except RuntimeError as refused:
        assert "mps" in str(refused)
        a.set_(_data(M * k).to("mps").untyped_storage(), 0, (M * k,))
    assert a.untyped_storage().nbytes() == M * k * 4 and _launch_memo.tensor_facts(a) != padded
    before = c.cpu().clone()
    for _ in range(2):
        with pytest.raises(MetalNonRecoverableError, match=REFUSED):
            _launch(a, b, c, k)
    assert len(executed) == 2 and torch.equal(c.cpu(), before)
    assert _diag("matmul_storage") == dict(misses=5, hits=1) and counted["proof"] == _checks(verify)["proof"]
