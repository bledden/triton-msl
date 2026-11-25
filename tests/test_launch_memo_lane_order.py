"""A remembered lane-order proof is skipped only where the uncached launch would find it again.

The launcher remembers a lane-order proof under every value the check reads (each
argument's storage, extent, offset, width and pointer or its exact scalar value, and the
grid) and skips the check on a launch with the same values
(triton_msl.backend._launch_memo). Every launch here is run against the check itself,
counted, and against the result the source computes or the refusal the uncached path
raises: a changed grid or scalar, a storage rebound onto another, an output that is an
input, and a hook that rebinds a storage each take the check again; an argument the memo
was not reviewed for takes it every time; and test mode agrees with the uncached path
over a random sequence.

This file imports the memo module, so it is not collected on a tree without it.
"""

import hashlib
import random

import pytest
import torch
import triton
import triton.language as tl

from tests.test_lane_order_launch import DEVICES, _wrapped, requires_mps
from triton_msl.backend import _lane_order, _launch_memo
from triton_msl.errors import MetalNonRecoverableError

pytestmark = requires_mps

BLOCK = 64


@triton.jit
def _reversed(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + (BLOCK - 1 - i), tl.load(X + i))


@triton.jit
def _by_programs(X, Y, BLOCK: tl.constexpr):
    # Each lane its own element when one program runs; at two programs lanes i and
    # i + BLOCK / 2 of every program load and store the same element (the same value,
    # so a launch on separate tensors is deterministic and in place it is refused).
    a = (tl.arange(0, BLOCK) * tl.num_programs(0)) % BLOCK
    tl.store(Y + a, tl.load(X + a) + 1.0)


KERNELS = (_reversed, _wrapped, _by_programs)


def _clear():
    for kernel in KERNELS:
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                check = getattr(getattr(compiled, "run", None), "_lane_check", None)
                memo = getattr(check, "memo", None)
                if memo is not None:
                    memo.entries.clear()


@pytest.fixture
def counted(monkeypatch):
    """Count the lane-order check; turn the diagnostic counters on; start from empty memos."""
    count = [0]
    unproved = _lane_order.unproved

    def counting(*args, **kwargs):
        count[0] += 1
        return unproved(*args, **kwargs)

    monkeypatch.setattr(_lane_order, "unproved", counting)
    monkeypatch.setattr(_launch_memo, "DIAG", True)
    monkeypatch.setattr(_launch_memo, "ENABLED", True)
    monkeypatch.setattr(_launch_memo, "VERIFY", False)
    _launch_memo.reset_diagnostics()
    _clear()
    return count


def _diag(kernel):
    return _launch_memo.diagnostics().get(kernel.fn.__name__, {}).get("lane_order", {})


def _values(device="mps", seed=0):
    return torch.arange(BLOCK, dtype=torch.float32).roll(seed).to(device)


def _launch(kernel, args, grid=1, device="mps"):
    kernel[(grid,)](*args, BLOCK=BLOCK)
    if device == "mps":
        torch.mps.synchronize()


@pytest.mark.parametrize("device", DEVICES)
def test_a_repeated_launch_skips_the_check_and_computes(device, counted):
    x = _values(device)
    y = torch.full((BLOCK,), -7.0, device=device)
    for _ in range(3):
        _launch(_reversed, (x, y), device=device)
        assert torch.equal(y.cpu(), x.cpu().flip(0))
        y.fill_(-7.0)
    assert counted[0] == 1 and _diag(_reversed) == dict(misses=1, hits=2)


def test_a_changed_scalar_takes_the_check_and_is_refused(counted):
    x = _values()
    _launch(_wrapped, (x, x, BLOCK))
    _launch(_wrapped, (x, x, BLOCK))
    assert torch.equal(x.cpu(), _values("cpu") * 4) and counted[0] == 1
    before = x.cpu().clone()
    # Half the modulus: two lanes of the one program store the same element.
    with pytest.raises(MetalNonRecoverableError, match="lane order") as refusal:
        _launch(_wrapped, (x, x, BLOCK // 2))
    assert "two lanes of one program reach the same element" in str(refusal.value)
    assert torch.equal(x.cpu(), before) and counted[0] == 2
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_wrapped, (x, x, BLOCK // 2))
    assert counted[0] == 3 and _diag(_wrapped) == dict(misses=3, hits=1)


def test_a_changed_grid_takes_the_check_and_is_refused(counted):
    x = _values()
    _launch(_by_programs, (x, x))
    _launch(_by_programs, (x, x))
    assert torch.equal(x.cpu(), _values("cpu") + 2) and counted[0] == 1
    before = x.cpu().clone()
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_by_programs, (x, x), grid=2)
    assert torch.equal(x.cpu(), before) and counted[0] == 2
    assert _diag(_by_programs) == dict(misses=2, hits=1)


def test_a_storage_rebound_onto_an_input_takes_the_check_and_is_refused(counted):
    x, y = _values(), torch.full((BLOCK,), -7.0, device="mps")
    _launch(_reversed, (x, y))
    _launch(_reversed, (x, y))
    assert counted[0] == 1
    # The same output object, rebound to the input's storage: an in-place reversal.
    y.set_(x.untyped_storage(), 0, (BLOCK,))
    with pytest.raises(MetalNonRecoverableError, match="lane order") as refusal:
        _launch(_reversed, (x, y))
    assert "argument(s) X, Y" in str(refusal.value) and counted[0] == 2
    assert torch.equal(x.cpu(), _values("cpu"))
    # The output as the input itself is the same refusal; a disjoint view of the same
    # storage is apart and computes; a second element width over the same bytes is refused.
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_reversed, (x, x))
    wide = torch.cat([_values("cpu"), torch.full((BLOCK,), -7.0)]).to("mps")
    head = torch.empty(0, device="mps").set_(wide.untyped_storage(), 0, (BLOCK,))
    tail = torch.empty(0, device="mps").set_(wide.untyped_storage(), BLOCK, (BLOCK,))
    _launch(_reversed, (head, tail))
    assert torch.equal(tail.cpu(), _values("cpu").flip(0)) and counted[0] == 4
    _launch(_reversed, (head, tail))
    assert counted[0] == 4
    assert _diag(_reversed) == dict(misses=4, hits=2)


def test_a_hook_that_rebinds_a_storage_is_seen_by_the_next_launch(counted):
    # The check runs before the hook, so what a hook changes reaches the next launch's
    # key, and a registered hook does not stop the memo (unlike the matmul checks).
    x, y, z = _values(), torch.full((BLOCK,), -7.0, device="mps"), torch.full((BLOCK,), -7.0, device="mps")
    _launch(_reversed, (x, y))
    seen = []

    def rebind_z(launch_metadata):
        seen.append(launch_metadata)
        z.set_(x.untyped_storage(), 0, (BLOCK,))

    hooks = triton.knobs.runtime.launch_enter_hook
    hooks.add(rebind_z)
    try:
        _launch(_reversed, (x, y))
        assert counted[0] == 1 and len(seen) == 1
        with pytest.raises(MetalNonRecoverableError, match="lane order"):
            _launch(_reversed, (x, z))
        assert counted[0] == 2 and len(seen) == 1
    finally:
        hooks.remove(rebind_z)
    assert torch.equal(x.cpu(), _values("cpu")) and _diag(_reversed) == dict(misses=2, hits=1)


class _Count(int):
    """An integer that is not an exact int: the memo was not reviewed for it."""


class _Wrapped(torch.Tensor):
    """A tensor subclass that may answer the storage questions as it likes."""


@pytest.mark.parametrize("unsupported", ["tensor subclass", "int subclass"])
def test_unsupported_inputs_take_the_check_every_time(unsupported, counted):
    x = _values()
    if unsupported == "tensor subclass":
        kernel, args = _reversed, (x.as_subclass(_Wrapped), torch.empty(BLOCK, device="mps"))
    else:
        kernel, args = _wrapped, (x, torch.empty(BLOCK, device="mps"), _Count(BLOCK))
    for _ in range(3):
        _launch(kernel, args)
    expected = _values("cpu").flip(0) if unsupported == "tensor subclass" else _values("cpu") * 2
    assert torch.equal(args[1].cpu(), expected)
    assert counted[0] == 3 and _diag(kernel) == dict(bypasses=3)


def test_a_parameter_is_remembered_like_a_tensor(counted):
    """A compiled model passes its weights as Parameters; their storage facts are the plain tensor's."""
    weight = torch.nn.Parameter(_values(), requires_grad=False)
    y = torch.empty(BLOCK, device="mps")
    for _ in range(3):
        _launch(_reversed, (weight, y))
        assert torch.equal(y.cpu(), _values("cpu").flip(0))
    assert counted[0] == 1 and _diag(_reversed) == dict(misses=1, hits=2)
    # Rebound onto the weight's storage it is refused, as a tensor would be.
    y.set_(weight.untyped_storage(), 0, (BLOCK,))
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_reversed, (weight, y))
    assert counted[0] == 2


def test_entries_are_bounded_and_cleared(counted, monkeypatch):
    monkeypatch.setattr(_launch_memo, "LIMIT", 3)
    x = _values()
    outputs = [torch.empty(BLOCK, device="mps") for _ in range(4)]
    for y in outputs:
        _launch(_reversed, (x, y))
    assert counted[0] == 4 and _diag(_reversed) == dict(misses=4, evictions=1)
    _launch(_reversed, (x, outputs[0]))
    _launch(_reversed, (x, outputs[0]))
    assert counted[0] == 5 and _diag(_reversed) == dict(misses=5, evictions=1, hits=1)


def test_fresh_outputs_under_one_launcher(counted, record_property):
    """A new output every call, as a compiled model allocates its intermediates."""
    x = _values()
    calls = 40
    for _ in range(calls):
        y = torch.empty(BLOCK, device="mps")
        _launch(_reversed, (x, y))
        assert torch.equal(y.cpu(), _values("cpu").flip(0))
        del y
    found = _diag(_reversed)
    assert found.get("hits", 0) + found.get("misses", 0) == calls and "bypasses" not in found
    assert counted[0] == found.get("misses", 0)
    record_property("lane_order_hits", found.get("hits", 0))
    record_property("lane_order_misses", found.get("misses", 0))


def _outcome(launch, output):
    try:
        launch()
    except MetalNonRecoverableError as refusal:
        return ("refused", str(refusal).split(". ")[0])
    return ("computed", hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest())


@pytest.mark.parametrize("device", DEVICES)
def test_verify_mode_agrees_with_the_uncached_path_on_a_random_sequence(device, counted, monkeypatch):
    rng = random.Random(20261124)
    storages = [torch.cat([_values("cpu", seed), torch.full((BLOCK,), -7.0)]).to(device) for seed in range(3)]

    def view(step, which):
        base = storages[step[which]]
        return torch.empty(0, device=device).set_(base.untyped_storage(), step[which + "_offset"], (BLOCK,))

    plan = []
    for _ in range(200):
        kernel = rng.choice(KERNELS)
        step = dict(
            kernel=kernel,
            x=rng.randrange(3),
            x_offset=rng.choice((0, BLOCK)),
            y=rng.randrange(3),
            y_offset=rng.choice((0, BLOCK)),
            grid=rng.choice((1, 1, 2)),
            m=rng.choice((BLOCK, BLOCK, BLOCK // 2)),
        )
        plan.append(step)

    def run(step):
        for seed, base in enumerate(storages):
            base[:BLOCK].copy_(_values("cpu", seed))
            base[BLOCK:].fill_(-7.0)
        x, y = view(step, "x"), view(step, "y")
        args = (x, y, step["m"]) if step["kernel"] is _wrapped else (x, y)
        return _outcome(lambda: _launch(step["kernel"], args, step["grid"], device), storages[step["y"]])

    monkeypatch.setattr(_launch_memo, "ENABLED", False)
    uncached = [run(step) for step in plan]
    assert len({outcome[0] for outcome in uncached}) == 2, "the sequence must both compute and refuse"
    checks_uncached = counted[0]
    _launch_memo.reset_diagnostics()

    monkeypatch.setattr(_launch_memo, "ENABLED", True)
    monkeypatch.setattr(_launch_memo, "VERIFY", True)
    remembered = [run(step) for step in plan]
    assert remembered == uncached
    # Test mode runs the check on every hit, so the count doubles exactly, while the
    # memo did find launches to remember.
    assert counted[0] == 2 * checks_uncached
    found = {}
    for kernel in KERNELS:
        for event, count in _diag(kernel).items():
            found[event] = found.get(event, 0) + count
    assert found.get("hits", 0) > 0 and "bypasses" not in found
    assert found.get("hits", 0) + found.get("misses", 0) == len(plan)


def test_test_mode_raises_a_disagreement_instead_of_a_refusal(counted, monkeypatch):
    x, y = _values(), torch.empty(BLOCK, device="mps")
    _launch(_reversed, (x, y))
    monkeypatch.setattr(_launch_memo, "VERIFY", True)
    # The launcher wraps a failure of the check into a refusal; a disagreement is not wrapped.
    monkeypatch.setattr(_lane_order, "unproved", lambda *args, **kwargs: ("a reason", ()))
    with pytest.raises(_launch_memo.LaunchMemoDisagreement, match="lane_order"):
        _launch(_reversed, (x, y))
