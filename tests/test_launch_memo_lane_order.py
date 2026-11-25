"""A remembered lane-order proof is skipped only where the uncached launch would find it again.

The launcher remembers a lane-order proof under every value the check reads (each
argument's storage, extent, offset, width and pointer or its exact scalar value, and the
grid) and skips the check on a launch with the same values
(triton_msl.backend._launch_memo). Every launch here is run against the check itself,
counted, and against the result the source computes or the refusal the uncached path
raises: a changed grid or scalar, a storage rebound onto another, an output that is an
input, and a hook that rebinds a storage each take the check again; an argument the memo
was not reviewed for takes it every time; and test mode agrees with the uncached path
over a random sequence. A storage that changes while the check itself runs is not
remembered at all. The last tests pin inputs whose keys could be confused or go stale:
scalars that compare equal, views of another shape, empty storages, a Parameter whose
data is swapped, a hook that is not Triton's own chain and a storage resized in place.

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


@triton.jit
def _gated(X, Y, S, BLOCK: tl.constexpr):
    # S of 1 or True: each lane its own element. S of 0 or False: every lane the first one.
    a = tl.arange(0, BLOCK) * S
    tl.store(Y + a, tl.load(X + a) * 2.0)


@triton.jit
def _scaled(X, Y, s, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * s)


@triton.jit
def _reversed_first(X, Y, n, BLOCK: tl.constexpr):
    # The first n elements reversed onto the last n of the block; with n of 0 nothing is read or stored.
    i = tl.arange(0, BLOCK)
    tl.store(Y + (BLOCK - 1 - i), tl.load(X + i, mask=i < n), mask=i < n)


KERNELS = (_reversed, _wrapped, _by_programs)
REMEMBERING = KERNELS + (_gated, _scaled, _reversed_first)


def _memos(kernels=None):
    """The lane-order memo of every launcher these kernels have built."""
    for kernel in REMEMBERING if kernels is None else kernels:
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                check = getattr(getattr(compiled, "run", None), "_lane_check", None)
                memo = getattr(check, "memo", None)
                if memo is not None:
                    yield check, memo


def _clear():
    for _, memo in _memos():
        memo.entries.clear()


def _remembered(kernel):
    return sum(len(memo.entries) for _, memo in _memos((kernel,)))


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


@pytest.fixture(params=[False, True], ids=["memo", "test mode"])
def verify(request, counted, monkeypatch):
    """Run the test with the memo alone and again in test mode, where every hit runs the check too."""
    monkeypatch.setattr(_launch_memo, "VERIFY", request.param)
    return request.param


def _diag(kernel):
    return _launch_memo.diagnostics().get(kernel.fn.__name__, {}).get("lane_order", {})


def _checks(verify, *kernels):
    """How often the check must have run: on every miss and bypass, and in test mode on every hit."""
    total = 0
    for kernel in kernels:
        found = _diag(kernel)
        total += found.get("misses", 0) + found.get("bypasses", 0) + (found.get("hits", 0) if verify else 0)
    return total


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


def _view(storage, offset, shape, stride=None, device="mps"):
    view = torch.empty(0, device=device)
    return view.set_(storage, offset, shape) if stride is None else view.set_(storage, offset, shape, stride)


@pytest.mark.parametrize("device", DEVICES)
def test_a_storage_rebound_while_the_check_runs_is_not_remembered(device, counted, monkeypatch):
    """The key is read, then the check reads the arguments again: a change in between is not a proof of the key.

    The check here rebinds the output apart from the input while it runs, as another
    thread or code inside a tensor accessor could. It passes on what it read, and the
    launch computes there. The key was read from the in-place binding, which the check
    refuses, so nothing may be remembered under it.
    """
    x = _values(device)
    apart = torch.full((BLOCK,), -7.0, device=device)
    y = _view(x.untyped_storage(), 0, (BLOCK,), device=device)
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_reversed, (x, y), device=device)
    assert counted[0] == 1 and torch.equal(x.cpu(), _values("cpu"))
    check, fired = _lane_order.unproved, []

    def rebinding(*args, **kwargs):
        if not fired:
            fired.append(True)
            y.set_(apart.untyped_storage(), 0, (BLOCK,))
        return check(*args, **kwargs)

    monkeypatch.setattr(_lane_order, "unproved", rebinding)
    _launch(_reversed, (x, y), device=device)
    assert fired and counted[0] == 2
    assert torch.equal(apart.cpu(), _values("cpu").flip(0)) and torch.equal(x.cpu(), _values("cpu"))
    assert _remembered(_reversed) == 0, "a pass found on other facts was remembered under the refused ones"
    assert _diag(_reversed) == dict(misses=1, bypasses=1)
    # The changed facts are a launch of their own: checked once, then remembered.
    _launch(_reversed, (x, y), device=device)
    _launch(_reversed, (x, y), device=device)
    assert counted[0] == 3 and _remembered(_reversed) == 1
    # Back on the input's storage, with nothing changing during the launch: refused.
    y.set_(x.untyped_storage(), 0, (BLOCK,))
    with pytest.raises(MetalNonRecoverableError, match="lane order") as refusal:
        _launch(_reversed, (x, y), device=device)
    assert "argument(s) X, Y" in str(refusal.value) and counted[0] == 4
    assert torch.equal(x.cpu(), _values("cpu"))
    assert _diag(_reversed) == dict(misses=3, bypasses=1, hits=1)


def test_true_and_one_are_each_launched_as_themselves(counted, verify, monkeypatch):
    """True and 1 compare and hash equal, so within one launcher they are one key.

    The check takes int() of both and the two pass or are refused together, so a proof
    remembered for one is the other's proof. Triton compiles a bool and an int argument
    separately, so the launches themselves do not share a launcher; each launch's own
    check and arguments are recorded and the equal scalar is then put to that check.
    """
    seen = []
    remembered_unproved = _lane_order.remembered_unproved

    def recording(check, arguments, grid):
        seen.append((check, list(arguments), grid))
        return remembered_unproved(check, arguments, grid)

    monkeypatch.setattr(_lane_order, "remembered_unproved", recording)
    x = _values()
    expected = _values("cpu")
    for scalar in (1, True, 1, True):
        _launch(_gated, (x, x, scalar))
        expected = expected * 2
        assert torch.equal(x.cpu(), expected)
    assert _diag(_gated) == dict(misses=2, hits=2)
    for scalar in (0, False, False, 0):
        with pytest.raises(MetalNonRecoverableError, match="lane order") as refusal:
            _launch(_gated, (x, x, scalar))
        assert "two lanes of one program reach the same element" in str(refusal.value)
        assert torch.equal(x.cpu(), expected)
    assert _diag(_gated) == dict(misses=6, hits=2) and counted[0] == _checks(verify, _gated)
    _launch(_gated, (x, x, True))
    assert torch.equal(x.cpu(), expected * 2) and _diag(_gated) == dict(misses=6, hits=3)
    # The equal scalar under each launch's own check: one key, and the check's own answer is the same.
    twin = {(int, 1): True, (bool, True): 1, (int, 0): False, (bool, False): 0}
    swapped = set()
    for check, arguments, grid in seen:
        if not any((type(value), value) in twin for value in arguments if type(value) in (int, bool)):
            continue
        other = [twin[type(v), v] if type(v) in (int, bool) and (type(v), v) in twin else v for v in arguments]
        assert _launch_memo.lane_order_key(other, grid) == _launch_memo.lane_order_key(arguments, grid)
        check.memo.entries.clear()
        first = remembered_unproved(check, arguments, grid)
        hits = _diag(_gated)["hits"]
        second = remembered_unproved(check, other, grid)
        assert (first is None) == (second is None) == (_lane_order.unproved(check, other, grid) is None)
        assert _diag(_gated)["hits"] == hits + (1 if first is None else 0)
        swapped.add((type(next(v for v in arguments if type(v) in (int, bool))), first is None))
    assert {(bool, True), (bool, False)} <= swapped, "True and False were each put to a check as 1 and 0"


@pytest.mark.parametrize("device", DEVICES)
def test_zero_and_negative_zero_are_each_launched_as_themselves(device, counted, verify):
    """0.0 and -0.0 are one key (the check never reads a float's value); the launch still passes its own scalar."""
    x = _values(device) + 1.0
    y = torch.full((BLOCK,), -7.0, device=device)
    for scale, negative in ((0.0, False), (-0.0, True), (0.0, False), (-0.0, True)):
        _launch(_scaled, (x, y, scale), device=device)
        got = y.cpu()
        assert torch.equal(got, torch.zeros(BLOCK)) and bool(torch.signbit(got).all()) is negative
        y.fill_(-7.0)
    assert _diag(_scaled) == dict(misses=1, hits=3) and counted[0] == _checks(verify, _scaled)
    _launch(_scaled, (x, y, 3.0), device=device)
    assert torch.equal(y.cpu(), (_values("cpu") + 1.0) * 3.0) and _diag(_scaled) == dict(misses=2, hits=3)


@pytest.mark.parametrize("device", DEVICES)
def test_a_view_of_another_shape_over_the_same_storage_and_offset_is_the_same_launch(device, counted, verify):
    """The check reads the storage, the offset and the element width of a view, never its shape or strides."""
    base = torch.cat([_values("cpu"), torch.full((BLOCK,), -7.0)]).to(device)
    storage = base.untyped_storage()
    head = _view(storage, 0, (BLOCK,), device=device)
    shapes = [((BLOCK,), None), ((8, 8), None), ((4, 16), None), ((BLOCK // 2,), (2,)), ((2, 2, 16), None)]
    for shape, stride in shapes:
        tail = _view(storage, BLOCK, shape, stride, device=device)
        _launch(_reversed, (head, tail), device=device)
        assert torch.equal(base.cpu()[BLOCK:], _values("cpu").flip(0))
        assert torch.equal(base.cpu()[:BLOCK], _values("cpu"))
        base[BLOCK:] = -7.0
    assert _diag(_reversed) == dict(misses=1, hits=len(shapes) - 1) and counted[0] == _checks(verify, _reversed)
    # The input's own cells under another shape are still the input's cells: refused every time.
    for shape in ((BLOCK,), (8, 8), (8, 8)):
        with pytest.raises(MetalNonRecoverableError, match="lane order"):
            _launch(_reversed, (head, _view(storage, 0, shape, device=device)), device=device)
        assert torch.equal(base.cpu()[:BLOCK], _values("cpu"))
    assert _diag(_reversed) == dict(misses=4, hits=len(shapes) - 1) and counted[0] == _checks(verify, _reversed)


@pytest.mark.parametrize("device", DEVICES)
def test_two_empty_storages_are_one_launch_that_touches_nothing(device, counted, verify):
    """Two distinct empty storages have the same facts as one (no pointer, no bytes), and the check agrees.

    An empty storage holds no byte, so the check finds nothing shared whichever way the
    empty arguments are bound, and the launch (which reads and stores nothing at n = 0)
    is the same launch. A storage that does hold bytes is a different key.
    """
    empties = [torch.empty(0, device=device) for _ in range(4)]
    assert _launch_memo.tensor_facts(empties[0]) == _launch_memo.tensor_facts(empties[1])
    for x, y in ((empties[0], empties[1]), (empties[0], empties[0]), (empties[2], empties[3])):
        _launch(_reversed_first, (x, y, 0), device=device)
        assert x.numel() == 0 and y.numel() == 0
    assert _diag(_reversed_first) == dict(misses=1, hits=2)
    real = _values(device)
    _launch(_reversed_first, (empties[0], real, 0), device=device)
    assert torch.equal(real.cpu(), _values("cpu")) and _diag(_reversed_first) == dict(misses=2, hits=2)
    out = torch.full((BLOCK,), -7.0, device=device)
    _launch(_reversed_first, (real, out, BLOCK), device=device)
    assert torch.equal(out.cpu(), _values("cpu").flip(0))
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_reversed_first, (real, real, BLOCK), device=device)
    assert torch.equal(real.cpu(), _values("cpu"))
    assert _diag(_reversed_first) == dict(misses=4, hits=2) and counted[0] == _checks(verify, _reversed_first)


def test_a_parameter_whose_data_is_swapped_takes_the_check(counted, verify):
    """The same Parameter object over another storage is another launch: its facts are read on every launch."""
    weight = torch.nn.Parameter(_values(), requires_grad=False)
    y = torch.full((BLOCK,), -7.0, device="mps")
    for _ in range(2):
        _launch(_reversed, (weight, y))
        assert torch.equal(y.cpu(), _values("cpu").flip(0))
    assert _diag(_reversed) == dict(misses=1, hits=1)
    # Onto the output's storage: an in-place reversal, refused, and never remembered.
    kept = weight.data
    weight.data = y
    before = y.cpu().clone()
    for _ in range(2):
        with pytest.raises(MetalNonRecoverableError, match="lane order") as refusal:
            _launch(_reversed, (weight, y))
        assert "argument(s) X, Y" in str(refusal.value) and torch.equal(y.cpu(), before)
    assert _diag(_reversed) == dict(misses=3, hits=1)
    # Onto a third storage: checked anew, and computed from the new data.
    weight.data = _values(seed=5)
    _launch(_reversed, (weight, y))
    assert torch.equal(y.cpu(), _values("cpu", 5).flip(0))
    # Back on the first storage, the first launch again.
    weight.data = kept
    _launch(_reversed, (weight, y))
    assert torch.equal(y.cpu(), _values("cpu").flip(0))
    assert _diag(_reversed) == dict(misses=4, hits=2) and counted[0] == _checks(verify, _reversed)


@pytest.mark.parametrize("shape", ["plain function", "HookChain subclass"])
def test_a_hook_that_is_not_a_plain_chain_is_seen_by_the_next_launch(shape, counted, verify, monkeypatch):
    """A hook Triton did not build (a bare function, a HookChain subclass) is treated like any other here.

    The lane-order check runs before the hook whatever the hook is, so the memo is
    consulted, and what the hook rebinds is read by the next launch's key.
    """
    x, y, z = _values(), torch.full((BLOCK,), -7.0, device="mps"), torch.full((BLOCK,), -7.0, device="mps")
    _launch(_reversed, (x, y))
    seen = []

    def rebind_z(launch_metadata):
        seen.append(launch_metadata)
        z.set_(x.untyped_storage(), 0, (BLOCK,))

    if shape == "plain function":
        hook = rebind_z
    else:

        class Chain(type(triton.knobs.runtime.launch_enter_hook)):
            pass

        hook = Chain()
        hook.add(rebind_z)
    monkeypatch.setattr(triton.knobs.runtime, "launch_enter_hook", hook)
    y.fill_(-7.0)
    _launch(_reversed, (x, y))
    assert len(seen) == 1 and torch.equal(y.cpu(), _values("cpu").flip(0))
    assert _diag(_reversed) == dict(misses=1, hits=1)
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_reversed, (x, z))
    assert len(seen) == 1 and torch.equal(x.cpu(), _values("cpu"))
    assert _diag(_reversed) == dict(misses=2, hits=1) and counted[0] == _checks(verify, _reversed)


def test_a_host_storage_resized_in_place_takes_the_check(counted, verify):
    """A real ``UntypedStorage.resize_`` on the host: the storage's size (and here its address) is in the key."""
    base = torch.zeros(4 * BLOCK)
    storage = base.untyped_storage()
    x, y = _view(storage, 0, (BLOCK,), device="cpu"), _view(storage, 2 * BLOCK, (BLOCK,), device="cpu")
    x.copy_(_values("cpu"))
    for _ in range(2):
        y.fill_(-7.0)
        _launch(_reversed, (x, y), device="cpu")
        assert torch.equal(y, _values("cpu").flip(0))
    assert _diag(_reversed) == dict(misses=1, hits=1)
    for count in (3 * BLOCK, 64 * BLOCK):
        before = (_launch_memo.tensor_facts(x), _launch_memo.tensor_facts(y))
        storage.resize_(count * 4)
        assert storage.nbytes() == count * 4
        assert (_launch_memo.tensor_facts(x), _launch_memo.tensor_facts(y)) != before
        x.copy_(_values("cpu"))
        y.fill_(-7.0)
        misses = _diag(_reversed)["misses"]
        _launch(_reversed, (x, y), device="cpu")
        assert torch.equal(y, _values("cpu").flip(0)) and torch.equal(x, _values("cpu"))
        assert _diag(_reversed)["misses"] == misses + 1
    # In place over the resized storage it is refused as before.
    with pytest.raises(MetalNonRecoverableError, match="lane order"):
        _launch(_reversed, (x, _view(storage, 0, (BLOCK,), device="cpu")), device="cpu")
    assert torch.equal(x, _values("cpu"))
    assert _diag(_reversed) == dict(misses=4, hits=1) and counted[0] == _checks(verify, _reversed)


def test_a_device_storage_that_torch_will_not_resize_is_rebound(counted, verify):
    """On MPS torch refuses ``UntypedStorage.resize_``, so this test rebinds instead of resizing in place.

    torch 2.12.1 raises "UntypedStorage.resize_: got unexpected device type mps", and
    ``Tensor.resize_`` to a larger size gives the tensor a new storage (pointer and size
    both change), which is a rebinding. Should a later torch resize a device storage in
    place, the first branch runs instead and the same outcome is required: the changed
    storage takes the check, and the launch computes.
    """
    x = _values()
    y = torch.full((BLOCK,), -7.0, device="mps")
    for _ in range(2):
        _launch(_reversed, (x, y))
        assert torch.equal(y.cpu(), _values("cpu").flip(0))
    assert _diag(_reversed) == dict(misses=1, hits=1)
    before = _launch_memo.tensor_facts(y)
    try:
        y.untyped_storage().resize_(4 * BLOCK * 4)
    except RuntimeError as refused:
        assert "mps" in str(refused)
        y.resize_(4 * BLOCK)
    y = y.resize_(BLOCK) if y.numel() != BLOCK else y
    assert _launch_memo.tensor_facts(y) != before and y.untyped_storage().nbytes() == 4 * BLOCK * 4
    y.fill_(-7.0)
    _launch(_reversed, (x, y))
    assert torch.equal(y.cpu(), _values("cpu").flip(0))
    assert _diag(_reversed) == dict(misses=2, hits=1) and counted[0] == _checks(verify, _reversed)
