"""What the launcher does around the lane-order check when it has no proof to read.

The check itself is exercised launch by launch in test_lane_order_launch.py. These tests
pin the rules beside it: a plan is always published, a plan that cannot be read or built
refuses, facts about one shader are not used for another binary, and a launcher that was
never given a plan is left alone.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_msl  # noqa: F401
from triton_msl.backend._lane_order import SAFE, LaunchCheck, launch_plan, unproved
from triton_msl.codegen._access_plan import _SHADER_FACTS, source_only
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")


@triton.jit
def _double(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(O + i, tl.load(X + i) * 2.0)


@triton.jit
def _reverse(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(O + (BLOCK - 1 - i), tl.load(X + i))


@triton.jit
def _dot(A, B, C, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a = tl.load(A + i[:, None] * BLOCK + i[None, :])
    b = tl.load(B + i[:, None] * BLOCK + i[None, :])
    tl.store(C + i[:, None] * BLOCK + i[None, :], tl.dot(a, b))


def _launcher(kernel):
    device = list(kernel.device_caches.values())[0]
    return list(device[0].values())[-1].run


def _values(count):
    return torch.arange(count, dtype=torch.float32, device="mps") + 1.0


def test_an_in_place_launch_whose_lanes_keep_to_their_own_elements_computes():
    x = _values(64)
    _double[(1,)](x, x, BLOCK=64)
    assert torch.equal(x.cpu(), (torch.arange(64, dtype=torch.float32) + 1.0) * 2.0)
    plan = _launcher(_double)._access_plan
    assert plan["version"] >= 1 and not plan.get("why") and plan["replaced"] is False
    assert set(_SHADER_FACTS) <= set(plan)


def test_an_in_place_reversal_refuses_and_names_the_argument():
    x, out = _values(64), torch.zeros(64, device="mps")
    _reverse[(1,)](x, out, BLOCK=64)
    assert torch.equal(out.cpu(), torch.flip(x.cpu(), [0]))
    before = x.clone()
    with pytest.raises(MetalNonRecoverableError, match=r"lane order: .*argument\(s\) X, O: two lanes"):
        _reverse[(1,)](x, x, BLOCK=64)
    assert torch.equal(x, before)


@triton.jit
def _double_inside(X, O, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(O + i, tl.load(X + i, mask=m) * 2.0, mask=m)


@pytest.mark.parametrize("count,programs", [(61, 1), (64, 1), (253, 4), (40, 1), (130, 3)])
def test_two_halves_of_one_buffer_compute_when_the_mask_keeps_them_apart(count, programs):
    # The block reaches past the half it stores into; the mask is what stops it there.
    # A mask that follows the program id is weighed program by program.
    host = torch.arange(2 * count, dtype=torch.float32) + 1.0
    buffer = host.to("mps")
    _double_inside[(programs,)](buffer[count:], buffer[:count], count, BLOCK=64)
    want = host.clone()
    want[:count] = host[count:] * 2.0
    assert torch.equal(buffer.cpu(), want)
    plan = _launcher(_double_inside)._access_plan
    found = LaunchCheck(plan).verdicts([buffer[count:], buffer[:count], count], (programs, 1, 1))
    assert [answer for answer, _ in found] == ["lanes"]


def test_two_overlapping_views_under_a_mask_still_refuse():
    count = 61
    buffer = (torch.arange(2 * count, dtype=torch.float32) + 1.0).to("mps")
    before = buffer.clone()
    # The input starts one cell above the output: lane i stores what lane i - 1 reads.
    with pytest.raises(MetalNonRecoverableError, match=r"lane order: .*argument\(s\) X, O: two lanes"):
        _double_inside[(1,)](buffer[1 : count + 1], buffer[:count], count, BLOCK=64)
    assert torch.equal(buffer, before)


def test_shader_facts_are_dropped_for_another_binary():
    x = _values(64)
    _double[(1,)](x, x, BLOCK=64)
    plan = _launcher(_double)._access_plan
    reduced = source_only(plan)
    assert not set(_SHADER_FACTS) & set(reduced) and reduced["accesses"] == plan["accesses"]
    grid = (1, 1, 1)
    assert [answer for answer, _ in LaunchCheck(plan).verdicts([x, x], grid)] == ["lanes"]
    # Without the shader's facts the same in-place launch has no proof left ...
    assert unproved(LaunchCheck(reduced), [x, x], grid) is not None
    # ... and a launch that stores into memory nothing else reads still has one.
    out = torch.zeros(64, device="mps")
    answers = [answer for answer, _ in LaunchCheck(reduced).verdicts([x, out], grid)]
    assert sorted(answers) == ["no-write", "one-store"]
    assert unproved(LaunchCheck(reduced), [x, out], grid) is None


def test_the_launcher_reduces_the_plan_on_the_other_binary_route():
    from types import SimpleNamespace

    x = _values(64)
    _double[(1,)](x, x, BLOCK=64)
    plan = _launcher(_double)._access_plan
    assert launch_plan(SimpleNamespace(access_plan=plan, binary_route="msl")) is plan
    assert launch_plan(SimpleNamespace(access_plan=plan)) is plan
    assert launch_plan(SimpleNamespace(access_plan=plan, binary_route="cpp")) == source_only(plan)
    assert launch_plan(SimpleNamespace(binary_route="cpp")) is None


@pytest.mark.parametrize(
    "plan,reason",
    [
        ({"version": 0, "why": "not built: KeyError"}, "could not be read"),
        ("not a plan", "could not be read"),
    ],
)
def test_a_plan_that_cannot_be_read_refuses_every_launch(plan, reason):
    x, out = _values(64), torch.zeros(64, device="mps")
    _double[(1,)](x, out, BLOCK=64)
    launcher = _launcher(_double)
    kept = launcher._access_plan, launcher._lane_check
    launcher._access_plan, launcher._lane_check = plan, None
    try:
        with pytest.raises(MetalNonRecoverableError, match="lane order: .*" + reason):
            _double[(1,)](x, out, BLOCK=64)
    finally:
        launcher._access_plan, launcher._lane_check = kept
    _double[(1,)](x, out, BLOCK=64)


def test_a_check_that_fails_refuses():
    x, out = _values(64), torch.zeros(64, device="mps")
    _double[(1,)](x, out, BLOCK=64)
    launcher = _launcher(_double)

    class Broken:
        def verdicts(self, arguments, grid):
            raise ZeroDivisionError

    kept = launcher._lane_check
    launcher._lane_check = Broken()
    try:
        with pytest.raises(
            MetalNonRecoverableError, match=r"lane order: .*the check itself failed \(ZeroDivisionError\)"
        ):
            _double[(1,)](x, out, BLOCK=64)
    finally:
        launcher._lane_check = kept


def test_a_plan_with_a_reason_it_proves_nothing_refuses_shared_memory_only_by_that_reason():
    x = _values(64)
    _double[(1,)](x, x, BLOCK=64)
    plan = dict(_launcher(_double)._access_plan, why="made up")
    found = unproved(LaunchCheck(plan), [x, x], (1, 1, 1))
    assert found is not None and "made up" in found[0]


def test_every_answer_is_a_proof_or_has_a_reason():
    from triton_msl.backend import _lane_order

    assert set(SAFE).isdisjoint(_lane_order._REASONS)
    assert "untraced" in _lane_order._REASONS and "lanes-meet" in _lane_order._REASONS


def test_a_replacement_shader_is_marked_and_left_to_its_own_checks():
    n = 64
    a = torch.randn(n, n, device="mps")
    b = torch.randn(n, n, device="mps")
    c = torch.zeros(n, n, device="mps")
    _dot[(1,)](a, b, c, BLOCK=n)
    torch.testing.assert_close(c.cpu(), a.cpu() @ b.cpu(), rtol=1e-4, atol=1e-4)
    launcher = _launcher(_dot)
    plan = launcher._access_plan
    if plan["replaced"]:
        # The shader is not this kernel lowered operation by operation, so the plan
        # says so and the launcher does not build a check from it.
        assert launcher._lane_check is None
    else:
        assert launcher._lane_check is not None


def test_a_launcher_that_was_given_no_plan_is_not_checked():
    x, out = _values(64), torch.zeros(64, device="mps")
    _double[(1,)](x, out, BLOCK=64)
    launcher = _launcher(_double)
    kept = launcher._access_plan, launcher._lane_check
    launcher._access_plan, launcher._lane_check = None, None
    try:
        _double[(1,)](x, out, BLOCK=64)
        assert launcher._lane_check is None
    finally:
        launcher._access_plan, launcher._lane_check = kept
