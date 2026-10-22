"""Keep the JSON observation order when optimizing ordinary small descriptors."""

import pytest

from triton_msl.backend import _launch_contract as contract
from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize("late_error", [False, True])
def test_early_mismatch_does_not_skip_later_custom_iterator(late_error):
    packed = [4, 1, 0, 128, [1], False, None, None, None, None, None, None]
    expected = contract._canonical(packed)
    # Prime any optional plan: the failure must remain observable on a warm call.
    assert contract.validate_packed_launch(packed, expected) == tuple(packed)
    events = []
    sentinel = RuntimeError("late custom iterator")

    class ObservedList(list):
        def __iter__(self):
            events.append("iterated")
            if late_error:
                raise sentinel
            return super().__iter__()

    packed[0] = 5
    packed[4] = ObservedList([1])
    with pytest.raises(RuntimeError if late_error else MetalNonRecoverableError) as raised:
        contract.validate_packed_launch(packed, expected)
    assert events == ["iterated"]
    if late_error:
        assert raised.value is sentinel
