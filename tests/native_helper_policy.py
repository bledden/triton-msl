"""Availability policy for tests of optional, selected native helpers.

A pure artifact may skip only after import machinery proves helper absence.
A discoverable/loaded helper that was not selected must fail the native tests,
including a binder whose configure() declined. Import/discovery failures propagate.

Installed native lanes should set TRITON_MSL_TEST_REQUIRE_VALIDATION_NATIVE=1
and TRITON_MSL_TEST_REQUIRE_BINDER_NATIVE=1 as applicable. An explicit matching
TRITON_MSL_BUILD_*_NATIVE=1 also requires selection. These flags affect tests only;
they do not change backend selection or supported Python/platform combinations.
"""

import importlib.util
import os
import sys

import pytest


def require_selected_native(selected, helper):
    """Continue with a selected helper, or skip only genuinely optional absence."""
    if helper not in ("validation", "binder"):
        raise ValueError(f"unknown native test helper: {helper}")
    requirement = f"TRITON_MSL_TEST_REQUIRE_{helper.upper()}_NATIVE"
    mode = os.environ.get(requirement, "0")
    if mode not in ("0", "1"):
        pytest.fail(f"{requirement} must be 0 or 1")
    if selected is not None:
        return
    name = f"triton_msl.backend._{helper}_native"
    expected = mode == "1" or os.environ.get(f"TRITON_MSL_BUILD_{helper.upper()}_NATIVE") == "1"
    if expected:
        pytest.fail(f"{name} was required but is not selected")
    # Membership matters: a None sentinel in sys.modules is a blocked/broken
    # import, not proof that the optional image is absent. Do not retry imports
    # or configure a declined binder just to turn a failing control into a pass.
    if name in sys.modules or importlib.util.find_spec(name) is not None:
        pytest.fail(f"{name} is present but is not selected")
    pytest.skip(f"Optional native helper is genuinely absent: {name}")
