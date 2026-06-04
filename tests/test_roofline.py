"""Unit tests for the roofline analysis (WS0/C6)."""
import math

import pytest

from triton_metal.profiling.roofline import (
    HardwareRoofs, RooflineResult, classify, format_roofline,
)


def test_memory_bound_classification():
    # vector_add: 16M fp32 elements, 3 arrays (2 read + 1 write) = 192 MB,
    # ~0 flops/byte -> deeply memory-bound.
    n = 16 * 1024 * 1024
    bytes_moved = 3 * n * 4
    r = classify(bytes_moved, flops=n, seconds=bytes_moved / 1e9 / 137.5,
                 dtype="fp32")
    assert r.bound == "memory"
    # achieved ~137.5 GB/s on the default 546 roof -> ~25%
    assert 0.24 < r.pct_of_bandwidth < 0.26
    assert r.limiting_pct == r.pct_of_bandwidth


def test_compute_bound_classification():
    # A dense matmul: high arithmetic intensity -> compute-bound.
    m = k = nn = 1024
    flops = 2 * m * k * nn
    bytes_moved = (m * k + k * nn + m * nn) * 2  # fp16
    # Fast enough that AI is above the ridge and we hit compute.
    r = classify(bytes_moved, flops, seconds=flops / 1e12 / 10.0, dtype="fp16")
    assert r.bound == "compute"
    assert r.arithmetic_intensity > r.ridge_point


def test_ridge_point_is_compute_over_bandwidth():
    roofs = HardwareRoofs()
    r = classify(1000, 1, seconds=1e-6, dtype="fp32")
    expected_ridge = roofs.fp32_tflops * 1e12 / (roofs.mem_bw_gbps * 1e9)
    assert math.isclose(r.ridge_point, expected_ridge, rel_tol=1e-9)


def test_fp16_roof_is_double_fp32():
    roofs = HardwareRoofs()
    assert math.isclose(roofs.compute_roof_tflops("fp16"),
                        2 * roofs.compute_roof_tflops("fp32"), rel_tol=1e-9)


def test_compute_roof_marked_estimate():
    roofs = HardwareRoofs()
    assert roofs.compute_roof_is_estimate("fp32") is True
    assert roofs.compute_roof_is_estimate("fp16") is True


def test_achieved_bandwidth_matches_input():
    # 100 GB moved in 1 s -> 100 GB/s exactly.
    r = classify(bytes_moved=100_000_000_000, flops=0, seconds=1.0)
    assert math.isclose(r.achieved_gbps, 100.0, rel_tol=1e-9)


def test_zero_seconds_rejected():
    with pytest.raises(ValueError):
        classify(1000, 1000, seconds=0.0)


def test_overridable_roofs():
    custom = HardwareRoofs(name="Custom", mem_bw_gbps=1000.0,
                           fp32_tflops=50.0, fp32_is_estimate=False)
    r = classify(1_000_000_000, 0, seconds=1.0, dtype="fp32", roofs=custom)
    assert r.mem_bw_gbps == 1000.0
    assert r.compute_roof_tflops == 50.0
    assert r.compute_roof_is_estimate is False


def test_format_roofline_is_readable():
    r = classify(3 * 16 * 1024 * 1024 * 4, 16 * 1024 * 1024,
                 seconds=0.0014, dtype="fp32")
    s = format_roofline("vector_add_16M", r)
    assert "vector_add_16M" in s and "bound" in s and "GB/s" in s
