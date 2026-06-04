"""Roofline analysis for the hardware harness (WS0/C6).

Turns a measured (bytes, flops, seconds) tuple into a roofline
classification: arithmetic intensity, achieved bandwidth / compute, the
fraction of each hardware roof reached, and whether the kernel is
memory-bound or compute-bound. This is the empirical definition of "how
close to the hardware bound are we" — combined with the static disassembly
metrics (registers / occupancy / MMA presence) in ``disasm.py``, it tells us
*which* bound a kernel is hitting.

Hardware roofs are for the Apple M4 Max GPU (REFERENCES.md [13]). The memory
bandwidth (546 GB/s) is Apple-published. The compute roofs are **estimates**
(Apple does not publish GPU FLOP rates): 40 cores x 128 ALUs/core x 2
(FMA) x ~1.4 GHz. They are marked as estimates and are overridable, because
"% of peak compute" should never be presented as more precise than the roof
it's measured against.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class HardwareRoofs:
    """Hardware roofline constants. Defaults: Apple M4 Max GPU."""

    name: str = "Apple M4 Max"
    # Apple-published.
    mem_bw_gbps: float = 546.0
    # ESTIMATES (Apple does not publish GPU FLOPs):
    #   40 cores * 128 ALUs * 2 (FMA) * 1.4 GHz ~= 14.3 TFLOP/s FP32.
    #   Apple GPUs run FP16 at ~2x FP32 throughput.
    fp32_tflops: float = 14.3
    fp16_tflops: float = 28.6
    fp32_is_estimate: bool = True
    fp16_is_estimate: bool = True

    def compute_roof_tflops(self, dtype: str) -> float:
        d = (dtype or "fp32").lower()
        if d in ("fp16", "float16", "f16", "bf16", "bfloat16"):
            return self.fp16_tflops
        return self.fp32_tflops

    def compute_roof_is_estimate(self, dtype: str) -> bool:
        d = (dtype or "fp32").lower()
        if d in ("fp16", "float16", "f16", "bf16", "bfloat16"):
            return self.fp16_is_estimate
        return self.fp32_is_estimate


@dataclass(frozen=True)
class RooflineResult:
    bytes_moved: int
    flops: int
    seconds: float
    dtype: str
    # achieved
    achieved_gbps: float
    achieved_tflops: float
    arithmetic_intensity: float       # flops / byte
    # roofs
    mem_bw_gbps: float
    compute_roof_tflops: float
    ridge_point: float                # AI at which compute == bandwidth roof
    # fractions of roof reached (0..1)
    pct_of_bandwidth: float
    pct_of_compute: float
    # classification
    bound: str                        # "memory" | "compute"
    limiting_pct: float               # fraction of the *limiting* roof reached
    compute_roof_is_estimate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def classify(bytes_moved: int, flops: int, seconds: float, *,
             dtype: str = "fp32",
             roofs: Optional[HardwareRoofs] = None) -> RooflineResult:
    """Classify a measured kernel against the roofline.

    Args:
        bytes_moved: total DRAM bytes the kernel must move (read + write of
            the data that doesn't fit in cache/threadgroup). For memory-bound
            kernels this is the dominant model term.
        flops: total floating-point ops (count an FMA as 2).
        seconds: measured GPU time (e.g. from MetalBenchmark median).
        dtype: governs which compute roof applies.
        roofs: hardware roofs; defaults to M4 Max.

    Bound is decided by arithmetic intensity vs the ridge point
    (ridge = compute_roof / bandwidth_roof): below the ridge a kernel is
    memory-bound, above it compute-bound. ``limiting_pct`` is the fraction of
    whichever roof actually caps the kernel — that's the number to drive
    toward 1.0.
    """
    roofs = roofs or HardwareRoofs()
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    achieved_gbps = (bytes_moved / 1e9) / seconds
    achieved_tflops = (flops / 1e12) / seconds
    ai = (flops / bytes_moved) if bytes_moved > 0 else float("inf")

    compute_roof = roofs.compute_roof_tflops(dtype)
    ridge = compute_roof * 1e12 / (roofs.mem_bw_gbps * 1e9)  # flops/byte

    pct_bw = achieved_gbps / roofs.mem_bw_gbps if roofs.mem_bw_gbps else 0.0
    pct_compute = achieved_tflops / compute_roof if compute_roof else 0.0

    if ai < ridge:
        bound = "memory"
        limiting_pct = pct_bw
    else:
        bound = "compute"
        limiting_pct = pct_compute

    return RooflineResult(
        bytes_moved=bytes_moved,
        flops=flops,
        seconds=seconds,
        dtype=dtype,
        achieved_gbps=achieved_gbps,
        achieved_tflops=achieved_tflops,
        arithmetic_intensity=ai,
        mem_bw_gbps=roofs.mem_bw_gbps,
        compute_roof_tflops=compute_roof,
        ridge_point=ridge,
        pct_of_bandwidth=pct_bw,
        pct_of_compute=pct_compute,
        bound=bound,
        limiting_pct=limiting_pct,
        compute_roof_is_estimate=roofs.compute_roof_is_estimate(dtype),
    )


def format_roofline(name: str, r: RooflineResult) -> str:
    """One-line human summary."""
    roof_note = " (est.)" if r.compute_roof_is_estimate and r.bound == "compute" else ""
    return (
        f"{name}: {r.bound}-bound at {r.limiting_pct * 100:.1f}% of "
        f"{'bandwidth' if r.bound == 'memory' else 'compute'} roof{roof_note} "
        f"| {r.achieved_gbps:.1f} GB/s ({r.pct_of_bandwidth * 100:.0f}% of "
        f"{r.mem_bw_gbps:.0f}), {r.achieved_tflops:.2f} TFLOP/s "
        f"({r.pct_of_compute * 100:.0f}% of {r.compute_roof_tflops:.1f}), "
        f"AI={r.arithmetic_intensity:.2f} flop/byte (ridge={r.ridge_point:.2f})"
    )
