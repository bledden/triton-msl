# GPU profiling beyond the harness: Instruments & Xcode GPU capture

The hardware harness (`benchmarks/hw_harness.py`, WS0/C6) gives you everything
that is available *programmatically* on Apple Silicon:

- GPU-timestamp timing → roofline (% of the 546 GB/s memory roof / compute
  roof), memory- vs compute-bound classification;
- pipeline-reflection occupancy (max threads/threadgroup, execution width,
  threadgroup-memory footprint);
- best-effort native-AGX disassembly (register/instruction signal, partial on
  M4 — see `triton_msl/profiling/disasm.py`);
- MLX comparison ratios.

Two things it **cannot** give you, because Apple does not expose them through
any public programmatic API — this document is where they live instead.

## Why these aren't in the harness

1. **Live GPU performance counters** — ALU utilization, live occupancy,
   register pressure, cache hit rates. On the M4 the device's only counter set
   is `timestamp` (`MTLDevice.counterSets()` → `['timestamp']`). The
   `MTLCommonCounterSetStageUtilization` / `…Statistic` constants exist in the
   framework but the device does not vend them, so `MTLCounterSampleBuffer`
   cannot sample them. This is a Metal *public-API* limitation, not a pyobjc
   one — a Swift/Objective-C++ helper hits the identical wall. The Metal
   driver's `n_regs` is a hardcoded placeholder (`driver.py`) for the same
   reason: "Apple doesn't expose per-kernel register usage to client code."

2. **Complete, authoritative native-ISA disassembly** — the vendored
   `applegpu` decoder targets the M1-era AGX ISA; the M4 is AGX2, so harness
   disassembly is partial (the harness reports a decode-coverage %). Apple's
   own tooling has the full, current decoder.

Both are accessible through Apple's **offline GPU tooling**, below.

## Live counters & occupancy: Instruments "Metal System Trace"

1. Build/run a script that dispatches the kernel under test (the harness
   itself works, or any `@triton.jit` driver).
2. `xcrun xctrace record --template "Metal System Trace" --launch -- \
   <python> <script>` — or open **Instruments.app → Metal System Trace**,
   target the Python process, and record.
3. In the trace: the **GPU track** shows per-encoder occupancy, ALU/FMA
   utilization, memory bandwidth, and limiter reasons (register-limited vs
   threadgroup-memory-limited vs thread-count-limited).
4. Export the relevant counters; cross-reference with the harness's roofline
   call for the same kernel (the harness already tells you *which* roof binds;
   the trace tells you *why* — e.g. occupancy capped by registers).

This is the authoritative source for the ALU%/occupancy numbers the harness
marks as unavailable.

## Authoritative native-ISA disassembly: Xcode GPU Frame Capture

1. Xcode → **Debug → Capture GPU Frame** (or `MTLCaptureManager` programmatic
   capture around the dispatch).
2. In the capture, select the compute pipeline → **Pipeline Statistics** /
   the shader disassembly view shows the *current* native AGX with Apple's
   full decoder (register allocation, spills, instruction selection, whether
   `simdgroup_matrix` lowered to the hardware MMA path).
3. Compare against the harness's best-effort `applegpu` disassembly: where the
   harness coverage is low, the Xcode view is ground truth.

## How this fits the WS1 optimization loop

"Optimal bounds given by the hardware" is defined empirically:

1. **Harness first** — `python benchmarks/hw_harness.py` gives the roofline
   %, the bound (memory/compute), MLX ratio, and occupancy hint for every
   kernel. This finds *where* the gap is (e.g. `reduce_sum` at 16% of the
   bandwidth roof, 1.3× slower than MLX → a clear target).
2. **Instruments/Xcode when the harness isn't enough** — for a kernel the
   harness flags as underperforming, capture it here to see the limiter
   reason (registers? occupancy? a missing MMA lowering?) that the live
   counters expose and the programmatic API does not.
3. **Fix, re-run the harness, diff against `baseline.json`.** A WS1 change is
   "done" for a kernel when its limiting counter saturates — verified in the
   harness roofline and, where needed, the Instruments trace.

## References

See `REFERENCES.md`: Apple GPU counter APIs [9], Metal Shading Language
spec [8], Asahi/AGX [10], `applegpu` [11], M4 Max hardware reference [13].
