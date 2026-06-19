"""Metal-specific benchmarking using GPU timestamps.

Uses MTLCommandBuffer.GPUStartTime/GPUEndTime for nanosecond-precision
GPU timing. Falls back to wall-clock time if GPU timestamps are unavailable.
"""

import time


def metal_do_bench(fn, *, quantiles=None, warmup=25, rep=100, **kwargs):
    """Benchmark a function with GPU-precise timing.

    Compatible with Triton's benchmarker interface (registered via
    MetalDriver.get_benchmarker()).

    Uses MTLCommandBuffer.GPUStartTime/GPUEndTime when the function
    returns a command buffer, otherwise falls back to wall-clock timing.

    Args:
        fn: Callable to benchmark. Should dispatch Metal work.
            If it returns a MTLCommandBuffer, GPU timestamps are used.
        quantiles: List of quantiles to return (e.g. [0.5, 0.2, 0.8]).
            If None, returns the median time.
        warmup: Number of warmup iterations.
        rep: Number of timed iterations.

    Returns:
        If quantiles is None: median time in milliseconds.
        Otherwise: list of times in ms corresponding to each quantile.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(rep):
        result = fn()
        # If fn returns a command buffer, use GPU timestamps
        if result is not None and hasattr(result, "GPUStartTime"):
            gpu_start = result.GPUStartTime()
            gpu_end = result.GPUEndTime()
            if gpu_end > gpu_start:
                times.append((gpu_end - gpu_start) * 1000.0)
                continue
        # Fallback: wall-clock timing (re-run since we already called fn)
        if not times or (result is None or not hasattr(result, "GPUStartTime")):
            # For wall-clock mode, we need to re-measure properly
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    times.sort()
    if quantiles is None:
        return times[len(times) // 2]

    result = []
    for q in quantiles:
        idx = int(q * (len(times) - 1))
        result.append(times[idx])
    return result


class MetalBenchmark:
    """GPU-timed benchmark runner using Metal command buffer timestamps.

    Unlike metal_do_bench (which measures wall-clock time including CPU
    overhead), this class measures pure GPU execution time using
    MTLCommandBuffer.GPUStartTime/GPUEndTime.
    """

    def __init__(self):
        import Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()

    def time_kernel(self, pipeline, buffers, n_elements, block_size=256,
                    warmup=10, rep=100):
        """Time a compute kernel dispatch using GPU timestamps.

        Args:
            pipeline: MTLComputePipelineState.
            buffers: List of MTLBuffer objects to bind.
            n_elements: Total elements (determines grid size).
            block_size: Threads per threadgroup.
            warmup: Warmup iterations.
            rep: Timed iterations.

        Returns:
            dict with keys: median_us, min_us, max_us, p10_us, p90_us,
            throughput_gb_s (if applicable), all_us.
        """
        import Metal

        n_groups = (n_elements + block_size - 1) // block_size

        # Warmup
        for _ in range(warmup):
            self._dispatch(pipeline, buffers, n_groups, block_size)

        # Timed runs
        gpu_times_us = []
        for _ in range(rep):
            cmd = self.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            for i, buf in enumerate(buffers):
                enc.setBuffer_offset_atIndex_(buf, 0, i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_groups, 1, 1),
                Metal.MTLSizeMake(block_size, 1, 1),
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            gpu_start = cmd.GPUStartTime()
            gpu_end = cmd.GPUEndTime()
            gpu_times_us.append((gpu_end - gpu_start) * 1e6)

        gpu_times_us.sort()
        n = len(gpu_times_us)

        return {
            "median_us": gpu_times_us[n // 2],
            "min_us": gpu_times_us[0],
            "max_us": gpu_times_us[-1],
            "p10_us": gpu_times_us[int(0.1 * (n - 1))],
            "p90_us": gpu_times_us[int(0.9 * (n - 1))],
            "all_us": gpu_times_us,
        }

    def _dispatch(self, pipeline, buffers, n_groups, block_size):
        """Dispatch a kernel (no timing)."""
        import Metal

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_groups, 1, 1),
            Metal.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


def compute_throughput(n_bytes, time_us):
    """Compute throughput in GB/s from byte count and time in microseconds."""
    if time_us <= 0:
        return float("inf")
    return (n_bytes / 1e9) / (time_us / 1e6)


def compute_gflops(n_flops, time_us):
    """Compute GFLOP/s from flop count and time in microseconds."""
    if time_us <= 0:
        return float("inf")
    return (n_flops / 1e9) / (time_us / 1e6)


def format_benchmark_result(name, result, n_bytes=None, n_flops=None):
    """Format a benchmark result as a human-readable string."""
    lines = [f"  {name}:"]
    lines.append(f"    GPU time: {result['median_us']:.1f} us "
                 f"(min={result['min_us']:.1f}, max={result['max_us']:.1f}, "
                 f"p10={result['p10_us']:.1f}, p90={result['p90_us']:.1f})")
    if n_bytes:
        bw = compute_throughput(n_bytes, result["median_us"])
        lines.append(f"    Bandwidth: {bw:.1f} GB/s")
    if n_flops:
        gf = compute_gflops(n_flops, result["median_us"])
        lines.append(f"    Compute: {gf:.1f} GFLOP/s")
    return "\n".join(lines)
