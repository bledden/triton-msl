"""Metal kernel autotuning infrastructure.

Sweeps kernel configuration parameters (block_size, tile sizes, etc.)
and selects the best-performing configuration using GPU-timed benchmarks.

Results are cached to disk so repeated runs skip the tuning step.

Usage:
    from triton_msl.autotuning import MetalAutotuner, AutotuneConfig

    configs = [
        AutotuneConfig(block_size=128),
        AutotuneConfig(block_size=256),
        AutotuneConfig(block_size=512),
        AutotuneConfig(block_size=1024),
    ]
    tuner = MetalAutotuner(configs)
    best = tuner.tune(make_vector_add_kernel, "vector_add",
                       buffers=[a, b, out, n_buf], n_elements=n)
    # best.block_size is now the fastest option
"""

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional


@dataclass
class AutotuneConfig:
    """A single autotuning configuration to try.

    Attributes:
        block_size: Threads per threadgroup (128, 256, 512, 1024).
        num_warps: Warp count (block_size / 32).
        tile_m: Tile size in M dimension (for matmul-type kernels).
        tile_n: Tile size in N dimension.
        tile_k: Tile size in K dimension.
        extra: Arbitrary extra parameters passed to kernel generator.
    """
    block_size: int = 256
    num_warps: int = 0  # auto-computed from block_size if 0
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.num_warps == 0:
            self.num_warps = self.block_size // 32

    def to_kwargs(self):
        """Convert to keyword arguments for a kernel generator function."""
        kwargs = {"block_size": self.block_size}
        if self.tile_m > 0:
            kwargs["block_m"] = self.tile_m
        if self.tile_n > 0:
            kwargs["block_n"] = self.tile_n
        if self.tile_k > 0:
            kwargs["block_k"] = self.tile_k
        kwargs.update(self.extra)
        return kwargs

    def signature(self):
        """Return a hashable signature for this config."""
        d = asdict(self)
        return json.dumps(d, sort_keys=True)


@dataclass
class AutotuneResult:
    """Result of an autotuning run."""
    best_config: AutotuneConfig
    best_time_us: float
    all_results: List[dict]  # [{config: ..., median_us: ..., min_us: ...}, ...]


class MetalAutotuner:
    """Sweep kernel configurations and select the fastest.

    Compiles each config, runs GPU-timed benchmarks, and returns the
    best configuration. Results are cached to disk.
    """

    def __init__(self, configs: List[AutotuneConfig],
                 cache_dir: Optional[str] = None,
                 warmup: int = 10, rep: int = 50):
        """
        Args:
            configs: List of configurations to try.
            cache_dir: Directory for caching results. Defaults to
                /tmp/triton_msl_autotune_cache.
            warmup: Warmup iterations per config.
            rep: Timed iterations per config.
        """
        self.configs = configs
        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "triton_msl_autotune_cache"
        )
        self.warmup = warmup
        self.rep = rep
        os.makedirs(self.cache_dir, exist_ok=True)

    def tune(self, kernel_gen_fn: Callable, kernel_name: str,
             buffers: list, n_elements: int,
             dispatch_fn: Optional[Callable] = None) -> AutotuneResult:
        """Run autotuning and return the best configuration.

        Args:
            kernel_gen_fn: Function that takes **config.to_kwargs() and
                returns MSL source code.
            kernel_name: Name of the kernel function in the MSL source.
            buffers: List of Metal buffers to pass to the kernel.
            n_elements: Total number of elements (used for grid sizing
                when dispatch_fn is None).
            dispatch_fn: Optional custom dispatch function. If provided,
                called as dispatch_fn(pipeline, buffers, config) for each
                iteration. If None, uses standard 1D dispatch.

        Returns:
            AutotuneResult with the best config and timing data.
        """
        import Metal

        # Check cache first
        cache_key = self._cache_key(kernel_gen_fn, kernel_name, n_elements)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        device = Metal.MTLCreateSystemDefaultDevice()
        queue = device.newCommandQueue()

        all_results = []
        for config in self.configs:
            try:
                # Generate and compile kernel with this config
                kwargs = config.to_kwargs()
                msl_src = kernel_gen_fn(**kwargs)
                pipeline = self._compile(device, msl_src, kernel_name)

                # Benchmark
                median_us = self._bench(
                    queue, pipeline, buffers, n_elements,
                    config.block_size, dispatch_fn, config
                )
                all_results.append({
                    "config": config,
                    "median_us": median_us,
                    "status": "ok",
                })
            except Exception as e:
                all_results.append({
                    "config": config,
                    "median_us": float("inf"),
                    "status": f"error: {e}",
                })

        # Select best
        best = min(all_results, key=lambda r: r["median_us"])
        result = AutotuneResult(
            best_config=best["config"],
            best_time_us=best["median_us"],
            all_results=all_results,
        )

        # Save to cache
        self._save_cache(cache_key, result)
        return result

    def _compile(self, device, msl_src, kernel_name):
        """Compile MSL source to a pipeline state."""
        import Foundation

        compile_cache = os.path.join(
            tempfile.gettempdir(), "triton_msl_autotune_compile"
        )
        os.makedirs(compile_cache, exist_ok=True)

        src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
        base = f"{kernel_name}_{src_hash}"
        metal_path = os.path.join(compile_cache, f"{base}.metal")
        air_path = os.path.join(compile_cache, f"{base}.air")
        metallib_path = os.path.join(compile_cache, f"{base}.metallib")

        if not os.path.exists(metallib_path):
            with open(metal_path, "w") as f:
                f.write(msl_src)
            subprocess.check_call(
                ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
                 "-o", air_path, "-std=metal3.2", "-O2"],
                stderr=subprocess.PIPE,
            )
            subprocess.check_call(
                ["xcrun", "-sdk", "macosx", "metallib", air_path,
                 "-o", metallib_path],
                stderr=subprocess.PIPE,
            )

        url = Foundation.NSURL.fileURLWithPath_(metallib_path)
        library, error = device.newLibraryWithURL_error_(url, None)
        if error is not None:
            raise RuntimeError(f"Library load failed: {error}")

        function = library.newFunctionWithName_(kernel_name)
        if function is None:
            raise RuntimeError(f"Kernel '{kernel_name}' not found in library")

        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if error is not None:
            raise RuntimeError(f"Pipeline creation failed: {error}")
        return pipeline

    def _bench(self, queue, pipeline, buffers, n_elements, block_size,
               dispatch_fn, config):
        """Benchmark a single configuration and return median time in us."""
        import Metal

        n_groups = (n_elements + block_size - 1) // block_size

        def dispatch():
            cmd = queue.commandBuffer()
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
            return cmd

        # Warmup
        for _ in range(self.warmup):
            if dispatch_fn:
                dispatch_fn(pipeline, buffers, config)
            else:
                dispatch()

        # Timed runs
        gpu_times_us = []
        for _ in range(self.rep):
            if dispatch_fn:
                cmd = dispatch_fn(pipeline, buffers, config)
            else:
                cmd = dispatch()
            gpu_start = cmd.GPUStartTime()
            gpu_end = cmd.GPUEndTime()
            gpu_times_us.append((gpu_end - gpu_start) * 1e6)

        gpu_times_us.sort()
        return gpu_times_us[len(gpu_times_us) // 2]

    def _cache_key(self, kernel_gen_fn, kernel_name, n_elements):
        """Generate a cache key from the kernel function and parameters."""
        config_sigs = [c.signature() for c in self.configs]
        combined = f"{kernel_name}:{n_elements}:{json.dumps(config_sigs)}"
        return hashlib.sha256(combined.encode()).hexdigest()[:24]

    def _load_cache(self, key):
        """Load cached autotuning result, or None."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            # Reconstruct AutotuneConfig
            best_cfg = AutotuneConfig(**data["best_config"])
            return AutotuneResult(
                best_config=best_cfg,
                best_time_us=data["best_time_us"],
                all_results=[],  # don't cache individual results
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _save_cache(self, key, result: AutotuneResult):
        """Save autotuning result to disk cache."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        data = {
            "best_config": asdict(result.best_config),
            "best_time_us": result.best_time_us,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Common config presets
# ---------------------------------------------------------------------------

ELEMENTWISE_CONFIGS = [
    AutotuneConfig(block_size=128),
    AutotuneConfig(block_size=256),
    AutotuneConfig(block_size=512),
    AutotuneConfig(block_size=1024),
]

REDUCTION_CONFIGS = [
    AutotuneConfig(block_size=128),
    AutotuneConfig(block_size=256),
    AutotuneConfig(block_size=512),
]

MATMUL_CONFIGS = [
    AutotuneConfig(block_size=128, tile_m=16, tile_n=16, tile_k=16),
    AutotuneConfig(block_size=128, tile_m=32, tile_n=32, tile_k=32),
    AutotuneConfig(block_size=256, tile_m=32, tile_n=32, tile_k=32),
]

ATTENTION_CONFIGS = [
    AutotuneConfig(block_size=128),
    AutotuneConfig(block_size=256),
    AutotuneConfig(block_size=512),
]
