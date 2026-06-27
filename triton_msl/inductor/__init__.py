"""triton-msl Inductor integration.

Replaces PyTorch's default MPS backend (MetalScheduling) with TritonScheduling,
routing torch.compile() through Triton → triton-msl → MSL → Metal GPU.

Usage:
    import triton_msl.inductor
    triton_msl.inductor.register_metal_triton_backend()

    model = torch.compile(my_model)
    output = model(input)  # compiles via Triton → Metal
"""

from textwrap import dedent

from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)

_registered = False


class MetalTritonDeviceOpOverrides(DeviceOpOverrides):
    """Device op overrides for MPS when using TritonScheduling.

    Metal has no stream concept, so stream-related methods return no-ops.
    Mirrors CpuDeviceOpOverrides for stream handling.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def set_device(self, device_idx: int) -> str:
        return "pass  # MPS single device"

    def synchronize(self) -> str:
        return "pass  # MPS synchronize handled by Metal command buffer"

    def device_guard(self, device_idx: int) -> str:
        return "torch._ops.contextlib.nullcontext()"

    def cpp_kernel_type(self) -> str:
        return "void*"


def register_metal_triton_backend():
    """Replace MPS's MetalScheduling with TritonScheduling.

    Must be called before the first torch.compile() invocation.
    PyTorch's init_backend_registration() checks
    `if get_scheduling_for_device("mps") is None` before registering
    its default, so pre-registering here takes priority.
    """
    global _registered
    if _registered:
        return

    from torch._inductor.codegen.common import register_backend_for_device
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

    # Force single-process compilation. Metal / PyObjC is NOT fork-safe: once a
    # Metal device/context is initialized in the parent, a forked inductor
    # compile-worker subprocess that touches Metal crashes ("A compilation
    # subprocess exited unexpectedly"). Worse, a worker that crashes mid-write
    # can leave a corrupted entry in inductor's on-disk FX/triton cache, which a
    # later compile may load and silently return WRONG values from. Pinning
    # compilation to the main process (and disabling subprocess autotuning)
    # removes the fork entirely -- this is a correctness requirement on Metal,
    # not a perf tweak. Set on the config object (not the env var) so it holds
    # regardless of import ordering. See docs/superpowers/plans/
    # 2026-06-18-inductor-backend-port.md.
    import torch._inductor.config as _ind_config
    _ind_config.compile_threads = 1
    if hasattr(_ind_config, "autotune_in_subproc"):
        _ind_config.autotune_in_subproc = False

    # Disable inductor's multi-config kernel autotuning. It benchmarks several
    # tile configs per kernel and keeps the "fastest" by wall-clock — but on
    # Metal that is doubly unsafe:
    #   (a) Metal timing is coarse/noisy, so the winning config varies run to
    #       run (nondeterministic kernel selection), and
    #   (b) some configs it explores (sizePerThread > 1, i.e. XBLOCK/R0_BLOCK
    #       wider than the threadgroup) our flat lid-based MSL lowering does NOT
    #       cover — it silently computes the wrong reduction (the same class the
    #       _patch_reduction_configs filter and the n>1 store refusal guard
    #       against).
    # Together these produced a NONDETERMINISTIC wrong gradient — a transformer
    # `head.weight` grad off by ~0.11 on roughly 1 in 4 cold runs, exact on the
    # rest. Pinning to the single default (correct, deterministic) config makes
    # the compiled result reproducible and correct. Correctness > the autotuning
    # speedup; our real perf comes from the hand-written fast paths, not from
    # autotuning inductor's generic tiles.
    # NOTE: this gates *pointwise* autotuning only. Persistent/reduction config
    # correctness is enforced separately by the under-filling filter below (which
    # drops configs whose dispatched threads would surplus-over-count a reduction).
    if hasattr(_ind_config, "triton") and hasattr(_ind_config.triton, "autotune_pointwise"):
        _ind_config.triton.autotune_pointwise = False

    register_backend_for_device(
        "mps",
        TritonScheduling,
        PythonWrapperCodegen,
        None,  # No C++ AOT wrapper initially
        WrapperFxCodegen,
    )

    # Override device op overrides with Triton-compatible version.
    #
    # torch 2.10+ ships a NATIVE MPS inductor backend; its
    # `mps_device_op_overrides` module registers torch's own
    # `MPSDeviceOpOverrides` for "mps" at import time. That import is driven
    # lazily by `_initialize_device_op_overrides()` on the first
    # `get_device_op_overrides()` call -- which happens AFTER this function in
    # the compile pipeline. If we register ours now and torch's lazy init runs
    # later, torch's override CLOBBERS ours, and torch's native MPS override
    # has no `import_get_raw_stream_as` (a Triton-scheduling-only method) ->
    # `NotImplementedError` when our TritonScheduling writes the kernel header.
    #
    # Force the native init to run FIRST (it sets a one-shot flag so it never
    # re-runs), THEN register ours last so it wins and is never clobbered.
    from torch._inductor.codegen.common import _initialize_device_op_overrides
    _initialize_device_op_overrides()
    register_device_op_overrides("mps", MetalTritonDeviceOpOverrides())

    # Patch MpsInterface with methods needed by TritonScheduling.
    # MPS is single-device, so exchange_device/set_device are no-ops.
    _patch_mps_device_interface()

    # Replace libdevice stubs with tl.math equivalents for Metal.
    # CUDA uses extern_elementwise → __nv_* functions. Metal has no libdevice,
    # but tl.math.* maps to standard MLIR math ops that our backend handles.
    _patch_libdevice_for_metal()

    # Limit persistent reduction configs to Metal's 1024 max threads per threadgroup.
    # Inductor uses XBLOCK * R0_BLOCK threads; default MAX_PERSISTENT_BLOCK_NUMEL=4096
    # allows configs that exceed Metal's hardware limit, causing silent kernel failures.
    _patch_persistent_reduction_configs()

    # Filter non-persistent reduction configs where R0_BLOCK > num_warps * 32.
    # Our MSL lowering uses flat lid indexing (1 element per thread). Triton's blocked
    # layout with sizePerThread > 1 requires each thread to process multiple elements,
    # which we don't support. Capping R0_BLOCK forces Triton to use scf.for loops.
    _patch_reduction_configs()

    _registered = True


def _patch_mps_device_interface():
    """Add missing DeviceInterface methods to MpsInterface for Triton compatibility."""
    from torch._dynamo.device_interface import MpsInterface

    if not hasattr(MpsInterface, "exchange_device") or \
       MpsInterface.exchange_device is not MpsInterface.__mro__[0].__dict__.get("exchange_device"):
        MpsInterface.exchange_device = staticmethod(lambda device: 0)

    if not hasattr(MpsInterface, "maybe_exchange_device") or \
       "maybe_exchange_device" not in MpsInterface.__dict__:
        MpsInterface.maybe_exchange_device = staticmethod(lambda device: 0)

    if not hasattr(MpsInterface, "set_device") or \
       "set_device" not in MpsInterface.__dict__:
        MpsInterface.set_device = staticmethod(lambda device: None)

    if not hasattr(MpsInterface, "get_raw_stream") or \
       "get_raw_stream" not in MpsInterface.__dict__:
        MpsInterface.get_raw_stream = staticmethod(lambda device_idx: 0)


def _filter_metal_persistent_configs(configs, rnumel):
    """Keep persistent-reduction configs whose tile fits Metal's 1024 thread/threadgroup limit.

    Earlier this ALSO dropped UNDER-FILLING configs (XBLOCK*rnumel < num_warps*32 — more
    threads than elements): the old simd_sum reduction let surplus lanes wrap `lid % rnumel`
    back over the columns and over-count (observed: __safe_softmax XBLOCK=1/rnumel=16 returned
    all-zeros, which corrupted a transformer gradient through autotuning, which picks configs
    nondeterministically). That over-refused small-reduction CNNs (BatchNorm) entirely.

    The current 2-D reduce lowering is a shared-memory TREE that MASKS surplus lanes — it
    stages `if (lid < total)` and reduces `if (lid < rows)`, so threads beyond the element
    count contribute nothing. Under-filling configs therefore now compute EXACTLY and are
    KEPT (verified: XBLOCK=1 sum/mean at rnumel=16, BatchNorm CNNs cosine=1.0, and the
    transformer gradient + training-convergence tests). Autotuning is also disabled here
    (register_metal_triton_backend pins a single deterministic config), removing the
    intermittency that surfaced the original bug. Only the hard 1024-thread Metal limit
    remains a drop.

    If every config exceeds 1024 threads, refuse loudly — Metal cannot launch it.
    """
    filtered = [c for c in configs if c.kwargs.get("XBLOCK", 1) * rnumel <= 1024]
    if filtered:
        return filtered
    from triton_msl.errors import MetalNonRecoverableError
    raise MetalNonRecoverableError(
        f"No Metal-safe persistent-reduction config for rnumel={rnumel}: every candidate "
        f"exceeds Metal's 1024 threads/threadgroup limit (XBLOCK*rnumel > 1024). Refusing "
        f"rather than emit a config Metal cannot launch.")


def _patch_persistent_reduction_configs():
    """Limit persistent reduction tile sizes to Metal's 1024 max threads/threadgroup.

    Inductor generates configs where XBLOCK * R0_BLOCK threads are needed.
    Metal supports at most 1024 threads per threadgroup. We cap
    MAX_PERSISTENT_BLOCK_NUMEL to 1024 so invalid configs are never generated.
    """
    import torch._inductor.runtime.triton_heuristics as th

    orig_fn = th._persistent_reduction_configs

    def _metal_persistent_reduction_configs(
        size_hints, reduction_hint=False, inductor_meta=None, triton_meta=None
    ):
        # Check if targeting MPS device
        device_props = triton_meta.get("device") if triton_meta else None
        is_mps = device_props and getattr(device_props, "type", "") == "mps"

        if is_mps:
            # Temporarily cap block numel to Metal's 1024 thread limit
            import torch._inductor.runtime.triton_heuristics as _th

            saved = getattr(_th, "_METAL_MAX_BLOCK_OVERRIDE", None)
            _th._METAL_MAX_BLOCK_OVERRIDE = True
            orig_max = 4096  # default MAX_PERSISTENT_BLOCK_NUMEL

            # Generate configs, then keep only the Metal-safe ones (or refuse).
            configs = orig_fn(size_hints, reduction_hint, inductor_meta, triton_meta)
            rnumel = th.get_total_reduction_numel(size_hints)
            if saved is None:
                delattr(_th, "_METAL_MAX_BLOCK_OVERRIDE")
            return _filter_metal_persistent_configs(configs, rnumel)
        else:
            return orig_fn(size_hints, reduction_hint, inductor_meta, triton_meta)

    th._persistent_reduction_configs = _metal_persistent_reduction_configs


def _patch_reduction_configs():
    """Filter non-persistent reduction configs for Metal compatibility.

    Our MSL lowering uses flat lid-based indexing (1 element per thread).
    When R0_BLOCK > num_warps * 32, Triton's blocked layout uses
    sizePerThread > 1 (each thread handles multiple elements), which our
    lowering doesn't support — producing silently wrong results.

    Fix: remove configs where R0_BLOCK exceeds the thread count.
    """
    import torch._inductor.runtime.triton_heuristics as th

    orig_fn = th._reduction_configs

    def _metal_reduction_configs(*, size_hints, inductor_meta=None, triton_meta=None, **kwargs):
        configs = orig_fn(size_hints=size_hints, inductor_meta=inductor_meta, triton_meta=triton_meta, **kwargs)

        device_props = triton_meta.get("device") if triton_meta else None
        is_mps = device_props and getattr(device_props, "type", "") == "mps"
        if not is_mps:
            return configs

        import copy
        result = []
        for c in configs:
            r0_block = c.kwargs.get("R0_BLOCK", 1)
            threads = c.num_warps * 32
            fixed = copy.deepcopy(c)
            # Force XBLOCK=1: our Welford/multi-value reduce does a flat
            # reduction of ALL threads. With XBLOCK>1, different rows share
            # the same SIMD groups, causing incorrect cross-row mixing.
            fixed.kwargs["XBLOCK"] = 1
            # Cap R0_BLOCK to thread count so sizePerThread stays at 1
            if r0_block > threads:
                p2 = 1
                while p2 * 2 <= threads:
                    p2 *= 2
                fixed.kwargs["R0_BLOCK"] = p2
            result.append(fixed)
        return result

    th._reduction_configs = _metal_reduction_configs


def _patch_libdevice_for_metal():
    """Replace libdevice stubs with Metal-compatible implementations.

    Inductor's triton_compat.py imports triton.language.extra.libdevice which
    is just stubs (``...``). On CUDA, the cuda/libdevice.py overrides these with
    extern_elementwise calls to __nv_* functions. On Metal, we provide a module
    that maps to tl.math (for available ops) or Triton primitives.
    """
    from triton_msl.inductor.metal_libdevice import metal_libdevice
    import torch._inductor.runtime.triton_compat as tc
    tc.libdevice = metal_libdevice

    try:
        import torch._inductor.runtime.triton_helpers as th
        th.libdevice = metal_libdevice
    except (ImportError, AttributeError):
        pass
