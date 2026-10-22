"""Prune proven capacity failures, never compiler-integrity failures.

CPU contract tests use the installed Autotuner, real emission/compiler boundaries,
and real matmul allocation arithmetic. A small graph/launcher adapter replaces GPU
execution; these are not throughput or source-pattern capability tests.
"""

from types import SimpleNamespace
import math

import pytest
from triton import Config
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources

from triton_msl.backend.compiler import MetalBackend, MetalOptions
from triton_msl.backend.driver import MetalUtils
from triton_msl.codegen._msl_templates import make_matmul_kernel
from triton_msl.errors import MetalNonRecoverableError


def _tuner(monkeypatch, tmp_path, blocks, failure=None):
    from triton._C.libtriton import ir
    import triton_msl.codegen.generic_lowerer as lowerer
    import triton_msl.codegen.mlir_walker as walker
    import triton_msl.codegen.msl_emitter as emitter

    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_MSL_LEGACY", "1")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setattr(emitter, "_legacy_fallback", lambda *args: pytest.fail("terminal error fell back"))

    class Lowerer:
        def __init__(self, graph, options):
            self.block_k = graph.block_k
            self.effective_block_size = 1024
            # Real template lowering initializes kb but returns before building it.
            self.kb = None
            self._assert_plan = None
            self._flash_attention = None
            self._argument_abi = None

        def lower(self):
            if failure is not None and self.block_k == 128:
                raise failure
            return make_matmul_kernel(64, 64, self.block_k)

        def get_output_arg_indices(self):
            return [2]

    # Keep the compiler's native operation census live. The prior string adapter
    # bypassed the module contract and failed before testing either error class.
    # Only graph construction/lowering is substituted: this is still a resource
    # classification test, not evidence of native matmul pattern recognition.
    ctx = ir.context()
    ir.load_dialects(ctx)
    modules = {}
    for k in set(blocks):
        path = tmp_path / f"resource_pin_{k}.ttir"
        path.write_text(f"module {{ tt.func public @resource_pin_{k}() {{ tt.return }} }}")
        modules[k] = ir.parse_mlir_module(str(path), ctx)
        modules[k].context = ctx

    real_walk = walker.walk_ttgir

    def graph_adapter(mod, options):
        graph = real_walk(mod, options)
        graph.block_k = int(graph.func_name.rsplit("_", 1)[1])
        return graph

    monkeypatch.setattr(walker, "walk_ttgir", graph_adapter)
    monkeypatch.setattr(lowerer, "GenericLowerer", Lowerer)
    calls = []

    def source():
        pass

    def launch(**kwargs):
        k = kwargs["BK"]
        calls.append(k)
        return MetalBackend.make_msl(modules[k], {}, MetalOptions())

    def bench(fn, quantiles):
        fn()
        return [1.0] * len(quantiles)

    tuner = Autotuner(
        SimpleNamespace(fn=source, run=launch), [], [Config({"BK": k}) for k in blocks], [], None, None, do_bench=bench
    )
    tuner.cache_results = False
    return tuner, calls


@pytest.mark.parametrize("blocks", [(128, 64), (64, 128), (128, 256)])
def test_real_autotuner_prunes_proven_capacity_only(monkeypatch, tmp_path, blocks):
    tuner, calls = _tuner(monkeypatch, tmp_path, blocks)
    if 64 not in blocks:
        # Installed Triton retries its best (infinite-cost) config when all fail.
        # That final launch MUST remain a loud resource failure, not bogus success.
        with pytest.raises(OutOfResources):
            tuner.run()
    else:
        assert "kernel void" in tuner.run()
        assert tuner.best_config.kwargs["BK"] == 64
        assert calls == [*blocks, 64]
        assert all(math.isinf(t[0]) == (c.kwargs["BK"] != 64) for c, t in tuner.configs_timings.items())


@pytest.mark.parametrize("message", ["unreplayed value", "exceeds 1024: unwritten staging tail"])
def test_integrity_failure_is_never_pruned(monkeypatch, tmp_path, message):
    import triton_msl.debug as debug

    monkeypatch.setenv("TRITON_MSL_FALLBACK", "warn")
    monkeypatch.setattr(debug, "_cached_fallback_mode", debug._UNSET)
    error = MetalNonRecoverableError(message)
    tuner, calls = _tuner(monkeypatch, tmp_path, (128, 64), failure=error)
    with pytest.warns(UserWarning, match="error is propagated to the caller"):
        with pytest.raises(MetalNonRecoverableError) as caught:
            tuner.run()
    assert caught.value is error
    assert not isinstance(error, OutOfResources)
    assert calls == [128]


def test_capacity_type_requires_numeric_excess_and_preserves_boundary():
    from triton_msl.errors import MetalResourceError

    assert "kernel void" in make_matmul_kernel(64, 64, 64)  # exactly 32768 bytes
    with pytest.raises(MetalResourceError) as caught:
        make_matmul_kernel(64, 64, 128)
    assert (caught.value.required, caught.value.limit, caught.value.resource) == (
        65536,
        32768,
        "threadgroup memory bytes",
    )
    assert not isinstance(caught.value, MetalNonRecoverableError)
    with pytest.raises(ValueError):
        MetalResourceError(32768, 32768, "threadgroup memory bytes")


@pytest.mark.parametrize("tile", [32, 128])
def test_pointer_role_integrity_precedes_capacity(tile):
    from triton_msl.codegen._lowerer_templates import _TemplateMixin

    # A source-role proof resolves A/B/C to noncanonical physical argument slots.
    # At 128 the allocation ALSO exceeds capacity. Integrity must win in both.
    args = [SimpleNamespace(index=i, name=n, is_ptr=True, elem_type="f32") for i, n in zip((2, 0, 1), ("A", "B", "C"))]
    lowerer = SimpleNamespace(
        graph=SimpleNamespace(args=args, ops=[], func_name="role_priority"),
        _dot_template_ptr_roles=lambda: tuple(args),
        _maybe_fast_matmul_descriptor=lambda: None,
    )
    with pytest.raises(MetalNonRecoverableError, match="canonical.*signature") as caught:
        _TemplateMixin._lower_dot_simple_template(lowerer, [tile], args, "fp32")
    assert not isinstance(caught.value, OutOfResources)


@pytest.mark.parametrize("message", ["Internal error: AGXMetal compiler", "threadgroup memory exceeds capacity"])
def test_pipeline_text_does_not_prove_resource_exhaustion(message):
    library = SimpleNamespace(newFunctionWithName_=lambda name: object())
    device = SimpleNamespace(
        newLibraryWithURL_error_=lambda *args: (library, None),
        newComputePipelineStateWithFunction_error_=lambda *args: (None, message),
    )
    # No Metal object or GPU call: all pipeline API outcomes are supplied explicitly.
    with pytest.raises(RuntimeError, match=message):
        MetalUtils.load_binary(SimpleNamespace(device=device), "kernel", "/not-opened.metallib", 0)
