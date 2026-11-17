"""Regression anchor for examples/local_triton_dev.py — the kernels in the
'develop Triton locally on Apple Silicon' example must keep matching the CPU
reference (they are real @triton.jit kernels run through the Metal backend)."""

import importlib.util
import os

import pytest

_EX = os.path.join(os.path.dirname(__file__), "..", "examples", "local_triton_dev.py")

try:
    import numpy as np  # noqa: F401
    import torch
    import triton  # noqa: F401

    _HAS = hasattr(torch, "mps") and torch.backends.mps.is_available()
except Exception:  # noqa: BLE001
    _HAS = False

requires = pytest.mark.skipif(not _HAS, reason="needs torch + an Apple MPS GPU")


def _load_example():
    spec = importlib.util.spec_from_file_location("local_triton_dev", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires
@pytest.mark.parametrize("runner", ["run_vector_add", "run_fused_softmax", "run_matmul"])
def test_local_dev_example_matches_cpu(runner):
    import numpy as np

    np.random.seed(0)
    mod = _load_example()
    name, shape, err, tol = getattr(mod, runner)()
    assert err < tol, f"{name} ({shape}): max|Δ vs NumPy| = {err:.2e} >= tol {tol:.0e}"


def test_local_dev_matmul_source_masks_every_operand_axis_cpu():
    """Compile the actual example source without acquiring a Metal device."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    fn = _load_example().matmul_kernel
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    constants = dict(BM=32, BN=32, BK=32)
    signature = {name: "*fp32" if i < 3 else "i32" for i, name in enumerate(fn.arg_names)}
    signature.update({name: "constexpr" for name in constants})
    context = ir.context()
    ir.load_dialects(context)
    source = ASTSource(fn, signature, constants)
    module = source.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    msl = lowerer.lower()
    assert "long _A_M = (long)(_mmarg3);" in msl
    assert "long _B_N = (long)(_mmarg4);" in msl
    assert "both operands prove a zero-masked K tail" in msl
    assert lowerer._tail_access_bounds is None
