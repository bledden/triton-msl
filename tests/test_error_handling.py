"""Tests for structured error types and error handling."""

import os
import platform
import tempfile

import pytest

from triton_msl.errors import (
    MetalCodegenError,
    MetalCompilationError,
    MetalLaunchError,
    MetalNotImplementedError,
    MetalUnsupportedError,
    MetalValidationError,
)
from triton_msl.debug import _debug_level, _reset_debug_cache

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)


class TestErrorTypes:
    """Test that error types have correct attributes and messages."""

    def test_codegen_error_basic(self):
        err = MetalCodegenError("test failure")
        assert "test failure" in str(err)
        assert err.op_name is None
        assert err.ssa_id is None

    def test_codegen_error_with_context(self):
        err = MetalCodegenError(
            "Cannot lower operation",
            op_name="tt.unknown",
            ssa_id="%42",
            type_str="tensor<32xf32>",
        )
        msg = str(err)
        assert "Cannot lower operation" in msg
        assert "tt.unknown" in msg
        assert "%42" in msg
        assert "tensor<32xf32>" in msg

    def test_unsupported_error_is_codegen_error(self):
        err = MetalUnsupportedError("FP64 not available on Apple GPUs")
        assert isinstance(err, MetalCodegenError)
        assert isinstance(err, RuntimeError)
        assert "Hardware unsupported" in str(err)

    def test_not_implemented_error_has_issue_link(self):
        err = MetalNotImplementedError(
            "cf.cond_br",
            op_name="cf.cond_br",
        )
        assert isinstance(err, MetalCodegenError)
        assert "Not yet implemented" in str(err)
        assert "github.com/bledden/triton-msl/issues" in str(err)

    def test_compilation_error_basic(self):
        err = MetalCompilationError("compilation failed")
        assert isinstance(err, RuntimeError)
        assert "compilation failed" in str(err)

    def test_compilation_error_with_stderr(self):
        err = MetalCompilationError(
            "MSL compilation failed",
            msl_source="/tmp/test.metal",
            stderr="error: use of undeclared identifier 'foo'",
        )
        msg = str(err)
        assert "MSL compilation failed" in msg
        assert "undeclared identifier" in msg
        assert err.msl_source == "/tmp/test.metal"

    # --- MetalValidationError tests ---

    def test_validation_error_basic(self):
        err = MetalValidationError("invalid block size")
        assert isinstance(err, MetalCodegenError)
        assert isinstance(err, RuntimeError)
        assert "Validation failed" in str(err)
        assert "invalid block size" in str(err)
        assert err.constraint is None

    def test_validation_error_with_constraint(self):
        err = MetalValidationError(
            "tensor rank too high",
            op_name="tt.reduce",
            ssa_id="%10",
            type_str="tensor<2x3x4x5xf32>",
            constraint="max_rank <= 3",
        )
        msg = str(err)
        assert "Validation failed" in msg
        assert "tensor rank too high" in msg
        assert "tt.reduce" in msg
        assert "%10" in msg
        assert "tensor<2x3x4x5xf32>" in msg
        assert "max_rank <= 3" in msg
        assert err.constraint == "max_rank <= 3"

    def test_validation_error_inherits_codegen_attrs(self):
        err = MetalValidationError(
            "type mismatch",
            op_name="arith.addf",
            ssa_id="%5",
        )
        assert err.op_name == "arith.addf"
        assert err.ssa_id == "%5"
        assert err.type_str is None

    # --- MetalLaunchError tests ---

    def test_launch_error_basic(self):
        err = MetalLaunchError("dispatch failed")
        assert isinstance(err, RuntimeError)
        # MetalLaunchError is NOT a MetalCodegenError
        assert not isinstance(err, MetalCodegenError)
        assert "dispatch failed" in str(err)
        assert err.kernel_name is None
        assert err.grid is None
        assert err.reason is None

    def test_launch_error_with_context(self):
        err = MetalLaunchError(
            "buffer allocation failed",
            kernel_name="add_kernel",
            grid=(256, 1, 1),
            reason="out of GPU memory",
        )
        msg = str(err)
        assert "buffer allocation failed" in msg
        assert "add_kernel" in msg
        assert "(256, 1, 1)" in msg
        assert "out of GPU memory" in msg
        assert err.kernel_name == "add_kernel"
        assert err.grid == (256, 1, 1)
        assert err.reason == "out of GPU memory"

    def test_launch_error_attributes(self):
        err = MetalLaunchError(
            "command buffer error",
            kernel_name="softmax_kernel",
        )
        assert err.kernel_name == "softmax_kernel"
        assert err.grid is None
        assert err.reason is None

    # --- Inheritance checks ---

    def test_error_hierarchy(self):
        """All error types have correct inheritance chains."""
        # MetalCompilationError -> RuntimeError
        assert issubclass(MetalCompilationError, RuntimeError)

        # MetalCodegenError -> RuntimeError
        assert issubclass(MetalCodegenError, RuntimeError)

        # MetalValidationError -> MetalCodegenError -> RuntimeError
        assert issubclass(MetalValidationError, MetalCodegenError)
        assert issubclass(MetalValidationError, RuntimeError)

        # MetalUnsupportedError -> MetalCodegenError -> RuntimeError
        assert issubclass(MetalUnsupportedError, MetalCodegenError)
        assert issubclass(MetalUnsupportedError, RuntimeError)

        # MetalNotImplementedError -> MetalCodegenError -> RuntimeError
        assert issubclass(MetalNotImplementedError, MetalCodegenError)
        assert issubclass(MetalNotImplementedError, RuntimeError)

        # MetalLaunchError -> RuntimeError (but NOT MetalCodegenError)
        assert issubclass(MetalLaunchError, RuntimeError)
        assert not issubclass(MetalLaunchError, MetalCodegenError)


class TestDebugMode:
    """Test TRITON_MSL_DEBUG env var and _debug_level helper."""

    def test_debug_env_recognized(self):
        """TRITON_MSL_DEBUG env var should be checked by compiler."""
        # Just verify the env var mechanism works at Python level
        os.environ["TRITON_MSL_DEBUG"] = "1"
        assert os.environ.get("TRITON_MSL_DEBUG") == "1"
        del os.environ["TRITON_MSL_DEBUG"]
        assert os.environ.get("TRITON_MSL_DEBUG") is None

    def test_debug_level_returns_1(self, monkeypatch):
        """_debug_level() returns 1 when TRITON_MSL_DEBUG=1."""
        _reset_debug_cache()
        monkeypatch.setenv("TRITON_MSL_DEBUG", "1")
        assert _debug_level() == 1
        _reset_debug_cache()

    def test_debug_level_returns_2(self, monkeypatch):
        """_debug_level() returns 2 when TRITON_MSL_DEBUG=2."""
        _reset_debug_cache()
        monkeypatch.setenv("TRITON_MSL_DEBUG", "2")
        assert _debug_level() == 2
        _reset_debug_cache()

    def test_debug_level_default_zero(self, monkeypatch):
        """_debug_level() returns 0 when TRITON_MSL_DEBUG is not set."""
        _reset_debug_cache()
        monkeypatch.delenv("TRITON_MSL_DEBUG", raising=False)
        assert _debug_level() == 0
        _reset_debug_cache()

    def test_debug_level_invalid_value(self, monkeypatch):
        """_debug_level() returns 0 for non-integer values."""
        _reset_debug_cache()
        monkeypatch.setenv("TRITON_MSL_DEBUG", "yes")
        assert _debug_level() == 0
        _reset_debug_cache()

    def test_debug_level_caches_result(self, monkeypatch):
        """_debug_level() caches the result across calls."""
        _reset_debug_cache()
        monkeypatch.setenv("TRITON_MSL_DEBUG", "1")
        assert _debug_level() == 1
        # Change env var — cached value should persist
        monkeypatch.setenv("TRITON_MSL_DEBUG", "2")
        assert _debug_level() == 1  # still cached as 1
        _reset_debug_cache()


class TestCodegenErrorWrapping:
    """Test that the generic lowerer wraps errors with context."""

    def test_lowerer_wraps_exceptions(self):
        """Verify _lower_op wraps unexpected errors in MetalCodegenError."""
        from triton_msl.codegen.generic_lowerer import GenericLowerer
        from triton_msl.codegen.mlir_walker import SSAValue

        # Create a minimal SSA value that will trigger an error
        ssa = SSAValue(
            id=99,
            name="v99",
            op="tt.load",
            operand_ids=[999],  # nonexistent operand
            attrs={},
            type_str="tensor<32xf32>",
            elem_type="f32",
            is_tensor=True,
        )

        # Create a minimal lowerer that will fail on this op
        # (missing env entries for operand_ids will cause KeyError)
        class FakeGraph:
            ops = []
            args = []
            called_funcs = []
            name = "test"
            raw_text = ""

        lowerer = GenericLowerer.__new__(GenericLowerer)
        lowerer.graph = FakeGraph()
        lowerer._skip_ids = set()
        lowerer.env = {}
        lowerer.env_types = {}
        lowerer.env_shapes = {}
        lowerer.env_is_ptr = {}
        lowerer.computed_values = {}

        with pytest.raises(MetalCodegenError) as exc_info:
            lowerer._lower_op(ssa)

        assert "tt.load" in str(exc_info.value)
