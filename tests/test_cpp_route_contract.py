"""C++ binary-production records and fail-closed route decisions.

Stage-control spies only: neither mocked LLVM nor GPU numerics earn correctness
credit for the experimental C++ implementation.
"""

from types import SimpleNamespace
import warnings

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

from triton_msl.backend import compiler
from triton_msl.backend import _cache_contract as cache


_SIMPLE = """module {
  tt.func public @k() {
    %0 = arith.constant 1.0 : f32
    tt.return
  }
}
"""


@pytest.fixture
def stage(monkeypatch):
    events = []
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    monkeypatch.delenv("TRITON_MSL_FORCE_PYTHON", raising=False)
    monkeypatch.delenv("TRITON_MSL_CPP_TRACE", raising=False)
    monkeypatch.setattr(cache, "execution_contract", lambda: "controlled-producer")
    monkeypatch.setattr(cache, "validate_execution_contract", lambda stamp: None)
    monkeypatch.setattr(compiler, "_cpp_warning_emitted", False, raising=False)
    monkeypatch.setattr(compiler.MetalBackend, "_initialize_compilation_runtime", lambda self: None)
    backend = compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32))
    monkeypatch.setattr(backend, "_cpp_load_failure", lambda: None)

    def msl(src, metadata, options):
        metadata["block_size"] = 32
        events.append("msl-source")
        return "controlled MSL"

    def llvm(src, metadata, options):
        metadata["block_size"] = 64
        events.append("cpp-lowering")
        return "controlled LLVM"

    monkeypatch.setattr(compiler.MetalBackend, "make_msl", staticmethod(msl))
    monkeypatch.setattr(compiler.MetalBackend, "make_llir", staticmethod(llvm))
    monkeypatch.setattr(compiler.MetalBackend, "make_metallib_from_llir", staticmethod(lambda *args: b"cpp-binary"))
    monkeypatch.setattr(compiler.MetalBackend, "make_metallib", staticmethod(lambda *args: b"msl-binary"))

    def run(text=_SIMPLE):
        stages = {}
        backend.add_stages(stages, SimpleNamespace(num_warps=4))
        metadata = {"name": "k", "num_warps": 4, "num_ctas": 1, "shared": 0}
        msl_source = stages["msl"](text, metadata)
        binary = stages["metallib"](msl_source, metadata)
        return binary, metadata

    return run, events, backend


@pytest.mark.parametrize("enabled", [False, True])
def test_binary_production_route_is_recorded(stage, monkeypatch, enabled):
    run, events, _ = stage
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1" if enabled else "0")
    binary, metadata = run()
    expected = "cpp" if enabled else "msl"
    assert binary == expected.encode() + b"-binary"
    assert metadata["binary_route"] == expected
    assert events.count("cpp-lowering") == int(enabled)


def test_experimental_binary_warning_is_once_per_process(stage):
    run, _, _ = stage
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
        run()
    messages = [str(item.message) for item in caught]
    assert sum("C++" in message and "unaudited" in message for message in messages) == 1


def test_cpp_failure_is_visible_and_preserves_msl_metadata(stage, monkeypatch):
    run, _, _ = stage

    def fail(*args):
        raise RuntimeError("forced-cpp-probe-failure")

    monkeypatch.setattr(compiler.MetalBackend, "make_metallib_from_llir", staticmethod(fail))
    with pytest.warns(UserWarning, match=r"C\+\+.*forced-cpp-probe-failure.*MSL"):
        binary, metadata = run()
    assert binary == b"msl-binary" and metadata["block_size"] == 32
    assert metadata["binary_route"] == "msl"
    assert "forced-cpp-probe-failure" in metadata["cpp_fallback_reason"]


def test_force_python_never_claims_a_cpp_binary(stage, monkeypatch):
    run, events, _ = stage
    monkeypatch.setenv("TRITON_MSL_FORCE_PYTHON", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binary, metadata = run()
    assert binary == b"msl-binary" and metadata["binary_route"] == "msl"
    assert "cpp-lowering" not in events
    assert not any("unaudited" in str(item.message) for item in caught)


@pytest.mark.parametrize(
    "dot", ["%d = tt.dot %a, %b, %c : tensor<32x32xf32>", '%d = "tt.dot"(%a, %b, %c) : () -> tensor<32x32xf32>']
)
def test_known_unsafe_dot_is_explicitly_derouted_not_compiled(stage, dot):
    run, events, _ = stage
    source = _SIMPLE.replace("    tt.return", "    " + dot + "\n    tt.return")
    with pytest.warns(UserWarning, match=r"C\+\+.*dot.*MSL"):
        binary, metadata = run(source)
    assert binary == b"msl-binary" and metadata["binary_route"] == "msl"
    assert "cpp-lowering" not in events
    assert "dot" in metadata["cpp_fallback_reason"]


def test_missing_optional_extension_has_an_explicit_msl_disposition(stage, monkeypatch):
    run, events, backend = stage
    monkeypatch.setattr(backend, "_cpp_load_failure", lambda: "optional extension unavailable: module not found")
    with pytest.warns(UserWarning, match=r"C\+\+.*unavailable.*MSL"):
        binary, metadata = run()
    assert binary == b"msl-binary" and metadata["binary_route"] == "msl"
    assert "cpp-lowering" not in events
    assert "unavailable" in metadata["cpp_fallback_reason"]


@pytest.mark.parametrize("dot", ["tt.dot", '"tt.dot"'])
def test_direct_cpp_entry_refuses_dot_before_lowering(monkeypatch, dot):
    import sys
    from triton_msl.errors import MetalNonRecoverableError

    monkeypatch.setitem(sys.modules, "triton_msl._triton_msl_cpp", SimpleNamespace())

    def forbidden(*args):
        raise AssertionError("unsafe dot reached annotation stripping before a C++ refusal")

    monkeypatch.setattr(compiler.MetalBackend, "_strip_ttg_annotations", staticmethod(forbidden))
    with pytest.raises(MetalNonRecoverableError, match=r"C\+\+.*dot"):
        compiler.MetalBackend.make_llir(f"%d = {dot} %a, %b, %c", {"name": "k"}, SimpleNamespace())


@triton.jit
def _unmasked_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets))


@pytest.mark.parametrize("use_cpp", [False, True])
def test_actual_metallib_route_and_numeric_control(monkeypatch, tmp_path, use_cpp):
    """Host inputs and compile_shader=0 force the measured metallib to execute."""
    import torch
    from triton.compiler.compiler import ASTSource
    from triton_msl.backend import driver

    if not torch.backends.mps.is_available():
        pytest.skip("Metal hardware unavailable")
    if use_cpp and not compiler.MetalBackend._has_cpp_passes():
        pytest.skip("optional C++ extension unavailable")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1" if use_cpp else "0")
    monkeypatch.delenv("TRITON_MSL_FORCE_PYTHON", raising=False)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    cpp_calls, launches = [], []
    original = compiler.MetalBackend.make_metallib_from_llir

    def compile_cpp(*args, **kwargs):
        result = original(*args, **kwargs)
        cpp_calls.append(result)
        return result

    monkeypatch.setattr(compiler.MetalBackend, "make_metallib_from_llir", staticmethod(compile_cpp))
    utils = driver._get_utils()
    launch = utils.launch

    def launch_binary(*args, **kwargs):
        launches.append(True)
        return launch(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", launch_binary)
    src = ASTSource(
        fn=_unmasked_add, signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"}, constexprs={"BLOCK": 128}
    )
    kernel = triton.compile(src, target=GPUTarget("metal", "apple-m4", 32))
    assert kernel.metadata.binary_route == ("cpp" if use_cpp else "msl")
    assert len(cpp_calls) == int(use_cpp)
    x = torch.arange(128, dtype=torch.float32)
    y = torch.full_like(x, 2.0)
    output = torch.full_like(x, -99.0)
    # CompiledKernel (unlike JITFunction) requires an explicit three-axis grid.
    kernel[(1, 1, 1)](x, y, output)
    assert launches == [True]
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)


@triton.jit
def _multi_program_dot(a, b, c, K: tl.constexpr, N: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 32)
    lhs = tl.load(a + rows[:, None] * K + inner[None, :])
    rhs = tl.load(b + inner[:, None] * N + cols[None, :])
    result = tl.dot(lhs, rhs)
    tl.store(c + rows[:, None] * N + cols[None, :], result)


@triton.jit
def _multi_program_dot_runtime(a, b, c, M, N):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 32)
    lhs = tl.load(a + rows[:, None] * 32 + inner[None, :], rows[:, None] < M, other=0.0)
    rhs = tl.load(b + inner[:, None] * N + cols[None, :], cols[None, :] < N, other=0.0)
    result = tl.dot(lhs, rhs)
    tl.store(c + rows[:, None] * N + cols[None, :], result, (rows[:, None] < M) & (cols[None, :] < N))


def test_constexpr_multi_program_dot_deroutes_with_proved_strides_and_grid(monkeypatch, tmp_path):
    """Literal strides and the proved 2D grid now define each full source tile.

    C++ still declines this dot; faithful MSL reaches binary production without
    requiring runtime scalars named M/N. No binary is built by this CPU control.
    """
    from triton.compiler.compiler import ASTSource

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))

    class SourceProduced(Exception):
        pass

    def forbidden(*args, **kwargs):
        raise AssertionError("multi-program dot entered experimental C++ binary production")

    def source_produced(source, metadata, options):
        assert "long _M = (long)((long)row_base + 32L);" in source
        assert "long _N = (long)((long)col_base + 32L);" in source
        assert "uint pid_m = pid3.x;" in source and "uint pid_n = pid3.y;" in source
        assert "(uint)(64)" in source  # source B/C leading stride, not the tile width
        raise SourceProduced

    monkeypatch.setattr(compiler.MetalBackend, "make_metallib_from_llir", staticmethod(forbidden))
    monkeypatch.setattr(compiler.MetalBackend, "make_metallib", staticmethod(source_produced))
    src = ASTSource(
        fn=_multi_program_dot, signature={"a": "*fp16", "b": "*fp16", "c": "*fp32"}, constexprs={"K": 32, "N": 64}
    )
    with pytest.raises(SourceProduced):
        triton.compile(src, target=GPUTarget("metal", "apple-m4", 32))


def test_multi_program_dot_deroutes_and_writes_every_output(monkeypatch, tmp_path):
    """Cover the actual multi-grid boundary that motivated the C++ dot guard."""
    import torch
    from triton.compiler.compiler import ASTSource
    from triton_msl.backend import driver

    if not torch.backends.mps.is_available():
        pytest.skip("Metal hardware unavailable")
    if not compiler.MetalBackend._has_cpp_passes():
        pytest.skip("optional C++ extension unavailable")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    monkeypatch.delenv("TRITON_MSL_FORCE_PYTHON", raising=False)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    cpp_calls, launches = [], []
    original_cpp = compiler.MetalBackend.make_metallib_from_llir

    def cpp_binary(*args, **kwargs):
        result = original_cpp(*args, **kwargs)
        cpp_calls.append(True)
        return result

    monkeypatch.setattr(compiler.MetalBackend, "make_metallib_from_llir", staticmethod(cpp_binary))
    utils = driver._get_utils()
    original_launch = utils.launch

    def launch_binary(*args, **kwargs):
        launches.append(True)
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", launch_binary)
    src = ASTSource(
        fn=_multi_program_dot_runtime,
        signature={"a": "*fp16", "b": "*fp16", "c": "*fp32", "M": "i32", "N": "i32"},
        constexprs={},
    )
    compiled = triton.compile(src, target=GPUTarget("metal", "apple-m4", 32))
    generator = torch.Generator().manual_seed(245247)
    a = torch.randn(64, 32, generator=generator).half()
    b = torch.randn(32, 64, generator=generator).half()
    c = torch.full((64, 64), float("nan"), dtype=torch.float32)
    compiled[(2, 2, 1)](a, b, c, 64, 64)
    print(
        "DOT_GRID",
        "cpp_binaries",
        len(cpp_calls),
        "metallib_launches",
        len(launches),
        "unwritten",
        torch.isnan(c).sum().item(),
        flush=True,
    )
    assert launches == [True]
    assert torch.isfinite(c).all()
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-3, atol=1e-3)
    assert compiled.metadata.binary_route == "msl"
    assert "tt.dot" in compiled.metadata.cpp_fallback_reason
    assert cpp_calls == []


def test_constexpr_multi_program_dot_writes_the_proved_full_grid(monkeypatch, tmp_path):
    import torch
    from triton.compiler.compiler import ASTSource

    if not torch.backends.mps.is_available():
        pytest.skip("Metal GPU needed")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    monkeypatch.setenv("TRITON_MSL_FAST_MATMUL", "0")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    src = ASTSource(_multi_program_dot, {"a": "*fp16", "b": "*fp16", "c": "*fp32"}, {"K": 32, "N": 64})
    kernel = triton.compile(src, target=GPUTarget("metal", "apple-m4", 32))
    assert kernel.metadata.binary_route == "msl"
    generator = torch.Generator().manual_seed(32064)
    ah = torch.randint(-3, 4, (64, 32), generator=generator).half()
    bh = torch.randint(-3, 4, (32, 64), generator=generator).half()
    a, b = ah.to("mps"), bh.to("mps")
    owner = torch.full((64 * 64 + 64,), -17391.0, device="mps")
    c = owner[32:-32].view(64, 64)
    kernel[(2, 2, 1)](a, b, c)
    expected = (ah.double() @ bh.double()).float()
    torch.testing.assert_close(c.cpu(), expected, atol=0, rtol=0)
    assert torch.all(owner[:32].cpu() == -17391.0) and torch.all(owner[-32:].cpu() == -17391.0)
    assert torch.equal(a.cpu(), ah) and torch.equal(b.cpu(), bh)
