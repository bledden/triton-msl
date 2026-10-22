"""Route-verified C++ memory composition; every input lane has allocated backing.

The CPU masked-load tests pin suppression of inactive reads. These GPU tests
check numerical retention without ever running an old out-of-bounds shader.
Fallback controls are explicit and do not earn C++ coverage.
"""

import hashlib
import os
import re

import numpy as np
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend import compiler, driver


@triton.jit
def _memory(X, O, N, R, MODE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MODE == "store":
        x = tl.load(X + i)
        value = x * 2 + 1
    elif MODE == "loop":
        value = tl.full((BLOCK,), 0, tl.float32)
        for k in range(R):
            x = tl.load(X + i, i < N, other=0)
            value += x + k
    elif MODE == "dual":
        x = tl.load(X + i, i < N, other=7)
        y = tl.load(X + i, i >= N, other=x)
        value = y * 2 + 1
    elif MODE == "where":
        x = tl.load(X + i, i < N, other=7)
        value = tl.where(x > 0, x * 2, -x)
    elif MODE == "volatile":
        x = tl.load(X + i, i < N, other=7, volatile=True)
        value = x * 2 + 1
    else:
        x = tl.load(X + i, i < N, other=7)
        value = x * 2 + 1
    tl.store(O + i, value, i < N)


@pytest.mark.parametrize(
    "mode,block,dtype,route",
    [
        ("tail", 128, "fp32", "cpp"),
        ("store", 128, "fp32", "cpp"),
        ("loop", 128, "fp32", "cpp"),
        ("dual", 128, "fp32", "cpp"),
        ("where", 128, "fp32", "msl"),
        ("tail", 2048, "fp32", "cpp"),
        ("tail", 128, "i32", "cpp"),
        ("tail", 128, "fp16", "msl"),
        ("volatile", 128, "fp32", "cpp"),
        ("volatile", 128, "i32", "cpp"),
    ],
)
def test_memory_bounds_and_composition(monkeypatch, tmp_path, record_property, mode, block, dtype, route):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("Metal hardware unavailable")
    if not compiler.MetalBackend._has_cpp_passes():
        pytest.skip("optional C++ extension unavailable; no C++ coverage")
    import triton_msl._triton_msl_cpp as cpp
    from pathlib import Path

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    monkeypatch.delenv("TRITON_MSL_FORCE_PYTHON", raising=False)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    binaries, launches, llvm_texts = [], [], []
    make_binary = compiler.MetalBackend.make_metallib_from_llir
    make_llir = compiler.MetalBackend.make_llir

    def tracked_llir(*args, **kwargs):
        text = make_llir(*args, **kwargs)
        llvm_texts.append(text)
        return text

    def tracked_binary(*args, **kwargs):
        binary = make_binary(*args, **kwargs)
        binaries.append(hashlib.sha256(binary).hexdigest())
        return binary

    monkeypatch.setattr(compiler.MetalBackend, "make_metallib_from_llir", staticmethod(tracked_binary))
    monkeypatch.setattr(compiler.MetalBackend, "make_llir", staticmethod(tracked_llir))
    utils = driver._get_utils()
    launch = utils.launch

    def tracked_launch(pipeline, grid, group_size, buffers, *args, **kwargs):
        launches.append((tuple(grid), tuple(group_size)))
        return launch(pipeline, grid, group_size, buffers, *args, **kwargs)

    monkeypatch.setattr(utils, "launch", tracked_launch)
    source = ASTSource(
        _memory,
        signature={"X": "*" + dtype, "O": "*" + dtype, "N": "i32", "R": "i32"},
        constexprs={"MODE": mode, "BLOCK": block},
    )
    kernel = triton.compile(source, target=GPUTarget("metal", "apple-m4", 32))
    digest = hashlib.sha256(kernel.asm["metallib"]).hexdigest()
    assert kernel.metadata.binary_route == route
    assert binaries == ([digest] if route == "cpp" else [])
    assert bool(kernel.metadata.cpp_fallback_reason) == (route == "msl")
    assert len(llvm_texts) == (1 if route == "cpp" else 0)
    if llvm_texts:
        assert ("load volatile " in llvm_texts[0]) == (mode == "volatile")
        assert kernel.metadata.cpp_scalar_cfg_typed_ir_sha256 == hashlib.sha256(llvm_texts[0].encode()).hexdigest()
        assert kernel.metadata.cpp_scalar_cfg_metallib_sha256 == digest
        cfg_route = kernel.metadata.cpp_scalar_cfg_route
        assert cfg_route in (
            {"optimized", "fallback"} if os.environ.get("TRITON_MSL_CPP_SCALAR_CFG") == "1" else {"disabled"}
        )
        if mode == "dual" and os.environ.get("TRITON_MSL_CPP_SCALAR_CFG") == "1":
            assert cfg_route == "optimized"
            assert len(re.findall(r"\bload float\b", llvm_texts[0])) == 1
            assert len(re.findall(r"\bbr i1\b", llvm_texts[0])) == 1
        record_property("cpp_scalar_cfg_route", cfg_route)
        record_property("cpp_scalar_cfg_reason", kernel.metadata.cpp_scalar_cfg_reason)
        record_property("llvm_sha256", hashlib.sha256(llvm_texts[0].encode()).hexdigest())
    record_property("binary_route", route)
    record_property("metallib_sha256", digest)
    record_property("extension_sha256", hashlib.sha256(Path(cpp.__file__).read_bytes()).hexdigest())
    # Retain the exact generated artifacts for same-toolchain diagnostic runs.
    (tmp_path / "kernel.metallib").write_bytes(kernel.asm["metallib"])
    if llvm_texts:
        (tmp_path / "kernel.llir").write_text(llvm_texts[0])
    record_property("artifact_directory", str(tmp_path))

    n_alloc = 3 * block
    tdtype = {"fp32": torch.float32, "i32": torch.int32, "fp16": torch.float16}[dtype]
    x = (torch.arange(n_alloc, dtype=torch.int32) % 127 - 63).to(tdtype)
    backing = torch.full((n_alloc + 32,), -999, dtype=tdtype)
    out = backing[16 : 16 + n_alloc]
    checked = 0
    for n in [-1, 0, 1, block - 1, block, block + 1, 2 * block + 1, n_alloc]:
        for repetitions in [0, 1, 3] if mode == "loop" else [1]:
            backing.fill_(-999)
            before = len(launches)
            kernel[(3, 1, 1)](x, out, n, repetitions)
            assert len(launches) == before + 1
            assert launches[-1] == ((3, 1, 1), (min(block, 1024), 1, 1))
            base = x.numpy().astype(np.float64)
            if mode == "loop":
                value = repetitions * base + repetitions * (repetitions - 1) / 2
            elif mode == "where":
                # Triton's tensor.__neg__ lowers to sub(0, x), not an IEEE
                # sign-bit toggle: positive zero remains positive zero.
                value = np.where(base > 0, base * 2, 0.0 - base)
            else:
                value = base * 2 + 1
            expected = np.full(n_alloc + 32, -999, dtype=out.numpy().dtype)
            active = max(0, min(n, n_alloc))
            expected[16 : 16 + active] = value[:active].astype(expected.dtype)
            assert backing.numpy().tobytes() == expected.tobytes(), (mode, block, dtype, n, repetitions)
            checked += 1
    record_property("checked_bound_loop_cases", checked)
