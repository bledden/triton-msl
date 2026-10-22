"""CPU-only native lowering checks. These do not widen C++ GPU admission."""

import re

import pytest
import triton  # noqa: F401

cpp = pytest.importorskip("triton_msl._triton_msl_cpp")


def source(mode="dynamic", other=True):
    mask = {
        "dynamic": """%n = tt.splat %extent : i32 -> tensor<128xi32>
    %i = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %m = arith.cmpi slt, %i, %n : tensor<128xi32>""",
        "false": "%m = arith.constant dense<false> : tensor<128xi1>",
        "true": "%m = arith.constant dense<true> : tensor<128xi1>",
        "unmasked": "",
    }[mode]
    operands = "%p" if mode == "unmasked" else "%p, %m" + (", %z" if other else "")
    return f"""module {{
  tt.func public @masked_load(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %extent: i32) {{
    %p = tt.splat %in : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %q = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %z = arith.constant dense<7.0> : tensor<128xf32>
    {mask}
    %x = tt.load {operands} : tensor<128x!tt.ptr<f32>>
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
    tt.return
  }}
}}"""


def lower(mode="dynamic", other=True):
    return cpp.run_to_llvm(source(mode, other))


def test_comparison_mask_is_reachable_in_opt_in_router(monkeypatch):
    from triton_msl.backend.compiler import MetalBackend

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    # The comparison result type is implicit. The text dtype guard does not
    # reject every boolean predicate, so it is not a masked-load safety proof.
    assert "i1" not in source()
    assert not MetalBackend._has_complex_ops(source())


@pytest.mark.parametrize("other", [True, False])
def test_dynamic_mask_branches_before_the_load(other):
    ir = lower(other=other)
    # A select on an already loaded value is not a masked memory access.
    branch = re.search(r"br i1 %\w+, label %([\w.]+), label %([\w.]+)", ir)
    assert branch is not None, ir
    taken, bypass = branch.groups()
    blocks = dict(re.findall(r"^([\w.]+):[^\n]*\n(.*?)(?=^[\w.]+:|^})", ir, re.M | re.S))
    assert "load float" in blocks[taken], ir
    assert "load float" not in blocks[bypass], ir
    assert "load float" not in ir[: branch.start()], ir
    assert "phi float" in blocks[bypass], ir
    assert ("7.000000e+00" if other else "undef") in blocks[bypass], ir


@pytest.mark.parametrize("other", [True, False])
def test_false_mask_performs_no_memory_read(other):
    ir = lower("false", other)
    assert "load float" not in ir, ir
    assert "store float" in ir, ir


@pytest.mark.parametrize("mode", ["unmasked", "true"])
def test_unconditional_access_has_no_mask_branch(mode):
    ir = lower(mode)
    assert len(re.findall(r"\bload float\b", ir)) == 1, ir
    assert "br i1" not in ir, ir


def test_consecutive_masked_loads_preserve_value_flow():
    text = source().replace(
        "tt.store %q, %x",
        """%second = arith.cmpi sgt, %i, %n : tensor<128xi32>
    %y = tt.load %p, %second, %x : tensor<128x!tt.ptr<f32>>
    tt.store %q, %y""",
    )
    ir = cpp.run_to_llvm(text)
    assert len(re.findall(r"\bbr i1\b", ir)) == 2, ir
    phis = re.findall(r"(%[\w.]+) = phi float (.*)", ir)
    assert len(phis) == 2, ir
    assert phis[0][0] in phis[1][1], ir


def test_masked_load_then_masked_store_keeps_store_on_true_edge():
    text = source().replace("tt.store %q, %x :", "tt.store %q, %x, %m :")
    ir = cpp.run_to_llvm(text)
    blocks = dict(re.findall(r"^([\w.]+):[^\n]*\n(.*?)(?=^[\w.]+:|^})", ir, re.M | re.S))
    branches = re.findall(r"br i1 %\w+, label %([\w.]+), label %([\w.]+)", ir)
    assert len(branches) == 2, ir
    taken, bypass = branches[-1]
    assert "store float" in blocks[taken], ir
    assert "store float" not in blocks[bypass], ir
    assert len(re.findall(r"\bstore float\b", ir)) == 1, ir


def test_masked_load_composes_with_loop_carried_value():
    text = source().replace(
        "%x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>",
        """%lo = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %x = scf.for %k = %lo to %extent step %step iter_args(%carry = %z) -> (tensor<128xf32>) : i32 {
      %v = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
      %next = arith.addf %carry, %v : tensor<128xf32>
      scf.yield %next : tensor<128xf32>
    }""",
    )
    ir = cpp.run_to_llvm(text)
    assert "fadd float" in ir, ir
    assert len(re.findall(r"\bphi float\b", ir)) >= 2, ir
    blocks = dict(re.findall(r"^([\w.]+):[^\n]*\n(.*?)(?=^[\w.]+:|^})", ir, re.M | re.S))
    edges = re.findall(r"br i1 %\w+, label %([\w.]+), label %([\w.]+)", ir)
    guarded_loads = [(a, b) for a, b in edges if "load float" in blocks[a] and "load float" not in blocks[b]]
    assert len(guarded_loads) == 1, ir
