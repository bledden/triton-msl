"""CPU-only controls for the pre-adapter translation proof and fallback selection."""

from pathlib import Path

import pytest

from triton_msl.backend.cpp_cfg_proved_gep import AIR_LAYOUT
from triton_msl.backend.cpp_cfg_translation_proof import UnprovedTranslation, prove_translatable, select_lowering

FIXTURES = Path(__file__).with_name("fixtures") / "cpp_cfg"
HEADER = f'target datalayout = "{AIR_LAYOUT}"\n'


def kernel(*body):
    return (
        HEADER
        + "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2, i32 %lid) {\n"
        + "\n".join("  " + line for line in body)
        + "\n  ret void\n}\n!0 = !{}\n"
    )


BASE = (
    "%.generic = addrspacecast ptr addrspace(1) %0 to ptr",
    "%.generic1 = addrspacecast ptr addrspace(1) %1 to ptr",
    "%.val = load i32, ptr addrspace(2) %2, align 4",
    "%5 = sext i32 %lid to i64",
)


def test_accepted_shapes_are_proved():
    assert prove_translatable(
        kernel(
            *BASE,
            "%6 = getelementptr float, ptr %.generic, i64 %5",
            "%7 = load float, ptr %6, align 4",
            "store float %7, ptr %.generic1, align 4",
        )
    ) == {"%0": "float", "%1": "float"}
    assert prove_translatable(
        kernel(
            *BASE,
            "%6 = getelementptr i32, ptr %.generic, i32 1",
            "%7 = load i32, ptr %6, align 4",
            "%8 = getelementptr i32, ptr %.generic1, i64 %5",
            "store i32 %7, ptr %8, align 4",
        )
    ) == {"%0": "i32", "%1": "i32"}
    # An unused buffer is proved with no type: the adapter's default (float) applies to nothing.
    assert prove_translatable(kernel(*BASE, "%7 = load float, ptr %.generic, align 4")) == {"%0": "float", "%1": None}
    # A chained GEP with the same element type is typable from the root (the accepted route emits it).
    assert prove_translatable(
        kernel(
            *BASE,
            "%6 = getelementptr float, ptr %.generic, i64 %5",
            "%7 = getelementptr float, ptr %6, i32 1",
            "%8 = load float, ptr %7, align 4",
        )
    ) == {"%0": "float", "%1": None}
    # Byte buffers may use byte GEPs.
    assert prove_translatable(
        kernel(*BASE, "%6 = getelementptr i8, ptr %.generic, i64 4", "%7 = load i8, ptr %6, align 1")
    ) == {"%0": "i8", "%1": None}


@pytest.mark.parametrize(
    "lines,reason",
    [
        (
            (*BASE, "%6 = getelementptr i8, ptr %.generic, i64 4", "%7 = load float, ptr %6, align 4"),
            "byte-offset GEP on a non-byte buffer",
        ),
        (
            (
                *BASE,
                "%6 = getelementptr float, ptr %.generic, i64 %5",
                "%7 = getelementptr i8, ptr %6, i64 4",
                "%8 = load float, ptr %7, align 4",
            ),
            "byte-offset GEP on a non-byte buffer",
        ),
        (
            (
                *BASE,
                "%6 = getelementptr float, ptr %.generic, i64 %5",
                "%7 = getelementptr i32, ptr %6, i64 1",
                "%8 = load i32, ptr %7, align 4",
            ),
            "mixed access types",
        ),
        (
            (*BASE, "%6 = getelementptr [4 x i8], ptr %.generic, i64 %5", "%7 = load float, ptr %6, align 4"),
            "array GEP survived",
        ),
        (
            (
                *BASE,
                "%6 = getelementptr float, ptr %.generic, i64 %5",
                "%7 = getelementptr float, ptr %.generic1, i64 %5",
                "%8 = select i1 true, ptr %6, ptr %7",
                "%9 = load float, ptr %8, align 4",
            ),
            "escapes into an unproved use",
        ),
        (
            (*BASE, "%6 = load float, ptr %.generic, align 4", "%7 = load i32, ptr %.generic, align 4"),
            "mixed access types",
        ),
        (
            (*BASE, "%6 = getelementptr float, ptr %unknown, i64 %5", "%7 = load float, ptr %6, align 4"),
            "not a kernel buffer",
        ),
        ((*BASE, "call void @escape(ptr %.generic)"), "escapes into an unproved use"),
        ((*BASE, "store ptr %.generic, ptr %.generic1, align 8"), "pointer stored as a value"),
        (
            (*BASE, "%6 = getelementptr inbounds float, ptr %.generic, i64 %5", "%7 = load float, ptr %6, align 4"),
            "unproved GEP form",
        ),
        (
            (*BASE, "%6 = getelementptr float, ptr %.generic, i64 %5, i32 0", "%7 = load float, ptr %6, align 4"),
            "unproved GEP form",
        ),
        ((*BASE, "%6 = load <4 x float>, ptr %.generic, align 16"), "unproved use"),
        ((*BASE, "%x = load float, ptr addrspace(2) %2, align 4"), "mixed constant-buffer types"),
    ],
)
def test_unproved_forms_refuse(lines, reason):
    with pytest.raises(UnprovedTranslation, match=reason):
        prove_translatable(kernel(*lines))


def _retained(arm, case):
    return (FIXTURES / arm / case / "opaque.ll").read_text()


@pytest.mark.parametrize(
    "case,expected_route,reason",
    [
        ("TWO_PTRS_SAME_MASK", "fallback", "byte-offset GEP on a non-byte buffer"),
        ("CHAINED_ADDPTR_F32", "fallback", "unproved result use"),
        ("SELECT_POINTER_F32", "fallback", "unproved result use"),
        ("LANE_INDEXED_F32", "optimized", None),
        ("LANE_INDEXED_I32", "optimized", None),
        ("STRIDED_2_F32", "optimized", None),
        ("NEGATED_MASK", "optimized", None),
        ("VOLATILE_LOAD", "optimized", None),
    ],
)
def test_capability_parity_fixtures_select_the_right_lowering(case, expected_route, reason):
    """Retained real lowerings: the optimized IR is used only when positively proved; otherwise the
    unoptimized (accepted-helper) lowering is returned unchanged, so no capability is lost."""
    optimized, unoptimized = _retained("cpp-cfg-native-build", case), _retained("cpp-volatile-build", case)
    decision = select_lowering(optimized, unoptimized)
    assert decision.route == expected_route, decision.reason
    if expected_route == "fallback":
        assert reason in decision.reason
        assert decision.ir == unoptimized
        if case != "SELECT_POINTER_F32":  # pointer select fails on the accepted route too (pre-existing)
            prove_translatable(unoptimized)  # the accepted lowering itself is within the proved subset
    else:
        assert "[" not in "".join(line for line in decision.ir.splitlines() if "getelementptr" in line)
        prove_translatable(decision.ir)


def test_fallback_is_never_decided_by_metal_compilation(monkeypatch):
    """The decision is a pure function of the two IR texts."""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no subprocess in the eligibility gate"))
    decision = select_lowering(
        _retained("cpp-cfg-native-build", "TWO_PTRS_SAME_MASK"), _retained("cpp-volatile-build", "TWO_PTRS_SAME_MASK")
    )
    assert decision.route == "fallback"


@pytest.mark.parametrize(
    "line",
    [
        "call void @escape(ptr addrspace(2) %2)",
        "%v = ptrtoint ptr addrspace(2) %2 to i64",
        "%p = inttoptr i64 0 to ptr",
        "%p = phi ptr [ null, %entry ], [ null, %loop ]",
    ],
)
def test_unproved_pointer_operations_fall_back(line):
    decision = select_lowering(kernel(*BASE, line), "unchanged unoptimized IR")
    assert decision.route == "fallback"
    assert decision.ir == "unchanged unoptimized IR"


def test_incomplete_input_is_a_refusal():
    assert select_lowering("not a kernel", "original").ir == "original"


def test_proved_zero_offset_nonfloat_types_reach_adapter():
    import triton  # noqa: F401
    from triton_msl.backend.compiler import MetalBackend

    ir = kernel(*BASE, "%v = load i32, ptr %.generic, align 4", "store i32 %v, ptr %.generic1, align 4")
    decision = select_lowering(ir, "original")
    assert decision.route == "optimized"
    typed = MetalBackend._opaque_to_typed_ptrs(decision.ir, proved_device_types=prove_translatable(decision.ir))
    assert "define void @k(i32 addrspace(1)* %0, i32 addrspace(1)* %1" in typed
