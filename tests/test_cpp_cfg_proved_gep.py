"""CPU-only positive/refusal controls for the evidence-only narrow adapter."""

import re

import pytest

from triton_msl.backend.cpp_cfg_proved_gep import AIR_LAYOUT, SIZE, UnprovedGEP, lower_disjoint_or, normalize_array_geps


def fixture(ty="float", width="i64", index="%idx", array_bytes=4, volatile=""):
    return f"""target datalayout = "{AIR_LAYOUT}"
define void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, {width} %idx) {{
  %p = addrspacecast ptr addrspace(1) %in to ptr
  %q = addrspacecast ptr addrspace(1) %out to ptr
  %a = getelementptr [{array_bytes} x i8], ptr %p, {width} {index}
  %b = getelementptr [{array_bytes} x i8], ptr %q, {width} {index}
  %v = load {volatile}{ty}, ptr %a, align 1
  store {ty} %v, ptr %b, align 1
  ret void
}}
"""


@pytest.mark.parametrize("ty", sorted(SIZE))
@pytest.mark.parametrize(
    "width,index", [("i32", "%idx"), ("i64", "%idx"), ("i32", "-2147483648"), ("i64", "9223372036854775807")]
)
def test_equal_size_preserves_every_other_byte(ty, width, index):
    text = fixture(ty, width, index, SIZE[ty], "volatile ")
    assert normalize_array_geps(text) == text.replace(f"[{SIZE[ty]} x i8]", ty)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("[4 x i8]", "[8 x i8]"),
        lambda s: s.replace("%a, align 1", "%p, align 1"),  # dead result, no use proof
        lambda s: s.replace("ret void", "call void @escape(ptr %a)\n  ret void"),
        lambda s: s.replace("ret void", "call void @escape(ptr %p)\n  ret void"),
        lambda s: s.replace("ret void", "%mixed = load i32, ptr %a, align 1\n  ret void"),
        lambda s: s.replace("ret void", "%mixed = load i32, ptr %p, align 1\n  ret void"),
        lambda s: s.replace("getelementptr [", "getelementptr inbounds ["),
        lambda s: s.replace("ptr %p, i64", "ptr %unknown, i64"),
        lambda s: s.replace("ptr %q, i64", "ptr %a, i64"),  # chained pointer
        lambda s: s.replace("[4 x i8], ptr %p, i64 %idx", "[4 x i8], ptr %p, i64 %idx, i32 0"),
        lambda s: s.replace("ptr addrspace(1) %in", "ptr addrspace(3) %in"),
        lambda s: s.replace("%v = load float", "%v = load atomic float"),
        lambda s: s.replace("i32:32:32", "i32:128:128"),
        lambda s: s.replace(
            "ret void", "%extra = getelementptr i32, ptr %p, i64 %idx\n  %x = load i32, ptr %extra, align 1\n  ret void"
        ),
    ],
)
def test_unproved_pointer_or_stride_refuses(mutate):
    with pytest.raises(UnprovedGEP):
        normalize_array_geps(mutate(fixture()))


def test_non_array_input_untouched():
    text = fixture().replace("[4 x i8]", "float")
    assert normalize_array_geps(text) == text


def test_disjoint_removes_only_promise():
    text = "  %result = or disjoint i32 %a, 1\n  %other = or i32 %a, 2\n"
    assert lower_disjoint_or(text) == text.replace("or disjoint", "or")
    for a in range(64):
        for b in range(64):
            if a & b == 0:
                assert (a | b) == a + b


def test_signed_index_spelling_not_recomputed():
    text = fixture(index="-1")
    assert re.findall(r"i64 -1", normalize_array_geps(text)) == ["i64 -1", "i64 -1"]
