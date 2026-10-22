"""CPU building-block controls; these do not enable a production route."""

import hashlib
import re
from types import SimpleNamespace

import pytest

from triton_msl.backend._attention_variant import (
    PipelineFacts,
    VariantSelection,
    select_variant,
    validate_variant_geometry,
)
from triton_msl.codegen._cooperative_attention import build_candidate
from triton_msl.codegen.msl_emitter import Arg, KernelBuilder
from triton_msl.errors import MetalNonRecoverableError


def fixture(renamed=False):
    names = (
        ("Query", "Key", "Value", "Addend", "Result", "QueryCount", "KeyCount")
        if renamed
        else ("Q", "K", "V", "Bias", "Out", "NQ", "NK")
    )
    kb = KernelBuilder("attention", block_size=32)
    kb.args = [Arg(name, "fp16" if i < 3 else "fp32" if i < 5 else "i32", is_ptr=i < 5) for i, name in enumerate(names)]
    plan = dict(zip(("q", "k", "v", "bias", "out", "nq", "nk"), names))
    plan.update(rows=32, cols=64, depth=64, causal=False)
    return SimpleNamespace(kb=kb), plan


@pytest.mark.parametrize("renamed", [False, True])
def test_builder_owns_tiles_and_preserves_scalar(renamed):
    lowerer, plan = fixture(renamed)
    scalar = lowerer.kb
    before = scalar.build()
    candidate = build_candidate(lowerer, plan)
    assert lowerer.kb is scalar and scalar.build() == before
    assert scalar.block_size == 32 and not scalar._threadgroup_arrays
    assert (candidate.threads, candidate.shared, candidate.execution_width) == (512, 16384, 32)
    assert candidate.entry == "attention__attncoop16"
    assert "threadgroup float Qs[2048];" in candidate.source
    assert "threadgroup float S[2048];" in candidate.source
    assert "uint sgitg [[simdgroup_index_in_threadgroup]]" in candidate.source
    assert "uint tiisg [[thread_index_in_simdgroup]]" in candidate.source
    assert candidate.source.count("volatile device ") == 5
    assert f"{plan['q']}[_p515_row * 64u + _p515_kd_q]" in candidate.source
    assert f"{plan['out']}[_p515_row * 64u + _p523_od]" in candidate.source


@pytest.mark.parametrize("field,value", [("rows", 64), ("cols", 32), ("depth", 32), ("causal", True)])
def test_other_proved_shapes_stay_scalar(field, value):
    lowerer, plan = fixture()
    plan[field] = value
    assert build_candidate(lowerer, plan) is None


def barrier_check(source):
    """No SIMD/threadgroup barrier may be nested in an active-row branch."""
    stack = []
    seen = []
    for raw in source.splitlines():
        line = raw.split("//", 1)[0].strip()
        if "barrier(" in line:
            assert not any(re.match(r"if\b", scope) for scope in stack)
            seen.append(line)
        if "{" in line:
            stack.append(line.split("{", 1)[0])
        for _ in range(line.count("}")):
            stack.pop()
    assert sum("simdgroup_barrier" in b for b in seen) == 5
    assert sum("threadgroup_barrier(mem_flags::mem_device)" in b for b in seen) == 1


def test_ordered_chains_and_barriers():
    lowerer, plan = fixture()
    source = build_candidate(lowerer, plan).source
    barrier_check(source)
    final = source.index("threadgroup_barrier(mem_flags::mem_device)")
    assert "Out[" not in source[:final]
    assert "return;" not in source
    assert "_p515_score_d0[_p523_m_k] += _p515_q_d0 * static_cast<float>(K[" in source
    assert "_p515_acc[_p523_c_v] += _p553_p * static_cast<float>(V[" in source
    assert "_p515_denom = _p515_denom * _p515_alpha_d + _p515_p_sum_d;" in source
    assert "Qs[_p523_slot * 64u + _p515_kd_d0]" in source
    assert "S[_p523_slot * 64u + _p515_j_d1]" in source
    assert source.index("_p515_acc[_p523_c_a] *= _p515_alpha_d;") < source.index("_p515_acc[_p523_c_v] +=") < final
    with pytest.raises(AssertionError):
        barrier_check(source.replace("simdgroup_barrier(mem_flags::mem_threadgroup); // SB4", "// deleted SB4"))
    with pytest.raises(AssertionError):
        barrier_check(
            source.replace(
                "simdgroup_barrier(mem_flags::mem_threadgroup); // SB0",
                "if (_p523_active) {\n simdgroup_barrier(mem_flags::mem_threadgroup);\n}",
            )
        )


def test_complete_body_matches_reviewed_sb4_schedule():
    # Frozen cooperative_16.msl f9df782d... from the exact-output A/B. Its whole
    # body was independently normalized only for SIMD variable names and tile
    # flattening. This is intentionally a statement/ordering pin, not a numerical
    # proof or permission to inherit the old GPU result after changing this hash.
    obj, plan = fixture()
    body = build_candidate(obj, plan).source.split(") {", 1)[1].rsplit("}", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"threadgroup float (?:Qs|S)\[2048\];", "", body)
    body = re.sub(r"\s+", "", body)
    assert (
        hashlib.sha256(body.encode()).hexdigest() == "4993c0fc680512922cbbdbb7515816b09a2021a50a9bc127ce164234956664a0"
    )


def test_flattened_tiles_cover_exact_original_cells():
    # Every expression indexing either tile has one of these two column forms.
    # The 2D-to-1D rewrite changes no address, including the four owned columns.
    linear = set()
    for slot in range(32):
        for lane in range(16):
            for part in range(4):
                column = lane + 16 * part
                offset = slot * 64 + lane + 16 * part
                assert divmod(offset, 64) == (slot, column)
                assert 0 <= offset < 2048
                linear.add(offset)
    assert linear == set(range(2048))


@pytest.mark.parametrize(
    "facts,threads,shared,proved",
    [
        (None, 1024, 32768, True),
        (PipelineFacts(16, 1024, 16384), 1024, 32768, True),
        (PipelineFacts(32, 256, 16384), 1024, 32768, True),
        (PipelineFacts(32, 1024, 16384), 256, 32768, True),
        (PipelineFacts(32, 1024, 16384), 1024, 8192, True),
        (PipelineFacts(32, 1024, 0), 1024, 32768, True),
        (PipelineFacts(32, 1024, 16384), 1024, 32768, False),
        (PipelineFacts(True, 1024, 16384), 1024, 32768, True),
    ],
)
def test_unqualified_facts_keep_scalar(facts, threads, shared, proved):
    selection = select_variant(facts, device_max_threads=threads, device_max_shared=shared, route_enabled=proved)
    assert not selection.cooperative and (selection.threads, selection.shared) == (32, 0)
    validate_variant_geometry(selection, threads=32, shared=0)


def test_selected_pipeline_cannot_receive_old_stash_geometry():
    selection = select_variant(
        PipelineFacts(32, 1024, 16384), device_max_threads=1024, device_max_shared=32768, route_enabled=True
    )
    assert selection.cooperative
    validate_variant_geometry(selection, threads=512, shared=16384)
    for threads, shared in ((32, 0), (32, 16384), (512, 0), (1024, 16384)):
        with pytest.raises(MetalNonRecoverableError, match="geometry"):
            validate_variant_geometry(selection, threads=threads, shared=shared)


def test_real_admitted_lowerer_remains_scalar_and_sibling_is_explicit():
    from tests.test_attention_capability import OPEN_CASES, compile_case
    from triton_msl.codegen._biased_replay import wide_row_plan

    for case in OPEN_CASES:
        obj = compile_case(case)
        plan = wide_row_plan(obj)
        assert plan is not None
        scalar = obj.lower()
        candidate = build_candidate(obj, plan)
        assert obj._actual_dispatch_threads == case[1]
        assert obj._replay_shared_bytes == 0
        assert "__attncoop16" not in scalar
        assert (candidate is not None) == (case[1] == 32)


@pytest.mark.parametrize(
    "selection",
    [VariantSelection(1, 512, 16384, "forged"), VariantSelection(True, 512.0, 16384, "forged")],
)
def test_geometry_rejects_noncanonical_selection(selection):
    with pytest.raises(MetalNonRecoverableError, match="geometry"):
        validate_variant_geometry(selection, threads=512, shared=16384)


@pytest.mark.parametrize("name", ["Qs", "S", "sgitg", "tiisg", "_p523_slot", "pid", "lid", "tid"])
def test_source_argument_cannot_collide_with_cooperative_locals(name):
    obj, plan = fixture()
    obj.kb.args[0].name = name
    plan["q"] = name
    before = obj.kb.build()
    candidate = build_candidate(obj, plan)
    assert obj.kb.build() == before and obj.kb.args[0].name == name
    assert "volatile device half* _tmattn_arg0 [[buffer(0)]]" in candidate.source
    assert "static_cast<float>(_tmattn_arg0[_p515_row * 64u + _p515_kd_q])" in candidate.source


def test_argument_renaming_avoids_existing_argument_name():
    obj, plan = fixture()
    obj.kb.args[0].name = "Qs"
    obj.kb.args[1].name = "_tmattn_arg0"
    plan.update(q="Qs", k="_tmattn_arg0")
    source = build_candidate(obj, plan).source
    assert "half* _tmattn_arg0_ [[buffer(0)]]" in source
    assert "half* _tmattn_arg0 [[buffer(1)]]" in source
