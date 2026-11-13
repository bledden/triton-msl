"""CPU-only fragment composition tests. Run with --noconftest (root probes Metal).

No torch, framework import, native compiler, driver activation or device query.
The TTGIR parser/emitter is exercised; that is not a GPU correctness test.
"""

import hashlib
import json
from pathlib import Path

import pytest
from triton._C.libtriton import ir
from triton_msl.backend.compiler import MetalOptions
from triton_msl.codegen import mlir_walker, msl_emitter
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen._fa_fragments import MODES, row_loop
from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup as maker

FIXTURES = Path(__file__).parent / "fixtures/fa_fragments"
ARMS = dict(off="C", expanded="C_U", private="S_U", output="B_U", **{"private-output": "SB_U", "shared": "F_U"})


def lower(mode, path=None):
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(path or FIXTURES / "_flash_attn_fwd.ttgir"), ctx)
    opts = MetalOptions(experimental_fa_fragment_lowering=mode)
    low = GenericLowerer(mlir_walker.walk_ttgir(mod, opts), opts)
    source = low.lower()
    metadata = {"num_warps": 4, "num_ctas": 1, "execution_contract": "controlled"}
    assert msl_emitter.emit_msl(mod, metadata, opts) == source
    return source, metadata, low


def test_option_cache_keys_and_default():
    assert MetalOptions().experimental_fa_fragment_lowering == "off"
    assert len({MetalOptions(experimental_fa_fragment_lowering=m).hash() for m in MODES}) == len(MODES)
    assert MetalOptions().hash() == MetalOptions(experimental_fa_fragment_lowering="off").hash()


@pytest.mark.parametrize("bad", [None, True, False, 1, [], "F_U", "SHARED", ""])
def test_unknown_or_coerced_options_refuse(bad):
    with pytest.raises(ValueError, match="experimental_fa_fragment_lowering"):
        MetalOptions(experimental_fa_fragment_lowering=bad)


@pytest.mark.parametrize("mode", MODES)
def test_real_lowerer_and_emitter_preserve_reviewed_sources(mode):
    source, metadata, low = lower(mode)
    if mode in ARMS:
        assert source == (FIXTURES / (ARMS[mode] + ".metal")).read_text()
    else:
        # F+B is a NEW source. Only its final private scratch barriers change.
        old = (FIXTURES / "F_U.metal").read_text()
        # Select all four epilogue rows, not just the last one.
        cut = old.index("    {\n\n        for (uint t=0u;t<TPG;t++) {", old.index("// one partial tail block"))
        assert old[cut:].count("threadgroup_barrier(") == 8
        assert source == old[:cut] + old[cut:].replace("threadgroup_barrier(", "simdgroup_barrier(")
    assert low.effective_block_size == 256
    assert low._flash_attention == ("flash_attention", source, 256)
    if mode == "off":
        assert "fa_fragment_lowering" not in metadata
    else:
        assert metadata["fa_fragment_lowering"] == dict(requested=mode, selected=mode, reason="selected")
        assert "for (uint rb=" not in source and "@ROW@" not in source
    assert "source KV tail: 32" in source
    assert "isfinite" in source and "? NAN" in source


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("scale", "scale"),
        ("initial", "source-initial-values"),
    ],
)
def test_real_source_neighbors_keep_original_route(tmp_path, mutation, reason):
    text = (FIXTURES / "_flash_attn_fwd.ttgir").read_text()
    if mutation == "scale":
        assert text.count("dense<1.250000e-01>") == 1
        text = text.replace("dense<1.250000e-01>", "dense<2.500000e-01>")
    else:
        old = "%cst_1 = arith.constant dense<0.000000e+00>"
        assert text.count(old) == 1
        text = text.replace(old, "%cst_1 = arith.constant dense<1.000000e+00>")
    path = tmp_path / "neighbor.ttgir"
    path.write_text(text)
    baseline, _, _ = lower("off", path)
    for mode in MODES[1:]:
        source, metadata, _ = lower(mode, path)
        assert source == baseline
        assert metadata["fa_fragment_lowering"] == dict(requested=mode, selected="off", reason=reason)


def test_half_accumulation_does_not_activate_composition(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "1")
    baseline, _, _ = lower("off")
    assert "simdgroup_half8x8 o[4][TPG]" in baseline
    source, metadata, _ = lower("shared")
    assert source == baseline
    assert metadata["fa_fragment_lowering"]["reason"] == "precision"


@pytest.mark.parametrize(
    "updates",
    [
        dict(head_dim=128),
        dict(head_dim=64.0),
        dict(v_head_dim=64.0),
        dict(initial_values=(-0.0, 0.0)),
        dict(v_head_dim=128),
        dict(out_dtype="fp32"),
        dict(half_accumulate=True),
        dict(causal=True),
        dict(source_kv_block=64),
        dict(source_kv_block=None),
        dict(mask_tail_scores=True),
        dict(unmasked_kv=("V",)),
        dict(source_causal_stop=True),
        dict(initial_values=(1.0, 0.0)),
        dict(round_q="half"),
        dict(round_p="half"),
        dict(scale_chain=("half",)),
        dict(scale=0.25),
        dict(bias=True),
        dict(mask=True),
        dict(lse=True),
        dict(runtime_scale=True),
        dict(grid_3d=True),
        dict(runtime_neg_inf=True),
    ],
)
def test_direct_maker_cannot_expand_scope(updates):
    args = dict(head_dim=64, out_dtype="fp16", source_kv_block=32, mask_tail_scores=False)
    args.update(updates)
    with pytest.raises(ValueError, match="fragment lowering outside prepared scope"):
        maker(**args, fragment_lowering="shared")


@pytest.mark.parametrize("row", json.loads((FIXTURES / "baseline_emissions.json").read_text()))
def test_other_default_emissions_byte_identical(row):
    args = row["args"]
    for name in ("initial_values", "unmasked_kv", "scale_chain"):
        if name in args and args[name] is not None:
            args[name] = tuple(args[name])
    source = maker(**args)
    assert hashlib.sha256(source.encode()).hexdigest() == row["sha256"]


def test_shared_output_lifetime_and_ownership():
    source, _, _ = lower("shared-output")
    # Both full/tail publish before every group reads coefficients. Their end
    # barriers retire those reads before any future coefficient/epilogue write.
    blocks = source.split("// Replicate each row coefficient")[1:]
    assert len(blocks) == 2
    for block in blocks:
        publish = block.index("on_scratch[lid] =")
        first_read = block.index("simdgroup_load(coefficient,")
        assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in block[publish:first_read]
        last_read = block.index("simdgroup_load(coefficient, on_scratch + 3u*64u")
        pv = block.index("simdgroup_multiply_accumulate(o[3][t]", last_read)
        assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in block[pv:]
    # 256 writers publish exactly 32x8 coefficients; each matrix row is constant.
    assert set(range(256)) == {row * 8 + col for row in range(32) for col in range(8)}
    for group in range(8):
        own = {group * 64 + lane + step for lane in range(32) for step in (0, 32)}
        assert own == set(range(group * 64, (group + 1) * 64))
        for other in range(8):
            if group != other:
                assert not own & set(range(other * 64, (other + 1) * 64))
    # Final normalization, masks and stores remain byte equal to the accepted F_U.
    assert source.count("simdgroup_barrier(mem_flags::mem_threadgroup);") == 9  # 8 output + 1 tail staging


def test_renderer_expands_only_owned_placeholder():
    body = "\nfloat a = tg_alpha[@ROW@*8u];\no[@ROW@][t] *= a;\n"
    expected = "\n".join("{\n" + body.replace("@ROW@", str(i) + "u") + "\n}" for i in range(4))
    assert row_loop(body, True) == expected
    assert row_loop(body, False) == "for (uint rb=0u;rb<4u;rb++) {" + body.replace("@ROW@", "rb") + "}"
    with pytest.raises(ValueError, match="placeholder"):
        row_loop("a += 1;", True)


def test_fixture_provenance():
    for row in json.loads((FIXTURES / "provenance.json").read_text()):
        assert hashlib.sha256((FIXTURES / row["file"]).read_bytes()).hexdigest() == row["sha256"]


def test_source_value_proof_still_precedes_experimental_maker(tmp_path, monkeypatch):
    from triton_msl.codegen import _msl_templates
    from triton_msl.errors import MetalNonRecoverableError

    text = (FIXTURES / "_flash_attn_fwd.ttgir").read_text()
    old = "%acc_71 = arith.mulf %acc_40, %acc_70"
    assert text.count(old) == 1
    path = tmp_path / "wrong_recurrence.ttgir"
    path.write_text(text.replace(old, "%acc_71 = arith.addf %acc_40, %acc_70"))

    def must_not_emit(*args, **kwargs):
        pytest.fail("unproved source reached experimental maker")

    monkeypatch.setattr(_msl_templates, "make_flash_attention_kernel_simdgroup", must_not_emit)
    with pytest.raises(MetalNonRecoverableError):
        lower("shared", path)
