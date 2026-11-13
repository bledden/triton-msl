"""Real native-IR proof and normal emitter, with no compiler or device calls."""

import hashlib
import json
import itertools
import pytest
from test_source_program_integration import parse, FIXTURES
from triton_msl.backend.compiler import MetalOptions
from triton_msl.codegen import msl_emitter
from triton_msl.codegen._source_program import entry, precision, semantics, cooperative
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.backend._source_program_admission import _range
from triton._C.libtriton import ir


def emit(branch, enabled):
    md = {}
    mod = parse(FIXTURES / (branch + ".ttgir"))
    opts = MetalOptions(experimental_source_program_cooperative=enabled)
    return mod, msl_emitter.emit_msl(mod, md, opts), md, opts


@pytest.mark.parametrize("value", [1, 0, "1", None, [], {}])
def test_option_not_truthy_coercion(value):
    with pytest.raises(ValueError, match="exact bool"):
        MetalOptions(experimental_source_program_cooperative=value)


def test_distinct_option_cache_keys():
    assert MetalOptions().hash() != MetalOptions(experimental_source_program_cooperative=True).hash()


@pytest.mark.parametrize("branch", ["masked", "padded"])
def test_normal_emitter_and_unchanged_recurrence(branch):
    mod, source, md, opts = emit(branch, True)
    p, _ = precision.capture(mod, ir, walk_ttgir, opts, semantics, GenericLowerer)
    assert source == cooperative.render(p, md["name"])[0]
    d = md["batched_dot_bounds"][1]
    assert md["block_size"] == 128 and md["shared"] == 13088
    assert d["threads"] == [128, 1, 1] and d["outputs"] == [0, 1]
    assert d["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert md["flash_attention"] is None
    assert "blocked:" in d["runtime_admission"]
    assert d["full_tile_domain"]["N"] == [32, 64, 128, 256]
    expected = (FIXTURES / "external_cooperative" / (branch + "-cooperative.metal")).read_text()
    assert source == expected


@pytest.mark.parametrize(
    "branch,n,h,batch", itertools.product(["masked", "padded"], [32, 64, 128, 256], [1, 2], [1, 2])
)
def test_full_tile_address_enclosure(branch, n, h, batch):
    _, _, md, _ = emit(branch, True)
    sizes = [
        batch * h * n * n * 32,
        batch * h * n * n,
        *([batch * h * n * n * 32] * 3),
        batch * h * n * n,
        batch * n * n,
    ]
    for record in md["batched_dot_bounds"][1]["addresses"]:
        bounds = [_range(e, n, h, batch) for e in record["links"]]
        lo, hi = sum(x[0] for x in bounds), sum(x[1] for x in bounds)
        assert lo == 0 and hi == sizes[record["root"]] - 1
        assert hi < 2**31


@pytest.mark.parametrize("branch", ["masked", "padded"])
def test_default_bytes_and_metadata_match_retained_parent(branch):
    _, source, md, _ = emit(branch, False)
    folder = FIXTURES / "external_cooperative"
    assert source == (folder / (branch + "-scalar.metal")).read_text()
    assert json.loads(json.dumps(md)) == json.loads((folder / (branch + "-scalar.json")).read_text())


@pytest.mark.parametrize("branch", ["masked", "padded"])
def test_enabled_still_refuses_native_precision_change(branch, tmp_path):
    text = (FIXTURES / (branch + ".ttgir")).read_text()
    line = next(s for s in text.splitlines() if "= tt.dot " in s)
    changed = text.replace(line, line.replace(" : ", ", inputPrecision = tf32 : ", 1), 1)
    path = tmp_path / "bad.ttgir"
    path.write_text(changed)
    with pytest.raises(Exception, match="native precision is not IEEE"):
        msl_emitter.emit_msl(parse(path), {}, MetalOptions(experimental_source_program_cooperative=True))
