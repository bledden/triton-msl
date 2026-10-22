"""CPU-only source/index proofs for the experimental tiled K layout.

These do not compile MSL, query a device, establish numeric bits/performance, or
qualify generic shapes for production. No dependency on local evidence paths.
"""

import difflib
import re

import pytest

from triton_msl.codegen._msl_templates import make_flash_attention_kernel_tiled


# Rectangular tiles in both directions, physical-loop iterations, multiple D
# chunks, tiny tiles, and source-query programs using smaller resident subtiles.
CASES = [
    (32, 32, 64, 64, {}),
    (16, 64, 64, 32, {"causal": True}),
    (64, 16, 128, 32, {"out_dtype": "fp16", "round_q": "half", "round_p": "half"}),
    (8, 16, 96, 16, {"out_dtype": "bf16", "round_q": "bfloat", "round_p": "bfloat"}),
    (16, 8, 48, 16, {"scale": 0.0}),
    (32, 64, 64, 16, {"bias": True, "mask": True, "lse": True, "runtime_scale": True, "grid_3d": True}),
    (64, 32, 64, 32, {"source_kv_block": 32, "causal": True, "source_causal_stop": True}),
    (1, 1, 16, 16, {}),
    (4, 8, 32, 8, {"lse": True}),
    (32, 16, 128, 64, {"source_kv_block": 16, "mask_tail_scores": False, "unmasked_kv": ("K", "V")}),
    (64, 32, 64, 32, {"query_tile": 32, "source_kv_block": 32, "causal": True, "source_causal_stop": True}),
    (64, 64, 64, 16, {"query_tile": 32, "source_kv_block": 64}),
]


def _emit(case, **options):
    bm, bn, d, dc, flags = case
    return make_flash_attention_kernel_tiled(d, bm, bn, Dc=dc, **(flags | options))


def _sites(source):
    result = {}
    for role in ("K", "V"):
        hits = re.findall(r"tg_KV\[([^\]]+)\] = [^\n]+\n\s*\? " + role + r"\[([^\]]+)\]", source)
        assert len(hits) == 1, (role, hits)
        result[role] = hits[0]
    score = re.findall(r"dot \+= qv \* tg_KV\[([^\]]+)\];", source)
    pv = re.findall(r"o_val \+= tg_S\[r \* BN \+ cj\] \* tg_KV\[([^\]]+)\];", source)
    assert len(score) == len(pv) == 1
    result.update(score=score[0], pv=pv[0])
    return result


def _index(expression, **values):
    assert re.fullmatch(r"[a-zA-Z0-9_ *+]+", expression)
    assert set(re.findall(r"[A-Za-z_]\w*", expression)) <= set(values)
    return eval(expression, {"__builtins__": {}}, values)


def _prove_layout(source):
    constants = {k: int(v) for k, v in re.findall(r"const uint (BM|BN|D|DC|TPG)\s*=\s*(\d+)u;", source)}
    assert set(constants) == {"BM", "BN", "D", "DC", "TPG"}
    bm, bn, d, dc, tpg = (constants[k] for k in ("BM", "BN", "D", "DC", "TPG"))
    sites = _sites(source)
    tile = int(re.search(r"threadgroup float tg_KV\[(\d+)\]", source).group(1))
    inverse = {}
    # Actual physical-thread stride, not an assumption of one element/thread.
    for lid in range(tpg):
        for i in range(lid, bn * dc, tpg):
            r, c = divmod(i, dc)
            index = _index(sites["K"][0], **constants, i=i, r=r, c=c)
            assert 0 <= index < tile and index not in inverse
            inverse[index] = (r, c)
    assert set(inverse) == set(range(bn * dc))
    # Exhaust all score threads/columns; chunks change the source K column but
    # preserve the tile bijection and ascending per-dot accumulation order.
    for chunk in range(0, d, dc):
        for lid in range(tpg):
            for i in range(lid, bm * bn, tpg):
                r, cj = divmod(i, bn)
                for c in range(dc):
                    index = _index(sites["score"], **constants, i=i, r=r, cj=cj, c=c)
                    assert inverse[index] == (cj, c)
                    assert 0 <= chunk + c < d
    return constants, sites


def _reverse(source):
    sites = _sites(source)
    assert sites["K"][0] == "c * BN + r" and sites["score"] == "c * BN + cj"
    old_write = "tg_KV[c * BN + r] ="
    old_read = "dot += qv * tg_KV[c * BN + cj];"
    assert source.count(old_write) == source.count(old_read) == 1
    return source.replace(old_write, "tg_KV[i] =", 1).replace(old_read, "dot += qv * tg_KV[cj * DC + c];", 1)


@pytest.mark.parametrize("case", CASES)
def test_default_off_and_exact_two_hunk_round_trip(case):
    original = _emit(case)
    assert original == _emit(case, transpose_k=False)
    candidate = _emit(case, transpose_k=True)
    assert _reverse(candidate) == original
    diff = list(difflib.unified_diff(original.splitlines(), candidate.splitlines()))
    assert sum(x.startswith("@@") for x in diff) == 2
    assert sum(x.startswith("+") and not x.startswith("+++") for x in diff) == 2
    assert sum(x.startswith("-") and not x.startswith("---") for x in diff) == 2


@pytest.mark.parametrize("case", CASES)
def test_generic_k_bijection_and_score_value_identity(case):
    reference, candidate = _emit(case), _emit(case, transpose_k=True)
    rc, rs = _prove_layout(reference)
    cc, cs = _prove_layout(candidate)
    assert rc == cc
    assert rs["K"][1] == cs["K"][1]  # Device address, including runtime inner stride.
    assert rs["V"] == cs["V"] and rs["pv"] == cs["pv"]
    assert cc["TPG"] == min(cc["BM"] * cc["BN"], 1024)


@pytest.mark.parametrize("case", CASES)
def test_barrier_numerical_abi_and_resource_declarations_unchanged(case):
    reference, candidate = _emit(case), _emit(case, transpose_k=True)
    assert _reverse(candidate) == reference  # Includes every arithmetic/rounding statement.
    for pattern in (r"threadgroup_barrier\([^\n]+", r"threadgroup float [^\n]+", r"[^\n]*\[\[buffer\(\d+\)\]\][^\n]*"):
        assert re.findall(pattern, reference) == re.findall(pattern, candidate)
    assert reference.count("threadgroup_barrier(") == candidate.count("threadgroup_barrier(")
    assert reference.count("i += TPG") == candidate.count("i += TPG") == 7
    assert _sites(reference)["V"] == _sites(candidate)["V"]
    assert _sites(reference)["pv"] == _sites(candidate)["pv"]


@pytest.mark.parametrize("invalid", [None, 0, 1, -1, 0.0, 1.0, "true", [], {}])
def test_transpose_requires_exact_bool(invalid):
    with pytest.raises((ValueError, TypeError), match="transpose_k"):
        _emit(CASES[0], transpose_k=invalid)


@pytest.mark.parametrize("case", [CASES[0], CASES[1], CASES[2]])
@pytest.mark.parametrize("mutant", ["write-only", "read-only", "wrong-stride", "V-change"])
def test_mutants_are_discriminated(case, mutant):
    candidate = _emit(case, transpose_k=True)
    if mutant == "write-only":
        changed = candidate.replace("dot += qv * tg_KV[c * BN + cj];", "dot += qv * tg_KV[cj * DC + c];", 1)
    elif mutant == "read-only":
        changed = candidate.replace("tg_KV[c * BN + r] =", "tg_KV[i] =", 1)
    elif mutant == "wrong-stride":
        changed = candidate.replace("tg_KV[c * BN + r] =", "tg_KV[c * DC + r] =", 1)
    else:
        changed = candidate.replace("tg_KV[i] =", "tg_KV[c * BN + r] =", 1)
    assert changed != candidate
    if mutant == "V-change":
        assert _sites(changed)["V"] != _sites(_emit(case))["V"]
        with pytest.raises(AssertionError):
            _reverse(changed)
    elif mutant == "wrong-stride" and case[1] == case[3]:
        # For square BN/DC, this spelling is algebraically identical. The
        # exact-transformation guard must still catch it; no false numeric claim.
        _prove_layout(changed)
        with pytest.raises(AssertionError):
            _reverse(changed)
    else:
        with pytest.raises((AssertionError, KeyError)):
            _prove_layout(changed)
