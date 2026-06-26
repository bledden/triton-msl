# tests/test_matmul_tuner.py
"""Unit tests for the deterministic, occupancy-gated matmul rr/rc selector.

No GPU needed: selection is pure logic (cores pinned to 40 -> gate = 8*40 = 320).
The selector NEVER returns an invalid (rr,rc); for low-occupancy unaligned shapes
it returns None (-> the caller uses the generic path, never-regress)."""
import pytest

from triton_msl.autotuning import matmul_tuner as MT
from triton_msl.autotuning.matmul_tuner import CANDIDATES, valid_candidates, best_rrrc


@pytest.fixture(autouse=True)
def _fixed_env(monkeypatch):
    monkeypatch.setattr(MT, "_CORES", 40)               # gate = 8*40 = 320
    monkeypatch.delenv("TRITON_MSL_MATMUL_AUTOTUNE", raising=False)
    yield


def test_candidates_default_first_register_safe_includes_rr1():
    assert CANDIDATES[0] == (4, 4)                       # default + aligned common case
    assert all(rr * rc <= 32 for rr, rc in CANDIDATES)   # register budget
    assert any(rr == 1 for rr, rc in CANDIDATES)         # rr=1 -> M%8 coverage


def test_valid_candidates_N_contract_is_strip_width_8rc():
    # N=2048 (%64==0): every rc valid; M=2048 (%64==0): every rr -> all candidates.
    assert set(valid_candidates(2048, 2048, 2048)) == set(CANDIDATES)
    # N must align to the per-simdgroup STRIP width (8*rc), NOT the full 32*rc tile — the
    # kernel's `col0 >= N` guard masks the overshoot (verified byte-exact in the dispatch).
    # So N=2080 (%32==0, %128!=0), which the old 32*rc gate WRONGLY excluded (the matmul
    # N%32 perf cliff), is valid via every rc whose strip 8*rc divides it.
    v = valid_candidates(256, 2080, 256)
    assert v and all(2080 % (8 * rc) == 0 for rr, rc in v)
    # A genuinely strip-unaligned N (N % 8 != 0) stays empty — a partial strip would write
    # past N, so it correctly routes to the generic path.
    assert valid_candidates(256, 2050, 256) == []
    # M=48 (%16==0, %32!=0): only rr in {2,1}; (4,4) excluded.
    v = valid_candidates(48, 2048, 2048)
    assert (4, 4) not in v and all(rr in (1, 2) for rr, rc in v)
    # K not %8 -> nothing valid.
    assert valid_candidates(2048, 2048, 60) == []


def test_aligned_returns_default():
    assert best_rrrc("fp32", "fp32", 2048, 2048, 2048) == (4, 4)


def test_unaligned_large_M16_enables_rr2():
    # M=2032 (%32==16 -> not aligned; %16==0 -> rr=2). Large -> passes gate.
    cfg = best_rrrc("fp32", "fp32", 2032, 2048, 2048)
    assert cfg is not None and cfg[0] == 2 and 2032 % (8 * cfg[0]) == 0


def test_unaligned_large_M8_uses_rr1():
    # M=2040 (%16==8 -> rr=2 invalid; %8==0 -> rr=1). Large -> passes gate.
    cfg = best_rrrc("fp32", "fp32", 2040, 2048, 2048)
    assert cfg is not None and cfg[0] == 1 and 2040 % 8 == 0


def test_low_parallelism_routes_to_generic():
    # A genuinely low-parallelism shape (small M AND small N) has too few threadgroups to
    # beat generic, so the occupancy gate returns None. NB: small M with LARGE N now uses
    # the fast rc=1 path (measured ~14x faster than generic, byte-exact) — so M alone no
    # longer routes to generic; only true low total parallelism does.
    assert best_rrrc("fp32", "fp32", 48, 48, 2048) is None
    assert best_rrrc("fp32", "fp32", 48, 2048, 2048) is not None   # large N -> fast (rc=1)


def test_killswitch_pins_default_or_none(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_MATMUL_AUTOTUNE", "0")
    assert best_rrrc("fp32", "fp32", 2048, 2048, 2048) == (4, 4)   # aligned -> (4,4)
    assert best_rrrc("fp32", "fp32", 2032, 2048, 2048) is None     # unaligned -> generic


def test_no_valid_candidate_returns_none():
    assert best_rrrc("fp32", "fp32", 64, 40, 64) is None           # N=40 not %64


def test_selector_never_returns_invalid_config():
    # Exhaustive-ish: across a grid of shapes, any returned (rr,rc) must satisfy the
    # size contract (the prime-directive safety property).
    for M in (2048, 2032, 2040, 1024, 512, 48, 24, 8, 100):
        for N in (2048, 1024, 512, 256, 100):
            for K in (2048, 512, 64):
                cfg = best_rrrc("fp32", "fp32", M, N, K)
                if cfg is not None:
                    rr, rc = cfg
                    assert M % (8 * rr) == 0 and N % (32 * rc) == 0 and K % 8 == 0
