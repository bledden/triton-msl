"""The biased FlashAttention backward templates replay only addresses they can prove.

The backward detector reads each pointer's strides by their position in the address chain, and
the dK/dV, dQ/delta and dBias templates then rebuild every address from their own coordinates
(program ids 0-2 and the loop). A row or depth index shifted or scaled, a side tensor's head or
instance offset moved, an output address shifted or a loop advancing differently all ran the
canonical address before. ``_witness_biased_fa_backward`` now evaluates the source addresses
exactly and binds the strides by value.

Each one-change variant of trifast's ``_bwd_kv``, ``_bwd_q`` and ``_bwd_b`` (copied in
tests/test_fa_bwd_routing.py) either refuses with "biased-FA backward:" or computes exactly what
the SOURCE computes: a float64 evaluation of the kernel's formula (lse and delta are inputs, not
recomputed) on the variant's own address spelling, into poisoned outputs.
"""

import math

import pytest
import torch
import triton

from tests.test_fa_bwd_routing import _bwd_b, _bwd_kv, _bwd_q
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = 77.0
HC, HH, INST, N, DIM = 4, 2, 3, 48, 32  # combined heads, heads per mask batch, instances, length, head dim
SM = 1.0 / math.sqrt(DIM)
FRAGMENT = "biased-FA backward: {}:"


def _variant(kernel, *edits):
    fn = triton.JITFunction(kernel.fn)
    src = fn.src
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    fn._unsafe_update_src(src)
    return fn


@pytest.fixture
def fa_hits(monkeypatch):
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits, real = [], fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


# ---------------------------------------------------------------------------------------
# Addresses: each role in (strides, head, instance, tile index, tile index)
# ---------------------------------------------------------------------------------------


def _a4(s, h, i, x, y):
    return h * s[0] + i * s[1] + x * s[2] + y * s[3]


def _a3(s, h, i, x):
    return h * s[0] + i * s[1] + x * s[2]


def _bias(s, h, i, j, k):
    return h * s[0] + j * s[1] + k * s[2]


def _mask(s, h, i, k):
    return (h // HH) * s[0] + i * s[1] + k * s[2]


def _db(s, h, j, k):
    return h * s[0] + j * s[1] + k * s[2]


CANON = dict(
    q=_a4, k=_a4, v=_a4, do=_a4, o=_a4, dq=_a4, dk=_a4, dv=_a4, b=_bias, lse=_a3, delta=_a3, mask=_mask, db=_db
)
ORDER = {
    "kv": ("delta", "q", "k", "v", "b", "lse", "mask", "do", "dk", "dv"),
    "q": ("delta", "q", "k", "v", "b", "lse", "mask", "o", "do", "dq"),
    "b": ("delta", "q", "k", "v", "b", "lse", "mask", "do", "db"),
}
OUTPUTS = {"kv": ("dk", "dv"), "q": ("dq", "delta"), "b": ("db",)}
KERNELS = {"kv": _bwd_kv, "q": _bwd_q, "b": _bwd_b}


def _inputs(kind, seed):
    """Every tensor larger than its logical shape in every axis, each with its own padding."""
    gen = torch.Generator().manual_seed(seed)
    # The leading pair holds (head, instance); kv/q swap them in some variants, b loops i to 2N.
    lead = (HC + 2, HC + 2) if kind != "b" else (HC + 1, 2 * N + 2)
    rows = 2 * N + 8
    t = {}
    for pad, name in enumerate(("q", "k", "v", "do", "o")):
        t[name] = torch.randn(*lead, rows + pad, DIM + 1 + pad % 3, generator=gen)
    t["b"] = torch.randn(HC + 1, rows + 1, rows + 3, generator=gen)
    t["lse"] = torch.randn(*lead, rows + 2, generator=gen) * 0.5 + 2.0
    t["delta"] = torch.randn(*lead, rows + 3, generator=gen) * 0.5
    t["mask"] = (torch.rand(*lead, rows + 4, generator=gen) < 0.15).to(torch.uint8)
    for name in ("dq", "dk", "dv"):
        t[name] = torch.full((*lead, rows + 5, DIM + 2), POISON)
    t["db"] = torch.full((HC + 1, rows + 3, rows + 5), POISON)
    if kind == "q":
        t["delta"] = torch.full(t["delta"].shape, POISON)  # delta is an output of _bwd_q
    return t


def _launch(kind, fn, t):
    dev = {name: x.to("mps") for name, x in t.items()}
    args = []
    for name in ORDER[kind]:
        args += [dev[name], *dev[name].stride()]
    if kind == "b":
        grid, tail = (triton.cdiv(N, 32), triton.cdiv(N, 32), HC), (SM, -1e9, HH, N, DIM, N, 32, 32)
    else:
        grid, tail = (triton.cdiv(N, 32), INST, HC), (SM, -1e9, N, HH, DIM, N, 32, 32)
    fn[grid](*args, *tail)
    torch.mps.synchronize()
    return [dev[name] for name in OUTPUTS[kind]]


def _gather(storage, offsets):
    return storage.reshape(-1).double()[offsets]


def _place(expected, offsets, values):
    cells = offsets.reshape(-1)
    assert cells.numel() == cells.unique().numel(), "a variant whose programs store one cell twice"
    expected.view(-1)[cells] = values.expand(offsets.shape).reshape(-1)


def _scores(a, h, i, j, k, sm_on_q):
    """P (the loaded-lse probabilities) and the effective Q, K, V, dO of one (head, instance)."""
    d = torch.arange(DIM)[None, :]
    q = _gather(a["t"]["q"], a["q"](a["s"]["q"], h, i, j, d))
    kk = _gather(a["t"]["k"], a["k"](a["s"]["k"], h, i, k, d))
    v = _gather(a["t"]["v"], a["v"](a["s"]["v"], h, i, k, d))
    do = _gather(a["t"]["do"], a["do"](a["s"]["do"], h, i, j, d))
    bias = _gather(a["t"]["b"], a["b"](a["s"]["b"], h, i, j, k.T))
    lse = _gather(a["t"]["lse"], a["lse"](a["s"]["lse"], h, i, j))
    masked = _gather(a["t"]["mask"], a["mask"](a["s"]["mask"], h, i, k.T)) != 0
    s = (q * SM) @ kk.T + bias if sm_on_q else q @ (kk * SM).T + bias
    p = torch.exp(s - lse).masked_fill(masked, 0.0)
    return p, q, kk, v, do


def _expected(kind, case, t):
    a = {**CANON, **case, "t": t, "s": {name: x.stride() for name, x in t.items()}}
    s = a["s"]
    out = {name: torch.full(t[name].shape, POISON, dtype=torch.float64) for name in OUTPUTS[kind]}
    # ``loop``: how many loop indices a changed loop step still visits (j for kv, k for q, i for b).
    j = torch.arange(case.get("loop", N) if kind == "kv" else N)[:, None]
    k = torch.arange(case.get("loop", N) if kind == "q" else N)[:, None]
    kd = torch.arange(N)[:, None]  # dK / dV rows: the program's own key tile, not the loop
    d = torch.arange(DIM)[None, :]
    if kind == "b":
        for h in range(HC):
            db = torch.zeros(N, N, dtype=torch.float64)
            for i in range(case.get("loop", N)):
                p, q, kk, v, do = _scores(a, h, i, j, k, sm_on_q=True)
                delta = _gather(t["delta"], a["delta"](s["delta"], h, i, j))
                db += p * (do @ v.T - delta)
            _place(out["db"], a["db"](s["db"], h, j, k.T), db)
        return [out["db"]]
    for h in range(HC):
        for i in range(INST):
            p, q, kk, v, do = _scores(a, h, i, j, k, sm_on_q=False)
            if kind == "kv":
                delta = _gather(t["delta"], a["delta"](s["delta"], h, i, j))
                ds = p * (do @ v.T - delta)
                _place(out["dk"], a["dk"](s["dk"], h, i, kd, d), SM * ds.T @ q)
                _place(out["dv"], a["dv"](s["dv"], h, i, kd, d), p.T @ do)
            else:
                o = _gather(t["o"], a["o"](s["o"], h, i, j, d))
                delta = (o * do).sum(-1, keepdim=True)
                ds = p * (do @ v.T - delta)
                _place(out["dq"], a["dq"](s["dq"], h, i, j, d), ds @ (kk * SM))
                _place(out["delta"], a["delta"](s["delta"], h, i, j), delta)
    return [out[name] for name in OUTPUTS[kind]]


def _case(edits=(), fragment=None, **overrides):
    return dict(edits=edits, fragment=fragment, **overrides)


# ---------------------------------------------------------------------------------------
# One-change variants per kernel (the address lines are trifast's own spelling)
# ---------------------------------------------------------------------------------------

_KV_Q = "q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)"
_KV_KT = "kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)"
_KV_L = "base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)"
_KV_DK = "dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd)"
_KV_M = "mask_ptrs = base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn)"
_MASK_BATCH = ("mask_start_h = pid_h // H", "mask_start_h = pid_h")


def _mask_by_head(s, h, i, k):
    return h * s[0] + i * s[1] + k * s[2]


KV_CASES = {
    "canonical": _case(),
    "q_row_plus_1": _case(
        [(_KV_Q, _KV_Q.replace("(j_idxs[", "((j_idxs + 1)["))],
        FRAGMENT.format("Q"),
        q=lambda s, h, i, j, d: _a4(s, h, i, j + 1, d),
    ),
    "k_depth_scaled": _case(
        [(_KV_KT, _KV_KT.replace("(d_idxs[:, None])", "((d_idxs * 2)[:, None])"))],
        FRAGMENT.format("K"),
        k=lambda s, h, i, k, d: _a4(s, h, i, k, 2 * d),
    ),
    "lse_head_plus_1": _case(
        [(_KV_L, _KV_L.replace("(start_h * stride_lh)", "((start_h + 1) * stride_lh)"))],
        FRAGMENT.format("Lse"),
        lse=lambda s, h, i, j: _a3(s, h + 1, i, j),
    ),
    "mask_key_plus_1": _case(
        [(_KV_M, _KV_M.replace("(k_idxs * stride_mn)", "((k_idxs + 1) * stride_mn)"))],
        FRAGMENT.format("Mask"),
        mask=lambda s, h, i, k: _mask(s, h, i, k + 1),
    ),
    "dk_row_plus_1": _case(
        [(_KV_DK, _KV_DK.replace("(k_idxs[", "((k_idxs + 1)["))],
        FRAGMENT.format("DK"),
        dk=lambda s, h, i, k, d: _a4(s, h, i, k + 1, d),
    ),
    "delta_loop_advance_doubled": _case(
        [("d_ptrs += BLOCK_J * stride_dn", "d_ptrs += 2 * BLOCK_J * stride_dn")],
        FRAGMENT.format("Delta"),
        delta=lambda s, h, i, j: _a3(s, h, i, j + 32 * (j // 32)),
    ),
    # The j loop visits only its first tile (N = 48 < 64).
    "loop_step_doubled": _case(
        [("for start_j in range(0, N, BLOCK_J):", "for start_j in range(0, N, 2 * BLOCK_J):")],
        "biased-FA backward: a source loop that is not range(0, sequence length",
        loop=32,
    ),
    # Must compute: the batch strides are bound by value, the mask batch is read where the source reads it.
    "lse_head_instance_swapped": _case(
        [(_KV_L, "base_l_ptr = l_ptr + (start_i * stride_lh) + (start_h * stride_lm)")],
        lse=lambda s, h, i, j: _a3(s, i, h, j),
    ),
    "mask_batch_is_the_head": _case([_MASK_BATCH], mask=_mask_by_head),
    "eq_q_index_term_order": _case(
        [(_KV_Q, "q_ptrs = base_q_ptr + (d_idxs[None, :] * stride_qd) + (j_idxs[:, None] * stride_qn)")]
    ),
}

_Q_K = "k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd)"
_Q_Q = "q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)"
_Q_DQ = "dq_ptrs = base_dq_ptr + (j_idxs[:, None] * stride_dqn) + (d_idxs[None, :] * stride_dqd)"
_Q_D = "base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)"
_Q_L = "l_ptrs = base_l_ptr + (j_idxs * stride_ln)"

Q_CASES = {
    "canonical": _case(),
    "k_row_plus_1": _case(
        [(_Q_K, _Q_K.replace("(k_idxs[", "((k_idxs + 1)["))],
        FRAGMENT.format("K"),
        k=lambda s, h, i, k, d: _a4(s, h, i, k + 1, d),
    ),
    "q_depth_scaled": _case(
        [(_Q_Q, _Q_Q.replace("(d_idxs[", "((d_idxs * 2)["))],
        FRAGMENT.format("Q"),
        q=lambda s, h, i, j, d: _a4(s, h, i, j, 2 * d),
    ),
    "bias_head_plus_1": _case(
        [("base_b_ptr = b_ptr + (start_h * stride_bh)", "base_b_ptr = b_ptr + ((start_h + 1) * stride_bh)")],
        FRAGMENT.format("Bias"),
        b=lambda s, h, i, j, k: _bias(s, h + 1, i, j, k),
    ),
    "dq_row_plus_1": _case(
        [(_Q_DQ, _Q_DQ.replace("(j_idxs[", "((j_idxs + 1)["))],
        FRAGMENT.format("DQ"),
        dq=lambda s, h, i, j, d: _a4(s, h, i, j + 1, d),
    ),
    "delta_store_head_plus_1": _case(
        [(_Q_D, _Q_D.replace("(start_h * stride_dh)", "((start_h + 1) * stride_dh)"))],
        FRAGMENT.format("Delta"),
        delta=lambda s, h, i, j: _a3(s, h + 1, i, j),
    ),
    "k_loop_advance_doubled": _case(
        [("k_ptrs += BLOCK_K * stride_kn", "k_ptrs += 2 * BLOCK_K * stride_kn")],
        FRAGMENT.format("K"),
        k=lambda s, h, i, k, d: _a4(s, h, i, k + 32 * (k // 32), d),
    ),
    "loop_step_doubled": _case(
        [("for start_k in range(0, N, BLOCK_K):", "for start_k in range(0, N, 2 * BLOCK_K):")],
        "biased-FA backward: a source loop that is not range(0, sequence length",
        loop=32,
    ),
    "lse_row_scaled": _case(
        [(_Q_L, "l_ptrs = base_l_ptr + ((j_idxs * 2) * stride_ln)")],
        FRAGMENT.format("Lse"),
        lse=lambda s, h, i, j: _a3(s, h, i, 2 * j),
    ),
    "delta_store_head_instance_swapped": _case(
        [(_Q_D, "base_d_ptr = d_ptr + (start_i * stride_dh) + (start_h * stride_dm)")],
        delta=lambda s, h, i, j: _a3(s, i, h, j),
    ),
    "mask_batch_is_the_head": _case([_MASK_BATCH], mask=_mask_by_head),
    "eq_dq_index_term_order": _case(
        [(_Q_DQ, "dq_ptrs = base_dq_ptr + (d_idxs[None, :] * stride_dqd) + (j_idxs[:, None] * stride_dqn)")]
    ),
}

_B_Q = "q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)"
_B_K = "k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :]) * stride_kd"
_B_DB = "db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)"

B_CASES = {
    "canonical": _case(),
    "q_row_plus_1": _case(
        [(_B_Q, _B_Q.replace("(j_idxs[", "((j_idxs + 1)["))],
        FRAGMENT.format("Q"),
        q=lambda s, h, i, j, d: _a4(s, h, i, j + 1, d),
    ),
    "k_depth_scaled": _case(
        [(_B_K, _B_K.replace("(d_idxs[None, :])", "((d_idxs * 2)[None, :])"))],
        FRAGMENT.format("K"),
        k=lambda s, h, i, k, d: _a4(s, h, i, k, 2 * d),
    ),
    "bias_head_plus_1": _case(
        [("base_b_ptr = b_ptr + (start_h * stride_bh)", "base_b_ptr = b_ptr + ((start_h + 1) * stride_bh)")],
        FRAGMENT.format("Bias"),
        b=lambda s, h, i, j, k: _bias(s, h + 1, i, j, k),
    ),
    "delta_row_plus_1": _case(
        [("d_ptrs = base_d_ptr + (j_idxs * stride_dn)", "d_ptrs = base_d_ptr + ((j_idxs + 1) * stride_dn)")],
        FRAGMENT.format("Delta"),
        delta=lambda s, h, i, j: _a3(s, h, i, j + 1),
    ),
    "db_key_plus_1": _case(
        [(_B_DB, _B_DB.replace("k_idxs[None, :]", "(k_idxs + 1)[None, :]"))],
        FRAGMENT.format("DB"),
        db=lambda s, h, j, k: _db(s, h, j, k + 1),
    ),
    # The i loop is pointer increments only: a doubled increment is another instance stride.
    "q_loop_advance_doubled": _case(
        [("q_ptrs += stride_qm * BLOCK_I", "q_ptrs += 2 * stride_qm * BLOCK_I")],
        FRAGMENT.format("Q"),
        q=lambda s, h, i, j, d: _a4(s, h, 2 * i, j, d),
    ),
    # Every other instance step: the pointers still advance one instance per iteration.
    "loop_step_doubled": _case(
        [("for start_i in range(0, N, BLOCK_I):", "for start_i in range(0, N, 2 * BLOCK_I):")],
        "biased-FA backward: a source loop that is not range(0, sequence length",
        loop=N // 2,
    ),
    "mask_batch_is_the_head": _case([_MASK_BATCH], mask=_mask_by_head),
    "eq_db_inner_term_order": _case(
        [(_B_DB, "db_ptrs = base_db_ptr + (k_idxs[None, :] * stride_dbn + j_idxs[:, None] * stride_dbm)")]
    ),
}

CASES = {"kv": KV_CASES, "q": Q_CASES, "b": B_CASES}


@requires_mps
@pytest.mark.parametrize("kind,name", [(kind, name) for kind in CASES for name in sorted(CASES[kind])])
def test_backward_variant_refuses_or_computes_the_source(kind, name, fa_hits):
    case = CASES[kind][name]
    fn = _variant(KERNELS[kind], *case["edits"])
    t = _inputs(kind, seed=1601)
    try:
        outputs = _launch(kind, fn, t)
    except MetalNonRecoverableError as exc:
        assert case["fragment"] is not None, f"refused a variant that must compute: {exc}"
        assert case["fragment"] in str(exc)
        assert fa_hits == []
        return
    if case["fragment"] is None:
        assert fa_hits == [True]
    for got, expected in zip(outputs, _expected(kind, case, t)):
        got = got.cpu().double()
        written = expected != POISON
        assert torch.equal(got != POISON, written), f"{int(((got != POISON) != written).sum())} cells differ in domain"
        scale = max(1.0, float(expected[written].abs().max()))
        torch.testing.assert_close(got[written], expected[written], atol=2e-4 * scale, rtol=2e-3)
