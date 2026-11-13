"""Explicit row-fragment emission for the default-off dense FA experiment.

No MSL parser or fragment-to-lane mapping: bodies use an emitter-owned @ROW@
placeholder and are emitted in row order, with a separate lexical scope per row.
"""

MODES = ("off", "expanded", "private", "output", "private-output", "shared", "shared-output")


def validate_mode(mode):
    if type(mode) is not str or mode not in MODES:
        raise ValueError("experimental_fa_fragment_lowering must be one of " + ", ".join(MODES))
    return mode


def scope_reason(
    *,
    head_dim,
    v_head_dim=None,
    out_dtype="fp32",
    causal=False,
    half_accumulate=False,
    source_kv_block=None,
    mask_tail_scores=True,
    unmasked_kv=(),
    source_causal_stop=None,
    initial_values=None,
    bias=False,
    mask=False,
    lse=False,
    runtime_scale=False,
    grid_3d=False,
    mask_batch_div=None,
    runtime_neg_inf=False,
    round_q=None,
    round_p=None,
    scale_chain=(),
    scale=None,
):
    """Rollout boundary, in addition to the caller's source/value/ABI proofs."""
    if (
        type(head_dim) is not int
        or head_dim != 64
        or (v_head_dim is not None and (type(v_head_dim) is not int or v_head_dim != 64))
    ):
        return "head-width"
    if out_dtype not in ("fp16", "f16") or half_accumulate:
        return "precision"
    if causal or any((bias, mask, lse, runtime_scale, grid_3d, runtime_neg_inf)) or mask_batch_div is not None:
        return "source-family"
    if round_q is not None or round_p is not None or scale_chain:
        return "source-rounding"
    if scale not in (None, 0.125):
        return "scale"
    if (
        type(source_kv_block) is not int
        or source_kv_block != 32
        or mask_tail_scores is not False
        or unmasked_kv
        or source_causal_stop not in (None, False)
    ):
        return "source-tail"
    if initial_values is not None and (
        type(initial_values) is not tuple
        or len(initial_values) != 2
        or any(type(v) not in (int, float) or float(v).hex() != "0x0.0p+0" for v in initial_values)
    ):
        return "source-initial-values"
    return None


def row_loop(body, expanded, *, single=False):
    """Render known template bodies, never rewrite arbitrary source text."""
    if "@ROW@" not in body:
        raise ValueError("row body has no placeholder")
    if not expanded:
        prefix = "for (uint rb=0u;rb<4u;rb++)"
        return prefix + (" " + body.replace("@ROW@", "rb") if single else " {" + body.replace("@ROW@", "rb") + "}")
    if single:
        body = body.strip()
    return "\n".join("{\n" + body.replace("@ROW@", str(row) + "u") + "\n}" for row in range(4))
