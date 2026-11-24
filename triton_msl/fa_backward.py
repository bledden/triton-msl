"""FlashAttention with a Metal backward pass — enables TRAINING.

This exposes ``flash_attention`` as a ``torch.autograd.Function``. The forward computes
the output in float32 with explicit torch matmul and softmax operations (``_attention_output``)
and the logsumexp with a separate Metal kernel. The backward dispatches this package's
tiled Metal dK/dV and dQ kernels (FA-2 backward).

The output is deliberately NOT torch's ``scaled_dot_product_attention``: on MPS, torch
2.12.1 caches that operation per shape and causal flag and ignores ``scale`` on later
calls, so a second scale for the same shape (or a scale the caller's own code used first)
returned the first scale's attention, and dQ/dK computed from it.

Head dim is fixed at 64, ``N % 16 == 0`` (the backward tile). The dedicated Metal
``make_fa_logsumexp_kernel`` does not materialize the N*N score matrix. That statement
describes the logsumexp kernel, not the output, whose float32 scores are one ``[ZH, N, N]``
tensor.
See ``make_fa_backward_dkv_kernel`` / ``make_fa_backward_dq_kernel`` for the backward kernels
and ``tests/test_fa_backward.py`` for the autograd cross-check.
"""

import torch

from triton_msl.codegen._msl_templates import (
    make_fa_backward_dkv_kernel,
    make_fa_backward_dq_kernel,
    make_fa_logsumexp_kernel,
)

_RT = None
_LIBS = {}  # bool(causal) -> (dkv_lib, dq_lib)
_LSE_LIBS = {}  # bool(causal) -> logsumexp lib


def _logsumexp(q, k, scale, causal):
    """Flash log-sum-exp on Metal (no N*N materialization). q,k: [ZH,N,64] mps float32 -> [ZH,N]."""
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    global _RT
    if _RT is None:
        _RT = CompileShaderRuntime()
    key = bool(causal)
    if key not in _LSE_LIBS:
        _LSE_LIBS[key] = _RT.get_library(make_fa_logsumexp_kernel(causal))
    ZH, N, _ = q.shape
    L = torch.empty(ZH, N, device=q.device, dtype=torch.float32)
    _RT.dispatch(
        _LSE_LIBS[key],
        "fa_lse",
        [q.contiguous(), k.contiguous(), L, N, scale],
        threads=((N // 16) * 256, ZH, 1),
        group_size=(256, 1, 1),
    )
    return L


def _dispatch_backward(q, k, v, dO, L, D, scale, causal):
    """Run the two Metal backward kernels. All tensors [ZH,N,64]/[ZH,N] contiguous mps float32."""
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    global _RT
    if _RT is None:
        _RT = CompileShaderRuntime()
    key = bool(causal)
    if key not in _LIBS:
        _LIBS[key] = (
            _RT.get_library(make_fa_backward_dkv_kernel(causal)),
            _RT.get_library(make_fa_backward_dq_kernel(causal)),
        )
    dkv_lib, dq_lib = _LIBS[key]
    ZH, N, _ = q.shape
    dK, dV, dQ = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
    grid = ((N // 16) * 256, ZH, 1)
    base = [q, k, v, dO, L, D]
    _RT.dispatch(dkv_lib, "fa_bwd_dkv", base + [dK, dV, N, scale], threads=grid, group_size=(256, 1, 1))
    _RT.dispatch(dq_lib, "fa_bwd_dq", base + [dQ, N, scale], threads=grid, group_size=(256, 1, 1))
    return dQ, dK, dV


def _attention_output(q, k, v, scale, causal):
    """softmax(scale * q @ k^T, causal) @ v in float32. q,k,v: [ZH,N,64] mps float32.

    Not torch's scaled_dot_product_attention: its MPS implementation in torch 2.12.1 keeps
    the first scale it sees for a shape and returns that attention for every later scale.
    Here the scale is an ordinary multiply on this call's scores.
    """
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v)


class _FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        O = _attention_output(q, k, v, scale, causal)
        L = _logsumexp(q, k, scale, causal)  # Metal flash logsumexp — no N*N materialization
        ctx.save_for_backward(q, k, v, O, L)
        ctx.scale, ctx.causal = scale, causal
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        D = (dO * O).sum(-1)
        dQ, dK, dV = _dispatch_backward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dO.contiguous(),
            L.contiguous(),
            D.contiguous(),
            ctx.scale,
            ctx.causal,
        )
        return dQ, dK, dV, None, None


def flash_attention(q, k, v, scale=None, causal=False):
    """FlashAttention whose backward runs on Metal (trainable).

    Args:
        q, k, v: Identically shaped, nonempty ``[..., N, 64]`` strided tensors,
            with the same float32/float16/bfloat16 dtype and MPS device.
            Leading dims fold to ZH; noncontiguous views are supported.
            This fixed self-attention API does not broadcast peers or accept
            different query/key sequence lengths.
            Half inputs compute in fp32 internally and cast back (grads match the input dtype).
        scale: softmax scale (default ``1/sqrt(64)``).
        causal: causal masking.

    Returns an output of the same shape; ``.backward()`` produces dQ/dK/dV via Metal.
    """
    # Establish the fixed backward ABI before a Q-driven cast or reshape can
    # erase a peer mismatch. Eager SDPA's broader broadcasting/cross-attention
    # capabilities do not establish the layout consumed by these Metal kernels.
    for name, value in (("q", q), ("k", k), ("v", v)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"flash_attention requires {name} to be a torch.Tensor")
        if value.ndim < 2:
            raise ValueError(f"flash_attention requires {name} to have rank >= 2")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("flash_attention requires identical q, k and v shapes; peer broadcasting is unsupported")
    if any(size == 0 for size in q.shape):
        raise ValueError("flash_attention requires nonempty q, k and v shapes")
    if q.shape[-1] != 64:
        raise ValueError(f"flash_attention (Metal backward) supports head_dim=64, got {q.shape[-1]}")
    if q.shape[-2] % 16 != 0:
        raise ValueError(f"N must be divisible by 16, got {q.shape[-2]}")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("flash_attention supports only float32, float16 or bfloat16 dtype")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("flash_attention requires identical q, k and v dtypes")
    if q.device.type != "mps" or k.device != q.device or v.device != q.device:
        raise ValueError("flash_attention requires q, k and v on the same MPS device")
    if any(value.layout != torch.strided for value in (q, k, v)):
        raise ValueError("flash_attention requires strided tensor layouts")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    in_dtype = q.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        # fp16/bf16 training: the kernels are fp32, so compute in fp32 internally; autograd
        # flows the grads back through the .float()/.to() casts to the original half tensors.
        q, k, v = q.float(), k.float(), v.float()
    orig = q.shape
    q2, k2, v2 = (t.reshape(-1, orig[-2], orig[-1]) for t in (q, k, v))
    out = _FlashAttentionFn.apply(q2, k2, v2, scale, causal)
    return out.reshape(orig).to(in_dtype)
