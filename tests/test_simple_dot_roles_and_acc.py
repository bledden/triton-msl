"""P0 (packet 065, 2026-09-01): the bare single-dot templates must never write the
wrong buffer nor drop a fused accumulator.

Three cooperating defects on every earlier tree (GPU-verified on clean ba21e9d):

1. Triton fuses ``dot + s`` into the dot's accumulator (``tt.dot %x, %y, splat(s)``),
   so no top-level epilogue op exists for the compute-epilogue guard to see.
2. ``_detect_simple_dot`` asked ``_acc_init_is_bias(acc)`` and proceeded when that was
   False — UNKNOWN counted as "not a bias" — so the inline template emitted ``acc0(0)``
   and silently DROPPED the init. Now: bare templates proceed only on a POSITIVE
   literal-zero proof (``_acc_init_is_literal_zero``); anything else declines to the
   generic path inside the envelope or refuses.
3. Pointer roles were POSITIONAL (``ptr_args[0/1/2]``) in all four dot templates, and
   ``_lower_simple_dot_inline`` also hard-coded ``[[buffer(0/1/2)]]``: with args
   ``(X, Y, P, C)`` the matmul was written INTO P — an unrelated INPUT — and C left
   untouched. Now: roles from dataflow (``_dot_template_ptr_roles``), every kernel
   arg declared at its own ``buffer(i)``; the delegated ``make_matmul_kernel`` route
   refuses unless the roles sit exactly at slots 0/1/2.

Packet 079 (2026-09-02): opcode presence is not value-path proof. The positive op
allowlist admitted ``arith.truncf`` anywhere because the fp16-OUTPUT cast is on the
store path, so the same opcode between a load and the dot was silently dropped (the
template stages the raw fp32 buffer: err vs raw exactly 0). Same class, all
GPU-verified on the 078 tree: a cast round-trip on the store path, a per-iteration
cast on the K-loop accumulator, a non-canonical K-loop load mask, a K-loop pointer
advance that is not BLOCK_K * stride, a scaled induction-variable K index, and K-loop
A/B roles bound by DECLARATION ORDER (loop-carried pointers did not trace: (B, A, C)
-> err 345). Now ``_dot_template_value_paths`` proves every role's complete path
against exactly what the template replays; the K-loop roles trace THROUGH the loop's
iter-args and compute correctly.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    import triton_msl.codegen._lowerer_templates as lowerer_templates
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")

if HAS:

    @triton.jit
    def _leak_065(X, Y, P, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        s = tl.sum(tl.load(P + tl.arange(0, 1024)), axis=0)
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y) + s)

    @triton.jit
    def _four_ptr_bare(X, Y, P, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _out_first(C, X, Y, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _full5(X, Y, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        acc = tl.full((S, S), 5.0, tl.float32)
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y, acc))

    @triton.jit
    def _two_store_p_first(X, Y, P, C, S: tl.constexpr):
        # Packet 067: a store of P BEFORE the dot's store — the first-store
        # convention made P the output (overwritten) and left C NaN.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(P + om[:, None] * S + on[None, :], x)
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _two_store_c_first(X, Y, P, C, S: tl.constexpr):
        # Packet 067: the dot's store first, then a store of P — C came out right
        # but the second store was silently DROPPED.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))
        tl.store(P + om[:, None] * S + on[None, :], x)

    @triton.jit
    def _atomic_add_then_store(X, Y, P, C, S: tl.constexpr):
        # Packet 071: an externally observable write that is NOT a tt.store.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.atomic_add(P, 1.0)
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _atomic_cas_then_store(X, Y, P, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.atomic_cas(P, 0, 1)
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _a_const_residual(X, Y, C, S: tl.constexpr):
        # Packet 073: allowed address ops, but a +1 base offset the template never
        # replays (it loaded raw X: matched X[:-1] @ Y, not X[1:] @ Y).
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + 1 + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _b_runtime_residual(X, Y, C, off, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + off + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _c_residual(X, Y, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + 1 + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _trans_b_positive(X, Y, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        yt = tl.load(Y + on[:, None] * S + ok[None, :])  # identical aranges: this IS row-major Y
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, tl.trans(yt)))

    @triton.jit
    def _kloop_positive(
        C, A, B, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            acc = tl.dot(
                tl.load(A + rm[:, None] * sam + kk[None, :] * sak),
                tl.load(B + kk[:, None] * sbk + rn[None, :] * sbn),
                acc,
            )
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc)

    @triton.jit
    def _canon(X, Y, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    # ---- packet 079: value paths ----
    @triton.jit
    def _ab_cast_079(X, Y, C, S: tl.constexpr):
        # fp32 buffers cast to fp16 BEFORE the dot: intended = X.half() @ Y.half().
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :]).to(tl.float16)
        y = tl.load(Y + ok[:, None] * S + on[None, :]).to(tl.float16)
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _b_only_cast_079(X, Y, C, S: tl.constexpr):
        # X is a native fp16 buffer, Y an fp32 buffer cast at the dot: only B's path
        # carries a transform (role-awareness: A is clean).
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y.to(tl.float16)))

    @triton.jit
    def _c_roundtrip_079(X, Y, C, S: tl.constexpr):
        # fp16 rounding of the RESULT stored into an fp32 buffer: truncf + extf.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y).to(tl.float16))

    @triton.jit
    def _f16_out_079(X, Y, C, S: tl.constexpr):
        # The ONE cast the template replays: f32 accumulator -> fp16 output buffer.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y).to(tl.float16))

    @triton.jit
    def _native_f16_079(X, Y, C, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(C + om[:, None] * S + on[None, :], tl.dot(x, y))

    @triton.jit
    def _kl_079(
        A,
        B,
        C,
        M,
        N,
        K,
        sam,
        sak,
        sbk,
        sbn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        MODE: tl.constexpr,
    ):
        # MODE 0 bare | 1 per-iteration fp16 round of the accumulator | 3 non-canonical
        # A mask (rows >= M-8 zeroed) | 4 half pointer advance on A | 5 canonical masks.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * sam + rk[None, :] * sak
        b_ptrs = B + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            if MODE == 3:
                a = tl.load(a_ptrs, mask=rm[:, None] < M - 8, other=0.0)
            elif MODE == 5:
                a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
            else:
                a = tl.load(a_ptrs)
            if MODE == 5:
                b = tl.load(b_ptrs, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
            else:
                b = tl.load(b_ptrs)
            if MODE == 1:
                acc = tl.dot(a, b, acc).to(tl.float16).to(tl.float32)
            else:
                acc = tl.dot(a, b, acc)
            if MODE == 4:
                a_ptrs += (BK // 2) * sak
            else:
                a_ptrs += BK * sak
            b_ptrs += BK * sbk
        if MODE == 5:
            tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
        else:
            tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc)

    @triton.jit
    def _kl_swapped_079(
        Bp, Ap, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
    ):
        # B's buffer is declared FIRST. Loop-carried A/B pointers did not trace, so the
        # resolver fell back to declaration order: A = Bp (GPU err 345, no exception).
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = Ap + rm[:, None] * sam + rk[None, :] * sak
        b_ptrs = Bp + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc)

    @triton.jit
    def _kl_iv_079(
        A,
        B,
        C,
        M,
        N,
        K,
        sam,
        sak,
        sbk,
        sbn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        # Induction-variable-addressed K index: SCALE=1 is the replayed ``k + range``;
        # SCALE=2 reads every other K tile, which the template silently ignores.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            rk = k * SCALE + tl.arange(0, BK)
            a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak)
            b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn)
            acc = tl.dot(a, b, acc)
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc)

    @triton.jit
    def _out16_079(X, Y, W, Z, sx, sy, sz, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        # Upstream test_dot shape: four pointers, runtime row strides, ``out_dtype=f16``.
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        ok = tl.arange(0, BK)
        x = tl.load(X + om[:, None] * sx + ok[None, :])
        y = tl.load(Y + ok[:, None] * sy + on[None, :])
        tl.store(Z + om[:, None] * sz + on[None, :], tl.dot(x, y, out_dtype=tl.float16))

    @triton.jit
    def _tut_1d_079(
        a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
    ):
        # The Triton-tutorial matmul: 1-D grid split by ``pid % num_pid_m`` /
        # ``pid // num_pid_m``, wrap-around ``% M`` / ``% N`` load indices, an
        # iteration-counted K loop with ``K - k*BK`` masks and loop-carried pointers.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BM)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        offs_am = (pid_m * BM + tl.arange(0, BM)) % M
        offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + (offs_am[:, None] * sam + offs_k[None, :] * sak)
        b_ptrs = b_ptr + (offs_k[:, None] * sbk + offs_bn[None, :] * sbn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
            acc = tl.dot(a, b, acc=acc)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        offs_cm = pid_m * BM + tl.arange(0, BM)
        offs_cn = pid_n * BN + tl.arange(0, BN)
        tl.store(
            c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :],
            acc,
            mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N),
        )

    @triton.jit
    def _i8_dot_079(X, Y, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # int8 x int8 -> i32 (Triton forces the i32 result), stored into an int32 buffer.
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(x, y))

    @triton.jit
    def _i8_dot_narrow_079(X, Y, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # the i32 result narrowed to an int8 buffer (arith.trunci on the store path)
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(x, y).to(tl.int8))

    @triton.jit
    def _i8_dot_plus1_079(X, Y, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # an integer consumer of the dot result (globally allowed integer arith class)
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(x, y) + 1)

    def _direct_lowerer_for(fn, signature, constexprs):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({"num_warps": 4})
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
        context = ir.context()
        ir.load_dialects(context)
        mod = src.make_ir(
            target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
        )
        metadata = {}
        mod = backend.make_ttir(mod, metadata, options)
        mod = backend.make_ttgir(mod, metadata, options)
        return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    for fn in (
        _leak_065,
        _four_ptr_bare,
        _out_first,
        _full5,
        _canon,
        _two_store_p_first,
        _two_store_c_first,
        _atomic_add_then_store,
        _atomic_cas_then_store,
        _a_const_residual,
        _b_runtime_residual,
        _c_residual,
        _trans_b_positive,
        _kloop_positive,
        _ab_cast_079,
        _b_only_cast_079,
        _c_roundtrip_079,
        _f16_out_079,
        _native_f16_079,
        _kl_079,
        _kl_swapped_079,
        _kl_iv_079,
        _out16_079,
        _tut_1d_079,
        _i8_dot_079,
        _i8_dot_narrow_079,
        _i8_dot_plus1_079,
    ):
        if hasattr(fn, "device_caches"):
            fn.device_caches.clear()


@pytest.fixture
def route_spies(monkeypatch):
    inline, generic = [], []
    oi = lowerer_templates._TemplateMixin._lower_simple_dot_inline
    og = generic_lowerer.GenericLowerer._lower_dot
    monkeypatch.setattr(
        generic_lowerer.GenericLowerer,
        "_lower_simple_dot_inline",
        lambda self, info: (inline.append(1), oi(self, info))[1],
    )
    monkeypatch.setattr(
        generic_lowerer.GenericLowerer, "_lower_dot", lambda self, ssa: (generic.append(1), og(self, ssa))[1]
    )
    return inline, generic


def _mk(S=32):
    torch.manual_seed(3)
    x = torch.randn(S, S, device="mps") * 0.1
    y = torch.randn(S, S, device="mps") * 0.1
    p = torch.randn(1024, device="mps")
    c = torch.full((S, S), float("nan"), device="mps")
    return x, y, p, p.clone(), c


@requires_gpu
def test_four_ptr_bare_matmul_writes_C_not_P(cold_gpu_caches, route_spies):
    """Roles + slots half: output LAST with an extra pointer before it."""
    x, y, p, p0, c = _mk()
    _four_ptr_bare[(1,)](x, y, p, c, S=32)
    torch.mps.synchronize()
    assert route_spies[0] and not route_spies[1], "expected the inline template"
    assert not bool(c.isnan().any()), "C left unwritten — output misbound"
    assert torch.equal(p, p0), "P (an input) was overwritten — the P0 corruption"
    torch.testing.assert_close(c, x @ y, rtol=1e-4, atol=1e-4)


@requires_gpu
def test_output_first_bare_matmul(cold_gpu_caches, route_spies):
    x, y, p, p0, c = _mk()
    _out_first[(1,)](c, x, y, S=32)
    torch.mps.synchronize()
    assert route_spies[0]
    assert not bool(c.isnan().any())
    torch.testing.assert_close(c, x @ y, rtol=1e-4, atol=1e-4)


@requires_gpu
def test_nonzero_constant_accumulator_declines_to_generic(cold_gpu_caches, route_spies):
    """NON-REGRESSION CONTROL (passes on base too): ``tl.full(5.0)`` was already a
    recognized bias form under the old ``_acc_init_is_bias`` test, so stage 1a
    already declined it to the generic path. The positive literal-zero proof must
    keep that behavior; the polarity BITE is the fused-reduce kernel below, whose
    accumulator the old test classified UNKNOWN and let through."""
    x, y, p, p0, c = _mk()
    _full5[(1,)](x, y, c, S=32)
    torch.mps.synchronize()
    assert route_spies[1] and not route_spies[0], "bare template took a non-zero accumulator"
    torch.testing.assert_close(c, x @ y + 5.0, rtol=1e-4, atol=1e-4)
    assert (c - x @ y).abs().max().item() > 1.0, "accumulator dropped — the silent-wrong"


@requires_gpu
def test_fused_reduce_accumulator_refuses_loudly(cold_gpu_caches):
    """The 063/065 kernel: acc = splat(tl.sum(P)) is UNKNOWN-not-zero; outside the
    default envelope it must refuse, never write P or leave C NaN silently."""
    x, y, p, p0, c = _mk()
    with pytest.raises(MetalNonRecoverableError):
        _leak_065[(1,)](x, y, p, c, S=32)
        torch.mps.synchronize()
    assert torch.equal(p, p0)


@requires_gpu
def test_canonical_dot_keeps_inline_template(cold_gpu_caches, route_spies):
    """Negative control: the fast route is unchanged for the canonical 3-ptr dot."""
    x, y, p, p0, c = _mk()
    _canon[(1,)](x, y, c, S=32)
    torch.mps.synchronize()
    assert route_spies[0] and not route_spies[1]
    torch.testing.assert_close(c, x @ y, rtol=1e-4, atol=1e-4)


@requires
def test_lowering_boundary_buffer_slots():
    """Rule 9 (bites on base): the 4-ptr bare kernel's MSL must declare EVERY arg at
    its own positional slot — C at buffer(3), P at buffer(2). Base declared only
    three buffers with C at buffer(2), which is the slot the launcher hands P."""
    lw = _direct_lowerer_for(_four_ptr_bare, {"X": "*fp32", "Y": "*fp32", "P": "*fp32", "C": "*fp32"}, {"S": 32})
    msl = lw.lower()
    assert "* _mmarg3 [[buffer(3)]]" in msl, "C must bind at its own arg slot (3), not 2"
    assert "* _mmarg2 [[buffer(2)]]" in msl, "every kernel arg must be declared at its slot"


@requires
def test_lowering_boundary_fused_reduce_acc_not_literal_zero():
    """Rule 9: the fused-reduce accumulator is NOT a literal zero, and lowering
    refuses instead of emitting the bare template (skips on trees without the
    positive-proof helper — the decline cannot exist there either)."""
    lw = _direct_lowerer_for(_leak_065, {"X": "*fp32", "Y": "*fp32", "P": "*fp32", "C": "*fp32"}, {"S": 32})
    probe = getattr(lw, "_acc_init_is_literal_zero", None)
    if probe is None:
        pytest.skip("pre-P0 tree: helper absent")
    dot = next(o for o in lw.graph.ops if o.op == "tt.dot")
    by_id = {o.id: o for o in lw.graph.ops}
    assert probe(dot.operand_ids[2], by_id) is False
    with pytest.raises(MetalNonRecoverableError):
        lw.lower()


@requires_gpu
@pytest.mark.parametrize("fn", [_two_store_p_first, _two_store_c_first], ids=["P-then-C", "C-then-P"])
def test_two_stores_refuse_on_bare_template(cold_gpu_caches, fn):
    """Packet 067: one dot + two stores must REFUSE (the template emits one store).
    On base: P-then-C overwrote P and left C NaN; C-then-P silently dropped the P
    store. Both orders now refuse before emission, with every buffer untouched."""
    x, y, p, p0, c = _mk()
    p2 = torch.full((32, 32), float("nan"), device="mps")
    with pytest.raises(MetalNonRecoverableError, match="exactly one tt.dot and exactly one tt.store"):
        fn[(1,)](x, y, p2, c, S=32)
        torch.mps.synchronize()
    assert bool(c.isnan().all()) and bool(p2.isnan().all()), "a refused kernel must touch nothing"


@requires
@pytest.mark.parametrize("fn", [_two_store_p_first, _two_store_c_first], ids=["P-then-C", "C-then-P"])
def test_lowering_boundary_two_stores_refuse(fn):
    """Rule 9: fresh TTGIR, both store orders refuse at the template role gate."""
    lw = _direct_lowerer_for(fn, {"X": "*fp32", "Y": "*fp32", "P": "*fp32", "C": "*fp32"}, {"S": 32})
    with pytest.raises(MetalNonRecoverableError, match="exactly one tt.dot and exactly one tt.store"):
        lw.lower()


@requires_gpu
@pytest.mark.parametrize(
    "fn,pdt",
    [(_atomic_add_then_store, torch.float32), (_atomic_cas_then_store, torch.int32)],
    ids=["atomic_rmw", "atomic_cas"],
)
def test_atomic_side_effect_refuses_on_bare_template(cold_gpu_caches, fn, pdt):
    """Packet 071: the bare template re-emits only its replayed vocabulary; an atomic
    (tt.atomic_rmw / tt.atomic_cas) is an externally observable effect it would
    silently DROP (base: C exact, P stayed 0). Both families must refuse, touching
    nothing."""
    x, y, _p, _p0, c = _mk()
    p = torch.zeros(1, device="mps", dtype=pdt)
    with pytest.raises(MetalNonRecoverableError, match="does not replay"):
        fn[(1,)](x, y, p, c, S=32)
        torch.mps.synchronize()
    assert p.item() == 0 and bool(c.isnan().all()), "a refused kernel must touch nothing"


@requires
@pytest.mark.parametrize(
    "fn,pty", [(_atomic_add_then_store, "*fp32"), (_atomic_cas_then_store, "*i32")], ids=["atomic_rmw", "atomic_cas"]
)
def test_lowering_boundary_atomic_refuses(fn, pty):
    """Rule 9: fresh TTGIR, both atomic families refuse at the template gate."""
    lw = _direct_lowerer_for(fn, {"X": "*fp32", "Y": "*fp32", "P": pty, "C": "*fp32"}, {"S": 32})
    with pytest.raises(MetalNonRecoverableError, match="does not replay"):
        lw.lower()


@requires_gpu
@pytest.mark.parametrize("which", ["A-const", "B-runtime", "C-const"])
def test_residual_base_offset_refuses(cold_gpu_caches, which):
    """Packet 073: an address term the template does NOT replay (a constant or
    runtime-scalar base offset on A, B or C) must REFUSE — the template addresses
    from strides alone and would read/write the wrong slice. On base: A-const
    computed X[:-1] @ Y (the raw buffer) instead of X[1:] @ Y, no exception."""
    torch.manual_seed(19)
    S = 32
    xb = torch.randn(S * S + 1, device="mps") * 0.1
    yb = torch.randn(S * S + 1, device="mps") * 0.1
    cb = torch.full((S * S + 1,), float("nan"), device="mps")
    with pytest.raises(MetalNonRecoverableError, match="stride could not be inferred"):
        if which == "A-const":
            _a_const_residual[(1,)](xb, yb, cb, S=S)
        elif which == "B-runtime":
            _b_runtime_residual[(1,)](xb, yb, cb, 1, S=S)
        else:
            _c_residual[(1,)](xb, yb, cb, S=S)
        torch.mps.synchronize()
    assert bool(cb.isnan().all()), "a refused kernel must touch nothing"


@requires
@pytest.mark.parametrize("which", ["A-const", "B-runtime", "C-const"])
def test_lowering_boundary_residual_refuses(which):
    """Rule 9: fresh TTGIR, residual base offsets refuse at the stride tracer."""
    if which == "A-const":
        lw = _direct_lowerer_for(_a_const_residual, {"X": "*fp32", "Y": "*fp32", "C": "*fp32"}, {"S": 32})
    elif which == "B-runtime":
        lw = _direct_lowerer_for(
            _b_runtime_residual, {"X": "*fp32", "Y": "*fp32", "C": "*fp32", "off": "i32"}, {"S": 32}
        )
    else:
        lw = _direct_lowerer_for(_c_residual, {"X": "*fp32", "Y": "*fp32", "C": "*fp32"}, {"S": 32})
    with pytest.raises(MetalNonRecoverableError, match="stride could not be inferred"):
        lw.lower()


@requires_gpu
def test_transposed_operand_positive_keeps_template(cold_gpu_caches, route_spies):
    """Positive control: tl.trans on B lowers to memdesc_trans, which the template
    replays (1B pin) — must stay on the inline route and be exact vs x @ Yᵀ."""
    x, y, p, p0, c = _mk()
    _trans_b_positive[(1,)](x, y, c, S=32)
    torch.mps.synchronize()
    assert route_spies[0]
    torch.testing.assert_close(c, x @ y.T, rtol=1e-4, atol=1e-4)


@requires_gpu
def test_kloop_pid_tiles_positive(cold_gpu_caches):
    """Positive control: K-loop matmul with pid*BLOCK tile terms and output FIRST
    — those tile terms are intentionally replayed and must not be refused."""
    torch.manual_seed(19)
    M = N = K = 64
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    c = torch.full((M, N), float("nan"), device="mps")
    _kloop_positive[(2, 2)](
        c,
        a,
        b,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=32,
        BN=32,
        BK=16,
    )
    torch.mps.synchronize()
    assert not bool(c.isnan().any())
    torch.testing.assert_close(c, a @ b, rtol=1e-3, atol=1e-2)


# ---------------------------------------------------------------------------
# Packet 079: per-role value-path proofs
# ---------------------------------------------------------------------------

_KL_SIG = {
    "A": "*fp32",
    "B": "*fp32",
    "C": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "sam": "i32",
    "sak": "i32",
    "sbk": "i32",
    "sbn": "i32",
    "scm": "i32",
    "scn": "i32",
}


def _witness(launch, c, raw, intended):
    """Correct-or-refuse for a GPU witness: a refusal must touch nothing; a compute
    must match the INTENDED semantics, never the raw-buffer result the template used
    to stage. The witness data separate the two by >= 10x the accepted error."""
    sep = (raw - intended).abs().max().item()
    assert sep > 0, "witness data do not separate raw from intended"
    try:
        launch()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert bool(c.isnan().all()), "a refused kernel must touch nothing"
        return "refused"
    err = (c.float() - intended).abs().max().item()
    assert err == err and err <= sep / 10, (
        f"computed the RAW-buffer result, not the intended one (err vs intended {err}, raw/intended separation {sep})"
    )
    return "computed"


def _cast_data(scale=30.0):
    torch.manual_seed(23)
    x = torch.randn(32, 32, device="mps") * scale + 0.1234567
    y = torch.randn(32, 32, device="mps") * scale + 0.7654321
    c = torch.full((32, 32), float("nan"), device="mps")
    return x, y, c


@requires_gpu
@pytest.mark.parametrize("which", ["A+B", "B-only"])
def test_operand_cast_not_dropped(cold_gpu_caches, which):
    """Packet 079 (bites on base: err vs raw exactly 0, vs intended ~0.065 at scale 3):
    an fp32 buffer cast to fp16 before the dot must not be staged raw."""
    x, y, c = _cast_data()
    if which == "A+B":
        intended = x.half().float() @ y.half().float()
        _witness(lambda: _ab_cast_079[(1,)](x, y, c, S=32), c, x @ y, intended)
    else:
        xh = x.half()
        intended = xh.float() @ y.half().float()
        _witness(lambda: _b_only_cast_079[(1,)](xh, y, c, S=32), c, xh.float() @ y, intended)


@requires_gpu
def test_store_roundtrip_not_dropped(cold_gpu_caches):
    """Packet 079: ``dot.to(f16)`` stored into an fp32 buffer is an fp16 rounding of the
    result (truncf + extf); the template stored the raw f32 accumulator (err vs raw 0)."""
    x, y, c = _cast_data()
    _witness(lambda: _c_roundtrip_079[(1,)](x, y, c, S=32), c, x @ y, (x @ y).half().float())


def _kl_data(M=64, N=64, K=64, scale=3.0, seed=29):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="mps") * scale
    b = torch.randn(K, N, device="mps") * scale
    c = torch.full((M, N), float("nan"), device="mps")
    return a, b, c


def _kl_launch(fn, grid, a, b, c, M, N, K, **kw):
    return lambda: fn[grid](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), **kw
    )


@requires_gpu
def test_kloop_body_cast_not_dropped(cold_gpu_caches):
    """Packet 079: rounding the accumulator to fp16 every iteration (truncf + extf
    between the dot and the yield) — the float accumulator dropped it (err 0.11)."""
    a, b, c = _kl_data()
    acc = torch.zeros(64, 64, device="mps")
    for k0 in range(0, 64, 32):
        acc = (acc + a[:, k0 : k0 + 32] @ b[k0 : k0 + 32, :]).half().float()
    _witness(_kl_launch(_kl_079, (2, 2), a, b, c, 64, 64, 64, BM=32, BN=32, BK=32, MODE=1), c, a @ b, acc)


@requires_gpu
def test_kloop_noncanonical_mask_not_dropped(cold_gpu_caches):
    """Packet 079: a load mask that is NOT the template's boundary clip (rows >= M-8
    zeroed) — the K-loop template applies only its own bounds (err 26)."""
    a, b, c = _kl_data()
    am = a.clone()
    am[64 - 8 :] = 0
    _witness(_kl_launch(_kl_079, (2, 2), a, b, c, 64, 64, 64, BM=32, BN=32, BK=32, MODE=3), c, a @ b, am @ b)


@requires_gpu
def test_kloop_pointer_advance_not_dropped(cold_gpu_caches):
    """Packet 079: A advanced by BLOCK_K/2 per iteration — the template advances by
    exactly BLOCK_K * stride and computed plain A @ B (err 43)."""
    a, b, c = _kl_data(M=32, N=32, K=128)
    intended = sum(a[:, 16 * i : 16 * i + 32] @ b[32 * i : 32 * i + 32, :] for i in range(4))
    _witness(_kl_launch(_kl_079, (1, 1), a, b, c, 32, 32, 128, BM=32, BN=32, BK=32, MODE=4), c, a @ b, intended)


@requires_gpu
def test_kloop_scaled_iv_index_not_dropped(cold_gpu_caches):
    """Packet 079: ``rk = 2*k + range`` reads every other K tile; the template's
    ``k + range`` addressing computed the contiguous product instead."""
    a, b, c = _kl_data(M=32, N=32, K=128)
    intended = a[:, 0:32] @ b[0:32] + a[:, 64:96] @ b[64:96]
    launch = lambda: _kl_iv_079[(1, 1)](
        a,
        b,
        c,
        32,
        32,
        64,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=32,
        BN=32,
        BK=32,
        SCALE=2,
    )
    _witness(launch, c, a[:, :64] @ b[:64], intended)


@requires_gpu
def test_kloop_swapped_declaration_computes(cold_gpu_caches):
    """Packet 079 (bites on base: err 345, no exception): with B's buffer declared
    first, the loop-carried A/B streams must trace THROUGH the iter-args to their
    args. Now computes exactly — the roles are dataflow, not position."""
    a, b, c = _kl_data()
    _kl_swapped_079[(2, 2)](
        b,
        a,
        c,
        64,
        64,
        64,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=32,
        BN=32,
        BK=32,
    )
    torch.mps.synchronize()
    assert not bool(c.isnan().any())
    torch.testing.assert_close(c, a @ b, rtol=1e-4, atol=1e-3)


@requires_gpu
def test_native_fp16_keeps_template(cold_gpu_caches, route_spies):
    """Positive: native fp16 buffers (no cast on any path) stay on the template."""
    x, y, c = _cast_data(scale=1.0)
    xh, yh = x.half(), y.half()
    _native_f16_079[(1,)](xh, yh, c, S=32)
    torch.mps.synchronize()
    assert route_spies[0]
    torch.testing.assert_close(c, xh.float() @ yh.float(), rtol=1e-4, atol=1e-3)


@requires_gpu
def test_fp16_output_cast_keeps_template(cold_gpu_caches, route_spies):
    """Positive: the ONE cast the template replays (f32 accumulator -> fp16 C)."""
    x, y, _ = _cast_data(scale=1.0)
    ch = torch.full((32, 32), float("nan"), device="mps", dtype=torch.float16)
    _f16_out_079[(1,)](x, y, ch, S=32)
    torch.mps.synchronize()
    assert route_spies[0]
    torch.testing.assert_close(ch.float(), (x @ y).half().float(), rtol=2e-3, atol=2e-2)


@requires_gpu
@pytest.mark.parametrize("shape", [(64, 64, 64), (50, 40, 44)], ids=["aligned", "ragged"])
def test_kloop_canonical_masks_positive(cold_gpu_caches, shape):
    """Positive: the canonical ``(rm < M) & (rk < K - k)`` / ``(rk < K - k) & (rn < N)``
    load masks and ``(rm < M) & (rn < N)`` store mask ARE the template's clip."""
    M, N, K = shape
    a, b, c = _kl_data(M=M, N=N, K=K)
    _kl_launch(_kl_079, ((M + 31) // 32, (N + 31) // 32), a, b, c, M, N, K, BM=32, BN=32, BK=32, MODE=5)()
    torch.mps.synchronize()
    assert not bool(c.isnan().any())
    torch.testing.assert_close(c, a @ b, rtol=1e-4, atol=1e-3)


@requires_gpu
def test_kloop_iv_addressed_positive(cold_gpu_caches):
    """Positive: K indexed as ``k + range`` (no loop-carried pointers) is replayed."""
    a, b, c = _kl_data()
    _kl_iv_079[(2, 2)](
        a,
        b,
        c,
        64,
        64,
        64,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=32,
        BN=32,
        BK=32,
        SCALE=1,
    )
    torch.mps.synchronize()
    assert not bool(c.isnan().any())
    torch.testing.assert_close(c, a @ b, rtol=1e-4, atol=1e-3)


_BOUNDARY_079 = [
    (
        "A+B-cast",
        lambda: _direct_lowerer_for(_ab_cast_079, {"X": "*fp32", "Y": "*fp32", "C": "*fp32"}, {"S": 32}),
        "dot operand A",
    ),
    (
        "B-only-cast",
        lambda: _direct_lowerer_for(_b_only_cast_079, {"X": "*fp16", "Y": "*fp32", "C": "*fp32"}, {"S": 32}),
        "dot operand B",
    ),
    (
        "C-roundtrip",
        lambda: _direct_lowerer_for(_c_roundtrip_079, {"X": "*fp32", "Y": "*fp32", "C": "*fp32"}, {"S": 32}),
        "round-trip on the store path",
    ),
    (
        "kloop-body-cast",
        lambda: _direct_lowerer_for(_kl_079, _KL_SIG, {"BM": 32, "BN": 32, "BK": 32, "MODE": 1}),
        "accumulator",
    ),
    (
        "kloop-noncanonical-mask",
        lambda: _direct_lowerer_for(_kl_079, _KL_SIG, {"BM": 32, "BN": 32, "BK": 32, "MODE": 3}),
        "load mask",
    ),
    (
        "kloop-half-advance",
        lambda: _direct_lowerer_for(_kl_079, _KL_SIG, {"BM": 32, "BN": 32, "BK": 32, "MODE": 4}),
        "pointer advance",
    ),
    (
        "kloop-scaled-iv",
        lambda: _direct_lowerer_for(_kl_iv_079, _KL_SIG, {"BM": 32, "BN": 32, "BK": 32, "SCALE": 2}),
        "scaled index",
    ),
]


@requires
@pytest.mark.parametrize("name,mk,expect", _BOUNDARY_079, ids=[t[0] for t in _BOUNDARY_079])
def test_lowering_boundary_value_paths_refuse(name, mk, expect):
    """Rule 9: fresh TTGIR, every 079 form refuses at the lowering boundary with a
    ROLE-naming reason (an opcode allowlist could only name the opcode)."""
    lw = mk()
    with pytest.raises(MetalNonRecoverableError, match=expect):
        lw.lower()


@requires
def test_lowering_boundary_kloop_roles_through_iter_args():
    """Rule 9 (bites on base): with B's buffer declared first, the emitted K-loop
    MSL must stage A from ``Ap`` — base staged it from ``Bp`` (declaration order)."""
    sig = {
        "Bp": "*fp32",
        "Ap": "*fp32",
        "C": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "sam": "i32",
        "sak": "i32",
        "sbk": "i32",
        "sbn": "i32",
        "scm": "i32",
        "scn": "i32",
    }
    lw = _direct_lowerer_for(_kl_swapped_079, sig, {"BM": 32, "BN": 32, "BK": 32})
    msl = lw.lower()
    # Unspecialized runtime strides route this harness to the strided SCALAR
    # template (``_sum += A[m*sam + k*sak] * B[k*sbk + n*sbn]``); a GPU launch
    # specializes the unit strides and takes the simdgroup one (``tg_A[i] = ...``).
    # Either way the A operand must be read through ``Ap`` and B through ``Bp``. The
    # template names arguments positionally: Bp is _mmarg0, Ap _mmarg1, sam _mmarg6,
    # sbk _mmarg8.
    lines = msl.splitlines()
    scalar = [i for i, ln in enumerate(lines) if "_sum +=" in ln]
    if scalar:
        read = lines[scalar[0]] + lines[scalar[0] + 1]  # the product spans two lines
        assert read.index("_mmarg1[") < read.index("_mmarg0["), f"A must be read from Ap and B from Bp, got {read!r}"
        assert "(uint)m * (uint)(_mmarg6)" in read and "(uint)k * (uint)(_mmarg8)" in read
    else:
        a_stage = [ln for ln in lines if "tg_A[i] =" in ln]
        b_stage = [ln for ln in lines if "tg_B[i] =" in ln]
        assert a_stage and all("_mmarg1[" in ln for ln in a_stage), f"A must be staged from Ap, got {a_stage}"
        assert b_stage and all("_mmarg0[" in ln for ln in b_stage), f"B must be staged from Bp, got {b_stage}"


@requires
@pytest.mark.parametrize("mode", [0, 5], ids=["bare", "canonical-masks"])
def test_lowering_boundary_kloop_positive_forms_lower(mode):
    """Positive control at the boundary: the bare and canonically-masked K-loops lower."""
    lw = _direct_lowerer_for(_kl_079, _KL_SIG, {"BM": 32, "BN": 32, "BK": 32, "MODE": mode})
    assert "kernel void" in lw.lower()


@requires_gpu
def test_out_dtype_f16_dot_to_f16_buffer_keeps_template(cold_gpu_caches, route_spies):
    """Positive: an ``out_dtype=float16`` dot (upstream test_dot rows) whose SOLE consumer is
    the store to an f16 buffer stays on the template, whose one final f32 -> f16 rounding is
    the only observable event: the result equals the half-rounded product."""
    torch.manual_seed(7)
    S = 32
    x = (torch.randn(S, S, device="mps") * 0.5).half()
    y = (torch.randn(S, S, device="mps") * 0.5).half()
    w = torch.zeros(S, S, device="mps", dtype=torch.float16)
    z = torch.full((S, S), float("nan"), device="mps", dtype=torch.float16)
    _out16_079[(1,)](x, y, w, z, S, S, S, BM=S, BK=S, BN=S)
    torch.mps.synchronize()
    assert route_spies[0]
    assert not bool(z.isnan().any())
    torch.testing.assert_close(z.float(), (x.float() @ y.float()).half().float(), rtol=1e-3, atol=1e-2)


@requires_gpu
def test_out_dtype_f16_dot_extended_to_f32_not_exposed(cold_gpu_caches):
    """Packet 083 (bites on ba21e9d AND on the frozen 080 tree, err 0.06 vs the IR oracle,
    0 vs the raw f32 product): ``tl.dot(x, y, out_dtype=f16)`` stored into an f32 buffer
    is ``extf(f16 result)`` — the stored values must be f16-representable. The template
    exposed its f32 accumulator instead; 080 called that a "precision refinement". It is
    not: a declared result type makes the rounding observable. The reference follows the
    IR's operations and dtypes in order."""
    torch.manual_seed(7)
    S = 32
    x = (torch.randn(S, S, device="mps") * 30).half()
    y = (torch.randn(S, S, device="mps") * 30).half()
    w = torch.zeros(S, S, device="mps", dtype=torch.float16)
    z = torch.full((S, S), float("nan"), device="mps", dtype=torch.float32)
    raw = x.float() @ y.float()
    intended = raw.half().float()
    _witness(lambda: _out16_079[(1,)](x, y, w, z, S, S, S, BM=S, BK=S, BN=S), z, raw, intended)


@requires
def test_lowering_boundary_f16_result_extended_refuses():
    """Rule 9: fresh TTGIR, the f16-result -> extf -> f32-store path refuses at the boundary."""
    sig = {"X": "*fp16", "Y": "*fp16", "W": "*fp16", "Z": "*fp32", "sx": "i32", "sy": "i32", "sz": "i32"}
    lw = _direct_lowerer_for(_out16_079, sig, {"BM": 32, "BK": 32, "BN": 32})
    with pytest.raises(MetalNonRecoverableError, match="f16 result extended"):
        lw.lower()


@requires_gpu
@pytest.mark.parametrize(
    "shape", [(96, 80, 64), (64, 64, 64), (128, 128, 128)], ids=["9-tiles-unaligned", "single-tile", "4-tiles-aligned"]
)
def test_tutorial_1d_grid_mapping_replayed(cold_gpu_caches, shape):
    """Packet 080 W8 (bites on base for the 9-tile shape: 60% of C never written): the
    tutorial matmul's 1-D grid split ``pid % num_pid_m`` / ``pid // num_pid_m`` was
    dispatched by the K-loop template as if 2-D (``pid3.x`` / ``pid3.y``), so every
    tile past the first column was skipped whenever the runtime fast path did not
    take over (unaligned shapes). The mapping is now proven (``num_pid_m ==
    cdiv(M, BLOCK_M)``) and replayed verbatim; the aligned and single-tile shapes are
    positives that were already right."""
    M, N, K = shape
    torch.manual_seed(3)
    a = torch.randn(M, K, device="mps", dtype=torch.float16)
    b = torch.randn(K, N, device="mps", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="mps")
    BM = BN = 32 if M % 64 else 64
    BK = 32
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _tut_1d_079[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=BM,
        BN=BN,
        BK=BK,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert not bool(c.isnan().any()), f"{c.isnan().float().mean().item():.0%} of C never written — tiles dropped"
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-3, atol=1e-2)


@requires
def test_lowering_boundary_1d_grid_mapping_emitted():
    """Rule 9 (bites on base): the emitted K-loop MSL for the tutorial form must split
    ``pid3.x`` by ``cdiv(_M, BLOCK_M)`` exactly as the IR does, never read ``pid3.y``."""
    sig = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "sam": "i32",
        "sak": "i32",
        "sbk": "i32",
        "sbn": "i32",
        "scm": "i32",
        "scn": "i32",
    }
    lw = _direct_lowerer_for(_tut_1d_079, sig, {"BM": 32, "BN": 32, "BK": 32})
    msl = lw.lower()
    assert "pid3.x % _npm" in msl and "pid3.x / _npm" in msl and "((uint)_M + 32u - 1u) / 32u" in msl, (
        "1-D grid split not replayed"
    )
    assert "pid3.y" not in msl, "the 2-D mapping leaked into a 1-D-grid kernel"


_I8_SIG = {"X": "*i8", "Y": "*i8", "C": "*i32"}


@requires_gpu
def test_int8_dot_worst_product_exact(cold_gpu_caches, route_spies):
    """Packet 087: the int8 -> i32 admission is an EXACTNESS proof, pinned at its worst
    product: every operand -128, K=128, so each output is exactly 128 * 16384 = 2,097,152
    (float accumulation of int8 products stays exact below 2^24 for K <= 1023). Bit
    equality, no tolerance."""
    M = N = 16
    K = 128
    x = torch.full((M, K), -128, device="mps", dtype=torch.int8)
    y = torch.full((K, N), -128, device="mps", dtype=torch.int8)
    c = torch.full((M, N), 7, device="mps", dtype=torch.int32)
    _i8_dot_079[(1,)](x, y, c, M=M, N=N, K=K)
    torch.mps.synchronize()
    assert route_spies[0], "expected the template route"
    assert torch.equal(c.cpu(), torch.full((M, N), 2_097_152, dtype=torch.int32))


@requires
def test_lowering_boundary_int8_k_cutoff_refuses():
    """Rule 9 (bites on base): K=1024 is the first width where int8 partial sums can reach
    2^24 in float; the single-tile admission stops at K <= 1023."""
    lw = _direct_lowerer_for(_i8_dot_079, _I8_SIG, {"M": 16, "N": 16, "K": 1024})
    with pytest.raises(MetalNonRecoverableError, match="K <= 1023"):
        lw.lower()


@requires
def test_lowering_boundary_int8_k1023_max_lowers():
    """Positive control at the boundary: K=512 (the largest power-of-two tile below the
    cutoff) lowers."""
    lw = _direct_lowerer_for(_i8_dot_079, _I8_SIG, {"M": 16, "N": 16, "K": 512})
    assert "kernel void" in lw.lower()


@requires
@pytest.mark.parametrize(
    "fn,sig,expect",
    [
        (_i8_dot_narrow_079, {"X": "*i8", "Y": "*i8", "C": "*i8"}, "trailing compute epilogue|i32 output|trunci"),
        (_i8_dot_plus1_079, _I8_SIG, "non-zero accumulator init|observed by|addi"),
    ],
    ids=["narrowed-i8-C", "integer-consumer"],
)
def test_lowering_boundary_int8_non_replayed_forms_refuse(fn, sig, expect):
    """Current-correctness pins (087 holdouts 3/4): an i32 result narrowed to an int8 buffer
    (modular in Triton, undefined for MSL float->char) and an integer consumer of the
    result both refuse. Each is caught by an EARLIER gate on this tree (the ``trunci`` is
    a compute epilogue; Triton fuses ``+ 1`` into the dot accumulator, which the
    literal-zero proof rejects); the value-path reasons are the defense in depth."""
    lw = _direct_lowerer_for(fn, sig, {"M": 16, "N": 16, "K": 128})
    with pytest.raises(MetalNonRecoverableError, match=expect):
        lw.lower()
