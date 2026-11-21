#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// simdgroup-MMA FlashAttention-2 (half in/out, fp32 compute, non-causal).
// Loop-split: branchless full-block fast loop + separate staged/masked tail block.
kernel void _flash_attn_fwd(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* Out [[buffer(3)]],
    constant uint& fa_arg4 [[buffer(4)]],
    constant uint& fa_arg5 [[buffer(5)]],
    constant uint& fa_arg6 [[buffer(6)]],
    constant uint& fa_arg7 [[buffer(7)]],
    constant uint& fa_arg8 [[buffer(8)]],
    constant uint& fa_arg9 [[buffer(9)]],
    constant uint& fa_arg10 [[buffer(10)]],
    constant uint& fa_arg11 [[buffer(11)]],
    constant uint& fa_arg12 [[buffer(12)]],
    constant uint& fa_arg13 [[buffer(13)]],
    constant uint& fa_arg14 [[buffer(14)]],
    constant uint& fa_arg15 [[buffer(15)]],
    constant uint& fa_arg16 [[buffer(16)]],
    constant uint& fa_arg17 [[buffer(17)]],
    uint3 pid3 [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint BM = 32u, BN = 64u, D = 64u, NT = 256u, TPG = 1u;
    const float scale = 0.125f;
    const uint q_sz = fa_arg4;
    const uint q_sh = fa_arg5;
    const uint q_sm = fa_arg6;
    const uint q_sk = 1u;
    const uint k_sz = fa_arg7;
    const uint k_sh = fa_arg8;
    const uint k_sn = fa_arg9;
    const uint k_sk = 1u;
    const uint v_sz = fa_arg10;
    const uint v_sh = fa_arg11;
    const uint v_sn = fa_arg12;
    const uint v_sk = 1u;
    const uint o_sz = fa_arg13;
    const uint o_sh = fa_arg14;
    const uint o_sm = fa_arg15;
    const uint o_sk = 1u;
    const uint Z = 1u;
    const uint H = fa_arg16;
    const uint N_CTX = fa_arg17;
    uint q_block = pid3.x, zh = pid3.y;
    uint z = zh / H, h = zh % H;
    uint q_start = q_block * BM;
    uint q_base = z*q_sz+h*q_sh, k_base = z*k_sz+h*k_sh, v_base = z*v_sz+h*v_sh, o_base = z*o_sz+h*o_sh;


    threadgroup half tgQ[32 * 64];
    threadgroup float  tg_S[32 * 64];
    threadgroup half tgP[32 * 64];
    threadgroup float on_scratch[8 * 64];
    threadgroup float  tg_m[32], tg_l[32], tg_alpha[32];
    threadgroup float  tg_pmax[32 * 8], tg_psum[32 * 8];  // parallel-softmax partials
    threadgroup half tgKV[64 * 8u];

    simdgroup_float8x8 o[4][TPG];
    {
for (uint t=0u;t<TPG;t++) o[0u][t]=simdgroup_float8x8(0.0f);
}
{
for (uint t=0u;t<TPG;t++) o[1u][t]=simdgroup_float8x8(0.0f);
}
{
for (uint t=0u;t<TPG;t++) o[2u][t]=simdgroup_float8x8(0.0f);
}
{
for (uint t=0u;t<TPG;t++) o[3u][t]=simdgroup_float8x8(0.0f);
}

    for (uint i = lid; i < BM*D; i += NT) {
        uint qr = q_start + i/D;
        tgQ[i] = (qr < N_CTX) ? Q[q_base + qr*q_sm + (i%D)*q_sk] : half(0);
    }
    if (lid < BM) { tg_m[lid]=-INFINITY; tg_l[lid]=0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_full = N_CTX / BN;            // floor: count of FULL kv-blocks
    for (uint kv_block = 0u; kv_block < n_full; kv_block++) {
        uint kv_start = kv_block * BN;
    {
        simdgroup_float8x8 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        simdgroup_half8x8 qf, kf;
        for (uint kc = 0u; kc < D; kc += 8u) {
            simdgroup_load(kf, K + k_base + (kv_start + sgitg*8u)*k_sn + kc*k_sk, k_sn, 0, true);
            simdgroup_load(qf, tgQ + 0u*D + kc, D);  simdgroup_multiply_accumulate(s0, qf, kf, s0);
            simdgroup_load(qf, tgQ + 8u*D + kc, D);  simdgroup_multiply_accumulate(s1, qf, kf, s1);
            simdgroup_load(qf, tgQ + 16u*D + kc, D); simdgroup_multiply_accumulate(s2, qf, kf, s2);
            simdgroup_load(qf, tgQ + 24u*D + kc, D); simdgroup_multiply_accumulate(s3, qf, kf, s3);
        }
        simdgroup_store(s0, tg_S + 0u*BN + sgitg*8u, BN);
        simdgroup_store(s1, tg_S + 8u*BN + sgitg*8u, BN);
        simdgroup_store(s2, tg_S + 16u*BN + sgitg*8u, BN);
        simdgroup_store(s3, tg_S + 24u*BN + sgitg*8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Parallel online-softmax: all 256 threads (8 per row, row=lid/8) scan
            // BN/8 cols each and reduce partial max/sum via tg_pmax/tg_psum scratch —
            // vs the old 32-thread serial BN-col scan (was ~14% of block time).
            // Masked cols get -INFINITY in the max pass, so exp() = 0 in the sum pass:
            // the causal/tail GUARD is preserved without a second guard.
            uint pr = lid / 8u, pc = lid % 8u; 
            uint cA = pc * (BN / 8u), cB = cA + (BN / 8u);
            float pm = -INFINITY;
            for (uint cj = cA; cj < cB; cj++) {
                
                float s = true ? (tg_S[pr*BN+cj]*scale) : -INFINITY;
                tg_S[pr*BN+cj] = s; pm = max(pm, s);
            }
            tg_pmax[pr*8u + pc] = pm;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < BM) {
                uint r = lid; float m_prev = tg_m[r], l_prev = tg_l[r], m_new = m_prev;
                bool m_ended = false;
                for (uint j = 0u; j < 8u; j++) {
                    m_new = max(m_new, tg_pmax[r*8u+j]);
                    if ((j % 4u) == 3u && m_new == -INFINITY) m_ended = true;
                }
                if (m_ended) m_new = NAN;
                tg_m[r] = m_new; tg_alpha[r] = exp(m_prev - m_new); tg_l[r] = l_prev * tg_alpha[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float ps = 0.0f; float mr = tg_m[pr];
            for (uint cj = cA; cj < cB; cj++) {
                float p = exp(tg_S[pr*BN+cj] - mr); tgP[pr*BN+cj] = half(p); ps += p;
            }
            tg_psum[pr*8u + pc] = ps;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < BM) {
                uint r = lid; float l = tg_l[r];
                for (uint j = 0u; j < 8u; j++) l += tg_psum[r*8u+j];
                tg_l[r] = l;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Keep a nonfinite source row local across the 8x8 MMAs.
            if (!isfinite(tg_l[pr])) {
                for (uint cj = cA; cj < cB; cj++) { tgP[pr*BN+cj] = half(0.0f); }
            }        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // CPU prototype: broadcast each row's alpha into an identically typed fragment.
        // Each SIMD group owns 256 slots; no cross-group scratch reader exists here.
        {

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[0u*8u+e/8u];
                tg_S[sgitg*256u+0u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
{

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[1u*8u+e/8u];
                tg_S[sgitg*256u+1u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
{

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[2u*8u+e/8u];
                tg_S[sgitg*256u+2u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
{

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[3u*8u+e/8u];
                tg_S[sgitg*256u+3u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 pf, vf, vfs[64/8];
        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*8u;
            // Same-type storage multiplication: no diagonal MMA and no lane-row formula.
            {

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 0u*64u, 8u);
                o[0u][t].thread_elements() = o[0u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
{

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 1u*64u, 8u);
                o[1u][t].thread_elements() = o[1u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
{

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 2u*64u, 8u);
                o[2u][t].thread_elements() = o[2u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
{

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 3u*64u, 8u);
                o[3u][t].thread_elements() = o[3u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
            for (uint kk=0u;kk<BN;kk+=8u)
                simdgroup_load(vfs[kk/8u], V + v_base + (kv_start + kk)*v_sn + (ct*8u)*v_sk, v_sn);
            for (uint kk=0u;kk<BN;kk+=8u) {
                vf = vfs[kk/8u];
                simdgroup_load(pf, tgP + 0u*BN + kk, BN);  simdgroup_multiply_accumulate(o[0][t], pf, vf, o[0][t]);
                simdgroup_load(pf, tgP + 8u*BN + kk, BN);  simdgroup_multiply_accumulate(o[1][t], pf, vf, o[1][t]);
                simdgroup_load(pf, tgP + 16u*BN + kk, BN); simdgroup_multiply_accumulate(o[2][t], pf, vf, o[2][t]);
                simdgroup_load(pf, tgP + 24u*BN + kk, BN); simdgroup_multiply_accumulate(o[3][t], pf, vf, o[3][t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }
    if (n_full * BN < N_CTX) {          // one partial tail block (masked staging)
        uint kv_start = n_full * BN;
    {
        simdgroup_float8x8 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        simdgroup_half8x8 qf, kf;
        for (uint kc = 0u; kc < D; kc += 8u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lid; e < 64u*8u; e += NT) {
                uint rr = e / 8u, cc = e % 8u; uint kvr = kv_start + rr;
                tgKV[rr*8u + cc] = (kvr < N_CTX) ? K[k_base + kvr*k_sn + (kc+cc)*k_sk] : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(kf, tgKV + (sgitg*8u)*8u, 8u, 0, true);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(qf, tgQ + 0u*D + kc, D);  simdgroup_multiply_accumulate(s0, qf, kf, s0);
            simdgroup_load(qf, tgQ + 8u*D + kc, D);  simdgroup_multiply_accumulate(s1, qf, kf, s1);
            simdgroup_load(qf, tgQ + 16u*D + kc, D); simdgroup_multiply_accumulate(s2, qf, kf, s2);
            simdgroup_load(qf, tgQ + 24u*D + kc, D); simdgroup_multiply_accumulate(s3, qf, kf, s3);
        }
        simdgroup_store(s0, tg_S + 0u*BN + sgitg*8u, BN);
        simdgroup_store(s1, tg_S + 8u*BN + sgitg*8u, BN);
        simdgroup_store(s2, tg_S + 16u*BN + sgitg*8u, BN);
        simdgroup_store(s3, tg_S + 24u*BN + sgitg*8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Parallel online-softmax: all 256 threads (8 per row, row=lid/8) scan
            // BN/8 cols each and reduce partial max/sum via tg_pmax/tg_psum scratch —
            // vs the old 32-thread serial BN-col scan (was ~14% of block time).
            // Masked cols get -INFINITY in the max pass, so exp() = 0 in the sum pass:
            // the causal/tail GUARD is preserved without a second guard.
            uint pr = lid / 8u, pc = lid % 8u; 
            uint cA = pc * (BN / 8u), cB = cA + (BN / 8u);
            float pm = -INFINITY;
            for (uint cj = cA; cj < cB; cj++) {
                uint kv_row = kv_start + cj;
                float s = ((kv_row < N_CTX) || (kv_row - N_CTX < ((32u - N_CTX % 32u) % 32u))) /* source KV tail: 32 */ ? (tg_S[pr*BN+cj]*scale) : -INFINITY;
                tg_S[pr*BN+cj] = s; pm = max(pm, s);
            }
            tg_pmax[pr*8u + pc] = pm;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < BM) {
                uint r = lid; float m_prev = tg_m[r], l_prev = tg_l[r], m_new = m_prev;
                bool m_ended = false;
                for (uint j = 0u; j < 8u; j++) {
                    m_new = max(m_new, tg_pmax[r*8u+j]);
                    if ((j % 4u) == 3u && m_new == -INFINITY) m_ended = true;
                }
                if (m_ended) m_new = NAN;
                tg_m[r] = m_new; tg_alpha[r] = exp(m_prev - m_new); tg_l[r] = l_prev * tg_alpha[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float ps = 0.0f; float mr = tg_m[pr];
            for (uint cj = cA; cj < cB; cj++) {
                float p = exp(tg_S[pr*BN+cj] - mr); tgP[pr*BN+cj] = half(p); ps += p;
            }
            tg_psum[pr*8u + pc] = ps;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < BM) {
                uint r = lid; float l = tg_l[r];
                for (uint j = 0u; j < 8u; j++) l += tg_psum[r*8u+j];
                tg_l[r] = l;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Keep a nonfinite source row local across the 8x8 MMAs.
            if (!isfinite(tg_l[pr])) {
                for (uint cj = cA; cj < cB; cj++) { tgP[pr*BN+cj] = half(0.0f); }
            }        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // CPU prototype: broadcast each row's alpha into an identically typed fragment.
        // Each SIMD group owns 256 slots; no cross-group scratch reader exists here.
        {

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[0u*8u+e/8u];
                tg_S[sgitg*256u+0u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
{

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[1u*8u+e/8u];
                tg_S[sgitg*256u+1u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
{

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[2u*8u+e/8u];
                tg_S[sgitg*256u+2u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
{

            for (uint e=lid%32u;e<64u;e+=32u) {
                float a=tg_alpha[3u*8u+e/8u];
                tg_S[sgitg*256u+3u*64u+e] = isfinite(a) ? a : 0.0f;
            }
        
}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 pf, vf, vfs[64/8];
        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*8u;
            // Same-type storage multiplication: no diagonal MMA and no lane-row formula.
            {

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 0u*64u, 8u);
                o[0u][t].thread_elements() = o[0u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
{

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 1u*64u, 8u);
                o[1u][t].thread_elements() = o[1u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
{

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 2u*64u, 8u);
                o[2u][t].thread_elements() = o[2u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
{

                simdgroup_float8x8 alpha_fragment;
                simdgroup_load(alpha_fragment, tg_S + sgitg*256u + 3u*64u, 8u);
                o[3u][t].thread_elements() = o[3u][t].thread_elements() * alpha_fragment.thread_elements();
            
}
            for (uint kk=0u;kk<BN;kk+=8u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lid%32u; e < 64u; e += 32u) {
                    uint rr = e / 8u, cc = e % 8u; uint kvr = kv_start + kk + rr;
                    tgKV[sgitg*64u + rr*8u + cc] = (kvr < N_CTX) ? V[v_base + kvr*v_sn + (ct*8u + cc)*v_sk] : half(0);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(vfs[kk/8u], tgKV + sgitg*64u, 8u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk=0u;kk<BN;kk+=8u) {
                vf = vfs[kk/8u];
                simdgroup_load(pf, tgP + 0u*BN + kk, BN);  simdgroup_multiply_accumulate(o[0][t], pf, vf, o[0][t]);
                simdgroup_load(pf, tgP + 8u*BN + kk, BN);  simdgroup_multiply_accumulate(o[1][t], pf, vf, o[1][t]);
                simdgroup_load(pf, tgP + 16u*BN + kk, BN); simdgroup_multiply_accumulate(o[2][t], pf, vf, o[2][t]);
                simdgroup_load(pf, tgP + 24u*BN + kk, BN); simdgroup_multiply_accumulate(o[3][t], pf, vf, o[3][t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }


    {

        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*8u;
            simdgroup_store(o[0u][t], on_scratch + sgitg*64u, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e=lid%32u;e<64u;e+=32u) {
                uint dr=e/8u, dc=e%8u;
                uint qr2=(q_start+0u*8u+dr), dc2=(ct*8u+dc);
                if (qr2 < N_CTX)
                    Out[o_base + qr2*o_sm + dc2*o_sk] = half(!isfinite(tg_l[0u*8u+dr]) ? NAN : float(on_scratch[sgitg*64u+e] * float((tg_l[0u*8u+dr]>0.0f) ? (1.0f/tg_l[0u*8u+dr]) : 0.0f)));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    
}
{

        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*8u;
            simdgroup_store(o[1u][t], on_scratch + sgitg*64u, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e=lid%32u;e<64u;e+=32u) {
                uint dr=e/8u, dc=e%8u;
                uint qr2=(q_start+1u*8u+dr), dc2=(ct*8u+dc);
                if (qr2 < N_CTX)
                    Out[o_base + qr2*o_sm + dc2*o_sk] = half(!isfinite(tg_l[1u*8u+dr]) ? NAN : float(on_scratch[sgitg*64u+e] * float((tg_l[1u*8u+dr]>0.0f) ? (1.0f/tg_l[1u*8u+dr]) : 0.0f)));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    
}
{

        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*8u;
            simdgroup_store(o[2u][t], on_scratch + sgitg*64u, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e=lid%32u;e<64u;e+=32u) {
                uint dr=e/8u, dc=e%8u;
                uint qr2=(q_start+2u*8u+dr), dc2=(ct*8u+dc);
                if (qr2 < N_CTX)
                    Out[o_base + qr2*o_sm + dc2*o_sk] = half(!isfinite(tg_l[2u*8u+dr]) ? NAN : float(on_scratch[sgitg*64u+e] * float((tg_l[2u*8u+dr]>0.0f) ? (1.0f/tg_l[2u*8u+dr]) : 0.0f)));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    
}
{

        for (uint t=0u;t<TPG;t++) {
            uint ct = sgitg + t*8u;
            simdgroup_store(o[3u][t], on_scratch + sgitg*64u, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e=lid%32u;e<64u;e+=32u) {
                uint dr=e/8u, dc=e%8u;
                uint qr2=(q_start+3u*8u+dr), dc2=(ct*8u+dc);
                if (qr2 < N_CTX)
                    Out[o_base + qr2*o_sm + dc2*o_sk] = half(!isfinite(tg_l[3u*8u+dr]) ? NAN : float(on_scratch[sgitg*64u+e] * float((tg_l[3u*8u+dr]>0.0f) ? (1.0f/tg_l[3u*8u+dr]) : 0.0f)));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    
}
}
