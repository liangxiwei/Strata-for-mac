// Prompt-attention variants. The production kernels are included verbatim (qsa_prompt_attn.metal). The variant
// keeps every value's operation sequence: scores (the same 8-dim lane dots, butterfly, * scale), per head the
// max over the chunk's 32 cells in ascending order (one thread, as before), exp(s - m_new) per cell, the sum of
// those in ascending order (one thread), alpha, lsum = fma(lsum, alpha, sum), acc *= alpha then fma(p, v, acc) in
// ascending cells. What changes: the running acc lives in each thread's registers (it is only ever touched by
// the thread owning that dim), which frees 12 KB of threadgroup memory; the 384 exps of a chunk run on 384 threads
// instead of 12 serial loops of 32.
#include "qsa_prompt_attn.metal"

#define PA2_KERNEL(NAME, KV_MODE)                                                                     \
    kernel void NAME(constant const float* q [[buffer(0)]],                                           \
                     constant const uint* k_pool [[buffer(1)]],                                       \
                     constant const uint* v_pool [[buffer(2)]],                                       \
                     constant const int8_t* k_q [[buffer(3)]],                                        \
                     constant const int8_t* v_q [[buffer(4)]],                                        \
                     constant const uint* k_scale [[buffer(5)]],                                      \
                     constant const uint* v_scale [[buffer(6)]],                                      \
                     constant const uint8_t* k_q4 [[buffer(7)]],                                      \
                     constant const uint8_t* v_q4 [[buffer(8)]],                                      \
                     constant const int* page_table [[buffer(9)]],                                    \
                     constant const int* ids [[buffer(10)]],                                          \
                     constant const int* steps [[buffer(11)]],                                        \
                     constant const int& n_kv_heads [[buffer(12)]],                                   \
                     constant const int& page_size [[buffer(13)]],                                    \
                     constant const long& cap [[buffer(14)]],                                         \
                     device float* attn [[buffer(15)]],                                               \
                     uint2 gpos [[threadgroup_position_in_grid]],                                     \
                     uint t [[thread_index_in_threadgroup]],                                          \
                     uint lane [[thread_index_in_simdgroup]],                                         \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                    \
        const int qi = (int) gpos.x, kvh = (int) gpos.y;                                              \
        const int n_head = n_kv_heads * PA_G;                                                         \
        constant const float* qp = q + (ulong) qi * n_head * PA_HD + (ulong) kvh * PA_G * PA_HD;      \
        device float* out = attn + (ulong) qi * n_head * PA_HD + (ulong) kvh * PA_G * PA_HD;          \
        constant const int* sel = ids + (ulong) qi * cap;                                             \
        const int n = steps[(ulong) qi * 4 + 3];                                                      \
        const float scale = 1.0f / metal::precise::sqrt((float) PA_HD);                               \
        threadgroup float sq[PA_G][PA_HD];                                                            \
        threadgroup float sp[PA_G][PA_CH];                                                            \
        threadgroup float alpha[PA_G], mnew[PA_G];                                                    \
        threadgroup float mrow[PA_G], lsum[PA_G];                                                     \
        threadgroup long srow[PA_CH];                                                                 \
        float acc[PA_G];                                                                              \
        _Pragma("unroll") for (int h = 0; h < PA_G; ++h) acc[h] = 0.0f;                               \
        for (int i = (int) t; i < PA_G * PA_HD; i += PA_THREADS) sq[i / PA_HD][i % PA_HD] = qp[i];    \
        if (t < (uint) PA_G) { mrow[t] = -INFINITY; lsum[t] = 0.0f; }                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
        for (int c0 = 0; c0 < n; c0 += PA_CH) {                                                       \
            const int nh = min(PA_CH, n - c0);                                                        \
            if (t < (uint) PA_CH) {                                                                   \
                long r = -1;                                                                          \
                if (t < nh) {                                                                         \
                    const int cell = sel[c0 + t];                                                     \
                    const long page = (long) page_table[(ulong) cell / (ulong) page_size];            \
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);    \
                }                                                                                     \
                srow[t] = r;                                                                          \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            for (int c = (int) sg; c < PA_CH; c += PA_WARPS) {                                        \
                if (c >= nh || srow[c] < 0) {                                                         \
                    if (lane < (uint) PA_G) sp[lane][c] = -INFINITY;                                  \
                    continue;                                                                         \
                }                                                                                     \
                float k8[8];                                                                          \
                if (KV_MODE == 0) pa_load8_k_f16(k_pool, srow[c], (int) lane * 8, k8);                \
                else pa_load8_k_q8(k_q, k_scale, srow[c], (int) lane * 8, k8);                        \
                _Pragma("unroll")                                                                     \
                for (int h = 0; h < PA_G; ++h) {                                                      \
                    const float4 qa = *(const threadgroup float4*) &sq[h][lane * 8];                     \
                    const float4 qb = *(const threadgroup float4*) &sq[h][lane * 8 + 4];                 \
                    float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +             \
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;              \
                    for (int o = 16; o > 0; o >>= 1) s += simd_shuffle_xor(s, (uint) o);              \
                    if (lane == 0) sp[h][c] = s * scale;                                              \
                }                                                                                     \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* the max in ascending cells, one thread per head (unchanged) */                         \
            if (t < (uint) PA_G) {                                                                    \
                float mx = -INFINITY;                                                                 \
                for (int c = 0; c < PA_CH; ++c) mx = metal::precise::fmax(mx, sp[t][c]);              \
                mnew[t] = metal::precise::fmax(mrow[t], mx);                                          \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* the exps, one per (head, cell) - the same expression per element */                   \
            for (int i = (int) t; i < PA_G * PA_CH; i += PA_THREADS) {                                \
                const int h = i / PA_CH, c = i % PA_CH;                                               \
                const float s = sp[h][c];                                                             \
                sp[h][c] = s == -INFINITY ? 0.0f : metal::precise::exp(s - mnew[h]);                  \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* the sum in ascending cells, alpha and the running sums, one thread per head */         \
            if (t < (uint) PA_G) {                                                                    \
                float sum = 0.0f;                                                                     \
                for (int c = 0; c < PA_CH; ++c) sum += sp[t][c];                                      \
                const float m_old = mrow[t], m_new = mnew[t];                                         \
                alpha[t] = m_old == -INFINITY ? 0.0f : metal::precise::exp(m_old - m_new);            \
                mrow[t] = m_new;                                                                      \
                lsum[t] = fma(lsum[t], alpha[t], sum);                                                \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            _Pragma("unroll")                                                                         \
            for (int h = 0; h < PA_G; ++h) acc[h] *= alpha[h];                                        \
            for (int c = 0; c < nh; ++c) {                                                            \
                if (srow[c] < 0) continue;                                                            \
                const float v = KV_MODE == 0 ? pa_load_v_f16(v_pool, srow[c], (int) t)                \
                              : KV_MODE == 1 ? pa_load_v_q8(v_q, v_scale, srow[c], (int) t)           \
                                             : pa_load_v_q4_h(v_q4, srow[c], (int) t);               \
                _Pragma("unroll")                                                                     \
                for (int h = 0; h < PA_G; ++h) acc[h] = fma(sp[h][c], v, acc[h]);                     \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        }                                                                                             \
        _Pragma("unroll")                                                                             \
        for (int h = 0; h < PA_G; ++h) {                                                              \
            const float inv = lsum[h] > 0.0f ? 1.0f / lsum[h] : 0.0f;                                 \
            out[(ulong) h * PA_HD + t] = acc[h] * inv;                                                \
        }                                                                                             \
    }
PA2_KERNEL(pa2_m0, 0)
PA2_KERNEL(pa2_m1, 1)

#define PA3_KERNEL(NAME, KV_MODE)                                                                     \
    kernel void NAME(constant const float* q [[buffer(0)]],                                           \
                     constant const uint* k_pool [[buffer(1)]],                                       \
                     constant const uint* v_pool [[buffer(2)]],                                       \
                     constant const int8_t* k_q [[buffer(3)]],                                        \
                     constant const int8_t* v_q [[buffer(4)]],                                        \
                     constant const uint* k_scale [[buffer(5)]],                                      \
                     constant const uint* v_scale [[buffer(6)]],                                      \
                     constant const uint8_t* k_q4 [[buffer(7)]],                                      \
                     constant const uint8_t* v_q4 [[buffer(8)]],                                      \
                     constant const int* page_table [[buffer(9)]],                                    \
                     constant const int* ids [[buffer(10)]],                                          \
                     constant const int* steps [[buffer(11)]],                                        \
                     constant const int& n_kv_heads [[buffer(12)]],                                   \
                     constant const int& page_size [[buffer(13)]],                                    \
                     constant const long& cap [[buffer(14)]],                                         \
                     device float* attn [[buffer(15)]],                                               \
                     uint2 gpos [[threadgroup_position_in_grid]],                                     \
                     uint t [[thread_index_in_threadgroup]],                                          \
                     uint lane [[thread_index_in_simdgroup]],                                         \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                    \
        const int qi = (int) gpos.x, kvh = (int) gpos.y;                                              \
        const int n_head = n_kv_heads * PA_G;                                                         \
        constant const float* qp = q + (ulong) qi * n_head * PA_HD + (ulong) kvh * PA_G * PA_HD;      \
        device float* out = attn + (ulong) qi * n_head * PA_HD + (ulong) kvh * PA_G * PA_HD;          \
        constant const int* sel = ids + (ulong) qi * cap;                                             \
        const int n = steps[(ulong) qi * 4 + 3];                                                      \
        const float scale = 1.0f / metal::precise::sqrt((float) PA_HD);                               \
        threadgroup float sq[PA_G][PA_HD];                                                            \
        threadgroup float sp[PA_G][PA_CH];                                                            \
        threadgroup float alpha[PA_G], mnew[PA_G];                                                    \
        threadgroup float mrow[PA_G], lsum[PA_G];                                                     \
        threadgroup long srow[PA_CH];                                                                 \
        float acc[PA_G];                                                                              \
        _Pragma("unroll") for (int h = 0; h < PA_G; ++h) acc[h] = 0.0f;                               \
        for (int i = (int) t; i < PA_G * PA_HD; i += PA_THREADS) sq[i / PA_HD][i % PA_HD] = qp[i];    \
        if (t < (uint) PA_G) { mrow[t] = -INFINITY; lsum[t] = 0.0f; }                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
        for (int c0 = 0; c0 < n; c0 += PA_CH) {                                                       \
            const int nh = min(PA_CH, n - c0);                                                        \
            if (t < (uint) PA_CH) {                                                                   \
                long r = -1;                                                                          \
                if (t < nh) {                                                                         \
                    const int cell = sel[c0 + t];                                                     \
                    const long page = (long) page_table[(ulong) cell / (ulong) page_size];            \
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);    \
                }                                                                                     \
                srow[t] = r;                                                                          \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            for (int c = (int) sg; c < PA_CH; c += PA_WARPS) {                                        \
                if (c >= nh || srow[c] < 0) {                                                         \
                    if (lane < (uint) PA_G) sp[lane][c] = -INFINITY;                                  \
                    continue;                                                                         \
                }                                                                                     \
                float k8[8];                                                                          \
                if (KV_MODE == 0) pa_load8_k_f16(k_pool, srow[c], (int) lane * 8, k8);                \
                else pa_load8_k_q8(k_q, k_scale, srow[c], (int) lane * 8, k8);                        \
                /* the 12 heads' lane dots, then their butterflies four at a time (per component the */ \
                /* same xor tree on the same values) */                                                \
                float sh[PA_G];                                                                       \
                _Pragma("unroll")                                                                     \
                for (int h = 0; h < PA_G; ++h) {                                                      \
                    const float4 qa = *(const threadgroup float4*) &sq[h][lane * 8];                     \
                    const float4 qb = *(const threadgroup float4*) &sq[h][lane * 8 + 4];                 \
                    sh[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +               \
                            k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;                \
                }                                                                                     \
                _Pragma("unroll")                                                                     \
                for (int h = 0; h < PA_G; h += 4) {                                                   \
                    float4 s4 = float4(sh[h], sh[h + 1], sh[h + 2], sh[h + 3]);                       \
                    for (int o = 16; o > 0; o >>= 1) s4 += simd_shuffle_xor(s4, (uint) o);            \
                    if (lane == 0) {                                                                  \
                        sp[h][c] = s4.x * scale; sp[h + 1][c] = s4.y * scale;                         \
                        sp[h + 2][c] = s4.z * scale; sp[h + 3][c] = s4.w * scale;                     \
                    }                                                                                 \
                }                                                                                     \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* the max in ascending cells, one thread per head (unchanged) */                         \
            if (t < (uint) PA_G) {                                                                    \
                float mx = -INFINITY;                                                                 \
                for (int c = 0; c < PA_CH; ++c) mx = metal::precise::fmax(mx, sp[t][c]);              \
                mnew[t] = metal::precise::fmax(mrow[t], mx);                                          \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* the exps, one per (head, cell) - the same expression per element */                   \
            for (int i = (int) t; i < PA_G * PA_CH; i += PA_THREADS) {                                \
                const int h = i / PA_CH, c = i % PA_CH;                                               \
                const float s = sp[h][c];                                                             \
                sp[h][c] = s == -INFINITY ? 0.0f : metal::precise::exp(s - mnew[h]);                  \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* the sum in ascending cells, alpha and the running sums, one thread per head */         \
            if (t < (uint) PA_G) {                                                                    \
                float sum = 0.0f;                                                                     \
                for (int c = 0; c < PA_CH; ++c) sum += sp[t][c];                                      \
                const float m_old = mrow[t], m_new = mnew[t];                                         \
                alpha[t] = m_old == -INFINITY ? 0.0f : metal::precise::exp(m_old - m_new);            \
                mrow[t] = m_new;                                                                      \
                lsum[t] = fma(lsum[t], alpha[t], sum);                                                \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            _Pragma("unroll")                                                                         \
            for (int h = 0; h < PA_G; ++h) acc[h] *= alpha[h];                                        \
            for (int c = 0; c < nh; ++c) {                                                            \
                if (srow[c] < 0) continue;                                                            \
                const float v = KV_MODE == 0 ? pa_load_v_f16(v_pool, srow[c], (int) t)                \
                              : KV_MODE == 1 ? pa_load_v_q8(v_q, v_scale, srow[c], (int) t)           \
                                             : pa_load_v_q4_h(v_q4, srow[c], (int) t);               \
                _Pragma("unroll")                                                                     \
                for (int h = 0; h < PA_G; ++h) acc[h] = fma(sp[h][c], v, acc[h]);                     \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        }                                                                                             \
        _Pragma("unroll")                                                                             \
        for (int h = 0; h < PA_G; ++h) {                                                              \
            const float inv = lsum[h] > 0.0f ? 1.0f / lsum[h] : 0.0f;                                 \
            out[(ulong) h * PA_HD + t] = acc[h] * inv;                                                \
        }                                                                                             \
    }
PA3_KERNEL(pa3_m0, 0)
