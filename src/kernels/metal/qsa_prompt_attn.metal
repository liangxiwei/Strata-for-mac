// src/kernels/metal/qsa_prompt_attn.metal - the port of src/kernels/cuda/qsa_prompt_attn.cu (K15).
//
// WHY THIS IS NOT THE CUDA KERNEL.  Both CUDA kernels are built on instructions this GPU does not have:
//   * prompt_attn_kernel<KV_MODE> stages q as FP16 hi/lo halves and runs m16n8k16 f16 MMA (sm_75+) over a
//     ~54 KB (mode 0) / ~38 KB (mode 1) extern __shared__ block - over the 32,768 B threadgroup budget this
//     GPU measures all by itself, before the rule-10 occupancy question;
//   * prompt_attn_i8_kernel cp.asyncs its K/V slices into a 45.6 KB Smem2 and is otherwise the same MMA
//     design (and cp.async is sm_80-only);
//   * prompt_attn_wmma_kernel is the gfx12 WMMA twin (HIP-only).
// The port therefore keeps the FILE's contract rather than its instruction mix: one block per (query, KV
// head) walks that query's whole selection in 32-cell chunks with an ONLINE softmax - no split-K scratch,
// no merge kernel - with the decode kernel's own f32 dot arithmetic (warp per cell, 8 dims per lane,
// ascending, butterfly), the int8/fp16/q4 reads of qsa_decode_attn.metal's helpers, and p.v one dim per
// thread.  qsa_prompt_attn.hpp's accuracy contract ("not bitwise with qsa_decode_attn_batch; the parity
// test bounds the difference") is exactly what this satisfies, and the threadgroup budget is
// 12 KB (q rows) + 12 KB (value accumulators) + 1.5 KB (probabilities) + change ~ 25.8 KB.
//
// KV_MODE 3 (hybrid K8V4) rounds V through fp16 at load, as the CUDA kernel's __half V rows do; mode 2
// (Q4_0 K) is refused on the host, as on CUDA.
#include "strata_port.metalh"

constant const int PA_HD = 256;          // head_dim
constant const int PA_G = 12;            // query heads per KV head
constant const int PA_CH = 32;           // cells per chunk
constant const int PA_THREADS = 256;
constant const int PA_WARPS = PA_THREADS / 32;
constant const int PA_KV_Q8_GROUP = 64;
constant const int PA_QK4_0 = 32;

// the K-side load of 8 values at dims [d0, d0+8) of a pool row (the decode kernel's load8 arithmetic)
static inline void pa_load8_k_f16(constant const uint* k_pool, long row, int d0, thread float* out) {
    constant const uint16_t* base = reinterpret_cast<constant const uint16_t*>(k_pool) + (ulong) row * PA_HD + d0;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = f32_from_f16(base[j]);
}
static inline void pa_load8_k_q8(constant const int8_t* k_q, constant const uint* k_scale, long row, int d0,
                                 thread float* out) {
    constant const uint16_t* scales = reinterpret_cast<constant const uint16_t*>(k_scale);
    const uint16_t sbits = scales[(ulong) row * (PA_HD / PA_KV_Q8_GROUP) + d0 / PA_KV_Q8_GROUP];
    const float sc = f32_from_f16(sbits);
    constant const int8_t* c = k_q + (ulong) row * PA_HD + d0;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}

// the V-side load of one dim of a pool row; mode 3 rounds the q4 value through fp16 (the CUDA kernel's
// dequantize-to-fp16-at-gather)
static inline float pa_load_v_f16(constant const uint* v_pool, long row, int t) {
    return f32_from_f16(reinterpret_cast<constant const uint16_t*>(v_pool)[(ulong) row * PA_HD + t]);
}
static inline float pa_load_v_q8(constant const int8_t* v_q, constant const uint* v_scale, long row, int t) {
    constant const uint16_t* vs = reinterpret_cast<constant const uint16_t*>(v_scale);
    const float sc = f32_from_f16(vs[(ulong) row * (PA_HD / PA_KV_Q8_GROUP) + t / PA_KV_Q8_GROUP]);
    return (float) v_q[(ulong) row * PA_HD + t] * sc;
}
static inline float pa_load_v_q4_h(constant const uint8_t* v_q4, long row, int t) {
    const uint bph = (PA_HD / PA_QK4_0) * 18;
    const int b = t / PA_QK4_0, rem = t % PA_QK4_0;
    constant const uint8_t* blk = v_q4 + (ulong) row * bph + (ulong) b * 18;
    const float d = f32_from_f16((uint16_t) ((uint) blk[0] | ((uint) blk[1] << 8)));
    const int j = rem < 16 ? rem : rem - 16;
    const uint8_t byte = blk[2 + j];
    const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
    return f32_from_f16(f16_from_f32((float) nibble * d));
}

#define PA_KERNEL(NAME, KV_MODE)                                                                      \
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
        const int n = steps[(ulong) qi * 4 + 3];                     /* kStepWidth */                 \
        const float scale = 1.0f / metal::precise::sqrt((float) PA_HD);   /* exactly 1/16 */          \
        threadgroup float sq[PA_G][PA_HD];                         /* 12 KB: this KV head's queries */ \
        threadgroup float acc[PA_G][PA_HD];                        /* 12 KB: running p.v sums */     \
        threadgroup float sp[PA_G][PA_CH];                         /* the chunk's probabilities */   \
        threadgroup float alpha[PA_G];                                                                \
        threadgroup float mrow[PA_G], lsum[PA_G];                                                     \
        threadgroup long srow[PA_CH];                                                                 \
        for (int i = (int) t; i < PA_G * PA_HD; i += PA_THREADS) {                                    \
            sq[i / PA_HD][i % PA_HD] = qp[i];    /* qp already sits at this KV head's 12 query rows */ \
            acc[i / PA_HD][i % PA_HD] = 0.0f;   /* threadgroup memory is uninitialized, even 0*NaN */ \
        }                                                                                            \
        if (t < (uint) PA_G) { mrow[t] = -INFINITY; lsum[t] = 0.0f; }                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
        for (int c0 = 0; c0 < n; c0 += PA_CH) {                                                       \
            const int nh = min(PA_CH, n - c0);                                                        \
            if (t < (uint) PA_CH) {                                                                   \
                long r = -1;                                                                          \
                if (t < nh) {                                                                         \
                    const int cell = sel[c0 + t];                                                     \
                    const long page = (long) page_table[(ulong) cell / (ulong) page_size];            \
                    /* a block the KV streaming left non-resident keeps page -1: masked, not read */  \
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);    \
                }                                                                                     \
                srow[t] = r;                                                                          \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* scores: warp w takes cells w, w+8, ...; each lane holds 8 of the 256 dims */            \
            for (int c = (int) sg; c < PA_CH; c += PA_WARPS) {                                        \
                if (c >= nh || srow[c] < 0) {                                                         \
                    if (lane < (uint) PA_G) sp[lane][c] = -INFINITY;                                  \
                    continue;                                                                         \
                }                                                                                     \
                float k8[8];                                                                          \
                if (KV_MODE == 0) pa_load8_k_f16(k_pool, srow[c], (int) lane * 8, k8);                \
                else if (KV_MODE == 1) pa_load8_k_q8(k_q, k_scale, srow[c], (int) lane * 8, k8);      \
                else pa_load8_k_q8(k_q, k_scale, srow[c], (int) lane * 8, k8);   /* mode 3: int8 K */ \
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
            /* the online update of the 12 heads' max and sum (one thread per head) */                \
            if (t < (uint) PA_G) {                                                                    \
                float mx = -INFINITY;                                                                 \
                for (int c = 0; c < PA_CH; ++c) mx = metal::precise::fmax(mx, sp[t][c]);              \
                const float m_old = mrow[t];                                                          \
                const float m_new = metal::precise::fmax(m_old, mx);                                  \
                float sum = 0.0f;                                                                     \
                for (int c = 0; c < PA_CH; ++c) {                                                     \
                    const float e = sp[t][c] == -INFINITY ? 0.0f : metal::precise::exp(sp[t][c] - m_new); \
                    sp[t][c] = e;                                                                     \
                    sum += e;                                                                         \
                }                                                                                     \
                alpha[t] = m_old == -INFINITY ? 0.0f : metal::precise::exp(m_old - m_new);            \
                mrow[t] = m_new;                                                                      \
                lsum[t] = fma(lsum[t], alpha[t], sum);                                                \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
            /* rescale the running sums, then accumulate this chunk's p.v one dim per thread */        \
            _Pragma("unroll")                                                                         \
            for (int h = 0; h < PA_G; ++h) acc[h][t] *= alpha[h];                                     \
            for (int c = 0; c < nh; ++c) {                                                            \
                if (srow[c] < 0) continue;                                                            \
                const float v = KV_MODE == 0    ? pa_load_v_f16(v_pool, srow[c], (int) t)             \
                              : KV_MODE == 1    ? pa_load_v_q8(v_q, v_scale, srow[c], (int) t)        \
                              : KV_MODE == 3    ? pa_load_v_q4_h(v_q4, srow[c], (int) t)              \
                                                 : pa_load_v_f16(v_pool, srow[c], (int) t);          \
                _Pragma("unroll")                                                                     \
                for (int h = 0; h < PA_G; ++h) acc[h][t] = fma(sp[h][c], v, acc[h][t]);               \
            }                                                                                         \
            threadgroup_barrier(mem_flags::mem_threadgroup);   /* sp/srow rewrite next chunk */       \
        }                                                                                             \
        _Pragma("unroll")                                                                             \
        for (int h = 0; h < PA_G; ++h) {                                                              \
            const float inv = lsum[h] > 0.0f ? 1.0f / lsum[h] : 0.0f;                                 \
            out[(ulong) h * PA_HD + t] = acc[h][t] * inv;                                             \
        }                                                                                             \
    }

PA_KERNEL(prompt_attn_kernel_m0, 0)   // FP16 pools
PA_KERNEL(prompt_attn_kernel_m1, 1)   // INT8 codes + fp16 scales per 64
PA_KERNEL(prompt_attn_kernel_m3, 3)   // hybrid K8V4: INT8 K, V dequantized from q4_0 through fp16

// The same attention with the running p.v sums in registers and the chunk's exps spread over the threadgroup.
// Every value's operations are the kernel above's: scores (8-dim lane dots, butterfly, * scale), per head the max
// over the chunk's 32 cells in ascending order (one thread), exp(s - m_new) per cell, their sum in ascending order
// (one thread), alpha, lsum = fma(lsum, alpha, sum), acc *= alpha, then fma(p, v, acc) in ascending cells. acc is
// only ever touched by the thread owning its dim, so it needs no threadgroup memory: 14.3 KB instead of 26.5 KB,
// two groups per core. Measured on the M2 Max (micro/prompt-attn.log, 1,024 queries x 2 KV heads at 1.5K-30K
// context, finite and inf / NaN / masked-page inputs): 117-136 -> 45-51 ms per launch, every output bitwise.
#define PA_KERNEL_REG(NAME, KV_MODE)                                                                     \
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
PA_KERNEL_REG(prompt_attn_reg_m0, 0)
PA_KERNEL_REG(prompt_attn_reg_m1, 1)
PA_KERNEL_REG(prompt_attn_reg_m3, 3)
