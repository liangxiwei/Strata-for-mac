// src/kernels/metal/qsa_decode_attn.metal - the port of src/kernels/cuda/qsa_decode_attn.cu's kernels
// (K15): the split-K decode attention over the selected cells, FP16 / INT8 / Q4_0 / hybrid K8V4 pools.
//
// RULE 9: the CUDA kernels take QsaAttnPools BY VALUE (nine device pointers in a struct) - a pointer stored
// in argument bytes is not a usable device pointer on this GPU (measured, docs/PORT_METAL/PROGRESS.md).
// Every pool pointer is therefore its own bound [[buffer(N)]], the null ones bind nil (R4), and the
// per-query offsets the CUDA kernel took by pointer arithmetic stay IN the kernel: the bound arguments are
// the base pointers the launcher receives, and query gpos.z's stride arrives as the `stride` scalar.  The
// CUDA template <int KV_MODE> becomes one body macro stamped four times (the iq_kernels.metal pattern;
// MSL's threadgroup locals live at kernel scope).
//
// The CUDA `__expf` (the fast intrinsic) became metal::precise::exp: the parity tolerances (vs an FP64
// reference at 4x the old kernel's error) sit far above either exp's error.
#include "strata_port.metalh"

constant const int DA_HD = 256;          // head_dim
constant const int DA_G = 12;            // query heads per KV head (24 / 2)
constant const int DA_CHUNK = 64;        // cells per block
constant const int DA_THREADS = 256;
constant const int DA_WARPS = DA_THREADS / 32;
constant const int DA_KV_Q8_GROUP = 64;  // kv_q8.hpp
constant const int DA_QK4_0 = 32;        // kv_q4.hpp: one q4_0 block is 32 values, 18 bytes
constant const float DA_FLT_MAX = 3.4028234663852886e+38f;

static inline float da_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, (uint) o);
    return v;
}
static inline float da_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = metal::precise::fmax(v, simd_shuffle_xor(v, (uint) o));
    return v;
}

// 8 consecutive values of one cell's key row for the KV head of `row`, dimensions [d0, d0+8)
static inline void da_load8_f16(constant const uint* pool, long row, int d0, thread float* out) {
    constant const uint16_t* base = reinterpret_cast<constant const uint16_t*>(pool) + (ulong) row * DA_HD + d0;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = f32_from_f16(base[j]);
}

static inline void da_load8_q8(constant const int8_t* codes, constant const uint* scale, long row, int d0,
                               thread float* out) {
    constant const uint16_t* scales = reinterpret_cast<constant const uint16_t*>(scale);
    const uint16_t sbits = scales[(ulong) row * (DA_HD / DA_KV_Q8_GROUP) + d0 / DA_KV_Q8_GROUP];
    const float sc = f32_from_f16(sbits);
    constant const int8_t* c = codes + (ulong) row * DA_HD + d0;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}

// q4_0: 18-byte blocks of 32 values (a fp16 scale + 16 bytes of nibbles, low nibble = j, high = j+16)
static inline void da_load8_q4(constant const uint8_t* q4, long row, int d0, thread float* out) {
    const uint bph = (DA_HD / DA_QK4_0) * 18;    // bytes per head-row
    const int b = d0 / DA_QK4_0, rem = d0 % DA_QK4_0;
    constant const uint8_t* blk = q4 + (ulong) row * bph + (ulong) b * 18;
    const float d = f32_from_f16((uint16_t) ((uint) blk[0] | ((uint) blk[1] << 8)));
    const int j = (rem < 16) ? rem : rem - 16;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (blk[2 + j + k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (blk[2 + j + k] >> 4) - 8) * d;
    }
}

// the one-thread V read of the value pass (dimension t of the cell's row)
static inline float da_load_v_f16(constant const uint* v_pool, long row, int t) {
    return f32_from_f16(reinterpret_cast<constant const uint16_t*>(v_pool)[(ulong) row * DA_HD + t]);
}
static inline float da_load_v_q8(constant const int8_t* v_q, constant const uint* v_scale, long row, int t) {
    constant const uint16_t* vs = reinterpret_cast<constant const uint16_t*>(v_scale);
    const float sc = f32_from_f16(vs[(ulong) row * (DA_HD / DA_KV_Q8_GROUP) + t / DA_KV_Q8_GROUP]);
    return (float) v_q[(ulong) row * DA_HD + t] * sc;
}
static inline float da_load_v_q4(constant const uint8_t* v_q4, long row, int t) {
    const uint bph = (DA_HD / DA_QK4_0) * 18;
    const int b = t / DA_QK4_0, rem = t % DA_QK4_0;
    constant const uint8_t* blk = v_q4 + (ulong) row * bph + (ulong) b * 18;
    const float d = f32_from_f16((uint16_t) ((uint) blk[0] | ((uint) blk[1] << 8)));
    const int j = rem < 16 ? rem : rem - 16;
    const uint8_t byte = blk[2 + j];
    const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
    return (float) nibble * d;
}

// the chunk kernel's whole body, stamped per KV_MODE (0 fp16, 1 int8, 2 q4 both sides, 3 hybrid K8V4)
#define DA_CHUNK_BODY(KV_MODE)                                                                        \
    constant const float* q0 = q;                                                                     \
    q0 += (ulong) gpos.z * (ulong) (n_kv_heads * DA_G) * DA_HD;                                       \
    constant const int* ids0 = ids + (ulong) gpos.z * (ulong) cap;                                    \
    constant const int* step0 = step + (ulong) gpos.z * 4;            /* kStepCount = 4 */            \
    device float* acc0 = part_acc + (ulong) gpos.z * (ulong) stride;                                  \
    device float* m0 = part_m + (ulong) gpos.z * (ulong) stride;                                      \
    device float* l0 = part_l + (ulong) gpos.z * (ulong) stride;                                      \
    threadgroup float sq[DA_G][DA_HD];               /* 12 KB: this KV head's query heads */          \
    threadgroup float sp[DA_G][DA_CHUNK];            /* scores, then probabilities */                 \
    threadgroup long srow[DA_CHUNK];                 /* pool row of each cell (page, kv head, slot) */\
    const int n_ids = step0[3];                                       /* kStepWidth */                \
    const int chunk = (int) gpos.x, kvh = (int) gpos.y;                                               \
    const int c0 = chunk * DA_CHUNK;                                                                  \
    const int n_here = min(DA_CHUNK, n_ids - c0);                                                     \
    const int slot = kvh * n_chunks + chunk;                                                          \
    if (n_here <= 0) {                                                                                \
        if (t < (uint) DA_G) {                                                                        \
            m0[(ulong) slot * DA_G + t] = -DA_FLT_MAX;                                                \
            l0[(ulong) slot * DA_G + t] = 0.0f;                                                       \
        }                                                                                             \
        return;                                                                                       \
    }                                                                                                 \
    for (int i = (int) t; i < DA_G * DA_HD; i += DA_THREADS)                                          \
        sq[i / DA_HD][i % DA_HD] = q0[(ulong) (kvh * DA_G) * DA_HD + i];                              \
    if (t < (uint) DA_CHUNK) {                                                                        \
        long r = -1;                                                                                  \
        if (t < n_here) {                                                                             \
            const int cell = ids0[c0 + t];                                                            \
            const long page = (long) page_table[(ulong) cell / (ulong) page_size];                    \
            /* a block the KV streaming could not make resident keeps page -1; its cells are masked */ \
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);            \
        }                                                                                             \
        srow[t] = r;                                                                                  \
    }                                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                  \
    /* scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions */     \
    for (int c = (int) sg; c < DA_CHUNK; c += DA_WARPS) {                                             \
        if (c >= n_here || srow[c] < 0) {                                                             \
            if (lane < (uint) DA_G) sp[lane][c] = -DA_FLT_MAX;                                        \
            continue;                                                                                 \
        }                                                                                             \
        float k8[8];                                                                                  \
        if (KV_MODE == 0) da_load8_f16(k_pool, srow[c], (int) lane * 8, k8);                          \
        else if (KV_MODE == 1) da_load8_q8(k_q, k_scale, srow[c], (int) lane * 8, k8);                \
        else if (KV_MODE == 3) da_load8_q8(k_q, k_scale, srow[c], (int) lane * 8, k8);                \
        else da_load8_q4(k_q4, srow[c], (int) lane * 8, k8);                                          \
        _Pragma("unroll")                                                                             \
        for (int h = 0; h < DA_G; ++h) {                                                              \
            const float4 qa = *(const threadgroup float4*) &sq[h][lane * 8];                             \
            const float4 qb = *(const threadgroup float4*) &sq[h][lane * 8 + 4];                         \
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +                     \
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;                      \
            s = da_warp_sum(s);                                                                       \
            if (lane == 0) sp[h][c] = s * scale;                                                      \
        }                                                                                             \
    }                                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                  \
    /* per-head chunk max and exp-sum: warp w handles heads w and w+8 */                              \
    for (int h = (int) sg; h < DA_G; h += DA_WARPS) {                                                 \
        const float a = sp[h][lane], b = sp[h][lane + 32];                                            \
        const float m = da_warp_max(metal::precise::fmax(a, b));                                      \
        const float ea = (lane < (uint) n_here && srow[lane] >= 0) ? metal::precise::exp(a - m) : 0.0f;   \
        const float eb = (lane + 32 < (uint) n_here && srow[lane + 32] >= 0) ? metal::precise::exp(b - m) \
                                                                             : 0.0f;                  \
        sp[h][lane] = ea;                                                                             \
        sp[h][lane + 32] = eb;                                                                        \
        const float l = da_warp_sum(ea + eb);                                                         \
        if (lane == 0) {                                                                              \
            m0[(ulong) slot * DA_G + h] = m;                                                          \
            l0[(ulong) slot * DA_G + h] = l;                                                          \
        }                                                                                             \
    }                                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                  \
    /* values: thread t owns dimension t for all 12 heads */                                          \
    float acc[DA_G];                                                                                  \
    _Pragma("unroll")                                                                                 \
    for (int h = 0; h < DA_G; ++h) acc[h] = 0.0f;                                                     \
    for (int c = 0; c < n_here; ++c) {                                                                \
        if (srow[c] < 0) continue;                   /* masked above, weight 0 */                     \
        const float v = KV_MODE == 0    ? da_load_v_f16(v_pool, srow[c], (int) t)                     \
                      : KV_MODE == 1    ? da_load_v_q8(v_q, v_scale, srow[c], (int) t)                \
                      : KV_MODE == 3    ? da_load_v_q4(v_q4, srow[c], (int) t)                        \
                                         : da_load_v_q4(v_q4, srow[c], (int) t);                      \
        _Pragma("unroll")                                                                             \
        for (int h = 0; h < DA_G; ++h) acc[h] = fma(sp[h][c], v, acc[h]);                             \
    }                                                                                                 \
    _Pragma("unroll")                                                                                 \
    for (int h = 0; h < DA_G; ++h) acc0[((ulong) slot * DA_G + h) * DA_HD + t] = acc[h];

#define DA_CHUNK_KERNEL(NAME, KV_MODE)                                                                \
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
                     constant const int* step [[buffer(11)]],                                         \
                     constant const int& n_kv_heads [[buffer(12)]],                                   \
                     constant const int& page_size [[buffer(13)]],                                    \
                     constant const float& scale [[buffer(14)]],                                      \
                     device float* part_acc [[buffer(15)]],                                           \
                     device float* part_m [[buffer(16)]],                                             \
                     device float* part_l [[buffer(17)]],                                             \
                     constant const int& n_chunks [[buffer(18)]],                                     \
                     constant const long& cap [[buffer(19)]],                                         \
                     constant const long& stride [[buffer(20)]],                                      \
                     uint3 gpos [[threadgroup_position_in_grid]],                                     \
                     uint t [[thread_index_in_threadgroup]],                                          \
                     uint lane [[thread_index_in_simdgroup]],                                         \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                    \
        DA_CHUNK_BODY(KV_MODE)                                                                        \
    }

DA_CHUNK_KERNEL(attn_chunk_kernel_m0, 0)
DA_CHUNK_KERNEL(attn_chunk_kernel_m1, 1)
DA_CHUNK_KERNEL(attn_chunk_kernel_m2, 2)
DA_CHUNK_KERNEL(attn_chunk_kernel_m3, 3)

// the merge: one block per (query head, query), 256 threads (one per dim), the usual log-sum-exp rescale
// over the chunks' partials.  CUDA reads gridDim.x for the attn row stride; the launcher passes n_head.
kernel void attn_merge_kernel(constant const float* part_acc [[buffer(0)]],
                              constant const float* part_m [[buffer(1)]],
                              constant const float* part_l [[buffer(2)]],
                              constant const int& n_chunks [[buffer(3)]],
                              device float* attn [[buffer(4)]],
                              constant const int& n_head [[buffer(5)]],
                              constant const long& stride [[buffer(6)]],
                              uint2 gpos [[threadgroup_position_in_grid]],
                              uint d [[thread_index_in_threadgroup]]) {
    constant const float* pa = part_acc + (ulong) gpos.y * (ulong) stride;
    constant const float* pm = part_m + (ulong) gpos.y * (ulong) stride;
    constant const float* pl = part_l + (ulong) gpos.y * (ulong) stride;
    device float* out = attn + (ulong) gpos.y * (ulong) n_head * DA_HD;
    const int h = (int) gpos.x;                 // global query head
    const int kvh = h / DA_G, hl = h % DA_G;
    float M = -DA_FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = metal::precise::fmax(M, pm[(kvh * n_chunks + c) * DA_G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = pm[slot * DA_G + hl];
        if (m == -DA_FLT_MAX) continue;
        const float w = metal::precise::exp(m - M);
        L = fma(pl[slot * DA_G + hl], w, L);
        acc = fma(pa[((ulong) slot * DA_G + hl) * DA_HD + d], w, acc);
    }
    out[(ulong) h * DA_HD + d] = L > 0.0f ? acc / L : 0.0f;
}
