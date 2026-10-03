// src/kernels/metal/qsa_select.metal - the port of src/kernels/cuda/qsa_select.cu's kernels (K15): the
// block scores of many QSA queries and the 4-pass radix top-k over them.
//
// NOT PORTED, DELIBERATELY:
//   * block_scores_tc_kernel (3xTF32 mma.sync, sm_80) and block_scores_wmma_kernel (gfx12) - Metal has
//     neither mma.sync nor that WMMA; qsa_block_scores_tc returns false and the warp kernel scores, exactly
//     as on a pre-sm_80 CUDA card or a non-gfx12 AMD one.  The tc kernel's own header disclaims bitwise
//     equality, so nothing downstream depends on it being the one that ran.
//   * block_topk_reg_kernel (the 1024-thread register top-k): its per-warp histograms alone are
//     32 x 256 ints = 32,768 B - the ENTIRE threadgroup-memory budget this GPU measures (Apple M2 Max,
//     maxThreadgroupMemoryLength 32768) - before s_warp/s_digit/s_above and the per-thread key[33] arrays,
//     and a 33-entry per-thread key table would pin occupancy under rule 10's drop.  qsa_block_topk
//     therefore always launches block_topk_kernel (the CUDA file's own STRATA_TOPK_OLD path), which the
//     CUDA file guarantees produces THE SAME IDS.
#include "strata_port.metalh"

constant const int QS_IDX_DIM = 128, QS_IDX_HEADS = 4, QS_R = 4;
constant const int QS_SCORE_WARPS = 8;
constant const int QS_TOPK_T = 256;

// Total order over f32 as an unsigned key (numpy's float semantics), qsa.metal's twin
static inline uint qs_order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// one warp per (query, block): 4-head relu'd dot of the block's pooled key (or the spare's `dead` key at
// b == n_bid, with the +1e9 tail bias when the tail has cells).  Grid (ceil(reach/8), nq), 256 threads.
kernel void block_scores_kernel(constant const float* pooled [[buffer(0)]],
                                constant const float* dead [[buffer(1)]],
                                constant const float* q_idx [[buffer(2)]],
                                constant const int* steps [[buffer(3)]],
                                constant const long& max_blocks [[buffer(4)]],
                                device float* out [[buffer(5)]],
                                uint2 gpos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const long qi = (long) gpos.y;
    constant const int* st = steps + (ulong) qi * 4;    // kStepCount = 4
    const long n_kv = (long) st[1], n_bid = (long) st[2];
    const long b = (long) gpos.x * QS_SCORE_WARPS + (long) sg;
    if (b > n_bid || b >= max_blocks) return;
    constant const float* key = (b == n_bid) ? dead : pooled + (ulong) b * QS_IDX_DIM;
    const float4 k4 = *(constant const float4*) (key + lane * 4);
    constant const float* q = q_idx + (ulong) qi * QS_IDX_HEADS * QS_IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < QS_IDX_HEADS; ++h) {
        const float4 q4 = *(constant const float4*) (q + h * QS_IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += simd_shuffle_xor(d, (uint) o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % QS_R != 0) score += 1e9f;
        out[(ulong) qi * max_blocks + b] = score;
    }
}

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block (rides the
// tensor-core launch on CUDA; kept for completeness - the .mm's tc path is off, so it idles here too)
kernel void block_scores_tail_kernel(constant const float* dead [[buffer(0)]],
                                     constant const float* q_idx [[buffer(1)]],
                                     constant const int* steps [[buffer(2)]],
                                     constant const long& max_blocks [[buffer(3)]],
                                     device float* out [[buffer(4)]],
                                     uint gpos [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const long qi = (long) gpos;
    constant const int* st = steps + (ulong) qi * 4;
    const long n_kv = (long) st[1], n_bid = (long) st[2];
    if (n_bid >= max_blocks) return;
    const float4 k4 = *(constant const float4*) (dead + lane * 4);
    constant const float* q = q_idx + (ulong) qi * QS_IDX_HEADS * QS_IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < QS_IDX_HEADS; ++h) {
        const float4 q4 = *(constant const float4*) (q + h * QS_IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += simd_shuffle_xor(d, (uint) o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % QS_R != 0) score += 1e9f;
        out[(ulong) qi * max_blocks + n_bid] = score;
    }
}

// Block scores with every key block read ONCE for all of a call's queries (nq <= MQ, no active-block count):
// a fixed grid strides over the blocks; per (block, query) the same arithmetic in the same order as
// block_scores_kernel.  The q rows stage in threadgroup memory (16 KB, well under the 32 KB budget).
constant const int QS_MQ = 8;
constant const int QS_MULTI_BLOCKS = 256;    // the launcher's fixed grid (CUDA's gridDim.x)

kernel void block_scores_multi_kernel(constant const float* pooled [[buffer(0)]],
                                      constant const float* dead [[buffer(1)]],
                                      constant const float* q_idx [[buffer(2)]],
                                      constant const int* steps [[buffer(3)]],
                                      constant const int& nq [[buffer(4)]],
                                      constant const long& max_blocks [[buffer(5)]],
                                      device float* out [[buffer(6)]],
                                      uint3 gpos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[QS_MQ * QS_IDX_HEADS * QS_IDX_DIM];
    threadgroup long s_nkv[QS_MQ], s_nbid[QS_MQ];
    for (int i = (int) tid; i < nq * QS_IDX_HEADS * QS_IDX_DIM; i += 256) qs[i] = q_idx[i];
    if (tid < (uint) nq) {
        s_nkv[tid] = (long) steps[(ulong) tid * 4 + 1];
        s_nbid[tid] = (long) steps[(ulong) tid * 4 + 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    const long wstride = (long) QS_MULTI_BLOCKS * QS_SCORE_WARPS;
    for (long b = (long) gpos.x * QS_SCORE_WARPS + (long) sg; b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *(constant const float4*) (pooled + (ulong) b * QS_IDX_DIM + lane * 4);
        const float4 kd = *(constant const float4*) (dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const long n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            const threadgroup float* q = qs + (ulong) qi * QS_IDX_HEADS * QS_IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < QS_IDX_HEADS; ++h) {
                const float4 q4 = *(const threadgroup float4*) (q + h * QS_IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += simd_shuffle_xor(d, (uint) o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % QS_R != 0) score += 1e9f;
                out[(ulong) qi * max_blocks + b] = score;
            }
        }
    }
}

// 4-pass radix select over one query's n_bid + 1 blocks, each weighted by its cell count, the cells emitted
// ascending with ties to the lowest index - the same selection as qsa.metal's topk_kernel, over blocks.
// (CUDA's `weight` lambda is this helper - MSL has no lambdas, as the sampler port recorded.)
static inline int qs_block_weight(long b, long n_bid, long n_kv) {
    return b < n_bid ? QS_R : (int) (n_kv - n_bid * QS_R);
}

kernel void block_topk_kernel(constant const float* scores [[buffer(0)]],
                              constant const int* steps [[buffer(1)]],
                              constant const long& max_blocks [[buffer(2)]],
                              constant const long& cap [[buffer(3)]],
                              device int* ids [[buffer(4)]],
                              uint gpos [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int hist[256];
    threadgroup int s_a[QS_TOPK_T], s_b[QS_TOPK_T];
    threadgroup int s_digit, s_above;
    const long qi = (long) gpos;
    constant const int* st = steps + (ulong) qi * 4;
    const long n_kv = (long) st[1], n_bid = (long) st[2], width = (long) st[3];
    device int* out = ids + (ulong) qi * cap;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (long j = (long) t; j < n_kv; j += QS_TOPK_T) out[j] = (int) j;
        return;
    }
    constant const float* sc = scores + (ulong) qi * max_blocks;
    const long nb = n_bid + 1;                         // blocks 0..n_bid, the last possibly empty
    const long per = (nb + QS_TOPK_T - 1) / QS_TOPK_T;
    const long b0 = (long) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = (int) t; i < 256; i += QS_TOPK_T) atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (long b = b0; b < b1; ++b) {
            const int w = qs_block_weight(b, n_bid, n_kv);
            if (w == 0) continue;
            const uint k = qs_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask))
                atomic_fetch_add_explicit(&hist[(k >> shift) & 255], w, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + atomic_load_explicit(&hist[d], memory_order_relaxed) >= width) break;
                cum += atomic_load_explicit(&hist[d], memory_order_relaxed);
            }
            s_digit = d;
            s_above = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= (uint) s_digit << shift;
        above = s_above;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix;
    const long eq_budget = width - above;              // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (long b = b0; b < b1; ++b) {
        const int w = qs_block_weight(b, n_bid, n_kv);
        if (w == 0) continue;
        const uint k = qs_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < QS_TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const long eq_before = s_b[t];
    long my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s_a[t] = sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < QS_TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long wpos = s_a[t];
    long eq_left = my_eq;
    for (long b = b0; b < b1; ++b) {
        const int w = qs_block_weight(b, n_bid, n_kv);
        if (w == 0) continue;
        const uint k = qs_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int) (b * QS_R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int) (b * QS_R + c);
        }
    }
}

// The same selection as block_topk_kernel with its serial parts made parallel: the threshold digit comes from a
// suffix sum over the 256 digit counts (the largest digit whose cells-at-or-above reach `width`, else 0 - what
// the original's descending walk finds), the per-thread exclusive prefixes from simd prefix sums, and each radix
// pass counts into one histogram per simdgroup with block-strided reads. Every count is an integer and none
// depends on who counts it; the emission keeps the original's contiguous per-thread ranges, so the ids are
// written in the same ascending order to the same positions. Measured on the M2 Max, one query, realistic
// scores (bench/results/2026-10-03-metal-decode-opt2/micro/topk.log): 53-59 -> 27-30 us at 10K cells,
// 106-110 -> 65-68 us at 32K; ties, all-equal, NaN and -0 inputs give the same ids.
kernel void block_topk_scan_kernel(constant const float* scores [[buffer(0)]],
                          constant const int* steps [[buffer(1)]],
                          constant const long& max_blocks [[buffer(2)]],
                          constant const long& cap [[buffer(3)]],
                          device int* ids [[buffer(4)]],
                          uint gpos [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_int hist[8][256];       // one histogram per simdgroup: less atomic contention
    threadgroup int tot[256];
    threadgroup int part[8];
    threadgroup int s_digit;
    threadgroup atomic_int s_max;
    const long qi = (long) gpos;
    constant const int* st = steps + (ulong) qi * 4;
    const long n_kv = (long) st[1], n_bid = (long) st[2], width = (long) st[3];
    device int* out = ids + (ulong) qi * cap;
    if (n_kv <= width) {
        for (long j = (long) t; j < n_kv; j += QS_TOPK_T) out[j] = (int) j;
        return;
    }
    constant const float* sc = scores + (ulong) qi * max_blocks;
    const long nb = n_bid + 1;
    uint prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = (int) t; i < 8 * 256; i += QS_TOPK_T)
            atomic_store_explicit(&hist[i / 256][i % 256], 0, memory_order_relaxed);
        if (t == 0) atomic_store_explicit(&s_max, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (long b = (long) t; b < nb; b += QS_TOPK_T) {      // a count does not depend on who counts
            const int w = qs_block_weight(b, n_bid, n_kv);
            if (w == 0) continue;
            const uint k = qs_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask))
                atomic_fetch_add_explicit(&hist[sg][(k >> shift) & 255], w, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // digit d = 255 - t: its count, and the inclusive sum of the counts of all digits >= d
        const int d = 255 - (int) t;
        int c = 0;
        for (int g = 0; g < 8; ++g) c += atomic_load_explicit(&hist[g][d], memory_order_relaxed);
        tot[d] = c;
        const int incl = simd_prefix_inclusive_sum(c);
        if (lane == 31) part[sg] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int off = 0;
        for (int g = 0; g < (int) sg; ++g) off += part[g];
        const int ge = above + off + incl;                   // cells with key digit >= d (above the prefix)
        // the original walks d = 255 .. 1 and stops at the first d with cum + hist[d] >= width: the largest
        // such d, else 0 - and the cells strictly above it are ge - hist[d]
        if (d > 0 && ge >= width) atomic_fetch_max_explicit(&s_max, d, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int dsel = atomic_load_explicit(&s_max, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // recompute ge for the chosen digit: above + sum of tot[e] for e > dsel
        if (t == 0) s_digit = dsel;
        int gt_part = 0;
        if (d > dsel) gt_part = c;
        const int gsum = simd_sum(gt_part);
        if (lane == 0) part[sg] = gsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int add = 0;
        for (int g = 0; g < 8; ++g) add += part[g];
        prefix |= (uint) s_digit << shift;
        above += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix;
    const long eq_budget = width - above;
    // contiguous ranges for the emission, as in the original (ascending block order across threads)
    const long per = (nb + QS_TOPK_T - 1) / QS_TOPK_T;
    const long b0 = (long) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    int gt = 0, eq = 0;
    for (long b = b0; b < b1; ++b) {
        const int w = qs_block_weight(b, n_bid, n_kv);
        if (w == 0) continue;
        const uint k = qs_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    // exclusive prefix of eq over threads, then of the selected counts
    const int eq_incl = simd_prefix_inclusive_sum(eq);
    if (lane == 31) part[sg] = eq_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int eo = 0;
    for (int g = 0; g < (int) sg; ++g) eo += part[g];
    const long eq_before = eo + eq_incl - eq;
    long my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int sel_incl = simd_prefix_inclusive_sum(sel);
    if (lane == 31) part[sg] = sel_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int so = 0;
    for (int g = 0; g < (int) sg; ++g) so += part[g];
    long wpos = so + sel_incl - sel;
    long eq_left = my_eq;
    for (long b = b0; b < b1; ++b) {
        const int w = qs_block_weight(b, n_bid, n_kv);
        if (w == 0) continue;
        const uint k = qs_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int) (b * QS_R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int) (b * QS_R + c);
        }
    }
    (void) tot;
}
