// src/kernels/metal/native_qsa_indexer.metal - the port of src/kernels/cuda/native_qsa_indexer.cu's kernels
// (K18): llama.cpp qwen4exp's native indexer key path - the F16 raw-key cache, the F32 four-cell pooling,
// the weighted RMSNorm and the text iM-RoPE rotation of the pooled keys, as one single-token append, the
// batched append's three kernels (first cell, completed blocks, tail), all f32 on both backends.
//
// THE TAB TEMPLATE.  The CUDA file compiles <bool TAB> twice and launches <true> only when a rope table is
// registered, because on CUDA a table read merely SKIPPED at run time still changed the compiled default
// path's results.  Metal has one kernel whose table pointers are null on the analytic path - the standing
// native_rope.metal spelling (and the arithmetic of the two CUDA variants is line-identical: the table
// branch only replaces where c and s come from).
//
// Numerical notes (the CUDA original's own pins, kept):
//   * every raw key is rounded through F16 and expanded back to F32 (SET_ROWS/GET_ROWS' storage);
//   * the pooled mean is an F32 sum in cell order (tail[0]+tail[1]+tail[2]+incoming), then SCALE's
//     fma(0.25f, sum, 0.0f) - the zero bias is part of the contract;
//   * the RMS reduction is ggml's XOR butterfly: warp partials through threadgroup memory, warp 0's
//     lane-parallel butterfly on top;
//   * the spare (position 0) keeps its zero angle, and under YaRN the mscale magnitude still rides in
//     through cos(0) - exactly what the queries are scaled by, so the top-k is unmoved;
//   * rsqrtf/powf/cosf/sinf/logf (fast-math on the CUDA build) are metal::precise:: here (R5).
#include "strata_port.metalh"

constant const int NQI_D = 128;        // idx_dim
constant const int NQI_R = 4;          // idx_block
constant const int NQI_ROT = 64;       // n_rot
constant const int NQI_THREADS = 256;  // the CUDA file's THREADS

// mrope.hpp's mrope_pos (as native_rope.metal / rope.metal spell it): text has t = h = w, the table maps
// the cell's position per sector; null table = the position itself
static inline int nqsi_mrope_pos(constant const int* tab, int pos, int pair) {
    return tab != nullptr ? tab[(ulong) pos * 3 + (uint) pair % 3] : pos;
}

// rope_scaling.hpp's rope_yarn_ramp + rope_scaled_angle, transcribed (the analytic path's pinned ggml
// arithmetic - the same spellings native_rope.metal carries)
static inline float nqsi_rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

static inline void nqsi_rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low, float corr_high,
                                          float ext_factor, float mscale_in, int pair, thread float& cos_out,
                                          thread float& sin_out) {
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = nqsi_rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * metal::precise::log(1.0f / freq_scale);
    }
    cos_out = metal::precise::cos(theta) * mscale;
    sin_out = metal::precise::sin(theta) * mscale;
}

// the rotation of one dim of a pooled row: the table when one applies (rope_tab_cs inlined), else the
// analytic angle - `zero_pos` is the spare's position 0, whose angle is zero in every scaling
static inline float nqsi_rotate(threadgroup const float* values, int d, int rope_pos, float theta_scale,
                                float freq_scale, float corr_low, float corr_high, float ext_factor,
                                float mscale, constant const int* mtab, bool zero_pos,
                                constant const float* tab_cos, constant const float* tab_sin, int tab_max_pos) {
    float y = values[d];
    if (d < NQI_ROT) {
        const int pair = d % (NQI_ROT / 2);
        const int p = zero_pos ? 0 : nqsi_mrope_pos(mtab, rope_pos, pair);
        float c, s;
        if (tab_cos != nullptr && p >= 0 && p < tab_max_pos) {
            c = tab_cos[(ulong) p * 32 + (uint) pair];
            s = tab_sin[(ulong) p * 32 + (uint) pair];
        } else {
            const float theta_extrap = (float) p * metal::precise::pow(theta_scale, (float) pair);
            nqsi_rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        const float a = values[pair], z = values[pair + NQI_ROT / 2];
        y = d < NQI_ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}

static inline float nqsi_warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += simd_shuffle_xor(x, (uint) offset);
    return x;
}

// the block-wide sum of squares of the mean: two warp-level butterflies through partials[32] (all 8 warps
// write theirs - warps past D write 0 - so the second butterfly reads only initialized slots)
static inline float nqsi_norm_scale(threadgroup float* partials, float mean, uint d, uint lane, uint sg) {
    float square_sum = d < (uint) NQI_D ? mean * mean : 0.0f;
    square_sum = nqsi_warp_sum(square_sum);
    if (lane == 0) partials[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square_sum = lane < (uint) (NQI_THREADS / 32) ? partials[lane] : 0.0f;
    square_sum = nqsi_warp_sum(square_sum);
    return square_sum;
}

// ================= append (the single-token path) =================
// One block of 256 threads over D = 128 dims.  Stores the raw key (F16-rounded) into its slot; on a block
// completion (slot == R-1, or position 0 for the spare) pools AND ROTATES the pooled row in the same launch,
// so the arguments are identical for every token and a captured graph replays the right position.
kernel void nqsi_append_kernel(constant const float* raw [[buffer(0)]],
                               constant const int* pos_dev [[buffer(1)]],
                               constant const int& pos_base [[buffer(2)]],
                               constant const float* gamma [[buffer(3)]],
                               constant const float& epsilon [[buffer(4)]],
                               device float* tail [[buffer(5)]],
                               device float* dead [[buffer(6)]],
                               device float* pooled [[buffer(7)]],
                               device int* block_pos [[buffer(8)]],
                               constant const int& max_cells [[buffer(9)]],
                               constant const float& theta_scale [[buffer(10)]],
                               constant const float& freq_scale [[buffer(11)]],
                               constant const float& corr_low [[buffer(12)]],
                               constant const float& corr_high [[buffer(13)]],
                               constant const float& ext_factor [[buffer(14)]],
                               constant const float& mscale [[buffer(15)]],
                               constant const int* mtab [[buffer(16)]],
                               constant const float* tab_cos [[buffer(17)]],
                               constant const float* tab_sin [[buffer(18)]],
                               constant const int& tab_max_pos [[buffer(19)]],
                               uint d [[thread_position_in_grid]]) {
    const int pos = pos_dev[0];
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % NQI_R;
    float incoming = 0.0f;
    if (d < (uint) NQI_D) {
        // SET_ROWS stores F16; GET_ROWS expands those exact values to F32
        incoming = f32_from_f16(f16_from_f32(raw[d]));
        if (slot < NQI_R - 1) tail[(ulong) slot * NQI_D + d] = incoming;
    }
    if (pos != 0 && slot != NQI_R - 1) return;
    threadgroup float values[NQI_D];
    threadgroup float partials[32];
    const uint lane = d % 32;
    float mean = 0.0f;
    if (d < (uint) NQI_D) {
        // the spare's four gather indices all name cell zero; completed blocks use chronological slices
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < NQI_R; ++j)
            sum = sum + (pos == 0 || j == NQI_R - 1 ? incoming : tail[(ulong) j * NQI_D + d]);
        mean = fma(0.25f, sum, 0.0f);    // SCALE includes a zero bias
    }
    const float square_sum = nqsi_norm_scale(partials, mean, d, lane, d / 32);
    const float scale = metal::precise::rsqrt(square_sum / (float) NQI_D + epsilon);
    if (d < (uint) NQI_D) values[d] = scale * mean * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint) NQI_D) return;
    const int b = pos / NQI_R;
    const int rope_pos = pos == 0 ? 0 : pos_base + NQI_R * b;
    const float y = nqsi_rotate(values, (int) d, rope_pos, theta_scale, freq_scale, corr_low, corr_high,
                                ext_factor, mscale, mtab, pos == 0, tab_cos, tab_sin, tab_max_pos);
    pooled[(ulong) b * NQI_D + d] = y;
    if (pos == 0) dead[d] = y;
    else pooled[(ulong) (b + 1) * NQI_D + d] = dead[d];
    if (d == 0 && pos != 0) block_pos[0] = rope_pos;
}

// ================= append_first (cell 0 of a sequence) =================
// The spare - every gather index names cell 0 - written to pooled[0] and dead.
kernel void nqsi_append_first_kernel(constant const float* raw [[buffer(0)]],
                                     constant const float* gamma [[buffer(1)]],
                                     constant const float& epsilon [[buffer(2)]],
                                     device float* dead [[buffer(3)]],
                                     device float* pooled [[buffer(4)]],
                                     constant const float& theta_scale [[buffer(5)]],
                                     constant const float& freq_scale [[buffer(6)]],
                                     constant const float& corr_low [[buffer(7)]],
                                     constant const float& corr_high [[buffer(8)]],
                                     constant const float& ext_factor [[buffer(9)]],
                                     constant const float& mscale [[buffer(10)]],
                                     constant const int* mtab [[buffer(11)]],
                                     constant const float* tab_cos [[buffer(12)]],
                                     constant const float* tab_sin [[buffer(13)]],
                                     constant const int& tab_max_pos [[buffer(14)]],
                                     uint d [[thread_position_in_grid]]) {
    threadgroup float values[NQI_D];
    threadgroup float partials[32];
    const uint lane = d % 32;
    float mean = 0.0f;
    if (d < (uint) NQI_D) {
        const float incoming = f32_from_f16(f16_from_f32(raw[d]));
        float sum = incoming;
#pragma unroll
        for (int j = 1; j < NQI_R; ++j) sum = sum + incoming;
        mean = fma(0.25f, sum, 0.0f);
    }
    const float square_sum = nqsi_norm_scale(partials, mean, d, lane, d / 32);
    const float scale = metal::precise::rsqrt(square_sum / (float) NQI_D + epsilon);
    if (d < (uint) NQI_D) values[d] = scale * mean * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint) NQI_D) return;
    const float y = nqsi_rotate(values, (int) d, 0, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
                                mscale, mtab, true, tab_cos, tab_sin, tab_max_pos);
    pooled[d] = y;
    dead[d] = y;
}

// ================= append_blocks (C-2: the batched append's completed blocks) =================
// Each with the single append's arithmetic: keys rounded through F16, summed tail[0]+tail[1]+tail[2]+
// incoming in that order, the RMS norm, gamma, the rotation at pos_base + 4b.  k(j) is the key of the
// block's cell j: a row of this batch (cell >= p0) or the tail the previous batch left (cell < p0).  Only
// the batch's last completed block writes the spare after it (every earlier one's is overwritten by the
// next block in order).
kernel void nqsi_append_blocks_kernel(constant const float* raw [[buffer(0)]],
                                      constant const long& n [[buffer(1)]],
                                      constant const long& p0 [[buffer(2)]],
                                      constant const int& pos_base [[buffer(3)]],
                                      constant const float* gamma [[buffer(4)]],
                                      constant const float& epsilon [[buffer(5)]],
                                      constant const float* tail [[buffer(6)]],
                                      constant const float* dead [[buffer(7)]],
                                      device float* pooled [[buffer(8)]],
                                      device int* block_pos [[buffer(9)]],
                                      constant const long& first_block [[buffer(10)]],
                                      constant const long& last_block [[buffer(11)]],
                                      constant const float& theta_scale [[buffer(12)]],
                                      constant const float& freq_scale [[buffer(13)]],
                                      constant const float& corr_low [[buffer(14)]],
                                      constant const float& corr_high [[buffer(15)]],
                                      constant const float& ext_factor [[buffer(16)]],
                                      constant const float& mscale [[buffer(17)]],
                                      constant const int* mtab [[buffer(18)]],
                                      constant const float* tab_cos [[buffer(19)]],
                                      constant const float* tab_sin [[buffer(20)]],
                                      constant const int& tab_max_pos [[buffer(21)]],
                                      uint3 gpos [[threadgroup_position_in_grid]],
                                      uint d [[thread_index_in_threadgroup]]) {
    const long b = first_block + (long) gpos.x;
    threadgroup float values[NQI_D];
    threadgroup float partials[32];
    const uint lane = d % 32;
    float mean = 0.0f;
    if (d < (uint) NQI_D) {
        // k(j): a row of this batch (cell >= p0) or the tail the previous batch left (cell < p0)
        float sum = b * NQI_R + 0 >= p0
                        ? f32_from_f16(f16_from_f32(raw[(ulong) (b * NQI_R + 0 - p0) * NQI_D + d]))
                        : tail[(ulong) 0 * NQI_D + d];
#pragma unroll
        for (int j = 1; j < NQI_R; ++j) {
            const long cell = b * NQI_R + j;
            const float k = cell >= p0 ? f32_from_f16(f16_from_f32(raw[(ulong) (cell - p0) * NQI_D + d]))
                                       : tail[(ulong) j * NQI_D + d];
            sum = sum + k;
        }
        mean = fma(0.25f, sum, 0.0f);
    }
    const float square_sum = nqsi_norm_scale(partials, mean, d, lane, d / 32);
    const float scale = metal::precise::rsqrt(square_sum / (float) NQI_D + epsilon);
    if (d < (uint) NQI_D) values[d] = scale * mean * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint) NQI_D) return;
    const int rope_pos = pos_base + NQI_R * (int) b;
    pooled[(ulong) b * NQI_D + d] = nqsi_rotate(values, (int) d, rope_pos, theta_scale, freq_scale,
                                                corr_low, corr_high, ext_factor, mscale, mtab, false,
                                                tab_cos, tab_sin, tab_max_pos);
    if (b == last_block) {
        pooled[(ulong) (b + 1) * NQI_D + d] = dead[d];
        if (d == 0) block_pos[0] = rope_pos;
    }
}

// ================= append_tail =================
// The tail after the batch: slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any.
kernel void nqsi_append_tail_kernel(constant const float* raw [[buffer(0)]],
                                    constant const long& n [[buffer(1)]],
                                    constant const long& p0 [[buffer(2)]],
                                    device float* tail [[buffer(3)]],
                                    uint3 gpos [[threadgroup_position_in_grid]],
                                    uint d [[thread_index_in_threadgroup]]) {
    const int s = (int) gpos.x;
    if (d >= (uint) NQI_D) return;
    const long last = p0 + n - 1;
    const long cell = last - ((last % NQI_R) - s + NQI_R) % NQI_R;   // last cell <= last, cell%4==s
    if (cell < p0) return;
    tail[(ulong) s * NQI_D + d] = f32_from_f16(f16_from_f32(raw[(ulong) (cell - p0) * NQI_D + d]));
}
