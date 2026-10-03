// src/kernels/metal/native_ple_postops.metal - the port of src/kernels/cuda/native_ple_postops.cu's kernels:
// the PLE block's post-projection half, single token and T-token batch, arithmetic adapted from llama.cpp's
// qwen4exp (reduce_rows/sumrows/unary).  The CUDA file's own notes carry over:
//   * the gate's eight partial lanes keep the MATERIALIZED multiply rounding (`__fmul_rn`, never a dot FMA)
//     and fold ascending, then two warp butterflies;
//   * the conv's taps are `__fmul_rn`/`__fadd_rn` chains - explicit there, materialized products here (the
//     build's -ffp-contract=off forbids cross-statement contraction, and the source spellings pin the same);
//   * the batch rms repeats the gamma row every H rows - the ported native_gr `weighted_rms_norm` arithmetic
//     (`partial += value*value` materialized, precise::rsqrt), spelled with the file's own constants.
// All seven kernels carry the `nple_` prefix: ple.metal already ships a `gate_kernel` of its own, and one
// metallib is one namespace.  `expf`/`sqrtf`/`fabsf` are the precise MSL intrinsics (xcrun metal's defaults
// are lossy - the port's standing rule).
#include "strata_port.metalh"

constant const int NPLE_N = 2560;        // N
constant const int NPLE_H = 4;           // H
constant const int NPLE_D = NPLE_N * NPLE_H;      // 10240
constant const int NPLE_HISTORY = 9;     // HISTORY

// silu, `sum / (1 + exp(-sum))` - the CUDA file's own expression
static inline float nple_silu(float x) { return x / (1.0f + metal::precise::exp(-x)); }

// the full-32 xor butterfly (CUDA's warp_sum / norm_warp_sum)
static inline float nple_warp_sum(float v) {
    for (uint off = 16; off; off >>= 1) v += simd_shuffle_xor(v, off);
    return v;
}

// ---- the gate: s[c] = scale * sum_d key[c,d]*query[c,d], then the signed square root and the sigmoid.
// One block per stream (CUDA blockIdx.x is the data index); 512 threads with the eight per-thread partials.
kernel void nple_gate_kernel(constant const float* key [[buffer(0)]],
                             constant const float* query [[buffer(1)]],
                             device float* gate [[buffer(2)]],
                             constant const float& scale [[buffer(3)]],
                             uint3 gpos [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[32];      // 16 warps at 512 threads; only the first 16 slots are written
    const ulong row = (ulong) gpos.x * NPLE_N;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < 8; ++j) {
        const int d = (int) t + j * 512;
        // the MUL is materialized (`__fmul_rn` in the CUDA text - never a dot FMA); -ffp-contract=off keeps it
        const float p = d < NPLE_N ? key[row + (ulong) d] * query[row + (ulong) d] : 0.0f;
        sums[j] += p;
    }
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += sums[j];
    sum = nple_warp_sum(sum);
    if (lane == 0) partials[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = nple_warp_sum(sum);
    if (t == 0) {
        const float s = fma(scale, sum, 0.0f);       // ggml SCALE's zero bias
        const float mag = metal::precise::sqrt(metal::fmax(metal::precise::fabs(s), 1e-6f));
        const float sign = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
        gate[gpos.x] = 1.0f / (1.0f + metal::precise::exp(-(sign * mag)));
    }
}

// ---- gated[i] = value[i % N] * gate[i / N], the value broadcast across the hc streams (flat launch).
kernel void nple_broadcast_kernel(constant const float* value [[buffer(0)]],
                                  constant const float* gate [[buffer(1)]],
                                  device float* gated [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i < (uint) NPLE_D) gated[i] = value[i % (uint) NPLE_N] * gate[i / (uint) NPLE_N];
}

// ---- the single-token dilated conv (taps 0,3,6 of the channel's history plus the new normalized row),
// then SiLU, then result = hidden + gated + conv.  history is channel-major here: [c][HISTORY].
kernel void nple_conv_residual_kernel(constant const float* history [[buffer(0)]],
                                      constant const float* normalized [[buffer(1)]],
                                      constant const ushort* weights [[buffer(2)]],   // f16 taps, k + 4*c
                                      constant const float* hidden [[buffer(3)]],
                                      constant const float* gated [[buffer(4)]],
                                      device float* conv [[buffer(5)]],
                                      device float* result [[buffer(6)]],
                                      uint c [[thread_position_in_grid]]) {
    if (c >= (uint) NPLE_D) return;
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[(ulong) c * NPLE_HISTORY + 3 * k];
        const float w = f32_from_f16(weights[(ulong) c * 4 + (ulong) k]);
        const float term = x * w;                    // __fmul_rn
        sum = k == 0 ? term : sum + term;            // __fadd_rn, the CUDA file's own chain
    }
    const float activation = nple_silu(sum);
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = hidden[c] + (gated[c] + activation);
}

// ---- the batch rms: weighted_rms_norm's arithmetic with the gamma row repeating every H rows (one token's
// H groups).  One block per row, 1024 threads (the CUDA file's own BlockSize for 2560 columns); in-place
// input==output is safe - loop 2 reads each element before the same thread writes it, as in the original.
kernel void nple_rms_rep_kernel(device const float* input [[buffer(0)]],
                                constant const float* gamma [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                constant const float& eps [[buffer(3)]],
                                uint3 gpos [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];          // 1024 threads = exactly 32 warps
    const ulong row = (ulong) gpos.x * NPLE_N;
    device const float* in = input + row;
    device float* out = output + row;
    constant const float* gam = gamma + (ulong) (gpos.x % (uint) NPLE_H) * NPLE_N;
    float partial = 0.0f;
    for (int col = (int) t; col < NPLE_N; col += 1024) {
        const float value = in[col];
        partial += value * value;                    // materialized product, the weighted_rms_norm precedent
    }
    partial = nple_warp_sum(partial);
    if (lane == 0) sums[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial = 0.0f;
    if (lane < 32) partial = sums[lane];             // BlockSize / 32 == 32
    partial = nple_warp_sum(partial);
    const float mean = partial / (float) NPLE_N;
    const float scale = metal::precise::rsqrt(mean + eps);
    for (int col = (int) t; col < NPLE_N; col += 1024) out[col] = scale * in[col] * gam[col];
}

// ---- the batch broadcast, element for element the single kernel's arithmetic (flat launch).
kernel void nple_broadcast_batch_kernel(constant const float* value [[buffer(0)]],
                                        constant const float* gate [[buffer(1)]],
                                        device float* gated [[buffer(2)]],
                                        constant const int& T [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    const ulong I = (ulong) i;
    if (I >= (ulong) T * (ulong) NPLE_D) return;
    const ulong t = I / (ulong) NPLE_D, d = I % (ulong) NPLE_D;
    gated[I] = value[t * (ulong) NPLE_N + d % (ulong) NPLE_N] * gate[t * (ulong) NPLE_H + d / (ulong) NPLE_N];
}

// ---- the batch dilated conv (taps 9, 6, 3 tokens back and this one; a tap before the chunk reads the
// history) and the residual.  hidden is updated in place, each thread owning one element.
kernel void nple_conv_residual_batch_kernel(constant const float* history [[buffer(0)]],
                                            constant const float* normalized [[buffer(1)]],
                                            constant const ushort* weights [[buffer(2)]],
                                            device float* hidden [[buffer(3)]],
                                            constant const float* gated [[buffer(4)]],
                                            constant const int& T [[buffer(5)]],
                                            uint i [[thread_position_in_grid]]) {
    const ulong I = (ulong) i;
    if (I >= (ulong) T * (ulong) NPLE_D) return;
    const uint t = (uint) (I / (ulong) NPLE_D), c = (uint) (I % (ulong) NPLE_D);
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const int p = (int) t - 9 + 3 * k;           // the token this tap reads (k == 3: this one)
        const float x = p >= 0 ? normalized[(ulong) p * NPLE_D + c]
                               : history[(ulong) c * NPLE_HISTORY + (9 + p)];
        const float wk = f32_from_f16(weights[(ulong) c * 4 + (ulong) k]);
        const float term = x * wk;
        sum = k == 0 ? term : sum + term;
    }
    const float activation = nple_silu(sum);
    hidden[I] = hidden[I] + (gated[I] + activation);
}

// ---- the history after the chunk: the last nine normalized rows (older ones from the history when T < 9).
kernel void nple_history_batch_kernel(device float* history [[buffer(0)]],
                                      constant const float* normalized [[buffer(1)]],
                                      constant const int& T [[buffer(2)]],
                                      uint c [[thread_position_in_grid]]) {
    if (c >= (uint) NPLE_D) return;
    float h[NPLE_HISTORY];
    for (int r = 0; r < NPLE_HISTORY; ++r) {
        const int p = T - NPLE_HISTORY + r;
        h[r] = p >= 0 ? normalized[(ulong) p * NPLE_D + c] : history[(ulong) c * NPLE_HISTORY + (T + r)];
    }
    for (int r = 0; r < NPLE_HISTORY; ++r) history[(ulong) c * NPLE_HISTORY + r] = h[r];
}
