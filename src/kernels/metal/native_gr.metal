// src/kernels/metal/native_gr.metal - the ports of src/kernels/cuda/native_gr_norm.cu and
// src/kernels/cuda/native_gr_postops.cu (parts of the gr family; their entry points ride gr's parity).
// Pinned llama.cpp arithmetic: the SCALE op keeps its explicit +0 bias, the XOR butterfly repeats in every
// warp, and pre_gated's two modes differ in exactly one FMA-vs-multiply-add (the CUDA file pins both).
#include "strata_port.metalh"

static inline float ngr_sigmoid(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }
// ggml SCALE is scale*x+bias, including its +0 bias: keep it so a compile-time zero cannot change
// signed-zero behaviour
static inline float scale_zero_bias(float x, float scale) { return fma(scale, x, 0.0f); }

static inline float norm_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += simd_shuffle_xor(value, offset);
    return value;
}

// BlockSize arrives as the thread count the launcher chose (256 or 1024, as the CUDA template's two paths)
kernel void weighted_rms_norm(constant const float* input [[buffer(0)]],
                              constant const float* gamma [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant const int& n_cols [[buffer(3)]],
                              constant const float& epsilon [[buffer(4)]],
                              uint gpos [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint tptg [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const ulong row_offset = (ulong) gpos * (ulong) n_cols;
    constant const float* in_r = input + row_offset;
    constant const float* gam_r = gamma + row_offset;
    device float* out_r = output + row_offset;
    float partial = 0.0f;
    for (int col = (int) tid; col < n_cols; col += (int) tptg) {
        const float value = in_r[col];
        partial += value * value;
    }
    threadgroup float sums[32];
    partial = norm_warp_sum(partial);
    if (lane == 0) sums[tid / 32] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial = 0.0f;
    if (lane < tptg / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);

    const float mean = partial / n_cols;
    const float scale = metal::precise::rsqrt(mean + epsilon);
    for (int col = (int) tid; col < n_cols; col += (int) tptg)
        out_r[col] = scale * in_r[col] * gam_r[col];
}

kernel void native_gr_down_silu_kernel(device float* lo [[buffer(0)]],
                                       constant const int& count [[buffer(1)]],
                                       constant const float& scale [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= (uint) count) return;
    const float x = scale_zero_bias(lo[i], scale);
    lo[i] = x / (1.0f + metal::precise::exp(-x));
}

// fused_layer as a runtime flag (the CUDA <bool Fused> template): the two readings differ in exactly one
// FMA-vs-separate-add, pinned by native_gr_postops_parity on CUDA - here both spellings are kept verbatim
kernel void native_gr_pre_gated_kernel(constant const float* xn [[buffer(0)]],
                                       device float* gate [[buffer(1)]],
                                       device float* mixed [[buffer(2)]],
                                       constant const int& n_embd [[buffer(3)]],
                                       constant const int& hc [[buffer(4)]],
                                       constant const float& scale [[buffer(5)]],
                                       constant const int& fused [[buffer(6)]],
                                       uint d_in [[thread_position_in_grid]]) {
    const ulong d = (ulong) d_in;
    if (d >= (ulong) n_embd) return;
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const ulong i = (ulong) c * (ulong) n_embd + d;
        const float x = xn[i], w = ngr_sigmoid(gate[i]);
        const float product = x * w;                       // __fmul_rn under no-contraction
        gate[i] = product;
        if (fused != 0) sum = fma(x, w, sum);              // __fmaf_rn
        else sum = c == 0 ? product : sum + product;       // __fadd_rn
    }
    mixed[d] = fused != 0 ? scale * sum : scale_zero_bias(sum, scale);
}

kernel void native_gr_post_kernel(constant const float* residual [[buffer(0)]],
                                  constant const float* block_out [[buffer(1)]],
                                  constant const float* inject [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant const int& n_embd [[buffer(4)]],
                                  constant const int& hc [[buffer(5)]],
                                  constant const float& scale [[buffer(6)]],
                                  uint i_in [[thread_position_in_grid]]) {
    const ulong i = (ulong) i_in;
    if (i >= (ulong) n_embd * (ulong) hc) return;
    const int c = (int) (i / (ulong) n_embd), d = (int) (i % (ulong) n_embd);
    const float weight = scale_zero_bias(ngr_sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // exact residual/output alias is supported; no other thread reads residual[i]
    output[i] = fma(block_out[d], weight, residual[i]);
}
