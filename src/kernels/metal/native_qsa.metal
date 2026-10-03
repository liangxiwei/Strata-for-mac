// src/kernels/metal/native_qsa.metal - the port of src/kernels/cuda/native_qsa.cu's kernels (K18): the
// pinned llama.cpp RMSNorm (ggml norm.cu's XOR reduction) and the QSA output gate (unary.cu's sigmoid
// multiply).  All f32 on both backends - no double sites to emulate.
//
//   * the CUDA template <int BlockSize>'s two widths (256 below n_cols 1024, else 1024) are ONE kernel
//     whose block size arrives as an argument the launcher sets to the width it chose (the standing
//     R7 pattern; the arithmetic is width-independent);
//   * rsqrtf/expf (fast-math on the CUDA build) are metal::precise::rsqrt / metal::precise::exp, the
//     port's R5 substitution everywhere;
//   * the in-place contract (output == input exact) is the CUDA kernel's own: all reads of `input` for
//     the reduction precede the barrier, and afterwards each thread reads and writes only its own
//   * columns - gamma is [n_cols] BROADCAST over rows here (native_gr.metal's weighted_rms_norm strides
//     gamma per row - same kernel shape, different gamma contract; do not merge them).
#include "strata_port.metalh"

// __shfl_xor_sync's butterfly, the CUDA file's warp_sum
static inline float nqs_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += simd_shuffle_xor(value, (uint) offset);
    return value;
}

// ================= norm<BlockSize> =================
// One block per row; every thread's strided columns accumulate x*x, a warp butterfly, the warp partials
// through threadgroup memory, then warp 0's lane-parallel butterfly - ggml's exact reduction ORDER.
kernel void nqs_norm_kernel(constant const float* input [[buffer(0)]],
                            constant const float* gamma [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            constant const int& n_cols [[buffer(3)]],
                            constant const float& epsilon [[buffer(4)]],
                            constant const int& block [[buffer(5)]],     // blockDim, the launcher's width
                            uint3 gpos [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    const ulong row_offset = (ulong) gpos.x * (ulong) n_cols;
    constant const float* in = input + row_offset;
    device float* out = output + row_offset;
    float partial = 0.0f;
    for (ulong col = tid; col < (ulong) n_cols; col += (ulong) block) {
        const float x = in[col];
        partial += x * x;
    }
    threadgroup float sums[32];
    partial = nqs_warp_sum(partial);
    if (lane == 0) sums[sg] = partial;
    // all reads of `input` for the reduction precede this barrier; afterwards each thread touches only
    // its own columns, so exact in-place output == input keeps working (the CUDA file's own contract)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial = lane < (uint) (block / 32) ? sums[lane] : 0.0f;
    partial = nqs_warp_sum(partial);
    const float mean = partial / (float) n_cols;
    const float scale = metal::precise::rsqrt(mean + epsilon);
    for (ulong col = tid; col < (ulong) n_cols; col += (ulong) block)
        out[col] = scale * in[col] * gamma[col];
}

// ================= gate =================
// attn * sigmoid(the SECOND half of each head's 2*head_dim block), one thread per output element.
kernel void nqs_gate_kernel(constant const float* attn [[buffer(0)]],
                            constant const float* q_full [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            constant const int& n_head [[buffer(3)]],
                            constant const int& head_dim [[buffer(4)]],
                            uint i_in [[thread_position_in_grid]]) {
    const ulong i = (ulong) i_in;
    if (i >= (ulong) n_head * (ulong) head_dim) return;
    const ulong head = i / (ulong) head_dim, channel = i % (ulong) head_dim;
    const float raw = q_full[head * 2 * (ulong) head_dim + (ulong) head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-raw));
    output[i] = attn[i] * sigmoid;
}
