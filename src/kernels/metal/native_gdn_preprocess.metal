// src/kernels/metal/native_gdn_preprocess.metal - the port of src/kernels/cuda/native_gdn_preprocess.cu's
// five kernels (K10's native sibling, wave 3): the conv+SiLU, the in-place L2, the beta sigmoid, the gate
// softplus, and the closing out-norm of the pinned gated_delta_net path.
//
// Grid mappings (rules 6-7): conv_silu/beta_sigmoid/gate_softplus compute c or i as
// blockIdx*blockDim+threadIdx, so they take the plain `uint [[thread_position_in_grid]]`; l2_norm and
// out_norm use blockIdx.x as the ROW index and a 256-wide block over a 128-wide row, so they read
// `uint3 [[threadgroup_position_in_grid]]` next to SCALAR thread/lane/simd indices (the standing width
// pattern), with the 2-level norm_sum reduction spelled on threadgroup memory exactly as the .cu's
// __shared__ sums does (the .cu declares 32 slots and uses the 8 a 256-thread block has).
//
// Precision seams (the .cu compiles --use_fast_math; the metallib builds -fno-fast-math, rule 5):
//   * expf -> metal::precise::exp; rsqrtf -> metal::precise::rsqrt (~1 ulp, where CUDA's fast rsqrt carries
//     ~2 - both far under the /tmp double-reference probe's tolerance);
//   * log1pf does not exist in MSL: gate_softplus's log1pf(expf(v)) is fused_gdn.metal's measured series
//     seam (below e ~ 0.05 the Horner form carries what f32 log(1+e) cancels away) - fgdn_softplus_f,
//     copied as ngd_softplus;
//   * the .cu's __fadd_rn(sum, 0.0f) / __fmul_rn / __fmaf_rn(...,0.0f) spellings are IEEE-rounded single
//     operations, which -fno-fast-math plain operators and fma() reproduce exactly.
// No fp64 anywhere in the .cu (all-float fast math, like native_gdn.cu) - nothing to emulate.
#include "strata_port.metalh"

constant const int NGP_S = 128;          ///< the .cu's constexpr int S (the norm width)
constant const int NGP_WARPS = 8;        ///< warps in the norms' 256-thread block

static inline float ngd_sigmoid(float value) { return 1.0f / (1.0f + metal::precise::exp(-value)); }

// ggml_compute_softplus_f32: log1p(exp(x)) with the large-x branch; MSL has no log1p, so below e ~ 0.05 the
// Horner'd series carries what f32's log(1+e) would cancel away (fused_gdn.metal's measured seam)
static inline float ngd_softplus(float x) {
    if (x > 20.0f) return x;
    const float e = metal::precise::exp(x);
    if (e < 0.05f) return e * (1.0f - e * (0.5f - e * (1.0f / 3.0f - e * 0.25f)));
    return metal::precise::log(1.0f + e);
}

// the .cu's warp_sum (five xor-shuffle levels over the simdgroup)
static inline float ngp_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += simd_shuffle_xor(value, (uint) offset);
    return value;
}

// the .cu's norm_sum: reduce each warp, park the per-warp totals in threadgroup memory, then reduce the
// first eight of them again - every thread leaves with the block total
static inline float ngp_norm_sum(float value, threadgroup float* sums, uint lane, uint sg) {
    value = ngp_warp_sum(value);
    if (lane == 0u) sums[sg] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = lane < 8u ? sums[lane] : 0.0f;
    return ngp_warp_sum(value);
}

// ---- the four-tap conv with the FP32 SiLU, and the history slide (oldest out, newest into the last tap);
// the .cu's `sum = __fadd_rn(sum, 0.0f)` zero-bias add kept as the IEEE add it spells ----
kernel void conv_silu(device float* history [[buffer(0)]],
                      constant const float* input [[buffer(1)]],
                      constant const float* weights [[buffer(2)]],
                      device float* raw_output [[buffer(3)]],
                      device float* silu_output [[buffer(4)]],
                      constant const int& channels [[buffer(5)]],
                      uint c [[thread_position_in_grid]]) {          // blockIdx*blockDim + threadIdx
    if (c >= (uint) channels) return;
    float values[4] = {history[(ulong) c * 3], history[(ulong) c * 3 + 1], history[(ulong) c * 3 + 2], input[c]};
    float sum = 0.0f;
    for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[(ulong) c * 4 + tap];
    // The native SSM kernel adds its zero bias even when there is no bias input (the .cu's comment).
    sum = sum + 0.0f;
    raw_output[c] = sum;
    silu_output[c] = sum / (1.0f + metal::precise::exp(-sum));
    for (int tap = 0; tap < 3; ++tap) history[(ulong) c * 3 + tap] = values[tap + 1];
}

// ---- in-place row L2 with the .cu's exact rounding chain: rsqrt(mean(x^2) + epsilon) then
// __fmul_rn/__fmaf_rn against scale_after - the FP32 store boundary between RMSNorm and ggml_scale ----
kernel void l2_norm(device float* input [[buffer(0)]],
                    constant const float& epsilon [[buffer(1)]],
                    constant const float& scale_after [[buffer(2)]],
                    uint3 gpos [[threadgroup_position_in_grid]],    // blockIdx.x is the row
                    uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[NGP_WARPS];
    const int col = (int) tid;                       // threadIdx.x; the block is 256 wide over a 128-wide row
    device float* row = input + (ulong) gpos.x * NGP_S;
    const float value = col < NGP_S ? row[col] : 0.0f;
    float partial = 0.0f;
    if (col < NGP_S) partial += value * value;
    partial = ngp_norm_sum(partial, sums, lane, sg);
    const float scale = metal::precise::rsqrt(partial / (float) NGP_S + epsilon);
    if (col < NGP_S) {
        // Preserve the FP32 store boundary between RMSNorm and ggml_scale (the .cu's comment).
        const float normalized = scale * value;
        row[col] = fma(normalized, scale_after, 0.0f);
    }
}

// ---- `beta = sigmoid(beta)`, in place, one flat elementwise pass ----
kernel void beta_sigmoid(device float* beta [[buffer(0)]],
                         constant const int& count [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < (uint) count) beta[i] = ngd_sigmoid(beta[i]);
}

// ---- gate[h] = softplus(alpha[h] + dt[h]) * ssm_a[h]; the >20 branch and the log1p series seam ----
kernel void gate_softplus(constant const float* alpha [[buffer(0)]],
                          constant const float* dt [[buffer(1)]],
                          constant const float* ssm_a [[buffer(2)]],
                          device float* gate [[buffer(3)]],
                          constant const int& count [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint) count) return;
    const float value = alpha[i] + dt[i];
    const float softplus = value > 20.0f ? value : ngd_softplus(value);   // 1 + e^v loses e^v below ~1e-7
    gate[i] = softplus * ssm_a[i];
}

// ---- rms_norm(output, epsilon) * gamma * sigmoid(z): one 256-thread block per head, gamma broadcast ----
kernel void out_norm(constant const float* input [[buffer(0)]],
                     constant const float* z [[buffer(1)]],
                     constant const float* gamma [[buffer(2)]],
                     device float* output [[buffer(3)]],
                     constant const float& epsilon [[buffer(4)]],
                     uint3 gpos [[threadgroup_position_in_grid]],    // blockIdx.x is the head
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[NGP_WARPS];
    const int col = (int) tid;
    const ulong offset = (ulong) gpos.x * NGP_S;
    const float value = col < NGP_S ? input[offset + col] : 0.0f;
    float partial = 0.0f;
    if (col < NGP_S) partial += value * value;
    partial = ngp_norm_sum(partial, sums, lane, sg);
    const float scale = metal::precise::rsqrt(partial / (float) NGP_S + epsilon);
    if (col < NGP_S) {
        // RMSNorm+gamma is one pinned fused operator, followed by sigmoid*mul (the .cu's comment).
        const float weighted = (scale * value) * gamma[col];
        output[offset + col] = weighted * ngd_sigmoid(z[offset + col]);
    }
}
