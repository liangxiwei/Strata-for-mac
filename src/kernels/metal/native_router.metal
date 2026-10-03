// src/kernels/metal/native_router.metal - the port of src/kernels/cuda/native_router.cu's `route` kernel:
// llama.cpp's 32x8-block topk-moe, finite F32, 512 experts, top 10, softmax, no bias, lower normalization
// clamp 2^-14.  The 32x8 block is flattened to a 256-thread group whose first 32 threads are row zero -
// exactly one simdgroup, so the CUDA warp shuffles map 1:1 onto simd_shuffle_xor.
//
// Two deviations from the CUDA text, both no-ops numerically:
//   * the vestigial `__syncthreads()` after the logits load guarded no shared data (values[] are private
//     registers) and MSL barriers must be reached by every thread of the group while rows 1..7 have already
//     returned - dropped;
//   * `-FLT_MAX` is spelled as its literal (MSL headers do not define it); NaN tested via metal::isnan.
#include "strata_port.metalh"

static inline float router_warp_sum(float v) {
    for (uint mask = 16; mask; mask >>= 1) v += simd_shuffle_xor(v, mask);
    return v;
}

static inline float router_warp_max(float v) {
    for (uint mask = 16; mask; mask >>= 1) v = fmax(v, simd_shuffle_xor(v, mask));
    return v;
}

kernel void route(constant const float* logits [[buffer(0)]],
                  device int* ids [[buffer(1)]],
                  device float* weights [[buffer(2)]],
                  uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x = the token
                  uint t [[thread_index_in_threadgroup]]) {
    logits += (ulong) gpos.x * 512;
    ids += (ulong) gpos.x * 10;
    weights += (ulong) gpos.x * 10;
    const uint lane = t & 31u;        // threadIdx.x of the 32x8 block, flattened
    if (t >= 32) return;              // threadIdx.y != 0: only row zero is active

    float values[16];
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + (uint) i * 32];
    float maximum = -INFINITY;
    for (int i = 0; i < 16; ++i) maximum = fmax(maximum, values[i]);
    maximum = router_warp_max(maximum);
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        values[i] = metal::precise::exp(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / router_warp_sum(sum);
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (metal::isnan(values[i])) values[i] = -3.402823466e+38f;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = (int) lane;
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = (int) lane + i * 32; }
        }
        for (uint mask = 16; mask; mask >>= 1) {
            const float other = simd_shuffle_xor(best, mask);
            const int other_id = simd_shuffle_xor(expert, mask);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == (int) lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == (int) lane) selected = best;
    }
    selected_sum = fmax(router_warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
