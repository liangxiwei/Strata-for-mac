// src/kernels/metal/bf16_gemv.metal - the ports of src/kernels/cuda/bf16_gemv.cu (K17) and the two
// fp32-MMVF kernels of src/kernels/cuda/native_bf16.cu (their entry points ride gr's family).
//
// MSL attribute discipline (learned the hard way, see PROGRESS.md): no thread_position_in_grid and no
// threads_per_threadgroup in these signatures - the group position is uint3, the thread index scalar, and
// the block size arrives as a plain argument the launcher already knows.
#include "strata_port.metalh"

static inline float f32_from_bf16(uint h) {
    return as_type<float>((h & 0xffffu) << 16);
}

// ---- bf16_gemv.cu ----

kernel void bf16_gemv_naive_kernel(constant const uint* x [[buffer(0)]],        // bf16 patterns
                                   constant const uint* w [[buffer(1)]],
                                   device float* y [[buffer(2)]],
                                   constant const long& n_in [[buffer(3)]],
                                   constant const long& n_out [[buffer(4)]],
                                   uint o [[thread_position_in_grid]]) {
    if (o >= (uint) n_out) return;
    constant const uint16_t* row16 = reinterpret_cast<constant const uint16_t*>(w) + (ulong) o * (ulong) n_in;
    constant const uint16_t* x16 = reinterpret_cast<constant const uint16_t*>(x);
    float acc = 0.0f;
    for (long i = 0; i < n_in; ++i)
        acc += f32_from_bf16(x16[i]) * f32_from_bf16(row16[i]);
    y[o] = acc;
}

kernel void bf16_gemv_warp_kernel(constant const uint* x [[buffer(0)]],
                                  constant const uint* w [[buffer(1)]],
                                  device float* y [[buffer(2)]],
                                  constant const long& n_in [[buffer(3)]],
                                  constant const long& n_out [[buffer(4)]],
                                  constant const uint& block [[buffer(5)]],
                                  uint3 gpos [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint warps_per_block = block / 32u;
    const long o = (long) gpos.x * (long) warps_per_block + sg;      // blockIdx*wpb + warp
    if (o >= n_out) return;
    constant const uint16_t* row16 = reinterpret_cast<constant const uint16_t*>(w) + (ulong) o * (ulong) n_in;
    constant const uint16_t* x16 = reinterpret_cast<constant const uint16_t*>(x);
    float acc = 0.0f;
    for (long i = lane; i < n_in; i += 32)
        acc += f32_from_bf16(x16[i]) * f32_from_bf16(row16[i]);
    for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    if (lane == 0) y[o] = acc;
}

kernel void bf16_gemv_split_kernel(constant const uint* x [[buffer(0)]],
                                   constant const uint* w [[buffer(1)]],
                                   device float* y [[buffer(2)]],
                                   constant const long& n_in [[buffer(3)]],
                                   constant const long& n_out [[buffer(4)]],
                                   constant const int& tpr [[buffer(5)]],
                                   threadgroup float* scratch [[threadgroup(0)]],
                                   uint3 gpos [[threadgroup_position_in_grid]],   // (row, 1, 1)
                                   uint t [[thread_index_in_threadgroup]]) {
    const long o = (long) gpos.x;
    if (o >= n_out) return;
    constant const uint16_t* row16 = reinterpret_cast<constant const uint16_t*>(w) + (ulong) o * (ulong) n_in;
    constant const uint16_t* x16 = reinterpret_cast<constant const uint16_t*>(x);
    float acc = 0.0f;
    for (long i = t; i < n_in; i += tpr)
        acc += f32_from_bf16(x16[i]) * f32_from_bf16(row16[i]);
    scratch[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) y[o] = scratch[0];
}

// ---- native_bf16.cu's fp32 MMVF ----

static inline float mmvf_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += simd_shuffle_xor(value, offset);
    return value;
}

kernel void bf16_f32_mmvf_kernel(constant const float* x [[buffer(0)]],
                                 constant const uint* w [[buffer(1)]],          // bf16 pairs as uint
                                 device float* y [[buffer(2)]],
                                 constant const int& n_in [[buffer(3)]],
                                 constant const uint& block [[buffer(4)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],   // one row per group
                                 uint t [[thread_index_in_threadgroup]]) {
    // the row strides in uint16 ELEMENTS first, then widens to uint pairs - casting w to uint* BEFORE the
    // stride walks 4-byte units and lands every row on row*2's weights (row 0 alone looked right)
    constant const uint* weights2 = reinterpret_cast<constant const uint*>(
        reinterpret_cast<constant const uint16_t*>(w) + (ulong) gpos.x * (ulong) n_in);
    constant const float2* inputs2 = reinterpret_cast<constant const float2*>(x);
    threadgroup float partials[32];
    if (block > 32) {
        if (t < 32) partials[t] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float acc = 0.0f;
    for (int pair = (int) t; pair < n_in / 2; pair += (int) block) {
        const uint weight = weights2[pair];
        const float2 input = inputs2[pair];
        // ggml_cuda_mad's two ORDERED multiply-adds, not a pair sum
        acc = fma(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = fma(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if (block > 32) {
        if ((t & 31) == 0) partials[t / 32] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[gpos.x] = acc;
}

// up to NTMAX activation rows; every output is bit-identical to a single-row launch
kernel void bf16_f32_mmvf_multi_kernel(constant const float* x [[buffer(0)]],
                                       constant const long& ldx [[buffer(1)]],
                                       constant const uint* w [[buffer(2)]],
                                       device float* y [[buffer(3)]],
                                       constant const long& ldy [[buffer(4)]],
                                       constant const int& n_in [[buffer(5)]],
                                       constant const int& n_tok_in [[buffer(6)]],
                                       constant const int& ntmax [[buffer(7)]],        // 4 or 8
                                       constant const uint& block [[buffer(8)]],
                                       threadgroup float* partials [[threadgroup(0)]],  // [NTMAX][32]
                                       uint3 gpos [[threadgroup_position_in_grid]],
                                       uint t [[thread_index_in_threadgroup]]) {
    // uint16-element stride first, then the uint-pair view (the single kernel's comment has the story)
    constant const uint* weights2 = reinterpret_cast<constant const uint*>(
        reinterpret_cast<constant const uint16_t*>(w) + (ulong) gpos.x * (ulong) n_in);
    if (block > 32 && t < 32)
        for (int k = 0; k < ntmax; ++k) partials[(ulong) k * 32 + t] = 0.0f;
    if (block > 32) threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[8];
    for (int k = 0; k < 8; ++k) acc[k] = 0.0f;
    for (int pair = (int) t; pair < n_in / 2; pair += (int) block) {
        const uint weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
        for (int k = 0; k < ntmax; ++k) {
            if (k < n_tok_in) {
                const float2 input = reinterpret_cast<constant const float2*>(x + (ulong) k * (ulong) ldx)[pair];
                acc[k] = fma(w0, input.x, acc[k]);
                acc[k] = fma(w1, input.y, acc[k]);
            }
        }
    }
    for (int k = 0; k < ntmax; ++k) acc[k] = mmvf_warp_sum(acc[k]);
    if (block > 32) {
        if ((t & 31) == 0)
            for (int k = 0; k < ntmax; ++k) partials[(ulong) k * 32 + t / 32] = acc[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 32)
            for (int k = 0; k < ntmax; ++k) acc[k] = mmvf_warp_sum(partials[(ulong) k * 32 + t]);
    }
    if (t == 0)
        for (int k = 0; k < ntmax; ++k)
            if (k < n_tok_in) y[(ulong) k * (ulong) ldy + gpos.x] = acc[k];
}
