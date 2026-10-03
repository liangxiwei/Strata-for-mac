// src/kernels/metal/shared_expert.metal - the port of src/kernels/cuda/shared_expert.cu's kernels (K6).
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T
//     g = sigmoid(x @ gate_inp_shexp)                     one SCALAR per token
//     return h * g[:, None]
//
// Three things are spelled differently by necessity, all following the port's established answers (see
// PROGRESS.md and elementwise.metal's silu):
//   * NO FP64 on this GPU.  swiglu's `x/(1+exp(-x))` in double becomes the same expression in f32 with
//     metal::precise::exp - the CUDA file itself notes its double was matching numpy, not buying accuracy,
//     and elementwise_parity measured the f32 form at 3.4e-08; the scalar gate's and moe_combine's DOUBLE
//     ACCUMULATIONS become KahanSum (Neumaier) f32 sums, the router's measured stand-in (error ~2^-46
//     relative, two orders under every tolerance this file is checked against).
//   * kernel names carry the `shexp_` prefix: `scale_kernel` and `to_f16_kernel` already exist in
//     elementwise.metal with different signatures.
#include "strata_port.metalh"

static inline float shexp_f32_from_bf16(uint h) {
    return as_type<float>((h & 0xffffu) << 16);
}

// silu(x) = x / (1 + exp(-x)) - `ref/moe.py`'s expression, f32 precise (see the file comment).  The
// multiplication by `up` stays at the call site, in the reference's order.
static inline float shexp_silu(float x) {
    return x / (1.0f + metal::precise::exp(-x));
}

// ---- the SwiGLU and the f32->f16 helpers ---------------------------------------------------------------

kernel void shexp_swiglu_kernel(constant const float* gate [[buffer(0)]],
                                constant const float* up [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant const int& n [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= (uint) n) return;
    out[i] = shexp_silu(gate[i]) * up[i];
}

// the pinned-CUDA fp32 form (`__fdividef(g, 1+__expf(-g)) * up`): the same expression, spelled with the
// precise MSL intrinsics this build has (no fast-math spellings exist under -fno-fast-math).
kernel void shexp_native_swiglu_kernel(constant const float* gate [[buffer(0)]],
                                       constant const float* up [[buffer(1)]],
                                       device float* out [[buffer(2)]],
                                       constant const int& n [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= (uint) n) return;
    out[i] = shexp_silu(gate[i]) * up[i];
}

// never launched by the CUDA file either (dead there too); kept so the file is the whole file.
kernel void shexp_to_f16_kernel(constant const float* in [[buffer(0)]],
                                device ushort* out [[buffer(1)]],
                                constant const int& n [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < (uint) n) out[i] = (ushort) f16_from_f32(in[i]);
}

// ---- the per-token scalar gate --------------------------------------------------------------------------
//
// sigmoid(dot(x, w)) with BOTH OPERANDS BF16 (the reference's contract - `gate_inp_shexp` is a BF16 weight,
// so the activation is its bf16 image; the CUDA file's long comment records what reading it as f32 did).
// The CUDA kernel accumulates in DOUBLE; this GPU has none, so each thread keeps a KahanSum (the router's
// measured stand-in for an ascending double sum) and the tree reduces the per-thread VALUES.  The launch is
// <<<1, 256>>> on both backends: 256 is the reduction's width, not the problem's size.

kernel void shexp_scalar_gate_kernel(constant const ushort* x_bf16 [[buffer(0)]],
                                     constant const ushort* w_bf16 [[buffer(1)]],
                                     device float* out [[buffer(2)]],
                                     constant const int& n_embd [[buffer(3)]],
                                     constant const int& block [[buffer(4)]],
                                     uint t [[thread_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];   // 8 warps: the launch is one 256-thread group
    KahanSum ks;
    for (int i = (int) t; i < n_embd; i += block)
        ks.add(shexp_f32_from_bf16(x_bf16[i]) * shexp_f32_from_bf16(w_bf16[i]));
    float acc = ks.value();
    for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    acc = simd_shuffle(acc, 0u);
    if (lane == 0) scratch[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nw = (block + 31) >> 5;
    if (sg == 0) {
        acc = (int) lane < nw ? scratch[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
        acc = simd_shuffle(acc, 0u);
        if (t == 0) out[0] = 1.0f / (1.0f + metal::precise::exp(-acc));
    }
}

kernel void shexp_scale_kernel(device float* out [[buffer(0)]],
                               constant const float* g [[buffer(1)]],
                               constant const int& n [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i < (uint) n) out[i] *= g[0];
}

// the 2D scale (one row per token): group .y is the token, .x*block + t the within-row element
kernel void shexp_scale_rows_kernel(device float* out [[buffer(0)]],
                                    constant const float* g [[buffer(1)]],
                                    constant const int& n [[buffer(2)]],
                                    constant const uint& block [[buffer(3)]],
                                    uint2 gpos [[threadgroup_position_in_grid]],
                                    uint t [[thread_index_in_threadgroup]]) {
    const ulong row = gpos.y;
    const int i = (int) (gpos.x * block + t);
    if (i < n) out[row * (ulong) n + i] *= g[row];
}

// the pinned-CUDA fp32 sigmoid of an already-reduced dot
kernel void shexp_native_scalar_sigmoid_kernel(device float* gate [[buffer(0)]]) {
    gate[0] = 1.0f / (1.0f + metal::precise::exp(-gate[0]));
}

// the same expression, thread t = token t (the <<<1, n_tok>>> launch)
kernel void shexp_native_scalar_sigmoid_multi_kernel(device float* gate [[buffer(0)]],
                                                     uint t [[thread_position_in_grid]]) {
    gate[t] = 1.0f / (1.0f + metal::precise::exp(-gate[t]));
}

// ---- the MoE block's final combination -------------------------------------------------------------------

kernel void shexp_moe_combine_kernel(constant const float* parts [[buffer(0)]],
                                     constant const float* weights [[buffer(1)]],
                                     constant const float* shared_out [[buffer(2)]],
                                     device float* y [[buffer(3)]],
                                     constant const int& n_embd [[buffer(4)]],
                                     constant const int& k [[buffer(5)]],
                                     constant const int& has_shared [[buffer(6)]],
                                     uint j [[thread_position_in_grid]]) {
    if (j >= (uint) n_embd) return;
    // the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`), accumulated compensated
    // where the CUDA file uses double - and the SHARED output is added PLAIN, not router-weighted
    KahanSum acc;
    for (int e = 0; e < k; ++e)
        acc.add(weights[e] * parts[(ulong) e * (ulong) n_embd + j]);
    if (has_shared) acc.add(shared_out[j]);
    y[j] = acc.value();
}
