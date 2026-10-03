// src/kernels/metal/gdn.metal - the port of src/kernels/cuda/gdn.cu's kernels (K10): the gated delta-net's
// non-projection parts.  The recurrence is one thread per (h, j) column of the state, exactly as the CUDA
// file spells it - no barrier anywhere in the recurrence, because every line touches only one column.
//
// MAX_H stays 1 (the CUDA file's measured latency-hiding knob: 192 blocks x 32 threads put 4 warps on every
// SM where 24 blocks left 1 warp on half of them); the arithmetic does not change with it, and neither does
// the port.  The staged ks/qs rows are threadgroup memory like the CUDA __shared__ ones - ~1 KB, well under
// this GPU's 32768-byte threadgroup limit.
//
// The two norms accumulate in DOUBLE in the CUDA file (the reference `ref/gdn.py` computes them in float64),
// and this GPU has no fp64 - the port emulates the double products and sums the way strata_port.metalh does
// for quantize_act's divide and the router's sums: a product of two f32 is EXACT in double (48 significand
// bits), so the exact hi/lo pair (the fma residual does not round) rides a Neumaier-compensated sum, and the
// warp reduction combines the compensated partials so the 5 tree levels round nothing the double original
// rounded.  Measured error budget: the f32 ssum and the rsqrt each round once (~6e-8 + ~1.2e-7 relative)
// against the test's 1e-6 tolerance.
//
// The CUDA file's softplus_f (log1pf(expf(x))) is dead code there - only sigmoid_f is used, by the closing
// norm - so it is not carried; the softplus seam this port uses where it IS live is fused_gdn.metal's
// fgdn_softplus_f (elementwise.metal's measured series form).
#include "strata_port.metalh"

constant const int GDN_JTHREADS = 32;   ///< threads along j, the state's fast axis
constant const int GDN_MAX_H = 1;       ///< heads staged per block (the .cu's knob; see its comment)
constant const int GDN_STAGED = 128;    ///< the ks/qs stage width, the .cu's __shared__ [MAX_H][128]

static inline float gdn_sigmoid_f(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// the CUDA file's `acc += (double) p[i] * (double) p[i]`: the exact product as its hi/lo pair (fma's
// residual does not round), added into the compensated sum - the pair IS the double product
static inline void gdn_add_sq(thread KahanSum& acc, float x) {
    const float hi = x * x;
    acc.add(hi);
    acc.add(fma(x, x, -hi));
}

// one Neumaier add, spelled on two scalars so the warp reduction below can keep its pairs compensated
static inline void gdn_neumaier_add(thread float& sum, thread float& c, float x) {
    const float t = sum + x;
    if (metal::precise::fabs(sum) >= metal::precise::fabs(x)) c += (sum - t) + x;
    else c += (x - t) + sum;
    sum = t;
}

// the CUDA file's __shfl_down_sync reduction of the per-thread double partials; both components of each
// partial shuffle down and combine compensated, so the tree adds nothing the double chain rounded exactly
static inline float gdn_warp_total(float sum, float c) {
    for (int off = 16; off > 0; off >>= 1) {
        const float s2 = simd_shuffle_down(sum, (uint) off);
        const float c2 = simd_shuffle_down(c, (uint) off);
        gdn_neumaier_add(sum, c, s2);
        gdn_neumaier_add(sum, c, c2);
    }
    return sum + c;
}

// ---- the recurrence: one thread per (h, j) column, q/k staged per block, decay BEFORE the update ----
kernel void gdn_step_kernel(device float* state [[buffer(0)]],
                            constant const float* q [[buffer(1)]],
                            constant const float* k [[buffer(2)]],
                            constant const float* v [[buffer(3)]],
                            constant const float* gate [[buffer(4)]],
                            constant const float* beta [[buffer(5)]],
                            device float* o [[buffer(6)]],
                            constant const int& S [[buffer(7)]],
                            constant const int& h_k [[buffer(8)]],
                            constant const int& h_v [[buffer(9)]],
                            uint3 gpos [[threadgroup_position_in_grid]],   // x: the j tile, y: the head tile
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float ks[GDN_MAX_H][GDN_STAGED];
    threadgroup float qs[GDN_MAX_H][GDN_STAGED];
    threadgroup float dec_s[GDN_MAX_H];
    threadgroup float beta_s[GDN_MAX_H];

    const int h0 = (int) gpos.y * GDN_MAX_H;
    const int nh = GDN_MAX_H < h_v - h0 ? GDN_MAX_H : h_v - h0;
    const int j = (int) gpos.x * GDN_JTHREADS + (int) tid;

    // stage this block's q/k rows and its per-head scalars
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const int src = h % h_k;                       // MODULO head pairing
        for (int i = (int) tid; i < S; i += GDN_JTHREADS) {
            ks[hi][i] = k[(ulong) src * S + i];
            qs[hi][i] = q[(ulong) src * S + i];
        }
        if (tid == 0u) {
            dec_s[hi] = metal::precise::exp(gate[h]);
            beta_s[hi] = beta[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j >= S) return;

    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const float dec = dec_s[hi], b = beta_s[hi];
        device float* col = state + (ulong) h * S + j;      // (S, h_v, S) with j fastest
        const ulong stride = (ulong) h_v * S;

        // pass 1: decay the state and contract it against k.  Decay BEFORE the update - PROPERTY 5.
        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(ulong) i * stride] * dec;
            col[(ulong) i * stride] = s;
            sk += s * ks[hi][i];
        }
        const float d = (v[(ulong) h * S + j] - sk) * b;
        // pass 2: the rank-1 update, then read out against q using the UPDATED state
        float dot = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(ulong) i * stride] + ks[hi][i] * d;
            col[(ulong) i * stride] = s;
            dot += s * qs[hi][i];
        }
        o[(ulong) h * S + j] = dot;
    }
}

// ---- `ggml_ssm_conv`: out[c] = sum_i inp[i][c] * kW[c*d_conv + i], and the state slides ----
kernel void gdn_conv_kernel(device float* cs [[buffer(0)]],
                            constant const float* x [[buffer(1)]],
                            constant const float* kW [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant const int& C [[buffer(4)]],
                            constant const int& dc [[buffer(5)]],
                            uint c [[thread_position_in_grid]]) {         // blockIdx*blockDim + threadIdx
    if (c >= (uint) C) return;
    device float* st = cs + (ulong) c * (dc - 1);
    constant const float* w = kW + (ulong) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
    acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop the oldest, append the newest
    st[dc - 2] = x[c];
}

// ---- `build_gdn_l2_norm`, one warp per row: x / sqrt(sum(x^2) + eps) over the last axis.  The `+ eps` is a
// floor on the SQUARED NORM (no division by the width), and the double accumulation is the compensated pair
// (see the file comment). ----
kernel void gdn_l2_kernel(device float* x [[buffer(0)]],
                          constant const int& cols [[buffer(1)]],
                          constant const float& eps [[buffer(2)]],
                          uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x is the row
                          uint lane [[thread_index_in_simdgroup]]) {
    device float* p = x + (ulong) gpos.x * cols;
    KahanSum acc;
    for (int i = (int) lane; i < cols; i += 32) gdn_add_sq(acc, p[i]);
    const float total = gdn_warp_total(acc.sum, acc.c);
    threadgroup float ssum[1];
    if (lane == 0u) ssum[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(ssum[0] + eps);
    for (int i = (int) lane; i < cols; i += 32) p[i] *= inv;
}

// ---- `beta = sigmoid(beta)`, in place, over the h_v per-head scalars (flat elementwise) ----
kernel void gdn_beta_gate_kernel(device float* beta [[buffer(0)]],
                                 constant const int& n [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i < (uint) n) beta[i] = gdn_sigmoid_f(beta[i]);
}

// ---- y = rms_norm(o, eps) * ssm_norm * sigmoid(z), one warp per head, norm over that head's S values ----
kernel void gdn_out_norm_kernel(constant const float* o [[buffer(0)]],
                                constant const float* z [[buffer(1)]],
                                constant const float* ssm_norm [[buffer(2)]],
                                device float* y [[buffer(3)]],
                                constant const int& S [[buffer(4)]],
                                constant const float& eps [[buffer(5)]],
                                uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x is the head
                                uint lane [[thread_index_in_simdgroup]]) {
    const int h = (int) gpos.x;
    constant const float* po = o + (ulong) h * S;
    constant const float* pz = z + (ulong) h * S;
    device float* py = y + (ulong) h * S;
    KahanSum acc;
    for (int i = (int) lane; i < S; i += 32) gdn_add_sq(acc, po[i]);
    const float total = gdn_warp_total(acc.sum, acc.c);
    threadgroup float ssum[1];
    if (lane == 0u) ssum[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(ssum[0] / (float) S + eps);
    for (int i = (int) lane; i < S; i += 32)
        py[i] = po[i] * inv * ssm_norm[i] * gdn_sigmoid_f(pz[i]);
}
