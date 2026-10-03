// src/kernels/metal/gr.metal - the port of src/kernels/cuda/gr.cu's kernels (K9).  The Activation template
// (float vs bf16 storage) becomes a runtime flag with both pointers passed (the inactive one may be null);
// every reduction, warp split and expression order is the CUDA file's, including the "no FP64 on decode
// paths" rule that file states - nothing here needs the double emulation at all.
#include "strata_port.metalh"

constant const int GR_THREADS = 256;
constant const int GR_WARPS = 8;

static inline float f32_from_bf16g(uint h) { return as_type<float>((h & 0xffffu) << 16); }
static inline float silu_f(float x) { return x / (1.0f + metal::precise::exp(-x)); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// activation value: f32 storage (act32) or bf16 storage
static inline float activation_val(int act32, constant const float* xf, constant const uint* xh, long i) {
    return act32 != 0 ? xf[i] : f32_from_bf16g(reinterpret_cast<constant const uint16_t*>(xh)[i]);
}

static inline float warp_sumg(float v) {
    for (int off = 16; off > 0; off >>= 1) v += simd_shuffle_down(v, off);
    return simd_shuffle(v, 0u);
}

// ---- gr_norm: hc groups, one per residual stream ----
kernel void gr_norm_kernel(constant const float* R [[buffer(0)]],
                           constant const float* w_norm [[buffer(1)]],
                           constant const float& eps [[buffer(2)]],
                           constant const int& n_embd [[buffer(3)]],
                           device float* xn [[buffer(4)]],
                           device uint* xq [[buffer(5)]],                  // bf16 patterns; unused when fp32
                           constant const int& act_is_f32 [[buffer(6)]],
                           uint3 gpos [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           constant const uint& block [[buffer(7)]],   // arg 8 of 8: index must equal position
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const int c = (int) gpos.x;
    constant const float* Rc = R + (ulong) c * n_embd;
    device float* xnc = xn + (ulong) c * n_embd;
    device uint16_t* xqc = reinterpret_cast<device uint16_t*>(xq) + (ulong) c * n_embd;

    float ss = 0.0f;
    for (int d = (int) tid; d < n_embd; d += (int) block) {
        const float v = Rc[d];
        ss += v * v;
    }
    // block_sumf: simdgroup trees, meet in shared, one broadcast
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = warp_sumg(ss);
    if (lane == 0) scratch[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nw = block / 32;
    float v = tid < nw ? scratch[tid] : 0.0f;
    if (sg == 0) v = warp_sumg(v);
    if (tid == 0) scratch[0] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ms = scratch[0] / (float) n_embd;
    const float rs = metal::precise::rsqrt(ms + eps);
    for (int d = (int) tid; d < n_embd; d += (int) block) {
        const float x = Rc[d] * rs * w_norm[(ulong) c * n_embd + d];
        xnc[d] = x;
        if (act_is_f32 == 0) xqc[d] = (uint16_t) bf16_from_f32(x);
    }
}

// ---- gr_down: ONE WARP per output row, the block's 8 warps splitting the reduction ----
kernel void gr_down_kernel(constant const float* xf [[buffer(0)]],
                           constant const uint* xh [[buffer(1)]],          // bf16 when xf unused
                           constant const uint* w_down [[buffer(2)]],      // bf16 patterns
                           constant const int& hc_dim [[buffer(3)]],
                           constant const int& hc_lr [[buffer(4)]],
                           constant const int& hc [[buffer(5)]],
                           device float* lo [[buffer(6)]],
                           device uint* lq [[buffer(7)]],                   // bf16 out when fp32 off
                           constant const int& act_is_f32 [[buffer(8)]],
                           constant const uint& block [[buffer(9)]],
                           uint3 gpos [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[GR_WARPS];
    const int k = (int) gpos.x;
    if (k >= hc_lr) return;
    const uint nw = block / 32;
    constant const uint16_t* row = reinterpret_cast<constant const uint16_t*>(w_down) + (ulong) k * hc_dim;
    float acc = 0.0f;
    for (int i = (int) sg * 32 + (int) lane; i < hc_dim; i += (int) nw * 32)
        acc += activation_val(act_is_f32, xf, xh, i) * f32_from_bf16g(row[i]);
    acc = warp_sumg(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < nw ? part[lane] : 0.0f;
        t = warp_sumg(t);
        if (lane == 0) {
            const float v = silu_f(t / (float) hc);
            if (act_is_f32 != 0) lo[k] = v;
            else reinterpret_cast<device uint16_t*>(lq)[k] = (uint16_t) bf16_from_f32(v);
        }
    }
}

// ---- gr_gate: one warp per output row, lanes striding hc_lr ----
kernel void gr_gate_kernel(constant const float* xf [[buffer(0)]],
                           constant const uint* xh [[buffer(1)]],
                           constant const uint* w_up [[buffer(2)]],
                           constant const float* xn [[buffer(3)]],
                           constant const int& hc_dim [[buffer(4)]],
                           constant const int& hc_lr [[buffer(5)]],
                           device float* gated [[buffer(6)]],
                           constant const int& act_is_f32 [[buffer(7)]],
                           uint3 gpos [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    const int i = (int) gpos.x * GR_WARPS + (int) sg;      // blockIdx*WARPS + warp
    if (i >= hc_dim) return;
    constant const uint16_t* row = reinterpret_cast<constant const uint16_t*>(w_up) + (ulong) i * hc_lr;
    float acc = 0.0f;
    for (int k = (int) lane; k < hc_lr; k += 32)
        acc += activation_val(act_is_f32, xf, xh, k) * f32_from_bf16g(row[k]);
    acc = warp_sumg(acc);
    if (lane == 0) gated[i] = xn[i] * sigmoid_f(acc);
}

// ---- gr_mean: flat elementwise ----
kernel void gr_mean_kernel(constant const float* gated [[buffer(0)]],
                           constant const int& n_embd [[buffer(1)]],
                           constant const int& hc [[buffer(2)]],
                           device float* mixed [[buffer(3)]],
                           uint d [[thread_position_in_grid]]) {
    if (d >= (uint) n_embd) return;
    float m = 0.0f;
    for (int c = 0; c < hc; ++c) m += gated[(ulong) c * n_embd + d];
    mixed[d] = m / (float) hc;
}

// ---- gr_inject: one block of 32*hc threads, one warp per stream ----
kernel void gr_inject_kernel(constant const float* xf [[buffer(0)]],
                             constant const uint* xh [[buffer(1)]],
                             constant const uint* w_inject [[buffer(2)]],
                             constant const int& hc_dim [[buffer(3)]],
                             constant const int& hc [[buffer(4)]],
                             device float* inject [[buffer(5)]],
                             constant const int& act_is_f32 [[buffer(6)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const int c = (int) (tid >> 5);
    if (c >= hc) return;
    constant const uint16_t* row = reinterpret_cast<constant const uint16_t*>(w_inject) + (ulong) c * hc_dim;
    float acc = 0.0f;
    for (int i = (int) lane; i < hc_dim; i += 32)
        acc += activation_val(act_is_f32, xf, xh, i) * f32_from_bf16g(row[i]);
    acc = warp_sumg(acc);
    if (lane == 0) inject[c] = acc;
}

// ---- gr_write: the per-stream weights in dynamic threadgroup memory ----
kernel void gr_write_kernel(constant const float* R [[buffer(0)]],
                            constant const float* block_out [[buffer(1)]],
                            constant const float* inject [[buffer(2)]],
                            constant const int& n_embd [[buffer(3)]],
                            constant const int& hc [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            threadgroup uint8_t* smem_raw [[threadgroup(0)]],
                            constant const long& stride_in [[buffer(6)]],   // blocks*THREADS, from the host
                            uint i0 [[thread_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float* w = reinterpret_cast<threadgroup float*>(smem_raw);
    if (tid < (uint) hc) w[tid] = 2.0f * sigmoid_f(inject[tid] / (float) hc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const long n = (long) hc * n_embd;
    for (long i = (long) i0; i < n; i += stride_in) {
        const int c = (int) (i / n_embd), d = (int) (i % n_embd);
        // every stream adds the SAME block output; only the weight differs per stream
        out[i] = R[i] + block_out[d] * w[c];
    }
}
