// src/kernels/metal/ple.metal - the port of src/kernels/cuda/ple.cu's kernels (K20): the PLE block's GPU
// half, one token at a time.
//
// The CUDA file's arithmetic notes all carry over verbatim (its own comments are the record; the short
// version): gnorm's `(double)(x*x)` widens an f32-ROUNDED product, the gate is a signed square root then a
// sigmoid, the conv's taps read a ROW-FASTEST history with GGML-NATIVE `kW[k + kern*c]` weights, and the
// legacy value projection rounds the activation to BF16 before the dot.  This GPU has no fp64, so the three
// double accumulations (gnorm's sum of squares, the gate's key.query dot, the legacy bf16_gemv's dot) run as
// Neumaier-compensated f32 sums - the router's measured stand-in (round 3): where the products are already
// exact in f32 (bf16xbf16, and every f32 square) only the summation was double's job, and the compensation
// holds that error at ~2^-46 relative, two orders under the parity tolerances the ple_parity stages use.
//
// ONE RENAMED KERNEL: the CUDA file's private `to_bf16_kernel` is `ple_to_bf16_kernel` here because
// elementwise.metal already ships a `to_bf16_kernel` (one metallib, one namespace).  (The private port of
// s2_gemv_q8.cu's kernel this file once carried is gone - the canonical Q2_0 key projection calls the real
// `s2_gemv_q8()` since that file's port landed.)
#include "strata_port.metalh"

constant const int PLE_THREADS = 256;
constant const int PLE_WARPS = 8;
constant const int PLE_N_EMBD = 2560;              // NG_N_EMBD
constant const int PLE_HC = 4;                     // NG_HC
constant const int PLE_HC_DIM = PLE_N_EMBD * PLE_HC;    // 10240
constant const int PLE_HIST = 9;                   // NG_HIST

// silu in f32, `ggml_silu_f32`'s own expression (precise exp: xcrun metal defaults to lossy fast math)
static inline float ple_silu(float x) { return x / (1.0f + metal::precise::exp(-x)); }

// bf16 bits -> f32: the exact widening (high half relocated), ple.cu's bf16_float
static inline float ple_f32_from_bf16(uint h) { return as_type<float>((h & 0xffffu) << 16); }

// the CUDA file's __shfl_down_sync warp reduction: after the loop LANE 0 holds the whole sum
static inline float ple_warp_sum(float v, uint lane) {
    for (int o = 16; o > 0; o >>= 1) v += simd_shuffle_down(v, (uint) o);
    (void) lane;
    return v;
}

/// ple.cu's block_sum(double) for this GPU: each thread arrives with a compensated partial, the warps fold
/// by the CUDA butterfly, and thread 0 folds the 8 warp values with the same compensation.  The double
/// original is exact to ~2^-52; this is exact to ~2^-46 where the products themselves are f32-rounded - the
/// same trade every double reduction in the port makes (router, shexp scalar gate).
static inline float ple_block_sum(float v, threadgroup float* part, uint tid, uint lane, uint sg) {
    const float warp_v = ple_warp_sum(v, lane);
    if (lane == 0) part[sg] = warp_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (tid == 0) {
        KahanSum t;
        for (int i = 0; i < PLE_WARPS; ++i) t.add(part[i]);
        total = t.value();
        part[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return part[0];
}

// ---- history_advance: one thread per channel slides the ROW-FASTEST state up by one and appends the new
// normalized row (flat launch: CUDA computed blockIdx*blockDim+threadIdx, which is the global thread id).
kernel void history_advance_kernel(device float* history [[buffer(0)]],
                                   constant const float* normalized [[buffer(1)]],
                                   uint channel [[thread_position_in_grid]]) {
    if (channel >= (uint) PLE_HC_DIM) return;
    device float* column = history + (ulong) channel * PLE_HIST;
    for (int row = 0; row + 1 < PLE_HIST; ++row) column[row] = column[row + 1];
    column[PLE_HIST - 1] = normalized[channel];
}

// ---- grouped_norm, ONE BLOCK PER STREAM (CUDA blockIdx.x is the data index -> the GROUP position).
// Stream c occupies [c*n_embd, (c+1)*n_embd); the PRODUCT is rounded in f32 before it is widened
// (`(ggml_float)(x[i00]*x[i00])`), which the KahanSum's f32 addand keeps, and the scale is 1/sqrt in f32.
// x may alias y (ple.cu normalizes d_key in place): both spellings are device, and loop 2's read of xc[d]
// happens after the reduction's barriers, so every loop-1 read has completed before the first write.
kernel void gnorm_kernel(device const float* x [[buffer(0)]],
                         constant const float* w [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant const int& n_embd [[buffer(3)]],
                         constant const float& eps [[buffer(4)]],
                         uint3 gpos [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[PLE_WARPS];
    const uint c = gpos.x;
    device const float* xc = x + (ulong) c * n_embd;
    constant const float* wc = w + (ulong) c * n_embd;
    device float* yc = y + (ulong) c * n_embd;

    KahanSum acc;
    for (int d = (int) tid; d < n_embd; d += PLE_THREADS) {
        const float sq = xc[d] * xc[d];               // rounded in f32, then accumulated (compensated)
        acc.add(sq);
    }
    const float mean = ple_block_sum(acc.value(), part, tid, lane, sg) / (float) n_embd;
    const float scale = 1.0f / metal::precise::sqrt(mean + eps);
    for (int d = (int) tid; d < n_embd; d += PLE_THREADS) yc[d] = xc[d] * scale * wc[d];
}

// ---- the gate: s[c] = sum_d key*query / sqrt(n_embd), then the signed square root and the sigmoid.  The
// clamp is a floor on |s|; the SIGN rides separately, which keeps the gate symmetric about 0.5.
kernel void gate_kernel(device const float* key [[buffer(0)]],
                        device const float* query [[buffer(1)]],
                        device float* gate [[buffer(2)]],
                        constant const int& n_embd [[buffer(3)]],
                        constant const float& inv_sqrt_n [[buffer(4)]],
                        uint3 gpos [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[PLE_WARPS];
    const uint c = gpos.x;
    device const float* kc = key + (ulong) c * n_embd;
    device const float* qc = query + (ulong) c * n_embd;
    KahanSum acc;
    for (int d = (int) tid; d < n_embd; d += PLE_THREADS) acc.add(kc[d] * qc[d]);
    const float s = ple_block_sum(acc.value(), part, tid, lane, sg) * inv_sqrt_n;
    const float mag = metal::precise::sqrt(metal::fmax(metal::precise::fabs(s), 1e-6f));
    const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    if (tid == 0) gate[c] = 1.0f / (1.0f + metal::precise::exp(-(sgn * mag)));
}

// ---- gated[c][d] = value[d] * gate[c]: the value broadcast across the hc streams (flat launch).
kernel void bcast_kernel(device const float* value [[buffer(0)]],
                         device const float* gate [[buffer(1)]],
                         device float* gated [[buffer(2)]],
                         constant const int& n_embd [[buffer(3)]],
                         constant const int& hc [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= (uint) (n_embd * hc)) return;
    gated[i] = value[i % (uint) n_embd] * gate[i / (uint) n_embd];
}

// ---- the depthwise causal dilated conv, then SiLU.  One thread per channel (flat launch).  Tap k reads
// row `nhist - (kern-1-k)*dil` of the ROW-FASTEST history (`hist[row + nhist*c]`); for the real geometry
// (kern 4, dil 3, hist 9) three taps come from the caller's history and one from the NEW normalized row.
// kW IS GGML-NATIVE: `kW[k + kern*c]`, F16 bits.
kernel void conv_kernel(device const float* hist [[buffer(0)]],
                        device const float* norm [[buffer(1)]],
                        constant const ushort* kW [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        constant const int& hc_dim [[buffer(4)]],
                        constant const int& kern [[buffer(5)]],
                        constant const int& dil [[buffer(6)]],
                        constant const int& nhist [[buffer(7)]],
                        uint c [[thread_position_in_grid]]) {
    if (c >= (uint) hc_dim) return;
    float acc = 0.0f;
    for (int k = 0; k < kern; ++k) {
        const int row = nhist - (kern - 1 - k) * dil;      // tap 0 reads the FURTHEST back
        const float v = (row == nhist) ? norm[c] : hist[(ulong) row + (ulong) nhist * c];
        acc += f32_from_f16(kW[(ulong) k + (ulong) kern * c]) * v;
    }
    out[c] = ple_silu(acc);
}

// ---- result = hidden + gated + conv, elementwise (flat launch).  result may alias hidden exactly; each
// element is read before the same element is written.
kernel void add3_kernel(device const float* hidden [[buffer(0)]],
                        device const float* gated [[buffer(1)]],
                        device const float* conv [[buffer(2)]],
                        device float* result [[buffer(3)]],
                        constant const int& n [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= (uint) n) return;
    result[i] = hidden[i] + gated[i] + conv[i];
}

// ---- the legacy value projection: y[o] = sum_i bf16(x[i]) * bf16(w[o*n_in + i]).  The activation is the
// BF16 image of the embedding (the BF16 tensor's contract), so every product is EXACT in f32 and only the
// summation order vs ggml differs - the CUDA kernel's double accumulation is the compensated f32 sum here.
kernel void bf16_gemv_kernel(constant const uint* x [[buffer(0)]],         // bf16 patterns
                             constant const uint* w [[buffer(1)]],         // bf16 patterns
                             device float* y [[buffer(2)]],
                             constant const int& n_in [[buffer(3)]],
                             constant const int& n_out [[buffer(4)]],
                             uint o [[thread_position_in_grid]]) {
    if (o >= (uint) n_out) return;
    // uint16 stride FIRST, then the pair view: a bare uint* row would walk 4-byte units (round 9's bug class)
    constant const uint16_t* row16 = reinterpret_cast<constant const uint16_t*>(w) + (ulong) o * n_in;
    constant const uint16_t* x16 = reinterpret_cast<constant const uint16_t*>(x);
    KahanSum acc;
    for (int i = 0; i < n_in; ++i) acc.add(ple_f32_from_bf16(x16[i]) * ple_f32_from_bf16(row16[i]));
    y[o] = acc.value();
}

// ---- f32 -> BF16 bits (flat launch); renamed from the CUDA file's private to_bf16_kernel because
// elementwise.metal already carries one under that name.
kernel void ple_to_bf16_kernel(device const float* x [[buffer(0)]],
                               device ushort* y [[buffer(1)]],
                               constant const int& n [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= (uint) n) return;
    y[i] = (ushort) bf16_from_f32(x[i]);
}
