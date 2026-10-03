// src/kernels/metal/fused_gdn.metal - the port of src/kernels/cuda/fused_gdn.cu's kernels: the fused
// step+norm, the fused conv+SiLU+L2, and the alpha/beta rows with their epilogues (plan v0.3 P3).
//
// The CUDA block dim3(S, RG) = (128, 4) becomes ONE FLAT 512-thread group: the CUDA kernel itself numbers its
// threads `tid = rg * S + col`, so col = tid & 127 and rg = tid >> 7 are that mapping read back, warp for
// warp (docs/PORT_METAL/PROGRESS.md rule 7: attributed group coordinates stay scalar next to a uint3 group
// position; the 2D spelling would drag thread_index_in_threadgroup to a width the toolchain rejects).
//
// The bf16 weight rows stride in uint16 ELEMENTS first and take the uint4 pair view after (the round-9
// bug class: casting to uint* before the stride lands row k on row 2k's weights).  The .cu's __expf sites
// spell metal::precise::exp - the metallib builds with -fno-fast-math, where the plain names do not exist
// (rule 5) - and its log1pf(__expf(v)) is elementwise.metal's measured series seam, copied below.
//
// MEASURED TRAP (M2 Max): a compute pipeline whose register pressure caps maxTotalThreadsPerThreadgroup
// BELOW the dispatched group size drops the whole dispatch SILENTLY - command buffer completes, err = 0,
// the kernel never runs.  gdn_step_norm_kernel's first port (the CUDA register array `float s[32]` kept)
// capped at 384 under its 512-thread launch and did exactly that; the kernel below carries no per-thread
// arrays for that reason and the launcher's group size must stay within the pipeline's cap.
#include "strata_port.metalh"

constant const int FGDN_S = 128;    // state size (rows = cols = 128)
constant const int FGDN_RG = 4;     // row groups
constant const int FGDN_RPG = FGDN_S / FGDN_RG;    // 32 rows per thread

static inline float fgdn_sigmoid_f(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// ggml_compute_softplus_f32: log1p(exp(x)) with the large-x branch; MSL has no log1p, so below e ~ 0.05 the
// Horner'd series carries what f32's log(1+e) would cancel away (elementwise.metal's measured seam)
static inline float fgdn_softplus_f(float x) {
    if (x > 20.0f) return x;
    const float e = metal::precise::exp(x);
    if (e < 0.05f) return e * (1.0f - e * (0.5f - e * (1.0f / 3.0f - e * 0.25f)));
    return metal::precise::log(1.0f + e);
}

// the .cu's eight ordered fmaf per uint4: bf16 pair k is (word k's low half, word k's high half) against
// x's elements 2k and 2k+1, chained straight into the caller's accumulator (no junction rounding)
static inline float fgdn_dot8(float acc, uint4 wv, float4 xa, float4 xb) {
    acc = fma(as_type<float>(wv.x << 16), xa.x, acc);
    acc = fma(as_type<float>(wv.x & 0xffff0000u), xa.y, acc);
    acc = fma(as_type<float>(wv.y << 16), xa.z, acc);
    acc = fma(as_type<float>(wv.y & 0xffff0000u), xa.w, acc);
    acc = fma(as_type<float>(wv.z << 16), xb.x, acc);
    acc = fma(as_type<float>(wv.z & 0xffff0000u), xb.y, acc);
    acc = fma(as_type<float>(wv.w << 16), xb.z, acc);
    acc = fma(as_type<float>(wv.w & 0xffff0000u), xb.w, acc);
    return acc;
}

// ---- the fused step + closing norm: one block of (128 cols x 4 row groups) per value head, the per-head
// RMS and the sigmoid(z) gate of gdn_out_norm running in the same kernel.  red[] carries each row group's
// partial through threadgroup memory exactly as the .cu's __shared__ red does, barriers included (the one
// after the rank-1 update exists because every thread has read red[] for kv_col). ----
kernel void gdn_step_norm_kernel(device float* state [[buffer(0)]],
                                 constant const float* q [[buffer(1)]],
                                 constant const float* k [[buffer(2)]],
                                 constant const float* v [[buffer(3)]],
                                 constant const float* gate [[buffer(4)]],
                                 constant const float* beta [[buffer(5)]],
                                 constant const float* z [[buffer(6)]],
                                 constant const float* gamma [[buffer(7)]],
                                 constant const float& eps [[buffer(8)]],
                                 device float* y [[buffer(9)]],
                                 constant const int& h_k [[buffer(10)]],
                                 constant const int& h_v [[buffer(11)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x is the head
                                 uint tid [[thread_index_in_threadgroup]]) {    // tid = rg * S + col
    threadgroup float sk[FGDN_S], sq[FGDN_S];
    threadgroup float red[FGDN_RG][FGDN_S];
    threadgroup float wsum[FGDN_S * FGDN_RG / 32];
    const int head = (int) gpos.x;
    const int col = (int) (tid & 127u);          // threadIdx.x, 0..127
    const int rg = (int) (tid >> 7);             // threadIdx.y, 0..3
    const int qh = head % h_k;
    if (tid < (uint) FGDN_S) {
        sk[tid] = k[(ulong) qh * FGDN_S + tid];
        sq[tid] = q[(ulong) qh * FGDN_S + tid];
    }
    device float* base = state + ((ulong) (rg * FGDN_RPG) * h_v + head) * FGDN_S + col;
    const ulong row_stride = (ulong) h_v * FGDN_S;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float g = metal::precise::exp(gate[head]);
    // the CUDA kernel holds its 32 state rows in REGISTERS across both passes (`float s[RPG]`); that array
    // pins this pipeline to maxTotalThreadsPerThreadgroup = 384 (measured on the M2 Max), and a 512-thread
    // dispatch over a 384-max pipeline is one Metal drops SILENTLY - the command buffer completes with no
    // error and the kernel never runs.  The port keeps the 512-thread shape and reloads each row instead:
    // the elements are this thread's own column, nothing writes them between the passes, so every load
    // returns the bits the register copy held and every fma chain is the CUDA one.
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < FGDN_RPG; ++r)
        kv = fma(base[r * row_stride], sk[rg * FGDN_RPG + r], kv);
    red[rg][col] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
    const float delta = (v[(ulong) head * FGDN_S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < FGDN_RPG; ++r) {
        const float s = fma(g, base[r * row_stride], sk[rg * FGDN_RPG + r] * delta);
        o = fma(s, sq[rg * FGDN_RPG + r], o);
        base[r * row_stride] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // every thread has read red[] for kv_col
    red[rg][col] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * metal::precise::rsqrt((float) FGDN_S);
        sq_part = oc * oc;
    }
    // RMS over the head's 128 outputs: warps of row group 0 are threads 0..127
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += simd_shuffle_xor(sq_part, (uint) o2);
    if ((tid & 31u) == 0u) wsum[tid >> 5] = sq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = metal::precise::rsqrt(ss / (float) FGDN_S + eps);
        const float zz = z[(ulong) head * FGDN_S + col];
        y[(ulong) head * FGDN_S + col] = oc * scale * gamma[col] * fgdn_sigmoid_f(zz);
    }
}

// ---- the 4-tap conv + SiLU + head-wise L2: one 128-thread block per 128-channel head; blockIdx.x both
// locates the head's channels (c = blockIdx.x * S + threadIdx.x) and decides whether the L2 runs (the
// first qk_heads heads are q then k), so the group position is read for both (rule 6). ----
kernel void gdn_conv_l2_kernel(device float* hist [[buffer(0)]],
                               constant const float* qkv [[buffer(1)]],
                               constant const float* w [[buffer(2)]],
                               device float* h [[buffer(3)]],
                               constant const int& qk_heads [[buffer(4)]],
                               constant const float& eps [[buffer(5)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[FGDN_S / 32];
    const int c = (int) gpos.x * FGDN_S + (int) tid;
    const float v0 = hist[(ulong) c * 3], v1 = hist[(ulong) c * 3 + 1], v2 = hist[(ulong) c * 3 + 2],
                x = qkv[c];
    float sum = v0 * w[(ulong) c * 4] + v1 * w[(ulong) c * 4 + 1] + v2 * w[(ulong) c * 4 + 2] +
                x * w[(ulong) c * 4 + 3];
    hist[(ulong) c * 3] = v1;
    hist[(ulong) c * 3 + 1] = v2;
    hist[(ulong) c * 3 + 2] = x;
    float y = sum / (1.0f + metal::precise::exp(-sum));
    if ((int) gpos.x < qk_heads) {               // uniform per block, like the .cu's blockIdx test
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += simd_shuffle_xor(sq, (uint) o);
        if ((tid & 31u) == 0u) part[tid >> 5] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= metal::precise::rsqrt(ss + eps);
    }
    h[c] = y;
}

// ---- alpha and beta rows: one warp per row of the (2*h_v, n_embd) bf16 pair, lane 0 running the epilogue
// (gate = softplus(alpha + dt) * ssm_a, beta = sigmoid(beta)).  The bf16 row strides in uint16 ELEMENTS
// before the uint4 view - the round-9 bug class. ----
kernel void gdn_ab_kernel(constant const float* x [[buffer(0)]],
                          constant const uint16_t* wa [[buffer(1)]],           // bf16 alpha rows
                          constant const uint16_t* wb [[buffer(2)]],           // bf16 beta rows
                          constant const float* dt [[buffer(3)]],
                          constant const float* ssm_a [[buffer(4)]],
                          device float* gate [[buffer(5)]],
                          device float* beta [[buffer(6)]],
                          constant const int& n [[buffer(7)]],
                          constant const int& h_v [[buffer(8)]],
                          uint3 gpos [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    const int row = (int) gpos.x * 8 + (int) sg;    // blockIdx.x * 8 + (threadIdx.x >> 5)
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>((is_beta ? wb : wa) +
                                                                       (ulong) r * n);
    float acc = 0.0f;
    for (int j = (int) lane; j < n / 8; j += 32) {
        const uint4 wv = w4[j];                     // __ldg: the constant address space caches it
        const float4 xa = *(constant const float4*) (x + (ulong) j * 8);
        const float4 xb = *(constant const float4*) (x + (ulong) j * 8 + 4);
        acc = fgdn_dot8(acc, wv, xa, xb);
    }
    for (int o = 16; o > 0; o >>= 1) acc += simd_shuffle_xor(acc, (uint) o);
    if (lane != 0u) return;
    if (is_beta) {
        beta[r] = fgdn_sigmoid_f(acc);
    } else {
        const float v = acc + dt[r];
        const float sp = fgdn_softplus_f(v);
        gate[r] = sp * ssm_a[r];
    }
}
