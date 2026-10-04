// src/kernels/metal/fused_gr.metal - the port of src/kernels/cuda/fused_gr.cu's kernels (K16): the
// hyper-connection read as three launches per token (norm, down projection, up projection).
//
// The CUDA file stages activations in shared memory - xn (40 KB) inside the single-token gr_down_kernel,
// and up to 8 x 2560 floats (80 KB) of tiles in gr_down_multi_kernel.  This GPU's maxThreadgroupMemoryLength
// is 32768 (measured, Apple M2 Max), so neither fits; the port stages xn in the caller's device scratch and
// the down kernel reads it straight back, and the up kernel reads lo from device memory instead of a shared
// tile.  That keeps the bits: the CUDA file's own invariant is that the tiling "only changes the staging
// granularity" - a lane accumulates its chunks of 8 in ascending j with either tile, so removing the tile
// altogether leaves every output bit-identical, and a shared-memory copy of lo is the same bits the down
// kernel wrote.
//
// EVERY POINTER IS A BOUND BUFFER ARGUMENT, ONE LAUNCH PER TOKEN (measured, docs/PORT_METAL/PROGRESS.md
// rule 9): a raw pointer stored in setBytes data or a buffer - the CUDA struct's whole argument style - is
// not usable on Apple Silicon.  Two spellings were measured dead: a pointer loaded from such data
// dereferences to zero with every write vanishing (no error), and even reconstructing it as anchor +
// byte-offset from one bound anchor only reaches ~8-16 KB from the anchor's base (offsets beyond that
// silently address nothing).  A bound [[buffer(N)]] argument reaches anywhere - including interior
// pointers, which the runtime binds with a large offset the same way every other ported kernel's do.  So
// the T-token GrMulti becomes T launches of one-token kernels, each with its own pointers as arguments;
// per token the arithmetic is the CUDA kernels' own, so the multi read is still bitwise the single-token
// read, exactly as fused_gr.hpp promises.
#include "strata_port.metalh"

constant const int FGR_THREADS = 256;
constant const int FGR_WARPS = 8;
constant const int FGR_N = 2560;                // n_embd
constant const int FGR_HC = 4;                  // streams
constant const int FGR_D = FGR_N * FGR_HC;      // 10240
constant const int FGR_LR = 320;                // hc_lr
constant const int FGR_DOWN_BLOCKS = FGR_LR / FGR_WARPS;    // 40; block 40 carries the inject rows

static inline float fgr_sigmoid(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// the CUDA file's __shfl_xor_sync butterfly: every lane ends up holding the whole sum
static inline float fgr_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, (uint) o);
    return v;
}

// dot8: 8 packed bf16 against 8 floats, the two ordered fmaf per word exactly as the CUDA file spells them
static inline float fgr_dot8(uint4 w, constant const float* x) {
    float acc = 0.0f;
    const uint v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fma(as_type<float>(v[j] << 16), x[2 * j], acc);
        acc = fma(as_type<float>(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

// ---- gr_norm_multi_kernel for ONE token: R' into xn = R' * w_norm, the per-stream sums of squares, rs,
// then the in-place scale.  Step 1 of the single-token gr_down_kernel is this kernel exactly (the CUDA
// file keeps them in step; both write a.rs[t] and scale xn by s_rs[i / N]). ----
kernel void fused_gr_norm_kernel(constant const float* R [[buffer(0)]],
                                 constant const float* bo_prev [[buffer(1)]],
                                 constant const float* inj_prev [[buffer(2)]],
                                 constant const float* w_norm [[buffer(3)]],
                                 constant const float& eps [[buffer(4)]],
                                 constant const int& apply [[buffer(5)]],
                                 device float* rs [[buffer(6)]],
                                 device float* xn [[buffer(7)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[FGR_WARPS][FGR_HC];
    threadgroup float s_rs[FGR_HC];
    float gw[FGR_HC];
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c)
        gw[c] = apply != 0 ? 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC) : 0.0f;
    float ss[FGR_HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = (int) tid * 4; i < FGR_D; i += FGR_THREADS * 4) {
        const int c = i / FGR_N, d = i - c * FGR_N;
        float4 r = *(constant const float4*) (R + i);
        if (apply != 0) {
            const float4 b = *(constant const float4*) (bo_prev + d);
            r.x = fma(b.x, gw[c], r.x); r.y = fma(b.y, gw[c], r.y);
            r.z = fma(b.z, gw[c], r.z); r.w = fma(b.w, gw[c], r.w);
        }
        const float4 g = *(constant const float4*) (w_norm + i);
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss[c] += sq;
        *(device float4*) (xn + i) = float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c) {
        const float v = fgr_warp_sum(ss[c]);
        if (lane == 0) part[sg][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FGR_HC) {
        float s = 0.0f;
        for (int w = 0; w < FGR_WARPS; ++w) s += part[w][tid];
        s_rs[tid] = metal::precise::rsqrt(s / (float) FGR_N + eps);
        rs[tid] = s_rs[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (int i = (int) tid; i < FGR_D; i += FGR_THREADS) xn[i] *= s_rs[i / FGR_N];
}

// ---- the same norm in one pass: each thread keeps its ten float4 of R' * w_norm in registers and scales them
// itself after the reduction. The kernel above stores them, then re-reads and scales with a different thread
// mapping behind a device barrier; here every element still gets (r * g) rounded, then * s_rs[c] rounded, and
// s_rs comes from the identical reduction, so xn and rs are the same bits. Only the dependent second pass over
// device memory goes. Measured on the M2 Max in isolation: 18.1-18.5 -> 8.2-9.5 us per call, 96 calls per token
// (bench/results/2026-10-03-metal-decode-opt2/micro/gr.log).
kernel void fused_gr_norm1_kernel(device const float* R [[buffer(0)]],
                                  device const float* bo_prev [[buffer(1)]],
                                  device const float* inj_prev [[buffer(2)]],
                                  device const float* w_norm [[buffer(3)]],
                                  constant const float& eps [[buffer(4)]],
                                  constant const int& apply [[buffer(5)]],
                                  device float* rs [[buffer(6)]],
                                  device float* xn [[buffer(7)]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[FGR_WARPS][FGR_HC];
    threadgroup float s_rs[FGR_HC];
    constexpr int K = FGR_D / (FGR_THREADS * 4);   // 10 float4 per thread; a float4 never straddles streams
    float gw[FGR_HC];
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c)
        gw[c] = apply != 0 ? 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC) : 0.0f;
    float ss[FGR_HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 xv[K];
    int cv[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = (int) tid * 4 + k * FGR_THREADS * 4;
        const int c = i / FGR_N, d = i - c * FGR_N;
        float4 r = *(device const float4*) (R + i);
        if (apply != 0) {
            const float4 b = *(device const float4*) (bo_prev + d);
            r.x = fma(b.x, gw[c], r.x); r.y = fma(b.y, gw[c], r.y);
            r.z = fma(b.z, gw[c], r.z); r.w = fma(b.w, gw[c], r.w);
        }
        const float4 g = *(device const float4*) (w_norm + i);
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss[c] += sq;
        xv[k] = float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        cv[k] = c;
    }
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c) {
        const float v = fgr_warp_sum(ss[c]);
        if (lane == 0) part[sg][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FGR_HC) {
        float s = 0.0f;
        for (int w = 0; w < FGR_WARPS; ++w) s += part[w][tid];
        s_rs[tid] = metal::precise::rsqrt(s / (float) FGR_N + eps);
        rs[tid] = s_rs[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = (int) tid * 4 + k * FGR_THREADS * 4;
        *(device float4*) (xn + i) = xv[k] * s_rs[cv[k]];
    }
}

// ---- the down projection for ONE token: one warp per output row, the 41st block carrying the four inject
// rows.  The CUDA kernel stages its activation tile in shared memory; this one reads xn from the device
// scratch, a lane's chunks of 8 still ascending in j (the tile invariant), so the sums are the tiled
// kernel's bit for bit. ----
kernel void fused_gr_down_kernel(constant const uint* w_down [[buffer(0)]],        // bf16 patterns
                                 constant const uint* w_inject [[buffer(1)]],      // bf16; nil = final mixer
                                 constant const float* xn [[buffer(2)]],
                                 device float* lo [[buffer(3)]],
                                 device float* inject_out [[buffer(4)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    const bool inject_block = gpos.x == FGR_DOWN_BLOCKS;
    const int row = inject_block ? (int) sg : (int) gpos.x * FGR_WARPS + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    constant const uint16_t* wrow16 =
        reinterpret_cast<constant const uint16_t*>(inject_block ? w_inject : w_down) +
        (ulong) (active ? row : 0) * FGR_D;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>(wrow16);
    float acc = 0.0f;
    if (active) {
        for (int j = (int) lane; j < FGR_D / 8; j += 32)
            acc += fgr_dot8(w4[j], xn + (ulong) j * 8);
    }
    acc = fgr_warp_sum(acc);
    // The inject block has eight warps but only four output scalars. Its inactive warps must not
    // overwrite the following allocation (often the previous half's still-live injection gates).
    if (lane != 0 || !active) return;
    if (inject_block) {
        inject_out[row] = acc;
    } else {
        const float x = acc / (float) FGR_HC;
        lo[row] = x / (1.0f + metal::precise::exp(-x));
    }
}

// ---- the up projection for ONE token: 160 blocks of 16 columns x 4 streams, 8 rows per warp; each row of
// w_up read once, reduced by the butterfly, lane 0 running the epilogue.  lo is read from device memory -
// the same bits the down kernel wrote, where the CUDA kernel stages a shared copy - and rs from the norm. ----
constant const int FGR_UPM_COLS = 16;

kernel void fused_gr_up_kernel(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][FGR_UPM_COLS];
    const int d0 = (int) gpos.x * FGR_UPM_COLS;
    for (int r = (int) sg; r < FGR_HC * FGR_UPM_COLS; r += FGR_WARPS) {
        const int c = r / FGR_UPM_COLS, dd = r - c * FGR_UPM_COLS, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FGR_UPM_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}


static inline float fgr_dot8_shared(uint4 w, threadgroup const float* x) {
    float acc = 0.0f;
    const uint v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fma(as_type<float>(v[j] << 16), x[2 * j], acc);
        acc = fma(as_type<float>(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

kernel void gr_norm_down_shared(device const float* R [[buffer(0)]],
                                  device const float* bo_prev [[buffer(1)]],
                                  device const float* inj_prev [[buffer(2)]],
                                  device const float* w_norm [[buffer(3)]],
                                  constant const float& eps [[buffer(4)]],
                                  constant const int& apply [[buffer(5)]],
                                  device float* rs [[buffer(6)]],
                                  device float* xn [[buffer(7)]],
                                  constant const uint* w_down [[buffer(8)]],
                                  constant const uint* w_inject [[buffer(9)]],
                                  device float* lo [[buffer(10)]],
                                  device float* inject_out [[buffer(11)]],
                                  uint3 gpos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[FGR_WARPS][FGR_HC];
    threadgroup float s_rs[FGR_HC];
    constexpr int K = FGR_D / (FGR_THREADS * 4);   // 10 float4 per thread; a float4 never straddles streams
    float gw[FGR_HC];
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c)
        gw[c] = apply != 0 ? 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC) : 0.0f;
    float ss[FGR_HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 xv[K];
    int cv[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = (int) tid * 4 + k * FGR_THREADS * 4;
        const int c = i / FGR_N, d = i - c * FGR_N;
        float4 r = *(device const float4*) (R + i);
        if (apply != 0) {
            const float4 b = *(device const float4*) (bo_prev + d);
            r.x = fma(b.x, gw[c], r.x); r.y = fma(b.y, gw[c], r.y);
            r.z = fma(b.z, gw[c], r.z); r.w = fma(b.w, gw[c], r.w);
        }
        const float4 g = *(device const float4*) (w_norm + i);
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss[c] += sq;
        xv[k] = float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        cv[k] = c;
    }
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c) {
        const float v = fgr_warp_sum(ss[c]);
        if (lane == 0) part[sg][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FGR_HC) {
        float s = 0.0f;
        for (int w = 0; w < FGR_WARPS; ++w) s += part[w][tid];
        s_rs[tid] = metal::precise::rsqrt(s / (float) FGR_N + eps);
        if (gpos.x == 0) rs[tid] = s_rs[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gpos.x == 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = (int) tid * 4 + k * FGR_THREADS * 4;
        *(device float4*) (xn + i) = xv[k] * s_rs[cv[k]];
    }
    }

    // Every group recomputes the original norm. Two halves fit the measured 32-KiB limit.
    threadgroup float tile[FGR_D / 2];
    const bool inject_block = gpos.x == FGR_DOWN_BLOCKS;
    const int row = inject_block ? (int) sg : (int) gpos.x * FGR_WARPS + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    constant const uint16_t* wrow16 =
        reinterpret_cast<constant const uint16_t*>(inject_block ? w_inject : w_down) +
        (ulong) (active ? row : 0) * FGR_D;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>(wrow16);
    float acc = 0.0f;
    for (int chunk = 0; chunk < 2; ++chunk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < K / 2; ++k) {
            const int src = k + chunk * (K / 2), i = (int) tid * 4 + k * FGR_THREADS * 4;
            *(threadgroup float4*) (tile + i) = xv[src] * s_rs[cv[src]];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int j = (int) lane + chunk * (FGR_D / 16); j < (chunk + 1) * (FGR_D / 16); j += 32)
                acc += fgr_dot8_shared(w4[j], tile + (j - chunk * (FGR_D / 16)) * 8);
        }
    }
    acc = fgr_warp_sum(acc);
    if (lane != 0 || !active) return;
    if (inject_block) inject_out[row] = acc;
    else {
        const float x = acc / (float) FGR_HC;
        lo[row] = x / (1.0f + metal::precise::exp(-x));
    }
}
