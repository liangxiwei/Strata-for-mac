// src/kernels/metal/prefill.metal - the port of src/prefill/kernels.cu's kernels (M4).  It joins the shared
// metallib glob (cmake/metal_backend.cmake), so every kernel carries a pfl_ prefix: this file's CUDA names
// gr_norm_kernel / gdn_* / kv_append_kernel / to_f16_kernel / to_bf16_kernel are ALREADY TAKEN by the decode
// path's ports (gr.metal, gdn.metal, kv_q8.metal, elementwise.metal).
//
// Transcription rules (docs/PORT_METAL/ rounds 9-13): every pointer is a bound [[buffer(N)]] whose index
// equals its position in the launcher's argument chain (bug-class A); nullptr binds nil and the kernel's
// nullptr test sees exactly that (R4); __expf/log1pf/powf/cosf/sinf become metal::precise:: spellings
// (R5), with the softplus seam measured in elementwise.metal (no log1p in MSL); blockIdx-as-data-index
// reads threadgroup_position_in_grid, blockIdx*blockDim+threadIdx reads thread_position_in_grid (R6);
// no struct ever carries a pointer (R9) - the CUDA rope_kernel's by-value RopeTab and kv_append_kernel's
// by-value KvHostPools arrive as bound buffers and scalars, in declaration order.
//
// The geometry is the artifact's and constexpr in the CUDA file (N 2560, HC 4, D 10240, LR 320, S 128,
// HK 16, HV 48, C 10240); it stays constexpr here - the launchers bake the same constants.
#include "strata_port.metalh"
#include <metal_simdgroup_matrix>

// 16 query rows x 32 output columns. Load each weight tile once for four SIMD groups, convert BF16
// inside threadgroup memory, and keep accumulators in FP32. No full-matrix F32 widening allocation.
template<typename S, bool BF16>
inline void pfl_matrix_product(const device ushort* x, const device ushort* w, device float* y,
                               uint M, uint N, uint K, uint ldy, float beta, uint2 tile, uint tid, uint sg,
                               threadgroup S* sx, threadgroup S* sw, threadgroup float* result) {
    simdgroup_float8x8 c0(0.0f), c1(0.0f);
    const uint mr = tile.y * 16, nr = tile.x * 32;
    const uint sm = (sg / 2) * 8, sn = (sg % 2) * 16;
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < 16 * 32; i += 128) {
            const uint m = i / 32, k = i % 32;
            const ushort v = mr + m < M && k0 + k < K ? x[(ulong)(mr + m) * K + k0 + k] : 0;
            sx[i] = BF16 ? S(as_type<float>(uint(v) << 16)) : S(as_type<half>(v));
        }
        for (uint i = tid; i < 32 * 32; i += 128) {
            const uint n = i / 32, k = i % 32;
            const ushort v = nr + n < N && k0 + k < K ? w[(ulong)(nr + n) * K + k0 + k] : 0;
            sw[k * 32 + n] = BF16 ? S(as_type<float>(uint(v) << 16)) : S(as_type<half>(v));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_matrix<S,8,8> a, b0, b1;
            simdgroup_load(a, sx + sm * 32 + k, 32);
            simdgroup_load(b0, sw + k * 32 + sn, 32);
            simdgroup_load(b1, sw + k * 32 + sn + 8, 32);
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0, result + sm * 32 + sn, 32);
    simdgroup_store(c1, result + sm * 32 + sn + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16 * 32; i += 128) {
        const uint m = mr + i / 32, n = nr + i % 32;
        if (m < M && n < N) {
            const ulong at = (ulong)m * ldy + n;
            y[at] = beta == 0.0f ? result[i] : fma(beta, y[at], result[i]);
        }
    }
}

#define PFL_MATRIX_KERNEL(NAME, ST, BF) \
kernel void NAME(const device ushort* x [[buffer(0)]], const device ushort* w [[buffer(1)]], \
                 device float* y [[buffer(2)]], constant uint& M [[buffer(3)]], \
                 constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], \
                 constant uint& ldy [[buffer(6)]], constant float& beta [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup ST sx[16 * 32], sw[32 * 32]; \
    threadgroup float result[16 * 32]; \
    pfl_matrix_product<ST, BF>(x, w, y, M, N, K, ldy, beta, tile, tid, sg, sx, sw, result); \
}
PFL_MATRIX_KERNEL(pfl_gemm_bf16, float, true)
PFL_MATRIX_KERNEL(pfl_gemm_f16, half, false)

// The same products in larger tiles: BM x BN per threadgroup, SGM x SGN SIMD groups of (BM/SGM) x (BN/SGN), operands
// staged as [row][32 k] with 4-element loads, the next k-chunk's loads issued before the current chunk's MMAs.
// Every output element gets exactly pfl_matrix_product's MMA sequence: operands of the same type (half for FP16,
// float for BF16) and values, 8-wide k steps ascending from a zero accumulator, then the same beta epilogue - so
// the results are the same bits (micro/prefill-gemm.log; the shapes, partial M tiles and beta = 1 compared).
// The launcher requires K % 32 == 0 and 8-byte aligned X / W rows (the vector loads); anything else keeps
// pfl_matrix_product. Measured on the M2 Max, isolated: FP16 2.4-2.7 -> 5.5-7.7 TFLOPS, BF16 1.5-2.3 -> 4.5-5.7.
template<typename S, bool BF16, int BM, int BN, int SGM, int SGN>
inline void pfl_matrix_product2(const device ushort* x, const device ushort* w, device float* y, uint M, uint N, uint K,
                       uint ldy, float beta, uint2 tile, uint tid, uint sg, uint lane,
                       threadgroup S* sx, threadgroup S* sw, threadgroup float* scratch) {
    constexpr int TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    constexpr int XL = BM * 8 / TH, WL = BN * 8 / TH;          // ushort4 loads per thread per chunk
    static_assert(BM * 8 % TH == 0 && BN * 8 % TH == 0, "even split");
    simdgroup_float8x8 c[MI][NJ];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) c[i][j] = simdgroup_float8x8(0.0f);
    const uint mr = tile.y * BM, nr = tile.x * BN;
    const uint sm = (sg / SGN) * (BM / SGM), sn = (sg % SGN) * (BN / SGN);
    ushort4 rx[XL], rw[WL];
    auto fetch = [&](uint k0) {
#pragma unroll
        for (int l = 0; l < XL; ++l) {
            const uint i = tid + (uint) l * TH, m = i / 8, kk = (i % 8) * 4;
            rx[l] = (mr + m < M && k0 < K) ? *(const device ushort4*) (x + (ulong) (mr + m) * K + k0 + kk) : ushort4(0);
        }
#pragma unroll
        for (int l = 0; l < WL; ++l) {
            const uint i = tid + (uint) l * TH, n = i / 8, kk = (i % 8) * 4;
            rw[l] = (nr + n < N && k0 < K) ? *(const device ushort4*) (w + (ulong) (nr + n) * K + k0 + kk) : ushort4(0);
        }
    };
    fetch(0);
    for (uint k0 = 0; k0 < K; k0 += 32) {
#pragma unroll
        for (int l = 0; l < XL; ++l) {
            const uint i = tid + (uint) l * TH, m = i / 8, kk = (i % 8) * 4;
            threadgroup S* d = sx + m * 32 + kk;
#pragma unroll
            for (int q = 0; q < 4; ++q) d[q] = BF16 ? S(as_type<float>(uint(rx[l][q]) << 16)) : S(as_type<half>(rx[l][q]));
        }
#pragma unroll
        for (int l = 0; l < WL; ++l) {
            const uint i = tid + (uint) l * TH, n = i / 8, kk = (i % 8) * 4;
            threadgroup S* d = sw + n * 32 + kk;
#pragma unroll
            for (int q = 0; q < 4; ++q) d[q] = BF16 ? S(as_type<float>(uint(rw[l][q]) << 16)) : S(as_type<half>(rw[l][q]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fetch(k0 + 32);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_matrix<S, 8, 8> a[MI], b[NJ];
#pragma unroll
            for (int i = 0; i < MI; ++i) simdgroup_load(a[i], sx + (sm + i * 8) * 32 + k, 32);
#pragma unroll
            for (int j = 0; j < NJ; ++j) simdgroup_load(b[j], sw + (sn + j * 8) * 32 + k, 32, ulong2(0, 0), true);
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NJ; ++j) simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* mine = scratch + sg * 64;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            simdgroup_store(c[i][j], mine, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint m = mr + sm + i * 8 + e / 8, n = nr + sn + j * 8 + e % 8;
                if (m < M && n < N) {
                    const ulong at = (ulong) m * ldy + n;
                    y[at] = beta == 0.0f ? mine[e] : fma(beta, y[at], mine[e]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
#define PFL_MATRIX2_KERNEL(NAME, ST, BF, BM, BN, SGM, SGN) \
kernel void NAME(const device ushort* x [[buffer(0)]], const device ushort* w [[buffer(1)]], \
                 device float* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], \
                 constant uint& K [[buffer(5)]], constant uint& ldy [[buffer(6)]], constant float& beta [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ST sx[BM * 32], sw[BN * 32]; threadgroup float scratch[SGM * SGN * 64]; \
    pfl_matrix_product2<ST, BF, BM, BN, SGM, SGN>(x, w, y, M, N, K, ldy, beta, tile, tid, sg, lane, sx, sw, scratch); \
}
PFL_MATRIX2_KERNEL(pfl_gemm2_f16, half, false, 64, 64, 2, 2)     // 128 threads
PFL_MATRIX2_KERNEL(pfl_gemm2_bf16, float, true, 64, 64, 2, 4)    // 256 threads

// Each descriptor is {expert, first sorted activation row, row count <= 16, reserved}. Decode only
// the Q2 weight tile needed by this matrix multiply, instead of writing a whole FP16 expert to RAM.
template<bool DOWN>
inline void pfl_q2_product(const device ushort* x, const device uchar* arena, const device int4* tiles,
                           device float* y, ulong blob_bytes, uint2 tile, uint tid, uint sg,
                           threadgroup half* sx, threadgroup half* sw, threadgroup float* result) {
    constexpr uint K = DOWN ? 640 : 2560, N = DOWN ? 2560 : 1280;
    constexpr ulong CODES = DOWN ? 1280 * 640 : 0;
    constexpr ulong SCALES = 1280 * 640 + 2560 * 160 + (DOWN ? 1280 * 40 * 2 : 0);
    const int4 desc = tiles[tile.y];
    const uint row0 = uint(desc.y), rows = uint(desc.z), nr = tile.x * 32;
    const device uchar* blob = arena + ulong(desc.x) * blob_bytes;
    simdgroup_float8x8 c0(0.0f), c1(0.0f);
    const uint sm = (sg / 2) * 8, sn = (sg % 2) * 16;
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < 16 * 32; i += 128) {
            const uint m = i / 32, k = k0 + i % 32;
            sx[i] = m < rows ? as_type<half>(x[ulong(row0 + m) * K + k]) : half(0);
        }
        for (uint i = tid; i < 32 * 32; i += 128) {
            const uint n = nr + i / 32, k = k0 + i % 32;
            const uchar code = blob[CODES + ulong(n) * (K / 4) + k / 4];
            const int q = int((code >> (2 * (k % 4))) & 3) - 1;
            const device half* scales = reinterpret_cast<const device half*>(blob + SCALES);
            sw[(i % 32) * 32 + i / 32] = half(q) * scales[ulong(n) * (K / 64) + k / 64];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a, b0, b1;
            simdgroup_load(a, sx + sm * 32 + k, 32);
            simdgroup_load(b0, sw + k * 32 + sn, 32);
            simdgroup_load(b1, sw + k * 32 + sn + 8, 32);
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0, result + sm * 32 + sn, 32);
    simdgroup_store(c1, result + sm * 32 + sn + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16 * 32; i += 128)
        if (i / 32 < rows) y[ulong(row0 + i / 32) * N + nr + i % 32] = result[i];
}
#define PFL_Q2_KERNEL(NAME, DOWN) \
kernel void NAME(const device ushort* x [[buffer(0)]], const device uchar* arena [[buffer(1)]], \
                 const device int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant ulong& blob_bytes [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup half sx[16 * 32], sw[32 * 32]; threadgroup float result[16 * 32]; \
    pfl_q2_product<DOWN>(x, arena, tiles, y, blob_bytes, tile, tid, sg, sx, sw, result); \
}
PFL_Q2_KERNEL(pfl_q2_gemm_gu, false)
PFL_Q2_KERNEL(pfl_q2_gemm_down, true)

constant const int PFL_N = 2560;
constant const int PFL_HC = 4;
constant const int PFL_D = PFL_N * PFL_HC;     // 10240
constant const int PFL_S = 128;
constant const int PFL_HK = 16;
constant const int PFL_HV = 48;
constant const int PFL_C = 10240;
constant const int PFL_CONV_TILE = 64;

// ---- the CUDA file's device helpers, transcribed -----------------------------------------

static inline float pfl_warp_sum(float v) {
    for (uint o = 16u; o > 0u; o >>= 1u) v += simd_shuffle_xor(v, o);   // __shfl_xor_sync's butterfly
    return v;
}
static inline float pfl_warp_max(float v) {
    for (uint o = 16u; o > 0u; o >>= 1u) v = fmax(v, simd_shuffle_xor(v, o));
    return v;
}
static inline float pfl_sigm(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }
static inline uint pfl_bf(float f) { return bf16_from_f32(f); }
// bf_lo: what the BF16 image `hi` left out of f, itself in BF16 (STRATA_PREFILL_BF16X2)
static inline uint pfl_bf_lo(float f, uint hi) {
    uint w = (uint) hi << 16;
    return bf16_from_f32(f - as_type<float>(w));
}
static inline uint pfl_hf(float f) { return f16_from_f32(f); }
// A SwiGLU product for an FP16 GEMM: saturated (a token with a massive activation cannot become inf and
// then NaN in the down projection); a NaN stays NaN.  fp16 ends at 65504, exactly as the CUDA file spells.
static inline uint pfl_hf_sat(float f) {
    if (isnan(f)) return pfl_hf(f);
    return pfl_hf(fmin(fmax(f, -65504.0f), 65504.0f));
}
// block-wide sum for <=1024 threads, result broadcast - the CUDA block_sum's two warp stages and its
// shared-memory hand-offs in the same order, so the bits match (the .cu's callers pass 256 threads)
static inline float pfl_block_sum(float v, threadgroup float* sh, uint tid, uint nt) {
    const uint lane = tid & 31u, w = tid >> 5u;
    v = pfl_warp_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sh[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nw = (nt + 31u) >> 5u;
    float t = tid < nw ? sh[tid] : 0.0f;
    if (w == 0u) t = pfl_warp_sum(t);
    if (tid == 0u) sh[0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}
// log1pf(__expf(v)): elementwise.metal's measured softplus seam (MSL has no log1p; the series covers the
// cancellation-bound small end, the x > 20 branch is x to within f32)
static inline float pfl_softplus(float x) {
    if (x > 20.0f) return x;
    const float e = metal::precise::exp(x);
    if (e < 0.05f) return e * (1.0f - e * (0.5f - e * (1.0f / 3.0f - e * 0.25f)));
    return metal::precise::log(1.0f + e);
}

// ---- hyper-connection ---------------------------------------------------------------------

kernel void pfl_gr_norm_kernel(constant const float* R [[buffer(0)]],
                               constant const float* w [[buffer(1)]],
                               constant const float& eps [[buffer(2)]],
                               device float* xn [[buffer(3)]],
                               device ushort* xn16 [[buffer(4)]],
                               device ushort* xn16_lo [[buffer(5)]],       // null: none
                               uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x = row = t*4 + c
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const ulong row = gpos.x;
    const uint c = (uint) (row % PFL_HC);
    constant const float* r = R + row * PFL_N;
    float ss = 0.0f;
    for (int d = (int) tid; d < PFL_N; d += (int) 256u) ss += r[d] * r[d];
    const float rs = metal::precise::rsqrt(pfl_block_sum(ss, sh, tid, 256u) / (float) PFL_N + eps);
    for (int d = (int) tid; d < PFL_N; d += (int) 256u) {
        const float v = r[d] * rs * w[c * PFL_N + d];
        xn[row * PFL_N + d] = v;
        const uint h = pfl_bf(v);
        xn16[row * PFL_N + d] = (ushort) h;
        if (xn16_lo != nullptr) xn16_lo[row * PFL_N + d] = (ushort) pfl_bf_lo(v, h);
    }
}

// F-1: the row scale only (and the BF16 image); pfl_gr_mix_r_kernel recomputes r * rs * w itself, in the
// same order, so the FP32 copy of the normalized rows is neither written nor read
kernel void pfl_gr_norm_rs_kernel(constant const float* R [[buffer(0)]],
                                  constant const float* w [[buffer(1)]],
                                  constant const float& eps [[buffer(2)]],
                                  device float* rs_out [[buffer(3)]],
                                  device ushort* xn16 [[buffer(4)]],
                                  device ushort* xn16_lo [[buffer(5)]],
                                  uint3 gpos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const ulong row = gpos.x;
    const uint c = (uint) (row % PFL_HC);
    constant const float* r = R + row * PFL_N;
    float ss = 0.0f;
    for (int d = (int) tid; d < PFL_N; d += (int) 256u) ss += r[d] * r[d];
    const float rs = metal::precise::rsqrt(pfl_block_sum(ss, sh, tid, 256u) / (float) PFL_N + eps);
    if (tid == 0u) rs_out[row] = rs;
    for (int d = (int) tid; d < PFL_N; d += (int) 256u) {
        const float v = r[d] * rs * w[c * PFL_N + d];
        const uint h = pfl_bf(v);
        xn16[row * PFL_N + d] = (ushort) h;
        if (xn16_lo != nullptr) xn16_lo[row * PFL_N + d] = (ushort) pfl_bf_lo(v, h);
    }
}

kernel void pfl_gr_mix_r_kernel(constant const float* R [[buffer(0)]],
                                constant const float* rs [[buffer(1)]],
                                constant const float* w [[buffer(2)]],
                                constant const float* g [[buffer(3)]],
                                device float* mixed [[buffer(4)]],
                                device ushort* mixed16 [[buffer(5)]],
                                constant const long& T [[buffer(6)]],
                                device ushort* mixed_h [[buffer(7)]],
                                device ushort* mixed16_lo [[buffer(8)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * PFL_N) return;
    const ulong t = i / PFL_N, d = i % PFL_N;
    float s = 0.0f;
    for (int c = 0; c < PFL_HC; ++c) {
        const ulong j = t * PFL_D + (ulong) c * PFL_N + d;
        const float x = R[j] * rs[t * PFL_HC + c] * w[c * PFL_N + d];   // pfl_gr_norm_kernel's value, bit for bit
        s = fma(x, pfl_sigm(g[j]), s);
    }
    s /= (float) PFL_HC;
    mixed[i] = s;
    if (mixed16 != nullptr) {
        const uint h = pfl_bf(s);
        mixed16[i] = (ushort) h;
        if (mixed16_lo != nullptr) mixed16_lo[i] = (ushort) pfl_bf_lo(s, h);
    }
    if (mixed_h != nullptr) mixed_h[i] = (ushort) pfl_hf(s);
}

// F-2: pfl_gr_write_kernel for one row (t, c), then pfl_gr_norm_rs_kernel's reduction over it with the next
// half's norm weights - the same thread-to-element mapping (256 threads, stride 256) and block_sum
constant const int PFL_GRW_PER = (PFL_N + 255) / 256;   // 10
kernel void pfl_gr_write_norm_rs_kernel(device float* R [[buffer(0)]],
                                        constant const float* bo [[buffer(1)]],
                                        constant const float* inj [[buffer(2)]],
                                        constant const long& inj_ld [[buffer(3)]],
                                        constant const float* w [[buffer(4)]],
                                        constant const float& eps [[buffer(5)]],
                                        device float* rs_out [[buffer(6)]],
                                        device ushort* xn16 [[buffer(7)]],
                                        device ushort* xn16_lo [[buffer(8)]],
                                        uint3 gpos [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const ulong row = gpos.x;
    const ulong t = row / PFL_HC;
    const uint c = (uint) (row % PFL_HC);
    device float* r = R + row * PFL_N;
    const float sc = 2.0f * pfl_sigm(inj[t * (ulong) inj_ld + c] / (float) PFL_HC);
    float v[PFL_GRW_PER];
    float ss = 0.0f;
    int k = 0;
    for (int d = (int) tid; d < PFL_N; d += 256, ++k) {
        const float x = fma(bo[t * PFL_N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = metal::precise::rsqrt(pfl_block_sum(ss, sh, tid, 256u) / (float) PFL_N + eps);
    if (tid == 0u) rs_out[row] = rs;
    k = 0;
    for (int d = (int) tid; d < PFL_N; d += 256, ++k) {
        const float x = v[k] * rs * w[c * PFL_N + d];
        const uint h = pfl_bf(x);
        xn16[row * PFL_N + d] = (ushort) h;
        if (xn16_lo != nullptr) xn16_lo[row * PFL_N + d] = (ushort) pfl_bf_lo(x, h);
    }
}

kernel void pfl_gr_silu_kernel(constant const float* lo [[buffer(0)]],
                               device ushort* lo16 [[buffer(1)]],
                               device ushort* lo16_lo [[buffer(2)]],
                               constant const ulong& n [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float x = lo[i] / (float) PFL_HC;
    const float v = x / (1.0f + metal::precise::exp(-x));
    const uint h = pfl_bf(v);
    lo16[i] = (ushort) h;
    if (lo16_lo != nullptr) lo16_lo[i] = (ushort) pfl_bf_lo(v, h);
}

kernel void pfl_gr_mix_kernel(constant const float* xn [[buffer(0)]],
                              constant const float* g [[buffer(1)]],
                              device float* mixed [[buffer(2)]],
                              device ushort* mixed16 [[buffer(3)]],
                              constant const long& T [[buffer(4)]],
                              device ushort* mixed_h [[buffer(5)]],
                              device ushort* mixed16_lo [[buffer(6)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * PFL_N) return;
    const ulong t = i / PFL_N, d = i % PFL_N;
    float s = 0.0f;
    for (int c = 0; c < PFL_HC; ++c) {
        const ulong j = t * PFL_D + (ulong) c * PFL_N + d;
        s = fma(xn[j], pfl_sigm(g[j]), s);
    }
    s /= (float) PFL_HC;
    mixed[i] = s;
    if (mixed16 != nullptr) {
        const uint h = pfl_bf(s);
        mixed16[i] = (ushort) h;
        if (mixed16_lo != nullptr) mixed16_lo[i] = (ushort) pfl_bf_lo(s, h);
    }
    if (mixed_h != nullptr) mixed_h[i] = (ushort) pfl_hf(s);
}

kernel void pfl_gr_write_kernel(device float* R [[buffer(0)]],
                                constant const float* bo [[buffer(1)]],
                                constant const float* inj [[buffer(2)]],
                                constant const long& inj_ld [[buffer(3)]],
                                constant const long& T [[buffer(4)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * PFL_D) return;
    const ulong t = i / PFL_D, c = (i % PFL_D) / PFL_N, d = i % PFL_N;
    R[i] = fma(bo[t * PFL_N + d], 2.0f * pfl_sigm(inj[t * (ulong) inj_ld + c] / (float) PFL_HC), R[i]);
}

kernel void pfl_gr_broadcast_kernel(constant const float* e [[buffer(0)]],
                                    device float* R [[buffer(1)]],
                                    constant const long& T [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * PFL_D) return;
    const ulong t = i / PFL_D, d = i % PFL_N;
    R[i] = e[t * PFL_N + d];
}

// ---- GDN ------------------------------------------------------------------------------------

kernel void pfl_gdn_gates_kernel(constant const float* ab [[buffer(0)]],
                                 constant const float* dt [[buffer(1)]],
                                 constant const float* ssm_a [[buffer(2)]],
                                 device float* gate [[buffer(3)]],
                                 device float* beta [[buffer(4)]],
                                 constant const long& T [[buffer(5)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * PFL_HV) return;
    const ulong t = i / PFL_HV, h = i % PFL_HV;
    const float v = ab[t * 2 * PFL_HV + h] + dt[h];
    gate[i] = pfl_softplus(v) * ssm_a[h];
    beta[i] = pfl_sigm(ab[t * 2 * PFL_HV + PFL_HV + h]);
}

// one thread per channel, walks the chunk; then pfl_gdn_l2_kernel normalises
kernel void pfl_gdn_conv_kernel(device float* hist [[buffer(0)]],
                                constant const float* qkv [[buffer(1)]],
                                constant const float* w [[buffer(2)]],
                                device float* h [[buffer(3)]],
                                constant const long& T [[buffer(4)]],
                                uint c [[thread_position_in_grid]]) {      // blockIdx.x*blockDim.x+threadIdx.x
    if (c >= PFL_C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (long t = 0; t < T; ++t) {
        const float x = qkv[(ulong) t * PFL_C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[(ulong) t * PFL_C + c] = s / (1.0f + metal::precise::exp(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}

// C-3: the same 4-tap causal conv, tiled over tokens: thread (c, tile) reads its tile's 3 predecessors
// from the chunk (or the history before it) - the conv reads inputs, not its own outputs, so the tiles are
// independent and the per-element expression (so the bits) is the serial walk's
kernel void pfl_gdn_conv_tiled_kernel(device float* hist [[buffer(0)]],
                                      constant const float* qkv [[buffer(1)]],
                                      constant const float* w [[buffer(2)]],
                                      device float* h [[buffer(3)]],
                                      constant const long& T [[buffer(4)]],
                                      uint2 tpg [[thread_position_in_grid]],          // x: the channel
                                      uint2 gpos [[threadgroup_position_in_grid]]) {  // y: the token tile
    const uint c = tpg.x;
    if (c >= PFL_C) return;
    const long t0 = (long) gpos.y * PFL_CONV_TILE;
    if (t0 >= T) return;
    const long t1 = t0 + PFL_CONV_TILE < T ? t0 + PFL_CONV_TILE : T;
    device float* hist_c = hist + c * 3;
    const float v_init[3] = {hist_c[0], hist_c[1], hist_c[2]};
    float v0 = t0 - 3 >= 0 ? qkv[(ulong) (t0 - 3) * PFL_C + c] : v_init[(int) (t0 - 3 + 3)];
    float v1 = t0 - 2 >= 0 ? qkv[(ulong) (t0 - 2) * PFL_C + c] : v_init[(int) (t0 - 2 + 3)];
    float v2 = t0 - 1 >= 0 ? qkv[(ulong) (t0 - 1) * PFL_C + c] : v_init[(int) (t0 - 1 + 3)];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (long t = t0; t < t1; ++t) {
        const float x = qkv[(ulong) t * PFL_C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[(ulong) t * PFL_C + c] = s / (1.0f + metal::precise::exp(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}

// the history after the chunk: its last three inputs (the older history where the chunk is shorter than 3)
kernel void pfl_gdn_conv_hist_kernel(device float* hist [[buffer(0)]],
                                     constant const float* qkv [[buffer(1)]],
                                     constant const long& T [[buffer(2)]],
                                     uint c [[thread_position_in_grid]]) {
    if (c >= PFL_C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const long t = T - 3 + k;
        v[k] = t >= 0 ? qkv[(ulong) t * PFL_C + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}

kernel void pfl_gdn_l2_kernel(device float* h [[buffer(0)]],
                              constant const float& eps [[buffer(1)]],
                              uint3 gpos [[threadgroup_position_in_grid]],   // x: head, y: token
                              uint tid [[thread_index_in_threadgroup]]) {
    // block (t, head) over the 32 q/k heads, 128 threads
    const ulong t = gpos.y;
    const uint head = gpos.x;
    device float* x = h + t * PFL_C + (ulong) head * PFL_S;
    const float v = x[tid];
    float sq = pfl_warp_sum(v * v);
    threadgroup float part[4];
    if ((tid & 31u) == 0u) part[tid >> 5u] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[tid] = v * metal::precise::rsqrt(ss + eps);
}

constant const int PFL_RG = 4;
constant const int PFL_RPG = PFL_S / PFL_RG;     // 32

// the one-block-per-head kernel (A/B; STRATA_GDN_REC_HEADS): 512 threads, one threadgroup array budget
// sk+sq+red+wsum = 9472 B, well under the measured 32768 B limit
kernel void pfl_gdn_rec_kernel(device float* state [[buffer(0)]],
                               constant const float* h [[buffer(1)]],
                               constant const float* gate [[buffer(2)]],
                               constant const float* beta [[buffer(3)]],
                               constant const float* z [[buffer(4)]],
                               constant const float* gamma [[buffer(5)]],
                               constant const float& eps [[buffer(6)]],
                               device float* y [[buffer(7)]],
                               device ushort* y16 [[buffer(8)]],
                               constant const long& T [[buffer(9)]],
                               uint2 gpos [[threadgroup_position_in_grid]],      // x: the head
                               uint2 tid2 [[thread_position_in_threadgroup]]) {  // (col, rg)
    threadgroup float sk[PFL_S], sq[PFL_S], red[PFL_RG][PFL_S], wsum[16];
    const uint head = gpos.x, col = tid2.x, rg = tid2.y;
    const uint tid = rg * PFL_S + col;
    const uint qh = head % PFL_HK;
    float s[PFL_RPG];
    device float* base = state + ((ulong) (rg * PFL_RPG) * PFL_HV + head) * PFL_S + col;
    const ulong rs = (ulong) PFL_HV * PFL_S;
    for (int r = 0; r < PFL_RPG; ++r) s[r] = base[(ulong) r * rs];
    const float g_col = gamma[col];
    for (long t = 0; t < T; ++t) {
        constant const float* ht = h + (ulong) t * PFL_C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < PFL_S) { sq[tid] = ht[qh * PFL_S + tid]; sk[tid] = ht[PFL_HK * PFL_S + qh * PFL_S + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float g = metal::precise::exp(gate[(ulong) t * PFL_HV + head]);
        float kv = 0.0f;
        for (int r = 0; r < PFL_RPG; ++r) kv = fma(s[r], sk[rg * PFL_RPG + r], kv);
        red[rg][col] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * PFL_HK * PFL_S + head * PFL_S + col] - g * kv_col) *
                            beta[(ulong) t * PFL_HV + head];
        float o = 0.0f;
        for (int r = 0; r < PFL_RPG; ++r) {
            s[r] = fma(g, s[r], sk[rg * PFL_RPG + r] * delta);
            o = fma(s[r], sq[rg * PFL_RPG + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg][col] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0u) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * metal::precise::rsqrt((float) PFL_S);
            sp = oc * oc;
        }
        sp = pfl_warp_sum(sp);
        if ((tid & 31u) == 0u) wsum[tid >> 5u] = sp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0u) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float v = oc * metal::precise::rsqrt(ss / (float) PFL_S + eps) * g_col *
                            pfl_sigm(z[(ulong) t * PFL_HV * PFL_S + head * PFL_S + col]);
            y[(ulong) t * PFL_HV * PFL_S + head * PFL_S + col] = v;
            y16[(ulong) t * PFL_HV * PFL_S + head * PFL_S + col] = (ushort) pfl_hf(v);
        }
    }
    for (int r = 0; r < PFL_RPG; ++r) base[(ulong) r * rs] = s[r];
}

// D-2: the recurrence with the value columns split over 4 blocks per head, the output norm in its own
// kernel; per column the same arithmetic in the same order (the 4 row-group partials added as
// red[0]+red[1]+red[2]+red[3]): the same bits
constant const int PFL_CB = 32, PFL_NCB = PFL_S / PFL_CB;
kernel void pfl_gdn_rec_cols_kernel(device float* state [[buffer(0)]],
                                    constant const float* h [[buffer(1)]],
                                    constant const float* gate [[buffer(2)]],
                                    constant const float* beta [[buffer(3)]],
                                    device float* oc_out [[buffer(4)]],
                                    constant const long& T [[buffer(5)]],
                                    uint2 gpos [[threadgroup_position_in_grid]],
                                    uint2 tid2 [[thread_position_in_threadgroup]]) {
    threadgroup float sk[PFL_S], sq[PFL_S], red[PFL_RG][PFL_CB];
    const uint head = gpos.x / PFL_NCB, cb = gpos.x % PFL_NCB;
    const uint c = tid2.x, rg = tid2.y, tid = rg * PFL_CB + c, col = cb * PFL_CB + c;
    const uint qh = head % PFL_HK;
    float s[PFL_RPG];
    device float* base = state + ((ulong) (rg * PFL_RPG) * PFL_HV + head) * PFL_S + col;
    const ulong rs = (ulong) PFL_HV * PFL_S;
    for (int r = 0; r < PFL_RPG; ++r) s[r] = base[(ulong) r * rs];
    for (long t = 0; t < T; ++t) {
        constant const float* ht = h + (ulong) t * PFL_C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < PFL_S) { sq[tid] = ht[qh * PFL_S + tid]; sk[tid] = ht[PFL_HK * PFL_S + qh * PFL_S + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float g = metal::precise::exp(gate[(ulong) t * PFL_HV + head]);
        float kv = 0.0f;
        for (int r = 0; r < PFL_RPG; ++r) kv = fma(s[r], sk[rg * PFL_RPG + r], kv);
        red[rg][c] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (ht[2 * PFL_HK * PFL_S + head * PFL_S + col] - g * kv_col) *
                            beta[(ulong) t * PFL_HV + head];
        float o = 0.0f;
        for (int r = 0; r < PFL_RPG; ++r) {
            s[r] = fma(g, s[r], sk[rg * PFL_RPG + r] * delta);
            o = fma(s[r], sq[rg * PFL_RPG + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg][c] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0u)
            oc_out[(ulong) t * PFL_HV * PFL_S + head * PFL_S + col] =
                (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * metal::precise::rsqrt((float) PFL_S);
    }
    for (int r = 0; r < PFL_RPG; ++r) base[(ulong) r * rs] = s[r];
}

// pfl_gdn_rec_cols_kernel with the next token's inputs loaded while this token computes (software
// pipelining; the same arithmetic in the same order: the same bits).  STRATA_GDN_PIPELINE=0: the plain one.
kernel void pfl_gdn_rec_cols_pipe_kernel(device float* state [[buffer(0)]],
                                         constant const float* h [[buffer(1)]],
                                         constant const float* gate [[buffer(2)]],
                                         constant const float* beta [[buffer(3)]],
                                         device float* oc_out [[buffer(4)]],
                                         constant const long& T [[buffer(5)]],
                                         uint2 gpos [[threadgroup_position_in_grid]],
                                         uint2 tid2 [[thread_position_in_threadgroup]]) {
    constexpr int NT = PFL_CB * PFL_RG, LPT = PFL_S / NT;   // 128 threads, 1 q/k row each
    threadgroup float sk[PFL_S], sq[PFL_S], red[PFL_RG][PFL_CB];
    const uint head = gpos.x / PFL_NCB, cb = gpos.x % PFL_NCB;
    const uint c = tid2.x, rg = tid2.y, tid = rg * PFL_CB + c, col = cb * PFL_CB + c;
    const uint qh = head % PFL_HK;
    float s[PFL_RPG];
    device float* base = state + ((ulong) (rg * PFL_RPG) * PFL_HV + head) * PFL_S + col;
    const ulong rs = (ulong) PFL_HV * PFL_S;
    for (int r = 0; r < PFL_RPG; ++r) s[r] = base[(ulong) r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    for (long t = 0; t < T + 1; ++t) {           // fetch(0) .. fetch(T-1): the loop's first iteration loads t=0
        if (t == 0) {
            if (T > 0) {
                constant const float* ht = h;
                for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * PFL_S + tid + u * NT]; nk[u] = ht[PFL_HK * PFL_S + qh * PFL_S + tid + u * NT]; }
                nv = ht[2 * PFL_HK * PFL_S + head * PFL_S + col];
                ng = gate[head];
                nb = beta[head];
            }
            continue;
        }
        const long cur = t - 1;
        float cq[LPT], ck[LPT];
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cur + 1 < T) {
            constant const float* ht = h + (ulong) (cur + 1) * PFL_C;
            for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * PFL_S + tid + u * NT]; nk[u] = ht[PFL_HK * PFL_S + qh * PFL_S + tid + u * NT]; }
            nv = ht[2 * PFL_HK * PFL_S + head * PFL_S + col];
            ng = gate[(ulong) (cur + 1) * PFL_HV + head];
            nb = beta[(ulong) (cur + 1) * PFL_HV + head];
        }
        const float g = metal::precise::exp(cg);
        float kv = 0.0f;
        for (int r = 0; r < PFL_RPG; ++r) kv = fma(s[r], sk[rg * PFL_RPG + r], kv);
        red[rg][c] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
        for (int r = 0; r < PFL_RPG; ++r) {
            s[r] = fma(g, s[r], sk[rg * PFL_RPG + r] * delta);
            o = fma(s[r], sq[rg * PFL_RPG + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg][c] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0u)
            oc_out[(ulong) cur * PFL_HV * PFL_S + head * PFL_S + col] =
                (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * metal::precise::rsqrt((float) PFL_S);
    }
    for (int r = 0; r < PFL_RPG; ++r) base[(ulong) r * rs] = s[r];
}

kernel void pfl_gdn_out_norm_kernel(constant const float* z [[buffer(0)]],
                                    constant const float* gamma [[buffer(1)]],
                                    constant const float& eps [[buffer(2)]],
                                    device float* y [[buffer(3)]],
                                    device ushort* y16 [[buffer(4)]],
                                    uint3 gpos [[threadgroup_position_in_grid]],   // x: token, y: head
                                    uint col [[thread_index_in_threadgroup]]) {
    threadgroup float wsum[4];
    const ulong t = gpos.x;
    const uint head = gpos.y;
    const ulong at = t * PFL_HV * PFL_S + (ulong) head * PFL_S + col;
    const float oc = y[at];
    float sp = pfl_warp_sum(oc * oc);
    if ((col & 31u) == 0u) wsum[col >> 5u] = sp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * metal::precise::rsqrt(ss / (float) PFL_S + eps) * gamma[col] *
                    pfl_sigm(z[t * PFL_HV * PFL_S + head * PFL_S + col]);
    y[at] = v;
    y16[at] = (ushort) pfl_hf(v);
}

// ---- MoE ------------------------------------------------------------------------------------

kernel void pfl_iota_kernel(device int* dst [[buffer(0)]], constant uint& n [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = (int)i;
}

// route_kernel<REG>: one warp per token, REG as a runtime scalar (the NCOLS/NW precedent - a column's
// accumulation chain never mixes registers).  v[16] is the CUDA file's largest instantiation.
kernel void pfl_route_kernel(constant const float* logits [[buffer(0)]],
                             device int* ids [[buffer(1)]],
                             device float* wout [[buffer(2)]],
                             constant const long& T [[buffer(3)]],
                             constant const int& reg [[buffer(4)]],
                             uint3 gpos [[threadgroup_position_in_grid]],    // blockIdx.x: the token's warp
                             uint tid [[thread_index_in_threadgroup]]) {
    const long t = (long) gpos.x * 8 + (long) (tid >> 5);
    if (t >= T) return;
    const uint lane = tid & 31u;
    constant const float* lg = logits + (ulong) t * (reg * 32);
    float v[16];
    for (int i = 0; i < reg; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
    for (int i = 0; i < reg; ++i) mx = fmax(mx, v[i]);
    mx = pfl_warp_max(mx);
    float sum = 0.0f;
    for (int i = 0; i < reg; ++i) {
        v[i] = metal::precise::exp(v[i] - mx);                     // the .cu's plain expf
        sum += v[i];
    }
    const float rcp = 1.0f / pfl_warp_sum(sum);
    for (int i = 0; i < reg; ++i) {
        v[i] *= rcp;
        if (isnan(v[i])) v[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = (int) lane;
        for (int i = 1; i < reg; ++i)
            if (v[i] > best) { best = v[i]; ex = (int) lane + i * 32; }
        for (int m = 16; m > 0; m >>= 1) {
            const float ob = simd_shuffle_xor(best, (uint) m);
            const int oi = simd_shuffle_xor(ex, (uint) m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == (int) lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0u) ids[(ulong) t * 10 + rank] = ex;
        if (rank == (int) lane) selected = best;
    }
    selected_sum = fmax(pfl_warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10u) wout[(ulong) t * 10 + lane] = selected / selected_sum;
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16,
// down scales [2560][10] f16.  One thread per 4 weights (one code byte); HALF is the CUDA template flag.
kernel void pfl_blob_dequant_kernel(constant const uint8_t* blob [[buffer(0)]],
                                    device ushort* gu16 [[buffer(1)]],
                                    device ushort* d16 [[buffer(2)]],
                                    constant const int& half_out [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                        O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const long n_gu = 1280L * 640, n_d = 2560L * 160;
    if (i < (uint) n_gu) {
        const ulong row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        constant const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const uint16_t raw = (uint16_t) (sp[0] | (sp[1] << 8));
        const float d = f32_from_f16(raw);
        device ushort* o = gu16 + row * 2560 + byte * 4;
        for (int k = 0; k < 4; ++k) {
            const float v = (float) (((c >> (2 * k)) & 3) - 1) * d;
            o[k] = (ushort) (half_out ? pfl_hf(v) : pfl_bf(v));
        }
    } else if (i < (uint) (n_gu + n_d)) {
        const ulong j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        constant const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const uint16_t raw = (uint16_t) (sp[0] | (sp[1] << 8));
        const float d = f32_from_f16(raw);
        device ushort* o = d16 + row * 640 + byte * 4;
        for (int k = 0; k < 4; ++k) {
            const float v = (float) (((c >> (2 * k)) & 3) - 1) * d;
            o[k] = (ushort) (half_out ? pfl_hf(v) : pfl_bf(v));
        }
    }
}

kernel void pfl_swiglu_il_kernel(constant const float* gu [[buffer(0)]],
                                 device ushort* h16 [[buffer(1)]],
                                 constant const long& n [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) n * 640) return;
    const ulong r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = (ushort) pfl_hf_sat(g / (1.0f + metal::precise::exp(-g)) * u);
}

kernel void pfl_swiglu_pair_kernel(constant const float* g [[buffer(0)]],
                                   constant const float* u [[buffer(1)]],
                                   device ushort* h16 [[buffer(2)]],
                                   constant const long& n [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) n * 640) return;
    const float a = g[i];
    h16[i] = (ushort) pfl_hf_sat(a / (1.0f + metal::precise::exp(-a)) * u[i]);
}

kernel void pfl_gather_rows16_kernel(constant const uint4* x [[buffer(0)]],
                                     constant const int* src [[buffer(1)]],
                                     device uint4* dst [[buffer(2)]],
                                     constant const long& n [[buffer(3)]],
                                     constant const long& width [[buffer(4)]],
                                     uint i [[thread_position_in_grid]]) {   // one uint4 (8 bf16/f16)
    const long per = width / 8;
    if (i >= (ulong) n * per) return;
    const ulong r = i / per, j = i % per;
    dst[r * per + j] = x[(ulong) src[r] * per + j];
}

kernel void pfl_moe_combine_kernel(constant const float* Dm [[buffer(0)]],
                                   constant const int* slot [[buffer(1)]],
                                   constant const float* w [[buffer(2)]],
                                   constant const float* shared [[buffer(3)]],
                                   constant const float* sg [[buffer(4)]],
                                   device float* bo [[buffer(5)]],
                                   constant const long& T [[buffer(6)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * PFL_N) return;
    const ulong t = i / PFL_N, d = i % PFL_N;
    float s = 0.0f;
    for (int k = 0; k < 10; ++k)
        s = fma(w[t * 10 + k], Dm[(ulong) slot[t * 10 + k] * PFL_N + d], s);
    bo[i] = s + shared[i] * pfl_sigm(sg[t]);
}

// ---- QSA helpers ------------------------------------------------------------------------------

kernel void pfl_rms_rows_kernel(device float* x [[buffer(0)]],
                                constant const float* w [[buffer(1)]],
                                constant const long& cols [[buffer(2)]],
                                constant const long& ld [[buffer(3)]],
                                constant const float& eps [[buffer(4)]],
                                uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x: the row
                                uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    device float* r = x + (ulong) gpos.x * ld;
    float ss = 0.0f;
    for (long c = tid; c < cols; c += 256) ss += r[c] * r[c];
    const float s = metal::precise::rsqrt(pfl_block_sum(ss, sh, tid, 256u) / (float) cols + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (long c = tid; c < cols; c += 256) r[c] = s * r[c] * w[c];
}

// mrope.hpp's mrope_pos and rope_tab_cs (CUDA-only there; the transcription matches rope.metal /
// native_rope.metal) and rope_scaling.hpp's rope_scaled_angle (native_rope.metal's copy)
static inline int pfl_mrope_pos(constant const int* tab, int pos, int pair) {
    return tab != nullptr ? tab[(ulong) pos * 3 + (uint) pair % 3] : pos;
}
static inline bool pfl_rope_tab_cs(constant const float* cos_tab, constant const float* sin_tab, int max_pos,
                                   int p, int pair, thread float& c, thread float& s) {
    if (cos_tab == nullptr || p < 0 || p >= max_pos) return false;
    c = cos_tab[(ulong) p * 32 + (uint) pair];
    s = sin_tab[(ulong) p * 32 + (uint) pair];
    return true;
}
static inline float pfl_rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}
static inline void pfl_rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low, float corr_high,
                                         float ext_factor, float mscale_in, int pair, thread float& cos_out,
                                         thread float& sin_out) {
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = pfl_rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * metal::precise::log(1.0f / freq_scale);
    }
    cos_out = metal::precise::cos(theta) * mscale;
    sin_out = metal::precise::sin(theta) * mscale;
}

// TAB (#280, STRATA_ROPE_TABLE=1): the CUDA template <bool TAB> is the runtime flag `use_tab` - the
// table read is behind it, so the default path is the same instructions the .cu's <false> compiled to.
// The by-value RopeTab is RULE 9's case: its two pointers arrive as bound (possibly nil) buffers.
kernel void pfl_rope_kernel(device float* x [[buffer(0)]],
                            constant const long& heads [[buffer(1)]],
                            constant const long& dim [[buffer(2)]],
                            constant const long& ld [[buffer(3)]],
                            constant const long& pos0 [[buffer(4)]],
                            constant const float& theta_scale [[buffer(5)]],
                            constant const float& freq_scale [[buffer(6)]],
                            constant const float& corr_low [[buffer(7)]],
                            constant const float& corr_high [[buffer(8)]],
                            constant const float& ext_factor [[buffer(9)]],
                            constant const float& mscale [[buffer(10)]],
                            constant const int* mtab [[buffer(11)]],       // may bind nil: no image path
                            constant const int& use_tab [[buffer(12)]],
                            constant const float* tab_cos [[buffer(13)]],  // may bind nil: no table
                            constant const float* tab_sin [[buffer(14)]],
                            constant const int& tab_max_pos [[buffer(15)]],
                            uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x = row = t*heads + h
                            uint pair [[thread_index_in_threadgroup]]) {   // 0..31
    const ulong row = gpos.x;
    const ulong t = row / heads, h = row % heads;
    device float* p = x + t * (ulong) ld + h * dim;
    float c, s;
    if (!(use_tab != 0 && pfl_rope_tab_cs(tab_cos, tab_sin, tab_max_pos,
                                          pfl_mrope_pos(mtab, (int) (pos0 + t), (int) pair), (int) pair, c, s))) {
        const float theta_extrap =
            (float) pfl_mrope_pos(mtab, (int) (pos0 + t), (int) pair) * metal::precise::pow(theta_scale, (float) pair);
        pfl_rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, (int) pair, c, s);
    }
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}

kernel void pfl_split_q_kernel(constant const float* qf [[buffer(0)]],
                               device float* q [[buffer(1)]],
                               constant const long& T [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * 24 * 256) return;
    const ulong t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}

kernel void pfl_gate_attn_kernel(constant const float* a [[buffer(0)]],
                                 constant const float* qf [[buffer(1)]],
                                 device ushort* o16 [[buffer(2)]],
                                 constant const long& T [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= (ulong) T * 24 * 256) return;
    const ulong t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = (ushort) pfl_hf(a[i] * (1.0f / (1.0f + metal::precise::exp(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// ---- KV append: one block per (token, kv head, 64-value group); the two by-value KvHostPools are
// RULE 9's other case: sixteen bound buffers in the struct's declaration order, nulls binding nil
// (the kv_q8 port's idiom).  Only the f16/int8 fields are read here - q4 rides kv_append_q4.
kernel void pfl_kv_append_kernel(constant const float* K [[buffer(0)]],
                                 constant const float* V [[buffer(1)]],
                                 constant const long& pos0 [[buffer(2)]],
                                 constant const int* table [[buffer(3)]],
                                 constant const long& page_size [[buffer(4)]],
                                 device ushort* k_pool [[buffer(5)]],
                                 device ushort* v_pool [[buffer(6)]],
                                 device int8_t* k_q [[buffer(7)]],
                                 device int8_t* v_q [[buffer(8)]],
                                 device ushort* k_scale [[buffer(9)]],
                                 device ushort* v_scale [[buffer(10)]],
                                 device ushort* host_k_pool [[buffer(11)]],
                                 device ushort* host_v_pool [[buffer(12)]],
                                 device int8_t* host_k_q [[buffer(13)]],
                                 device int8_t* host_v_q [[buffer(14)]],
                                 device ushort* host_k_scale [[buffer(15)]],
                                 device ushort* host_v_scale [[buffer(16)]],
                                 device uint8_t* host_k_q4 [[buffer(17)]],
                                 device uint8_t* host_v_q4 [[buffer(18)]],
                                 device ushort* stage_k_pool [[buffer(19)]],
                                 device ushort* stage_v_pool [[buffer(20)]],
                                 device int8_t* stage_k_q [[buffer(21)]],
                                 device int8_t* stage_v_q [[buffer(22)]],
                                 device ushort* stage_k_scale [[buffer(23)]],
                                 device ushort* stage_v_scale [[buffer(24)]],
                                 device uint8_t* stage_k_q4 [[buffer(25)]],
                                 device uint8_t* stage_v_q4 [[buffer(26)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],   // (t, kvh, 2g+is_v)
                                 uint tid [[thread_index_in_threadgroup]]) {
    const ulong t = gpos.x;
    const uint kvh = gpos.y, g = gpos.z >> 1;
    const bool is_v = (gpos.z & 1) != 0;
    const uint d = g * 64u + tid;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const ulong pos = pos0 + t;
    const long page = table[pos / (ulong) page_size];
    const ulong row = ((ulong) page * 2 + kvh) * (ulong) page_size + pos % (ulong) page_size;
    const ulong row_id = ((pos / (ulong) page_size) * 2 + kvh) * (ulong) page_size + pos % (ulong) page_size;
    if (k_pool != nullptr) {
        const uint h = pfl_hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = (ushort) h;
        if (host_k_pool != nullptr) (is_v ? host_v_pool : host_k_pool)[row_id * 256 + d] = (ushort) h;
        if (stage_k_pool != nullptr) (is_v ? stage_v_pool : stage_k_pool)[row_id * 256 + d] = (ushort) h;
        return;
    }
    float a = fabs(x);
    for (uint o = 16u; o > 0u; o >>= 1u) a = fmax(a, simd_shuffle_xor(a, o));
    threadgroup float wm[2];
    if ((tid & 31u) == 0u) wm[tid >> 5u] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = fmax(wm[0], wm[1]);
    const uint sb = pfl_hf(amax / 127.0f);
    const float sf = f32_from_f16(sb);
    int q = 0;
    if (sf > 0.0f) {
        q = (int) metal::precise::rint(x / sf);                       // __float2int_rn: round to nearest even
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (tid == 0u) (is_v ? v_scale : k_scale)[row * 4 + g] = (ushort) sb;
    }
    if (host_k_q != nullptr) {
        (is_v ? host_v_q : host_k_q)[row_id * 256 + d] = (int8_t) q;
        if (tid == 0u) (is_v ? host_v_scale : host_k_scale)[row_id * 4 + g] = (ushort) sb;
    }
    if (stage_k_q != nullptr) {
        (is_v ? stage_v_q : stage_k_q)[row_id * 256 + d] = (int8_t) q;
        if (tid == 0u) (is_v ? stage_v_scale : stage_k_scale)[row_id * 4 + g] = (ushort) sb;
    }
}

// ---- the activation images (grid-stride, the .cu's capped grids) -------------------------------

kernel void pfl_to_f16_kernel(constant const float* x [[buffer(0)]],
                              device ushort* y [[buffer(1)]],
                              constant const ulong& n [[buffer(2)]],
                              uint i [[thread_position_in_grid]],
                              uint ng [[threads_per_grid]]) {
    for (ulong k = i; k < n; k += ng) y[k] = (ushort) pfl_hf(x[k]);
}

kernel void pfl_round_f16_kernel(constant const float* x [[buffer(0)]],
                                 device float* y [[buffer(1)]],
                                 constant const ulong& n [[buffer(2)]],
                                 uint i [[thread_position_in_grid]],
                                 uint ng [[threads_per_grid]]) {
    for (ulong k = i; k < n; k += ng) y[k] = f32_from_f16(pfl_hf(x[k]));
}

kernel void pfl_to_bf16_kernel(constant const float* x [[buffer(0)]],
                               device ushort* y [[buffer(1)]],
                               device ushort* ylo [[buffer(2)]],
                               constant const ulong& n [[buffer(3)]],
                               uint i [[thread_position_in_grid]],
                               uint ng [[threads_per_grid]]) {
    for (ulong k = i; k < n; k += ng) {
        const uint h = pfl_bf(x[k]);
        y[k] = (ushort) h;
        if (ylo != nullptr) ylo[k] = (ushort) pfl_bf_lo(x[k], h);
    }
}

kernel void pfl_copy_i32_kernel(device int* dst [[buffer(0)]],
                                constant const int* src [[buffer(1)]],
                                constant const ulong& n [[buffer(2)]],
                                uint i [[thread_position_in_grid]],
                                uint ng [[threads_per_grid]]) {
    for (ulong k = i; k < n; k += ng) dst[k] = src[k];
}

// the bf16 GEMM's widen pass (src/prefill/gemm.mm): a BF16 bit pattern widened to FP32 is the same bits
// shifted left 16 - exact, NaN patterns included
kernel void pfl_widen_f32_kernel(constant const ushort* x [[buffer(0)]],
                                 device float* y [[buffer(1)]],
                                 constant const ulong& n [[buffer(2)]],
                                 uint i [[thread_position_in_grid]],
                                 uint ng [[threads_per_grid]]) {
    for (ulong k = i; k < n; k += ng) y[k] = as_type<float>((uint) x[k] << 16);
}
