// Prefill GEMM variants. The production kernels are included verbatim (prefill.metal). Every output element of a
// variant gets the same sequence of 8x8x8 SIMD-group MMAs as pfl_matrix_product: operands of the same type (half
// for FP16, float for BF16), the same values (k-chunks of 8, ascending, zero accumulator start), then the same
// beta epilogue. Only the output tile per threadgroup / SIMD group and the operand staging change.
#include "prefill.metal"

// BM x BN per threadgroup, 4 SIMD groups in a 2 x 2 grid, each (BM/2) x (BN/2) = MI x NJ 8x8 accumulators.
// Operands staged as [row][32 k] (x) and [col][32 k] (w), 4-element vector loads; b is loaded transposed.
template<typename S, bool BF16, int BM, int BN>
inline void pg_product(const device ushort* x, const device ushort* w, device float* y, uint M, uint N, uint K,
                       uint ldy, float beta, uint2 tile, uint tid, uint sg, uint lane,
                       threadgroup S* sx, threadgroup S* sw, threadgroup float* scratch) {
    constexpr int MI = BM / 16, NJ = BN / 16;
    simdgroup_float8x8 c[MI][NJ];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) c[i][j] = simdgroup_float8x8(0.0f);
    const uint mr = tile.y * BM, nr = tile.x * BN;
    const uint sm = (sg / 2) * (BM / 2), sn = (sg % 2) * (BN / 2);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < BM * 8; i += 128) {
            const uint m = i / 8, kk = (i % 8) * 4;
            ushort4 v = ushort4(0);
            if (mr + m < M) v = *(const device ushort4*) (x + (ulong) (mr + m) * K + k0 + kk);
            threadgroup S* d = sx + m * 32 + kk;
#pragma unroll
            for (int q = 0; q < 4; ++q) d[q] = BF16 ? S(as_type<float>(uint(v[q]) << 16)) : S(as_type<half>(v[q]));
        }
        for (uint i = tid; i < BN * 8; i += 128) {
            const uint n = i / 8, kk = (i % 8) * 4;
            ushort4 v = ushort4(0);
            if (nr + n < N) v = *(const device ushort4*) (w + (ulong) (nr + n) * K + k0 + kk);
            threadgroup S* d = sw + n * 32 + kk;
#pragma unroll
            for (int q = 0; q < 4; ++q) d[q] = BF16 ? S(as_type<float>(uint(v[q]) << 16)) : S(as_type<half>(v[q]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
#define PG_KERNEL(NAME, ST, BF, BM, BN) \
kernel void NAME(const device ushort* x [[buffer(0)]], const device ushort* w [[buffer(1)]], \
                 device float* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], \
                 constant uint& K [[buffer(5)]], constant uint& ldy [[buffer(6)]], constant float& beta [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ST sx[BM * 32], sw[BN * 32]; threadgroup float scratch[4 * 64]; \
    pg_product<ST, BF, BM, BN>(x, w, y, M, N, K, ldy, beta, tile, tid, sg, lane, sx, sw, scratch); }
PG_KERNEL(pg_f16_64x64, half, false, 64, 64) PG_KERNEL(pg_f16_32x64, half, false, 32, 64)
PG_KERNEL(pg_f16_64x32, half, false, 64, 32) PG_KERNEL(pg_f16_32x32, half, false, 32, 32)
PG_KERNEL(pg_bf16_64x64, float, true, 64, 64) PG_KERNEL(pg_bf16_32x64, float, true, 32, 64)
PG_KERNEL(pg_bf16_64x32, float, true, 64, 32) PG_KERNEL(pg_bf16_32x32, float, true, 32, 32)

// SGM x SGN SIMD groups (32 * SGM * SGN threads), each (BM/SGM) x (BN/SGN); same per-element MMA sequence.
template<typename S, bool BF16, int BM, int BN, int SGM, int SGN>
inline void pg2_product(const device ushort* x, const device ushort* w, device float* y, uint M, uint N, uint K,
                        uint ldy, float beta, uint2 tile, uint tid, uint sg, uint lane,
                        threadgroup S* sx, threadgroup S* sw, threadgroup float* scratch) {
    constexpr int TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    simdgroup_float8x8 c[MI][NJ];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) c[i][j] = simdgroup_float8x8(0.0f);
    const uint mr = tile.y * BM, nr = tile.x * BN;
    const uint sm = (sg / SGN) * (BM / SGM), sn = (sg % SGN) * (BN / SGN);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < BM * 8; i += TH) {
            const uint m = i / 8, kk = (i % 8) * 4;
            ushort4 v = ushort4(0);
            if (mr + m < M) v = *(const device ushort4*) (x + (ulong) (mr + m) * K + k0 + kk);
            threadgroup S* d = sx + m * 32 + kk;
#pragma unroll
            for (int q = 0; q < 4; ++q) d[q] = BF16 ? S(as_type<float>(uint(v[q]) << 16)) : S(as_type<half>(v[q]));
        }
        for (uint i = tid; i < BN * 8; i += TH) {
            const uint n = i / 8, kk = (i % 8) * 4;
            ushort4 v = ushort4(0);
            if (nr + n < N) v = *(const device ushort4*) (w + (ulong) (nr + n) * K + k0 + kk);
            threadgroup S* d = sw + n * 32 + kk;
#pragma unroll
            for (int q = 0; q < 4; ++q) d[q] = BF16 ? S(as_type<float>(uint(v[q]) << 16)) : S(as_type<half>(v[q]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
#define PG2_KERNEL(NAME, ST, BF, BM, BN, SGM, SGN) \
kernel void NAME(const device ushort* x [[buffer(0)]], const device ushort* w [[buffer(1)]], \
                 device float* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], \
                 constant uint& K [[buffer(5)]], constant uint& ldy [[buffer(6)]], constant float& beta [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ST sx[BM * 32], sw[BN * 32]; threadgroup float scratch[SGM * SGN * 64]; \
    pg2_product<ST, BF, BM, BN, SGM, SGN>(x, w, y, M, N, K, ldy, beta, tile, tid, sg, lane, sx, sw, scratch); }
// names: pq_<type>_<BM>x<BN>_<SGM><SGN>
PG2_KERNEL(pq_f16_128x64_42, half, false, 128, 64, 4, 2) PG2_KERNEL(pq_f16_64x128_24, half, false, 64, 128, 2, 4)
PG2_KERNEL(pq_f16_128x128_44, half, false, 128, 128, 4, 4) PG2_KERNEL(pq_f16_64x64_22, half, false, 64, 64, 2, 2)
PG2_KERNEL(pq_bf16_128x32_42, float, true, 128, 32, 4, 2) PG2_KERNEL(pq_bf16_64x64_24, float, true, 64, 64, 2, 4)
PG2_KERNEL(pq_bf16_64x32_22, float, true, 64, 32, 2, 2) PG2_KERNEL(pq_bf16_128x64_44, float, true, 128, 64, 4, 4)

// pg2 with the next k-chunk's global loads issued (into registers) before the current chunk's MMAs
template<typename S, bool BF16, int BM, int BN, int SGM, int SGN>
inline void pp_product(const device ushort* x, const device ushort* w, device float* y, uint M, uint N, uint K,
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
#define PP_KERNEL(NAME, ST, BF, BM, BN, SGM, SGN) \
kernel void NAME(const device ushort* x [[buffer(0)]], const device ushort* w [[buffer(1)]], \
                 device float* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], \
                 constant uint& K [[buffer(5)]], constant uint& ldy [[buffer(6)]], constant float& beta [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ST sx[BM * 32], sw[BN * 32]; threadgroup float scratch[SGM * SGN * 64]; \
    pp_product<ST, BF, BM, BN, SGM, SGN>(x, w, y, M, N, K, ldy, beta, tile, tid, sg, lane, sx, sw, scratch); }
PP_KERNEL(pq_f16p_64x64_22, half, false, 64, 64, 2, 2) PP_KERNEL(pq_f16p_128x64_42, half, false, 128, 64, 4, 2)
PP_KERNEL(pq_bf16p_64x64_24, float, true, 64, 64, 2, 4) PP_KERNEL(pq_bf16p_64x32_22, float, true, 64, 32, 2, 2)
