// Prefill expert GEMM variants. The production file is included verbatim (iq_kernels.metal: iqk_gemm). A variant
// keeps every output element's MMA sequence: the same half operands (x rows; weights decoded by the same
// iqk_dq_dispatch into the same 32-value k-slices), 8-wide k steps ascending from a zero accumulator, stored as is.
// Only the tile (BM rows of one expert x BN columns), the SIMD-group split and the epilogue staging change.
#include "iq_kernels.metal"

// IqTileStoreN: production (iq_kernels.metal, added with iqk_gemm2)
template<int TY, bool DOWN, int BM, int BN, int SGM, int SGN>
inline void mg_gemm(constant const ushort* x, constant const uint8_t* arena, constant const int4* tiles, device float* y,
                    long H, long FF, ulong row_bytes, ulong weight_offset, uint2 tile, uint tid, uint sg, uint lane,
                    threadgroup half* sx, threadgroup half* sw, threadgroup float* scratch) {
    constexpr int TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    const uint K = (uint) (DOWN ? FF : H), N = (uint) (DOWN ? H : 2 * FF);
    const int4 desc = tiles[tile.y];
    const ulong offset = ulong(uint(desc.x)) | (ulong(uint(desc.w)) << 32);
    constant const uint8_t* blob = arena + offset;
    const uint row0 = uint(desc.y), rows = uint(desc.z), nr = tile.x * BN;
    const uint sm = (sg / SGN) * (BM / SGM), sn = (sg % SGN) * (BN / SGN);
    simdgroup_float8x8 c[MI][NJ];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) c[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < BM * 32; i += TH)
            sx[i] = i / 32 < rows ? as_type<half>(x[(ulong) (row0 + i / 32) * K + k0 + i % 32]) : half(0);
        const uint within = k0 % 256;
        for (uint j = tid; j < (uint) BN * 4; j += TH) {
            const uint r = j / 4, n = nr + r, quarter = j % 4;
            constant const uint8_t* w = blob + (DOWN ? weight_offset : (n & 1) * weight_offset)
                                       + (ulong) (DOWN ? n : n / 2) * row_bytes;
            const uint dq_tid = TY == 42 ? (within / 64) * 8 + (within % 64) / 8 + quarter : quarter * 8 + within / 32;
            IqTileStoreN store{sw, (long) r, (long) within, 0, BN};
            iqk_dq_dispatch<half>(TY, w, k0 / 256, store, dq_tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a[MI], b[NJ];
#pragma unroll
            for (int i = 0; i < MI; ++i) simdgroup_load(a[i], sx + (sm + i * 8) * 32 + k, 32);
#pragma unroll
            for (int j = 0; j < NJ; ++j) simdgroup_load(b[j], sw + k * BN + sn + j * 8, BN);
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
                const uint m = sm + i * 8 + e / 8, n = nr + sn + j * 8 + e % 8;
                if (m < rows) y[(ulong) (row0 + m) * N + n] = mine[e];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
#define MG(NAME, TY, DOWN, BM, BN, SGM, SGN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], constant const uint8_t* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[BM * 32], sw[32 * BN]; threadgroup float scratch[SGM * SGN * 64]; \
    mg_gemm<TY, DOWN, BM, BN, SGM, SGN>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); }
// mg_<gu|dn><type>_<BM>x<BN>_<SGM><SGN>
#define MG_SET(T, D, TAG) \
    MG(mg_##TAG##_16x32_22, T, D, 16, 32, 2, 2) MG(mg_##TAG##_16x64_22, T, D, 16, 64, 2, 2) \
    MG(mg_##TAG##_32x32_22, T, D, 32, 32, 2, 2) MG(mg_##TAG##_32x64_22, T, D, 32, 64, 2, 2) \
    MG(mg_##TAG##_32x64_24, T, D, 32, 64, 2, 4) MG(mg_##TAG##_64x64_24, T, D, 64, 64, 2, 4) \
    MG(mg_##TAG##_16x128_14, T, D, 16, 128, 1, 4) MG(mg_##TAG##_32x128_24, T, D, 32, 128, 2, 4)
MG_SET(22, false, gu22) MG_SET(42, true, dn42) MG_SET(16, false, gu16)
