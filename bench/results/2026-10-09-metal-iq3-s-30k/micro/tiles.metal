#include "iq_kernels.metal"
template<int TY, bool DOWN, int BM, int BN, int SGM, int SGN, int BK>
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
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i = tid; i < BM * BK; i += TH)
            sx[i] = i / BK < rows ? as_type<half>(x[(ulong) (row0 + i / BK) * K + k0 + i % BK]) : half(0);
        const uint within = k0 % 256;
        for (uint j = tid; j < (uint) BN * (BK / 8); j += TH) {
            const uint r = j / (BK / 8), n = nr + r, part = j % (BK / 8);
            const uint slice = within + (part / 4) * 32, quarter = part % 4;
            constant const uint8_t* w = blob + (DOWN ? weight_offset : (n & 1) * weight_offset)
                                       + (ulong) (DOWN ? n : n / 2) * row_bytes;
            const uint dq_tid = TY == 42 ? (slice / 64) * 8 + (slice % 64) / 8 + quarter : quarter * 8 + slice / 32;
            IqTileStoreN store{sw, (long) r, (long) within, 0, BN};
            iqk_dq_dispatch<half>(TY, w, k0 / 256, store, dq_tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < BK; k += 8) {
            simdgroup_half8x8 a[MI], b[NJ];
#pragma unroll
            for (int i = 0; i < MI; ++i) simdgroup_load(a[i], sx + (sm + i * 8) * BK + k, BK);
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

#define MG(NAME, TY, DOWN, BM, BN, SGM, SGN, BK) \
kernel void NAME(constant const ushort* x [[buffer(0)]], constant const uint8_t* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[BM * BK], sw[BK * BN]; threadgroup float scratch[SGM * SGN * 64]; \
    mg_gemm<TY, DOWN, BM, BN, SGM, SGN, BK>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); }

MG(mg_gu18_16x64_22_k32, 18, false, 16, 64, 2, 2, 32)
MG(mg_gu18_16x64_22_k64, 18, false, 16, 64, 2, 2, 64)
MG(mg_gu18_16x64_22_k128, 18, false, 16, 64, 2, 2, 128)
MG(mg_gu18_32x64_24_k32, 18, false, 32, 64, 2, 4, 32)
MG(mg_gu18_32x64_24_k64, 18, false, 32, 64, 2, 4, 64)
MG(mg_gu18_32x64_24_k128, 18, false, 32, 64, 2, 4, 128)
MG(mg_gu18_16x128_14_k32, 18, false, 16, 128, 1, 4, 32)
MG(mg_gu18_16x128_14_k64, 18, false, 16, 128, 1, 4, 64)
MG(mg_gu18_16x128_14_k128, 18, false, 16, 128, 1, 4, 128)
MG(mg_gu18_32x64_22_k32, 18, false, 32, 64, 2, 2, 32)
MG(mg_gu18_32x64_22_k64, 18, false, 32, 64, 2, 2, 64)
MG(mg_gu18_32x64_22_k128, 18, false, 32, 64, 2, 2, 128)
MG(mg_gu21_16x64_22_k32, 21, false, 16, 64, 2, 2, 32)
MG(mg_gu21_16x64_22_k64, 21, false, 16, 64, 2, 2, 64)
MG(mg_gu21_16x64_22_k128, 21, false, 16, 64, 2, 2, 128)
MG(mg_gu21_32x64_24_k32, 21, false, 32, 64, 2, 4, 32)
MG(mg_gu21_32x64_24_k64, 21, false, 32, 64, 2, 4, 64)
MG(mg_gu21_32x64_24_k128, 21, false, 32, 64, 2, 4, 128)
MG(mg_gu21_16x128_14_k32, 21, false, 16, 128, 1, 4, 32)
MG(mg_gu21_16x128_14_k64, 21, false, 16, 128, 1, 4, 64)
MG(mg_gu21_16x128_14_k128, 21, false, 16, 128, 1, 4, 128)
MG(mg_gu21_32x64_22_k32, 21, false, 32, 64, 2, 2, 32)
MG(mg_gu21_32x64_22_k64, 21, false, 32, 64, 2, 2, 64)
MG(mg_gu21_32x64_22_k128, 21, false, 32, 64, 2, 2, 128)
MG(mg_gu22_16x64_22_k32, 22, false, 16, 64, 2, 2, 32)
MG(mg_gu22_16x64_22_k64, 22, false, 16, 64, 2, 2, 64)
MG(mg_gu22_16x64_22_k128, 22, false, 16, 64, 2, 2, 128)
MG(mg_gu22_32x64_24_k32, 22, false, 32, 64, 2, 4, 32)
MG(mg_gu22_32x64_24_k64, 22, false, 32, 64, 2, 4, 64)
MG(mg_gu22_32x64_24_k128, 22, false, 32, 64, 2, 4, 128)
MG(mg_gu22_16x128_14_k32, 22, false, 16, 128, 1, 4, 32)
MG(mg_gu22_16x128_14_k64, 22, false, 16, 128, 1, 4, 64)
MG(mg_gu22_16x128_14_k128, 22, false, 16, 128, 1, 4, 128)
MG(mg_gu22_32x64_22_k32, 22, false, 32, 64, 2, 2, 32)
MG(mg_gu22_32x64_22_k64, 22, false, 32, 64, 2, 2, 64)
MG(mg_gu22_32x64_22_k128, 22, false, 32, 64, 2, 2, 128)
MG(mg_gu23_16x64_22_k32, 23, false, 16, 64, 2, 2, 32)
MG(mg_gu23_16x64_22_k64, 23, false, 16, 64, 2, 2, 64)
MG(mg_gu23_16x64_22_k128, 23, false, 16, 64, 2, 2, 128)
MG(mg_gu23_32x64_24_k32, 23, false, 32, 64, 2, 4, 32)
MG(mg_gu23_32x64_24_k64, 23, false, 32, 64, 2, 4, 64)
MG(mg_gu23_32x64_24_k128, 23, false, 32, 64, 2, 4, 128)
MG(mg_gu23_16x128_14_k32, 23, false, 16, 128, 1, 4, 32)
MG(mg_gu23_16x128_14_k64, 23, false, 16, 128, 1, 4, 64)
MG(mg_gu23_16x128_14_k128, 23, false, 16, 128, 1, 4, 128)
MG(mg_gu23_32x64_22_k32, 23, false, 32, 64, 2, 2, 32)
MG(mg_gu23_32x64_22_k64, 23, false, 32, 64, 2, 2, 64)
MG(mg_gu23_32x64_22_k128, 23, false, 32, 64, 2, 2, 128)
MG(mg_dn20_16x64_22_k32, 20, true, 16, 64, 2, 2, 32)
MG(mg_dn20_16x64_22_k64, 20, true, 16, 64, 2, 2, 64)
MG(mg_dn20_16x64_22_k128, 20, true, 16, 64, 2, 2, 128)
MG(mg_dn20_32x64_24_k32, 20, true, 32, 64, 2, 4, 32)
MG(mg_dn20_32x64_24_k64, 20, true, 32, 64, 2, 4, 64)
MG(mg_dn20_32x64_24_k128, 20, true, 32, 64, 2, 4, 128)
MG(mg_dn20_16x128_14_k32, 20, true, 16, 128, 1, 4, 32)
MG(mg_dn20_16x128_14_k64, 20, true, 16, 128, 1, 4, 64)
MG(mg_dn20_16x128_14_k128, 20, true, 16, 128, 1, 4, 128)
MG(mg_dn20_32x64_22_k32, 20, true, 32, 64, 2, 2, 32)
MG(mg_dn20_32x64_22_k64, 20, true, 32, 64, 2, 2, 64)
MG(mg_dn20_32x64_22_k128, 20, true, 32, 64, 2, 2, 128)
MG(mg_dn42_16x64_22_k32, 42, true, 16, 64, 2, 2, 32)
MG(mg_dn42_16x64_22_k64, 42, true, 16, 64, 2, 2, 64)
MG(mg_dn42_16x64_22_k128, 42, true, 16, 64, 2, 2, 128)
MG(mg_dn42_32x64_24_k32, 42, true, 32, 64, 2, 4, 32)
MG(mg_dn42_32x64_24_k64, 42, true, 32, 64, 2, 4, 64)
MG(mg_dn42_32x64_24_k128, 42, true, 32, 64, 2, 4, 128)
MG(mg_dn42_16x128_14_k32, 42, true, 16, 128, 1, 4, 32)
MG(mg_dn42_16x128_14_k64, 42, true, 16, 128, 1, 4, 64)
MG(mg_dn42_16x128_14_k128, 42, true, 16, 128, 1, 4, 128)
MG(mg_dn42_32x64_22_k32, 42, true, 32, 64, 2, 2, 32)
MG(mg_dn42_32x64_22_k64, 42, true, 32, 64, 2, 2, 64)
MG(mg_dn42_32x64_22_k128, 42, true, 32, 64, 2, 2, 128)