#include "iq_kernels.metal"
template<int TY, bool DOWN, int BM, int BN, int SGM, int SGN>
inline void cg_gemm(constant const ushort* x, constant const half* arena, constant const int4* tiles, device float* y,
                    long H, long FF, ulong row_bytes, ulong weight_offset, ulong blob_bytes, uint2 tile, uint tid, uint sg, uint lane,
                    threadgroup half* sx, threadgroup half* sw, threadgroup float* scratch) {
    constexpr int TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    const uint K = (uint) (DOWN ? FF : H), N = (uint) (DOWN ? H : 2 * FF);
    const int4 desc = tiles[tile.y];
    const ulong offset = ulong(uint(desc.x)) | (ulong(uint(desc.w)) << 32);
    const ulong expert = offset / blob_bytes;
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
        for (uint j = tid; j < (uint) BN * 32; j += TH) {
            const uint n = nr + j / 32, k = j % 32;
            sw[k * BN + j / 32] = arena[(expert * N + n) * K + k0 + k];
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

#define CG(NAME, TY, DOWN, BM, BN, SGM, SGN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], constant const half* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 constant const ulong& blob_bytes [[buffer(8)]], uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[BM * 32], sw[32 * BN]; threadgroup float scratch[SGM * SGN * 64]; \
    cg_gemm<TY, DOWN, BM, BN, SGM, SGN>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, blob_bytes, tile, tid, sg, lane, sx, sw, scratch); }


struct CacheStore {
    device half* dst; long first, logical;
    CacheStore operator+(long off) const thread { return {dst, first, logical + off}; }
    device half& operator[](long i) const thread { return dst[logical + i - first]; }
};
template<int TY, bool DOWN>
inline void cg_unpack(constant const uint8_t* arena, device half* weights, long H, long FF, ulong rb, ulong wo, ulong blob,
                      uint2 gp, uint tid) {
    const uint K = DOWN ? FF : H, N = DOWN ? H : 2*FF;
    const uint c = gp.x*32 + tid/4, quarter=tid%4;
    if (c >= N*(K/32)) return;
    const uint row=c/(K/32), k0=(c%(K/32))*32, within=k0%256;
    constant const uint8_t* w=arena+gp.y*blob+(DOWN?wo:(row&1)*wo)+ulong(DOWN?row:row/2)*rb;
    const uint dq_tid=TY==42?(within/64)*8+(within%64)/8+quarter:quarter*8+within/32;
    CacheStore store{weights+(ulong(gp.y)*N+row)*K+k0, long(within), 0};
    iqk_dq_dispatch<half>(TY,w,k0/256,store,dq_tid);
}
#define CU(NAME, TY, DOWN) \
kernel void NAME(constant const uint8_t* arena [[buffer(0)]], device half* weights [[buffer(1)]], \
                 constant const long& H [[buffer(2)]], constant const long& FF [[buffer(3)]], \
                 constant const ulong& rb [[buffer(4)]], constant const ulong& wo [[buffer(5)]], \
                 constant const ulong& blob [[buffer(6)]], uint2 gp [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]]) {cg_unpack<TY,DOWN>(arena,weights,H,FF,rb,wo,blob,gp,tid);}

CU(cu_gu18, 18, false)
CG(cg_gu18_16x64_22, 18, false, 16, 64, 2, 2)
CG(cg_gu18_32x64_24, 18, false, 32, 64, 2, 4)
CU(cu_gu21, 21, false)
CG(cg_gu21_16x64_22, 21, false, 16, 64, 2, 2)
CG(cg_gu21_32x64_24, 21, false, 32, 64, 2, 4)
CU(cu_gu22, 22, false)
CG(cg_gu22_16x64_22, 22, false, 16, 64, 2, 2)
CG(cg_gu22_32x64_24, 22, false, 32, 64, 2, 4)
CU(cu_gu23, 23, false)
CG(cg_gu23_16x64_22, 23, false, 16, 64, 2, 2)
CG(cg_gu23_32x64_24, 23, false, 32, 64, 2, 4)
CU(cu_dn20, 20, true)
CG(cg_dn20_16x64_22, 20, true, 16, 64, 2, 2)
CG(cg_dn20_32x64_24, 20, true, 32, 64, 2, 4)
CU(cu_dn42, 42, true)
CG(cg_dn42_16x64_22, 42, true, 16, 64, 2, 2)
CG(cg_dn42_32x64_24, 42, true, 32, 64, 2, 4)