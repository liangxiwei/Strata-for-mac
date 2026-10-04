// Research only. Keep the production dequantizer and every output's MMA order.
#include "iq_kernels.metal"

struct LinearSliceStore {
    device half* dst;
    long first, logical, stride;
    LinearSliceStore operator+(long off) const thread { return {dst, first, logical + off, stride}; }
    device half& operator[](long i) const thread { return dst[(logical + i - first) * stride]; }
};

template<int TY, bool DOWN>
inline void expand_experts(constant const uint8_t* arena, device half* expanded,
                           uint K, uint N, uint ne, ulong blob_bytes, ulong row_bytes,
                           ulong weight_offset, uint gid) {
    const ulong quarters = ulong(ne) * N * (K / 32) * 4;
    if (gid >= quarters) return;
    const uint quarter = gid % 4;
    const ulong slice = gid / 4;
    const uint k0 = uint(slice % (K / 32)) * 32;
    const ulong en = slice / (K / 32);
    const uint e = uint(en / N), n = uint(en % N);
    constant const uint8_t* w = arena + ulong(e) * blob_bytes
        + (DOWN ? weight_offset : (n & 1) * weight_offset)
        + ulong(DOWN ? n : n / 2) * row_bytes;
    const uint within = k0 % 256;
    const uint dq_tid = TY == 42 ? (within / 64) * 8 + (within % 64) / 8 + quarter
                                 : quarter * 8 + within / 32;
    // [expert][K][N] makes the later matrix tile's reads contiguous.
    LinearSliceStore store{expanded + ulong(e) * N * K + ulong(k0) * N + n, long(within), 0, long(N)};
    iqk_dq_dispatch<half>(TY, w, k0 / 256, store, dq_tid);
}

#define EXPAND(NAME, TY, DOWN) \
kernel void NAME(constant const uint8_t* arena [[buffer(0)]], device half* expanded [[buffer(1)]], \
                 constant uint& K [[buffer(2)]], constant uint& N [[buffer(3)]], \
                 constant uint& ne [[buffer(4)]], constant ulong& blob_bytes [[buffer(5)]], \
                 constant ulong& row_bytes [[buffer(6)]], constant ulong& weight_offset [[buffer(7)]], \
                 uint gid [[thread_position_in_grid]]) { \
    expand_experts<TY, DOWN>(arena, expanded, K, N, ne, blob_bytes, row_bytes, weight_offset, gid); }
EXPAND(expand_gu22, 22, false)
EXPAND(expand_gu16, 16, false)
EXPAND(expand_gu29, 29, false)
EXPAND(expand_dn42, 42, true)

template<int BM, int SGM, int SGN>
inline void cached_gemm(constant const ushort* x, constant const half* weights,
                        constant const int4* tiles, device float* y, uint K, uint N,
                        uint2 tile, uint tid, uint sg, uint lane,
                        threadgroup half* sx, threadgroup half* sw, threadgroup float* scratch) {
    constexpr int BN = 64, TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    const int4 desc = tiles[tile.y];
    const ulong e = ulong(uint(desc.x));
    const uint row0 = uint(desc.y), rows = uint(desc.z), nr = tile.x * BN;
    const uint sm = (sg / SGN) * (BM / SGM), sn = (sg % SGN) * (BN / SGN);
    simdgroup_float8x8 c[MI][NJ];
    for (int i = 0; i < MI; ++i) for (int j = 0; j < NJ; ++j) c[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < BM * 32; i += TH)
            sx[i] = i / 32 < rows ? as_type<half>(x[ulong(row0 + i / 32) * K + k0 + i % 32]) : half(0);
        for (uint i = tid; i < BN * 32; i += TH)
            sw[i] = weights[e * N * K + ulong(k0 + i / BN) * N + nr + i % BN];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a[MI], b[NJ];
            for (int i = 0; i < MI; ++i) simdgroup_load(a[i], sx + (sm + i * 8) * 32 + k, 32);
            for (int j = 0; j < NJ; ++j) simdgroup_load(b[j], sw + k * BN + sn + j * 8, BN);
            for (int i = 0; i < MI; ++i) for (int j = 0; j < NJ; ++j)
                simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* mine = scratch + sg * 64;
    for (int i = 0; i < MI; ++i) for (int j = 0; j < NJ; ++j) {
        simdgroup_store(c[i][j], mine, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint v = lane; v < 64; v += 32) {
            const uint m = sm + i * 8 + v / 8, n = nr + sn + j * 8 + v % 8;
            if (m < rows) y[ulong(row0 + m) * N + n] = mine[v];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define CACHED(NAME, BM, SGM, SGN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], constant const half* weights [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[BM * 32], sw[32 * 64]; threadgroup float scratch[SGM * SGN * 64]; \
    cached_gemm<BM, SGM, SGN>(x, weights, tiles, y, K, N, tile, tid, sg, lane, sx, sw, scratch); }
CACHED(cached_r16, 16, 2, 2)
CACHED(cached_r32, 32, 2, 4)

// A large streaming copy gives a measured bandwidth reference on the same run.
kernel void bandwidth_copy(constant const uint4* a [[buffer(0)]], device uint4* b [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) b[gid] = a[gid];
}
