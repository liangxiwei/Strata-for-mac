#include "weights.metal"
#define WDEV_FINE(NAME, TY, DOWN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], device const uint8_t* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[32 * 32], sw[32 * 64]; threadgroup float scratch[8 * 64]; \
    if (tiles[tile.y].z <= 8) \
        wdev_gemm<TY, DOWN, 8, 64, 1, 8>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); \
    else if (tiles[tile.y].z <= 16) \
        wdev_gemm<TY, DOWN, 16, 64, 2, 4>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); \
    else if (tiles[tile.y].z <= 24) \
        wdev_gemm<TY, DOWN, 24, 64, 1, 8>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); \
    else \
        wdev_gemm<TY, DOWN, 32, 64, 2, 4>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); \
}
WDEV_FINE(wdev_fine_gu18,18,false) WDEV_FINE(wdev_fine_gu21,21,false)
WDEV_FINE(wdev_fine_gu22,22,false) WDEV_FINE(wdev_fine_gu23,23,false)
WDEV_FINE(wdev_fine_dn20,20,true) WDEV_FINE(wdev_fine_dn42,42,true)
