#include "iq_kernels.metal"
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq2_xxs(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq2_xxs* x = reinterpret_cast<device const block_iq2_xxs*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    device const uint16_t* q2 = x[ibs].qs + 4 * ib;
    device const uint8_t* aux8 = reinterpret_cast<device const uint8_t*>(q2);
    constant const uint8_t* grid = reinterpret_cast<constant const uint8_t*>(iq2xxs_grid + aux8[il]);
    const uint aux32 = (uint) q2[2] | ((uint) q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (float) (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * (uint) il)) & 127u];
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq2_xs(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq2_xs* x = reinterpret_cast<device const block_iq2_xs*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    device const uint16_t* q2 = x[ibs].qs + 4 * ib;
    constant const uint8_t* grid = reinterpret_cast<constant const uint8_t*>(iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + (float) ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq2_s(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq2_s* x = reinterpret_cast<device const block_iq2_s*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint8_t* grid = reinterpret_cast<constant const uint8_t*>(
        iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + (float) ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq3_xxs(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq3_xxs* x = reinterpret_cast<device const block_iq3_xxs*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    device const uint8_t* q3 = x[ibs].qs + 8 * ib;
    device const uint16_t* gas = reinterpret_cast<device const uint16_t*>(x[ibs].qs + QK_K / 4) + 2 * ib;
    constant const uint8_t* grid1 = reinterpret_cast<constant const uint8_t*>(iq3xxs_grid + q3[2 * il + 0]);
    constant const uint8_t* grid2 = reinterpret_cast<constant const uint8_t*>(iq3xxs_grid + q3[2 * il + 1]);
    const uint aux32 = (uint) gas[0] | ((uint) gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * (uint) il)) & 127u];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f));
        y[j + 4] = (dst_t) (d * (float) grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f));
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq3_s(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq3_s* x = reinterpret_cast<device const block_iq3_s*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    device const uint8_t* qs = x[ibs].qs + 8 * ib;
    constant const uint8_t* grid1 =
        reinterpret_cast<constant const uint8_t*>(iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    constant const uint8_t* grid2 =
        reinterpret_cast<constant const uint8_t*>(iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (float) (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f));
        y[j + 4] = (dst_t) (d * (float) grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f));
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq1_m(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq1_m* x = reinterpret_cast<device const block_iq1_m*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    device const uint16_t* sc = reinterpret_cast<device const uint16_t*>(x[ibs].scales);
    const uint scale_u16 = (uint) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const long ib16 = 2 * ib + il / 2;
    const float d = f32_from_f16(scale_u16) * (float) (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta =
        (x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2))) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
    uint grid32[2];
    const thread int8_t* q = reinterpret_cast<const thread int8_t*>(grid32);
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | ((((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7)) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0fu;
    grid32[0] &= 0x0f0f0f0fu;
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * ((float) q[j] + delta));
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq4_nl(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq4_nl* x = reinterpret_cast<device const block_iq4_nl*>(vx) + ibs * (QK_K / QK4_NL);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 4 * il;
    device const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t, typename Ptr>
static inline void wdev_dq_q3_k(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_q3_K* x = reinterpret_cast<device const block_q3_K*>(vx) + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                        is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                        is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                  (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (float) (us - 32);
        auto y = yy + 128 * n + 32 * j;
        device const uint8_t* q = x->qs + 32 * n;
        device const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l)
            y[l] = (dst_t) (dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_iq4_xs(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_iq4_xs* x = reinterpret_cast<device const block_iq4_xs*>(vx) + ibs;
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 4 * il;
    device const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d =
        (float) x->d * (float) ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_q2_0(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    device const block_q2_0* x = reinterpret_cast<device const block_q2_0*>(vx) + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = (dst_t) (d * (float) (code - 1));
    }
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the
// 32-value blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid % 8).
static inline void wdev_scale(int j, device const uint8_t* q, thread uint8_t& d, thread uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_q4_k(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_q4_K* x = reinterpret_cast<device const block_q4_K*>(vx);
    const long il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    auto y = yy + 64 * il + n * ir;
    const float dall = (float) x[ibs].dm.x;
    const float dmin = (float) x[ibs].dm.y;
    device const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    wdev_scale((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * (float) sc, m1 = dmin * (float) m;
    wdev_scale((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * (float) sc, m2 = dmin * (float) m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = (dst_t) (d1 * (float) (q[l] & 0xF) - m1);
        y[l + 32] = (dst_t) (d2 * (float) (q[l] >> 4) - m2);
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_q5_k(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_q5_K* x = reinterpret_cast<device const block_q5_K*>(vx);
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        auto y = yy + 64 * il + 2 * ir;
        const float dall = (float) x[ibs].dm.x;
        const float dmin = (float) x[ibs].dm.y;
        device const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        device const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        wdev_scale(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * (float) sc, m1 = dmin * (float) m;
        wdev_scale(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * (float) sc, m2 = dmin * (float) m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = (dst_t) (d1 * (float) ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = (dst_t) (d1 * (float) ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = (dst_t) (d2 * (float) ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = (dst_t) (d2 * (float) ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_q5_1(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_q5_1* x = reinterpret_cast<device const block_q5_1*>(vx) + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float dmx = (float) x[ib].dm.x, dmy = (float) x[ib].dm.y;
    const uint qh = (uint) x[ib].qh[0] | ((uint) x[ib].qh[1] << 8) | ((uint) x[ib].qh[2] << 16) |
                    ((uint) x[ib].qh[3] << 24);   // CUDA memcpy'd the four bytes into a uint - same bytes
    auto y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = (int) ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = (int) ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = (dst_t) ((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dmx + dmy);
        y[iqs + 16] = (dst_t) ((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dmx + dmy);
    }
}
template<typename dst_t, typename Ptr>
static inline void wdev_dq_q8_0(device const uint8_t* vx, long ibs, Ptr yy, int tid) {
    device const block_q8_0* x = reinterpret_cast<device const block_q8_0*>(vx) + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    auto y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) ((float) x[ib].qs[8 * il + j] * d);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template<typename dst_t, typename Ptr>
static inline void wdev_dq_bf16(device const uint8_t* vxb, long ibs, Ptr yy, int tid) {
    device const uint16_t* x = reinterpret_cast<device const uint16_t*>(vxb) + ibs * 256 + tid * 8;
    for (int j = 0; j < 8; ++j)
        yy[tid * 8 + j] = (dst_t) as_type<float>((uint) x[j] << 16);
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template<typename dst_t, typename Ptr>
static inline void wdev_dq_dispatch(int ty, device const uint8_t* vx, long ibs, Ptr y, int tid) {
    switch (ty) {
        case 16: wdev_dq_iq2_xxs<dst_t>(vx, ibs, y, tid); break;
        case 17: wdev_dq_iq2_xs<dst_t>(vx, ibs, y, tid); break;
        case 18: wdev_dq_iq3_xxs<dst_t>(vx, ibs, y, tid); break;
        case 20: wdev_dq_iq4_nl<dst_t>(vx, ibs, y, tid); break;
        case 21: wdev_dq_iq3_s<dst_t>(vx, ibs, y, tid); break;
        case 22: wdev_dq_iq2_s<dst_t>(vx, ibs, y, tid); break;
        case 29: wdev_dq_iq1_m<dst_t>(vx, ibs, y, tid); break;
        case 23: wdev_dq_iq4_xs<dst_t>(vx, ibs, y, tid); break;
        case 11: wdev_dq_q3_k<dst_t>(vx, ibs, y, tid); break;
        case 42: wdev_dq_q2_0<dst_t>(vx, ibs, y, tid); break;
        case 12: wdev_dq_q4_k<dst_t>(vx, ibs, y, tid); break;
        case 13: wdev_dq_q5_k<dst_t>(vx, ibs, y, tid); break;
        case 7: wdev_dq_q5_1<dst_t>(vx, ibs, y, tid); break;
        case 8: wdev_dq_q8_0<dst_t>(vx, ibs, y, tid); break;
        case 30: wdev_dq_bf16<dst_t>(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<int TY, bool DOWN, int BM, int BN, int SGM, int SGN>
inline void wdev_gemm(constant const ushort* x, device const uint8_t* arena, constant const int4* tiles, device float* y,
                    long H, long FF, ulong row_bytes, ulong weight_offset, uint2 tile, uint tid, uint sg, uint lane,
                    threadgroup half* sx, threadgroup half* sw, threadgroup float* scratch) {
    constexpr int TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    const uint K = (uint) (DOWN ? FF : H), N = (uint) (DOWN ? H : 2 * FF);
    const int4 desc = tiles[tile.y];
    const ulong offset = ulong(uint(desc.x)) | (ulong(uint(desc.w)) << 32);
    device const uint8_t* blob = arena + offset;
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
            device const uint8_t* w = blob + (DOWN ? weight_offset : (n & 1) * weight_offset)
                                       + (ulong) (DOWN ? n : n / 2) * row_bytes;
            const uint dq_tid = TY == 42 ? (within / 64) * 8 + (within % 64) / 8 + quarter : quarter * 8 + within / 32;
            IqTileStoreN store{sw, (long) r, (long) within, 0, BN};
            wdev_dq_dispatch<half>(TY, w, k0 / 256, store, dq_tid);
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
#define WDEV(NAME, TY, DOWN, BM, BN, SGM, SGN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], device const uint8_t* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[BM * 32], sw[32 * BN]; threadgroup float scratch[SGM * SGN * 64]; \
    wdev_gemm<TY, DOWN, BM, BN, SGM, SGN>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); }

WDEV(wdev_gu18,18,false,32,64,2,4)
WDEV(wdev_gu21,21,false,32,64,2,4)
WDEV(wdev_gu22,22,false,32,64,2,4)
WDEV(wdev_gu23,23,false,32,64,2,4)
WDEV(wdev_dn20,20,true,32,64,2,4)
WDEV(wdev_dn42,42,true,32,64,2,4)
