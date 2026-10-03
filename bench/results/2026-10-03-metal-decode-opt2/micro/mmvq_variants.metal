// Microbenchmark variants of the IQ4_XS decode MMVQ. The production kernels are included verbatim; the
// variants change only how the exact integer values are computed (dp4a, codebook lookup), never the
// thread mapping, the integer results or the float expression/accumulation order.
#include "native_mmvq.metal"

// dp4a without the byte loop: four sign-extended products added to c. Integer addition is exact (no overflow:
// |sum| <= 4 * 128 * 128), so the association differs from the original but the int32 result does not.
static inline int fast_dp4a(int a, int b, int c) {
    const char4 x = as_type<char4>(a), y = as_type<char4>(b);
    return c + ((int) x.x * (int) y.x + (int) x.y * (int) y.y) + ((int) x.z * (int) y.z + (int) x.w * (int) y.w);
}

// the same 16-entry codebook bytes, picked by nibble from a constant table
static inline int2 lut_constant(int q4) {
    const uint u = as_type<uint>(q4);
    char4 lo, hi;
    lo.x = kvalues_iq4nl[(u >> 0) & 15];  hi.x = kvalues_iq4nl[(u >> 4) & 15];
    lo.y = kvalues_iq4nl[(u >> 8) & 15];  hi.y = kvalues_iq4nl[(u >> 12) & 15];
    lo.z = kvalues_iq4nl[(u >> 16) & 15]; hi.z = kvalues_iq4nl[(u >> 20) & 15];
    lo.w = kvalues_iq4nl[(u >> 24) & 15]; hi.w = kvalues_iq4nl[(u >> 28) & 15];
    return int2(as_type<int>(lo), as_type<int>(hi));
}

// register-only: four packed words, word chosen by nibble bits 2..3, byte by bits 0..1
static inline char reg_pick(uint n) {
    const uint w0 = 0xBFAD9881u, w1 = 0xF6EADDCFu, w2 = 0x26190D01u, w3 = 0x71594535u;   // kvalues_iq4nl
    const uint lo = (n & 4u) ? w1 : w0, hi = (n & 4u) ? w3 : w2;
    const uint w = (n & 8u) ? hi : lo;
    return as_type<char4>(w >> (8u * (n & 3u))).x;
}
static inline int2 lut_reg(int q4) {
    const uint u = as_type<uint>(q4);
    char4 lo, hi;
    lo.x = reg_pick((u >> 0) & 15);  hi.x = reg_pick((u >> 4) & 15);
    lo.y = reg_pick((u >> 8) & 15);  hi.y = reg_pick((u >> 12) & 15);
    lo.z = reg_pick((u >> 16) & 15); hi.z = reg_pick((u >> 20) & 15);
    lo.w = reg_pick((u >> 24) & 15); hi.w = reg_pick((u >> 28) & 15);
    return int2(as_type<int>(lo), as_type<int>(hi));
}

// F = 105: original lookup, fast dp4a
template<> struct nmv_Fmt<105> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) { return nmv_Fmt<5>::load(blk, iqs); }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        (void) blk;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[iqs / 4].qs), j);
            const int u1 = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[iqs / 4].qs), j + 4);
            sumi = fast_dp4a(r.v[j].x, u0, sumi);
            sumi = fast_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * (float) bq8[iqs / 4].ds.x;
        return d * (float) sumi;
    }
};
// F = 106: constant-table lookup, fast dp4a
template<> struct nmv_Fmt<106> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) {
        constant const IQ4XSBlock* w = reinterpret_cast<constant const IQ4XSBlock*>(blk);
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = lut_constant(nmv_get_int_b4(w->qs, iqs + j));
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) w->d;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<105>::apply(r, blk, bq8, iqs);
    }
};
// F = 107: register lookup, fast dp4a
template<> struct nmv_Fmt<107> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) {
        constant const IQ4XSBlock* w = reinterpret_cast<constant const IQ4XSBlock*>(blk);
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = lut_reg(nmv_get_int_b4(w->qs, iqs + j));
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) w->d;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<105>::apply(r, blk, bq8, iqs);
    }
};
// F = 108: original lookup, original dp4a would be F 5; this is the expanded view with fast dp4a
template<> struct nmv_Fmt<108> {
    enum : int { BYTES = 264, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) { return nmv_Fmt<11>::load(blk, iqs); }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<105>::apply(r, blk, bq8, iqs);
    }
};

NMV_SINGLE_PAIR(v105, 105)
NMV_SINGLE_PAIR(v106, 106)
NMV_SINGLE_PAIR(v107, 107)
NMV_SINGLE_PAIR(v108, 108)

// lookups with the ORIGINAL dp4a/apply (nmv_Fmt<5>::apply)
template<> struct nmv_Fmt<116> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) { return nmv_Fmt<106>::load(blk, iqs); }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<5>::apply(r, blk, bq8, iqs);
    }
};
template<> struct nmv_Fmt<117> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) { return nmv_Fmt<107>::load(blk, iqs); }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<5>::apply(r, blk, bq8, iqs);
    }
};
// SWAR register lookup: two 8-entry halves selected per byte lane, byte picked by shifting a 64-bit word
static inline uint lut8_swar(uint idx4, ulong lo8, ulong hi8) {   // idx4: four bytes, each 0..15
    uint r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint n = (idx4 >> (8 * i)) & 15u;
        const ulong t = (n & 8u) ? hi8 : lo8;
        r |= (uint) ((t >> (8u * (n & 7u))) & 0xFFul) << (8 * i);
    }
    return r;
}
static inline int2 lut_swar(int q4) {
    const ulong lo8 = 0xF6EADDCFBFAD9881ul, hi8 = 0x7159453526190D01ul;
    const uint u = as_type<uint>(q4);
    return int2(as_type<int>(lut8_swar(u & 0x0F0F0F0Fu, lo8, hi8)),
                as_type<int>(lut8_swar((u >> 4) & 0x0F0F0F0Fu, lo8, hi8)));
}
template<> struct nmv_Fmt<118> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) {
        constant const IQ4XSBlock* w = reinterpret_cast<constant const IQ4XSBlock*>(blk);
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = lut_swar(nmv_get_int_b4(w->qs, iqs + j));
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) w->d;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<5>::apply(r, blk, bq8, iqs);
    }
};
NMV_SINGLE_PAIR(v116, 116)
NMV_SINGLE_PAIR(v117, 117)
NMV_SINGLE_PAIR(v118, 118)

// threadgroup codebook: the table copied into threadgroup memory once per threadgroup
template<int ROWS>
static inline void tg_body(constant const uint8_t* w, constant const Q81Block* x, device float* y, int n_in, int n_out,
                           threadgroup float* partial, threadgroup char* tbl, uint gpos_x, uint tid) {
    if (tid < 16) tbl[tid] = kvalues_iq4nl[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ty = (int) (tid >> 5), lane = (int) (tid & 31u), row0 = ROWS * (int) gpos_x, bpr = n_in / 256;
    float tmp[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = (int) tid / 8; kbx < bpr; kbx += 16) {
        const int kby = kbx * 8, iqs = 4 * ((int) tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                constant const IQ4XSBlock* b =
                    reinterpret_cast<constant const IQ4XSBlock*>(w + ((size_t) (row0 + i) * bpr + kbx) * 136);
                nmv_Fmt<5>::W r;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint u = as_type<uint>(nmv_get_int_b4(b->qs, iqs + j));
                    char4 lo, hi;
                    lo.x = tbl[(u >> 0) & 15];  hi.x = tbl[(u >> 4) & 15];
                    lo.y = tbl[(u >> 8) & 15];  hi.y = tbl[(u >> 12) & 15];
                    lo.z = tbl[(u >> 16) & 15]; hi.z = tbl[(u >> 20) & 15];
                    lo.w = tbl[(u >> 24) & 15]; hi.w = tbl[(u >> 28) & 15];
                    r.v[j] = int2(as_type<int>(lo), as_type<int>(hi));
                }
                r.ls = ((b->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((b->scales_h >> (iqs / 2)) & 0x03) << 4);
                r.dw = (float) b->d;
                tmp[i] += nmv_Fmt<5>::apply(r, (constant const uint8_t*) b, x + kby, iqs);
            }
        }
    }
    if (ty > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[((ty - 1) * ROWS + i) * 32 + lane] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) tmp[i] += partial[(l * ROWS + i) * 32 + lane];
        tmp[i] = nmv_warp_sum(tmp[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}
kernel void vtg_small(NMV_SINGLE_PARAMS) {
    threadgroup float partial[3 * 4 * 32]; threadgroup char tbl[16];
    tg_body<4>(w, x, y, n_in, n_out, partial, tbl, gpos.x, tid);
}
kernel void vtg_large(NMV_SINGLE_PARAMS) {
    threadgroup float partial[3 * 1 * 32]; threadgroup char tbl[16];
    tg_body<1>(w, x, y, n_in, n_out, partial, tbl, gpos.x, tid);
}

// Direct accumulation: each looked-up code is multiplied by its activation byte without packing into a dp4a
// word. The products and their sum are exact integers (|sumi| <= 32 * 127 * 128 < 2^24), so int and float
// accumulation both give the original int32 sumi; the float expression after it is the original one.
template<int ROWS, bool FLOATS, bool TGTABLE>
static inline void direct_body(constant const uint8_t* w, constant const Q81Block* x, device float* y, int n_in,
                               int n_out, threadgroup float* partial, threadgroup float* ftbl,
                               threadgroup int* itbl, uint gpos_x, uint tid) {
    if (TGTABLE) {
        if (tid < 16) { ftbl[tid] = (float) kvalues_iq4nl[tid]; itbl[tid] = (int) kvalues_iq4nl[tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int ty = (int) (tid >> 5), lane = (int) (tid & 31u), row0 = ROWS * (int) gpos_x, bpr = n_in / 256;
    float tmp[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = (int) tid / 8; kbx < bpr; kbx += 16) {
        const int kby = kbx * 8, iqs = 4 * ((int) tid % 8);
        constant const Q81Block* bq8 = x + kby + iqs / 4;
        constant const int* q8w = reinterpret_cast<constant const int*>(bq8->qs);
        const float d8 = (float) bq8->ds.x;
        char4 a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = as_type<char4>(q8w[j]);
        float4 af[8];
        if (FLOATS) {
#pragma unroll
            for (int j = 0; j < 8; ++j) af[j] = float4(a[j]);
        }
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                constant const IQ4XSBlock* b =
                    reinterpret_cast<constant const IQ4XSBlock*>(w + ((size_t) (row0 + i) * bpr + kbx) * 136);
                constant const uint* qw = reinterpret_cast<constant const uint*>(b->qs) + iqs;
                int sumi;
                if (FLOATS) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint u = qw[j];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint lo = (u >> (8 * k)) & 15u, hi = (u >> (8 * k + 4)) & 15u;
                            const float vl = TGTABLE ? ftbl[lo] : (float) kvalues_iq4nl[lo];
                            const float vh = TGTABLE ? ftbl[hi] : (float) kvalues_iq4nl[hi];
                            s = fma(vl, af[j][k], s);
                            s = fma(vh, af[j + 4][k], s);
                        }
                    }
                    sumi = (int) s;
                } else {
                    sumi = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint u = qw[j];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint lo = (u >> (8 * k)) & 15u, hi = (u >> (8 * k + 4)) & 15u;
                            const int vl = TGTABLE ? itbl[lo] : (int) kvalues_iq4nl[lo];
                            const int vh = TGTABLE ? itbl[hi] : (int) kvalues_iq4nl[hi];
                            sumi += vl * (int) a[j][k];
                            sumi += vh * (int) a[j + 4][k];
                        }
                    }
                }
                const int ls = ((b->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((b->scales_h >> (iqs / 2)) & 0x03) << 4);
                sumi *= ls - 32;
                const float d = (float) b->d * d8;
                tmp[i] += d * (float) sumi;
            }
        }
    }
    if (ty > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[((ty - 1) * ROWS + i) * 32 + lane] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) tmp[i] += partial[(l * ROWS + i) * 32 + lane];
        tmp[i] = nmv_warp_sum(tmp[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}
#define DIRECT_PAIR(NAME, FLOATS, TG) \
kernel void NAME##_small(NMV_SINGLE_PARAMS) { \
    threadgroup float partial[3 * 4 * 32]; threadgroup float ft[16]; threadgroup int it[16]; \
    direct_body<4, FLOATS, TG>(w, x, y, n_in, n_out, partial, ft, it, gpos.x, tid); } \
kernel void NAME##_large(NMV_SINGLE_PARAMS) { \
    threadgroup float partial[3 * 1 * 32]; threadgroup float ft[16]; threadgroup int it[16]; \
    direct_body<1, FLOATS, TG>(w, x, y, n_in, n_out, partial, ft, it, gpos.x, tid); }
DIRECT_PAIR(vdi_tg, false, true)
DIRECT_PAIR(vdf_tg, true, true)
DIRECT_PAIR(vdi_c, false, false)
DIRECT_PAIR(vdf_c, true, false)

// Pair table: one threadgroup float2 (code of the low nibble, code of the high nibble) per weight byte.
template<int ROWS>
static inline void pair_body(constant const uint8_t* w, constant const Q81Block* x, device float* y, int n_in,
                             int n_out, threadgroup float* partial, threadgroup float2* ptbl, uint gpos_x, uint tid) {
    for (uint i = tid; i < 256; i += 128)
        ptbl[i] = float2((float) kvalues_iq4nl[i & 15], (float) kvalues_iq4nl[i >> 4]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ty = (int) (tid >> 5), lane = (int) (tid & 31u), row0 = ROWS * (int) gpos_x, bpr = n_in / 256;
    float tmp[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = (int) tid / 8; kbx < bpr; kbx += 16) {
        const int kby = kbx * 8, iqs = 4 * ((int) tid % 8);
        constant const Q81Block* bq8 = x + kby + iqs / 4;
        constant const int* q8w = reinterpret_cast<constant const int*>(bq8->qs);
        const float d8 = (float) bq8->ds.x;
        float4 af[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) af[j] = float4(as_type<char4>(q8w[j]));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                constant const IQ4XSBlock* b =
                    reinterpret_cast<constant const IQ4XSBlock*>(w + ((size_t) (row0 + i) * bpr + kbx) * 136);
                constant const uint* qw = reinterpret_cast<constant const uint*>(b->qs) + iqs;
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uchar4 q = as_type<uchar4>(qw[j]);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float2 v = ptbl[q[k]];
                        s = fma(v.x, af[j][k], s);
                        s = fma(v.y, af[j + 4][k], s);
                    }
                }
                int sumi = (int) s;
                const int ls = ((b->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((b->scales_h >> (iqs / 2)) & 0x03) << 4);
                sumi *= ls - 32;
                const float d = (float) b->d * d8;
                tmp[i] += d * (float) sumi;
            }
        }
    }
    if (ty > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[((ty - 1) * ROWS + i) * 32 + lane] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) tmp[i] += partial[(l * ROWS + i) * 32 + lane];
        tmp[i] = nmv_warp_sum(tmp[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}
kernel void vpair_small(NMV_SINGLE_PARAMS) {
    threadgroup float partial[3 * 4 * 32]; threadgroup float2 pt[256];
    pair_body<4>(w, x, y, n_in, n_out, partial, pt, gpos.x, tid); }
kernel void vpair_large(NMV_SINGLE_PARAMS) {
    threadgroup float partial[3 * 1 * 32]; threadgroup float2 pt[256];
    pair_body<1>(w, x, y, n_in, n_out, partial, pt, gpos.x, tid); }

// dp4a through float lanes: the four products are exact integers in float, and so is their sum and c's.
static inline int fdp4a(int a, int b, int c) {
    return c + (int) dot(float4(as_type<char4>(a)), float4(as_type<char4>(b)));
}
template<> struct nmv_Fmt<130> {   // expanded view, float-lane dp4a
    enum : int { BYTES = 264, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) { return nmv_Fmt<11>::load(blk, iqs); }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[iqs / 4].qs), j);
            const int u1 = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[iqs / 4].qs), j + 4);
            sumi = fdp4a(r.v[j].x, u0, sumi);
            sumi = fdp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * (float) bq8[iqs / 4].ds.x;
        return d * (float) sumi;
    }
};
template<> struct nmv_Fmt<131> {   // original lookup, float-lane dp4a
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) { return nmv_Fmt<5>::load(blk, iqs); }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        return nmv_Fmt<130>::apply(r, blk, bq8, iqs);
    }
};
NMV_SINGLE_PAIR(v130, 130)
NMV_SINGLE_PAIR(v131, 131)

#define DIRECT_R(R) \
kernel void vdf_r##R(NMV_SINGLE_PARAMS) { \
    threadgroup float partial[3 * R * 32]; threadgroup float ft[16]; threadgroup int it[16]; \
    direct_body<R, true, true>(w, x, y, n_in, n_out, partial, ft, it, gpos.x, tid); }
DIRECT_R(1) DIRECT_R(2) DIRECT_R(4) DIRECT_R(8)

// R rows, pair table or nibble table, one or two accumulators
template<int ROWS, bool PAIR, bool TWO>
static inline void v3_body(constant const uint8_t* w, constant const Q81Block* x, device float* y, int n_in, int n_out,
                           threadgroup float* partial, threadgroup float* ftbl, threadgroup float2* ptbl,
                           uint gpos_x, uint tid) {
    if (PAIR) {
        for (uint i = tid; i < 256; i += 128) ptbl[i] = float2((float) kvalues_iq4nl[i & 15], (float) kvalues_iq4nl[i >> 4]);
    } else if (tid < 16) {
        ftbl[tid] = (float) kvalues_iq4nl[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ty = (int) (tid >> 5), lane = (int) (tid & 31u), row0 = ROWS * (int) gpos_x, bpr = n_in / 256;
    float tmp[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = (int) tid / 8; kbx < bpr; kbx += 16) {
        const int kby = kbx * 8, iqs = 4 * ((int) tid % 8);
        constant const Q81Block* bq8 = x + kby + iqs / 4;
        constant const int* q8w = reinterpret_cast<constant const int*>(bq8->qs);
        const float d8 = (float) bq8->ds.x;
        float4 af[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) af[j] = float4(as_type<char4>(q8w[j]));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                constant const IQ4XSBlock* b =
                    reinterpret_cast<constant const IQ4XSBlock*>(w + ((size_t) (row0 + i) * bpr + kbx) * 136);
                constant const uint* qw = reinterpret_cast<constant const uint*>(b->qs) + iqs;
                float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint u = qw[j];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        float vl, vh;
                        if (PAIR) { const float2 v = ptbl[(u >> (8 * k)) & 255u]; vl = v.x; vh = v.y; }
                        else { vl = ftbl[(u >> (8 * k)) & 15u]; vh = ftbl[(u >> (8 * k + 4)) & 15u]; }
                        s0 = fma(vl, af[j][k], s0);
                        if (TWO) s1 = fma(vh, af[j + 4][k], s1); else s0 = fma(vh, af[j + 4][k], s0);
                    }
                }
                int sumi = TWO ? (int) s0 + (int) s1 : (int) s0;
                const int ls = ((b->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((b->scales_h >> (iqs / 2)) & 0x03) << 4);
                sumi *= ls - 32;
                const float d = (float) b->d * d8;
                tmp[i] += d * (float) sumi;
            }
        }
    }
    if (ty > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[((ty - 1) * ROWS + i) * 32 + lane] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) tmp[i] += partial[(l * ROWS + i) * 32 + lane];
        tmp[i] = nmv_warp_sum(tmp[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}
#define V3(NAME, R, PAIR, TWO) \
kernel void NAME(NMV_SINGLE_PARAMS) { \
    threadgroup float partial[3 * R * 32]; threadgroup float ft[16]; threadgroup float2 pt[PAIR ? 256 : 1]; \
    v3_body<R, PAIR, TWO>(w, x, y, n_in, n_out, partial, ft, pt, gpos.x, tid); }
V3(vdf_r4n1, 4, false, false) V3(vdf_r4n2, 4, false, true) V3(vdf_r4p1, 4, true, false) V3(vdf_r4p2, 4, true, true)

// ---- simdgroup-per-row IQ4_XS: lane l runs the original's four warps' threads l, l+32, l+64, l+96 in turn
// (their kbx loops unchanged), then adds the four partials in the original's order (warp 0 + 1 + 2 + 3) and
// runs the same butterfly. No cross-warp threadgroup partials; W simdgroups x R rows per threadgroup.
template<int W, int R>
static inline void sr_body(constant const uint8_t* w, constant const Q81Block* x, device float* y, int n_in, int n_out,
                           threadgroup float* tbl, uint gpos_x, uint tid) {
    using D = nmv_Direct<5>;
    if (tid < 16) tbl[tid] = (float) kvalues_iq4nl[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int lane = (int) (tid & 31u), row0 = ((int) gpos_x * W + (int) (tid >> 5)) * R, bpr = n_in / 256;
    if (row0 >= n_out) return;
    float acc[R];
#pragma unroll
    for (int q = 0; q < R; ++q) acc[q] = 0.0f;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int vt = lane + 32 * v;
        float t[R];
#pragma unroll
        for (int q = 0; q < R; ++q) t[q] = 0.0f;
        for (int kbx = vt / 8; kbx < bpr; kbx += 16) {
            const int iqs = 4 * (vt % 8);
            const D::A c = D::act(x, kbx * 8, iqs);
#pragma unroll
            for (int q = 0; q < R; ++q)
                if (row0 + q < n_out) t[q] += D::dot(w + ((size_t) (row0 + q) * bpr + kbx) * 136, c, iqs, tbl);
        }
#pragma unroll
        for (int q = 0; q < R; ++q) acc[q] = v == 0 ? t[q] : acc[q] + t[q];
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float s = nmv_warp_sum(acc[q]);
        if (lane == 0 && row0 + q < n_out) y[row0 + q] = s;
    }
}
#define SR(W, R) \
kernel void vsr##W##R(NMV_SINGLE_PARAMS) { threadgroup float tbl[16]; sr_body<W, R>(w, x, y, n_in, n_out, tbl, gpos.x, tid); }
SR(2, 1) SR(4, 1) SR(2, 2) SR(4, 2) SR(8, 1) SR(4, 4)
SR(4, 8) SR(2, 8) SR(2, 4) SR(8, 4)
