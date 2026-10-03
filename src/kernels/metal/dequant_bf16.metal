// src/kernels/metal/dequant_bf16.metal - the port of src/kernels/cuda/dequant_bf16.cu (part of K1's parity
// target; the file's own row is K17 in docs/PORT_METAL/STATUS.md).
//
// Same arithmetic, transcribed from the same ggml source the CUDA file cites.  One difference: CUDA templates
// group32 over the type constant; here the type is a runtime switch - one kernel instead of thirty, and the
// switch is per 32-element GROUP, not per element.  The i-quant types (16..29) are refused by the launcher
// until K19 ports iq_kernels.cu.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

static inline float h2f(constant const uint8_t* p) {
    const uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
    return float(as_type<half>(h));
}

static inline uint f2bf(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 64u;    // a NaN stays a quiet NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

static inline void put(device uchar* out, long base, int j, float v, int kind) {
    if (kind == 0) {                                     // bf16
        reinterpret_cast<device ushort*>(out)[base + j] = (ushort) f2bf(v);
    } else if (kind == 1) {                              // f16, round-to-nearest-even
        reinterpret_cast<device half*>(out)[base + j] = half(v);
    } else {
        reinterpret_cast<device float*>(out)[base + j] = v;
    }
}

constant int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static inline void scale_min_k4(int j, constant const uint8_t* q, thread int& d, thread int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static void group32(constant const uint8_t* row_blocks, int gi_in_row, device uchar* out, long out_base,
                    int type, int kind) {
    if (type == 42) {                                    // Q2_0: 64 per block of 18 B
        constant const uint8_t* b = row_blocks + (ulong) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, out_base, j, (float) (q - 1) * d, kind);
        }
    } else if (type == 2) {                              // Q4_0
        constant const uint8_t* b = row_blocks + (ulong) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, out_base, j, (float) ((b[2 + j] & 0x0F) - 8) * d, kind);
            put(out, out_base, j + 16, (float) ((b[2 + j] >> 4) - 8) * d, kind);
        }
    } else if (type == 6) {                              // Q5_0
        constant const uint8_t* b = row_blocks + (ulong) gi_in_row * 22;
        const float d = h2f(b);
        const uint qh = (uint) b[2] | ((uint) b[3] << 8) | ((uint) b[4] << 16) | ((uint) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, out_base, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d, kind);
            put(out, out_base, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d, kind);
        }
    } else if (type == 7) {                              // Q5_1
        constant const uint8_t* b = row_blocks + (ulong) gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint qh = (uint) b[4] | ((uint) b[5] << 8) | ((uint) b[6] << 16) | ((uint) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) & 1) << 4;
            const int xh1 = ((qh >> (j + 16)) & 1) << 4;
            put(out, out_base, j, (float) ((b[8 + j] & 15) | xh0) * d + m, kind);
            put(out, out_base, j + 16, (float) ((b[8 + j] >> 4) | xh1) * d + m, kind);
        }
    } else if (type == 8) {                              // Q8_0
        constant const uint8_t* b = row_blocks + (ulong) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, out_base, j, (float) (int8_t) b[2 + j] * d, kind);
    } else if (type == 20) {                             // IQ4_NL
        constant const uint8_t* b = row_blocks + (ulong) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, out_base, j, d * (float) kv_iq4nl[b[2 + j] & 0xf], kind);
            put(out, out_base, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4], kind);
        }
    } else if (type == 11) {                             // Q3_K
        constant const uint8_t* b = row_blocks + (ulong) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        constant const uint8_t* hm = b;
        constant const uint8_t* q = b + 32 + n * 32;
        constant const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint aux[4] = {(uint) sc[0] | ((uint) sc[1] << 8) | ((uint) sc[2] << 16) | ((uint) sc[3] << 24),
                       (uint) sc[4] | ((uint) sc[5] << 8) | ((uint) sc[6] << 16) | ((uint) sc[7] << 24),
                       (uint) sc[8] | ((uint) sc[9] << 8) | ((uint) sc[10] << 16) | ((uint) sc[11] << 24), 0};
        const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const thread int8_t* scales = reinterpret_cast<const thread int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, out_base, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)), kind);
        }
    } else if (type == 12) {                             // Q4_K
        constant const uint8_t* b = row_blocks + (ulong) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        constant const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, out_base, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1, kind);
    } else if (type == 13) {                             // Q5_K
        constant const uint8_t* b = row_blocks + (ulong) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        constant const uint8_t* qh = b + 16;
        constant const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, out_base, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1, kind);
        }
    } else if (type == 14) {                             // Q6_K
        constant const uint8_t* b = row_blocks + (ulong) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        constant const uint8_t* ql = b + 64 * n;
        constant const uint8_t* qh = b + 128 + 32 * n;
        constant const int8_t* sc = reinterpret_cast<constant const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, out_base, l, d * (float) sc[is + 2 * qu] * (float) (q - 32), kind);
        }
    } else if (type == 23) {                             // IQ4_XS
        constant const uint8_t* b = row_blocks + (ulong) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        constant const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, out_base, j, dl * (float) kv_iq4nl[qs[j] & 0xf], kind);
            put(out, out_base, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4], kind);
        }
    }
}

kernel void dequant_kernel(constant const uint8_t* blocks [[buffer(0)]],
                           constant const long& row_bytes [[buffer(1)]],
                           constant const long& row0 [[buffer(2)]],
                           constant const long& rows [[buffer(3)]],
                           constant const long& groups_per_row [[buffer(4)]],
                           constant const int& type [[buffer(5)]],
                           constant const int& kind [[buffer(6)]],
                           device uchar* out [[buffer(7)]],
                           uint g [[thread_position_in_grid]]) {
    if (g >= rows * groups_per_row) return;
    const long r = g / groups_per_row, gi = g % groups_per_row;
    const long base = r * groups_per_row * 32 + gi * 32;      // in OUTPUT elements; put() views out by kind
    group32(blocks + (row0 + r) * row_bytes, (int) gi, out, base, type, kind);
}
