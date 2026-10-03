// Direct-dot variants of the resident expert / IQ3_S MMVQ kernels. The production file is included verbatim.
// Each dd_* dot computes the original call's integer sum exactly (every term and partial sum is an integer
// below 2^24 in magnitude, so float accumulation is exact in any order), then runs the original integer
// scale steps and float expression unchanged. Thread mapping, lane order and the butterfly are the original.
#include "iq_kernels.metal"

static inline float4 dd_q8(constant const block_q8_1* b, int j) {
    return float4(as_type<char4>(reinterpret_cast<constant const int*>(b->qs)[j]));
}
// the four bytes of a grid word as floats, byte b negated when bit b of s is set (exact: a sign flip)
static inline float4 dd_signed4(uint w, uint s) {
    const uint4 m = uint4(s & 1u, (s >> 1) & 1u, (s >> 2) & 1u, (s >> 3) & 1u) << 31;
    return as_type<float4>(as_type<uint4>(float4(as_type<uchar4>(w))) ^ m);
}
static inline float dd_fma4(float4 v, float4 a, float s) {
    s = fma(v.x, a.x, s); s = fma(v.y, a.y, s); s = fma(v.z, a.z, s); return fma(v.w, a.w, s);
}

// Q2_0: element 8j+m of the call is ((qs16[j] >> 2m) & 3) - 1 (the byte_perm table {-1, 0, 1, 2})
static inline float dd_q2_0(constant const uint8_t* vbq, constant const block_q8_1* bq8_1, int kbx, int iqs) {
    constant const block_q2_0* b = reinterpret_cast<constant const block_q2_0*>(vbq) + kbx;
    const float d2 = (float) b->d;
    constant const uint16_t* qs = reinterpret_cast<constant const uint16_t*>(b->qs) + iqs * 4;
    constant const block_q8_1* c = bq8_1 + iqs;
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint q = qs[j];
        const float4 v0 = float4(float(q & 3u), float((q >> 2) & 3u), float((q >> 4) & 3u), float((q >> 6) & 3u)) - 1.0f;
        const float4 v1 = float4(float((q >> 8) & 3u), float((q >> 10) & 3u), float((q >> 12) & 3u), float((q >> 14) & 3u)) - 1.0f;
        s = dd_fma4(v0, dd_q8(c, 2 * j), s);
        s = dd_fma4(v1, dd_q8(c, 2 * j + 1), s);
    }
    const int sumi = (int) s;
    const float d8 = iqk_lo2f(c->ds);
    return d2 * d8 * (float) sumi;
}

static inline float dd_iq2_s(constant const uint8_t* vbq, constant const block_q8_1* bq8_1, int kbx, int iqs) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const uint qs_packed = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const uint qh = bq2->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    constant const block_q8_1* c = bq8_1 + iqs / 2;
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint idx = ((qs_packed >> (8 * p)) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x300u);
        const uint2 g = reinterpret_cast<constant const uint2*>(iq2s_grid)[idx];
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        float s = p < 2 ? s0 : s1;
        s = dd_fma4(dd_signed4(g.x, sg), dd_q8(c, 2 * p), s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), dd_q8(c, 2 * p + 1), s);
        if (p < 2) s0 = s; else s1 = s;
    }
    const int sumi0 = (int) s0, sumi1 = (int) s1;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * iqk_lo2f(c->ds);
    return d * (float) sumi;
}

static inline float dd_iq2_xxs(constant const uint8_t* vbq, constant const block_q8_1* bq8_1, int kbx, int iqs) {
    constant const block_iq2_xxs* bq2 = reinterpret_cast<constant const block_iq2_xxs*>(vbq) + kbx;
    const uint q2 = (uint) iqk_get_int_b2(bq2->qs, iqs);
    const uint aux32 = as_type<uint>(iqk_get_int_b2(bq2->qs, iqs + 1));
    constant const block_q8_1* c = bq8_1 + iqs / 2;
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint2 g = reinterpret_cast<constant const uint2*>(iq2xxs_grid)[(q2 >> (8 * p)) & 0xFFu];
        const uint sg = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * p))) & 0xFFu;
        s = dd_fma4(dd_signed4(g.x, sg), dd_q8(c, 2 * p), s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), dd_q8(c, 2 * p + 1), s);
    }
    const int ls = (int) (aux32 >> 27) | 1;
    int sumi = (int) s;
    sumi = sumi * ls / 8;
    const float d = (float) bq2->d * iqk_lo2f(c->ds);
    return d * (float) sumi;
}

static inline float dd_iq3_s(constant const uint8_t* vbq, constant const block_q8_1* bq8_1, int kbx, int iqs) {
    constant const block_iq3_s* bq3 = reinterpret_cast<constant const block_iq3_s*>(vbq) + kbx;
    const uint q0 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 0);
    const uint q1 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1);
    const uint qh = bq3->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->signs), iqs / 2);
    constant const block_q8_1* c = bq8_1 + iqs / 2;
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint qq = p < 2 ? q0 : q1, sh = 16u * (uint) (p & 1);
        const uint i0 = ((qq >> sh) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x100u);
        const uint i1 = ((qq >> (sh + 8)) & 0xFFu) | ((qh << (7 - 2 * p)) & 0x100u);
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        s = dd_fma4(dd_signed4(iq3s_grid[i0], sg), dd_q8(c, 2 * p), s);
        s = dd_fma4(dd_signed4(iq3s_grid[i1], sg >> 4), dd_q8(c, 2 * p + 1), s);
    }
    int sumi = (int) s;
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (float) bq3->d * iqk_lo2f(c->ds);
    return d * (float) sumi;
}

template<int TY>
static inline float dd_dot(constant const uint8_t* row, constant const block_q8_1* x, int kbx, int iqs) {
    if (TY == 42) return dd_q2_0(row, x, kbx, iqs);
    if (TY == 22) return dd_iq2_s(row, x, kbx, iqs);
    if (TY == 16) return dd_iq2_xxs(row, x, kbx, iqs);
    if (TY == 21) return dd_iq3_s(row, x, kbx, iqs);
    return 0.0f;
}
template<int TY>
static inline float dd_row_dot(constant const uint8_t* row, constant const block_q8_1* x, int nb, int lane) {
    using F = iqk_Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += dd_dot<TY>(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return iqk_warp_sum(s);
}

template<int TY, bool DOWN>
static inline void dd_resident_body(constant const uint8_t* arena, constant const ulong* offsets,
                                    constant const int* ids, constant const int* residency,
                                    constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes,
                                    ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                                    device float* out, device float* up, uint3 gp, uint tid) {
    const int e = (int) gp.y, row = (int) gp.x * 8 + (int) (tid >> 5), lane = (int) (tid & 31);
    if (row >= (DOWN ? n_embd : 2 * n_ff)) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = !DOWN && row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    float value = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* wr = arena + off + ((DOWN || is_up) ? weight_offset : 0) + (ulong) r * row_bytes;
        const int width = (int) (DOWN ? n_ff : n_embd);
        const int tok = DOWN ? e : e / k;
        value = dd_row_dot<TY>(wr, xq + (size_t) tok * (width / 32), width / iqk_Fmt<TY>::qk, lane);
    }
    if (lane == 0) (is_up ? up : out)[(size_t) e * (DOWN ? n_embd : n_ff) + r] = value;
}
#define DD_RESIDENT_GU(T) \
kernel void dd_resident_gu_##T(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                               uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    dd_resident_body<T, false>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, \
                               slot_bytes, n_expert, k, has_offsets, out, up, gp, tid); }
#define DD_RESIDENT_DOWN(T) \
kernel void dd_resident_down_##T(IQK_RESIDENT_PARAMS, \
                                 uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    dd_resident_body<T, true>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, \
                              slot_bytes, n_expert, k, has_offsets, out, out, gp, tid); }
DD_RESIDENT_GU(22) DD_RESIDENT_GU(16) DD_RESIDENT_DOWN(42)

// IQ3_S single-column MMVQ (the multi kernel's ncols == 1 path: one row per warp, lane 0 stores)
kernel void dd_mmvq_21(IQK_MMVQ_PARAMS) {
    const int row = (int) gpos.x * 4 + (int) (tid >> 5);
    if (row >= n_out) return;
    const int lane = (int) (tid & 31u);
    const int nb = n_in / 256;
    const float s = dd_row_dot<21>(w + (size_t) row * row_bytes, x, nb, lane);
    if (lane == 0) y[row] = s;
}

// ---- R rows per warp: a lane's calls (kbx, iqs) are the same for every row, so its activation block is
// loaded and converted once per call and applied to R rows. Each row keeps its own ascending chain and butterfly.
struct DDAct { float4 a[8]; float d8; };
static inline DDAct dd_act(constant const block_q8_1* c) {
    DDAct r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.a[j] = dd_q8(c, j);
    r.d8 = iqk_lo2f(c->ds);
    return r;
}
static inline float ddw_q2_0(constant const uint8_t* vbq, int kbx, int iqs, const thread DDAct& c) {
    constant const block_q2_0* b = reinterpret_cast<constant const block_q2_0*>(vbq) + kbx;
    const float d2 = (float) b->d;
    constant const uint16_t* qs = reinterpret_cast<constant const uint16_t*>(b->qs) + iqs * 4;
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint q = qs[j];
        const float4 v0 = float4(float(q & 3u), float((q >> 2) & 3u), float((q >> 4) & 3u), float((q >> 6) & 3u)) - 1.0f;
        const float4 v1 = float4(float((q >> 8) & 3u), float((q >> 10) & 3u), float((q >> 12) & 3u), float((q >> 14) & 3u)) - 1.0f;
        s = dd_fma4(v0, c.a[2 * j], s);
        s = dd_fma4(v1, c.a[2 * j + 1], s);
    }
    return d2 * c.d8 * (float) (int) s;
}
static inline float ddw_iq2_s(constant const uint8_t* vbq, int kbx, int iqs, const thread DDAct& c) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const uint qs_packed = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const uint qh = bq2->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint idx = ((qs_packed >> (8 * p)) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x300u);
        const uint2 g = reinterpret_cast<constant const uint2*>(iq2s_grid)[idx];
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        float s = p < 2 ? s0 : s1;
        s = dd_fma4(dd_signed4(g.x, sg), c.a[2 * p], s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), c.a[2 * p + 1], s);
        if (p < 2) s0 = s; else s1 = s;
    }
    const int sumi0 = (int) s0, sumi1 = (int) s1;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * c.d8;
    return d * (float) sumi;
}
static inline float ddw_iq2_xxs(constant const uint8_t* vbq, int kbx, int iqs, const thread DDAct& c) {
    constant const block_iq2_xxs* bq2 = reinterpret_cast<constant const block_iq2_xxs*>(vbq) + kbx;
    const uint q2 = (uint) iqk_get_int_b2(bq2->qs, iqs);
    const uint aux32 = as_type<uint>(iqk_get_int_b2(bq2->qs, iqs + 1));
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint2 g = reinterpret_cast<constant const uint2*>(iq2xxs_grid)[(q2 >> (8 * p)) & 0xFFu];
        const uint sg = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * p))) & 0xFFu;
        s = dd_fma4(dd_signed4(g.x, sg), c.a[2 * p], s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), c.a[2 * p + 1], s);
    }
    const int ls = (int) (aux32 >> 27) | 1;
    int sumi = (int) s;
    sumi = sumi * ls / 8;
    const float d = (float) bq2->d * c.d8;
    return d * (float) sumi;
}
static inline float ddw_iq3_s(constant const uint8_t* vbq, int kbx, int iqs, const thread DDAct& c) {
    constant const block_iq3_s* bq3 = reinterpret_cast<constant const block_iq3_s*>(vbq) + kbx;
    const uint q0 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 0);
    const uint q1 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1);
    const uint qh = bq3->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->signs), iqs / 2);
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint qq = p < 2 ? q0 : q1, sh = 16u * (uint) (p & 1);
        const uint i0 = ((qq >> sh) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x100u);
        const uint i1 = ((qq >> (sh + 8)) & 0xFFu) | ((qh << (7 - 2 * p)) & 0x100u);
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        s = dd_fma4(dd_signed4(iq3s_grid[i0], sg), c.a[2 * p], s);
        s = dd_fma4(dd_signed4(iq3s_grid[i1], sg >> 4), c.a[2 * p + 1], s);
    }
    int sumi = (int) s;
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (float) bq3->d * c.d8;
    return d * (float) sumi;
}
template<int TY>
static inline float ddw(constant const uint8_t* row, int kbx, int iqs, const thread DDAct& c) {
    if (TY == 42) return ddw_q2_0(row, kbx, iqs, c);
    if (TY == 22) return ddw_iq2_s(row, kbx, iqs, c);
    if (TY == 16) return ddw_iq2_xxs(row, kbx, iqs, c);
    if (TY == 21) return ddw_iq3_s(row, kbx, iqs, c);
    return 0.0f;
}
// the q8_1 block a call reads: Q2_0's iqs is a 32-value chunk index, the 256-blocks' iqs/2 is
template<int TY> static inline int dd_xblk(int kbx, int iqs) {
    return TY == 42 ? kbx * 2 + iqs : kbx * 8 + iqs / 2;
}

// R rows per warp; DIRECT picks the direct dot (shared activations) or the original dot
template<int TY, bool DOWN, int R, bool DIRECT>
static inline void rr_resident_body(constant const uint8_t* arena, constant const ulong* offsets,
                                    constant const int* ids, constant const int* residency,
                                    constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes,
                                    ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                                    device float* out, device float* up, uint3 gp, uint tid) {
    using F = iqk_Fmt<TY>;
    const int e = (int) gp.y, lane = (int) (tid & 31), row0 = ((int) gp.x * 8 + (int) (tid >> 5)) * R;
    const int nrows = (int) (DOWN ? n_embd : 2 * n_ff);
    if (row0 >= nrows) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const int width = (int) (DOWN ? n_ff : n_embd);
    const int nb = width / F::qk;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* wr[R];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const int row = min(row0 + q, nrows - 1);
            const bool is_up = !DOWN && row >= n_ff;
            const int r = is_up ? row - (int) n_ff : row;
            wr[q] = arena + off + ((DOWN || is_up) ? weight_offset : 0) + (ulong) r * row_bytes;
        }
        const int tok = DOWN ? e : e / k;
        constant const block_q8_1* xt = xq + (size_t) tok * (width / 32);
        for (int kk = lane; kk < nb * F::ipb; kk += 32) {
            const int kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
            if (DIRECT) {
                const DDAct c = dd_act(xt + dd_xblk<TY>(kbx, iqs));
#pragma unroll
                for (int q = 0; q < R; ++q) s[q] += ddw<TY>(wr[q], kbx, iqs, c);
            } else {
#pragma unroll
                for (int q = 0; q < R; ++q) s[q] += iqk_fmt_dot<TY>(wr[q], xt + kbx * (F::qk / 32), kbx, iqs);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = slot >= 0 ? iqk_warp_sum(s[q]) : 0.0f;
        const int row = row0 + q;
        if (lane != 0 || row >= nrows) continue;
        const bool is_up = !DOWN && row >= n_ff;
        const int r = is_up ? row - (int) n_ff : row;
        (is_up ? up : out)[(size_t) e * (DOWN ? n_embd : n_ff) + r] = v;
    }
}
#define RR_GU(T, R, D, NAME) \
kernel void NAME(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                 uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    rr_resident_body<T, false, R, D>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, \
                                     slot_bytes, n_expert, k, has_offsets, out, up, gp, tid); }
#define RR_DOWN(T, R, D, NAME) \
kernel void NAME(IQK_RESIDENT_PARAMS, \
                 uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    rr_resident_body<T, true, R, D>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, \
                                    slot_bytes, n_expert, k, has_offsets, out, out, gp, tid); }
RR_GU(22, 2, false, or2_gu_22) RR_GU(22, 4, false, or4_gu_22) RR_GU(22, 2, true, dd2_gu_22) RR_GU(22, 4, true, dd4_gu_22)
RR_GU(16, 2, false, or2_gu_16) RR_GU(16, 4, false, or4_gu_16) RR_GU(16, 2, true, dd2_gu_16) RR_GU(16, 4, true, dd4_gu_16)
RR_DOWN(42, 2, false, or2_down_42) RR_DOWN(42, 4, false, or4_down_42) RR_DOWN(42, 2, true, dd2_down_42) RR_DOWN(42, 4, true, dd4_down_42)

template<int R, bool DIRECT>
static inline void rr_mmvq21(constant const uint8_t* w, ulong row_bytes, constant const block_q8_1* x, device float* y,
                             int n_in, int n_out, uint gx, uint tid) {
    const int row0 = ((int) gx * 4 + (int) (tid >> 5)) * R;
    if (row0 >= n_out) return;
    const int lane = (int) (tid & 31u), nb = n_in / 256;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    for (int kk = lane; kk < nb * 8; kk += 32) {
        const int kbx = kk / 8, iqs = 2 * (kk % 8);
        if (DIRECT) {
            const DDAct c = dd_act(x + kbx * 8 + iqs / 2);
#pragma unroll
            for (int q = 0; q < R; ++q) s[q] += ddw_iq3_s(w + (size_t) min(row0 + q, n_out - 1) * row_bytes, kbx, iqs, c);
        } else {
#pragma unroll
            for (int q = 0; q < R; ++q) s[q] += iqk_fmt_dot<21>(w + (size_t) min(row0 + q, n_out - 1) * row_bytes, x + kbx * 8, kbx, iqs);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = iqk_warp_sum(s[q]);
        if (lane == 0 && row0 + q < n_out) y[row0 + q] = v;
    }
}
kernel void or2_mmvq_21(IQK_MMVQ_PARAMS) { rr_mmvq21<2, false>(w, row_bytes, x, y, n_in, n_out, gpos.x, tid); }
kernel void dd2_mmvq_21(IQK_MMVQ_PARAMS) { rr_mmvq21<2, true>(w, row_bytes, x, y, n_in, n_out, gpos.x, tid); }
kernel void dd4_mmvq_21(IQK_MMVQ_PARAMS) { rr_mmvq21<4, true>(w, row_bytes, x, y, n_in, n_out, gpos.x, tid); }

// ---- grid codebooks staged in threadgroup memory (same values; only where they are read from)
static inline float ddt_iq2_s(constant const uint8_t* vbq, int kbx, int iqs, const thread DDAct& c, threadgroup const uint2* grid) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const uint qs_packed = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const uint qh = bq2->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint idx = ((qs_packed >> (8 * p)) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x300u);
        const uint2 g = grid[idx];
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        float s = p < 2 ? s0 : s1;
        s = dd_fma4(dd_signed4(g.x, sg), c.a[2 * p], s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), c.a[2 * p + 1], s);
        if (p < 2) s0 = s; else s1 = s;
    }
    const int sumi0 = (int) s0, sumi1 = (int) s1;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * c.d8;
    return d * (float) sumi;
}
static inline float ddt_iq2_xxs(constant const uint8_t* vbq, int kbx, int iqs, const thread DDAct& c, threadgroup const uint2* grid) {
    constant const block_iq2_xxs* bq2 = reinterpret_cast<constant const block_iq2_xxs*>(vbq) + kbx;
    const uint q2 = (uint) iqk_get_int_b2(bq2->qs, iqs);
    const uint aux32 = as_type<uint>(iqk_get_int_b2(bq2->qs, iqs + 1));
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint2 g = grid[(q2 >> (8 * p)) & 0xFFu];
        const uint sg = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * p))) & 0xFFu;
        s = dd_fma4(dd_signed4(g.x, sg), c.a[2 * p], s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), c.a[2 * p + 1], s);
    }
    const int ls = (int) (aux32 >> 27) | 1;
    int sumi = (int) s;
    sumi = sumi * ls / 8;
    const float d = (float) bq2->d * c.d8;
    return d * (float) sumi;
}
template<int TY, int R>
static inline void tg_gu_body(constant const uint8_t* arena, constant const int* ids, constant const int* residency,
                              constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes, ulong weight_offset,
                              ulong slot_bytes, int n_expert, int k, device float* gate, device float* up, uint3 gp, uint tid,
                              threadgroup uint2* grid) {
    constexpr int G = TY == 22 ? 1024 : 256;
    constant const uint2* src = reinterpret_cast<constant const uint2*>(TY == 22 ? iq2s_grid : iq2xxs_grid);
    for (int i = (int) tid; i < G; i += 256) grid[i] = src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int e = (int) gp.y, lane = (int) (tid & 31), row0 = ((int) gp.x * 8 + (int) (tid >> 5)) * R;
    const int nrows = (int) (2 * n_ff);
    if (row0 >= nrows) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const int nb = (int) n_embd / 256;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    if (slot >= 0) {
        const ulong off = (ulong) slot * slot_bytes;
        constant const uint8_t* wr[R];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const int row = min(row0 + q, nrows - 1);
            const bool is_up = row >= n_ff;
            wr[q] = arena + off + (is_up ? weight_offset : 0) + (ulong) (is_up ? row - (int) n_ff : row) * row_bytes;
        }
        constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
        for (int kk = lane; kk < nb * 8; kk += 32) {
            const int kbx = kk / 8, iqs = 2 * (kk % 8);
            const DDAct c = dd_act(xt + kbx * 8 + iqs / 2);
#pragma unroll
            for (int q = 0; q < R; ++q) s[q] += TY == 22 ? ddt_iq2_s(wr[q], kbx, iqs, c, grid) : ddt_iq2_xxs(wr[q], kbx, iqs, c, grid);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = slot >= 0 ? iqk_warp_sum(s[q]) : 0.0f;
        const int row = row0 + q;
        if (lane != 0 || row >= nrows) continue;
        const bool is_up = row >= n_ff;
        (is_up ? up : gate)[(size_t) e * n_ff + (is_up ? row - (int) n_ff : row)] = v;
    }
}
#define TG_GU(T, R) \
kernel void tg##R##_gu_##T(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                           uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup uint2 grid[T == 22 ? 1024 : 256]; \
    tg_gu_body<T, R>(arena, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, out, up, gp, tid, grid); }
TG_GU(22, 1) TG_GU(22, 2) TG_GU(22, 4) TG_GU(16, 1) TG_GU(16, 2) TG_GU(16, 4)

// ---- threadgroup size experiment: W warps per threadgroup (the production kernels use 8), R rows per warp,
// original dot (iqk_fmt_dot) or the direct one. Each row's arithmetic is unchanged.
template<int TY, int W, int R, bool DIRECT>
static inline void tgw_gu_body(constant const uint8_t* arena, constant const int* ids, constant const int* residency,
                               constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes, ulong weight_offset,
                               ulong slot_bytes, int n_expert, int k, device float* gate, device float* up, uint3 gp, uint tid) {
    using F = iqk_Fmt<TY>;
    const int e = (int) gp.y, lane = (int) (tid & 31), row0 = ((int) gp.x * W + (int) (tid >> 5)) * R;
    const int nrows = (int) (2 * n_ff);
    if (row0 >= nrows) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const int nb = (int) n_embd / F::qk;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    if (slot >= 0) {
        const ulong off = (ulong) slot * slot_bytes;
        constant const uint8_t* wr[R];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const int row = min(row0 + q, nrows - 1);
            const bool is_up = row >= n_ff;
            wr[q] = arena + off + (is_up ? weight_offset : 0) + (ulong) (is_up ? row - (int) n_ff : row) * row_bytes;
        }
        constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
        for (int kk = lane; kk < nb * F::ipb; kk += 32) {
            const int kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
            if (DIRECT) {
                const DDAct c = dd_act(xt + dd_xblk<TY>(kbx, iqs));
#pragma unroll
                for (int q = 0; q < R; ++q) s[q] += ddw<TY>(wr[q], kbx, iqs, c);
            } else {
#pragma unroll
                for (int q = 0; q < R; ++q) s[q] += iqk_fmt_dot<TY>(wr[q], xt + kbx * (F::qk / 32), kbx, iqs);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = slot >= 0 ? iqk_warp_sum(s[q]) : 0.0f;
        const int row = row0 + q;
        if (lane != 0 || row >= nrows) continue;
        const bool is_up = row >= n_ff;
        (is_up ? up : gate)[(size_t) e * n_ff + (is_up ? row - (int) n_ff : row)] = v;
    }
}
#define TGW(T, W, R, D, NAME) \
kernel void NAME(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                 uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    tgw_gu_body<T, W, R, D>(arena, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, out, up, gp, tid); }
TGW(22, 1, 1, false, w1r1o_gu_22) TGW(22, 2, 1, false, w2r1o_gu_22) TGW(22, 4, 1, false, w4r1o_gu_22)
TGW(22, 1, 2, true, w1r2d_gu_22) TGW(22, 2, 2, true, w2r2d_gu_22) TGW(22, 4, 2, true, w4r2d_gu_22)
TGW(22, 2, 1, true, w2r1d_gu_22) TGW(22, 1, 1, true, w1r1d_gu_22)

// ---- floors (not exact; timing only): same mapping and dispatch as native_resident_gu_22
// L: load exactly the bytes a call reads (weights + activation block), integer-sum them, no grid
// G: L plus the four grid lookups
template<int MODE>
static inline void floor_gu_body(constant const uint8_t* arena, constant const int* ids, constant const int* residency,
                                 constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes, ulong weight_offset,
                                 ulong slot_bytes, int n_expert, int k, device float* gate, device float* up, uint3 gp, uint tid) {
    const int e = (int) gp.y, row = (int) gp.x * 8 + (int) (tid >> 5), lane = (int) (tid & 31);
    if (row >= 2 * n_ff) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    uint acc = 0;
    if (slot >= 0) {
        constant const uint8_t* wr = arena + (ulong) slot * slot_bytes + (is_up ? weight_offset : 0) + (ulong) r * row_bytes;
        constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
        for (int kk = lane; kk < (int) (n_embd / 256) * 8; kk += 32) {
            const int kbx = kk / 8, iqs = 2 * (kk % 8);
            constant const block_iq2_s* b = reinterpret_cast<constant const block_iq2_s*>(wr) + kbx;
            const uint qs = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(b->qs), iqs / 2);
            const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(b->qs), QK_K / 32 + iqs / 2);
            acc += qs + sp + b->qh[iqs / 2] + b->scales[iqs / 2] + as_type<ushort>(b->d);
            constant const block_q8_1* c = xt + kbx * 8 + iqs / 2;
            for (int j = 0; j < 8; ++j) acc += (uint) reinterpret_cast<constant const int*>(c->qs)[j];
            if (MODE == 1)
                for (int p = 0; p < 4; ++p) {
                    const uint idx = ((qs >> (8 * p)) & 0xFFu) | ((b->qh[iqs / 2] << (8 - 2 * p)) & 0x300u);
                    const uint2 g = reinterpret_cast<constant const uint2*>(iq2s_grid)[idx];
                    acc += g.x ^ g.y;
                }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) (is_up ? up : gate)[(size_t) e * n_ff + r] = as_type<float>(acc);
}
kernel void floorL_gu_22(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    floor_gu_body<0>(arena, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, out, up, gp, tid); }
kernel void floorG_gu_22(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    floor_gu_body<1>(arena, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, out, up, gp, tid); }

// ---- activation in threadgroup memory: the token's q8_1 row converted to floats once per threadgroup (8 rows
// of one expert share it); each call reads its 32 values there. Same exact direct dot otherwise.
static inline float tad_iq2_s(constant const uint8_t* vbq, int kbx, int iqs, threadgroup const float4* xa, float d8) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const uint qs_packed = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const uint qh = bq2->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint idx = ((qs_packed >> (8 * p)) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x300u);
        const uint2 g = reinterpret_cast<constant const uint2*>(iq2s_grid)[idx];
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        float s = p < 2 ? s0 : s1;
        s = dd_fma4(dd_signed4(g.x, sg), xa[2 * p], s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), xa[2 * p + 1], s);
        if (p < 2) s0 = s; else s1 = s;
    }
    const int sumi0 = (int) s0, sumi1 = (int) s1;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * d8;
    return d * (float) sumi;
}
template<int R>
static inline void ta_gu_body(constant const uint8_t* arena, constant const int* ids, constant const int* residency,
                              constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes, ulong weight_offset,
                              ulong slot_bytes, int n_expert, int k, device float* gate, device float* up, uint3 gp, uint tid,
                              threadgroup float4* xa, threadgroup float* xd) {
    const int e = (int) gp.y;
    constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
    const int nblk = (int) n_embd / 32;
    for (int i = (int) tid; i < nblk * 8; i += 256)
        xa[i] = float4(as_type<char4>(reinterpret_cast<constant const int*>(xt[i / 8].qs)[i % 8]));
    for (int i = (int) tid; i < nblk; i += 256) xd[i] = iqk_lo2f(xt[i].ds);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int lane = (int) (tid & 31), row0 = ((int) gp.x * 8 + (int) (tid >> 5)) * R;
    const int nrows = (int) (2 * n_ff);
    if (row0 >= nrows) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    if (slot >= 0) {
        const ulong off = (ulong) slot * slot_bytes;
        constant const uint8_t* wr[R];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const int row = min(row0 + q, nrows - 1);
            const bool is_up = row >= n_ff;
            wr[q] = arena + off + (is_up ? weight_offset : 0) + (ulong) (is_up ? row - (int) n_ff : row) * row_bytes;
        }
        for (int kk = lane; kk < (int) (n_embd / 256) * 8; kk += 32) {
            const int kbx = kk / 8, iqs = 2 * (kk % 8), xb = kbx * 8 + iqs / 2;
#pragma unroll
            for (int q = 0; q < R; ++q) s[q] += tad_iq2_s(wr[q], kbx, iqs, xa + xb * 8, xd[xb]);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = slot >= 0 ? iqk_warp_sum(s[q]) : 0.0f;
        const int row = row0 + q;
        if (lane != 0 || row >= nrows) continue;
        const bool is_up = row >= n_ff;
        (is_up ? up : gate)[(size_t) e * n_ff + (is_up ? row - (int) n_ff : row)] = v;
    }
}
#define TA_GU(R) \
kernel void ta##R##_gu_22(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                          uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float4 xa[2560 / 4]; threadgroup float xd[2560 / 32]; \
    ta_gu_body<R>(arena, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, out, up, gp, tid, xa, xd); }
TA_GU(1) TA_GU(2) TA_GU(4)
