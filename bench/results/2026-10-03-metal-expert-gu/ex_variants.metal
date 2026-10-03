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

// ---- round 24: IQ2_S / IQ2_XXS resident gate/up access patterns ------------------------------------------------
// Call k of a row reads q8_1 block k of the token (block kbx * 8 + iqs / 2, with kbx = k / 8, iqs = 2 (k % 8)); for
// n_embd = 2560 a lane makes at most three calls, k = lane, lane + 32, lane + 64. Every variant keeps a row's
// per-lane order (calls ascending), the Split load/apply arithmetic (bitwise the per-call dot, iq_multi_parity)
// and the butterfly; only when loads are issued, and where the activation is read from, changes.
struct GuAct { int u[8]; float d8; };
template<int TY> struct GuApply;
template<> struct GuApply<22> {
    static float apply(const thread iqk_Split<22>::W& r, const thread GuAct& a) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = iqk_dp4a(r.g[j], a.u[j], sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = iqk_dp4a(r.g[j], a.u[j], sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * a.d8;
        return d * (float) sumi;
    }
};
template<> struct GuApply<16> {
    static float apply(const thread iqk_Split<16>::W& r, const thread GuAct& a) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = iqk_dp4a(r.g[j], a.u[j], sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * a.d8;
        return d * (float) sumi;
    }
};

// R rows per warp. REGACT: the lane's (at most three) activation blocks are loaded into registers once per warp.
// UNROLL: every row's weight loads (and grid lookups) are issued before any row is reduced.
template<int TY, int R, bool REGACT, bool UNROLL>
static inline void gx_body(constant const uint8_t* arena, constant const ulong* offsets, constant const int* ids,
                           constant const int* residency, constant const block_q8_1* xq, long n_embd, long n_ff,
                           ulong row_bytes, ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                           device float* gate, device float* up, uint3 gp, uint tid) {
    using F = iqk_Fmt<TY>;
    using S = iqk_Split<TY>;
    constexpr int MAXJ = 3;
    const int e = (int) gp.y, lane = (int) (tid & 31), row0 = ((int) gp.x * 8 + (int) (tid >> 5)) * R;
    const int nrows = (int) (2 * n_ff);
    if (row0 >= nrows) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const int ncalls = (int) (n_embd / F::qk) * F::ipb;
    constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
    const ulong off = slot >= 0 ? (has_offsets ? offsets[slot] : (ulong) slot * slot_bytes) : 0;
    GuAct act[MAXJ];
    if (REGACT && slot >= 0) {
#pragma unroll
        for (int j = 0; j < MAXJ; ++j) {
            const int kk = lane + 32 * j;
            if (kk < ncalls) {
                constant const int* q = reinterpret_cast<constant const int*>(xt[kk].qs);
#pragma unroll
                for (int i = 0; i < 8; ++i) act[j].u[i] = q[i];
                act[j].d8 = iqk_lo2f(xt[kk].ds);
            }
        }
    }
    auto row_ptr = [&](int row) {
        const bool is_up = row >= n_ff;
        return arena + off + (is_up ? weight_offset : 0) + (ulong) (is_up ? row - (int) n_ff : row) * row_bytes;
    };
    float res[R];
    if (UNROLL) {
        typename S::W w[R][MAXJ];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            constant const uint8_t* wr = row_ptr(min(row0 + q, nrows - 1));
#pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int kk = lane + 32 * j;
                if (slot >= 0 && kk < ncalls) w[q][j] = S::load(wr, kk / F::ipb, F::step * (kk % F::ipb));
            }
        }
#pragma unroll
        for (int q = 0; q < R; ++q) {
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int kk = lane + 32 * j;
                if (slot >= 0 && kk < ncalls)
                    s += REGACT ? GuApply<TY>::apply(w[q][j], act[j])
                                : S::apply(w[q][j], xt + (kk / F::ipb) * (F::qk / 32), F::step * (kk % F::ipb));
            }
            res[q] = slot >= 0 ? iqk_warp_sum(s) : 0.0f;
        }
    } else {
        for (int q = 0; q < R; ++q) {
            constant const uint8_t* wr = row_ptr(min(row0 + q, nrows - 1));
            typename S::W w[MAXJ];
#pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int kk = lane + 32 * j;
                if (slot >= 0 && kk < ncalls) w[j] = S::load(wr, kk / F::ipb, F::step * (kk % F::ipb));
            }
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int kk = lane + 32 * j;
                if (slot >= 0 && kk < ncalls)
                    s += REGACT ? GuApply<TY>::apply(w[j], act[j])
                                : S::apply(w[j], xt + (kk / F::ipb) * (F::qk / 32), F::step * (kk % F::ipb));
            }
            res[q] = slot >= 0 ? iqk_warp_sum(s) : 0.0f;
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const int row = row0 + q;
        if (lane != 0 || row >= nrows) continue;
        const bool is_up = row >= n_ff;
        (is_up ? up : gate)[(size_t) e * n_ff + (is_up ? row - (int) n_ff : row)] = res[q];
    }
}
#define GX(T, R, RA, UN, NAME) \
kernel void NAME##_gu_##T(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                          uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    gx_body<T, R, RA, UN>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, \
                          n_expert, k, has_offsets, out, up, gp, tid); }
#define GX_ALL(T) GX(T, 1, false, false, un1) GX(T, 1, true, false, ra1) GX(T, 2, true, false, ra2) \
    GX(T, 4, true, false, ra4) GX(T, 8, true, false, ra8) GX(T, 2, true, true, ru2) GX(T, 4, true, true, ru4) \
    GX(T, 2, false, true, uu2)
GX_ALL(22) GX_ALL(16)

// ---- cost floors for the IQ2_S gate/up (timing only, not exact). Same dispatch as native_resident_gu_22.
// WM 0: the production per-call loads (2-byte qs/signs pairs, qh, scales, d); WM 1: the warp reads the row as
// 4-byte words, lane-contiguous (coalesced), the same bytes. ACT: the q8_1 block per call. GRID: four lookups
// per call (indices from the loaded bytes). SETUP: only ids / residency / offset and the store.
template<int WM, bool ACT, bool GRID, bool SETUP>
static inline void fl_body(constant const uint8_t* arena, constant const int* ids, constant const int* residency,
                           constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes, ulong weight_offset,
                           ulong slot_bytes, int n_expert, int k, device float* gate, device float* up, uint3 gp, uint tid) {
    const int e = (int) gp.y, row = (int) gp.x * 8 + (int) (tid >> 5), lane = (int) (tid & 31);
    if (row >= 2 * n_ff) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    uint acc = 0;
    if (slot >= 0 && !SETUP) {
        constant const uint8_t* wr = arena + (ulong) slot * slot_bytes + (is_up ? weight_offset : 0) + (ulong) r * row_bytes;
        constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
        if (WM == 0) {
            for (int kk = lane; kk < (int) (n_embd / 256) * 8; kk += 32) {
                const int kbx = kk / 8, iqs = 2 * (kk % 8);
                constant const block_iq2_s* b = reinterpret_cast<constant const block_iq2_s*>(wr) + kbx;
                const uint qs = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(b->qs), iqs / 2);
                const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(b->qs), QK_K / 32 + iqs / 2);
                const uint qh = b->qh[iqs / 2];
                acc += qs + sp + qh + b->scales[iqs / 2] + as_type<ushort>(b->d);
                if (ACT) {
                    constant const block_q8_1* c = xt + kk;
                    for (int j = 0; j < 8; ++j) acc += (uint) reinterpret_cast<constant const int*>(c->qs)[j];
                    acc += as_type<uint>(c->ds);
                }
                if (GRID)
                    for (int p = 0; p < 4; ++p) {
                        const uint idx = ((qs >> (8 * p)) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x300u);
                        const uint2 g = reinterpret_cast<constant const uint2*>(iq2s_grid)[idx];
                        acc += g.x ^ g.y;
                    }
            }
        } else {
            constant const uint* w4 = reinterpret_cast<constant const uint*>(wr);   // rows are 4-byte aligned (820 B)
            const int words = (int) (row_bytes / 4);
            for (int i = lane; i < words; i += 32) {
                const uint v = w4[i];
                acc += v;
                if (GRID)
                    for (int p = 0; p < 2; ++p) {
                        const uint2 g = reinterpret_cast<constant const uint2*>(iq2s_grid)[(v >> (10 * p)) & 1023u];
                        acc += g.x ^ g.y;
                    }
            }
            if (ACT)
                for (int kk = lane; kk < (int) (n_embd / 32); kk += 32) {
                    constant const block_q8_1* c = xt + kk;
                    for (int j = 0; j < 8; ++j) acc += (uint) reinterpret_cast<constant const int*>(c->qs)[j];
                    acc += as_type<uint>(c->ds);
                }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) (is_up ? up : gate)[(size_t) e * n_ff + r] = as_type<float>(acc);
}
#define FL(NAME, WM, A, G, S) \
kernel void NAME(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], uint3 gp [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]]) { \
    fl_body<WM, A, G, S>(arena, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, out, up, gp, tid); }
FL(flS_gu_22, 0, false, false, true)
FL(flW_gu_22, 0, false, false, false) FL(flWA_gu_22, 0, true, false, false) FL(flWG_gu_22, 0, false, true, false)
FL(flWAG_gu_22, 0, true, true, false)
FL(fcW_gu_22, 1, false, false, false) FL(fcWA_gu_22, 1, true, false, false) FL(fcWG_gu_22, 1, false, true, false)
FL(fcWAG_gu_22, 1, true, true, false)

// ---- the token's q8_1 row in threadgroup memory, TRANSPOSED: word w of block b at tq[w * NB + b], so the lanes
// of a warp (consecutive blocks) read consecutive words - no bank conflicts. A call's u words and d8 are the very
// values the original reads from device memory; the dot is the original iqk_vd_* text with those reads swapped.
constant const int TQ_NB = 80;   // n_embd / 32 for 2560 (harness)
static inline float tq_iq2_s(constant const uint8_t* vbq, int kbx, int iqs, threadgroup const int* tq,
                             threadgroup const float* td, int blk) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const int qs_packed = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        constant const int* grid_pos = reinterpret_cast<constant const int*>(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = iqk_vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = iqk_vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = tq[(l0 + 0) * TQ_NB + blk];
        const int u1 = tq[(l0 + 1) * TQ_NB + blk];
        if (l0 < 4) { sumi0 = iqk_dp4a(grid_l, u0, sumi0); sumi0 = iqk_dp4a(grid_h, u1, sumi0); }
        else { sumi1 = iqk_dp4a(grid_l, u0, sumi1); sumi1 = iqk_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * td[blk];
    return d * (float) sumi;
}
static inline float tq_iq2_xxs(constant const uint8_t* vbq, int kbx, int iqs, threadgroup const int* tq,
                               threadgroup const float* td, int blk) {
    constant const block_iq2_xxs* bq2 = reinterpret_cast<constant const block_iq2_xxs*>(vbq) + kbx;
    const int q2 = iqk_get_int_b2(bq2->qs, iqs);
    const thread uint8_t* aux8 = reinterpret_cast<const thread uint8_t*>(&q2);
    const uint aux32 = as_type<uint>(iqk_get_int_b2(bq2->qs, iqs + 1));
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = reinterpret_cast<constant const uint2*>(iq2xxs_grid)[aux8[k0 / 2]];
        const uint signs = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * (k0 / 2))));
        const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
        const int grid0 = iqk_vsub4(as_type<int>(grid_pos.x) ^ signs0, signs0);
        sumi = iqk_dp4a(grid0, tq[(k0 + 0) * TQ_NB + blk], sumi);
        const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
        const int grid1 = iqk_vsub4(as_type<int>(grid_pos.y) ^ signs1, signs1);
        sumi = iqk_dp4a(grid1, tq[(k0 + 1) * TQ_NB + blk], sumi);
    }
    const int ls = (int) (aux32 >> 27) | 1;
    sumi = sumi * ls / 8;
    const float d = (float) bq2->d * td[blk];
    return d * (float) sumi;
}
// direct float dot with the transposed threadgroup activation (dd_iq2_s's arithmetic)
static inline float tqd_iq2_s(constant const uint8_t* vbq, int kbx, int iqs, threadgroup const int* tq,
                              threadgroup const float* td, int blk) {
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
        s = dd_fma4(dd_signed4(g.x, sg), float4(as_type<char4>(tq[(2 * p) * TQ_NB + blk])), s);
        s = dd_fma4(dd_signed4(g.y, sg >> 4), float4(as_type<char4>(tq[(2 * p + 1) * TQ_NB + blk])), s);
        if (p < 2) s0 = s; else s1 = s;
    }
    const int sumi0 = (int) s0, sumi1 = (int) s1;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * td[blk];
    return d * (float) sumi;
}
template<int TY, int MODE>   // MODE 0: original dot text, 1: direct float dot (IQ2_S only)
static inline void tq_gu_body(constant const uint8_t* arena, constant const ulong* offsets, constant const int* ids,
                              constant const int* residency, constant const block_q8_1* xq, long n_embd, long n_ff,
                              ulong row_bytes, ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                              device float* gate, device float* up, uint3 gp, uint tid, threadgroup int* tq, threadgroup float* td) {
    const int e = (int) gp.y;
    constant const block_q8_1* xt = xq + (size_t) (e / k) * TQ_NB;
    for (int i = (int) tid; i < TQ_NB * 8; i += 256) {
        const int b = i % TQ_NB, w = i / TQ_NB;
        tq[w * TQ_NB + b] = reinterpret_cast<constant const int*>(xt[b].qs)[w];
    }
    for (int i = (int) tid; i < TQ_NB; i += 256) td[i] = iqk_lo2f(xt[i].ds);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row = (int) gp.x * 8 + (int) (tid >> 5), lane = (int) (tid & 31);
    if (row >= 2 * n_ff) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    float value = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* wr = arena + off + (is_up ? weight_offset : 0) + (ulong) r * row_bytes;
        float s = 0.0f;
        for (int kk = lane; kk < (int) (n_embd / 256) * 8; kk += 32) {
            const int kbx = kk / 8, iqs = 2 * (kk % 8);
            s += TY == 22 ? (MODE ? tqd_iq2_s(wr, kbx, iqs, tq, td, kk) : tq_iq2_s(wr, kbx, iqs, tq, td, kk))
                          : tq_iq2_xxs(wr, kbx, iqs, tq, td, kk);
        }
        value = iqk_warp_sum(s);
    }
    if (lane == 0) (is_up ? up : gate)[(size_t) e * n_ff + r] = value;
}
#define TQ(T, M, NAME) \
kernel void NAME##_gu_##T(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                          uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup int tq[TQ_NB * 8]; threadgroup float td[TQ_NB]; \
    tq_gu_body<T, M>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, \
                     has_offsets, out, up, gp, tid, tq, td); }
TQ(22, 0, tq) TQ(22, 1, tqd) TQ(16, 0, tq)

// ---- (a) per-warp weight staging: the warp copies its 820-byte row into threadgroup memory with lane-contiguous
// 4-byte loads (independent, coalesced), then every call decodes from there. Only a simdgroup barrier: no warp
// waits for another. The dot is iqk_vd_iq2_s_q8_1's text with the weight reads from threadgroup memory.
// (b) packed grid: every iq2s_grid byte is 8, 25 or 43 (checked by the pack kernel), so an entry is 8 two-bit codes;
// byte = 8 + 17 c + (c >> 1) rebuilds the same int. 2 KB instead of 8 KB.
kernel void pack_iq2s_grid(device ushort* pg [[buffer(0)]], device int* bad [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= 1024) return;
    const ulong g = iq2s_grid[i];
    ushort w = 0;
    for (int b = 0; b < 8; ++b) {
        const uint v = (uint) ((g >> (8 * b)) & 0xFF);
        const uint c = v == 8 ? 0 : v == 25 ? 1 : v == 43 ? 2 : 3;
        if (c == 3) atomic_fetch_add_explicit((device atomic_int*) bad, 1, memory_order_relaxed);
        w |= (ushort) (c << (2 * b));
    }
    pg[i] = w;
}
static inline int pg_word(uint w8) {   // four 2-bit codes -> four grid bytes
    const uint x = (w8 & 3u) | (((w8 >> 2) & 3u) << 8) | (((w8 >> 4) & 3u) << 16) | (((w8 >> 6) & 3u) << 24);
    return as_type<int>(0x08080808u + (x << 4) + x + ((x >> 1) & 0x01010101u));
}
template<bool TGW, bool PG>
static inline float sw_iq2_s(const threadgroup uint8_t* tgrow, constant const uint8_t* devrow, int kbx, int iqs,
                             constant const block_q8_1* bq8_1, constant const ushort* pg) {
    uint qs_packed, signs_packed_32; int qh, ls0, ls1; float dw;
    if (TGW) {
        const threadgroup uint8_t* b = tgrow + kbx * 82;
        const threadgroup uint16_t* q16 = reinterpret_cast<const threadgroup uint16_t*>(b + 2);
        qs_packed = (uint) q16[2 * (iqs / 2)] | ((uint) q16[2 * (iqs / 2) + 1] << 16);
        signs_packed_32 = (uint) q16[2 * (QK_K / 32 + iqs / 2)] | ((uint) q16[2 * (QK_K / 32 + iqs / 2) + 1] << 16);
        qh = b[66 + iqs / 2];
        ls0 = b[74 + iqs / 2] & 0x0F; ls1 = b[74 + iqs / 2] >> 4;
        dw = (float) as_type<half>(*reinterpret_cast<const threadgroup ushort*>(b));
    } else {
        constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(devrow) + kbx;
        qs_packed = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
        signs_packed_32 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
        qh = bq2->qh[iqs / 2];
        ls0 = bq2->scales[iqs / 2] & 0x0F; ls1 = bq2->scales[iqs / 2] >> 4;
        dw = (float) bq2->d;
    }
    const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
    const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int idx = qs[l0 / 2] | ((qh << (8 - l0)) & 0x300);
        int g0, g1;
        if (PG) { const uint w = pg[idx]; g0 = pg_word(w & 0xFFu); g1 = pg_word(w >> 8); }
        else { constant const int* grid_pos = reinterpret_cast<constant const int*>(iq2s_grid + idx); g0 = grid_pos[0]; g1 = grid_pos[1]; }
        const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = iqk_vsub4(g0 ^ signs0, signs0);
        const int grid_h = iqk_vsub4(g1 ^ signs1, signs1);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 0);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 1);
        if (l0 < 4) { sumi0 = iqk_dp4a(grid_l, u0, sumi0); sumi0 = iqk_dp4a(grid_h, u1, sumi0); }
        else { sumi1 = iqk_dp4a(grid_l, u0, sumi1); sumi1 = iqk_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = dw * iqk_lo2f(bq8_1[iqs / 2].ds);
    return d * (float) sumi;
}
template<bool TGW, bool PG>
static inline void sw_gu_body(constant const uint8_t* arena, constant const ulong* offsets, constant const int* ids,
                              constant const int* residency, constant const block_q8_1* xq, long n_embd, long n_ff,
                              ulong row_bytes, ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                              device float* gate, device float* up, uint3 gp, uint tid, threadgroup uint* stage,
                              constant const ushort* pg) {
    const int e = (int) gp.y, warp = (int) (tid >> 5), row = (int) gp.x * 8 + warp, lane = (int) (tid & 31);
    if (row >= 2 * n_ff) return;                               // whole warps only: no simdgroup barrier is split
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    float value = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* wr = arena + off + (is_up ? weight_offset : 0) + (ulong) r * row_bytes;
        threadgroup uint* mine = stage + warp * 208;
        if (TGW) {
            const int words = (int) (row_bytes / 4);
            constant const uint* w4 = reinterpret_cast<constant const uint*>(wr);
            for (int i = lane; i < words; i += 32) mine[i] = w4[i];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        constant const block_q8_1* xt = xq + (size_t) (e / k) * (n_embd / 32);
        float s = 0.0f;
        for (int kk = lane; kk < (int) (n_embd / 256) * 8; kk += 32) {
            const int kbx = kk / 8, iqs = 2 * (kk % 8);
            s += sw_iq2_s<TGW, PG>(reinterpret_cast<const threadgroup uint8_t*>(mine), wr, kbx, iqs, xt + kbx * 8, pg);
        }
        value = iqk_warp_sum(s);
    }
    if (lane == 0) (is_up ? up : gate)[(size_t) e * n_ff + r] = value;
}
#define SW(NAME, TGW, PG) \
kernel void NAME(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], constant const ushort* pg [[buffer(15)]], \
                 uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup uint stage[8 * 208]; \
    sw_gu_body<TGW, PG>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, n_expert, k, \
                        has_offsets, out, up, gp, tid, stage, pg); }
SW(swc_gu_22, false, false) SW(sws_gu_22, true, false) SW(swp_gu_22, false, true) SW(swsp_gu_22, true, true)
// threadgroup size experiment, larger groups: W warps of one row each, the original dot (no threadgroup memory)
TGW(22, 16, 1, false, wG16_gu_22) TGW(22, 32, 1, false, wG32_gu_22) TGW(22, 8, 1, false, wG08_gu_22)
// the production body verbatim under another name (compiler / placement check)
kernel void cpy_resident_gu_22(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]],
                               uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    IQK_RESIDENT_CALL(22, false, up)
}

// ---- the token's q8_1 row word-major in DEVICE memory (what a transposing quantizer would write): word j of block b
// at xT[j * nblk + b], scale d at dT[b]. A warp's load of word j is then 128 contiguous bytes instead of 32
// 36-byte-strided ones (9 cache lines). Same values, the production body otherwise (iqk_vd_iq2_s_q8_1's text).
static inline float xt_iq2_s(constant const uint8_t* vbq, int kbx, int iqs, constant const int* xT, constant const float* dT,
                             int blk, int nblk) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const int qs_packed = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        constant const int* grid_pos = reinterpret_cast<constant const int*>(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = iqk_vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = iqk_vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = xT[(l0 + 0) * nblk + blk];
        const int u1 = xT[(l0 + 1) * nblk + blk];
        if (l0 < 4) {
            sumi0 = iqk_dp4a(grid_l, u0, sumi0);
            sumi0 = iqk_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = iqk_dp4a(grid_l, u0, sumi1);
            sumi1 = iqk_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * dT[blk];
    return d * (float) sumi;
}
kernel void xt_resident_gu_22(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]],
                              constant const int* xT [[buffer(16)]], constant const float* dT [[buffer(17)]],
                              uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int e = (int) gp.y, row = (int) gp.x * 8 + (int) (tid >> 5), lane = (int) (tid & 31);
    if (row >= 2 * n_ff) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    float value = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* wr = arena + off + (is_up ? weight_offset : 0) + (ulong) r * row_bytes;
        const int nblk = (int) (n_embd / 32), tok = e / k;
        float s = 0.0f;
        for (int kk = lane; kk < (int) (n_embd / 256) * 8; kk += 32) {
            s += xt_iq2_s(wr, kk / 8, 2 * (kk % 8), xT + (size_t) tok * nblk * 8, dT + (size_t) tok * nblk, kk, nblk);
        }
        value = iqk_warp_sum(s);
    }
    if (lane == 0) (is_up ? up : out)[(size_t) e * n_ff + r] = value;
}
