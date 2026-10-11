#include "native_mmvq.metal"

// Integer products and their sums fit exactly in float. The outer scales,
// block accumulation and virtual-warp reduction retain the packed oracle order.
template<int F> struct CandidateDirect;
struct CandidateKWeight { float4 q[4]; int sc[2], m[2], off, word; float d, dm; };
template<int F> static inline CandidateKWeight candidate_k_load(constant const uint8_t* blk, int iqs) {
    const typename nmv_Fmt<F>::W old = nmv_Fmt<F>::load(blk, iqs);
    CandidateKWeight w;
    w.off = old.bq8_offset; w.word = (iqs / 2) % 4;
    w.d = (float) old.dm.x; w.dm = (float) old.dm.y;
    const thread uint8_t* s = reinterpret_cast<const thread uint8_t*>(old.aux);
    const uint4 shifts = uint4(0, 8, 16, 24);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        w.sc[i] = s[i]; w.m[i] = s[i + 2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint v;
            if constexpr (F == 0)
                v = ((as_type<uint>(old.vl[j]) >> (4 * i)) & 0x0f0f0f0fu) |
                    (((as_type<uint>(old.vh[j]) >> i) << 4) & 0x10101010u);
            else v = (as_type<uint>(old.v[j]) >> (4 * i)) & 0x0f0f0f0fu;
            w.q[2 * i + j] = float4((uint4(v) >> shifts) & 255u);
        }
    }
    return w;
}
static inline float candidate_k_apply(const thread CandidateKWeight& w, constant const Q81Block* x) {
    float sd = 0.0f, sm = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        constant const Q81Block* b = x + w.off + i;
        constant const int* q = reinterpret_cast<constant const int*>(b->qs) + w.word;
        const float4 a0 = float4(as_type<char4>(q[0])), a1 = float4(as_type<char4>(q[4]));
        const int dotq = (int) (dot(w.q[2 * i], a0) + dot(w.q[2 * i + 1], a1));
        const int sumq = (int) (dot(float4(1.0f), a0) + dot(float4(1.0f), a1));
        sd += (float) b->ds.x * (float) (dotq * w.sc[i]);
        sm += (float) b->ds.x * (float) (sumq * w.m[i]);
    }
    return w.d * sd - w.dm * sm;
}
template<> struct CandidateDirect<0> {
    using W = CandidateKWeight;
    static W load(constant const uint8_t* b, int i, threadgroup const float*) { return candidate_k_load<0>(b, i); }
    static float apply(const thread W& w, constant const Q81Block* x, int) { return candidate_k_apply(w, x); }
};
template<> struct CandidateDirect<1> {
    using W = CandidateKWeight;
    static W load(constant const uint8_t* b, int i, threadgroup const float*) { return candidate_k_load<1>(b, i); }
    static float apply(const thread W& w, constant const Q81Block* x, int) { return candidate_k_apply(w, x); }
};
template<int V> struct CandidateIqWeight { float4 q[V]; int scale; float d; };
template<int F, int V> static inline CandidateIqWeight<V> candidate_iq_load(constant const uint8_t* b, int iqs,
                                                                          threadgroup const float* tbl) {
    CandidateIqWeight<V> w;
    w.d = (float) *reinterpret_cast<constant const half*>(b);
    w.scale = 1;
    if constexpr (F == 5) {
        constant const IQ4XSBlock* x = reinterpret_cast<constant const IQ4XSBlock*>(b);
        w.scale = (((x->scales_l[iqs / 8] >> (iqs & 4)) & 15) |
                   (((x->scales_h >> (iqs / 2)) & 3) << 4)) - 32;
    }
#pragma unroll
    for (int j = 0; j < V / 2; ++j) {
        uint u;
        if constexpr (F == 5)
            u = reinterpret_cast<constant const uint*>(reinterpret_cast<constant const IQ4XSBlock*>(b)->qs)[iqs + j];
        else u = as_type<uint>(nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(
                     reinterpret_cast<constant const IQ4NLBlock*>(b)->qs), iqs + j));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w.q[j][k] = tbl[(u >> (8 * k)) & 15u];
            w.q[j + V / 2][k] = tbl[(u >> (8 * k + 4)) & 15u];
        }
    }
    return w;
}
template<> struct CandidateDirect<5> {
    using W = CandidateIqWeight<8>;
    static W load(constant const uint8_t* b, int i, threadgroup const float* t) { return candidate_iq_load<5, 8>(b, i, t); }
    static float apply(const thread W& w, constant const Q81Block* x, int i) {
        const NmvActXs a = nmv_Direct<5>::act(x, 0, i);
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) s += dot(w.q[j], a.a[j]);
        return (w.d * a.d8) * (float) ((int) s * w.scale);
    }
};
template<> struct CandidateDirect<9> {
    using W = CandidateIqWeight<4>;
    static W load(constant const uint8_t* b, int i, threadgroup const float* t) { return candidate_iq_load<9, 4>(b, i, t); }
    static float apply(const thread W& w, constant const Q81Block* x, int i) {
        const NmvActNl a = nmv_Direct<9>::act(x, 0, i);
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) s += dot(w.q[j], a.a[j]);
        return (w.d * a.d8) * (float) (int) s;
    }
};
template<int F, int NT, int R>
static inline void candidate_batch(constant const uint8_t* w, constant const Q81Block* x, device float* y,
                                   int n_in, int n_out, uint gx, uint tid, threadgroup float* table) {
    using Fmt = nmv_Fmt<F>; using D = CandidateDirect<F>;
    if constexpr (F == 5 || F == 9) {
        if (tid < 16) table[tid] = (float) kvalues_iq4nl[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int lane = (int) (tid & 31u), sg = (int) (tid >> 5);
    const int row0 = ((int) gx * 2 + sg) * R;
    const int nb = n_in / Fmt::DIV, xs = n_in / 32;
    if (row0 >= n_out) return;
    float acc[NT][R];
#pragma unroll
    for (int vw = 0; vw < 4; ++vw) {
        const int vt = lane + 32 * vw, iqs = Fmt::kqs(vt);
        float part[NT][R];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) part[j][i] = 0.0f;
        for (int kb = vt / Fmt::T; kb < nb; kb += Fmt::BPI) {
#pragma unroll
            for (int i = 0; i < R; ++i) if (row0 + i < n_out) {
                const typename D::W v = D::load(w + ((size_t) (row0 + i) * nb + kb) * Fmt::BYTES, iqs, table);
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    part[j][i] += D::apply(v, x + (size_t) j * xs + kb * Fmt::KBY, iqs);
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) acc[j][i] = vw == 0 ? part[j][i] : acc[j][i] + part[j][i];
    }
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const float value = nmv_warp_sum(acc[j][i]);
            const int sl = (F == 9 || (F == 5 && n_out < 2560) || (F < 2 && nb < Fmt::BPI)) ? (row0 + i) % 4 : 0;
            if (lane == sl && row0 + i < n_out) y[(size_t) j * n_out + row0 + i] = value;
        }
}
#define CANDIDATE(F, NT, R) \
kernel void candidate_f##F##_nt##NT##_r##R(NMV_SINGLE_PARAMS) { \
    threadgroup float table[16]; candidate_batch<F, NT, R>(w, x, y, n_in, n_out, gpos.x, tid, table); }
#define CANDIDATE_ROWS(F, NT) CANDIDATE(F, NT, 1) CANDIDATE(F, NT, 2) CANDIDATE(F, NT, 4)
#define CANDIDATE_COLS(F) CANDIDATE_ROWS(F, 1) CANDIDATE_ROWS(F, 2) CANDIDATE_ROWS(F, 3) CANDIDATE_ROWS(F, 4) \
    CANDIDATE_ROWS(F, 5) CANDIDATE_ROWS(F, 6) CANDIDATE_ROWS(F, 7) CANDIDATE_ROWS(F, 8)
CANDIDATE_COLS(0) CANDIDATE_COLS(1) CANDIDATE_COLS(5) CANDIDATE_COLS(9)
