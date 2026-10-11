#include "native_mmvq.metal"

// The integer four-byte dot is exactly representable as float. Keep the Q8 scales,
// signed integer scale product, block accumulation and XOR reduction unchanged.
struct Q6DirectWeight { float4 q0, q1; int s0, s1; float d; };
static inline Q6DirectWeight q6_direct_load(constant const uint8_t* blk, int lane) {
    const nmv_Fmt<4>::W wv = nmv_Fmt<4>::load(blk, lane);
    constant const Q6KBlock* w = reinterpret_cast<constant const Q6KBlock*>(blk);
    const int so = 8 * (lane / 16) + (lane % 16) / 4;
    const uint4 sh = uint4(0, 8, 16, 24);
    const uint vl = as_type<uint>(wv.vl), vh = as_type<uint>(wv.vh);
    const uint c0 = (vl & 0x0f0f0f0fu) | ((vh << 4) & 0x30303030u);
    const uint c1 = ((vl >> 4) & 0x0f0f0f0fu) | (vh & 0x30303030u);
    return Q6DirectWeight{float4((uint4(c0) >> sh) & 63u) - 32.0f,
                          float4((uint4(c1) >> sh) & 63u) - 32.0f,
                          (int) w->scales[so], (int) w->scales[so + 4], wv.d};
}
static inline float q6_direct_apply(const thread Q6DirectWeight& w, constant const Q81Block* x, int lane) {
    const int off = 4 * (lane / 16) + (lane % 16) / 8;
    const int iqs = lane % 8;
    const float4 a0 = float4(as_type<char4>(reinterpret_cast<constant const int*>(x[off].qs)[iqs]));
    const float4 a1 = float4(as_type<char4>(reinterpret_cast<constant const int*>(x[off + 2].qs)[iqs]));
    const int d0 = (int) dot(w.q0, a0), d1 = (int) dot(w.q1, a1);
    float s = 0.0f;
    s += (float) x[off].ds.x * (float) (d0 * w.s0);
    s += (float) x[off + 2].ds.x * (float) (d1 * w.s1);
    return w.d * s;
}
template<int NT, int R>
static inline void q6_exact_sg(constant const uint8_t* w, constant const Q81Block* x, device float* y,
                              int n_in, int n_out, uint gx, uint tid) {
    const int lane = (int) (tid & 31u), sg = (int) (tid >> 5);
    const int row0 = ((int) gx * 2 + sg) * R;
    const int nb = n_in / 256, xs = n_in / 32;
    if (row0 >= n_out) return;
    float acc[NT][R];
#pragma unroll
    for (int vw = 0; vw < 4; ++vw) {
        float part[NT][R];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) part[j][i] = 0.0f;
        for (int kb = vw; kb < nb; kb += 4) {
#pragma unroll
            for (int i = 0; i < R; ++i) if (row0 + i < n_out) {
                const Q6DirectWeight wv = q6_direct_load(w + ((size_t) (row0 + i) * nb + kb) * 210, lane);
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    part[j][i] += q6_direct_apply(wv, x + (size_t) j * xs + kb * 8, lane);
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
            const int store_lane = nb < 4 ? (row0 + i) % 4 : 0;
            if (lane == store_lane && row0 + i < n_out) y[(size_t) j * n_out + row0 + i] = value;
        }
}
#define Q6_SG(NT, R) \
kernel void candidate_q6_nt##NT##_r##R(NMV_SINGLE_PARAMS) { q6_exact_sg<NT, R>(w, x, y, n_in, n_out, gpos.x, tid); }
#define Q6_SHAPES(NT) Q6_SG(NT, 1) Q6_SG(NT, 2) Q6_SG(NT, 4)
Q6_SHAPES(1) Q6_SHAPES(2) Q6_SHAPES(3) Q6_SHAPES(4)
Q6_SHAPES(5) Q6_SHAPES(6) Q6_SHAPES(7) Q6_SHAPES(8)
