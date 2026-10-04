#include "native_mmvq.metal"
kernel void dim_iq4_2560(NMV_SINGLE_PARAMS) {
    constexpr int R = 4;
    using D = nmv_Direct<5>;
    using Fmt = nmv_Fmt<5>;
    threadgroup float tbl[16];
    if (tid < 16) tbl[tid] = (float) kvalues_iq4nl[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int lane = (int) (tid & 31u);
    const int row0 = ((int) gpos.x * NMV_WARPS + (int) (tid >> 5)) * R;
    const int blocks_per_row = 2560 / Fmt::DIV;
    if (row0 >= n_out) return;
    float acc[R];
#pragma unroll
    for (int v = 0; v < NMV_WARPS; ++v) {
        const int vt = lane + 32 * v;                  // the four-warp layout's thread index
        float t[R];
#pragma unroll
        for (int i = 0; i < R; ++i) t[i] = 0.0f;
        for (int kbx = vt / Fmt::T; kbx < blocks_per_row; kbx += Fmt::BPI) {
            const int kqs = Fmt::kqs(vt);
            const D::A c = D::act(x, kbx * Fmt::KBY, kqs);
#pragma unroll
            for (int i = 0; i < R; ++i)
                if (row0 + i < n_out)
                    t[i] += D::dot(w + ((size_t) (row0 + i) * (size_t) blocks_per_row + (size_t) kbx) * Fmt::BYTES, c,
                                   kqs, tbl);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = v == 0 ? t[i] : acc[i] + t[i];
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float s = nmv_warp_sum(acc[i]);
        if (lane == 0 && row0 + i < n_out) y[row0 + i] = s;
    }
}

kernel void dim_iq4_6144(NMV_SINGLE_PARAMS) {
    constexpr int R = 4;
    using D = nmv_Direct<5>;
    using Fmt = nmv_Fmt<5>;
    threadgroup float tbl[16];
    if (tid < 16) tbl[tid] = (float) kvalues_iq4nl[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int lane = (int) (tid & 31u);
    const int row0 = ((int) gpos.x * NMV_WARPS + (int) (tid >> 5)) * R;
    const int blocks_per_row = 6144 / Fmt::DIV;
    if (row0 >= n_out) return;
    float acc[R];
#pragma unroll
    for (int v = 0; v < NMV_WARPS; ++v) {
        const int vt = lane + 32 * v;                  // the four-warp layout's thread index
        float t[R];
#pragma unroll
        for (int i = 0; i < R; ++i) t[i] = 0.0f;
        for (int kbx = vt / Fmt::T; kbx < blocks_per_row; kbx += Fmt::BPI) {
            const int kqs = Fmt::kqs(vt);
            const D::A c = D::act(x, kbx * Fmt::KBY, kqs);
#pragma unroll
            for (int i = 0; i < R; ++i)
                if (row0 + i < n_out)
                    t[i] += D::dot(w + ((size_t) (row0 + i) * (size_t) blocks_per_row + (size_t) kbx) * Fmt::BYTES, c,
                                   kqs, tbl);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = v == 0 ? t[i] : acc[i] + t[i];
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float s = nmv_warp_sum(acc[i]);
        if (lane == 0 && row0 + i < n_out) y[row0 + i] = s;
    }
}
