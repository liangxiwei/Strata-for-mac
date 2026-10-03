// fused_gr variants. The production kernels are included verbatim; variants keep every output's operation
// sequence (per-thread accumulation order, butterfly, epilogue expression) and change only staging/scheduling.
#include "fused_gr.metal"

// norm, one pass: each thread keeps its ten float4 of R' * w_norm in registers and scales them itself after
// the reduction, instead of storing, re-reading and scaling them with a different thread mapping. Every
// element still sees (r * g) rounded, then * s_rs[c] rounded.
kernel void gr_norm_v2(device const float* R [[buffer(0)]],
                       device const float* bo_prev [[buffer(1)]],
                       device const float* inj_prev [[buffer(2)]],
                       device const float* w_norm [[buffer(3)]],
                       constant const float& eps [[buffer(4)]],
                       constant const int& apply [[buffer(5)]],
                       device float* rs [[buffer(6)]],
                       device float* xn [[buffer(7)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[FGR_WARPS][FGR_HC];
    threadgroup float s_rs[FGR_HC];
    float gw[FGR_HC];
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c)
        gw[c] = apply != 0 ? 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC) : 0.0f;
    float ss[FGR_HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr int K = FGR_D / (FGR_THREADS * 4);   // 10
    float4 xv[K];
    int cv[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = (int) tid * 4 + k * FGR_THREADS * 4;
        const int c = i / FGR_N, d = i - c * FGR_N;
        float4 r = *(device const float4*) (R + i);
        if (apply != 0) {
            const float4 b = *(device const float4*) (bo_prev + d);
            r.x = fma(b.x, gw[c], r.x); r.y = fma(b.y, gw[c], r.y);
            r.z = fma(b.z, gw[c], r.z); r.w = fma(b.w, gw[c], r.w);
        }
        const float4 g = *(device const float4*) (w_norm + i);
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss[c] += sq;
        xv[k] = float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        cv[k] = c;
    }
#pragma unroll
    for (int c = 0; c < FGR_HC; ++c) {
        const float v = fgr_warp_sum(ss[c]);
        if (lane == 0) part[sg][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FGR_HC) {
        float s = 0.0f;
        for (int w = 0; w < FGR_WARPS; ++w) s += part[w][tid];
        s_rs[tid] = metal::precise::rsqrt(s / (float) FGR_N + eps);
        rs[tid] = s_rs[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = (int) tid * 4 + k * FGR_THREADS * 4;
        *(device float4*) (xn + i) = xv[k] * s_rs[cv[k]];
    }
}

// down, device address space and R rows per warp (each row's lane-ascending chain unchanged)
template<int R>
static inline void gr_down_body(device const uint* w_down, device const uint* w_inject, device const float* xn,
                                device float* lo, device float* inject_out, uint gx, uint lane, uint sg, uint nsg) {
    const bool inject_block = gx == (uint) (FGR_LR / (nsg * R));
    float acc[R];
    int rows[R];
    bool act[R];
#pragma unroll
    for (int q = 0; q < R; ++q) {
        rows[q] = inject_block ? (int) (sg * R + q) : (int) ((gx * nsg + sg) * R + q);
        act[q] = !(inject_block && (w_inject == nullptr || rows[q] >= FGR_HC));
        acc[q] = 0.0f;
    }
    device const uint4* w4[R];
#pragma unroll
    for (int q = 0; q < R; ++q)
        w4[q] = reinterpret_cast<device const uint4*>(reinterpret_cast<device const uint16_t*>(inject_block ? w_inject : w_down) +
                                                      (ulong) (act[q] ? rows[q] : 0) * FGR_D);
    for (int j = (int) lane; j < FGR_D / 8; j += 32) {
        const float4 xa = *(device const float4*) (xn + (ulong) j * 8);
        const float4 xb = *(device const float4*) (xn + (ulong) j * 8 + 4);
#pragma unroll
        for (int q = 0; q < R; ++q) {
            if (!act[q]) continue;
            const uint4 w = w4[q][j];
            float a = 0.0f;
            a = fma(as_type<float>(w.x << 16), xa.x, a); a = fma(as_type<float>(w.x & 0xffff0000u), xa.y, a);
            a = fma(as_type<float>(w.y << 16), xa.z, a); a = fma(as_type<float>(w.y & 0xffff0000u), xa.w, a);
            a = fma(as_type<float>(w.z << 16), xb.x, a); a = fma(as_type<float>(w.z & 0xffff0000u), xb.y, a);
            a = fma(as_type<float>(w.w << 16), xb.z, a); a = fma(as_type<float>(w.w & 0xffff0000u), xb.w, a);
            acc[q] += a;
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float s = fgr_warp_sum(acc[q]);
        if (lane != 0 || !act[q]) continue;
        if (inject_block) inject_out[rows[q]] = s;
        else { const float x = s / (float) FGR_HC; lo[rows[q]] = x / (1.0f + metal::precise::exp(-x)); }
    }
}
#define GR_DOWN(NAME, R) \
kernel void NAME(device const uint* w_down [[buffer(0)]], device const uint* w_inject [[buffer(1)]], \
                 device const float* xn [[buffer(2)]], device float* lo [[buffer(3)]], device float* inject_out [[buffer(4)]], \
                 uint3 gpos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) { \
    gr_down_body<R>(w_down, w_inject, xn, lo, inject_out, gpos.x, lane, sg, nsg); }
GR_DOWN(gr_down_r1, 1) GR_DOWN(gr_down_r2, 2) GR_DOWN(gr_down_r4, 4)

// up: every warp loads its eight rows first, reduces them, and lane q runs row q's epilogue (the same
// expression lane 0 ran; after the xor butterfly every lane holds the row's sum)
kernel void gr_up_v2(device const uint* w_up [[buffer(0)]],
                     device const float* lo [[buffer(1)]],
                     device const float* R [[buffer(2)]],
                     device float* R_out [[buffer(3)]],
                     device const float* bo_prev [[buffer(4)]],
                     device const float* inj_prev [[buffer(5)]],
                     device const float* w_norm [[buffer(6)]],
                     device const float* rs [[buffer(7)]],
                     device float* mixed [[buffer(8)]],
                     constant const int& apply [[buffer(9)]],
                     uint3 gpos [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][FGR_UPM_COLS];
    const int d0 = (int) gpos.x * FGR_UPM_COLS;
    constexpr int RPW = FGR_HC * FGR_UPM_COLS / FGR_WARPS;   // 8 rows per warp
    const float4 la0 = *(device const float4*) (lo + (ulong) lane * 8);
    const float4 la1 = *(device const float4*) (lo + (ulong) lane * 8 + 4);
    const bool tail = lane < (uint) (FGR_LR / 8 - 32);
    float4 lb0 = 0.0f, lb1 = 0.0f;
    if (tail) { lb0 = *(device const float4*) (lo + (ulong) (32 + lane) * 8); lb1 = *(device const float4*) (lo + (ulong) (32 + lane) * 8 + 4); }
    float acc[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = (int) sg + q * FGR_WARPS;
        const int c = r / FGR_UPM_COLS, dd = r - c * FGR_UPM_COLS, i = c * FGR_N + d0 + dd;
        device const uint4* w4 =
            reinterpret_cast<device const uint4*>(reinterpret_cast<device const uint16_t*>(w_up) + (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        float a = 0.0f;
        a = fma(as_type<float>(wa.x << 16), la0.x, a); a = fma(as_type<float>(wa.x & 0xffff0000u), la0.y, a);
        a = fma(as_type<float>(wa.y << 16), la0.z, a); a = fma(as_type<float>(wa.y & 0xffff0000u), la0.w, a);
        a = fma(as_type<float>(wa.z << 16), la1.x, a); a = fma(as_type<float>(wa.z & 0xffff0000u), la1.y, a);
        a = fma(as_type<float>(wa.w << 16), la1.z, a); a = fma(as_type<float>(wa.w & 0xffff0000u), la1.w, a);
        if (tail) {
            const uint4 wb = w4[32 + lane];
            float b = 0.0f;
            b = fma(as_type<float>(wb.x << 16), lb0.x, b); b = fma(as_type<float>(wb.x & 0xffff0000u), lb0.y, b);
            b = fma(as_type<float>(wb.y << 16), lb0.z, b); b = fma(as_type<float>(wb.y & 0xffff0000u), lb0.w, b);
            b = fma(as_type<float>(wb.z << 16), lb1.x, b); b = fma(as_type<float>(wb.z & 0xffff0000u), lb1.y, b);
            b = fma(as_type<float>(wb.w << 16), lb1.z, b); b = fma(as_type<float>(wb.w & 0xffff0000u), lb1.w, b);
            a += b;
        }
        acc[q] = a;
    }
#pragma unroll
    for (int q = 0; q < RPW; ++q) acc[q] = fgr_warp_sum(acc[q]);
    if (lane < (uint) RPW) {
        float mine = acc[0];
#pragma unroll
        for (int q = 1; q < RPW; ++q) if ((int) lane == q) mine = acc[q];
        const int r = (int) sg + (int) lane * FGR_WARPS;
        const int c = r / FGR_UPM_COLS, dd = r - c * FGR_UPM_COLS, i = c * FGR_N + d0 + dd;
        float rv = R[i];
        if (apply != 0) {
            rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
            R_out[i] = rv;
        }
        const float x = rv * w_norm[i] * rs[c];
        g[c][dd] = x * fgr_sigmoid(mine);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FGR_UPM_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

// down with the lane's 40 weight loads issued four at a time ahead of the (unchanged, ascending) accumulation
kernel void gr_down_pf(device const uint* w_down [[buffer(0)]], device const uint* w_inject [[buffer(1)]],
                       device const float* xn [[buffer(2)]], device float* lo [[buffer(3)]],
                       device float* inject_out [[buffer(4)]], uint3 gpos [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const bool inject_block = gpos.x == FGR_DOWN_BLOCKS;
    const int row = inject_block ? (int) sg : (int) gpos.x * FGR_WARPS + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    device const uint4* w4 = reinterpret_cast<device const uint4*>(
        reinterpret_cast<device const uint16_t*>(inject_block ? w_inject : w_down) + (ulong) (active ? row : 0) * FGR_D);
    float acc = 0.0f;
    if (active) {
        for (int j0 = (int) lane; j0 < FGR_D / 8; j0 += 128) {
            uint4 w[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) w[u] = j0 + 32 * u < FGR_D / 8 ? w4[j0 + 32 * u] : uint4(0u);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = j0 + 32 * u;
                if (j >= FGR_D / 8) break;
                const float4 xa = *(device const float4*) (xn + (ulong) j * 8);
                const float4 xb = *(device const float4*) (xn + (ulong) j * 8 + 4);
                float a = 0.0f;
                a = fma(as_type<float>(w[u].x << 16), xa.x, a); a = fma(as_type<float>(w[u].x & 0xffff0000u), xa.y, a);
                a = fma(as_type<float>(w[u].y << 16), xa.z, a); a = fma(as_type<float>(w[u].y & 0xffff0000u), xa.w, a);
                a = fma(as_type<float>(w[u].z << 16), xb.x, a); a = fma(as_type<float>(w[u].z & 0xffff0000u), xb.y, a);
                a = fma(as_type<float>(w[u].w << 16), xb.z, a); a = fma(as_type<float>(w[u].w & 0xffff0000u), xb.w, a);
                acc += a;
            }
        }
    }
    acc = fgr_warp_sum(acc);
    if (lane != 0 || !active) return;
    if (inject_block) inject_out[row] = acc;
    else { const float x = acc / (float) FGR_HC; lo[row] = x / (1.0f + metal::precise::exp(-x)); }
}
