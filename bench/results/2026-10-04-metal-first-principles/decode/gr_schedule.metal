#include "fused_gr.metal"
kernel void gr_down_w4(constant const uint* w_down [[buffer(0)]],        // bf16 patterns
                                 constant const uint* w_inject [[buffer(1)]],      // bf16; nil = final mixer
                                 constant const float* xn [[buffer(2)]],
                                 device float* lo [[buffer(3)]],
                                 device float* inject_out [[buffer(4)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    const bool inject_block = gpos.x == 80;
    const int row = inject_block ? (int) sg : (int) gpos.x * 4 + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    constant const uint16_t* wrow16 =
        reinterpret_cast<constant const uint16_t*>(inject_block ? w_inject : w_down) +
        (ulong) (active ? row : 0) * FGR_D;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>(wrow16);
    float acc = 0.0f;
    if (active) {
        for (int j = (int) lane; j < FGR_D / 8; j += 32)
            acc += fgr_dot8(w4[j], xn + (ulong) j * 8);
    }
    acc = fgr_warp_sum(acc);
    // The inject block has eight warps but only four output scalars. Its inactive warps must not
    // overwrite the following allocation (often the previous half's still-live injection gates).
    if (lane != 0 || !active) return;
    if (inject_block) {
        inject_out[row] = acc;
    } else {
        const float x = acc / (float) FGR_HC;
        lo[row] = x / (1.0f + metal::precise::exp(-x));
    }
}

kernel void gr_up_w4_c8(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][8];
    const int d0 = (int) gpos.x * 8;
    for (int r = (int) sg; r < FGR_HC * 8; r += 4) {
        const int c = r / 8, dd = r - c * 8, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_up_w4_c16(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][16];
    const int d0 = (int) gpos.x * 16;
    for (int r = (int) sg; r < FGR_HC * 16; r += 4) {
        const int c = r / 16, dd = r - c * 16, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_up_w4_c32(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][32];
    const int d0 = (int) gpos.x * 32;
    for (int r = (int) sg; r < FGR_HC * 32; r += 4) {
        const int c = r / 32, dd = r - c * 32, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_down_w8(constant const uint* w_down [[buffer(0)]],        // bf16 patterns
                                 constant const uint* w_inject [[buffer(1)]],      // bf16; nil = final mixer
                                 constant const float* xn [[buffer(2)]],
                                 device float* lo [[buffer(3)]],
                                 device float* inject_out [[buffer(4)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    const bool inject_block = gpos.x == 40;
    const int row = inject_block ? (int) sg : (int) gpos.x * 8 + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    constant const uint16_t* wrow16 =
        reinterpret_cast<constant const uint16_t*>(inject_block ? w_inject : w_down) +
        (ulong) (active ? row : 0) * FGR_D;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>(wrow16);
    float acc = 0.0f;
    if (active) {
        for (int j = (int) lane; j < FGR_D / 8; j += 32)
            acc += fgr_dot8(w4[j], xn + (ulong) j * 8);
    }
    acc = fgr_warp_sum(acc);
    // The inject block has eight warps but only four output scalars. Its inactive warps must not
    // overwrite the following allocation (often the previous half's still-live injection gates).
    if (lane != 0 || !active) return;
    if (inject_block) {
        inject_out[row] = acc;
    } else {
        const float x = acc / (float) FGR_HC;
        lo[row] = x / (1.0f + metal::precise::exp(-x));
    }
}

kernel void gr_up_w8_c8(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][8];
    const int d0 = (int) gpos.x * 8;
    for (int r = (int) sg; r < FGR_HC * 8; r += 8) {
        const int c = r / 8, dd = r - c * 8, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_up_w8_c16(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][16];
    const int d0 = (int) gpos.x * 16;
    for (int r = (int) sg; r < FGR_HC * 16; r += 8) {
        const int c = r / 16, dd = r - c * 16, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_up_w8_c32(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][32];
    const int d0 = (int) gpos.x * 32;
    for (int r = (int) sg; r < FGR_HC * 32; r += 8) {
        const int c = r / 32, dd = r - c * 32, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_down_w16(constant const uint* w_down [[buffer(0)]],        // bf16 patterns
                                 constant const uint* w_inject [[buffer(1)]],      // bf16; nil = final mixer
                                 constant const float* xn [[buffer(2)]],
                                 device float* lo [[buffer(3)]],
                                 device float* inject_out [[buffer(4)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    const bool inject_block = gpos.x == 20;
    const int row = inject_block ? (int) sg : (int) gpos.x * 16 + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    constant const uint16_t* wrow16 =
        reinterpret_cast<constant const uint16_t*>(inject_block ? w_inject : w_down) +
        (ulong) (active ? row : 0) * FGR_D;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>(wrow16);
    float acc = 0.0f;
    if (active) {
        for (int j = (int) lane; j < FGR_D / 8; j += 32)
            acc += fgr_dot8(w4[j], xn + (ulong) j * 8);
    }
    acc = fgr_warp_sum(acc);
    // The inject block has eight warps but only four output scalars. Its inactive warps must not
    // overwrite the following allocation (often the previous half's still-live injection gates).
    if (lane != 0 || !active) return;
    if (inject_block) {
        inject_out[row] = acc;
    } else {
        const float x = acc / (float) FGR_HC;
        lo[row] = x / (1.0f + metal::precise::exp(-x));
    }
}

kernel void gr_up_w16_c8(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][8];
    const int d0 = (int) gpos.x * 8;
    for (int r = (int) sg; r < FGR_HC * 8; r += 16) {
        const int c = r / 8, dd = r - c * 8, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_up_w16_c16(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][16];
    const int d0 = (int) gpos.x * 16;
    for (int r = (int) sg; r < FGR_HC * 16; r += 16) {
        const int c = r / 16, dd = r - c * 16, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}

kernel void gr_up_w16_c32(constant const uint* w_up [[buffer(0)]],            // bf16 patterns
                               constant const float* lo [[buffer(1)]],
                               constant const float* R [[buffer(2)]],
                               device float* R_out [[buffer(3)]],
                               constant const float* bo_prev [[buffer(4)]],
                               constant const float* inj_prev [[buffer(5)]],
                               constant const float* w_norm [[buffer(6)]],
                               constant const float* rs [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant const int& apply [[buffer(9)]],
                               uint3 gpos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float g[FGR_HC][32];
    const int d0 = (int) gpos.x * 32;
    for (int r = (int) sg; r < FGR_HC * 32; r += 16) {
        const int c = r / 32, dd = r - c * 32, i = c * FGR_N + d0 + dd;
        constant const uint4* w4 =
            reinterpret_cast<constant const uint4*>(reinterpret_cast<constant const uint16_t*>(w_up) +
                                                    (ulong) i * FGR_LR);
        const uint4 wa = w4[lane];
        const uint4 wb = lane < (uint) (FGR_LR / 8 - 32) ? w4[32 + lane] : uint4(0u);
        float acc = fgr_dot8(wa, lo + (ulong) lane * 8);
        if (lane < (uint) (FGR_LR / 8 - 32)) acc += fgr_dot8(wb, lo + (ulong) (32 + lane) * 8);
        acc = fgr_warp_sum(acc);
        if (lane == 0) {
            float rv = R[i];
            if (apply != 0) {
                rv = fma(bo_prev[d0 + dd], 2.0f * fgr_sigmoid(inj_prev[c] / (float) FGR_HC), rv);
                R_out[i] = rv;                        // this block owns column d0+dd of every stream
            }
            const float x = rv * w_norm[i] * rs[c];
            g[c][dd] = x * fgr_sigmoid(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < FGR_HC; ++c) s += g[c][tid];
        mixed[d0 + tid] = s / (float) FGR_HC;
    }
}
