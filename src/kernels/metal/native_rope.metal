// src/kernels/metal/native_rope.metal - the port of src/kernels/cuda/native_rope.cu's kernel (part of K18;
// its parity rides rope_parity).  Numerical contract: pinned ggml rope_multi/rope_yarn.  The CUDA template
// <bool TAB> becomes a runtime flag with null table pointers - the "default path unchanged" argument the
// CUDA file makes is a fast-math-compilation concern there; here the analytic path is spelled with precise::
// calls, which is the no-fast-math build's default anyway.
#include "strata_port.metalh"

// mrope.hpp's mrope_pos (see rope.metal)
static inline int mrope_pos(constant const int* tab, int pos, int pair) {
    return tab != nullptr ? tab[(ulong) pos * 3 + (uint) pair % 3] : pos;
}

// rope_scaling.hpp's rope_scaled_angle + ramp, transcribed (the analytic paths' pinned ggml arithmetic)
static inline float rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

static inline void rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low, float corr_high,
                                     float ext_factor, float mscale_in, int pair, thread float& cos_out,
                                     thread float& sin_out) {
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * metal::precise::log(1.0f / freq_scale);
    }
    cos_out = metal::precise::cos(theta) * mscale;
    sin_out = metal::precise::sin(theta) * mscale;
}

kernel void native_rope_apply_kernel(constant const float* x [[buffer(0)]],
                                     device float* out [[buffer(1)]],
                                     constant const int& rows [[buffer(2)]],
                                     constant const int& width [[buffer(3)]],
                                     constant const int& n_rot [[buffer(4)]],
                                     constant const float& theta_scale [[buffer(5)]],
                                     constant const float& freq_scale [[buffer(6)]],
                                     constant const float& corr_low [[buffer(7)]],
                                     constant const float& corr_high [[buffer(8)]],
                                     constant const float& ext_factor [[buffer(9)]],
                                     constant const float& mscale [[buffer(10)]],
                                     constant const int* positions [[buffer(11)]],
                                     constant const int* mtab [[buffer(12)]],
                                     constant const float* tab_cos [[buffer(13)]],   // null: analytic path
                                     constant const float* tab_sin [[buffer(14)]],
                                     constant const int& tab_max_pos [[buffer(15)]],
                                     uint2 tpg [[threadgroup_position_in_grid]],
                                     uint2 gpos [[thread_position_in_grid]]) {
    // the CUDA grid is (pairs/128, rows) x 128 threads: `pair` is the GLOBAL thread x, `row` the group y
    const int row = (int) tpg.y;
    const int pair = (int) gpos.x;
    if (row >= rows || pair >= width / 2) return;
    const ulong start = (ulong) row * width;
    if (pair >= n_rot / 2) {
        out[start + 2 * pair] = x[start + 2 * pair];             // the untouched tail
        out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        return;
    }
    float c, s;
    const int p = mrope_pos(mtab, positions[row], pair);
    if (tab_cos != nullptr && p >= 0 && p < tab_max_pos) {       // rope_tab_cs, inlined
        c = tab_cos[(ulong) p * 32 + (uint) pair];
        s = tab_sin[(ulong) p * 32 + (uint) pair];
    } else {
        const float theta_extrap = (float) p * metal::precise::pow(theta_scale, (float) pair);
        rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
    }
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
}
