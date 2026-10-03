// src/kernels/metal/quantize_act.metal - the port of src/kernels/cuda/quantize_act.cu (K4).
// The file's three subtleties are the contract; the MSL spellings:
//   * subtlety 2 (the double divide + rint) becomes rint_of_ratio (strata_port.metalh);
//   * the Q8_K `__fmul_rn` is a plain multiply under the build's -ffp-contract=off, which is the same
//     "round the product before nearest_int's magic add" the CUDA kernel pins;
//   * everything else is f32 and verbatim.
#include "strata_port.metalh"

kernel void quantize_q8_0_kernel(constant const float* x [[buffer(0)]],
                                 device uint8_t* blocks [[buffer(1)]],
                                 constant const long& n_blocks [[buffer(2)]],
                                 uint b [[thread_position_in_grid]]) {
    if (b >= (uint) n_blocks) return;
    constant const float* xb = x + (ulong) b * 32;
    device uint8_t* out = blocks + (ulong) b * 34;              // { fp16 d ; int8 qs[32] }

    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = metal::precise::fmax(amax, metal::precise::fabs(xb[i]));
    if (amax == 0.0f) {
        const uint zb = f16_from_f32(0.0f);
        out[0] = (uint8_t) (zb & 0xFF);
        out[1] = (uint8_t) (zb >> 8);
        for (int i = 0; i < 32; ++i) out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint d16bits = f16_from_f32(d32);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);

    for (int i = 0; i < 32; ++i) {
        float q = rint_of_ratio(xb[i], d32);                    // the double divide, emulated
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        out[2 + i] = as_type<uint8_t>((int8_t) q);
    }
}

kernel void quantize_q8_0_scaled_kernel(constant const float* x [[buffer(0)]],
                                        device uint8_t* blocks [[buffer(1)]],
                                        device float* scales [[buffer(2)]],
                                        constant const long& n_blocks [[buffer(3)]],
                                        uint b [[thread_position_in_grid]]) {
    if (b >= (uint) n_blocks) return;
    constant const float* xb = x + (ulong) b * 32;
    device uint8_t* out = blocks + (ulong) b * 34;

    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = metal::precise::fmax(amax, metal::precise::fabs(xb[i]));
    // VERBATIM from cpu/expert.cpp, as the CUDA file's comment requires
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    scales[b] = s;

    const uint d16bits = f16_from_f32(s);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < 32; ++i) {
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = as_type<uint8_t>((int8_t) v);
    }
}

kernel void dequant_q8_0_kernel(constant const uint8_t* blocks [[buffer(0)]],
                                device float* x [[buffer(1)]],
                                constant const long& n_blocks [[buffer(2)]],
                                uint b [[thread_position_in_grid]]) {
    if (b >= (uint) n_blocks) return;
    constant const uint8_t* blk = blocks + (ulong) b * 34;
    const uint dbits = (uint) (blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    device float* out = x + (ulong) b * 32;
    for (int i = 0; i < 32; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
}

// ggml's nearest_int (the magic constant 12582912.0f = 1.5 * 2^23), transcribed - it decides exact ties
// differently from rint and that is its whole point.
static inline int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    return (as_type<int>(val) & 0x007fffff) - 0x00400000;
}

kernel void quantize_q8_K_kernel(constant const float* x [[buffer(0)]],
                                 device uint8_t* blocks [[buffer(1)]],
                                 constant const long& n_blocks [[buffer(2)]],
                                 uint b [[thread_position_in_grid]]) {
    if (b >= (uint) n_blocks) return;
    constant const float* xb = x + (ulong) b * 256;
    device uint8_t* out = blocks + (ulong) b * 292;
    device float* d = reinterpret_cast<device float*>(out);
    device int8_t* qs = reinterpret_cast<device int8_t*>(out + 4);
    device int16_t* bsums = reinterpret_cast<device int16_t*>(out + 4 + 256);

    // the SIGNED value at the largest magnitude; strictly-greater keeps the FIRST maximum, as np.argmax
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < 256; ++j) {
        const float ax = metal::precise::fabs(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }
    if (amax == 0.0f) {
        *d = 0.0f;
        for (int j = 0; j < 256; ++j) qs[j] = 0;
        for (int j = 0; j < 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / max;             // -127, NOT -128
    for (int j = 0; j < 256; ++j) {
        // __fmul_rn: the product rounds to f32 BEFORE nearest_int's add (no contraction, as built)
        const int v = nearest_int_dev(iscale * xb[j]);
        qs[j] = (int8_t) min(127, v);               // MIN only - the source has no lower clamp
    }
    for (int j = 0; j < 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += (int) qs[j * 16 + ii];
        bsums[j] = (int16_t) sum;
    }
    *d = 1.0f / iscale;
}

kernel void dequant_q8_K_kernel(constant const uint8_t* blocks [[buffer(0)]],
                                device float* x [[buffer(1)]],
                                constant const long& n_blocks [[buffer(2)]],
                                uint b [[thread_position_in_grid]]) {
    if (b >= (uint) n_blocks) return;
    constant const uint8_t* blk = blocks + (ulong) b * 292;
    const float d = as_type<float>(uint(blk[0] | (blk[1] << 8) | (blk[2] << 16) | (blk[3] << 24)));
    constant const int8_t* qs = reinterpret_cast<constant const int8_t*>(blk + 4);
    device float* out = x + (ulong) b * 256;
    for (int i = 0; i < 256; ++i) out[i] = (float) qs[i] * d;
}
