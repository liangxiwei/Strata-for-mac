// include/strata/metal_compat/cuda_fp16.h - the CUDA fp16 header's surface for the Metal backend.
//
// The parity tests include <cuda_fp16.h> for __half/__float2half/__half2float; the conversions are
// f16_bits.hpp's bit-for-bit round-to-nearest-even (the same one the kernels use), spelled under the CUDA
// names so the test sources stay identical across backends.
#pragma once

#include <cstdint>
#include <cstring>

struct __half {
    uint16_t bits = 0;
};

static inline float __half2float(__half h) {
    const uint16_t x = h.bits;
    const uint32_t sign = (x & 0x8000u) << 16;
    const uint32_t exp = (x >> 10) & 0x1Fu, man = x & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {                                    // subnormal: normalise
            uint32_t e = 127 - 15 + 1, m = man;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            f = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (man << 13);       // inf / NaN
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

static inline __half __float2half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const int exp = (int) rawexp - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    __half h;
    if (rawexp == 0xFFu) {                          // inf / NaN
        h.bits = (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));
        return h;
    }
    if (exp >= 31) {                                // overflow -> inf
        h.bits = (uint16_t) (sign | 0x7C00u);
        return h;
    }
    if (exp <= 0) {                                 // subnormal or zero
        if (exp < -10) {
            h.bits = (uint16_t) sign;
            return h;
        }
        man |= 0x800000u;
        const uint32_t sh = (uint32_t) (14 - exp);
        uint32_t out = (man >> sh) & 0x3FFu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (out & 1u))) ++out;    // round to even
        h.bits = (uint16_t) (sign | out);
        return h;
    }
    uint32_t out = sign | ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) ++out;
    h.bits = (uint16_t) out;
    return h;
}
