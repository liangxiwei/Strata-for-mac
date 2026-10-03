// src/kernels/metal/s2_gemv.metal - the port of src/kernels/cuda/s2_gemv.cu (K3).  One thread per output
// row, dequantize on the fly, FP32 accumulation in the CUDA kernel's exact order (no contraction - the
// build's -ffp-contract=off stands in for the CUDA file's reliance on nvcc NOT contracting this loop).
#include "strata_port.metalh"

constant const int STRATA_GEMV_QK = 64;

kernel void s2_gemv_kernel(constant const ushort* x [[buffer(0)]],         // fp16 patterns
                           constant const uint8_t* codes [[buffer(1)]],
                           constant const float* scales [[buffer(2)]],
                           device float* y [[buffer(3)]],
                           constant const long& n_in [[buffer(4)]],
                           constant const long& n_out [[buffer(5)]],
                           uint o [[thread_position_in_grid]]) {
    if (o >= (uint) n_out) return;

    const long nb = (long) n_in / STRATA_GEMV_QK;
    constant const uint8_t* c = codes + (ulong) o * (ulong) nb * (STRATA_GEMV_QK / 4);
    constant const float* s = scales + (ulong) o * (ulong) nb;

    float acc = 0.0f;
    for (long b = 0; b < nb; ++b) {
        const float d = s[b];
        constant const uint8_t* cb = c + b * (STRATA_GEMV_QK / 4);
        constant const ushort* xb = x + (ulong) b * STRATA_GEMV_QK;
        for (int j = 0; j < STRATA_GEMV_QK; ++j) {
            // (code - 1) in the INTEGER domain, then the group scale, then the activation
            const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
            acc += (float) (code - 1) * d * f32_from_f16(xb[j]);
        }
    }
    y[o] = acc;
}
