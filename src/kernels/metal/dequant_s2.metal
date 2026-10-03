// src/kernels/metal/dequant_s2.metal - the port of src/kernels/cuda/dequant_s2.cu (K2,
// docs/PORT_METAL/STATUS.md).  The decode is exact integer-to-float arithmetic and the parity test compares
// bits; nothing here may round differently than the CUDA kernel (the single multiply is why the -1 stays in
// the integer domain - see the CUDA file's header note).
#include "strata_port.metalh"

constant const int STRATA_S2_QK = 64;
constant const int STRATA_S2_PER_BYTE = 4;

kernel void dequant_s2_kernel(constant const uint8_t* codes [[buffer(0)]],
                              constant const float* scales [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant const long& n_blocks [[buffer(3)]],
                              uint b [[thread_position_in_grid]]) {
    if (b >= (uint) n_blocks) return;
    const float d = scales[b];
    constant const uint8_t* c = codes + (ulong) b * (STRATA_S2_QK / STRATA_S2_PER_BYTE);
    device float* y = out + (ulong) b * STRATA_S2_QK;
    for (int j = 0; j < STRATA_S2_QK; ++j) {
        const int code = (c[j / STRATA_S2_PER_BYTE] >> ((j % STRATA_S2_PER_BYTE) * 2)) & 0x03;
        y[j] = (float) (code - 1) * d;              // the -1 on the CODE, then ONE multiply
    }
}
