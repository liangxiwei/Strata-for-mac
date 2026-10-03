// src/kernels/metal/s2_gemv_quads.metal - the port of src/kernels/cuda/s2_gemv_quads.cu.
//
// One BLOCK per output row, the code load amortised over four elements (S2: one code byte per quad) and the
// activation load over four halves.  The reduction is the CUDA file's fixed shared-memory tree, so the
// result is deterministic for a given threads_per_row - the parity bench compares it against the naive
// kernel with a tolerance for exactly the summation-order change.
//
// The CUDA kernel reads the four halves as one aligned `uint2`; here they are four ushort reads the
// compiler may merge - the VALUES and the summation order are unchanged.
#include "strata_port.metalh"

// QK = 64 elements per group; a quad of 4 elements is 1/16 of a group, so group = (quad * 4) >> 6 = quad >> 4
constant const int S2Q_QK_S2 = 64;

kernel void s2_gemv_quads_kernel(constant const ushort* x [[buffer(0)]],          // fp16 patterns
                                 constant const uint8_t* codes [[buffer(1)]],
                                 constant const float* scales [[buffer(2)]],
                                 device float* y [[buffer(3)]],
                                 constant const long& n_in [[buffer(4)]],
                                 constant const long& n_out [[buffer(5)]],
                                 constant const int& tpr [[buffer(6)]],
                                 threadgroup float* partial [[threadgroup(0)]],
                                 uint3 gpos [[threadgroup_position_in_grid]],     // one row per group
                                 uint t [[thread_index_in_threadgroup]]) {
    const long o = (long) gpos.x;
    if (o >= n_out) return;

    const long n_quads = n_in / 4;
    constant const uint8_t* c = codes + (ulong) o * n_quads;      // exactly one code byte per quad
    constant const float* s = scales + (ulong) o * (n_in / S2Q_QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long q = (long) t; q < n_quads; q += tpr) {
        const uint8_t byte = c[q];                                // ONE load for four codes
        const float d = s[q >> 4];                                // (q*4) >> 6, the group index as a shift
        // the four codes, each carrying the -1 bias in the INTEGER domain; the scale is applied once per
        // element here rather than once per group, which is the same expression the generic kernel uses
        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * f32_from_f16(x[q * 4 + 0]);
        a1 += w1 * f32_from_f16(x[q * 4 + 1]);
        a2 += w2 * f32_from_f16(x[q * 4 + 2]);
        a3 += w3 * f32_from_f16(x[q * 4 + 3]);
    }
    partial[t] = (a0 + a1) + (a2 + a3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = tpr / 2; step > 0; step >>= 1) {
        if ((int) t < step) partial[t] += partial[t + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) y[o] = partial[0];
}
