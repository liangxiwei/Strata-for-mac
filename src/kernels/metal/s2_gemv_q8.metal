// src/kernels/metal/s2_gemv_q8.metal - the port of src/kernels/cuda/s2_gemv_q8.cu (K11): the S2 GEMV that
// consumes Q8_0 ACTIVATIONS, as ggml does.  One threadgroup per output row (CUDA blockIdx.x is the data
// index), `tpr` threads of it each walking quads strided by `tpr`, the quad loop and the threadgroup
// reduction tree verbatim; the Q8_0 byte layout is walked with NARROW strides first (byte blocks of 34,
// the fp16 scale read as two bytes, the int8 payload reinterpreted only after `blk + 2`).
//
// This is the file's ONE kernel; the private copies that carried it until now (shexp_s2_gemv_q8_kernel in
// shared_expert.metal, ple_s2_gemv_q8_kernel in ple.metal) are deleted - the launchers call s2_gemv_q8().
#include "strata_port.metalh"

kernel void s2_gemv_q8_kernel(constant const uint8_t* act [[buffer(0)]],
                              constant const uint8_t* codes [[buffer(1)]],
                              constant const float* scales [[buffer(2)]],
                              device float* y [[buffer(3)]],
                              constant const long& n_in [[buffer(4)]],
                              constant const long& n_out [[buffer(5)]],
                              constant const int& tpr [[buffer(6)]],
                              threadgroup float* partial [[threadgroup(0)]],   // the launcher's smem bytes
                              uint3 gpos [[threadgroup_position_in_grid]],     // one row per group
                              uint t [[thread_index_in_threadgroup]]) {
    const long o = (long) gpos.x;
    if (o >= n_out) return;

    const long n_quads = n_in / 4;
    constant const uint8_t* c = codes + (ulong) o * n_quads;
    constant const float* s = scales + (ulong) o * (n_in / 64);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long q = (long) t; q < n_quads; q += tpr) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];                      // (q*4) >> 6, the S2 group index as a shift
        // the activation quad: four int8 in one 32-element block, so one block scale
        const long ablk = (q * 4) / 32;
        constant const uint8_t* blk = act + ablk * 34;
        const ushort dbits = (ushort) blk[0] | ((ushort) blk[1] << 8);
        const float dx = f32_from_f16(dbits);
        constant const int8_t* xq = reinterpret_cast<constant const int8_t*>(blk + 2);
        const int off = (int) ((q * 4) % 32);

        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * ((float) xq[off + 0] * dx);
        a1 += w1 * ((float) xq[off + 1] * dx);
        a2 += w2 * ((float) xq[off + 2] * dx);
        a3 += w3 * ((float) xq[off + 3] * dx);
    }
    partial[t] = (a0 + a1) + (a2 + a3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = tpr / 2; step > 0; step >>= 1) {
        if ((int) t < step) partial[t] += partial[t + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) y[o] = partial[0];
}
