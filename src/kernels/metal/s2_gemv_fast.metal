// src/kernels/metal/s2_gemv_fast.metal - the port of src/kernels/cuda/s2_gemv_fast.cu's two kernels.
//
// THE CONSTANT TABLE, REPLACED (the shim's cudaMemcpyToSymbol is not ported and returns an error).  The
// CUDA file uploads `__constant__ float c_codes[256][4]` with c_codes[b][k] = (float)(((b >> (2*k)) & 3) -
// 1) - a broadcast float4 load per quad replacing the shifts/masks/convert.  Those are exactly the integers
// -1, 0, 1, 2, which f32 represents exactly, so this port COMPUTES the value at the use site:
// `(float)(((byte >> (2*k)) & 3) - 1)` produces the identical float the table held, and the multiplies
// keep the CUDA order `(cv * d) * act`.  No table, no upload, bit-identical arithmetic.
//
// The shared staging of `x` (lever 1) is kept as-is: 4096 halves = 8 KB plus the partials fits this GPU's
// 32 KB threadgroup memory (measured), so no restructure was needed.
#include "strata_port.metalh"

constant const int S2F_QK_S2 = 64;
constant const int S2F_MAX_SHARED_HALVES = 4096;      // 8 KB of threadgroup for x; n_embd 2560 fits

template <bool STAGE_X>
static inline void s2f_body(constant const ushort* x, constant const uint8_t* codes,
                            constant const float* scales, device float* y, constant const long& n_in,
                            constant const long& n_out, constant const int& tpr, threadgroup uint8_t* smem,
                            uint3 gpos, uint t) {
    // the CUDA file's one `extern __shared__` slab, carved the same way: sx first (staged only), the
    // per-thread partials after it
    threadgroup ushort* sx = reinterpret_cast<threadgroup ushort*>(smem);
    threadgroup float* partial =
        reinterpret_cast<threadgroup float*>(smem + (STAGE_X ? S2F_MAX_SHARED_HALVES * 2 : 0));

    const long o = (long) gpos.x;
    if (o >= n_out) return;

    if (STAGE_X) {
        for (long i = (long) t; i < n_in; i += tpr) sx[i] = x[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const long n_quads = n_in / 4;
    constant const uint8_t* c = codes + (ulong) o * n_quads;
    constant const float* s = scales + (ulong) o * (n_in / S2F_QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long q = (long) t; q < n_quads; q += tpr) {
        const uint8_t byte = c[q];                          // ONE load for four codes
        const float d = s[q >> 4];                          // (q*4) >> 6
        // c_codes[byte][k], computed - see the file comment
        const float c0 = (float) (((byte >> 0) & 3) - 1);
        const float c1 = (float) (((byte >> 2) & 3) - 1);
        const float c2 = (float) (((byte >> 4) & 3) - 1);
        const float c3 = (float) (((byte >> 6) & 3) - 1);
        if (STAGE_X) {
            a0 += c0 * d * f32_from_f16(sx[q * 4 + 0]);
            a1 += c1 * d * f32_from_f16(sx[q * 4 + 1]);
            a2 += c2 * d * f32_from_f16(sx[q * 4 + 2]);
            a3 += c3 * d * f32_from_f16(sx[q * 4 + 3]);
        } else {
            a0 += c0 * d * f32_from_f16(x[q * 4 + 0]);
            a1 += c1 * d * f32_from_f16(x[q * 4 + 1]);
            a2 += c2 * d * f32_from_f16(x[q * 4 + 2]);
            a3 += c3 * d * f32_from_f16(x[q * 4 + 3]);
        }
    }
    partial[t] = (a0 + a1) + (a2 + a3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = tpr / 2; step > 0; step >>= 1) {
        if ((int) t < step) partial[t] += partial[t + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) y[o] = partial[0];
}

#define S2F_ARGS                                                                                           \
    constant const ushort* x [[buffer(0)]], constant const uint8_t* codes [[buffer(1)]],                    \
        constant const float* scales [[buffer(2)]], device float* y [[buffer(3)]],                           \
        constant const long& n_in [[buffer(4)]], constant const long& n_out [[buffer(5)]],                  \
        constant const int& tpr [[buffer(6)]], threadgroup uint8_t* smem [[threadgroup(0)]],                \
        uint3 gpos [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]

kernel void s2_gemv_fast_kernel_staged(S2F_ARGS) {
    s2f_body<true>(x, codes, scales, y, n_in, n_out, tpr, smem, gpos, t);
}

kernel void s2_gemv_fast_kernel_global(S2F_ARGS) {
    s2f_body<false>(x, codes, scales, y, n_in, n_out, tpr, smem, gpos, t);
}
