// src/kernels/metal/s_gemv.metal - the port of src/kernels/cuda/s_gemv.cu (K12).
//
// The CUDA file is templated on CODE_BITS (2/4/8, a compile-time property of the unpacking loop) and on the
// activation kind (Q8K or Q8_0 for the warp-per-row kernel); MSL entry points cannot be templated, so each
// instantiation is a thin wrapper over a templated body and carries its width/kind in the kernel NAME.
//
// Everything arithmetic is the CUDA order verbatim: the decode `cb[code]*scale + offset` with the offset on
// the WEIGHT (before the activation multiply), the quad/octet bit order `element i+k at bits
// [k*CODE_BITS,(k+1)*CODE_BITS) of the little-endian word at byte i/PER_BYTE`, and the fixed accumulator
// trees.  The build's -ffp-contract=off stands in for the CUDA file's reliance on nvcc not contracting the
// naive loop; the parity tolerances cover that difference (measured ~1e-5 against 1e-4, s_gemv_parity).
//
// The IQ4NL codebook: the CUDA original stages the 16 bytes in shared memory because a DIVERGENT read of
// constant memory is its worst pattern (a speed question, measured 2.12x - see s_gemv.cu).  Here it is a
// program-scope constant array read directly; the VALUES are identical, and latency is not this port's
// correctness problem.
#include "strata_port.metalh"

constant const signed char sg_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,   13,   25,   38,   53,   69,   89,  113};

// block geometries as ENUM values, not `constant` variables: MSL parks program-scope variables in the
// constant address space, and those are not constant expressions - the templated kernels below need them
// at compile time (constexpr ternaries)
enum { SG_Q8K_BLOCK_BYTES = 292,   // { f32 d ; int8 qs[256] ; int16 bsums[16] }
       SG_Q8K_BLOCK_ELEMS = 256,
       SG_Q80_BLOCK_BYTES = 34,    // { fp16 d ; int8 qs[32] }
       SG_Q80_BLOCK_ELEMS = 32 };

template <int CODE_BITS>
static inline float sg_decode(int code, int bias, int codebook) {
    if (codebook == 1) return (float) sg_iq4nl[code & 0x0F];   // Codebook::Iq4Nl
    return (float) (code + bias);                              // the bias is applied to the CODE, in the integer domain
}

// `q8k_at`: dequantize element i of a Q8_K activation (292 bytes per 256 elements).  The float `d` is read
// by byte assembly - the same four bytes the CUDA `__ldg((const float*) blk)` reads, alignment-proof.
static inline float sg_q8k_at(constant const uint8_t* x, long i) {
    constant const uint8_t* blk = x + (i / SG_Q8K_BLOCK_ELEMS) * SG_Q8K_BLOCK_BYTES;
    const uint db = (uint) blk[0] | ((uint) blk[1] << 8) | ((uint) blk[2] << 16) | ((uint) blk[3] << 24);
    const float d = as_type<float>(db);
    const int q = (int) (int8_t) blk[4 + (i % SG_Q8K_BLOCK_ELEMS)];
    return d * (float) q;
}

// `q8_0_at`: block_q8_0 is {fp16 d; int8 qs[32]}, 34 bytes per 32 elements.
static inline float sg_q8_0_at(constant const uint8_t* x, long i) {
    constant const uint8_t* blk = x + (i / SG_Q80_BLOCK_ELEMS) * SG_Q80_BLOCK_BYTES;
    const ushort dbits = (ushort) blk[0] | ((ushort) blk[1] << 8);
    const float d = f32_from_f16(dbits);
    const int q = (int) (int8_t) blk[2 + (i % SG_Q80_BLOCK_ELEMS)];
    return d * (float) q;
}

// ---- s_gemv_kernel: one thread per output row, FP16 activation --------------------------------------

template <int CODE_BITS>
static inline void sg_naive_body(constant const ushort* x, constant const uint8_t* codes,
                                 constant const float* scales, constant const float* offset,
                                 device float* y, constant const long& n_in, constant const long& n_out,
                                 constant const int& bias, constant const int& codebook,
                                 constant const int& group_elems, constant const int& has_offset, uint o) {
    if (o >= (uint) n_out) return;
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long n_groups = n_in / group_elems;
    const long codes_per_row = n_in / PER_BYTE;
    constant const uint8_t* c = codes + (ulong) o * codes_per_row;
    constant const float* s = scales + (ulong) o * n_groups;
    constant const float* off = has_offset ? offset + (ulong) o * n_groups : nullptr;

    float acc = 0.0f;
    for (long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off != nullptr ? off[g] : 0.0f;
        const long base = g * (long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            // the offset belongs to the WEIGHT and is applied before the activation multiply - the Q4_K case
            // can tell the two orders apart (see the CUDA file's note)
            const float w = sg_decode<CODE_BITS>(code, bias, codebook) * d + b;
            acc += w * f32_from_f16(x[i]);
        }
    }
    y[o] = acc;
}

kernel void s_gemv_kernel_s2(constant const ushort* x [[buffer(0)]], constant const uint8_t* codes [[buffer(1)]],
                             constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],
                             device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],
                             constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],
                             constant const int& codebook [[buffer(8)]], constant const int& group_elems [[buffer(9)]],
                             constant const int& has_offset [[buffer(10)]],
                             uint o [[thread_position_in_grid]]) {
    sg_naive_body<2>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, has_offset, o);
}

kernel void s_gemv_kernel_s4(constant const ushort* x [[buffer(0)]], constant const uint8_t* codes [[buffer(1)]],
                             constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],
                             device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],
                             constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],
                             constant const int& codebook [[buffer(8)]], constant const int& group_elems [[buffer(9)]],
                             constant const int& has_offset [[buffer(10)]],
                             uint o [[thread_position_in_grid]]) {
    sg_naive_body<4>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, has_offset, o);
}

kernel void s_gemv_kernel_s8(constant const ushort* x [[buffer(0)]], constant const uint8_t* codes [[buffer(1)]],
                             constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],
                             device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],
                             constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],
                             constant const int& codebook [[buffer(8)]], constant const int& group_elems [[buffer(9)]],
                             constant const int& has_offset [[buffer(10)]],
                             uint o [[thread_position_in_grid]]) {
    sg_naive_body<8>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, has_offset, o);
}

// ---- s_gemv_q8k_kernel: one thread per row, Q8_K activation -----------------------------------------

template <int CODE_BITS>
static inline void sg_q8k_naive_body(constant const uint8_t* x, constant const uint8_t* codes,
                                     constant const float* scales, constant const float* offset,
                                     device float* y, constant const long& n_in, constant const long& n_out,
                                     constant const int& bias, constant const int& codebook,
                                     constant const int& group_elems, constant const int& has_offset, uint o) {
    if (o >= (uint) n_out) return;
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long n_groups = n_in / group_elems;
    const long codes_per_row = n_in / PER_BYTE;
    constant const uint8_t* c = codes + (ulong) o * codes_per_row;
    constant const float* s = scales + (ulong) o * n_groups;
    constant const float* off = has_offset ? offset + (ulong) o * n_groups : nullptr;

    float acc = 0.0f;
    for (long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off != nullptr ? off[g] : 0.0f;
        const long base = g * (long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            const float w = sg_decode<CODE_BITS>(code, bias, codebook) * d + b;
            acc += w * sg_q8k_at(x, i);
        }
    }
    y[o] = acc;
}

kernel void s_gemv_q8k_kernel_s4(constant const uint8_t* x [[buffer(0)]],
                                 constant const uint8_t* codes [[buffer(1)]],
                                 constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],
                                 device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],
                                 constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],
                                 constant const int& codebook [[buffer(8)]], constant const int& group_elems [[buffer(9)]],
                                 constant const int& has_offset [[buffer(10)]],
                                 uint o [[thread_position_in_grid]]) {
    sg_q8k_naive_body<4>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, has_offset, o);
}

kernel void s_gemv_q8k_kernel_s8(constant const uint8_t* x [[buffer(0)]],
                                 constant const uint8_t* codes [[buffer(1)]],
                                 constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],
                                 device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],
                                 constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],
                                 constant const int& codebook [[buffer(8)]], constant const int& group_elems [[buffer(9)]],
                                 constant const int& has_offset [[buffer(10)]],
                                 uint o [[thread_position_in_grid]]) {
    sg_q8k_naive_body<8>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, has_offset, o);
}

// ---- s_gemv_split_kernel: one BLOCK per output row, FP16 activation, quad loads ---------------------

template <int CODE_BITS>
static inline void sg_split_body(constant const ushort* x, constant const uint8_t* codes,
                                 constant const float* scales, constant const float* offset, device float* y,
                                 constant const long& n_in, constant const long& n_out, constant const int& bias,
                                 constant const int& codebook, constant const int& group_elems,
                                 constant const int& group_shift, constant const int& has_offset,
                                 constant const int& tpr, threadgroup float* partial, uint3 gpos, uint t) {
    const long o = (long) gpos.x;                 // blockIdx.x IS the row
    if (o >= n_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long n_groups = n_in / group_elems;
    constant const uint8_t* c = codes + (ulong) o * (n_in / PER_BYTE);
    constant const float* s = scales + (ulong) o * n_groups;
    constant const float* off = has_offset ? offset + (ulong) o * n_groups : nullptr;

    constexpr int QE = 4;
    constexpr int QB = QE * CODE_BITS / 8;        // 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    long i = (long) t * QE;
    for (; i + QE <= n_in; i += (long) tpr * QE) {
        const long g = i >> group_shift;          // the whole quad is in one group: group_elems % 4 == 0
        const float d = s[g];
        const float b = off != nullptr ? off[g] : 0.0f;
        constant const uint8_t* cp = c + i / PER_BYTE;
        unsigned v;
        if (QB == 1) v = cp[0];
        else if (QB == 2) v = *reinterpret_cast<constant const ushort*>(cp);
        else v = *reinterpret_cast<constant const uint*>(cp);
        const float f0 = f32_from_f16(x[i]);
        const float f1 = f32_from_f16(x[i + 1]);
        const float f2 = f32_from_f16(x[i + 2]);
        const float f3 = f32_from_f16(x[i + 3]);
        acc0 += (sg_decode<CODE_BITS>((int) (v & MASK), bias, codebook) * d + b) * f0;
        acc1 += (sg_decode<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook) * d + b) * f1;
        acc2 += (sg_decode<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b) * f2;
        acc3 += (sg_decode<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b) * f3;
    }
    // the last, partial quad - at most one per thread
    for (; i < n_in; i += (long) tpr * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long e = i + k;
            const long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (sg_decode<CODE_BITS>(code, bias, codebook) * s[g] +
                     (off != nullptr ? off[g] : 0.0f)) * f32_from_f16(x[e]);
        }
    }
    partial[t] = (acc0 + acc1) + (acc2 + acc3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = tpr / 2; step > 0; step >>= 1) {
        if ((int) t < step) partial[t] += partial[t + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) y[o] = partial[0];
}

#define SG_SPLIT_ARGS                                                                                        \
    constant const ushort* x [[buffer(0)]], constant const uint8_t* codes [[buffer(1)]],                     \
        constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],              \
        device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],                              \
        constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],                    \
        constant const int& codebook [[buffer(8)]], constant const int& group_elems [[buffer(9)]],           \
        constant const int& group_shift [[buffer(10)]], constant const int& has_offset [[buffer(11)]],       \
        constant const int& tpr [[buffer(12)]], threadgroup float* partial [[threadgroup(0)]],               \
        uint3 gpos [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]

kernel void s_gemv_split_kernel_s2(SG_SPLIT_ARGS) {
    sg_split_body<2>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, group_shift,
                     has_offset, tpr, partial, gpos, t);
}

kernel void s_gemv_split_kernel_s4(SG_SPLIT_ARGS) {
    sg_split_body<4>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, group_shift,
                     has_offset, tpr, partial, gpos, t);
}

kernel void s_gemv_split_kernel_s8(SG_SPLIT_ARGS) {
    sg_split_body<8>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, group_shift,
                     has_offset, tpr, partial, gpos, t);
}

// ---- s_gemv_q8_split_kernel: one WARP per output row, quantized activation, QE = 16 ------------------

template <int CODE_BITS, bool Q8K>
static inline void sg_q8_split_body(constant const uint8_t* x, constant const uint8_t* codes,
                                    constant const float* scales, constant const float* offset, device float* y,
                                    constant const long& n_in, constant const long& n_out, constant const int& bias,
                                    constant const int& codebook, constant const int& group_shift,
                                    constant const int& has_offset, constant const uint& block, uint3 gpos,
                                    uint lane, uint sg) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const uint warps_per_block = block >> 5;
    const long o = (long) gpos.x * (long) warps_per_block + sg;   // blockIdx*wpb + warp-in-block
    if (o >= n_out) return;                                       // per-WARP: whole warps leave together

    const long n_groups = n_in >> group_shift;
    const long codes_per_row = n_in / PER_BYTE;
    constant const uint8_t* c = codes + (ulong) o * codes_per_row;
    constant const float* s = scales + (ulong) o * n_groups;
    constant const float* off = has_offset ? offset + (ulong) o * n_groups : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
    // EIGHT CONSECUTIVE ELEMENTS PER LANE was the CUDA experiment; this port keeps QE = 16 (a double quad,
    // four code words) exactly as the file stands - two or four independent words per lane, one scale.
    constexpr int QE = 16;
    constexpr int QW = 4 * CODE_BITS / 8;         // bytes per FOUR-code word: 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    constexpr int blk_elems = Q8K ? SG_Q8K_BLOCK_ELEMS : SG_Q80_BLOCK_ELEMS;
    constexpr int blk_bytes = Q8K ? SG_Q8K_BLOCK_BYTES : SG_Q80_BLOCK_BYTES;
    long i = (long) lane * QE;
    for (; i + QE <= n_in; i += 32 * QE) {
        const long g = i >> group_shift;          // the whole octet is in one group: group_elems % 16 == 0
        const float d = s[g];
        const float b = off != nullptr ? off[g] : 0.0f;
        constant const uint8_t* cp = c + i / PER_BYTE;
        unsigned v, v2, v3, v4;
        if (QW == 1) {
            v = cp[0]; v2 = cp[1]; v3 = cp[2]; v4 = cp[3];
        } else if (QW == 2) {
            v = *reinterpret_cast<constant const ushort*>(cp);
            v2 = *reinterpret_cast<constant const ushort*>(cp + 2);
            v3 = *reinterpret_cast<constant const ushort*>(cp + 4);
            v4 = *reinterpret_cast<constant const ushort*>(cp + 6);
        } else {
            v = *reinterpret_cast<constant const uint*>(cp);
            v2 = *reinterpret_cast<constant const uint*>(cp + 4);
            v3 = *reinterpret_cast<constant const uint*>(cp + 8);
            v4 = *reinterpret_cast<constant const uint*>(cp + 12);
        }
        // the activation block is hoisted out of the sixteen elements: i % blk_elems is always a multiple of
        // 16, so no lane-iteration straddles a block boundary and one scale/one base pointer serve all 16
        constant const uint8_t* xb = x + (i / blk_elems) * blk_bytes;
        const int xi = (int) (i % blk_elems);
        float xd;
        if (Q8K) {
            const uint db = (uint) xb[0] | ((uint) xb[1] << 8) | ((uint) xb[2] << 16) | ((uint) xb[3] << 24);
            xd = as_type<float>(db);
        } else {
            const ushort dbits = (ushort) xb[0] | ((ushort) xb[1] << 8);
            xd = f32_from_f16(dbits);
        }
        constant const int8_t* xq = reinterpret_cast<constant const int8_t*>(xb + (Q8K ? 4 : 2)) + xi;

        const float w0 = sg_decode<CODE_BITS>((int) (v & MASK), bias, codebook) * d + b;
        const float w1 = sg_decode<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w2 = sg_decode<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w3 = sg_decode<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w4 = sg_decode<CODE_BITS>((int) (v2 & MASK), bias, codebook) * d + b;
        const float w5 = sg_decode<CODE_BITS>((int) ((v2 >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w6 = sg_decode<CODE_BITS>((int) ((v2 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w7 = sg_decode<CODE_BITS>((int) ((v2 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        acc0 += w0 * (xd * (float) xq[0]);
        acc1 += w1 * (xd * (float) xq[1]);
        acc2 += w2 * (xd * (float) xq[2]);
        acc3 += w3 * (xd * (float) xq[3]);
        acc4 += w4 * (xd * (float) xq[4]);
        acc5 += w5 * (xd * (float) xq[5]);
        acc6 += w6 * (xd * (float) xq[6]);
        acc7 += w7 * (xd * (float) xq[7]);
        const float w8 = sg_decode<CODE_BITS>((int) (v3 & MASK), bias, codebook) * d + b;
        const float w9 = sg_decode<CODE_BITS>((int) ((v3 >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w10 = sg_decode<CODE_BITS>((int) ((v3 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w11 = sg_decode<CODE_BITS>((int) ((v3 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w12 = sg_decode<CODE_BITS>((int) (v4 & MASK), bias, codebook) * d + b;
        const float w13 = sg_decode<CODE_BITS>((int) ((v4 >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w14 = sg_decode<CODE_BITS>((int) ((v4 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w15 = sg_decode<CODE_BITS>((int) ((v4 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        acc8 += w8 * (xd * (float) xq[8]);
        acc9 += w9 * (xd * (float) xq[9]);
        acc10 += w10 * (xd * (float) xq[10]);
        acc11 += w11 * (xd * (float) xq[11]);
        acc12 += w12 * (xd * (float) xq[12]);
        acc13 += w13 * (xd * (float) xq[13]);
        acc14 += w14 * (xd * (float) xq[14]);
        acc15 += w15 * (xd * (float) xq[15]);
    }
    // the last, partial quad - at most one per lane
    for (; i < n_in; i += 32 * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long e = i + k;
            const long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (sg_decode<CODE_BITS>(code, bias, codebook) * s[g] + (off != nullptr ? off[g] : 0.0f)) *
                    (Q8K ? sg_q8k_at(x, e) : sg_q8_0_at(x, e));
        }
    }
    float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
                (((acc8 + acc9) + (acc10 + acc11)) + ((acc12 + acc13) + (acc14 + acc15)));
    for (int step = 16; step > 0; step >>= 1) acc += simd_shuffle_down(acc, step);
    if (lane == 0) y[o] = acc;
}

#define SG_Q8_SPLIT_ARGS                                                                                    \
    constant const uint8_t* x [[buffer(0)]], constant const uint8_t* codes [[buffer(1)]],                   \
        constant const float* scales [[buffer(2)]], constant const float* offset [[buffer(3)]],             \
        device float* y [[buffer(4)]], constant const long& n_in [[buffer(5)]],                             \
        constant const long& n_out [[buffer(6)]], constant const int& bias [[buffer(7)]],                   \
        constant const int& codebook [[buffer(8)]], constant const int& group_shift [[buffer(9)]],          \
        constant const int& has_offset [[buffer(10)]], constant const uint& block [[buffer(11)]],           \
        uint3 gpos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],               \
        uint sg [[simdgroup_index_in_threadgroup]]

kernel void s_gemv_q8_split_kernel_s4_q8k(SG_Q8_SPLIT_ARGS) {
    sg_q8_split_body<4, true>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_shift,
                              has_offset, block, gpos, lane, sg);
}

kernel void s_gemv_q8_split_kernel_s8_q8k(SG_Q8_SPLIT_ARGS) {
    sg_q8_split_body<8, true>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_shift,
                              has_offset, block, gpos, lane, sg);
}

kernel void s_gemv_q8_split_kernel_s4_q80(SG_Q8_SPLIT_ARGS) {
    sg_q8_split_body<4, false>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_shift,
                               has_offset, block, gpos, lane, sg);
}

kernel void s_gemv_q8_split_kernel_s8_q80(SG_Q8_SPLIT_ARGS) {
    sg_q8_split_body<8, false>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_shift,
                               has_offset, block, gpos, lane, sg);
}
