// src/kernels/metal/native_mmvq.metal - the port of src/kernels/cuda/native_mmvq.cu: the pinned
// llama.cpp sm_120 generic MMVQ family (Q4_0, Q5_0, Q8_0, IQ4_NL, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS)
// plus its Q8_1 quantizer.  Every dot expression, integer trick, load width and accumulation order is the
// CUDA file's own; the structures that changed are listed at the bottom of this comment.
//
// The CUDA file's arithmetic is dp4a integer dots, __byte_perm SWAR and fp16->fp32 scale products - no
// double anywhere, so none of strata_port.metalh's fp64 emulations are needed.  What is spelled instead of
// a CUDA intrinsic (each bit-identical, the sources named - the same set iq_kernels.metal carries):
//   * __dp4a -> nmv_dp4a: dp4a.hpp's sm_60 fallback, that header's documented bit-exact __dp4a.
//   * __byte_perm -> nmv_byte_perm: "result byte i is byte s.nibble[i] & 7 of the pair {y:x}, x the low
//     word" (hip_compat/intrinsics.hpp's CUDA default-mode semantics).
//   * __vsubss4 -> nmv_vsubss4: the packed signed-byte SATURATING subtract from the same header (Q3_K and
//     Q6_K use it); built on nmv_vsub4, the Hacker's Delight 2-18 wrapping SWAR subtract.
//   * __shfl_xor_sync -> simd_shuffle_xor (the warp butterfly); __low2float / __half22float2 -> (float)
//     casts on the half2 members (half -> float is exact); roundf -> nmv_roundf (ties away from zero; MSL
//     has no precise::round); fabs/fmax -> metal::precise:: (rule 5).
//   * the IQ4_NL codebook: the CUDA file's __device__ iq4nl_values table IS ggml-common.h's
//     kvalues_iq4nl (same 16 values), so the table rides the metallib as the program-scope constant the
//     GGML_COMMON_IMPL_METAL pair provides - the iq_kernels precedent for cudaMemcpyToSymbol.
//
// The block structs are the ggml-common.h METAL decls (the same header the CUDA file includes; every
// layout the CUDA file pins with static_asserts is the same struct here: block_q5_K == Q5KBlock etc.).
//
// THE RESTRUCTURES (structure only; every output value's operation sequence is the CUDA one):
//   * template<bool SmallK> / template<typename Weight, int Qi> / template<typename F, int NCOLS, int NW,
//     int ROWS> become concrete instantiations with the template arguments in the name, as the port
//     always does: native_q5_k_mmvq_kernel_small/_large, native_small_mmvq_kernel_q8_0_small/_large,
//     native_mmvq_multi_kernel_<fmt>_r1/_r2/_r4 (ROWS).  NCOLS and NW stay RUNTIME scalars: a column's
//     accumulation chain (tmp[j][i] += ... in ascending kbx, cross-warp partials in ascending l, then the
//     XOR butterfly) never mixes columns, so a column's bits do not depend on how many other columns the
//     same call carries - which is exactly the multi == single contract iq_parity and mmvq_multi_parity
//     check.  NW is runtime so one _r2 instantiation serves the upstream layout's two warp counts
//     (NW = 4 for ncols <= 4, 2 above), with BPI = F::BPI * NW / WARPS computed in the kernel exactly as
//     the CUDA template did at compile time.
//   * the per-column kernels call the SAME Fmt::load/Fmt::apply pair the multi kernel calls.  The CUDA
//     file writes the dot twice (an inline q*_q8_dot and the traits' load/apply split) and its own
//     mmvq_multi_parity holds the two bitwise equal; transcribing the split once and using it for both
//     paths keeps that equality by construction instead of by duplication.
//   * blockIdx-as-data-index (rule 6): the mmvq kernels' blockIdx.x is the ROW GROUP index, so the port
//     reads threadgroup_position_in_grid; tid = 32 * threadIdx.y + threadIdx.x is the flat
//     thread_index_in_threadgroup.  The quantizer's i = blockIdx * blockDim + threadIdx is
//     thread_position_in_grid (rule 6's second form).
#include "strata_port.metalh"

// llama.cpp's block structs and the IQ4_NL codebook, unchanged - the same include the CUDA file makes
#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#include "../../../third_party/ggml/ggml-common.h"

// the CUDA file's block names, for reading the dots against its source
typedef block_q5_K  Q5KBlock;
typedef block_q8_1  Q81Block;
typedef block_q2_0  Q20Block;
typedef block_q3_K  Q3KBlock;
typedef block_iq4_xs IQ4XSBlock;
typedef block_q4_K  Q4KBlock;
typedef block_q6_K  Q6KBlock;
typedef block_q4_0  Q40Block;
typedef block_q5_0  Q50Block;
typedef block_q8_0  Q80Block;
typedef block_iq4_nl IQ4NLBlock;

constant const int NMV_WARPS = 4;        // warps per block (the CUDA file's WARPS)

// ---------------------------------------------------------------- the CUDA intrinsic stand-ins
// CUDA's __byte_perm (default mode), per this repo's HIP compat shim: result byte i is byte (s.nibble[i] & 7)
// of the eight bytes {y:x}, x the low word - no sign-replication mode bit.
static inline uint nmv_byte_perm(uint x, uint y, uint s) {
    uint r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint idx = (s >> (4 * i)) & 0x7u;
        const uint byte_ = idx < 4u ? (x >> (8 * idx)) & 0xFFu : (y >> (8 * (idx - 4u))) & 0xFFu;
        r |= byte_ << (8 * i);
    }
    return r;
}

// CUDA's __vsub4 (wrapping per-byte subtract), Hacker's Delight 2-18 SWAR - nmv_vsubss4's base.
static inline int nmv_vsub4(int a, int b) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    return as_type<int>(((ua | 0x80808080u) - (ub & 0x7F7F7F7Fu)) ^ ((ua ^ ~ub) & 0x80808080u));
}
// CUDA's packed signed-byte SATURATING subtract __vsubss4 (hip_compat's vsubss4): the wrapping difference,
// and in each lane that overflowed on the minuend's side, 0x7f or 0x80.
static inline int nmv_vsubss4(int a, int b) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    const uint d = as_type<uint>(nmv_vsub4(a, b));
    const uint overflow = (ua ^ ub) & (ua ^ d) & 0x80808080u;
    const uint mask = (overflow >> 7) * 0xFFu;
    const uint bound = 0x7F7F7F7Fu + ((ua & 0x80808080u) >> 7);
    return as_type<int>((d & ~mask) | (bound & mask));
}

// CUDA's signed __dp4a: dp4a.hpp's own (bit-exact) definition - four signed-byte products, wrapping int32.
static inline int nmv_dp4a(int a, int b, int c) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    int r = c;
    for (int lane = 0; lane < 4; ++lane) {
        const uint va = (ua >> (8 * lane)) & 0xFFu, vb = (ub >> (8 * lane)) & 0xFFu;
        const int sa = va < 0x80u ? (int) va : (int) va - 0x100;
        const int sb = vb < 0x80u ? (int) vb : (int) vb - 0x100;
        r += sa * sb;
    }
    return r;
}

// load_int_b2: the pinned helper's pair of 16-bit loads and little-endian combine (Q3_K's 110-byte and
// Q2_0's 18-byte strides give the small blocks only two-byte alignment - the loads stay narrow, rule B).
static inline int nmv_load_int_b2(constant const uint16_t* x, int i32) {
    int value = (int) x[2 * i32 + 0] << 0;
    value |= (int) x[2 * i32 + 1] << 16;
    return value;
}
// a 4-byte load from a naturally aligned constant byte pointer (the activation codes' view)
static inline int nmv_get_int_b4(constant const uint8_t* x, int i32) {
    return reinterpret_cast<constant const int*>(x)[i32];
}

// CUDA roundf: nearest integer, ties AWAY from zero (MSL's rint is ties-to-even, no precise::round).
static inline float nmv_roundf(float x) {
    const float r = floor(metal::precise::fabs(x) + 0.5f);   // floor is exact under any math mode
    return x < 0.0f ? -r : r;
}

// the CUDA file's __shfl_xor_sync butterflies: after warp_sum every lane holds the sum, after warp_max the
// max (the literal 16 is WARP / 2 - program-scope constants do not make array or unroll bounds)
static inline float nmv_warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += simd_shuffle_xor(x, (uint) offset);
    return x;
}
static inline float nmv_warp_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1)
        x = metal::precise::fmax(x, simd_shuffle_xor(x, (uint) offset));
    return x;
}

// the pinned nonlinear IQ4 codebook two-stage byte lookup (iq4_table_lookup); the table is kvalues_iq4nl,
// which is exactly the CUDA file's iq4nl_values (16 values, unchanged).
static inline int2 nmv_iq4_table_lookup(int q4) {
    constant const uint* table32 = reinterpret_cast<constant const uint*>(kvalues_iq4nl);
    uint tmp[2];
    const uint uq4 = as_type<uint>(q4);
    const uint low_high_selection_indices = 0x32103210u | ((uq4 & 0x88888888u) >> 1);
    for (uint i = 0; i < 2; ++i) {
        const uint shift = 16 * i;
        const uint low = nmv_byte_perm(table32[0], table32[1], uq4 >> shift);
        const uint high = nmv_byte_perm(table32[2], table32[3], uq4 >> shift);
        tmp[i] = nmv_byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return int2(as_type<int>(nmv_byte_perm(tmp[0], tmp[1], 0x6420u)),
                as_type<int>(nmv_byte_perm(tmp[0], tmp[1], 0x7531u)));
}

// ---------------------------------------------------------------- the *_impl dots (transcribed verbatim)
// q5_q8_dot_impl: the exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
static inline float nmv_q5_q8_dot_impl(const thread int* vl, const thread int* vh, const thread int* u,
                                       const thread uint8_t* sc, const thread uint8_t* m, half2 dm5,
                                       const thread float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = nmv_dp4a(v0i, u[2 * i], nmv_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = nmv_dp4a(0x01010101, u[2 * i], nmv_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);
    }
    const float dm5x = (float) dm5.x, dm5y = (float) dm5.y;
    return dm5x * sumf_d - dm5y * sumf_m;
}

// q4_q8_dot_impl: the exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
static inline float nmv_q4_q8_dot_impl(const thread int* v, const thread int* u, const thread uint8_t* sc,
                                       const thread uint8_t* m, half2 dm4, const thread float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = nmv_dp4a(v1i, u[2 * i + 1], nmv_dp4a(v0i, u[2 * i], 0));
        const int dot2 = nmv_dp4a(0x01010101, u[2 * i + 1], nmv_dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);
    }
    const float dm4x = (float) dm4.x, dm4y = (float) dm4.y;
    return dm4x * sumf_d - dm4y * sumf_m;
}

// the 6-bit scales and mins of the group pair, branchless (llama.cpp; shared by Q4_K and Q5_K)
static inline void nmv_k_scale_min(constant const uint8_t* scales8, int bq8_offset, thread uint16_t* aux) {
    constant const uint16_t* scales = reinterpret_cast<constant const uint16_t*>(scales8);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint s0 = scales[jm];
    const uint s2 = scales[jm + 2];
    const uint s4 = scales[jm + 4];
    const uint hi = (uint) -(int) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

// q3_q8_dot_impl: signed scales, __vsubss4, DP4A order, the float accumulation sequence.
static inline float nmv_q3_q8_dot_impl(int vl, int vh, const thread int* u, constant const uint8_t* scales,
                                       int scale_offset, float d3, const thread float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = nmv_vsubss4(vil, vih);
        sumf += d8[i] * (float) (nmv_dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

// q6_q8_dot_impl: signed per-16-element scales, signed-byte saturating subtraction, DP4A order.
static inline float nmv_q6_q8_dot_impl(int vl, int vh, const thread int* u, constant const int8_t* scales,
                                       float d, const thread float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = nmv_vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (float) (nmv_dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

// ---------------------------------------------------------------- the load/apply traits
// Each format splits its dot into `load` (everything that depends only on the weight block) and `apply`
// (the activation loads and the original *_impl expression), exactly the CUDA file's plan-v0.3-P3 traits:
// load runs once per (row, block), apply once per column, and apply's operations are the ncols == 1 dot's
// own, in the same order - which is what makes every column of a multi call bitwise equal to a
// single-column call.  `blk` is the weight block's first byte; the small formats' decode is a few integer
// ops, so their load keeps nothing (the CUDA SmallTraits kept the pointer; the pointer is passed instead).
//
// enum members carry the compile-time ints (MSL program-scope variables must live in the constant address
// space; an enum is the spelling that carries a compile-time int without one - iq_kernels' precedent):
//   BYTES = block stride in bytes, DIV = elements per block, T = the tid-to-block divisor, KBY = q8_1
//   blocks per weight block, BPI = blocks per iteration at NW = 4 warps (the multi kernel scales it).
template<int F> struct nmv_Fmt;

// Q5_K: QK=256, QI=32, VDR=2
template<> struct nmv_Fmt<0> {
    enum : int { BYTES = 176, DIV = 256, T = 16, KBY = 8, BPI = 8 };
    static int kqs(int tid) { return 2 * (tid % 16); }                     // VDR * (tid % (QI / VDR))
    struct W { int vl[2]; int vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    static W load(constant const uint8_t* blk, int iqs) {
        constant const Q5KBlock* bq5 = reinterpret_cast<constant const Q5KBlock*>(blk);
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        constant const int* ql = reinterpret_cast<constant const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        constant const int* qh = reinterpret_cast<constant const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        nmv_k_scale_min(bq5->scales, r.bq8_offset, r.aux);
        r.dm = bq5->dm;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        (void) blk;
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            constant const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = (float) bq8i->ds.x;
            constant const int* q8 = reinterpret_cast<constant const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const thread uint8_t* sc = reinterpret_cast<const thread uint8_t*>(r.aux);
        return nmv_q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};

// Q4_K: QK=256, QI=32, VDR=2
template<> struct nmv_Fmt<1> {
    enum : int { BYTES = 144, DIV = 256, T = 16, KBY = 8, BPI = 8 };
    static int kqs(int tid) { return 2 * (tid % 16); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    static W load(constant const uint8_t* blk, int iqs) {
        constant const Q4KBlock* bq4 = reinterpret_cast<constant const Q4KBlock*>(blk);
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        constant const int* ql = reinterpret_cast<constant const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        nmv_k_scale_min(bq4->scales, r.bq8_offset, r.aux);
        r.dm = bq4->dm;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        (void) blk;
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            constant const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = (float) bq8i->ds.x;
            constant const int* q8 = reinterpret_cast<constant const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const thread uint8_t* sc = reinterpret_cast<const thread uint8_t*>(r.aux);
        return nmv_q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};

// Q2_0: QK=64, QI=2, VDR=1 - the weight block is only 2-byte aligned, so qs is loaded as int16_t pairs
// (the exact pinned vec_dot_q2_0_q8_1 note; rule B's narrow-first order).
template<> struct nmv_Fmt<2> {
    enum : int { BYTES = 18, DIV = 64, T = 2, KBY = 2, BPI = 64 };
    static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4]; int qy[4]; float d2; };
    static W load(constant const uint8_t* blk, int iqs) {
        constant const Q20Block* w = reinterpret_cast<constant const Q20Block*>(blk);
        W r;
        r.d2 = (float) w->d;
        constant const int16_t* qs = reinterpret_cast<constant const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = as_type<int>(nmv_byte_perm(0x020100ffu, 0x020100ffu, as_type<uint>(q) >> 0));
            const int qo = as_type<int>(nmv_byte_perm(0x020100ffu, 0x020100ffu, as_type<uint>(q) >> 2));
            r.qx[j] = as_type<int>(nmv_byte_perm(as_type<uint>(qe), as_type<uint>(qo), 0x5140u));
            r.qy[j] = as_type<int>(nmv_byte_perm(as_type<uint>(qe), as_type<uint>(qo), 0x7362u));
        }
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        (void) blk;
        constant const Q81Block* chunk = bq8 + iqs;
        constant const int* q8 = reinterpret_cast<constant const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = nmv_dp4a(q8[j * 2], r.qx[j], sumi);
            sumi = nmv_dp4a(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = (float) chunk->ds.x;
        return r.d2 * d8 * (float) sumi;
    }
};

// Q3_K: QK=256, QI=16, VDR=1 (the 110-byte stride gives alternate blocks two-byte alignment - b2 loads)
template<> struct nmv_Fmt<3> {
    enum : int { BYTES = 110, DIV = 256, T = 16, KBY = 8, BPI = 8 };
    static int kqs(int tid) { return tid % 16; }
    struct W { int vl; int vh; float d; int scale_offset; int bq8_offset; };
    static W load(constant const uint8_t* blk, int iqs) {
        constant const Q3KBlock* w = reinterpret_cast<constant const Q3KBlock*>(blk);
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = (float) w->d;
        r.vl = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qs), iqs);
        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        r.vh = ~nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->hmask), iqs % 8) >> r.bq8_offset;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        constant const Q3KBlock* w = reinterpret_cast<constant const Q3KBlock*>(blk);
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[r.bq8_offset + i].qs), iqs % 8);
            d8[i] = (float) bq8[r.bq8_offset + i].ds.x;
        }
        return nmv_q3_q8_dot_impl(r.vl, r.vh, u, w->scales, r.scale_offset, r.d, d8);
    }
};

// Q6_K: QK=256, QI=32, VDR=1
template<> struct nmv_Fmt<4> {
    enum : int { BYTES = 210, DIV = 256, T = 32, KBY = 8, BPI = 4 };
    static int kqs(int tid) { return tid % 32; }
    struct W { int vl; int vh; float d; int bq8_offset; };
    static W load(constant const uint8_t* blk, int iqs) {
        constant const Q6KBlock* w = reinterpret_cast<constant const Q6KBlock*>(blk);
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->ql), iqs);
        r.vh = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qh), 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.d = (float) w->d;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        constant const Q6KBlock* w = reinterpret_cast<constant const Q6KBlock*>(blk);
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[r.bq8_offset + 2 * i].qs), iqs % 8);
            d8[i] = (float) bq8[r.bq8_offset + 2 * i].ds.x;
        }
        return nmv_q6_q8_dot_impl(r.vl, r.vh, u, w->scales + scale_offset, r.d, d8);
    }
};

// IQ4_XS: QK=256, QI=32, VDR=4
template<> struct nmv_Fmt<5> {
    enum : int { BYTES = 136, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    static W load(constant const uint8_t* blk, int iqs) {
        constant const IQ4XSBlock* w = reinterpret_cast<constant const IQ4XSBlock*>(blk);
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            r.v[j] = nmv_iq4_table_lookup(nmv_get_int_b4(w->qs, iqs + j));
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) w->d;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* bq8, int iqs) {
        (void) blk;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[iqs / 4].qs), j);
            const int u1 = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8[iqs / 4].qs), j + 4);
            sumi = nmv_dp4a(r.v[j].x, u0, sumi);
            sumi = nmv_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * (float) bq8[iqs / 4].ds.x;
        return d * (float) sumi;
    }
};

// Lossless IQ4_XS view: the original eight header bytes and 256 signed codebook bytes.
// No weight scale or floating-point operation changes. The dot/accumulation stays Fmt<5>'s.
template<> struct nmv_Fmt<11> {
    enum : int { BYTES = 264, DIV = 256, T = 8, KBY = 8, BPI = 16 };
    static int kqs(int tid) { return 4 * (tid % 8); }
    using W = nmv_Fmt<5>::W;
    static W load(constant const uint8_t* blk, int iqs) {
        constant const IQ4XSBlock* header = reinterpret_cast<constant const IQ4XSBlock*>(blk);
        constant const int* codes = reinterpret_cast<constant const int*>(blk + 8 + (iqs / 4) * 32);
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = int2(codes[j], codes[j + 4]);
        r.ls = ((header->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
               (int) (((header->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) header->d;
        return r;
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* x, int iqs) {
        return nmv_Fmt<5>::apply(r, blk, x, iqs);
    }
};

kernel void native_iq4_xs_expand(constant const uint8_t* src [[buffer(0)]],
                                device uint8_t* dst [[buffer(1)]],
                                constant const ulong& words [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= words) return;
    const ulong block = i / 32; const uint word = i % 32;
    constant const uint8_t* in = src + block * 136;
    device uint8_t* out = dst + block * 264;
    if (word == 0) *reinterpret_cast<device uint2*>(out) = *reinterpret_cast<constant const uint2*>(in);
    const uchar4 q = as_type<uchar4>(reinterpret_cast<constant const uint*>(in + 8)[word]);
    const char4 lo(kvalues_iq4nl[q.x & 15], kvalues_iq4nl[q.y & 15], kvalues_iq4nl[q.z & 15], kvalues_iq4nl[q.w & 15]);
    const char4 hi(kvalues_iq4nl[q.x >> 4], kvalues_iq4nl[q.y >> 4], kvalues_iq4nl[q.z >> 4], kvalues_iq4nl[q.w >> 4]);
    device int* codes = reinterpret_cast<device int*>(out + 8 + (word / 4) * 32);
    codes[word % 4] = as_type<int>(lo); codes[4 + word % 4] = as_type<int>(hi);
}

// The four 32-element formats: QI = 4 for Q4_0 / Q5_0 / IQ4_NL, 8 for Q8_0.
// The affine Q4_0 / Q5_0 correction consumes the original-input sum stored in Q8_1, exactly as the pinned
// CUDA dot does; a signed-integer code substitution would differ.

// Q4_0 (the exact pinned small_q8_dot(Q40Block*, ...))
template<> struct nmv_Fmt<6> {
    enum : int { BYTES = 18, DIV = 32, T = 2, KBY = 1, BPI = 64 };
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int dummy; };                                    // the CUDA W held the block pointer; it is passed
    static W load(constant const uint8_t* blk, int iqs) {
        (void) blk; (void) iqs;
        return W{0};
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* x, int iqs) {
        (void) r;
        constant const Q40Block* w = reinterpret_cast<constant const Q40Block*>(blk);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qs), iqs + i);
            const int vi0 = (v >> 0) & 0x0F0F0F0F;
            const int vi1 = (v >> 4) & 0x0F0F0F0F;
            sumi = nmv_dp4a(vi0, nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(x->qs), iqs + i), sumi);
            sumi = nmv_dp4a(vi1, nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(x->qs), iqs + i + 4), sumi);
        }
        const float dsx = (float) x->ds.x, dsy = (float) x->ds.y;
        const float d = (float) w->d;
        return d * ((float) sumi * dsx - 4.0f * dsy);
    }
};

// Q5_0
template<> struct nmv_Fmt<7> {
    enum : int { BYTES = 22, DIV = 32, T = 2, KBY = 1, BPI = 64 };
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int dummy; };
    static W load(constant const uint8_t* blk, int iqs) {
        (void) blk; (void) iqs;
        return W{0};
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* x, int iqs) {
        (void) r;
        constant const Q50Block* w = reinterpret_cast<constant const Q50Block*>(blk);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qs), iqs + i);
            const int vh = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qh), 0) >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            sumi = nmv_dp4a(vi0, nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(x->qs), iqs + i), sumi);
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            sumi = nmv_dp4a(vi1, nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(x->qs), iqs + i + 4), sumi);
        }
        const float dsx = (float) x->ds.x, dsy = (float) x->ds.y;
        const float d = (float) w->d;
        return d * ((float) sumi * dsx - 8.0f * dsy);
    }
};

// Q8_0 (QI = 8)
template<> struct nmv_Fmt<8> {
    enum : int { BYTES = 34, DIV = 32, T = 4, KBY = 1, BPI = 32 };
    static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int dummy; };
    static W load(constant const uint8_t* blk, int iqs) {
        (void) blk; (void) iqs;
        return W{0};
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* x, int iqs) {
        (void) r;
        constant const Q80Block* w = reinterpret_cast<constant const Q80Block*>(blk);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qs), iqs + i);
            const int u = nmv_get_int_b4(reinterpret_cast<constant const uint8_t*>(x->qs), iqs + i);
            sumi = nmv_dp4a(v, u, sumi);
        }
        const float d0 = (float) w->d;
        const float d1 = (float) x->ds.x;
        return d0 * d1 * (float) sumi;
    }
};

// IQ4_NL: the codebook lookup pair per 32-element block
template<> struct nmv_Fmt<9> {
    enum : int { BYTES = 18, DIV = 32, T = 2, KBY = 1, BPI = 64 };
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int dummy; };
    static W load(constant const uint8_t* blk, int iqs) {
        (void) blk; (void) iqs;
        return W{0};
    }
    static float apply(const thread W& r, constant const uint8_t* blk, constant const Q81Block* x, int iqs) {
        (void) r;
        constant const IQ4NLBlock* w = reinterpret_cast<constant const IQ4NLBlock*>(blk);
        constant const int* q8 = reinterpret_cast<constant const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int2 v = nmv_iq4_table_lookup(
                nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qs), iqs + i));
            sumi = nmv_dp4a(v.x, q8[i], sumi);
            sumi = nmv_dp4a(v.y, q8[i + 4], sumi);
        }
        const float d = (float) w->d * (float) x->ds.x;
        return d * (float) sumi;
    }
};

// load + apply composed: the ncols == 1 dot the CUDA kernels call (its mmvq_multi_parity holds the two
// bitwise equal, so the split transcribed once serves both paths)
template<int F>
static inline float nmv_dot(constant const uint8_t* blk, constant const Q81Block* x, int iqs) {
    const typename nmv_Fmt<F>::W wv = nmv_Fmt<F>::load(blk, iqs);
    return nmv_Fmt<F>::apply(wv, blk, x, iqs);
}

// ---------------------------------------------------------------- the kernel bodies
// The ncols = 1 oracle: NW = WARPS = 4 warps and ROWS rows (1, or 4 for small K), the CUDA thread-to-block
// mapping (kbx = tid / T striding BPI), warp-ascending shared partials, then the XOR tree.  The partial
// array lives in the KERNEL (MSL: threadgroup variables are kernel-scope) and arrives as a flat pointer.
template<int F, int ROWS>
static inline void nmv_single_body(constant const uint8_t* w, constant const Q81Block* x, device float* y,
                                   int n_in, int n_out, threadgroup float* partial, uint gpos_x, uint tid) {
    using Fmt = nmv_Fmt<F>;
    const int ty = (int) (tid >> 5);
    const int lane = (int) (tid & 31u);
    const int row0 = ROWS * (int) gpos_x;
    const int blocks_per_row = n_in / Fmt::DIV;
    float tmp[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = (int) tid / Fmt::T; kbx < blocks_per_row; kbx += Fmt::BPI) {
        const int kby = kbx * Fmt::KBY;
        const int kqs = Fmt::kqs((int) tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // the source assumes allocator padding for partial row groups; this guard preserves every
            // valid row's math without an out-of-bounds read
            if (row0 + i < n_out) {
                const size_t block = (size_t) (row0 + i) * (size_t) blocks_per_row + (size_t) kbx;
                tmp[i] += nmv_dot<F>(w + block * (size_t) Fmt::BYTES, x + kby, kqs);
            }
        }
    }
    if (ty > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[((ty - 1) * ROWS + i) * 32 + lane] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) tmp[i] += partial[(l * ROWS + i) * 32 + lane];
        tmp[i] = nmv_warp_sum(tmp[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The multi-column kernel: one body for both layouts (NW runtime, rule-discussed in the file comment).  The
// EXACT layout (nw = 4, rows = 1 or 4) is the ncols = 1 layout, so every column is bitwise equal to a
// single-column call; the UPSTREAM layout (rows = 2, nw = 4 for ncols <= 4 else 2) is llama.cpp's generic
// multi-column table - faster, equal only to float rounding, selected by native_mmvq_set_multi_exact(false).
// partial is [(NW-1) <= 3][MAX_NCOLS = 8][ROWS][32] at the kernel scope, addressed flat.
template<int F, int ROWS>
static inline void nmv_multi_body(constant const uint8_t* w, constant const Q81Block* x, device float* y,
                                  int n_in, int n_out, int ncols, int nw, threadgroup float* partial,
                                  uint gpos_x, uint tid) {
    using Fmt = nmv_Fmt<F>;
    const int ty = (int) (tid >> 5);
    const int lane = (int) (tid & 31u);
    const int row0 = ROWS * (int) gpos_x;
    const int blocks_per_row = n_in / Fmt::DIV;
    const int x_stride = n_in / 32;                        // Q8_1 blocks per activation column
    const int bpi = Fmt::BPI * nw / NMV_WARPS;             // blocks per iteration scale with the warp count
    float tmp[8][ROWS];                                    // MAX_NCOLS = 8
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < ROWS; ++i) tmp[j][i] = 0.0f;
    for (int kbx = (int) tid / Fmt::T; kbx < blocks_per_row; kbx += bpi) {
        const int kby = kbx * Fmt::KBY;
        const int kqs = Fmt::kqs((int) tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const size_t block = (size_t) (row0 + i) * (size_t) blocks_per_row + (size_t) kbx;
                constant const uint8_t* blk = w + block * (size_t) Fmt::BYTES;
                const typename Fmt::W wv = Fmt::load(blk, kqs);       // once per (row, block)
                for (int j = 0; j < ncols; ++j)                        // then per column
                    tmp[j][i] += Fmt::apply(wv, blk, x + (size_t) j * x_stride + kby, kqs);
            }
        }
    }
    if (ty > 0) {
        for (int j = 0; j < ncols; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[((ty - 1) * 8 * ROWS + j * ROWS + i) * 32 + lane] = tmp[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            for (int l = 0; l < nw - 1; ++l) tmp[j][i] += partial[(l * 8 * ROWS + j * ROWS + i) * 32 + lane];
            tmp[j][i] = nmv_warp_sum(tmp[j][i]);
            if (lane == i && row0 + i < n_out) y[(size_t) j * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// ---------------------------------------------------------------- q8_1 (the pinned quantizer)
// One thread per element; the warp butterfly gives every lane its 32 values' amax and sum (blockDim is a
// multiple of 32 and i is warp-contiguous, so a warp spans exactly one 32-value block; only whole warps
// return).  n is n_in * ncols - the columns are contiguous, so every block stays inside one column.
kernel void native_quantize_q8_1_kernel(constant const float* x [[buffer(0)]],
                                        device uint8_t* yb [[buffer(1)]],
                                        constant const long& n [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
    if ((long) i >= n) return;
    const float xi = x[i];
    const float amax = nmv_warp_max(metal::precise::fabs(xi));
    const float sum = nmv_warp_sum(xi);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? (int8_t) 0 : (int8_t) nmv_roundf(xi / d);
    const long ib = (long) i / 32, iqs = (long) i % 32;
    device uint8_t* blk = yb + (size_t) ib * 36;             // Q81Block: { fp16 d ; fp16 sum ; int8 qs[32] }
    blk[4 + iqs] = as_type<uint8_t>(q);
    if (iqs == 0) {
        const uint dbits = f16_from_f32(d), sbits = f16_from_f32(sum);
        blk[0] = (uint8_t) (dbits & 0xFF);
        blk[1] = (uint8_t) (dbits >> 8);
        blk[2] = (uint8_t) (sbits & 0xFF);
        blk[3] = (uint8_t) (sbits >> 8);
    }
}

// ---------------------------------------------------------------- the MMVQ kernels
// blockDim is (32, WARPS): tid's warp index is tid >> 5 and its lane tid & 31; blockIdx.x is the row group.
#define NMV_SINGLE_PARAMS \
    constant const uint8_t* w [[buffer(0)]], \
    constant const Q81Block* x [[buffer(1)]], \
    device float* y [[buffer(2)]], \
    constant const int& n_in [[buffer(3)]], \
    constant const int& n_out [[buffer(4)]], \
    uint3 gpos [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]
#define NMV_SINGLE_KERNEL(FN, F, ROWS, TAG) \
kernel void FN##_##TAG(NMV_SINGLE_PARAMS) { \
    threadgroup float partial[3 * ROWS * 32];   /* [WARPS - 1][ROWS][WARP], flat */ \
    nmv_single_body<F, ROWS>(w, x, y, n_in, n_out, partial, gpos.x, tid); \
}
#define NMV_SINGLE_PAIR(FN, F) \
    NMV_SINGLE_KERNEL(FN, F, 4, small) \
    NMV_SINGLE_KERNEL(FN, F, 1, large)

NMV_SINGLE_PAIR(native_q5_k_mmvq_kernel, 0)
NMV_SINGLE_PAIR(native_q4_k_mmvq_kernel, 1)
NMV_SINGLE_PAIR(native_q2_0_mmvq_kernel, 2)
NMV_SINGLE_PAIR(native_q3_k_mmvq_kernel, 3)
NMV_SINGLE_PAIR(native_q6_k_mmvq_kernel, 4)
NMV_SINGLE_PAIR(native_iq4_xs_mmvq_kernel, 5)
NMV_SINGLE_PAIR(native_iq4_xs_expanded, 11)
NMV_SINGLE_PAIR(native_small_mmvq_kernel_q4_0, 6)
NMV_SINGLE_PAIR(native_small_mmvq_kernel_q5_0, 7)
NMV_SINGLE_PAIR(native_small_mmvq_kernel_q8_0, 8)
NMV_SINGLE_PAIR(native_small_mmvq_kernel_iq4_nl, 9)

// ---------------------------------------------------------------- direct-codebook single-column kernels
// The same thread-to-block mapping, per-thread accumulation, cross-warp partial order and butterfly as
// nmv_single_body<5 or 9>, and the same integer scale step and float expression per call. What changes is how
// a call's integer sum is formed: the original builds packed codebook words (byte_perm SWAR, ~150 ALU ops per
// word) and feeds the emulated dp4a; these look each 4-bit code up in a threadgroup float codebook and FMA it
// against the activation byte. Every product and partial sum is an integer of magnitude below
// 32 * 127 * 128 < 2^24, so the float accumulation is exact in any order and (int) of it is the original
// int32 sumi. ROWS is only how many rows a threadgroup carries (a row's arithmetic does not depend on it).
// Measured on the M2 Max, isolated (bench/results/2026-10-03-metal-decode-opt2/micro/mmvq-iq4.log; absolute times
// move 20-30% between sessions, the order does not): the 2560 x 248320 head 3.2-4.3 ms -> 1.6-2.3 ms, as fast as
// or faster than the 264-byte expanded view, which this needs neither the memory nor the preparation for.
struct NmvActXs { float4 a[8]; float d8; };
struct NmvActNl { float4 a[4]; float d8; };

template<int F> struct nmv_Direct;
template<> struct nmv_Direct<5> {    // IQ4_XS: iqs = 4 * (tid % 8), one q8_1 block of 32 values per call
    using A = NmvActXs;
    static A act(constant const Q81Block* x, int kby, int iqs) {
        constant const Q81Block* b = x + kby + iqs / 4;
        constant const int* q = reinterpret_cast<constant const int*>(b->qs);
        A r;
#pragma unroll
        for (int j = 0; j < 8; ++j) r.a[j] = float4(as_type<char4>(q[j]));
        r.d8 = (float) b->ds.x;
        return r;
    }
    static float dot(constant const uint8_t* blk, const thread A& c, int iqs, threadgroup const float* tbl) {
        constant const IQ4XSBlock* w = reinterpret_cast<constant const IQ4XSBlock*>(blk);
        constant const uint* qw = reinterpret_cast<constant const uint*>(w->qs) + iqs;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {          // word j: low nibbles meet q8 ints j, high nibbles ints j + 4
            const uint u = qw[j];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s = fma(tbl[(u >> (8 * k)) & 15u], c.a[j][k], s);
                s = fma(tbl[(u >> (8 * k + 4)) & 15u], c.a[j + 4][k], s);
            }
        }
        int sumi = (int) s;
        const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (int) (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        sumi *= ls - 32;
        const float d = (float) w->d * c.d8;
        return d * (float) sumi;
    }
};
template<> struct nmv_Direct<9> {    // IQ4_NL: iqs = 2 * (tid % 2), two 16-bit-aligned words per call
    using A = NmvActNl;
    static A act(constant const Q81Block* x, int kby, int iqs) {
        constant const Q81Block* b = x + kby;
        constant const int* q = reinterpret_cast<constant const int*>(b->qs) + iqs;
        A r;
        r.a[0] = float4(as_type<char4>(q[0])); r.a[1] = float4(as_type<char4>(q[1]));
        r.a[2] = float4(as_type<char4>(q[4])); r.a[3] = float4(as_type<char4>(q[5]));
        r.d8 = (float) b->ds.x;
        return r;
    }
    static float dot(constant const uint8_t* blk, const thread A& c, int iqs, threadgroup const float* tbl) {
        constant const IQ4NLBlock* w = reinterpret_cast<constant const IQ4NLBlock*>(blk);
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {          // word i: low nibbles meet q8 int iqs + i, high ones iqs + i + 4
            const uint u = as_type<uint>(nmv_load_int_b2(reinterpret_cast<constant const uint16_t*>(w->qs), iqs + i));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s = fma(tbl[(u >> (8 * k)) & 15u], c.a[i][k], s);
                s = fma(tbl[(u >> (8 * k + 4)) & 15u], c.a[i + 2][k], s);
            }
        }
        const float d = (float) w->d * c.d8;
        return d * (float) (int) s;
    }
};

template<int F, int ROWS>
static inline void nmv_direct_body(constant const uint8_t* w, constant const Q81Block* x, device float* y, int n_in,
                                   int n_out, threadgroup float* partial, threadgroup float* tbl, uint gpos_x,
                                   uint tid) {
    using Fmt = nmv_Fmt<F>;
    using D = nmv_Direct<F>;
    if (tid < 16) tbl[tid] = (float) kvalues_iq4nl[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ty = (int) (tid >> 5);
    const int lane = (int) (tid & 31u);
    const int row0 = ROWS * (int) gpos_x;
    const int blocks_per_row = n_in / Fmt::DIV;
    float tmp[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = (int) tid / Fmt::T; kbx < blocks_per_row; kbx += Fmt::BPI) {
        const int kby = kbx * Fmt::KBY;
        const int kqs = Fmt::kqs((int) tid);
        const typename D::A c = D::act(x, kby, kqs);     // once per call, shared by the ROWS rows
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const size_t block = (size_t) (row0 + i) * (size_t) blocks_per_row + (size_t) kbx;
                tmp[i] += D::dot(w + block * (size_t) Fmt::BYTES, c, kqs, tbl);
            }
        }
    }
    if (ty > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[((ty - 1) * ROWS + i) * 32 + lane] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < 3; ++l) tmp[i] += partial[(l * ROWS + i) * 32 + lane];
        tmp[i] = nmv_warp_sum(tmp[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}
#define NMV_DIRECT_KERNEL(NAME, F, ROWS) \
kernel void NAME(NMV_SINGLE_PARAMS) { \
    threadgroup float partial[3 * ROWS * 32]; \
    threadgroup float tbl[16]; \
    nmv_direct_body<F, ROWS>(w, x, y, n_in, n_out, partial, tbl, gpos.x, tid); \
}
NMV_DIRECT_KERNEL(native_iq4_xs_direct_r4, 5, 4)
NMV_DIRECT_KERNEL(native_iq4_nl_direct_r4, 9, 4)

// IQ4_XS, one simdgroup per row: lane l runs the four warps' threads l, l + 32, l + 64, l + 96 of the layout
// above in turn (each one's kbx loop and accumulation unchanged), adds their partials in the warp order
// (0 + 1 + 2 + 3, the cross-warp sum above) and runs the same butterfly; lane 0 stores. No threadgroup partials
// or second barrier, and no idle warps when a row has fewer than 16 blocks. 4 simdgroups x 4 rows per group.
// Measured (M2 Max, isolated, every output bitwise; micro/mmvq-iq4.log): 8-24% faster than _r4 for n_out >= 2560,
// slower below (n_out 640: about 2x) - the launcher's crossover.
kernel void native_iq4_xs_direct_sg4(NMV_SINGLE_PARAMS) {
    constexpr int R = 4;
    using D = nmv_Direct<5>;
    using Fmt = nmv_Fmt<5>;
    threadgroup float tbl[16];
    if (tid < 16) tbl[tid] = (float) kvalues_iq4nl[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int lane = (int) (tid & 31u);
    const int row0 = ((int) gpos.x * NMV_WARPS + (int) (tid >> 5)) * R;
    const int blocks_per_row = n_in / Fmt::DIV;
    if (row0 >= n_out) return;
    float acc[R];
#pragma unroll
    for (int v = 0; v < NMV_WARPS; ++v) {
        const int vt = lane + 32 * v;                  // the four-warp layout's thread index
        float t[R];
#pragma unroll
        for (int i = 0; i < R; ++i) t[i] = 0.0f;
        for (int kbx = vt / Fmt::T; kbx < blocks_per_row; kbx += Fmt::BPI) {
            const int kqs = Fmt::kqs(vt);
            const D::A c = D::act(x, kbx * Fmt::KBY, kqs);
#pragma unroll
            for (int i = 0; i < R; ++i)
                if (row0 + i < n_out)
                    t[i] += D::dot(w + ((size_t) (row0 + i) * (size_t) blocks_per_row + (size_t) kbx) * Fmt::BYTES, c,
                                   kqs, tbl);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = v == 0 ? t[i] : acc[i] + t[i];
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float s = nmv_warp_sum(acc[i]);
        if (lane == 0 && row0 + i < n_out) y[row0 + i] = s;
    }
}

// rows and nw are runtime scalars: r4/r1 are the exact layout's two shapes (nw is 4), r2 the upstream one
// (nw 4 or 2, chosen by the launcher, per the file comment).
#define NMV_MULTI_KERNEL(FMT, F, ROWS) \
kernel void native_mmvq_multi_kernel_##FMT##_r##ROWS( \
    constant const uint8_t* w [[buffer(0)]], \
    constant const Q81Block* x [[buffer(1)]], \
    device float* y [[buffer(2)]], \
    constant const int& n_in [[buffer(3)]], \
    constant const int& n_out [[buffer(4)]], \
    constant const int& ncols [[buffer(5)]], \
    constant const int& nw [[buffer(6)]], \
    uint3 gpos [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float partial[3 * 8 * ROWS * 32];   /* [WARPS - 1][MAX_NCOLS][ROWS][WARP], flat */ \
    nmv_multi_body<F, ROWS>(w, x, y, n_in, n_out, ncols, nw, partial, gpos.x, tid); \
}
#define NMV_MULTI_TRIPLE(FMT, F) \
    NMV_MULTI_KERNEL(FMT, F, 1) \
    NMV_MULTI_KERNEL(FMT, F, 2) \
    NMV_MULTI_KERNEL(FMT, F, 4)

NMV_MULTI_TRIPLE(q5_k, 0)
NMV_MULTI_TRIPLE(q4_k, 1)
NMV_MULTI_TRIPLE(q2_0, 2)
NMV_MULTI_TRIPLE(q3_k, 3)
NMV_MULTI_TRIPLE(q6_k, 4)
NMV_MULTI_TRIPLE(iq4_xs, 5)
NMV_MULTI_TRIPLE(q4_0, 6)
NMV_MULTI_TRIPLE(q5_0, 7)
NMV_MULTI_TRIPLE(q8_0, 8)
NMV_MULTI_TRIPLE(iq4_nl, 9)
