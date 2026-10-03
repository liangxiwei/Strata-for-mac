// src/kernels/metal/iq_kernels.metal - the port of src/kernels/cuda/iq_kernels.cu: the i-quants family
// (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0, and the Q4_K / Q5_K / Q5_1 / Q8_0
// that ride the same launchers), the q8_1 quantizer, the dequantizers and the grouped native experts.
//
// The CUDA file's arithmetic is integer bit tricks and fp16->fp32 scale products; there is NO double in its
// device code (checked: every scale path is a half widened to float), so nothing needs the port's fp64
// emulations.  What had to be spelled instead of a CUDA intrinsic (each bit-identical, sources named):
//   * the codebook grids / kmask / ksigns / kvalues tables: ggml-common.h's own METAL decl+impl pair, the
//     same header the CUDA file includes - the tables compile into the metallib as program-scope
//     `static const constant` arrays (what cudaMemcpyToSymbol would have provided on CUDA; not needed here).
//   * __byte_perm -> iqk_byte_perm: "result byte i is byte s.nibble[i] & 7 of the pair {y:x}, x the low
//     word" - the semantics this repo's include/strata/hip_compat/intrinsics.hpp documents for CUDA's
//     default mode (no sign-replication bit; selector nibbles may exceed 7 and wrap).
//   * __vcmpne4 / __vsub4 -> the SWAR formulas from the same hip_compat header (Hacker's Delight 2-18),
//     bit-identical to CUDA's per-byte ops.
//   * __dp4a -> iqk_dp4a: the signed-byte dot of include/strata/kernels/dp4a.hpp's sm_60 fallback, which is
//     that header's documented bit-exact definition of __dp4a.
//   * __shfl_xor_sync -> simd_shuffle_xor; __expf -> metal::precise::exp; roundf -> iqk_roundf (CUDA's
//     roundf is ties-away-from-zero; spelled on |x|+0.5 because MSL has no precise::round).
//   * fp16 block scales are read through the struct members (half -> float is exact) and written through
//     f16_from_f32 (round-to-nearest-even, the f16_bits.hpp twin of __float2half).
//
// The CUDA file's template kernels become concrete instantiations: mmvq_kernel_<T>, and for the
// decode-once family mmvq_multi_kernel_<T>_<NC>, native_gu_multi_kernel_<T>, native_down_multi_kernel_<T>.
// The Split load/apply decomposition is transcribed verbatim, so a column of a multi kernel does its
// integer and float operations in the same order on the same values as the per-column kernel - bitwise
// equal, the contract iq_multi_parity checks against iq_set_old_kernels.
//
// RULE 9 (docs/PORT_METAL/PROGRESS.md): the grouped expert kernels' grp_ptr table holds raw device
// pointers IN DEVICE MEMORY, and such a pointer is not a usable device pointer on this GPU (it loads fine
// and dereferences to zero, writes vanish).  So the native_gu/down kernels take each group's BLOB as a
// bound [[buffer(0)]] argument and the group index as a scalar; the launcher reads the small table back to
// the host and launches once per group.  The counts (n_groups, grp_start, ent_dst, ent_tok) stay
// device-resident and are read inside the kernel exactly where the CUDA kernel reads them.
#include "strata_port.metalh"

// llama.cpp's block structs and codebook grids, unchanged - the same include the CUDA file makes
#define GGML_COMMON_DECL_METAL
#include <metal_simdgroup_matrix>
#define GGML_COMMON_IMPL_METAL
#include "../../../third_party/ggml/ggml-common.h"

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
static inline int iqk_get_int_b2(constant const uint16_t* x, int i32) {
    int x32 = (int) x[2 * i32 + 0] << 0;
    x32 |= (int) x[2 * i32 + 1] << 16;
    return x32;
}
static inline int iqk_get_int_b4(constant const uint8_t* x, int i32) {
    return reinterpret_cast<constant const int*>(x)[i32];
}
static inline uint iqk_unpack_ksigns(uint8_t v) {
    const uint p = popcount((uint) v) & 1u;
    const uint s = (uint) v ^ (p << 7u);
    return s * 0x01010101u;
}

// CUDA's __byte_perm (default mode), per this repo's HIP compat shim: result byte i is byte (s.nibble[i] & 7)
// of the eight bytes {y:x}, x the low word - no sign-replication mode bit.
static inline uint iqk_byte_perm(uint x, uint y, uint s) {
    uint r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint idx = (s >> (4 * i)) & 0x7u;
        const uint byte_ = idx < 4u ? (x >> (8 * idx)) & 0xFFu : (y >> (8 * (idx - 4u))) & 0xFFu;
        r |= byte_ << (8 * i);
    }
    return r;
}

// CUDA's __vcmpne4 (0xff per unequal byte) and __vsub4 (wrapping per-byte subtract): the SWAR forms of
// include/strata/hip_compat/intrinsics.hpp, bit-identical to the PTX ops.
static inline int iqk_vcmpne4(int a, int b) {
    const uint t = as_type<uint>(a) ^ as_type<uint>(b);
    const uint nonzero = (((t & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | t) & 0x80808080u;
    return as_type<int>((nonzero >> 7) * 0xFFu);
}
static inline int iqk_vsub4(int a, int b) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    return as_type<int>(((ua | 0x80808080u) - (ub & 0x7F7F7F7Fu)) ^ ((ua ^ ~ub) & 0x80808080u));
}
// CUDA's packed signed-byte SATURATING subtract (hip_compat's vsubss4): the wrapping difference, and in
// each lane that overflowed the bound on the minuend's side, 0x7f or 0x80.  Only the Q3_K bridge below
// needs it.
static inline int iqk_vsubss4(int a, int b) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    const uint d = as_type<uint>(iqk_vsub4(a, b));
    const uint overflow = (ua ^ ub) & (ua ^ d) & 0x80808080u;
    const uint mask = (overflow >> 7) * 0xFFu;
    const uint bound = 0x7F7F7F7Fu + ((ua & 0x80808080u) >> 7);
    return as_type<int>((d & ~mask) | (bound & mask));
}

// CUDA's signed __dp4a: dp4a.hpp's own (bit-exact) definition - four signed-byte products, wrapping int32.
static inline int iqk_dp4a(int a, int b, int c) {
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

static inline int2 iqk_get_int_from_table_16(int q4, constant const int8_t* table) {
    constant const uint* table32 = reinterpret_cast<constant const uint*>(table);
    uint tmp[2];
    const uint uq4 = as_type<uint>(q4);
    const uint low_high_selection_indices = 0x32103210u | ((uq4 & 0x88888888u) >> 1);
    for (uint i = 0; i < 2; ++i) {
        const uint shift = 16 * i;
        const uint low = iqk_byte_perm(table32[0], table32[1], uq4 >> shift);
        const uint high = iqk_byte_perm(table32[2], table32[3], uq4 >> shift);
        tmp[i] = iqk_byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return int2(as_type<int>(iqk_byte_perm(tmp[0], tmp[1], 0x6420u)),
                as_type<int>(iqk_byte_perm(tmp[0], tmp[1], 0x7531u)));
}

// CUDA roundf: nearest integer, ties AWAY from zero (MSL's rint is ties-to-even and there is no
// metal::precise::round, so the identity roundf(x) = copysign(floor(|x|+0.5), x) is spelled out).
static inline float iqk_roundf(float x) {
    const float r = floor(metal::precise::fabs(x) + 0.5f);   // floor is exact under any math mode
    return x < 0.0f ? -r : r;
}

// the CUDA file's __shfl_xor_sync butterfly: every lane ends up holding the whole sum
static inline float iqk_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, (uint) o);
    return v;
}

// __low2float / __half2float / __half22float2 on the block unions (half -> float is exact)
static inline float iqk_lo2f(half2 h) { return (float) h.x; }

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
static inline float iqk_vd_q2_0_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                    int kbx, int iqs) {
    constant const block_q2_0* bq2_0 = reinterpret_cast<constant const block_q2_0*>(vbq) + kbx;
    const float d2 = (float) bq2_0->d;
    constant const int16_t* qs = reinterpret_cast<constant const int16_t*>(bq2_0->qs) + iqs * 4;
    constant const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1_chunk->qs), j * 2 + 0);
        const int v = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1_chunk->qs), j * 2 + 1);
        const int qe = as_type<int>(iqk_byte_perm(0x020100FFu, 0x020100FFu, as_type<uint>(q) >> 0));
        const int qo = as_type<int>(iqk_byte_perm(0x020100FFu, 0x020100FFu, as_type<uint>(q) >> 2));
        const int qx = as_type<int>(iqk_byte_perm(as_type<uint>(qe), as_type<uint>(qo), 0x5140u));
        const int qy = as_type<int>(iqk_byte_perm(as_type<uint>(qe), as_type<uint>(qo), 0x7362u));
        sumi = iqk_dp4a(u, qx, sumi);
        sumi = iqk_dp4a(v, qy, sumi);
    }
    const float d8 = iqk_lo2f(bq8_1_chunk->ds);
    return d2 * d8 * (float) sumi;
}

static inline float iqk_vd_iq2_xxs_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                       int kbx, int iqs) {
    constant const block_iq2_xxs* bq2 = reinterpret_cast<constant const block_iq2_xxs*>(vbq) + kbx;
    const int q2 = iqk_get_int_b2(bq2->qs, iqs);
    const thread uint8_t* aux8 = reinterpret_cast<const thread uint8_t*>(&q2);
    const uint aux32 = as_type<uint>(iqk_get_int_b2(bq2->qs, iqs + 1));
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = reinterpret_cast<constant const uint2*>(iq2xxs_grid)[aux8[k0 / 2]];
        const uint signs = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * (k0 / 2))));
        const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
        const int grid0 = iqk_vsub4(as_type<int>(grid_pos.x) ^ signs0, signs0);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), k0 + 0);
        sumi = iqk_dp4a(grid0, u0, sumi);
        const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
        const int grid1 = iqk_vsub4(as_type<int>(grid_pos.y) ^ signs1, signs1);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), k0 + 1);
        sumi = iqk_dp4a(grid1, u1, sumi);
    }
    const int ls = (int) (aux32 >> 27) | 1;
    sumi = sumi * ls / 8;
    const float d = (float) bq2->d * iqk_lo2f(bq8_1[iqs / 2].ds);
    return d * (float) sumi;
}

static inline float iqk_vd_iq2_xs_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                      int kbx, int iqs) {
    constant const block_iq2_xs* bq2 = reinterpret_cast<constant const block_iq2_xs*>(vbq) + kbx;
    const int2 q2_packed = int2(iqk_get_int_b2(bq2->qs, iqs + 0), iqk_get_int_b2(bq2->qs, iqs + 1));
    const thread uint16_t* q2 = reinterpret_cast<const thread uint16_t*>(&q2_packed);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = reinterpret_cast<constant const uint2*>(iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint signs = iqk_unpack_ksigns((uint8_t) (q2[l0 / 2] >> 9));
        const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
        const int grid_l = iqk_vsub4(as_type<int>(grid_pos.x) ^ signs0, signs0);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 0);
        const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
        const int grid_h = iqk_vsub4(as_type<int>(grid_pos.y) ^ signs1, signs1);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 1);
        if (l0 < 4) {
            sumi0 = iqk_dp4a(grid_l, u0, sumi0);
            sumi0 = iqk_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = iqk_dp4a(grid_l, u0, sumi1);
            sumi1 = iqk_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * iqk_lo2f(bq8_1[iqs / 2].ds);
    return d * (float) sumi;
}

static inline float iqk_vd_iq2_s_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                     int kbx, int iqs) {
    constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
    const int qs_packed = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
    const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 =
        iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
    const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        constant const int* grid_pos = reinterpret_cast<constant const int*>(
            iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                       0x00000000);
        const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                       0x00000000);
        const int grid_l = iqk_vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = iqk_vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 0);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 1);
        if (l0 < 4) {
            sumi0 = iqk_dp4a(grid_l, u0, sumi0);
            sumi0 = iqk_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = iqk_dp4a(grid_l, u0, sumi1);
            sumi1 = iqk_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * iqk_lo2f(bq8_1[iqs / 2].ds);
    return d * (float) sumi;
}

static inline float iqk_vd_iq3_xxs_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                       int kbx, int iqs) {
    constant const block_iq3_xxs* bq3 = reinterpret_cast<constant const block_iq3_xxs*>(vbq) + kbx;
    const int2 q3_packed = int2(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs),
                                iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1));
    const thread uint8_t* q3 = reinterpret_cast<const thread uint8_t*>(&q3_packed);
    const uint aux32 =
        as_type<uint>(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), QK_K / 16 + iqs / 2));
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = int2((int) iq3xxs_grid[q3[l0 + 0]], (int) iq3xxs_grid[q3[l0 + 1]]);
        const uint signs = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * (l0 / 2))));
        const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
        const int grid_l = iqk_vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 0);
        const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
        const int grid_h = iqk_vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 1);
        sumi = iqk_dp4a(grid_l, u0, sumi);
        sumi = iqk_dp4a(grid_h, u1, sumi);
    }
    const int ls = (int) (aux32 >> 28);
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = (float) bq3->d * iqk_lo2f(bq8_1[iqs / 2].ds);
    return d * (float) sumi;
}

static inline float iqk_vd_iq3_s_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                     int kbx, int iqs) {
    constant const block_iq3_s* bq3 = reinterpret_cast<constant const block_iq3_s*>(vbq) + kbx;
    const int2 qs_packed = int2(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 0),
                                iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1));
    const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->signs), iqs / 2);
    const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = int2((int) iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                   (int) iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                       0x00000000);
        const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                       0x00000000);
        const int grid_l = iqk_vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = iqk_vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 0);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), l0 + 1);
        sumi = iqk_dp4a(grid_l, u0, sumi);
        sumi = iqk_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (float) bq3->d * iqk_lo2f(bq8_1[iqs / 2].ds);
    return d * (float) sumi;
}

static inline float iqk_vd_iq1_m_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                     int kbx, int iqs) {
    constant const block_iq1_m* bq1 = reinterpret_cast<constant const block_iq1_m*>(vbq) + kbx;
    const int qs_packed = iqk_get_int_b4(bq1->qs, iqs);
    const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = (int) iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs].qs), l0 + 0);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs].qs), l0 + 1);
        sumi[l0 / 4] = iqk_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = iqk_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (float) (qhl & 0x08) * (2.0f * IQ1M_DELTA / 8.0f);
        int sumy = 0;
        sumy = iqk_dp4a(u0, 0x01010101, sumy);
        sumy = iqk_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * (float) sumy;
    }
    constant const uint16_t* sc = reinterpret_cast<constant const uint16_t*>(bq1->scales);
    // iq1m_scale_t: the fp16 scale packed across the four scale words' top nibbles
    const uint scale_u16 = (uint) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = f32_from_f16(scale_u16) * iqk_lo2f(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((float) sumi[0] + sumf[0]) * (float) sc0 + d * ((float) sumi[1] + sumf[1]) * (float) sc1;
}

static inline float iqk_vd_iq4_nl_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                      int kbx, int iqs) {
    constant const block_iq4_nl* bq4 = reinterpret_cast<constant const block_iq4_nl*>(vbq) + kbx;
    constant const int* q8 = reinterpret_cast<constant const int*>(bq8_1->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq4->qs), iqs + l);
        const int2 v = iqk_get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = iqk_dp4a(v.x, q8[l + 0], sumi);
        sumi = iqk_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = (float) bq4->d * iqk_lo2f(bq8_1->ds);
    return d * (float) sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
static inline float iqk_vd_iq4_xs_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                      int kbx, int iqs) {
    constant const block_iq4_xs* bq4 = reinterpret_cast<constant const block_iq4_xs*>(vbq) + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = iqk_get_int_b4(bq4->qs, iqs + j);
        const int2 v = iqk_get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 4].qs), j + 0);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 4].qs), j + 4);
        sumi = iqk_dp4a(v.x, u0, sumi);
        sumi = iqk_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (int) (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = (float) bq4->d * iqk_lo2f(bq8_1[iqs / 4].ds);
    return d * (float) sumi;
}

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed; the
// Q5_1 min term is the one departure (below).  Via eddoursul/Strata 8029fa9 (iq_dot.cuh) and #255 (Q8_0,
// gopinath87607), which agree with llama.cpp and with each other.
constant const int IQK_VDR_Q4_K = 2;
constant const int IQK_VDR_Q5_K = 2;
constant const int IQK_VDR_Q5_1 = 2;
constant const int IQK_VDR_Q8_0 = 2;
constant const int IQK_QI8_1 = 8;    // QK8_1 / (4 * QR8_1) = 32 / 4
constant const int IQK_QI5_1 = 4;    // QK5_1 / (4 * QR5_1) = 32 / 8 (the QI macros are the CUDA-only decl)

static inline float iqk_vd_q4_K_q8_1_impl(const thread int* v, const thread int* u,
                                          const thread uint8_t* sc, const thread uint8_t* m,
                                          half2 dm4, const thread float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < IQK_VDR_Q4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = iqk_dp4a(v1i, u[2 * i + 1], iqk_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = iqk_dp4a(0x01010101, u[2 * i + 1], iqk_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    const float dm4x = (float) dm4.x, dm4y = (float) dm4.y;
    return dm4x * sumf_d - dm4y * sumf_m;
}
static inline float iqk_vd_q5_K_q8_1_impl(const thread int* vl, const thread int* vh, const thread int* u,
                                          const thread uint8_t* sc, const thread uint8_t* m,
                                          half2 dm5, const thread float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < IQK_VDR_Q5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = iqk_dp4a(v0i, u[2 * i + 0], iqk_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = iqk_dp4a(0x01010101, u[2 * i + 0], iqk_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (float) (dot1 * sc[i]);
        sumf_m += d8[i] * (float) (dot2 * m[i]);
    }
    const float dm5x = (float) dm5.x, dm5y = (float) dm5.y;
    return dm5x * sumf_d - dm5y * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
static inline void iqk_k_scale_min(constant const uint8_t* scales8, int bq8_offset, thread uint16_t* aux) {
    constant const uint16_t* scales = reinterpret_cast<constant const uint16_t*>(scales8);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint s0 = scales[jm + 0];
    const uint s2 = scales[jm + 2];
    const uint s4 = scales[jm + 4];
    const uint hi = (uint) -(int) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
static inline float iqk_vd_q4_K_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                    int kbx, int iqs) {
    constant const block_q4_K* bq4_K = reinterpret_cast<constant const block_q4_K*>(vbq) + kbx;
    int v[2];
    int u[2 * IQK_VDR_Q4_K];
    float d8[IQK_VDR_Q4_K];
    const int bq8_offset = IQK_VDR_Q4_K * ((iqs / 2) / (IQK_QI8_1 / 2));
    constant const uint8_t* q4 = bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    v[0] = iqk_get_int_b4(q4, 0);
    v[1] = iqk_get_int_b4(q4, 4);
    uint16_t aux[2];
    iqk_k_scale_min(bq4_K->scales, bq8_offset, aux);
    const thread uint8_t* sc = reinterpret_cast<const thread uint8_t*>(aux);
    const thread uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < IQK_VDR_Q4_K; ++i) {
        constant const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = iqk_lo2f(bq8i->ds);
        const int q8off = ((iqs / 2) % 4);
        u[2 * i + 0] = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8i->qs), q8off + 0);
        u[2 * i + 1] = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8i->qs), q8off + 4);
    }
    return iqk_vd_q4_K_q8_1_impl(v, u, sc, m, bq4_K->dm, d8);
}
static inline float iqk_vd_q5_K_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                    int kbx, int iqs) {
    constant const block_q5_K* bq5_K = reinterpret_cast<constant const block_q5_K*>(vbq) + kbx;
    int vl[2];
    int vh[2];
    int u[2 * IQK_VDR_Q5_K];
    float d8[IQK_VDR_Q5_K];
    const int bq8_offset = IQK_VDR_Q5_K * ((iqs / 2) / (IQK_QI8_1 / 2));
    constant const uint8_t* ql = bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    vl[0] = iqk_get_int_b4(ql, 0);
    vl[1] = iqk_get_int_b4(ql, 4);
    constant const int* qh = reinterpret_cast<constant const int*>(bq5_K->qh + 4 * ((iqs / 2) % 4));
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    iqk_k_scale_min(bq5_K->scales, bq8_offset, aux);
    const thread uint8_t* sc = reinterpret_cast<const thread uint8_t*>(aux);
    const thread uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < IQK_VDR_Q5_K; ++i) {
        constant const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = iqk_lo2f(bq8i->ds);
        const int q8off = ((iqs / 2) % 4);
        u[2 * i + 0] = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8i->qs), q8off + 0);
        u[2 * i + 1] = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8i->qs), q8off + 4);
    }
    return iqk_vd_q5_K_q8_1_impl(vl, vh, u, sc, m, bq5_K->dm, d8);
}
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which our quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations.  That is ggml-cpu's convention (its q8_1 `s` is d * sum(q)) and the one the K-quant
// mins above use; the scaled and the min term then see the same activation (eddoursul/Strata measured 1.1-1.2%
// against 1.9% relative error per expert).  Result: sumi * (d5 * d8) + sumu * (m5 * d8).
static inline float iqk_vd_q5_1_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                    int kbx, int iqs) {
    constant const block_q5_1* bq5_1 = reinterpret_cast<constant const block_q5_1*>(vbq) + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < IQK_VDR_Q5_1; ++i) {
        const int vl = iqk_get_int_b4(bq5_1->qs, iqs + i);
        const int vh = iqk_get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1->qs), iqs + i);
        const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1->qs), iqs + i + IQK_QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = iqk_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = iqk_dp4a(vi1, u1, sumi);
        sumu = iqk_dp4a(0x01010101, u1, iqk_dp4a(0x01010101, u0, sumu));
    }
    const float dm5x = (float) bq5_1->dm.x, dm5y = (float) bq5_1->dm.y;
    const float d8 = iqk_lo2f(bq8_1->ds);
    return (float) sumi * (dm5x * d8) + (float) sumu * (dm5y * d8);
}
static inline float iqk_vd_q8_0_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                    int kbx, int iqs) {
    constant const block_q8_0* bq8_0 = reinterpret_cast<constant const block_q8_0*>(vbq) + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < IQK_VDR_Q8_0; ++i)
        sumi = iqk_dp4a(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq8_0->qs), iqs + i),
                        iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1->qs), iqs + i), sumi);
    const float d8_0 = (float) bq8_0->d, d8_1 = iqk_lo2f(bq8_1->ds);
    return d8_0 * d8_1 * (float) sumi;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
// (enum members, not static constexpr data: MSL requires program-scope variables to live in the constant
// address space, and an enum is the spelling that carries a compile-time int without one)
template<int TY> struct iqk_Fmt;
template<> struct iqk_Fmt<16> { enum : int { qk = 256, ipb = 8, step = 2 }; };   // IQ2_XXS
template<> struct iqk_Fmt<17> { enum : int { qk = 256, ipb = 8, step = 2 }; };   // IQ2_XS
template<> struct iqk_Fmt<18> { enum : int { qk = 256, ipb = 8, step = 2 }; };   // IQ3_XXS
template<> struct iqk_Fmt<20> { enum : int { qk = 32, ipb = 2, step = 2 }; };    // IQ4_NL
template<> struct iqk_Fmt<21> { enum : int { qk = 256, ipb = 8, step = 2 }; };   // IQ3_S
template<> struct iqk_Fmt<23> { enum : int { qk = 256, ipb = 8, step = 4 }; };   // IQ4_XS
template<> struct iqk_Fmt<22> { enum : int { qk = 256, ipb = 8, step = 2 }; };   // IQ2_S
template<> struct iqk_Fmt<29> { enum : int { qk = 256, ipb = 8, step = 1 }; };   // IQ1_M
template<> struct iqk_Fmt<42> { enum : int { qk = 64, ipb = 2, step = 1 }; };    // Q2_0
template<> struct iqk_Fmt<12> { enum : int { qk = 256, ipb = 16, step = 2 }; };  // Q4_K (QI4_K / VDR 2)
template<> struct iqk_Fmt<13> { enum : int { qk = 256, ipb = 16, step = 2 }; };  // Q5_K (QI5_K / VDR 2)
template<> struct iqk_Fmt<7>  { enum : int { qk = 32, ipb = 2, step = 2 }; };    // Q5_1 (QI5_1 / VDR 2)
template<> struct iqk_Fmt<8>  { enum : int { qk = 32, ipb = 4, step = 2 }; };    // Q8_0 (QI8_0 / VDR 2)
// Q3_K exists in iq_kernels.cu only as a DEQUANT format (the token embedding); its MMVQ dot lives in
// native_mmvq.cu, which is not ported.  The bridge below gives iq_parity's native_mmvq(11) a correct dot -
// transcribed from the same vendored vecdotq.cuh the CUDA tree's other dots come from (see the .mm notes).
template<> struct iqk_Fmt<11> { enum : int { qk = 256, ipb = 16, step = 1 }; };  // Q3_K (QI3_K / 2 calls per block)

// Q3_K x q8_1 (vecdotq.cuh's vec_dot_q3_K_q8_1 + its _impl_mmvq): the bridge for iq_parity's
// native_mmvq(11) - native_mmvq.cu is not ported, and its Q3_K MMVQ is what the test exercises.  iq_mmvq
// itself still refuses type 11 exactly as iq_kernels.cu does.
static inline float iqk_vd_q3_K_q8_1_impl(int vl, int vh, const thread int* u,
                                          constant const uint8_t* scales, int scale_offset, float d3,
                                          const thread float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {                       // QR3_K = 4
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = iqk_vsubss4(vil, vih);
        sumf += d8[i] * (float) (iqk_dp4a(vi, u[i], 0) * sc);   // SIMD dot product
    }
    return d3 * sumf;
}
static inline float iqk_vd_q3_K_q8_1(constant const uint8_t* vbq, constant const block_q8_1* bq8_1,
                                    int kbx, int iqs) {
    constant const block_q3_K* bq3_K = reinterpret_cast<constant const block_q3_K*>(vbq) + kbx;
    const int bq8_offset = 4 * (iqs / (16 / 2));        // QR3_K * (iqs / (QI3_K/2)), QI3_K = 16
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / (8 / 2);   // iqs - iqs%QI8_1 + (iqs%QI8_1)/(QI8_1/2)
    const float d = (float) bq3_K->d;
    const int vl = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3_K->qs), iqs);
    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = (~iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3_K->hmask), iqs % (16 / 2))) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[bq8_offset + i].qs), iqs % 8);
        d8[i] = iqk_lo2f(bq8_1[bq8_offset + i].ds);
    }
    return iqk_vd_q3_K_q8_1_impl(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

// Fmt<TY>::dot: the chain folds to the one branch at compile time (TY is a template constant)
template<int TY>
static inline float iqk_fmt_dot(constant const uint8_t* vbq, constant const block_q8_1* bq8_1, int kbx, int iqs) {
    if (TY == 16) return iqk_vd_iq2_xxs_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 17) return iqk_vd_iq2_xs_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 18) return iqk_vd_iq3_xxs_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 20) return iqk_vd_iq4_nl_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 21) return iqk_vd_iq3_s_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 22) return iqk_vd_iq2_s_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 23) return iqk_vd_iq4_xs_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 29) return iqk_vd_iq1_m_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 11) return iqk_vd_q3_K_q8_1(vbq, bq8_1, kbx, iqs);   // the native_mmvq bridge
    if (TY == 42) return iqk_vd_q2_0_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 12) return iqk_vd_q4_K_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 13) return iqk_vd_q5_K_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 7) return iqk_vd_q5_1_q8_1(vbq, bq8_1, kbx, iqs);
    if (TY == 8) return iqk_vd_q8_0_q8_1(vbq, bq8_1, kbx, iqs);
    return 0.0f;   // unreachable: every launcher's switch lists its formats
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
static inline float iqk_row_dot(constant const uint8_t* row, constant const block_q8_1* x, int nb, int lane) {
    using F = iqk_Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += iqk_fmt_dot<TY>(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return iqk_warp_sum(s);
}

// ---------------------------------------------------------------- decode once, apply to every column
// Each dot is split, as native_mmvq.cu's multi-column traits are, into `load` (everything that depends only on
// the weight) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale step
// and the same float expression).  apply(load(...)) does the dot's integer and float operations in the same
// order on the same values, so a column of the multi kernels is BITWISE equal to the same column of the
// per-column kernels (iq_multi_parity checks it).
template<int TY> struct iqk_Split;
// (MSL has no derived classes: the CUDA file's shared SplitLs2 apply - "two half sums, two 4-bit scales" -
// is transcribed into both IQ2_XS and IQ2_S verbatim)
template<> struct iqk_Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq2_xxs* bq2 = reinterpret_cast<constant const block_iq2_xxs*>(vbq) + kbx;
        const int q2 = iqk_get_int_b2(bq2->qs, iqs);
        const thread uint8_t* aux8 = reinterpret_cast<const thread uint8_t*>(&q2);
        const uint aux32 = as_type<uint>(iqk_get_int_b2(bq2->qs, iqs + 1));
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = reinterpret_cast<constant const uint2*>(iq2xxs_grid)[aux8[k0 / 2]];
            const uint signs = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * (k0 / 2))));
            const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
            r.g[k0 + 0] = iqk_vsub4(as_type<int>(grid_pos.x) ^ signs0, signs0);
            const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
            r.g[k0 + 1] = iqk_vsub4(as_type<int>(grid_pos.y) ^ signs1, signs1);
        }
        r.ls = (int) (aux32 >> 27) | 1;
        r.dw = (float) bq2->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * iqk_lo2f(bq8_1[iqs / 2].ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<17> {   // IQ2_XS (and IQ2_S below share this apply: two half sums, two 4-bit scales)
    struct W { int g[8]; int ls0, ls1; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq2_xs* bq2 = reinterpret_cast<constant const block_iq2_xs*>(vbq) + kbx;
        const int2 q2_packed = int2(iqk_get_int_b2(bq2->qs, iqs + 0), iqk_get_int_b2(bq2->qs, iqs + 1));
        const thread uint16_t* q2 = reinterpret_cast<const thread uint16_t*>(&q2_packed);
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = reinterpret_cast<constant const uint2*>(iq2xs_grid)[q2[l0 / 2] & 0x1FF];
            const uint signs = iqk_unpack_ksigns((uint8_t) (q2[l0 / 2] >> 9));
            const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
            r.g[l0 + 0] = iqk_vsub4(as_type<int>(grid_pos.x) ^ signs0, signs0);
            const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
            r.g[l0 + 1] = iqk_vsub4(as_type<int>(grid_pos.y) ^ signs1, signs1);
        }
        r.dw = (float) bq2->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * iqk_lo2f(bq8_1[iqs / 2].ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<22> {   // IQ2_S
    struct W { int g[8]; int ls0, ls1; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq2_s* bq2 = reinterpret_cast<constant const block_iq2_s*>(vbq) + kbx;
        const int qs_packed = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), iqs / 2);
        const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 =
            iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq2->qs), QK_K / 32 + iqs / 2);
        const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            constant const int* grid_pos = reinterpret_cast<constant const int*>(
                iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                           0x00000000);
            const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                           0x00000000);
            r.g[l0 + 0] = iqk_vsub4(grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = iqk_vsub4(grid_pos[1] ^ signs1, signs1);
        }
        r.dw = (float) bq2->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * iqk_lo2f(bq8_1[iqs / 2].ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq3_xxs* bq3 = reinterpret_cast<constant const block_iq3_xxs*>(vbq) + kbx;
        const int2 q3_packed = int2(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs),
                                    iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1));
        const thread uint8_t* q3 = reinterpret_cast<const thread uint8_t*>(&q3_packed);
        const uint aux32 =
            as_type<uint>(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), QK_K / 16 + iqs / 2));
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = int2((int) iq3xxs_grid[q3[l0 + 0]], (int) iq3xxs_grid[q3[l0 + 1]]);
            const uint signs = iqk_unpack_ksigns((uint8_t) (aux32 >> (7 * (l0 / 2))));
            const int signs0 = iqk_vcmpne4(as_type<int>(signs & 0x08040201u), 0);
            r.g[l0 + 0] = iqk_vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = iqk_vcmpne4(as_type<int>(signs & 0x80402010u), 0);
            r.g[l0 + 1] = iqk_vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = (int) (aux32 >> 28);
        r.dw = (float) bq3->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * iqk_lo2f(bq8_1[iqs / 2].ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq3_s* bq3 = reinterpret_cast<constant const block_iq3_s*>(vbq) + kbx;
        const int2 qs_packed = int2(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 0),
                                    iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1));
        const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->signs), iqs / 2);
        const thread uint8_t* signs_packed_8 = reinterpret_cast<const thread uint8_t*>(&signs_packed_32);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = int2((int) iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                       (int) iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                           0x00000000);
            const int signs1 = iqk_vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                           0x00000000);
            r.g[l0 + 0] = iqk_vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = iqk_vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = (float) bq3->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = iqk_dp4a(r.g[j], iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 2].qs), j), sumi);
        sumi *= r.ls;
        const float d = r.dw * iqk_lo2f(bq8_1[iqs / 2].ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq1_m* bq1 = reinterpret_cast<constant const block_iq1_m*>(vbq) + kbx;
        const int qs_packed = iqk_get_int_b4(bq1->qs, iqs);
        const thread uint8_t* qs = reinterpret_cast<const thread uint8_t*>(&qs_packed);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = (int) iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (float) (qhl & 0x08) * (2.0f * IQ1M_DELTA / 8.0f);
        }
        constant const uint16_t* sc = reinterpret_cast<constant const uint16_t*>(bq1->scales);
        const uint scale_u16 = (uint) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        r.dw = f32_from_f16(scale_u16);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs].qs), l0 + 0);
            const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs].qs), l0 + 1);
            sumi[l0 / 4] = iqk_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = iqk_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = iqk_dp4a(u0, 0x01010101, sumy);
            sumy = iqk_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * (float) sumy;
        }
        const float d = r.dw * iqk_lo2f(bq8_1[iqs].ds);
        return d * ((float) sumi[0] + sumf[0]) * (float) r.sc0 + d * ((float) sumi[1] + sumf[1]) * (float) r.sc1;
    }
};
template<> struct iqk_Split<20> {   // IQ4_NL
    struct W { int2 v[2]; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq4_nl* bq4 = reinterpret_cast<constant const block_iq4_nl*>(vbq) + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l)
            r.v[l] = iqk_get_int_from_table_16(iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq4->qs), iqs + l),
                                               kvalues_iq4nl);
        r.dw = (float) bq4->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        constant const int* q8 = reinterpret_cast<constant const int*>(bq8_1->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = iqk_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = iqk_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        const float d = r.dw * iqk_lo2f(bq8_1->ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<23> {   // IQ4_XS
    struct W { int2 v[4]; int ls; float dw; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_iq4_xs* bq4 = reinterpret_cast<constant const block_iq4_xs*>(vbq) + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iqk_get_int_from_table_16(iqk_get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (int) (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) bq4->d;
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 4].qs), j + 0);
            const int u1 = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1[iqs / 4].qs), j + 4);
            sumi = iqk_dp4a(r.v[j].x, u0, sumi);
            sumi = iqk_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * iqk_lo2f(bq8_1[iqs / 4].ds);
        return d * (float) sumi;
    }
};
template<> struct iqk_Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    static W load(constant const uint8_t* vbq, int kbx, int iqs) {
        constant const block_q2_0* bq2_0 = reinterpret_cast<constant const block_q2_0*>(vbq) + kbx;
        W r;
        r.d2 = (float) bq2_0->d;
        constant const int16_t* qs = reinterpret_cast<constant const int16_t*>(bq2_0->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = as_type<int>(iqk_byte_perm(0x020100FFu, 0x020100FFu, as_type<uint>(q) >> 0));
            const int qo = as_type<int>(iqk_byte_perm(0x020100FFu, 0x020100FFu, as_type<uint>(q) >> 2));
            r.qx[j] = as_type<int>(iqk_byte_perm(as_type<uint>(qe), as_type<uint>(qo), 0x5140u));
            r.qy[j] = as_type<int>(iqk_byte_perm(as_type<uint>(qe), as_type<uint>(qo), 0x7362u));
        }
        return r;
    }
    static float apply(const thread W& r, constant const block_q8_1* bq8_1, int iqs) {
        constant const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1_chunk->qs), j * 2 + 0);
            const int v = iqk_get_int_b4(reinterpret_cast<constant const uint8_t*>(bq8_1_chunk->qs), j * 2 + 1);
            sumi = iqk_dp4a(u, r.qx[j], sumi);
            sumi = iqk_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = iqk_lo2f(bq8_1_chunk->ds);
        return r.d2 * d8 * (float) sumi;
    }
};

// One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks), the
// whole warp.  Per activation this is row_dot: the same calls k, lane-strided the same way, summed in the same
// order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.
template<int TY, int NC>
static inline void iqk_row_dot_multi(constant const uint8_t* row, constant const block_q8_1* x,
                                     const thread int* off, int n, int nb, int lane, thread float* s) {
    using F = iqk_Fmt<TY>;
    using S = iqk_Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < n) s[c] = iqk_warp_sum(s[c]);
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// One thread per element; the warp butterfly gives every lane its 32 values' amax and sum (blockDim is a
// multiple of 32 and i is warp-contiguous, so a warp spans exactly one 32-value block).
kernel void quantize_q8_1_kernel(constant const float* x [[buffer(0)]],
                                 device uint8_t* yb [[buffer(1)]],
                                 constant const long& n [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if ((long) i >= n) return;
    const float xi = x[i];
    float amax = metal::precise::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = metal::precise::fmax(amax, simd_shuffle_xor(amax, (uint) o));
        sum += simd_shuffle_xor(sum, (uint) o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? (int8_t) 0 : (int8_t) iqk_roundf(xi / d);
    const long ib = (long) i / 32, iqs = (long) i % 32;
    device uint8_t* blk = yb + (size_t) ib * sizeof(block_q8_1);   // { fp16 d ; fp16 sum ; int8 qs[32] }
    blk[4 + iqs] = as_type<uint8_t>(q);
    if (iqs == 0) {
        const uint dbits = f16_from_f32(d), sbits = f16_from_f32(sum);
        blk[0] = (uint8_t) (dbits & 0xFF);
        blk[1] = (uint8_t) (dbits >> 8);
        blk[2] = (uint8_t) (sbits & 0xFF);
        blk[3] = (uint8_t) (sbits >> 8);
    }
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq2_xxs(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq2_xxs* x = reinterpret_cast<constant const block_iq2_xxs*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint16_t* q2 = x[ibs].qs + 4 * ib;
    constant const uint8_t* aux8 = reinterpret_cast<constant const uint8_t*>(q2);
    constant const uint8_t* grid = reinterpret_cast<constant const uint8_t*>(iq2xxs_grid + aux8[il]);
    const uint aux32 = (uint) q2[2] | ((uint) q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (float) (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * (uint) il)) & 127u];
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq2_xs(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq2_xs* x = reinterpret_cast<constant const block_iq2_xs*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint16_t* q2 = x[ibs].qs + 4 * ib;
    constant const uint8_t* grid = reinterpret_cast<constant const uint8_t*>(iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + (float) ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq2_s(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq2_s* x = reinterpret_cast<constant const block_iq2_s*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint8_t* grid = reinterpret_cast<constant const uint8_t*>(
        iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + (float) ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f));
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq3_xxs(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq3_xxs* x = reinterpret_cast<constant const block_iq3_xxs*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint8_t* q3 = x[ibs].qs + 8 * ib;
    constant const uint16_t* gas = reinterpret_cast<constant const uint16_t*>(x[ibs].qs + QK_K / 4) + 2 * ib;
    constant const uint8_t* grid1 = reinterpret_cast<constant const uint8_t*>(iq3xxs_grid + q3[2 * il + 0]);
    constant const uint8_t* grid2 = reinterpret_cast<constant const uint8_t*>(iq3xxs_grid + q3[2 * il + 1]);
    const uint aux32 = (uint) gas[0] | ((uint) gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * (uint) il)) & 127u];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f));
        y[j + 4] = (dst_t) (d * (float) grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f));
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq3_s(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq3_s* x = reinterpret_cast<constant const block_iq3_s*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint8_t* qs = x[ibs].qs + 8 * ib;
    constant const uint8_t* grid1 =
        reinterpret_cast<constant const uint8_t*>(iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    constant const uint8_t* grid2 =
        reinterpret_cast<constant const uint8_t*>(iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (float) (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f));
        y[j + 4] = (dst_t) (d * (float) grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f));
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq1_m(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq1_m* x = reinterpret_cast<constant const block_iq1_m*>(vx);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 8 * il;
    constant const uint16_t* sc = reinterpret_cast<constant const uint16_t*>(x[ibs].scales);
    const uint scale_u16 = (uint) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const long ib16 = 2 * ib + il / 2;
    const float d = f32_from_f16(scale_u16) * (float) (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta =
        (x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2))) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
    uint grid32[2];
    const thread int8_t* q = reinterpret_cast<const thread int8_t*>(grid32);
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | ((((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7)) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0fu;
    grid32[0] &= 0x0f0f0f0fu;
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) (d * ((float) q[j] + delta));
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq4_nl(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq4_nl* x = reinterpret_cast<constant const block_iq4_nl*>(vx) + ibs * (QK_K / QK4_NL);
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 4 * il;
    constant const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t, typename Ptr>
static inline void iqk_dq_q3_k(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_q3_K* x = reinterpret_cast<constant const block_q3_K*>(vx) + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                        is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                        is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                  (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (float) (us - 32);
        auto y = yy + 128 * n + 32 * j;
        constant const uint8_t* q = x->qs + 32 * n;
        constant const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l)
            y[l] = (dst_t) (dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_iq4_xs(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_iq4_xs* x = reinterpret_cast<constant const block_iq4_xs*>(vx) + ibs;
    const long il = tid / 8, ib = tid % 8;
    auto y = yy + 32 * ib + 4 * il;
    constant const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d =
        (float) x->d * (float) ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = (dst_t) (d * (float) kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_q2_0(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    constant const block_q2_0* x = reinterpret_cast<constant const block_q2_0*>(vx) + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = (dst_t) (d * (float) (code - 1));
    }
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the
// 32-value blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid % 8).
static inline void iqk_get_scale_min_k4(int j, constant const uint8_t* q, thread uint8_t& d, thread uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_q4_k(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_q4_K* x = reinterpret_cast<constant const block_q4_K*>(vx);
    const long il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    auto y = yy + 64 * il + n * ir;
    const float dall = (float) x[ibs].dm.x;
    const float dmin = (float) x[ibs].dm.y;
    constant const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    iqk_get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * (float) sc, m1 = dmin * (float) m;
    iqk_get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * (float) sc, m2 = dmin * (float) m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = (dst_t) (d1 * (float) (q[l] & 0xF) - m1);
        y[l + 32] = (dst_t) (d2 * (float) (q[l] >> 4) - m2);
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_q5_k(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_q5_K* x = reinterpret_cast<constant const block_q5_K*>(vx);
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        auto y = yy + 64 * il + 2 * ir;
        const float dall = (float) x[ibs].dm.x;
        const float dmin = (float) x[ibs].dm.y;
        constant const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        constant const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        iqk_get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * (float) sc, m1 = dmin * (float) m;
        iqk_get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * (float) sc, m2 = dmin * (float) m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = (dst_t) (d1 * (float) ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = (dst_t) (d1 * (float) ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = (dst_t) (d2 * (float) ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = (dst_t) (d2 * (float) ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_q5_1(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_q5_1* x = reinterpret_cast<constant const block_q5_1*>(vx) + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float dmx = (float) x[ib].dm.x, dmy = (float) x[ib].dm.y;
    const uint qh = (uint) x[ib].qh[0] | ((uint) x[ib].qh[1] << 8) | ((uint) x[ib].qh[2] << 16) |
                    ((uint) x[ib].qh[3] << 24);   // CUDA memcpy'd the four bytes into a uint - same bytes
    auto y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = (int) ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = (int) ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = (dst_t) ((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dmx + dmy);
        y[iqs + 16] = (dst_t) ((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dmx + dmy);
    }
}
template<typename dst_t, typename Ptr>
static inline void iqk_dq_q8_0(constant const uint8_t* vx, long ibs, Ptr yy, int tid) {
    constant const block_q8_0* x = reinterpret_cast<constant const block_q8_0*>(vx) + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    auto y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = (dst_t) ((float) x[ib].qs[8 * il + j] * d);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template<typename dst_t, typename Ptr>
static inline void iqk_dq_bf16(constant const uint8_t* vxb, long ibs, Ptr yy, int tid) {
    constant const uint16_t* x = reinterpret_cast<constant const uint16_t*>(vxb) + ibs * 256 + tid * 8;
    for (int j = 0; j < 8; ++j)
        yy[tid * 8 + j] = (dst_t) as_type<float>((uint) x[j] << 16);
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template<typename dst_t, typename Ptr>
static inline void iqk_dq_dispatch(int ty, constant const uint8_t* vx, long ibs, Ptr y, int tid) {
    switch (ty) {
        case 16: iqk_dq_iq2_xxs<dst_t>(vx, ibs, y, tid); break;
        case 17: iqk_dq_iq2_xs<dst_t>(vx, ibs, y, tid); break;
        case 18: iqk_dq_iq3_xxs<dst_t>(vx, ibs, y, tid); break;
        case 20: iqk_dq_iq4_nl<dst_t>(vx, ibs, y, tid); break;
        case 21: iqk_dq_iq3_s<dst_t>(vx, ibs, y, tid); break;
        case 22: iqk_dq_iq2_s<dst_t>(vx, ibs, y, tid); break;
        case 29: iqk_dq_iq1_m<dst_t>(vx, ibs, y, tid); break;
        case 23: iqk_dq_iq4_xs<dst_t>(vx, ibs, y, tid); break;
        case 11: iqk_dq_q3_k<dst_t>(vx, ibs, y, tid); break;
        case 42: iqk_dq_q2_0<dst_t>(vx, ibs, y, tid); break;
        case 12: iqk_dq_q4_k<dst_t>(vx, ibs, y, tid); break;
        case 13: iqk_dq_q5_k<dst_t>(vx, ibs, y, tid); break;
        case 7: iqk_dq_q5_1<dst_t>(vx, ibs, y, tid); break;
        case 8: iqk_dq_q8_0<dst_t>(vx, ibs, y, tid); break;
        case 30: iqk_dq_bf16<dst_t>(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
kernel void dequant_flat_kernel_f32(constant const int& ty [[buffer(0)]],
                                    constant const uint8_t* vx [[buffer(1)]],
                                    device float* y [[buffer(2)]],
                                    uint3 gpos [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    iqk_dq_dispatch<float>(ty, vx, (long) gpos.x, y + (size_t) gpos.x * QK_K, (int) tid);
}
kernel void dequant_flat_kernel_f16(constant const int& ty [[buffer(0)]],
                                    constant const uint8_t* vx [[buffer(1)]],
                                    device half* y [[buffer(2)]],
                                    uint3 gpos [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    iqk_dq_dispatch<half>(ty, vx, (long) gpos.x, y + (size_t) gpos.x * QK_K, (int) tid);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
kernel void dequant_gu_kernel(constant const int& ty [[buffer(0)]],
                              constant const uint8_t* gate [[buffer(1)]],
                              constant const uint8_t* up [[buffer(2)]],
                              constant const long& per_row [[buffer(3)]],
                              device half* y [[buffer(4)]],
                              uint3 gpos [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    const long i = (long) gpos.x;
    const int parity = (int) gpos.y;
    const long r = i / per_row, c = i % per_row;
    iqk_dq_dispatch<half>(ty, parity ? up : gate, i, y + (size_t) ((2 * r + parity) * per_row + c) * QK_K, (int) tid);
}
kernel void embed_rows_kernel(constant const int& ty [[buffer(0)]],
                              constant const uint8_t* table [[buffer(1)]],
                              constant const ulong& row_bytes [[buffer(2)]],
                              constant const int* tokens [[buffer(3)]],
                              constant const long& n_embd [[buffer(4)]],
                              device float* y [[buffer(5)]],
                              uint3 gpos [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    const int t = (int) gpos.y;
    const long b = (long) gpos.x;
    constant const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    iqk_dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + (size_t) b * QK_K, (int) tid);
}

// ---------------------------------------------------------------- the MMVQ kernels
// CUDA's mmvq_kernel: block (32, 4) = four warps of one row each; threadIdx.y is the row-within-block, so on a
// flat thread index tid (x fastest) it is tid >> 5 and threadIdx.x is tid & 31.
template<int TY>
static inline void iqk_mmvq_body(constant const uint8_t* w, ulong row_bytes, constant const block_q8_1* x,
                                 device float* y, int n_in, int n_out, int ncols, uint gpos_x, uint tid) {
    const int row = (int) gpos_x * 4 + (int) (tid >> 5);
    if (row >= n_out) return;
    const int lane = (int) (tid & 31u);
    const int nb = n_in / iqk_Fmt<TY>::qk;
    constant const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = iqk_row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}
// mmvq_kernel with the columns taken NC at a time.  After warp_sum every lane holds the same sum, so lane c stores
// column c.
template<int TY, int NC>
static inline void iqk_mmvq_multi_body(constant const uint8_t* w, ulong row_bytes, constant const block_q8_1* x,
                                       device float* y, int n_in, int n_out, int ncols, uint gpos_x, uint tid) {
    const int row = (int) gpos_x * 4 + (int) (tid >> 5);
    if (row >= n_out) return;
    const int lane = (int) (tid & 31u);
    const int nb = n_in / iqk_Fmt<TY>::qk, xb = n_in / 32;
    constant const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        const int n = min(NC, ncols - c0);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (c0 + min(c, n - 1)) * xb;
        float s[NC];
        iqk_row_dot_multi<TY, NC>(wr, x, off, n, nb, lane, s);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < n && (int) lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
    }
}

#define IQK_MMVQ_PARAMS \
    constant const uint8_t* w [[buffer(0)]], \
    constant const ulong& row_bytes [[buffer(1)]], \
    constant const block_q8_1* x [[buffer(2)]], \
    device float* y [[buffer(3)]], \
    constant const int& n_in [[buffer(4)]], \
    constant const int& n_out [[buffer(5)]], \
    constant const int& ncols [[buffer(6)]], \
    uint3 gpos [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]
#define IQK_MMVQ_KERNEL(T) \
kernel void mmvq_kernel_##T(IQK_MMVQ_PARAMS) { iqk_mmvq_body<T>(w, row_bytes, x, y, n_in, n_out, ncols, gpos.x, tid); }
#define IQK_MMVQ_MULTI_KERNEL(T, NC) \
kernel void mmvq_multi_kernel_##T##_##NC(IQK_MMVQ_PARAMS) { \
    iqk_mmvq_multi_body<T, NC>(w, row_bytes, x, y, n_in, n_out, ncols, gpos.x, tid); \
}
IQK_MMVQ_KERNEL(16) IQK_MMVQ_KERNEL(17) IQK_MMVQ_KERNEL(18) IQK_MMVQ_KERNEL(20) IQK_MMVQ_KERNEL(21)
IQK_MMVQ_KERNEL(22) IQK_MMVQ_KERNEL(23) IQK_MMVQ_KERNEL(29) IQK_MMVQ_KERNEL(42) IQK_MMVQ_KERNEL(12)
IQK_MMVQ_KERNEL(13) IQK_MMVQ_KERNEL(7) IQK_MMVQ_KERNEL(8)
IQK_MMVQ_KERNEL(11)   // the native_mmvq Q3_K bridge (per-column kernel only; iq_mmvq never dispatches it)
IQK_MMVQ_MULTI_KERNEL(16, 1) IQK_MMVQ_MULTI_KERNEL(16, 2) IQK_MMVQ_MULTI_KERNEL(16, 4) IQK_MMVQ_MULTI_KERNEL(16, 8)
IQK_MMVQ_MULTI_KERNEL(17, 1) IQK_MMVQ_MULTI_KERNEL(17, 2) IQK_MMVQ_MULTI_KERNEL(17, 4) IQK_MMVQ_MULTI_KERNEL(17, 8)
IQK_MMVQ_MULTI_KERNEL(18, 1) IQK_MMVQ_MULTI_KERNEL(18, 2) IQK_MMVQ_MULTI_KERNEL(18, 4) IQK_MMVQ_MULTI_KERNEL(18, 8)
IQK_MMVQ_MULTI_KERNEL(20, 1) IQK_MMVQ_MULTI_KERNEL(20, 2) IQK_MMVQ_MULTI_KERNEL(20, 4) IQK_MMVQ_MULTI_KERNEL(20, 8)
IQK_MMVQ_MULTI_KERNEL(21, 1) IQK_MMVQ_MULTI_KERNEL(21, 2) IQK_MMVQ_MULTI_KERNEL(21, 4) IQK_MMVQ_MULTI_KERNEL(21, 8)
IQK_MMVQ_MULTI_KERNEL(22, 1) IQK_MMVQ_MULTI_KERNEL(22, 2) IQK_MMVQ_MULTI_KERNEL(22, 4) IQK_MMVQ_MULTI_KERNEL(22, 8)
IQK_MMVQ_MULTI_KERNEL(23, 1) IQK_MMVQ_MULTI_KERNEL(23, 2) IQK_MMVQ_MULTI_KERNEL(23, 4) IQK_MMVQ_MULTI_KERNEL(23, 8)
IQK_MMVQ_MULTI_KERNEL(29, 1) IQK_MMVQ_MULTI_KERNEL(29, 2) IQK_MMVQ_MULTI_KERNEL(29, 4) IQK_MMVQ_MULTI_KERNEL(29, 8)
IQK_MMVQ_MULTI_KERNEL(42, 1) IQK_MMVQ_MULTI_KERNEL(42, 2) IQK_MMVQ_MULTI_KERNEL(42, 4) IQK_MMVQ_MULTI_KERNEL(42, 8)

// ---------------------------------------------------------------- grouped native experts
constant const int IQK_GU_ROWS = 8;     // rows per block (one warp each)
constant const int IQK_GRP_NC = 4;      // entries per decode-once pass (kVerifyMaxT = 8; see the CUDA file)

// native_gu_kernel for ONE group (rule 9: blob as a bound buffer, g as a scalar - the CUDA kernel's
// blockIdx.y iterated the groups through the grp_ptr table, which the launcher now walks host-side)
template<int TG>
static inline void iqk_native_gu_body(constant const uint8_t* blob, constant const int* grp_start,
                                      constant const int* n_groups, constant const int* ent_tok,
                                      constant const block_q8_1* xq, long n_embd, long n_ff, ulong gu_row,
                                      ulong up_off, device float* gate, device float* up, int g,
                                      uint gpos_x, uint tid) {
    if (g >= n_groups[0]) return;
    const uint warp = tid >> 5, lane = tid & 31u;
    const int row = (int) gpos_x * IQK_GU_ROWS + (int) warp;             // 0 .. 2*n_ff
    if (row >= 2 * (int) n_ff) return;
    const bool is_up = row >= (int) n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    constant const uint8_t* wr = blob + (is_up ? up_off : 0) + (size_t) r * gu_row;
    const int nb = (int) (n_embd / iqk_Fmt<TG>::qk), xb = (int) (n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = iqk_row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, (int) lane);
        if (lane == 0) (is_up ? up : gate)[(size_t) e * (size_t) n_ff + r] = s;
    }
}
// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.
template<int TG>
static inline void iqk_native_gu_multi_body(constant const uint8_t* blob, constant const int* grp_start,
                                            constant const int* n_groups, constant const int* ent_tok,
                                            constant const block_q8_1* xq, long n_embd, long n_ff, ulong gu_row,
                                            ulong up_off, device float* gate, device float* up, int g,
                                            uint gpos_x, uint tid) {
    if (g >= n_groups[0]) return;
    const uint warp = tid >> 5, lane = tid & 31u;
    const int row = (int) gpos_x * IQK_GU_ROWS + (int) warp;             // 0 .. 2*n_ff
    if (row >= 2 * (int) n_ff) return;
    const bool is_up = row >= (int) n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    constant const uint8_t* wr = blob + (is_up ? up_off : 0) + (size_t) r * gu_row;
    const int nb = (int) (n_embd / iqk_Fmt<TG>::qk), xb = (int) (n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    device float* dst = is_up ? up : gate;
    for (int e = e0; e < e1; e += IQK_GRP_NC) {
        const int n = min(IQK_GRP_NC, e1 - e);
        int off[IQK_GRP_NC];
#pragma unroll
        for (int c = 0; c < IQK_GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
        float s[IQK_GRP_NC];
        iqk_row_dot_multi<TG, IQK_GRP_NC>(wr, xq, off, n, nb, (int) lane, s);
#pragma unroll
        for (int c = 0; c < IQK_GRP_NC; ++c)
            if (c < n && (int) lane == c) dst[(size_t) (e + c) * (size_t) n_ff + r] = s[c];
    }
}
// native_down_kernel for ONE group (same per-group form)
template<int TD>
static inline void iqk_native_down_body(constant const uint8_t* blob, constant const int* grp_start,
                                        constant const int* n_groups, constant const int* ent_dst,
                                        constant const block_q8_1* hq, long n_embd, long n_ff, ulong d_row,
                                        ulong down_off, device float* out, int g, uint gpos_x, uint tid) {
    if (g >= n_groups[0]) return;
    const uint warp = tid >> 5, lane = tid & 31u;
    const int r = (int) gpos_x * 8 + (int) warp;
    if (r >= (int) n_embd) return;
    constant const uint8_t* wr = blob + down_off + (size_t) r * d_row;
    const int nb = (int) (n_ff / iqk_Fmt<TD>::qk), hb = (int) (n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = iqk_row_dot<TD>(wr, hq + (size_t) e * hb, nb, (int) lane);
        if (lane == 0) out[(size_t) ent_dst[e] * (size_t) n_embd + r] = s;
    }
}
// native_down_kernel with the entries taken GRP_NC at a time
template<int TD>
static inline void iqk_native_down_multi_body(constant const uint8_t* blob, constant const int* grp_start,
                                              constant const int* n_groups, constant const int* ent_dst,
                                              constant const block_q8_1* hq, long n_embd, long n_ff, ulong d_row,
                                              ulong down_off, device float* out, int g, uint gpos_x, uint tid) {
    if (g >= n_groups[0]) return;
    const uint warp = tid >> 5, lane = tid & 31u;
    const int r = (int) gpos_x * 8 + (int) warp;
    if (r >= (int) n_embd) return;
    constant const uint8_t* wr = blob + down_off + (size_t) r * d_row;
    const int nb = (int) (n_ff / iqk_Fmt<TD>::qk), hb = (int) (n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += IQK_GRP_NC) {
        const int n = min(IQK_GRP_NC, e1 - e);
        int off[IQK_GRP_NC];
#pragma unroll
        for (int c = 0; c < IQK_GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
        float s[IQK_GRP_NC];
        iqk_row_dot_multi<TD, IQK_GRP_NC>(wr, hq, off, n, nb, (int) lane, s);
#pragma unroll
        for (int c = 0; c < IQK_GRP_NC; ++c)
            if (c < n && (int) lane == c) out[(size_t) ent_dst[e + c] * (size_t) n_embd + r] = s[c];
    }
}

#define IQK_GU_PARAMS \
    constant const uint8_t* blob [[buffer(0)]], \
    constant const int* grp_start [[buffer(1)]], \
    constant const int* n_groups [[buffer(2)]], \
    constant const int* ent_tok [[buffer(3)]], \
    constant const block_q8_1* xq [[buffer(4)]], \
    constant const long& n_embd [[buffer(5)]], \
    constant const long& n_ff [[buffer(6)]], \
    constant const ulong& gu_row [[buffer(7)]], \
    constant const ulong& up_off [[buffer(8)]], \
    device float* gate [[buffer(9)]], \
    device float* up [[buffer(10)]], \
    constant const int& g [[buffer(11)]], \
    uint3 gpos [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]
#define IQK_GU_KERNEL(T) \
kernel void native_gu_kernel_##T(IQK_GU_PARAMS) { \
    iqk_native_gu_body<T>(blob, grp_start, n_groups, ent_tok, xq, n_embd, n_ff, gu_row, up_off, gate, up, g, gpos.x, tid); \
}
#define IQK_GU_MULTI_KERNEL(T) \
kernel void native_gu_multi_kernel_##T(IQK_GU_PARAMS) { \
    iqk_native_gu_multi_body<T>(blob, grp_start, n_groups, ent_tok, xq, n_embd, n_ff, gu_row, up_off, gate, up, g, gpos.x, tid); \
}
IQK_GU_KERNEL(16) IQK_GU_KERNEL(17) IQK_GU_KERNEL(18) IQK_GU_KERNEL(21) IQK_GU_KERNEL(22) IQK_GU_KERNEL(23)
IQK_GU_KERNEL(29) IQK_GU_KERNEL(42) IQK_GU_KERNEL(12) IQK_GU_KERNEL(13) IQK_GU_KERNEL(8)
IQK_GU_MULTI_KERNEL(16) IQK_GU_MULTI_KERNEL(17) IQK_GU_MULTI_KERNEL(18) IQK_GU_MULTI_KERNEL(21)
IQK_GU_MULTI_KERNEL(22) IQK_GU_MULTI_KERNEL(23) IQK_GU_MULTI_KERNEL(29) IQK_GU_MULTI_KERNEL(42)

#define IQK_DOWN_PARAMS \
    constant const uint8_t* blob [[buffer(0)]], \
    constant const int* grp_start [[buffer(1)]], \
    constant const int* n_groups [[buffer(2)]], \
    constant const int* ent_dst [[buffer(3)]], \
    constant const block_q8_1* hq [[buffer(4)]], \
    constant const long& n_embd [[buffer(5)]], \
    constant const long& n_ff [[buffer(6)]], \
    constant const ulong& d_row [[buffer(7)]], \
    constant const ulong& down_off [[buffer(8)]], \
    device float* out [[buffer(9)]], \
    constant const int& g [[buffer(10)]], \
    uint3 gpos [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]
#define IQK_DOWN_KERNEL(T) \
kernel void native_down_kernel_##T(IQK_DOWN_PARAMS) { \
    iqk_native_down_body<T>(blob, grp_start, n_groups, ent_dst, hq, n_embd, n_ff, d_row, down_off, out, g, gpos.x, tid); \
}
#define IQK_DOWN_MULTI_KERNEL(T) \
kernel void native_down_multi_kernel_##T(IQK_DOWN_PARAMS) { \
    iqk_native_down_multi_body<T>(blob, grp_start, n_groups, ent_dst, hq, n_embd, n_ff, d_row, down_off, out, g, gpos.x, tid); \
}
IQK_DOWN_KERNEL(20) IQK_DOWN_KERNEL(23) IQK_DOWN_KERNEL(42) IQK_DOWN_KERNEL(7) IQK_DOWN_KERNEL(8)
IQK_DOWN_MULTI_KERNEL(20) IQK_DOWN_MULTI_KERNEL(23) IQK_DOWN_MULTI_KERNEL(42)

// Fully resident native experts. Routes and residency are read at replay time; only the arena is a
// bound pointer. Byte offsets stay integers, including for mixed-size slots in the same allocation.
template<int TY, bool DOWN>
static inline void iqk_resident_body(constant const uint8_t* arena, constant const ulong* offsets,
                                     constant const int* ids, constant const int* residency,
                                     constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes,
                                     ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                                     device float* out, device float* up, uint3 gp, uint tid) {
    const int e = (int) gp.y, row = (int) gp.x * 8 + (int) (tid >> 5), lane = (int) (tid & 31);
    if (row >= (DOWN ? n_embd : 2 * n_ff)) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    const bool is_up = !DOWN && row >= n_ff;
    const int r = is_up ? row - (int) n_ff : row;
    float value = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* wr = arena + off + ((DOWN || is_up) ? weight_offset : 0) + (ulong) r * row_bytes;
        const int width = (int) (DOWN ? n_ff : n_embd);
        const int tok = DOWN ? e : e / k;
        value = iqk_row_dot<TY>(wr, xq + (size_t) tok * (width / 32), width / iqk_Fmt<TY>::qk, lane);
    }
    if (lane == 0) (is_up ? up : out)[(size_t) e * (DOWN ? n_embd : n_ff) + r] = value;
}
#define IQK_RESIDENT_PARAMS \
    constant const uint8_t* arena [[buffer(0)]], constant const ulong* offsets [[buffer(1)]], \
    constant const int* ids [[buffer(2)]], constant const int* residency [[buffer(3)]], \
    constant const block_q8_1* xq [[buffer(4)]], constant const long& n_embd [[buffer(5)]], \
    constant const long& n_ff [[buffer(6)]], constant const ulong& row_bytes [[buffer(7)]], \
    constant const ulong& weight_offset [[buffer(8)]], constant const ulong& slot_bytes [[buffer(9)]], \
    constant const int& n_expert [[buffer(10)]], constant const int& k [[buffer(11)]], \
    constant const int& has_offsets [[buffer(12)]], device float* out [[buffer(13)]]
#define IQK_RESIDENT_CALL(T, DOWN, UP) \
    iqk_resident_body<T, DOWN>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, \
                               slot_bytes, n_expert, k, has_offsets, out, UP, gp, tid);
#define IQK_RESIDENT_GU(T) \
kernel void native_resident_gu_##T(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                                   uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    IQK_RESIDENT_CALL(T, false, up) \
}
#define IQK_RESIDENT_DOWN(T) \
kernel void native_resident_down_##T(IQK_RESIDENT_PARAMS, \
                                     uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    IQK_RESIDENT_CALL(T, true, out) \
}
IQK_RESIDENT_GU(16) IQK_RESIDENT_GU(17) IQK_RESIDENT_GU(18) IQK_RESIDENT_GU(21)
IQK_RESIDENT_GU(22) IQK_RESIDENT_GU(23) IQK_RESIDENT_GU(29) IQK_RESIDENT_GU(42)
IQK_RESIDENT_GU(12) IQK_RESIDENT_GU(13) IQK_RESIDENT_GU(8)
IQK_RESIDENT_DOWN(20) IQK_RESIDENT_DOWN(23) IQK_RESIDENT_DOWN(42) IQK_RESIDENT_DOWN(7) IQK_RESIDENT_DOWN(8)

// ---------------------------------------------------------------- direct dots (decode, single token)
// The same lane-to-call mapping, ascending per-lane accumulation, butterfly and per-call integer scale step and
// float expression as iqk_row_dot / the resident body; only a call's integer sum is formed differently. The
// original packs the codes into dp4a words (byte_perm / vcmpne4 / vsub4 SWAR) and runs the emulated dp4a; these
// expand each code to a float (a sign flip is an XOR of the sign bit) and FMA it against the activation byte.
// Every product and partial sum is an integer below 2^24 in magnitude (|code| <= 62, |q8| <= 128, 32 terms), so
// the float sums are exact in any order and (int) of them is the original int32. A lane's activation block is
// the same for every row, so R rows per warp convert it once. Measured, M2 Max, isolated
// (bench/results/2026-10-03-metal-decode-opt2/micro/experts-iq3s.log): Q2_0 resident down about 1.4-1.5x faster
// (110 -> 73 us in that log), IQ3_S 2560 x 6144 about 1.1x; IQ2_S / IQ2_XXS gate/up showed no stable gain in
// any variant there and keep the original kernels.
struct IqkAct { float4 a[8]; float d8; };
static inline IqkAct iqk_act(constant const block_q8_1* c) {
    IqkAct r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.a[j] = float4(as_type<char4>(reinterpret_cast<constant const int*>(c->qs)[j]));
    r.d8 = iqk_lo2f(c->ds);
    return r;
}
// the four bytes of a grid word as floats, byte b negated when bit b of s is set
static inline float4 iqk_signed4(uint w, uint s) {
    const uint4 m = uint4(s & 1u, (s >> 1) & 1u, (s >> 2) & 1u, (s >> 3) & 1u) << 31;
    return as_type<float4>(as_type<uint4>(float4(as_type<uchar4>(w))) ^ m);
}
static inline float iqk_fma4(float4 v, float4 a, float s) {
    s = fma(v.x, a.x, s); s = fma(v.y, a.y, s); s = fma(v.z, a.z, s); return fma(v.w, a.w, s);
}
// Q2_0: element 8j+m of the call is ((qs16[j] >> 2m) & 3) - 1, the {-1, 0, 1, 2} table byte_perm builds
static inline float iqk_direct_q2_0(constant const uint8_t* vbq, int kbx, int iqs, const thread IqkAct& c) {
    constant const block_q2_0* b = reinterpret_cast<constant const block_q2_0*>(vbq) + kbx;
    const float d2 = (float) b->d;
    constant const uint16_t* qs = reinterpret_cast<constant const uint16_t*>(b->qs) + iqs * 4;
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint q = qs[j];
        const float4 v0 = float4(float(q & 3u), float((q >> 2) & 3u), float((q >> 4) & 3u), float((q >> 6) & 3u)) - 1.0f;
        const float4 v1 = float4(float((q >> 8) & 3u), float((q >> 10) & 3u), float((q >> 12) & 3u), float((q >> 14) & 3u)) - 1.0f;
        s = iqk_fma4(v0, c.a[2 * j], s);
        s = iqk_fma4(v1, c.a[2 * j + 1], s);
    }
    return d2 * c.d8 * (float) (int) s;
}
// IQ3_S: grid index i = qs[i] | (bit i of qh) << 8; sign byte p covers the eight values of pair p
static inline float iqk_direct_iq3_s(constant const uint8_t* vbq, int kbx, int iqs, const thread IqkAct& c) {
    constant const block_iq3_s* bq3 = reinterpret_cast<constant const block_iq3_s*>(vbq) + kbx;
    const uint q0 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 0);
    const uint q1 = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->qs), iqs + 1);
    const uint qh = bq3->qh[iqs / 2];
    const uint sp = (uint) iqk_get_int_b2(reinterpret_cast<constant const uint16_t*>(bq3->signs), iqs / 2);
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint qq = p < 2 ? q0 : q1, sh = 16u * (uint) (p & 1);
        const uint i0 = ((qq >> sh) & 0xFFu) | ((qh << (8 - 2 * p)) & 0x100u);
        const uint i1 = ((qq >> (sh + 8)) & 0xFFu) | ((qh << (7 - 2 * p)) & 0x100u);
        const uint sg = (sp >> (8 * p)) & 0xFFu;
        s = iqk_fma4(iqk_signed4(iq3s_grid[i0], sg), c.a[2 * p], s);
        s = iqk_fma4(iqk_signed4(iq3s_grid[i1], sg >> 4), c.a[2 * p + 1], s);
    }
    int sumi = (int) s;
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (float) bq3->d * c.d8;
    return d * (float) sumi;
}

// Q2_0 resident down, R rows per warp (the resident body's per-row arithmetic; grid x = ceil(n_embd / (8R)))
template<int R>
static inline void iqk_resident_down_direct_q2_0(constant const uint8_t* arena, constant const ulong* offsets,
                                                 constant const int* ids, constant const int* residency,
                                                 constant const block_q8_1* xq, long n_embd, long n_ff,
                                                 ulong row_bytes, ulong weight_offset, ulong slot_bytes, int n_expert,
                                                 int has_offsets, device float* out, uint3 gp, uint tid) {
    using F = iqk_Fmt<42>;
    const int e = (int) gp.y, lane = (int) (tid & 31), row0 = ((int) gp.x * 8 + (int) (tid >> 5)) * R;
    if (row0 >= (int) n_embd) return;
    const int id = ids[e], slot = (id >= 0 && id < n_expert) ? residency[id] : -1;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : (ulong) slot * slot_bytes;
        constant const uint8_t* base = arena + off + weight_offset;
        constant const block_q8_1* xt = xq + (size_t) e * (n_ff / 32);
        const int nb = (int) (n_ff / F::qk);
        for (int kk = lane; kk < nb * F::ipb; kk += 32) {
            const int kbx = kk / F::ipb, iqs = F::step * (kk % F::ipb);
            const IqkAct c = iqk_act(xt + kbx * 2 + iqs);
#pragma unroll
            for (int q = 0; q < R; ++q)
                s[q] += iqk_direct_q2_0(base + (ulong) min(row0 + q, (int) n_embd - 1) * row_bytes, kbx, iqs, c);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = slot >= 0 ? iqk_warp_sum(s[q]) : 0.0f;
        if (lane == 0 && row0 + q < (int) n_embd) out[(size_t) e * n_embd + row0 + q] = v;
    }
}
kernel void native_resident_down_direct_42_r4(IQK_RESIDENT_PARAMS, uint3 gp [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]]) {
    iqk_resident_down_direct_q2_0<4>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset,
                                     slot_bytes, n_expert, has_offsets, out, gp, tid);
}

// IQ3_S single column (mmvq_multi_kernel_21_1's arithmetic), R rows per warp: grid x = ceil(n_out / (4R))
template<int R>
static inline void iqk_mmvq_direct_iq3_s(constant const uint8_t* w, ulong row_bytes, constant const block_q8_1* x,
                                         device float* y, int n_in, int n_out, uint gx, uint tid) {
    const int row0 = ((int) gx * 4 + (int) (tid >> 5)) * R;
    if (row0 >= n_out) return;
    const int lane = (int) (tid & 31u), nb = n_in / 256;
    float s[R];
#pragma unroll
    for (int q = 0; q < R; ++q) s[q] = 0.0f;
    for (int kk = lane; kk < nb * 8; kk += 32) {
        const int kbx = kk / 8, iqs = 2 * (kk % 8);
        const IqkAct c = iqk_act(x + kbx * 8 + iqs / 2);
#pragma unroll
        for (int q = 0; q < R; ++q) s[q] += iqk_direct_iq3_s(w + (size_t) min(row0 + q, n_out - 1) * row_bytes, kbx, iqs, c);
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        const float v = iqk_warp_sum(s[q]);
        if (lane == 0 && row0 + q < n_out) y[row0 + q] = v;
    }
}
kernel void mmvq_direct_21_r2(IQK_MMVQ_PARAMS) { iqk_mmvq_direct_iq3_s<2>(w, row_bytes, x, y, n_in, n_out, gpos.x, tid); }

// Store adapter for one 32-value slice of a 256-value IQ block. Four threads per output row
// call the original dequantizer and write straight into a transposed matrix tile. This needs only
// 5 KiB of threadgroup memory, and partial 640-wide down blocks read only their valid 32-value slices.
struct IqTileStore {
    threadgroup half* tile;
    long row, first, logical;
    IqTileStore operator+(long offset) const thread {
        return {tile, row, first, logical + offset};
    }
    threadgroup half& operator[](long index) const thread {
        return tile[(logical + index - first) * 32 + row];
    }
};

template<int TY, bool DOWN>
inline void iqk_gemm(constant const ushort* x, constant const uint8_t* arena, constant const int4* tiles,
                     device float* y, long H, long FF, ulong row_bytes, ulong weight_offset,
                     uint2 tile, uint tid, uint sg,
                     threadgroup half* sx, threadgroup half* sw, threadgroup float* result) {
    const uint K = (uint) (DOWN ? FF : H), N = (uint) (DOWN ? H : 2 * FF);
    const int4 desc = tiles[tile.y];
    const ulong offset = ulong(uint(desc.x)) | (ulong(uint(desc.w)) << 32);
    constant const uint8_t* blob = arena + offset;
    const uint row0 = uint(desc.y), rows = uint(desc.z), nr = tile.x * 32;
    const uint sm = (sg / 2) * 8, sn = (sg % 2) * 16;
    simdgroup_float8x8 c0(0.0f), c1(0.0f);
    const uint r = tid / 4, n = nr + r, quarter = tid % 4;
    constant const uint8_t* w = blob + (DOWN ? weight_offset : (n & 1) * weight_offset)
                               + (ulong) (DOWN ? n : n / 2) * row_bytes;
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < 16 * 32; i += 128)
            sx[i] = i / 32 < rows ? as_type<half>(x[(ulong)(row0 + i / 32) * K + k0 + i % 32]) : half(0);
        const uint within = k0 % 256;
        const uint dq_tid = TY == 42 ? (within / 64) * 8 + (within % 64) / 8 + quarter
                                    : quarter * 8 + within / 32;
        IqTileStore store{sw, (long) r, (long) within, 0};
        iqk_dq_dispatch<half>(TY, w, k0 / 256, store, dq_tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a, b0, b1;
            simdgroup_load(a, sx + sm * 32 + k, 32);
            simdgroup_load(b0, sw + k * 32 + sn, 32);
            simdgroup_load(b1, sw + k * 32 + sn + 8, 32);
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0, result + sm * 32 + sn, 32);
    simdgroup_store(c1, result + sm * 32 + sn + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16 * 32; i += 128)
        if (i / 32 < rows) y[(ulong)(row0 + i / 32) * N + nr + i % 32] = result[i];
}
#define IQK_GEMM(NAME, TY, DOWN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], constant const uint8_t* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup half sx[16 * 32], sw[32 * 32]; threadgroup float result[16 * 32]; \
    iqk_gemm<TY, DOWN>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, sx, sw, result); \
}
IQK_GEMM(native_gemm_gu_16, 16, false) IQK_GEMM(native_gemm_gu_17, 17, false)
IQK_GEMM(native_gemm_gu_18, 18, false) IQK_GEMM(native_gemm_gu_21, 21, false)
IQK_GEMM(native_gemm_gu_22, 22, false) IQK_GEMM(native_gemm_gu_23, 23, false)
IQK_GEMM(native_gemm_gu_29, 29, false) IQK_GEMM(native_gemm_gu_42, 42, false)
IQK_GEMM(native_gemm_down_20, 20, true) IQK_GEMM(native_gemm_down_23, 23, true)
IQK_GEMM(native_gemm_down_42, 42, true)

// The same expert products in wider / taller tiles: BM routed rows of one expert x BN columns per threadgroup,
// SGM x SGN SIMD groups. Every output element gets iqk_gemm's MMA sequence - the same half operands (the x rows;
// the weights decoded by the same iqk_dq_dispatch call per (column, quarter) into the same 32-value k-slices),
// 8-wide k steps ascending from a zero accumulator - stored as is; so the results are the same bits
// (micro/prefill-moe.log). A BN = 64 tile shares each x slice over twice the columns; BM = 32 decodes each weight
// slice once for twice the rows, worth it when experts get many rows. Measured on the M2 Max, isolated: 16 x 64
// 12-18% faster at ~20 rows per expert, 32 x 64 about 21% faster at ~78.
struct IqTileStoreN {           // IqTileStore with a BN-wide transposed tile
    threadgroup half* tile;
    long row, first, logical, stride;
    IqTileStoreN operator+(long offset) const thread { return {tile, row, first, logical + offset, stride}; }
    threadgroup half& operator[](long index) const thread { return tile[(logical + index - first) * stride + row]; }
};

template<int TY, bool DOWN, int BM, int BN, int SGM, int SGN>
inline void iqk_gemm2(constant const ushort* x, constant const uint8_t* arena, constant const int4* tiles, device float* y,
                    long H, long FF, ulong row_bytes, ulong weight_offset, uint2 tile, uint tid, uint sg, uint lane,
                    threadgroup half* sx, threadgroup half* sw, threadgroup float* scratch) {
    constexpr int TH = 32 * SGM * SGN, MI = BM / SGM / 8, NJ = BN / SGN / 8;
    const uint K = (uint) (DOWN ? FF : H), N = (uint) (DOWN ? H : 2 * FF);
    const int4 desc = tiles[tile.y];
    const ulong offset = ulong(uint(desc.x)) | (ulong(uint(desc.w)) << 32);
    constant const uint8_t* blob = arena + offset;
    const uint row0 = uint(desc.y), rows = uint(desc.z), nr = tile.x * BN;
    const uint sm = (sg / SGN) * (BM / SGM), sn = (sg % SGN) * (BN / SGN);
    simdgroup_float8x8 c[MI][NJ];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) c[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < BM * 32; i += TH)
            sx[i] = i / 32 < rows ? as_type<half>(x[(ulong) (row0 + i / 32) * K + k0 + i % 32]) : half(0);
        const uint within = k0 % 256;
        for (uint j = tid; j < (uint) BN * 4; j += TH) {
            const uint r = j / 4, n = nr + r, quarter = j % 4;
            constant const uint8_t* w = blob + (DOWN ? weight_offset : (n & 1) * weight_offset)
                                       + (ulong) (DOWN ? n : n / 2) * row_bytes;
            const uint dq_tid = TY == 42 ? (within / 64) * 8 + (within % 64) / 8 + quarter : quarter * 8 + within / 32;
            IqTileStoreN store{sw, (long) r, (long) within, 0, BN};
            iqk_dq_dispatch<half>(TY, w, k0 / 256, store, dq_tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a[MI], b[NJ];
#pragma unroll
            for (int i = 0; i < MI; ++i) simdgroup_load(a[i], sx + (sm + i * 8) * 32 + k, 32);
#pragma unroll
            for (int j = 0; j < NJ; ++j) simdgroup_load(b[j], sw + k * BN + sn + j * 8, BN);
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NJ; ++j) simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* mine = scratch + sg * 64;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            simdgroup_store(c[i][j], mine, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint m = sm + i * 8 + e / 8, n = nr + sn + j * 8 + e % 8;
                if (m < rows) y[(ulong) (row0 + m) * N + n] = mine[e];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
#define IQK_GEMM2(NAME, TY, DOWN, BM, BN, SGM, SGN) \
kernel void NAME(constant const ushort* x [[buffer(0)]], constant const uint8_t* arena [[buffer(1)]], \
                 constant const int4* tiles [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant const long& H [[buffer(4)]], constant const long& FF [[buffer(5)]], \
                 constant const ulong& row_bytes [[buffer(6)]], constant const ulong& weight_offset [[buffer(7)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half sx[BM * 32], sw[32 * BN]; threadgroup float scratch[SGM * SGN * 64]; \
    iqk_gemm2<TY, DOWN, BM, BN, SGM, SGN>(x, arena, tiles, y, H, FF, row_bytes, weight_offset, tile, tid, sg, lane, sx, sw, scratch); }
#define IQK_GEMM2_PAIR(TAG, TY, DOWN) \
    IQK_GEMM2(native_gemm2_##TAG##_r16, TY, DOWN, 16, 64, 2, 2) IQK_GEMM2(native_gemm2_##TAG##_r32, TY, DOWN, 32, 64, 2, 4)
IQK_GEMM2_PAIR(gu_16, 16, false) IQK_GEMM2_PAIR(gu_17, 17, false) IQK_GEMM2_PAIR(gu_18, 18, false)
IQK_GEMM2_PAIR(gu_21, 21, false) IQK_GEMM2_PAIR(gu_22, 22, false) IQK_GEMM2_PAIR(gu_23, 23, false)
IQK_GEMM2_PAIR(gu_29, 29, false) IQK_GEMM2_PAIR(gu_42, 42, false)
IQK_GEMM2_PAIR(down_20, 20, true) IQK_GEMM2_PAIR(down_23, 23, true) IQK_GEMM2_PAIR(down_42, 42, true)

// ---------------------------------------------------------------- the element-wise steps
kernel void swiglu_entries_kernel(constant const float* gate [[buffer(0)]],
                                  constant const float* up [[buffer(1)]],
                                  device float* h [[buffer(2)]],
                                  constant const long& n [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
    if ((long) i >= n) return;
    const float g = gate[i];
    // __expf -> metal::precise::exp (the port's substitute at every exp site)
    h[i] = (g / (1.0f + metal::precise::exp(-g))) * up[i];
}
