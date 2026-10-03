// src/kernels/metal/s2_expert_grouped.metal - the port of src/kernels/cuda/s2_expert_grouped.cu: R4's grouped
// GPU expert, all four families of it (the previous one-warp-per-row kernels, the rewritten pair kernels, the
// grouped verify-window kernels in both stagings, and the CPU-order experimental path), plus the token-graph
// helpers (hit select, hit add, resident-group build).
//
// The arithmetic is the CUDA file's own, transcription for transcription, because `s2_expert_grouped_parity`
// compares the PREVIOUS kernels against the NEW ones BITWISE on identical inputs - every integer sum is exact
// (`dp4a`), every float expression keeps its operand order, every reduction keeps its shuffle tree, and the
// lane->chunk assignment (`c = lane; c += 32`) is untouched.  What MSL forced, each following a measured port
// rule (docs/PORT_METAL/PROGRESS.md):
//   * __dp4a -> s2eg_dp4a: dp4a.hpp's own sm_60 fallback, that header's documented bit-exact definition (the
//     iq port's iqk_dp4a, same source).
//   * __byte_perm -> s2eg_byte_perm: the hip_compat semantics (result byte i is byte s.nibble[i]&7 of {y:x}),
//
//     the iq port's iqk_byte_perm.
//   * __shfl_down_sync(0xffffffff, v, off) -> simd_shuffle_down (gr.metal's reduction idiom);
//     __shfl_down_sync(mask, v, off, 8) - the CPU-order path's width-8 segment shuffle - is emulated as a
//     simdgroup shuffle_down with the out-of-segment lanes restored to their own value, which is exactly
//     CUDA's width-w rule.
//   * __ballot_sync -> s2eg_ballot: every lane reads all 32 predicates through simd_shuffle and builds the
//     same mask (these kernels are one warp; the 32 shuffles are noise next to the dot products).
//   * __expf / expf -> metal::precise::exp (MSL has no plain exp under -fno-fast-math; both kernel
//     generations share the swiglu, so their bitwise contract is unaffected, and against the double host
//     reference the elementwise port measured this form at 3.4e-08 rel - two orders under the test's
//     4e-5-of-sum|term| tolerance).
//   * __float2half_rn -> f16_from_f32 (strata_port.metalh, bit for bit); __fmul_rn/__fadd_rn/__fsub_rn/
//     __fdiv_rn are plain * + - / (exact IEEE ops under -fno-fast-math -ffp-contract=off), __fmaf_rn is fma.
//   * load_x_chunk's word offset: CUDA derives it from the RUNTIME pointer ((uintptr_t)q & 3); MSL cannot
//     read a pointer's integer value, so the chunk PARITY carries it - the launcher's gate (4-byte aligned
//     activation rows) makes chunk c's int8 sit 34c+2 bytes into the row, i.e. 2 bytes into an aligned word
//     at even c and 0 at odd c, which is what the pointer would have said.  The gate is the CUDA file's own
//     (new_grouped/new_hit check the alignments and fall back to the previous kernels).
//
// RULE 9 (the big one): the grouped kernels' grp_ptr table holds raw device pointers IN DEVICE MEMORY, and
// such a pointer is not a usable device pointer on this GPU.  So gu_grouped/down_grouped(_t) take each
// group's BLOB as a bound [[buffer(0)]] argument and the group index as a scalar; the launcher reads the
// small table back to the host and launches once per group (iq_kernels.mm's native_expert_grouped is the
// working precedent).  The counts (n_groups, grp_start, ent_dst, ent_tok) stay device-resident and are read
// inside the kernels exactly where the CUDA ones read them.  group_resident_kernel WRITES that table - it
// receives the blob base's ADDRESS as a scalar integer instead (MSL cannot turn a bound pointer back into an
// integer), and the value it writes is the same one the CUDA kernel computes, the host pointer the caller
// passed - which is what the launcher's k.buf() resolves against the registry on the way back in.
#include "strata_port.metalh"

// ---------------------------------------------------------------- the blob's own geometry (expert.hpp's)
constant const int s2eg_H = 2560;
constant const int s2eg_FF = 640;
constant const int s2eg_QK = 64;                        // Q2_0's group: one fp16 scale per 64 weights
constant const int s2eg_ROW_GU = s2eg_H / 4;                 // 640 B of codes per gate/up row (2 bits per element)
constant const int s2eg_ROW_D = s2eg_FF / 4;                 // 160 B per down row
constant const int s2eg_SC_GU = s2eg_H / s2eg_QK;                 // 40 fp16 scales per gate/up row
constant const int s2eg_SC_D = s2eg_FF / s2eg_QK;                 // 10 per down row
constant const ulong s2eg_O_D_CODES = (ulong) 2 * s2eg_FF * s2eg_ROW_GU;
constant const ulong s2eg_O_GU_SCALES = s2eg_O_D_CODES + (ulong) s2eg_H * s2eg_ROW_D;
constant const ulong s2eg_O_D_SCALES = s2eg_O_GU_SCALES + (ulong) 2 * s2eg_FF * s2eg_SC_GU * 2;

constant const int s2eg_THREADS = 256;                  // every launch in the file is <<<..., 256>>>
constant const int s2eg_WPB = s2eg_THREADS / 32;             // warps per block, the .cu's blockDim.x >> 5

// CUDA's signed __dp4a: dp4a.hpp's own (bit-exact) sm_60 definition - four signed-byte products, wrapping
// int32.  Same source as the iq port's iqk_dp4a.
static inline int s2eg_dp4a(int a, int b, int c) {
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

// CUDA's __byte_perm (default mode), per this repo's HIP compat shim: result byte i is byte (s.nibble[i] & 7)
// of the eight bytes {y:x}, x the low word.
static inline uint s2eg_byte_perm(uint x, uint y, uint s) {
    uint r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint idx = (s >> (4 * i)) & 0x7u;
        const uint byte_ = idx < 4u ? (x >> (8 * idx)) & 0xFFu : (y >> (8 * (idx - 4u))) & 0xFFu;
        r |= byte_ << (8 * i);
    }
    return r;
}

// fp16 at any address, one byte at a time (the .cu's f16_at).  f32_from_f16 is exact.
static inline float s2eg_f16_at(constant const uint8_t* p) {
    return f32_from_f16((uint) p[0] | ((uint) p[1] << 8));
}
// the same value through one 16-bit pattern (the .cu's f16_ld; the scales sit at even addresses, but the
// bytes are composed all the same so no alignment is ever assumed).
static inline float s2eg_f16_ld(constant const uint8_t* p) {
    return f32_from_f16((uint) p[0] | ((uint) p[1] << 8));
}

// the .cu's warp_sum: the same shuffle-down butterfly, lane 0 holds the sum.
static inline float s2eg_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += simd_shuffle_down(v, (uint) off);
    return v;
}

// __shfl_down_sync(mask, v, delta, 8): CUDA's width-8 segment shuffle - lanes whose source leaves their
// 8-lane segment receive their OWN value.  A simdgroup shuffle_down covers 32 lanes, so the lanes that read
// into the NEXT segment (but still inside the simdgroup) are restored by hand; past the simdgroup's end the
// MSL op already returns the caller's own value.
static inline float s2eg_shfl_down_8(float v, uint delta, uint lane) {
    const float s = simd_shuffle_down(v, delta);
    return ((lane & 7u) + delta < 8u) ? s : v;
}

// __ballot_sync(0xffffffff, pred): every lane ends up holding the same 32-bit mask.  Built from 32 simd
// shuffles - uniform, divergence-free, and exactly the mask CUDA's op produces.
static inline uint s2eg_ballot(int pred) {
    uint hit = 0;
    for (uint l = 0; l < 32u; ++l)
        if (simd_shuffle(pred, l) != 0) hit |= 1u << l;
    return hit;
}

// one 4-byte little-endian word at any address - the .cu's `memcpy(&xw, p, 4)`, which exists precisely
// because the activation's int8 are never 4-byte aligned (a cast there faulted on the kernel's first run).
static inline int s2eg_word_at(constant const uint8_t* p) {
    return as_type<int>((uint) p[0] | ((uint) p[1] << 8) | ((uint) p[2] << 16) | ((uint) p[3] << 24));
}

// a chunk's 8 code bytes as a uint2 - byte-composed, so the grouped kernels (whose CUDA bodies read
// `*(const uint2*)` off blobs the launcher never alignment-gates) carry no vector-alignment assumption.
static inline uint2 s2eg_u2_at(constant const uint8_t* p) {
    uint2 v;
    v.x = (uint) p[0] | ((uint) p[1] << 8) | ((uint) p[2] << 16) | ((uint) p[3] << 24);
    v.y = (uint) p[4] | ((uint) p[5] << 8) | ((uint) p[6] << 16) | ((uint) p[7] << 24);
    return v;
}

// ---------------------------------------------------------------- the previous kernels' inner body

/// ONE S2 ROW AGAINST A Q8_0 ACTIVATION, WARP-WIDE (the .cu's row_dot_s2_q8, verbatim).  `x_scales` non-null
/// replaces the block's fp16 `d` with the fp32 activation scale (R4.2h); null keeps the fp16 behaviour.
static inline float s2eg_row_dot(constant const uint8_t* codes, constant const uint8_t* scales,
                                 constant const uint8_t* x_q8_0, int n_chunks, uint lane,
                                 constant const float* x_scales) {
    float acc = 0.0f;
    for (int c = (int) lane; c < n_chunks; c += 32) {
        constant const uint8_t* cb = codes + (ulong) c * 8;             // 8 code bytes = 32 elements
        constant const uint8_t* xb = x_q8_0 + (ulong) c * 34;           // one block_q8_0
        const float dx = x_scales != nullptr ? x_scales[c] : s2eg_f16_at(xb);
        constant const uint8_t* xq = xb + 2;

        // the packed form is LSB-first: element 4j+k is bits [2k, 2k+2) of code byte j; each code byte
        // becomes a word whose four bytes are its four 2-bit fields, then dp4a does four exact MACs.
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x - the weight-independent term, as ones * x
        const int ones = 0x01010101;
        for (int j = 0; j < 8; ++j) {
            const uint cbyte = cb[j];
            const int cw = as_type<int>((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                       (((cbyte >> 6) & 3u) << 24));
            const int xw = s2eg_word_at(xq + 4 * j);
            s = s2eg_dp4a(cw, xw, s);
            hx = s2eg_dp4a(ones, xw, hx);
        }
        // one weight scale per 64 elements, so per TWO 32-element chunks
        const float dw = s2eg_f16_at(scales + (ulong) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

// ---------------------------------------------------------------- the new kernels' helpers

// a chunk's two code words as the eight dp4a operands that pair with X[0..7] (the .cu's expand_codes):
// m[4h + f] byte b is the code of element 16h + 4b + f.
static inline void s2eg_expand_codes(uint2 cb, int m[8]) {
    const uint M = 0x03030303u;
    m[0] = as_type<int>(cb.x & M);
    m[1] = as_type<int>((cb.x >> 2) & M);
    m[2] = as_type<int>((cb.x >> 4) & M);
    m[3] = as_type<int>((cb.x >> 6) & M);
    m[4] = as_type<int>(cb.y & M);
    m[5] = as_type<int>((cb.y >> 2) & M);
    m[6] = as_type<int>((cb.y >> 4) & M);
    m[7] = as_type<int>((cb.y >> 6) & M);
}

/// The 32 int8 of the `block_q8_0` at `xb` regrouped into X[0..7], and their sum hx (the .cu's
/// load_x_chunk).  `word_off` is (uintptr_t)(xb + 2) & 3 in the CUDA original; MSL cannot read a pointer's
/// integer value, so the launcher's gate carries it: rows are 4-byte aligned (else the PREVIOUS kernels run),
/// which fixes chunk c's int8 at 34c + 2 - 2 bytes into an aligned word at even c, 0 at odd c.  The ninth
/// word is read only in the 2-byte case and then lies inside block c + 1 of the same row (both row lengths,
/// 80 and 20 chunks, are even, so an even chunk is never a row's last).
static inline int s2eg_load_x_chunk(constant const uint8_t* xb, uint word_off, int X[8]) {
    constant const uint8_t* q = xb + 2;
    constant const uint* p = reinterpret_cast<constant const uint*>(q - word_off);   // 4-aligned by the gate
    const uint sh = word_off * 8;
    uint v[9];
    for (int j = 0; j < 8; ++j) v[j] = p[j];
    v[8] = word_off != 0u ? p[8] : 0u;
    uint n[8];
    int hx = 0;
    for (int j = 0; j < 8; ++j) {
        // natural word j: x[4j .. 4j+3]; a 64-bit shift (sh is 0 or 16) rather than __funnelshift_r
        n[j] = (uint) ((((ulong) v[j + 1] << 32) | v[j]) >> sh);
        hx = s2eg_dp4a(0x01010101, as_type<int>(n[j]), hx);
    }
    for (int h = 0; h < 2; ++h) {
        const uint t0 = s2eg_byte_perm(n[4 * h], n[4 * h + 1], 0x5140);      // a0 b0 a1 b1
        const uint t1 = s2eg_byte_perm(n[4 * h], n[4 * h + 1], 0x7362);      // a2 b2 a3 b3
        const uint t2 = s2eg_byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);  // c0 d0 c1 d1
        const uint t3 = s2eg_byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);  // c2 d2 c3 d3
        X[4 * h + 0] = as_type<int>(s2eg_byte_perm(t0, t2, 0x5410));         // a0 b0 c0 d0
        X[4 * h + 1] = as_type<int>(s2eg_byte_perm(t0, t2, 0x7632));         // a1 b1 c1 d1
        X[4 * h + 2] = as_type<int>(s2eg_byte_perm(t1, t3, 0x5410));         // a2 b2 c2 d2
        X[4 * h + 3] = as_type<int>(s2eg_byte_perm(t1, t3, 0x7632));         // a3 b3 c3 d3
    }
    return hx;
}

// `s` for one chunk: the eight expanded code words against the eight regrouped activation words.
static inline int s2eg_chunk_s(const int m[8], const int X[8]) {
    int s = 0;
    for (int j = 0; j < 8; ++j) s = s2eg_dp4a(m[j], X[j], s);
    return s;
}

// one activation chunk's contribution for the grouped kernels' first staging (the .cu's chunk_dot): the
// same dp4a sequence and float expression as s2eg_row_dot's inner body, bitwise.  The activations come from
// THREADGROUP memory (the group's staging), so that is the address space the words arrive in.
static inline float s2eg_chunk_dot(uint2 cb, threadgroup const int* xw, float dw, float dx) {
    const thread uint8_t* cbytes = reinterpret_cast<const thread uint8_t*>(&cb);
    int s = 0, hx = 0;
    const int ones = 0x01010101;
    for (int j = 0; j < 8; ++j) {
        const uint cbyte = cbytes[j];
        const int cw = as_type<int>((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                   (((cbyte >> 6) & 3u) << 24));
        s = s2eg_dp4a(cw, xw[j], s);
        hx = s2eg_dp4a(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

// ---------------------------------------------------------------- the previous per-hit kernels

/// GATE AND UP, ONE WARP PER ROW.  Slot i is DECODED from row-slot i (even = gate r, odd = up r - the rows
/// are interleaved in the blob, and the first version of this kernel got that wrong; moe_hit_parity caught
/// it at worst rel 2.2e+03) and WRITTEN to the gate-major output slot its parity says.  Grid:
/// ceil(n_hits * 2FF / warps) x 256; `bg` is the CUDA blockIdx.x, `t` the threadIdx.x.
kernel void gu_kernel(constant const uint8_t* blob_base [[buffer(0)]],
                      constant const int* slot_index [[buffer(1)]],
                      constant const long& blob_bytes [[buffer(2)]],
                      constant const uint8_t* x_q8_0 [[buffer(3)]],
                      constant const float* x_scales [[buffer(4)]],
                      device float* gate_up [[buffer(5)]],
                      constant const int& n_hits [[buffer(6)]],
                      constant const int* d_count [[buffer(7)]],
                      constant const int* dst_index [[buffer(8)]],
                      constant const int& tok_div [[buffer(9)]],
                      uint bg [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
    const long slot = (long) bg * s2eg_WPB + (long) (t >> 5);
    const long rows_per_hit = 2L * s2eg_FF;
    const long total = (long) n_hits * rows_per_hit;
    if (slot >= total) return;
    const int h = (int) (slot / rows_per_hit);
    if (d_count != nullptr && h >= d_count[0]) return;     // token graph: capacity layout, device count
    const int i = (int) (slot % rows_per_hit);
    const uint lane = t & 31u;

    constant const uint8_t* blob = blob_base + (ulong) slot_index[h] * (ulong) blob_bytes;
    constant const uint8_t* xq = x_q8_0;
    constant const float* xs = x_scales;
    if (tok_div > 0) {   // plan v0.3 P6 verify window: each hit reads its own token's activation
        const int tok = dst_index[h] / tok_div;
        xq += (ulong) tok * (ulong) (s2eg_H / 32) * 34;
        if (xs != nullptr) xs += (ulong) tok * (ulong) (s2eg_H / 32);
    }
    const float acc = s2eg_row_dot(blob + (ulong) i * s2eg_ROW_GU, blob + s2eg_O_GU_SCALES + (ulong) i * s2eg_SC_GU * 2,
                                   xq, s2eg_H / 32, lane, xs);
    const float s = s2eg_warp_sum(acc);
    if (lane != 0u) return;
    // gate-major: every hit's gate rows contiguous from 0, every hit's up rows from n_hits * s2eg_FF - the
    // swiglu and the quantizer walk contiguous ranges, so the layout is what makes the shared steps work.
    const int r = i >> 1;
    const ulong base = (i & 1) ? ((ulong) n_hits * s2eg_FF + (ulong) h * s2eg_FF) : ((ulong) h * s2eg_FF);
    gate_up[base + (ulong) r] = s;
}

/// `silu(gate) * up`, in place, over the GATE-MAJOR buffer (hit h's row r meets itself at h*s2eg_FF + r).  SiLU
/// on the GATE - docs/semantics.md's reading.  __expf is metal::precise::exp here (file comment); both
/// kernel generations share this kernel, so their bitwise contract does not depend on the spelling.
kernel void swiglu_kernel(device float* gate_up [[buffer(0)]],
                          constant const long& n_pairs [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if ((long) i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + metal::precise::exp(-g))) * u;
}

/// DOWN, ONE WARP PER ROW.  `dst_index[h]` is the router's slot, not h - the two halves' meeting rule the
/// header documents.
kernel void down_kernel(constant const uint8_t* blob_base [[buffer(0)]],
                        constant const int* slot_index [[buffer(1)]],
                        constant const int* dst_index [[buffer(2)]],
                        constant const long& blob_bytes [[buffer(3)]],
                        constant const uint8_t* h_q8_0 [[buffer(4)]],
                        constant const float* h_scales [[buffer(5)]],
                        device float* out [[buffer(6)]],
                        constant const int& n_hits [[buffer(7)]],
                        constant const int* d_count [[buffer(8)]],
                        uint bg [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
    const long row = (long) bg * s2eg_WPB + (long) (t >> 5);
    const long total = (long) n_hits * s2eg_H;
    if (row >= total) return;
    const int h = (int) (row / s2eg_H);
    if (d_count != nullptr && h >= d_count[0]) return;
    const int r = (int) (row % s2eg_H);
    const uint lane = t & 31u;

    constant const uint8_t* blob = blob_base + (ulong) slot_index[h] * (ulong) blob_bytes;
    constant const uint8_t* xb = h_q8_0 + (ulong) h * (ulong) (s2eg_FF / 32) * 34;
    const float acc = s2eg_row_dot(blob + s2eg_O_D_CODES + (ulong) r * s2eg_ROW_D,
                                   blob + s2eg_O_D_SCALES + (ulong) r * s2eg_SC_D * 2, xb, s2eg_FF / 32, lane,
                                   h_scales != nullptr ? h_scales + (ulong) h * (ulong) (s2eg_FF / 32) : nullptr);
    const float s = s2eg_warp_sum(acc);
    if (lane == 0u) out[(ulong) dst_index[h] * s2eg_H + (ulong) r] = s;
}

// ---------------------------------------------------------------- the new per-hit kernels (pairs)

/// `gu_kernel`, new: ONE WARP PER (gate, up) PAIR - row-slots 2r and 2r+1, adjacent in the blob - so each
/// activation chunk is loaded and regrouped once for both rows.  The uint2 code loads are the reason the
/// launcher gates on an 8-byte aligned arena and slot size (the .cu's own contract; otherwise the previous
/// kernels run).
kernel void gu_pair_kernel(constant const uint8_t* blob_base [[buffer(0)]],
                           constant const int* slot_index [[buffer(1)]],
                           constant const long& blob_bytes [[buffer(2)]],
                           constant const uint8_t* x_q8_0 [[buffer(3)]],
                           constant const float* x_scales [[buffer(4)]],
                           device float* gate_up [[buffer(5)]],
                           constant const int& n_hits [[buffer(6)]],
                           constant const int* d_count [[buffer(7)]],
                           constant const int* dst_index [[buffer(8)]],
                           constant const int& tok_div [[buffer(9)]],
                           uint bg [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]]) {
    const long pair = (long) bg * s2eg_WPB + (long) (t >> 5);
    const long total = (long) n_hits * s2eg_FF;
    if (pair >= total) return;
    const int h = (int) (pair / s2eg_FF);
    if (d_count != nullptr && h >= d_count[0]) return;
    const int r = (int) (pair % s2eg_FF);
    const uint lane = t & 31u;

    constant const uint8_t* blob = blob_base + (ulong) slot_index[h] * (ulong) blob_bytes;
    constant const uint8_t* xq = x_q8_0;
    constant const float* xs = x_scales;
    if (tok_div > 0) {
        const int tok = dst_index[h] / tok_div;
        xq += (ulong) tok * (ulong) (s2eg_H / 32) * 34;
        if (xs != nullptr) xs += (ulong) tok * (ulong) (s2eg_H / 32);
    }
    constant const uint8_t* codes = blob + (ulong) (2 * r) * s2eg_ROW_GU;             // gate row; the up row follows
    constant const uint8_t* scales = blob + s2eg_O_GU_SCALES + (ulong) (2 * r) * s2eg_SC_GU * 2;
    float acc_g = 0.0f, acc_u = 0.0f;
    for (int c = (int) lane; c < s2eg_H / 32; c += 32) {
        constant const uint8_t* xb = xq + (ulong) c * 34;
        const float dx = xs != nullptr ? xs[c] : s2eg_f16_ld(xb);
        const uint woff = (c & 1) ? 0u : 2u;        // load_x_chunk's word offset, from the gate (file comment)
        int X[8], m[8];
        const int hx = s2eg_load_x_chunk(xb, woff, X);
        s2eg_expand_codes(s2eg_u2_at(codes + (ulong) c * 8), m);
        const float dw_g = s2eg_f16_ld(scales + (ulong) (c >> 1) * 2);
        acc_g += dw_g * dx * (float) (s2eg_chunk_s(m, X) - hx);
        s2eg_expand_codes(s2eg_u2_at(codes + s2eg_ROW_GU + (ulong) c * 8), m);
        const float dw_u = s2eg_f16_ld(scales + s2eg_SC_GU * 2 + (ulong) (c >> 1) * 2);
        acc_u += dw_u * dx * (float) (s2eg_chunk_s(m, X) - hx);
    }
    const float sg = s2eg_warp_sum(acc_g);
    const float su = s2eg_warp_sum(acc_u);
    if (lane != 0u) return;
    gate_up[(ulong) h * s2eg_FF + (ulong) r] = sg;                                // gate-major, as in `gu_kernel`
    gate_up[(ulong) n_hits * s2eg_FF + (ulong) h * s2eg_FF + (ulong) r] = su;
}

/// `down_kernel`, new: ONE WARP PER PAIR OF ROWS r, r + 1 of one hit, the intermediate's chunk loaded once.
kernel void down_pair_kernel(constant const uint8_t* blob_base [[buffer(0)]],
                             constant const int* slot_index [[buffer(1)]],
                             constant const int* dst_index [[buffer(2)]],
                             constant const long& blob_bytes [[buffer(3)]],
                             constant const uint8_t* h_q8_0 [[buffer(4)]],
                             constant const float* h_scales [[buffer(5)]],
                             device float* out [[buffer(6)]],
                             constant const int& n_hits [[buffer(7)]],
                             constant const int* d_count [[buffer(8)]],
                             uint bg [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]]) {
    const long pair = (long) bg * s2eg_WPB + (long) (t >> 5);
    const long total = (long) n_hits * (s2eg_H / 2);
    if (pair >= total) return;
    const int h = (int) (pair / (s2eg_H / 2));
    if (d_count != nullptr && h >= d_count[0]) return;
    const int r = 2 * (int) (pair % (s2eg_H / 2));
    const uint lane = t & 31u;

    constant const uint8_t* blob = blob_base + (ulong) slot_index[h] * (ulong) blob_bytes;
    constant const uint8_t* xrow = h_q8_0 + (ulong) h * (ulong) (s2eg_FF / 32) * 34;
    constant const float* xs = h_scales != nullptr ? h_scales + (ulong) h * (ulong) (s2eg_FF / 32) : nullptr;
    constant const uint8_t* codes = blob + s2eg_O_D_CODES + (ulong) r * s2eg_ROW_D;
    constant const uint8_t* scales = blob + s2eg_O_D_SCALES + (ulong) r * s2eg_SC_D * 2;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int c = (int) lane; c < s2eg_FF / 32; c += 32) {
        constant const uint8_t* xb = xrow + (ulong) c * 34;
        const float dx = xs != nullptr ? xs[c] : s2eg_f16_ld(xb);
        const uint woff = (c & 1) ? 0u : 2u;
        int X[8], m[8];
        const int hx = s2eg_load_x_chunk(xb, woff, X);
        s2eg_expand_codes(s2eg_u2_at(codes + (ulong) c * 8), m);
        const float dw0 = s2eg_f16_ld(scales + (ulong) (c >> 1) * 2);
        acc0 += dw0 * dx * (float) (s2eg_chunk_s(m, X) - hx);
        s2eg_expand_codes(s2eg_u2_at(codes + s2eg_ROW_D + (ulong) c * 8), m);
        const float dw1 = s2eg_f16_ld(scales + s2eg_SC_D * 2 + (ulong) (c >> 1) * 2);
        acc1 += dw1 * dx * (float) (s2eg_chunk_s(m, X) - hx);
    }
    const float s0 = s2eg_warp_sum(acc0);
    const float s1 = s2eg_warp_sum(acc1);
    if (lane != 0u) return;
    const ulong o = (ulong) dst_index[h] * s2eg_H + (ulong) r;
    out[o] = s0;
    out[o + 1] = s1;
}

// ---------------------------------------------------------------- the CPU-order experimental path

// The CPU subtracts the weight bias after its eight FMA accumulators have been reduced; the correction is
// computed once per chunk here, as in the .cu.
kernel void activation_correction_kernel(constant const uint8_t* q8 [[buffer(0)]],
                                         constant const float* scales [[buffer(1)]],
                                         device float* hx [[buffer(2)]],
                                         constant const int& chunks [[buffer(3)]],
                                         uint c [[thread_position_in_grid]]) {
    if ((int) c >= chunks) return;
    constant const int8_t* q = reinterpret_cast<constant const int8_t*>(q8 + (ulong) c * 34 + 2);
    int sum = 0;
    for (int j = 0; j < 32; ++j) sum += (int) q[j];
    hx[c] = scales[c] * (float) sum;
}

static inline int s2eg_dot4(constant const uint8_t* codes, constant const int8_t* q) {
    const uint c = codes[0];
    const int cw = as_type<int>((c & 3u) | (((c >> 2) & 3u) << 8) | (((c >> 4) & 3u) << 16) |
                               (((c >> 6) & 3u) << 24));
    return s2eg_dp4a(cw, s2eg_word_at(reinterpret_cast<constant const uint8_t*>(q)), 0);
}

// the .cu's row_dot_cpu_order: the CPU VNNI lane order, FMA accumulation, separate correction, and the
// _mm_hadd_ps pair order (4, 1, 2) - a standard 4,2,1 tree is a different float expression.
static inline float s2eg_row_dot_cpu_order(constant const uint8_t* codes, constant const uint8_t* scales,
                                           constant const uint8_t* xq, constant const float* xs,
                                           constant const float* hx, int blocks, uint lane) {
    float acc = 0.0f;
    float corr = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float d = s2eg_f16_at(scales + 2 * b);
        const int lo = s2eg_dot4(codes + b * 16 + (int) lane,
                                 reinterpret_cast<constant const int8_t*>(xq + (ulong) (2 * b) * 34 + 2) +
                                     (int) lane * 4);
        const int hi = s2eg_dot4(codes + b * 16 + 8 + (int) lane,
                                 reinterpret_cast<constant const int8_t*>(xq + (ulong) (2 * b + 1) * 34 + 2) +
                                     (int) lane * 4);
        acc = fma(d * xs[2 * b], (float) lo, acc);          // __fmaf_rn(__fmul_rn(d, xs), lo, acc)
        acc = fma(d * xs[2 * b + 1], (float) hi, acc);
        if (lane == 0u)
            corr = corr + d * (hx[2 * b] + hx[2 * b + 1]);  // __fadd_rn(corr, __fmul_rn(d, __fadd_rn))
    }
    // _mm_add_ps(low128, high128), then two _mm_hadd_ps: the width-8 shuffle in the order 4, 1, 2
    acc = acc + s2eg_shfl_down_8(acc, 4u, lane);
    acc = acc + s2eg_shfl_down_8(acc, 1u, lane);
    acc = acc + s2eg_shfl_down_8(acc, 2u, lane);
    return acc - corr;   // __fsub_rn; only lane zero is consumed
}

// the .cu's cpu_order_projection_kernel<DOWN>, both instantiations spelled out (template kernels become
// concrete ones on this port; the launcher names carry the argument).  Eight lanes per row, the CPU's
// accumulator lane count.
kernel void cpu_order_projection_kernel_false(constant const uint8_t* blob_base [[buffer(0)]],
                                              constant const int* slots [[buffer(1)]],
                                              constant const int* destinations [[buffer(2)]],
                                              constant const long& blob_bytes [[buffer(3)]],
                                              constant const uint8_t* xq [[buffer(4)]],
                                              constant const float* xs [[buffer(5)]],
                                              constant const float* hx [[buffer(6)]],
                                              device float* out [[buffer(7)]],
                                              constant const int& n_hits [[buffer(8)]],
                                              uint bg [[threadgroup_position_in_grid]],
                                              uint t [[thread_index_in_threadgroup]]) {
    constexpr int rows_per_hit = 2 * s2eg_FF;   // DOWN = false
    const int row = (int) bg * (s2eg_THREADS / 8) + (int) (t / 8u);
    if (row >= n_hits * rows_per_hit) return;
    const int h = row / rows_per_hit;
    const int r = row % rows_per_hit;
    const uint lane = t & 7u;
    constant const uint8_t* blob = blob_base + (ulong) slots[h] * (ulong) blob_bytes;
    constant const uint8_t* codes = blob + (ulong) r * s2eg_ROW_GU;
    constant const uint8_t* scales = blob + s2eg_O_GU_SCALES + (ulong) r * s2eg_SC_GU * 2;
    const float value = s2eg_row_dot_cpu_order(codes, scales, xq, xs, hx, s2eg_SC_GU, lane);
    if (lane != 0u) return;
    out[((r & 1) ? (ulong) n_hits * s2eg_FF : 0ul) + (ulong) h * s2eg_FF + (ulong) (r >> 1)] = value;
}

kernel void cpu_order_projection_kernel_true(constant const uint8_t* blob_base [[buffer(0)]],
                                             constant const int* slots [[buffer(1)]],
                                             constant const int* destinations [[buffer(2)]],
                                             constant const long& blob_bytes [[buffer(3)]],
                                             constant const uint8_t* xq [[buffer(4)]],
                                             constant const float* xs [[buffer(5)]],
                                             constant const float* hx [[buffer(6)]],
                                             device float* out [[buffer(7)]],
                                             constant const int& n_hits [[buffer(8)]],
                                             uint bg [[threadgroup_position_in_grid]],
                                             uint t [[thread_index_in_threadgroup]]) {
    constexpr int rows_per_hit = s2eg_H;        // DOWN = true
    const int row = (int) bg * (s2eg_THREADS / 8) + (int) (t / 8u);
    if (row >= n_hits * rows_per_hit) return;
    const int h = row / rows_per_hit;
    const int r = row % rows_per_hit;
    const uint lane = t & 7u;
    constant const uint8_t* blob = blob_base + (ulong) slots[h] * (ulong) blob_bytes;
    constant const uint8_t* codes = blob + s2eg_O_D_CODES + (ulong) r * s2eg_ROW_D;
    constant const uint8_t* scales = blob + s2eg_O_D_SCALES + (ulong) r * s2eg_SC_D * 2;
    const float value = s2eg_row_dot_cpu_order(codes, scales,
                                               xq + (ulong) h * (s2eg_FF / 32) * 34, xs + h * (s2eg_FF / 32),
                                               hx + h * (s2eg_FF / 32), s2eg_SC_D, lane);
    if (lane != 0u) return;
    out[(ulong) destinations[h] * s2eg_H + (ulong) r] = value;
}

// Accurate fp32 exponential (the .cu uses expf, not __expf, here on purpose); metal::precise::exp is this
// port's spelling (file comment).
kernel void cpu_order_swiglu_kernel(device float* gu [[buffer(0)]],
                                    constant const int& pairs [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
    if ((int) i >= pairs) return;
    const float g = gu[i];
    const float eg = metal::precise::exp(-g);
    gu[i] = (g / (1.0f + eg)) * gu[pairs + i];
}

// the intermediate's CPU-contract quantizer: round-half-away int8 codes, fp16 `d` in the block, fp32 scale
// and correction carried alongside (all __f*__rn are plain ops under this build's flags; __float2half_rn
// is f16_from_f32, bit for bit).
kernel void cpu_order_quantize_kernel(constant const float* x [[buffer(0)]],
                                      device uint8_t* blocks [[buffer(1)]],
                                      device float* scales [[buffer(2)]],
                                      device float* hx [[buffer(3)]],
                                      constant const int& chunks [[buffer(4)]],
                                      uint c [[thread_position_in_grid]]) {
    if ((int) c >= chunks) return;
    constant const float* xb = x + (ulong) c * 32;
    device uint8_t* out = blocks + (ulong) c * 34;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = metal::precise::fmax(amax, metal::precise::fabs(xb[j]));
    const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    scales[c] = s;
    const uint bits = f16_from_f32(s);
    out[0] = (uint8_t) bits;
    out[1] = (uint8_t) (bits >> 8);
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
        const float tt = xb[j] * inv;
        int v = (int) (tt + (tt >= 0.0f ? 0.5f : -0.5f));
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + j] = (uint8_t) (int8_t) v;
        sum += v;
    }
    hx[c] = s * (float) sum;
}

// ---------------------------------------------------------------- the token-graph helpers

// which of this layer's routed experts are resident, decided ON THE DEVICE from the static residency row.
// One warp; k <= 32; the ballot compacts the hits in routing order.
kernel void hit_select_kernel(constant const int* ids [[buffer(0)]],
                              constant const int* res_row [[buffer(1)]],
                              constant const int& k [[buffer(2)]],
                              constant const int& n_expert [[buffer(3)]],
                              device int* slot [[buffer(4)]],
                              device int* dst [[buffer(5)]],
                              device int* count [[buffer(6)]],
                              uint lane [[thread_index_in_threadgroup]]) {
    int s = -1;
    if ((int) lane < k) {
        const int e = ids[lane];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const uint hit = s2eg_ballot(s >= 0 ? 1 : 0);
    if (s >= 0) {
        const int at = (int) popcount(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = (int) lane;
    }
    if (lane == 0u) *count = (int) popcount(hit);
}

// the same for up to 128 routed entries (a verify window of T tokens x k): four warps, ballots compacted
// in entry order through a threadgroup warp-count prefix.
kernel void hit_select_multi_kernel(constant const int* ids [[buffer(0)]],
                                    constant const int* res_row [[buffer(1)]],
                                    constant const int& n [[buffer(2)]],
                                    constant const int& n_expert [[buffer(3)]],
                                    device int* slot [[buffer(4)]],
                                    device int* dst [[buffer(5)]],
                                    device int* count [[buffer(6)]],
                                    uint t [[thread_index_in_threadgroup]]) {
    threadgroup int warp_count[4];
    const int i = (int) t, lane = (int) (t & 31u), warp = (int) (t >> 5);
    int s = -1;
    if (i < n) {
        const int e = ids[i];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const uint hit = s2eg_ballot(s >= 0 ? 1 : 0);
    if (lane == 0) warp_count[warp] = (int) popcount(hit);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int before = 0;
    for (int w = 0; w < warp; ++w) before += warp_count[w];
    if (s >= 0) {
        const int at = before + (int) popcount(hit & ((1u << (uint) lane) - 1u));
        slot[at] = s;
        dst[at] = i;
    }
    if (i == 0) *count = warp_count[0] + warp_count[1] + warp_count[2] + warp_count[3];
}

// each hit's row of `hit_out` added into `parts` (rows the CPU left at zero).  grid.y is the hit, a DATA
// index - so the group position carries it (R6); the x stride is the launcher's grid width.
kernel void add_hits_kernel(device float* parts [[buffer(0)]],
                            constant const float* hit_out [[buffer(1)]],
                            constant const int* dst [[buffer(2)]],
                            constant const int* count [[buffer(3)]],
                            constant const int& n_embd [[buffer(4)]],
                            constant const int& grid_x [[buffer(5)]],
                            uint2 gpos [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
    const int h = (int) gpos.y;
    if (h >= count[0]) return;
    const ulong row = (ulong) dst[h] * (ulong) n_embd;
    for (int i = (int) gpos.x * s2eg_THREADS + (int) t; i < n_embd; i += grid_x * s2eg_THREADS)
        parts[row + (ulong) i] += hit_out[row + (ulong) i];
}

// ---------------------------------------------------------------- the grouped kernels (verify window)

constant const int s2eg_GU_CHUNKS = (s2eg_H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constant const int s2eg_GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
constant const int s2eg_GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constant const int s2eg_D_ROWS = 64;                       // down rows per block: 8 per warp

// Gate/up: a block = s2eg_GU_ROWS rows of ONE group (RULE 9: the blob arrives bound, `g` as a scalar - the file
// comment).  The group's activations are staged once into threadgroup memory as words; each warp then walks
// its rows, loading each lane's code chunks once and dotting them with every entry.  23 KB of threadgroup
// memory (this GPU's measured limit is 32 KB).
kernel void gu_grouped_kernel(constant const uint8_t* blob [[buffer(0)]],
                              constant const int* grp_start [[buffer(1)]],
                              constant const int* n_groups [[buffer(2)]],
                              constant const int* ent_tok [[buffer(3)]],
                              constant const uint8_t* x_q8_0 [[buffer(4)]],
                              constant const float* x_scales [[buffer(5)]],
                              device float* gate_up [[buffer(6)]],
                              constant const int& cap_entries [[buffer(7)]],
                              constant const int& g [[buffer(8)]],
                              uint bg [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]]) {
    threadgroup int xs_q[s2eg_GMAX][s2eg_H / 4];          // the entries' int8 activations as words (2560 B each)
    threadgroup float xs_d[s2eg_GMAX][s2eg_H / 32];
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, s2eg_GMAX);
    const int lane = (int) (t & 31u), warp = (int) (t >> 5);
    for (int i = (int) t; i < ne * (s2eg_H / 32); i += s2eg_THREADS) {
        const int k = i / (s2eg_H / 32), c = i - k * (s2eg_H / 32);
        constant const uint8_t* xb =
            x_q8_0 + (ulong) ent_tok[e0 + k] * (ulong) (s2eg_H / 32) * 34 + (ulong) c * 34;
        xs_d[k][c] = x_scales != nullptr ? x_scales[(ulong) ent_tok[e0 + k] * (s2eg_H / 32) + c] : s2eg_f16_at(xb);
        constant const uint8_t* q = xb + 2;
        for (int w = 0; w < 8; ++w) xs_q[k][c * 8 + w] = s2eg_word_at(q + 4 * w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row0 = (int) bg * s2eg_GU_ROWS;
    for (int rr = warp; rr < s2eg_GU_ROWS; rr += 8) {
        const int i = row0 + rr;
        constant const uint8_t* codes = blob + (ulong) i * s2eg_ROW_GU;
        constant const uint8_t* scales = blob + s2eg_O_GU_SCALES + (ulong) i * s2eg_SC_GU * 2;
        uint2 cb[s2eg_GU_CHUNKS];
        float dw[s2eg_GU_CHUNKS];
        for (int q = 0; q < s2eg_GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < s2eg_H / 32) {
                cb[q] = s2eg_u2_at(codes + (ulong) c * 8);
                dw[q] = s2eg_f16_at(scales + (ulong) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
            for (int q = 0; q < s2eg_GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= s2eg_H / 32) break;
                acc += s2eg_chunk_dot(cb[q], &xs_q[k][c * 8], dw[q], xs_d[k][c]);
            }
            const float sum = s2eg_warp_sum(acc);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                const ulong base = (i & 1) ? ((ulong) cap_entries * s2eg_FF + (ulong) e * s2eg_FF) : ((ulong) e * s2eg_FF);
                gate_up[base + (ulong) r] = sum;
            }
        }
    }
}

// Down: a block = s2eg_D_ROWS rows of ONE group; the entries' quantized intermediates staged once.  A down row
// is 20 chunks, so lanes 0..19 each hold one chunk, as in the per-entry kernel.
kernel void down_grouped_kernel(constant const uint8_t* blob [[buffer(0)]],
                                constant const int* grp_start [[buffer(1)]],
                                constant const int* n_groups [[buffer(2)]],
                                constant const int* ent_dst [[buffer(3)]],
                                constant const uint8_t* h_q8_0 [[buffer(4)]],
                                constant const float* h_scales [[buffer(5)]],
                                device float* out [[buffer(6)]],
                                constant const int& g [[buffer(7)]],
                                uint bg [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_threadgroup]]) {
    threadgroup int hs_q[s2eg_GMAX][s2eg_FF / 4];
    threadgroup float hs_d[s2eg_GMAX][s2eg_FF / 32];
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, s2eg_GMAX);
    const int lane = (int) (t & 31u), warp = (int) (t >> 5);
    for (int i = (int) t; i < ne * (s2eg_FF / 32); i += s2eg_THREADS) {
        const int k = i / (s2eg_FF / 32), c = i - k * (s2eg_FF / 32);
        constant const uint8_t* xb = h_q8_0 + (ulong) (e0 + k) * (ulong) (s2eg_FF / 32) * 34 + (ulong) c * 34;
        hs_d[k][c] = h_scales != nullptr ? h_scales[(ulong) (e0 + k) * (s2eg_FF / 32) + c] : s2eg_f16_at(xb);
        constant const uint8_t* q = xb + 2;
        for (int w = 0; w < 8; ++w) hs_q[k][c * 8 + w] = s2eg_word_at(q + 4 * w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row0 = (int) bg * s2eg_D_ROWS;
    for (int rr = warp; rr < s2eg_D_ROWS; rr += 8) {
        const int r = row0 + rr;
        constant const uint8_t* codes = blob + s2eg_O_D_CODES + (ulong) r * s2eg_ROW_D;
        constant const uint8_t* scales = blob + s2eg_O_D_SCALES + (ulong) r * s2eg_SC_D * 2;
        const int c = lane;
        uint2 cb = uint2(0, 0);
        float dw = 0.0f;
        if (c < s2eg_FF / 32) {
            cb = s2eg_u2_at(codes + (ulong) c * 8);
            dw = s2eg_f16_at(scales + (ulong) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
            if (c < s2eg_FF / 32) acc += s2eg_chunk_dot(cb, &hs_q[k][c * 8], dw, hs_d[k][c]);
            const float sum = s2eg_warp_sum(acc);
            if (lane == 0) out[(ulong) ent_dst[e0 + k] * s2eg_H + (ulong) r] = sum;
        }
    }
}

// ---- the grouped kernels with the staged activations laid out for the reads (the .cu's own fix for the
// 8-way bank conflict): word-major staging, lanes read consecutive words, (dx, hx) one 64-bit read, and
// each warp takes its rows TWO at a time so every shared read serves both.  25.6 KB of threadgroup memory.

// stages chunk i = k * NC + c of a group: its regrouped words at xs_w[j * STRIDE + i] and (dx, hx) at
// xs_dh[i] (the .cu's stage_chunk<STRIDE>).
static inline void s2eg_stage_chunk(constant const uint8_t* xb, float dx, uint word_off, int stride,
                                    threadgroup int* xs_w, threadgroup int2* xs_dh, int i) {
    int X[8];
    const int hx = s2eg_load_x_chunk(xb, word_off, X);
    for (int j = 0; j < 8; ++j) xs_w[j * stride + i] = X[j];
    xs_dh[i] = int2(as_type<int>(dx), hx);
}

kernel void gu_grouped_t_kernel(constant const uint8_t* blob [[buffer(0)]],
                                constant const int* grp_start [[buffer(1)]],
                                constant const int* n_groups [[buffer(2)]],
                                constant const int* ent_tok [[buffer(3)]],
                                constant const uint8_t* x_q8_0 [[buffer(4)]],
                                constant const float* x_scales [[buffer(5)]],
                                device float* gate_up [[buffer(6)]],
                                constant const int& cap_entries [[buffer(7)]],
                                constant const int& g [[buffer(8)]],
                                uint bg [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_threadgroup]]) {
    constexpr int NC = s2eg_H / 32;
    threadgroup int xs_w[8 * s2eg_GMAX * NC];          // 20 KB: word j of entry k's chunk c at [j][k * NC + c]
    threadgroup int2 xs_dh[s2eg_GMAX * NC];            // 5 KB: (dx as bits, hx)
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, s2eg_GMAX);
    const int lane = (int) (t & 31u), warp = (int) (t >> 5);
    for (int i = (int) t; i < ne * NC; i += s2eg_THREADS) {
        const int k = i / NC, c = i - k * NC;
        const int tok = ent_tok[e0 + k];
        constant const uint8_t* xb = x_q8_0 + (ulong) tok * (ulong) NC * 34 + (ulong) c * 34;
        const float dx = x_scales != nullptr ? x_scales[(ulong) tok * NC + c] : s2eg_f16_ld(xb);
        s2eg_stage_chunk(xb, dx, (c & 1) ? 0u : 2u, s2eg_GMAX * NC, xs_w, xs_dh, i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row0 = (int) bg * s2eg_GU_ROWS;                 // even: a pair is always (gate r, up r)
    for (int pp = warp; pp < s2eg_GU_ROWS / 2; pp += 8) {
        const int i = row0 + 2 * pp;
        constant const uint8_t* codes = blob + (ulong) i * s2eg_ROW_GU;
        constant const uint8_t* scales = blob + s2eg_O_GU_SCALES + (ulong) i * s2eg_SC_GU * 2;
        int m0[s2eg_GU_CHUNKS][8], m1[s2eg_GU_CHUNKS][8];
        float dw0[s2eg_GU_CHUNKS], dw1[s2eg_GU_CHUNKS];
        for (int q = 0; q < s2eg_GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < NC) {
                s2eg_expand_codes(s2eg_u2_at(codes + (ulong) c * 8), m0[q]);
                s2eg_expand_codes(s2eg_u2_at(codes + s2eg_ROW_GU + (ulong) c * 8), m1[q]);
                dw0[q] = s2eg_f16_ld(scales + (ulong) (c >> 1) * 2);
                dw1[q] = s2eg_f16_ld(scales + s2eg_SC_GU * 2 + (ulong) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int q = 0; q < s2eg_GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= NC) break;
                const int at = k * NC + c;
                int X[8];
                for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (s2eg_GMAX * NC) + at];
                const int2 dh = xs_dh[at];
                const float dx = as_type<float>(dh.x);
                acc0 += dw0[q] * dx * (float) (s2eg_chunk_s(m0[q], X) - dh.y);
                acc1 += dw1[q] * dx * (float) (s2eg_chunk_s(m1[q], X) - dh.y);
            }
            const float s0 = s2eg_warp_sum(acc0);
            const float s1 = s2eg_warp_sum(acc1);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                gate_up[(ulong) e * s2eg_FF + (ulong) r] = s0;
                gate_up[(ulong) cap_entries * s2eg_FF + (ulong) e * s2eg_FF + (ulong) r] = s1;
            }
        }
    }
}

kernel void down_grouped_t_kernel(constant const uint8_t* blob [[buffer(0)]],
                                  constant const int* grp_start [[buffer(1)]],
                                  constant const int* n_groups [[buffer(2)]],
                                  constant const int* ent_dst [[buffer(3)]],
                                  constant const uint8_t* h_q8_0 [[buffer(4)]],
                                  constant const float* h_scales [[buffer(5)]],
                                  device float* out [[buffer(6)]],
                                  constant const int& g [[buffer(7)]],
                                  uint bg [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]]) {
    constexpr int NC = s2eg_FF / 32;
    threadgroup int hs_w[8 * s2eg_GMAX * NC];
    threadgroup int2 hs_dh[s2eg_GMAX * NC];
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, s2eg_GMAX);
    const int lane = (int) (t & 31u), warp = (int) (t >> 5);
    for (int i = (int) t; i < ne * NC; i += s2eg_THREADS) {
        const int k = i / NC, c = i - k * NC;
        constant const uint8_t* xb = h_q8_0 + (ulong) (e0 + k) * (ulong) NC * 34 + (ulong) c * 34;
        const float dx = h_scales != nullptr ? h_scales[(ulong) (e0 + k) * NC + c] : s2eg_f16_ld(xb);
        s2eg_stage_chunk(xb, dx, (c & 1) ? 0u : 2u, s2eg_GMAX * NC, hs_w, hs_dh, i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row0 = (int) bg * s2eg_D_ROWS;
    for (int pp = warp; pp < s2eg_D_ROWS / 2; pp += 8) {
        const int r = row0 + 2 * pp;
        constant const uint8_t* codes = blob + s2eg_O_D_CODES + (ulong) r * s2eg_ROW_D;
        constant const uint8_t* scales = blob + s2eg_O_D_SCALES + (ulong) r * s2eg_SC_D * 2;
        const int c = lane;                                // lanes 0..19 hold one chunk each, as above
        int m0[8], m1[8];
        float dw0 = 0.0f, dw1 = 0.0f;
        if (c < NC) {
            s2eg_expand_codes(s2eg_u2_at(codes + (ulong) c * 8), m0);
            s2eg_expand_codes(s2eg_u2_at(codes + s2eg_ROW_D + (ulong) c * 8), m1);
            dw0 = s2eg_f16_ld(scales + (ulong) (c >> 1) * 2);
            dw1 = s2eg_f16_ld(scales + s2eg_SC_D * 2 + (ulong) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            if (c < NC) {
                const int at = k * NC + c;
                int X[8];
                for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (s2eg_GMAX * NC) + at];
                const int2 dh = hs_dh[at];
                const float dx = as_type<float>(dh.x);
                acc0 += dw0 * dx * (float) (s2eg_chunk_s(m0, X) - dh.y);
                acc1 += dw1 * dx * (float) (s2eg_chunk_s(m1, X) - dh.y);
            }
            const float s0 = s2eg_warp_sum(acc0);
            const float s1 = s2eg_warp_sum(acc1);
            if (lane == 0) {
                const ulong o = (ulong) ent_dst[e0 + k] * s2eg_H + (ulong) r;
                out[o] = s0;
                out[o + 1] = s1;
            }
        }
    }
}

// ---------------------------------------------------------------- the resident-group builder

// groups built on the device when every expert is resident at `base + id * blob` (the MTP layer): one block
// of 128 threads, groups in first-appearance order, entries of a group in routing order.  The table entry
// it writes is the blob's ADDRESS as an integer - RULE 9 means the grouped kernels' launcher will read it
// back to the host and bind the blob itself, and MSL cannot take a bound pointer's integer value, so the
// base arrives as a scalar (the same number the CUDA kernel computes: the pointer the caller passed).
kernel void group_resident_kernel(constant const int* ids [[buffer(0)]],
                                  constant const int& n [[buffer(1)]],
                                  constant const int& k_per_tok [[buffer(2)]],
                                  constant const ulong& base_addr [[buffer(3)]],
                                  constant const long& blob [[buffer(4)]],
                                  device ulong* grp_ptr [[buffer(5)]],
                                  device int* grp_start [[buffer(6)]],
                                  device int* counts [[buffer(7)]],
                                  device int* ent_dst [[buffer(8)]],
                                  device int* ent_tok [[buffer(9)]],
                                  uint i [[thread_index_in_threadgroup]]) {
    threadgroup int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129];
    const int ii = (int) i;
    const int e = ii < n ? ids[ii] : -1;
    e_s[ii] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int first = ii, rank = 0, size = 0;
    if (ii < n) {
        for (int j = 0; j < ii; ++j)
            if (e_s[j] == e) { if (first == ii) first = j; ++rank; }
        if (first == ii)
            for (int j = ii; j < n; ++j) size += e_s[j] == e;
    }
    first_s[ii] = first;
    size_s[ii] = (ii < n && first == ii) ? size : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ii == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = base_addr + (ulong) e_s[j] * (ulong) blob;
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ii < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = ii;
        ent_tok[at] = ii / k_per_tok;
    }
}
