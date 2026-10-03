// src/kernels/metal/elementwise.metal - the port of src/kernels/cuda/elementwise.cu (K1 of
// docs/PORT_METAL/STATUS.md).  Same kernels, same contracts, same launch geometry as the CUDA file; the
// comment trail of the CUDA file is the authority, and anything that had to differ is marked HERE and in
// PROGRESS.md.
//
// Three things are spelled differently by necessity:
//   * MSL gives thread coordinates as attributed parameters, and setBytes maps to `constant T&` references;
//   * the doorbell ring/wait/publish use RELAXED device atomics + a device fence where CUDA used volatile +
//     __threadfence_system() - on Apple Silicon's unified memory that is the documented way to make a GPU
//     store visible to a CPU poll (validated by the M1 micro-test before this file was trusted);
//   * silu computes in fp32, not fp64: Apple GPUs have no native double.  The CUDA file itself notes the
//     difference is in the last bits (its fp64 was matching numpy, not for accuracy's sake).
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// f16_bits.hpp's f16_from_f32, bit for bit (round-to-nearest-even, finite overflow saturates to inf -
// the conflation the CUDA file's round-198 comment warns about).
static inline uint f16_from_f32(float f) {
    const uint x = as_type<uint>(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint rawexp = (x >> 23) & 0xFFu;
    const int exp = (int) rawexp - 127 + 15;
    uint man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return sign | 0x7C00u | (man ? 0x200u : 0u);
    if (exp >= 31) return sign | 0x7C00u;
    if (exp <= 0) {
        if (exp < -10) return sign;
        man |= 0x800000u;
        const uint sh = (uint) (14 - exp);
        uint h = (man >> sh) & 0x3FFu;
        const uint rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
        return sign | h;
    }
    uint h = sign | ((uint) exp << 10) | (man >> 13);
    const uint rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return h;
}

// bf16_bits.hpp's bf16_from_f32: NaN/inf stays, else round-to-nearest-even by truncation trick.
static inline uint bf16_from_f32(float f) {
    uint i = as_type<uint>(f);
    if ((i & 0x7FFFFFFFu) > 0x7F800000u) return (i >> 16) | 64u;
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return i >> 16;
}

// ggml_compute_softplus_f32: log1p(exp(x)) with the large-x branch that avoids overflow (x > 20 is x to
// within f32).  MSL has no log1p; the series covers the small end where log(1+x) would lose it.  precise::
// because xcrun metal's default fast math is lossy enough to fail the parity tolerance (measured, this port).
static inline float softplus_dev(float x) {
    if (x > 20.0f) return x;
    const float e = metal::precise::exp(x);
    // below e ~ 0.05, f32's log(1+e) is cancellation-bound (the sum keeps only ~4 good digits); the series
    // e - e^2/2 + e^3/3 - e^4/4 is Horner'd and good to e^5/5 ~ 6e-8 relative there
    if (e < 0.05f) return e * (1.0f - e * (0.5f - e * (1.0f / 3.0f - e * 0.25f)));
    return metal::precise::log(1.0f + e);
}

kernel void embedding_gather_kernel(constant const uint8_t* codes [[buffer(0)]],
                                    constant const float* scales [[buffer(1)]],
                                    constant const float* offsets [[buffer(2)]],
                                    constant const ulong& n [[buffer(3)]],
                                    constant const int& code_bits [[buffer(4)]],
                                    constant const int& code_bias [[buffer(5)]],
                                    constant const int& group_elems [[buffer(6)]],
                                    device float* out [[buffer(7)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const uint mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const ulong group = i / group_elems;
    const float product = (float) (code + code_bias) * scales[group];       // __fmul_rn: IEEE mul
    out[i] = product + (offsets ? offsets[group] : 0.0f);                   // __fadd_rn: IEEE add
}

kernel void gdn_gate_kernel(constant const float* alpha [[buffer(0)]],
                            constant const float* dt [[buffer(1)]],
                            constant const float* ssm_a [[buffer(2)]],
                            device float* gate [[buffer(3)]],
                            constant const ulong& h_v [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= h_v) return;
    gate[i] = softplus_dev(alpha[i] + dt[i % h_v]) * ssm_a[i % h_v];
}

kernel void scale_kernel(device float* x [[buffer(0)]], constant const long& n [[buffer(1)]],
                         constant const float& s [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < (ulong) n) x[i] *= s;
}

kernel void add_kernel(device float* dst [[buffer(0)]], constant const float* src [[buffer(1)]],
                       constant const long& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < (ulong) n) dst[i] += src[i];
}

kernel void to_f16_kernel(constant const float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                          constant const ulong& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = (ushort) f16_from_f32(x[i]);
}

kernel void to_bf16_kernel(constant const float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                           constant const ulong& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = (ushort) bf16_from_f32(x[i]);
}

kernel void silu_kernel(device float* x [[buffer(0)]], constant const ulong& n [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = x[i];                   // fp32: Apple GPUs have no native double (see the file comment)
    x[i] = v / (1.0f + metal::precise::exp(-v));
}

// One SIMDGROUP per row, reduced through shuffles - the same warp-per-row shape and reduction ORDER as the
// CUDA kernel (shuffle-down tree), so the sums agree bit for bit.  The row guard is load-bearing (the CUDA
// file's QSA bug): the launcher rounds the grid to whole 4-simdgroup blocks.
kernel void rms_norm_weighted_kernel(device float* x [[buffer(0)]],
                                      constant const float* w [[buffer(1)]],
                                      constant const ulong& rows [[buffer(2)]],
                                      constant const ulong& cols [[buffer(3)]],
                                      constant const float& eps [[buffer(4)]],
                                      uint tptg [[threads_per_threadgroup]],
                                      uint tgpg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = (ulong) tgpg * (tptg / 32u) + sg;
    if (row >= rows) return;
    device float* r = x + row * cols;
    float acc = 0.0f;
    for (ulong c = lane; c < cols; c += 32) acc += r[c] * r[c];
    for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
    float inv = 0.0f;
    if (lane == 0) inv = metal::precise::rsqrt(acc / (float) cols + eps);
    inv = simd_shuffle(inv, 0u);
    for (ulong c = lane; c < cols; c += 32) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

// ---- the doorbells: seq_cst atomics on shared storage stand in for volatile + __threadfence_system() ----
kernel void doorbell_ring_kernel(device atomic_uint* seq [[buffer(0)]]) {
    const uint old = atomic_load_explicit(seq, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);   // the __threadfence_system stand-in
    atomic_store_explicit(seq, old + 1u, memory_order_relaxed);
}

kernel void doorbell_wait_kernel(const device atomic_uint* flag [[buffer(0)]],
                                 const device atomic_uint* seq [[buffer(1)]]) {
    const uint want = atomic_load_explicit(seq, memory_order_relaxed);
    while (atomic_load_explicit(flag, memory_order_relaxed) != want) {}
}

kernel void copy_from_mapped_kernel(device float4* dst [[buffer(0)]],
                                    constant const float4* src [[buffer(1)]],
                                    constant const long& n4 [[buffer(2)]],
                                    uint tpg [[threads_per_grid]],
                                    uint gid [[thread_position_in_grid]]) {
    const long stride = (long) tpg;
    for (long i = (long) gid; i < n4; i += stride) dst[i] = src[i];
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself: threadgroup = row.
kernel void copy_rows_from_mapped_kernel(device float4* dst [[buffer(0)]],
                                         constant const float4* src [[buffer(1)]],
                                         constant const long& row4 [[buffer(2)]],
                                         constant const int* hit_rows [[buffer(3)]],
                                         constant const int* count [[buffer(4)]],
                                         uint tgpg [[threadgroup_position_in_grid]],
                                         uint t_in_tg [[thread_index_in_threadgroup]],
                                         uint tptg [[threads_per_threadgroup]]) {
    threadgroup int hit;
    if (t_in_tg == 0) {
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == (int) tgpg;
        hit = h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float4* d = dst + (ulong) tgpg * (ulong) row4;
    if (hit) {
        for (long i = (long) t_in_tg; i < row4; i += (long) tptg) d[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        constant const float4* sr = src + (ulong) tgpg * (ulong) row4;
        for (long i = (long) t_in_tg; i < row4; i += (long) tptg) d[i] = sr[i];
    }
}

kernel void doorbell_publish_kernel(constant const float* x [[buffer(0)]],
                                    constant const int* ids [[buffer(1)]],
                                    constant const float* w [[buffer(2)]],
                                    constant const int& n [[buffer(3)]],
                                    constant const int& k [[buffer(4)]],
                                    device float* x_out [[buffer(5)]],
                                    device int* ids_out [[buffer(6)]],
                                    device float* w_out [[buffer(7)]],
                                    device atomic_uint* seq [[buffer(8)]],
                                    uint t [[thread_index_in_threadgroup]],
                                    uint tptg [[threads_per_threadgroup]]) {
    for (int i = (int) t; i < n; i += (int) tptg) x_out[i] = x[i];
    if ((int) t < k) { ids_out[t] = ids[t]; w_out[t] = w[t]; }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (t == 0) {
        const uint old = atomic_load_explicit(seq, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);   // the __threadfence_system stand-in
    atomic_store_explicit(seq, old + 1u, memory_order_relaxed);
    }
}

kernel void copy_i32_from_mapped_kernel(device int* dst [[buffer(0)]],
                                        constant const int* src [[buffer(1)]],
                                        constant const int& n [[buffer(2)]],
                                        uint t [[thread_index_in_threadgroup]],
                                        uint tptg [[threads_per_threadgroup]]) {
    for (int i = (int) t; i < n; i += (int) tptg) dst[i] = src[i];
}
