// src/kernels/metal/router_top10.metal - the port of src/kernels/cuda/router_top10.cu's GENERIC kernel
// (K5; the HIP-only "fast" variant stays HIP-only, and router_top10_variant keeps returning false).
//
// One THREADGROUP per token, as CUDA one block per token.  The fp64 parts are the port's one real
// divergence and each is emulated, not approximated:
//   * exp((double) l - (double) mx): the ARGUMENT is exact in f32 (a difference of two f32s has no rounding
//     in f32 either - Sterbenz), and precise::exp carries ~1 ulp; against the parity tolerance (weights
//     rel 1e-5) that is two orders of margin;
//   * the ascending DOUBLE sum -> the ascending NEUMAIER-COMPENSATED f32 sum (strata_port.metalh): its
//     error is ~2^-46 relative where a plain f32 sum would already sit at the tolerance;
//   * the renormalisation's double sum -> the same compensated sum.
// The selection structure (k passes of a block-wide stable argmax, ties by lowest index) is CUDA's own
// serial-friendly shape, spelled with simdgroups instead of warps.
#include "strata_port.metalh"

constant const int RT_MAX_THREADS = 512;

// MSL has no `extern __shared__`: the dynamic scratch arrives as a threadgroup-memory argument
// ([[threadgroup(0)]], sized by the launcher's shared_bytes), laid out exactly as the CUDA kernel's.
kernel void router_top10_impl(constant const float* logits [[buffer(0)]],
                              constant const int& n_tokens [[buffer(1)]],
                              constant const int& n_expert [[buffer(2)]],
                              constant const int& k [[buffer(3)]],
                              device int* ids [[buffer(4)]],
                              device float* weights [[buffer(5)]],
                              threadgroup uint8_t* s_raw [[threadgroup(0)]],
                              uint t_in [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint nt [[threads_per_threadgroup]],
                              uint tptg [[threads_per_grid]]) {
    const int t = (int) t_in;
    if (t >= n_tokens) return;
    const uint lane = tid % 32u, warp = tid / 32u;
    const uint nw = (nt + 31u) / 32u;
    constant const float* l = logits + (ulong) t * (ulong) n_expert;

    const uint taken_bytes = ((uint) n_expert + 15u) & ~15u;
    threadgroup uint8_t* s_taken = s_raw;
    threadgroup float* s_ex = reinterpret_cast<threadgroup float*>(s_raw + taken_bytes);
    threadgroup float* s_p = s_ex + n_expert;                   // the CUDA file's float tail
    // small fixed scratch for the cross-simdgroup reductions
    threadgroup float s_red[16];
    threadgroup int s_rid[16];
    threadgroup float s_inv;

    // ---- the max: a tree of fmax is EXACT and order-independent
    float mx = -INFINITY;
    for (int e = (int) tid; e < n_expert; e += (int) nt) mx = metal::precise::fmax(mx, l[e]);
    for (int off = 16; off > 0; off >>= 1) mx = metal::precise::fmax(mx, simd_shuffle_down(mx, off));
    if (lane == 0) s_red[warp] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < nw) ? s_red[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) v = metal::precise::fmax(v, simd_shuffle_down(v, off));
        if (tid == 0) s_red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = s_red[0];

    // ---- the 512 exponentials, once each, in parallel (f32: see the file comment)
    for (int e = (int) tid; e < n_expert; e += (int) nt)
        s_ex[e] = metal::precise::exp(l[e] - mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- the sum, ASCENDING, on one thread: compensated, in the reference's order
    if (tid == 0) {
        KahanSum s;
        for (int e = 0; e < n_expert; ++e) s.add(s_ex[e]);
        s_inv = 1.0f / s.value();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = s_inv;

    // ---- p[] once (the same expression the CUDA kernel hoisted)
    for (int e = (int) tid; e < n_expert; e += (int) nt) s_p[e] = s_ex[e] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- the selection: k passes of a stable block argmax, ties by lowest index
    for (int e = (int) tid; e < n_expert; e += (int) nt) s_taken[e] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;                                      // loses to every real index
        for (int e = (int) tid; e < n_expert; e += (int) nt) {
            if (s_taken[e]) continue;
            const float pe = s_p[e];
            if (pe > bv) { bv = pe; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_red[warp] = bv; s_rid[warp] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            float v = (tid < nw) ? s_red[tid] : -INFINITY;
            int ix = (tid < nw) ? s_rid[tid] : n_expert;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_down(v, off);
                const int oi = simd_shuffle_down(ix, off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (tid == 0 && ix < n_expert) {
                ids[(ulong) t * k + i] = ix;
                weights[(ulong) t * k + i] = v;
                s_taken[ix] = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- renormalise, ggml's lower clamp, order preserved (the compensated sum again)
    if (tid == 0) {
        KahanSum s;
        for (int i = 0; i < k; ++i) s.add(weights[(ulong) t * k + i]);
        const float sc = metal::precise::fmax(s.value(), 6.103515625e-05f);   // 2**-14
        for (int i = 0; i < k; ++i)
            weights[(ulong) t * k + i] = weights[(ulong) t * k + i] / sc;
    }
}
