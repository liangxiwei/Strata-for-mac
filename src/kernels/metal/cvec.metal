// src/kernels/metal/cvec.metal - the port of src/kernels/cuda/cvec.cu's kernel (K13).  One threadgroup per
// (stream, token): the pending write, then the h . v dot, then the update - the CUDA file's shape, with the
// warp tree reductions spelled as simdgroup butterflies.  __expf becomes precise::exp (the build is
// -fno-fast-math; the parity test owns the tolerance).
#include "strata_port.metalh"

constant const int CV_THREADS = 256;
constant const int CV_MAXK = 16;    // n_embd up to 4096, held in registers between the dot and the update

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

kernel void cvec_kernel(device float* R [[buffer(0)]],
                        constant const float* dir [[buffer(1)]],
                        constant const float* s_l [[buffer(2)]],
                        constant const int* on [[buffer(3)]],
                        constant const int& mode [[buffer(4)]],
                        constant const long& layer [[buffer(5)]],
                        constant const int& n [[buffer(6)]],
                        constant const int& hc [[buffer(7)]],
                        constant const long& r_ld [[buffer(8)]],
                        constant const float* bo [[buffer(9)]],
                        constant const long& bo_ld [[buffer(10)]],
                        constant const float* inj [[buffer(11)]],
                        constant const long& inj_ld [[buffer(12)]],
                        constant const int& write [[buffer(13)]],
                        uint2 gp [[threadgroup_position_in_grid]],   // (stream, token) are GROUP indices
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    const int c = (int) gp.x;
    const long t = (long) gp.y;
    device float* r = R + (ulong) t * (ulong) r_ld + (ulong) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the threadgroup
    if (!steer && !write) return;
    constant const float* v = dir + (ulong) layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[(ulong) t * (ulong) inj_ld + c] / (float) hc) : 0.0f;
    constant const float* b = write ? bo + (ulong) t * (ulong) bo_ld : nullptr;
    float x[CV_MAXK];
    float dot = 0.0f;
    for (int k = 0; k < CV_MAXK; ++k) {
        const int d = (int) tid + k * CV_THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fma(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fma(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
        threadgroup float part[CV_THREADS / 32];
        for (int o = 16; o > 0; o >>= 1) dot += simd_shuffle_xor(dot, o);
        if (lane == 0) part[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            float p = tid < (uint) (CV_THREADS / 32) ? part[tid] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) p += simd_shuffle_xor(p, o);
            if (tid == 0) part[0] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = part[0] * s;   // s (h . v)
    }
    for (int k = 0; k < CV_MAXK; ++k) {
        const int d = (int) tid + k * CV_THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fma(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}
