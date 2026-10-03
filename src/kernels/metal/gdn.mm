// src/kernels/metal/gdn.mm - the port of src/kernels/cuda/gdn.cu's host half (K10).  Same header contract
// (include/strata/kernels/gdn.hpp); every k<<<grid, block, 0, stream>>> becomes a metal::Launch whose chained
// .buf()/.scalar() order is EXACTLY the kernel signature's [[buffer(N)]] order (docs/PORT_METAL/PROGRESS.md:
// an unset binding once read garbage as a loop stride and hung the GPU).  All pointers ride as bound buffer
// arguments, only the ints/floats as scalars (rule 9).  Host-side validation and the MAX_H grid math port
// verbatim from the .cu, comments included - they record the measured behaviour this port must keep.
#include "strata/kernels/gdn.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;   ///< threads along j, the state's fast axis
/// Heads staged per block.  **THIS IS A LATENCY-HIDING KNOB, NOT A CAPACITY ONE, AND IT WAS SET WRONG.**
/// (the .cu's comment, kept: at MAX_H = 8 the kernel ran 24 blocks x 32 threads - 1 warp on 24 of 48 SMs -
/// and measured 0.2800 ms per GDN block, 14% of a 71.5 ms token; at MAX_H = 1 it is 192 blocks, 4 warps on
/// every SM.  The arithmetic does not change at all: the same columns are computed by the same code.)
constexpr int MAX_H = 1;       ///< heads staged in shared memory per block

}  // namespace

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [8][128])\n",
                     (long long) s.S);
        std::exit(1);
    }
    const unsigned gx = (unsigned) ((s.S + JTHREADS - 1) / JTHREADS);
    const unsigned gy = (unsigned) ((s.h_v + MAX_H - 1) / MAX_H);
    metal::Launch kern("gdn_step_kernel", gx, gy, 1, (unsigned) JTHREADS, 1, 1, 0, stream);
    kern.buf(state).buf(q).buf(k).buf(v).buf(gate).buf(beta).buf(o)
        .scalar((int) s.S).scalar((int) s.h_k).scalar((int) s.h_v);
    kern.done();
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_step: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int blocks = (int) ((channels + 255) / 256);
    metal::Launch kern("gdn_conv_kernel", (unsigned) blocks, 1, 1, 256, 1, 1, 0, stream);
    kern.buf(conv_state).buf(x).buf(kW).buf(out).scalar((int) channels).scalar((int) d_conv);
    kern.done();
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_conv_step: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide warp reduction\n", (long long) cols);
        std::exit(1);
    }
    metal::Launch kern("gdn_l2_kernel", (unsigned) rows, 1, 1, 32, 1, 1, 0, stream);
    kern.buf(x).scalar((int) cols).scalar(eps);
    kern.done();
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_l2_norm: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

/// `beta = sigmoid(beta)`, in place, over the `h_v` per-head scalars.  (The .cu's warning, kept: this is
/// `build_layer_attn_linear` L889 and leaving it out was the C1 bug - the kernel is documented as
/// `d[h,j] = (v[h,j] - sk[h,j]) * beta[h]` and applies no sigmoid of its own, so the layer glue owes it one.
/// It survived gdn_parity because the test supplies its own beta.)
void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int) h_v;
    metal::Launch kern("gdn_beta_gate_kernel", (unsigned) ((n + 127) / 128), 1, 1, 128, 1, 1, 0, stream);
    kern.buf(beta).scalar(n);
    kern.done();
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_beta_gate: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream) {
    if (h_v <= 0 || S <= 0) return;
    metal::Launch kern("gdn_out_norm_kernel", (unsigned) h_v, 1, 1, 32, 1, 1, 0, stream);
    kern.buf(o).buf(z).buf(ssm_norm).buf(y).scalar((int) S).scalar(eps);
    kern.done();
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_out_norm: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
