// src/kernels/metal/fused_gdn.mm - the port of src/kernels/cuda/fused_gdn.cu's host half.  Same header
// contract (include/strata/kernels/fused_gdn.hpp); every pointer is a bound buffer argument and every
// int/float a scalar (docs/PORT_METAL/PROGRESS.md rule 9), chained in EXACTLY the kernel signatures'
// [[buffer(N)]] order (the unset-binding bug class).  Host-side validation ports verbatim from the .cu.
// The .cu's dim3(S, RG) block is one flat 512-thread group here - the kernel's own `tid = rg * S + col`
// numbering, read back (fused_gdn.metal's file comment).
#include "strata/kernels/fused_gdn.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    metal::Launch kern("gdn_conv_l2_kernel", (unsigned) (channels / S), 1, 1, (unsigned) S, 1, 1, 0, stream);
    kern.buf(history).buf(qkv).buf(conv_w).buf(h).scalar(qk_heads).scalar(eps);
    kern.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_conv_l2: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    metal::Launch kern("gdn_ab_kernel", (unsigned) ((2 * h_v + 7) / 8), 1, 1, 256, 1, 1, 0, stream);
    kern.buf(x).buf(w_alpha).buf(w_beta).buf(dt).buf(ssm_a).buf(gate).buf(beta)
        .scalar(n_embd).scalar(h_v);
    kern.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_ab: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    metal::Launch kern("gdn_step_norm_kernel", (unsigned) h_v, 1, 1, (unsigned) (S * RG), 1, 1, 0, stream);
    kern.buf(state).buf(q).buf(k).buf(v).buf(gate).buf(beta).buf(z).buf(gamma).scalar(eps).buf(y)
        .scalar(h_k).scalar(h_v);
    kern.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gdn_step_norm: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
