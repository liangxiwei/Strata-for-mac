// src/kernels/metal/native_gr.mm - the ports of src/kernels/cuda/native_gr_norm.cu's and
// native_gr_postops.cu's host halves (the gr family's native path).
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR requires non-null four-byte aligned pointers");
}

void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR requires positive bounded dimensions");
}

void check_launch(const char* what) {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + cudaGetErrorString(error));
}

constexpr int THREADS = 256;
unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }

}  // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    const int block = n_cols < 1024 ? 256 : 1024;       // the CUDA template's two paths
    metal::Launch k("weighted_rms_norm", (unsigned) n_rows, 1, 1, (unsigned) block, 1, 1, 0, stream);
    k.buf(input).buf(gamma).buf(output).scalar(n_cols).scalar(epsilon);
    k.done();
    check_launch("native GR RMSNorm launch: ");
}

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    metal::Launch k("native_gr_down_silu_kernel", blocks((std::size_t) hc_lr), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(lo).scalar(hc_lr).scalar(1.0f / float(hc));
    k.done();
    check_launch("native GR postops launch: ");
}

void native_gr_pre_gated(const float* xn, float* gate, float* mixed,
                         int n_embd, int hc, bool fused_layer, void* stream) {
    check_shape(n_embd, hc);
    check_pointer(xn); check_pointer(gate); check_pointer(mixed);
    metal::Launch k("native_gr_pre_gated_kernel", blocks((std::size_t) n_embd), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(xn).buf(gate).buf(mixed).scalar(n_embd).scalar(hc).scalar(1.0f / float(hc))
     .scalar(fused_layer ? 1 : 0);
    k.done();
    check_launch("native GR postops launch: ");
}

void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream) {
    check_shape(n_embd, hc);
    check_pointer(residual); check_pointer(block_out); check_pointer(inject); check_pointer(output);
    metal::Launch k("native_gr_post_kernel", blocks((std::size_t) n_embd * (std::size_t) hc), 1, 1, THREADS, 1,
                    1, 0, stream);
    k.buf(residual).buf(block_out).buf(inject).buf(output).scalar(n_embd).scalar(hc)
     .scalar(1.0f / float(hc));
    k.done();
    check_launch("native GR postops launch: ");
}

}  // namespace strata::kernels
