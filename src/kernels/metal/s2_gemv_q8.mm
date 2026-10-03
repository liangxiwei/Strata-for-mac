// src/kernels/metal/s2_gemv_q8.mm - the port of src/kernels/cuda/s2_gemv_q8.cu's host half (K11).  Same
// contract as the header, same guards, same grid (one group per row) and same dynamic-shared sizing; the
// launch is a metal::Launch chain whose argument order IS the kernel's [[buffer(N)]] order.
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;
}  // namespace

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    metal::Launch k("s2_gemv_q8_kernel", (unsigned) n_out, 1, 1, (unsigned) threads_per_row, 1, 1,
                    (size_t) threads_per_row * sizeof(float), stream);
    k.buf(act).buf(codes).buf(scales).buf(y).scalar(n_in).scalar(n_out).scalar(threads_per_row);
    k.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv_q8 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
