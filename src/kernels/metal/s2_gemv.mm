// src/kernels/metal/s2_gemv.mm - the port of src/kernels/cuda/s2_gemv.cu's launcher (K3).
#include "strata/kernels/s2_gemv.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    constexpr int64_t QK = 64;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %lld\n", (long long) n_in,
                     (long long) QK);
        std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_out + 127) / 128);
    metal::Launch k("s2_gemv_kernel", grid, 1, 1, 128, 1, 1, 0, nullptr);
    k.buf(x).buf(codes).buf(scales).buf(y).scalar(n_in).scalar(n_out);
    k.done();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
