// src/kernels/metal/s2_gemv_quads.mm - the port of src/kernels/cuda/s2_gemv_quads.cu's launcher.
#include "strata/kernels/s_gemv.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                   int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long) n_in);
        std::exit(1);
    }
    metal::Launch k("s2_gemv_quads_kernel", (unsigned) n_out, 1, 1, (unsigned) threads_per_row, 1, 1,
                    (size_t) threads_per_row * sizeof(float), nullptr);
    k.buf(x).buf(codes).buf(scales).buf(y).scalar(n_in).scalar(n_out).scalar(threads_per_row);
    k.done();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv_quads: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
