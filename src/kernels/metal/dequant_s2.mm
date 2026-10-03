// src/kernels/metal/dequant_s2.mm - the port of src/kernels/cuda/dequant_s2.cu's launcher (K2).
#include "strata/kernels/dequant_s2.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    const unsigned grid = (unsigned) ((n_blocks + 255) / 256);
    metal::Launch k("dequant_s2_kernel", grid, 1, 1, 256, 1, 1, 0, nullptr);
    k.buf(codes).buf(scales).buf(out).scalar(n_blocks);
    k.done();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_s2: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
