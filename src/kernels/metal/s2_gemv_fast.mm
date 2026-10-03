// src/kernels/metal/s2_gemv_fast.mm - the port of src/kernels/cuda/s2_gemv_fast.cu's launcher.
//
// DEVIATION FROM THE CUDA FILE, FORCED BY THE SHIM: `ensure_lut()` uploaded the 256x4 float code table
// with cudaMemcpyToSymbol, which the Metal shim does not implement (it returns an error).  The table held
// exactly the integers ((b >> 2k) & 3) - 1, so the .metal computes them at the use site - the same float
// values, the same multiply order, no upload at all (s2_gemv_fast.metal's file comment has the reasoning).
// The per-device `g_lut_ready` state goes with it: there is nothing left to upload.
#include "strata/kernels/s_gemv.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                  int64_t n_out, int threads_per_row, bool stage_x) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % 64 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long) n_in, 64);
        std::exit(1);
    }
    if (stage_x && n_in > 4096) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld exceeds the %d-half shared staging limit\n",
                     (long long) n_in, 4096);
        std::exit(1);
    }
    const size_t smem = (stage_x ? 4096 * sizeof(uint16_t) : 0) + (size_t) threads_per_row * sizeof(float);
    metal::Launch k(stage_x ? "s2_gemv_fast_kernel_staged" : "s2_gemv_fast_kernel_global", (unsigned) n_out,
                    1, 1, (unsigned) threads_per_row, 1, 1, smem, nullptr);
    k.buf(x).buf(codes).buf(scales).buf(y).scalar(n_in).scalar(n_out).scalar(threads_per_row);
    k.done();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv_fast: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
