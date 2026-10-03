// src/kernels/metal/quantize_act.mm - the port of src/kernels/cuda/quantize_act.cu's launchers (K4).
// Same five entry points, same block geometry; the double-divide subtlety lives in the MSL (rint_of_ratio).
#include "strata/kernels/quantize_act.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void sync_if_needed(void* stream) {
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 32 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of 32\n", (long long) n);
        std::exit(1);
    }
    const int64_t nb = n / 32;
    metal::Launch k("quantize_q8_0_kernel", (unsigned) ((nb + 127) / 128), 1, 1, 128, 1, 1, 0, stream);
    k.buf(x).buf(blocks).scalar(nb);
    k.done();
    launch_check("quantize_q8_0");
    sync_if_needed(stream);
}

void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 32 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of 32\n", (long long) n);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const int64_t nb = n / 32;
    metal::Launch k("quantize_q8_0_scaled_kernel", (unsigned) ((nb + 127) / 128), 1, 1, 128, 1, 1, 0,
                    stream);
    k.buf(x).buf(blocks).buf(scales).scalar(nb);
    k.done();
    launch_check("quantize_q8_0_scaled");
    sync_if_needed(stream);
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t nb = n / 32;
    metal::Launch k("dequant_q8_0_kernel", (unsigned) ((nb + 127) / 128), 1, 1, 128, 1, 1, 0, stream);
    k.buf(blocks).buf(x).scalar(nb);
    k.done();
    launch_check("dequant_q8_0");
    sync_if_needed(stream);
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 256 != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of 256\n", (long long) n);
        std::exit(1);
    }
    const int64_t nb = n / 256;
    metal::Launch k("quantize_q8_K_kernel", (unsigned) ((nb + 63) / 64), 1, 1, 64, 1, 1, 0, stream);
    k.buf(x).buf(blocks).scalar(nb);
    k.done();
    launch_check("quantize_q8_K");
    sync_if_needed(stream);
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t nb = n / 256;
    metal::Launch k("dequant_q8_K_kernel", (unsigned) ((nb + 63) / 64), 1, 1, 64, 1, 1, 0, stream);
    k.buf(blocks).buf(x).scalar(nb);
    k.done();
    launch_check("dequant_q8_K");
    sync_if_needed(stream);
}

}  // namespace strata::kernels
