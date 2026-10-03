// src/kernels/metal/bf16_gemv.mm - the ports of src/kernels/cuda/bf16_gemv.cu's and native_bf16.cu's
// fp32-MMVF host halves (K17 + the gr family's MMVF).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

void finish(void* stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream != nullptr) return;
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(s));
        std::exit(1);
    }
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_out >= 64) {                                   // the warp-per-row path, as the CUDA threshold
        const int warps = THREADS / 32;
        metal::Launch k("bf16_gemv_warp_kernel", (unsigned) ((n_out + warps - 1) / warps), 1, 1, THREADS, 1, 1,
                        0, stream);
        k.buf(x).buf(w).buf(y).scalar(n_in).scalar(n_out).scalar((unsigned) THREADS);
        k.done();
        finish(stream, "bf16_gemv(warp)");
        return;
    }
    metal::Launch k("bf16_gemv_naive_kernel", (unsigned) ((n_out + THREADS - 1) / THREADS), 1, 1, THREADS, 1, 1,
                    0, stream);
    k.buf(x).buf(w).buf(y).scalar(n_in).scalar(n_out).scalar((unsigned) THREADS);
    k.done();
    finish(stream, "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row == 32) {
        const int warps = THREADS / 32;
        metal::Launch k("bf16_gemv_warp_kernel", (unsigned) ((n_out + warps - 1) / warps), 1, 1, THREADS, 1, 1,
                        0, stream);
        k.buf(x).buf(w).buf(y).scalar(n_in).scalar(n_out).scalar((unsigned) THREADS);
        k.done();
        finish(stream, "bf16_gemv_split(warp)");
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                             "warp-per-row path)\n", threads_per_row);
        std::exit(1);
    }
    metal::Launch k("bf16_gemv_split_kernel", (unsigned) n_out, 1, 1, (unsigned) threads_per_row, 1, 1,
                    (size_t) threads_per_row * sizeof(float), stream);
    k.buf(x).buf(w).buf(y).scalar(n_in).scalar(n_out).scalar(threads_per_row);
    k.done();
    finish(stream, "bf16_gemv_split");
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const int bs = mmvf_block_size(n_in);
    metal::Launch k("bf16_f32_mmvf_kernel", (unsigned) n_out, 1, 1, (unsigned) bs, 1, 1, 0, stream);
    k.buf(x).buf(w).buf(y).scalar((int) n_in).scalar((unsigned) bs);   // block: buffer(4), always set - an
    k.done();                                                          // unset binding reads garbage as the stride
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(e));
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const int ntmax = n_tok <= 4 ? 4 : 8;
    const int bs = mmvf_block_size(n_in);
    metal::Launch k("bf16_f32_mmvf_multi_kernel", (unsigned) n_out, 1, 1, (unsigned) bs, 1, 1,
                    (size_t) ntmax * 32 * sizeof(float), stream);
    k.buf(x).scalar(ldx).buf(w).buf(y).scalar(ldy).scalar((int) n_in).scalar(n_tok).scalar(ntmax)
     .scalar((unsigned) bs);            // block: buffer(8), always set - an unset binding reads garbage
    k.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
