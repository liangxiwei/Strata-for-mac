// src/kernels/metal/router_top10.mm - the port of src/kernels/cuda/router_top10.cu's launcher (K5).
// The generic kernel only: the HIP-only "fast" variant stays HIP-only, so router_top10_variant returns
// false exactly as the CUDA (non-HIP) build does.
#include "strata/kernels/router_top10.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

bool router_top10_variant(const float*, int, int, int, int*, float*, void*, int) { return false; }

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    constexpr int RT_MAX_THREADS = 512;
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full simdgroup, for the reductions
    // taken-mask (16B aligned) + n_expert f32 exps + n_expert f32 probabilities; the CUDA kernel's doubles
    // become floats here (see the .metal file's note on the emulations)
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem = taken_bytes + (size_t) n_expert * 4 + (size_t) n_expert * 4;
    metal::Launch kr("router_top10_impl", (unsigned) n_tokens, 1, 1, (unsigned) threads, 1, 1, smem, stream);
    kr.buf(logits).scalar(n_tokens).scalar(n_expert).scalar(k).buf(ids).buf(weights);
    kr.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "router_top10 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "router_top10: %s\n", cudaGetErrorString(s));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
