#include "strata/kernels/elementwise.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Independent byte oracle, padding guards, live graph sources and compute/copy dependencies.
int main(int argc, char**) {
    if (argc > 1) setenv("STRATA_METAL_COMPUTE_COPY", "0", 1);
    cudaStream_t s{}; cudaStreamCreate(&s);
    int failures = 0, cases = 0;
    for (size_t width : {size_t(1), 3ul, 4ul, 15ul, 16ul, 17ul, 32ul, 127ul, 512ul, 513ul, 8193ul,
                         65536ul, 65537ul}) {
        for (size_t rows : {1ul, 3ul, 24ul}) for (size_t offset : {0ul, 1ul, 3ul, 16ul}) {
            const size_t sp = width * 2, dp = width + (offset ? 7 : 16);
            const size_t sn = offset + rows * sp + 32, dn = offset + rows * dp + 32;
            unsigned char *src, *dst;
            cudaMallocHost(&src, sn); cudaMalloc(&dst, dn);
            std::vector<unsigned char> expected(dn), got(dn);
            auto copy = [&] {
                if (rows == 1) return cudaMemcpyAsync(dst + offset, src + offset, width, cudaMemcpyDefault, s);
                return cudaMemcpy2DAsync(dst + offset, dp, src + offset, sp, width, rows, cudaMemcpyDefault, s);
            };
            cudaGraph_t graph{}; cudaGraphExec_t exec{};
            cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            failures += copy() != cudaSuccess;
            cudaStreamEndCapture(s, &graph); cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            for (int replay = 0; replay < 3; ++replay) {
                std::fill(expected.begin(), expected.end(), 0xa5);
                cudaMemcpy(dst, expected.data(), dn, cudaMemcpyHostToDevice);
                if (replay) cudaGraphLaunch(exec, s);
                // Graph encoding must not consume mapped host writes until the stream is flushed.
                for (size_t i = 0; i < sn; ++i) src[i] = (unsigned char) (i * 73 + replay * 43);
                if (!replay) failures += copy() != cudaSuccess;
                for (size_t row = 0; row < rows; ++row)
                    std::memcpy(expected.data() + offset + row * dp, src + offset + row * sp, width);
                cudaStreamSynchronize(s);
                cudaMemcpy(got.data(), dst, dn, cudaMemcpyDeviceToHost);
                if (got != expected) {
                    std::fprintf(stderr, "copy mismatch width=%zu rows=%zu offset=%zu replay=%d\n",
                                 width, rows, offset, replay);
                    ++failures;
                }
                ++cases;
            }
            cudaGraphExecDestroy(exec); cudaFreeHost(src); cudaFree(dst);
        }
    }
    // Compute -> copy -> compute in a single captured serial pass.
    float *src, *dst;
    cudaMallocHost(&src, 512 * 4); cudaMalloc(&dst, 512 * 4);
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    strata::kernels::scale_inplace(src, 512, 2.f, s);
    cudaMemcpy2DAsync(dst, 64, src, 128, 64, 16, cudaMemcpyDeviceToDevice, s);
    strata::kernels::scale_inplace(dst, 256, 3.f, s);
    cudaStreamEndCapture(s, &graph); cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    for (int replay = 0; replay < 3; ++replay) {
        cudaGraphLaunch(exec, s);
        for (int i = 0; i < 512; ++i) src[i] = float(i + replay);
        cudaStreamSynchronize(s);
        for (int i = 0; i < 256; ++i) failures += dst[i] != 6.f * float((i / 16) * 32 + i % 16 + replay);
        ++cases;
    }
    cudaGraphExecDestroy(exec); cudaFreeHost(src); cudaFree(dst); cudaStreamDestroy(s);
    std::printf("Metal copy: %d cases, %d failures (%s)\n", cases, failures, argc > 1 ? "blit" : "compute");
    return failures != 0;
}
