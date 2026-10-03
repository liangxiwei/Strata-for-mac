#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/metal_mmvq.hpp"
#include "strata/platform/metal_launch.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Compare against the original four-warp shader, not a second call through the new dispatcher.
int main(int argc, char**) {
    const bool bench = argc > 1;
    if (!bench) setenv("STRATA_METAL_IQ4_EXPAND", "1", 1);
    cudaStream_t stream{}; cudaStreamCreate(&stream);
    long long different = 0, nonfinite = 0, checked = 0, byte_diff = 0;
    std::printf("K,N,layout,us\n");
    const std::vector<std::pair<int, int>> shapes = bench
        ? std::vector<std::pair<int, int>>{{2560, 248320}, {2560, 10240}, {2560, 6144}, {2560, 12288},
                                         {6144, 2560}, {2560, 512}, {2560, 640}}
        : std::vector<std::pair<int, int>>{{256, 513}, {512, 513}, {768, 513}, {2560, 513},
                                         {3840, 513}, {4096, 513}, {4352, 513}, {10240, 513}, {16384, 513}};
    for (auto [K, N] : shapes) {
        const size_t wbytes = strata::kernels::native_mmvq_weight_bytes(23, K, N);
        std::mt19937 rng(K + N);
        std::vector<unsigned char> weights(wbytes);
        for (auto& v : weights) v = (unsigned char) rng();
        for (size_t i = 0; i < wbytes; i += 136) {
            const unsigned r = rng();
            const uint16_t scale = uint16_t(((r >> 31) << 15) | ((5 + (r >> 10) % 5) << 10) | (r & 1023));
            std::memcpy(weights.data() + i, &scale, 2);
        }
        std::vector<float> x(K), expected(N), got(N);
        std::normal_distribution<float> normal;
        for (auto& v : x) v = normal(rng);
        void *dw, *expanded, *qx; float *dx, *dy;
        const size_t qcol = strata::kernels::native_q8_1_bytes(K, 1);
        cudaMalloc(&dw, wbytes); cudaMalloc(&qx, qcol * 2);
        cudaMalloc(&expanded, (wbytes / 136) * 264);
        cudaMalloc(&dx, K * 4); cudaMalloc(&dy, N * 4 * 2);
        cudaMemcpy(dw, weights.data(), wbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice);
        strata::kernels::native_quantize_q8_1(dx, qx, K, 1, stream);
        {
            const unsigned long words = (wbytes / 136) * 32;
            strata::metal::Launch expand("native_iq4_xs_expand", (words + 255) / 256, 1, 1, 256, 1, 1, 0, stream);
            expand.buf(dw).buf(expanded).scalar(words); expand.done();
        }
        auto launch = [&](const std::string& layout) {
            const bool small = K / 256 < 16;
            const std::string suffix = small ? "small" : "large";
            const int rows = small ? 4 : 1;
            const int warps = 4;
            const std::string kernel_name = (layout == "expanded" ? "native_iq4_xs_expanded_" :
                                             "native_iq4_xs_mmvq_kernel_") + suffix;
            strata::metal::Launch kernel(kernel_name.c_str(),
                                        (N + rows - 1) / rows, 1, 1, 32, warps, 1, 0, stream);
            kernel.buf(layout == "expanded" ? expanded : dw)
                  .buf(qx).buf(dy).scalar(K).scalar(N); kernel.done();
        };
        launch("original"); cudaStreamSynchronize(stream);
        cudaMemcpy(expected.data(), dy, N * 4, cudaMemcpyDeviceToHost);
        for (const std::string layout : {"original", "expanded"}) {
            launch(layout); cudaStreamSynchronize(stream);
            cudaMemcpy(got.data(), dy, N * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                different += std::memcmp(&expected[i], &got[i], 4) != 0;
                nonfinite += !std::isfinite(expected[i]) || !std::isfinite(got[i]);
                ++checked;
            }
            if (bench) {
                cudaGraph_t graph{}; cudaGraphExec_t exec{};
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
                for (int i = 0; i < 32; ++i) launch(layout);
                cudaStreamEndCapture(stream, &graph); cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                cudaGraphDestroy(graph);
                cudaGraphLaunch(exec, stream); cudaStreamSynchronize(stream);
                const auto start = std::chrono::steady_clock::now();
                for (int trial = 0; trial < 5; ++trial) {
                    cudaGraphLaunch(exec, stream); cudaStreamSynchronize(stream);
                }
                const double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / 160;
                std::printf("%d,%d,%s,%.3f\n", K, N, layout.c_str(), us);
                cudaGraphExecDestroy(exec);
            }
        }
        if (!bench) {
            // Check the representation against a scalar byte oracle, including all untouched scale bytes.
            constexpr int table[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            std::vector<unsigned char> packed((wbytes / 136) * 264);
            cudaMemcpy(packed.data(), expanded, packed.size(), cudaMemcpyDeviceToHost);
            for (size_t b = 0; b < wbytes / 136; ++b) {
                byte_diff += std::memcmp(packed.data() + b * 264, weights.data() + b * 136, 8) != 0;
                for (int sub = 0; sub < 8; ++sub) for (int j = 0; j < 16; ++j) {
                    const unsigned char q = weights[b * 136 + 8 + sub * 16 + j];
                    byte_diff += packed[b * 264 + 8 + sub * 32 + j] != (unsigned char) table[q & 15];
                    byte_diff += packed[b * 264 + 8 + sub * 32 + 16 + j] != (unsigned char) table[q >> 4];
                }
            }
            const size_t prepared = strata::kernels::metal_iq4_prepare(23, dw, K, N);
            byte_diff += prepared != packed.size();
            byte_diff += strata::kernels::metal_iq4_prepare(23, dw, K, N) != 0;
            cudaMemcpy((char*) qx + qcol, qx, qcol, cudaMemcpyDeviceToDevice);
            for (int T : {1, 2}) {
                cudaGraph_t graph{}; cudaGraphExec_t exec{};
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
                strata::kernels::native_mmvq(23, dw, qx, dy, K, N, T, stream);
                cudaStreamEndCapture(stream, &graph); cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                cudaGraphDestroy(graph);
                cudaGraphLaunch(exec, stream); cudaStreamSynchronize(stream);
                std::vector<float> multi(T * N);
                cudaMemcpy(multi.data(), dy, T * N * 4, cudaMemcpyDeviceToHost);
                for (int i = 0; i < T * N; ++i) {
                    different += std::memcmp(&expected[i % N], &multi[i], 4) != 0;
                    nonfinite += !std::isfinite(multi[i]); ++checked;
                }
                cudaGraphExecDestroy(exec);
            }
            strata::kernels::metal_iq4_release(dw);
            strata::kernels::metal_iq4_release(dw);  // Idempotent; the original remains usable.
            strata::kernels::native_mmvq(23, dw, qx, dy, K, N, 1, stream);
            cudaStreamSynchronize(stream); cudaMemcpy(got.data(), dy, N * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                different += std::memcmp(&expected[i], &got[i], 4) != 0;
                nonfinite += !std::isfinite(got[i]); ++checked;
            }
        }
        cudaFree(dw); cudaFree(expanded); cudaFree(qx); cudaFree(dx); cudaFree(dy);
    }
    cudaStreamDestroy(stream);
    std::printf("IQ4 decode: %lld outputs, %lld bit differences, %lld nonfinite, %lld representation/lifetime errors\n",
                checked, different, nonfinite, byte_diff);
    return different != 0 || nonfinite != 0 || byte_diff != 0;
}
