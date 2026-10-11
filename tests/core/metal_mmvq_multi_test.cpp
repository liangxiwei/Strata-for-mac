#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/metal_mmvq.hpp"
#include "strata/platform/metal_launch.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

// The unchanged packed-byte single-token shaders are the GPU oracle. Independent
// columns, row tails, guards and changed-input tape replay detect indexing errors.
int main() {
    setenv("STRATA_METAL_IQ4_EXPAND", "1", 1);
    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    long long checked = 0, different = 0, guards = 0, cases = 0;
    for (int type : {12, 13, 14, 20, 23}) {
        const int block = type == 20 ? 32 : 256;
        const int bytes = type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : type == 20 ? 18 : 136;
        const std::string stem = type == 12 ? "native_q4_k_mmvq_kernel_" :
                                 type == 13 ? "native_q5_k_mmvq_kernel_" : type == 14 ? "native_q6_k_mmvq_kernel_" :
                                 type == 20 ? "native_small_mmvq_kernel_iq4_nl_" : "native_iq4_xs_mmvq_kernel_";
        for (auto [K, N] : {std::pair{256, 1}, {512, 17}, {768, 513}, {1024, 3},
                            {2560, 513}, {6144, 2561}, {16384, 513}}) {
            const size_t wb = strata::kernels::native_mmvq_weight_bytes(type, K, N);
            std::mt19937 random(type + K + N);
            std::vector<unsigned char> weights(wb);
            for (auto& v : weights) v = (unsigned char) random();
            for (size_t b = 0; b < wb; b += bytes) {
                const unsigned r = random();
                const uint16_t scale = uint16_t(((r >> 31) << 15) | ((5 + (r >> 10) % 7) << 10) | (r & 1023));
                std::memcpy(weights.data() + b + (type == 14 ? 208 : 0), &scale, 2);
                if (type == 12 || type == 13) {
                    const unsigned m = random();
                    const uint16_t minimum = uint16_t(((m >> 31) << 15) | ((5 + (m >> 10) % 7) << 10) | (m & 1023));
                    std::memcpy(weights.data() + b + 2, &minimum, 2);
                }
            }
            void *dw, *qx;
            float *dx, *ref, *out;
            const size_t qcol = strata::kernels::native_q8_1_bytes(K, 1);
            cudaMalloc(&dw, wb); cudaMalloc(&qx, qcol * 8); cudaMalloc(&dx, size_t(K) * 8 * 4);
            cudaMalloc(&ref, size_t(N) * 8 * 4); cudaMalloc(&out, (size_t(N) * 8 + 32) * 4);
            cudaMemcpy(dw, weights.data(), wb, cudaMemcpyHostToDevice);
            const int small_limit = type == 12 || type == 13 ? 8 : type == 14 ? 4 : type == 20 ? 64 : 16;
            const int oracle_rows = K / block < small_limit ? 4 : 1;
            const std::string oracle = stem + (oracle_rows == 4 ? "small" : "large");
            for (int expanded = 0; expanded <= (type == 23 && K == 2560 ? 1 : 0); ++expanded) {
                if (expanded) strata::kernels::metal_iq4_prepare(type, dw, K, N);
                for (int T = 1; T <= 8; ++T) {
                    cudaGraph_t graph{}; cudaGraphExec_t exec{};
                    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
                    strata::kernels::native_mmvq(type, dw, qx, out + 16, K, N, T, stream);
                    cudaStreamEndCapture(stream, &graph);
                    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                    cudaGraphDestroy(graph);
                    for (int pass = 0; pass < 2; ++pass) {
                        std::vector<float> input(size_t(K) * T);
                        for (auto& v : input) v = float(int(random() % 20001) - 10000) / 713.0f;
                        if (!pass) std::fill(input.begin(), input.begin() + K, 0.0f);
                        cudaMemcpy(dx, input.data(), input.size() * 4, cudaMemcpyHostToDevice);
                        strata::kernels::native_quantize_q8_1(dx, qx, K, T, stream);
                        for (int j = 0; j < T; ++j) {
                            strata::metal::Launch legacy(oracle.c_str(), (N + oracle_rows - 1) / oracle_rows,
                                                         1, 1, 32, 4, 1, 0, stream);
                            legacy.buf(dw).buf((const unsigned char*) qx + size_t(j) * qcol)
                                  .buf(ref + size_t(j) * N).scalar(K).scalar(N);
                            legacy.done();
                        }
                        cudaStreamSynchronize(stream);
                        std::vector<float> expected(size_t(N) * T), got(size_t(N) * 8 + 32, 123456.0f);
                        cudaMemcpy(expected.data(), ref, expected.size() * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(out, got.data(), got.size() * 4, cudaMemcpyHostToDevice);
                        cudaGraphLaunch(exec, stream); cudaStreamSynchronize(stream);
                        cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost);
                        long diff = 0, guard = 0;
                        for (size_t i = 0; i < expected.size(); ++i)
                            diff += !std::isfinite(got[i + 16]) || std::memcmp(&got[i + 16], &expected[i], 4) != 0;
                        for (size_t i = 0; i < got.size(); ++i)
                            if (i < 16 || i >= 16 + expected.size()) guard += got[i] != 123456.0f;
                        checked += expected.size(); different += diff; guards += guard; ++cases;
                        if (diff || guard) std::fprintf(stderr, "type=%d K=%d N=%d T=%d expanded=%d pass=%d diff=%ld guard=%ld\n",
                                                       type, K, N, T, expanded, pass, diff, guard);
                    }
                    cudaGraphExecDestroy(exec);
                }
                if (expanded) strata::kernels::metal_iq4_release(dw);
            }
            cudaFree(dw); cudaFree(qx); cudaFree(dx); cudaFree(ref); cudaFree(out);
        }
    }
    cudaStreamDestroy(stream);
    std::printf("Metal MMVQ multi: %lld cases, %lld outputs, %lld bit differences, %lld guard errors\n",
                cases, checked, different, guards);
    return different != 0 || guards != 0;
}
