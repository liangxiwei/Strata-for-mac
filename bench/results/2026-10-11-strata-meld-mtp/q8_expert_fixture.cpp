// Validate the existing resident Q8 kernels on real imported MTP expert blocks.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

float dot(const uint8_t* w, const uint8_t* x, int n) {
    double sum = 0;
    for (int b = 0; b < n / 32; ++b) {
        uint16_t wd, xd;
        std::memcpy(&wd, w + b * 34, 2);
        std::memcpy(&xd, x + b * 36, 2);
        int s = 0;
        for (int i = 0; i < 32; ++i)
            s += int((int8_t)w[b * 34 + 2 + i]) * int((int8_t)x[b * 36 + 4 + i]);
        sum += double(k::f32_from_f16(wd)) * double(k::f32_from_f16(xd)) * s;
    }
    return float(sum);
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    constexpr int H = 2560, FF = 640, NE = 512, T = 2, K = 2, E = T * K;
    const auto L = k::native_expert_layout(8, 8, H, FF);
    std::vector<uint8_t> blobs(3 * L.bytes);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    const int source_ids[] = {0, 7, 511};
    for (int s = 0; s < 3; ++s) {
        if (fseeko(f, (off_t)source_ids[s] * L.bytes, SEEK_SET) != 0 ||
            std::fread(blobs.data() + s * L.bytes, 1, L.bytes, f) != L.bytes) return 2;
    }
    std::fclose(f);
    std::vector<int32_t> residency(NE, -1);
    for (int s = 0; s < 3; ++s) residency[source_ids[s]] = s;
    std::vector<float> input(T * H);
    for (int i = 0; i < T * H; ++i) input[i] = std::sin(float(i) * .019f) * .13f;
    uint8_t *arena = nullptr, *xq = nullptr;
    float *x = nullptr, *out = nullptr;
    int32_t *ids = nullptr, *res = nullptr;
    void* scratch = nullptr;
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    check(cudaMalloc(&arena, blobs.size()));
    check(cudaMalloc(&x, input.size() * 4));
    check(cudaMalloc(&xq, k::native_q8_1_bytes(H, T)));
    check(cudaMalloc(&out, E * H * 4));
    check(cudaMalloc(&ids, E * 4));
    check(cudaMalloc(&res, NE * 4));
    const size_t scratch_bytes = k::native_expert_scratch_bytes(E, FF);
    check(cudaMalloc(&scratch, scratch_bytes));
    check(cudaMemcpy(arena, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(res, residency.data(), NE * 4, cudaMemcpyHostToDevice));
    k::native_quantize_q8_1(x, xq, H, T, stream);
    check(cudaStreamSynchronize(stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    k::native_expert_resident(L, arena, nullptr, L.bytes, ids, res, NE, T, K, xq, scratch, out, stream);
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, 0));
    cudaGraphDestroy(graph);
    const size_t fa = (E * FF * sizeof(float) + 255) & ~size_t(255);
    std::vector<uint8_t> input_q(k::native_q8_1_bytes(H, T)), state(scratch_bytes);
    std::vector<float> output(E * H);
    check(cudaMemcpy(input_q.data(), xq, input_q.size(), cudaMemcpyDeviceToHost));
    int failures = 0;
    for (int pass = 0; pass < 2; ++pass) {
        const int32_t routes0[] = {0, 7, 511, 7}, routes1[] = {511, 0, 7, 511};
        const auto* routes = pass == 0 ? routes0 : routes1;
        check(cudaMemcpy(ids, routes, E * 4, cudaMemcpyHostToDevice));
        check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(state.data(), scratch, state.size(), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output.data(), out, output.size() * 4, cudaMemcpyDeviceToHost));
        const float* gate = (const float*)state.data();
        const float* up = (const float*)(state.data() + fa);
        const float* h = (const float*)(state.data() + 2 * fa);
        const uint8_t* hq = state.data() + 3 * fa;
        double max_gu = 0, max_sw = 0, max_down = 0;
        auto compare = [&](float got, float want, double& peak) {
            const double d = std::abs(double(got) - want);
            peak = std::max(peak, d);
            if (!std::isfinite(got) || d > 1e-5 + 1e-4 * std::abs(want)) ++failures;
        };
        for (int e = 0; e < E; ++e) {
            const uint8_t* b = blobs.data() + residency[routes[e]] * L.bytes;
            const uint8_t* iq = input_q.data() + (e / K) * H / 32 * 36;
            for (int row = 0; row < FF; ++row) {
                compare(gate[e * FF + row], dot(b + row * L.gu_row, iq, H), max_gu);
                compare(up[e * FF + row], dot(b + L.up_off + row * L.gu_row, iq, H), max_gu);
                const float g = gate[e * FF + row];
                compare(h[e * FF + row], g / (1.f + std::exp(-g)) * up[e * FF + row], max_sw);
            }
            for (int row = 0; row < H; ++row)
                compare(output[e * H + row], dot(b + L.down_off + row * L.d_row,
                                                hq + e * FF / 32 * 36, FF), max_down);
        }
        std::printf("capture replay %d: gate/up max_abs %.9g, swiglu %.9g, down %.9g, cumulative failures %d\n",
                    pass, max_gu, max_sw, max_down, failures);
    }
    cudaGraphExecDestroy(exec);
    cudaFree(arena); cudaFree(x); cudaFree(xq); cudaFree(out); cudaFree(ids); cudaFree(res); cudaFree(scratch);
    cudaStreamDestroy(stream);
    return failures ? 1 : 0;
}
