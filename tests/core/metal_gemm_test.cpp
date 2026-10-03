#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#include <cstring>

// Same native blocks as the independent IQ dequant parity fixtures: random legal codes with small,
// finite scales. Compare tiled products with the validated full FP16 dequantizer + ordinary GEMM.
int native_gemm_case(int gu_type, int down_type, int H, int FF, strata::prefill::Gemm& gemm,
                     cudaStream_t stream, std::mt19937& rng) {
    namespace k = strata::kernels;
    const auto L = k::native_expert_layout(gu_type, down_type, H, FF);
    const size_t offset = L.bytes + 128;
    std::vector<uint8_t> blobs(offset + L.bytes);
    auto matrix = [&](size_t at, int type, int rows, int cols) {
        const size_t size = rows * k::iq_row_bytes(type, cols);
        for (size_t i = 0; i < size; ++i) blobs[at + i] = (uint8_t) rng();
        const int qk = type == 42 ? 64 : type == 20 ? 32 : 256;
        const size_t block = k::iq_row_bytes(type, qk);
        for (size_t i = 0; i < size; i += block) {
            if (type == 29) {
                blobs[at + i + 55] = (blobs[at + i + 55] & 15) | 0x10;
            } else {
                const uint16_t d = k::f16_from_f32((float(int(rng() % 63) - 31)) / 8192);
                std::memcpy(blobs.data() + at + i, &d, 2);
            }
        }
    };
    for (size_t at : {size_t(0), offset}) {
        matrix(at, gu_type, 2 * FF, H);
        matrix(at + L.down_off, down_type, H, FF);
    }
    constexpr int rows = 37;
    const int32_t desc[] = {0, 1, 3, 0, (int32_t)offset, 4, 16, 0,
                           (int32_t)offset, 20, 16, 0, 0, 36, 1, 0};
    uint8_t* arena = nullptr;
    uint16_t *x = nullptr, *w = nullptr;
    int32_t* tiles = nullptr;
    float *actual = nullptr, *expected = nullptr;
    cudaMalloc(&arena, blobs.size()); cudaMalloc(&tiles, sizeof desc);
    cudaMalloc(&x, rows * H * 2); cudaMalloc(&w, 2 * FF * H * 2);
    const size_t out_cap = (size_t) rows * std::max(H, 2 * FF) + 16;
    cudaMalloc(&actual, out_cap * 4); cudaMalloc(&expected, out_cap * 4);
    cudaMemcpyAsync(arena, blobs.data(), blobs.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tiles, desc, sizeof desc, cudaMemcpyHostToDevice, stream);
    int failures = 0;
    for (bool down : {false, true}) {
        const int inner = down ? FF : H, cols = down ? H : 2 * FF;
        std::vector<uint16_t> input(rows * inner);
        for (auto& v : input) v = k::f16_from_f32(float(int(rng() % 2001) - 1000) / 113);
        std::vector<float> got((size_t) rows * cols + 16, -987.f), want(got);
        cudaMemcpyAsync(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(actual, got.data(), got.size() * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(expected, want.data(), want.size() * 4, cudaMemcpyHostToDevice, stream);
        for (int expert = 0; expert < 2; ++expert) {
            auto* blob = arena + expert * offset;
            if (down) k::iq_dequant_f16(down_type, blob + L.down_off, (int64_t) H * FF, w, stream);
            else k::iq_dequant_gu_f16(gu_type, blob, blob + L.up_off, FF, H, w, stream);
            const int first = expert ? 4 : 1, count = expert ? 32 : 3;
            gemm.f16(x + first * inner, w, expected + first * cols, count, cols, inner);
            if (!expert) gemm.f16(x + 36 * inner, w, expected + 36 * cols, 1, cols, inner);
        }
        k::native_expert_gemm(L, x, arena, tiles, actual, 4, down, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(got.data(), actual, got.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(want.data(), expected, want.size() * 4, cudaMemcpyDeviceToHost);
        size_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) diff += std::memcmp(&got[i], &want[i], 4) != 0;
        std::printf("Native tiled %d/%d %dx%d %s, offsets/tails/guards: %zu differences\n",
                    gu_type, down_type, H, FF, down ? "down" : "gate/up", diff);
        failures += diff != 0;
    }
    cudaFree(arena); cudaFree(tiles); cudaFree(x); cudaFree(w); cudaFree(actual); cudaFree(expected);
    return failures;
}

int main() {
    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(stream, 0, err)) return 1;
    std::mt19937 rng(17);
    int fail = 0;
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42})
        for (int dt : {20, 23, 42}) fail += native_gemm_case(gu, dt, 512, 256, gemm, stream, rng);
    for (int gu : {16, 22, 29}) fail += native_gemm_case(gu, 42, 2560, 640, gemm, stream, rng);
    {
        constexpr size_t gu_codes = 1280 * 640, d_codes = 2560 * 160;
        constexpr size_t gu_scales = 1280 * 40, d_scales = 2560 * 10;
        std::vector<uint8_t> blob(gu_codes + d_codes + (gu_scales + d_scales) * 2);
        for (size_t i = 0; i < gu_codes + d_codes; ++i) blob[i] = uint8_t(rng());
        for (size_t i = 0; i < gu_scales + d_scales; ++i) {
            const uint16_t h = strata::kernels::f16_from_f32((float(int(i % 127) - 63)) / 1024);
            std::memcpy(blob.data() + gu_codes + d_codes + 2 * i, &h, 2);
        }
        uint8_t* b = nullptr; uint16_t *gu = nullptr, *d = nullptr;
        cudaMalloc(&b, blob.size()); cudaMalloc(&gu, gu_codes * 8); cudaMalloc(&d, d_codes * 8);
        cudaMemcpyAsync(b, blob.data(), blob.size(), cudaMemcpyHostToDevice, stream);
        strata::prefill::blob_dequant_f16(b, gu, d, stream);
        cudaStreamSynchronize(stream);
        size_t bad = 0;
        for (int part = 0; part < 2; ++part) {
            const size_t codes = part ? d_codes : gu_codes, co = part ? gu_codes : 0;
            const size_t so = gu_codes + d_codes + (part ? gu_scales * 2 : 0);
            std::vector<uint16_t> got(codes * 4);
            cudaMemcpy(got.data(), part ? d : gu, got.size() * 2, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < got.size(); ++i) {
                uint16_t h; std::memcpy(&h, blob.data() + so + (i / 64) * 2, 2);
                const int q = int((blob[co + i / 4] >> (2 * (i % 4))) & 3) - 1;
                const uint16_t want = strata::kernels::f16_from_f32(q * strata::kernels::f32_from_f16(h));
                bad += got[i] != want;
            }
        }
        std::printf("Canonical expert dequant: %zu differences\n", bad);
        fail += bad != 0;
        // Different experts, partial tiles and a nonzero row offset. Compare the
        // fused Q2 product against independently unpacked FP16 weights + GEMM.
        const size_t stride = blob.size();
        blob.resize(stride * 2);
        std::memcpy(blob.data() + stride, blob.data(), stride);
        for (size_t i = 0; i < gu_codes + d_codes; ++i) blob[stride + i] ^= 0xff;
        uint8_t* arena = nullptr;
        int32_t* tiles = nullptr;
        uint16_t* x = nullptr;
        float *actual = nullptr, *expected = nullptr;
        constexpr int rows = 36;
        const int32_t tile_data[] = {0, 0, 3, 0, 1, 3, 16, 0, 1, 19, 16, 0, 0, 35, 1, 0};
        cudaMalloc(&arena, blob.size()); cudaMalloc(&tiles, sizeof(tile_data));
        cudaMalloc(&x, rows * 2560 * 2);
        cudaMalloc(&actual, (rows * 2560 + 16) * 4); cudaMalloc(&expected, rows * 2560 * 4);
        cudaMemcpyAsync(arena, blob.data(), blob.size(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(tiles, tile_data, sizeof(tile_data), cudaMemcpyHostToDevice, stream);
        for (bool down : {false, true}) {
            const int inner = down ? 640 : 2560, cols = down ? 2560 : 1280;
            std::vector<uint16_t> input(rows * inner);
            for (auto& v : input) v = strata::kernels::f16_from_f32(float(int(rng() % 2001) - 1000) / 113);
            std::vector<float> got(rows * cols + 16, -987.0f), want(rows * cols);
            cudaMemcpyAsync(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(actual, got.data(), got.size() * 4, cudaMemcpyHostToDevice, stream);
            strata::prefill::blob_dequant_f16(arena, gu, d, stream);
            gemm.f16(x, down ? d : gu, expected, 3, cols, inner, cols, 0);
            gemm.f16(x + 35 * inner, down ? d : gu, expected + 35 * cols, 1, cols, inner, cols, 0);
            strata::prefill::blob_dequant_f16(arena + stride, gu, d, stream);
            gemm.f16(x + 3 * inner, down ? d : gu, expected + 3 * cols, 32, cols, inner, cols, 0);
            strata::prefill::moe_q2_gemm(x, arena, tiles, actual, 4, stride, down, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpy(got.data(), actual, got.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(want.data(), expected, want.size() * 4, cudaMemcpyDeviceToHost);
            double se = 0, sr = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const double diff = double(got[i]) - want[i]; se += diff * diff; sr += double(want[i]) * want[i];
            }
            const double rel = std::sqrt(se / sr);
            std::printf("Q2 tiled %s: relative L2 %.8g\n", down ? "down" : "gate/up", rel);
            fail += !std::isfinite(rel) || rel > 2e-6;
            for (size_t i = want.size(); i < got.size(); ++i) fail += got[i] != -987.0f;
        }
        cudaFree(arena); cudaFree(tiles); cudaFree(x); cudaFree(actual); cudaFree(expected);
        cudaFree(b); cudaFree(gu); cudaFree(d);
    }
    for (bool bf : {false, true}) for (int T : {1, 3, 16, 23, 33}) {
        for (int N : {1, 35, 320}) {
            const int K = N == 35 ? 67 : 2560, ld = N + 7;
            std::vector<uint16_t> x(T * K), w(N * K);
            auto encode = [&](float f) { return bf ? strata::kernels::bf16_from_f32(f) : strata::kernels::f16_from_f32(f); };
            auto decode = [&](uint16_t v) { return bf ? strata::kernels::f32_from_bf16(v) : strata::kernels::f32_from_f16(v); };
            for (auto& v : x) v = encode(float(int(rng() % 2001) - 1000) / 113);
            for (auto& v : w) v = encode(float(int(rng() % 2001) - 1000) / 271);
            uint16_t *dx = nullptr, *dw = nullptr;
            float* dy = nullptr;
            cudaMalloc(&dx, x.size() * 2); cudaMalloc(&dw, w.size() * 2); cudaMalloc(&dy, T * ld * 4);
            cudaMemcpyAsync(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice, stream);
            for (float beta : {0.0f, 1.0f}) {
                std::vector<float> y(T * ld, beta == 0 ? std::numeric_limits<float>::quiet_NaN() : 0.25f);
                cudaMemcpyAsync(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice, stream);
                if (bf) gemm.bf16(dx, dw, dy, T, N, K, ld, beta);
                else gemm.f16(dx, dw, dy, T, N, K, ld, beta);
                // Record commits the CB; synchronize must still wait when there is no open CB.
                cudaEvent_t e{}; cudaEventCreate(&e); cudaEventRecord(e, stream);
                cudaStreamSynchronize(stream); cudaEventDestroy(e);
                cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
                double se = 0, sr = 0;
                for (int t = 0; t < T; ++t) for (int n = 0; n < N; ++n) {
                    double ref = beta * 0.25;
                    for (int k = 0; k < K; ++k) ref += double(decode(x[t * K + k])) * decode(w[n * K + k]);
                    const double d = y[t * ld + n] - ref;
                    se += d * d; sr += ref * ref;
                }
                const double rel = std::sqrt(se / sr);
                if (!std::isfinite(rel) || rel > 5e-5) {
                    std::printf("FAIL %s %dx%dx%d beta %.0f: rel %.8g\n", bf ? "bf16" : "f16", T, N, K, beta, rel);
                    ++fail;
                }
                for (int t = 0; t < T; ++t) for (int n = N; n < ld; ++n)
                    fail += beta == 0 ? !std::isnan(y[t * ld + n]) : y[t * ld + n] != 0.25f;
            }
            cudaFree(dx); cudaFree(dw); cudaFree(dy);
        }
    }
    cudaStreamDestroy(stream);
    std::printf("Metal GEMM: %d failures (60 FP16/BF16 cases, tails, strided outputs, beta=0/1)\n", fail);
    return fail != 0;
}
