// Compare ARM's vectorized integer dots with an independent scalar Q2_0 reference.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

int main() {
    std::mt19937 rng(20261003);
    int bad = 0;
    std::array<c::ActQ, 8> acts{};
    const c::ActQ* ap[8];
    std::array<std::array<float, 7>, 8> out{};
    float* yp[8];
    for (int t = 0; t < 8; ++t) {
        ap[t] = &acts[t]; yp[t] = out[t].data();
        for (int b = 0; b < c::MAXC; ++b) {
            int sum = 0;
            for (int j = 0; j < c::QKA; ++j) {
                const int q = int(rng() % 256) - 128;
                acts[t].q[b * c::QKA + j] = (int8_t) q;
                sum += q;
            }
            acts[t].scale[b] = float(1 + rng() % 31) / 4096;
            acts[t].hx[b] = acts[t].scale[b] * float(sum);
        }
    }
    for (int nb : {1, 3, 10, 40}) {
        const size_t stride = nb * 18 + 6;
        std::vector<uint8_t> weight(3 + 7 * stride);
        uint8_t* w = weight.data() + 3;  // deliberately unaligned, including the FP16 scales
        for (int r = 0; r < 7; ++r) for (int b = 0; b < nb; ++b) {
            const uint16_t scales[] = {0, 1, 0x03ff, 0x1400, 0x2400, 0x9400};
            const uint16_t scale = scales[rng() % 6];
            std::memcpy(w + r * stride + b * 18, &scale, 2);
            for (int j = 0; j < 16; ++j) w[r * stride + b * 18 + 2 + j] = (uint8_t) rng();
        }
        for (int nt = 1; nt <= 8; ++nt) {
            for (auto& row : out) row.fill(9876.0f);
            c::q2_0_gguf_rows_multi(w, stride, nb, ap, nt, yp, 1, 6);
            for (int t = 0; t < nt; ++t) {
                bad += out[t][0] != 9876.0f || out[t][6] != 9876.0f;
                for (int r = 1; r < 6; ++r) {
                    float expected = 0;
                    for (int b = 0; b < nb; ++b) {
                        const uint8_t* block = w + r * stride + b * 18;
                        uint16_t scale;
                        std::memcpy(&scale, block, 2);
                        const float d = strata::kernels::f32_from_f16(scale);
                        for (int h = 0; h < 2; ++h) {
                            int dot = 0;
                            const int chunk = 2 * b + h;
                            for (int j = 0; j < 32; ++j)
                                dot += ((block[2 + h * 8 + j / 4] >> (2 * (j % 4))) & 3) *
                                       acts[t].q[chunk * 32 + j];
                            expected += d * (acts[t].scale[chunk] * float(dot) - acts[t].hx[chunk]);
                        }
                    }
                    if (std::memcmp(&expected, &out[t][r], sizeof(float)) != 0) ++bad;
                }
            }
        }
    }
    std::printf("Q2 NEON vs scalar: %d failures (unaligned rows, subnormal scales, 1..8 tokens)\n", bad);
    return bad != 0;
}
