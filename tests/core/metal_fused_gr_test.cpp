#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// Real allocation adjacency: the old eight-warp inject block zeroed the next four live gates.
// Compare the composed operation with an independent FP64 reference and guard both ends of the gates.
int main() {
    using namespace strata::kernels;
    constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
    std::mt19937 rng(71);
    auto random = [&] { return float(int(rng() % 2001) - 1000) / 1000; };
    std::vector<float> r(D), norm(D), bo(N);
    std::vector<uint16_t> wd(LR * D), wu(D * LR), wi(HC * D);
    for (auto& v : r) v = random();
    for (auto& v : norm) v = 0.8f + 0.1f * random();
    for (auto& v : bo) v = random();
    for (auto* w : {&wd, &wu, &wi}) for (auto& v : *w) v = bf16_from_f32(0.02f * random());
    float *dr, *dn, *db, *gates, *lo, *rs, *out, *xn;
    uint16_t *dd, *du, *di;
    cudaStream_t stream{}; cudaStreamCreate(&stream);
    auto alloc = [](auto** p, size_t n) { cudaMalloc(p, n * sizeof(**p)); };
    alloc(&dr, D); alloc(&dn, D); alloc(&db, N); alloc(&gates, 12);
    alloc(&lo, LR); alloc(&rs, HC); alloc(&out, N); alloc(&xn, D);
    alloc(&dd, wd.size()); alloc(&du, wu.size()); alloc(&di, wi.size());
    cudaMemcpy(dn, norm.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bo.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(du, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(di, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    int fail = 0;
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    for (bool apply : {false, true}) for (bool inject : {false, true}) for (bool multi : {false, true}) {
        std::vector<float> guard(12, 123.0f);
        for (int c = 0; c < HC; ++c) guard[8 + c] = (c - 2) * 1.1f;
        cudaMemcpyAsync(dr, r.data(), D * 4, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(gates, guard.data(), 48, cudaMemcpyHostToDevice, stream);
        FusedGrArgs a;
        a.R = a.R_out = dr; a.apply = apply; a.bo_prev = db; a.inj_prev = gates + 8;
        a.w_norm = dn; a.w_down = dd; a.w_up = du; a.w_inject = inject ? di : nullptr;
        a.lo = lo; a.rs = rs; a.inject_out = inject ? gates + 4 : nullptr; a.mixed = out;
        if (multi) fused_gr_read_multi(&a, 1, xn, stream);
        else fused_gr_read(a, stream);
        cudaStreamSynchronize(stream);
        std::vector<float> got(N), gr(D), gg(12);
        cudaMemcpy(got.data(), out, N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gr.data(), dr, D * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gg.data(), gates, 48, cudaMemcpyDeviceToHost);
        std::vector<double> xr(D), xl(LR);
        for (int c = 0; c < HC; ++c) {
            double sq = 0;
            for (int d = 0; d < N; ++d) {
                const int i = c * N + d;
                xr[i] = r[i] + (apply ? bo[d] * 2 * sig(guard[8 + c] / 4.0) : 0);
                sq += xr[i] * xr[i];
                fail += std::abs(gr[i] - xr[i]) > 2e-6;
            }
            const double scale = 1 / std::sqrt(sq / N + 1e-6);
            for (int d = 0; d < N; ++d) xr[c * N + d] *= norm[c * N + d] * scale;
        }
        for (int j = 0; j < LR; ++j) {
            double v = 0;
            for (int i = 0; i < D; ++i) v += f32_from_bf16(wd[j * D + i]) * xr[i];
            v /= HC; xl[j] = v * sig(v);
        }
        double se = 0, sr = 0;
        for (int d = 0; d < N; ++d) {
            double ref = 0;
            for (int c = 0; c < HC; ++c) {
                double v = 0;
                for (int j = 0; j < LR; ++j) v += f32_from_bf16(wu[(c * N + d) * LR + j]) * xl[j];
                ref += xr[c * N + d] * sig(v) / HC;
            }
            se += (got[d] - ref) * (got[d] - ref); sr += ref * ref;
        }
        const double rel = std::sqrt(se / sr);
        fail += !std::isfinite(rel) || rel > 3e-6;
        for (int i = 0; i < 12; ++i) if (i < 4 || i >= 8 || !inject) fail += gg[i] != guard[i];
        std::printf("fused GR apply=%d inject=%d multi=%d: relative L2 %.3g\n", apply, inject, multi, rel);
    }
    for (void* p : {(void*)dr, (void*)dn, (void*)db, (void*)gates, (void*)lo, (void*)rs, (void*)out,
                    (void*)xn, (void*)dd, (void*)du, (void*)di}) cudaFree(p);
    cudaStreamDestroy(stream);
    std::printf("Metal fused GR: %d failures\n", fail);
    return fail != 0;
}
