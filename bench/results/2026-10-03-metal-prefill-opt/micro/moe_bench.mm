// Prefill expert GEMM: native_gemm_{gu_22,gu_16,down_42} (16 x 32 tiles) vs mg_* variants, bitwise, timed.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        const int rounds = argc > 1 ? atoi(argv[1]) : 5;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"moe.metallib"] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> P;
        auto pso = [&](const std::string& n) {
            if (P.count(n)) return P[n];
            id<MTLFunction> f = [lib newFunctionWithName:@(n.c_str())];
            if (!f) { printf("no %s\n", n.c_str()); exit(1); }
            return P[n] = [dev newComputePipelineStateWithFunction:f error:nil];
        };
        std::mt19937 rng(21);
        const long H = 2560, FF = 640;
        const int n_exp = 256;
        for (int gu_t : {22, 16})
        for (int per : {20, 80}) {
            const size_t gu_block = gu_t == 22 ? 82 : 66;
            const unsigned long gu_row = (H / 256) * gu_block, d_row = (FF / 64) * 18;
            const unsigned long up_off = FF * gu_row, down_off = 2 * up_off, blob = down_off + H * d_row;
            std::vector<uint8_t> arena(blob * n_exp);
            for (auto& v : arena) v = (uint8_t) rng();
            for (int e = 0; e < n_exp; ++e) {
                for (size_t i = 0; i < 2 * up_off; i += gu_block) { const __fp16 d = (__fp16) ((int(rng() % 63) - 31) / 8192.0f); std::memcpy(&arena[e * blob + i], &d, 2); }
                for (size_t i = 0; i < H * d_row; i += 18) { const __fp16 d = (__fp16) ((int(rng() % 63) - 31) / 8192.0f); std::memcpy(&arena[e * blob + down_off + i], &d, 2); }
            }
            id<MTLBuffer> A = [dev newBufferWithBytes:arena.data() length:arena.size() options:0];
            // skewed routing: counts ~ mean `per`, a long tail
            std::vector<int> cnt(n_exp);
            std::exponential_distribution<double> ex(1.0 / per);
            int T = 0;
            for (auto& c : cnt) { c = std::max(1, (int) ex(rng)); T += c; }
            std::vector<int> off(n_exp + 1, 0);
            for (int e = 0; e < n_exp; ++e) off[e + 1] = off[e] + cnt[e];
            auto make_tiles = [&](int BM) {
                std::vector<int> t;
                for (int e = 0; e < n_exp; ++e) {
                    const unsigned long o = (unsigned long) e * blob;
                    for (int r = 0; r < cnt[e]; r += BM) {
                        t.push_back((int) (uint32_t) o); t.push_back(off[e] + r); t.push_back(std::min(BM, cnt[e] - r)); t.push_back((int) (uint32_t) (o >> 32));
                    }
                }
                return t;
            };
            std::map<int, id<MTLBuffer>> TB; std::map<int, int> NT;
            for (int BM : {16, 32, 64}) { auto t = make_tiles(BM); NT[BM] = (int) t.size() / 4; TB[BM] = [dev newBufferWithBytes:t.data() length:t.size() * 4 options:0]; }
            for (bool down : {false, true}) {
                const long K = down ? FF : H, N = down ? H : 2 * FF;
                std::vector<__fp16> xv((size_t) T * K);
                std::normal_distribution<float> nd;
                for (auto& v : xv) v = (__fp16) nd(rng);
                id<MTLBuffer> X = [dev newBufferWithBytes:xv.data() length:xv.size() * 2 options:0];
                id<MTLBuffer> Y0 = [dev newBufferWithLength:(size_t) T * N * 4 options:0], Y1 = [dev newBufferWithLength:(size_t) T * N * 4 options:0];
                const unsigned long rb = down ? d_row : gu_row, wo = down ? down_off : up_off;
                const std::string base = down ? "native_gemm_down_42" : "native_gemm_gu_" + std::to_string(gu_t);
                const std::string tag = down ? "dn42" : "gu" + std::to_string(gu_t);
                if (down && gu_t == 16) continue;   // the down kernel is the same for both gate/up types
                auto launch = [&](id<MTLComputeCommandEncoder> e, const std::string& kn, id<MTLBuffer> Y) {
                    int BM = 16, BN = 32, TH = 128;
                    if (kn.rfind("mg_", 0) == 0) {
                        const auto p1 = kn.find('_', 3), p2 = kn.rfind('_');
                        BM = atoi(kn.c_str() + p1 + 1); BN = atoi(kn.c_str() + kn.find('x', p1) + 1);
                        TH = 32 * (kn[p2 + 1] - '0') * (kn[p2 + 2] - '0');
                    }
                    [e setComputePipelineState:pso(kn)];
                    [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:A offset:0 atIndex:1]; [e setBuffer:TB[BM] offset:0 atIndex:2];
                    [e setBuffer:Y offset:0 atIndex:3]; [e setBytes:&H length:8 atIndex:4]; [e setBytes:&FF length:8 atIndex:5];
                    [e setBytes:&rb length:8 atIndex:6]; [e setBytes:&wo length:8 atIndex:7];
                    [e dispatchThreadgroups:MTLSizeMake((N + BN - 1) / BN, NT[BM], 1) threadsPerThreadgroup:MTLSizeMake(TH, 1, 1)];
                };
                auto run = [&](const std::string& kn, id<MTLBuffer> Y, int reps) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    for (int r = 0; r < reps; ++r) launch(e, kn, Y);
                    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / reps;
                };
                std::vector<std::string> ks = {base};
                for (const char* v : {"16x32_22", "16x64_22", "32x32_22", "32x64_22", "32x64_24", "64x64_24", "16x128_14", "32x128_24"})
                    ks.push_back("mg_" + tag + "_" + v);
                memset(Y0.contents, 0xAA, (size_t) T * N * 4);
                run(base, Y0, 1);
                std::map<std::string, std::vector<double>> t;
                std::map<std::string, long> d;
                for (const auto& kn : ks) {
                    memset(Y1.contents, 0x55, (size_t) T * N * 4);
                    run(kn, Y1, 1);
                    long dd = 0;
                    for (size_t i = 0; i < (size_t) T * N; ++i) dd += ((uint32_t*) Y0.contents)[i] != ((uint32_t*) Y1.contents)[i];
                    d[kn] = dd;
                }
                for (int r = 0; r < rounds; ++r) for (const auto& kn : ks) t[kn].push_back(run(kn, Y1, 3));
                printf("%s rows %d (%.1f per expert):", base.c_str(), T, (double) T / n_exp);
                for (const auto& kn : ks) { auto v = t[kn]; std::sort(v.begin(), v.end()); printf("  %s %.2f ms%s", kn == base ? "orig" : kn.c_str() + 3 + tag.size() + 1, v[v.size() / 2], d[kn] ? " (DIFF)" : ""); }
                printf("\n"); fflush(stdout);
                for (const auto& kn : ks) if (d[kn]) printf("   %s: %ld differences\n", kn.c_str(), d[kn]);
            }
        }
    }
    return 0;
}
