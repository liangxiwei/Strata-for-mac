// Expert GEMM on the IQ3_S pack shapes: compare every output before timing.
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
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"tiles.metallib"] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> P;
        auto pso = [&](const std::string& n) {
            if (P.count(n)) return P[n];
            id<MTLFunction> f = [lib newFunctionWithName:@(n.c_str())];
            if (!f) { printf("no %s\n", n.c_str()); exit(1); }
            NSError* pe = nil; id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&pe];
            if (!ps) { printf("pipeline %s: %s\n", n.c_str(), pe.localizedDescription.UTF8String); exit(1); }
            return P[n] = ps;
        };
        std::mt19937 rng(21);
        const long H = 2560, FF = 640;
        const int n_exp = 512;
        for (int gu_t : {18, 21, 22, 23})
        for (int per : {20,40}) {
            const size_t gu_block = gu_t == 22 ? 82 : gu_t == 18 ? 98 : gu_t == 21 ? 110 : 136;
            const unsigned long gu_row = (H / 256) * gu_block, d_row = gu_t == 22 ? (FF / 64) * 18 : (FF / 32) * 18;
            const int dt = gu_t == 22 ? 42 : 20;
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
                    for (int r = BM == 2 ? (cnt[e] / 32)*32 : 0; r < (BM == 1 ? (cnt[e]/32)*32 : cnt[e]); r += BM == 1 ? 32 : BM == 2 ? 16 : BM) {
                        t.push_back((int) (uint32_t) o); t.push_back(off[e] + r); t.push_back(std::min(BM == 1 ? 32 : BM == 2 ? 16 : BM, cnt[e] - r)); t.push_back((int) (uint32_t) (o >> 32));
                    }
                }
                return t;
            };
            std::map<int, id<MTLBuffer>> TB; std::map<int, int> NT;
            for (int BM : {1, 2, 16, 32, 64}) { auto t = make_tiles(BM); NT[BM] = (int) t.size() / 4; TB[BM] = [dev newBufferWithBytes:t.data() length:t.size() * 4 options:0]; }
            for (bool down : {false, true}) {
                const long K = down ? FF : H, N = down ? H : 2 * FF;
                std::vector<__fp16> xv((size_t) T * K);
                std::normal_distribution<float> nd;
                for (auto& v : xv) v = (__fp16) nd(rng);
                id<MTLBuffer> X = [dev newBufferWithBytes:xv.data() length:xv.size() * 2 options:0];
                id<MTLBuffer> Y0 = [dev newBufferWithLength:(size_t) T * N * 4 options:0], Y1 = [dev newBufferWithLength:(size_t) T * N * 4 options:0];
                const unsigned long rb = down ? d_row : gu_row, wo = down ? down_off : up_off;
                const std::string base = down ? "native_gemm2_down_"+std::to_string(dt)+"_r16" : "native_gemm2_gu_"+std::to_string(gu_t)+"_r16";
                const std::string tag = down ? "dn" + std::to_string(dt) : "gu" + std::to_string(gu_t);
                if (down && (gu_t == 21 || gu_t == 23)) continue;   // the down kernel is the same for both gate/up types
                auto launch = [&](id<MTLComputeCommandEncoder> e, const std::string& kn, id<MTLBuffer> Y) {
                    const bool full = kn=="full32"; const bool tail = kn=="tail16";
                    const int BM = full ? 1 : tail ? 2 : 16, BN=64, TH=full ? 256 : 128;
                    if (!NT[BM]) return;
                    const std::string kernel = full ? base.substr(0,base.size()-2)+"32" : base;
                    [e setComputePipelineState:pso(kernel)];
                    [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:A offset:0 atIndex:1]; [e setBuffer:TB[BM] offset:0 atIndex:2];
                    [e setBuffer:Y offset:0 atIndex:3]; [e setBytes:&H length:8 atIndex:4]; [e setBytes:&FF length:8 atIndex:5];
                    [e setBytes:&rb length:8 atIndex:6]; [e setBytes:&wo length:8 atIndex:7];
                    [e dispatchThreadgroups:MTLSizeMake((N + BN - 1) / BN, NT[BM], 1) threadsPerThreadgroup:MTLSizeMake(TH, 1, 1)];
                };
                auto run = [&](const std::string& kn, id<MTLBuffer> Y, int reps) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    for (int r = 0; r < reps; ++r) {if (kn=="adaptive") {launch(e,"full32",Y);launch(e,"tail16",Y);}else launch(e,kn,Y);}
                    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / reps;
                };
                std::vector<std::string> ks = {base};
                ks.push_back("adaptive");
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
                for (const auto& kn : ks) { auto v = t[kn]; std::sort(v.begin(), v.end()); printf("  %s %.2f ms%s", kn == base ? "orig" : kn.c_str(), v[v.size() / 2], d[kn] ? " (DIFF)" : ""); }
                printf("\n"); fflush(stdout);
                for (const auto& kn : ks) if (d[kn]) printf("   %s: %ld differences\n", kn.c_str(), d[kn]);
            }
        }
    }
    return 0;
}
