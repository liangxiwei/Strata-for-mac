// Bitwise + timing harness for fused_gr variants (n_embd 2560, 4 streams, lr 320).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

int main(int argc, char** argv) {
    @autoreleasepool {
        const int rounds = argc > 2 ? atoi(argv[2]) : 7;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> P;
        auto pso = [&](const std::string& n) {
            if (P.count(n)) return P[n];
            id<MTLFunction> f = [lib newFunctionWithName:@(n.c_str())];
            if (!f) { printf("no %s\n", n.c_str()); exit(1); }
            return P[n] = [dev newComputePipelineStateWithFunction:f error:nil];
        };
        const int N = 2560, HC = 4, D = N * HC, LR = 320;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd;
        auto fbuf = [&](size_t n, float scale) {
            std::vector<float> v(n); for (auto& x : v) x = nd(rng) * scale;
            return [dev newBufferWithBytes:v.data() length:n * 4 options:MTLResourceStorageModeShared];
        };
        auto bbuf = [&](size_t n, float scale) {
            std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(nd(rng) * scale);
            return [dev newBufferWithBytes:v.data() length:n * 2 options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> R = fbuf(D, 1.0f), bo = fbuf(N, 0.5f), inj = fbuf(4, 1.0f), wn = fbuf(D, 1.0f);
        const int copies = 32;
        std::vector<id<MTLBuffer>> wd, wu;
        id<MTLBuffer> wi = bbuf((size_t) 4 * D, 0.02f);
        for (int c = 0; c < copies; ++c) { wd.push_back(bbuf((size_t) LR * D, 0.02f)); wu.push_back(bbuf((size_t) D * LR, 0.05f)); }
        struct Out { id<MTLBuffer> rs, xn, lo, io, Rout, mixed; };
        auto mk = [&]() { return Out{[dev newBufferWithLength:16 options:0], [dev newBufferWithLength:D * 4 options:0],
                                     [dev newBufferWithLength:LR * 4 + 64 options:0], [dev newBufferWithLength:64 options:0],
                                     [dev newBufferWithLength:D * 4 options:0], [dev newBufferWithLength:N * 4 options:0]}; };
        Out A = mk(), B = mk();
        const float eps = 1e-6f;
        const int apply = 1;
        auto norm = [&](id<MTLComputeCommandEncoder> e, const std::string& k, Out& o) {
            [e setComputePipelineState:pso(k)];
            [e setBuffer:R offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1]; [e setBuffer:inj offset:0 atIndex:2];
            [e setBuffer:wn offset:0 atIndex:3]; [e setBytes:&eps length:4 atIndex:4]; [e setBytes:&apply length:4 atIndex:5];
            [e setBuffer:o.rs offset:0 atIndex:6]; [e setBuffer:o.xn offset:0 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto down = [&](id<MTLComputeCommandEncoder> e, const std::string& k, Out& o, int c) {
            int warps = 8; sscanf(k.c_str(), "gr_down_w%d", &warps);
            [e setComputePipelineState:pso(k)];
            [e setBuffer:wd[c % copies] offset:0 atIndex:0]; [e setBuffer:wi offset:0 atIndex:1];
            [e setBuffer:o.xn offset:0 atIndex:2]; [e setBuffer:o.lo offset:0 atIndex:3]; [e setBuffer:o.io offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(LR / warps + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * warps, 1, 1)];
        };
        auto up = [&](id<MTLComputeCommandEncoder> e, const std::string& k, Out& o, int c) {
            int warps = 8, cols = 16; sscanf(k.c_str(), "gr_up_w%d_c%d", &warps, &cols);
            [e setComputePipelineState:pso(k)];
            [e setBuffer:wu[c % copies] offset:0 atIndex:0]; [e setBuffer:o.lo offset:0 atIndex:1]; [e setBuffer:R offset:0 atIndex:2];
            [e setBuffer:o.Rout offset:0 atIndex:3]; [e setBuffer:bo offset:0 atIndex:4]; [e setBuffer:inj offset:0 atIndex:5];
            [e setBuffer:wn offset:0 atIndex:6]; [e setBuffer:o.rs offset:0 atIndex:7]; [e setBuffer:o.mixed offset:0 atIndex:8];
            [e setBytes:&apply length:4 atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake(N / cols, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * warps, 1, 1)];
        };
        auto run = [&](auto&& f) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            f(e);
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return cb.GPUEndTime - cb.GPUStartTime;
        };
        auto same = [&](id<MTLBuffer> a, id<MTLBuffer> b, size_t n) {
            return (long) std::count_if((const uint32_t*) a.contents, (const uint32_t*) a.contents + n,
                                        [&, i = (size_t) 0](uint32_t v) mutable { return v != ((const uint32_t*) b.contents)[i++]; });
        };
        std::vector<std::string> dns = {"fused_gr_down_kernel", "gr_down_w4", "gr_down_w8", "gr_down_w16"}, ups = {"fused_gr_up_kernel", "gr_up_w4_c8", "gr_up_w4_c16", "gr_up_w4_c32", "gr_up_w8_c8", "gr_up_w8_c16", "gr_up_w8_c32", "gr_up_w16_c8", "gr_up_w16_c16", "gr_up_w16_c32"};
        run([&](id<MTLComputeCommandEncoder> e) { norm(e, "fused_gr_norm1_kernel", A); down(e, dns[0], A, 0); up(e, ups[0], A, 0); });
        for (const auto& dk : dns) for (const auto& uk : ups) {
            run([&](id<MTLComputeCommandEncoder> e) { norm(e, "fused_gr_norm1_kernel", B); down(e, dk, B, 0); up(e, uk, B, 0); });
            long different = same(A.rs, B.rs, 4) + same(A.xn, B.xn, D) + same(A.lo, B.lo, LR) + same(A.io, B.io, 4) + same(A.Rout, B.Rout, D) + same(A.mixed, B.mixed, N);
            if (different) { fprintf(stderr, "%s/%s different=%ld\n", dk.c_str(), uk.c_str(), different); return 2; }
        }
        printf("# chain_checks=%zu bitdiff=0 apply=1 copies=32 reps=96\n", dns.size() * ups.size());
        const int reps = 96;
        std::map<std::string, std::vector<double>> t;
        for (int r = 0; r < rounds; ++r) {
            if (r % 2) { std::reverse(dns.begin(), dns.end()); std::reverse(ups.begin(), ups.end()); }
            for (const auto& k : dns) t[k].push_back(run([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) down(e, k, A, i); }));
            for (const auto& k : ups) t[k].push_back(run([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) up(e, k, A, i); }));
        }
        for (auto& [k, v] : t) {
            std::sort(v.begin(), v.end());
            printf("%-24s %8.2f us/dispatch\n", k.c_str(), v[v.size() / 2] * 1e6 / reps);
        }
    }
    return 0;
}
