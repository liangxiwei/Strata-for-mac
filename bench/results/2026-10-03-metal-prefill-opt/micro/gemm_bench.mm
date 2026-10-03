// Prefill GEMM: production pfl_gemm_f16/bf16 vs tiled variants, bitwise on every output, timed per call.
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

static uint16_t f2h(float f) { __fp16 h = (__fp16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; }
static uint16_t f2b(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

int main(int argc, char** argv) {
    @autoreleasepool {
        const int rounds = argc > 1 ? atoi(argv[1]) : 5;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"gemm.metallib"] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> P;
        auto pso = [&](const std::string& n) {
            if (P.count(n)) return P[n];
            id<MTLFunction> f = [lib newFunctionWithName:@(n.c_str())];
            if (!f) { printf("no %s\n", n.c_str()); exit(1); }
            return P[n] = [dev newComputePipelineStateWithFunction:f error:nil];
        };
        std::mt19937 rng(3);
        std::normal_distribution<float> nd;
        struct Shape { bool bf; uint M, N, K; float beta; };
        const std::vector<Shape> shapes = {
            {false, 1024, 10240, 2560, 0}, {false, 1024, 2560, 6144, 0}, {false, 1024, 6144, 2560, 0},
            {false, 1024, 12288, 2560, 0}, {false, 1024, 640, 2560, 0}, {false, 777, 2560, 6144, 0},
            {false, 1024, 2560, 6144, 1}, {true, 1024, 320, 10240, 0}, {true, 1024, 10240, 320, 0},
            {true, 1024, 512, 2560, 0}, {true, 1024, 32, 2560, 0}, {true, 777, 320, 10240, 1}};
        printf("type,M,N,K,beta,kernel,us,TFLOPS,bitdiff\n");
        for (const Shape& sh : shapes) {
            const int copies = 4;
            std::vector<uint16_t> xh((size_t) sh.M * sh.K), wh((size_t) sh.N * sh.K);
            for (auto& v : xh) v = sh.bf ? f2b(nd(rng)) : f2h(nd(rng) * 0.5f);
            id<MTLBuffer> X = [dev newBufferWithBytes:xh.data() length:xh.size() * 2 options:0];
            std::vector<id<MTLBuffer>> W;
            for (int c = 0; c < copies; ++c) {
                for (auto& v : wh) v = sh.bf ? f2b(nd(rng) * 0.05f) : f2h(nd(rng) * 0.05f);
                W.push_back([dev newBufferWithBytes:wh.data() length:wh.size() * 2 options:0]);
            }
            std::vector<float> yinit((size_t) sh.M * sh.N);
            for (auto& v : yinit) v = nd(rng);
            id<MTLBuffer> Y0 = [dev newBufferWithLength:yinit.size() * 4 options:0], Y1 = [dev newBufferWithLength:yinit.size() * 4 options:0];
            auto launch = [&](id<MTLComputeCommandEncoder> e, const std::string& kn, int c, id<MTLBuffer> Y) {
                int BM = 16, BN = 32, TH = 128;
                if (kn.rfind("pg_", 0) == 0) { const auto p = kn.rfind('_'); BM = atoi(kn.c_str() + p + 1); BN = atoi(kn.c_str() + kn.find('x', p) + 1); }
                if (kn.rfind("pq_", 0) == 0) {
                    const auto p1 = kn.find('_', 3), p2 = kn.rfind('_');   // pq_<type>_<BM>x<BN>_<SGM><SGN>
                    BM = atoi(kn.c_str() + p1 + 1); BN = atoi(kn.c_str() + kn.find('x', p1) + 1);
                    TH = 32 * (kn[p2 + 1] - '0') * (kn[p2 + 2] - '0');
                }
                [e setComputePipelineState:pso(kn)];
                [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:W[c % copies] offset:0 atIndex:1]; [e setBuffer:Y offset:0 atIndex:2];
                [e setBytes:&sh.M length:4 atIndex:3]; [e setBytes:&sh.N length:4 atIndex:4]; [e setBytes:&sh.K length:4 atIndex:5];
                [e setBytes:&sh.N length:4 atIndex:6]; [e setBytes:&sh.beta length:4 atIndex:7];
                [e dispatchThreadgroups:MTLSizeMake((sh.N + BN - 1) / BN, (sh.M + BM - 1) / BM, 1) threadsPerThreadgroup:MTLSizeMake(TH, 1, 1)];
            };
            auto run = [&](const std::string& kn, int reps, id<MTLBuffer> Y) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                for (int r = 0; r < reps; ++r) launch(e, kn, r, Y);
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
            };
            const std::string base = sh.bf ? "pfl_gemm_bf16" : "pfl_gemm_f16", pre = sh.bf ? "pg_bf16_" : "pg_f16_";
            std::vector<std::string> ks = {base};
            if (sh.bf) for (const char* t : {"pq_bf16_64x64_24", "pq_bf16p_64x64_24", "pq_bf16p_64x32_22"}) ks.push_back(t);
            else for (const char* t : {"pg_f16_64x64", "pq_f16p_64x64_22", "pq_f16p_128x64_42"}) ks.push_back(t);
            memcpy(Y0.contents, yinit.data(), yinit.size() * 4);
            run(base, 1, Y0);
            std::map<std::string, long> diff;
            for (const auto& kn : ks) {
                memcpy(Y1.contents, yinit.data(), yinit.size() * 4);
                run(kn, 1, Y1);
                long d = 0;
                for (size_t i = 0; i < yinit.size(); ++i) d += ((uint32_t*) Y0.contents)[i] != ((uint32_t*) Y1.contents)[i];
                diff[kn] = d;
            }
            std::map<std::string, std::vector<double>> t;
            for (int r = 0; r < rounds; ++r) for (const auto& kn : ks) t[kn].push_back(run(kn, 8, Y1));
            const double flop = 2.0 * sh.M * sh.N * sh.K;
            for (const auto& kn : ks) {
                auto v = t[kn]; std::sort(v.begin(), v.end());
                printf("%s,%u,%u,%u,%g,%s,%.1f,%.2f,%ld\n", sh.bf ? "bf16" : "f16", sh.M, sh.N, sh.K, sh.beta, kn.c_str(),
                       v[v.size() / 2], flop / (v[v.size() / 2] * 1e6), diff[kn]);
            }
            fflush(stdout);
        }
    }
    return 0;
}
