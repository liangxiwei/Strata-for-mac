// Prompt attention: prompt_attn_kernel_m0 vs pa2_m0, bitwise on every output, timed per chunk-sized launch.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        const int rounds = argc > 1 ? atoi(argv[1]) : 5;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"attn.metallib"] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto pso = [&](const char* n) {
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:nil];
            if (!p) { printf("no %s\n", n); exit(1); }
            printf("%s: maxThreads %lu, static threadgroup %lu B\n", n, (unsigned long) p.maxTotalThreadsPerThreadgroup,
                   (unsigned long) p.staticThreadgroupMemoryLength);
            return p;
        };
        id<MTLComputePipelineState> A = pso("prompt_attn_kernel_m0"), B = pso("pa2_m0"), C = pso("pa3_m0");
        std::mt19937 rng(9);
        std::normal_distribution<float> nd;
        const int HD = 256, G = 12, KVH = 2, NH = KVH * G, page = 64;
        for (int specials : {0, 1}) for (int pos0 : {9000, 1500, 30000}) {
            const int nq = 1024, cells = pos0 + nq + 64, pages = (cells + page - 1) / page;
            const long cap = 2051;
            std::vector<__fp16> kp((size_t) pages * KVH * page * HD), vp(kp.size());
            for (auto& v : kp) v = (__fp16) (nd(rng) * 0.5f);
            for (auto& v : vp) v = (__fp16) nd(rng);
            if (specials) kp[5 * HD + 3] = (__fp16) INFINITY;         // one non-finite K entry
            std::vector<int> table(pages);
            for (int p = 0; p < pages; ++p) table[p] = p;
            if (specials) table[3] = -1;                              // a non-resident page: masked cells
            std::vector<float> qv((size_t) nq * NH * HD);
            for (auto& v : qv) v = nd(rng);
            std::vector<int> ids((size_t) nq * cap, 0), steps((size_t) nq * 4);
            std::vector<int> base;                                    // shared selection, perturbed per query
            for (int i = 0; i < 520; ++i) base.push_back((int) (rng() % (pos0 / 4 + 1)));
            for (int i = 0; i < nq; ++i) {
                const int pos = pos0 + i, n_kv = pos + 1, width = (int) std::min<long>(n_kv, cap);
                steps[i * 4 + 0] = pos; steps[i * 4 + 1] = n_kv; steps[i * 4 + 2] = n_kv / 4; steps[i * 4 + 3] = width;
                std::set<int> s;
                for (int c = std::max(0, n_kv - 8); c < n_kv; ++c) s.insert(c);
                for (int b : base) { if (rng() % 10 == 0) b = (int) (rng() % (n_kv / 4 + 1)); for (int c = 0; c < 4; ++c) if (b * 4 + c < n_kv) s.insert(b * 4 + c); if ((int) s.size() >= width) break; }
                while ((int) s.size() < width) s.insert((int) (rng() % n_kv));
                int j = 0;
                for (int c : s) { if (j >= width) break; ids[(size_t) i * cap + j++] = c; }
            }
            id<MTLBuffer> Q = [dev newBufferWithBytes:qv.data() length:qv.size() * 4 options:0];
            id<MTLBuffer> KP = [dev newBufferWithBytes:kp.data() length:kp.size() * 2 options:0];
            id<MTLBuffer> VP = [dev newBufferWithBytes:vp.data() length:vp.size() * 2 options:0];
            id<MTLBuffer> PT = [dev newBufferWithBytes:table.data() length:table.size() * 4 options:0];
            id<MTLBuffer> ID = [dev newBufferWithBytes:ids.data() length:ids.size() * 4 options:0];
            id<MTLBuffer> ST = [dev newBufferWithBytes:steps.data() length:steps.size() * 4 options:0];
            id<MTLBuffer> D = [dev newBufferWithLength:64 options:0];
            id<MTLBuffer> O1 = [dev newBufferWithLength:qv.size() * 4 options:0], O2 = [dev newBufferWithLength:qv.size() * 4 options:0], O3 = [dev newBufferWithLength:qv.size() * 4 options:0];
            memset(O1.contents, 0xAA, qv.size() * 4); memset(O2.contents, 0x55, qv.size() * 4); memset(O3.contents, 0x33, qv.size() * 4);
            const int nkvh = KVH, ps = page;
            auto run = [&](id<MTLComputePipelineState> p, id<MTLBuffer> O, int reps) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                for (int r = 0; r < reps; ++r) {
                    [e setComputePipelineState:p];
                    [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:KP offset:0 atIndex:1]; [e setBuffer:VP offset:0 atIndex:2];
                    for (int b = 3; b <= 8; ++b) [e setBuffer:D offset:0 atIndex:b];
                    [e setBuffer:PT offset:0 atIndex:9]; [e setBuffer:ID offset:0 atIndex:10]; [e setBuffer:ST offset:0 atIndex:11];
                    [e setBytes:&nkvh length:4 atIndex:12]; [e setBytes:&ps length:4 atIndex:13]; [e setBytes:&cap length:8 atIndex:14];
                    [e setBuffer:O offset:0 atIndex:15];
                    [e dispatchThreadgroups:MTLSizeMake(nq, KVH, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / reps;
            };
            run(A, O1, 1); run(B, O2, 1); run(C, O3, 1);
            long d = 0, d3 = 0, nonfinite = 0;
            for (size_t i = 0; i < qv.size(); ++i) {
                d += ((uint32_t*) O1.contents)[i] != ((uint32_t*) O2.contents)[i];
                d3 += ((uint32_t*) O1.contents)[i] != ((uint32_t*) O3.contents)[i];
                nonfinite += !std::isfinite(((float*) O1.contents)[i]);
            }
            std::vector<double> ta, tb, tc;
            for (int r = 0; r < rounds; ++r) { ta.push_back(run(A, O1, 2)); tb.push_back(run(B, O2, 2)); tc.push_back(run(C, O3, 2)); }
            std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end()); std::sort(tc.begin(), tc.end());
            printf("%s pos %5d: original %.2f ms  pa2 %.2f ms  pa3 %.2f ms  (differ: pa2 %ld, pa3 %ld of %zu; %ld non-finite in the reference)\n",
                   specials ? "specials" : "finite  ", pos0, ta[ta.size() / 2], tb[tb.size() / 2], tc[tc.size() / 2], d, d3, qv.size(), nonfinite);
        }
    }
    return 0;
}
