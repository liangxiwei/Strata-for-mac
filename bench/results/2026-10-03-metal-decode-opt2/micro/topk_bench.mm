// block_topk: production kernel vs variant, bitwise on the written ids, timed per query.
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
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"topk.metallib"] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto pso = [&](const char* n) { return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:nil]; };
        id<MTLComputePipelineState> A = pso("block_topk_kernel"), B = pso("block_topk_v2");
        std::mt19937 rng(5);
        const long width_max = 2048 + 3, cap = 2051;
        long total = 0, diff = 0;
        for (int dist = 0; dist < 4; ++dist)
        for (long n_kv : {2052L, 2100L, 3000L, 10000L, 10001L, 10003L, 32000L, 32767L}) {
            const long n_bid = n_kv / 4, nb = n_bid + 1, max_blocks = 32768 / 4 + 2;
            std::vector<float> sc(max_blocks, 0.0f);
            std::normal_distribution<float> nd;
            for (long b = 0; b < nb; ++b) {
                float v = 0;
                if (dist == 0) for (int h = 0; h < 4; ++h) v += std::max(0.0f, nd(rng) * 3.0f);    // realistic: ReLU sums
                if (dist == 1) v = (float) (rng() % 7);                                           // heavy ties
                if (dist == 2) v = 0.0f;                                                          // all equal
                if (dist == 3) { v = nd(rng); if (rng() % 9 == 0) v = NAN; if (rng() % 11 == 0) v = -0.0f; }
                sc[b] = v;
            }
            if (n_kv % 4 != 0) sc[n_bid] += 1e9f;
            const int steps[4] = {(int) (n_kv - 1), (int) n_kv, (int) n_bid, (int) std::min(n_kv, width_max)};
            id<MTLBuffer> S = [dev newBufferWithBytes:sc.data() length:sc.size() * 4 options:0];
            id<MTLBuffer> St = [dev newBufferWithBytes:steps length:16 options:0];
            id<MTLBuffer> O1 = [dev newBufferWithLength:cap * 4 options:0], O2 = [dev newBufferWithLength:cap * 4 options:0];
            memset(O1.contents, 0xAA, cap * 4); memset(O2.contents, 0xAA, cap * 4);
            auto run = [&](id<MTLComputePipelineState> p, id<MTLBuffer> O, int reps) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                for (int r = 0; r < reps; ++r) {
                    [e setComputePipelineState:p];
                    [e setBuffer:S offset:0 atIndex:0]; [e setBuffer:St offset:0 atIndex:1];
                    [e setBytes:&max_blocks length:8 atIndex:2]; [e setBytes:&cap length:8 atIndex:3];
                    [e setBuffer:O offset:0 atIndex:4];
                    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
            };
            run(A, O1, 1); run(B, O2, 1);
            long d = 0;
            for (long i = 0; i < cap; ++i) d += ((int*) O1.contents)[i] != ((int*) O2.contents)[i];
            total += cap; diff += d;
            if (dist == 0) {
                std::vector<double> ta, tb;
                for (int r = 0; r < 7; ++r) { ta.push_back(run(A, O1, 64)); tb.push_back(run(B, O2, 64)); }
                std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
                printf("n_kv %6ld: original %7.2f us  v2 %7.2f us  diff %ld\n", n_kv, ta[3], tb[3], d);
            } else if (d) printf("dist %d n_kv %ld: %ld differ\n", dist, n_kv, d);
        }
        printf("block_topk: %ld ids compared, %ld differ\n", total, diff);
    }
    return 0;
}
