// Rotating resident-expert decode benchmark. Compare lossless nibbles with the current kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static double median(std::vector<double> a) { std::sort(a.begin(), a.end()); return a[a.size() / 2]; }

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* path = argc > 1 ? argv[1] : "expert_nibble.metallib";
        const int rounds = argc > 2 ? atoi(argv[2]) : 9;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        auto pso = [&](std::string name) {
            if (pipelines.count(name)) return pipelines[name];
            auto p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(name.c_str())] error:&err];
            if (!p) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); exit(1); }
            return pipelines[name] = p;
        };
        auto timed = [&](auto encode) {
            auto cb = [q commandBuffer]; auto e = [cb computeCommandEncoder]; encode(e);
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "%s\n", cb.error.localizedDescription.UTF8String); exit(1); }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
        };
        printf("format,experts,reps,current_us,nibble_us,speedup,original_MiB,nibble_MiB,expand_ms,invalid_codes,bitdiff,checked_outputs\n");
        const long H = 2560, FF = 640;
        const int ne = 128, kk = 10, has = 0, reps = ne * 4 / kk;
        for (int ty : {22, 16}) {
            @autoreleasepool {
                std::mt19937 rng(4500 + ty);
                const unsigned long block = ty == 22 ? 82 : 66, rb0 = H / 256 * block, rb1 = H / 256 * 140;
                const unsigned long up0 = FF * rb0, up1 = FF * rb1, slot0 = 2 * up0, slot1 = 2 * up1;
                auto A = [dev newBufferWithLength:ne * slot0 options:0];
                auto C = [dev newBufferWithLength:ne * slot1 options:0];
                auto BAD = [dev newBufferWithLength:4 options:0]; memset(BAD.contents, 0, 4);
                for (size_t i = 0; i < A.length / 4; ++i) ((uint32_t*) A.contents)[i] = rng();
                for (size_t i = 0; i < A.length; i += block) {
                    const uint16_t d = uint16_t(((rng() >> 31) << 15) | ((5 + rng() % 5) << 10) | (rng() & 1023));
                    memcpy((char*) A.contents + i, &d, 2);
                }
                const uint nb = A.length / block;
                const double expand_us = timed([&](id<MTLComputeCommandEncoder> e) {
                    [e setComputePipelineState:pso("expand_nibble_" + std::to_string(ty))];
                    [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:C offset:0 atIndex:1];
                    [e setBuffer:BAD offset:0 atIndex:2]; [e setBytes:&nb length:4 atIndex:3];
                    [e dispatchThreadgroups:MTLSizeMake((nb * 8 + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });
                const uint invalid = *(uint*) BAD.contents;
                if (invalid) { fprintf(stderr, "invalid=%u\n", invalid); return 3; }
                std::vector<int> ids(ne * 4), residency(ne), perm(ne);
                for (int i = 0; i < ne; ++i) residency[i] = perm[i] = i;
                for (int j = 0; j < 4; ++j) {
                    std::shuffle(perm.begin(), perm.end(), rng);
                    std::copy(perm.begin(), perm.end(), ids.begin() + ne * j);
                }
                auto I = [dev newBufferWithBytes:ids.data() length:ids.size() * 4 options:0];
                auto R = [dev newBufferWithBytes:residency.data() length:residency.size() * 4 options:0];
                auto OFF = [dev newBufferWithLength:8 options:0];
                auto X = [dev newBufferWithLength:H / 32 * 36 options:0];
                for (size_t i = 0; i < X.length; i += 36) {
                    const uint16_t d = uint16_t(((9 + rng() % 3) << 10) | (rng() & 1023)), sum = 0;
                    memcpy((char*) X.contents + i, &d, 2); memcpy((char*) X.contents + i + 2, &sum, 2);
                    for (int j = 0; j < 32; ++j) ((uint8_t*) X.contents)[i + 4 + j] = uint8_t(rng());
                }
                const size_t outputs = size_t(reps) * kk * FF;
                auto G0 = [dev newBufferWithLength:outputs * 4 options:0], U0 = [dev newBufferWithLength:outputs * 4 options:0];
                auto G1 = [dev newBufferWithLength:outputs * 4 options:0], U1 = [dev newBufferWithLength:outputs * 4 options:0];
                auto launch = [&](id<MTLComputeCommandEncoder> e, bool cache, int rep) {
                    const auto arena = cache ? C : A;
                    const unsigned long rb = cache ? rb1 : rb0, up = cache ? up1 : up0, slot = cache ? slot1 : slot0;
                    const auto G = cache ? G1 : G0, U = cache ? U1 : U0;
                    [e setComputePipelineState:pso((cache ? "cached_nibble_gu_" : "native_resident_gu_") + std::to_string(ty))];
                    [e setBuffer:arena offset:0 atIndex:0]; [e setBuffer:OFF offset:0 atIndex:1];
                    [e setBuffer:I offset:size_t(rep) * kk * 4 atIndex:2]; [e setBuffer:R offset:0 atIndex:3];
                    [e setBuffer:X offset:0 atIndex:4]; [e setBytes:&H length:8 atIndex:5]; [e setBytes:&FF length:8 atIndex:6];
                    [e setBytes:&rb length:8 atIndex:7]; [e setBytes:&up length:8 atIndex:8]; [e setBytes:&slot length:8 atIndex:9];
                    [e setBytes:&ne length:4 atIndex:10]; [e setBytes:&kk length:4 atIndex:11]; [e setBytes:&has length:4 atIndex:12];
                    [e setBuffer:G offset:size_t(rep) * kk * FF * 4 atIndex:13]; [e setBuffer:U offset:size_t(rep) * kk * FF * 4 atIndex:14];
                    [e dispatchThreadgroups:MTLSizeMake(2 * FF / 8, kk, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                };
                auto run = [&](bool cache) { return timed([&](auto e) { for (int r = 0; r < reps; ++r) launch(e, cache, r); }) / reps; };
                memset(G0.contents, 0xaa, G0.length); memset(U0.contents, 0xaa, U0.length);
                memset(G1.contents, 0x55, G1.length); memset(U1.contents, 0x55, U1.length);
                run(false); run(true);
                size_t diff = 0;
                for (size_t i = 0; i < outputs; ++i) {
                    diff += ((uint32_t*) G0.contents)[i] != ((uint32_t*) G1.contents)[i];
                    diff += ((uint32_t*) U0.contents)[i] != ((uint32_t*) U1.contents)[i];
                }
                if (diff) { fprintf(stderr, "bitdiff=%zu\n", diff); return 2; }
                std::vector<double> old, cache;
                for (int r = 0; r < rounds; ++r) {
                    if (r % 2) { cache.push_back(run(true)); old.push_back(run(false)); }
                    else { old.push_back(run(false)); cache.push_back(run(true)); }
                }
                printf("gu%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%u,%zu,%zu\n",
                       ty, ne, reps, median(old), median(cache), median(old) / median(cache),
                       A.length / double(1 << 20), C.length / double(1 << 20), expand_us / 1000, invalid, diff, outputs * 2);
                fflush(stdout);
            }
        }
    }
    return 0;
}
