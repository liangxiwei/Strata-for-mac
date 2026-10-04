// Isolated first-principles experiment, not an engine optimization.
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

using ulong = unsigned long;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* lib_path = argc > 1 ? argv[1] : "expert_cache.metallib";
        const int rounds = argc > 2 ? atoi(argv[2]) : 5;
        const uint ne = argc > 3 ? atoi(argv[3]) : 512;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib_path)] error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        auto pso = [&](const std::string& name) {
            if (pipelines.count(name)) return pipelines[name];
            id<MTLFunction> f = [lib newFunctionWithName:@(name.c_str())];
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
            if (!p) { fprintf(stderr, "%s: %s\n", name.c_str(), err.localizedDescription.UTF8String); exit(1); }
            return pipelines[name] = p;
        };
        auto timed = [&](auto encode, int reps) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int r = 0; r < reps; ++r) encode(enc);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "GPU: %s\n", cb.error.localizedDescription.UTF8String); exit(1);
            }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / reps;
        };
        {
            const uint n = (256u << 20) / 16;
            id<MTLBuffer> A = [dev newBufferWithLength:size_t(n) * 16 options:0];
            id<MTLBuffer> B = [dev newBufferWithLength:size_t(n) * 16 options:0];
            memset(A.contents, 0x37, A.length); memset(B.contents, 0, B.length);
            auto copy = [&](id<MTLComputeCommandEncoder> e) {
                [e setComputePipelineState:pso("bandwidth_copy")];
                [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1];
                [e setBytes:&n length:4 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            };
            timed(copy, 2);
            std::vector<double> t;
            for (int r = 0; r < rounds; ++r) t.push_back(timed(copy, 4));
            const double ms = median(t);
            printf("# device=%s copy_bytes_each=%lu copy_ms=%.6f read_plus_write_GBs=%.3f validated=%d\n",
                   dev.name.UTF8String, A.length, ms, (A.length + B.length) / (ms * 1e6),
                   memcmp(A.contents, B.contents, A.length) == 0);
        }
        printf("format,experts,rows,mean_rows,tile_rows,expanded_GiB,fused_ms,cached_ms,expand_ms,expand_plus_cached_ms,bitdiff,checked_outputs\n");
        const long H = 2560, FF = 640;
        for (int ty : {22, 16, 29, 42}) {
            @autoreleasepool {
                const bool down = ty == 42;
                const uint K = uint(down ? FF : H), N = uint(down ? H : 2 * FF);
                const ulong block = ty == 22 ? 82 : ty == 16 ? 66 : 56;
                const ulong gu_row = (H / 256) * block, d_row = (FF / 64) * 18;
                const ulong up_off = FF * gu_row, down_off = 2 * up_off;
                const ulong blob = down_off + H * d_row;
                const ulong rb = down ? d_row : gu_row, wo = down ? down_off : up_off;
                id<MTLBuffer> A = [dev newBufferWithLength:blob * ne options:0];
                id<MTLBuffer> E = [dev newBufferWithLength:ulong(ne) * N * K * 2 options:0];
                std::mt19937 rng(2100 + ty);
                uint32_t* raw = (uint32_t*) A.contents;
                for (size_t i = 0; i < A.length / 4; ++i) raw[i] = rng();
                for (uint e = 0; e < ne; ++e) {
                    auto* b = (uint8_t*) A.contents + e * blob;
                    for (ulong i = 0; i < 2 * up_off; i += block) {
                        if (ty == 29) {
                            auto* sc = (uint16_t*) (b + i + 48);
                            const uint16_t d = 0x1c00;
                            for (int j = 0; j < 4; ++j) sc[j] = (sc[j] & 0x0fff) | (((d >> (4 * j)) & 15) << 12);
                        } else {
                            const __fp16 d = (__fp16) ((int(rng() % 63) - 31) / 8192.0f);
                            memcpy(b + i, &d, 2);
                        }
                    }
                    for (ulong i = 0; i < H * d_row; i += 18) {
                        const __fp16 d = (__fp16) ((int(rng() % 63) - 31) / 8192.0f);
                        memcpy(b + down_off + i, &d, 2);
                    }
                }
                const std::string tag = down ? "dn42" : "gu" + std::to_string(ty);
                auto expand = [&](id<MTLComputeCommandEncoder> e) {
                    const uint n = ne * N * (K / 32) * 4;
                    [e setComputePipelineState:pso("expand_" + tag)];
                    [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:E offset:0 atIndex:1];
                    [e setBytes:&K length:4 atIndex:2]; [e setBytes:&N length:4 atIndex:3];
                    [e setBytes:&ne length:4 atIndex:4]; [e setBytes:&blob length:8 atIndex:5];
                    [e setBytes:&rb length:8 atIndex:6]; [e setBytes:&wo length:8 atIndex:7];
                    [e dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                };
                timed(expand, 1);
                for (int per : {20, 80}) {
                    @autoreleasepool {
                        const int BM = per >= 40 ? 32 : 16, TH = BM == 32 ? 256 : 128;
                        std::vector<int> cnt(ne), off(ne + 1, 0);
                        std::exponential_distribution<double> skew(1.0 / per);
                        for (uint i = 0; i < ne; ++i) { cnt[i] = std::max(1, int(skew(rng))); off[i + 1] = off[i] + cnt[i]; }
                        const uint rows = off[ne];
                        std::vector<int> qt, et;
                        for (uint e = 0; e < ne; ++e) for (int r = 0; r < cnt[e]; r += BM) {
                            const ulong o = ulong(e) * blob;
                            qt.insert(qt.end(), {int(uint32_t(o)), off[e] + r, std::min(BM, cnt[e] - r), int(uint32_t(o >> 32))});
                            et.insert(et.end(), {int(e), off[e] + r, std::min(BM, cnt[e] - r), 0});
                        }
                        const uint nt = qt.size() / 4;
                        id<MTLBuffer> QT = [dev newBufferWithBytes:qt.data() length:qt.size() * 4 options:0];
                        id<MTLBuffer> ET = [dev newBufferWithBytes:et.data() length:et.size() * 4 options:0];
                        id<MTLBuffer> X = [dev newBufferWithLength:ulong(rows) * K * 2 options:0];
                        id<MTLBuffer> Y0 = [dev newBufferWithLength:ulong(rows) * N * 4 options:0];
                        id<MTLBuffer> Y1 = [dev newBufferWithLength:Y0.length options:0];
                        std::normal_distribution<float> nd;
                        auto* xv = (__fp16*) X.contents;
                        for (ulong i = 0; i < ulong(rows) * K; ++i) xv[i] = (__fp16) nd(rng);
                        auto fused = [&](id<MTLComputeCommandEncoder> e) {
                            const std::string name = "native_gemm2_" + std::string(down ? "down_42" : "gu_" + std::to_string(ty)) + "_r" + std::to_string(BM);
                            [e setComputePipelineState:pso(name)];
                            [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:A offset:0 atIndex:1];
                            [e setBuffer:QT offset:0 atIndex:2]; [e setBuffer:Y0 offset:0 atIndex:3];
                            [e setBytes:&H length:8 atIndex:4]; [e setBytes:&FF length:8 atIndex:5];
                            [e setBytes:&rb length:8 atIndex:6]; [e setBytes:&wo length:8 atIndex:7];
                            [e dispatchThreadgroups:MTLSizeMake(N / 64, nt, 1) threadsPerThreadgroup:MTLSizeMake(TH, 1, 1)];
                        };
                        auto cached = [&](id<MTLComputeCommandEncoder> e) {
                            [e setComputePipelineState:pso("cached_r" + std::to_string(BM))];
                            [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:E offset:0 atIndex:1];
                            [e setBuffer:ET offset:0 atIndex:2]; [e setBuffer:Y1 offset:0 atIndex:3];
                            [e setBytes:&K length:4 atIndex:4]; [e setBytes:&N length:4 atIndex:5];
                            [e dispatchThreadgroups:MTLSizeMake(N / 64, nt, 1) threadsPerThreadgroup:MTLSizeMake(TH, 1, 1)];
                        };
                        auto combined = [&](id<MTLComputeCommandEncoder> e) { expand(e); cached(e); };
                        memset(Y0.contents, 0xaa, Y0.length); memset(Y1.contents, 0x55, Y1.length);
                        timed(fused, 1); timed(cached, 1);
                        ulong diff = 0, checked = ulong(rows) * N;
                        for (ulong i = 0; i < checked; ++i) diff += ((uint32_t*) Y0.contents)[i] != ((uint32_t*) Y1.contents)[i];
                        if (diff) { fprintf(stderr, "%s bitdiff=%lu\n", tag.c_str(), diff); return 2; }
                        std::vector<double> ft, ct, dt, bt;
                        for (int r = 0; r < rounds; ++r) {
                            // Alternate which product runs first to reduce ordering bias.
                            if (r % 2) { ct.push_back(timed(cached, 2)); ft.push_back(timed(fused, 2)); }
                            else { ft.push_back(timed(fused, 2)); ct.push_back(timed(cached, 2)); }
                            dt.push_back(timed(expand, 2)); bt.push_back(timed(combined, 2));
                        }
                        printf("%s,%u,%u,%.2f,%d,%.5f,%.6f,%.6f,%.6f,%.6f,%lu,%lu\n",
                               tag.c_str(), ne, rows, double(rows) / ne, BM, E.length / double(1ull << 30),
                               median(ft), median(ct), median(dt), median(bt), diff, checked);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    return 0;
}
