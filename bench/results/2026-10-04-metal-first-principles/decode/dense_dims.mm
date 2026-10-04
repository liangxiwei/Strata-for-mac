// Bitwise + timing harness for IQ4_XS decode MMVQ variants. GPU time from command-buffer timestamps;
// weights rotate over enough copies to exceed the system-level cache (each dispatch reads DRAM, as in the model).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <map>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int kTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

int main(int argc, char** argv) {
    @autoreleasepool {
        const char* lib_path = argc > 1 ? argv[1] : "mmvq.metallib";
        const int rounds = argc > 2 ? atoi(argv[2]) : 7;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib_path)] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto pso = [&](const std::string& n) {
            id<MTLFunction> f = [lib newFunctionWithName:@(n.c_str())];
            if (!f) { printf("no %s\n", n.c_str()); exit(1); }
            return [dev newComputePipelineStateWithFunction:f error:nil];
        };
        std::vector<std::string> variants = {"native_iq4_xs_direct_sg4", "dim_iq4"};
        const std::vector<std::pair<int, int>> shapes = {{2560, 248320}, {2560, 10240}, {2560, 6144}, {2560, 12288}, {6144, 2560}};
        printf("K,N,variant,us,GB/s,bitdiff\n");
        for (auto [K, N] : shapes) {
            const int nb = K / 256;
            const size_t wbytes = (size_t) N * nb * 136, ebytes = (size_t) N * nb * 264;
            std::mt19937 rng(K * 7 + N);
            std::vector<uint8_t> w(wbytes);
            for (auto& v : w) v = (uint8_t) rng();
            for (size_t i = 0; i < wbytes; i += 136) {
                const unsigned r = rng();
                const uint16_t scale = uint16_t(((r >> 31) << 15) | ((5 + (r >> 10) % 5) << 10) | (r & 1023));
                std::memcpy(w.data() + i, &scale, 2);
            }
            std::vector<uint8_t> x((size_t) (K / 32) * 36);
            for (size_t i = 0; i < x.size(); i += 36) {
                const uint16_t d = uint16_t((((rng() % 3) + 9) << 10) | (rng() & 1023)), s = uint16_t(rng() & 0x7bff);
                std::memcpy(x.data() + i, &d, 2); std::memcpy(x.data() + i + 2, &s, 2);
                for (int j = 0; j < 32; ++j) x[i + 4 + j] = (uint8_t) rng();
            }
            const size_t target = 256ull << 20;
            const int wc = (int) std::min<size_t>(32, std::max<size_t>(1, (target + wbytes - 1) / wbytes));
            const int ec = (int) std::min<size_t>(32, std::max<size_t>(1, (target + ebytes - 1) / ebytes));
            std::vector<id<MTLBuffer>> wb, eb;
            for (int i = 0; i < wc; ++i) wb.push_back([dev newBufferWithBytes:w.data() length:wbytes options:MTLResourceStorageModeShared]);

            id<MTLBuffer> xb = [dev newBufferWithBytes:x.data() length:x.size() options:MTLResourceStorageModeShared];
            const int reps = std::max(8, (int) std::min<size_t>(256, (2ull << 30) / wbytes));
            id<MTLBuffer> yb = [dev newBufferWithLength:(size_t) N * 4 * reps options:MTLResourceStorageModeShared];
            const bool small = nb < 16;
            auto encode = [&](id<MTLComputeCommandEncoder> enc, const std::string& v, int i) {
                const bool expanded = v.find("expanded") != std::string::npos || v == "v108" || v == "v130";
                static std::map<std::string, id<MTLComputePipelineState>> cache;
                const bool fixed = true;
                const bool sr = v.rfind("vsr", 0) == 0;
                int rows = 16;
                int warps = 4;
                if (sr) { warps = v[3] - '0'; rows = (v[4] - '0') * warps; }
                const std::string name = v == "dim_iq4" ? v + "_" + std::to_string(K) : v;
                if (!cache.count(name)) cache[name] = pso(name);
                [enc setComputePipelineState:cache[name]];
                [enc setBuffer:(expanded ? eb[i % ec] : wb[i % wc]) offset:0 atIndex:0];
                [enc setBuffer:xb offset:0 atIndex:1];
                [enc setBuffer:yb offset:(size_t) (i % reps) * N * 4 atIndex:2];
                [enc setBytes:&K length:4 atIndex:3];
                [enc setBytes:&N length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((N + rows - 1) / rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, warps, 1)];
            };
            std::vector<float> ref(N), got(N);
            std::map<std::string, std::vector<double>> times;
            std::map<std::string, long> diffs;
            for (int round = 0; round < rounds; ++round) {
                if (round > 0) std::reverse(variants.begin(), variants.end());
                for (const auto& v : variants) {
                    // correctness: one dispatch into slot 0
                    {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        encode(enc, v, 0);
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        std::memcpy(got.data(), yb.contents, N * 4);
                        if (v == "native_iq4_xs_direct_sg4" && round == 0) ref = got;
                        long d = 0;
                        for (int i = 0; i < N; ++i) d += std::memcmp(&ref[i], &got[i], 4) != 0;
                        diffs[v] += d;
                    }
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int i = 0; i < reps; ++i) encode(enc, v, i);
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    times[v].push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps);
                }
            }
            for (const auto& v : variants) {
                auto t = times[v];
                std::sort(t.begin(), t.end());
                const double med = t[t.size() / 2];
                const bool expanded = v.find("expanded") != std::string::npos || v == "v108" || v == "v130";
                printf("%d,%d,%s,%.2f,%.1f,%ld\n", K, N, v.c_str(), med, (expanded ? ebytes : wbytes) / med / 1e3, diffs[v]);
            }
            fflush(stdout);
        }
    }
    return 0;
}
