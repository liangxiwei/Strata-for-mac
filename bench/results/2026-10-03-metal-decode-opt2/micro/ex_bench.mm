// Bitwise + timing harness: resident expert kernels (gate/up IQ2_S or IQ2_XXS, down Q2_0) and IQ3_S MMVQ.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        const int rounds = argc > 1 ? atoi(argv[1]) : 7;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"ex.metallib"] error:&err];
        if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::map<std::string, id<MTLComputePipelineState>> P;
        auto pso = [&](const std::string& n) {
            if (P.count(n)) return P[n];
            id<MTLFunction> f = [lib newFunctionWithName:@(n.c_str())];
            if (!f) { printf("no %s\n", n.c_str()); exit(1); }
            return P[n] = [dev newComputePipelineStateWithFunction:f error:nil];
        };
        std::mt19937 rng(11);
        auto fill_blocks = [&](uint8_t* p, size_t bytes, size_t block, size_t dpos) {
            for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
            for (size_t i = 0; i + block <= bytes; i += block) {
                const unsigned r = rng();
                const uint16_t d = uint16_t(((r >> 31) << 15) | ((5 + (r >> 10) % 5) << 10) | (r & 1023));
                std::memcpy(p + i + dpos, &d, 2);
            }
        };
        auto q8 = [&](int n) {
            std::vector<uint8_t> x((size_t) (n / 32) * 36);
            for (size_t i = 0; i < x.size(); i += 36) {
                const uint16_t d = uint16_t((((rng() % 3) + 9) << 10) | (rng() & 1023)), s = uint16_t(rng() & 0x7bff);
                std::memcpy(x.data() + i, &d, 2); std::memcpy(x.data() + i + 2, &s, 2);
                for (int j = 0; j < 32; ++j) x[i + 4 + j] = (uint8_t) rng();
            }
            return [dev newBufferWithBytes:x.data() length:x.size() options:MTLResourceStorageModeShared];
        };
        auto same = [&](id<MTLBuffer> a, id<MTLBuffer> b, size_t n) {
            const uint32_t* x = (const uint32_t*) a.contents, * y = (const uint32_t*) b.contents;
            long d = 0; for (size_t i = 0; i < n; ++i) d += x[i] != y[i]; return d;
        };
        auto timed = [&](auto&& f) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            f(e);
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return cb.GPUEndTime - cb.GPUStartTime;
        };
        // ---- resident experts: n_embd 2560, n_ff 640, k = 10 routes, one token
        const long n_embd = 2560, n_ff = 640;
        const int k = 10, n_expert = 128;
        for (int gu_t : {22, 16}) {
            const size_t gu_block = gu_t == 22 ? 82 : 66;
            const unsigned long gu_row = (unsigned long) (n_embd / 256) * gu_block, d_row = (unsigned long) (n_ff / 64) * 18;
            const unsigned long up_off = (unsigned long) n_ff * gu_row, down_off = 2 * up_off;
            const unsigned long slot = down_off + (unsigned long) n_embd * d_row;
            std::vector<uint8_t> arena(slot * n_expert);
            for (int e = 0; e < n_expert; ++e) {
                fill_blocks(arena.data() + e * slot, 2 * up_off, gu_block, 0);
                fill_blocks(arena.data() + e * slot + down_off, (size_t) n_embd * d_row, 18, 0);
            }
            id<MTLBuffer> A = [dev newBufferWithBytes:arena.data() length:arena.size() options:MTLResourceStorageModeShared];
            std::vector<int> ids(n_expert * 4), res(n_expert);
            for (int i = 0; i < n_expert; ++i) res[i] = i;
            std::vector<int> perm(n_expert); for (int i = 0; i < n_expert; ++i) perm[i] = i;
            for (size_t i = 0; i < ids.size(); ++i) { if (i % n_expert == 0) std::shuffle(perm.begin(), perm.end(), rng); ids[i] = perm[i % n_expert]; }
            id<MTLBuffer> I = [dev newBufferWithBytes:ids.data() length:ids.size() * 4 options:0];
            id<MTLBuffer> Rz = [dev newBufferWithBytes:res.data() length:res.size() * 4 options:0];
            id<MTLBuffer> off = [dev newBufferWithLength:8 options:0];
            id<MTLBuffer> X = q8((int) n_embd), H = q8((int) n_ff * k);
            const int reps = n_expert * 4 / k;   // every dispatch routes to ten different experts
            id<MTLBuffer> G1 = [dev newBufferWithLength:k * n_ff * 4 * reps options:0], U1 = [dev newBufferWithLength:k * n_ff * 4 * reps options:0];
            id<MTLBuffer> G2 = [dev newBufferWithLength:k * n_ff * 4 * reps options:0], U2 = [dev newBufferWithLength:k * n_ff * 4 * reps options:0];
            id<MTLBuffer> O1 = [dev newBufferWithLength:k * n_embd * 4 * reps options:0], O2 = [dev newBufferWithLength:k * n_embd * 4 * reps options:0];
            const int has = 0, kk = k, ne = n_expert;
            const unsigned long sb = slot;
            auto res_args = [&](id<MTLComputeCommandEncoder> e, int rep, id<MTLBuffer> xq, unsigned long rb, unsigned long wo) {
                [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:off offset:0 atIndex:1];
                [e setBuffer:I offset:(size_t) rep * k * 4 atIndex:2]; [e setBuffer:Rz offset:0 atIndex:3];
                [e setBuffer:xq offset:0 atIndex:4]; [e setBytes:&n_embd length:8 atIndex:5]; [e setBytes:&n_ff length:8 atIndex:6];
                [e setBytes:&rb length:8 atIndex:7]; [e setBytes:&wo length:8 atIndex:8]; [e setBytes:&sb length:8 atIndex:9];
                [e setBytes:&ne length:4 atIndex:10]; [e setBytes:&kk length:4 atIndex:11]; [e setBytes:&has length:4 atIndex:12];
            };
            auto gu = [&](id<MTLComputeCommandEncoder> e, const std::string& kn, int rep, id<MTLBuffer> G, id<MTLBuffer> U) {
                [e setComputePipelineState:pso(kn)];
                res_args(e, rep, X, gu_row, up_off);
                [e setBuffer:G offset:(size_t) rep * k * n_ff * 4 atIndex:13]; [e setBuffer:U offset:(size_t) rep * k * n_ff * 4 atIndex:14];
                int R = (kn[0] == 'o' || kn[0] == 'd' || kn[0] == 't') && kn[2] >= '2' && kn[2] <= '8' ? kn[2] - '0' : 1, W = 8;
                if (kn[0] == 'w') { W = kn[1] - '0'; R = kn[3] - '0'; }
                [e dispatchThreadgroups:MTLSizeMake((2 * n_ff + W * R - 1) / (W * R), k, 1) threadsPerThreadgroup:MTLSizeMake(32 * W, 1, 1)];
            };
            auto dn = [&](id<MTLComputeCommandEncoder> e, const std::string& kn, int rep, id<MTLBuffer> O) {
                [e setComputePipelineState:pso(kn)];
                res_args(e, rep, H, d_row, down_off);
                [e setBuffer:O offset:(size_t) rep * k * n_embd * 4 atIndex:13];
                int R = (kn[0] == 'o' || kn[0] == 'd' || kn[0] == 't') && kn[2] >= '2' && kn[2] <= '8' ? kn[2] - '0' : 1;
                [e dispatchThreadgroups:MTLSizeMake((n_embd + 8 * R - 1) / (8 * R), k, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            };
            const std::string ogu = "native_resident_gu_" + std::to_string(gu_t), dgu = "dd_resident_gu_" + std::to_string(gu_t);
            const std::string T = std::to_string(gu_t);
            std::vector<std::string> guv = {"dd_resident_gu_" + T, "dd2_gu_" + T, "or2_gu_" + T, "tg2_gu_" + T};
            if (gu_t == 22) for (const char* v : {"w2r2d_gu_22", "w1r1o_gu_22", "ta2_gu_22", "floorL_gu_22", "floorG_gu_22"}) guv.push_back(v);
            const std::vector<std::string> dnv = {"dd_resident_down_42", "or4_down_42", "dd2_down_42", "dd4_down_42"};
            timed([&](id<MTLComputeCommandEncoder> e) { for (int r = 0; r < reps; ++r) { gu(e, ogu, r, G1, U1); dn(e, "native_resident_down_42", r, O1); } });
            for (const auto& kn : guv) {
                timed([&](id<MTLComputeCommandEncoder> e) { for (int r = 0; r < reps; ++r) gu(e, kn, r, G2, U2); });
                printf("%-24s gate %ld up %ld bit differences of %ld each%s\n", kn.c_str(), same(G1, G2, (size_t) k * n_ff * reps),
                       same(U1, U2, (size_t) k * n_ff * reps), (long) k * n_ff * reps, kn.rfind("floor", 0) == 0 ? " (timing floor, not exact)" : "");
            }
            if (gu_t == 22)
                for (const auto& kn : dnv) {
                    timed([&](id<MTLComputeCommandEncoder> e) { for (int r = 0; r < reps; ++r) dn(e, kn, r, O2); });
                    printf("%-24s down %ld bit differences of %ld\n", kn.c_str(), same(O1, O2, (size_t) k * n_embd * reps), (long) k * n_embd * reps);
                }
            std::map<std::string, std::vector<double>> t;
            for (int r = 0; r < rounds; ++r) {
                t[ogu].push_back(timed([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) gu(e, ogu, i, G1, U1); }));
                for (const auto& kn : guv) t[kn].push_back(timed([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) gu(e, kn, i, G2, U2); }));
                if (gu_t != 22) continue;
                t["native_resident_down_42"].push_back(timed([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) dn(e, "native_resident_down_42", i, O1); }));
                for (const auto& kn : dnv) t[kn].push_back(timed([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) dn(e, kn, i, O2); }));
            }
            for (auto& [kn, v] : t) { std::sort(v.begin(), v.end()); printf("  %-26s %8.2f us\n", kn.c_str(), v[v.size() / 2] * 1e6 / reps); }
        }
        // ---- IQ3_S MMVQ, one column
        for (auto [K, N] : std::vector<std::pair<int, int>>{{2560, 6144}, {2560, 640}, {4096, 2560}, {2560, 12288}}) {
            const unsigned long rb = (unsigned long) (K / 256) * 110;
            const size_t wbytes = rb * N;
            const int copies = (int) std::min<size_t>(24, std::max<size_t>(1, (256ull << 20) / wbytes));
            std::vector<id<MTLBuffer>> W;
            std::vector<uint8_t> w(wbytes);
            for (int c = 0; c < copies; ++c) { fill_blocks(w.data(), wbytes, 110, 0); W.push_back([dev newBufferWithBytes:w.data() length:wbytes options:0]); }
            id<MTLBuffer> X = q8(K);
            id<MTLBuffer> Y1 = [dev newBufferWithLength:N * 4 * copies options:0], Y2 = [dev newBufferWithLength:N * 4 * copies options:0];
            const int one = 1;
            auto mm = [&](id<MTLComputeCommandEncoder> e, const std::string& kn, int c, id<MTLBuffer> Y) {
                [e setComputePipelineState:pso(kn)];
                [e setBuffer:W[c % copies] offset:0 atIndex:0]; [e setBytes:&rb length:8 atIndex:1]; [e setBuffer:X offset:0 atIndex:2];
                [e setBuffer:Y offset:(size_t) (c % copies) * N * 4 atIndex:3]; [e setBytes:&K length:4 atIndex:4]; [e setBytes:&N length:4 atIndex:5];
                [e setBytes:&one length:4 atIndex:6];
                int R = (kn[0] == 'o' || kn[0] == 'd' || kn[0] == 't') && kn[2] >= '2' && kn[2] <= '8' ? kn[2] - '0' : 1;
                [e dispatchThreadgroups:MTLSizeMake((N + 4 * R - 1) / (4 * R), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            };
            timed([&](id<MTLComputeCommandEncoder> e) { for (int c = 0; c < copies; ++c) mm(e, "mmvq_multi_kernel_21_1", c, Y1); });
            std::string diffs;
            for (const std::string kn : {"dd_mmvq_21", "or2_mmvq_21", "dd2_mmvq_21", "dd4_mmvq_21"}) {
                timed([&](id<MTLComputeCommandEncoder> e) { for (int c = 0; c < copies; ++c) mm(e, kn, c, Y2); });
                diffs += " " + kn + "=" + std::to_string(same(Y1, Y2, (size_t) N * copies));
            }
            std::map<std::string, std::vector<double>> t;
            const int reps = std::max(copies, 32);
            for (int r = 0; r < rounds; ++r)
                for (const std::string kn : {"mmvq_multi_kernel_21_1", "dd_mmvq_21", "or2_mmvq_21", "dd2_mmvq_21", "dd4_mmvq_21"})
                    t[kn].push_back(timed([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) mm(e, kn, i, kn[0] == 'm' ? Y1 : Y2); }));
            printf("iq3_s %dx%d bit differences (of %ld):%s\n", K, N, (long) N * copies, diffs.c_str());
            for (auto& [kn, v] : t) { std::sort(v.begin(), v.end()); printf("  %-26s %8.2f us  %.1f GB/s\n", kn.c_str(), v[v.size() / 2] * 1e6 / reps, wbytes / (v[v.size() / 2] * 1e9 / reps)); }
        }
    }
    return 0;
}
