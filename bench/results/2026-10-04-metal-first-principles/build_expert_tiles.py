#!/usr/bin/env python3
"""Extend the expert-cache harness with exact mixed 16/32-row tiles, reusing production kernels."""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
scratch = Path('/tmp/strata-first-principles')
source = (OUT / 'micro/expert_cache.mm').read_text()
source = source.replace('bitdiff,checked_outputs\\n', 'bitdiff,checked_outputs,mixed_ms,mixed_bitdiff,tiles16,tiles32\\n')
before = '                        auto combined = [&](id<MTLComputeCommandEncoder> e) { expand(e); cached(e); };'
after = '''                        auto combined = [&](id<MTLComputeCommandEncoder> e) { expand(e); cached(e); };
                        std::vector<int> mixed16, mixed32;
                        for (uint ei = 0; ei < ne; ++ei) {
                            for (int r = 0; r < cnt[ei]; ) {
                                const int remain = cnt[ei] - r;
                                const int bm = remain > 16 && (BM == 32 || cnt[ei] >= 32) ? 32 : 16;
                                const ulong o = ulong(ei) * blob;
                                auto& desc = bm == 32 ? mixed32 : mixed16;
                                desc.insert(desc.end(), {int(uint32_t(o)), off[ei] + r, std::min(bm, remain), int(uint32_t(o >> 32))});
                                r += bm;
                            }
                        }
                        auto M16 = [dev newBufferWithBytes:mixed16.data() length:mixed16.size() * 4 options:0];
                        auto M32 = [dev newBufferWithBytes:mixed32.data() length:mixed32.size() * 4 options:0];
                        auto mixed = [&](id<MTLComputeCommandEncoder> e) {
                            for (int bm : {16, 32}) {
                                const auto tile = bm == 16 ? M16 : M32;
                                const uint count = (bm == 16 ? mixed16.size() : mixed32.size()) / 4;
                                if (count == 0) continue;
                                const std::string name = "native_gemm2_" + std::string(down ? "down_42" : "gu_" + std::to_string(ty)) + "_r" + std::to_string(bm);
                                [e setComputePipelineState:pso(name)];
                                [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:A offset:0 atIndex:1];
                                [e setBuffer:tile offset:0 atIndex:2]; [e setBuffer:Y1 offset:0 atIndex:3];
                                [e setBytes:&H length:8 atIndex:4]; [e setBytes:&FF length:8 atIndex:5];
                                [e setBytes:&rb length:8 atIndex:6]; [e setBytes:&wo length:8 atIndex:7];
                                [e dispatchThreadgroups:MTLSizeMake(N / 64, count, 1) threadsPerThreadgroup:MTLSizeMake(bm == 16 ? 128 : 256, 1, 1)];
                            }
                        };'''
assert source.count(before) == 1
source = source.replace(before, after)
before = '                        std::vector<double> ft, ct, dt, bt;'
after = '''                        timed(mixed, 1);
                        ulong mixed_diff = 0;
                        for (ulong i = 0; i < ulong(rows) * N; ++i)
                            mixed_diff += ((uint32_t*) Y0.contents)[i] != ((uint32_t*) Y1.contents)[i];
                        if (mixed_diff) { fprintf(stderr, "mixed bitdiff=%lu\\n", mixed_diff); return 4; };
                        std::vector<double> mt;
                        for (int r = 0; r < rounds; ++r) mt.push_back(timed(mixed, 2));
                        std::vector<double> ft, ct, dt, bt;'''
assert source.count(before) == 1
source = source.replace(before, after)
before = '%.6f,%lu,%lu\\n",'
after = '%.6f,%lu,%lu,%.6f,%lu,%zu,%zu\\n",'
assert source.count(before) == 1
source = source.replace(before, after)
before = 'median(ft), median(ct), median(dt), median(bt), diff, checked);'
after = 'median(ft), median(ct), median(dt), median(bt), diff, checked, median(mt), mixed_diff, mixed16.size() / 4, mixed32.size() / 4);'
assert source.count(before) == 1
source = source.replace(before, after)
(scratch / 'expert_tiles.mm').write_text(source)
subprocess.run(['clang++', '-std=c++20', '-O2', '-fobjc-arc', str(scratch / 'expert_tiles.mm'),
                '-framework', 'Metal', '-framework', 'Foundation', '-o', str(scratch / 'expert_tiles')], check=True)
print(scratch / 'expert_tiles')
