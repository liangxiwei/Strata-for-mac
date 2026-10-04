#!/usr/bin/env python3
"""Test exact norm/down fusion: repeat norm per down group, stage two 20-KiB halves."""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TMP = Path('/tmp/strata-first-principles/decode')
TMP.mkdir(parents=True, exist_ok=True)
gr = (ROOT / 'src/kernels/metal/fused_gr.metal').read_text()
dot = gr[gr.index('static inline float fgr_dot8('):gr.index('// ---- gr_norm_multi_kernel')]
dot = dot.replace('fgr_dot8(', 'fgr_dot8_shared(').replace('constant const float* x', 'threadgroup const float* x')
norm = gr[gr.index('kernel void fused_gr_norm1_kernel'):gr.index('// ---- the down projection')]
body = norm[norm.index('    threadgroup float part'):norm.rindex('}')]
body = body.replace('        rs[tid] = s_rs[tid];', '        if (gpos.x == 0) rs[tid] = s_rs[tid];')
tail = body.index('#pragma unroll\n    for (int k = 0; k < K; ++k) {', body.index('s_rs[tid] ='))
body = body[:tail] + '    if (gpos.x == 0) {\n' + body[tail:] + '    }\n'
header = norm[:norm.index('    threadgroup float part')]
header = header.replace('fused_gr_norm1_kernel', 'gr_norm_down_shared')
header = header.replace('                                  uint tid [[thread_index_in_threadgroup]],', '''                                  constant const uint* w_down [[buffer(8)]],
                                  constant const uint* w_inject [[buffer(9)]],
                                  device float* lo [[buffer(10)]],
                                  device float* inject_out [[buffer(11)]],
                                  uint3 gpos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],''')
down = '''
    // Every group recomputes the original norm. Two halves fit the measured 32-KiB limit.
    threadgroup float tile[FGR_D / 2];
    const bool inject_block = gpos.x == FGR_DOWN_BLOCKS;
    const int row = inject_block ? (int) sg : (int) gpos.x * FGR_WARPS + (int) sg;
    const bool active = !(inject_block && (w_inject == nullptr || sg >= (uint) FGR_HC));
    constant const uint16_t* wrow16 =
        reinterpret_cast<constant const uint16_t*>(inject_block ? w_inject : w_down) +
        (ulong) (active ? row : 0) * FGR_D;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>(wrow16);
    float acc = 0.0f;
    for (int chunk = 0; chunk < 2; ++chunk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < K / 2; ++k) {
            const int src = k + chunk * (K / 2), i = (int) tid * 4 + k * FGR_THREADS * 4;
            *(threadgroup float4*) (tile + i) = xv[src] * s_rs[cv[src]];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int j = (int) lane + chunk * (FGR_D / 16); j < (chunk + 1) * (FGR_D / 16); j += 32)
                acc += fgr_dot8_shared(w4[j], tile + (j - chunk * (FGR_D / 16)) * 8);
        }
    }
    acc = fgr_warp_sum(acc);
    if (lane != 0 || !active) return;
    if (inject_block) inject_out[row] = acc;
    else {
        const float x = acc / (float) FGR_HC;
        lo[row] = x / (1.0f + metal::precise::exp(-x));
    }
}
'''
metal = OUT / 'decode/gr_fusion.metal'
metal.write_text(gr + '\n' + dot + header + body + down)
host = (OUT / 'decode/gr_schedule.mm').read_text().replace('const int apply = 1;', 'int apply = 1; bool inject = true;')
host = host.replace('[e setBuffer:wi offset:0 atIndex:1];', '[e setBuffer:inject ? wi : nil offset:0 atIndex:1];')
anchor = '        auto run = [&](auto&& f) {'
fusion = '''        auto fused = [&](id<MTLComputeCommandEncoder> e, Out& o, int c) {
            [e setComputePipelineState:pso("gr_norm_down_shared")];
            [e setBuffer:R offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1]; [e setBuffer:inj offset:0 atIndex:2];
            [e setBuffer:wn offset:0 atIndex:3]; [e setBytes:&eps length:4 atIndex:4]; [e setBytes:&apply length:4 atIndex:5];
            [e setBuffer:o.rs offset:0 atIndex:6]; [e setBuffer:o.xn offset:0 atIndex:7];
            [e setBuffer:wd[c % copies] offset:0 atIndex:8]; [e setBuffer:inject ? wi : nil offset:0 atIndex:9];
            [e setBuffer:o.lo offset:0 atIndex:10]; [e setBuffer:o.io offset:0 atIndex:11];
            [e dispatchThreadgroups:MTLSizeMake(LR / 8 + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
'''
host = host.replace(anchor, fusion + anchor)
start = host.index('        std::vector<std::string> dns =')
end = host.index('        for (auto& [k, v] : t)', start)
host = host[:start] + '''        size_t checked = 0;
        for (int a : {0, 1}) for (int with_inject : {0, 1}) {
            apply = a; inject = with_inject;
            run([&](id<MTLComputeCommandEncoder> e) { norm(e, "fused_gr_norm1_kernel", A); down(e, "fused_gr_down_kernel", A, 0); up(e, "fused_gr_up_kernel", A, 0); });
            run([&](id<MTLComputeCommandEncoder> e) { fused(e, B, 0); up(e, "fused_gr_up_kernel", B, 0); });
            long different = same(A.rs, B.rs, 4) + same(A.xn, B.xn, D) + same(A.lo, B.lo, LR) +
                (inject ? same(A.io, B.io, 4) : 0) + same(A.Rout, B.Rout, D) + same(A.mixed, B.mixed, N);
            checked += 4 + D + LR + (inject ? 4 : 0) + D + N;
            if (different) { fprintf(stderr, "apply=%d inject=%d bitdiff=%ld\\n", apply, (int) inject, different); return 2; }
        }
        printf("# checked=%zu bitdiff=0 copies=32 reps=96\\n", checked);
        apply = 1; inject = true;
        const int reps = 96;
        std::map<std::string, std::vector<double>> t;
        std::vector<std::string> variants = {"original_norm_down", "fused_norm_down", "original_chain", "fused_chain"};
        for (int r = 0; r < rounds; ++r) {
            if (r % 2) std::reverse(variants.begin(), variants.end());
            for (const auto& k : variants) t[k].push_back(run([&](id<MTLComputeCommandEncoder> e) {
                for (int i = 0; i < reps; ++i) {
                    if (k.find("fused") == 0) fused(e, A, i);
                    else { norm(e, "fused_gr_norm1_kernel", A); down(e, "fused_gr_down_kernel", A, i); }
                    if (k.find("chain") != std::string::npos) up(e, "fused_gr_up_kernel", A, i);
                }
            }));
        }
''' + host[end:]
source = OUT / 'decode/gr_fusion.mm'
source.write_text(host)
subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-fno-fast-math', '-ffp-contract=off',
                '-I', str(ROOT / 'src/kernels/metal'), str(metal), '-o', str(TMP / 'gr_fusion.metallib')], check=True)
subprocess.run(['clang++', '-O2', '-std=c++17', '-fobjc-arc', '-framework', 'Foundation', '-framework', 'Metal',
                str(source), '-o', str(TMP / 'gr_fusion')], check=True)
print(TMP / 'gr_fusion')
