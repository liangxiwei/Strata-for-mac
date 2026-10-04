#!/usr/bin/env python3
"""Exact decode experiments: GR scheduling, fixed matrix geometry, and expert down row reuse."""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DEST = OUT / 'decode'
TMP = Path('/tmp/strata-first-principles/decode')
DEST.mkdir(exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
oldbench = ROOT / 'bench/results/2026-10-03-metal-decode-opt2/micro'


def build(name, metal, host):
    (DEST / (name + '.metal')).write_text(metal)
    (DEST / (name + '.mm')).write_text(host)
    subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-fno-fast-math', '-ffp-contract=off',
                    '-I', str(ROOT / 'src/kernels/metal'), str(DEST / (name + '.metal')),
                    '-o', str(TMP / (name + '.metallib'))], check=True)
    subprocess.run(['clang++', '-std=c++20', '-O2', '-fobjc-arc', str(DEST / (name + '.mm')),
                    '-framework', 'Metal', '-framework', 'Foundation', '-o', str(TMP / name)], check=True)
    print(TMP / name, flush=True)


# Every output row retains its dot8 accumulation, XOR reduction and epilogue.
gr = (ROOT / 'src/kernels/metal/fused_gr.metal').read_text()
down = gr[gr.index('kernel void fused_gr_down_kernel'):gr.index('// ---- the up projection')]
up = gr[gr.index('kernel void fused_gr_up_kernel'):]
variants = '#include "fused_gr.metal"\n'
dn_names = ['fused_gr_down_kernel']
up_names = ['fused_gr_up_kernel']
for warps in [4, 8, 16]:
    name = f'gr_down_w{warps}'
    dn_names.append(name)
    variants += down.replace('fused_gr_down_kernel', name).replace('FGR_DOWN_BLOCKS', str(320 // warps)).replace('FGR_WARPS', str(warps))
    for cols in [8, 16, 32]:
        name = f'gr_up_w{warps}_c{cols}'
        up_names.append(name)
        variants += up.replace('fused_gr_up_kernel', name).replace('FGR_UPM_COLS', str(cols)).replace('FGR_WARPS', str(warps))
host = (oldbench / 'gr_bench.mm').read_text()
host = host.replace('@"gr.metallib"', '@(argv[1])').replace('argc > 1 ? atoi(argv[1]) : 7', 'argc > 2 ? atoi(argv[2]) : 7')
host = host.replace('const int copies = 16;', 'const int copies = 32;')
host = host.replace('int rows = 1;\n            if (k == "gr_down_r2") rows = 2;\n            if (k == "gr_down_r4") rows = 4;',
                    'int warps = 8; sscanf(k.c_str(), "gr_down_w%d", &warps);')
host = host.replace('LR / (8 * rows) + 1', 'LR / warps + 1')
host = host.replace('threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];\n        };\n        auto up',
                    'threadsPerThreadgroup:MTLSizeMake(32 * warps, 1, 1)];\n        };\n        auto up')
host = host.replace('auto up = [&](id<MTLComputeCommandEncoder> e, const std::string& k, Out& o, int c) {',
                    'auto up = [&](id<MTLComputeCommandEncoder> e, const std::string& k, Out& o, int c) {\n            int warps = 8, cols = 16; sscanf(k.c_str(), "gr_up_w%d_c%d", &warps, &cols);')
host = host.replace('MTLSizeMake(N / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)',
                    'MTLSizeMake(N / cols, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * warps, 1, 1)')
begin = host.index('        // correctness:')
end = host.index('        for (auto& [k, v] : t)', begin)
checks = '''        std::vector<std::string> dns = {DN_NAMES}, ups = {UP_NAMES};
        run([&](id<MTLComputeCommandEncoder> e) { norm(e, "fused_gr_norm1_kernel", A); down(e, dns[0], A, 0); up(e, ups[0], A, 0); });
        for (const auto& dk : dns) for (const auto& uk : ups) {
            run([&](id<MTLComputeCommandEncoder> e) { norm(e, "fused_gr_norm1_kernel", B); down(e, dk, B, 0); up(e, uk, B, 0); });
            long different = same(A.rs, B.rs, 4) + same(A.xn, B.xn, D) + same(A.lo, B.lo, LR) + same(A.io, B.io, 4) + same(A.Rout, B.Rout, D) + same(A.mixed, B.mixed, N);
            if (different) { fprintf(stderr, "%s/%s different=%ld\\n", dk.c_str(), uk.c_str(), different); return 2; }
        }
        printf("# chain_checks=%zu bitdiff=0 apply=1 copies=32 reps=96\\n", dns.size() * ups.size());
        const int reps = 96;
        std::map<std::string, std::vector<double>> t;
        for (int r = 0; r < rounds; ++r) {
            if (r % 2) { std::reverse(dns.begin(), dns.end()); std::reverse(ups.begin(), ups.end()); }
            for (const auto& k : dns) t[k].push_back(run([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) down(e, k, A, i); }));
            for (const auto& k : ups) t[k].push_back(run([&](id<MTLComputeCommandEncoder> e) { for (int i = 0; i < reps; ++i) up(e, k, A, i); }));
        }
'''.replace('DN_NAMES', ', '.join('"' + n + '"' for n in dn_names)).replace('UP_NAMES', ', '.join('"' + n + '"' for n in up_names))
host = host[:begin] + checks + host[end:]
build('gr_schedule', variants, host)

# Specialize the K bound; leave every virtual warp's float partial chain in the original order.
nmv = (ROOT / 'src/kernels/metal/native_mmvq.metal').read_text()
body = nmv[nmv.index('kernel void native_iq4_xs_direct_sg4'):nmv.index('// rows and nw are runtime scalars')]
variants = '#include "native_mmvq.metal"\n'
for k in [2560, 6144]:
    variants += body.replace('native_iq4_xs_direct_sg4', f'dim_iq4_{k}').replace('n_in / Fmt::DIV', f'{k} / Fmt::DIV')
host = (oldbench / 'mmvq_bench.mm').read_text()
begin = host.index('        std::vector<std::string> variants')
end = host.index('        printf("K,N,variant', begin)
host = host[:begin] + '''        std::vector<std::string> variants = {"native_iq4_xs_direct_sg4", "dim_iq4"};
        const std::vector<std::pair<int, int>> shapes = {{2560, 248320}, {2560, 10240}, {2560, 6144}, {2560, 12288}, {6144, 2560}};
''' + host[end:]
# Retain only the raw view to keep the microbenchmark's working set bounded.
begin = host.index('            std::vector<uint8_t> e(ebytes);')
end = host.index('            std::vector<uint8_t> x(', begin)
host = host[:begin] + host[end:]
host = host.replace('const size_t target = 512ull << 20;', 'const size_t target = 256ull << 20;')
host = host.replace('            for (int i = 0; i < ec; ++i) eb.push_back([dev newBufferWithBytes:e.data() length:ebytes options:MTLResourceStorageModeShared]);', '')
host = host.replace('const bool fixed = v.rfind("vdf_r", 0) == 0 || v == "native_iq4_xs_direct_r4";', 'const bool fixed = true;')
host = host.replace('int rows = fixed ? (v == "native_iq4_xs_direct_r4" ? 4 : (int) (v[5] - 48)) : (small ? 4 : 1);', 'int rows = 16;')
host = host.replace('const std::string name = (fixed || sr) ? v : v + (small ? "_small" : "_large");', 'const std::string name = v == "dim_iq4" ? v + "_" + std::to_string(K) : v;')
host = host.replace('            for (int round = 0; round < rounds; ++round) {', '            for (int round = 0; round < rounds; ++round) {\n                if (round > 0) std::reverse(variants.begin(), variants.end());')
# The reference must always be the current kernel even when timing order reverses.
host = host.replace('if (v == variants[0] && round == 0)', 'if (v == "native_iq4_xs_direct_sg4" && round == 0)')
build('dense_dims', variants, host)

# Existing expert harness, retaining only dimension specialization and down row reuse candidates.
variants = '#include "iq_kernels.metal"\n'
for ty in [22, 16]:
    variants += f'''kernel void dim_gu_{ty}(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {{
        iqk_resident_body<{ty}, false>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, k, has_offsets, out, up, gp, tid);
    }}\n'''
for r in [2, 4, 8, 16]:
    variants += f'''kernel void dd{r}_down(IQK_RESIDENT_PARAMS, uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {{
        iqk_resident_down_direct_q2_0<{r}>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes, weight_offset, slot_bytes, n_expert, has_offsets, out, gp, tid);
    }}\n'''
host = (ROOT / 'bench/results/2026-10-03-metal-expert-gu/ex_bench.mm').read_text()
host = host.replace('@"ex.metallib"', '@(argv[1])').replace('argc > 1 ? atoi(argv[1]) : 7', 'argc > 2 ? atoi(argv[2]) : 7')
begin = host.index('            {\n                id<MTLCommandBuffer> cb', host.index('memset(BAD.contents'))
end = host.index('            const int reps =', begin)
host = host[:begin] + host[end:]
begin = host.index('            std::vector<std::string> guv;')
end = host.index('            timed(', begin)
host = host[:begin] + '''            std::vector<std::string> guv = {"dim_gu_" + T};
            const std::vector<std::string> dnv = {"dd2_down", "dd4_down", "dd8_down", "dd16_down"};
''' + host[end:]
host = host.replace('"native_resident_down_42"', '"native_resident_down_direct_42_r4"')
host = host.replace('                int R = kn.size() > 3', '                int R = kn.size() > 3', 1)
start = host.index('            auto dn =')
region_end = host.index('            const std::string ogu', start)
part = host[start:region_end]
begin = part.index('                int R =')
end = part.index('                [e dispatchThreadgroups', begin)
part = part[:begin] + '                int R = 4; sscanf(kn.c_str(), "dd%d_down", &R);\n' + part[end:]
host = host[:start] + part + host[region_end:]
build('expert_dims', variants, host)
