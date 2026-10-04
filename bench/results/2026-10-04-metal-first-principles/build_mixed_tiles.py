#!/usr/bin/env python3
"""Build an isolated mixed-expert-tile engine. Production files and the daily executable stay untouched."""
import difflib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
build = ROOT / 'build-metal'
scratch = Path('/tmp/strata-first-principles')
old = (ROOT / 'src/prefill/prefill.cpp').read_text()
anchor = old.index('                    if (native_tiles) {')
begin = old.index('                        std::vector<int32_t> tiles;', anchor)
end = old.index('                        stats_.experts_resident += order.size();', begin)
original = old[begin:end]
new_block = '''                        static const bool mixed_tiles = [] {
                            const char* e = std::getenv("STRATA_METAL_PREFILL_MIXED_TILES");
                            return e && std::atoi(e) != 0;
                        }();
                        if (mixed_tiles && tile_rows > 0) {
                            std::vector<int32_t> tiles16, tiles32;
                            for (const int32_t e : order) {
                                const int slot = m.host_res[(size_t) l * m.g->n_expert + e];
                                const uint64_t offset = (uint64_t) (m.cache->device_slot(slot) - base);
                                for (int32_t r = 0; r < m.cnt[(size_t) e]; ) {
                                    const int32_t remain = m.cnt[(size_t) e] - r;
                                    const int bm = remain > 16 && (tile_rows == 32 || m.cnt[(size_t) e] >= 32) ? 32 : 16;
                                    auto& tiles = bm == 16 ? tiles16 : tiles32;
                                    tiles.push_back((int32_t) (uint32_t) offset);
                                    tiles.push_back(m.off[(size_t) e] + r);
                                    tiles.push_back(std::min(bm, remain));
                                    tiles.push_back((int32_t) (uint32_t) (offset >> 32));
                                    r += bm;
                                }
                            }
                            const size_t split = tiles16.size();
                            tiles16.insert(tiles16.end(), tiles32.begin(), tiles32.end());
                            cudaMemcpyAsync(m.metal_tiles, tiles16.data(), tiles16.size() * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, m.cs);
                            auto product = [&](const uint16_t* x, float* y, bool down) {
                                if (split) strata::kernels::native_expert_gemm(native_tile_layout, x, base,
                                    m.metal_tiles, y, split / 4, down, m.cs, 16);
                                if (!tiles32.empty()) strata::kernels::native_expert_gemm(native_tile_layout, x, base,
                                    m.metal_tiles + split, y, tiles32.size() / 4, down, m.cs, 32);
                            };
                            pt.mark(kPfGemmGU, cs);
                            product(m.Xs, m.GU, false);
                            swiglu_interleaved(m.GU, m.Hh, T * K, m.cs);
                            pt.mark(kPfGemmD, cs);
                            product(m.Hh, m.Dm, true);
                        } else {
''' + original + '''                        }
'''
new = old[:begin] + new_block + old[end:]
(OUT / 'mixed-tiles.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), fromfile='a/src/prefill/prefill.cpp', tofile='b/src/prefill/prefill.cpp')))
candidate = scratch / 'prefill.cpp'
candidate.write_text(new)
flags = {}
for line in (build / 'CMakeFiles/strata_prefill.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
obj = scratch / 'prefill.cpp.o'
cmd = ['clang++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
       '-c', str(candidate), '-o', str(obj)]
subprocess.run(cmd, check=True)
archive = scratch / 'libstrata_prefill_mixed.a'
shutil.copy2(build / 'libstrata_prefill.a', archive)
subprocess.run(['ar', 'r', str(archive), str(obj)], check=True)
subprocess.run(['ranlib', str(archive)], check=True)
link = shlex.split((build / 'CMakeFiles/strata.dir/link.txt').read_text())
exe = scratch / 'strata-mixed-tiles'
link[link.index('-o') + 1] = str(exe)
link[link.index('libstrata_prefill.a')] = str(archive)
subprocess.run(link, cwd=build, check=True)
(OUT / 'run/mixed-build.json').write_text(json.dumps({'compile': cmd, 'link': link}, indent=2))
print(exe)
