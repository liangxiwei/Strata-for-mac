#!/usr/bin/env python3
"""Build an isolated opt-in cache experiment from the existing Metal build, without editing production source."""
import argparse
import difflib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, default=ROOT / 'build-metal')
p.add_argument('--scratch', type=Path, default=Path('/tmp/strata-first-principles'))
a = p.parse_args()
a.build = a.build.resolve()
a.scratch.mkdir(parents=True, exist_ok=True)
source = ROOT / 'src/prefill/gemm.mm'
old = source.read_text()
new = old.replace('#include <mutex>', '#include <mutex>\n#include <algorithm>\n#include <cstdint>\n#include <tuple>')
before = '    ~WidenScratch() { if (x) cudaFree(x); if (w) cudaFree(w); }'
after = '''    // Research only: the source weights are immutable for this Gemm's lifetime.
    using CacheKey = std::tuple<uintptr_t, int, int64_t, int64_t>;
    std::map<CacheKey, uint16_t*> cache;
    size_t cache_budget = 0, cache_used = 0;
    uint64_t cache_hits = 0, cache_misses = 0;
    bool cache_initialized = false;
    ~WidenScratch() {
        if (cache_initialized)
            std::fprintf(stderr, "prefill cache: used=%zu MiB budget=%zu MiB matrices=%zu hits=%llu misses=%llu\\n",
                         cache_used >> 20, cache_budget >> 20, cache.size(),
                         (unsigned long long) cache_hits, (unsigned long long) cache_misses);
        for (auto& item : cache) cudaFree(item.second);
        if (x) cudaFree(x);
        if (w) cudaFree(w);
    }'''
assert new.count(before) == 1
new = new.replace(before, after)
before = '''                  int64_t K, int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {'''
after = '''                  int64_t K, int64_t ldy, float beta) {
    static const uint64_t requested_mib = [] {
        const char* e = std::getenv("STRATA_METAL_PREFILL_CACHE_MIB");
        return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    if (requested_mib > 0 && N > 0 && K > 0 && N * K <= scratch_elems_) {
        auto* s = static_cast<WidenScratch*>(hipblaslt_state_);
        if (!s->cache_initialized) {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                // Bound this prototype by both the device budget and post-prefill-allocation headroom.
                const size_t requested = size_t(std::min<uint64_t>(requested_mib, 1ull << 20)) << 20;
                s->cache_budget = std::min({requested, total_bytes / 12, free_bytes / 4});
            }
            s->cache_initialized = true;
            std::fprintf(stderr, "prefill cache: initial budget=%zu MiB\\n", s->cache_budget >> 20);
        }
        const WidenScratch::CacheKey key{reinterpret_cast<uintptr_t>(W_blocks), ggml_type, N, K};
        if (const auto it = s->cache.find(key); it != s->cache.end()) {
            ++s->cache_hits;
            f16(X, it->second, Y, T, N, K, ldy, beta);
            return;
        }
        const size_t bytes = size_t(N * K) * 2;
        const size_t charged = (bytes + 16383) & ~size_t(16383);
        if (charged <= s->cache_budget - s->cache_used) {
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, bytes) == cudaSuccess) {
                auto* weight = static_cast<uint16_t*>(ptr);
                strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, weight, stream_);
                s->cache.emplace(key, weight);
                s->cache_used += charged;
                ++s->cache_misses;
                // The same dequantized bits, shape, tile and accumulation order as the original path.
                f16(X, weight, Y, T, N, K, ldy, beta);
                return;
            }
            cudaGetLastError();
        }
    }
    if (N * K > scratch_elems_) {'''
assert new.count(before) == 1
new = new.replace(before, after)
patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                   fromfile='a/src/prefill/gemm.mm', tofile='b/src/prefill/gemm.mm'))
(OUT / 'dense-cache.patch').write_text(patch)
candidate = a.scratch / 'gemm.mm'
candidate.write_text(new)
flags = {}
for line in (a.build / 'CMakeFiles/strata_prefill.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
obj = a.scratch / 'gemm.mm.o'
compile_cmd = ['clang++', *flags['OBJCXX_DEFINES'], *flags['OBJCXX_INCLUDES'], *flags['OBJCXX_FLAGS'],
               '-c', str(candidate), '-o', str(obj)]
subprocess.run(compile_cmd, check=True)
archive = a.scratch / 'libstrata_prefill.a'
shutil.copy2(a.build / 'libstrata_prefill.a', archive)
subprocess.run(['ar', 'r', str(archive), str(obj)], check=True)
subprocess.run(['ranlib', str(archive)], check=True)
link_cmd = shlex.split((a.build / 'CMakeFiles/strata.dir/link.txt').read_text())
exe = a.scratch / 'strata-dense-cache'
link_cmd[link_cmd.index('-o') + 1] = str(exe)
link_cmd[link_cmd.index('libstrata_prefill.a')] = str(archive)
subprocess.run(link_cmd, cwd=a.build, check=True)
(OUT / 'run/build.json').write_text(json.dumps({'compile': compile_cmd, 'link': link_cmd, 'engine': str(exe)}, indent=2))
print(exe)
