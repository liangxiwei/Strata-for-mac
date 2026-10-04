#!/usr/bin/env python3
"""Reuse independent references for focused dimension-specialization boundary tests."""
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUILD = ROOT / 'build-metal'
TMP = Path('/tmp/strata-first-principles/decode')
src = (ROOT / 'tests/core/metal_direct_dot_test.cpp').read_text()
src = src.replace('Launch d(direct ? "native_resident_down_direct_42_r4" : "native_resident_down_42",',
                  'Launch d(direct ? (n_embd == 2560 && n_ff == 640 ? "native_resident_down_dim_42_r4" : "native_resident_down_direct_42_r4") : "native_resident_down_42",')
src = src.replace('void resident_public_case(int n_tok) {\n    const long n_embd = 2560, n_ff = 640;',
                  'void resident_public_case(int n_tok, long n_embd = 2560, long n_ff = 640) {')
src = src[:src.index('int main() {')] + '''int main() {
    cudaStreamCreate(&stream);
    for (int mode = 0; mode < 3; ++mode) {
        for (int cap : {1, 10, 23, 40}) for (bool offsets : {false, true})
            resident_down_case(2560, 640, cap, offsets, mode);
        resident_down_case(2596, 640, 23, true, mode);
        resident_down_case(512, 128, 7, true, mode);
    }
    for (int t : {1, 3, 4}) { resident_public_case(t); resident_public_case(t, 512, 128); }
    cudaStreamDestroy(stream);
    printf("down dimensions: %lld outputs compared, %lld bit differences\\n", g_checked, g_diff);
    return g_diff != 0;
}
'''
candidate = TMP / 'down_dims_check.cpp'
candidate.write_text(src)
flags = {}
for line in (BUILD / 'CMakeFiles/metal_direct_dot_test.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
obj = TMP / 'down_dims_check.cpp.o'
subprocess.run(['clang++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
                '-c', str(candidate), '-o', str(obj)], check=True)
link = shlex.split((BUILD / 'CMakeFiles/metal_direct_dot_test.dir/link.txt').read_text())
link[link.index('CMakeFiles/metal_direct_dot_test.dir/tests/core/metal_direct_dot_test.cpp.o')] = str(obj)
link[link.index('-o') + 1] = str(TMP / 'down_dims_check')
link[link.index('libstrata_kernels.a')] = str(TMP / 'libstrata_kernels.a')
link[link.index('libstrata_metal_kernels.a')] = str(TMP / 'libstrata_metal_kernels.a')
subprocess.run(link, cwd=BUILD, check=True)
print(TMP / 'down_dims_check')
