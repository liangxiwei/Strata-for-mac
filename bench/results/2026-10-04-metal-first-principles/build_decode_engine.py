#!/usr/bin/env python3
"""Build a same-binary off/on expert-down experiment without changing production files."""
from pathlib import Path
import difflib
import json
import shlex
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUILD = ROOT / 'build-metal'
TMP = Path('/tmp/strata-first-principles/decode')
TMP.mkdir(parents=True, exist_ok=True)

metal_old = (ROOT / 'src/kernels/metal/iq_kernels.metal').read_text()
addition = '''
// The canonical model geometry, preserving every lane's call and float accumulation sequence.
kernel void native_resident_down_dim_42_r4(IQK_RESIDENT_PARAMS,
                                          uint3 gp [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
    iqk_resident_down_direct_q2_0<4>(arena, offsets, ids, residency, xq, 2560, 640, row_bytes,
                                    weight_offset, slot_bytes, n_expert, has_offsets, out, gp, tid);
}
'''
metal_new = metal_old if 'kernel void native_resident_down_dim_42_r4' in metal_old else metal_old + addition
metal = TMP / 'iq_kernels.metal'
metal.write_text(metal_new)
sources = [metal if s.name == 'iq_kernels.metal' else s for s in sorted((ROOT / 'src/kernels/metal').glob('*.metal'))]
subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-fno-fast-math', '-ffp-contract=off',
                '-I', str(ROOT / 'src/kernels/metal'), *map(str, sources), '-o', str(TMP / 'strata.metallib')], check=True)
gen = TMP / 'strata_metallib.mm'
subprocess.run(['cmake', '-DIN=' + str(TMP / 'strata.metallib'), '-DOUT=' + str(gen),
                '-DNAME=strata_metallib_bytes', '-P', str(ROOT / 'cmake/embed_binary.cmake')], check=True)

old = (ROOT / 'src/kernels/metal/iq_kernels.mm').read_text()
anchor = '    if (direct) std::snprintf(name, sizeof name, "native_resident_down_direct_42_r4");'
replacement = '''    static const bool specialize_down = [] {
        const char* e = std::getenv("STRATA_METAL_DECODE_DOWN_DIMS");
        return e && std::atoi(e) != 0;
    }();
    const bool canonical_down = specialize_down && L.n_embd == 2560 && L.n_ff == 640;
    if (direct) std::snprintf(name, sizeof name, canonical_down ? "native_resident_down_dim_42_r4"
                                                               : "native_resident_down_direct_42_r4");'''
if anchor in old:
    assert old.count(anchor) == 1
    new = old.replace(anchor, replacement)
else:
    assert 'native_resident_down_dim_42_r4' in old
    new = old
host = TMP / 'iq_kernels.mm'
host.write_text(new)
patch = ''
for before, after, path in [(metal_old, metal_new, 'src/kernels/metal/iq_kernels.metal'),
                           (old, new, 'src/kernels/metal/iq_kernels.mm')]:
    patch += ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile='a/' + path, tofile='b/' + path))
if patch:
    (OUT / 'decode/down-dims.patch').write_text(patch)


def replace_object(target, source, object_name):
    flags = {}
    for line in (BUILD / 'CMakeFiles' / (target + '.dir') / 'flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            flags[key] = shlex.split(value)
    obj = TMP / object_name
    subprocess.run(['clang++', *flags['OBJCXX_DEFINES'], *flags['OBJCXX_INCLUDES'], *flags['OBJCXX_FLAGS'],
                    '-c', str(source), '-o', str(obj)], check=True)
    archive = TMP / ('lib' + target + '.a')
    shutil.copy2(BUILD / ('lib' + target + '.a'), archive)
    subprocess.run(['ar', 'r', str(archive), str(obj)], check=True)
    subprocess.run(['ranlib', str(archive)], check=True)
    return archive


hostlib = replace_object('strata_kernels', host, 'iq_kernels.mm.o')
metallib = replace_object('strata_metal_kernels', gen, 'strata_metallib.mm.o')
link = shlex.split((BUILD / 'CMakeFiles/strata.dir/link.txt').read_text())
exe = TMP / 'strata-down-dims'
link[link.index('-o') + 1] = str(exe)
link[link.index('libstrata_kernels.a')] = str(hostlib)
link[link.index('libstrata_metal_kernels.a')] = str(metallib)
subprocess.run(link, cwd=BUILD, check=True)
(OUT / 'run/decode-engine-build.json').write_text(json.dumps({'sources': list(map(str, sources)), 'link': link}, indent=2))
print(exe)
