#!/usr/bin/env python3
"""Compile the isolated cache benchmarks with the engine's strict floating-point flags."""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
scratch = Path('/tmp/strata-first-principles')
scratch.mkdir(parents=True, exist_ok=True)
for name in ['expert_cache', 'expert_nibble']:
    subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-fno-fast-math', '-ffp-contract=off',
                    '-I', str(ROOT / 'src/kernels/metal'), str(OUT / 'micro' / (name + '.metal')),
                    '-o', str(scratch / (name + '.metallib'))], check=True)
    subprocess.run(['clang++', '-std=c++20', '-O2', '-fobjc-arc', str(OUT / 'micro' / (name + '.mm')),
                    '-framework', 'Metal', '-framework', 'Foundation',
                    '-o', str(scratch / name)], check=True)
print(scratch)
