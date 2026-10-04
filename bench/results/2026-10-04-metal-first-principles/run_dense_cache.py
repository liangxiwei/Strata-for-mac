#!/usr/bin/env python3
"""One isolated run on the frozen 10K input; sample physical footprint outside timed requests."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine', type=Path, default=Path('/tmp/strata-first-principles/strata-dense-cache'))
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cache-mib', type=int, default=0)
p.add_argument('--tokens', type=int, default=128)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
cfg = json.loads((ROOT / 'strata-iq2_xs.json').read_text())
env = child_env(cfg)
env.update(STRATA_DECODE_TIMING='1', STRATA_TRACE='1', STRATA_METAL_PREFILL_CACHE_MIB=str(a.cache_mib))
prompt = ROOT / 'bench/results/2026-10-03-metal-iq2-xs-opt/tiled/prompt-tokens.txt'
ids = [int(x) for x in prompt.read_text().strip().split(',')]
config = {'engine': str(a.engine), 'sha256': hashlib.sha256(a.engine.read_bytes()).hexdigest(),
          'args': cfg['args'], 'cache_mib': a.cache_mib, 'prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
          'env': {k: v for k, v in env.items() if k.startswith('STRATA_')}}
(a.output / 'config.json').write_text(json.dumps(config, indent=2))
def swap():
    return subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
metrics = {'swap_before': swap(), 'memory_samples': []}
started = time.monotonic()
e = StrataEngine(str(a.engine), cfg['args'], cwd=str(ROOT), log=str(a.output / 'engine.log'), env=env)
metrics['startup_s'] = time.monotonic() - started
info = e.info
records = []
def memory():
    s = subprocess.check_output(['vmmap', '-summary', str(e.proc.pid)], text=True, stderr=subprocess.STDOUT)
    found = re.findall(r'^Physical footprint[^\n]*', s, re.MULTILINE)
    metrics['memory_samples'].append(found)
    with (a.output / 'footprint.txt').open('a') as f:
        f.write(s + '\n')
try:
    memory()
    for i in range(a.repeats):
        output = [t for t in e.generate(ids, a.tokens, {'temperature': 0}, threading.Event()) if t is not None]
        record = {'run': i, 'ids': output, **e.last}
        assert output and record['drafts_offered'] == 0
        if i:
            assert output == records[0]['ids']
        records.append(record)
        memory()
        print(json.dumps({k: v for k, v in record.items() if k != 'ids'}), flush=True)
finally:
    e.close()
    metrics['swap_after'] = swap()
    (a.output / 'results.json').write_text(json.dumps({'config': config, 'info': info,
                                                      'records': records, 'metrics': metrics}, indent=2))
