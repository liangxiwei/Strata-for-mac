#!/usr/bin/env python3
"""Profile the installed optimized engine; sampled pass costs are diagnostic only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = ROOT / 'bench/results/2026-10-11-metal-mtp-optimization'
spec = importlib.util.spec_from_file_location('bench', PREVIOUS / 'run_benchmark.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
measure = bench.measure
measure.URL = 'http://127.0.0.1:8082'

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')

assert not bench.processes()
cfg = json.loads((ROOT / 'bench/results/2026-10-11-meld-comparison/config-r0-strata.json').read_text())
cfg['args'][cfg['args'].index('--mtp-max-t') + 1] = '4'
cfg['args'][cfg['args'].index('--mtp') + 1] = str(PREVIOUS / 'rt-subset')
cfg['env'].update(STRATA_METAL_KPROFILE=str(OUT / 'kernels.csv'), STRATA_METAL_KPROFILE_SKIP='4',
                  STRATA_DECODE_TIMING='1', STRATA_METAL_HC_BATCH='1', STRATA_METAL_GDN_SNAPSHOT='0')
cfg['log'] = str(OUT / 'engine.log')
save('config.json', cfg)
log = (OUT / 'server.log').open('w')
p = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'serve/server.py'), '--engine', 'strata',
                      '--config', str(OUT / 'config.json'), '--host', '127.0.0.1', '--port', '8082'],
                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    start = time.monotonic()
    while True:
        if p.poll() is not None:
            raise RuntimeError('Server exited; inspect log')
        try:
            h = measure.request_json('/health')
            if h.get('status') == 'ok' and h.get('loaded', True):
                break
        except OSError:
            pass
        assert time.monotonic() - start < 900
        time.sleep(2)
    print(json.dumps({'ready_seconds': time.monotonic() - start}), flush=True)
    prompt = (ROOT / 'bench/results/2026-10-11-meld-comparison/prompt-r0-1000.txt').read_text()
    for window in (1, 4):
        bench.tuned.extra = {'strata_tune': {'mtp_max_t': window, 'spec_min_p': 0.0}}
        offset = (OUT / 'engine.log').stat().st_size
        result = measure.run(prompt, 160)
        save(f'result-t{window}.json', result)
        with (OUT / 'engine.log').open('rb') as f:
            f.seek(offset)
            (OUT / f'trace-t{window}.log').write_bytes(f.read())
        print(json.dumps({'window': window, 'tokens': result['gen_tokens'], 'timings': result['timings']}), flush=True)
        if (OUT / 'kernels.csv').exists():
            (OUT / f'kernels-after-t{window}.csv').write_bytes((OUT / 'kernels.csv').read_bytes())
finally:
    bench.comparison.shutdown(p)
    log.close()
    save('cleanup.json', {'remaining_inference_processes': bench.processes()})
    assert not bench.processes()
