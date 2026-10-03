#!/usr/bin/env python3
"""Summarize the same-binary 10K/256 decode A/B; every run must reproduce round 21's reference output ids."""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
reference = json.loads((ROOT.parent / '2026-10-03-metal-decode-opt/baseline/results.json').read_text())
expected = reference['records'][0]['ids']
rows, shas = {}, set()
for folder in sorted(ROOT.iterdir()):
    path = folder / 'results.json'
    if not path.is_file() or folder.name.startswith(('audit', 'profile')):
        continue
    data = json.loads(path.read_text())
    config = data['config']
    if config.get('tokens') != 256:
        continue  # profiling runs change scheduling; they are not performance results
    assert config['args'] == reference['config']['args'], folder
    assert config['prompt_sha256'] == reference['config']['prompt_sha256'], folder
    for i, record in enumerate(data['records']):
        assert record['ids'] == expected, (folder, i, 'output differs from round 21')
        assert record['generated'] == 256 and record['prompt_tokens'] == 10000, folder
        assert record['drafts_offered'] == record['drafts_accepted'] == 0, folder
        assert record['hits'] == record['lookups'], (folder, 'expert cache misses')
        assert record['reused'] == (9993 if i else 0), (folder, 'prefix reuse differs')
    shas.add(config['sha256'])
    cold, *warm = data['records']
    warm_ms = [x['decode_ms'] for x in warm]
    log = (folder / 'engine.log').read_text()
    rows[folder.name] = {
        'sha256': config['sha256'],
        'env': {k: v for k, v in config['env'].items() if k not in ('STRATA_DECODE_TIMING', 'STRATA_TRACE')},
        'extra_view_mib': sum(float(x) for x in re.findall(r'lossless IQ4 .*? ([\d.]+) MiB', log)),
        'prefill_tok_s': 10000 / cold['prompt_ms'] * 1000,
        'cold_decode_tok_s': 256000 / cold['decode_ms'],
        'warm_decode_ms': warm_ms,
        'warm_median_ms': statistics.median(warm_ms),
        'warm_median_tok_s': 256000 / statistics.median(warm_ms),
        'metrics': data.get('metrics', {}),
        'output_ids_identical_to_round21': True,
    }
assert len(shas) == 1, ('runs used different binaries', shas)
control = statistics.median([r['warm_median_ms'] for n, r in rows.items() if n.startswith('control')])
for row in rows.values():
    row['vs_control_median_pct'] = (control / row['warm_median_ms'] - 1) * 100
    row['saved_ms_per_token'] = (control - row['warm_median_ms']) / 256
(ROOT / 'summary.json').write_text(json.dumps({'binary_sha256': shas.pop(), 'runs': rows}, indent=2))
print('run            extra MiB  prefill t/s  cold t/s  warm median t/s  warm t/s (3 repeats)   vs control')
for name, r in rows.items():
    warm = ' / '.join(f'{256000 / x:.2f}' for x in r['warm_decode_ms'])
    print(f'{name:14s} {r["extra_view_mib"]:9.1f} {r["prefill_tok_s"]:12.1f} {r["cold_decode_tok_s"]:9.2f} '
          f'{r["warm_median_tok_s"]:16.2f}  {warm:22s} {r["vs_control_median_pct"]:+6.1f}%')
