#!/usr/bin/env python3
"""Summarize sequential 10K/256 decode runs and enforce comparable outputs/settings."""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
reference = json.loads((ROOT / 'baseline/results.json').read_text())
expected = reference['records'][0]['ids']
rows = {}
for folder in sorted(ROOT.iterdir()):
    path = folder / 'results.json'
    if not path.is_file() or folder.name.startswith('audit'):
        continue
    data = json.loads(path.read_text())
    if data.get('config', {}).get('tokens') != 256:
        continue  # Profiling changes scheduling and is not a performance result.
    config = data['config']
    assert config['args'] == reference['config']['args'], folder
    assert config['prompt_sha256'] == reference['config']['prompt_sha256'], folder
    for i, record in enumerate(data['records']):
        assert record['ids'] == expected, (folder, i, 'output differs')
        assert record['generated'] == 256 and record['prompt_tokens'] == 10000, folder
        assert record['drafts_offered'] == record['drafts_accepted'] == 0, folder
        assert record['hits'] == record['lookups'], (folder, 'expert cache misses')
        assert record['reused'] == (9993 if i else 0), (folder, 'prefix reuse differs')
    cold, *warm = data['records']
    warm_ms = [x['decode_ms'] for x in warm]
    log = (folder / 'engine.log').read_text()
    extra_mib = sum(float(x) for x in re.findall(r'lossless IQ4 .*? ([\d.]+) MiB', log))
    rows[folder.name] = {
        'sha256': config['sha256'], 'env': config['env'],
        'extra_view_mib': extra_mib,
        'prefill_tok_s': 10000 / cold['prompt_ms'] * 1000,
        'cold_decode_ms': cold['decode_ms'],
        'cold_decode_tok_s': 256000 / cold['decode_ms'],
        'warm_decode_ms': warm_ms,
        'warm_median_ms': statistics.median(warm_ms) if warm_ms else None,
        'warm_tok_s': [256000 / x for x in warm_ms],
        'metrics': data.get('metrics', {}),
        'output_ids_identical': True,
    }
for row in rows.values():
    if row['warm_median_ms']:
        row['warm_median_tok_s'] = 256000 / row['warm_median_ms']
        row['warm_speedup_vs_round20_pct'] = (rows['baseline']['warm_median_ms'] / row['warm_median_ms'] - 1) * 100
        row['saved_ms_per_token_vs_round20'] = (rows['baseline']['warm_median_ms'] - row['warm_median_ms']) / 256
(ROOT / 'summary.json').write_text(json.dumps(rows, indent=2))
print('variant               extra MiB  cold t/s  warm t/s  vs r20')
for name, row in rows.items():
    print(f'{name:22s} {row["extra_view_mib"]:9.1f} {row["cold_decode_tok_s"]:9.3f} '
          f'{row.get("warm_median_tok_s", 0):9.3f} {row.get("warm_speedup_vs_round20_pct", 0):+7.2f}%')
