#!/usr/bin/env python3
"""Summarize all whole-decode pairs, preserving cold and every warm sample."""
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
runs = json.loads((OUT / 'stage-runs.json').read_text())
pairs = [
    ('private-off-on', 'decode-down-off-a', 'decode-down-on-a'),
    ('private-on-off', 'decode-down-off-b', 'decode-down-on-b'),
    ('production-build-off-on', 'decode-build-off', 'decode-build-on'),
]
data = {'method': 'Record 0 is cold; median uses every subsequent request. Timing excludes model load, tokenizer and HTTP.',
        'pairs': [], 'weight_cache_bytes_added': 0, 'scratch_bytes_added': 0}
hashes = set()
for label, off_key, on_key in pairs:
    off, on = runs[off_key], runs[on_key]
    assert off['config']['sha256'] == on['config']['sha256']
    assert off['config']['args'] == on['config']['args']
    assert off['config']['prompt_sha256'] == on['config']['prompt_sha256']
    for side, expected in [(off, '0'), (on, '1')]:
        assert side['config']['env']['STRATA_METAL_DECODE_DOWN_DIMS'] == expected
        for r in side['records']:
            assert r['output_count'] == r['generated'] == 256
            assert r['drafts_offered'] == 0 and r['hits'] == r['lookups']
            hashes.add(r['ids_sha256'])
    a = [r['decode_ms'] for r in off['records'][1:]]
    b = [r['decode_ms'] for r in on['records'][1:]]
    x, y = statistics.median(a), statistics.median(b)
    cold = lambda side: {k: side['records'][0][k] for k in ('prompt_ms', 'decode_ms')}
    total = lambda side: statistics.median(r['prompt_ms'] + r['decode_ms'] for r in side['records'][1:])
    data['pairs'].append({'label': label, 'off': off_key, 'on': on_key, 'engine_sha256': off['config']['sha256'],
                          'warm_decode_ms_off': a, 'warm_decode_ms_on': b, 'median_ms_off': x, 'median_ms_on': y,
                          'decode_time_reduction_percent': 100 * (1 - y / x),
                          'decode_tok_s_increase_percent': 100 * (x / y - 1),
                          'median_tok_s_off': 256000 / x, 'median_tok_s_on': 256000 / y,
                          'saved_ms_per_token': (x - y) / 256,
                          'cold_off': cold(off), 'cold_on': cold(on),
                          'warm_prompt_plus_decode_median_ms_off': total(off),
                          'warm_prompt_plus_decode_median_ms_on': total(on),
                          'metrics_off': off['metrics'], 'metrics_on': on['metrics']})
assert len(hashes) == 1
data['same_256_output_ids_all_runs'] = True
data['output_ids_sha256'] = next(iter(hashes))
audit = json.loads((OUT / 'decode/audit-comparison.json').read_text())
data['audit'] = {k: audit[k] for k in ('same_requests', 'same_output_ids', 'same_committed_states',
                                     'same_model_settings', 'no_drafts', 'rows', 'values', 'different', 'bitwise_pass')}
(OUT / 'decode/summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
for p in data['pairs']:
    print(f"{p['label']}: {p['median_ms_off']:.1f} -> {p['median_ms_on']:.1f} ms; tok/s +{p['decode_tok_s_increase_percent']:.3f}%")
