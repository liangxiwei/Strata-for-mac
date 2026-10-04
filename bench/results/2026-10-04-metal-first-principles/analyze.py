#!/usr/bin/env python3
"""Reproduce the whole-stage distribution and tensor inventory from local evidence."""
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from tools.gguf_reader import GGUFFile


def category(kernel, stage):
    if stage == 'prefill':
        if kernel.startswith('native_gemm2_'):
            return 'expert_gemm'
        if kernel.startswith('pfl_gemm2_'):
            return 'dense_gemm'
        if kernel.startswith('prompt_attn_'):
            return 'attention'
        if kernel.startswith('dequant'):
            return 'dequant'
        if kernel.startswith('pfl_gdn_rec_'):
            return 'gdn_recurrence'
    else:
        if kernel.startswith('native_resident_'):
            return 'expert_gu_down'
        if kernel.startswith('fused_gr_'):
            return 'hyper_connection'
        if kernel == 'bf16_f32_mmvf_kernel':
            return 'bf16_projection'
        if kernel.startswith(('native_iq4_', 'native_q6_', 'mmvq_direct_',
                              'native_small_mmvq_', 'native_q2_0_')):
            return 'quant_dense_projection'
        if kernel.startswith(('attn_', 'block_scores_', 'block_topk_', 'nqsi_')):
            return 'qsa_attention_indexer'
        if kernel.startswith('gdn_'):
            return 'gdn'
        if 'quantize_q8_1' in kernel:
            return 'activation_quantize'
    return 'other'


def profile(stage):
    src = OUT / ('run/profile/kernels.csv.direct.csv' if stage == 'prefill' else 'run/profile/kernels.csv')
    dest = OUT / (stage + '-kernels.csv')
    kernels = collections.defaultdict(lambda: {'ms': 0.0, 'count': 0})
    if src.exists():
        for r in csv.DictReader(src.open()):
            k = r['kernel_grid_threads'].split()[0] if stage == 'prefill' else r['kernel']
            kernels[k]['ms'] += float(r['total_ms']) if stage == 'prefill' else float(r['mean_us']) / 1000
            kernels[k]['count'] += int(r['count']) if stage == 'prefill' else 1
        with dest.open('w') as f:
            w = csv.DictWriter(f, fieldnames=['kernel', 'category', 'ms', 'count'], lineterminator='\n')
            w.writeheader()
            for k, v in sorted(kernels.items(), key=lambda x: -x[1]['ms']):
                w.writerow({'kernel': k, 'category': category(k, stage), **v})
    else:
        for r in csv.DictReader(dest.open()):
            kernels[r['kernel']] = {'ms': float(r['ms']), 'count': int(r['count'])}
    categories = collections.defaultdict(float)
    for k, v in kernels.items():
        categories[category(k, stage)] += v['ms']
    total = sum(categories.values())
    return {'sampled_kernel_ms': total, 'count': sum(v['count'] for v in kernels.values()),
            'categories': [{'name': k, 'ms': v, 'percent': 100 * v / total}
                           for k, v in sorted(categories.items(), key=lambda x: -x[1])]}


def inventory():
    cfg = json.loads((ROOT / 'strata-iq2_xs.json').read_text())
    gguf = ROOT / cfg['args'][cfg['args'].index('--native') + 1]
    tensors = GGUFFile(gguf).tensors
    matrices = [t for t in tensors if len(t.shape) == 2 and '_exps' not in t.name
                and t.name != 'token_embd.weight' and not t.name.startswith('mtp.')]
    quant = [t for t in matrices if t.type_name not in ('F32', 'F16', 'BF16')]
    pack = ROOT / cfg['args'][cfg['args'].index('--pack') + 1]
    # Native experts manifest stores exact unpadded blob bytes for each layer.
    fmt = [list(map(int, s.split()[:5])) for s in (pack / 'native_experts.txt').read_text().splitlines()
           if s and not s.startswith('#')]
    experts = sum(r[4] * 512 for r in fmt)
    # Flag 4 in index.txt denotes the engine's BF16 view. This excludes FP16 conv/norm vectors.
    raw16 = sum(int(p[4]) for s in (pack / 'index.txt').read_text().splitlines()
                if not s.startswith('#') and (p := s.split()) and int(p[2]) & 4)
    quant_bytes = sum(t.expected_bytes() for t in quant)
    active = experts * 10 // 512
    weight_bytes = active + quant_bytes + raw16
    return {'model': str(gguf.relative_to(ROOT)), 'layers': 48, 'experts_per_layer': 512,
            'active_experts': 10, 'expert_weight_bytes': experts, 'active_expert_bytes_per_token': active,
            'dense_quant_matrix_count': len(quant), 'dense_quant_weight_bytes': quant_bytes,
            'pack_bf16_weight_bytes': raw16, 'approx_weight_bytes_per_decode_token': weight_bytes,
            'dense_matrix_flops_per_token': 2 * sum(t.elements for t in matrices),
            'expert_flops_per_token': 48 * 10 * 3 * 2 * 2560 * 640,
            'dense_quant_f16_view_bytes': 2 * sum(t.elements for t in quant),
            'full_experts_f16_view_bytes': 48 * 512 * 3 * 2560 * 640 * 2,
            'iq2_nibble_extra_bytes': sum(512 * 2 * 640 * (2560 // 256) * (140 - (82 if r[1] == 22 else 66))
                                         for r in fmt if r[1] in (22, 16)),
            'weights_only_floor_ms_at_400GBs': weight_bytes / 400e9 * 1000,
            'note': 'Tensor inventory estimate, not measured DRAM traffic; excludes state, KV, activations and redundant transactions.'}


def runs():
    dest = OUT / 'stage-runs.json'
    prior = json.loads(dest.read_text()) if dest.exists() else {}
    for f in sorted((OUT / 'run').glob('*/results.json')):
        d = json.loads(f.read_text())
        if 'records' not in d or 'config' not in d:
            continue
        rows = []
        for r in d['records']:
            ids = r['ids']
            rows.append({k: v for k, v in r.items() if k != 'ids'} | {
                'output_count': len(ids), 'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        prior[f.parent.name] = {'config': d['config'], 'info': d['info'], 'records': rows,
                                'metrics': d.get('metrics', {})}
    dest.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + '\n')
    return prior


working_set = int(re.search(r'GiB \((\d+) B\)', (OUT / 'device.txt').read_text())[1])
data = {'device': {'name': 'Apple M2 Max', 'gpu_cores': 38, 'physical_ram_bytes': 96 * 2**30,
                   'power_mode': 'Automatic, AC', 'metal_working_set_bytes': working_set},
        'head': 'e0307a0', 'prefill_profile': profile('prefill'), 'decode_profile': profile('decode'),
        'inventory': inventory(), 'runs': runs()}
(OUT / 'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in data.items() if k != 'runs'}, ensure_ascii=False, indent=2))
