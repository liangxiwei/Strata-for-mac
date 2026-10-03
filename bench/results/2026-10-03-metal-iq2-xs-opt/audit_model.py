#!/usr/bin/env python3
"""Real-model numerical audit; run variants sequentially, with no public server."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env
from tools.conversation_cache_parity import load_tokenizer, state_hashes
from serve.frontend import ChatTemplate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--engine', type=Path)
p.add_argument('--mode', choices=['reference','prefill','resident'], required=True)
p.add_argument('--case', choices=['10k','short'], default='short')
p.add_argument('--shadow-windows', type=int, default=0)
p.add_argument('--logits-windows', type=int, default=1000)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
config = json.loads((Path(__file__).parent / 'tiled/server-config.json').read_text())
if a.engine: config['exe'] = str(a.engine.resolve())
args = config['args'] + ['--adapt-swaps', '0', '--prompt-cache', '6']
if a.case == 'short':
    # All prompt rows go through verify windows: targets stay identical across variants.
    args += ['--short-read', '1024']
env = child_env(config)
for key in ['STRATA_METAL_IQ_RESIDENT', 'STRATA_METAL_LAYER_SYNC', 'STRATA_METAL_PREFILL_F16_EXPERTS']:
    env.pop(key, None)
env['STRATA_METAL_IQ_RESIDENT'] = '1' if a.mode == 'resident' else '0'
if a.mode == 'reference': env['STRATA_METAL_PREFILL_F16_EXPERTS'] = '1'
env['STRATA_STATE_HASH'] = '1'
env['STRATA_STATE_HASH_GDN'] = '1'
env['STRATA_METAL_LOGITS_PREFIX'] = str((a.output/'logits').resolve())
env['STRATA_METAL_LOGITS_WINDOWS'] = str(a.logits_windows)
if a.shadow_windows:
    env['STRATA_METAL_CHECK_IQ_RESIDENT'] = str(a.shadow_windows)
tokenizer = load_tokenizer(Path(config['tokenizer']))
tpl = ChatTemplate(Path(config['tokenizer'])/'chat_template.jinja')
if a.case == '10k':
    ids = [int(x) for x in (Path(__file__).parent/'tiled/prompt-tokens.txt').read_text().strip().split(',')]
    cases = [('frozen_10k', ids, 64)]
else:
    # Fixed text includes reasoning, code and multilingual retrieval; one sampled row at each end.
    texts = [
        ('math', '请检查下面的计算过程是否正确，只回答正确或错误：17×23=17×(20+3)=340+51=391；391-125=266；266÷7=38。'),
        ('code', '请检查Python函数，并只输出调用结果。\ndef f(xs):\n    seen=set()\n    out=[]\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\nprint(f([3,1,3,2,1,2]))'),
        ('logic', '档案写道：红盒子里有蓝钥匙，绿盒子里有白钥匙，蓝盒子里有红钥匙。取出红盒子中的钥匙放入绿盒子，不移动其余物品。现在蓝钥匙在哪个盒子？只回答颜色。'),
        ('english', 'A shop starts with 24 apples. It sells 7, receives 15, and then sells half of the remaining apples. The calculation is 24-7+15=32, then 32/2=16. Return only the number of apples left.'),
    ]
    cases = [(name, tokenizer.encode(tpl.render([{'role':'user','content':text}], enable_thinking=False), parse_special=True), 1)
             for name, text in texts]
(a.output/'config.json').write_text(json.dumps({'exe': config['exe'], 'args':args,'mode':a.mode,'case':a.case,
    'env':{k:v for k,v in env.items() if k.startswith('STRATA_')},
    'exe_sha256':hashlib.sha256(Path(config['exe']).read_bytes()).hexdigest()}, indent=2))
engine = StrataEngine(config['exe'], args, cwd=config['cwd'], log=str(a.output/'engine.log'), env=env)
records=[]
try:
    for name, ids, count in cases:
        print(f'{a.mode}: {name}, {len(ids)} prompt tokens', flush=True)
        out=[t for t in engine.generate(ids,count,{'temperature':0},threading.Event()) if t is not None]
        record={'name':name,'prompt_ids':ids,'ids':out,'text':tokenizer.decode(out),**engine.last}
        records.append(record)
        if name == 'frozen_10k':
            suffix=tokenizer.encode('<|im_end|>\n<|im_start|>user\n再次给出校验码。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
            cases.append(('cached_followup',ids+out+suffix,24))
        if (record.get('reused', 0) > 0) != (name == 'cached_followup'):
            raise AssertionError('unexpected prompt cache reuse')
        print(f'{name}: {record["text"]!r}',flush=True)
        (a.output/'results.json').write_text(json.dumps({'records':records,'info':engine.info},ensure_ascii=False,indent=2))
finally:
    engine.close()
log_text=(a.output/'engine.log').read_text()
hashes=state_hashes(log_text)
if a.shadow_windows:
    checks=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9e.+-]+)',line)}
            for line in log_text.splitlines() if 'strata IQ check:' in line]
    if len(checks) != a.shadow_windows*48 or any(c['different'] or c['nonfinite'] for c in checks):
        raise AssertionError('real expert outputs differ or checks are incomplete')
(a.output/'results.json').write_text(json.dumps({'records':records,'state_hashes':hashes,'info':engine.info},ensure_ascii=False,indent=2))
