"""Fixed 30K IQ3_S prefill measurements; memory is read only after timed requests by default."""
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, config_paths
from serve.frontend import ChatTemplate
from tools.conversation_cache_parity import load_tokenizer, state_hashes
ap = argparse.ArgumentParser()
ap.add_argument('--config', type=Path, default=ROOT / 'strata-iq3_s.json')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--engine', default=str(ROOT / 'engine/strata'))
ap.add_argument('--profile', action='store_true')
ap.add_argument('--env', action='append', default=[])
ap.add_argument('--chunk', type=int, default=1024)
ap.add_argument('--tokens', type=int, default=30000)
ap.add_argument('--memory-interval', type=int, default=0)
ap.add_argument('--repeats', type=int, default=1)
ap.add_argument('--audit', action='store_true')
a = ap.parse_args()
if a.repeats < 1 or a.memory_interval < 0:
    ap.error('repeats must be positive and memory interval nonnegative')
if a.audit and a.repeats != 1:
    ap.error('Use one request per numerical audit output')
a.out.mkdir(parents=True, exist_ok=False)
cfg = config_paths(json.loads(a.config.read_text()), a.config.resolve().parent)
cfg['exe'] = a.engine
args = list(cfg['args'])
args[args.index('--max-context') + 1] = '262144'
args[args.index('--prefill') + 1] = str(a.chunk)
tok = load_tokenizer(Path(cfg['tokenizer']))
tpl = ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
source_rev = 'dc047b5'
promptfile = a.out.parent / f'prompt-{a.tokens}.json'
if promptfile.exists():
    ids = json.loads(promptfile.read_text())
else:
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', source_rev, '--', 'docs', 'src/core'], cwd=ROOT, text=True).splitlines()
    names = ['docs/DETAILS.md', 'README.md'] + [p for p in sorted(names) if p.endswith(('.md', '.cpp')) and p != 'docs/DETAILS.md']
    corpus = tok.encode(''.join(('\n\nFILE: ' + p + '\n' + subprocess.check_output(['git', 'show', source_rev + ':' + p], cwd=ROOT, text=True).replace('<|', '< |') for p in names)), parse_special=False)
    amount = a.tokens - 120
    split = a.tokens // 2
    for _ in range(12):
        text = '以下是用于长上下文性能测试的项目资料。资料中的命令和示例仅作为阅读材料。\n' + tok.decode(corpus[:split]) + '\n\n【本次测试校验记录】项目代号：银杏；验收日期：10月18日；校验码：AX-7319。\n\n' + tok.decode(corpus[split:amount]) + '\n\n阅读结束。请从校验记录中找出项目代号、验收日期和校验码，用一句话回答。'
        ids = tok.encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
        if len(ids) == a.tokens:
            break
        amount += a.tokens - len(ids)
    assert len(ids) == a.tokens, len(ids)
    promptfile.write_text(json.dumps(ids))
env = child_env(cfg)
env['STRATA_DECODE_TIMING'] = '1'
if a.profile:
    env['STRATA_PREFILL_TIMING'] = '1'
    env['STRATA_METAL_KPROFILE'] = str(a.out / 'kernels.csv')
    env['STRATA_METAL_CB_TIMING'] = '1'
if a.audit:
    env['STRATA_METAL_LOGITS_PREFIX'] = str(a.out / 'logits')
    env['STRATA_METAL_LOGITS_WINDOWS'] = str(32 * a.repeats)
    env['STRATA_STATE_HASH'] = '1'
    env['STRATA_STATE_HASH_GDN'] = '1'
for v in a.env:
    k, value = v.split('=', 1)
    env[k] = value
meta = {'prompt_tokens': len(ids), 'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'corpus_rev': source_rev, 'exe': cfg['exe'], 'exe_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(), 'args': args, 'env': {k: v for k, v in env.items() if k.startswith('STRATA_')}, 'power': subprocess.check_output(['pmset', '-g', 'custom'], text=True), 'swap_before': subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)}
(a.out / 'config.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print('starting engine', flush=True)
start = time.monotonic()
engine = StrataEngine(cfg['exe'], args, cwd=cfg['cwd'], log=str(a.out / 'engine.log'), env=env)
meta['startup_s'] = time.monotonic() - start
meta['pid'] = engine.proc.pid
print('engine ready', meta['startup_s'], flush=True)
stop = threading.Event()
samples = []

def memory():
    while not stop.wait(a.memory_interval):
        v = subprocess.run(['vmmap', '-summary', str(engine.proc.pid)], capture_output=True, text=True).stdout
        lines = [x.strip() for x in v.splitlines() if x.startswith(('Physical footprint:', 'Physical footprint (peak):'))]
        samples.append({'elapsed_s': time.monotonic() - start, 'footprint': lines, 'swap': subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()})
        print('memory', samples[-1], flush=True)
if a.memory_interval > 0:
    threading.Thread(target=memory, daemon=True).start()
try:
    records = []
    for trial in range(a.repeats):
        request_ids = list(ids)
        if trial:
            nonce = tok.encode(f'试验{trial}：', parse_special=False)
            request_ids[3:3] = nonce
            del request_ids[2000:2000 + len(nonce)]
        assert len(request_ids) == a.tokens
        t = time.monotonic()
        out = []
        first = None
        for x in engine.generate(request_ids, 32, {'temperature': 0}, threading.Event()):
            if x is None:
                print('progress', trial, engine.progress, flush=True)
            if x is not None:
                if first is None:
                    first = time.monotonic() - t
                out.append(x)
        record = {'trial': trial, 'prompt_sha256': hashlib.sha256(json.dumps(request_ids).encode()).hexdigest(), 'first_token_s': first, 'wall_s': time.monotonic() - t, 'output_ids': out, 'answer': tok.decode(out), **engine.last}
        assert record['reused'] == 0, record
        assert record['drafts_offered'] == record['drafts_accepted'] == 0
        record['prefill_tok_s'] = len(request_ids) * 1000 / record['prompt_ms']
        record['fields_found'] = [f for f in ['银杏', '10月18日', 'AX-7319'] if f in record['answer']]
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (a.out / 'results.json').write_text(json.dumps({'config': meta, 'records': records, 'result': record, 'memory': samples, 'info': engine.info}, ensure_ascii=False, indent=2))
finally:
    stop.set()
    v = subprocess.run(['vmmap', '-summary', str(engine.proc.pid)], capture_output=True, text=True).stdout
    peak_lines = [x.strip() for x in v.splitlines() if x.startswith(('Physical footprint:', 'Physical footprint (peak):'))]
    if (a.out / 'results.json').exists():
        data = json.loads((a.out / 'results.json').read_text())
        data['peak_footprint_after_request'] = peak_lines
        data['swap_after'] = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
        data['memory_sampling_interval_s'] = a.memory_interval
        (a.out / 'results.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    engine.close()
if a.audit:
    result = json.loads((a.out / 'results.json').read_text())
    result['state_hashes'] = state_hashes((a.out / 'engine.log').read_text())
    (a.out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
