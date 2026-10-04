#!/usr/bin/env python3
"""The README's two speed numbers on this Mac, the way the README defines them: "writes answers" = decode speed in a
short chat (a fresh short prompt, 256 tokens, temperature 0, thinking off), "reads your prompt" = a ~30K-token
document read into a fresh context (no prefix reuse). One engine, the daily config, run with no other engine loaded."""
import argparse, glob, hashlib, json, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
from tools.conversation_cache_parity import load_tokenizer

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config', type=Path, required=True)   # setup's strata-<model>.json
ap.add_argument('--doc-tokens', type=int, default=30000)
a = ap.parse_args()
out = Path(__file__).parent
name = a.config.stem[len('strata-'):]
run = out / 'run' / name
run.mkdir(parents=True, exist_ok=True)
cfg = json.loads(a.config.read_text())
env = child_env(cfg)
env['STRATA_DECODE_TIMING'] = '1'
tok = load_tokenizer(Path(cfg['tokenizer']))
tpl = ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja') if (Path(cfg['tokenizer']) / 'chat_template.jinja').exists() else None
def chat(text):
    if tpl is None:
        text = f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return tok.encode(text, parse_special=True)
    return tok.encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
chats = ['Write a detailed, 800-word article about how bridges are designed to survive earthquakes.',
         '请写一篇约 800 字的文章，介绍城市地铁系统是如何规划和建设的。',
         'Explain, in about 800 words, how a modern CPU executes instructions, from fetch to retire.',
         '请用约 800 字详细说明，一杯咖啡从咖啡豆到端上桌经历了哪些步骤。']
# the document: this repository's own English docs, cut to --doc-tokens with the chat template around it
doc = ''
for p in ['docs/DETAILS.md', 'docs/HOW_IT_WORKS.md', 'docs/MODELS.md', 'docs/INSTALL.md', 'docs/TROUBLESHOOTING.md',
          'docs/PORT_METAL/PROGRESS.md', 'docs/MULTI_GPU.md', 'docs/AMD_HIP.md']:
    doc += (ROOT / p).read_text() + '\n\n'
body = tok.encode(doc, parse_special=False)
head = chat('Read the document below, then summarize what it says about memory use in three sentences.\n\n')
doc_ids = head[:-8] + body[: a.doc_tokens - len(head)] + head[-8:]
config = {'exe': cfg['exe'], 'exe_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(), 'args': cfg['args'],
          'env': {k: v for k, v in env.items() if k.startswith('STRATA_')}, 'doc_tokens': len(doc_ids)}
t0 = time.monotonic()
e = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(run / 'engine.log'), env=env)
startup = time.monotonic() - t0
rows = []
try:
    for i, text in enumerate(chats):
        ids = chat(text)
        o = [t for t in e.generate(ids, 256, {'temperature': 0}, threading.Event()) if t is not None]
        r = dict(e.last)
        rows.append({'kind': 'short chat', 'prompt_tokens': len(ids), 'generated': len(o), 'reused': r.get('reused'),
                     'prompt_ms': r['prompt_ms'], 'decode_ms': r['decode_ms'],
                     'decode_tok_s': 1000 * len(o) / r['decode_ms'], 'text_start': tok.decode(o[:24])})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    o = [t for t in e.generate(doc_ids, 64, {'temperature': 0}, threading.Event()) if t is not None]
    r = dict(e.last)
    rows.append({'kind': 'document', 'prompt_tokens': len(doc_ids), 'generated': len(o), 'reused': r.get('reused'),
                 'prompt_ms': r['prompt_ms'], 'prefill_tok_s': 1000 * len(doc_ids) / r['prompt_ms'],
                 'decode_ms': r['decode_ms'], 'decode_tok_s': 1000 * len(o) / r['decode_ms'], 'text_start': tok.decode(o[:40])})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
finally:
    e.close()
(out / f'{name}.json').write_text(json.dumps({'config': config, 'startup_s': startup, 'runs': rows}, ensure_ascii=False, indent=2))
