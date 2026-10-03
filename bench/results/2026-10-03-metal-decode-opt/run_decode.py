#!/usr/bin/env python3
"""Private engine decode measurement on the frozen 10K prompt; profile runs are diagnostic only."""
import argparse, hashlib, json, os, resource, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from serve.server import StrataEngine,child_env
from tools.conversation_cache_parity import load_tokenizer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',type=Path,default=ROOT/'build-metal/strata')
p.add_argument('--tokens',type=int,default=256);p.add_argument('--repeats',type=int,default=1)
p.add_argument('--config',type=Path,default=ROOT/'bench/results/2026-10-03-metal-iq2-xs-opt/tiled/server-config.json')
p.add_argument('--env',action='append',default=[]);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
old=ROOT/'bench/results/2026-10-03-metal-iq2-xs-opt'
cfg=json.loads(a.config.read_text());env=child_env(cfg)
env['STRATA_DECODE_TIMING']='1';env['STRATA_TRACE']='1'
for item in a.env:
    key,value=item.split('=',1);env[key]=value
ids=[int(x) for x in (old/'tiled/prompt-tokens.txt').read_text().strip().split(',')]
tok=load_tokenizer(Path(cfg['tokenizer']))
config={'exe':str(a.engine.resolve()),'sha256':hashlib.sha256(a.engine.read_bytes()).hexdigest(),'args':cfg['args'],
        'config_path':str(a.config.resolve()),'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
        'env':{k:v for k,v in env.items() if k.startswith('STRATA_')},'tokens':a.tokens,'repeats':a.repeats,
        'prompt_tokens':len(ids),'prompt_sha256':hashlib.sha256((old/'tiled/prompt-tokens.txt').read_bytes()).hexdigest()}
(a.output/'config.json').write_text(json.dumps(config,indent=2))
def swap_usage():
    return subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()
metrics={'swap_before':swap_usage()}
started=time.monotonic()
e=StrataEngine(str(a.engine.resolve()),cfg['args'],cwd=cfg['cwd'],log=str(a.output/'engine.log'),env=env);records=[]
metrics['startup_seconds']=time.monotonic()-started
info=e.info
try:
    for i in range(a.repeats):
        request_start=time.monotonic();out=[];timeline=[]
        for t in e.generate(ids,a.tokens,{'temperature':0},threading.Event()):
            if t is not None:
                out.append(t)
                if len(out)==1 or len(out)%32==0:
                    timeline.append({'tokens':len(out),'elapsed_ms':(time.monotonic()-request_start)*1000})
        r={'run':i,'ids':out,'text':tok.decode(out),'timeline':timeline,**e.last};records.append(r)
        if not out or r['finish'] not in ['stop','length'] or r['drafts_offered']!=0: raise AssertionError(r)
        if i and out!=records[0]['ids']: raise AssertionError('repeat output differs')
        print(json.dumps({k:v for k,v in r.items() if k not in ['ids','text','timeline']}),flush=True)
        (a.output/'results.json').write_text(json.dumps({'config':config,'info':e.info,'records':records},ensure_ascii=False,indent=2))
finally:
    e.close()
    metrics['swap_after']=swap_usage()
    metrics['child_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    (a.output/'results.json').write_text(json.dumps({'config':config,'info':info,'records':records,'metrics':metrics},ensure_ascii=False,indent=2))
