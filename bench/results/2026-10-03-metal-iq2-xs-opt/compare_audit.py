#!/usr/bin/env python3
"""Compare actual-model audit logits and committed state, refusing unmatched inputs."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
x=json.loads((a.reference/'results.json').read_text());y=json.loads((a.candidate/'results.json').read_text())
files=lambda d: sorted(d.glob('logits-*.bin'),key=lambda p:int(p.stem.split('-')[-1]))
rf,cf=files(a.reference),files(a.candidate)
summary={'reference':str(a.reference),'candidate':str(a.candidate),'trace_files':[len(rf),len(cf)],'windows':[]}
summary['same_requests']=len(x['records'])==len(y['records']) and all(u['name']==v['name'] and u['prompt_ids']==v['prompt_ids'] for u,v in zip(x['records'],y['records']))
summary['same_output_ids']=len(x['records'])==len(y['records']) and all(u['ids']==v['ids'] for u,v in zip(x['records'],y['records']))
summary['same_committed_states']=len(x['state_hashes'])==len(y['state_hashes'])==len(x['records']) and bool(x['state_hashes']) and x['state_hashes']==y['state_hashes']
keys=('kv','kv_resident','context','spec','mtp_max','lookup','cvec','expert_slots')
summary['same_model_settings']=all(key in x['info'] and key in y['info'] and x['info'][key]==y['info'][key] for key in keys)
summary['no_drafts']=all(r['drafts_offered']==0 and r['drafts_accepted']==0 for r in x['records']+y['records'])
all_equal=bool(rf) and len(rf)==len(cf) and summary['same_requests'] and summary['same_output_ids'] and summary['same_committed_states'] and summary['same_model_settings'] and summary['no_drafts']
total_values=total_rows=total_different=0

def read(path):
    with path.open('rb') as f:
        pos,t,nv=struct.unpack('<qqq',f.read(24))
        inp=np.fromfile(f,dtype='<i4',count=t);out=np.fromfile(f,dtype='<i4',count=t)
        logits=np.fromfile(f,dtype='<f4')
    assert logits.size==t*nv and nv>0 and t>0,path
    return (pos,t,nv),inp,out,logits.reshape(t,nv)

for r,c in zip(rf,cf):
    hm,hi,ho,h=read(r);km,ki,ko,k=read(c)
    equal_inputs=hm==km and np.array_equal(hi,ki)
    row={'file':r.name,'position':hm[0],'rows':hm[1],'same_inputs':equal_inputs}
    if equal_inputs:
        different=int(np.count_nonzero(h.view('u4')!=k.view('u4')))
        finite=bool(np.isfinite(h).all() and np.isfinite(k).all())
        d=h.astype('f8')-k.astype('f8')
        row.update(values=int(h.size),different=different,finite=finite,max_abs=float(np.abs(d).max()),
                   rel_l2=float(np.sqrt(np.sum(d*d)/max(float(np.sum(h.astype('f8')**2)),1e-30))),
                   same_top1_rows=int(np.count_nonzero(np.argmax(h,axis=1)==np.argmax(k,axis=1))))
        total_values+=h.size;total_rows+=hm[1];total_different+=different
        all_equal=all_equal and different==0 and finite
    else:
        all_equal=False
    summary['windows'].append(row)
summary.update(rows=total_rows,values=total_values,different=total_different,bitwise_pass=bool(all_equal))
a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='windows'},ensure_ascii=False,indent=2))
if not all_equal: raise SystemExit(1)
