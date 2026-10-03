import csv,sys,collections
rows=[r for r in csv.DictReader(open(sys.argv[1])) if int(r['entries'])==int(sys.argv[2])]
agg=collections.OrderedDict()
for r in rows:
    k=r['kernel']; a=agg.setdefault(k,[0,0.0,0.0,set()])
    a[0]+=1; a[1]+=float(r['mean_us']); a[2]+=float(r['gap_us']); a[3].add((r['grid_x'],r['grid_y'],r['threads']))
tot=sum(v[1] for v in agg.values()); gap=sum(v[2] for v in agg.values())
print(f"entries {len(rows)} busy {tot/1000:.3f} ms gaps {gap/1000:.3f} ms")
for k,v in sorted(agg.items(),key=lambda x:-x[1][1])[:int(sys.argv[3]) if len(sys.argv)>3 else 40]:
    print(f"{k:44s} n={v[0]:4d} {v[1]/1000:7.3f}ms {100*v[1]/tot:5.1f}% mean={v[1]/v[0]:7.2f}us shapes={len(v[3])}")
