#!/usr/bin/env python3
"""strata-vision on the Mac's GPU (Metal) against the same program on the CPU: the same pictures, the same token grid,
and embeddings that agree to float rounding (they are different kernels, so not bit for bit).  Also the time per
picture and the encoder's memory.  -> encoder.json"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--exe", default=str(ROOT / "build-vision-metal" / "bin" / "strata-vision"))
ap.add_argument("--mmproj", default=str(ROOT / "Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"))
ap.add_argument("--model", default=str(ROOT / "Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"))
a = ap.parse_args()
run = HERE / "run"
run.mkdir(exist_ok=True)


def read_sve(p):
    raw = Path(p).read_bytes()
    magic, n, nx, ny, d = np.frombuffer(raw[:20], dtype=np.int32)
    assert magic == 0x31455653, "not an SVE1 file"
    return int(n), int(nx), int(ny), np.frombuffer(raw[20:], dtype=np.float32).reshape(n, d)


def encode(gpu, extra=()):
    args = [a.exe, "--mmproj", a.mmproj, "--model", a.model, *(["--gpu"] if gpu else []), *extra]
    t0 = time.monotonic()
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(run / f"vision-{'gpu' if gpu else 'cpu'}.log", "w"),
                         text=True, bufsize=1)
    ready = p.stdout.readline().strip()
    start_s = time.monotonic() - t0
    assert ready.startswith("READY"), ready
    rss = subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
    foot = subprocess.run(["vmmap", "-summary", str(p.pid)], capture_output=True, text=True).stdout
    foot = next((l.split(":", 1)[1].strip() for l in foot.splitlines() if l.startswith("Physical footprint:")), None)
    res = {}
    for img in sorted((HERE / "images").iterdir()):
        out = run / f"{img.stem}-{'gpu' if gpu else 'cpu'}.sve"
        p.stdin.write(f"ENC {img} {out}\n")
        p.stdin.flush()
        line = p.stdout.readline().strip()
        assert line.startswith("OK"), line
        res[img.name] = {"line": line, "file": out, "ms": float(line.split()[4])}
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=60)
    return {"start_s": start_s, "rss_kib": int(rss or 0), "footprint": foot, "images": res}


gpu, cpu = encode(True), encode(False, ("--max-tokens", "1024"))     # the CPU at the GPU's token cap: same grid
rows = []
for name in gpu["images"]:
    n1, nx1, ny1, e1 = read_sve(gpu["images"][name]["file"])
    n2, nx2, ny2, e2 = read_sve(cpu["images"][name]["file"])
    same = (n1, nx1, ny1) == (n2, nx2, ny2)
    cos = float(np.min(np.sum(e1 * e2, 1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)))) if same else None
    rel = float(np.max(np.abs(e1 - e2)) / np.max(np.abs(e2))) if same else None
    rows.append({"image": name, "tokens": n1, "grid": [nx1, ny1], "same_grid_as_cpu": same,
                 "min_row_cosine": cos, "max_abs_diff_over_max": rel, "finite": bool(np.isfinite(e1).all()),
                 "gpu_ms": gpu["images"][name]["ms"], "cpu_ms": cpu["images"][name]["ms"]})
    print(json.dumps(rows[-1]))
summary = {"exe": a.exe, "gpu": {k: v for k, v in gpu.items() if k != "images"},
           "cpu": {k: v for k, v in cpu.items() if k != "images"}, "images": rows}
(HERE / "encoder.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary["gpu"]), json.dumps(summary["cpu"]))
sys.exit(0 if all(r["same_grid_as_cpu"] and r["finite"] and r["min_row_cosine"] > 0.99 for r in rows) else 1)
