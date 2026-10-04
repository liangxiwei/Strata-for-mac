#!/usr/bin/env python3
"""Pictures through the whole Mac path: setup's config (strata-iq2_xs.json, images on), the server, strata-vision on
Metal, the engine's GENI/M-RoPE path.  Each test picture gets a question whose answer is known; a follow-up about a
picture in the conversation, and a text-only question, ride along.  One engine; temperature 0, thinking off.
-> e2e.json (answers, checks, times, memory, the engine's expert cache)"""
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
CFG = ROOT / "strata-iq2_xs.json"
PORT = 8080
IMG = HERE / "images"


def url(name):
    data = (IMG / name).read_bytes()
    return f"data:image/{'jpeg' if name.endswith('.jpg') else 'png'};base64," + base64.b64encode(data).decode()


def ask(messages, max_tokens=160):
    body = json.dumps({"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "none"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"], r.get("usage", {}), time.monotonic() - t0


def pic(name, text):
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(name)}},
                                         {"type": "text", "text": text}]}]


CASES = [
    ("receipt", pic("receipt.png", "这张小票上的店名和总金额是多少？"), [["STRATA"], ["COFFEE"], ["12.60"]]),
    ("shapes", pic("shapes.png", "图里有哪几个形状？分别是什么颜色？"), [["圆"], ["方", "正方"], ["三角"], ["红"], ["蓝"], ["绿"]]),
    ("dots", pic("dots.png", "图里一共有几个圆点？只回答数字。"), [["7", "七"]]),
    ("chinese", pic("chinese.png", "把图里的文字原样写出来。"), [["苹果电脑运行大模型"]]),
    ("screenshot", pic("screenshot.jpg", "Describe this screenshot in two sentences."), []),
    ("text only", [{"role": "user", "content": "17 乘以 23 等于多少？只回答数字。"}], [["391"]]),
]


def footprint(pid):
    s = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True).stdout
    return next((l.split(":", 1)[1].strip() for l in s.splitlines() if l.startswith("Physical footprint")), None)


log = ROOT / json.loads(CFG.read_text())["log"]
log_start = log.stat().st_size if log.exists() else 0
run = HERE / "run"
run.mkdir(exist_ok=True)
t0 = time.monotonic()
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "strata",
                        "--config", str(CFG), "--port", str(PORT)], cwd="/tmp",
                       stdout=open(run / "server.log", "w"), stderr=subprocess.STDOUT)
try:
    health = None
    for _ in range(300):
        time.sleep(2)
        try:
            health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2))
            if health.get("loaded"):
                break
        except OSError:
            pass
    ready_s = time.monotonic() - t0
    assert health and health.get("loaded") and health.get("images"), health
    rows = []
    for name, msgs, checks in CASES:
        text, usage, s = ask(msgs)
        ok = all(any(w in text for w in alts) for alts in checks)
        rows.append({"case": name, "answer": text, "checks": checks, "pass": ok if checks else None,
                     "seconds": round(s, 2), "usage": usage})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    # a follow-up about the picture already in the conversation
    first = rows[2]["answer"]
    msgs = pic("dots.png", "图里一共有几个圆点？只回答数字。") + [{"role": "assistant", "content": first},
                                                     {"role": "user", "content": "这些圆点是什么颜色的？"}]
    text, usage, s = ask(msgs)
    rows.append({"case": "follow-up (dots)", "answer": text, "checks": [["橙", "orange", "Orange"]],
                 "pass": any(w in text for w in ("橙", "orange", "Orange")), "seconds": round(s, 2), "usage": usage})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    pids = subprocess.run(["pgrep", "-f", "engine/strata"], capture_output=True, text=True).stdout.split()
    mem = {}
    for p in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", p], capture_output=True, text=True).stdout
        mem["strata-vision" if "strata-vision" in cmd else "engine"] = footprint(p)
finally:
    srv.send_signal(signal.SIGTERM)
    srv.wait(timeout=120)
engine_log = log.read_bytes()[log_start:].decode("utf-8", "replace") if log.exists() else ""
(run / "engine.log").write_text(engine_log)
cache = [l for l in engine_log.splitlines() if "expert cache" in l or "token graph hit path" in l or "VRAM free" in l]
summary = {"config": str(CFG.relative_to(ROOT)), "ready_s": round(ready_s, 1), "health": health, "memory": mem,
           "engine": cache, "cases": rows}
(HERE / "e2e.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print(json.dumps({k: summary[k] for k in ("ready_s", "memory", "engine")}, ensure_ascii=False, indent=1))
sys.exit(0 if all(r["pass"] is not False for r in rows) else 1)
