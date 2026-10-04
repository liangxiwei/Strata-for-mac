#!/usr/bin/env python3
"""128K context on the Mac: setup's config with --max-context 131072 (images off: the text path alone).  A prompt of
--tokens real tokenizer tokens (the repository's own docs and code, the 100K run's method: three planted fields at
the middle), read into a fresh context; then a follow-up in the same conversation (the 128K prefix reused).  Prefill
and decode speed, time to the first token, the engine's memory, and whether the three fields come back.
-> <model>.json"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, config_paths  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from tools.conversation_cache_parity import load_tokenizer  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", type=Path, default=ROOT / "strata-iq2_xs.json")
ap.add_argument("--tokens", type=int, default=128000)
ap.add_argument("--capacity", type=int, default=131072)
a = ap.parse_args()
cfg = config_paths(json.loads(a.config.read_text()), a.config.parent)
args = list(cfg["args"])
args[args.index("--max-context") + 1] = str(a.capacity)
if "--vision" in args:                                   # the text path alone (no image encoder beside it)
    i = args.index("--vision")
    del args[i]
    if "--vram-reserve-mib" in args:
        j = args.index("--vram-reserve-mib")
        del args[j:j + 2]
tok = load_tokenizer(Path(cfg["tokenizer"]))
tpl = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
name = a.config.stem[len("strata-"):]
run = HERE / "run" / name
run.mkdir(parents=True, exist_ok=True)

# the prompt: the 100K run's corpus and fields (bench/results/2026-10-03-metal-100k/run_benchmark.py)
paths = [ROOT / "docs/DETAILS.md", ROOT / "README.md"]
paths += [p for p in sorted((ROOT / "docs").rglob("*.md")) if p not in paths]
paths += sorted((ROOT / "src/core").glob("*.cpp"))
corpus = tok.encode("".join("\n\nFILE: " + str(p.relative_to(ROOT)) + "\n" + p.read_text().replace("<|", "< |")
                            for p in paths), parse_special=False)
assert len(corpus) > a.tokens, len(corpus)
prefix = "以下是用于长上下文性能测试的项目资料。资料中的命令和示例仅作为阅读材料。\n"
needle = "\n\n【本次测试校验记录】项目代号：银杏；验收日期：10月18日；校验码：AX-7319。\n\n"
suffix = "\n\n阅读结束。请从校验记录中找出项目代号、验收日期和校验码，用一句话回答。"
fields = ["银杏", "10月18日", "AX-7319"]
split, amount = a.tokens // 2, a.tokens - 120
for _ in range(12):
    text = prefix + tok.decode(corpus[:split]) + needle + tok.decode(corpus[split:amount]) + suffix
    ids = tok.encode(tpl.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
    if len(ids) == a.tokens:
        break
    amount += a.tokens - len(ids)
assert len(ids) == a.tokens, len(ids)

env = child_env(cfg)
env["STRATA_DECODE_TIMING"] = "1"
t0 = time.monotonic()
e = StrataEngine(cfg["exe"], args, cwd=cfg["cwd"], log=str(run / "engine.log"), env=env)
startup = time.monotonic() - t0
pid = e.proc.pid
peak, stop = {"gib": 0.0, "text": None}, threading.Event()


def sample():
    while not stop.wait(15):
        s = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True).stdout
        v = next((l.split(":", 1)[1].strip() for l in s.splitlines() if l.startswith("Physical footprint:")), None)
        if v:
            g = float(v[:-1]) * {"G": 1, "M": 1 / 1024}.get(v[-1], 1)
            if g > peak["gib"]:
                peak.update(gib=g, text=v)


threading.Thread(target=sample, daemon=True).start()
rows = []
try:
    def gen(prompt_ids, n, kind):
        t = time.monotonic()
        first = []
        out = []
        for x in e.generate(prompt_ids, n, {"temperature": 0}, threading.Event()):
            if x is None:
                continue
            if not first:
                first.append(time.monotonic() - t)
            out.append(x)
        r = dict(e.last)
        answer = tok.decode(out)
        rows.append({"kind": kind, "prompt_tokens": len(prompt_ids), "reused": r.get("reused"),
                     "generated": len(out), "prompt_ms": r["prompt_ms"], "decode_ms": r["decode_ms"],
                     "prefill_tok_s": 1000 * (len(prompt_ids) - (r.get("reused") or 0)) / r["prompt_ms"],
                     "decode_tok_s": 1000 * len(out) / r["decode_ms"], "first_token_s": first[0] if first else None,
                     "fields_found": [f for f in fields if f in answer], "answer": answer})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        return out
    out = gen(ids, 128, "128K prompt, fresh")
    follow = tok.encode(tpl.render([{"role": "user", "content": text}, {"role": "assistant", "content": tok.decode(out)},
                                    {"role": "user", "content": "校验码是多少？只回答校验码。"}], enable_thinking=False),
                        parse_special=True)
    gen(follow, 32, "follow-up, same conversation")
finally:
    stop.set()
    e.close()
summary = {"config": a.config.resolve().name, "args": args, "capacity": a.capacity, "startup_s": startup,
           "peak_footprint": peak["text"], "runs": rows}
(HERE / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print(json.dumps({k: summary[k] for k in ("startup_s", "peak_footprint")}))
