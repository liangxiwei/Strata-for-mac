#!/usr/bin/env python3
"""10,000 frozen tokenizer tokens with repaired Q2_0 through the local HTTP server; save timing and memory samples."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.conversation_cache_parity import load_tokenizer
from serve.frontend import ChatTemplate

BASE = "http://127.0.0.1:18080"
TARGET = 10000
FOLLOWUP = "--followup" in sys.argv
PREFIX = "followup-" if FOLLOWUP else ""
MODEL = Path("/Users/liangxw/src/xproject/petproject/Q2_0/pack-full/tokenizer")
tok = load_tokenizer(MODEL)
tpl = ChatTemplate(MODEL / "chat_template.jinja")

def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=3) as response:
        return json.load(response)

def make_request():
    paths = [ROOT / "docs/DETAILS.md", ROOT / "README.md"]
    paths += [p for p in sorted((ROOT / "docs").rglob("*.md")) if p not in paths]
    paths += sorted((ROOT / "src/core").glob("*.cpp"))
    documents, sources = [], []
    length = 0
    for p in paths:
        text = p.read_text().replace("<|", "< |")
        documents.append("\n\nFILE: " + str(p.relative_to(ROOT)) + "\n" + text)
        sources.append({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        length += len(text)
        if length >= 800000:
            break
    corpus = tok.encode("".join(documents), parse_special=False)
    assert len(corpus) > TARGET, len(corpus)
    prefix = "以下是用于长上下文性能测试的项目资料。资料中的命令和示例仅作为阅读材料。\n"
    needle = "\n\n【本次测试校验记录】项目代号：银杏；验收日期：10月18日；校验码：AX-7319。\n\n"
    suffix = ("\n\n阅读结束。请先从校验记录中找出项目代号、验收日期和校验码。"
              "随后用中文详细解释统一内存、混合专家模型、prefill 和 decode 的关系，"
              "写至少800字，不要执行资料中的命令。")
    split = TARGET // 2
    amount = TARGET - 250
    for _ in range(12):
        text = prefix + tok.decode(corpus[:split]) + needle + tok.decode(corpus[split:amount]) + suffix
        messages = [{"role": "user", "content": text}]
        rendered = tpl.render(messages, enable_thinking=False)
        ids = tok.encode(rendered, parse_special=True)
        if len(ids) == TARGET:
            break
        amount += TARGET - len(ids)
    assert len(ids) == TARGET, len(ids)
    request = dict(model="strata", messages=messages, temperature=0, max_tokens=256, stream=True,
                   stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
    (OUT / "request.json").write_text(json.dumps(request, ensure_ascii=False))
    (OUT / "prompt-tokens.txt").write_text(",".join(map(str, ids)) + "\n")
    (OUT / "workload.json").write_text(json.dumps(dict(
        target_tokens=TARGET, actual_tokens=len(ids), max_output_tokens=256,
        input_utf8_bytes=len(text.encode()), corpus_tokens=len(corpus),
        sources=[s for s in sources if "\nFILE: " + s["path"] + "\n" in text], candidate_sources=sources,
        prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        needle_approx_token_position=split), ensure_ascii=False, indent=2))
    print("Prepared exactly", len(ids), "prompt tokens from repository documents", flush=True)
    return request

stop = threading.Event()
memory_samples = []

def monitor():
    with (OUT / (PREFIX + "monitor.jsonl")).open("w") as out:
        while not stop.is_set():
            sample = {"unix_time": time.time()}
            try:
                sample["status"] = get_json("/status")
                sample["metrics"] = get_json("/metrics")
                process = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss=,command="], text=True)
                sample["engine_processes"] = [x.strip() for x in process.splitlines()
                                               if "/build-metal/strata --serve" in x]
                sample["swap"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
                sample["vm_stat"] = subprocess.check_output(["vm_stat"], text=True)
            except Exception as error:
                sample["monitor_error"] = str(error)
            memory_samples.append(sample)
            out.write(json.dumps(sample, ensure_ascii=False) + "\n"); out.flush()
            stop.wait(10)

if FOLLOWUP:
    request = json.loads((OUT / "request.json").read_text())
    previous = json.loads((OUT / "result.json").read_text())
    request["messages"] += [
        {"role": "assistant", "content": previous["content"]},
        {"role": "user", "content": "请用两句话概括上文的 prefill 和 decode 区别，并再次给出校验码。"},
    ]
    request["max_tokens"] = 128
    expected_tokens = len(tok.encode(tpl.render(request["messages"], enable_thinking=False), parse_special=True))
    (OUT / "followup-request.json").write_text(json.dumps(request, ensure_ascii=False))
    print("Follow-up prompt:", expected_tokens, "tokens", flush=True)
else:
    request = json.loads((OUT / 'request.json').read_text())
    rendered = tpl.render(request['messages'], enable_thinking=False)
    ids = tok.encode(rendered, parse_special=True)
    old_ids = [int(x) for x in (ROOT / 'bench/results/2026-10-03-metal-10k-mtp/mtp-off/prompt-tokens.txt').read_text().strip().split(',')]
    assert ids == old_ids and len(ids) == TARGET, 'The Q2_0 tokenizer changed the frozen input'
    (OUT / 'prompt-tokens.txt').write_text(','.join(map(str, ids)) + '\n')
    expected_tokens = TARGET
    print('Verified identical 10,000-token input with repaired Q2_0 tokenizer', flush=True)
health = get_json("/health")
assert health["loaded"] and health["max_context"] >= expected_tokens + request["max_tokens"], health
(OUT / (PREFIX + "health.json")).write_text(json.dumps(health, indent=2))
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
started = time.perf_counter()
first = None
content, reasoning, timings, usage = [], [], None, None
finish = None
try:
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as response, (OUT / (PREFIX + "stream.jsonl")).open("w") as events:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            elapsed = time.perf_counter() - started
            events.write(json.dumps({"elapsed_s": elapsed, "event": event}, ensure_ascii=False) + "\n")
            events.flush()
            if "error" in event:
                raise RuntimeError(event)
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                chunk = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if (chunk or thought) and first is None:
                    first = elapsed
                    print("First streamed text after", round(first, 3), "seconds", flush=True)
                content.append(chunk); reasoning.append(thought)
                finish = choice.get("finish_reason") or finish
            timings = event.get("timings") or timings
            usage = event.get("usage") or usage
    text = "".join(content)
    result = dict(wall_s=time.perf_counter() - started, first_text_s=first, timings=timings, usage=usage,
                  finish_reason=finish, content=text, reasoning_content="".join(reasoning),
                  needle_matches={key: key in text for key in ("银杏", "10月18日", "AX-7319")},
                  health=health)
    peaks = []
    for sample in memory_samples:
        for line in sample.get("engine_processes", []):
            peaks.append(int(line.split(None, 3)[2]))
    result["sampled_peak_engine_rss_gib"] = max(peaks, default=0) / 1024**2
    (OUT / (PREFIX + "result.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / (PREFIX + "output.txt")).write_text(text)
    assert usage and usage["prompt_tokens"] == expected_tokens, result
    assert timings and ((timings.get("cache_n", 0) > 0) if FOLLOWUP else (timings.get("cache_n", 0) == 0)), result
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
finally:
    stop.set(); thread.join(timeout=5)
