#!/usr/bin/env python3
"""One loaded IQ3_S engine at a time; named points permit alternating paired runs."""
import argparse
import hashlib
import importlib.util
import json
import os
import math
from pathlib import Path
import subprocess
import time
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REFERENCE = ROOT / "bench/results/2026-10-11-meld-comparison"
spec = importlib.util.spec_from_file_location("comparison", REFERENCE / "compare_api.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
measure = comparison.measure
measure.OUT = OUT
measure.URL = "http://127.0.0.1:8082"
original_http = measure.HTTP


class TunedHttp:
    extra = {}

    def open(self, req, *args, **kwargs):
        if isinstance(req, urllib.request.Request) and req.full_url.endswith("/v1/chat/completions"):
            body = json.loads(req.data)
            body.update(self.extra)
            req = urllib.request.Request(req.full_url, data=json.dumps(body).encode(), headers=dict(req.headers))
        return original_http.open(req, *args, **kwargs)


tuned = TunedHttp()
measure.HTTP = tuned


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def processes():
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    return [r for r in rows.splitlines() if r.split()[-1].split("/")[-1] in
            ("strata", "llama-server", "llama-bench")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="/tmp/strata-metal-mtp-20261011/build/strata")
    parser.add_argument("--point", action="append", required=True,
                        help="name:window:hc_batch:full_or_subset:round[:gdn_snapshot]")
    parser.add_argument("--sweep", action="append", help="request_label:window:round[:confidence_threshold]")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="after the screening sweep, alternate the best MTP window and off for rounds 1 and 2")
    args = parser.parse_args()
    assert not processes(), "Previous inference process is alive"
    subset = OUT / "rt-subset"
    subset.mkdir(exist_ok=True)
    for p in (ROOT / "Strata-data/mtp/rt").iterdir():
        if p.is_file() and p.name != "draft_vocab.bin" and not (subset / p.name).exists():
            (subset / p.name).symlink_to(p.resolve())
    vocab = ROOT / "data/draft_vocab.bin"
    if not (subset / vocab.name).exists():
        (subset / vocab.name).symlink_to(vocab.resolve())
    save("machine.json", {"power": subprocess.check_output(["pmset", "-g", "custom"], text=True),
                          "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
                          "engine": args.engine,
                          "engine_sha256": hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),
                          "subset_tokens": vocab.stat().st_size // 4,
                          "context_capacity": 131072, "output_tokens": 512, "points": args.point})
    results = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else []
    for point in args.point:
        parts = point.split(":")
        label, window, batch, vocabulary, round_id = parts[:5]
        engine_hash = hashlib.sha256(Path(args.engine).read_bytes()).hexdigest()
        cfg = json.loads((REFERENCE / "config-r0-strata.json").read_text())
        cfg["exe"] = args.engine
        cfg["args"][cfg["args"].index("--mtp-max-t") + 1] = window
        if vocabulary == "subset":
            cfg["args"][cfg["args"].index("--mtp") + 1] = str(subset)
        cfg["env"] = dict(cfg["env"], STRATA_METAL_HC_BATCH=batch,
                          STRATA_DECODE_TIMING="1", STRATA_METAL_GDN_SNAPSHOT=parts[5] if len(parts) > 5 else "0")
        cfg["log"] = str(OUT / f"engine-{label}.log")
        save(f"config-{label}.json", cfg)
        log = (OUT / f"server-{label}.log").open("w")
        process = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"),
                                    "--engine", "strata", "--config", str(OUT / f"config-{label}.json"),
                                    "--host", "127.0.0.1", "--port", "8082"],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited; inspect log")
                try:
                    health = measure.request_json("/health")
                    if health.get("status") == "ok" and health.get("loaded", True):
                        break
                except OSError:
                    pass
                if time.monotonic() - started > 900:
                    raise RuntimeError("Startup timeout")
                time.sleep(2)
            print(json.dumps({"label": label, "ready_s": time.monotonic() - started}), flush=True)
            save(f"warmup-{label}.json", measure.run("Write a detailed tutorial on Python generators with examples.", 64))
            requests = args.sweep or [f"{label}:{window}:{round_id}"]
            screen_count = len(requests)
            screened_labels = {r.split(":")[0] for r in requests}
            for request in requests:
                request_parts = request.split(":")
                request_label, request_window, request_round = request_parts[:3]
                if args.sweep:
                    tuned.extra = {"strata_tune": {"mtp_max_t": int(request_window),
                                                   "spec_min_p": float(request_parts[3]) if len(request_parts) > 3 else .5}}
                for length in (1000, 4000):
                  prompt = (REFERENCE / f"prompt-r{request_round}-{length}.txt").read_text()
                  path = Path(cfg["log"])
                  offset = path.stat().st_size
                  result = measure.run(prompt, 512)
                  assert result["gen_tokens"] == 512 and result["timings"]["cache_n"] == 0
                  assert not result["reasoning_content"]
                  assert (result["timings"]["draft_n"] > 0) == (int(request_window) > 1)
                  golden = json.loads((REFERENCE / f"result-r{request_round}-strata-{length}.json").read_text())
                  result.update(label=request_label, round=int(request_round), window=int(request_window), hc_batch=int(batch),
                                engine_sha256=engine_hash, gdn_snapshot=int(parts[5]) if len(parts) > 5 else 0,
                                vocabulary=vocabulary, approx_prompt_tokens=length,
                                identical_to_strata_off=result["content"] == golden["content"],
                                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                  with path.open("rb") as f:
                      f.seek(offset)
                      trace = f.read().decode()
                  (OUT / f"trace-{request_label}-{length}.log").write_text(trace)
                  result["decode_round_timing"] = [s for s in trace.splitlines() if s.startswith("strata decode timing:")]
                  save(f"result-{request_label}-{length}.json", result)
                  compact = {k: v for k, v in result.items() if k not in ("events", "content", "reasoning_content")}
                  results.append(compact)
                  save("summary.json", results)
                  print(json.dumps(compact), flush=True)
                if args.auto_confirm and request == requests[screen_count - 1]:
                    cases = [r for r in results if r["label"] in screened_labels]
                    scores = {w: math.exp(sum(math.log(r["decode_tps"]) for r in cases if r["window"] == w) /
                                          sum(r["window"] == w for r in cases))
                              for w in {r["window"] for r in cases} if w > 1}
                    best = max(scores, key=scores.get)
                    save("selected-window.json", {"window": best, "screen_scores": scores,
                                                  "criterion": "geometric mean of both input-length decode rates"})
                    requests.extend([f"final-r1:{best}:1", "control-r1:1:1",
                                     "control-r2:1:2", f"final-r2:{best}:2"])
                    print(json.dumps({"selected_window": best, "confirmation_order": requests[screen_count:]}), flush=True)
        finally:
            comparison.shutdown(process)
            log.close()
            save("cleanup.json", {"remaining_inference_processes": processes(),
                                  "swap_after": subprocess.check_output(["sysctl", "vm.swapusage"], text=True)})
            assert not processes()


if __name__ == "__main__":
    main()
