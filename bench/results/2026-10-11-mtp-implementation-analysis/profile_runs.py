#!/usr/bin/env python3
"""Profile existing MTP implementations and isolate vocabulary/threshold settings."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MELD = ROOT.parent / "meld-turbo"
REFERENCE = ROOT / "bench/results/2026-10-11-meld-comparison"
spec = importlib.util.spec_from_file_location("comparison", REFERENCE / "compare_api.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
measure = comparison.measure
measure.OUT = OUT
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


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def inference_processes():
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    return [r for r in rows.splitlines() if r.split()[-1].split("/")[-1] in ("strata", "llama-server", "llama-bench")]


def main():
    assert not inference_processes(), "Previous inference process is still alive"
    subset = OUT / "rt-subset"
    subset.mkdir(exist_ok=True)
    for p in (ROOT / "Strata-data/mtp/rt").iterdir():
        if p.is_file() and p.name != "draft_vocab.bin":
            dst = subset / p.name
            if not dst.exists():
                dst.symlink_to(p.resolve())
    vocab = MELD / "data/draft_vocab.bin"
    if not (subset / vocab.name).exists():
        (subset / vocab.name).symlink_to(vocab.resolve())
    save("metadata.json", {
        "purpose": "diagnostic per-round wall times, not a replacement for the three-round performance benchmarks",
        "strata_engine": str(ROOT / "engine/strata"), "strata_mtp_experts": "original Q2_0",
        "subset_tokens": vocab.stat().st_size // 4, "request_tokens": 512,
        "strata_stage_counter_profiler_disabled": "Metal gpu_stamp counts launches rather than nanoseconds",
        "meld_trace_caveat": "LLAMA_DECODE_TRACE forces synchronization; trace times are diagnostic",
        "power": subprocess.check_output(["pmset", "-g", "custom"], text=True),
    })
    summary = []
    prompt = (REFERENCE / "prompt-r0-1000.txt").read_text()
    (OUT / "prompt.txt").write_text(prompt)
    for mode in ("strata_full", "strata_subset", "meld_trace"):
        env = dict(os.environ)
        if mode.startswith("strata"):
            cfg = json.loads((ROOT / "bench/results/2026-10-11-strata-mtp/config.json").read_text())
            if mode == "strata_subset":
                cfg["args"][cfg["args"].index("--mtp") + 1] = str(subset)
            cfg["env"] = dict(cfg["env"], STRATA_DECODE_TIMING="1", STRATA_METAL_GRAPH_TIMING="1")
            cfg["log"] = str(OUT / f"engine-{mode}.log")
            save(f"config-{mode}.json", cfg)
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "strata",
                       "--config", str(OUT / f"config-{mode}.json"), "--host", "127.0.0.1", "--port", "8082"]
            cwd = ROOT
            measure.URL = "http://127.0.0.1:8082"
            points = (.5, 0.0)
        else:
            env.update(SPEC="mtp", PORT="8081", CTX="131072", LLAMA_DECODE_TRACE="1")
            command = [str(MELD / "run-iq3_s.sh")]
            cwd = MELD
            measure.URL = "http://127.0.0.1:8081"
            points = (0.0,)
        log = (OUT / f"server-{mode}.log").open("w")
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{mode} exited; inspect log")
                try:
                    health = measure.request_json("/health")
                    if health.get("status") == "ok" and health.get("loaded", True):
                        break
                except OSError:
                    pass
                if time.monotonic() - started > 900:
                    raise RuntimeError("Startup timeout")
                time.sleep(2)
            print(json.dumps({"mode": mode, "ready_s": time.monotonic() - started}), flush=True)
            tuned.extra = {}
            save(f"warmup-{mode}.json", measure.run("Write a detailed tutorial on Python generators with examples.", 64))
            for threshold in points:
                label = f"{mode}-p{threshold:g}"
                tuned.extra = {"strata_tune": {"spec_min_p": threshold}} if mode.startswith("strata") else {}
                path = OUT / (f"engine-{mode}.log" if mode.startswith("strata") else f"server-{mode}.log")
                start_offset = path.stat().st_size
                r = measure.run(prompt, 512)
                assert r["gen_tokens"] == 512 and r["timings"]["cache_n"] == 0 and not r["reasoning_content"]
                assert r["timings"]["draft_n"] > 0
                time.sleep(.1)
                with path.open("rb") as f:
                    f.seek(start_offset)
                    segment = f.read().decode()
                (OUT / f"trace-{label}.log").write_text(segment)
                if mode.startswith("strata"):
                    timing = [line for line in segment.splitlines() if line.startswith("strata decode timing:")]
                    assert len(timing) == 1, timing
                    golden = json.loads((REFERENCE / "result-r0-strata-1000.json").read_text())["content"]
                    r["identical_to_strata_off"] = r["content"] == golden
                    r["decode_round_timing"] = timing[0]
                r.update(mode=mode, spec_min_p=threshold)
                save(f"result-{label}.json", r)
                compact = {k: v for k, v in r.items() if k not in ("events", "content", "reasoning_content")}
                summary.append(compact)
                save("summary.json", summary)
                print(json.dumps(compact), flush=True)
        finally:
            comparison.shutdown(process)
            log.close()
    save("cleanup.json", {"remaining_inference_processes": inference_processes()})
    assert not inference_processes()


if __name__ == "__main__":
    main()
