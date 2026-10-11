#!/usr/bin/env python3
"""Compare the installed Strata engine and Meld MTP with identical API prompts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time
import urllib.error

OUT = Path(__file__).resolve().parent
STRATA = OUT.parents[2]
MELD = STRATA.parent / "meld-turbo"
PREVIOUS = MELD / "bench/results/2026-10-11-iq3-s"
spec = importlib.util.spec_from_file_location("measurement", PREVIOUS / "run_benchmark.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)
measure.OUT = OUT


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def processes():
    text = subprocess.check_output(["ps", "-axo", "pid=,pgid=,rss="], text=True)
    return [list(map(int, line.split())) for line in text.splitlines() if line.strip()]


def monitor(group, stop, samples):
    while not stop.is_set():
        members = [r for r in processes() if r[1] == group]
        if members:
            samples.append({"time": time.time(), "rss_kib": sum(r[2] for r in members), "processes": members})
        stop.wait(1)


def shutdown(process):
    if any(r[1] == process.pid for r in processes()):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any(r[1] == process.pid for r in processes()):
        process.poll()
        if time.monotonic() >= deadline:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            if any(r[1] == process.pid for r in processes()):
                raise RuntimeError("Previous engine still alive; refusing to load another")
            break
        time.sleep(.2)
    process.wait(timeout=10)


def main():
    cfg = json.loads((STRATA / "strata-iq3_s.json").read_text())
    cfg["exe"] = str(STRATA / cfg["exe"])
    cfg["cwd"] = str(STRATA)
    cfg["tokenizer"] = str(STRATA / cfg["tokenizer"])
    cfg["port"] = 8082
    cfg["args"][cfg["args"].index("--max-context") + 1] = "131072"
    cfg["args"] += ["--prompt-cache", "0"]
    save("machine.json", {"hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True),
                         "power": subprocess.check_output(["pmset", "-g", "custom"], text=True),
                         "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
                         "strata_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=STRATA, text=True).strip(),
                         "strata_engine_sha256": hashlib.sha256((STRATA / "engine/strata").read_bytes()).hexdigest(),
                         "meld_engine_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.home() / ".meld-turbo/llama.cpp", text=True).strip(),
                         "context_capacity": 131072, "output_token_cap": 512,
                         "order": [["strata", "meld_mtp"], ["meld_mtp", "strata"], ["strata", "meld_mtp"]]})
    results = []
    for round_id in range(3):
        order = ("meld_mtp", "strata") if round_id == 1 else ("strata", "meld_mtp")
        for mode in order:
            label = f"r{round_id}-{mode}"
            stop, samples = threading.Event(), []
            env = dict(os.environ)
            if mode == "strata":
                current = dict(cfg, log=str(OUT / f"engine-{label}.log"))
                save(f"config-{label}.json", current)
                command = [str(STRATA / ".venv/bin/python"), str(STRATA / "serve/server.py"),
                           "--engine", "strata", "--config", str(OUT / f"config-{label}.json"),
                           "--host", "127.0.0.1", "--port", "8082"]
                measure.URL = "http://127.0.0.1:8082"
                cwd = STRATA
            else:
                env.update(SPEC="mtp", PORT="8081", CTX="131072")
                command, cwd = [str(MELD / "run-iq3_s.sh")], MELD
                measure.URL = "http://127.0.0.1:8081"
            log = (OUT / f"server-{label}.log").open("w")
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            watcher = threading.Thread(target=monitor, args=(process.pid, stop, samples), daemon=True)
            watcher.start()
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
                        raise RuntimeError(f"{label} startup timed out")
                    time.sleep(2)
                print(json.dumps({"label": label, "ready_s": time.monotonic() - started}), flush=True)
                save(f"health-{label}.json", health)
                save(f"props-{label}.json", measure.request_json("/props"))
                save(f"warmup-{label}.json", measure.run("Write a detailed tutorial on Python generators with examples.", 64))
                for length in (1000, 4000):
                    name = f"prompt-r{round_id}-{length}.txt"
                    prompt = (PREVIOUS / name).read_text()
                    (OUT / name).write_text(prompt)
                    result = measure.run(prompt, 512)
                    result.update(label=label, mode=mode, round=round_id, approx_prompt_tokens=length,
                                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                    if result["timings"]["cache_n"] != 0:
                        raise RuntimeError(f"{label} reused prompt cache")
                    save(f"result-{label}-{length}.json", result)
                    summary = {k: v for k, v in result.items() if k not in ("events", "content", "reasoning_content")}
                    results.append(summary)
                    save("summary.json", results)
                    print(json.dumps(summary), flush=True)
                save(f"swap-after-{label}.json", {"swap": subprocess.check_output(["sysctl", "vm.swapusage"], text=True)})
            finally:
                shutdown(process)
                stop.set()
                watcher.join(timeout=3)
                save(f"memory-{label}.json", samples)
                log.close()
    groups = []
    for length in (1000, 4000):
        group = {"approx_prompt_tokens": length}
        for mode in ("strata", "meld_mtp"):
            cases = [r for r in results if r["mode"] == mode and r["approx_prompt_tokens"] == length]
            group[mode] = {"runs": len(cases), "prompt_tokens": [r["prompt_tokens"] for r in cases],
                           "gen_tokens": [r["gen_tokens"] for r in cases],
                           "decode_tps": statistics.median(r["decode_tps"] for r in cases),
                           "prefill_tps": statistics.median(r["timings"]["prompt_per_second"] for r in cases),
                           "ttft_s": statistics.median(r["ttft_s"] for r in cases),
                           "total_s": statistics.median(r["elapsed_s"] for r in cases)}
        group["meld_decode_gain_percent"] = (group["meld_mtp"]["decode_tps"] / group["strata"]["decode_tps"] - 1) * 100
        group["meld_prefill_gain_percent"] = (group["meld_mtp"]["prefill_tps"] / group["strata"]["prefill_tps"] - 1) * 100
        groups.append(group)
    memory = {}
    for mode in ("strata", "meld_mtp"):
        memory[mode] = max(s["rss_kib"] for f in OUT.glob(f"memory-r*-{mode}.json") for s in json.loads(f.read_text())) / 1048576
    report = {"groups": groups, "peak_combined_process_rss_gib": memory,
              "all_prompts_uncached": all(r["timings"]["cache_n"] == 0 for r in results),
              "all_thinking_disabled": all(not json.loads(f.read_text())["reasoning_content"] for f in OUT.glob("result-*.json")),
              "engines_stopped_after_test": True}
    save("report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
