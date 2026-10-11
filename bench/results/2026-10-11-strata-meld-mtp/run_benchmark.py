#!/usr/bin/env python3
"""Benchmark the experimental Strata engine with Meld's imported Q8 MTP weights."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = ROOT / "bench/results/2026-10-11-meld-comparison"
ORIGINAL_MTP = ROOT / "bench/results/2026-10-11-strata-mtp"
EXPERIMENT = Path("/tmp/strata-meld-mtp-20261011")
spec = importlib.util.spec_from_file_location("comparison", PREVIOUS / "compare_api.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
measure = comparison.measure
measure.OUT = OUT
measure.URL = "http://127.0.0.1:8082"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inference_processes():
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss=,comm="], text=True)
    return [r for r in rows.splitlines()
            if r.split()[-1].split("/")[-1] in ("strata", "llama-server", "llama-bench", "mtp_q8_fixture")]


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main():
    assert not inference_processes(), "Previous inference process is still active"
    assert not any(port_open(p) for p in (8081, 8082)), "Previous server is still active"
    fixture = (OUT / "q8-fixture.log").read_text()
    assert fixture.count("cumulative failures 0") == 2, "Q8 expert validation has not passed"
    audit = json.loads((EXPERIMENT / "rt/conversion-audit.json").read_text())
    assert audit["all_expert_blocks_byte_identical"]
    cfg = json.loads((ORIGINAL_MTP / "config.json").read_text())
    cfg["exe"] = str(EXPERIMENT / "build/strata")
    cfg["args"][cfg["args"].index("--mtp") + 1] = str(EXPERIMENT / "rt")
    cfg["log"] = str(OUT / "engine.log")
    save("config.json", cfg)
    save("machine.json", {
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True),
        "power": subprocess.check_output(["pmset", "-g", "custom"], text=True),
        "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
        "daily_engine_sha256": hashlib.sha256((ROOT / "engine/strata").read_bytes()).hexdigest(),
        "experimental_engine_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
        "mtp_source_sha256": audit["source_sha256"], "expert_bytes": audit["expert_bytes"],
        "max_context": 131072, "spec": 4, "mtp_max_t": 4, "spec_min_p": .5,
        "suffix_draft": 0, "prefill_chunk": 2048, "strata_forward_algorithm_retained": True,
        "references_measured_earlier": True, "single_server_for_three_rounds": True,
        "draft_head_vocabulary": "original Strata full vocabulary; no draft_vocab.bin in original runtime",
        "legacy_q2_expert_support_retained": True,
    })
    log = (OUT / "server.log").open("w")
    process = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"),
                                "--engine", "strata", "--config", str(OUT / "config.json"),
                                "--host", "127.0.0.1", "--port", "8082"],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    stop, samples = threading.Event(), []
    watcher = threading.Thread(target=comparison.monitor, args=(process.pid, stop, samples), daemon=True)
    watcher.start()
    results = []
    try:
        began = time.monotonic()
        while True:
            if process.poll() is not None:
                raise RuntimeError("Server exited; inspect server.log and engine.log")
            try:
                health = measure.request_json("/health")
                if health.get("status") == "ok" and health.get("loaded"):
                    break
            except OSError:
                pass
            if time.monotonic() - began > 900:
                raise RuntimeError("Startup timed out")
            time.sleep(2)
        ready_s = time.monotonic() - began
        assert "expert format native-q8_0, blob bytes 5222400" in (OUT / "engine.log").read_text(), "Q8 MTP expert path not loaded"
        print(json.dumps({"ready_s": ready_s, "mtp_expert_format": "native-q8_0"}), flush=True)
        save("health.json", health)
        save("props.json", measure.request_json("/props"))
        save("warmup.json", measure.run("Write a detailed tutorial on Python generators with examples.", 64))
        for round_id in range(3):
            for length in (1000, 4000):
                name = f"prompt-r{round_id}-{length}.txt"
                prompt = (PREVIOUS / name).read_text()
                (OUT / name).write_text(prompt)
                result = measure.run(prompt, 512)
                result.update(round=round_id, approx_prompt_tokens=length, mode="strata_meld_q8_mtp",
                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                assert result["gen_tokens"] == 512, result["gen_tokens"]
                assert result["timings"]["cache_n"] == 0, result["timings"]
                assert result["timings"].get("draft_n", 0) > 0, "No MTP drafts"
                assert not result["reasoning_content"], "Thinking was enabled"
                save(f"result-r{round_id}-{length}.json", result)
                summary = {k: v for k, v in result.items() if k not in ("events", "content", "reasoning_content")}
                results.append(summary)
                save("summary.json", results)
                print(json.dumps(summary), flush=True)
        reference = json.loads((PREVIOUS / "report.json").read_text())
        original = json.loads((ORIGINAL_MTP / "report.json").read_text())
        groups = []
        for length in (1000, 4000):
            cases = [r for r in results if r["approx_prompt_tokens"] == length]
            prior = next(g for g in reference["groups"] if g["approx_prompt_tokens"] == length)
            prior_mtp = next(g for g in original["groups"] if g["approx_prompt_tokens"] == length)
            offered = sum(r["timings"]["draft_n"] for r in cases)
            accepted = sum(r["timings"]["draft_n_accepted"] for r in cases)
            result = {
                "approx_prompt_tokens": length, "prompt_tokens": cases[0]["prompt_tokens"],
                "runs": len(cases), "gen_tokens": 512,
                "decode_tps": statistics.median(r["decode_tps"] for r in cases),
                "decode_tps_range": [min(r["decode_tps"] for r in cases), max(r["decode_tps"] for r in cases)],
                "prefill_tps": statistics.median(r["timings"]["prompt_per_second"] for r in cases),
                "ttft_s": statistics.median(r["ttft_s"] for r in cases),
                "total_s": statistics.median(r["elapsed_s"] for r in cases),
                "draft_n": offered, "draft_n_accepted": accepted, "draft_acceptance": accepted / offered,
                "previous_strata_off": prior["strata"], "previous_meld_mtp": prior["meld_mtp"],
                "previous_strata_original_mtp": prior_mtp,
            }
            result["gain_over_previous_strata_off_percent"] = (result["decode_tps"] / prior["strata"]["decode_tps"] - 1) * 100
            result["gain_over_previous_strata_original_mtp_percent"] = (result["decode_tps"] / prior_mtp["decode_tps"] - 1) * 100
            groups.append(result)
        parity = []
        for round_id in range(3):
            for length in (1000, 4000):
                a = json.loads((PREVIOUS / f"result-r{round_id}-strata-{length}.json").read_text())
                b = json.loads((OUT / f"result-r{round_id}-{length}.json").read_text())
                assert a["prompt_sha256"] == b["prompt_sha256"] and a["prompt_tokens"] == b["prompt_tokens"]
                parity.append({"round": round_id, "approx_prompt_tokens": length,
                               "identical_greedy_text": a["content"] == b["content"]})
        save("text-parity.json", parity)
        save("report.json", {"groups": groups, "ready_s": ready_s,
                             "identical_output_pairs": sum(r["identical_greedy_text"] for r in parity),
                             "output_pairs": len(parity), "reference_runs_measured_earlier": True,
                             "single_process_for_three_mtp_rounds": True})
    finally:
        comparison.shutdown(process)
        stop.set()
        watcher.join(timeout=3)
        log.close()
        save("memory.json", samples)
        remaining = inference_processes()
        ports = {str(p): port_open(p) for p in (8081, 8082)}
        save("cleanup.json", {"server_group": process.pid,
                              "remaining_group_members": [r for r in comparison.processes() if r[1] == process.pid],
                              "inference_processes": remaining, "ports_open": ports,
                              "swap_after": subprocess.check_output(["sysctl", "vm.swapusage"], text=True)})
        assert not remaining and not any(ports.values()), "Inference process remains"
        before = json.loads((OUT / "source-manifest.json").read_text())["daily_engine_sha256"]
        assert hashlib.sha256((ROOT / "engine/strata").read_bytes()).hexdigest() == before, "Daily engine was modified"


if __name__ == "__main__":
    main()
