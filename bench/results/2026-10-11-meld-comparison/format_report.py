#!/usr/bin/env python3
import json
from pathlib import Path
import statistics

out = Path(__file__).resolve().parent
report = json.loads((out / "report.json").read_text())
rows = json.loads((out / "summary.json").read_text())
starts = []
for line in (out / "benchmark.log").read_text().splitlines():
    try:
        item = json.loads(line)
    except ValueError:
        continue
    if isinstance(item, dict) and "ready_s" in item:
        starts.append(item)
report["startup_median_s"] = {mode: statistics.median(r["ready_s"] for r in starts if r["label"].endswith("-" + mode)) for mode in ("strata", "meld_mtp")}
report["rss_caveat"] = "Process RSS is not a comparable total physical-memory/GPU-footprint metric across engines."
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
lines = ["Strata / Meld Turbo IQ3_S comparison, 2026-10-11", "",
         "Apple M2 Max, 38-core GPU, 96 GiB RAM, AC power, Low Power Mode kept enabled.",
         "Existing IQ3_S GGUF model weights were reused without requantization.",
         "Strata: installed engine/strata and current strata-iq3_s.json settings,",
         "except context capacity set to 131072 and prompt cache disabled in temporary configs.",
         "Strata uses prefill chunk 2048, mtp-max-t 1, suffix-draft 0 (no speculative drafts).",
         "Meld: patched llama.cpp, MTP shared-Q8_0 head, 106K draft vocabulary, 3 drafts maximum.",
         "Meld also uses 131072 context capacity and one slot.",
         "Both: identical source-code chat prompts, temperature 0, thinking off, 512 output tokens.",
         "Three alternating pairs; order Strata/Meld, Meld/Strata, Strata/Meld.",
         "Only one engine was loaded at a time. A 64-token warmup followed each model load.",
         "All measured prompts were uncached. Each decode run lasted over 10 seconds.",
         "Startup and warmup were excluded. Background user apps remained open.",
         "Decode uses client time from first to last streamed content; prefill uses server timing.",
         "All values below are medians across three runs.", ""]
for group in report["groups"]:
    lines.append(f"Input: {group['strata']['prompt_tokens'][0]} tokens; output: 512 tokens")
    for mode in ("strata", "meld_mtp"):
        r = group[mode]
        lines.append(f"  {mode:8s}: decode {r['decode_tps']:.2f} tok/s; prefill {r['prefill_tps']:.1f} tok/s; TTFT {r['ttft_s']:.2f} s; total {r['total_s']:.2f} s")
    lines.append(f"  Meld vs Strata: decode {group['meld_decode_gain_percent']:+.1f}%; prefill {group['meld_prefill_gain_percent']:+.1f}%")
    saved = (1 - group["meld_mtp"]["total_s"] / group["strata"]["total_s"]) * 100
    lines.append(f"  Total request wall time reduction with Meld: {saved:.1f}%")
    lines.append("")
parity = []
for round_id in range(3):
    for length in (1000, 4000):
        a = json.loads((out / f"result-r{round_id}-strata-{length}.json").read_text())
        b = json.loads((out / f"result-r{round_id}-meld_mtp-{length}.json").read_text())
        assert a["prompt_sha256"] == b["prompt_sha256"]
        assert a["prompt_tokens"] == b["prompt_tokens"]
        parity.append({"round": round_id, "approx_prompt_tokens": length,
                       "identical_greedy_text": a["content"] == b["content"]})
(out / "text-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
lines += [f"Exact paired output text match: {sum(r['identical_greedy_text'] for r in parity)}/{len(parity)}.",
          "This is a throughput test; no task-quality assessment was performed.",
          "Actual input lengths tested were 1139 and 4289 tokens; 128K capacity was not filled.",
          "The engines differ in implementation and draft settings; this does not isolate the engine from MTP.",
          "RSS samples are preserved but are not a comparable total physical-memory/GPU-footprint metric.",
          "All test engines were stopped after the comparison; no model server was left resident.", "",
          f"Repeated model load to healthy API, median: Strata {report['startup_median_s']['strata']:.2f} s; Meld {report['startup_median_s']['meld_mtp']:.2f} s.",
          "These are repeated loads in the current cache/background-load conditions, not cold-boot measurements.", "",
          "Reproduce from the Strata checkout with ports 8081 and 8082 free:",
          "  python3 bench/results/2026-10-11-meld-comparison/compare_api.py",
          "  python3 bench/results/2026-10-11-meld-comparison/format_report.py", ""]
(out / "REPORT.txt").write_text("\n".join(lines))
print("\n".join(lines))
