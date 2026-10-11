#!/usr/bin/env python3
import json
from pathlib import Path

out = Path(__file__).resolve().parent
report = json.loads((out / "report.json").read_text())
lines = ["Strata IQ3_S with MTP enabled, 2026-10-11", "",
         "Apple M2 Max, 38-core GPU, 96 GiB RAM, AC power, Low Power Mode kept enabled.",
         "Same installed Strata engine binary and main IQ3_S GGUF files as the previous comparison.",
         "Context capacity 131072, prefill chunk 2048, temperature 0, thinking off, prompt cache disabled.",
         "MTP window cap changed from 1 to 4: up to 3 speculative draft tokens plus the current token.",
         "spec-min-p remains 0.5; suffix-draft remains 0. Existing Strata MTP runtime files reused.",
         "MTP model quantization differs from Meld's shared-Q8_0 head; this is an installed-config comparison.",
         "A 64-token warmup was excluded. The same process served 3 rounds of identical saved ~1K/~4K prompts.",
         "Each measured request generated 512 tokens, with no prefix reuse and with real MTP drafts.",
         "Baseline Strata-off and Meld-MTP values below are medians from the immediately preceding test.",
         "They were not rerun interleaved with this MTP-on test. Other user apps remained open.",
         "Decode uses client first-to-last streamed content; prefill uses engine timing. Values are medians.", ""]
for group in report["groups"]:
    lines.append(f"Input: {group['prompt_tokens']} tokens; output: {group['gen_tokens']} tokens")
    for label, r in (("Strata off (previous)", group["previous_strata_off"]),
                     ("Strata MTP (new)", group),
                     ("Meld MTP (previous)", group["previous_meld_mtp"])):
        lines.append(f"  {label:21s}: decode {r['decode_tps']:.2f} tok/s; prefill {r['prefill_tps']:.1f} tok/s; TTFT {r['ttft_s']:.2f} s; total {r['total_s']:.2f} s")
    lines.append(f"  Strata MTP vs previous Strata off: {group['gain_over_previous_strata_off_percent']:+.1f}% decode")
    lines.append(f"  Draft acceptance: {group['draft_acceptance']:.1%} ({group['draft_n_accepted']}/{group['draft_n']})")
    lines.append(f"  Strata MTP decode range: {group['decode_tps_range'][0]:.2f}-{group['decode_tps_range'][1]:.2f} tok/s")
    lines.append("")
lines += [f"Exact output text match with previous Strata-off: {report['identical_output_pairs']}/{report['output_pairs']} pairs.",
          "Text parity is not a quality evaluation. No task-accuracy assessment was performed.",
          "Only 1139/4289 input token lengths were tested; 128K capacity was not filled.",
          "MTP window sizes 2 and 3, other thresholds, and other draft weights were not tested here.",
          "The daily strata-iq3_s.json config was not changed. Test processes were closed at completion.", "",
          "Reproduce from the Strata checkout with port 8082 free:",
          "  python3 bench/results/2026-10-11-strata-mtp/run_benchmark.py",
          "  python3 bench/results/2026-10-11-strata-mtp/format_report.py", ""]
(out / "REPORT.txt").write_text("\n".join(lines))
print("\n".join(lines))
