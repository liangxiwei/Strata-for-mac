#!/usr/bin/env python3
"""Format the imported-MTP experiment with its conversion and comparison limits."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
report = json.loads((OUT / "report.json").read_text())
cleanup = json.loads((OUT / "cleanup.json").read_text())
audit = json.loads((OUT / "conversion-audit.json").read_text())
assert not cleanup["remaining_group_members"] and not cleanup["inference_processes"]
assert not any(cleanup["ports_open"].values())
lines = [
    "Strata IQ3_S with imported Meld shared-Q8_0 MTP — 2026-10-11",
    "",
    "Hardware: Apple M2 Max, 38 GPU cores, 96 GiB RAM, macOS 26.5, AC power.",
    "The existing Low Power Mode setting was retained. Other user applications remained open.",
    "Main model: the existing Strata Qwen3.8-Flash-Next GSQ-RCO IQ3_S GGUF shards.",
    "MTP source: ~/.meld-turbo/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
    f"MTP source SHA256: {audit['source_sha256']}",
    "Independent experimental engine built from a snapshot of the current working-tree sources.",
    "The daily engine, configuration and original MTP runtime weights were preserved.",
    "",
    "Compatibility and validation:",
    "- All 512 expert gate/up/down matrices retain the exact original Q8_0 blocks.",
    "- The combined embedding/hidden input projection was split at a Q8 block boundary, without requantization.",
    "- Attention and shared-expert Q8 projections retain the exact original blocks.",
    "- Norm tensors were copied as effective F32 gamma; all seven match the original Strata runtime exactly.",
    "- HC Q8 tensors were dequantized and rounded to BF16 for the existing Strata GR kernels.",
    "- Router and shared gate F32-to-BF16 conversion is exact for these tensors.",
    "- Expert byte validation and a CPU reference check of three real experts passed.",
    "- Captured GPU graph replay with changed routing passed the same expert reference check.",
    "- The engine log confirms native-q8_0 expert loading; each measured request generated real MTP drafts.",
    f"- Expert storage: {audit['expert_bytes'] / 2**30:.3f} GiB; dense storage: {audit['dense_bytes'] / 2**20:.2f} MiB.",
    "",
    "Measurement:",
    "Context capacity 131072; prefill chunk 2048; max speculation window 4 (up to 3 drafts).",
    "spec-min-p=0.5; suffix-draft=0; original Strata full draft vocabulary retained.",
    "Temperature 0, thinking disabled, prompt caching disabled, exactly 512 output tokens per request.",
    "A 64-token warmup was excluded. One server handled all three short/long prompt rounds.",
    "Prompts exactly match the earlier saved comparison inputs.",
    "Decode rate: (completion_tokens - 1) / (last-content time - first-content time).",
    "Results are medians of three runs. Reference values below were measured earlier, not interleaved.",
    "",
    "Input tokens | Strata off | Strata original MTP | Strata Meld-Q8 MTP | Meld MTP",
]
for g in report["groups"]:
    lines.append(f"{g['prompt_tokens']:>12} | {g['previous_strata_off']['decode_tps']:>10.2f} | "
                 f"{g['previous_strata_original_mtp']['decode_tps']:>19.2f} | "
                 f"{g['decode_tps']:>18.2f} | {g['previous_meld_mtp']['decode_tps']:>8.2f}")
    lines += [
        f"  New decode range: {g['decode_tps_range'][0]:.2f}-{g['decode_tps_range'][1]:.2f} tokens/s.",
        f"  Change from original Strata MTP: {g['gain_over_previous_strata_original_mtp_percent']:+.1f}%; "
        f"from Strata off: {g['gain_over_previous_strata_off_percent']:+.1f}%.",
        f"  Draft acceptance: {g['draft_acceptance']:.1%} ({g['draft_n_accepted']}/{g['draft_n']}); "
        f"original Strata MTP: {g['previous_strata_original_mtp']['draft_acceptance']:.1%}.",
        f"  Prefill: {g['prefill_tps']:.1f} tokens/s; TTFT: {g['ttft_s']:.2f} s; total: {g['total_s']:.2f} s.",
    ]
lines += [
    "",
    f"Exact final output matches against the earlier Strata-off runs: {report['identical_output_pairs']}/{report['output_pairs']}.",
    "Final-output parity validates these greedy cases; it is not a task-quality evaluation.",
    "",
    "Interpretation limits:",
    "This tests imported Meld weights in Strata's MTP implementation, rather than a port of Meld's graph.",
    "Strata's global hidden-input RMS, dense draft attention, projection split, full draft vocabulary and",
    "confidence stopping rule are retained. Meld uses per-stream hidden RMS and a reduced draft vocabulary.",
    "HC BF16 rounding has relative L2 error of approximately 0.12%-0.15%; individual tensors are audited.",
    "Only 1139/4289 input tokens were tested. 128K context capacity was not filled.",
    "Process RSS is not comparable total physical/GPU memory between the two engines.",
    "",
    f"Repeated-load server readiness: {report['ready_s']:.2f} s; not a cold-boot measurement.",
    "All test processes stopped at completion. Ports 8081 and 8082 are closed.",
    "Source manifest, experimental patch, importer, kernel fixture, config, conversion audit, engine/server",
    "logs, machine metadata, exact prompts, raw SSE requests/results and cleanup check are saved here.",
    "",
    "Experimental workspace: /tmp/strata-meld-mtp-20261011/{source,build,rt}.",
    "The workspace is temporary; retain the audit/patch/importer to reproduce the experiment after cleanup.",
]
(OUT / "REPORT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
