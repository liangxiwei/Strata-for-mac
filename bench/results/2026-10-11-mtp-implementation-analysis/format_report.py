#!/usr/bin/env python3
"""Summarize measured phase costs and source-level implementation differences."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
summary = json.loads((OUT / "summary.json").read_text())
cleanup = json.loads((OUT / "cleanup.json").read_text())
assert not cleanup["remaining_inference_processes"]
timing_re = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, "
    r"([\d.]+) ms/window = verify ([\d.]+).*commit/emit ([\d.]+) \+ draft ([\d.]+)"
)
strata = []
for row in summary:
    if not row["mode"].startswith("strata"):
        continue
    match = timing_re.search(row["decode_round_timing"])
    assert match
    win, avg_t, out_t, round_ms, verify_ms, commit_ms, draft_ms = map(float, match.groups())
    strata.append({
        "mode": row["mode"], "spec_min_p": row["spec_min_p"], "windows": int(win),
        "average_input_window_tokens": avg_t, "output_tokens_per_window": out_t,
        "round_ms": round_ms, "verify_wall_ms": verify_ms, "commit_emit_cpu_ms": commit_ms,
        "draft_wall_ms": draft_ms, "decode_tps": row["decode_tps"],
        "verification_share": verify_ms / round_ms, "draft_share": draft_ms / round_ms,
        "draft_acceptance": row["timings"]["draft_n_accepted"] / row["timings"]["draft_n"],
        "output_equal_to_original_strata_off": row["identical_to_strata_off"],
    })
trace = (OUT / "trace-meld_trace-p0.log").read_text()
rows = [{"start_ms": float(a), "duration_ms": float(b), "context": c, "tokens": int(n)}
        for a, b, c, n in re.findall(r"dectrace ([\d.]+) ([\d.]+) ctx=(\S+) n=(\d+)", trace)]
assert rows
contexts = sorted({r["context"] for r in rows})
assert len(contexts) == 2
target = max(contexts, key=lambda c: sum(r["duration_ms"] for r in rows if r["context"] == c and r["tokens"] > 8))
draft = next(c for c in contexts if c != target)
last_prefill = max(i for i, r in enumerate(rows) if r["tokens"] > 8)
decode = rows[last_prefill + 1:]
target_rows = [r for r in decode if r["context"] == target]
draft_rows = [r for r in decode if r["context"] == draft]
target4 = [r["duration_ms"] for r in target_rows if r["tokens"] == 4]
draft1 = [r["duration_ms"] for r in draft_rows if r["tokens"] == 1]
meld_request = next(r for r in summary if r["mode"] == "meld_trace")
meld = {
    "target_context": target, "draft_context": draft,
    "target_decode_calls": len(target_rows),
    "target_tokens_per_call": dict(collections.Counter(r["tokens"] for r in target_rows)),
    "target_4_token_calls": len(target4), "target_4_token_mean_ms": statistics.mean(target4),
    "target_4_token_median_ms": statistics.median(target4),
    "target_mean_call_ms": statistics.mean(r["duration_ms"] for r in target_rows),
    "draft_1_token_calls": len(draft1), "draft_1_token_mean_ms": statistics.mean(draft1),
    "draft_1_token_median_ms": statistics.median(draft1),
    "target_total_ms": sum(r["duration_ms"] for r in target_rows),
    "draft_total_ms": sum(r["duration_ms"] for r in draft_rows),
    "draft_ms_per_target_call": sum(r["duration_ms"] for r in draft_rows) / len(target_rows),
    "output_tokens_per_target_call": meld_request["gen_tokens"] / len(target_rows),
    "decode_tps": meld_request["decode_tps"],
    "timing_caveat": "GPU-synchronized llama_decode wall time, not a kernel-only timestamp; includes catch-up calls",
}
encoding = {}
for mode in ("strata_full", "strata_subset"):
    text = (OUT / f"engine-{mode}.log").read_text()
    groups = {}
    for n, ms in re.findall(r"metal graph: entries=(\d+) encode_ms=([\d.]+)", text):
        groups.setdefault(int(n), []).append(float(ms))
    encoding[mode] = {str(n): {"samples": len(v), "median_ms": statistics.median(v), "max_ms": max(v)}
                      for n, v in groups.items()}
report = {
    "strata": strata, "meld": meld, "strata_encoding_diagnostic": encoding,
    "all_strata_output_equal_to_original_strata_off": all(r["output_equal_to_original_strata_off"] for r in strata),
    "daily_engine_sha256": hashlib.sha256((ROOT / "engine/strata").read_bytes()).hexdigest(),
    "processes_closed": True,
    "comparison_limit": "One 1139-input/512-output request per setting; diagnostic, not an interleaved three-round speed comparison",
    "verify_wall_caveat": "Strata commit is asynchronous; some preceding commit GPU work can be charged to verify wall time",
}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "MTP implementation comparison on Apple M2 Max — 2026-10-11",
    "",
    "Question: why does Meld MTP speed up the existing IQ3_S model while the current Strata Metal path does not?",
    "Same main GGUF files, same saved 1139-token input, 512 greedy output tokens, thinking/cache disabled.",
    "This diagnostic uses the installed Strata engine and its original Q2_0 MTP expert runtime.",
    "Meld uses its shared-Q8_0 MTP and the 106299-token draft subset. Low Power Mode retained.",
    "One warmup followed by one request per setting. Every previous server was stopped before the next load.",
    "Strata settings: full or subset vocabulary; max window=4; threshold .5 or 0 per request.",
    "Meld settings: max drafts=3; default threshold=0; LLAMA_DECODE_TRACE=1.",
    "",
    "Measured Strata per-round wall times:",
    "Setting | Avg input tokens | Output tokens | Verify ms | Draft ms | Round ms | Decode tok/s",
]
for row in strata:
    lines.append(f"{row['mode']}, p={row['spec_min_p']:g} | {row['average_input_window_tokens']:.2f} | "
                 f"{row['output_tokens_per_window']:.2f} | {row['verify_wall_ms']:.2f} | "
                 f"{row['draft_wall_ms']:.2f} | {row['round_ms']:.2f} | {row['decode_tps']:.2f}")
lines += [
    "",
    f"Meld: {meld['target_4_token_calls']} four-token target calls, mean {meld['target_4_token_mean_ms']:.2f} ms, "
    f"median {meld['target_4_token_median_ms']:.2f} ms.",
    f"Meld draft: one-token mean {meld['draft_1_token_mean_ms']:.2f} ms; median {meld['draft_1_token_median_ms']:.2f} ms.",
    f"All draft calls including catch-up total {meld['draft_ms_per_target_call']:.2f} ms per target call.",
    f"Output: {meld['output_tokens_per_target_call']:.2f} tokens per target call; request decode {meld['decode_tps']:.2f} tok/s.",
    "",
    "Interpretation:",
    "Cost per emitted token = (verification + drafting + state/host overhead) / emitted tokens per round.",
    "Strata verification is about 89%-92% of round wall time in these settings.",
    "At threshold 0, four-token Strata verification costs about 166-168 ms versus about 104 ms in Meld.",
    "The subset lowers Strata draft cost (16.82 -> 11.77 ms at threshold .5), but barely changes verification.",
    "Reducing the confidence threshold increases accepted output per round but slows overall Strata throughput here.",
    "A high conditional draft-acceptance percentage alone therefore does not establish a speed benefit.",
    "",
    "Source findings and implementation priorities:",
    "1. Batched Metal hyper-connection projections (highest priority for the verification path).",
    "   Strata fused_gr_read_multi launches norm/down/up separately for each token. Meld adds small-batch",
    "   float/BF16 matvec selection for K=320 and split-K for short-wide matrices such as 320x10240 and 4x10240.",
    "   Strata already batches other projections and its LM head: it is not a fully sequential verifier.",
    "   Proposed change: bind contiguous activation/output buffers and use token strides; share weight loads",
    "   across 2-4 token accumulators. Preserve each output's arithmetic order when possible.",
    "   Source: src/kernels/metal/fused_gr.mm:125; src/core/verify.cpp:632;",
    "   Meld patches/0002-metal-small-batch-matvec-for-f32-f16-bf16-rows-with-.patch and 0003-metal-split-K-...patch.",
    "   The phase timing establishes an expensive verifier; HC's individual time share still needs a kernel-level A/B.",
    "2. Restore the existing draft vocabulary subset when MTP is enabled on Metal (low effort, measured benefit).",
    "   Strata already implements it in src/core/mtp.cpp:398, but setup.py:3154 skips refresh_draft_vocab on Metal.",
    "   data/mac-metal.json deliberately disables drafts after prior M2 Max measurements, explaining that choice.",
    "   Meld's subset is credited to Strata. The current installation has no rt/draft_vocab.bin.",
    "   This experiment enabled it in a separate runtime directory, leaving original files unchanged.",
    "3. Evaluate GDN state snapshots instead of accepted-prefix recurrence replay.",
    "   Strata multi-token verification reads the old state, then commit runs the accepted-prefix recurrence again.",
    "   One-token Metal windows already have an in-place optimization, so that repeated work is specific to multi-token windows.",
    "   Meld's delta-net-base.cpp stores rollback snapshots as part of the verification recurrence.",
    "   Tradeoff: snapshot RAM and write bandwidth versus repeated computation. Benefit not measured in this diagnostic.",
    "   Source: src/core/verify.cpp:668,1317; Meld src/models/delta-net-base.cpp:566.",
    "4. Tune MTP by tokens per millisecond and add a measured fallback to ordinary decode.",
    "   Strata DraftPolicy chooses suffix lookup versus MTP; it does not generally disable a losing MTP path.",
    "   Threshold 0 from Meld was slower in Strata here. Window size and threshold need joint calibration.",
    "   Source: src/spec/draft_policy.cpp:66; src/program/generate.cpp:5369; Meld common/common.h:330.",
    "5. Evaluate Metal-native dense-quant kernels and optional Q8_0 dense re-encoding after the above.",
    "   Strata's multi-column kernels preserve CUDA arithmetic and software dp4a/byte_perm semantics.",
    "   Meld documents Q2_0-d8 four-token verification 53.9 -> 46.5 ms, with increased storage and measured KLD 0.0026.",
    "   Those are the author's Q2_0-d8 results, not our IQ3_S results: all tests here used the original IQ3_S main file.",
    "   Source: src/kernels/metal/native_mmvq.metal:94; Meld docs/PERFORMANCE.md:24 and tools/requant_dense.py.",
    "",
    "Additional observations:",
    "- All main experts are resident on the GPU in both tested paths. CPU expert fallback/PCIe misses do not explain the gap.",
    "- Strata graph replay re-encodes a tape, but sampled encoding for the 2733-entry tape was about 1.1-1.2 ms.",
    "  CPU graph encoding is much smaller than the 60+ ms target verification difference.",
    "- Strata and Meld both have autoregressive draft steps and catch-up work; Meld does not eliminate all CPU synchronization.",
    "- Strata normalizes the 10240-element MTP hidden input globally; Meld normalizes each 2560-element stream.",
    "  This semantic difference needs checking against the trained-model reference, rather than assuming either is a speed fix.",
    "- Meld's optional fused HC op is off by default, with no clear end-to-end gain in its performance notes.",
    "  The useful lesson is the measured small-batch kernel work, not adopting every fused op or debug experiment.",
    "- Strata output was exactly equal to the prior non-MTP output for all four settings here.",
    "  The earlier Meld MTP/non-MTP comparison had different greedy texts; floating-point equivalence needs validation when porting kernels.",
    "",
    "Limits:",
    "Diagnostic one-request phase measurements, not a repeated/interleaved performance benchmark.",
    "Meld trace explicitly synchronizes llama_decode. Strata verify wall time can include an asynchronous preceding commit.",
    "Strata's legacy GPU stage stamp counter on Metal is a launch counter, not nanoseconds; it was not used as a time profile.",
    "Kernel-level diagnosis should use real Metal command-buffer/counter timestamps and verify end-to-end gains afterwards.",
    "Only this M2 Max / IQ3_S / Metal setup was tested; the conclusion does not generalize to Strata's CUDA/HIP engines.",
    "No daily engine, model weight or config changes were made. All inference processes are closed.",
]
(OUT / "REPORT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:27]))
