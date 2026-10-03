#!/usr/bin/env python3
"""Validate the frozen-input comparison and save the measured differences."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
MODES = {"mtp-off": 1, "mtp-t2": 2, "mtp-on": 4}
results = {name: json.loads((OUT / name / "result.json").read_text()) for name in MODES}
baseline = results["mtp-off"]
base_input = (OUT / "mtp-off" / "prompt-tokens.txt").read_bytes()
base_config = None
rows = []

for name, max_t in MODES.items():
    result = results[name]
    timing = result["timings"]
    assert (OUT / name / "prompt-tokens.txt").read_bytes() == base_input, name
    assert result["usage"]["prompt_tokens"] == 10000, name
    assert result["usage"]["completion_tokens"] == 256, name
    assert timing["cache_n"] == 0 and timing["prompt_n"] == 10000, name
    config = json.loads((OUT / name / "server-config.json").read_text())
    argv = config["args"]
    assert argv[argv.index("--mtp-max-t") + 1] == str(max_t), name
    assert argv[argv.index("--spec") + 1] == "4", name
    assert argv[argv.index("--suffix-draft") + 1] == "0", name
    config.pop("log")
    argv[argv.index("--mtp-max-t") + 1] = "MODE"
    if base_config is None:
        base_config = config
    assert config == base_config, name
    offered = timing["draft_n"]
    if max_t > 1:
        assert offered > 0, name
    else:
        assert offered == 0, name
    rate = timing["predicted_n"] * 1000 / timing["predicted_ms"]
    base_rate = baseline["timings"]["predicted_n"] * 1000 / baseline["timings"]["predicted_ms"]
    rows.append({
        "mode": name,
        "mtp_max_t": max_t,
        "prefill_tokens_per_s": timing["prompt_per_second"],
        "first_text_s": result["first_text_s"],
        "decode_tokens_per_s": rate,
        "decode_ms": timing["predicted_ms"],
        "decode_throughput_change_percent": (rate / base_rate - 1) * 100,
        "draft_accepted": timing["draft_n_accepted"],
        "draft_offered": offered,
        "acceptance_percent": timing["draft_n_accepted"] / offered * 100 if offered else None,
        "output_identical_to_baseline": result["content"] == baseline["content"],
        "needle_matches": result["needle_matches"],
        "http_wall_s": result["wall_s"],
    })

environment = json.loads((OUT / "environment.json").read_text())
binary_hash = hashlib.sha256(Path(base_config["exe"]).read_bytes()).hexdigest()
assert binary_hash == environment["binary_sha256"], "Engine changed since the benchmark started"
environment["speculative_drafts"] = "spec=4, suffix_draft=0; compare mtp_max_t=1, 2, 4"
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
summary = {
    "input_tokens_identical": True,
    "token_file_sha256": hashlib.sha256(base_input).hexdigest(),
    "configs_identical_except_mtp_max_t_and_log": True,
    "rows": rows,
    "all_outputs_identical": all(row["output_identical_to_baseline"] for row in rows),
    "quality_note": "Planted-field checks are separate from throughput and text equality; inspect each row.",
    "scope": "One frozen 10,000-token prompt; 256 output tokens; one fresh server per configuration.",
}
(OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
