"""Compare one-request IQ3_S audits, including full logits and active state."""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.conversation_cache_parity import state_hashes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def load(folder):
    data = json.loads((folder / "results.json").read_text())
    records = data.get("records") or [data["result"]]
    assert len(records) == 1, "Use one request per numerical audit"
    states = data.get("state_hashes") or state_hashes((folder / "engine.log").read_text())
    return data, records[0], states


def strip_chunk(engine_args):
    result = list(engine_args)
    index = result.index("--prefill")
    del result[index:index + 2]
    return result


def traces(folder):
    return sorted(folder.glob("logits-*.bin"), key=lambda p: int(p.stem.split("-")[-1]))


def read_trace(path):
    with path.open("rb") as stream:
        header = struct.unpack("<qqq", stream.read(24))
        position, count, vocab = header
        assert position >= 0 and count > 0 and vocab > 0
        inputs = np.fromfile(stream, dtype="<i4", count=count)
        outputs = np.fromfile(stream, dtype="<i4", count=count)
        logits = np.fromfile(stream, dtype="<f4")
    assert inputs.size == outputs.size == count and logits.size == count * vocab
    return header, inputs, outputs, logits


reference, left, left_states = load(args.reference)
candidate, right, right_states = load(args.candidate)
settings = ("kv", "kv_resident", "context", "spec", "mtp_max", "lookup", "cvec", "expert_slots")
left_files, right_files = traces(args.reference), traces(args.candidate)
summary = {
    "same_request": (left.get("prompt_sha256") or reference["config"]["prompt_sha256"])
                    == (right.get("prompt_sha256") or candidate["config"]["prompt_sha256"]),
    "same_binary": reference["config"]["exe_sha256"] == candidate["config"]["exe_sha256"],
    "same_engine_args_except_chunk": strip_chunk(reference["config"]["args"])
                                    == strip_chunk(candidate["config"]["args"]),
    "same_model_settings": all(reference["info"][key] == candidate["info"][key] for key in settings),
    "same_output_ids": left["output_ids"] == right["output_ids"],
    "same_active_states": len(left_states) == len(right_states) == 1 and left_states == right_states,
    "fresh_prompts_no_drafts": all(r["reused"] == r["drafts_offered"] == r["drafts_accepted"] == 0
                                  for r in (left, right)),
    "trace_files": [len(left_files), len(right_files)],
    "rows": 0,
    "values": 0,
    "different": 0,
    "same_trace_inputs": True,
    "finite": True,
}
for lp, rp in zip(left_files, right_files):
    lh, li, lo, lv = read_trace(lp)
    rh, ri, ro, rv = read_trace(rp)
    matching = lh == rh and np.array_equal(li, ri) and np.array_equal(lo, ro)
    summary["same_trace_inputs"] &= matching
    if matching:
        summary["rows"] += lh[1]
        summary["values"] += lv.size
        summary["different"] += int(np.count_nonzero(lv.view("u4") != rv.view("u4")))
        summary["finite"] &= bool(np.isfinite(lv).all() and np.isfinite(rv).all())
checks = ("same_request", "same_binary", "same_engine_args_except_chunk", "same_model_settings",
          "same_output_ids", "same_active_states", "fresh_prompts_no_drafts", "same_trace_inputs", "finite")
summary["bitwise_pass"] = bool(
    all(summary[key] for key in checks) and len(left_files) == len(right_files) > 0
    and summary["rows"] == 32 and summary["values"] > 0 and summary["different"] == 0
)
args.output.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["bitwise_pass"] else 1)
