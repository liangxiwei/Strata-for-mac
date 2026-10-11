#!/usr/bin/env python3
"""Import Meld's shared Q8 MTP into an experimental Strata runtime directory.

Experts and large projections retain the exact GGUF Q8_0 block bytes. HC and
router tensors use the existing Strata BF16 interface, with rounding audited.
GGUF norm weights already include the Gemma +1 and are copied without adding it.
The Strata forward algorithm and speculation settings are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf

H, FF, NE = 2560, 640, 512
PREFIX = "blk.48."
Q8_ROW = H // 32 * 34
Q8_DOWN_ROW = FF // 32 * 34
BLOB = 3 * H * FF // 32 * 34


def bf16(x):
    bits = np.asarray(x, dtype="<f4").view("<u4")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    source = Path(args.gguf).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(str(source))
    tensors = {t.name[len(PREFIX):]: t for t in r.tensors if t.name.startswith(PREFIX)}
    assert len(tensors) == len(r.tensors), "Unexpected block or shared tensor in MTP GGUF"
    audit = {"source": str(source), "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
             "expert_format": "q8_0", "expert_bytes": NE * BLOB,
             "strata_forward_algorithm_retained": True, "norms_already_effective_gamma": True,
             "tensors": [], "unused_tensors": []}
    consumed = set()
    gate, up, down = [tensors[name] for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")]
    for t, shape in ((gate, (NE, FF, Q8_ROW)), (up, (NE, FF, Q8_ROW)), (down, (NE, H, Q8_DOWN_ROW))):
        assert int(t.tensor_type) == 8 and t.data.shape == shape, (t.name, t.data.shape)
        consumed.add(t.name[len(PREFIX):])
    with (out / "experts.bin").open("wb") as f:
        for e in range(NE):
            for t in (gate, up, down):
                f.write(t.data[e].tobytes())
    assert (out / "experts.bin").stat().st_size == NE * BLOB
    # Read the complete converted file back to validate every expert block byte.
    with (out / "experts.bin").open("rb") as f:
        for e in range(NE):
            for t in (gate, up, down):
                raw = t.data[e].tobytes()
                assert f.read(len(raw)) == raw, (e, t.name)
        assert not f.read(1)
    audit["all_expert_blocks_byte_identical"] = True
    (out / "experts.txt").write_text(f"q8_0 {NE} {H} {FF}\n")
    mapping = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
        "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
        "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
        "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
        "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
        "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
        "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
        "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
        "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
        "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
        "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
        "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
        "nextn.hc_head_norm.weight": "hyper_connection_mixer.hc_norm.weight",
        "nextn.hc_head_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
        "nextn.hc_head_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
    }
    original_layout = {}
    for line in (ROOT / "Strata-data/mtp/rt/dense.txt").read_text().splitlines():
        name, kind, rows, cols, off, size = line.split()
        original_layout[name] = (kind, int(rows), int(cols), int(off), int(size))
    lines, raw_by_name = [], {}
    with (out / "dense.bin").open("wb") as f:
        def write(name, kind, rows, cols, raw, source_name, error=None):
            want_kind, want_rows, want_cols, _, _ = original_layout[name]
            assert (kind, rows, cols) == (want_kind, want_rows, want_cols), (name, rows, cols)
            off = f.tell()
            f.write(b"\0" * (-off % 256))
            off = f.tell()
            f.write(raw)
            lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
            raw_by_name[name] = raw
            audit["tensors"].append({"name": name, "source_name": source_name, "kind": kind,
                                     "rows": rows, "cols": cols, "sha256": hashlib.sha256(raw).hexdigest(),
                                     "bf16_rounding": error})

        fc = tensors["nextn.eh_proj.weight"]
        assert int(fc.tensor_type) == 8 and fc.data.shape == (H, 2 * Q8_ROW)
        for i, name in enumerate(("fc_embedding.weight", "fc_hidden.weight")):
            write(name, "q8_0", H, H, np.ascontiguousarray(fc.data[:, i * Q8_ROW:(i + 1) * Q8_ROW]).tobytes(), fc.name)
        consumed.add("nextn.eh_proj.weight")
        for src, name in mapping.items():
            t = tensors[src]
            kind, rows, cols, _, _ = original_layout[name]
            assert tuple(reversed(tuple(map(int, t.shape)))) in ((cols,), (rows, cols)), (src, t.shape, rows, cols)
            error = None
            if kind == "q8_0":
                assert int(t.tensor_type) == 8
                raw = t.data.tobytes()
            elif kind == "f32":
                assert int(t.tensor_type) == 0
                raw = np.asarray(t.data, dtype="<f4").tobytes()
            else:
                assert kind == "bf16"
                x = gguf.quants.dequantize(t.data, t.tensor_type)
                assert x.size == rows * cols and np.isfinite(x).all()
                values = bf16(x)
                restored = (values.astype("<u4") << 16).view("<f4")
                diff = restored - x
                error = {"source_type": int(t.tensor_type), "max_abs": float(np.max(np.abs(diff))),
                         "relative_l2": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(x.ravel()), 1e-30)),
                         "exact_values": int(np.count_nonzero(diff == 0)), "total_values": int(diff.size)}
                raw = values.tobytes()
            write(name, kind, rows, cols, raw, t.name, error)
            consumed.add(src)
    (out / "dense.txt").write_text("\n".join(lines) + "\n")
    with (out / "dense.bin").open("rb") as f:
        for line in lines:
            name, kind, rows, cols, off, size = line.split()
            f.seek(int(off)); assert f.read(int(size)) == raw_by_name[name]
    with (ROOT / "Strata-data/mtp/rt/dense.bin").open("rb") as f:
        for row in audit["tensors"]:
            name = row["name"]
            kind, _, _, off, size = original_layout[name]
            if kind == "f32":
                f.seek(off)
                row["identical_to_original_strata_norm"] = f.read(size) == raw_by_name[name]
    audit["unused_tensors"] = sorted(set(tensors) - consumed)
    assert all(name.startswith("indexer.") for name in audit["unused_tensors"]), audit["unused_tensors"]
    audit["unused_reason"] = "Strata's MTP uses dense attention and never reads indexer tensors."
    # Retain Strata's existing subset so only the auxiliary model is changed.
    vocab = ROOT / "Strata-data/mtp/rt/draft_vocab.bin"
    if vocab.exists():
        shutil.copy2(vocab, out / "draft_vocab.bin")
        audit["draft_vocab_source"] = str(vocab)
        audit["draft_vocab_size"] = vocab.stat().st_size // 4
        audit["draft_vocab_sha256"] = hashlib.sha256(vocab.read_bytes()).hexdigest()
    audit["dense_bytes"] = (out / "dense.bin").stat().st_size
    audit["experts_sha256"] = hashlib.file_digest((out / "experts.bin").open("rb"), "sha256").hexdigest()
    (out / "conversion-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"out": str(out), "source_sha256": audit["source_sha256"],
                      "experts_bytes": audit["expert_bytes"], "dense_bytes": audit["dense_bytes"],
                      "all_expert_blocks_byte_identical": True, "tensors": len(lines)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
