#!/usr/bin/env python3
"""Reproduce the Metal measurements; unknown arguments are passed to strata."""
import argparse
import pathlib
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--model-dir", type=pathlib.Path, required=True)
p.add_argument("--exe", type=pathlib.Path, default=pathlib.Path("build-metal/strata"))
p.add_argument("--log", type=pathlib.Path, required=True)
p.add_argument("--canonical", action="store_true")
p.add_argument("--prompt-tokens", type=int, default=24)
p.add_argument("--new", type=int, default=16)
p.add_argument("--prompt-file", type=pathlib.Path)
a, extra = p.parse_known_args()
prompt = [9707,11,6319,1440,11,2834,10265,11,19147,1079,3844,11,32114,12488,
          131464,144162,1773,11,1871,1879,3300,51016,45589,13]
tokens = (prompt * ((a.prompt_tokens + 23) // 24))[:a.prompt_tokens]
shard1 = next(a.model_dir.glob("*-00001-of-00002.gguf"))
shard2 = next(a.model_dir.glob("*-00002-of-00002.gguf"))
cmd = [str(a.exe.resolve()), "--pack", str(a.model_dir / "pack-full"),
       "--ple-gguf", str(shard2), "--max-context", "4096", "--max-new", str(a.new),
       "--stats", "--check-logits"]
cmd += ["--tokens-file", str(a.prompt_file)] if a.prompt_file else ["--tokens", ",".join(map(str, tokens))]
if not a.canonical:
    cmd += ["--native", str(shard1)]
cmd += extra
a.log.parent.mkdir(parents=True, exist_ok=True)
with a.log.open("w") as f:
    f.write("argv: " + repr(cmd) + "\n")
    f.flush()
    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
print(a.log.read_text())
raise SystemExit(result.returncode)
