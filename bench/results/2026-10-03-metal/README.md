# M2 Max Metal optimization measurements

Machine: M2 Max, 38 GPU cores, 96 GB unified memory, macOS 26.5. Release build, full 48-layer Q2_0
pack plus native dense projections and real PLE table. Model loading is excluded from throughput.

The current implementation and correctness repairs are documented in
[`PROGRESS.md`, round 16](../../../docs/PORT_METAL/PROGRESS.md).

| Measurement | Log | Result |
| --- | --- | --- |
| Old executable, canonical CPU-expert path, Chinese prompt, 56 outputs | `baseline-real-chat.log` | Decode 2.31 tok/s, 14-token prefill 2.28 tok/s |
| Native/resident Metal path, same prompt and output count | `fgr-fixed-prefill-chat.log` | Decode 18.72 tok/s, prefill 12.12 tok/s |
| 387-token prompt, direct Q8 expert GEMV, chunk 256 | `direct-prefill-long.log` | Prefill 48.54 tok/s |
| Same prompt, fused Q2 expert tiles, chunk 256 | `q2-tiled-prefill-long.log` | Prefill 84.52 tok/s |
| HTTP, 971-token prompt, single-token verify | `server-single-long.json` | Prefill 128.0 tok/s, decode 18.7 tok/s |
| HTTP, warm short question, 40 outputs | `server-single-repeat.json` | Decode 19.8 tok/s |

The first two rows compare different execution configurations and rounding; their decoded Chinese is
saved in the corresponding `.txt` files. The repaired speculative and single-token HTTP configurations
produced exactly the same text in all four requests. These are correctness fixtures, not a model-quality
evaluation. The first HTTP request includes cold pipeline compilation and has a longer prefill delay.

Build and tests:

```bash
cmake --build build-metal -j8
ctest --test-dir build-metal --output-on-failure
```

`ctest-final.log`: 46/46 passed. `resident-graph-parity.log` additionally checks changed expert IDs on
captured replay after destroying the source graph. `metal-gemm-final.log` checks dense GEMM and Q2 tiles
against independent references, including guards and partial tiles.

Repeat CLI prefill measurement from the repository root:

```bash
python3 bench/results/2026-10-03-metal/benchmark.py \
  --model-dir /Users/liangxw/src/xproject/petproject/Q2_0 \
  --log /tmp/strata-prefill-repeat.log \
  --prompt-file bench/results/2026-10-03-metal/prompt-long.txt \
  --new 96 --prefill 256 --expert-cache auto
```

For the HTTP sequence, start the server with `server-single.json` on port 18080, then run
`python3 bench/results/2026-10-03-metal/server_benchmark.py --prefix repeat`.
The test saves every request and response. Do not run CLI inference, GPU tests and the HTTP benchmark
concurrently. The daily-use config is `/Users/liangxw/src/xproject/petproject/Q2_0/strata-mac-optimized.json`.

Other logs here are retained diagnostics. Runs named `runtime-fixed`, `simd-*`, `shared-prefill`,
`seq-half-*` and similar predate all correctness repairs or use synchronizing dumps; do not use them
as final speed or quality evidence. The synthetic token list in the old handoff was not a meaningful
Chinese chat prompt; the new `prompt-chat.txt` uses the real tokenizer and chat delimiters.
