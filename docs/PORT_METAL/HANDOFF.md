# HANDOFF - the Strata Metal port, 2026-10-03 (updated after round 26)

For the agent continuing this work. Read AGENTS.md first (repo rules), then this file, then
docs/PORT_METAL/PROGRESS.md rounds 9-26 (the platform rules with their measurements) and STATUS.md
(the per-file state). PLAN.md is the original plan. Everything below is measured, not assumed.

## Quality constraint

The user explicitly requires speed improvements without reducing model capability. Preserve the current
IQ2_XS weights/quantization, full layers and routed expert count, FP16 KV and requested sampling. Retain
independent kernel references and compare real-model logits on identical input tokens plus committed
state; retrieval scores or plausible wording alone are insufficient. A candidate with unexplained
numerical deviations must stay out of the default path until investigated. Current round-21 audits
pass for the tested short, 10K and cached-follow-up inputs; 100K still needs its own validation.

## Where things stand

The CUDA/HIP engine is ported to Metal (Apple M2 Max, macOS 26.5) end to end:

* **Round 26 (latest): GDN commit without the second recurrence.** A one-token decode window writes its GDN state
  back (n_keep = 1) and the commit skips the recurrence: commit GPU 2.43 -> 0.71 ms/token, warm decode median
  21.78 -> 22.55 tok/s, bitwise (`STRATA_METAL_GDN_INPLACE=0` restores). Binary `build-metal/strata-round26`.
* **Round 25: prefill.** First-request prefill of the frozen 10K prompt 80.6-81.6 s -> 51.1-51.9 s
  (about 123 -> 194 tok/s, same binary), bitwise: 64 x 64 GEMM tiles, register prompt attention, wider expert
  tiles. Switches `STRATA_METAL_PREFILL_GEMM2`, `STRATA_METAL_PROMPT_ATTN_REG`, `STRATA_METAL_PREFILL_MOE2`
  (`=0` restores). Prefill time now: expert tiles 39% (IQ2 decode-bound), FP16 GEMM 22%, attention 14%, BF16
  GEMM 11%. Round 24 (expert gate/up in decode): nothing kept - see PROGRESS.
* **Rounds 22-23: exact direct decode kernels.** Warm decode on the frozen 10K / 256 test, same
  binary per round: 18.83-18.90 -> 21.78-21.99 tok/s (round 22) -> 22.23-22.46 tok/s (round 23), about
  44.7 ms/token, with **no IQ4 view** (2.24 GiB less; the daily config now sets `STRATA_METAL_IQ4_EXPAND=0`
  because the view is slower than the direct kernels). Every output id, all 96,596,480 audited logits and all
  six committed states stay bitwise equal. Evidence: `bench/results/2026-10-03-metal-decode-opt2/` and `-opt3/`;
  the switches are in the next section and PROGRESS rounds 22-23. Rounds 21's notes below are history.
* **The current model is IQ2_XS**, with both GGUF shards SHA-256 verified. Round 21's matched-binary
  10K / 256-output test, MTP off, improves warm decode **18.69 -> 19.53 tok/s (+4.5%)** with only
  **2.244 GiB** additional cache. Three repeated requests recompute all output tokens; prompt reuse
  is identical on both sides. First-request decode varies (17.69-19.52 tok/s), so cold-request speedup
  is not established. Prefill is around 127-130 tok/s in these runs, with no claimed stable gain.
  The final cleaned build, tested through the daily config, repeats at **19.65-19.69 tok/s** warm
  (median 19.67), with first-request decode 19.52. All private test engines are stopped.
  Evidence: `bench/results/2026-10-03-metal-decode-opt/`. Round 20's earlier HTTP test measured
  120.3 / 18.0 tok/s, versus round 19's 44.7 / 7.6. These are workload measurements, not task accuracy.
* **Cache only repeated, measured work.** The 179 dense IQ4 matrices keep a lossless 264-byte view
  of each original 136-byte block: unchanged 8-byte header plus signed int8 codebook values. Integer
  dot, scales and float accumulation use the original shader's operations. Views take about 0.12 s
  to build and speed up decode about 4.4% in an isolated same-binary cache on/off comparison. The
  625 MiB output-head view produced no clear overall gain and was removed, along with larger FP16
  views, GR rewrites and a gate fusion that only moved the full-model result by about 0.1%.
  Short and 2D copies also avoid compute/blit pass switches. No eager graph commit is introduced.
* **2026-10-04: pictures, Q2_0, 128K, the model list.**
  - **Pictures.** `strata-vision` (llama.cpp mtmd, `-DGGML_METAL=ON`) matches the CPU encoder: median row cosine
    0.99998, 2 of 2,178 rows under 0.9. It takes 0.4-1.4 s a picture on Metal; the CPU takes 1-4.5 min. Through the
    server, all 7 picture/text checks pass. With `--vision`, text answers are the same as without, at the same speed.
    Evidence: `bench/results/2026-10-04-metal-vision/`.
  - **Q2_0** (native pack): 22.5-22.8 tok/s writing, 205 reading, 31.64 GiB of experts
    (`bench/results/2026-10-04-metal-models/`).
  - **128K** (IQ2_XS, corrected engine): 141 tok/s reading, 24.5 writing, the planted fields 3/3, follow-up 0.95 s
    (`bench/results/2026-10-04-metal-128k/`).
  - **setup.** It shows the model list every run (arrow keys; data/mac-metal.json's `menu`), puts images on by
    default, downloads a big file over four connections and checks each file's published SHA-256.
* **Installed by setup now (2026-10-03, after round 26):** `./download-model.sh` + `./setup.sh` compile the Metal
  engine into `engine/strata` and write `strata-iq2_xs.json` from `data/mac-metal.json` (the settings below, paths
  relative to the repo); the model files live in the repo's git-ignored `Strata-data/` (models/IQ2_XS, packs/iq2_xs,
  mtp/rt - moved there from the old IQ2_XS/ folder). Same output text and speed as the round-26 binary:
  `bench/results/2026-10-03-metal-setup/`. The old `IQ2_XS/strata-mac-optimized.json` named below points at the
  files' old place and is superseded.
* **Current local server config (until setup):** `/Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json`.
  It uses 32,768 capacity, FP16 KV, 1,024-token prefill, and
  `--spec 4 --mtp-max-t 1 --suffix-draft 0`. The measured draft count is zero. All 24,576 experts
  reside in the GPU cache (33.02 GiB); `--mmap-experts --no-prefill-borrow` avoids another pinned
  host copy. Since round 22 the config sets `STRATA_METAL_IQ4_EXPAND=0` (it was 1 in round 21; copies of
  both versions are in `bench/results/2026-10-03-metal-decode-opt2/daily-config-*.json`). The view stays
  opt-in, bounded by min(4 GiB, GPU budget/16). MTP runtime files remain required by the serve interface
  even with drafts disabled.
* **Fully resident native IQ now runs as one Metal graph.** GPU routing indexes the bound cache arena
  through a device residency table and 64-bit slot offsets. The no-borrow configuration keeps those
  slots stable; partial caches retain the segmented path. Per-layer host time and GPU-reach wait are
  both zero in the final run. IQ data-producing kernels are asynchronous; synchronous legacy copies
  now wait for blocking streams in the runtime, while nonblocking streams require explicit ordering.
* **Native IQ prefill is fused.** Quantized weight slices decode directly into transposed threadgroup
  tiles, feeding FP16 matrix products with FP32 accumulation. The final kernel uses 5 KiB shared
  memory and matches unpacked FP16 GEMM bit for bit in the added tests. This replaces full expert
  dequantization and many small dispatches when the layer's used experts are all resident.
* **Round-20 cached follow-up and memory (before the extra views):** reuses 10,255 tokens, reads 35 new tokens, first text in 2.335 s,
  and returns the expected check code. Three short/repeated requests also pass. Engine physical
  footprint peaks at 40.3G; sampled swap stays at its initial 4,679.88 MiB. All test services are stopped.
  The four optimization-stage outputs match each other. Re-running the original round-19 binary also
  yields the same 256 output tokens, although its historical run differed; the old discrepancy's cause
  is unresolved. New audits compare 389 matched-input positions / 96,596,480 logits with zero bitwise
  differences, including 10K prefill and cached continuation. Six committed-state fingerprints match.
  Real-weight expert checks compare 23,347,200 values with zero differences. Four short requests also
  match round 19's committed states and output IDs. These are numerical-equivalence checks on tested
  inputs, not a general benchmark of model intelligence.
* **Two additional correctness bugs are fixed.** Public FP16/FP32 dequantization wrappers still
  rejected IQ types despite the kernels existing; they now dispatch correctly with source row offsets.
  Q3_K scale reconstruction omitted byte 12 of its packed scales; restoring it removes 2,681 fixture
  mismatches. The Q2_0 model used Q3_K tensors. With this fix, its same 10K check improves from 0/3 to
  3/3 fields and measures **111.9 tok/s prefill, 19.2 tok/s decode**. Earlier retrieval failures cannot
  establish that Q2_0 has worse model quality.
* **Q2_0 was deleted at the user's request after the corrected control run.** Shared PLE and MTP
  runtime files were preserved in IQ2_XS first. Historical logs remain, but the old model paths no
  longer exist. The removal manifest is `bench/results/2026-10-03-metal-iq2-xs/migration.json`.
* **100K still needs validation on the corrected engine.** Round 17 measured 104.1 tok/s prefill,
  18.6 tok/s decode and cached-follow-up first text in 3.081 s, but its outputs repeated "RAM".
  That executable contained the Q3_K error; do not use it to judge model quality or claim usable
  100K quality. Round 18's MTP on/off comparison also used that executable and is historical.
  IQ2_XS has only been tested with MTP disabled here.
* **Every kernel file has a Metal port** (44 .cu files; native_gr_norm/native_bf16 ride native_gr.mm /
  bf16_gemv.mm). Final round-21 CTest: **49 entries, no failures, 148.22 s**; three x86-only checks
  internally skip on ARM. Added copy checks cover 471 cases per compute/blit path, including mapped
  writes after graph launch. IQ4 views match the original shader over 27,702 outputs with zero bit
  differences, non-finite values or representation/lifetime errors. Round-21 real-model short + 10K +
  cached-follow-up audits again match all **389 positions / 96,596,480 logits and 6 committed states**.
* **The old first-request serve crash is fixed.** Graph executables now own their tape independently
  of the source graph. Stream/event lifetime and synchronization were repaired too. Two other real
  inference bugs were found: uninitialized prompt-attention accumulators, and fused-GR inactive warps
  overwriting adjacent injection gates. Passing the earlier parity suite did not cover these cases.

## Build, test, run (all from the repo root)

```bash
# the integration build (already configured; STRATA_METAL_ENGINE and STRATA_NATIVE_EXPERTS are ON,
# ggml FetchContent uses the existing build-iq/_deps/strata_llamacpp-src checkout)
cmake --build build-metal -j 8
ctest --test-dir build-metal --output-on-failure  # 50 entries (metal_mmvq_decode_test sets its own IQ4 view)

# focused numerical checks for the round-20 native IQ changes
ctest --test-dir build-metal --output-on-failure -R '^(iq_parity|iq_multi_parity|metal_gemm_test|metal_smoke)$'

# the server (all addresses stay local)
.venv/bin/python -m serve.server --engine strata --config strata-iq2_xs.json --port 8080   # setup's config (engine/strata)
curl http://127.0.0.1:8080/v1/models                                   # works (READY)
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"请简短介绍你自己。"}],"max_tokens":96,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

For numerical A/B, `STRATA_METAL_IQ_RESIDENT=0` (or `STRATA_METAL_LAYER_SYNC=1`) restores segmented
decode; `STRATA_METAL_PREFILL_F16_EXPERTS=1` restores complete expert dequantization before prefill
GEMM. Native whole-window decode is enabled by default only when all experts are resident and slots
cannot be borrowed. It was temporarily opt-in during the quality audit, then restored after the
matched-input logits and committed states passed. `STRATA_METAL_CHECK_IQ_RESIDENT=N` shadows the
first N segmented windows' expert calculations; `STRATA_METAL_LOGITS_PREFIX=/path/prefix` and
`STRATA_METAL_LOGITS_WINDOWS=N` save full logit traces. Diagnostics synchronize/read back and must be
off for speed measurements. The `audit_model.py` / `compare_audit.py` scripts in the evidence directory
configure these switches and fail on unmatched inputs, missing states, or numerical differences.

Rounds 22-23 kernels, each on by default and `=0` restores the original (bitwise the same outputs either way):
`STRATA_METAL_IQ4_DIRECT`, `STRATA_METAL_IQ4_SG`, `STRATA_METAL_EXPERT_DIRECT`, `STRATA_METAL_GR_NORM1`,
`STRATA_METAL_GDN_TAIL`, `STRATA_METAL_TOPK_SCAN`. Order-preserving diagnostics: `STRATA_METAL_CB_TIMING=1`
(GPU busy vs wall per command buffer) and `STRATA_METAL_KPROFILE=<csv>` (+ `STRATA_METAL_KPROFILE_SKIP=N`): one
timestamped pass per tape entry, relative numbers only (each pass adds 2-3 us); aggregate with
`bench/results/2026-10-03-metal-decode-opt2/micro/agg.py <csv> <entries>`. Run real-model tests one engine at a
time - memory is limited (user request).

Useful switches: `STRATA_METAL_DEBUG=1` traces every launch/copy/drain; `STRATA_TG_FLUSH_US` overrides
the ring-poll cadence (200 us is the Metal default and the measured knee - do not set 2000).

## Paths

* Model data now: `Strata-data/` in the repo (see "Installed by setup now" above); the lines below are the layout
  before that move.
* Repo: /Users/liangxw/src/xproject/petproject/Strata (branch main, committed in f90370f and after - this note was
  written when ALL WORK WAS UNCOMMITTED - commit is
  the new owner's call together with the user).
* Model data: /Users/liangxw/src/xproject/petproject/IQ2_XS/ - two GGUF shards (68.03 GB), pack-full/
  (1.43 GiB native pack + tokenizer, no separate experts.bin), mtp/rt (preserved draft runtime),
  strata-mac-optimized.json, and source/verification manifests. Source:
  ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF at ed59f92082b1e93c0e96d60a8b11aab089b52f09.
  The download used hf-mirror.com. Use `.venv/bin/python tools/iq_pack.py` for this pack; the system
  Python 3.9 does not support the tool's Path.write_text(newline=...) call.
* Old model directory /Users/liangxw/src/xproject/petproject/Q2_0/ was removed. Current config paths
  were checked after removal. The daily IQ2_XS config points at the optimized build-metal/strata.
* The Metal port's files: src/kernels/metal/*.mm + *.metal (one metallib, embedded), src/platform/
  metal_runtime.mm (the whole runtime: streams, tape graphs, the inflight-CB stream-query semantics),
  include/strata/metal_compat/ (the CUDA shim). Platform rules live in the .metal/.mm file headers too.
* Round-25 executable: `build-metal/strata-round25`, SHA-256
  `69aea4c61ebbce6ac0ce5a28a1dde2e54d8fae9ac1f827c49e6c1277f3e378a8` (decode = round 23's kernels).
* Round-23 executable: `build-metal/strata-round23`, SHA-256
  `1a5300ad257e19162346d599d24cb465efb6a51cc11a9e9d6248bfd6578d7b31` (round 22's `bc95a289...` was rebuilt over).
* Round-21 reference executable: `build-metal/strata-round21`, SHA-256
  `0c65709c5d8dd410578d2f2378a8258b7e2dbdfe2bbef43f719bf2b57ddbb86c`.
  `build-metal/strata-round20` is also preserved. Current performance and quality evidence is in
  `bench/results/2026-10-03-metal-decode-opt/`; use its README to distinguish retained changes from experiments.

## The rules that bite (full versions with measurements: PROGRESS rounds 9-13)

1. Pointers NEVER travel as data: a pointer stored in setBytes/buffer memory dereferences to zero and
   writes vanish (rule 9). Every pointer is its own bound [[buffer(N)]]; scalars only via k.scalar().
   The [[buffer(N)]] index must equal the argument's position in the .buf()/.scalar() chain.
2. A dispatch wider than the pipeline's maxTotalThreadsPerThreadgroup is silently DROPPED (rule 10) -
   the runtime aborts naming the kernel.
3. A kernel already spinning on a mapped flag NEVER sees later CPU stores (rule 11), and a killed
   spinning kernel WEDGES the whole GPU until a real sleep/wake. The verify window and token graph
   still need SEGMENTS when CPU experts participate: raise flags between launches, then commit.
   With the immutable canonical Q2 expert arena, both token and verify graphs run entirely on the GPU,
   without host flags or per-layer round trips. Round 20 extends this to native IQ with all experts
   resident and prefill borrowing disabled; partial native caches still use segmentation.
   Do not insert flag waits into the fully resident graph path.
4. -fno-fast-math build: only metal::precise:: spellings exist. No fp64 on the GPU (emulation patterns
   in strata_port.metalh and qsa.metal's double-float layer). 32 KB threadgroup cap.
5. Do not add an eager commit inside cudaGraphLaunch - measured to break parity (round 14).

## What to do next, in order

1. Remaining decode GPU time after round 23 (KPROFILE shares, `-opt3/profile/`): IQ4_XS 18.7% (still ALU-bound,
   ~175-215 GB/s), IQ2_S + IQ2_XXS gate/up 16.2% (55-65 GB/s; every variant tried in round 22 failed - a
   byte-load-only floor is half the kernel, so it needs a different access pattern, e.g. a simdgroup loading
   whole 82-byte blocks), Q2_0 down 10%, GR up/down/norm 18% (down/up near 250 GB/s), GDN recurrence 2.3% plus
   (round 26 removed the commit's second recurrence at T = 1), `route` 2.3% (one simdgroup, serial top-10; bitwise-sensitive
   order), attention chunk 3.2%. Earlier round-21 note, kept: Round 21 warm decode takes about 51 ms/token. Preserve
   the user's memory/value constraint: measure full decode and additional bytes, and discard noise-level
   gains. Round 21's per-entry command-buffer profiler was removed because it changes submission timing
   and violates mapped-source late-write ordering; its CSV is diagnostic history only. Use the safe
   CPU-only `STRATA_METAL_GRAPH_TIMING=1` or an order-preserving GPU profiler. Keep original operations,
   independent references and round 20's matched-input real-model audit scripts as correctness gates.
2. (Done 2026-10-04 at 128K, one sample per model: `bench/results/2026-10-04-metal-128k/`.) Repeat 100K and
   cached-follow-up validation with IQ2_XS and the corrected engine before claiming
   usable long-context quality. The 10K check now passes, but it does not validate 100K. Reconsider
   speculative windows only with fresh measurements; the old Q2 comparison predates the Q3_K fix.
   For captured resident experts, bind an immutable arena and index it on the GPU: reading a
   GPU-generated pointer table during capture binds stale experts even if execution does not crash.
3. **Housekeeping**: the working tree is one big uncommitted change set (the whole port + phase-1 macOS
   support). tests/bench and fixtures live inside build dirs (iq fixtures, ple stand-in pack) - the
   deterministic generators are tools/iq_fixture.py and tools/ple_fixture.py.

## Ledger pointers

STATUS.md (per-file, keep in step with cmake/metal_backend.cmake's STRATA_METAL_PORTED), PROGRESS.md
(rounds 1-21: every rule with its measurement - read before touching runtime/shader code),
PLAN.md (original architecture), docs/MAC.md (Mac dev environment).
