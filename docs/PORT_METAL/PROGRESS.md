# The Metal port - progress log

Append-only, newest last. Every entry: the date, what landed, and what was measured (and on what).
Plan: [PLAN.md](PLAN.md). Live state: [STATUS.md](STATUS.md).

## 2026-10-02

* Decided the port and wrote the plan. Survey that fixed the module cut (all on this checkout):
  * kernel launch seam: `include/strata/kernels/*.hpp` declares ~214 host entry points; 44 `.cu` files hold
    ~190 `__global__` kernels (16,895 lines); the host engine (layer/session/verify/mtp/expert_source,
    6,800 lines) never launches a kernel directly.
  * host CUDA surface: ~25 `cuda*` functions; graph logic isolated in `src/core/graph.cpp`; the only device
    code outside the kernel files is the 6-line `poison_kernel` in `src/core/device.cu`.
  * (already in from the macOS work earlier today: C++ CPU side builds and passes 12/12 ctest on this Mac,
    scalar expert paths in place - see docs/MAC.md.)

## 2026-10-02 (second session)

* **M0, M1, M2 and kernel file K1 landed; `elementwise_parity` passes with 0 failures** on the M2 Max
  (ctest 15/15 in the Metal build: the two Metal tests + the 13 CPU-side ones).
  * `strata-device`: "Apple M2 Max", working set 38.9 GiB, shared memory per threadgroup 32 KiB.
  * `metal_smoke` (all measured on this machine): launch round-trip exact; **GPU->CPU doorbell ring visible
    to a spinning CPU in 0.2-0.6 ms with NO stream sync** (relaxed device atomics + a device fence - the risk
    item M1 existed to retire); CPU->GPU wait ordering holds through a captured stream; graph-tape replay
    0.14-0.30 ms per replay (one scale kernel) and replays read live buffer contents.
  * `elementwise_parity --selftest`: 0 failures. Exact where CUDA is exact (f16/bf16 conversions, Q8_0
    dequant, scale/add), rel 3.4e-08 (silu, softplus) and 3.6e-08 (rms_norm) where CUDA rounds differently.
* The port's debugging ledger - five behaviours that are now ENCODED in the runtime, each found by a
  measurement (this is the part a next session must not relearn):
  1. **A CPU memcpy into a shared buffer is not guaranteed GPU-visible.** cudaMemcpyAsync is therefore a
     blit encoder on the stream (GPU-ordered, hazard-tracked); unowned host sources stage through a temp
     buffer first. Symmetrically, CPU-direct writes to a tracked buffer can be masked by the CPU's own
     cache lines - the blit path avoids both directions of the gap.
  2. **`cudaDeviceSynchronize` must take the pending command buffers out, commit, THEN wait** - nilling the
     cursors first made the wait a no-op, and every "kernel did nothing" report below was that race.
  3. **Manual reference counting**: this ObjC++ TU stores Metal objects in C++ containers; every store
     retains, and an `@autoreleasepool` anywhere in the call path drains the pile that unretained objects
     depended on (a per-copy pool freed every registered buffer; the third copy segfaulted).
  4. **A null kernel argument binds nil** (`setBuffer:nil`), not a dummy buffer - the kernels test their
     pointers, and a dummy flips the branch (rms_norm's null-weight case read garbage).
  5. **`xcrun metal` defaults to lossy fast math**: `metal::precise::exp/log/rsqrt` per call site plus
     `-ffp-contract=off` (the HIP backend's own discipline) is what makes the parity tolerances pass; MSL has
     no log1p, and the softplus series seam needed the e<0.05 Horner form (f32 log(1+e) is cancellation-bound
     there - 2 of 48 fixture heads were off before it).
* MSL reality notes: scalars ride as `constant T&` + setBytes; thread coordinates come as attributed
  parameters (`uint ... [[thread_position_in_grid]]` - not free identifiers, and 32-bit only); device-space
  atomics support only relaxed order on this toolchain (the seq_cst spelling does not compile), so the
  doorbells are relaxed atomics + `atomic_thread_fence(mem_device, seq_cst)`.
* Next (see STATUS.md): K2 dequant_s2, K3 s2_gemv, K4 quantize_act, K5 router_top10 - the decode path; then
  the GEMM (M4) and the engine link (M5).

## 2026-10-02 (third session)

* **K2-K5 landed in one pass; the Metal build's ctest is 19/19.**  The decode path's first five kernel files
  are ported and parity-green on the M2 Max:
  * `dequant_s2_parity`: bit-exact (the decode is exact integer-to-float arithmetic; a tolerance would only
    hide a bug, the CUDA test says - and none was needed).
  * `s2_gemv_parity`: the row dot's accumulation order kept; within the phase tolerance.
  * `quantize_act_parity`: **byte-exact** - Q8_0 over 5 distributions x 131,072 elements including the exact
    .5-tie fixture, and Q8_K including bsums.  Two fp64 spots emulated (see below).
  * `router_top10_parity`: ids exact, weights within the 1e-5 tolerance, |sum-1| fine.  The HIP-only "fast"
    variant stays off exactly as on the CUDA build.
* **The fp64 problem and its answer** (Apple GPUs have no double - the compiler says so outright):
  * `rint((double) a / (double) b)` (quantize_q8_0's rule) -> `rint_of_ratio` in strata_port.metalh: the
    correctly-rounded f32 quotient refined by its EXACT residual (`fma(-b, q0, a)` does not round), which
    decides .5 boundaries at ~46 good bits.  The exact-tie fixture (522,240 ties, half where half-to-even
    != half-away) passes.
  * the router's ascending double sums -> ascending Neumaier-compensated f32 sums (error ~2^-46 relative,
    where a plain f32 sum would already sit at the 1e-5 tolerance over 512 terms).
  * the router's `exp((double)...)`: the ARGUMENT is exact in f32 (Sterbenz), `metal::precise::exp` carries
    ~1 ulp - two orders under the tolerance, measured by the parity run.
* **The compiler flags that made bits match**: `-fno-fast-math -ffp-contract=off` (the default fast math
  approximates RECIPROCAL DIVISION - Q8_K's `1.0f / iscale` came back 1 ulp off until this; the port's
  .metal files already spell `metal::precise::` everywhere, which is what makes -fno-fast-math usable at
  all, since the plain exp/log names disappear without it).
* Shared MSL helpers moved to src/kernels/metal/strata_port.metalh (f16/bf16 converters, rint_of_ratio,
  KahanSum) so the kernel files stop carrying private copies - the CUDA tree's round-198 lesson.
* Next: K6-K14 of the kernel table (shared_expert, rope, sampler, gr, gdn, the qsa family, kv_*), then M4
  (MPS GEMM for prefill) and M5 (link the engine).

* **K7 landed: `rope_parity` passes** (with native_rope.cu and the mrope/rope-table registries that parity
  pulls in - the CUDA tree keeps those in native_rope.cu, and the port follows that split).  The float64
  frequency tables stay HOST-side exactly as the CUDA file insists; the MSL gotchas of the day: `half` is a
  reserved TYPE NAME in MSL (the rope kernel's local `half` became `half_n`), and 2D grids read as
  `uint2 [[thread_position_in_grid]]`.

* **K14's kv_q8 half landed (with qsa.cu's two FP16 KV steps): `kv_q8_parity` passes** - INT8 codes and
  FP16 scales BITWISE equal to the host reference, worst INT8-vs-FP16 error 0.587 quantization steps (the
  phase bound).  cvec (K13) is ported too, but its parity needs `fused_gr_read`, so it waits for K16.
* **A whole bug class found and fixed: `thread_position_in_grid` vs `threadgroup_position_in_grid`.**  In
  multi-dimensional dispatches where the CUDA kernel used `blockIdx` as a DATA index (kv_append's
  head/group/KV triple, cvec's stream/token pair), the MSL port must read the GROUP position, not the
  global THREAD position - and where CUDA computed `blockIdx.x * blockDim.x + threadIdx.x`, the global
  thread position IS the right thing.  Mixing them up writes nothing at the right coordinates (kv_q8 was
  98.7% wrong with no Metal error at all); rope_neox had passed only because its row count fit one
  threadgroup.  MSL also requires every attributed grid coordinate in one signature to have the SAME
  vector width (uint2 + uint does not compile).
* The `KvHostPools` struct (eight device pointers) rides through a setBytes argument VERBATIM - on unified
  memory the CPU pointers are valid GPU pointers, so the C layout copies straight into the constant-buffer
  bytes.
* Metal build ctest: **21/21**.

* **K17's bf16_gemv landed with the fp32-MMVF pair from native_bf16.cu: `bf16_gemv_parity` passes; ctest
  22/22.**  The whole gr FAMILY is now ported and compiling - gr.cu's six kernels in all three activation
  modes (bf16 storage, fp32, native MMVF through the bf16_gemv/native_gr ports) - but `gr_parity` cannot
  link until fused_gr.cu's read/read_multi exist, so K9 is "ported, parity blocked" exactly like cvec/K13.
* The MSL attribute-width rule, finally pinned down: within one kernel signature, `thread_position_in_grid`
  must agree in width with `thread_index_in_threadgroup`/`threads_per_threadgroup`, and `uint3` is not even
  valid for `thread_index_in_threadgroup`.  The port's standing pattern is now: flat kernels take plain
  scalars; group-indexed kernels take `uint3 [[threadgroup_position_in_grid]]` + SCALAR thread/lane/simd
  indices, and any block size the body needs arrives as a plain argument (the launcher knows it) instead of
  `threads_per_threadgroup`.

## 2026-10-02 (round 9): K16 fused_gr lands, and rule 9 - pointers that are not pointers

* **K16's fused_gr.{metal,mm} is green: ctest 24/24** - gr_parity 0 failures (all three activation modes,
  the scalar capture fixtures, and the fused multi/single/graph checks) and cvec_parity all passed,
  including its folded write being BITWISE the fused read's.  The read is three one-token launches (norm,
  down, up) with xn staged in the caller's device scratch: the CUDA kernels stage 40-80 KB of activations
  in shared memory and this GPU's maxThreadgroupMemoryLength is 32768 (measured, Apple M2 Max), but the
  CUDA file's own tile invariant - a lane's chunks of 8 accumulate in ascending j with either tile - makes
  reading them straight back bit-identical.  The T-token read is T of those launches; per token the
  arithmetic is the CUDA kernels' own, so multi == single bit for bit, as fused_gr.hpp promises.  The v3
  read (STRATA_GR_V3) and the split/staged variants are CUDA shared-memory and cp.async stagings of the
  same arithmetic and do not exist here; fused_gr_variant() says plain.
* **RULE 9 - a pointer stored in buffer memory or setBytes data is NOT a usable device pointer on this
  GPU.**  The first fused_gr port passed FusedGrArgs/GrMulti as bytes, the way KvHostPools rides kv_q8 -
  and every kernel "ran" (command buffer completes, no error) while reading zeros and writing nothing.  A
  standalone probe series measured it cleanly:
  - the struct's bytes arrive VERBATIM (a 912-byte setBytes echo'd back word-for-word; layout, scalars,
    even the pointer VALUES load right - printing one gave exactly the buffer's contents address), but the
    first DEREFERENCE of such a pointer reads 0.0 and writes through it vanish - flat 8-pointer struct,
    nested arrays, constant AND device parameter spellings, real buffers as well as setBytes;
  - reconstructing the pointer as `anchor + byte_offset` from one argument-bound anchor WORKS - reads and
    writes, multiple buffers, negative offsets - but only within ~8-16 KB of the anchor's base: with the
    anchor buffer sized 8192 the same kernel stores land, sized 16384 they vanish, and a 1 MB anchor with
    correctly recomputed 64-bit offsets (verified word-exact on the GPU) still writes nothing;
  - a bound `[[buffer(N)]]` argument reaches anywhere, interior pointers included (the runtime binds them
    with large offsets, exactly like every other ported kernel).
  So the port's standing pattern is: **every pointer is a bound buffer argument; the only things that
  travel as bytes are scalars.**  This also DISPROVES the round-7 note below that said KvHostPools copies
  verbatim into constant bytes "so the CPU pointers are valid GPU pointers" - the bytes copy fine, the
  pointers just do not work; kv_q8's parity never noticed because it passes host=null (K14's row in
  STATUS.md now says so, fix with K15).
* **Two latent bugs the blocked tests were hiding, both found by gr_parity's first runs:**
  - gr_norm_kernel declared its block-size scalar `[[buffer(9)]]` while the launcher set 8 arguments - the
    unset binding read garbage as `d += block`'s stride, and with a 0 there the loop never ended: the GPU
    span at 100% and cudaDeviceSynchronize hung in waitUntilCompleted (the whole suite previously never
    executed this kernel).  A signature-vs-launcher audit then flagged the same gap on both MMVF kernels,
    whose `block` argument was never bound at all - they had been running on whatever the previous dispatch
    left at that encoder slot.  Rule: the buffer index must equal the argument's position, always.
  - both MMVF kernels cast the bf16 weight pointer to `uint*` BEFORE striding `+ row * n_in`, so every row
    walked 4-byte units where the CUDA original strides uint16 elements - row k computed row 2k's dot
    (row 0 alone looked right, which is why nothing that only checked row-adjacent behaviour caught it).
    Fixed with the uint16 stride first, then the pair view; a 6-shape probe (n_in 640..10240, n_out 4..512)
    now reads 0 rows wrong, and bf16_gemv_parity stays green.
* Next: K10 gdn.cu, then K6 shared_expert + the s_gemv family, then K8 sampler (K15's qsa work carries the
  KvHostPools rule-9 fix).

## 2026-10-02 (round 10): the parallel waves begin - four ports in one round, ctest 30/30

The port now runs in waves of parallel agents, each owning disjoint `src/kernels/metal/<stem>.*` files and
its own build directory; the CMake registration was pre-seeded with every remaining stem and the parity
gate now also requires the stem's `.mm` to exist, so an unported stem never breaks another agent's build.
Integration stays central: one build dir (`build-metal`), STATUS/PROGRESS, and the shared runtime.

* **Wave 1, all four green in the integration build (ctest 24 -> 30/30):**
  - K10 gdn + fused_gdn: gdn_parity 0 failures (worst rel 2.0e-07, conv state slide bitwise).
  - K8 sampler: 0 failures on all three sampled paths; the penalty bitmap moved OUT of dynamic threadgroup
    memory into a device-memory ring (32,768 B at a 262144 vocab was the whole measured budget), and the
    merge stages read candidate lists straight from device memory instead of 32 KB shared copies.
  - K12 s_gemv + s2_gemv_quads + s2_gemv_fast + K6 shared_expert: s_gemv_parity 0 failures (S2/Q2_0
    bitwise, worst 9.6e-06); shared_expert_parity 0 failures; s2_gemv_fast's constant table is COMPUTED
    in-kernel (the table held exactly representable integers, so the arithmetic is bit-identical and
    cudaMemcpyToSymbol is not needed); swiglu/scalar-gate/moe_combine doubles became precise::exp /
    KahanSum per the established precedents (structure tol 3.5e-04 vs CUDA's 2.2e-07, 29x under the 1e-2
    tolerance, byte-stable).
  - K19 iq_kernels: iq_parity 0 failures - dequant BIT-EXACT over all 10 formats; iq_multi_parity 0
    failures (multi==single bitwise, native_expert_grouped 24 checks bitwise vs old kernels).  Codebook
    grids ride the metallib as `static const constant` arrays via the GGML_COMMON_DECL/IMPL_METAL pair;
    `__byte_perm`/`__vcmpne4`/`__vsub4`/`__dp4a` transcribed from the repo's own hip_compat/dp4a
    definitions; roundf ties-away via an iqk_roundf (MSL has no precise::round).  Parity caught a
    transcription bug (Q2_0's Fmt step is 1, not 2 - with 2, half the dot read past the block).  Fixtures
    are deterministic (tools/iq_fixture.py) and generated into each build dir.  A native_mmvq BRIDGE rides
    iq_kernels.mm (forwards type codes to iq_mmvq, exactly what native_mmvq.cu does; Q3_K via a vendored
    vecdotq dot) - DELETE it when K18 ports native_mmvq.cu or the symbols collide.
* **RULE 10 (measured, gdn's port): a dispatch wider than the pipeline's occupancy limit is silently
  DROPPED.**  A register-heavy kernel pinned its PSO to maxTotalThreadsPerThreadgroup = 384; the
  512-thread dispatch completed its command buffer clean and executed nothing - no error, no writes.
  `Launch::done()` now refuses such launches by name (metal_runtime.mm); the gdn kernel itself was
  restructured (drop the per-thread s[32] array, reload each row in both passes - bits unchanged).
* Wave 2 dispatched: qsa chain (qsa/select/decode_attn/prompt_attn), the kv family (kv_q8's rule-9 fix
  for non-null KvHostPools + kv_q4 + kv_stream), s2_expert_grouped, ple.

## 2026-10-02 (round 11): wave 2 green - the qsa/kv/ple/grouped families, ctest 37/37

* **Wave 2, all four green in the integration build (ctest 30 -> 37/37):**
  - K15 the qsa chain (qsa, qsa_select, qsa_decode_attn, qsa_prompt_attn): qsa_parity 0 failures (spare
    key BIT-EXACT through a double-float emulation layer - exact two-products, compensated sums, 2-step
    Newton rsqrt, residual-refined divide, all in the CUDA order; indexer scores 1.23e-08; all 31
    static+captured tail-replay cases); qsa_prompt_attn_parity 0 failures at `32768 2048 5`.  The prompt
    kernels' mma.sync/cp.async/WMMA became an f32 online-softmax kernel - 2.4x slower than the decode path,
    a correctness stand-in; qsa_block_scores_tc returns false (no TF32; warp scores as pre-sm_80 CUDA).
  - K14 the kv family: the KvHostPools rule-9 fix (eight bound buffers in declaration order, nulls bind
    nil), kv_q4 (FWHT bitwise; a gather-scale `const uint` truncation bug found by a standalone probe -
    every |d|<1 read as 0), kv_stream (306 batches / 728,342 lookups / 90.1% hit, 0 failures), kv_hybrid
    (0 bad of 786,432 K-appends; mode-3 attention 1.15e-07).
  - K20 ple: 0 failures on a DETERMINISTIC SPARSE STAND-IN GGUF (every IQ4_NL value a 2^-24 multiple, so
    the probe rows solve exactly; head/tail bit-equal to the oracle include) - the 68 GB artifact wall
    moved, not faked; with the real shard provisioned the env var becomes unnecessary.
  - K22 s2_expert_grouped: 12 old-vs-new comparisons BITWISE, host reference worst 1.27e-07 of the 4e-5
    bound; moe_grouped_s2 needs a sync before reading the device-written pointer table, so it is NOT
    graph-capturable (only the P6 verify path calls it; documented).
* **Integrator-side gaps fixed:** a metal `cuda_fp16.h` (bit-level f16_bits.hpp under the CUDA names), a
  one-argument cudaEventRecord overload, per-test ctest arguments (qsa_prompt_attn_parity takes positional
  geometry; `--selftest` would parse as context length 0), and the ple artifact wiring (sparse
  ple_table.gguf + pack/full/dense.bin + bench/micro/ple_{in,out}.bin cloned into the build dir).
* **Bridges to delete when K18's files land (duplicate symbols otherwise):** iq_kernels.mm's native_mmvq
  bridge, ple.mm's native_mmvq block, qsa.mm's native_qsa_indexer bridge.
* Wave 3 dispatched: native_mmvq (+ bridge deletions), the native_qsa trio (+ bridge deletion),
  native_gdn pair, native_router/flash_attn/ple_postops + s2_gemv_q8 as a real port (its private copies
  in shared_expert.mm and ple.mm migrate away).  The native_* parities live in bench/micro (not in the
  published source, absent here), so those files verify via the existing green tests that exercise their
  bridges plus /tmp host-reference probes - the fused_gdn precedent.

## 2026-10-02 (round 12): wave 3 green - the native_* family and K11, ctest 39/39

* **Wave 3, all four green in the integration build (ctest 37 -> 39/39):**
  - K18 native_mmvq: 51 kernels, 34 entry points.  The iq_kernels.mm and ple.mm bridges are DELETED; the
    four formats they used to answer now run real kernels with iq_parity/ple_parity still 0 failures.
    CUDA's own src/kernels/mmvq_multi_parity.cpp (never bridge-passable) ran against the Metal build:
    multi_exact ON -> 148,480 outputs, 0 bitwise diffs vs single-column calls; OFF (negative control)
    differs at every T>4 - the comparison has power.  Now registered and passing.  NCOLS/NW became
    runtime scalars (~280 template instantiations -> 30 kernels): a column's accumulation chain never
    mixes columns, measured 0 diffs over the 148k outputs plus 10 formats in the probe.
  - K18 native_qsa trio: the qsa.mm bridge deleted; qsa_parity 0 failures behind the real entry points
    (batch vs sequential bitwise over 33,409 state words x 3 scalings).  Uncovered entries probed vs
    double references (worst 8.2e-06, the f32-angle trig the CUDA table feature exists to remove).
    score_kernel's TF32 path uses the CUDA file's own gfx1100 scalar fallback.
  - K18 native_gdn pair: probes worst scale-rel 3.4e-07 (chained gate recurrence), conv history slide
    bitwise; native_gdn_pipeline has no code of its own (only a CUDA oracle test name) - covered.
  - K18 native_router / native_flash_attn / native_ple_postops: probes 0 failures each (router ids
    exact, flash attn widths 1/137/256 x mask on/off, ple_postops aliases and T=12 batches).
  - K11 s2_gemv_q8 became a real port: parity 0 failures; shared_expert's and ple's private copies
    deleted, both parities stayed green.
* What is LEFT of the kernel phase: native_moe.cu (86 lines, one combine kernel) and K21
  verify_kernels.cu (559 lines, includes gpu_stamp), plus the two engine-side wirings the waves refused
  (shared_expert's native-MMVQ path, ple_block's native-postops path) - dispatched as wave 4.  Then the
  serial segment: M4 (MPS prefill GEMM), M5 (engine link), M6 (real model, download deferred).

## 2026-10-03 (round 13): wave 4 - the kernel phase is COMPLETE

* **Every src/kernels/cuda/*.cu file now has a Metal port** (native_gr_norm/postops and native_bf16 ride
  native_gr.mm / bf16_gemv.mm; native_gdn_pipeline never had code of its own).  Integration ctest 39/39,
  plus probe verification where no in-repo parity exists (bench/micro is absent from the published source).
* Wave 4 (finished after a machine reboot ate the first agent mid-run - its files survived on disk and
  were audited line-by-line rather than trusted):
  - native_moe: 46,216 probe checks, 0 failures, worst rel 1.7e-07; the f32 contract (first product
    rounds, then the FMA chain in expert order, spelled out because -ffp-contract=off) matches BITWISE on
    every output; multi == per-token singles bitwise.  shared_expert's combine is its OWN distinct CUDA
    code (double accumulation, k<=64), not a duplicate - left alone.
  - verify_kernels (K21): 238,708 probe checks, 0 failures.  ONE REAL BUG found in the interrupted
    session's files: step_state_shadow was 32x undersized (h_v*S*RPG instead of S*S*h_v) - any T>=2
    window corrupted neighbouring buffers with values exploding to 1e28/NaN.  Fixed to the (row,head,col)
    layout gdn_parity's st_dev uses.  gpu_stamp is a strictly-increasing execution-time launch counter
    (MSL has no readable GPU clock): stamps count LAUNCHES, not nanoseconds - verify.cpp's ns->ms print
    will read launch-counts/1e6; captured replays keep the sequence increasing.  fused_gr.mm's two stamps
    (after the norm pass, after the down pass) are wired, matching launch_multi's.
  - The interrupted session's shared_expert/ple native wirings had survived the reboot too and verified
    green in the full suite.
* **RULE 11 (measured, verify_kernels' wait flags): a GPU kernel already spinning on a mapped-host flag
  does NOT observe CPU stores made after its launch.**  A plain store was never seen in 8 seconds; a
  volatile + seq_cst + full-barrier store was seen once after 4.1 s and never again across two 15 s runs.
  The wait kernels' read paths are verified pre-set and through a captured replay with the flag raised
  before replay (the realistic engine shape), but M5's split-verify CANNOT release a mid-flight window
  from the host on this backend as CUDA does - the engine's verify overlap must raise flags between
  launches (stream-ordered), not during one.  COROLLARY, also measured: a killed spinning kernel WEDGES
  THE WHOLE GPU until a true sleep/wake cycle (maintenance sleeps do not clear it; system daemons can
  delay the sleep a long time) - never leave a spin-wait kernel running without its exit condition.
* What remains is the serial segment: M4 (MPS prefill GEMM), M5 (engine link - mind rule 11 in the
  verify overlap design), M6 (real model end-to-end, ~68 GB download, deferred as planned).

## 2026-10-03 (round 14): M4 + M5 - the prefill GEMM on MPS and the engine boots

* **M4 (prefill GEMM on MetalPerformanceShaders), probe-verified:** gemm.mm + kernels.mm + 27 pfl_-
  prefixed kernels in the shared metallib.  The cublas->MPS mapping is measured, not assumed:
  cublasGemmEx(OP_T, OP_N, ..., COMPUTE_32F) is MPS's transposeRight:YES flag only (no data moves), MPS's
  f16 GEMM is BITWISE a sequential ascending-k f32 accumulation (16,896/16,896 outputs), and beta==0 does
  not read C like cuBLAS's.  MPS HARD-REJECTS BF16 matrices on this machine (an NSAssert abort, measured)
  so the bf16 path widens exactly to f32 first (bf16xbf16 products are exact in f32 - COMPUTE_32F
  semantics, one extra pass).  44/44 GEMM shapes vs double (worst scale-rel 3.6e-07), 34 kernel cases,
  and the custom-kernel -> MPS -> custom-kernel order proof both directions.  Three additive runtime
  exports carry the seam (mtl_command_buffer / resolve_buffer / mtl_device, opaque void*).
* **M5 (engine link + boot):** the eleven engine .cpp files compile and link on Metal (whole-archive
  check: ZERO undefined symbols; the only missing symbol in the entire engine was bf16_rows_dot_multi,
  now a scalar ARM transcription).  Three additive shim entries (cudaGraphNodeType, cudaMemcpy2DAsync
  per-row, 4-arg cudaMemcpyAsync).  `strata --help` prints usage; with a 1-layer real pack and the real
  PLE shard it loads dense weights, opens the PLE table, reports the device, and stops exactly at the
  expert-arena size refusal - the planned M5 bar.  Engine tests registered and green in the main build.
* **RULE 11 made concrete - the segmented window:** the CUDA verify window (one captured graph released
  mid-flight by host flag stores) became SEGMENTS on Metal: pre(l,grp) chains captured once and replayed
  between host steps, each post run LIVE (serve the pool, raise flags, THEN launch the waiters whose exit
  conditions are pre-set), tail as a last segment.  The token path got the same segmentation (its
  doorbell_wait kernels have the identical hazard).  Two NEW measured contract facts the design leans on:
  (a) TAPE LIFECYCLE - cudaGraphInstantiate is identity and cudaGraphDestroy deletes the tape, so the
  CUDA tree's destroy-after-instantiate spelling is use-after-free here (guarded in verify.cpp/session.cpp);
  (b) COMMIT + VISIBILITY - launches sit in the stream's open command buffer until a commit, and a host
  polling mapped memory sees a GPU kernel's writes only through driver entry (a plain or seq_cst poll
  stayed blind for seconds in isolation; with a commit + a cudaEventQuery every ~0.1-2 ms the ring is
  visible in 0.08-2.3 ms).  The engine's loops already carry the cure; explicit cudaEventRecord kicks
  follow every segmented launch.
* **A runtime-level eager commit was tried and REJECTED by measurement:** committing inside
  cudaGraphLaunch (to make launches "look async" without caller kicks) broke sampler/qsa/ple parity
  (captured-replay rows read -1 - host writes to mapped staging race the eagerly-committed replay).
  Reverted; the engine's explicit kicks are the correct mechanism.  Do not re-add.
* **Integration state: ctest 43/43** (39 kernel parities + 4 engine tests), STRATA_METAL_ENGINE and
  STRATA_NATIVE_EXPERTS ON in build-metal (the ggml FetchContent pointed at the local _deps copy - the
  sandbox has no network for it).  The M6 weights are down (66.4 GB via hf-mirror, strata-gguf-verified);
  a full 48-layer pack build is next, then the first real generate.

## 2026-10-03 (round 15): M6 - the model GENERATES on the Mac

* **First real end-to-end run: the full Qwen3.8-Flash-Next Q2_0 on the M2 Max.**  The 48-layer pack
  (tools/strata_pack.py from shard 1, 38 GB) loads: 4630 MiB dense, the PLE table's 320,001,536 rows from
  shard 2, the 31.64 GiB expert arena at 2.65 GiB/s; the token graph captures (48 layers); a 23-token
  prompt prefills and 24 tokens generate: **decode 2.06 tok/s, prefill 1.96 tok/s** (greedy).  ple_parity
  re-verified against the REAL shard 2 in place of the sparse stand-in: all passed.
* **One runtime lie caught by the first real token (and fixed):** cudaStreamQuery returned Success as
  soon as nothing was PENDING host-side - after the kick event commits the segment's command buffer the
  poll read "graph finished" while the GPU was still executing, and session_run_token declared "layer 0
  never rang" falsely.  Streams now retain their last committed command buffer and the query answers
  NotReady until it truly completes (metal_runtime.mm).  The eager-commit experiment of round 14 stays
  rejected; this is the correct half of that idea.
* Performance is correctness-first: 2 tok/s vs a PC's ~30-40 comes from the known stand-ins (the prompt
  attention's f32 online-softmax, the bf16-GEMM f32 widen, per-group launches with D2H pointer-table
  reads, the segmented token path's 48 host round-trips per token).  Optimization is future work with
  measurements, not part of M6's bar.
* M6 remainder in flight: the MTP draft layer (tools/mtp_fetch.py now honours HF_ENDPOINT for mirrors)
  for serve mode's --spec/--mtp, then serve/server.py --engine strata handshake.
* **First measured optimization (the engine's own --stats table):** at the default settings the token
  spent 641.9 ms/token - 562.5 of it inside LAYERS: "wait for rings" 316.4 + CPU pool 246.0 (PLE 0.2,
  embed 2.5, head 15.2, sample 1.1).  The ring poll's driver-entry cadence (STRATA_TG_FLUSH_US) defaulted
  to 2000 us - on this backend that is ring-DETECTION latency the pool pays too: 200 us measured
  641.9 -> 449.2 ms/token (+43%, pool 246 -> 154 just from starting sooner); 50 us over-flushes and
  regresses (490.4; 290k flushes).  200 us is now the STRATA_METAL_BACKEND default (session.cpp), env
  override intact.  NOTE: the MTP download was running during these measurements - re-baseline when idle.
  The two remaining hot spots, measured: the CPU expert pool (scalar ARM transcription of the AVX-512
  bf16 dots - NEON dot-product intrinsics are the obvious next win) and the 48 per-layer ring waits
  (segment batching).
* **OPEN (the one known bug, first thing for the next pair of hands): serve + --native crashes on the
  first request.**  The engine reaches READY (the server's /v1/models answers, n_ctx 32768) and exits
  SIGSEGV (-11) on the first GEN line - the server auto-restarts it, but every request dies.  The plain
  generate path is fine (round 15's runs, with and without --stats); the crash is in the combination only
  serve exercises with --native: the verify window's segmented capture under the real fast-path flags,
  the prefill pipeline (MPS + pfl_ kernels) on real data, or the MTP spec path - all first exercised
  here.  Reproduce: the config /Users/liangxw/src/xproject/petproject/Q2_0/strata-mac.json + `python3 -m
  serve.server --engine strata --config ... --port 8080` then any chat completion; or feed
  `GEN 8 <ids>` on stdin to the same strata --serve command line directly (the exact argv is in the
  config).  A first lldb --batch attempt produced no backtrace (killed by timeout) - retry with a longer
  patience or sample the spinning state.  NOTE: serve REQUIRES --native <shard1> (Verifier::init refuses
  otherwise: "the native BF16 projections are off ... the verify window reproduces the default native
  decode path") - that flag was missing at first and is now in the config.

## 2026-10-03 (round 16): full-resident Metal inference, faster prefill, and real correctness fixes

Measured on the same M2 Max (38 GPU cores, 8 performance + 4 efficiency CPU cores, 96 GB), macOS 26.5,
Release build, the full Q2_0 model in `/Users/liangxw/src/xproject/petproject/Q2_0`. Logs and repeatable
drivers are in `bench/results/2026-10-03-metal/`. Model weights were not modified. Earlier rounds remain
a historical record; the graph-ownership workaround and the first-request serve failure above are superseded.

### Correctness before speed

* `cudaGraphInstantiate` now copies the tape; an executable owns its entries independently of the source
  graph. Captured copies retain their buffers. Source-graph destruction followed by replay is tested.
  Stream synchronization waits for already committed work even when there is no open command buffer.
  Events retain their command buffer instead of using a callback that can touch a deleted event.
  Metal objects have explicit ownership, and short autorelease pools keep temporaries bounded.
* The segmented token runner had skipped its final segment. It now executes the final post-layer work.
* `qsa_prompt_attn.metal` initialized its online-softmax accumulator too late: multiplying uninitialized
  NaNs by zero still produced NaNs. Every accumulator element is now initialized before use.
* `fused_gr_down_kernel` launched eight warps but had only four injection outputs. Inactive warps wrote
  four zeros past the output and corrupted the previous injection gates. They now return before storing.
  The norm kernel also uses a device-memory barrier for cross-thread reads of its normalized values.
  The new `metal_fused_gr_test` checks against an independent double reference, including adjacent guards;
  comparing two implementations sharing the same bug had not caught this in the old tests.
* The MTP graph previously read a device-written expert pointer table while recording, before the router
  ran. It could capture stale or null expert bindings. Metal MTP now binds its immutable arena and reads
  expert indices on the GPU. The fixed Chinese short-answer fixture accepts 23/30 drafts; before this
  repair the same request accepted 15/36. This is one fixture, not a general acceptance-rate claim.

### Changes to the hot paths

* `--expert-cache auto` on the contiguous canonical Metal arena borrows the already registered weight
  allocation. All 24,576 experts are resident without allocating another 31.64 GiB copy. It is immutable:
  cache fills cannot overwrite weights, prompt scratch is separate, and adaptive eviction is disabled.
* Token and verify graphs with that arena perform routing, expert evaluation and combination on the GPU.
  No per-layer CPU expert call, flag wait, pointer readback or command-buffer round trip is needed.
  CPU-participating paths retain the segmented implementation and its no-live-spin rule.
* Adjacent kernels share a serial compute encoder. Blits and MPS close it at their boundaries. Graph launch
  still records into the open command buffer; it does not eagerly commit.
* AArch64 Q2 experts use NEON unpacking and integer dot products, reusing unpacked weights across tokens
  while preserving scalar floating-point accumulation order. A separate scalar reference checks
  1..8 tokens, unaligned rows, subnormal/negative scales and output guards, bit for bit.
* Prefill FP16/BF16 GEMM uses SIMD-group matrices, FP32 accumulation and partial tiles. BF16 is expanded in
  threadgroup memory, avoiding MPS's full FP32 weight copy. The MPS path remains available for comparisons.
* Small resident prompt chunks use direct Q8 expert kernels. Chunks of at least 32 tokens group routed
  rows into 16-row tiles and decode Q2 weights inside the matrix multiply. Two projection dispatches
  replace per-expert dequantization and many small GEMMs. Tile results exactly match unpacked FP16 GEMM
  in the multi-expert test, including partial tiles, offset rows and output guards.

### Measurements

| Workload | Before / comparison | Optimized | Evidence |
| --- | --- | --- | --- |
| 15-token Chinese chat prompt, generate 56, greedy decode | Old executable, canonical dense + CPU experts: 2.31 tok/s | Native dense + resident Metal experts: 18.72 tok/s | `baseline-real-chat.log`, `fgr-fixed-prefill-chat.log`; both decoded texts saved |
| Same chat, 14 prompt tokens processed before the first decode token | 2.28 tok/s sequential | 12.12 tok/s batched, including cold setup | Same logs; TTFT 6.574 s -> 1.214 s, model loading excluded |
| 387-token prefill, chunk 256 | Resident direct Q8 expert GEMV: 48.54 tok/s | Q2 matrix tiles: 84.52 tok/s | `direct-prefill-long.log`, `q2-tiled-prefill-long.log` |
| 387-token prefill, automatic chunk (one chunk) | — | 79.90 tok/s | `q2-tiled-prefill-auto.log`; one sample, not evidence that a larger chunk always wins |
| 971-token HTTP prompt, original spec-4 configuration | Original program crashed on first request | 124.5 tok/s prefill, 18.2 tok/s decode over 103 outputs | `server-final-long.json` |
| Same 971-token HTTP prompt, `--spec 2 --mtp-max-t 1 --suffix-draft 0` | Repaired speculative configuration above | 128.0 tok/s prefill, 18.7 tok/s decode | `server-single-long.json`; exact same response text |
| Warm 25-token HTTP request, generate 40 | Repaired speculation: 16.7 tok/s decode | No drafts: 19.8 tok/s decode | `server-final-repeat.json`, `server-single-repeat.json`; exact same response text |
| HTTP follow-up to that prompt | — | Reused 964 tokens, read 136 fresh tokens at 62.1 tok/s | `server-final-followup.json` |
| CPU expert microbenchmark, one token | 1048.4 microseconds | 307.9 microseconds | `baseline-cpu-expert.log`, `neon-cpu-expert.log`; same Q2 arithmetic |

The first row is an end-to-end configuration comparison, not a bitwise A/B: native projections and GPU
experts have different rounding from the canonical CPU path. The 387-token and 971-token fixtures differ;
their rates should not be used as a speedup ratio. Cold first-request pipeline compilation is visible in
HTTP short-prompt timings. The ordinary token graph measured approximately 19-21 tok/s over the other
saved runs. No general model-quality or 32K-context performance claim follows from these fixtures.

The local `strata-mac-optimized.json` keeps the model paths and 32K capacity, with `--spec 2` as verifier
capacity and `--mtp-max-t 1 --suffix-draft 0` selecting single-token decode. The engine now skips draft
generation when neither MTP windows nor suffix lookup can consume it. Original and optimized configs
gave exactly the same text on all four HTTP requests (40, 103, 85, 40 output tokens). The optimized
configuration's decode measured 18.7-21.1 tok/s; small-prompt prefill can be slower with the smaller
verifier window, so this is a measured tradeoff. MTP files remain required by the existing serve interface.

The complete Metal suite passed **46/46** (`ctest-final.log`, 146.85 s). It includes graph/event lifetime,
all existing kernel parities, the new GEMM/GR/NEON references and immutable shared-cache checks. The real
HTTP sequence checks Chinese output, long-prompt fact retrieval, context reuse and a repeated request.
An additional resident-graph regression changes routed expert IDs after capture and after source-graph
destruction, then compares each replay against the independent double reference (`resident-graph-parity.log`).

### Comparison switches

* `STRATA_METAL_NO_SHARED_EXPERTS=1`: keep the previous separately allocated expert-cache policy.
* `STRATA_METAL_LAYER_SYNC=1`: keep the segmented token/verify route even with shared weights.
* `STRATA_METAL_ENCODER_PER_KERNEL=1`: close the compute encoder after every kernel.
* `STRATA_METAL_MPS_GEMM=1`: use the earlier MPS dense GEMM implementation.
* `STRATA_METAL_PREFILL_Q2_GEMV=1`: use the direct Q8 expert path for all prompt chunk sizes.
* `STRATA_METAL_PREFILL_F16_EXPERTS=1`: use the earlier whole-expert dequantize + FP16 GEMM path.
* `STRATA_CPU_SCALAR=1`: keep scalar CPU Q2 experts on AArch64.

Do not benchmark with the layer/half dump hooks enabled: they synchronize for every dump and disable
fusion. Some intermediate logs in the results directory contain invalid output from debugging; only
the files named in the measurement table are valid final comparisons.

## 2026-10-03 (round 17): 100K throughput and cached follow-up

Same M2 Max / 96 GB and executable as round 16. The benchmark-only configuration raises capacity to
131,072, retains FP16 KV, uses automatic 1,024-token prefill chunks and disables speculation with
`--spec 2 --mtp-max-t 1 --suffix-draft 0`. The daily-use 32K config was not changed. Full evidence and
the saved input are in `bench/results/2026-10-03-metal-100k/`.

The first HTTP request contains exactly **100,000 tokenizer tokens including the chat template**, with
no prefix hit. It reads those tokens in **960.978 s (104.1 tok/s)**, streams its first text at **961.498 s**,
and generates 256 tokens in **13.766 s (18.6 tok/s)**. Total HTTP time is 975.218 s; model load is excluded.
The subsequent conversation has 100,290 input tokens, reuses **100,255**, and reads only **35** in
**2.590 s**. First text arrives at **3.081 s**; 128 output tokens take 7.674 s (16.7 tok/s).

**The quality check failed.** Both responses repeat "RAM" and retrieve none of three planted fields
near the middle of the first input. This is a throughput measurement, not a successful 100K quality
validation. The cause is not yet isolated; do not attribute it to either the model or a particular
Metal kernel without a reference comparison.

`vmmap` reports an engine physical-footprint peak of **41.3G**. The approximately 9.98 GiB process RSS
omits much of the Metal allocation and must not be used as total memory. System swap usage was already
5,047.94 MiB at the first sample and ended at 5,007.94 MiB; no sample exceeded its initial value. The
test service was stopped after both requests and the final memory measurement.

## 2026-10-03 (round 18): MTP on/off at 10K

Same M2 Max / 96 GB, model and executable. Three fresh server processes each receive the same frozen
**10,000-token prompt** and generate **256 tokens** at temperature 0 with thinking disabled. All use
32,768 capacity, FP16 KV, full-resident experts, automatic 1,024-token prefill and
`--spec 4 --spec-min-p 0.5 --suffix-draft 0`. Only `--mtp-max-t` and the log path differ. No request
reuses a prefix; model loading is excluded. Evidence: `bench/results/2026-10-03-metal-10k-mtp/`.

| MTP maximum window | Prefill tok/s | HTTP first text | Decode tok/s | Change vs disabled | Drafts accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1: disabled | 107.6 | 93.039 s | 19.00 | baseline | 0/0 |
| 2: at most one draft | 107.1 | 93.466 s | 16.88 | -11.2% | 90/119 (75.6%) |
| 4: at most three drafts | 105.5 | 94.891 s | 14.17 | -25.4% | 128/201 (63.7%) |

The timing log explains the loss: disabled decode spends 52.17 ms per one-token verification; MTP
window 2 averages 83.58 ms verification plus 7.30 ms drafting for 1.54 emitted tokens per round;
window 4 averages 122.80 ms verification plus 15.81 ms drafting for 1.97 tokens per round. Fewer
rounds do not offset the higher cost. Keep speculation disabled in the daily-use config for this
tested workload; these are one-request measurements, not a general claim about MTP.

The saved comparison script validates exact input equality, configuration differences, zero cache
hits, 256 output tokens, nonzero drafts in both MTP modes, and the unchanged engine binary hash.
**All three responses are identical, but all three fail the planted-field retrieval check.** They
do not have the 100K response's long RAM repetition; nevertheless, matching outputs only establish
agreement among these modes, not long-context quality. This failure also exists without MTP.
All three test services were stopped; no inference code or daily-use configuration changed in this round.

## 2026-10-03 (round 19): IQ2_XS migration, MTP disabled, and dequantization fixes

The user requested IQ2_XS in place of Q2_0, a repeat of the 10K benchmark with MTP disabled, and
deletion of the old model. Evidence, input, output, logs and the migration manifest are in
`bench/results/2026-10-03-metal-iq2-xs/`. The machine is unchanged: M2 Max, 38 GPU cores, 96 GB,
macOS 26.5. This round does not retest 100K or enable MTP.

### Model and configuration

Both IQ2_XS shards are SHA-256 verified against
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The second shard is byte-identical to Q2_0's PLE shard
and was preserved with the MTP runtime before deleting the old directory. `.venv/bin/python
tools/iq_pack.py` builds the 1.43 GiB native pack; expert weights retain their GGUF quantization
and need no separate experts.bin.

The current model and `strata-mac-optimized.json` are under
`/Users/liangxw/src/xproject/petproject/IQ2_XS/`. The config uses 32,768 capacity, FP16 KV,
1,024-token prefill, `--spec 4 --mtp-max-t 1 --suffix-draft 0`, and
`--mmap-experts --no-prefill-borrow`. All 24,576 experts fit in the 33.02 GiB GPU cache. The mmap
source avoids a second pinned host copy. MTP runtime files are retained because serve still
requires them; the measured draft count is zero.

### Correctness fixes

* The first IQ request failed at prefill: `dequant_f16` / `dequant_f32` in
  `src/kernels/metal/dequant_bf16.mm` still rejected IQ types 16/17/18/21/22/29 with a K19
  placeholder error even though the kernels existed. They now dispatch to IQ kernels and apply
  the source row offset using `iq_row_bytes`. The failed request is saved separately and excluded
  from timing results.
* Expanded `iq_parity` uses independent gguf-py / Q2_0-codec references to check the public FP16
  and FP32 row-slice entries for all 10 formats, including nonzero row offset and output guards.
  This exposed a separate error in `dequant_bf16.metal`: Q3_K reconstruction omitted the last
  packed scale byte (`sc[11] << 24`). Restoring it reduces 2,681 slice mismatches to zero.
  Existing matrix-vector checks over 1..8 columns also pass. `strata` and `iq_parity` rebuilt;
  focused CTest passes 1/1 in 1.40 s. The full 46-test suite was not rerun in this round.

Q2_0 contains Q3_K tensors in dense weights, while IQ2_XS does not. The old Q2_0 shard itself was
SHA-256 verified before the control run, ruling out a damaged download in this comparison.
The corrected Q2_0 run retrieves 3/3 fields where round 18 had retrieved 0/3. Earlier quality
failures therefore cannot be attributed entirely to Q2_0 quantization. Round 17's 100K results
still require a new run on the corrected engine; the 10K recovery does not validate 100K.

### Measurements

Two fresh server processes receive exactly the same frozen 10,000-token prompt, including its
chat template, and generate 256 tokens at temperature 0 with thinking disabled. Token IDs match
the previous 10K test exactly. Both use the corrected binary, no prefix hit and no drafts. Model
loading is excluded; first-request pipeline initialization is included. Each configuration is
measured once. Q2_0 uses its canonical shared expert arena, while IQ2_XS uses the native IQ path.

| Model, MTP disabled | Prefill tok/s | HTTP first text | Decode tok/s | HTTP total | Retrieved fields |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_XS | 44.7 | 223.932 s | 7.6 | 257.508 s | 3/3 |
| Q2_0, corrected control | 111.9 | 89.448 s | 19.2 | 102.710 s | 3/3 |

IQ2_XS is slower on the current implementation: its native path still schedules each layer on
the host and lacks Q2's whole-graph execution and fused expert prefill. Verification takes
131.34 ms/token, including 67.05 ms waiting for GPU reach and 58.37 ms in the host scheduling
phase. The host phase includes GPU execution/waits; it is not all CPU computation. CPU expert
work and drafting are both zero, and decode expert-cache hit rate is 100%. The comparison does
not isolate the cost or general quality of the quantization formats themselves.

IQ2_XS engine physical footprint peaks at 40.2G in `vmmap`; RSS does not include all Metal memory.
System swap was already 4,775.88 MiB at the first sample and ended at 4,759.88 MiB; no sample
exceeded the initial value. Both test services were stopped after completion.

### Removal and retained evidence

After successful IQ2_XS inference, verification of all new configuration dependencies, and the
corrected Q2_0 control, `/Users/liangxw/src/xproject/petproject/Q2_0/` was deleted as requested.
The old directory contained about 105.85 GiB; about 27.59 GiB of shared PLE and MTP runtime files
were preserved for IQ2_XS first. About 78.26 GiB of single-link files were removed. Actual free
disk space also depends on APFS and other processes. `migration.json` records the file manifest,
deletion time and disk-space readings. New config paths were checked again after removal.
Historical Q2_0 logs and scripts remain as evidence, but rerunning them requires downloading
the old model again. Current IQ2_XS data occupies about 65.56 GiB.

## 2026-10-03 (round 20): native IQ scheduling, asynchronous prefill and fused tiles

The user's constraint is performance without reducing model capability. The IQ2_XS GGUF weights,
quantization, full layer/expert computation, FP16 KV, and disabled MTP remain as in round 19.
Evidence is in `bench/results/2026-10-03-metal-iq2-xs-opt/` on the same M2 Max / 96 GB machine.

### Changes and performance

* Native experts can bind the complete immutable arena and read routes, residency and 64-bit slot
  offsets on the GPU. Gate/up and down each take one batched dispatch instead of one per expert.
  With every expert resident and prefill borrowing disabled, the verifier captures the entire window;
  partial caches still require host-planned segments. Output ordering stays token-major, then route.
* IQ data-producing wrappers no longer synchronize after every quantization/dequantization. A
  synchronous legacy `cudaMemcpy` now drains blocking streams before copying, while nonblocking
  streams require explicit synchronization/dependencies. The runtime test covers both cases.
* Native prefill uses 16-by-32 matrix tiles: the existing IQ dequantizers write a 32-value slice
  directly into transposed threadgroup storage. Matrix operands stay FP16, accumulators FP32,
  and the GEMM accumulation order stays the same. A 640-wide down projection only reads its valid
  slices. The first fused version used 21 KiB per group; removing the 256-value intermediate tile
  reduces this to 5 KiB.

Frozen 10,000-token prompt, 256 generated tokens, temperature 0, no prefix reuse and no drafts.
One fresh process/request per configuration; model load excluded, first-request pipeline setup included.

| Stage | Prefill tok/s | Decode tok/s | HTTP first text | Total request |
| --- | ---: | ---: | ---: | ---: |
| Round 19 baseline | 44.7 | 7.6 | 223.932 s | 257.508 s |
| Native resident graph | 43.4 | 18.8 | 230.485 s | 244.093 s |
| Async IQ wrappers | 71.3 | 18.1 | 140.420 s | 154.483 s |
| Fused prefill, 21 KiB | 75.5 | 18.1 | 132.559 s | 146.628 s |
| Fused prefill, 5 KiB | 120.3 | 18.0 | 83.244 s | 97.416 s |

The final measured speedups are 2.69x prefill and 2.37x decode, with 62.2% less total request time.
Small decode differences between the intermediate single runs are not separate conclusions.
Cached follow-up reuses 10,255 tokens and adds 35: first text 2.335 s, decode 18.2 tok/s, the expected
AX-7319 returned. Three short/repeated smoke requests also pass. `vmmap` physical footprint peaks
at 40.3G; sampled swap does not exceed its initial 4,679.88 MiB.

### Numerical and regression gates

Kernel tests compare captured resident experts with the existing grouped path, including changing
routes/residency at replay, mixed slot sizes, missing/repeated experts, graph destruction and actual
2560-by-640 model shapes. Fused prefill tests compare against complete FP16 dequantization and
ordinary GEMM across 8 gate/up formats and 3 down formats, partial tiles, row offsets and guards.
These checks are bitwise equal. The full suite before the extra model audit reports 46 CTest
entries with no failures in 152.98 s; three x86-only checks skip on ARM.

The four optimization-stage responses match one another but differ from the historical round-19
response. On the user's explicit quality reminder, native whole-window decode was temporarily made
opt-in for an audit. Keeping segmented decode with the new prefill produces the same 256 tokens.
Re-running the saved **round-19 binary with its original SHA-256** also produces exactly those same
256 tokens. The historical wording difference therefore does not isolate an optimization effect;
its original cause has not been established, and retrieval 3/3 alone is not a quality proof.

Opt-in diagnostics now compare resident/grouped experts on the same real-model activations, and
save complete logits with position/input/output IDs for matched-input comparisons. They are disabled
in performance runs. Four short fixed-input cases (arithmetic, Python, Chinese retrieval and English)
cover 290 teacher-forced positions: all 72,012,800 logits and the committed recurrent/KV/PLE state
fingerprints match between the segmented reference and resident candidate. The first three windows'
48-layer expert audit compares 14,745,600 values with zero differences and zero non-finite values.
The short requests generate one token each: these are numerical-equivalence fixtures, not task scores.

The 10K audit then compares the original full-FP16 expert prefill plus segmented decode against fused
prefill plus resident decode. Both read 10,000 tokens, generate 64, then reuse 10,063 tokens and read
20 fresh tokens for a cached follow-up (10 output tokens including EOS; AX-7319 is correct). All 99
recorded positions / 24,583,680 logits are bitwise equal, and both committed states match. Together
with the short cases this is 389 positions / 96,596,480 logits with no differences. Real expert
checks across both audits total 288 layer checks / 23,347,200 values, no differences or non-finite
values. The original round-19 executable also passes a separate short-case comparison: all four
output IDs and all four committed-state fingerprints match the current reference.

The native resident path is default again after these gates; `STRATA_METAL_IQ_RESIDENT=0` or
`STRATA_METAL_LAYER_SYNC=1` restores segmentation, and `STRATA_METAL_PREFILL_F16_EXPERTS=1` restores
full expert dequantization before GEMM. Independent synthetic resident checks have also been extended
to every supported format (11 gate/up, 5 down), with finite outputs required as well as bitwise equality.
The model-audit scripts record the binary hash and configurations; diagnostic timings are excluded
from performance results. These measurements establish equality on the covered inputs and settings,
not universal task accuracy or 100K quality. 100K and MTP-enabled IQ2_XS remain untested here.

Final rebuild and full regression after the audit hooks and all-format resident tests: **46 CTest
entries, zero failures, 126.93 s**, with the same three x86-only checks skipped on ARM. The resident
test now runs 67 fixtures / 201 changing-route graph replays, all bitwise equal and finite. See
`ctest-quality.log` and `quality-summary.json`. All audit engines and HTTP test services are stopped.

## 2026-10-03 (round 21): bounded lossless dense views and fewer copy passes

The user explicitly requires useful memory/time tradeoffs for repeatedly expensive work, with no
loss of model capability. Retained two changes: lossless dense IQ4 decode views and short/2D compute
copies. The model, quantization, layers, routed experts, FP16 KV and sampling remain unchanged.
Evidence and all intermediate experiments: `bench/results/2026-10-03-metal-decode-opt/`.

### Measured benefit and rejected memory

On M2 Max / 96 GB, IQ2_XS, frozen 10,000-token prompt, 256 recomputed output tokens, MTP off:

| Same-binary comparison | Median warm decode, ms/256 | tok/s | Extra view memory |
| --- | ---: | ---: | ---: |
| Both optimizations off (`final-control`) | 13,700.4 | 18.69 | 0 |
| Dense views and compute copies (`final-repeat`) | 13,110.9 | 19.53 | 2.244 GiB |

Three warm requests reuse the same 9,993 prompt tokens on both sides, but recompute all outputs.
The improvement is **4.5%**, saving **0.59 s per 256 tokens / 2.30 ms per token**. A separate matched-
binary cache-only comparison holds compute copies enabled: **18.84 -> 19.66 tok/s (+4.4%)**.
These are full decode timings, not multiplied microbenchmark estimates. Cold decode varies from
17.69 to 19.52 tok/s, so no stable cold-request gain is claimed. Prefill is around 127-130 tok/s;
this round establishes no stable prefill improvement. All 256 output IDs match the original reference.

The cache covers 179 dense projection matrices used every token; the inventory records all names,
shapes and bytes. It allocates exactly **2,409,031,680 bytes**, builds in about **0.12 s**, and keeps
all 24,576 experts resident (33.02 GiB, 100% decode hits). Matched final runs' before/after swap
is unchanged at 2,754.75 MiB. Child RSS is recorded but is not unified-memory physical footprint.

The extra 625.2 MiB output-head view did not improve full-model warm decode clearly and was removed.
FP16 integer-code views doubled space and slowed large matrices. GR SIMD changes fell to about
18.93 tok/s; scalar-gate fusion changed throughput only about 0.1%. These and unhelpful warp/integer
dot variants were removed. Their failed measurements remain in the evidence directory.

### Implementation and numerical gate

Each cached IQ4_XS block keeps the original eight scale/header bytes and expands the 256 codebook
values to signed int8, for 264 bytes. This is an exact representation, not Q8 requantization. It
calls the original integer-dot/scale/float-accumulation implementation with the same row/warp layout.
Original 136-byte blocks remain for prefill and multi-token calls. The loader releases immutable
views before original weights; shape mismatches and budget/allocation failures retain the original
path. Independent PLE-key kernels get no unused MMVQ view. The cache is explicitly enabled only in
the measured daily IQ2_XS config (`STRATA_METAL_IQ4_EXPAND=1`), with a cap of min(4 GiB, GPU budget/16)
and a free-budget check. Generic builds default to no view allocation.

Short registered-buffer copies (<=64 KiB) use exact uint4/byte compute copies in the serial encoder;
2D registered-buffer copies become one dispatch. Host staging and deferred GraphLaunch submission
are preserved. The experimental per-entry command-buffer profiler was removed: committing during
replay violates the mapped-source late-write contract. Its historical CSV is not a valid throughput
or correctness measurement. CPU-only graph encoding timing is safe and remains optional. Metal
header dependencies are now included in the build so `.metalh` edits actually rebuild the library.

Final CTest: **49 entries, zero failures, 148.22 s**; three x86-only checks internally skip on ARM.
Compute/blit copy tests each cover 471 cases including padding, alignment, source-graph destruction,
late mapped writes and compute/copy dependencies. IQ4 tests check byte layout, original-shader
outputs, captured single/multi calls and release/fallback: 27,702 values, zero bit differences,
non-finite values or representation/lifetime errors.

The final build is compared to saved round-20 traces on identical inputs: four short fixtures cover
290 positions / 72,012,800 logits; 10K prefill, 64 decoded tokens and a KV-cached follow-up cover
99 positions / 24,583,680 logits. **All 96,596,480 logits and all six committed state fingerprints
match bit for bit.** Model settings and output IDs match. Follow-up reuses 10,063 tokens, reads 20
new ones and returns AX-7319. These are numerical-equivalence checks, not general task accuracy or
100K/MTP-enabled validation.

Final binary SHA-256: `0c65709c5d8dd410578d2f2378a8258b7e2dbdfe2bbef43f719bf2b57ddbb86c`.
After removing temporary diagnostics, the daily-config `release/` check repeats warm decode at
**19.670 / 19.687 / 19.645 tok/s** (median **19.67**), cold decode 19.52 and prefill 129.67.
Its binary matches both numerical audits. Daily config is updated; all private test engines are stopped.

## 2026-10-03 (round 22): measured first, then exact direct-codebook decode kernels

Same M2 Max / 96 GB, IQ2_XS, MTP off, frozen 10K prompt, 256 recomputed outputs, runs one engine at a time.
Evidence: `bench/results/2026-10-03-metal-decode-opt2/` (README in Chinese, every log, scripts, microbenchmarks).

### Where the time went

* `STRATA_METAL_CB_TIMING=1` (new, attaches GPU start/end handlers, changes no encoding): the GPU is busy about
  96% of decode wall time; CPU encoding of the 1,743-entry token tape is 1-2.5 ms. The kernels are the cost.
* `STRATA_METAL_KPROFILE=<csv>` (new): each replayed tape entry in its own compute/blit pass with stage-boundary
  timestamps, inside the command buffer the stream already builds - no earlier commit, so the mapped-source
  late-write contract holds. This GPU samples only at stage boundaries and caps a sample buffer at 4,096 samples
  (`micro/counter-probe.log`). Each pass adds 2-3 us; the numbers are relative, not throughput.
* Weight reads ran at 70-100 GB/s. The IQ4 kernels were ALU-bound on the emulated `__byte_perm` codebook lookup
  (about 150 ops per word); the emulated `dp4a` byte loop itself compiles well - `char4` and `float4` rewrites of
  it were slower.

### Retained (each behind a switch whose 0 restores the original kernel)

| Change | Switch | Isolated, M2 Max (`micro/`) |
| --- | --- | --- |
| IQ4_XS / IQ4_NL single column: threadgroup float codebook, each code FMA'd against its activation byte, original 136/18-byte weights | `STRATA_METAL_IQ4_DIRECT` | head 3.9 -> 2.3 ms; dense 1.6-1.9x |
| Q2_0 resident down: direct dot, 4 rows per warp share the converted activation | `STRATA_METAL_EXPERT_DIRECT` | 110 -> 73 us |
| IQ3_S single column: direct dot, 2 rows per warp | same | about 1.1x |
| Hyper-connection norm in one pass (registers, no second device pass) | `STRATA_METAL_GR_NORM1` | 18.5 -> 9.5 us, 96 per token |
| Tape replay reads its debug switches once instead of `getenv` per entry | - | CPU only |

Exactness: every product and partial sum of a call's integer dot is an integer below 2^24 in magnitude, so the
float sum is exact in any order and `(int)` of it is the original int32; the integer scale step, the float
expression, the thread mapping, the per-lane order and the butterfly are unchanged.

### Measured result (same binary, switches)

| Run | Extra memory | Warm decode median | ms/token |
| --- | ---: | ---: | ---: |
| Original kernels + round 21's IQ4 view | 2,297.4 MiB | 18.83-18.90 tok/s | 53.07 |
| Direct kernels, no view | 0 | 21.78-21.99 tok/s (+15.5-16.6%) | 45.84 |
| Direct kernels + view | 2,297.4 MiB | 19.76 tok/s | 50.50 |

The view now slows decode (it takes precedence for its matrices), so the daily config sets
`STRATA_METAL_IQ4_EXPAND=0` and saves 2.24 GiB. Prefill 115-118 tok/s, no prefill claim. All 256 output ids match
the round-21 reference; swap 2,722.75 MiB before and after every run.

Rejected, logs kept: constant / register-select / 64-bit SWAR lookups (10-20%), `char4`/`float4` dp4a (slower),
IQ2_S / IQ2_XXS gate/up direct dot, more rows per warp, threadgroup codebook or activations, smaller threadgroups
(no stable gain; a byte-load-only floor is already 63-84 us of the 112-172 us kernel), GR down/up `device` address
space, more rows per warp and prefetch (they already run near 250 GB/s).

### Gates

New `metal_direct_dot_test` compares each direct kernel to the original by name and through the public entry
points: random, extreme (integer sums at their bound) and inf / NaN / -0 / subnormal scales, 0 bit differences;
breaking each new kernel on purpose fails it. CTest 50 entries pass. Real model (`audit.sh`, round 20's scripts,
no view): 389 positions / 96,596,480 logits bitwise equal, all 6 committed states equal, AX-7319 returned.

## 2026-10-03 (round 23): wide IQ4_XS rows, GDN state write-back, parallel top-k

Evidence: `bench/results/2026-10-03-metal-decode-opt3/`. Binary SHA-256 `1a5300ad...d578d7b31`, kept as
`build-metal/strata-round23`.

| Change | Switch | Isolated |
| --- | --- | --- |
| IQ4_XS one simdgroup per row, 4 rows (`native_iq4_xs_direct_sg4`), used for n_out >= 2560 | `STRATA_METAL_IQ4_SG` | head 2.3 -> 1.9 ms; slower at n_out 640, so not used there |
| GDN recurrence: the last token's state is not staged in `shadow`; a commit writes it straight to `state` | `STRATA_METAL_GDN_TAIL` | half the state traffic |
| QSA block top-k with parallel scans (`block_topk_scan_kernel`) | `STRATA_METAL_TOPK_SCAN` | 10K: 53-59 -> 27-30 us; 32K: 106-110 -> 65-68 us |

Same-binary warm decode: round 22's kernels 21.75 / 21.85 tok/s, everything on 22.46 / 22.23 tok/s
(**+1.9-3.0%**, 45.76 -> 44.65 ms/token), no added memory, all output ids equal to the reference. The top-k test
sweeps contexts 2,052-32,767 so the equal-key cut falls in every thread's range: two injected single-thread
faults passed the first version of the test and are caught by the sweep. `metal_direct_dot_test` 20,540,177
outputs, 0 differences; CTest 50 pass; real-model audit again 96,596,480 logits / 6 states bitwise equal.
Round 21's control (18.83-18.90) was measured in round 22's session; the cross-session total is about +18%.

## 2026-10-03 (round 24): expert gate/up access patterns - nothing kept

Evidence: `bench/results/2026-10-03-metal-expert-gu/`. Breaking the resident IQ2_S gate/up down on the same dispatch:
setup 7.7 us, weight bytes alone 34-44 us, + activations 81 us, + grid 66 us, all loads 107 us, the kernel 164-175 us.
About thirty exact layouts (calls unrolled, activations in registers or transposed threadgroup memory, weight
staging, a 2 KB packed grid, other group sizes, word-major activations, overlap with a shared-expert MMVQ) were all
slower or flat; the kernel loses 20-40% with any extra register or threadgroup memory. Two rows per warp was 6-12%
faster isolated but flat in the full model (22.70 vs 22.69 tok/s over three alternating passes) and was removed.

## 2026-10-03 (round 25): prefill - larger GEMM tiles, register attention, wider expert tiles

Evidence: `bench/results/2026-10-03-metal-prefill-opt/`. `STRATA_METAL_KPROFILE` now also samples direct (uncaptured)
launches, so the prompt path can be profiled per kernel. Before: FP16 GEMM 31% at about 2.5 TFLOPS (a 16 x 32
tile per threadgroup), prompt attention 21%, BF16 GEMM 14%, expert tiles 25%.

| Change | Switch (`=0` restores) | Isolated |
| --- | --- | --- |
| Dense GEMM in 64 x 64 tiles, vector loads, next k-chunk prefetched (`pfl_gemm2_f16` / `_bf16`) | `STRATA_METAL_PREFILL_GEMM2` | FP16 2.4-2.7 -> 6.0-7.7 TFLOPS, BF16 1.5-2.3 -> 4.5-5.7 |
| Prompt attention: running sums in registers (26.5 -> 14.3 KB), the chunk's exps over all threads | `STRATA_METAL_PROMPT_ATTN_REG` | 113-136 -> 42-51 ms per launch |
| Expert tiles of 16 or 32 rows x 64 columns (32 when >= 40 rows per expert) | `STRATA_METAL_PREFILL_MOE2` | 12-18% at ~20 rows per expert, ~21% at ~78 |

Every output element keeps its MMA sequence (same operand types and values, 8-wide k steps ascending from zero,
the same beta epilogue); the attention max and sum stay one thread per head over the 32 cells in order.
Same-binary first-request prefill of the frozen 10K prompt: 80.58 / 81.60 s (124.1 / 122.6 tok/s) -> 51.89 /
51.11 s (192.7 / 195.7 tok/s), **+57%**, output ids equal to the reference, decode unchanged (23.2-23.4 tok/s).
`metal_direct_dot_test` 51,078,679 outputs bitwise (each new kernel broken on purpose is caught), CTest 50 pass,
real-model audit 389 positions / 96,596,480 logits / 6 states bitwise (the 10K case runs the batched prefill).
Rejected: prefill chunks 2,048 / 4,096 (same ids, 144.3 / 138.1 tok/s), an early router commit to overlap the
shared expert with host grouping (194.9 vs 204.7 tok/s), a float4-packed attention butterfly (register-bound).
Binary `build-metal/strata-round25`, SHA-256 `69aea4c61ebbce6ac0ce5a28a1dde2e54d8fae9ac1f827c49e6c1277f3e378a8`.

## 2026-10-03 (round 26): the GDN commit no longer repeats the recurrence

Evidence: `bench/results/2026-10-03-metal-gdn-inplace/`. A decode window verifies one token, and that token is
always committed (1 <= n_keep <= T). Verify ran the 36 GDN recurrences without writing the state; the commit ran
them again from the same state and inputs only to write it. Now, on Metal with every expert resident and one
stage, the one-token window is recorded with the round-23 tail kernel and a device constant n_keep = 1, so it
writes the state the commit would have computed; the commit then replays a second graph that is the commit
without the GDN recurrence (conv history, indexer appends and PLE history unchanged). A window that wrote its
state back can only be followed by that graph (anything else is an error, never a second advance). T > 1, the
segmented path and the other backends are unchanged; `STRATA_METAL_GDN_INPLACE=0` restores the old commit.

Commit graph 134 -> 98 entries, GPU 2.43 -> 0.71 ms per token (KPROFILE); verify's recurrence 1.31 -> 1.28 ms (the
extra state write costs nothing measurable). Warm decode, median of 13 runs per side, same binary: 21.78 -> 22.55
tok/s (+3.5%), with run-to-run noise of 20.2-22.8 in this session (a background `dasd` at ~88% CPU). Every
256-token output equals the reference; real-model audit 389 positions / 96,596,480 logits / 6 committed states
(GDN included) bitwise, AX-7319 returned; CTest 50 pass. Binary `build-metal/strata-round26`, SHA-256
`02bf0dfec002de459972fbeac2440d2394206fa0f4f0c621be6c84f8f62336aa`.

## 2026-10-04: the README numbers again, in Energy Mode Automatic

The user said the measurements so far were taken in Low Power Mode, and switched the Mac (plugged in) to Energy Mode
Automatic (`pmset -g`: `powermode 0`). Same binary (`engine/strata`, SHA-256 `71cb4981…`), same configs and inputs,
one engine at a time:

| | Low Power Mode | Automatic |
| --- | ---: | ---: |
| IQ2_XS short-chat decode (median of 4) | 23.93 tok/s | 28.60 tok/s (+19.5%) |
| IQ2_XS 30,000-token prefill | 201.8 tok/s | 276.8 tok/s (+37.2%) |
| Q2_0 short-chat decode | 22.68 tok/s | 26.68 tok/s (+17.6%) |
| Q2_0 30,000-token prefill | 205.3 tok/s | 276.5 tok/s (+34.7%) |
| IQ2_XS / Q2_0 128,000-token prefill | 141.1 / 142.6 tok/s | 202.6 / 204.4 tok/s (+43.6 / +43.4%) |

The 30K runs' output is the same in both modes; the GPU memory left is the same. MTP drafts stay off: in
Automatic, `--mtp-max-t 2` decodes 20.86 tok/s against 28.60 (-27.1%, drafts 76-80% accepted, the same output). The 128K prompt had changed
between the two runs (the script read the working tree's docs, edited in between), so it is now pinned to commit
`dc047b5`. Evidence: `bench/results/2026-10-04-metal-models/`, `-128k/`, `-vision/` (the `*.low-power.json` files
are the earlier runs). The absolute tok/s in rounds 9-26 above are Low Power Mode numbers; each round compared
its two sides in the same mode.

High Power Mode (`powermode 2`) afterwards, the same binary and inputs: IQ2_XS short-chat decode 27.98 / 28.93
tok/s (two runs; Automatic 28.60), 30K prefill 271.7 (276.8); Q2_0 27.03 (26.68), 275.0 (276.5); 128K prefill
193.7 / 200.5 (202.6 / 204.4); the image encoder within 4%, the picture checks all pass. No measurable gain. The
output is the same, and so is the memory. Chrome, WindowServer and Spotlight were running in the background.
`--mtp-max-t 2` is still slower there: 21.57 tok/s. The README keeps the Automatic numbers.
