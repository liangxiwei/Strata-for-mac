# The Metal port - the plan

Porting Strata's engine to Apple Silicon, so the model runs on a Mac's own GPU. This page is the architecture
and the module list; [STATUS.md](STATUS.md) is the live state of every module; [PROGRESS.md](PROGRESS.md) is
the dated log with its measurements. **Update those two files as work lands** - they, not anyone's memory, are
the record.

The port follows the repo's own rule: every ported kernel must pass the parity test that already exists for it
(synthetic data, no model), and every number written down here says what it was measured on. What cannot be
measured on this machine is marked **estimate**.

## Why it is possible (and what it costs)

The engine is CUDA-shaped, but the seam is narrow and clean:

* **The host code never launches a kernel directly.** Every kernel sits behind a launcher declared in
  `include/strata/kernels/*.hpp` (`void scale_inplace(float*, int64_t, float, void* stream)` and 213 more entry
  points); the `__global__` bodies and `<<<>>>` calls live in `src/kernels/cuda/*.cu` (44 files, ~190 kernels,
  16.9k lines). The HIP backend already exploits this seam at the API level (a `cuda_runtime.h` compat header
  force-included into every unit). The Metal backend exploits it at the launcher level.
* **The host engine's CUDA surface is about 25 API functions**: allocation/copy, stream/event, graph capture
  and replay, device properties. `graph.cpp` (capture bookkeeping) and all 6.8k lines of engine `.cpp` call
  only those.
* **The only device code outside `src/kernels/cuda/` is a 6-line poison kernel** in `src/core/device.cu`.

What has no Metal equivalent is **CUDA graph replay** and **cuBLAS**:

* Graphs become a **tape**: `cudaStreamBeginCapture` starts recording, each launch appends
  `(pipeline, buffers, bytes, grid)` to a list, `cudaGraphLaunch` re-encodes the list into fresh command
  buffers. This is correct *because the engine's replays already rely only on persistent buffers whose
  contents change* (the doorbell protocol carries the dynamics; capture-time scalars are baked on CUDA too).
  It costs CPU re-encode time per token - measured once the engine runs; **estimate**: a few hundred kernels
  per token of encoder calls, well under a millisecond per token on an M2 Max.
* cuBLAS (the prompt GEMMs, `src/prefill/gemm.cu`) becomes **Metal Performance Shaders** matrix kernels in a
  first pass, with bf16 routed through fp16/fp32 where MPS has no bf16. Prompt speed will be well below an
  RTX 5070; decode speed is the goal. **Estimate, not measured**: decode on an M2 Max (400 GB/s unified
  memory, IQ2_XS weights ~40 GB, ~660 MB of experts touched per token plus dense/KV traffic) has a
  bandwidth-bound ceiling near 10 tokens/s; landing anywhere from 3 to that is success. Once it runs, the
  number gets measured and this line is replaced.

## The mapping

| CUDA / HIP | Metal backend |
| --- | --- |
| `cudaMalloc` / `cudaFree` | `MTLDevice newBufferWithLength` (shared storage; unified memory) |
| `cudaMallocHost` mapped / `cudaHostRegister(Mapped)` | plain `malloc` wrapped with `newBufferWithBytesNoCopy` (zero-copy on Apple Silicon) |
| `cudaMemcpy[Async]` | `memcpy` (CPU) or a blit compute pass on the stream, per direction |
| `cudaStream_t` | a serial chain of `MTLCommandBuffer`s on one `MTLCommandQueue` |
| `cudaEventRecord/Query` | a `MTLSharedEvent` / completed-flag on the command buffer |
| stream capture + `cudaGraphLaunch` | the tape, re-encoded per launch (see above) |
| `<<<grid, block, shared>>>` | launcher .mm: `dispatchThreadgroups` with the same geometry |
| warp shuffles, `__syncthreads()` | simdgroup ops, `threadgroup_barrier` |
| cuBLAS GEMM | MPS matrix multiplication |
| pinned-memory doorbells (kernel spins on host flag) | shared-buffer atomics - **validated early, M1**, before mass-porting |

## The modules

Work lands in this order; each has a row in STATUS.md.

| Module | What it is | Done means |
| --- | --- | --- |
| **M0 runtime base** | `STRATA_ENABLE_METAL` in CMake; `include/strata/metal_compat/cuda_runtime.h` (the API shim, force-included like HIP's); `src/platform/metal_runtime.mm` (device, buffers, copies, streams, events, errors) | `strata-device` builds and answers on a Mac; a buffer round-trip through the shim works |
| **M1 launch + doorbell proof** | MSL compiled into the build (`src/kernels/metal/*.metal` -> one metallib, embedded); pipeline cache; grid dispatch; **the CPU↔GPU shared-memory doorbell micro-test** (the risk item: a kernel spinning on a host-written flag) | a kernel launched through the shim returns correct data; the doorbell handshake round-trips |
| **M2 graph tape** | capture/replay inside the shim's stream objects; `graph.cpp` compiled unchanged | a captured sequence replays with changed buffer contents and picks them up (a selftest in the shim) |
| **M3 kernels K1..K44** | one module per `.cu` file, each = MSL bodies + launcher `.mm`s, in the same header contract; order: `elementwise` (the glue + doorbells) first, then the decode path (`dequant_s2`, `s2_gemv*`, `quantize_act`, `router_top10`, `shared_expert`, `rope`, `sampler`, `gr`, `gdn`, `qsa*`, `kv_*`, `fused_*`, `native_*`, `iq_kernels`, `verify_kernels`, `s2_expert_grouped`, ...), prompt path last | that file's existing parity test (`*_parity`) passes on the Mac |
| **M4 prefill GEMM** | `src/prefill` under Metal: MPS GEMM, bf16 via conversion first | `prefill`-related parity checks pass |
| **M5 engine link** | `strata_engine`, `strata_prefill`, `strata` (`generate.cpp`) compile and link under `-DSTRATA_ENABLE_METAL=ON`; `poison_kernel` and the one launch in device.cu get a Metal home | `strata --serve` starts, prints READY (weights absent: it stops at load, and that is what "done" means here) |
| **M6 end-to-end** | the model on a Mac: `setup.py` macOS path grows a real-engine install (pack preparation already works - the tools are portable Python) | a chat answered by the real model, tokens/s measured and written into PROGRESS.md. **Needs the ~68 GB download; deferred until M3-M5 are green.** |

## What is deliberately NOT ported

* Multi-GPU, CUDA-specific tuning (pcie-frac probing, copy-engine grouping), MMQ prompt kernels - later or never.
* The x86 AVX-512 CPU experts: the Mac's CPU side already runs the scalar paths (docs/MAC.md); NEON versions
  are an optimization, not a blocker.
* Windows/`_MSC_VER` paths of the shim: Apple-Silicon Macs only.

## Verification ladder

1. Per-kernel: the repo's parity tests, unmodified, on the Mac (synthetic data; this is what flips a STATUS row).
2. Per-module shims: selftests inside `metal_runtime` (tape replay, doorbell, buffer round-trip).
3. Engine: `strata-device --selftest` (writes real GPU memory), `strata --serve` startup sequence.
4. Model: M6, once the download is acceptable.

No step is marked done by "it compiles".
