# Strata on a Mac

On an Apple-Silicon Mac the model runs on the **Metal backend** (`-DSTRATA_ENABLE_METAL=ON`,
[docs/PORT_METAL/](PORT_METAL/HANDOFF.md)). Measured on an M2 Max (96 GB), plugged in, Energy Mode Automatic,
with a 30,000-token prompt:

| Model | Writes | Reads |
| --- | --- | --- |
| IQ2_XS | 29 tokens/s | 277 tokens/s |
| Q2_0 | 27 tokens/s | 276 tokens/s |

In Low Power Mode the same Mac writes 15-16% and reads 26-30% slower. High Power Mode measured no faster than
Automatic (within 5%)
([bench/results/2026-10-04-metal-models](../bench/results/2026-10-04-metal-models/README.md)).

It also reads pictures, encoded on the Mac's GPU.

To install it:

```sh
./download-model.sh     # the model files, into Strata-data/ (git-ignored)
./setup.sh              # every run: the model list; builds, writes the config, starts it
```

[What setup does on a Mac](#what-setup-does-on-a-mac) below says what each step does; the
[README](../README.md#install) gives the short version. The default build (without
`-DSTRATA_ENABLE_METAL=ON`) has no engine on a Mac. Its engine is CUDA (NVIDIA) or HIP (AMD), and its CPU expert
kernels are x86 intrinsics (AVX-512 VNNI/VBMI, AVX2).

A Mac is also a good machine to **work on everything around the engine**: the server (the OpenAI- and
Anthropic-compatible API and the web app), the GGUF tools, the planner, and the CPU-side kernels and their tests all
build and pass on Apple Silicon.

> **On this page:** [What works](#what-works) · [Set up](#set-up) · [Build and test the C++](#build-and-test-the-c-side)
> · [Run the model on the Metal backend](#run-the-model-on-the-metal-backend) · [The Python side](#the-python-side) ·
> [The server and web app](#the-server-and-web-app) · [What setup does on a Mac](#what-setup-does-on-a-mac) ·
> [How the build differs](#how-the-build-differs)

## What works

| | on a Mac (Apple Silicon) |
| --- | --- |
| The model, installed by `./setup.sh` | **yes** - the Metal engine with IQ2_XS or Q2_0, measured on an M2 Max with 96 GB, text and pictures ([what setup does](#what-setup-does-on-a-mac)) |
| The model, the default build's `strata --serve` | **no** - the default build needs an NVIDIA or AMD GPU ([why](#strata-on-a-mac)) |
| The server, the API, the web app | **yes** - against the mock engine (a scripted answer) for development, or the real model on the Metal backend |
| C++ build + `ctest` | **yes** - the CPU-side targets; the AVX2/AVX-512 kernels are compiled out and their scalar transcriptions stand in |
| `tools/test_setup_*.py`, `serve/test_*.py` | **yes** |
| The GGUF tools (`strata-gguf`, `strata-dequant`, `strata-plan`) | **yes** |

Everything on this page was measured on an **Apple M2 Max (12 cores, 96 GB RAM), macOS 26.5, Xcode 17's clang,
CMake 3.30 from Homebrew**; other Macs will differ in speed only.

## Set up

You need the Xcode command-line tools (for clang and make), CMake, and Python 3.10 or newer:

```sh
xcode-select --install            # if you have no Xcode / CLT yet
brew install cmake python@3.12    # any Python 3.10+ works
```

Then one private Python environment inside the folder (nothing is installed system-wide):

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Build and test the C++ side

```sh
cmake -S . -B build -DSTRATA_BUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build
```

Measured on the M2 Max above: the configure-and-build (including the one-time llama.cpp fetch for ggml, see
[below](#how-the-build-differs)) takes **49 s**, and `ctest` runs **12 tests in about 7 s, all passing** - the
GGUF reader and split view, the speculation components (suffix drafter, controller, draft policy), the
conversation-cache policy, the bf16 rounding table, the CPU expert kernels and the expert pool. The conversation
tests (a CMake option, not in `ctest`) run as:

```sh
cmake -S . -B build -DSTRATA_BUILD_CONVERSATION_TESTS=ON && cmake --build build --target conversation_cache_test conversation_memory_test
./build/conversation_cache_test && ./build/conversation_memory_test     # 4175 + 23 checks, under a second
```

## Run the model on the Metal backend

`./setup.sh` does all of this ([below](#what-setup-does-on-a-mac)). This section is for working on the port: a
build with the tests, and running it by hand.

The port is recorded in [docs/PORT_METAL/](PORT_METAL/HANDOFF.md). So far it is validated only with the IQ2_XS model
on an Apple M2 Max (38-core GPU, 96 GB):

- all 24,576 experts are resident in GPU memory (33.02 GiB);
- the engine's physical footprint peaks at 40.3G (`vmmap`, round 20);
- Macs with less memory are untested.

Run one engine at a time.

Build in its own directory (the compiler is Xcode's: the app, with its Metal Toolchain):

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DSTRATA_METAL_ENGINE=ON \
      -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-metal -j 8
ctest --test-dir build-metal --output-on-failure      # 50 entries, about 2.5 minutes
```

The model files come from `./download-model.sh` (or `./setup.sh`), in `Strata-data/`:

- the two IQ2_XS GGUF shards (68.03 GB);
- the native pack built by `tools/iq_pack.py` (1.43 GiB);
- the MTP runtime files, which the serve interface requires even with drafts disabled.

Sources and checksums: [PROGRESS round 19](PORT_METAL/PROGRESS.md).

The config `./setup.sh` writes, `strata-iq2_xs.json`, names the engine and model by paths relative to the Strata
folder. Its arguments are [data/mac-metal.json](../data/mac-metal.json)'s, the measured ones:

- 32K context, FP16 KV, 1,024-token prefill chunks;
- `--spec 4 --mtp-max-t 1 --suffix-draft 0`, which makes zero drafts in practice;
- `--mmap-experts --no-prefill-borrow`;
- `STRATA_METAL_IQ4_EXPAND=0`.

To run your own build with it, point `"exe"` at `build-metal/strata`, then:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-iq2_xs.json --port 8080 --open
```

`--open` opens the browser once the model is ready; `http://127.0.0.1:8080/v1/chat/completions` and `/v1/messages`
work directly. As everywhere, do not expose the server beyond `127.0.0.1` without `--api-key`.

Measured on the frozen 10,000-token prompt: prefill 192.7-195.7 tokens/s, warm decode median 22.55 tokens/s, outputs
bitwise equal to the reference. Per-round numbers, switches and numerical audits:
[PORT_METAL/PROGRESS.md](PORT_METAL/PROGRESS.md) and `bench/results/2026-10-03-metal-*/`. 100K context and MTP
enabled are not validated yet.

## The Python side

The server's and setup's own tests need no GPU and no downloads:

```sh
.venv/bin/python -m unittest discover -s serve -p "test_*.py"     # 146 tests, 73 s, OK (9 skipped: Windows-only)
for t in tools/test_setup_*.py; do .venv/bin/python "$t"; done    # all OK
```

## The server and web app

With the mock engine - a scripted answer, not the model - the whole app runs: the browser interface at
`http://127.0.0.1:8080`, the OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages`
endpoints, streaming, the Monitor tab. This is what you develop against when you change the server or the web
app; the [engine boundary](DETAILS.md#using-it) is the same line protocol either way.

```sh
.venv/bin/python -m serve.server --engine mock --port 8080
```

`--answer "some text"` scripts a different reply (given more than once, requests cycle through them). As
everywhere, do not expose the server beyond `127.0.0.1` without `--api-key`.

## What setup does on a Mac

`./setup.sh` runs `setup.py`'s `mac_main`, then `install_mac`. These are the PC's steps for a Mac; each is skipped
when it is already done.

1. **Checks the Mac:**
   - It needs Apple Silicon and a native arm64 Python. An Intel Mac, or a Python under Rosetta, stops with the fix.
   - Under 20 GB of memory (a 16 or 18 GB Mac) it stops: the smallest model needs about 30 GiB.
   - It looks for Xcode's Metal compiler in every Xcode in `/Applications` (through `DEVELOPER_DIR`, no sudo).
     - A missing Metal Toolchain is downloaded after asking.
     - Xcode itself must come from the App Store.
     - A license that has not been accepted is reported with the command that accepts it.
2. **The model list, every run.** It shows the models of [data/mac-metal.json](../data/mac-metal.json)'s `menu`,
   downloaded ones first. You pick with the arrow keys or a number; Enter takes it, q stops.
   - Each line's brackets say whether the model is downloaded, how it fits this Mac's memory, and its measured
     speed (or "untested on a Mac").
   - Fit is "recommended for this Mac", "fits", "tight" (part of the experts on the CPU: slower, untested) or "too
     little memory". The estimate is the model's experts, plus the 7.3 GiB the measured IQ2_XS run used beside
     them, plus the 1.3 GiB image encoder, against 81% of the memory: the share macOS gave the GPU on the 96 GB Mac,
     77.8 GiB.
   - The recommended model is the first measured one that fits.
   - Without a terminal, or with `--yes`, setup takes the model used last, else the recommended one, and prints the
     list. `--model` / `--family` choose without it.
   - An installed model starts right away. A model that is not downloaded is downloaded after asking. A model with
     too little memory is asked about first (with `--yes` alone it stops).
3. **The settings** are data/mac-metal.json's measured ones: 32K context, FP16 KV, images on.
   - `--context` and `--kv` are kept, with a note that they are not measured on a Mac. A setup again keeps the
     context, KV, port, host, API key and any key you added to the config by hand.
   - `--vision no` sets a model up without pictures. The speed projection, the GPU flags and the low-RAM mode are PC
     features.
4. **Installs** the Python packages into `.venv` (CMake and Ninja too) and gets llama.cpp (ggml, gguf-py).
5. **Compiles** two programs. On the M2 Max: 113 build steps, 25 s, plus about 25 s for the encoder.
   - The Metal engine, into `engine/strata`.
   - The image encoder (`tools/vision`, llama.cpp's mtmd with ggml's Metal backend), into `engine/strata-vision`.
   - `engine/BUILD.json` keeps a hash of both sources; after a `git pull` that changes one, the next start compiles
     it again.
6. **The model files**, into `Strata-data/models/<size>/`:
   - A file of a gigabyte or more comes over four connections at once: hf-mirror serves about 3 MB/s per
     connection, and four reached 14-44 MB/s here. A stopped download resumes from its `.part.plan`.
   - Every file is checked against the SHA-256 Hugging Face publishes; a wrong file is deleted. The result is kept
     in its `.done` mark, so a file is never checked or downloaded again.
   - The original model's shard 2 is the same file for every size. With a checked copy already here, it is linked
     (no space, no download): Q2_0 took only its 37.6 GB shard 1.
   - The image encoder's weights (0.9 GB) go to `Strata-data/models/`.
7. Writes the pack (seconds) and the MTP layer, then `strata-<size>.json` and `run-<size>.sh` (both relative to the
   folder), and starts the server.

Where the model files go:

- **`Strata-data/` inside the Strata folder** (git-ignored) by default. The config names it relatively, so the
  folder can move.
- `--data-dir DIR`: another place. Remembered for later runs.
- `--gguf-dir DIR`: GGUFs you already have. Remembered for this model while all shards are still there; in the
  config they are absolute paths.
- **`./download-model.sh`** is `./setup.sh --download-only`. It runs steps 1 (without the compiler check), 2, 4
  (gguf-py only) and 6, plus the MTP layer. It writes no engine and no config.

`--setup` sets the chosen model up again. `--check` only checks the Mac and prints the list.

Measured results:

- Setup's install gives the same output text and speed as the hand-built engine
  ([bench/results/2026-10-03-metal-setup](../bench/results/2026-10-03-metal-setup/README.md)).
- IQ2_XS and Q2_0: [bench/results/2026-10-04-metal-models](../bench/results/2026-10-04-metal-models/README.md).
- Pictures: [bench/results/2026-10-04-metal-vision](../bench/results/2026-10-04-metal-vision/README.md).
- 128K context: [bench/results/2026-10-04-metal-128k](../bench/results/2026-10-04-metal-128k/README.md).

## How the build differs

The default build rewrites nothing for the Mac; the x86-only parts are simply compiled out, and the portable code
that already existed beside them carries the same entry points:

- `src/kernels/cpu/expert.cpp` - the AVX-512/AVX-2 kernels live inside `#if defined(__x86_64__) || defined(_M_X64)`;
  off x86, the same file compiles the scalar transcription of the same contracts (it is the oracle the vector
  kernels are checked against on x86, so it exists and is tested on both). `cpu_features()` truthfully reports
  no AVX-512, and `q2_rows_any`/`act_quant_any` dispatch to the scalar paths.
- `q2_avx2.cpp`, `iq_avx512.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp` stay out of the build (CMake leaves the files and
  their `-mavx*` flags out on a non-x86 CPU); native experts fall back to ggml-cpu's own dot products, which are
  NEON on Apple Silicon.
- `src/kernels/cpu/pool.cpp` - the spin-wait hint is `yield` on ARM instead of `_mm_pause`; macOS has no
  thread-to-core pinning (the scheduler places the workers) and no sysfs topology, so every logical CPU counts
  as a worker core.
- The GPU libraries (`strata_core`, `strata_kernels`, `strata_engine`, `strata_prefill` and the `strata` program)
  are only defined with `-DSTRATA_ENABLE_CUDA=ON` or `-DSTRATA_ENABLE_HIP=ON` on a PC; on a Mac they exist only in
  a `-DSTRATA_ENABLE_METAL=ON` build, where the Metal backend provides them ([PORT_METAL/PLAN.md](PORT_METAL/PLAN.md)).

The default build fetches the pinned llama.cpp commit (as on Linux) for ggml's CPU dots; on this Mac it compiled
with `-mcpu=native+dotprod+i8mm` and Apple's Accelerate BLAS. `-DSTRATA_NATIVE_EXPERTS=OFF` skips that fetch if
you only want the Strata-side targets.
