<h1 align="center">Strata for Mac</h1>

<p align="center"><b>Run a 125-billion-parameter AI model on your own Mac</b><br>
Apple Silicon, on the Mac's own GPU (Metal) · text and pictures · free and open source</p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a Mac. It is a large, smart AI
model that normally needs a server. It chats, writes code, reads pictures and works with your apps and coding agents.
Nothing leaves your Mac.

This is a fork of [Strata](https://github.com/Niko1221/Strata), which runs on Windows and Linux PCs with an NVIDIA or
AMD card. The fork adds a **Metal backend**: the whole engine runs on an Apple Silicon Mac's GPU. Every kernel is
checked against an independent reference. Every speed-up after that keeps the real model's output bit for bit (the
check: 96,596,480 logits compared, all equal; [the Metal port](docs/PORT_METAL/HANDOFF.md)).

## How fast is it?

Measured on an **Apple M2 Max (38-core GPU, 96 GB), macOS 26.5**, plugged in, Energy Mode **Automatic**:

- **Writes answers** is how fast the reply appears in a short chat.
- **Reads your prompt** is how fast it takes in what you send (here a 30,000-token document).
- A token is about ¾ of a word.

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **IQ2_XS** | 29 tokens/s | 277 tokens/s |
| **Q2_0** | 27 tokens/s | 276 tokens/s |

**Long texts (128K context).** With `--context 131072` the model takes a 128,000-token prompt (about 96,000 words):

| Size | Reads 128,000 tokens | Writes after it | Follow-up in the same chat | Found the 3 facts planted in the middle |
| --- | ---: | ---: | ---: | :---: |
| **IQ2_XS** | 203 tokens/s (10.5 min) | 25.3 tokens/s | first word after 0.94 s | 3 of 3 |
| **Q2_0** | 204 tokens/s (10.5 min) | 24.5 tokens/s | first word after 0.88 s | 3 of 3 |

**Full 256K context (IQ3_S).** On the fixed 256,000-token text, IQ3_S read the whole prompt, answered the three
facts placed in it, then answered a follow-up in the same conversation:

| Size | Reads 256,000 tokens | Writes after it | Found the 3 facts planted in the text | Follow-up: reused / newly read | Follow-up first word | Peak physical footprint |
| --- | ---: | ---: | :---: | ---: | ---: | ---: |
| **IQ3_S** | 168.1 tokens/s (25.4 min) | 21.13 tokens/s | 3 of 3 | 256,025 / 24 tokens, 1.994 s | 6.69 s | 61.6G |

This is one fixed long-context run, not a general quality score. Its 26-token first answer makes the decode figure a
small sample. The exact inputs, engine settings, output and timings are in
[the 256K measurement](bench/results/2026-10-04-metal-256k/README.md).

- **Reading:** a long prompt is read once. The follow-up reused all 128,026 tokens already read and read only the
  24 new ones. Reading gets slower as the text grows: 277 tokens/s at 30,000 tokens, 203-204 at 128,000.
- **Writing:** 27-29 tokens/s in a short chat, about 25 after 128,000 tokens.
- **Context:** after choosing a model, setup offers the measured 128K and 256K contexts. It selects 256K when that
  model and context fit its memory estimate; otherwise it selects 128K and says why. On the measured M2 Max, 128K
  and 256K reserve 2,677MiB and 6,248MiB more shared memory than 32K. On the 256,000-token text test, IQ2_XS,
  Q2_0 and IQ3_S read 159.8, 164.5 and 168.1 tokens/s; all found the three planted facts. `./setup.sh --setup
  --context 131072` selects 128K explicitly. The 256K text-path footprint peaked at 47.0G, 45.4G and 61.6G.
- **Energy Mode** (System Settings → Battery), measured on the same Mac with the same engine and inputs:
  - **Low Power Mode** makes it slower. It wrote 15-16% slower (24 / 23 tokens/s) and read 26-30% slower (202-205
    tokens/s at 30,000 tokens, 141-143 at 128,000).
  - **High Power Mode** measured no faster than Automatic. It wrote 27-29 tokens/s and read 272-275 tokens/s at
    30,000 tokens and 194-201 at 128,000, all within 5% of Automatic. Two High Power runs of the same test differed
    by 3.4%.
  - The answers to the same inputs were word for word the same in all three modes.
- **Pictures** are encoded on the Mac's GPU in 0.3-1.2 s. Turning them on changes nothing in text answers: the same
  output, the same speed.
- **A small local code check (256K setup):** IQ2_XS and IQ3_S each produced a patch that passed the official
  SWE-bench Verified harness for `matplotlib__matplotlib-20488`. On the same setup, a fixed, deterministic 10-problem
  slice of LiveCodeBench v6 scored 5/10 and 6/10 single-answer pass@1. This slice is a local smoke test, not an
  overall SWE-bench or LiveCodeBench score; its method, prompts, time and limits are recorded with the result.

How these were measured, and every optimization behind them:

- [bench/results/2026-10-04-metal-models](bench/results/2026-10-04-metal-models/README.md) (IQ2_XS, Q2_0);
- [bench/results/2026-10-04-metal-128k](bench/results/2026-10-04-metal-128k/README.md) (128K context);
- [bench/results/2026-10-04-metal-256k](bench/results/2026-10-04-metal-256k/README.md) (256K context);
- [bench/results/2026-10-04-metal-code-evals](bench/results/2026-10-04-metal-code-evals/README.md) (256K local code checks);
- [bench/results/2026-10-04-metal-vision](bench/results/2026-10-04-metal-vision/README.md) (pictures);
- [docs/PORT_METAL/PROGRESS.md](docs/PORT_METAL/PROGRESS.md) (the optimizations).

## What you need

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 or newer). **16 GB is not supported.** The memory decides the model; setup's list says what fits: [which model](#which-model) |
| **Disk** | about 80 GB free while the model is prepared; the finished model is about 70 GB (in `Strata-data/` in the Strata folder) |
| **System** | macOS with **Xcode** (from the App Store; the command-line tools alone are not enough). Measured on macOS 26.5 with Xcode 26.1 |

Setup installs everything else: Python with Homebrew if there is none, the Python packages, CMake, Xcode's Metal
Toolchain (after asking), the engine and the image encoder.

## Install

Install **Xcode** from the App Store and open it once. Then, in Terminal:

```sh
git clone https://github.com/liangxiwei/Strata-for-mac.git Strata && cd Strata
./download-model.sh
./setup.sh
```

**Every run shows the supported models in a list.** Pick one with the arrow keys and press Enter. Downloaded models
come first. The brackets say for each one:

- whether it is downloaded;
- how it fits this Mac's memory: **recommended for this Mac**, fits, tight, or too little memory;
- its measured speed, or "untested on a Mac".

```text
  Which model? (96 GB Mac)   (up/down, Enter; q to stop)
> 1) Qwen3.8-Flash-Next IQ2_XS  - the original model, 2-bit, writes a little faster  (downloaded, set up; recommended for this Mac; measured: 29 tok/s)
  2) Qwen3.8-Flash-Next Q2_0    - the original model, 2-bit, a little smaller  (downloaded, set up; fits this Mac; measured: 27 tok/s)
  3) Qwen3.8-Flash-Next Coder IQ1_M   - the Coder: half the experts, for code  (not downloaded, 58 GB; fits this Mac; untested on a Mac)
  4) Swift 1.5 IQ2_XS  - a fine-tune that thinks shorter  (not downloaded, 68 GB; fits this Mac; untested on a Mac)
  ...
```

Setup follows the model menu with the context menu:

```text
  Context length?   (up/down, Enter; q to stop)
  1) 128K tokens  (measured on this Mac; adds 2.6 GiB beyond 32K; ~44 GiB, fits this Mac)
> 2) 256K tokens  (measured on this Mac; adds 6.1 GiB beyond 32K; ~48 GiB, fits this Mac)
```

**`./download-model.sh`** downloads the chosen model's files only:

- the GGUFs, 58-84 GB from Hugging Face, fetched over four connections at once;
- the MTP layer, about 5 GB of the original Qwen checkpoint, which the server loads;
- the image encoder, 0.9 GB.

Each file is checked against the SHA-256 Hugging Face publishes. A stopped download continues where it left off, and a
file that is already there is never downloaded again. If huggingface.co is slow or blocked where you are, use the
mirror: `HF_ENDPOINT=https://hf-mirror.com ./download-model.sh`.

The files go into **`Strata-data/`** in the Strata folder. It is git-ignored, and the config names it by relative
paths, so the whole folder can move. To keep the files somewhere else:

- `./download-model.sh --data-dir /Volumes/Disk/Strata-data` puts all model files on another disk.
- `./download-model.sh --gguf-dir /path/to/IQ2_XS` uses GGUFs you already have: files that are there are kept,
  missing ones are downloaded into that folder.

Setup remembers either choice.

**`./setup.sh`** does the rest:

1. Checks the Mac: Apple Silicon, memory, and Xcode's Metal compiler (it offers to download a missing Metal
   Toolchain).
2. Compiles the Metal engine and the image encoder (under a minute on an M2 Max).
3. Prepares the model (seconds).
4. Writes `strata-<model>.json` with the selected context and the measured engine settings
   ([data/mac-metal.json](data/mac-metal.json)).
5. Starts the model.

The model loads in 30 seconds to 2 minutes. Then your browser opens the Strata app at `http://127.0.0.1:8080`.
Ctrl+C stops it. **Next time,** `./setup.sh` shows both lists again. An installed configuration starts right away
when its selected context is unchanged; a different context rewrites its config. A model that is not downloaded yet
is downloaded first, after asking.

Without a terminal, or with `--yes`, setup takes the model used last, else the recommended one, then takes 256K if
it fits the model-and-memory estimate (128K otherwise). `--model Q2_0` (and `--family coder` / `swift`) chooses
without the model list; `--context N` chooses the context without its list.

### Let your AI set it up

Use an AI coding assistant (Claude Code, Cursor, Codex, ...)? Paste this into it:

```text
Set up Strata on this Mac for me: https://github.com/liangxiwei/Strata-for-mac - follow the Mac section of docs/AI_SETUP.md in that repository.
```

## Which model?

Setup's list recommends one for your Mac's memory. The estimate counts every expert in the share of memory macOS
gives the GPU (77.8 of 96 GiB on the measured Mac), plus the rest the measured run used and the image encoder.

| Model | Download | Memory it takes (estimate) | On a Mac |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next IQ2_XS** | 68 GB | ~42 GiB | **measured**: 29 tok/s writing, 277 reading, pictures |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | ~40 GiB | **measured**: 27 tok/s writing, 276 reading |
| Qwen3.8-Flash-Next Coder IQ1_M | 58 GB | ~30 GiB | untested; for code (91% of the full model's SWE-bench Verified, by its authors) |
| Swift 1.5 IQ2_XS | 68 GB | ~42 GiB | untested; a fine-tune that thinks shorter |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | ~49 GiB | untested; 3-bit |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | ~55 GiB at 32K; ~62 GiB at 256K | measured at 256K: 168 tok/s prefill, 21 tok/s decode; 3.5-bit, the closest to the full model |

By this estimate:

| Memory | What fits |
| --- | --- |
| 96 GB and more | every size |
| 64 GB | every size but IQ3_S, which is tight |
| 48 GB | the Coder; the 2-bit sizes are tight (part of the experts on the CPU, slower, untested) |
| 36 GB | the Coder is tight |
| 24-32 GB | too little memory for every size |
| 16-18 GB | not supported |

## Using it

- **In the browser:** `http://127.0.0.1:8080`. **Chat**, **Monitor** (the model, CPU and RAM) and **About** (the
  settings and addresses).
- **Your apps and coding agents:** add an "OpenAI-compatible" provider. Base URL **`http://127.0.0.1:8080/v1`**, any
  API key, any model name. For apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages`. For Claude Code:
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`.
- **Pictures:** click **Picture** in the chat, or attach them in your app (an `image_url` part, or Anthropic's
  `image` block). The Mac's GPU encodes a picture in 0.3-1.2 s.
  - Checked on this Mac: it read a receipt's shop name and total, named three shapes and their colours, counted
    seven dots, read Chinese text back exactly, described a screenshot, and answered a follow-up about a picture
    earlier in the chat ([bench/results/2026-10-04-metal-vision](bench/results/2026-10-04-metal-vision/README.md)).
  - `./setup.sh --setup --vision no` sets a model up without pictures.
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is
  fastest; high is best for hard questions.
- **From your phone or another computer:** `./setup.sh --host 0.0.0.0 --api-key <secret>`. Always use a key.
- **Good to know:** it answers one request at a time. The first message of a chat is read in full (about 2 minutes
  per 30,000 tokens on the M2 Max). Follow-ups start in seconds.

The API: [docs/DETAILS.md](docs/DETAILS.md#using-it).

## Something went wrong?

- **Setup says Xcode is not installed, or there is no Metal compiler.** The command-line tools alone have no Metal
  compiler: install Xcode from the App Store and open it once. Since Xcode 26 the compiler is a separate download:
  answer `y` when setup offers it, or run `xcodebuild -downloadComponent MetalToolchain`. Then run `./setup.sh`
  again.
- **The download is slow or stopped.** Run the same command again; it continues where it stopped. From China, use
  the mirror: `HF_ENDPOINT=https://hf-mirror.com ./download-model.sh`.
- **It is very slow, or macOS says it is out of memory.** Close other apps (browsers use a lot), or pick a model the
  list marks "fits" or "recommended".
- **It says port 8080 is already in use.** Strata is already running: look for its Terminal window.
- **The engine stopped.** Its log is `strata-<model>.log` in the Strata folder (for example `strata-iq2_xs.log`).

More, and the details of the Mac port: [docs/MAC.md](docs/MAC.md).

## How does it work?

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them.
- **A Mac's GPU and processor share one memory,** so all 24,576 experts stay where the GPU reads them. The GPU
  computes every word on its own, as one graph, with no trips back to the processor.
- **Long texts are read in pieces** of 1,024 tokens at a time, about 277 tokens per second on the M2 Max.
- **Pictures** go through the model's own vision encoder (llama.cpp's mtmd, on the GPU). The model reads them in
  place of image tokens in the chat.

The Metal port, kernel by kernel, with every measurement: [docs/PORT_METAL/](docs/PORT_METAL/HANDOFF.md). How the
engine works: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It is compressed
by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) and UkisAI (Swift 1.5).

[Strata](https://github.com/Niko1221/Strata) is by Niko1221; this Mac version is a fork that adds the Metal backend.
It is built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).

Strata is open source under the [MIT License](LICENSE). A few parts and every model carry their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. A coffee for its author keeps the work on it going.</sub></p>
