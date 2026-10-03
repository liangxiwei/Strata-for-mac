<h1 align="center">Strata for Mac</h1>

<p align="center"><b>Run a 125-billion-parameter AI model on your own Mac</b><br>
Apple Silicon, on the Mac's own GPU (Metal) · also NVIDIA or AMD PCs (Windows or Linux) · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on a normal computer. It chats, writes code and works with your apps and coding agents,
and nothing leaves your machine.

This is a fork of [Strata](https://github.com/Niko1221/Strata) with a **Metal backend**: the whole engine runs on an
Apple Silicon Mac's GPU. Every kernel is checked against an independent reference, and every speed-up after that
keeps the real model's output bit for bit (96,596,480 logits compared, all equal: [the Metal port](docs/PORT_METAL/HANDOFF.md)).
The Metal code sits behind its own build switch; the CUDA (NVIDIA) and HIP (AMD) engines for PCs are the original
project's, and the PC numbers below are its measurements.

## How fast is it?

Measured on an **Apple M2 Max (38-core GPU, 96 GB), macOS 26.5**. "Writes answers" is how fast the reply appears in
a short chat; "reads your prompt" is how fast it takes in what you send (here a 30,000-token document). A token is
about ¾ of a word.

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **IQ2_XS** | 24 tokens/s | 196 tokens/s |

Writing stays at about 22-23 tokens/s with a 10,000-token conversation in context. A 30,000-token document takes
about 2.5 minutes to read the first time; follow-ups in the same chat reuse what was read (after a 10,000-token
conversation the next answer started in 2.3 seconds). How these were measured, and every optimization behind them:
[bench/results/2026-10-03-metal-readme](bench/results/2026-10-03-metal-readme/README.md),
[docs/PORT_METAL/PROGRESS.md](docs/PORT_METAL/PROGRESS.md). On a PC with an NVIDIA or AMD card the same model is
faster (an RTX 5070 writes 79 tokens/s and reads 2,090): [speed of each model on a PC](docs/MODELS.md#how-fast-is-each-size).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your computer, a coffee keeps the work on it going.</sub></p>

## What you need

**On a Mac:**

| | |
| --- | --- |
| **Mac** | Apple Silicon. Measured on an M2 Max with 96 GB: all 24,576 experts stay in the Mac's memory (33 GiB) and the engine uses about 40 GB in total. Macs with less memory are untested |
| **Disk** | about 80 GB free while you prepare the model; the finished IQ2_XS files (in `Strata-data/`) are about 70 GB |
| **System** | macOS with **Xcode** (the app, not only the command-line tools; setup downloads its Metal Toolchain if missing) and Python 3.10+ (`setup.sh` installs it with Homebrew if there is none; CMake comes with setup's Python packages). Measured on macOS 26.5, Xcode 26.1 |

So far the Mac runs the **IQ2_XS** size with 32K tokens of context, measured with prompts up to 30,000 tokens.
100K context, the "guess, then check" helper (MTP) and pictures are not validated on a Mac yet.

**On a PC:**

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series - with **12 GB of VRAM or more** |
| **RAM** | 32 GB or more - how much decides [which model](#which-model-should-i-pick) fits; 64 GB runs every size |
| **Disk** | about 80 GB free, on an SSD if you can (the first start is much faster) |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD |

On a PC everything else is installed for you. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).
The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### On a Mac

Install **Xcode** from the App Store and open it once (the engine's GPU kernels are compiled with its Metal
compiler). Then:

```sh
git clone https://github.com/liangxiwei/Strata-for-mac.git Strata && cd Strata
./download-model.sh
./setup.sh
```

**`./download-model.sh`** downloads the model files only: the IQ2_XS GGUFs (68 GB from Hugging Face) and the MTP
layer (~5 GB of the original Qwen checkpoint, which the server loads). It can be stopped and continues where it left
off, and a file that is already there is never downloaded again. If huggingface.co is slow or blocked where you
are, use a mirror: `HF_ENDPOINT=https://hf-mirror.com ./download-model.sh`.

The files go into **`Strata-data/`** inside the Strata folder. That folder is git-ignored, and the config names it
by relative paths, so you can move the whole Strata folder. To use another place:

- `./download-model.sh --data-dir /Volumes/Disk/Strata-data` puts all model files on another disk.
- `./download-model.sh --gguf-dir /path/to/IQ2_XS` uses a folder that already holds the GGUFs: present files are
  kept, missing ones are downloaded there.

`./setup.sh` remembers either choice, so the flag is not needed again.

**`./setup.sh`** then does the rest. Its only question is whether to download a missing Metal Toolchain:

- checks the Mac (Apple Silicon, memory, Xcode's Metal compiler);
- installs its Python packages into `.venv`;
- compiles the Metal engine (25 s on an M2 Max);
- prepares the model (4 s);
- writes `strata-iq2_xs.json` with the measured settings from [data/mac-metal.json](data/mac-metal.json): 32K
  context, every expert in the Mac's memory, MTP off;
- starts it.

The model loads in 30 seconds to about 1.5 minutes, then your browser opens the Strata app at
`http://127.0.0.1:8080`. Ctrl+C stops it. **Next time** run `./setup.sh` (or `./run-iq2_xs.sh`): it starts right
away. Without `./download-model.sh` first, `./setup.sh` downloads the model itself.

This install is checked against the measured one: the same output text and speed
([bench/results/2026-10-03-metal-setup](bench/results/2026-10-03-metal-setup/README.md)). Everything under
[Using it](#using-it) works on a Mac too, except pictures and the GPU readings in the Monitor. Building by hand,
tests and the port's switches: [docs/MAC.md](docs/MAC.md).

### Let your AI set it up

Use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this computer for me: https://github.com/liangxiwei/Strata-for-mac - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk, picks the model that fits, installs it, starts it and tells you how to
connect your apps. AI tools can also install, start and stop Strata themselves through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself (PC)

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

The same steps for NVIDIA and AMD: the installer finds your card and sets up the right engine for it. It asks which
model, which size, how much context (how much text it keeps in mind) and whether it should read pictures - press
Enter each time for the recommended answer. Then it downloads the model (~70 GB; you can stop and it continues where
it left off) and starts it. Your browser opens the Strata app at `http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again: it starts right away, nothing is downloaded twice. Close
its window to stop the model. Updating, Docker, several cards, where the files go and every option:
[docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

**On a Mac: IQ2_XS**, the only size measured there so far. On a PC the installer recommends one for your RAM. The
same model comes in sizes that are compressed more or less: smaller is faster, larger is a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best, and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's 4-bit (experimental) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder)** - a coding version with half of the experts removed: 91% of the full model's
  SWE-bench Verified score (by its authors), fits 32 GB of RAM. Weaker outside coding.
- **[Swift 1.5](docs/MODELS.md#swift-15)** - a fine-tune that thinks much shorter before it answers, so you get the
  answer sooner, at about the same quality.
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental) - the closest to the full
  model, but most of it is read from the SSD while it answers: 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** - a manual setup, not in the
  installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). You can add another model later with
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** `http://127.0.0.1:8080` - **Chat**, a live **Monitor** of the model and your GPU/CPU/RAM, and
  **About** with the settings and addresses. (On a Mac the Monitor shows the model, CPU and RAM; the GPU readings
  come from NVIDIA's and AMD's drivers and stay empty.)
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with base URL **`http://127.0.0.1:8080/v1`**,
  any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
- **Thinking:** choose **off, low, medium or high** in the chat menu or your app's "reasoning effort". Off is
  fastest; high is best for hard questions.
- **Pictures:** say yes to "Images?" in setup, then click **Picture** in the chat, or attach them in your app
  (AMD cards: on Linux through the processor, not on Windows yet; not validated on a Mac yet).
- **From your phone or another computer:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>` (on a Mac:
  `./setup.sh --host 0.0.0.0 --api-key <secret>`) - always with a key.
- **Good to know:** it answers one request at a time. The first message of a chat is read in full (on the M2 Max
  about 2.5 minutes per 30,000 tokens, on an RTX 5070 PC about 1 minute); follow-ups start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time Strata started.** Normal while it loads the model: wait, don't close the window.
  Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again: it continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or "the engine stopped unexpectedly".** Not enough free RAM:
  close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running - look for its window.
- **Mac: setup says Xcode is not installed, or there is no Metal compiler.** The command-line tools alone have no
  Metal compiler: install Xcode from the App Store and open it once. Since Xcode 26 the compiler is a separate
  download: answer `y` when setup offers it, or run `xcodebuild -downloadComponent MetalToolchain`. Then run
  `./setup.sh` again.
- **Mac: the engine stopped.** Its log is `strata-iq2_xs.log` in the Strata folder.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck on a PC? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. The
original project does not cover the Metal backend; for that, see [docs/PORT_METAL/](docs/PORT_METAL/HANDOFF.md).

## How does it work?

Models like this one normally run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes it fit by **sharing the work across your whole PC** - like a kitchen, where the things you use
all the time stay on the counter and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are asked most often; **your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.
- **On a Mac** the graphics card and the processor share one memory, so all 24,576 experts stay where the GPU reads
  them, and the GPU computes every word on its own.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words and the big model checks them all at once, so
  you get the same answer, 1.6-1.8x sooner on a PC. (On the M2 Max an earlier measurement, with the Q2_0 size and
  an engine build that has a bug fixed since, found the checking cost more than it saved, so it is off there for
  now; it is not re-measured: [round 18](docs/PORT_METAL/PROGRESS.md).)
- **Long texts are read in big pieces:** up to 8,192 tokens at a time and over 1,000 tokens per second on a PC;
  1,024 at a time and about 196 per second on the M2 Max.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers: [the
details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth;
Strata is built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). This Mac version is a fork
of [Niko1221/Strata](https://github.com/Niko1221/Strata) and adds the Metal backend. All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE); a few
parts and every model carry their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
