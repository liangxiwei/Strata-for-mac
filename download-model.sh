#!/bin/sh
# Strata on a Mac: download the model files only (~68 GB of GGUFs from Hugging Face + the ~5 GB MTP layer), resumable.
# ./setup.sh then finds them, builds the engine, writes the config and starts it - it downloads nothing again.
#
#   ./download-model.sh                          into Strata-data/ inside this folder (git-ignored)
#   ./download-model.sh --data-dir /Volumes/X    all model files on another disk (remembered)
#   ./download-model.sh --gguf-dir /path/IQ2_XS  the GGUFs in that folder (already there: kept; remembered)
#   HF_ENDPOINT=https://hf-mirror.com ./download-model.sh     through a mirror
#
# Other options are setup's (--model, --family; ./setup.sh --help).
exec "$(dirname "$0")/setup.sh" --download-only "$@"
