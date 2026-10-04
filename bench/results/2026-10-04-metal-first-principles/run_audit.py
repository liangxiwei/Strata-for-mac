#!/usr/bin/env python3
"""Use the established real-model audit with the current model config and frozen 10K input."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
base = ROOT / 'bench/results/2026-10-03-metal-iq2-xs-opt'
source = (base / 'audit_model.py').read_text()
source = source.replace("(Path(__file__).parent / 'tiled/server-config.json')", "(ROOT / 'strata-iq2_xs.json')")
source = source.replace("(Path(__file__).parent/'tiled/prompt-tokens.txt')", "(ROOT / 'bench/results/2026-10-03-metal-iq2-xs-opt/tiled/prompt-tokens.txt')")
exec(compile(source, str(base / 'audit_model.py'), 'exec'))
