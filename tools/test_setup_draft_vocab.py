"""Tests for setup.py's draft-vocabulary choice (#287): a setup run again without --draft-vocab keeps the subset the
model's config chose before (cyrillic, en), and refresh_draft_vocab copies the chosen shipped subset.  Pure file work
in a temporary folder - no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_draft_vocab
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class SavedChoice(unittest.TestCase):
    def test_a_config_keeps_its_choice(self):
        with tempfile.TemporaryDirectory() as d:
            for choice in setup.DRAFT_VOCABS:
                p = Path(d) / "strata-x.json"
                p.write_text(json.dumps({"args": [], "draft_vocab": choice}), encoding="utf-8")
                self.assertEqual(setup.saved_draft_vocab(p), choice)

    def test_no_choice_without_a_valid_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.saved_draft_vocab(Path(d) / "missing.json"))
            for text in ("{}", "[1]", "not json", json.dumps({"draft_vocab": "klingon"})):
                p = Path(d) / "strata-y.json"
                p.write_text(text, encoding="utf-8")
                self.assertIsNone(setup.saved_draft_vocab(p), text)


class Refresh(unittest.TestCase):
    def test_metal_drafts_get_a_subset_from_the_config_folder(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "refresh_draft_vocab") as refresh:
            p = Path(d) / "strata.json"
            cfg = {"backend": "metal", "cwd": "model", "draft_vocab": "en",
                   "args": ["--mtp", "rt", "--spec", "4", "--mtp-max-t", "4"]}
            setup.refresh_config_draft_vocab(p, cfg)
            refresh.assert_called_once_with(Path(d) / "model/rt", "en")
            # Zero means the --spec cap; omitting --mtp-max-t has the same meaning.
            for args in (["--mtp", "rt", "--spec", "4", "--mtp-max-t", "0"],
                         ["--mtp", "rt", "--spec", "4"]):
                refresh.reset_mock()
                setup.refresh_config_draft_vocab(p, {**cfg, "args": args})
                refresh.assert_called_once()

    def test_metal_without_drafts_keeps_its_memory_profile(self):
        with mock.patch.object(setup, "refresh_draft_vocab") as refresh:
            for args in (["--mtp", "rt", "--spec", "4", "--mtp-max-t", "1"],
                         ["--mtp", "rt", "--spec", "1", "--mtp-max-t", "4"],
                         ["--mtp", "rt"], ["--spec", "4"]):
                setup.refresh_config_draft_vocab(Path("strata.json"), {"backend": "metal", "args": args})
            refresh.assert_not_called()

    def test_custom_subset_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            custom = b"custom vocabulary"
            (rt / "draft_vocab.bin").write_bytes(custom)
            setup.refresh_draft_vocab(rt)
            self.assertEqual((rt / "draft_vocab.bin").read_bytes(), custom)

    def test_cyrillic_replaces_a_shipped_subset_and_stays(self):
        if not (ROOT / "data" / setup.DRAFT_VOCABS["cyrillic"]).exists():
            self.skipTest("data/draft_vocab_cyrillic.bin is not in this checkout")
        want = hashlib.sha256((ROOT / "data" / setup.DRAFT_VOCABS["cyrillic"]).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            (rt / "draft_vocab.bin").write_bytes((ROOT / "data" / setup.DRAFT_VOCABS["cjk"]).read_bytes())
            setup.refresh_draft_vocab(rt, "cyrillic")
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)
            setup.refresh_draft_vocab(rt, "cyrillic")                       # again: nothing changes
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)


if __name__ == "__main__":
    unittest.main()
