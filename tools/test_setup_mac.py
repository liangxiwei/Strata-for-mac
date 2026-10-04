"""setup.py on an Apple-Silicon Mac (install_mac): the run config it writes is the measured one (data/mac-metal.json,
checked against the arguments of bench/results/2026-10-03-metal-readme/summary.json), with paths relative to the
Strata folder; the memory, Apple-Silicon and Xcode checks; the Metal engine's build and rebuild; a start that never
asks nvidia-smi.  Every outside effect is mocked: no GPU, no Xcode, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_mac
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from serve import server  # noqa: E402

MEASURED = json.loads((ROOT / "bench/results/2026-10-03-metal-readme/summary.json").read_text(encoding="utf-8"))["config"]
SHARD = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{}-of-00002.gguf"


class FakeGGUF:
    """gguf_reader.GGUFFile: shard 2 holds the PLE table."""
    def __init__(self, path):
        names = ["per_layer_token_embd.weight"] if "00002-of" in str(path) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


def never(what):
    return mock.Mock(side_effect=AssertionError(f"{what} called on a Mac"))


def install(ram, argv, answers=None, extra=(), machine="arm64", tty=False, before=None):
    """setup.main() on a mocked Mac -> (exit code, printed text, the written config or None, questions asked, folder
    files).  answers: None = --yes, "" = Enter for every question, {words of a question: answer}.  tty: a terminal
    (the arrow-key list; its keys come from an arrow_menu patch).  before(folder): files to put there first."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        if before:
            before(t)
        data = t / "Strata-data"
        (data / "mtp" / "rt").mkdir(parents=True)
        (data / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        eng = t / "engine"
        eng.mkdir()
        (eng / "strata").write_bytes(b"")
        asked = []

        def fake_input(prompt=""):
            asked.append(prompt)
            if answers is None:
                raise AssertionError(f"asked {prompt!r} with --yes")
            if isinstance(answers, dict):
                return next((v for k, v in answers.items() if k in prompt), "")
            return answers

        def fake_download(url, dst, what=None):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        patches = [
            mock.patch.object(setup, "ROOT", t),
            mock.patch.object(setup, "MAC", True),
            mock.patch.object(setup.platform, "machine", lambda: machine),
            mock.patch.object(setup, "out", lambda cmd: "1" if machine == "rosetta" else ""),
            mock.patch.object(setup, "data_folder", lambda d: (data, [])),
            mock.patch.object(setup, "load_settings", lambda: {}),
            mock.patch.object(setup, "save_settings", lambda s: None),
            mock.patch.object(setup, "gpus", never("nvidia-smi")),
            mock.patch.object(setup, "amd_gpus", never("the AMD probe")),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Apple M2 Max", False, False)),
            mock.patch.object(setup, "free_gb", lambda p: 900.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: t / "third_party" / "llama.cpp"),
            mock.patch.object(setup, "metal_probe", lambda: (True, "Apple metal version 32023.830", {})),
            mock.patch.object(setup, "build_engine_metal", lambda yes, llama: eng),
            mock.patch.object(setup, "build_vision_metal", lambda llama: eng / "strata-vision"),
            mock.patch.object(setup, "verify_hf", lambda s, url: None),
            mock.patch.object(setup, "hf_sha256", lambda url: None),
            mock.patch.object(setup, "menu_tty", lambda: tty),
            mock.patch.object(setup, "update_installed_engine", lambda *a, **k: None),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "run", lambda *a, **k: None),
            mock.patch.object(setup, "mtp_corrupt", lambda *a, **k: False),
            mock.patch.object(setup, "refresh_draft_vocab", never("the draft subset copy")),
            mock.patch.object(setup, "calibrate_config", never("tuning")),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", *argv] + (["--yes"] if answers is None else [])),
            mock.patch("builtins.input", fake_input),
            *extra,
        ]
        out = io.StringIO()
        code = None
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            with contextlib.redirect_stdout(out):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = e.code
        written = sorted(t.glob("strata-*.json"))
        cfg = json.loads(written[-1].read_text(encoding="utf-8")) if written and code == 0 else None
        files = {p.name: p.read_text(encoding="utf-8") for p in t.glob("run-*.sh")}
        return code, out.getvalue(), cfg, asked, files


def measured_args() -> list:
    """The measured run's arguments with its paths replaced by the ones setup writes (relative to the Strata folder)."""
    paths = {"--pack": "Strata-data/packs/iq2_xs", "--native": "Strata-data/models/IQ2_XS/" + SHARD.format(1),
             "--ple-gguf": "Strata-data/models/IQ2_XS/" + SHARD.format(2),
             "--expert-profile": "data/expert-profile.bin", "--mtp": "Strata-data/mtp/rt"}
    a = list(MEASURED["args"])
    for flag, p in paths.items():
        a[a.index(flag) + 1] = p
    return a + VISION_ARGS


VISION_ARGS = ["--vision", "--vram-reserve-mib", "700"]   # images on (the default): the encoder's room, as on a PC


class MeasuredConfig(unittest.TestCase):
    def test_yes_writes_the_measured_config(self):
        code, out, cfg, asked, files = install(96.0, ["--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["args"], measured_args())
        for k, v in setup.mac_profile()["env"].items():
            self.assertEqual(cfg["env"][k], v)
            self.assertEqual(MEASURED["env"][k], v)                # the measured run had the same switch
        self.assertEqual((cfg["exe"], cfg["cwd"], cfg["tokenizer"], cfg["log"], cfg["backend"]),
                         ("engine/strata", ".", "Strata-data/packs/iq2_xs/tokenizer", "strata-iq2_xs.log", "metal"))
        self.assertNotIn("gpu", cfg)
        self.assertNotIn("lib_dirs", cfg)
        self.assertEqual(cfg["vision"], {"exe": "engine/strata-vision",
                                         "mmproj": "Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                                         "model": "Strata-data/models/IQ2_XS/" + SHARD.format(1), "gpu": True,
                                         "max_tokens": 1024})
        self.assertEqual(asked, [])
        script = files["run-iq2_xs.sh"]
        self.assertIn('cd "$(dirname "$0")"', script)
        self.assertIn('--config "strata-iq2_xs.json"', script)
        self.assertNotIn(tempfile.gettempdir(), script)

    def test_enter_for_every_question_is_the_same(self):
        code, out, cfg, asked, _ = install(96.0, ["--no-start"], answers="")
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["args"], measured_args())
        self.assertEqual(len(asked), 1)                   # the only question: the 68 GB download (Enter = yes)
        self.assertIn("download about 68 GB", asked[0])

    def test_no_to_the_download_stops_before_it(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start"], answers={"download about": "n"},
                                       extra=[mock.patch.object(setup, "download", never("a download"))])
        self.assertNotEqual(code, 0)
        self.assertIn("./download-model.sh downloads it alone", out)

    def test_images_off_on_request(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--vision", "no"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("vision", cfg)
        self.assertEqual(cfg["args"], measured_args()[:-len(VISION_ARGS)])

    def test_the_profile_is_the_measured_one(self):
        p = setup.mac_profile()
        self.assertEqual((p["family"], p["model"], p["context"]), ("qwen", "IQ2_XS", 32768))
        self.assertEqual(str(p["context"]), MEASURED["args"][MEASURED["args"].index("--max-context") + 1])
        for flag in ("--mmap-experts", "--no-prefill-borrow"):
            self.assertIn(flag, p["args"])
        self.assertNotIn("--kv", p["args"])                      # FP16 KV, the engine's default


class Memory(unittest.TestCase):
    def test_64gb_takes_the_measured_model(self):
        code, out, cfg, _, _ = install(64.0, ["--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["args"], measured_args())

    def test_48gb_is_recommended_the_coder(self):
        code, out, cfg, _, _ = install(48.0, ["--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", cfg["args"][cfg["args"].index("--native") + 1])
        self.assertIn("not measured on a Mac yet", out)

    def test_48gb_with_the_bigger_model_is_tight(self):
        code, out, cfg, _, _ = install(48.0, ["--no-start", "--model", "IQ2_XS"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("does not all fit the memory macOS gives the GPU", out)

    def test_too_little_memory_stops_with_yes_alone(self):
        code, out, cfg, _, _ = install(24.0, ["--no-start"])
        self.assertNotEqual(code, 0)
        self.assertIn("needs more memory than this Mac has", out)
        code, out, cfg, _, _ = install(24.0, ["--no-start", "--model", "IQ2_XS"])     # named: the user's risk
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("as you chose", out)

    def test_16gb_is_not_supported(self):
        for ram in (16.0, 18.0):
            code, out, cfg, _, _ = install(ram, ["--no-start", "--model", "IQ2_XS"])
            self.assertNotEqual(code, 0)
            self.assertIn("is too little", out)
            self.assertIn("24 GB of memory or more", out)


def never_shards(url, dst, what=None):
    """download() for a test where the model's GGUFs are there: the image encoder's file may come, a shard may not."""
    if dst.name.endswith(".gguf") and "-of-" in dst.name:
        raise AssertionError(f"downloaded {dst.name} again")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"")
    setup.mark(dst)


class TheMac(unittest.TestCase):
    def test_an_intel_mac_stops(self):
        code, out, _, _, _ = install(96.0, ["--no-start"], machine="x86_64")
        self.assertNotEqual(code, 0)
        self.assertIn("Intel Mac", out)

    def test_a_rosetta_python_stops_with_the_fix(self):
        code, out, _, _, _ = install(96.0, ["--no-start"], machine="rosetta")
        self.assertNotEqual(code, 0)
        self.assertIn("Rosetta", out)

    def test_check_only_checks(self):
        code, out, cfg, _, _ = install(96.0, ["--check"],
                                       extra=[mock.patch.object(setup, "build_engine_metal", never("the build"))])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("This Mac can run Strata", out)
        self.assertIsNone(cfg)

    def test_pc_flags_are_said_and_left_out(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--gpu", "1", "--low-ram", "on"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("--gpu is for a PC", out)
        self.assertIn("--low-ram / --kv-streaming are for a PC", out)
        self.assertNotIn("gpu", cfg)
        self.assertEqual(cfg["args"], measured_args())


class DownloadOnly(unittest.TestCase):
    def test_it_downloads_and_stops(self):
        got = []

        def fake_download(url, dst, what=None):
            got.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)
        code, out, cfg, _, files = install(96.0, ["--download-only"], extra=[
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "metal_probe", never("the Metal compiler check")),
            mock.patch.object(setup, "build_engine_metal", never("the build"))])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual([u.rsplit("/", 1)[1] for u in got],
                         [SHARD.format(1), SHARD.format(2), "mmproj-Qwen3.8-Flash-Next-BF16.gguf"])
        self.assertIsNone(cfg)                               # no config, no start script
        self.assertEqual(files, {})
        self.assertIn("Next:        ./setup.sh", out)

    def test_present_files_are_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as g:
            for i in (1, 2):
                (Path(g) / SHARD.format(i)).write_bytes(b"")
            code, out, cfg, _, _ = install(96.0, ["--download-only", "--gguf-dir", g], extra=[
                mock.patch.object(setup, "download", never_shards),
                mock.patch.object(setup, "whole_shard", lambda s: True)])
            self.assertEqual(code, 0, out[-3000:])
            self.assertIn("--gguf-dir " + str(Path(g).resolve()), out)

    def test_a_given_model_folder_is_remembered(self):
        saved = {}
        with tempfile.TemporaryDirectory() as g:
            for i in (1, 2):
                (Path(g) / SHARD.format(i)).write_bytes(b"")
            keep = [mock.patch.object(setup, "load_settings", lambda: dict(saved)),
                    mock.patch.object(setup, "save_settings", lambda s: saved.update(s)),
                    mock.patch.object(setup, "download", never_shards),
                    mock.patch.object(setup, "whole_shard", lambda s: True)]
            install(96.0, ["--download-only", "--gguf-dir", g], extra=keep)
            self.assertEqual(saved["gguf_dirs"]["IQ2_XS"], str(Path(g).resolve()))
            code, out, cfg, _, _ = install(96.0, ["--no-start"], extra=keep)     # no flag: the remembered folder
            self.assertEqual(code, 0, out[-3000:])
            self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1],
                             str(Path(g).resolve() / SHARD.format(1)))           # outside the folder: absolute

    def test_a_pc_is_told(self):
        with mock.patch.object(setup, "MAC", False), mock.patch.object(sys, "argv", ["setup.py", "--download-only"]), \
                contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(SystemExit):
            setup.main()
        self.assertIn("Mac's model download", err.getvalue())


class Choices(unittest.TestCase):
    def test_an_unmeasured_size_is_asked(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--model", "IQ3_S"], answers={"Go on anyway": "n"})
        self.assertNotEqual(code, 0)
        self.assertIn("not measured on a Mac", out)
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--model", "IQ3_S"])      # --yes + explicit: consent
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", cfg["args"][cfg["args"].index("--native") + 1])

    def test_the_unsloth_budget_mode_is_a_pc_mode(self):
        code, out, _, _, _ = install(96.0, ["--no-start", "--family", "unsloth"])
        self.assertNotEqual(code, 0)
        self.assertIn("not set up on a Mac", out)

    def test_context_and_kv_are_kept_as_chosen(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--context", "65536", "--kv", "int8"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("not validated on a Mac", out)
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "65536")
        self.assertEqual(a[a.index("--kv") + 1], "int8")

    def test_128k_is_measured(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--context", "131072"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("not validated on a Mac", out)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")

    def test_the_projection_stays_off(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--vision", "yes", "--experimental-speed-projection", "on"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("vision", cfg)
        self.assertNotIn("--control-vector-scaled", cfg["args"])
        self.assertEqual(cfg["args"], measured_args())


def mac_config(t: Path) -> Path:
    """A Mac run config in folder t, as install_mac writes it, with the files it names."""
    (t / "engine").mkdir(parents=True, exist_ok=True)
    (t / "engine" / "strata").write_bytes(b"")
    (t / "engine" / "BUILD.json").write_text(json.dumps({"source": "local", "backend": "metal", "src": "x"}))
    m = t / "Strata-data" / "models" / "IQ2_XS"
    m.mkdir(parents=True)
    for i in (1, 2):
        (m / SHARD.format(i)).write_bytes(b"x" * 10)
    args = [x.format(pack="Strata-data/packs/iq2_xs", gguf=f"Strata-data/models/IQ2_XS/{SHARD.format(1)}",
                     ple=f"Strata-data/models/IQ2_XS/{SHARD.format(2)}", context="32768", mtp="Strata-data/mtp/rt",
                     profile="data/expert-profile.bin") for x in setup.mac_profile()["args"]]
    cfg = {"exe": "engine/strata", "args": args, "cwd": ".", "tokenizer": "Strata-data/packs/iq2_xs/tokenizer",
           "model_name": "qwen3.8-flash-next-iq2_xs", "log": "strata-iq2_xs.log", "port": 8080, "backend": "metal",
           "env": {"STRATA_METAL_IQ4_EXPAND": "0"}}
    p = t / "strata-iq2_xs.json"
    p.write_text(json.dumps(cfg))
    return p


class Start(unittest.TestCase):
    def test_a_mac_start_asks_no_gpu_and_finds_its_files_from_anywhere(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as elsewhere:
            p = mac_config(Path(tmp))
            call = mock.Mock(return_value=0)
            cwd = os.getcwd()
            os.chdir(elsewhere)                             # relative paths are the config folder's, not the shell's
            try:
                with mock.patch.object(setup, "gpus", never("nvidia-smi")), \
                        mock.patch.object(setup, "amd_gpus", never("the AMD probe")), \
                        mock.patch.object(setup, "refresh_draft_vocab", never("the draft subset copy")), \
                        mock.patch.object(setup.subprocess, "call", call), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(setup.start(p, None, gpu=1), 0)
            finally:
                os.chdir(cwd)
            cmd = call.call_args[0][0]
            self.assertIn(str(p), cmd)
            self.assertNotIn("--gpu", cmd)
            self.assertIn("--open", cmd)
            self.assertIn("this Mac's GPU", out.getvalue())
            self.assertIn("a Mac has one GPU", out.getvalue())

    def test_a_missing_model_file_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = mac_config(Path(tmp))
            (Path(tmp) / "Strata-data/models/IQ2_XS" / SHARD.format(1)).unlink()
            with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                setup.start(p, None)
            self.assertIn(SHARD.format(1), out.getvalue())

    def test_the_server_reads_the_paths_relative_to_the_config(self):
        cfg = {"cwd": ".", "tokenizer": "Strata-data/packs/iq2_xs/tokenizer", "log": "strata-iq2_xs.log", "x": "y"}
        r = server.config_paths(cfg, Path("/S"))
        self.assertEqual((r["cwd"], r["tokenizer"], r["log"], r["x"]),
                         (os.path.normpath("/S"), os.path.normpath("/S/Strata-data/packs/iq2_xs/tokenizer"),
                          os.path.normpath("/S/strata-iq2_xs.log"), "y"))
        absolute = {"cwd": "/A", "tokenizer": "/B/tok", "log": "l.log"}
        self.assertEqual(server.config_paths(absolute, Path("/S")),
                         {"cwd": os.path.normpath("/A"), "tokenizer": "/B/tok", "log": os.path.normpath("/A/l.log")})
        self.assertEqual(server.config_paths({"tokenizer": "t"}, Path("/S")), {"tokenizer": "t"})   # no cwd: as before


class Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.root = mock.patch.object(setup, "ROOT", self.t)
        self.root.start()

    def tearDown(self):
        self.root.stop()
        self.tmp.cleanup()

    def stamp(self, src):
        (self.t / "engine").mkdir(exist_ok=True)
        (self.t / "engine" / "strata").write_bytes(b"old")
        (self.t / "engine" / "BUILD.json").write_text(json.dumps({"source": "local", "backend": "metal", "src": src}))

    def test_a_changed_source_is_compiled_again(self):
        self.stamp("old")
        build = mock.Mock()
        with mock.patch.object(setup, "source_hash", lambda parts: "new"), \
                mock.patch.object(setup, "build_engine_metal", build), \
                mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama"), \
                mock.patch.object(setup, "gpu_info", never("nvidia-smi")):
            setup.update_installed_engine("unused")
        build.assert_called_once()

    def test_the_same_source_is_not(self):
        self.stamp("same")
        with mock.patch.object(setup, "source_hash", lambda parts: "same"), \
                mock.patch.object(setup, "metal_env", never("the build")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.build_engine_metal(True, self.t / "llama"), self.t / "engine")
            setup.update_installed_engine("unused")

    def test_the_build(self):
        built = []

        def cmake_build(src, bdir, target, defs, vcvars, bat, env=None):
            built.append((target, defs, env))
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / "strata").write_bytes(b"new engine")
        with mock.patch.object(setup, "source_hash", lambda parts: "h"), \
                mock.patch.object(setup, "source_version", lambda: "0.1.34"), \
                mock.patch.object(setup, "metal_env", lambda yes: {"DEVELOPER_DIR": "/X"}), \
                mock.patch.object(setup, "cmake_build", cmake_build), \
                contextlib.redirect_stdout(io.StringIO()):
            eng = setup.build_engine_metal(True, self.t / "llama")
        target, defs, env = built[0]
        self.assertEqual(target, "strata")
        for d in ("-DSTRATA_ENABLE_METAL=ON", "-DSTRATA_METAL_ENGINE=ON", "-DSTRATA_NATIVE_EXPERTS=ON",
                  f"-DSTRATA_GGML_DIR={self.t / 'llama'}"):
            self.assertIn(d, defs)
        self.assertEqual(env, {"DEVELOPER_DIR": "/X"})
        self.assertEqual((eng / "strata").read_bytes(), b"new engine")
        meta = json.loads((eng / "BUILD.json").read_text())
        self.assertEqual((meta["backend"], meta["src"], meta["source"]), ("metal", "h", "local"))
        self.assertIn("cmake", setup.METAL_SOURCES)                # the metallib's build rules count as source


class Xcode(unittest.TestCase):
    def run_env(self, compiler, apps, answer="y", runner=None):
        calls = []
        with mock.patch.object(setup, "metal_compiler", compiler), \
                mock.patch.object(setup, "xcode_apps", lambda: apps), \
                mock.patch.object(setup, "run", runner or (lambda cmd, **k: calls.append(cmd))), \
                mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch("builtins.input", lambda p="": answer), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            os.environ.pop("DEVELOPER_DIR", None)
            try:
                env = setup.metal_env(False)
            except SystemExit:
                env = None
        return env, calls, out.getvalue()

    def test_xcode_selected_elsewhere_is_found_without_sudo(self):
        app = Path("/Applications/Xcode.app")
        env, calls, _ = self.run_env(lambda env=None: (bool(env and "DEVELOPER_DIR" in env), "Apple metal version 1"),
                                     [app])
        self.assertEqual(env["DEVELOPER_DIR"], str(app / "Contents" / "Developer"))
        self.assertEqual(calls, [])

    def test_a_missing_toolchain_is_downloaded_after_asking(self):
        state = {"have": False}

        def compiler(env=None):
            return (True, "Apple metal version 1") if state["have"] else (False, "error: missing Metal Toolchain")

        ran = []

        def run(cmd, **k):
            ran.append(cmd)
            state["have"] = True
        env, _, out = self.run_env(compiler, [Path("/Applications/Xcode.app")], runner=run)
        self.assertIsNotNone(env)
        self.assertEqual(ran, [["xcodebuild", "-downloadComponent", "MetalToolchain"]])
        self.assertIn("Metal Toolchain", out)

    def test_a_refused_toolchain_download_stops(self):
        env, calls, out = self.run_env(lambda env=None: (False, "error: missing Metal Toolchain"),
                                       [Path("/Applications/Xcode.app")], answer="n")
        self.assertIsNone(env)
        self.assertEqual(calls, [])
        self.assertIn("xcodebuild -downloadComponent MetalToolchain", out)

    def test_no_xcode_stops_with_the_app_store(self):
        env, _, out = self.run_env(lambda env=None: (False, "xcrun: error: unable to find utility \"metal\""), [])
        self.assertIsNone(env)
        self.assertIn("App Store", out)

    def test_an_unaccepted_license_says_so(self):
        env, _, out = self.run_env(lambda env=None: (False, "You have not agreed to the Xcode license agreements"),
                                   [Path("/Applications/Xcode.app")])
        self.assertIsNone(env)
        self.assertIn("xcodebuild -license accept", out)


class DataFolder(unittest.TestCase):
    def folder(self, t: Path, mac: bool, settings=None):
        saved = {}
        with mock.patch.object(setup, "ROOT", t / "Strata"), mock.patch.object(setup, "MAC", mac), \
                mock.patch.object(setup, "load_settings", lambda: dict(settings or {})), \
                mock.patch.object(setup, "save_settings", lambda s: saved.update(s)), \
                contextlib.redirect_stdout(io.StringIO()):
            dest, elsewhere = setup.data_folder(None)
        return dest, elsewhere, saved

    def test_a_mac_keeps_the_model_inside_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp).resolve()
            (t / "Strata").mkdir()
            dest, _, saved = self.folder(t, True)
            self.assertEqual(dest, t / "Strata" / "Strata-data")
            self.assertNotIn("data_dir", saved)                # not remembered: it moves with the folder
            dest, _, saved = self.folder(t, False)
            self.assertEqual(dest, t / "Strata-data")          # a PC: next to it, as before
            self.assertEqual(saved["data_dir"], str(t / "Strata-data"))

    def test_another_folders_model_is_used_where_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp).resolve()
            (t / "Strata").mkdir()
            other = t / "Strata2"
            (other / "Strata-data" / "models" / "IQ2_XS").mkdir(parents=True)
            (other / "setup.py").write_text("")
            (other / "Strata-data" / "models" / "IQ2_XS" / "a.gguf").write_bytes(b"x")
            dest, elsewhere, _ = self.folder(t, True, {"installs": [str(other)]})
            self.assertIn(other / "Strata-data", elsewhere)
            self.assertTrue((other / "Strata-data" / "models" / "IQ2_XS" / "a.gguf").exists())   # not moved


class Mirror(unittest.TestCase):
    def test_hf_endpoint(self):
        self.assertEqual(setup.HF_HOST, os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/"))
        with mock.patch.object(setup, "HF_HOST", "https://hf-mirror.com"):
            u = setup.hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "IQ2_XS/x.gguf"
            self.assertTrue(u.startswith("https://hf-mirror.com/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/"))
            self.assertEqual(setup.hf_unpinned(u),
                             "https://hf-mirror.com/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ2_XS/x.gguf")

    def test_tuning_is_for_nvidia(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = mac_config(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertFalse(setup.calibrate_config(p))
            self.assertIn("NVIDIA PCs", out.getvalue())


def put_model(t: Path, tag: str, q: str, cfg: dict | None = None):
    m = t / "Strata-data" / "models" / tag
    m.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        p = m / f"Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf"
        p.write_bytes(b"")
        setup.mark(p)
    if cfg is not None:
        (t / f"strata-{tag.lower()}.json").write_text(json.dumps(cfg))


class Menu(unittest.TestCase):
    def pick(self, ram, index, before=None, argv=("--no-start",)):
        seen = []

        def menu(title, lines, default, keys=None):
            seen.append((title, lines, default))
            return index
        res = install(ram, list(argv), answers="", tty=True, before=before,
                      extra=[mock.patch.object(setup, "arrow_menu", menu)])
        return res, seen[0] if seen else None

    def test_every_run_lists_the_models_downloaded_first(self):
        (code, out, cfg, _, _), (title, lines, default) = self.pick(96.0, 0, lambda t: put_model(t, "Q2_0", "Q2_0"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("96 GB Mac", title)
        self.assertTrue(lines[0].startswith("Qwen3.8-Flash-Next Q2_0"), lines)
        self.assertIn("(downloaded;", lines[0])
        iq2 = next(x for x in lines if "IQ2_XS" in x and "Swift" not in x)
        self.assertIn("not downloaded, 68 GB", iq2)
        self.assertIn("recommended for this Mac", iq2)
        self.assertIn("measured: 24 tok/s", iq2)
        self.assertEqual(sum("recommended" in x for x in lines), 1)
        self.assertEqual(len(lines), len(setup.mac_profile()["menu"]))
        self.assertIn("Q2_0-00001-of-00002.gguf", cfg["args"][cfg["args"].index("--native") + 1])   # the pick

    def test_the_memory_decides_the_brackets(self):
        _, (_, lines, default) = self.pick(48.0, 0)
        by = {x.split(" - ")[0].strip(): x for x in lines}
        self.assertIn("recommended for this Mac", by["Qwen3.8-Flash-Next Coder IQ1_M"])
        self.assertIn("tight: part of it on the CPU", by["Qwen3.8-Flash-Next IQ2_XS"])
        self.assertIn("too little memory", by["Qwen3.8-Flash-Next IQ3_S"])
        self.assertIn("Coder IQ1_M", lines[default])                  # nothing installed: the recommended one

    def test_the_model_used_last_is_the_default(self):
        _, (_, lines, default) = self.pick(96.0, 0, lambda t: put_model(t, "Q2_0", "Q2_0", {"args": []}))
        self.assertIn("Q2_0", lines[default])
        self.assertIn("downloaded, set up", lines[default])

    def test_keys(self):
        self.assertEqual(setup.menu_step("down", 2, 3), (0, False, False))     # wraps
        self.assertEqual(setup.menu_step("up", 0, 3), (2, False, False))
        self.assertEqual(setup.menu_step("k", 1, 3), (0, False, False))
        self.assertEqual(setup.menu_step("3", 0, 3), (2, True, False))
        self.assertEqual(setup.menu_step("enter", 1, 3), (1, True, False))
        self.assertEqual(setup.menu_step("q", 1, 3), (1, False, True))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.arrow_menu("t", ["a", "b", "c"], 0, keys=iter(["down", "down", "enter"])), 2)
            with self.assertRaises(SystemExit):
                setup.arrow_menu("t", ["a", "b"], 0, keys=iter(["quit"]))


class Installed(unittest.TestCase):
    def test_a_set_up_model_starts_without_a_setup(self):
        started = []
        code, out, cfg, _, _ = install(96.0, [], before=lambda t: put_model(t, "IQ2_XS", "IQ2_XS", {"args": [], "port": 8090}),
                                       extra=[mock.patch.object(setup, "start", lambda p, port, gpu, **k: started.append((p.name, port)) or 0),
                                              mock.patch.object(setup, "build_engine_metal", never("a build"))])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(started, [("strata-iq2_xs.json", None)])

    def test_a_setup_again_keeps_what_was_changed_by_hand(self):
        old = {"args": ["--max-context", "65536", "--kv", "int8"], "port": 8090, "host": "0.0.0.0", "api_key": "k",
               "sampling": {"temperature": 0.3}, "vision": {"exe": "old"}}
        code, out, cfg, _, _ = install(96.0, ["--setup", "--no-start"], before=lambda t: put_model(t, "IQ2_XS", "IQ2_XS", old))
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((cfg["port"], cfg["host"], cfg["api_key"], cfg["sampling"]), (8090, "0.0.0.0", "k", {"temperature": 0.3}))
        a = cfg["args"]
        self.assertEqual((a[a.index("--max-context") + 1], a[a.index("--kv") + 1]), ("65536", "int8"))
        self.assertEqual(cfg["vision"]["exe"], "engine/strata-vision")   # setup's own keys are setup's


class Response:
    def __init__(self, body: bytes, status=200, headers=None):
        self.body, self.status, self.headers, self.pos = body, status, headers or {}, 0

    def read(self, n=-1):
        b = self.body[self.pos:] if n < 0 else self.body[self.pos:self.pos + n]
        self.pos += len(b)
        return b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ranged(data: bytes, served: list):
    def urlopen(req, timeout=None):
        r = req.headers.get("Range")
        if not r:
            return Response(data, 200, {"Content-Length": str(len(data))})
        lo, hi = r[len("bytes="):].split("-")
        lo, hi = int(lo), int(hi) if hi else len(data) - 1
        served.append((lo, hi))
        return Response(data[lo:hi + 1], 206, {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"})
    return urlopen


class Downloads(unittest.TestCase):
    def test_parallel_ranges_resume_and_keep_what_is_there(self):
        data = bytes(range(256)) * 41
        with tempfile.TemporaryDirectory() as d:
            part = Path(d) / "m.gguf.part"
            part.write_bytes(data[:1000])                    # a single-stream download stopped here
            served = []
            with mock.patch.object(setup.urllib.request, "urlopen", ranged(data, served)), \
                    mock.patch.object(setup, "DOWNLOAD_RANGES", 3), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(setup.ranges_ok("https://x/m.gguf"))
                setup.download_ranges("https://x/m.gguf", part, len(data), "m")
            self.assertEqual(part.read_bytes(), data)
            self.assertFalse(part.with_name(part.name + ".plan").exists())
            self.assertGreaterEqual(min(lo for lo, hi in served if lo > 0), 1000)   # the first 1000 not again
            # a plan from an interrupted run: only the rest of each range is fetched
            part.write_bytes(data[:2000] + bytes(len(data) - 2000))
            plan = {"total": len(data), "ranges": [{"start": 1000, "end": len(data), "at": 2000}]}
            part.with_name(part.name + ".plan").write_text(json.dumps(plan))
            served.clear()
            with mock.patch.object(setup.urllib.request, "urlopen", ranged(data, served)), \
                    contextlib.redirect_stdout(io.StringIO()):
                setup.download_ranges("https://x/m.gguf", part, len(data), "m")
            self.assertEqual(part.read_bytes(), data)
            self.assertEqual(served, [(2000, len(data) - 1)])

    def test_a_server_that_ignores_ranges_is_not_used_for_them(self):
        with mock.patch.object(setup.urllib.request, "urlopen", lambda req, timeout=None: Response(b"x" * 10, 200)):
            self.assertFalse(setup.ranges_ok("https://x/m.gguf"))

    def test_the_published_sha256_is_checked_once(self):
        tree = [{"path": "IQ2_XS/a.gguf", "lfs": {"oid": "ab" * 32}}]
        seen = []

        def urlopen(req, timeout=None):
            seen.append(req.full_url)
            return Response(json.dumps(tree).encode())
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            url = "https://hf-mirror.com/ISTA-DASLab/R/resolve/" + "c" * 40 + "/IQ2_XS/a.gguf"
            self.assertEqual(setup.hf_sha256(url), "ab" * 32)
            self.assertEqual(seen[-1], "https://hf-mirror.com/api/models/ISTA-DASLab/R/tree/" + "c" * 40 + "/IQ2_XS")
            s = Path(d) / "a.gguf"
            s.write_bytes(b"model")
            with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                setup.verify_hf(s, url)                     # wrong bytes: deleted, so the next run fetches it again
            self.assertFalse(s.exists())
            tree[0]["lfs"]["oid"] = setup.hashlib.sha256(b"model").hexdigest()
            s.write_bytes(b"model")
            with contextlib.redirect_stdout(io.StringIO()):
                setup.verify_hf(s, url)
            n = len(seen)
            setup.verify_hf(s, url)                          # recorded in its finish mark: not asked again
            self.assertEqual(len(seen), n)

    def test_the_same_shard_is_linked_not_downloaded(self):
        def before(t):
            m = t / "Strata-data" / "models" / "IQ2_XS"
            m.mkdir(parents=True)
            (m / SHARD.format(2)).write_bytes(b"ple table")
            setup.mark(m / SHARD.format(2), "sha256 " + "ee" * 32)
        got = []

        def dl(url, dst, what=None):
            got.append(dst.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--model", "Q2_0"], before=before,
                                       extra=[mock.patch.object(setup, "hf_sha256", lambda url: "ee" * 32),
                                              mock.patch.object(setup, "download", dl)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("shared with IQ2_XS", out)
        self.assertNotIn("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", got)
        self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", got)


if __name__ == "__main__":
    unittest.main()
