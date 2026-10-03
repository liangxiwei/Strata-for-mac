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


def install(ram, argv, answers=None, extra=(), machine="arm64"):
    """setup.main() on a mocked Mac -> (exit code, printed text, the written config or None, questions asked, folder
    files).  answers: None = --yes, "" = Enter for every question, {words of a question: answer}."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
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
    return a


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
        self.assertEqual(asked, [])
        script = files["run-iq2_xs.sh"]
        self.assertIn('cd "$(dirname "$0")"', script)
        self.assertIn('--config "strata-iq2_xs.json"', script)
        self.assertNotIn(tempfile.gettempdir(), script)

    def test_enter_for_every_question_is_the_same(self):
        code, out, cfg, asked, _ = install(96.0, ["--no-start"], answers="")
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["args"], measured_args())
        self.assertEqual(asked, [])

    def test_the_profile_is_the_measured_one(self):
        p = setup.mac_profile()
        self.assertEqual((p["family"], p["model"], p["context"]), ("qwen", "IQ2_XS", 32768))
        self.assertEqual(str(p["context"]), MEASURED["args"][MEASURED["args"].index("--max-context") + 1])
        for flag in ("--mmap-experts", "--no-prefill-borrow"):
            self.assertIn(flag, p["args"])
        self.assertNotIn("--kv", p["args"])                      # FP16 KV, the engine's default


class Memory(unittest.TestCase):
    def test_64gb_is_untested_but_goes_on(self):
        code, out, cfg, _, _ = install(64.0, ["--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("untested", out)
        self.assertEqual(cfg["args"], measured_args())

    def test_48gb_stops_with_yes_alone(self):
        code, out, cfg, _, _ = install(48.0, ["--no-start"])
        self.assertNotEqual(code, 0)
        self.assertIn("too little", out)
        self.assertIsNone(cfg)

    def test_48gb_with_an_explicit_model_is_the_users_risk(self):
        code, out, cfg, _, _ = install(48.0, ["--no-start", "--model", "IQ2_XS"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("as you chose", out)


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
        self.assertEqual([u.rsplit("/", 1)[1] for u in got], [SHARD.format(1), SHARD.format(2)])
        self.assertIsNone(cfg)                               # no config, no start script
        self.assertEqual(files, {})
        self.assertIn("Next:        ./setup.sh", out)

    def test_present_files_are_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as g:
            for i in (1, 2):
                (Path(g) / SHARD.format(i)).write_bytes(b"")
            code, out, cfg, _, _ = install(96.0, ["--download-only", "--gguf-dir", g], extra=[
                mock.patch.object(setup, "download", never("a download")),
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
                    mock.patch.object(setup, "download", never("a download")),
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

    def test_images_and_the_projection_stay_off(self):
        code, out, cfg, _, _ = install(96.0, ["--no-start", "--vision", "yes", "--experimental-speed-projection", "on"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("vision", cfg)
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


if __name__ == "__main__":
    unittest.main()
