"""tools/export_public.py: every exported file is scanned, and nothing is read through a link."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "tools" / "export_public.py"  # private tool: absent from the public tree, where this is skipped
ex = None
if EXPORTER.exists():
    _spec = importlib.util.spec_from_file_location("export_public", EXPORTER)
    ex = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ex)

FAKE_TOKEN = "hf_" + "Ab3" * 12  # built at run time: this file is exported and scanned too


@unittest.skipIf(ex is None, "tools/export_public.py absent (public tree)")
class TestScan(unittest.TestCase):
    def scan(self, files: dict[str, bytes]):
        with tempfile.TemporaryDirectory() as d:
            for rel, data in files.items():
                p = Path(d) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            return ex.scan(Path(d), list(files))[0]

    def test_reviewed_fixture_lines_only(self):
        # The privacy guard's synthetic fixtures are excused line by line (whole line hashed): another
        # private key or home path in the same file still blocks (Codex review of the allowlist).
        rel = "app/tests/test_e2e_privacy.py"
        real = Path(ROOT / rel).read_text(encoding="utf-8")
        self.assertEqual(self.scan({rel: real.encode()}), [])
        key = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"  # split: this file is exported and scanned
        home = "C:" + "\\Users\\someone\\Documents\\private.txt"
        for extra in (f'X = "{key}\\nreal\\n"', f'P = r"{home}"', '"api_key": ["sk-' + "x" * 30 + '"],'):
            with self.subTest(extra=extra[:20]):
                self.assertTrue(self.scan({rel: (real + "\n" + extra + "\n").encode()}))

    def test_code_pilot_artifacts_still_block_secrets_and_private_paths(self):
        base = "phase0/results/code_pilot_cpu_20261010_01/"
        home = "C:" + "\\Users\\someone\\Documents\\private.txt"
        for name in ("manifest.json", "single__affine-cipher.jsonl", "accuracy_seconds_per_task.svg"):
            self.assertTrue(ex.selected(base + name))
            for text in (FAKE_TOKEN, home):
                self.assertTrue(self.scan({base + name: text.encode()}))

    def test_repo_pilot_diagnostic_whitelist_is_exact_and_scanned(self):
        allowed = {"phase0/repo_pilot/" + name for name in
                   ("tasks.json", "README.md", "pyproject.toml", "uv.lock", "final_cases.json",
                    "requests_diagnostic.json", "licenses/sympy-11400.txt", "licenses/sympy-11897.txt",
                    "licenses/sympy-12171.txt", "licenses/mpmath-0.19.txt", "cpu_20261010_private_hashes.json")} | {
                        "docs/13_repo_pilot.md"}
        for rel in allowed:
            self.assertTrue(ex.selected(rel), rel)
            self.assertTrue(self.scan({rel: FAKE_TOKEN.encode()}))
        for name in ("sources.json", "test.parquet", "reference.patch", "tests.py", "private.txt",
                     "sub/tasks.json", "tasks.json.part", "smoke.json", "licenses/extra.txt"):
            self.assertFalse(ex.selected("phase0/repo_pilot/" + name), name)
        for name in ("smoke.json", "verdicts.json.part", "single__psf__requests-1963.jsonl", "reference.patch"):
            self.assertFalse(ex.selected("phase0/results/repo_pilot_cpu_20261010_01/" + name), name)
        for name in ex.REPO_PILOT_RESULTS:
            self.assertFalse(ex.selected("phase0/results/repo_pilot_cpu_20261010_01/" + name), name)
            self.assertTrue(ex.selected("phase0/results/repo_pilot_compare_20261010_01/" + name), name)
            self.assertTrue(self.scan({"phase0/results/repo_pilot_compare_20261010_01/" + name: FAKE_TOKEN.encode()}))
            self.assertFalse(ex.selected("phase0/results/repo_pilot_other_20261010_01/" + name), name)
            self.assertFalse(ex.selected("phase0/results/repo_pilot_cpu_20261010_01/sub/" + name), name)
        self.assertFalse(ex.selected("phase0/data/repo-pilot/test.parquet"))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ex, "dirty_paths", return_value=[]), \
                mock.patch.object(ex, "git_files", return_value={f: "100644" for f in allowed}), \
                mock.patch.object(ex, "REWRITES", []), mock.patch("sys.stdout"):
            out = Path(d) / "public"
            self.assertEqual(ex.main([str(out)]), 0)
            exported = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(exported, allowed)
            for rel in allowed:
                self.assertEqual((out / rel).read_bytes(), (ROOT / rel).read_bytes())

    def test_repo_public_derivation_exact_selection_and_injected_secrets(self):
        public = "phase0/results/" + ex.REPO_PUBLIC_DIRECTORY + "/"
        private = "phase0/results/" + ex.REPO_PRIVATE_DIRECTORY + "/"
        self.assertEqual(len(ex.REPO_PUBLIC_RESULTS), 14)
        home = "/" + "home/fixture/private.txt"
        windows = "C:" + "\\Users\\fixture\\private.txt"
        for name in ex.REPO_PUBLIC_RESULTS:
            self.assertTrue(ex.selected(public + name), name)
            self.assertFalse(ex.selected(private + name), name)
            for injected in (FAKE_TOKEN, home, windows):
                self.assertTrue(self.scan({public + name: injected.encode()}))
            self.assertFalse(ex.selected(public + "sub/" + name))
            self.assertFalse(ex.selected(public.replace("_01/", "_02/") + name))
        for neighbor in ("smoke.json", "cache.json", ".campaign.lock", "report.md.part", "vote__sympy__sympy-11400.jsonl"):
            self.assertFalse(ex.selected(public + neighbor))
            self.assertFalse(ex.selected(private + neighbor))
            self.assertFalse(ex.selected(private + "sub/" + neighbor))

    def test_e10_synthetic_result_exact_line_only(self):
        rel = "phase0/results/sot_base_Qwen__Qwen3.5-4B_colab_other.jsonl"
        data = (ROOT / rel).read_bytes()
        lines = data.splitlines(keepends=True)
        self.assertEqual(json.loads(lines[34])["id"], "mtbench-121")
        self.assertEqual(self.scan({rel: data}), [])
        # Changing the example, adding a secret or whitespace changes the whole-line hash.
        changed_path = lines[34].replace(b"Name", b"RealUser")
        secret = lines[34].rstrip(b"\n") + b" " + FAKE_TOKEN.encode() + b"\n"
        for replacement in (changed_path, secret, b" " + lines[34]):
            changed = b"".join(lines[:34] + [replacement] + lines[35:])
            with self.subTest(replacement=replacement[:20]):
                self.assertTrue(self.scan({rel: changed}))
        # Neither another line in the same file nor the exact line in another file qualifies.
        self.assertTrue(self.scan({rel: data + lines[34]}))
        self.assertTrue(self.scan({rel: b"\n" + data}))
        self.assertTrue(self.scan({rel.replace("_other.", "_test."): data}))

    def test_pii_synthetic_fixtures_require_exact_path_and_line(self):
        # Parent-approved PII fixtures: the whole line and path must match; nothing else is excused.
        paths = ("app/README.md", "app/bench/pii/handmade.py", "app/tests/test_privacy.py")
        checked = 0
        for rel in paths:
            data = (ROOT / rel).read_bytes()
            self.assertEqual(self.scan({rel: data}), [])
            self.assertTrue(self.scan({rel: data + b"\n" + FAKE_TOKEN.encode() + b"\n"}))
            for line in data.splitlines(keepends=True):
                if (rel, ex.fixture_hash(line.decode("utf-8"), 0)) not in ex.FAKE_FIXTURES:
                    continue
                checked += 1
                with self.subTest(rel=rel, fixture=checked):
                    self.assertEqual(self.scan({rel: line}), [])
                    self.assertTrue(self.scan({rel: line.rstrip(b"\r\n") + b" " + FAKE_TOKEN.encode() + b"\n"}))
                    self.assertTrue(self.scan({"app/tests/unreviewed.py": line}))
        self.assertEqual(checked, 15)

    def test_e11_decorators_require_exact_path_line_and_hash(self):
        entries = [(p, n) for p, n, _ in ex.SYNTHETIC_RESULT_LINES if p.startswith("phase0/results/code_")]
        self.assertEqual(len(entries), 11)
        for rel, line in entries:
            data = (ROOT / rel).read_bytes()
            lines = data.splitlines(keepends=True)
            with self.subTest(rel=rel, line=line):
                self.assertEqual(self.scan({rel: data}), [])
                changed = b"".join(lines[:line - 1] + [b" " + lines[line - 1]] + lines[line:])
                self.assertTrue(self.scan({rel: changed}))
                self.assertTrue(self.scan({rel: data + b"\n" + FAKE_TOKEN.encode()}))
                self.assertTrue(self.scan({rel: b"\n" + data}))
                self.assertTrue(self.scan({rel.replace("_colab_", "_other_"): data}))

    def test_e11_artifact_whitelist_is_exact_and_scanned(self):
        for name in ex.E11_FILES:
            rel = "phase0/results/" + name
            self.assertTrue(ex.selected(rel))
            self.assertTrue(self.scan({rel: FAKE_TOKEN.encode()}))
        for name in ("e11_validation_private.json", "e11_accuracy_energy_colab.svg", "e11_accuracy_ms_smoke.pdf"):
            self.assertFalse(ex.selected("phase0/results/" + name))

    def test_any_extension_and_notebooks(self):
        # Audit 2026-10-09 (paper, 1): a token in a .ipynb or a .env variant gave zero alerts.
        for rel, data in (("phase0/colab/x.ipynb", json.dumps({"cells": [{"source": [FAKE_TOKEN]}]}).encode()),
                          ("app/config.env", f"HF={FAKE_TOKEN}\n".encode()), ("a/b.weird", FAKE_TOKEN.encode()),
                          ("c/blob.bin", b"\x00\xff" + FAKE_TOKEN.encode() + b"\x00")):
            with self.subTest(rel=rel):
                self.assertTrue(any("Hugging Face token" in b for b in self.scan({rel: data})))
        # a token split over two JSON lines of a notebook cell is reassembled
        split = {"cells": [{"source": [FAKE_TOKEN[:12], FAKE_TOKEN[12:]]}]}
        self.assertTrue(any("notebook" in b for b in self.scan({"n.ipynb": json.dumps(split).encode()})))
        out = {"cells": [{"source": [], "outputs": [{"output_type": "stream", "text": ["HF=", FAKE_TOKEN[:9], FAKE_TOKEN[9:]]}]}]}
        self.assertTrue(any("notebook" in b for b in self.scan({"o.ipynb": json.dumps(out).encode()})))

    def test_env_files_never_selected(self):
        for rel in ("app/config.env", "app/.env.local", "app/.envrc", "phase0/prod.env.bak", ".env"):
            self.assertFalse(ex.selected(rel), rel)
        self.assertTrue(ex.selected("app/myriad/node.py"))

    def test_gpqa_raw_files_never_selected(self):
        # E12: GPQA's terms forbid revealing its questions online; a model output may quote one.
        for rel in ("phase0/results/aa_Qwen__Qwen3.5-4B_colab_gpqa_all.jsonl",
                    "phase0/results/aa_Qwen__Qwen3.5-4B_colab_gpqa_all.jsonl.meta.json",
                    "phase0/results/GPQA_dump.jsonl"):
            self.assertFalse(ex.selected(rel), rel)
        self.assertTrue(ex.selected("phase0/results/aa_Qwen__Qwen3.5-4B_colab_scicode_test.jsonl"))
        self.assertTrue(ex.selected("phase0/results/e12_report_colab.md"))

    def test_scicode_official_files_only_selected(self):
        base = "phase0/essaim/scicode_skipped/"
        for name in ("13.6.txt", "62.1.txt", "76.3.txt", "README.md", "LICENSE"):
            self.assertTrue(ex.selected(base + name), name)
        for rel in (base + "private.txt", base + "fixture.json", base + "gpqa_diamond.csv",
                    base + "notes.md", base + "13.6.txt.bak", base + "nested/README.md",
                    "phase0/essaim/other/13.6.txt", "phase0/data/gpqa_diamond.csv",
                    "phase0/data/scicode_test_data.h5"):
            self.assertFalse(ex.selected(rel), rel)

    def test_scicode_official_files_exported_unchanged(self):
        base = "phase0/essaim/scicode_skipped/"
        files = [base + name for name in ("13.6.txt", "62.1.txt", "76.3.txt", "README.md", "LICENSE")]
        private = [base + "private.txt", base + "fixture.json", "phase0/data/gpqa_diamond.csv"]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ex, "dirty_paths", return_value=[]), \
                mock.patch.object(ex, "git_files", return_value={f: "100644" for f in files + private}), \
                mock.patch.object(ex, "REWRITES", []), mock.patch("sys.stdout"):
            out = Path(d) / "public"
            self.assertEqual(ex.main([str(out)]), 0)
            exported = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(exported, set(files))
            for rel in files:
                self.assertEqual((out / rel).read_bytes(), (ROOT / rel).read_bytes())

    def test_terminal_bench_resources_only_selected(self):
        base = "phase0/harbor/"
        for name in ("pyproject.toml", "uv.lock", "tasks.json"):
            self.assertTrue(ex.selected(base + name), name)
        self.assertTrue(ex.selected("docs/11_terminal_bench.md"))
        for rel in (base + "private.json", base + "tasks.json.bak", base + "notes.md", base + "credentials.env",
                    base + "extra.py", base + "nested/tasks.json", base + "nested/uv.lock",
                    "phase0/data/terminal-bench/selected/LICENSE", "docs/10_bancs_reels.md",
                    "docs/11_terminal_bench_private.md"):
            self.assertFalse(ex.selected(rel), rel)

    def test_terminal_bench_resources_and_guide_exported_unchanged(self):
        files = ["phase0/harbor/" + name for name in ("pyproject.toml", "uv.lock", "tasks.json")]
        files += ["docs/11_terminal_bench.md", "phase0/README.md"]
        private = ["phase0/harbor/private.json", "phase0/harbor/nested/tasks.json", "docs/10_bancs_reels.md"]
        rewrites = [rule for rule in ex.REWRITES if rule[0] == "phase0/README.md"]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ex, "dirty_paths", return_value=[]), \
                mock.patch.object(ex, "git_files", return_value={f: "100644" for f in files + private}), \
                mock.patch.object(ex, "REWRITES", rewrites), mock.patch("sys.stdout"):
            out = Path(d) / "public"
            self.assertEqual(ex.main([str(out)]), 0)
            exported = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(exported, set(files))
            for rel in files[:-1]:
                self.assertEqual((out / rel).read_bytes(), (ROOT / rel).read_bytes())
            readme = (out / "phase0/README.md").read_text(encoding="utf-8")
            self.assertIn("[guide](../docs/11_terminal_bench.md)", readme)
            self.assertNotIn("docs/10_bancs_reels.md", readme)

    def test_code_pilot_resources_are_an_exact_allowlist(self):
        base = "phase0/code_pilot/"
        for name in ("pyproject.toml", "uv.lock", "tasks.json", "sources.json"):
            self.assertTrue(ex.selected(base + name), name)
        self.assertTrue(ex.selected("docs/12_code_pilot.md"))
        for rel in (base + "private.json", base + "credentials.env", base + "nested/tasks.json",
                    base + ".venv/pyvenv.cfg", "phase0/data/code-pilot/README.md"):
            self.assertFalse(ex.selected(rel), rel)

    def test_code_pilot_results_are_an_exact_flat_allowlist(self):
        base = "phase0/results/code_pilot_cpu_20261010_01/"
        names = {"manifest.json", "verdicts.json", "summary.json", "report.md"}
        names.update(f"{mode}__{task}.jsonl" for mode in ("single", "vote", "cascade", "reference")
                     for task in ("affine-cipher", "beer-song", "book-store"))
        names.update(f"accuracy_{metric}.{ext}" for metric in ("seconds_per_task", "estimated_eur_per_task")
                     for ext in ("svg", "png", "pdf"))
        self.assertEqual(len(names), 22)
        for name in names:
            self.assertTrue(ex.selected(base + name), name)
        for name in ("manifest.json.part", "manifest_other.json", "verdict.json", "smoke.json", ".campaign.lock",
                     "cache.json", "archive.zip", "single__affine-cipher.jsonl.meta.json", "single__allergies.jsonl",
                     "single__affine_cipher.jsonl", "verify__affine-cipher.jsonl", "accuracy_seconds_per_task.jpg",
                     "accuracy_seconds_per_task_extra.svg", "GPQA_dump.jsonl", "nested/manifest.json"):
            self.assertFalse(ex.selected(base + name), name)
        for folder in ("code_pilot_cpu_20261010_01_extra", "code_pilot_cpu_20261010", "code_pilot_cpu_20261010_1",
                       "code_pilot_gpqa_20261010_01", "archive_v1"):
            self.assertFalse(ex.selected(f"phase0/results/{folder}/manifest.json"))
        self.assertTrue(ex.selected("phase0/results/code_pilot_compare_20261010_01/reference__book-store.jsonl"))


@unittest.skipIf(ex is None, "tools/export_public.py absent (public tree)")
class TestSafeRead(unittest.TestCase):
    def test_regular_file_and_outside(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            root = Path(d)
            (root / "a").mkdir()
            (root / "a" / "f.txt").write_bytes(b"ok")
            self.assertEqual(ex.safe_read("a/f.txt", root), b"ok")
            (Path(other) / "secret.txt").write_bytes(b"secret")
            try:
                os.symlink(Path(other) / "secret.txt", root / "a" / "link.txt")
                os.symlink(other, root / "d", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("no symbolic links on this system (Windows without the privilege)")
            # Audit 2026-10-09 (paper, 2): a tracked regular file replaced on disk by a link was followed.
            for rel in ("a/link.txt", "d/secret.txt"):
                with self.subTest(rel=rel), self.assertRaises(ex.Refused):
                    ex.safe_read(rel, root)

    @unittest.skipUnless(os.name == "nt", "junctions: Windows")
    def test_junction_refused(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            (Path(other) / "s.txt").write_bytes(b"secret")
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(Path(d) / "j"), other], capture_output=True)
            if r.returncode != 0:
                self.skipTest("mklink /J unavailable")
            with self.assertRaises(ex.Refused):
                ex.safe_read("j/s.txt", Path(d))


if __name__ == "__main__":
    unittest.main()
