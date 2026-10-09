"""tools/export_public.py: every exported file is scanned, and nothing is read through a link."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

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
