"""Read-only Colab diagnostics, without Colab, GPU or network."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "diagnose.py"
spec = importlib.util.spec_from_file_location("diagnose", SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class Diagnose(unittest.TestCase):
    def process(self, proc, pid, ppid, state="S"):
        path = proc / str(pid)
        path.mkdir()
        fields = [state, str(ppid)] + ["0"] * 9 + ["5", "7"] + ["0"] * 8
        (path / "stat").write_text(f"{pid} (name with ) spaces) " + " ".join(fields))
        (path / "io").write_text("rchar: 100\nwchar: 200\nread_bytes: 300\nwrite_bytes: 400\n")
        (path / "cmdline").write_text("SECRET-COMMAND")
        (path / "environ").write_text("SECRET-ENVIRONMENT")

    def test_live_descendants_files_resources_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as d:
            root, proc = Path(d) / "root", Path(d) / "proc"
            results = root / "phase0" / "results"
            results.mkdir(parents=True)
            models = root / "models"
            models.mkdir()
            (models / "model.gguf.part").write_bytes(b"123")
            (results / "colab_bootstrap_status.json").write_text('{"pid": 10, "token": "SECRET-BOOT"}')
            (results / "colab_launcher.log").write_text("ValueError: SECRET-LOG\nSECRET-URL\n")
            proc.mkdir()
            (proc / "meminfo").write_text("MemTotal: 100 kB\nMemAvailable: 50 kB\nSwapFree: 0 kB\n")
            self.process(proc, 10, 1)
            self.process(proc, 11, 10)
            self.process(proc, 12, 11)
            self.process(proc, 20, 1)
            before = {p: p.read_bytes() for p in Path(d).rglob("*") if p.is_file()}
            report = diag.diagnose(root, proc)
            self.assertTrue(report["launcher"]["alive"])
            self.assertEqual(report["launcher"]["cpu_ticks"], 12)
            self.assertEqual(report["launcher"]["write_bytes"], 400)
            self.assertEqual({p["pid"] for p in report["descendants"]}, {11, 12})
            self.assertEqual(report["models"][0]["bytes"], 3)
            self.assertIn("mtime", report["models"][0])
            self.assertEqual(report["memory_bytes"]["MemAvailable"], 51200)
            self.assertGreater(report["disk_free_bytes"], 0)
            self.assertEqual(report["logs"][0]["exceptions"], ["ValueError"])
            self.assertNotIn("SECRET", json.dumps(report))
            self.assertEqual(before, {p: p.read_bytes() for p in Path(d).rglob("*") if p.is_file()})

    def test_absent_and_zombie_launcher(self):
        with tempfile.TemporaryDirectory() as d:
            root, proc = Path(d), Path(d) / "proc"
            results = root / "phase0" / "results"
            results.mkdir(parents=True)
            proc.mkdir()
            self.assertFalse(diag.diagnose(root, proc)["launcher"]["alive"])
            for raw in ('{"pid": []}', '[10]', 'not json'):
                (results / "colab_bootstrap_status.json").write_text(raw)
                self.assertFalse(diag.diagnose(root, proc)["launcher"]["alive"])
            (results / "colab_bootstrap_status.json").write_text('{"pid": 10}')
            self.assertFalse(diag.diagnose(root, proc)["launcher"]["alive"])
            self.process(proc, 10, 1, "Z")
            report = diag.diagnose(root, proc)
            self.assertFalse(report["launcher"]["alive"])
            self.assertEqual(report["launcher"]["state"], "Z")

    def test_numeric_string_pid_and_invalid_pids(self):
        with tempfile.TemporaryDirectory() as d:
            root, proc = Path(d), Path(d) / "proc"
            results = root / "phase0" / "results"
            results.mkdir(parents=True)
            proc.mkdir()
            self.process(proc, 10, 1)
            self.process(proc, 11, 10)
            status = results / "colab_bootstrap_status.json"
            for pid in (10, "10", "0010"):
                with self.subTest(pid=pid):
                    status.write_text(json.dumps({"pid": pid}))
                    report = diag.diagnose(root, proc)
                    self.assertEqual(report["launcher"]["pid"], 10)
                    self.assertTrue(report["launcher"]["alive"])
                    self.assertEqual([p["pid"] for p in report["descendants"]], [11])
            for pid in (True, False, 10.0, None, [], {}, 0, -10, "0", "-10", "+10", "10.0", " 10", "10 ",
                        "10\n", "", "abc", "1e1", "١٠"):
                with self.subTest(pid=pid):
                    status.write_text(json.dumps({"pid": pid}))
                    report = diag.diagnose(root, proc)
                    self.assertIsNone(report["launcher"]["pid"])
                    self.assertFalse(report["launcher"]["alive"])
                    self.assertEqual(report["descendants"], [])

    def test_qualified_exceptions_without_messages_or_urls(self):
        with tempfile.TemporaryDirectory() as d:
            root, proc = Path(d), Path(d) / "proc"
            results = root / "phase0" / "results"
            results.mkdir(parents=True)
            proc.mkdir()
            types = ["ValueError", "httpx.HTTPStatusError", "huggingface_hub.errors.HfHubHTTPError",
                     "httpx.ReadTimeout", "KeyboardInterrupt", "module.CustomFailure"]
            log = "\n".join(f"{name}: SECRET-MESSAGE https://example.com/?token=SECRET-TOKEN" for name in types)
            log += "\nhttps://example.com/SECRET-URL\n  File SECRET-PATH, line 1\n"
            (results / "colab_launcher.log").write_text(log)
            report = diag.diagnose(root, proc)
            self.assertEqual(report["logs"][0]["exceptions"], types)
            self.assertNotIn("SECRET", json.dumps(report))
            self.assertNotIn("https://", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
