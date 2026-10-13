"""Execute recovery on temporary synthetic fixtures only, without a cloud CLI."""
import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import verif1_campaign as campaign


class VerifRecoveryTests(unittest.TestCase):
    def test_recovery_collects_only_direct_server_logs_with_stable_names(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            content, output = root / "content", root / "recovered"
            files = {
                "verif1_launch.json": json.dumps({"archive_sha256": "a" * 64, "complete": True,
                                                  "exit_code": 0, "scope": "pilot"}),
                "verif1_bootstrap.log": "synthetic bootstrap",
                "verif-1/results/ledger.jsonl": '{"synthetic":true}\n',
                "verif-1/runtime/models/server-logs/z.solo.123.456.log": "synthetic offload",
                "verif-1/runtime/models/server-logs/a.solo.123.457.log": "synthetic startup",
                "verif-1/runtime/models/server-logs/ignored.txt": "synthetic non-log",
                "verif-1/runtime/models/server-logs/nested/ignored.log": "synthetic nested log",
                "verif-1/runtime/models/ignored.log": "synthetic unrelated log",
                "verif-1/runtime/models/weights.gguf": "synthetic weights placeholder",
                "verif-1/runtime/hf/ignored.json": "synthetic cache",
                "verif-1/runtime/.venv/ignored.txt": "synthetic environment",
            }
            for name, text in files.items():
                path = content / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            calls = []

            def cli(args, script=None):
                calls.append(args)
                if args[0] == "exec":
                    self.assertEqual(args, ["exec", "-s", "phase0", "--timeout", "180"])
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        exec(script.replace("/content/", content.as_posix() + "/"), {})
                    return stream.getvalue()
                self.assertEqual(args, ["download", "-s", "phase0", "/content/verif1_recovered.tgz",
                                        str(output / "verif1_recovered.tgz")])
                self.assertIsNone(script)
                shutil.copyfile(content / "verif1_recovered.tgz", args[-1])
                return ""

            record = {"archive_sha256": "a" * 64, "utc": campaign.utc(), "hourly_rate": 0,
                      "session": {"synthetic": True}}
            with patch.object(campaign, "cli", side_effect=cli), redirect_stdout(io.StringIO()):
                result = campaign.recover(record, output)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["sha256"], campaign.file_sha(output / "verif1_recovered.tgz"))
            logs = ["verif-1/model-logs/a.solo.123.457.log", "verif-1/model-logs/z.solo.123.456.log"]
            with tarfile.open(output / "verif1_recovered.tgz") as archive:
                names = archive.getnames()
                self.assertEqual([n for n in names if n.startswith("verif-1/model-logs/")], logs)
                self.assertFalse(any(p in ("models", "hf", ".venv") for n in names for p in Path(n).parts))
                self.assertFalse(any("ignored" in n or n.endswith(".gguf") for n in names))
            for name, text in zip(logs, ("synthetic startup", "synthetic offload")):
                self.assertEqual((output / name).read_text(encoding="utf-8"), text)
            self.assertEqual((output / "verif-1/bootstrap.log").read_text(encoding="utf-8"), "synthetic bootstrap")
            self.assertEqual((output / "verif-1/results/ledger.jsonl").read_text(encoding="utf-8"),
                             '{"synthetic":true}\n')
            self.assertEqual(json.loads((output / "verif1_recovery_receipt.json").read_text())["sha256"],
                             result["sha256"])


if __name__ == "__main__":
    unittest.main()
