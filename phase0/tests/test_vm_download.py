"""Pinned VM inputs and account reserves, using synthetic files and processes only."""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

COLAB = Path(__file__).resolve().parents[1] / "colab"


def load(name):
    spec = importlib.util.spec_from_file_location(name, COLAB / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Inputs(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "real archive fixture requires WSL")
    def test_archive_preserves_small_checkpoints_and_refuses_oversized_companions(self):
        import tarfile
        from tests.test_colab_transfer import WrapperTransfer
        for oversized in (False, True):
            with self.subTest(oversized=oversized), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
                env = fixture.setup_root(root, "success")
                results = root / "phase0/results"
                results.mkdir()
                name = "aa_synthetic_colab_scicode_dev.jsonl"
                contents = {name: b'{"id": "synthetic"}\n', name + ".meta.json": b'{"synthetic": true}',
                            name + ".timing.jsonl": b'{"kind": "call", "wall_s": 1}\n'}
                for filename, raw in contents.items(): (results / filename).write_bytes(raw)
                if oversized:
                    with (results / (name + ".timing.jsonl")).open("wb") as output:
                        output.seek(8 * 1024 * 1024)
                        output.write(b"x")
                result = subprocess.run(["bash", str(root / "phase0/colab/colab_phase0.sh"), "up", "aa-1", "A100"],
                                        env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 65 if oversized else 0, result.stdout + result.stderr)
                self.assertEqual((root / "boot").exists(), not oversized)
                if oversized:
                    self.assertNotIn("upload", (root / "calls").read_text().splitlines())
                else:
                    with tarfile.open(root / "content/dllm.tgz") as archive:
                        for filename, raw in contents.items():
                            self.assertEqual(archive.extractfile("checkpoints/" + filename).read(), raw)
                        self.assertFalse(any(name.startswith("phase0/data/") and "scicode" in name
                                             for name in archive.getnames()))

    def test_prepare_uses_pinned_blobs_rebuilds_cache_and_reuses_valid_vm_inputs(self):
        import huggingface_hub
        from essaim import data, scicode
        from tests.test_aa import synthetic_problem
        helper = load("prepare_scicode")
        problem = synthetic_problem()
        row = {"problem_id": "7", "problem_name": "synthetic", "problem_description_main": "synthetic",
               "problem_background_main": "", "problem_io": "", "required_dependencies": "",
               "sub_steps": [{"step_number": step["number"], "step_description_prompt": step["description"],
                   "step_background": "", "function_header": step["header"], "return_line": step["return_line"],
                   "test_cases": step["tests"], "ground_truth_code": step["reference"]} for step in problem["steps"]]}
        raw = (json.dumps(row) + "\n").encode()
        blobs = {name: scicode.git_blob_sha1(raw) for name in scicode.FILES.values()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "hf-fixture"
            input_file.write_bytes(raw)
            calls = []
            def hf(repo, name, **kw):
                calls.append((repo, name, kw))
                return str(input_file)
            def drive(command, **kw):
                self.assertIn(helper.DRIVE_ID, command)
                self.assertEqual(kw["timeout"], 900)
                Path(command[command.index("-O") + 1]).write_bytes(b"synthetic targets")
            with mock.patch.object(data, "DATA", root / "data"), mock.patch.object(scicode, "BLOB_SHA1", blobs), \
                 mock.patch.object(scicode, "COUNTS", {"dev": (1, 3), "test": (1, 3)}), \
                 mock.patch.object(scicode, "H5_SHA256", hashlib.sha256(b"synthetic targets").hexdigest()), \
                 mock.patch.object(helper.importlib.metadata, "version", return_value="5.2.0"), \
                 mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=hf), \
                 mock.patch.object(helper.subprocess, "run", side_effect=drive) as downloader:
                helper.prepare()
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(kw["revision"] == data.REVISIONS["scicode"] for _, _, kw in calls))
                cache = data.DATA / "scicode_all_optional_reference_v2.jsonl"
                cache_before = cache.read_bytes()
                helper.prepare()
                self.assertEqual(cache.read_bytes(), cache_before)
                downloader.assert_called_once()
                self.assertEqual(len(calls), 2)
                cache.write_bytes(b"corrupt")
                with self.assertRaises(ValueError): helper.prepare()
                self.assertEqual(cache.read_bytes(), b"corrupt")
                cache.write_bytes(cache_before)
                (data.DATA / "scicode_inputs" / scicode.FILES["test"]).write_bytes(b"corrupt blob")
                with self.assertRaises(ValueError): helper.prepare()

    def test_streaming_atomic_success_cache_corruption_and_interruption(self):
        helper = load("prepare_scicode")
        raw = b"synthetic" * 200000
        expected = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "targets.h5"
            def download(output):
                self.assertFalse(target.exists())
                output.write_bytes(raw)
            helper.publish(target, expected, download)
            helper.publish(target, expected, mock.Mock(side_effect=AssertionError("cache must be reused")))
            self.assertEqual(target.read_bytes(), raw)
            target.write_bytes(b"corrupt")
            with self.assertRaises(ValueError): helper.publish(target, expected, download)
            self.assertEqual(target.read_bytes(), b"corrupt")
            target.unlink()
            for interrupted in (False, True):
                def fail(output):
                    output.write_bytes(b"partial")
                    if interrupted: raise subprocess.TimeoutExpired("synthetic", 1)
                with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                    helper.publish(target, expected, fail)
                self.assertFalse(target.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_git_blob_identity(self):
        helper = load("prepare_scicode")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "synthetic.jsonl"
            raw = b'{"synthetic": true}\n'
            expected = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
            helper.publish(target, expected, lambda p: p.write_bytes(raw), "sha1", True)
            self.assertEqual(helper.digest(target, "sha1", True), expected)

    def test_bootstrap_preparation_failure_prevents_launcher(self):
        source = COLAB.parent / "colab_bootstrap.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        body = next(node for node in tree.body if isinstance(node, ast.Try)).body
        index = next(i for i, node in enumerate(body) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "PLAN.startswith('aa-')")
        for failure in (subprocess.CalledProcessError(1, "synthetic"),
                        subprocess.TimeoutExpired("synthetic", 1)):
            namespace = {"PLAN": "aa-1", "WORK": "/synthetic", "RESULTS": "/synthetic/results", "env": {},
                         "groups": [], "run": mock.Mock(side_effect=[None, failure]), "status": mock.Mock(),
                         "subprocess": mock.Mock(DEVNULL=subprocess.DEVNULL)}
            with self.assertRaises(type(failure)):
                exec(compile(ast.Module(body=body[index:index + 3], type_ignores=[]), str(source), "exec"), namespace)
            namespace["subprocess"].Popen.assert_not_called()
            commands = namespace["run"].call_args_list
            self.assertIn("gdown==5.2.0", commands[0].args[0])
            self.assertIn("colab/prepare_scicode.py", commands[1].args[0])
            self.assertEqual(commands[1].kwargs["timeout"], 1200)
        namespace["run"] = mock.Mock()
        namespace["open"] = mock.Mock()
        exec(compile(ast.Module(body=body[index:index + 3], type_ignores=[]), str(source), "exec"), namespace)
        namespace["subprocess"].Popen.assert_called_once()


class Budget(unittest.TestCase):
    def setUp(self):
        self.helper = load("budget")
        self.policy = {"budget": 45, "minimum": 15, "hours": 7.8, "rate": 5.3, "cleanup": 1800, "margin": 155}
        self.before = {"schema": 1, "balance_units": 60.62, "rate_units_hour": 0, "assignments": 0}

    def test_preallocation_budget_duration_unknown_and_real_rate_refusal(self):
        self.helper.preflight(self.before, self.policy)
        for policy in ({**self.policy, "budget": 60}, {**self.policy, "hours": 10},
                       {**self.policy, "minimum": float("nan")}):
            with self.assertRaises(ValueError): self.helper.preflight(self.before, policy)
        for usage in ({}, {**self.before, "assignments": 1}, {**self.before, "rate_units_hour": 1}):
            with self.assertRaises(ValueError): self.helper.preflight(usage, self.policy)
        active = {**self.before, "assignments": 1, "rate_units_hour": 8}
        with self.assertRaises(ValueError): self.helper.preflight(active, self.policy, True)

    def test_observed_rate_and_elapsed_usage_bound_time_with_cleanup_reserve(self):
        active = {**self.before, "assignments": 1, "rate_units_hour": 5.3}
        seconds, rate = self.helper.remaining(active, self.policy, 60.62, 3600, 0)
        self.assertAlmostEqual(seconds, (45 - 5.3) / 5.3 * 3600 - 1955)
        faster, rate = self.helper.remaining({**active, "rate_units_hour": 8}, self.policy, 60.62, 3600, rate)
        self.assertLess(faster, seconds)
        self.assertEqual(rate, 8)
        for usage in ({**active, "balance_units": 16}, {**active, "rate_units_hour": 0},
                      {**active, "assignments": 2}, {**active, "schema": True}, {}):
            with self.assertRaises(ValueError): self.helper.remaining(usage, self.policy, 60.62, 3600, rate)
        with self.assertRaises(ValueError):
            self.helper.remaining(active, {**self.policy, "budget": float("nan")}, 60.62, 3600, rate)

    @unittest.skipUnless(sys.platform == "linux", "real CLI fixture requires WSL")
    def test_cleanup_only_ignores_new_campaign_budget_and_rate_limits(self):
        from tests.test_colab_transfer import WrapperTransfer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
            env = fixture.setup_root(root, "success")
            wrapper = root / "phase0/colab/colab_phase0.sh"
            allocated = subprocess.run(["bash", str(wrapper), "up", "aa-1", "A100"], env=env,
                                       capture_output=True, text=True, timeout=15)
            self.assertEqual(allocated.returncode, 0, allocated.stdout + allocated.stderr)
            command = ["bash", str(root / "phase0/colab/chain4.sh"), "--cleanup-only", "--lock", str(root / "lock"),
                       "--budget-units", "nan", "--min-balance-units", "nan", "--max-rate-units-hour", "0",
                       "--usage-seconds", "0", "--cleanup-seconds", "5"]
            cleaned = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(cleaned.returncode, 0, cleaned.stdout + cleaned.stderr)
            self.assertFalse((root / "active").exists())
            self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)

    @unittest.skipUnless(sys.platform == "linux", "real CLI fixture requires WSL")
    def test_live_usage_during_blocked_upload_stops_process_group_and_cleans_up(self):
        from tests.test_colab_transfer import WrapperTransfer, CLI
        for mode in ("low", "unknown", "rate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
                env = fixture.setup_root(root, "signal")
                injection = """
    if (root / 'pids').exists() and (root / 'active').exists():
        print('Current balance: BALANCE compute units\\nUsage rate: RATE/hr\\nActive assignments: 1')
        sys.exit(0)
""".replace("BALANCE", "16.00" if mode == "low" else "100.00")
                injection = injection.replace("RATE", "10000.00" if mode == "rate" else "5.30")
                if mode == "unknown": injection = injection.replace("        print(", "        # print(")
                fake = CLI.replace("elif args[0] == 'usage':", "elif args[0] == 'usage':" + injection)
                (root / ".local/bin/colab").write_text(f"#!{sys.executable}\n" + fake)
                command = ["bash", str(root / "phase0/colab/chain4.sh"), "--lock", str(root / "lock"),
                           "--hours", ".01", "--read-seconds", "2", "--up-seconds", "15", "--usage-seconds", ".1",
                           "--cleanup-seconds", "8", "--transfer-seconds", "2", "--grace", ".1"]
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=25)
                self.assertEqual(result.returncode, 65, result.stdout + result.stderr)
                self.assertIn("accounting unknown or reserve exhausted", result.stdout)
                self.assertTrue((root / "pids").exists())
                self.assertFalse((root / "active").exists())
                self.assertFalse((root / "boot").exists())
                calls = (root / "calls").read_text().splitlines()
                self.assertEqual(calls.count("new"), 1)
                self.assertEqual(calls.count("stop"), 1)
                self.assertGreaterEqual(calls.count("usage"), 4)
                for pid in json.loads((root / "pids").read_text()):
                    path = Path(f"/proc/{pid}/stat")
                    if path.exists():
                        self.assertIn(path.read_text().rsplit(")", 1)[1].split()[0], ("Z", "X", "x"))

    @unittest.skipUnless(sys.platform == "linux", "real CLI fixture requires WSL")
    def test_wrapper_refuses_before_new_and_real_rate_before_upload_cleanup_allowed(self):
        from tests.test_colab_transfer import WrapperTransfer, CLI
        for mode in ("reserve", "duration", "rate", "unknown"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
                env = fixture.setup_root(root, "success")
                env.update(DLLM_BUDGET_UNITS="45", DLLM_HOURS="7.8")
                if mode == "reserve": env["DLLM_BUDGET_UNITS"] = "60"
                if mode == "duration": env["DLLM_HOURS"] = "10"
                fake = CLI.replace("100.00 compute units", "60.62 compute units")
                if mode == "rate": fake = fake.replace("5.30/hr", "8.00/hr")
                if mode == "unknown":
                    fake = fake.replace("elif args[0] == 'usage':", "elif args[0] == 'usage':\n    sys.exit(0)")
                (root / ".local/bin/colab").write_text(f"#!{sys.executable}\n" + fake)
                wrapper = root / "phase0/colab/colab_phase0.sh"
                result = subprocess.run(["bash", str(wrapper), "up", "aa-1", "A100"], env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertNotIn("upload", calls)
                self.assertNotIn("boot", calls)
                self.assertEqual(calls.count("new"), int(mode == "rate"))
                if mode == "rate":
                    down = subprocess.run(["bash", str(wrapper), "down"], env=env,
                                          capture_output=True, text=True, timeout=15)
                    self.assertEqual(down.returncode, 0, down.stdout + down.stderr)
                    self.assertFalse((root / "active").exists())
