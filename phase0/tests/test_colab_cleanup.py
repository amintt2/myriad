"""Real supervisor, lifecycle lock and wrapper regressions with isolated, fail-closed CLI fixtures."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests import test_colab_transfer as transfer


INJECT = r'''
with (root / 'operations').open('a') as out:
    out.write(json.dumps({'operation': args[0], 'mode': mode,
                          'active': (root / 'active').exists(), 'boot': (root / 'boot').exists(),
                          'campaign_id': os.environ.get('DLLM_CAMPAIGN_ID')}) + '\n')
if mode == 'foreign-race' and args[0] == 'sessions' and not (root / 'released-gate').exists():
    (root / 'ready').touch()
    while not (root / 'release').exists(): time.sleep(.01)
    (root / 'released-gate').touch()
if args[0] == 'usage' and mode == 'expire-up-empty' and (root / 'boot').exists():
    sys.exit(0)
'''

FAKE = transfer.CLI.replace("if args[0] == 'sessions':", INJECT + "\nif args[0] == 'sessions':")
FAKE = FAKE.replace("    elif args[-1].endswith('snapshot.py'):\n"
                    "        print(json.dumps({'schema': 1, 'campaign_present': False}))", r'''
    elif args[-1].endswith('snapshot.py'):
        if not (root / 'boot').exists():
            print(json.dumps({'schema': 1, 'campaign_present': False}))
        else:
            jobs = json.loads(os.environ['EXPECTED'])
            print(json.dumps({'schema': 1, 'campaign_present': True,
                'bootstrap': {'step': 'lanceur en marche', 'plan': 'aa-1', 'pid': 42,
                              'campaign_id': os.environ['DLLM_CAMPAIGN_ID']},
                'jobs': {job: 'prévu' for job in jobs}, 'diagnostics': {'launcher_alive': True}}))
            (root / 'observed').touch()
            if mode == 'expire-observed': (root / 'active').unlink()
            if mode == 'replace-observed': (root / 'replaced').touch()
''')
FAKE = FAKE.replace("        (root / 'boot').touch()", "        (root / 'boot').touch()\n"
                    "        if mode in ('expire-up', 'expire-up-empty'): (root / 'active').unlink()")
FAKE = FAKE.replace("print('[phase0] synthetic | Hardware: A100 | Shape: Standard | Variant: GPU'",
                    "print('[phase0] ' + ('foreign' if (root / 'replaced').exists() else 'synthetic') "
                    "+ ' | Hardware: A100 | Shape: Standard | Variant: GPU'")

RUNNER = r'''import json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import campaign
campaign.WRAPPER = Path(os.environ['TEST_WRAPPER'])
if os.environ.get('ADOPTION_GATE') == '1':
    original = campaign.Supervisor.call
    def gated(self, operation, seconds, structured=False):
        if operation == 'down':
            root = Path(os.environ['TEST_ROOT'])
            (root / 'adopted').write_text(self.env['DLLM_OPERATION_IDENTITY'])
            (root / 'ready').touch()
            while not (root / 'release').exists(): time.sleep(.01)
        return original(self, operation, seconds, structured)
    campaign.Supervisor.call = gated  # Pause at a boundary, preserve the actual wrapper/down implementation.
sys.exit(campaign.main())
'''


@unittest.skipUnless(sys.platform == "linux", "real lifecycle processes require Linux/WSL")
class Cleanup(unittest.TestCase):
    def fixture(self, root, mode):
        helper = transfer.WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
        env = helper.setup_root(root, mode)
        fake = root / ".local/bin/colab"
        fake.write_text(f"#!{sys.executable}\n" + FAKE, encoding="utf-8")
        runner = root / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        sys.path.insert(0, str(transfer.COLAB))
        import campaign
        env.update(TEST_WRAPPER=str(root / "phase0/colab/colab_phase0.sh"),
                   EXPECTED=json.dumps(sorted(campaign.EXPECTED)), DLLM_CLEANUP_SECONDS="5")
        command = [sys.executable, str(runner), str(transfer.COLAB), "--receipt", str(root / "receipt"),
                   "--lock", str(root / "supervisor.lock"), "--budget-units", "20", "--hours", ".01",
                   "--read-seconds", "2", "--up-seconds", "15", "--cleanup-seconds", "5",
                   "--transfer-seconds", "2", "--poll-seconds", ".01", "--pull-seconds", "3600", "--grace", ".1"]
        return env, command

    def operations(self, root):
        path = root / "operations"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait_ready(self, root, process):
        deadline = time.monotonic() + 10
        while not (root / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue((root / "ready").exists())

    def finish(self, root, process):
        (root / "release").touch()
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    def test_foreign_manual_up_before_first_pin_is_never_adopted_by_supervisor_down(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "foreign-race")
            # The deliberately paused read must cover the competing manual up's entire 15-second budget.
            command.extend(("--read-seconds", "20"))
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                self.wait_ready(root, process)
                manual_env = {**env, "MODE": "success", "DLLM_CAMPAIGN_ID": "manual-campaign"}
                allocated = subprocess.run(["bash", env["TEST_WRAPPER"], "up", "aa-1", "A100"],
                                           env=manual_env, capture_output=True, text=True, timeout=15)
                self.assertEqual(allocated.returncode, 0, allocated.stdout + allocated.stderr)
                receipt = (root / "receipt").read_bytes()
                (root / "release").touch()
                output, _ = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 73, output)
                self.assertNotIn("wrapper down", output)
                self.assertEqual((root / "receipt").read_bytes(), receipt)
                self.assertTrue((root / "active").exists())
                calls = [c for c in self.operations(root) if c["mode"] == "foreign-race"]
                self.assertEqual([c["operation"] for c in calls], ["sessions"])
                # Even a direct call at the next layer must reject the unexpected campaign without a pin.
                refused = subprocess.run(["bash", env["TEST_WRAPPER"], "down"],
                                         env={**env, "DLLM_SUPERVISED": "1", "DLLM_CAMPAIGN_ID": calls[0]["campaign_id"]},
                                         capture_output=True, text=True, timeout=3)
                self.assertEqual(refused.returncode, 73, refused.stdout + refused.stderr)
                self.assertEqual([c for c in self.operations(root) if c["mode"] == "foreign-race"], calls)
                self.assertEqual((root / "receipt").read_bytes(), receipt)
                cleaned = subprocess.run([*command, "--cleanup-only"], env=manual_env,
                                         capture_output=True, text=True, timeout=8)
                self.assertEqual(cleaned.returncode, 0, cleaned.stdout + cleaned.stderr)
                self.assertFalse((root / "active").exists())
                self.assertFalse((root / "receipt").exists())
            finally:
                self.finish(root, process)

    def test_expiration_after_up_or_observation_keeps_failure_and_accounts_without_stop(self):
        for mode in ("expire-up", "expire-observed", "replace-observed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 73 if mode == "replace-observed" else 1,
                                 result.stdout + result.stderr)
                calls = self.operations(root)
                self.assertNotIn("stop", [c["operation"] for c in calls])
                self.assertNotIn("download", [c["operation"] for c in calls])
                self.assertNotIn("SUCCESS:", result.stdout)
                if mode == "replace-observed":
                    self.assertTrue((root / "active").exists())
                    self.assertEqual(json.loads((root / "receipt").read_text())["status"], "foreign-campaign")
                    self.assertEqual(sum(c["operation"] == "usage" and c["boot"] and not c["active"]
                                         for c in calls), 0)
                else:
                    self.assertFalse((root / "receipt").exists())
                    self.assertIn("explicitly absent", result.stdout)
                    self.assertIn("explicit absence confirmed", result.stdout)
                    evidence = [json.loads(line) for line in (root / "receipt.usage.jsonl").read_text().splitlines()]
                    self.assertTrue(evidence[-1]["release_confirmed"])
                    self.assertIsNotNone(evidence[-1]["after"])
                    self.assertEqual(sum(c["operation"] == "usage" and c["boot"] and not c["active"]
                                         for c in calls), 1)
                if mode != "expire-up": self.assertTrue((root / "observed").exists())

    def test_expiration_with_accounting_failure_retries_only_usage_on_cleanup_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "expire-up-empty")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            saved = json.loads((root / "receipt").read_text())
            self.assertEqual(saved["status"], "released/accounting-pending")
            self.assertTrue(saved["release_confirmed"])
            self.assertIn("release confirmed, accounting pending", result.stdout)
            calls = self.operations(root)
            self.assertEqual(sum(c["operation"] == "usage" and c["boot"] and not c["active"] for c in calls), 3)
            self.assertNotIn("stop", [c["operation"] for c in calls])
            retried = subprocess.run([*command, "--cleanup-only"], env={**env, "MODE": "success"},
                                     capture_output=True, text=True, timeout=8)
            self.assertEqual(retried.returncode, 0, retried.stdout + retried.stderr)
            self.assertEqual([c["operation"] for c in self.operations(root)[len(calls):]], ["usage"])
            self.assertFalse((root / "receipt").exists())

    def test_cleanup_only_pins_under_shared_lock_and_rejects_replacement_before_down_entry(self):
        for pending in (False, True):
            with self.subTest(accounting_pending=pending), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, "success")
                allocated = subprocess.run(["bash", env["TEST_WRAPPER"], "up", "aa-1", "A100"], env=env,
                                           capture_output=True, text=True, timeout=15)
                self.assertEqual(allocated.returncode, 0, allocated.stdout + allocated.stderr)
                saved = json.loads((root / "receipt").read_text())
                if pending:
                    (root / "active").unlink()
                    saved.update(status="released/accounting-pending", release_confirmed=True)
                    (root / "receipt").write_text(json.dumps(saved))
                calls = self.operations(root)
                process = subprocess.Popen([*command, "--cleanup-only"], env={**env, "ADOPTION_GATE": "1"},
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    self.wait_ready(root, process)
                    pinned = json.loads((root / "adopted").read_text())
                    self.assertEqual(pinned["allocation_id"], saved["allocation_id"])
                    for operation in (("down",), ("up", "aa-1", "A100")):
                        blocked = subprocess.run(["bash", env["TEST_WRAPPER"], *operation], env=env,
                                                 capture_output=True, text=True, timeout=3)
                        self.assertEqual(blocked.returncode, 73, blocked.stdout + blocked.stderr)
                        self.assertIn("lifecycle operation holds the lock", blocked.stdout)
                    replacement = {**saved, "allocation_id": "replacement-allocation"}
                    (root / "receipt").write_text(json.dumps(replacement))
                    (root / "release").touch()
                    output, _ = process.communicate(timeout=8)
                    self.assertEqual(process.returncode, 73, output)
                    self.assertEqual(self.operations(root), calls)
                    self.assertEqual(json.loads((root / "receipt").read_text()), replacement)
                    self.assertEqual((root / "active").exists(), not pending)
                finally:
                    self.finish(root, process)
