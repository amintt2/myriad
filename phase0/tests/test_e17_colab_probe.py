"""Synthetic observations and isolated fake CLI processes; never execute the remote probe locally."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

COLAB = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB))
import e17_probe
import e17_probe_remote
from budget import preflight
from e17_probe_remote import FLAGS, NAMESPACES


def compatible(request):
    return {**request, "schema": 1, "origin": "colab-exec", "kernel": "synthetic-kernel",
            "system": "Linux", "machine": "x86_64", "namespace_flags": FLAGS,
            "namespaces": dict.fromkeys(NAMESPACES, True), "namespace_combined": True,
            "landlock_abi": 7, "seccomp": True, "verdict": "compatible",
            "smoke": {"code": 0, "output_ok": True, "isolation": {
                "isolation_ready": "linux-shell-v2", "landlock_abi": 7, "seccomp": "on",
                "namespaces": NAMESPACES}}}


class Validation(unittest.TestCase):
    def test_ephemeral_remote_observations_without_landlock(self):
        source, vm_source = "synthetic local module", "synthetic vm module"
        request = {"sandbox_backend": "ephemeral-vm",
                   "local_linux_sha256": e17_probe.hashlib.sha256(source.encode()).hexdigest(),
                   "ephemeral_vm_sha256": e17_probe.hashlib.sha256(vm_source.encode()).hexdigest()}
        for code, expected in ((0, "compatible"), (1, "incompatible"), (124, "infrastructure-unknown")):
            with self.subTest(code=code):
                ready = {"isolation_ready": "ephemeral-vm-v1", "landlock_abi": None, "seccomp": "on",
                         "namespaces": NAMESPACES}
                env = mock.Mock(isolation=ready)
                env.run.return_value = (code, "E17-SYNTHETIC-OK\n" if code == 0 else "")
                local = SimpleNamespace(LocalLinuxEnv=mock.Mock(), VERSION="linux-shell-v2")
                vm = SimpleNamespace(colab_markers=lambda: {"hostname_file": True, "drive_binary": True},
                                     select_backend=mock.Mock(return_value=mock.Mock(return_value=env)))
                libc = mock.Mock()
                libc.syscall.return_value = -1
                with mock.patch.object(e17_probe_remote.platform, "system", return_value="Linux"), \
                     mock.patch.object(e17_probe_remote.platform, "machine", return_value="x86_64"), \
                     mock.patch.object(e17_probe_remote.ctypes, "CDLL", return_value=libc), \
                     mock.patch.object(e17_probe_remote.subprocess, "run",
                                       return_value=SimpleNamespace(returncode=0, stdout="SECCOMP-DENIED\n")), \
                     mock.patch.object(e17_probe_remote.importlib.util, "module_from_spec", side_effect=[local, vm]), \
                     mock.patch.object(e17_probe_remote.importlib.util, "spec_from_file_location",
                                       return_value=SimpleNamespace(loader=mock.Mock())), \
                     mock.patch.object(Path, "symlink_to"):
                    value = e17_probe_remote.observe(request, source, vm_source)
                self.assertEqual(value["verdict"], expected)
                self.assertEqual(value["landlock_abi"], -1)
                self.assertEqual(value["smoke"]["isolation"], ready)
                env.close.assert_called_once()
                local.LocalLinuxEnv.assert_not_called()
                vm.select_backend.assert_called_once_with("ephemeral-vm")
                e17_probe.validate_result(value, request)
        self.assertEqual(e17_probe_remote.observe(request, source, "wrong hash")["verdict"],
                         "infrastructure-unknown")

    def test_ephemeral_requires_markers_and_exact_receipt(self):
        request = {"sandbox_backend": "ephemeral-vm", "ephemeral_vm_sha256": "a" * 64}
        value = compatible(request)
        value.update(landlock_abi=-1, colab_markers={"hostname_file": True, "drive_binary": True})
        value["smoke"]["isolation"].update(isolation_ready="ephemeral-vm-v1", landlock_abi=None)
        self.assertIs(e17_probe.validate_result(value, request), value)
        for change in ({"colab_markers": {}}, {"colab_markers": {"hostname_file": 1, "drive_binary": True}},
                       {"seccomp": False}, {"namespace_combined": False},
                       {"smoke": compatible({})["smoke"]}, {"ephemeral_vm_sha256": "b" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                e17_probe.validate_result({**value, **change}, request)


    def test_smoke_observation_distinguishes_timeout_failure_and_success(self):
        ready = compatible({})["smoke"]["isolation"]
        source = "synthetic module fixture"
        request = {"local_linux_sha256": e17_probe.hashlib.sha256(source.encode()).hexdigest()}
        for code, expected in ((124, "infrastructure-unknown"), (1, "incompatible"), (0, "compatible")):
            with self.subTest(code=code):
                env = mock.Mock(isolation=ready)
                env.run.return_value = (code, "E17-SYNTHETIC-OK\n" if code == 0 else "")
                module = SimpleNamespace(LocalLinuxEnv=mock.Mock(return_value=env), VERSION="linux-shell-v2",
                                         IsolationError=type("SyntheticIsolationError", (RuntimeError,), {}))
                libc = mock.Mock()
                libc.syscall.return_value = 7
                with mock.patch.object(e17_probe_remote.platform, "system", return_value="Linux"), \
                     mock.patch.object(e17_probe_remote.platform, "machine", return_value="x86_64"), \
                     mock.patch.object(e17_probe_remote.ctypes, "CDLL", return_value=libc), \
                     mock.patch.object(e17_probe_remote.subprocess, "run",
                                       return_value=SimpleNamespace(returncode=0, stdout="SECCOMP-DENIED\n")), \
                     mock.patch.object(e17_probe_remote.importlib.util, "module_from_spec", return_value=module), \
                     mock.patch.object(e17_probe_remote.importlib.util, "spec_from_file_location",
                                       return_value=SimpleNamespace(loader=mock.Mock())), \
                     mock.patch.object(Path, "symlink_to"):
                    value = e17_probe_remote.observe(request, source)
                self.assertEqual(value["verdict"], expected)
                self.assertEqual(value["smoke"]["code"], code)
                env.close.assert_called_once()

    def test_compatible_and_negative_observations(self):
        request = {"campaign_id": "synthetic", "nonce": "fresh", "phase0": {"fingerprint": "a" * 64},
                   "local_linux_sha256": "b" * 64, "allocation_id": "synthetic-allocation"}
        value = compatible(request)
        self.assertIs(e17_probe.validate_result(value, request), value)
        for verdict in ("incompatible", "infrastructure-unknown"):
            negative = {**value, "verdict": verdict, "seccomp": None}
            self.assertEqual(e17_probe.validate_result(negative, request)["verdict"], verdict)
        for change in ({"nonce": "old"}, {"phase0": None}, {"origin": "wsl"}, {"seccomp": False},
                       {"landlock_abi": 2}, {"namespace_combined": False}, {"smoke": None},
                       {"namespaces": dict.fromkeys(NAMESPACES, 1)}, {"machine": "aarch64"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                e17_probe.validate_result({**value, **change}, request)

    def test_default_envelope_includes_accounting_margin(self):
        args = type("Args", (), dict(seconds=600, budget_units=4, min_balance_units=15,
                                    cleanup_seconds=1800, max_rate_units_hour=5.3, read_seconds=120, grace=5))()
        policy = e17_probe.Probe(args).policy
        usage = {"schema": 1, "balance_units": 59.62, "rate_units_hour": 0, "assignments": 0}
        self.assertEqual(preflight(usage, policy), 5.3)
        self.assertAlmostEqual(5.3 * (600 + 1800 + 161) / 3600, 3.770361111111111)
        with self.assertRaises(ValueError):
            preflight(usage, {**policy, "budget": 3.7})


FAKE = r'''import ast, json, os, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
mode = os.environ['MODE']
args = sys.argv[1:]
op = args[0]
active = root / 'active'
with (root / 'calls').open('a') as out: out.write(op + '\n')
if op == 'sessions':
    if mode == 'foreign-preflight': active.touch()
    if mode == 'session-read-once' and active.exists() and not (root / 'read-failed').exists():
        (root / 'read-failed').touch(); sys.exit(27)
    endpoint = 'foreign' if (root / 'replaced').exists() else 'synthetic'
    print(f'[phase0] {endpoint} | Hardware: L4 | Shape: Standard | Variant: GPU'
          if active.exists() else '[colab] No active sessions found on server.')
elif op == 'usage':
    if mode in ('usage-timeout-once', 'usage-timeout-always', 'usage-wait-term') and active.exists():
        if mode != 'usage-timeout-once' or not (root / 'usage-timeout').exists():
            (root / 'usage-timeout').touch()
            if mode == 'usage-wait-term': (root / 'usage-waiting').touch()
            sys.exit(124)
    if mode == 'usage-unknown' and active.exists(): sys.exit(0)
    if mode == 'usage-final-unknown' and (root / 'allocated').exists() and not active.exists(): sys.exit(0)
    if mode == 'usage-unknown-during' and (root / 'probe-started').exists() and active.exists(): sys.exit(0)
    balance = '16.00' if mode == 'low-balance' else '59.62'
    rate = '5.40' if mode == 'rate-ceiling' else '5.30'
    print(f'Current balance: {balance} compute units\nUsage rate: ' +
          (rate + '/hr\nActive assignments: 1' if active.exists() else '0.00/hr\nActive assignments: 0'))
elif op == 'new':
    if mode in ('new-timeout', 'new-term'):
        (root / 'new-pid').write_text(str(os.getpid()))
        (root / 'new-mask').write_text(Path('/proc/self/status').read_text())
        time.sleep(2 if mode == 'new-timeout' else .3)
        if mode == 'new-timeout': (root / 'late-allocation').touch()
    if mode != 'new-false-success': active.touch()
    (root / 'allocated').touch()
    if mode == 'new-ambiguous': sys.exit(27)
elif op == 'stop':
    if mode == 'stop-retry' and not (root / 'stop-failed').exists():
        (root / 'stop-failed').touch(); sys.exit(27)
    active.unlink(missing_ok=True)
    if mode == 'stop-ack-lost': sys.exit(27)
elif op == 'exec':
    if args[-1].endswith('snapshot.py'):
        if mode == 'slow-snapshot': time.sleep(1.2)
        if mode == 'snapshot-fail': sys.exit(27)
        if mode == 'replaced-session': (root / 'replaced').touch()
        print(json.dumps({'schema': 1, 'campaign_present': mode == 'foreign-campaign'}))
    else:
        if mode == 'slow-snapshot': time.sleep(1.2)
        (root / 'probe-started').touch()
        if mode in ('timeout', 'term', 'usage-unknown-during'): time.sleep(30)
        tree = ast.parse(Path(args[-1]).read_text())
        request = ast.literal_eval(tree.body[-1].value.args[0].right.args[0].args[0])
        flags = ['--user','--map-root-user','--mount','--net','--pid','--ipc','--uts','--fork','--kill-child=KILL']
        names = ['user','mount','pid','net','ipc','uts']
        value = {**request, 'schema': 1, 'origin': 'colab-exec', 'kernel': 'synthetic', 'system': 'Linux',
            'machine': 'x86_64', 'namespace_flags': flags, 'namespaces': dict.fromkeys(names, True),
            'namespace_combined': True, 'landlock_abi': 7, 'seccomp': True, 'verdict': 'compatible',
            'smoke': {'code': 0, 'output_ok': True, 'isolation': {'isolation_ready': 'linux-shell-v2',
                'landlock_abi': 7, 'seccomp': 'on', 'namespaces': names}}}
        if request.get('sandbox_backend') == 'ephemeral-vm':
            value.update(landlock_abi=-1, colab_markers={'hostname_file': True, 'drive_binary': True})
            value['smoke']['isolation'].update(isolation_ready='ephemeral-vm-v1', landlock_abi=None)
        if mode == 'incompatible': value.update(verdict='incompatible', namespace_combined=False)
        if mode == 'missing-ack': sys.exit(0)
        if mode == 'forged-success': value['smoke'] = None
        if mode == 'foreign-ack': value['nonce'] = 'old'
        print('E17_PROBE ' + json.dumps(value))
else: sys.exit(81)
'''

RUNNER = r'''import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import e17_probe
original = e17_probe.lifecycle.save
mode = os.environ['MODE']
if mode == 'slow-snapshot':
    original_run = e17_probe.run
    def run(command, **kwargs):
        if '--timeout' in command:
            kwargs['seconds'] = min(kwargs['seconds'], 2)
        return original_run(command, **kwargs)
    e17_probe.run = run  # Scale only the per-exec bound; keep actual helpers and the global budget.
def save(path, value):
    if mode == 'output-save-error' and Path(path).name == 'report.json':
        raise OSError('synthetic report persistence failure')
    stage = 'owned' if value.get('status') == 'owned' else 'returned' if value.get('allocation_returned') else 'pending'
    if mode.startswith('persist-' + stage) and Path(path).name == 'receipt':
        if mode.endswith('-after'): original(path, value)
        if '-term' in mode: os.kill(os.getpid(), signal.SIGTERM)
        raise OSError('synthetic persistence failure')
    return original(path, value)
e17_probe.lifecycle.save = save
sys.exit(e17_probe.main())
'''


@unittest.skipUnless(sys.platform == "linux", "fake CLI lifecycle processes require Linux/WSL")
class Processes(unittest.TestCase):
    def test_usage_timeout_retries_then_valid_or_global_cleanup(self):
        for mode, success in (("usage-timeout-once", True), ("usage-timeout-always", False)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                command.extend(("--seconds", "6", "--cleanup-seconds", "3"))
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
                report, calls = self.check(root, result.stdout + result.stderr, success)
                self.assertEqual(result.returncode, 0 if success else 1)
                self.assertEqual(calls.count("new"), 1)
                self.assertGreater(calls.count("usage"), 3)
                if not success:
                    self.assertNotIn("exec", calls)
                    self.assertEqual(report["error"], "UsageTimeout")

    def test_term_interrupts_usage_wait_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "usage-wait-term")
            command.extend(("--seconds", "10", "--retry-seconds", "2", "--cleanup-seconds", "3"))
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / "usage-waiting").exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "usage-waiting").exists())
                os.kill(process.pid, signal.SIGTERM)
                output, _ = process.communicate(timeout=8)
                report, calls = self.check(root, output)
                self.assertEqual(process.returncode, 1, output)
                self.assertEqual(report["error"], "SystemExit")
                self.assertNotIn("exec", calls)
                self.assertEqual(calls.count("new"), 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_explicit_ephemeral_uses_same_allocation_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "compatible")
            command.extend(("--sandbox-backend", "ephemeral-vm"))
            process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            value = json.loads((root / "report.json").read_text())
            self.assertTrue(value["closed"])
            self.assertEqual(value["sandbox_backend"], "ephemeral-vm")
            self.assertEqual(value["probe"]["smoke"]["isolation"]["isolation_ready"], "ephemeral-vm-v1")
            self.assertEqual(len(value["probe"]["ephemeral_vm_sha256"]), 64)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(calls.count("new"), 1)
            self.assertEqual(calls.count("stop"), 1)


    def test_new_timeout_kills_official_cli_before_absence_and_no_late_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "new-timeout")
            command.extend(("--read-seconds", ".5", "--grace", ".2"))
            # The inner safe_cli deadline stays at 120 s: only the outer controller expires.
            env["DLLM_CLI_SECONDS"] = "120"
            env["DLLM_CLI_GRACE"] = ".2"
            pid = None
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=8)
                pid = int((root / "new-pid").read_text())
                report, calls = self.check(root, result.stdout + result.stderr, stop=False)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(calls.count("new"), 1)
                mask = next(line for line in (root / "new-mask").read_text().splitlines()
                            if line.startswith("SigBlk:")).split()[1]
                self.assertEqual(int(mask, 16) & ((1 << (signal.SIGTERM - 1)) | (1 << (signal.SIGINT - 1))), 0)
                self.assertFalse(Path(f"/proc/{pid}").exists(), "official CLI survived outer timeout")
                time.sleep(2.1)
                self.assertFalse((root / "late-allocation").exists())
                self.assertFalse((root / "active").exists())
            finally:
                if pid is None and (root / "new-pid").exists(): pid = int((root / "new-pid").read_text())
                if pid is not None:
                    try: os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError: pass

    def test_term_during_new_keeps_confirmed_transition_and_cleanup_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "new-term")
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / "new-pid").exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "new-pid").exists())
                os.kill(process.pid, signal.SIGTERM)
                output, _ = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 1, output)
                report, calls = self.check(root, output)
                self.assertEqual(report["error"], "SystemExit")
                self.assertEqual(calls.count("new"), 1)
                self.assertFalse(Path("/proc/" + (root / "new-pid").read_text()).exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_slow_snapshot_usage_read_preserves_global_budget_for_next_exec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "slow-snapshot")
            command.extend(("--seconds", "8"))
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
            report, calls = self.check(root, result.stdout + result.stderr, success=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(calls.count("exec"), 2)
            self.assertGreaterEqual(calls.count("usage"), 5)

    def fixture(self, root, mode):
        bins = root / ".local/bin"
        bins.mkdir(parents=True)
        fake = bins / "colab"
        fake.write_text(f"#!{sys.executable}\n" + FAKE, encoding="utf-8")
        fake.chmod(0o755)
        runner = root / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        env = {**os.environ, "HOME": str(root), "PATH": str(bins) + ":" + os.environ["PATH"],
               "TEST_ROOT": str(root), "MODE": mode, "DLLM_REQUIRE_OWNER": "0"}
        for key in ("DLLM_LIFECYCLE_LOCK_FD", "DLLM_OPERATION_IDENTITY", "DLLM_CAMPAIGN_ID"):
            env.pop(key, None)
        command = [sys.executable, str(runner), str(COLAB), "--receipt", str(root / "receipt"),
                   "--output", str(root / "report.json"), "--seconds", "4", "--read-seconds", "1",
                   "--cleanup-seconds", "1.5", "--retry-seconds", ".01", "--grace", ".1",
                   "--usage-seconds", ".1"]
        return env, command

    def check(self, root, output, success=False, stop=True):
        report = json.loads((root / "report.json").read_text())
        calls = (root / "calls").read_text().splitlines()
        self.assertLessEqual(calls.count("new"), 1, output)
        self.assertEqual("stop" in calls, stop, output)
        self.assertTrue(report["closed"], output)
        self.assertTrue(report["absence_confirmed"], output)
        self.assertEqual(calls[-1], "usage", output)
        self.assertEqual(report["usage"][-1]["assignments"], 0, output)
        self.assertFalse((root / "active").exists(), output)
        self.assertFalse((root / "receipt").exists(), output)
        self.assertEqual(report["verdict"] == "compatible", success, output)
        return report, calls

    def test_results_errors_and_cleanup_with_exactly_one_allocation(self):
        for mode in ("success", "incompatible", "missing-ack", "forged-success", "foreign-ack",
                     "snapshot-fail", "usage-unknown", "usage-unknown-during", "rate-ceiling",
                     "session-read-once", "stop-retry", "stop-ack-lost", "timeout", "new-false-success"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
                output = result.stdout + result.stderr
                success = mode in ("success", "stop-retry", "stop-ack-lost")
                self.assertEqual(result.returncode, 0 if success else 1, output)
                report, calls = self.check(root, output, success, mode != "new-false-success")
                self.assertEqual(calls.count("new"), 1)
                if mode == "incompatible": self.assertEqual(report["verdict"], "incompatible")
                if mode == "stop-retry": self.assertGreaterEqual(calls.count("stop"), 2)
                if mode == "usage-unknown-during": self.assertLess(calls.index("stop"), 15)

    def test_persistence_exception_or_term_after_new_always_closes(self):
        for stage in ("returned", "owned"):
            for kind in ("error", "term"):
                for suffix in ("", "-after"):
                    mode = f"persist-{stage}-{kind}{suffix}"
                    with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        env, command = self.fixture(root, mode)
                        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.check(root, result.stdout + result.stderr)

    def test_preflight_refuses_without_allocating_or_stopping(self):
        for mode in ("foreign-preflight", "low-balance", "persist-pending-error", "persist-pending-error-after"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertNotIn("new", calls)
                self.assertNotIn("stop", calls)

    def test_report_persistence_failure_happens_after_closed_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "output-save-error")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 65, result.stdout + result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(calls.count("new"), 1)
            self.assertIn("stop", calls)
            self.assertEqual(calls[-1], "usage")
            self.assertFalse((root / "receipt").exists())
            self.assertFalse((root / "active").exists())
            journal = [json.loads(line) for line in (root / "receipt.usage.jsonl").read_text().splitlines()]
            self.assertTrue(journal[-1]["release_confirmed"])
            self.assertEqual(journal[-1]["after"]["assignments"], 0)

    def test_shared_lifecycle_lock_refuses_before_any_cli_call(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "success")
            with (root / "receipt.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
            self.assertFalse((root / "calls").exists())

    def test_ambiguous_new_foreign_session_and_unknown_final_usage_keep_failure(self):
        for mode in ("new-ambiguous", "replaced-session", "foreign-campaign", "usage-final-unknown"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertEqual(calls.count("new"), 1)
                self.assertEqual("stop" in calls, mode == "usage-final-unknown")
                report = json.loads((root / "report.json").read_text())
                self.assertFalse(report["closed"])
                self.assertEqual(report["verdict"], "infrastructure-unknown")
                self.assertTrue((root / "receipt").exists())
                self.assertEqual(calls[-1], "usage")

    def test_external_term_during_probe_and_wrapper_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "term")
            command = ["bash", str(COLAB / "colab_phase0.sh"), "e17-probe", *command[3:]]
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / "probe-started").exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "probe-started").exists())
                # Signal the Python controller, not its bash parent or a remote session.
                children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
                self.assertEqual(len(children), 1)
                os.kill(int(children[0]), signal.SIGTERM)
                output, _ = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 1, output)
                self.check(root, output)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
