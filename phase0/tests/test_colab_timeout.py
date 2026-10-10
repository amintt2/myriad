"""Exercise the Colab wrapper with a fake local CLI and short real deadlines (Linux/WSL, no Colab)."""
import json
import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

COLAB = Path(__file__).resolve().parents[1] / "colab"


WRAPPER = Path(__file__).resolve().parents[1] / "colab" / "colab_phase0.sh"
CLI = '''#!/usr/bin/python3
import json, os, signal, sys, time
from pathlib import Path

args = sys.argv[1:]
expected = ['exec', '-s', 'phase0', '--timeout', '60']
if os.environ['ACTION'] == 'diagnose':
    expected += ['-f', os.environ['DIAGNOSE']]
if args != expected:
    sys.exit(80)
if os.environ['ACTION'] == 'status':
    source = sys.stdin.read()
    if 'colab_status.json' not in source or 'nvidia-smi' not in source:
        sys.exit(81)
mode = os.environ['MODE']
if mode in ('success', 'error'):
    print('sortie CLI')
    print('message CLI', file=sys.stderr)
    sys.exit(0 if mode == 'success' else 23)
if mode == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.fork()
if child == 0:
    time.sleep(60)
    sys.exit(0)
pids = [os.getpid(), child]
ignored = []
for pid in pids:
    status = Path(f'/proc/{pid}/status').read_text().splitlines()
    mask = int(next(line.split()[1] for line in status if line.startswith('SigIgn:')), 16)
    ignored.append(bool(mask & (1 << (signal.SIGTERM - 1))))
Path(os.environ['STATE_FILE']).write_text(json.dumps({'pids': pids, 'ignores_term': ignored}))
time.sleep(60)
'''
# Run the actual supervisor; shorten only the test's wall-clock deadlines.
SUPERVISOR = '''#!/usr/bin/python3
import importlib.util, os, sys
if sys.argv[1].endswith('safe_cli.py'):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
if sys.argv[1] != os.environ['SUPERVISOR']:
    sys.exit(82)
spec = importlib.util.spec_from_file_location('read_timeout', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.exit(module.run(sys.argv[2:], seconds=0.5, grace=0.5))
'''


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X", "x")
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform == "linux", "Process groups: run under WSL/Linux")
class ColabTimeout(unittest.TestCase):
    def test_finished_child_does_not_turn_late_monitor_into_success(self):
        sys.path.insert(0, str(COLAB))
        from read_timeout import run
        def monitor(deadline):
            self.assertGreater(deadline, started)
            time.sleep(.3)
            return 100
        started = time.monotonic()
        code = run([sys.executable, "-c", "import time; time.sleep(.1)"], seconds=.05,
                   grace=.01, monitor=monitor)
        self.assertEqual(code, 124)
        self.assertGreaterEqual(time.monotonic() - started, .3)

    def test_nested_usage_is_bounded_by_operation_before_read_and_global_limits(self):
        sys.path.insert(0, str(COLAB))
        import campaign
        cli = '''import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
operation = sys.argv[1]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.fork()
if child == 0: time.sleep(60); sys.exit(0)
(root / (operation + '.pids')).write_text(json.dumps([os.getpid(), child]))
time.sleep(.1 if operation == 'snapshot' else 3)
(root / (operation + '.finished')).touch()
print(json.dumps({'schema': 1, 'balance_units': 100, 'rate_units_hour': 5.3, 'assignments': 1}))
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "cli.py"
            fake.write_text(cli)
            wrapper = root / "wrapper.sh"
            wrapper.write_text('exec "$PYTHON" "$TEST_ROOT/cli.py" "$@"\n')
            args = argparse.Namespace(hours=.01, budget_units=20, cleanup_seconds=3, min_balance_units=15,
                                      max_rate_units_hour=5.3, usage_seconds=.1, read_seconds=5, grace=.02)
            supervisor = campaign.Supervisor(args, root / "receipt.json")
            supervisor.env.update(TEST_ROOT=directory, PYTHON=sys.executable)
            supervisor.receipt.write_text(json.dumps({'status': 'owned', 'campaign_id': supervisor.campaign_id,
                'phase0': {'fingerprint': 'a' * 64, 'hardware': 'A100'}, 'usage_before': {'balance_units': 100},
                'budget_policy': {'budget': 20, 'minimum': 15, 'rate': 5.3, 'hours': .01,
                                  'cleanup': 3, 'margin': 1}}))
            supervisor.monitoring = True
            started = time.monotonic()
            try:
                with mock.patch.object(campaign, "WRAPPER", wrapper), \
                     self.assertRaises(campaign.AccountingFailure) as failure:
                    supervisor.call("snapshot", .5, True)
                elapsed = time.monotonic() - started
                self.assertEqual(failure.exception.code, 124)
                self.assertGreaterEqual(elapsed, .5)
                self.assertLess(elapsed, 1.5, "nested usage must not wait for its own 5 s or the global 36 s")
                self.assertTrue((root / "snapshot.finished").exists(), "parent must finish during slow usage")
                self.assertFalse((root / "usage-json.finished").exists())
                for operation in ("snapshot", "usage-json"):
                    self.assertTrue((root / (operation + ".pids")).exists())
                    pids = json.loads((root / (operation + ".pids")).read_text())
                    deadline = time.monotonic() + 1
                    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertFalse(any(alive(pid) for pid in pids))
            finally:
                for path in root.glob("*.pids"):
                    for pid in json.loads(path.read_text()):
                        if alive(pid):
                            try: os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError: pass

    def test_accounting_timeout_is_not_retried_by_observe_or_execute(self):
        sys.path.insert(0, str(COLAB))
        import campaign
        cli = '''import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
operation = sys.argv[1]
calls = root / 'calls'
previous = calls.read_text().splitlines() if calls.exists() else []
with calls.open('a') as output: output.write(operation + '\\n')
session = {'fingerprint': 'a' * 64, 'hardware': 'A100'}
if operation == 'sessions-json':
    print(json.dumps({'schema': 1, 'phase0': session if (root / 'active').exists() else None}))
elif operation == 'up':
    (root / 'active').touch()
    Path(os.environ['DLLM_CAMPAIGN_RECEIPT']).write_text(json.dumps({'status': 'owned',
        'campaign_id': os.environ['DLLM_CAMPAIGN_ID'], 'phase0': session, 'usage_before': {'balance_units': 100},
        'budget_policy': {'budget': 20, 'minimum': 15, 'rate': 5.3, 'hours': .01, 'cleanup': 3, 'margin': 1}}))
elif operation == 'usage-json':
    if os.environ['MODE'] == 'execute' and previous.count('usage-json') < 2:
        print(json.dumps({'schema': 1, 'balance_units': 100, 'rate_units_hour': 5.3, 'assignments': 1}))
    else:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = os.fork()
        if child == 0: time.sleep(60); sys.exit(0)
        (root / 'usage.pids').write_text(json.dumps([os.getpid(), child]))
        (root / 'usage.started').write_text(str(time.monotonic()))
        time.sleep(10)
elif operation == 'snapshot': print(json.dumps({'schema': 1, 'campaign_present': False}))
elif operation == 'pull': pass
elif operation == 'down':
    (root / 'active').unlink()
    (root / 'cleaned').write_text(str(time.monotonic()))
else: sys.exit(80)
'''
        for mode in ("observe", "execute"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "cli.py").write_text(cli)
                wrapper = root / "wrapper.sh"
                wrapper.write_text('exec "$PYTHON" "$TEST_ROOT/cli.py" "$@"\n')
                args = argparse.Namespace(hours=.01, budget_units=20, cleanup_seconds=3, min_balance_units=15,
                    max_rate_units_hour=5.3, usage_seconds=.001, read_seconds=.4, grace=.02, read_failures=3,
                    poll_seconds=.1, pull_seconds=1, up_seconds=2, transfer_seconds=.4, gpu="A100")
                supervisor = campaign.Supervisor(args, root / "receipt.json")
                supervisor.env.update(TEST_ROOT=directory, PYTHON=sys.executable, MODE=mode)
                if mode == "observe":
                    (root / "active").touch()
                    supervisor.receipt.write_text(json.dumps({'status': 'owned',
                        'campaign_id': supervisor.campaign_id,
                        'phase0': {'fingerprint': 'a' * 64, 'hardware': 'A100'}}))
                    supervisor.monitoring = True
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                try:
                    with mock.patch.object(campaign, "WRAPPER", wrapper):
                        if mode == "observe":
                            with self.assertRaises(campaign.AccountingFailure) as failure:
                                supervisor.observe()
                            self.assertEqual(failure.exception.code, 124)
                        else:
                            with mock.patch.object(supervisor, "observe", wraps=supervisor.observe) as observe:
                                self.assertEqual(supervisor.execute(), 124)
                            self.assertEqual(observe.call_count, 1)
                    calls = (root / "calls").read_text().splitlines()
                    self.assertEqual(calls.count("usage-json"), 1 if mode == "observe" else 3)
                    if mode == "execute":
                        self.assertEqual(calls.count("up"), 1)
                        self.assertEqual(calls.count("down"), 1)
                        self.assertIn("pull", calls)
                        self.assertFalse((root / "active").exists())
                        delay = float((root / "cleaned").read_text()) - float((root / "usage.started").read_text())
                        self.assertLess(delay, 1.5, "cleanup must follow the first failed accounting read")
                    pids = json.loads((root / "usage.pids").read_text())
                    deadline = time.monotonic() + 1
                    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertFalse(any(alive(pid) for pid in pids))
                finally:
                    for sig, handler in handlers.items(): signal.signal(sig, handler)
                    if (root / "usage.pids").exists():
                        for pid in json.loads((root / "usage.pids").read_text()):
                            if alive(pid):
                                try: os.kill(pid, signal.SIGKILL)
                                except ProcessLookupError: pass

    def run_wrapper(self, action, mode):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            bins = home / ".local" / "bin"
            bins.mkdir(parents=True)
            for name, source in (("colab", CLI), ("python3", SUPERVISOR)):
                path = bins / name
                path.write_text(source)
                path.chmod(0o755)
            state = home / "pids.json"
            env = {**os.environ, "HOME": d, "ACTION": action, "MODE": mode, "STATE_FILE": str(state),
                   "DLLM_REPO": str(WRAPPER.parents[2]), "DIAGNOSE": str(WRAPPER.parent / "diagnose.py"),
                   "SUPERVISOR": str(WRAPPER.parent / "read_timeout.py")}
            p = subprocess.Popen(["bash", str(WRAPPER), action], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, start_new_session=True)
            t0 = time.monotonic()
            try:
                out, err = p.communicate(timeout=10)
                elapsed = time.monotonic() - t0
                if mode in ("block", "ignore"):
                    self.assertTrue(state.exists(), "the fake CLI must start before the deadline")
                    observed = json.loads(state.read_text())
                    pids = observed["pids"]
                    self.assertEqual(observed["ignores_term"], [mode == "ignore"] * 2)
                    deadline = time.monotonic() + 2
                    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertFalse(any(alive(pid) for pid in pids), "CLI and child must both be stopped")
                return p.returncode, out, err, elapsed
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                if state.exists():  # cleanup even when a regression leaves a fake CLI or child alive
                    for pid in json.loads(state.read_text())["pids"]:
                        if alive(pid):
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                p.communicate()

    def test_success_output_unchanged(self):
        for action in ("status", "diagnose"):
            with self.subTest(action=action):
                rc, out, err, _ = self.run_wrapper(action, "success")
                self.assertEqual((rc, out, err), (0, "sortie CLI\n", "message CLI\n"))

    def test_cli_error_propagated(self):
        for action in ("status", "diagnose"):
            with self.subTest(action=action):
                rc, out, err, _ = self.run_wrapper(action, "error")
                self.assertEqual((rc, out, err), (23, "sortie CLI\n", "message CLI\ncolab exec: exit 23\n"))

    def test_connection_block_stops_group(self):
        for action in ("status", "diagnose"):
            with self.subTest(action=action):
                rc, _, _, elapsed = self.run_wrapper(action, "block")
                self.assertEqual(rc, 124)
                self.assertLess(elapsed, 8)

    def test_term_resistant_cli_and_child_killed(self):
        for action in ("status", "diagnose"):
            with self.subTest(action=action):
                rc, _, _, elapsed = self.run_wrapper(action, "ignore")
                self.assertEqual(rc, 124)
                self.assertLess(elapsed, 8)


if __name__ == "__main__":
    unittest.main()
