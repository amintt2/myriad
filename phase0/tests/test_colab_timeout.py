"""Exercise the Colab wrapper with a fake local CLI and short real deadlines (Linux/WSL, no Colab)."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


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
                self.assertEqual((rc, out, err), (23, "sortie CLI\n", "message CLI\n"))

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
