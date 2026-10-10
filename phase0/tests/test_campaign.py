"""Real Linux processes with a fake wrapper only; no cloud, dataset, model or candidate execution."""
import importlib.util
import ast
import json
import os
import signal
import shutil
import glob
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import run_aa
from tests.test_aa import FakeOut, synthetic_problem

COLAB = Path(__file__).resolve().parents[1] / "colab"


def load(name):
    spec = importlib.util.spec_from_file_location(name, COLAB / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Measurements(unittest.TestCase):
    def test_bootstrap_restores_timing_as_companion_without_metadata(self):
        source = COLAB.parent / "colab_bootstrap.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.Try))
        loop = next(n for n in main.body if isinstance(n, ast.For))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoints, results = root / "checkpoints", root / "phase0" / "results"
            checkpoints.mkdir()
            results.mkdir(parents=True)
            name = "aa_synthetic_colab_scicode_dev.jsonl"
            for suffix, text in (("", '{"id": "synthetic"}\n'), (".meta.json", '{"synthetic": true}'),
                                 (".timing.jsonl", '{"wall_s": 1}\n')):
                (checkpoints / (name + suffix)).write_text(text)
            namespace = {"glob": glob, "os": os, "shutil": shutil, "WORK": root.as_posix(),
                         "RESULTS": results.as_posix(), "restored": []}
            exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), "exec"), namespace)
            self.assertEqual(namespace["restored"], [name])
            self.assertEqual((results / (name + ".timing.jsonl")).read_text(), '{"wall_s": 1}\n')

    def test_bootstrap_refusal_never_updates_preexisting_status(self):
        source = COLAB.parent / "colab_bootstrap.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.Try))
        namespace = {"launcher_alive": lambda: "123", "status": mock.Mock(), "print": mock.Mock()}
        with self.assertRaises(SystemExit) as raised:
            exec(compile(ast.Module(body=main.body[:2], type_ignores=[]), str(source), "exec"), namespace)
        self.assertEqual(raised.exception.code, 73)
        namespace["status"].assert_not_called()

    def test_snapshot_does_not_hide_read_errors(self):
        module = load("snapshot")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertFalse(module.snapshot(root)["campaign_present"])
            results = root / "phase0" / "results"
            results.mkdir(parents=True)
            status = results / "colab_bootstrap_status.json"
            status.write_text("{")
            with self.assertRaises(ValueError): module.snapshot(root)
            with mock.patch.object(Path, "read_text", side_effect=PermissionError("synthetic")):
                with self.assertRaises(PermissionError): module.snapshot(root)

    def test_final_pull_requires_complete_current_artifacts_and_provenance(self):
        verifier = load("verify_e12_pull")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            provenance = {"schema": 1, "plan": "aa-1", "benches": ["scicode"],
                          "sources": {"synthetic.py": {"sha256": "a" * 64}}}
            expected = root / "expected.json"
            expected.write_text(json.dumps(provenance))
            (root / "e12_campaign_sources.json").write_text(json.dumps(provenance))
            names = [("sciexec_oracle_colab_scicode_dev.jsonl", 50)]
            for model in verifier.SOLO:
                for split, n in (("dev", 50), ("test", 288)):
                    names.extend((f"{kind}_{model.replace('/', '__')}_colab_scicode_{split}.jsonl", n)
                                 for kind in ("aa", "sciexec"))
            for name, n in names:
                (root / name).write_text("".join(json.dumps({"id": str(i)}) + "\n" for i in range(n)))
                (root / (name + ".meta.json")).write_text('{"synthetic": true}')
                if name.startswith("aa_"):
                    (root / (name + ".timing.jsonl")).write_text('{"wall_s": 1}\n')
            verifier.verify(root, expected)
            file = root / names[-1][0]
            raw = file.read_text()
            for invalid in ("", raw[:-1], '{"id": "duplicate"}\n' * names[-1][1]):
                file.write_text(invalid)
                with self.assertRaises(ValueError): verifier.verify(root, expected)
            file.write_text(raw)
            expected.write_text('{"different": true}')
            with self.assertRaises(ValueError): verifier.verify(root, expected)
            provenance["benches"] = ["gpqa", "scicode"]
            expected.write_text(json.dumps(provenance))
            (root / "e12_campaign_sources.json").write_text(json.dumps(provenance))
            with self.assertRaises(FileNotFoundError): verifier.verify(root, expected)

    def test_observed_clock_and_error_have_unknown_tokens(self):
        import threading
        for error in (False, True):
            out = FakeOut()
            def chat():
                if error:
                    raise RuntimeError("synthetic")
                return {"n_tokens": 8, "prompt_tokens": 13, "ms": 999}
            with mock.patch.object(run_aa.time, "perf_counter", side_effect=[10, 12.5]):
                if error:
                    with self.assertRaises(RuntimeError):
                        run_aa.observed_chat(chat, out, threading.Lock(), id="s")
                else:
                    g = run_aa.observed_chat(chat, out, threading.Lock(), id="s")
                    self.assertEqual((g["wall_s"], g["ms"]), (2.5, 999))
            row = json.loads(Path(str(out.path) + ".timing.jsonl").read_text())
            self.assertEqual(row["wall_s"], 2.5)
            self.assertEqual(row["n_tokens"], None if error else 8)
            self.assertEqual(row["error_type"], "RuntimeError" if error else None)

    def test_problem_clock_does_not_sum_concurrent_call_durations(self):
        import threading
        out = FakeOut()
        a = type("A", (), {"no_background": False, "model": "synthetic"})()
        g = {"text": "", "finish": "stop", "n_tokens": 5, "prompt_tokens": 7, "ms": 100000,
             "reasoning": False}
        srv = type("Server", (), {"http": object()})()
        with mock.patch.object(run_aa.sot, "chat", return_value=g), \
                mock.patch.object(run_aa.time, "perf_counter", side_effect=range(20)):
            run_aa.solve_problem(srv, a, synthetic_problem(), out, threading.Lock(), {})
        rows = [json.loads(l) for l in Path(str(out.path) + ".timing.jsonl").read_text().splitlines()]
        self.assertEqual([r["wall_s"] for r in rows if r["kind"] == "call"], [1, 1, 1])
        self.assertEqual(rows[-1]["kind"], "problem")
        self.assertEqual(rows[-1]["wall_s"], 10)

    def test_failed_scicode_step_keeps_unknown_tokens(self):
        import threading
        out = FakeOut()
        a = type("A", (), {"no_background": False, "model": "synthetic"})()
        srv = type("Server", (), {"http": object()})()
        with mock.patch.object(run_aa.sot, "chat", side_effect=RuntimeError("synthetic")):
            run_aa.solve_problem(srv, a, synthetic_problem(), out, threading.Lock(), {})
        self.assertTrue(all(r["n_tokens"] is None and r["ms"] is None and r["wall_s"] >= 0 for r in out.rows))

    def test_strict_session_parser(self):
        adapter = load("session_json")
        self.assertIsNone(adapter.parse("[colab] No active sessions found on server.\n")["phase0"])
        line = "[phase0] synthetic | Hardware: A100 | Shape: Standard | Variant: GPU"
        self.assertEqual(adapter.parse(line)["phase0"]["hardware"], "A100")
        for text in ("", "connection failed", "{}", line + "\n" + line, line.replace("phase0", "?")):
            with self.assertRaises(ValueError):
                adapter.parse(text)


FAKE = r'''import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
mode = os.environ['MODE']
operation = sys.argv[1]
with (root / 'calls').open('a') as f: f.write(operation + '\n')
session = {'schema': 1, 'phase0': {'fingerprint': 'a' * 64, 'hardware': 'A100'}}
active = root / 'active'
if operation == 'sessions-json':
    print(json.dumps(session if active.exists() or mode == 'existing' else {'schema': 1, 'phase0': None}))
elif operation == 'usage-json':
    print(json.dumps({'schema': 1, 'balance_units': 100, 'rate_units_hour': 5.3, 'assignments': 1}))
elif operation == 'up':
    if mode == 'up-refused': sys.exit(23)
    if mode == 'bootstrap-refused': sys.exit(73)
    active.touch()
    Path(os.environ['DLLM_CAMPAIGN_RECEIPT']).write_text(json.dumps({**session, 'status': 'owned',
        'campaign_id': os.environ['DLLM_CAMPAIGN_ID'],
        'usage_before': {'balance_units': 100}, 'budget_policy': {'budget': 20, 'minimum': 15,
            'rate': 5.3, 'hours': .002, 'cleanup': 3, 'margin': 1}}))
    if mode in ('refused-without-bootstrap', 'foreign-receipt-dns'):
        if mode == 'foreign-receipt-dns':
            Path(os.environ['DLLM_CAMPAIGN_RECEIPT']).write_text(json.dumps({**session,
                'status': 'foreign-campaign', 'campaign_id': os.environ['DLLM_CAMPAIGN_ID']}))
            sys.exit(73)
        sys.exit(1)
    if mode == 'timeout':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = os.fork()
        if child == 0: time.sleep(60); sys.exit(0)
        (root / 'pids').write_text(json.dumps([os.getpid(), child]))
        time.sleep(60)
elif operation == 'snapshot':
    if mode == 'foreign-receipt-dns': sys.exit(27)
    if mode == 'refused-without-bootstrap':
        print(json.dumps({'schema': 1, 'bootstrap': None, 'jobs': None, 'campaign_present': True}))
        sys.exit(0)
    count = root / 'count'
    n = int(count.read_text()) + 1 if count.exists() else 1
    count.write_text(str(n))
    if mode == 'signal': time.sleep(60)
    if mode == 'connection': sys.exit(27)
    if mode == 'malformed' or mode == 'transient' and n == 1: print('{'); sys.exit(0)
    jobs = {k: 'fini en 1.0 min' for k in json.loads(os.environ['EXPECTED'])}
    if mode == 'incomplete': jobs.pop(next(iter(jobs)))
    if mode in ('failed', 'cancelled'):
        jobs[next(iter(jobs))] = 'ÉCHEC code 7' if mode == 'failed' else 'annulé (arrêt du lanceur)'
    if mode == 'deadline' or mode in ('periodic', 'periodic-failed') and n <= 3: jobs = {k: 'prévu' for k in jobs}
    if mode in ('downloads', 'dead-preflight') and n <= 2: jobs = None
    print(json.dumps({'schema': 1, 'bootstrap': {'step': 'lanceur en marche', 'plan': 'aa-1', 'pid': 42,
                                               'campaign_id': 'foreign' if mode == 'foreign' else os.environ['DLLM_CAMPAIGN_ID']},
                      'jobs': jobs, 'diagnostics': {'launcher_alive': mode != 'dead-preflight'}}))
elif operation == 'pull':
    if mode == 'periodic-failed' and not (root / 'pull-lost').exists():
        (root / 'pull-lost').touch()
        sys.exit(27)
    if mode == 'pull-failed': sys.exit(31)
    if mode == 'downloads' and int((root / 'count').read_text()) <= 2: sys.exit(83)
elif operation == 'down':
    if mode == 'down-failed': sys.exit(32)
    active.unlink()
else: sys.exit(80)
'''

RUNNER = '''import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import campaign
campaign.WRAPPER = Path(os.environ['FAKE_WRAPPER'])
sys.exit(campaign.main())
'''


@unittest.skipUnless(sys.platform == "linux", "real process-group tests require Linux/WSL")
class Campaign(unittest.TestCase):
    def test_real_wrapper_signal_and_remote_refusal_with_local_zero(self):
        cli = r'''import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
mode = os.environ['MODE']
args = sys.argv[1:]
with (root / 'calls').open('a') as f: f.write(args[0] + '\n')
if args[0] == 'sessions':
    print('[phase0] synthetic | Hardware: A100 | Shape: Standard | Variant: GPU'
          if (root / 'allocated').exists() else '[colab] No active sessions found on server.')
elif args[0] == 'usage':
    print('Current balance: 100.00 compute units\nUsage rate: ' + ('5.30/hr\nActive assignments: 1'
          if (root / 'allocated').exists() else '0.00/hr\nActive assignments: 0'))
elif args[0] == 'new': (root / 'allocated').touch()
elif args[0] == 'upload': pass
elif args[0] == 'exec':
    if args[-1].endswith('capacity.py'):
        print('DLLM_CAPACITY ' + json.dumps({'free_mib': 40960}))
        sys.exit(0)
    elif args[-1].endswith('verify_part.py'):
        import ast
        config = ast.literal_eval(Path(args[-1]).read_text().splitlines()[0].split(' = ', 1)[1])
        print('DLLM_PART_ACK ' + json.dumps({**config, 'matches': True}))
    elif args[-1].endswith('reconstruct.py'):
        import ast
        config = ast.literal_eval(Path(args[-1]).read_text().splitlines()[0].split(' = ', 1)[1])
        print('DLLM_TRANSFER_ACK ' + json.dumps({'schema': 1, 'status': 'ok', **config}))
    elif args[-1].endswith('colab_bootstrap.py'):
        (root / 'boot').touch()
        print('SystemExit: 73' if mode == 'refused' else 'started')
        sys.exit(0)  # Real CLI can report remote exceptions as outputs with local success.
    if args[-1].endswith('snapshot.py'):
        if not (root / 'boot').exists():
            print(json.dumps({'schema': 1, 'campaign_present': False}))
        elif mode == 'refused':
            print(json.dumps({'schema': 1, 'campaign_present': True,
                              'bootstrap': {'campaign_id': 'preexisting', 'step': 'lanceur en marche'}}))
        elif not (root / 'pids').exists():
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            child = os.fork()
            if child == 0: time.sleep(60); sys.exit(0)
            (root / 'pids.tmp').write_text(json.dumps([os.getpid(), child]))
            (root / 'pids.tmp').replace(root / 'pids')
            time.sleep(60)
        else:
            print(json.dumps({'schema': 1, 'campaign_present': True,
                              'bootstrap': {'campaign_id': os.environ['DLLM_CAMPAIGN_ID']}}))
    else:
        sys.stdin.read()
        print('LISTE-OK')
elif args[0] == 'download': sys.exit(31)
elif args[0] == 'stop': (root / 'allocated').unlink()
else: sys.exit(81)
'''
        for mode in ("refused", "signal"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="real wrapper spaces ") as d:
                root = Path(d)
                shutil.copytree(COLAB, root / "phase0" / "colab")
                shutil.copyfile(COLAB.parent / "colab_jobs.py", root / "phase0" / "colab_jobs.py")
                shutil.copyfile(COLAB.parent / "colab_bootstrap.py", root / "phase0" / "colab_bootstrap.py")
                (root / "phase0" / "data").mkdir()
                (root / "phase0" / "data" / "scicode_test_data.h5").write_bytes(b"synthetic")
                bins = root / ".local" / "bin"
                bins.mkdir(parents=True)
                fake = bins / "colab"
                fake.write_text(f"#!{sys.executable}\n" + cli)
                fake.chmod(0o755)
                runner = root / "runner.py"
                runner.write_text(RUNNER)
                env = {**os.environ, "HOME": d, "TEST_ROOT": d, "MODE": mode, "DLLM_REPO": d,
                       "FAKE_WRAPPER": str(root / "phase0" / "colab" / "colab_phase0.sh"), "DLLM_BUDGET_UNITS": "20", "DLLM_HOURS": ".01"}
                cmd = [sys.executable, str(runner), str(COLAB), "--lock", str(root / "lock"),
                       "--receipt", str(root / "receipt.json"), "--budget-units", "20",
                       "--hours", ".01", "--read-seconds", "3", "--up-seconds", "10", "--transfer-seconds", "2",
                       "--cleanup-seconds", "10", "--grace", ".1"]
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                pids = []
                try:
                    if mode == "signal":
                        deadline = time.monotonic() + 10
                        while not (root / "pids").exists() and p.poll() is None and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue((root / "pids").exists())
                        pids = json.loads((root / "pids").read_text())
                        for pid in pids:
                            status = Path(f"/proc/{pid}/status").read_text()
                            mask = int(next(l.split()[1] for l in status.splitlines() if l.startswith("SigIgn:")), 16)
                            self.assertTrue(mask & (1 << (signal.SIGTERM - 1)), status)
                        p.send_signal(signal.SIGTERM)
                    output, _ = p.communicate(timeout=20)
                    self.assertEqual(p.returncode, 73 if mode == "refused" else 143, output)
                    calls = (root / "calls").read_text().splitlines()
                    self.assertEqual(calls.count("new"), 1)
                    if mode == "refused":
                        self.assertNotIn("stop", calls)
                        self.assertNotIn("download", calls)
                    else:
                        self.assertIn("stop", calls)
                        for pid in pids:
                            path = Path(f"/proc/{pid}/stat")
                            if path.exists():
                                self.assertIn(path.read_text().rsplit(")", 1)[1].split()[0], ("Z", "X", "x"))
                finally:
                    if p.poll() is None:
                        p.kill()
                        p.communicate()
                    for pid in pids:
                        try: os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError: pass

    def test_wrapper_pulls_timing_only_as_companion(self):
        cli = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
name = 'aa_synthetic_colab_scicode_dev.jsonl'
if args[0] == 'sessions':
    print('[phase0] synthetic | Hardware: A100 | Shape: Standard | Variant: GPU')
elif args[0] == 'exec':
    sys.stdin.read()
    print('LISTE-OK')
    print(name)
    print(name + '.timing.jsonl')
elif args[0] == 'download':
    remote = Path(args[3]).name
    if remote.endswith(('.timing.jsonl.meta.json', '.timing.jsonl.timing.jsonl')): sys.exit(82)
    if remote == name: raw = '{"id": "synthetic"}\\n'
    elif remote == name + '.timing.jsonl': raw = '{"kind": "call", "wall_s": 1}\\n'
    elif remote == 'colab_bootstrap_status.json': raw = '{"campaign_id": "synthetic"}'
    else: raw = '{"synthetic": true, "measurements": "e12-wall-v1"}'
    Path(args[4]).write_text(raw)
else: sys.exit(81)
'''
        with tempfile.TemporaryDirectory(prefix="pull spaces ") as d:
            root = Path(d)
            shutil.copytree(COLAB, root / "phase0" / "colab")
            shutil.copy(COLAB.parent / "aa_timing.py", root / "phase0")
            shutil.copy(COLAB.parent / "colab_jobs.py", root / "phase0")
            bins = root / ".local" / "bin"
            bins.mkdir(parents=True)
            fake = bins / "colab"
            fake.write_text(f"#!{sys.executable}\n" + cli)
            fake.chmod(0o755)
            receipt = root / "receipt"
            import hashlib
            receipt.write_text(json.dumps({"status": "owned", "plan": "aa-1", "campaign_id": "synthetic",
                "phase0": {"fingerprint": hashlib.sha256(b"synthetic").hexdigest(), "hardware": "A100"}}))
            env = {**os.environ, "HOME": d, "DLLM_REPO": d, "DLLM_CAMPAIGN_RECEIPT": str(receipt)}
            result = subprocess.run(["bash", str(root / "phase0" / "colab" / "colab_phase0.sh"), "pull"],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            companion = root / "phase0" / "results" / "aa_synthetic_colab_scicode_dev.jsonl.timing.jsonl"
            self.assertEqual(json.loads(companion.read_text())["wall_s"], 1)
            self.assertFalse(Path(str(companion) + ".meta.json").exists())

    def test_exclusive_wrapper_refuses_existing_campaign_before_upload(self):
        cli = '''import json, os, sys
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
mode = os.environ['MODE']
args = sys.argv[1:]
with (root / 'calls').open('a') as f: f.write(args[0] + '\\n')
if args[0] == 'sessions':
    if mode == 'connection': sys.exit(23)
    if mode == 'existing' or (root / 'allocated').exists():
        print('[phase0] synthetic | Hardware: A100 | Shape: Standard | Variant: GPU')
    else: print('[colab] No active sessions found on server.')
elif args[0] == 'usage':
    print('Current balance: 100.00 compute units\\nUsage rate: ' + ('5.30/hr\\nActive assignments: 1'
          if (root / 'allocated').exists() else '0.00/hr\\nActive assignments: 0'))
elif args[0] == 'new': (root / 'allocated').touch()
elif args[0] == 'exec':
    if args[-1].endswith('capacity.py'):
        print('DLLM_CAPACITY ' + json.dumps({'free_mib': 40960}))
    elif args[-1].endswith('verify_part.py'):
        import ast
        config = ast.literal_eval(Path(args[-1]).read_text().splitlines()[0].split(' = ', 1)[1])
        print('DLLM_PART_ACK ' + json.dumps({**config, 'matches': True}))
    elif args[-1].endswith('snapshot.py'):
        print(json.dumps({'schema': 1, 'campaign_present': mode == 'campaign-existing'}))
    elif args[-1].endswith('reconstruct.py'):
        import ast
        config = ast.literal_eval(Path(args[-1]).read_text().splitlines()[0].split(' = ', 1)[1])
        print('DLLM_TRANSFER_ACK ' + json.dumps({'schema': 1, 'status': 'ok', **config}))
    elif mode == 'bootstrap-refused': sys.exit(73)
elif args[0] == 'upload': pass
else: sys.exit(81)
'''
        for mode in ("existing", "campaign-existing", "bootstrap-refused", "success", "connection"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="wrapper spaces ") as d:
                root = Path(d)
                shutil.copytree(COLAB, root / "phase0" / "colab")
                shutil.copyfile(COLAB.parent / "colab_jobs.py", root / "phase0" / "colab_jobs.py")
                shutil.copyfile(COLAB.parent / "colab_bootstrap.py", root / "phase0" / "colab_bootstrap.py")
                (root / "phase0" / "data").mkdir()
                (root / "phase0" / "data" / "scicode_test_data.h5").write_bytes(b"synthetic, never executed")
                bins = root / ".local" / "bin"
                bins.mkdir(parents=True)
                fake = bins / "colab"
                fake.write_text(f"#!{sys.executable}\n" + cli)
                fake.chmod(0o755)
                receipt = root / "receipt.json"
                env = {**os.environ, "HOME": d, "TEST_ROOT": d, "DLLM_REPO": d, "MODE": mode,
                       "DLLM_CAMPAIGN_RECEIPT": str(receipt), "DLLM_CAMPAIGN_ID": "synthetic", "DLLM_BUDGET_UNITS": "20", "DLLM_HOURS": ".01"}
                result = subprocess.run(["bash", str(root / "phase0" / "colab" / "colab_phase0.sh"),
                                         "up", "aa-1", "A100"], env=env, capture_output=True, text=True, timeout=10)
                code = 0 if mode == "success" else 65 if mode == "connection" else 73
                self.assertEqual(result.returncode, code, result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertNotIn("stop", calls)
                if mode in ("existing", "campaign-existing", "connection"):
                    self.assertNotIn("upload", calls)
                self.assertEqual(receipt.exists(), mode in ("success", "campaign-existing", "bootstrap-refused"))

    def run_case(self, mode, expected_code, signal_case=False):
        sys.path.insert(0, str(COLAB))
        import campaign
        with tempfile.TemporaryDirectory(prefix="campaign with spaces ") as d:
            root = Path(d)
            (root / "fake.py").write_text(FAKE)
            wrapper = root / "fake wrapper.sh"
            wrapper.write_text('exec "$PYTHON" "$TEST_ROOT/fake.py" "$@"\n')
            runner = root / "runner.py"
            runner.write_text(RUNNER)
            env = {**os.environ, "TEST_ROOT": d, "MODE": mode, "PYTHON": sys.executable,
                   "FAKE_WRAPPER": str(wrapper), "EXPECTED": json.dumps(sorted(campaign.EXPECTED))}
            cmd = [sys.executable, str(runner), str(COLAB), "--lock", str(root / "lock"),
                   "--receipt", str(root / "receipt.json"), "--budget-units", "20", "--hours",
                   ".00015" if mode == "deadline" else ".002",
                   "--poll-seconds", ".02", "--pull-seconds", ".02", "--read-seconds", ".3",
                   "--up-seconds", ".4", "--transfer-seconds", ".3", "--cleanup-seconds", "3", "--grace", ".1"]
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if signal_case:
                deadline = time.monotonic() + 5
                while not (root / "count").exists() and p.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "count").exists())
                competitor = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2)
                self.assertEqual(competitor.returncode, 73, competitor.stdout)
                self.assertIn("another local campaign", competitor.stdout)
                p.send_signal(signal.SIGTERM)
            output, _ = p.communicate(timeout=10)
            self.assertEqual(p.returncode, expected_code, output)
            calls = (root / "calls").read_text().splitlines()
            if mode in ("existing", "up-refused", "bootstrap-refused", "foreign", "refused-without-bootstrap",
                        "foreign-receipt-dns"):
                self.assertNotIn("pull", calls)
                self.assertNotIn("down", calls)
            else:
                self.assertIn("pull", calls)
                self.assertIn("down", calls)
                self.assertEqual(calls.count("up"), 1)
            if expected_code:
                self.assertNotIn("SUCCESS:", output)
            if mode == "pull-failed":
                self.assertIn("results not confirmed recovered", output)
            if mode in ("failed", "cancelled"):
                self.assertIn("ÉCHEC code 7" if mode == "failed" else "annulé", output)
            if mode in ("periodic", "periodic-failed"):
                self.assertGreaterEqual(calls.count("pull"), 2)
            if mode == "timeout":
                for pid in json.loads((root / "pids").read_text()):
                    path = Path(f"/proc/{pid}/stat")
                    if path.exists():
                        self.assertIn(path.read_text().rsplit(")", 1)[1].split()[0], ("Z", "X", "x"))

    def test_success_and_transient_reads(self):
        for mode in ("success", "transient", "periodic", "periodic-failed", "downloads"):
            with self.subTest(mode=mode): self.run_case(mode, 0)

    def test_refusals_and_failures(self):
        for mode, code in (("existing", 73), ("up-refused", 23), ("bootstrap-refused", 73),
                           ("failed", 1), ("cancelled", 1), ("malformed", 65), ("incomplete", 65),
                           ("connection", 27), ("dead-preflight", 1), ("pull-failed", 31), ("down-failed", 32),
                           ("timeout", 124), ("deadline", 124), ("foreign", 73), ("refused-without-bootstrap", 1),
                           ("foreign-receipt-dns", 73)):
            with self.subTest(mode=mode): self.run_case(mode, code)

    def test_signal_cleanup(self):
        self.run_case("signal", 143, True)


if __name__ == "__main__":
    unittest.main()
