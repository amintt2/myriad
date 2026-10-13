"""Synthetic plan admission and hermetic fake official CLI; no cloud or scientific computation."""
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
import e16_supervision as e16
from campaign import Failure, PLANS, state


class Admission(unittest.TestCase):
    def usage(self, balance):
        return {"schema": 1, "balance_units": balance, "rate_units_hour": 0, "assignments": 0}

    def test_bounded_full_attempt_and_threshold(self):
        from budget import remaining
        for balance, plan, admitted in ((59.62, "e16-1", True), (45, "e16-1", True),
                                        (44.9, "e16-mc", True), (19, "e16-mc", False), (15, "e16-mc", False)):
            value = e16.admission(self.usage(balance))
            self.assertEqual(value["plan"], plan)
            self.assertEqual(value["admitted"], admitted)
            policy = value["policy"]
            self.assertEqual(policy["minimum"], 15)
            self.assertEqual(policy["cleanup"], 1800)
            self.assertEqual(policy["margin"], 161)
            self.assertLessEqual(policy["budget"], balance - 15)
            self.assertAlmostEqual(value["available_science_seconds"], value["seconds"] - 900)
            if admitted:
                cost = 5.3 * (value["seconds"] + 1800 + 161) / 3600
                self.assertLessEqual(cost, policy["budget"] + 1e-10)
                self.assertEqual(value["incompletion_risk"], balance < 59.5)
                self.assertTrue(value["warning"])
                allocated = {**self.usage(balance), "rate_units_hour": 5.3, "assignments": 1}
                for elapsed in (0, value["seconds"] / 2, value["seconds"]):
                    usage = {**allocated, "balance_units": balance - elapsed * 5.3 / 3600}
                    seconds, rate = remaining(usage, policy, balance, elapsed, 5.3)
                    self.assertGreaterEqual(usage["balance_units"] - rate * (seconds + 1800 + 161) / 3600, 15 - 1e-10)
        for usage in ({**self.usage(60), "assignments": 1}, {**self.usage(60), "rate_units_hour": 1},
                      {}, None, self.usage(float("nan")), self.usage(-1), self.usage(True)):
            with self.assertRaises(ValueError):
                e16.admission(usage)

    def test_exact_jobs_selected_plan_and_scientific_failure(self):
        for plan in ("aa-1", "e16-1", "e16-mc"):
            snapshot = {"schema": 1, "bootstrap": {"step": "lanceur en marche", "plan": plan, "pid": 123},
                        "diagnostics": {"launcher_alive": False},
                        "jobs": {" ".join(j): "fini en 1.0 min" for j in PLANS[plan]}}
            self.assertEqual(state(snapshot, plan), "complete")
            snapshot["jobs"].pop(next(iter(snapshot["jobs"])))
            with self.assertRaises(Failure):
                state(snapshot, plan)
        with self.assertRaises(Failure):
            state({}, "e16-2")

    def test_seal_identity_distinguishes_bootstrap_initialization(self):
        import e16_remote
        request = {"campaign_id": "owned", "plan": "e16-mc", "nonce": "synthetic"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            for boot, error in (({"campaign_id": "foreign", "step": "ÉCHEC"}, PermissionError),
                                ({"campaign_id": "owned", "plan": "e16-1", "step": "ÉCHEC"}, PermissionError),
                                ({"campaign_id": "owned", "plan": None, "step": "ÉCHEC"}, PermissionError),
                                ({"campaign_id": "owned", "step": "ÉCHEC"}, ValueError),
                                ({"campaign_id": "owned", "step": "environnement Python (uv)"}, ValueError)):
                (root / "results/colab_bootstrap_status.json").write_text(json.dumps(boot))
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    with mock.patch.object(e16_remote.os, "chdir"), self.assertRaises(error):
                        e16_remote.seal(request, final=True)
                finally:
                    os.chdir(previous)

    def test_vm_completion_requires_exact_rows_pins_and_all_execution_programs(self):
        import essaim
        from essaim import results
        import e16_remote
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results"
            folder.mkdir()
            quant = SimpleNamespace(model="synthetic/model", tag="iq", suffix="_iq_colab",
                                    exec_tag="e16synthetic", sha256="a" * 64)
            table = SimpleNamespace(QUANT=[quant], CODE_BOUNDS={(quant.model, quant.tag)}, SPEED_Q8=["small"],
                                    label=lambda model, tag: model + "-" + tag)
            items = [{"id": "one"}, {"id": "two"}]
            loader = lambda n, split: items
            data = SimpleNamespace(PART={"mc": 2, "code": 2}, SPLITS=("dev", "test"),
                                   CODE_LOADERS={"code": loader}, dataset_identity=lambda bench: {"synthetic": bench})
            code = SimpleNamespace(gen_path=lambda model, suffix, bench, split: folder / f"code_{split}.jsonl",
                                   exec_path=lambda tag, suffix, bench, split: folder / f"exec_{split}.jsonl",
                                   entry_point=lambda bench, item: "entry", extract_code=lambda text, entry: (text, "ok"),
                                   prog_id=lambda text: text)
            manifest = {"weights": {"sha256": quant.sha256}, "n": 2, "samples": 4, "model": quant.model}

            def write(path, rows, man):
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                Path(str(path) + ".meta.json").write_text(json.dumps(man))

            for split in data.SPLITS:
                write(folder / f"solo_synthetic__model_iq_colab_mc_{split}.jsonl", items,
                      {**manifest, "data": {"synthetic": "mc"}, "split": split})
                write(folder / f"code_{split}.jsonl", [{"id": it["id"], "sample": s, "text": "program"}
                      for it in items for s in range(5)], {**manifest, "data": {"synthetic": "code"}})
                write(folder / f"exec_{split}.jsonl", [{"id": it["id"], "prog": p} for it in items
                      for p in ("reference", "program")], {"models": [quant.model], "data": {"synthetic": "code"}})
            speed = folder / "solo_e16speed_colab_bench_all.jsonl"
            write(speed, [{"label": "synthetic/model-iq", "weights": {"sha256": quant.sha256}},
                          {"label": "small-Q8_0"}], {"alone_on_gpu": True, "gpu": "A100"})
            previous = Path.cwd()
            try:
                os.chdir(root)
                with mock.patch.dict(sys.modules, {"e16_quant": table, "run_solo": SimpleNamespace(LOADERS={"mc": loader})}), \
                     mock.patch.object(essaim, "data", data, create=True), mock.patch.object(essaim, "code", code, create=True):
                    self.assertEqual(len(e16_remote.complete("e16-1")), 14)
                    for target, corruption in ((folder / "exec_dev.jsonl", "missing"),
                                               (folder / "code_dev.jsonl", "duplicate"),
                                               (speed, "weight"), (speed, "tail")):
                        old = target.read_bytes()
                        rows = results.read_rows(target)
                        if corruption == "missing":
                            rows.pop()
                        if corruption == "duplicate":
                            rows.append(rows[0])
                        if corruption == "weight":
                            rows[0]["weights"]["sha256"] = "b" * 64
                        target.write_text("".join(json.dumps(r) + "\n" for r in rows) +
                                          ('{"unfinished":' if corruption == "tail" else ''))
                        with self.assertRaises((ValueError, KeyError)):
                            e16_remote.complete("e16-1")
                        target.write_bytes(old)
            finally:
                os.chdir(previous)


FAKE = r'''import ast, base64, hashlib, json, os, shutil, sys, tarfile, time, zlib
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
mode = os.environ['MODE']
args = sys.argv[1:]
op = args[0]
active = root / 'active'
with (root / 'calls').open('a') as out: out.write(op + '\n')
if op == 'sessions':
    if mode == 'foreign-preflight': active.touch()
    endpoint = 'foreign' if (root / 'replaced').exists() else 'synthetic'
    print(f'[phase0] {endpoint} | Hardware: A100 | Shape: Standard | Variant: GPU' if active.exists()
          else '[colab] No active sessions found on server.')
elif op == 'usage':
    if mode == 'usage-unknown' and active.exists(): sys.exit(0)
    if mode == 'usage-during' and (root / 'booted').exists() and active.exists(): sys.exit(0)
    if mode == 'final-unknown' and (root / 'allocated').exists() and not active.exists(): sys.exit(0)
    balance = f'{float(mode[8:]):.2f}' if mode.startswith('balance-') else '19.00' if mode == 'low' else '59.62'
    rate = '5.40' if mode == 'rate' else '5.30'
    print(f'Current balance: {balance} compute units\nUsage rate: ' +
          (rate + '/hr\nActive assignments: 1' if active.exists() else '0.00/hr\nActive assignments: 0'))
elif op == 'new':
    (root / 'new-started').touch()
    if mode in ('new-term', 'new-timeout'): time.sleep(.3 if mode == 'new-term' else 3)
    active.touch(); (root / 'allocated').touch()
    if mode == 'new-ambiguous': sys.exit(27)
elif op == 'stop':
    if mode == 'stop-retry' and not (root / 'stop-failed').exists():
        (root / 'stop-failed').touch(); sys.exit(27)
    if mode == 'stop-never': sys.exit(27)
    active.unlink(missing_ok=True)
    if mode == 'stop-lost': sys.exit(27)
elif op == 'upload':
    shutil.copyfile(args[-2], root / Path(args[-1]).name)
elif op == 'download':
    shutil.copyfile(root / Path(args[-2]).name, args[-1])
    if mode == 'corrupt-download': Path(args[-1]).write_bytes(b'corrupted')
elif op == 'exec':
    source = Path(args[-1]).read_text()
    if 'CONFIG = ' in source:
        config = ast.literal_eval(ast.parse(source).body[0].value)
        if 'DLLM_PART_ACK ' in source:
            path = root / config['name']
            matches = path.exists() and path.stat().st_size == config['size'] and hashlib.sha256(path.read_bytes()).hexdigest() == config['sha256']
            print('DLLM_PART_ACK ' + json.dumps({**config, 'matches': matches}))
        else:
            sys.path.insert(0, os.environ['HELPERS'])
            from reconstruct import reconstruct
            print('DLLM_TRANSFER_ACK ' + json.dumps(reconstruct(config, root)))
    elif 'E16_CAPACITY ' in source:
        print('E16_CAPACITY ' + json.dumps({'free_mib': 39000}))
    elif 'E16_SEAL ' in source:
        write = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'write')
        payload = ast.literal_eval(write.args[0])
        call = next(node for node in ast.walk(ast.parse(payload)) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'seal')
        request, final = [ast.literal_eval(a) for a in call.args]
        if mode == 'foreign-seal':
            print('E16_SEAL ' + base64.b64encode(zlib.compress(json.dumps({**request,
                  'foreign_campaign': True}).encode())).decode()); sys.exit(0)
        if mode.startswith('bootstrap-'):
            namespace = {}
            remote = payload.split('\nimport contextlib\n')[0]
            remote = remote.replace('/content/dllm', str(root / 'dllm')).replace('Path("/content")', 'Path(' + repr(str(root)) + ')')
            exec(compile(remote, 'real-e16-remote', 'exec'), namespace)
            try:
                seal = namespace['seal'](request, final)
            except PermissionError:
                seal = {**request, 'foreign_campaign': True}
            print('E16_SEAL ' + base64.b64encode(zlib.compress(json.dumps(seal).encode())).decode())
            sys.exit(0)
        if mode == 'missing-artifact' and final: sys.exit(27)
        provenance = json.loads((root / 'sources').read_text())
        content = {'e16_sources.json': json.dumps(provenance).encode(),
                   'synthetic.jsonl': (json.dumps({'text': 'x' * 100000}) + '\n').encode()}
        files = {}
        for name, raw in content.items():
            (root / name).write_bytes(raw)
            files[name] = {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        if mode == 'unsafe-inventory': files['../outside'] = files.pop('synthetic.jsonl')
        if mode == 'stale-seal': request['nonce'] = 'stale'
        print('E16_SEAL ' + base64.b64encode(zlib.compress(json.dumps({**request, 'schema': 1,
              'complete': final, 'files': files}).encode())).decode())
    elif 'WORK = "/content/dllm"' in source:
        (root / 'booted').touch()
        if mode.startswith('bootstrap-'):
            from unittest import mock
            with tarfile.open(root / 'dllm.tgz') as archive:
                archive.extractall(root / 'dllm', filter='data')
            (root / 'sources').write_bytes((root / 'dllm/phase0/results/e16_sources.json').read_bytes())
            boot = source.replace('/content/dllm', str(root / 'dllm')).replace('/content/llama', str(root / 'llama'))
            for rel, raw in [('colab/e16_remote.py', Path(os.environ['HELPERS']) / 'e16_remote.py'),
                             ('colab_bootstrap.py', Path(os.environ['HELPERS']).parent / 'colab_bootstrap.py')]:
                target = root / 'dllm/phase0' / rel
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(raw, target)
            def command(args, **kw):
                if args[0] == 'curl': Path(args[args.index('-o') + 1]).write_bytes(b'synthetic archive')
                if args[:2] == ['uv', 'sync']: (root / 'environment-installed').touch()
                if args[0] == 'nvidia-smi' and mode == 'bootstrap-gpu': raise OSError('synthetic GPU failure')
                return type('Result', (), {'stdout': 'CUDA0 version synthetic', 'stderr': '', 'returncode': 0})()
            def launch(*args, **kw):
                if mode.startswith('bootstrap-popen'): raise OSError('synthetic launch failure')
                return type('Launcher', (), {'pid': 123, 'returncode': 1,
                            'poll': lambda self: 1 if mode == 'bootstrap-exited' else None})()
            with mock.patch('subprocess.run', side_effect=command), mock.patch('subprocess.Popen', side_effect=launch), \
                 mock.patch('glob.glob', side_effect=lambda pattern, **kw: [str(root / 'llama/llama-server')] if 'llama-server' in pattern else []), \
                 mock.patch('shutil.which', return_value='synthetic-uv'), mock.patch('time.sleep'):
                try: exec(compile(boot, 'real-colab-bootstrap', 'exec'), {})
                except (SystemExit, OSError): pass
            if mode == 'bootstrap-popen-recovery-failed': (root / 'dllm/phase0/results/e16_sources.json').unlink()
            sys.exit(27)
        with tarfile.open(root / 'dllm.tgz') as archive:
            for name in ('colab_campaign_id.txt', 'colab_plan.txt'):
                (root / name).write_bytes(archive.extractfile('phase0/' + name).read())
            (root / 'sources').write_bytes(archive.extractfile('phase0/results/e16_sources.json').read())
    elif 'safe_snapshot(snapshot())' in source or args[-1].endswith('snapshot.py'):
        booted = (root / 'booted').exists()
        if mode == 'replaced' and booted: (root / 'replaced').touch()
        if mode == 'foreign-campaign' and not booted:
            print(json.dumps({'schema': 1, 'campaign_present': True})); sys.exit(0)
        if not booted:
            print(json.dumps({'schema': 1, 'campaign_present': False}))
        else:
            sys.path.insert(0, str(Path(os.environ['HELPERS']).parent))
            from colab_jobs import PLANS
            plan = 'e16-mc' if mode == 'foreign-plan' else (root / 'colab_plan.txt').read_text()
            campaign = 'foreign' if mode == 'foreign-bootstrap' else (root / 'colab_campaign_id.txt').read_text()
            jobs = {' '.join(j): 'fini en 1.0 min' for j in PLANS[plan]}
            if mode == 'failed-job': jobs[next(iter(jobs))] = 'ÉCHEC code 1'
            if mode in ('wait', 'expire'): jobs[next(iter(jobs))] = 'en cours'
            print(json.dumps({'schema': 1, 'campaign_present': True,
                'bootstrap': {'step': 'lanceur en marche', 'pid': 123, 'plan': plan,
                              'campaign_id': campaign},
                'jobs': jobs, 'diagnostics': {'launcher_alive': mode in ('wait', 'expire')}}))
    else: sys.exit(98)
else: sys.exit(99)
'''

RUNNER = r'''import hashlib, io, json, os, signal, sys, tarfile
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(sys.argv[1]).parent))
import e16_supervision as e16
original_init = e16.E16.__init__
def init(self, args):
    original_init(self, args)
    args.read_seconds, args.grace, args.cleanup_seconds, args.usage_seconds, args.retry_seconds = .7, .1, 3, .1, .01
e16.E16.__init__ = init
original_admission = e16.admission
def admission(value):
    result = original_admission(value)
    result['seconds'] = 12 if os.environ['MODE'] == 'expire' else 8
    result['available_science_seconds'] = 7
    return result
e16.admission = admission
def package(directory, campaign, plan):
    paths = {'phase0/colab/e16_remote.py': e16.HERE / 'e16_remote.py',
             'phase0/colab_bootstrap.py': e16.HERE.parent / 'colab_bootstrap.py'}
    sources = {'schema': 1, 'campaign_id': campaign, 'plan': plan, 'references': {},
               'sources': {rel: hashlib.sha256(path.read_bytes()).hexdigest() for rel, path in paths.items()}}
    archive = directory / 'prepared.tgz'
    with tarfile.open(archive, 'w:gz') as output:
        for name, content in [('phase0/colab_plan.txt', plan), ('phase0/colab_campaign_id.txt', campaign),
                              ('phase0/results/e16_sources.json', json.dumps(sources))]:
            member = tarfile.TarInfo(name); raw = content.encode(); member.size = len(raw)
            output.addfile(member, io.BytesIO(raw))
    return archive, sources
e16.package = package
original_save = e16.lifecycle.save
def save(path, value):
    stage = 'owned' if value.get('status') == 'owned' else 'returned' if value.get('allocation_returned') else 'pending'
    mode = os.environ['MODE']
    if mode.startswith('persist-' + stage) and Path(path).name == 'receipt':
        if mode.endswith('-after'): original_save(path, value)
        if '-term' in mode: os.kill(os.getpid(), signal.SIGTERM)
        raise OSError('synthetic persistence error')
    return original_save(path, value)
e16.lifecycle.save = save
sys.argv = [sys.argv[0], *sys.argv[2:]]
sys.exit(e16.main())
'''


@unittest.skipUnless(sys.platform == "linux", "hermetic official CLI processes require WSL/Linux")
class Processes(unittest.TestCase):
    def fixture(self, root, mode, execute=True):
        bins = root / ".local/bin"
        bins.mkdir(parents=True)
        fake = bins / "colab"
        fake.write_text(f"#!{sys.executable}\n" + FAKE)
        fake.chmod(0o755)
        runner = root / "runner.py"
        runner.write_text(RUNNER)
        env = {**os.environ, "HOME": str(root), "PATH": str(bins) + ":" + os.environ["PATH"],
               "TEST_ROOT": str(root), "MODE": mode, "HELPERS": str(COLAB), "DLLM_CLI_GRACE": ".1"}
        for key in ("DLLM_LIFECYCLE_LOCK_FD", "DLLM_OPERATION_IDENTITY", "DLLM_CAMPAIGN_ID", "DLLM_REQUIRE_OWNER"):
            env.pop(key, None)
        command = [sys.executable, str(runner), str(COLAB), "--output", str(root / "report.json"),
                   "--receipt", str(root / "receipt"), "--poll-seconds", ".01"]
        if execute:
            command.append("--execute")
        return env, command

    def test_preflight_never_allocates_or_mutates_cloud(self):
        for mode in ("success", "low", "foreign-preflight"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode, False)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode == 0, mode in ("success",), result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertLessEqual(set(calls), {"sessions", "usage"})
                self.assertFalse((root / "receipt").exists())

    def test_full_flow_and_failures_with_one_allocation_and_confirmed_cleanup(self):
        modes = ("success", "usage-unknown", "rate", "missing-artifact", "failed-job", "stale-seal",
                 "corrupt-download", "unsafe-inventory", "stop-retry", "stop-lost", "usage-during",
                 "persist-returned-before", "persist-returned-after", "persist-owned-before", "persist-owned-after",
                 "persist-returned-term-before", "persist-owned-term-before", "new-timeout",
                 "bootstrap-popen", "bootstrap-exited", "bootstrap-gpu", "bootstrap-popen-recovery-failed", "balance-45", "balance-44.9", "expire")
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=17 if mode == "expire" else 12)
                success = mode in ("success", "stop-retry", "stop-lost", "balance-45", "balance-44.9")
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertEqual(calls.count("new"), 1)
                self.assertFalse((root / "active").exists(), result.stdout + result.stderr)
                self.assertFalse((root / "receipt").exists(), result.stdout + result.stderr)
                report = json.loads((root / "report.json").read_text())
                self.assertTrue(report["closed"])
                self.assertTrue(report["absence_confirmed"])
                self.assertEqual(calls[-1], "usage")
                if mode.startswith("balance-"):
                    balance = float(mode[8:])
                    self.assertEqual(report["plan"], "e16-1" if balance >= 45 else "e16-mc")
                    self.assertEqual(report["admission"]["policy"]["budget"], balance - 15)
                    self.assertTrue(report["admission"]["incompletion_risk"])
                if mode == "expire":
                    self.assertEqual(report["verdict"], "incomplete")
                    self.assertTrue(report["recovered"])
                    self.assertTrue(all("partial-" in path for path in report["recovered"]))
                if mode.startswith("bootstrap-"):
                    self.assertTrue((root / "environment-installed").exists())
                    boot = json.loads((root / "dllm/phase0/results/colab_bootstrap_status.json").read_text())
                    self.assertEqual(boot["campaign_id"], report["campaign_id"])
                    self.assertNotIn("plan", boot)
                    self.assertTrue(boot["step"].startswith("ÉCHEC"))
                    self.assertFalse(report["foreign_campaign"])
                    self.assertEqual(calls.count("stop"), 1)
                    self.assertEqual(report["usage"][-1]["assignments"], 0)
                if success:
                    final = Path(report["recovered"][-1])
                    self.assertTrue((final / "verified.json").exists())
                    self.assertGreater((final / "synthetic.jsonl").stat().st_size, 65536)
                    journal = [json.loads(line) for line in (root / "receipt.usage.jsonl").read_text().splitlines()]
                    self.assertTrue(any(item.get("after", {}).get("assignments") == 1 for item in journal))
                if mode == "missing-artifact":
                    self.assertIn("partial-", report["recovered"][-1])

    def test_ambiguous_foreign_and_unconfirmed_stop_keep_receipt(self):
        for mode in ("new-ambiguous", "replaced", "foreign-campaign", "foreign-bootstrap", "foreign-plan", "foreign-seal",
                     "final-unknown", "stop-never"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertEqual(calls.count("new"), 1)
                self.assertTrue((root / "receipt").exists())
                self.assertFalse(json.loads((root / "report.json").read_text())["closed"])
                if mode in ("new-ambiguous", "replaced", "foreign-campaign", "foreign-bootstrap", "foreign-plan", "foreign-seal"):
                    self.assertNotIn("stop", calls)

    def test_wrapper_keeps_old_supervised_guard_and_exposes_dedicated_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, _ = self.fixture(root, "success")
            env.update(DLLM_SUPERVISED="1", DLLM_CAMPAIGN_RECEIPT=str(root / "receipt"))
            result = subprocess.run(["bash", str(COLAB / "colab_phase0.sh"), "up", "e16-1", "A100"],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse((root / "calls").exists())
            result = subprocess.run(["bash", str(COLAB / "colab_phase0.sh"), "e16-supervise", "--help"],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("--execute", result.stdout)
            self.assertFalse((root / "calls").exists())

    def test_signals_during_new_and_wait_are_cleaned_without_reallocation(self):
        for mode, marker, sig in (("new-term", "new-started", signal.SIGTERM),
                                  ("wait", "booted", signal.SIGINT)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, mode)
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while not (root / marker).exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue((root / marker).exists())
                    os.kill(process.pid, sig)
                    output, _ = process.communicate(timeout=8)
                    self.assertNotEqual(process.returncode, 0, output)
                    self.assertFalse((root / "active").exists(), output)
                    self.assertFalse((root / "receipt").exists(), output)
                    self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

    def test_output_and_receipt_with_spaces_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "success")
            command[command.index("--output") + 1] = str(root / "report with spaces.json")
            command[command.index("--receipt") + 1] = str(root / "receipt with spaces")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads((root / "report with spaces.json").read_text())["closed"])
            self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)

    def test_all_companions_and_recovery_namespace_refuse_before_cli(self):
        for companion in ("", ".sources.json", ".snapshots.jsonl"):
            for suffix in ("", ".lock", ".usage.jsonl", ".archive.tgz", ".archive.tgz.colab_bootstrap.py"):
                for relative in (False, True):
                    with self.subTest(companion=companion, suffix=suffix, relative=relative), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        env, command = self.fixture(root, "success")
                        output = root / "run with spaces"
                        receipt = root / "owner with spaces"
                        if not suffix:
                            receipt = Path(str(output) + companion)
                        elif not companion:
                            output = Path(str(receipt) + suffix)
                        else:
                            Path(str(receipt) + suffix).symlink_to(Path(str(output) + companion))
                        # Construct an actual collision, including fresh dangling aliases.
                        command[command.index("--output") + 1] = str(output)
                        command[command.index("--receipt") + 1] = str(receipt)
                        if relative:
                            command[command.index("--output") + 1] = "./" + output.name
                            command[command.index("--receipt") + 1] = "./nested/../" + receipt.name
                            (root / "nested").mkdir()
                        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=5)
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertFalse((root / "calls").exists())
                        self.assertFalse(Path(str(receipt) + ".lock").exists())
        for target in ("e16-" + "a" * 32 + "/partial-any/receipt", "run/receipt"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, "success")
                command[command.index("--output") + 1] = str(root / "run")
                command[command.index("--receipt") + 1] = str(root / target)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "calls").exists())

    def test_resolved_symlink_companions_refuse_before_cli(self):
        for alias, target in (("report.json.sources.json", "receipt"), ("receipt.lock", "receipt"),
                              ("receipt.archive.tgz", "receipt.usage.jsonl"),
                              ("report.json.snapshots.jsonl", "report.json")):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, command = self.fixture(root, "success")
                (root / alias).symlink_to(root / target)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "calls").exists())

    def test_shared_lock_and_existing_receipt_refuse_before_new(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "success")
            with (root / "receipt.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "calls").exists())
            (root / "receipt").write_text('{"foreign": true}')
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "calls").exists())

    def test_cleanup_only_resumes_accounting_without_new_or_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, command = self.fixture(root, "final-unknown")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
            self.assertNotEqual(result.returncode, 0)
            saved = json.loads((root / "receipt").read_text())
            self.assertEqual(saved["status"], "released/accounting-pending")
            before = (root / "calls").read_text().splitlines()
            env["MODE"] = "success"
            command = [str(root / "cleanup.json") if arg == str(root / "report.json") else arg for arg in command]
            command.remove("--execute")
            command.append("--cleanup-only")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            after = (root / "calls").read_text().splitlines()
            self.assertEqual(after[len(before):], ["usage"])
            self.assertFalse((root / "receipt").exists())
            self.assertTrue(json.loads((root / "cleanup.json").read_text())["closed"])


if __name__ == "__main__":
    unittest.main()
