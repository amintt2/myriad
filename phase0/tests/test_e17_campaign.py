"""Synthetic runtime/data/receipts and an isolated fake CLI; never contact Colab or load a model."""
import base64
import importlib.util
import io
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab"))
# Import the pure module before the supervisor of the same basename.
spec = importlib.util.spec_from_file_location("e17_campaign", ROOT / "e17_campaign.py")
campaign = importlib.util.module_from_spec(spec)
sys.modules["e17_campaign"] = campaign
spec.loader.exec_module(campaign)
spec = importlib.util.spec_from_file_location("e17_colab_campaign", ROOT / "colab/e17_campaign.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
import e17_vm


def synthetic(request, mode="success", *, provenance=None, bootstrap=None):
    plan = request["plan"]
    receipt = {"isolation_ready": "ephemeral-vm-v1", "landlock_abi": None, "seccomp": "on",
               "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}
    call = {"isolation": receipt, "complete": True, "sandbox_backend": "ephemeral-vm", "python_audit_hook": True}
    smoke = {"kind": "deterministic_harness_smoke", "model_measurement": False, "tasks": [],
             "sandbox_backend": "ephemeral-vm", "sandbox": "ephemeral-vm-v1"}
    for task in campaign.TASKS:
        grade = {"status": "graded", "tests_complete": True, "tests": e17_vm.SMOKE_TESTS[task],
                 "tests_expected": e17_vm.SMOKE_TESTS[task],
                 "sandbox_backend": "ephemeral-vm", "calls": [call]}
        smoke["tasks"].append({"task": task, "reference": {**grade, "passed": True},
                               "incorrect": {**grade, "passed": False}})
    if provenance is None: provenance = {"models": {name: {**info, "declared_at": "synthetic", "model": name, "alias": name,
        "endpoint_kind": "synthetic-loopback", "licence": "apache-2.0", "quantization": "Q8_0",
        "server": {k: v for k, v in plan["server"].items() if k != "assets"} |
                  {"artifact": campaign.ASSETS[0]["file"], "sha256": campaign.ASSETS[0]["sha256"]},
        "hardware": {"cpu": "synthetic", "host_memory_gib": 32, "execution": "synthetic",
                     "gpu": plan["gpu"], "cpu_threads": "1"},
        "energy": {"watts": 100, "eur_kwh": .25, "measured": False, "scope": "synthetic"}}
        for name, info in plan["models"].items()}}
    image = {"markers": {"hostname_file": True, "drive_binary": True}, "kernel": "synthetic-kernel",
             "system": "Linux", "machine": "x86_64"}
    runtime = {"sources": request["sources"], "sandbox_backend": "ephemeral-vm", "colab_image": image,
               "cuda_devices": "CUDA0: synthetic device", "library_path": "/synthetic/server:/usr/lib64-nvidia"}
    if bootstrap is not None: runtime["uv_bootstrap"] = bootstrap
    dataset = e17_vm.dev_manifest(json.loads((ROOT / "code_pilot/tasks.json").read_text(encoding="utf-8")))
    manifest = {"request": request, "runtime": runtime, "execution_provenance": provenance,
                "dataset_hash": campaign.digest(campaign.canonical(dataset)),
                "mode_order": plan["mode_order"], "selection_sha256": plan["selection_sha256"],
                "limits": campaign.LIMITS, "models": plan["modes"], "isolation": receipt, "colab_image": image,
                "sandbox_backend": "ephemeral-vm", "sandbox": "ephemeral-vm-v1"}
    registry = {"request": request, "revision": "R00", "proposal": "P01", "decision": "no-harness-change",
                "selection_sha256": plan["selection_sha256"], "scientific_gain_claimed": False, "attempts": [], "dev_attempt": plan["dev_attempt"]}
    files = {"manifest.json": campaign.canonical(manifest), "smoke.json": campaign.canonical(smoke),
             "dataset-manifest.json": campaign.canonical(dataset),
             "provenance.json": campaign.canonical(provenance), "runtime.json": campaign.canonical(runtime),
             "controller.log": b"synthetic controller closed\n", "server-events.jsonl": b""}
    rows = []
    for selected in plan["mode_order"]:
        for task in campaign.TASKS:
            row = {"mode": selected, "task": task, "campaign_hash": campaign.digest(campaign.canonical(manifest)),
                   "status": "graded", "passed": False, "isolation": receipt,
                   "verifier": {"status": "graded", "passed": False,
                   "sandbox_backend": "ephemeral-vm", "calls": [call]}}
            events = [{"kind": "start"}, {"kind": "isolation", "receipt": receipt}, {"kind": "end", **row}]
            data = b"\n".join(campaign.canonical(e) for e in events) + b"\n"
            files[f"{selected}__{task}.jsonl"] = data
            row["transcript_sha256"] = campaign.digest(data)
            rows.append(row)
            registry["attempts"].append({"mode": selected, "task": task, "state": "graded"})
    if mode == "false-success": rows.pop()
    files["verdicts.json"] = campaign.canonical(rows)
    files["registry.json"] = campaign.canonical(registry)
    files["vm-status.json"] = campaign.canonical({"request": request, "state": "complete", "error": None})
    if mode == "partial": files.pop("smoke.json")
    inventory = {"schema": 1, "request": request, "complete": mode != "partial",
                 "files": {name: {"bytes": len(data), "sha256": campaign.digest(data)} for name, data in files.items()}}
    if mode == "bad-hash": inventory["files"]["manifest.json"]["sha256"] = "0" * 64
    return files, inventory


class Validation(unittest.TestCase):
    def request(self, comparison=False):
        return {"campaign_id": "a" * 32, "allocation_id": "synthetic", "nonce": "fresh", "sources": {
                campaign.SELECTION: campaign.plan("e17-dev-single")["selection_sha256"],
                campaign.PROPOSAL: campaign.plan("e17-dev-single")["proposal_sha256"]},
                "plan": campaign.plan("e17-dev-compare") if comparison else campaign.plan("e17-dev-single")}

    def test_plans_catalog_pins_and_unchanged_budgets(self):
        plan = campaign.plan("e17-dev-single")
        self.assertEqual(plan["limits"], campaign.LIMITS)
        self.assertEqual(plan["gpu"], "L4")
        self.assertEqual(list(plan["models"]), [campaign.QWEN])
        self.assertEqual(plan["server"]["parallel"], 1)
        with self.assertRaises(ValueError): campaign.plan("TEST")
        with self.assertRaises(ValueError): campaign.plan("e17-dev-compare", "b" * 40)
        self.assertEqual(self.request(True)["plan"]["gpu"], "A100")
        files = campaign.source_bundle()
        self.assertNotIn("results", files)
        self.assertTrue(all(not n.startswith(("data/", "results/", "app/")) for n in files))

    def test_complete_recovery_and_false_success_partial_hashes(self):
        for mode in ("success", "false-success", "partial", "bad-hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root, request = Path(temporary), self.request()
                files, inventory = synthetic(request, mode)
                for name, data in files.items(): (root / name).write_bytes(data)
                if mode == "success": self.assertTrue(campaign.validate_retrieved(root, inventory, request))
                else:
                    with self.assertRaises(ValueError): campaign.validate_retrieved(root, inventory, request)

    def test_selection_order_and_registry_tampering_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, request = Path(temporary), self.request(True)
            files, inventory = synthetic(request)
            for name, data in files.items(): (root / name).write_bytes(data)
            self.assertTrue(campaign.validate_retrieved(root, inventory, request))
            for filename, key in (("registry.json", "selection_sha256"), ("manifest.json", "mode_order")):
                value = json.loads(files[filename])
                value[key] = "tampered"
                data = campaign.canonical(value)
                (root / filename).write_bytes(data)
                altered = {**inventory, "files": {**inventory["files"], filename: {
                    "bytes": len(data), "sha256": campaign.digest(data)}}}
                with self.assertRaises(ValueError): campaign.validate_retrieved(root, altered, request)
                (root / filename).write_bytes(files[filename])

    def test_unsafe_inventory_and_receipts_refused(self):
        request = self.request()
        _, inventory = synthetic(request)
        for change in ({"request": {}}, {"schema": 2}, {"complete": 1},
                       {"files": {"../foreign": {"bytes": 1, "sha256": "a" * 64}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                campaign.validate_inventory({**inventory, **change}, request)
        for value in (None, {}, {"backend": "strict"}, {"backend": "ephemeral-vm", "colab_detected": False}):
            with self.assertRaises(ValueError): e17_vm.receipt(value)

    def test_unknown_account_is_latched_until_cleanup_not_retried_by_recovery(self):
        args = SimpleNamespace(seconds=90, budget_units=4, min_balance_units=15, cleanup_seconds=1,
                               max_rate_units_hour=5.3, read_seconds=1, grace=.1, recover_seconds=15)
        value = supervisor.Campaign(args, campaign.plan("e17-dev-single"), {})
        with mock.patch.object(supervisor.e17_probe.Probe, "monitor", side_effect=TimeoutError("synthetic")) as monitor:
            with self.assertRaises(TimeoutError): value.monitor()
            with self.assertRaises(ValueError): value.monitor()
        self.assertEqual(monitor.call_count, 1)

    def test_download_hash_size_timeout_and_network_are_synthetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, data = Path(temporary) / "synthetic.bin", b"synthetic"
            info = {"bytes": len(data), "sha256": campaign.digest(data)}
            for returned in (data, data + b"x", b"corrupt"):
                with mock.patch.object(e17_vm.urllib.request, "urlopen", return_value=io.BytesIO(returned)):
                    if returned == data:
                        e17_vm.fetch("https://synthetic.invalid/fixture", path, info, time.monotonic() + 10)
                        self.assertEqual(path.read_bytes(), data)
                    else:
                        with self.assertRaises(ValueError):
                            e17_vm.fetch("https://synthetic.invalid/fixture", path, info, time.monotonic() + 10)
            with mock.patch.object(e17_vm.urllib.request, "urlopen", side_effect=OSError("synthetic network")):
                with self.assertRaises(OSError): e17_vm.fetch("synthetic", path, info, time.monotonic() + 10)
            with mock.patch.object(e17_vm.urllib.request, "urlopen", return_value=io.BytesIO(data)):
                with self.assertRaises(TimeoutError): e17_vm.fetch("synthetic", path, info, time.monotonic() - 1)
            self.assertFalse(path.with_suffix(".bin.part").exists())

    def test_no_local_science_and_missing_backend_refuses(self):
        from essaim.local_linux import IsolationError
        with mock.patch.object(e17_vm.sys, "platform", "win32"):
            with self.assertRaises(IsolationError): e17_vm.colab()
        with self.assertRaises(ValueError): supervisor.require_backend({"essaim/code_pilot.py": b"synthetic strict"})

    def test_dataset_scope_preserves_pins_and_never_stages_unseen_bytes(self):
        manifest = json.loads((ROOT / "code_pilot/tasks.json").read_text())
        view = e17_vm.dev_manifest(manifest)
        self.assertLess(len(view["files"]), len(manifest["files"]))
        self.assertEqual(view["tasks"], manifest["tasks"])
        for name, info in view["files"].items():
            self.assertEqual(info, manifest["files"][name])
            self.assertTrue(name in ("LICENSE.python", "README.md") or any(
                name.startswith("python/exercises/practice/" + t + "/") for t in campaign.TASKS))

    def test_sequential_client_switches_one_owned_server_and_preserves_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            output.mkdir()
            binary = root / "synthetic-server"
            binary.write_bytes(b"synthetic executable fixture, never executed")
            factory = mock.Mock()
            factory.return_value.post.side_effect = [
                mock.Mock(json=lambda: {"model": campaign.QWEN}),
                mock.Mock(json=lambda: {"model": campaign.GRANITE}),
                mock.Mock(json=lambda: {"model": campaign.GRANITE})]
            process1, process2 = mock.Mock(pid=10101, returncode=0), mock.Mock(pid=20202, returncode=0)
            process1.poll.return_value = process2.poll.return_value = None
            http = mock.Mock()
            http.__enter__ = mock.Mock(return_value=mock.Mock(get=mock.Mock(return_value=SimpleNamespace(status_code=200))))
            http.__exit__ = mock.Mock(return_value=False)
            httpx = SimpleNamespace(Client=mock.Mock(return_value=http), TransportError=OSError)
            with mock.patch.object(e17_vm.subprocess, "Popen", side_effect=[process1, process2]) as popen, \
                 mock.patch.object(e17_vm.os, "killpg", create=True) as kill, \
                 mock.patch.object(e17_vm.signal, "SIGKILL", 9, create=True), \
                 mock.patch.dict(sys.modules, {"httpx": httpx}):
                environment = {"LD_LIBRARY_PATH": "synthetic-library-path"}
                client = e17_vm.SequentialClient(binary, campaign.MODELS, root, output,
                                                 time.monotonic() + 10, factory, environment)
                try:
                    for model in (campaign.QWEN, campaign.GRANITE, campaign.GRANITE):
                        payload = {"model": model, "temperature": 0.0, "seed": 0, "max_tokens": 4096}
                        client.post("chat/completions", json=payload, timeout=5)
                        self.assertEqual(factory.return_value.post.call_args.kwargs["json"], payload)
                finally:
                    client.close()
            self.assertEqual(popen.call_count, 2)
            for call in popen.call_args_list:
                self.assertIs(call.kwargs["env"], environment)
                args = call.args[0]
                self.assertEqual(args[args.index("--device") + 1], "CUDA0")
                self.assertEqual(args[args.index("-np") + 1], "1")
                self.assertEqual(args[args.index("--reasoning-budget") + 1], "0")
            self.assertEqual({call.args[0] for call in kill.call_args_list}, {10101, 20202})
            self.assertEqual(process1.wait.call_count, 1)
            self.assertEqual(process2.wait.call_count, 1)
            events = [json.loads(line) for line in (output / "server-events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(e["event"] == "loaded" for e in events), 2)
            self.assertEqual(sum(e["event"] == "request" for e in events), 3)
            self.assertTrue(all(e["seconds"] >= 0 for e in events if e["event"] in ("loaded", "request")))

    def test_cuda_library_environment_and_cpu_fallback_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "cudart/lib/libcudart.so"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"synthetic shared library, never loaded")
            environment = e17_vm.cuda_environment(root / "bin/llama-server", root)
            self.assertIn(str(library.parent), environment["LD_LIBRARY_PATH"])
            self.assertIn("/usr/lib64-nvidia", environment["LD_LIBRARY_PATH"])
            for text, allowed in (("CUDA0: synthetic device", True), ("CPU0: synthetic device", False)):
                with mock.patch.object(e17_vm.subprocess, "run", return_value=SimpleNamespace(stdout=text, stderr="")) as run:
                    if allowed: self.assertEqual(e17_vm.cuda_devices(root / "server", environment), text)
                    else:
                        with self.assertRaises(ValueError): e17_vm.cuda_devices(root / "server", environment)
                    self.assertIs(run.call_args.kwargs["env"], environment)

    def test_server_startup_failure_is_not_retried_and_owned_child_is_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            output.mkdir()
            binary = root / "synthetic-server"
            binary.write_bytes(b"synthetic")
            process = mock.Mock(pid=30303, returncode=1)
            process.poll.return_value = 1
            with mock.patch.object(e17_vm.subprocess, "Popen", return_value=process) as popen, \
                 mock.patch.object(e17_vm.os, "killpg", create=True) as kill, \
                 mock.patch.object(e17_vm.signal, "SIGKILL", 9, create=True):
                client = e17_vm.SequentialClient(binary, campaign.MODELS, root, output, time.monotonic() + 10, mock.Mock())
                try:
                    with self.assertRaises(TimeoutError):
                        client.post("chat/completions", json={"model": campaign.QWEN}, timeout=5)
                finally:
                    client.close()
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(all(call.args[0] == 30303 for call in kill.call_args_list))


FAKE = r'''import ast, base64, json, os, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT']); mode = os.environ['MODE']; args = sys.argv[1:]; op = args[0]
active = root / 'active'
with (root / 'calls').open('a') as out: out.write(op + '\n')
if op == 'sessions':
    if mode == 'foreign-preflight': active.touch()
    endpoint = 'foreign' if (root / 'replaced').exists() else 'synthetic'
    print(f'[phase0] {endpoint} | Hardware: L4 | Shape: Standard | Variant: GPU' if active.exists()
          else '[colab] No active sessions found on server.')
elif op == 'usage':
    if mode == 'usage-unknown' and active.exists() and (root / 'remote-started').exists(): sys.exit(0)
    if mode == 'final-unknown' and not active.exists() and (root / 'allocated').exists(): sys.exit(0)
    print('Current balance: ' + ('16.00' if mode == 'low-balance' else '59.62') + ' compute units\nUsage rate: ' +
          ('5.30/hr\nActive assignments: 1' if active.exists() else '0.00/hr\nActive assignments: 0'))
elif op == 'new':
    active.touch(); (root / 'allocated').touch()
    if mode == 'ambiguous-new': sys.exit(27)
elif op == 'stop':
    if mode == 'stop-retry' and not (root / 'stop-failed').exists():
        (root / 'stop-failed').touch(); sys.exit(27)
    active.unlink(missing_ok=True)
elif op == 'exec':
    if args[-1].endswith('snapshot.py'):
        print(json.dumps({'schema': 1, 'campaign_present': mode == 'foreign-campaign'})); sys.exit(0)
    tree = ast.parse(Path(args[-1]).read_text())
    call = tree.body[-1].value.args[0]
    request, action, payload = [ast.literal_eval(a) for a in call.args]
    value = {'request': request, 'operation': action}
    if action == 'start':
        if mode == 'foreign-start': value.update(foreign_campaign=True, started=False)
        (root / 'remote-started').touch()
        if mode != 'foreign-start': value['started'] = True
        if mode == 'lost-start-ack': sys.exit(27)
    elif action == 'cancel': value['cancel_requested'] = True
    elif action == 'status':
        if mode in ('term', 'timeout'): time.sleep(30)
        if mode == 'network-fail': sys.exit(27)
        if mode == 'replaced-session': (root / 'replaced').touch()
        sys.path.insert(0, os.environ['TEST_MODULES'])
        from test_e17_campaign import synthetic
        _, value['inventory'] = synthetic(request, mode)
    elif action == 'chunk':
        sys.path.insert(0, os.environ['TEST_MODULES'])
        from test_e17_campaign import synthetic
        files, _ = synthetic(request, mode)
        value.update(name=payload['name'], offset=payload['offset'],
                     data=base64.b64encode(files[payload['name']][payload['offset']:payload['offset']+(4<<20)]).decode())
    if mode == 'foreign-ack': value['request'] = {}
    sys.path.insert(0, str(Path(os.environ['TEST_MODULES']).parent / 'colab'))
    from e17_campaign_remote import emit_response
    emit_response(value)
else: sys.exit(81)
'''

RUNNER = r'''import importlib.util, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import test_e17_campaign as fixture
module = fixture.supervisor
# Substitute only the backend readiness check: every official operation still goes through safe_cli and the fake.
module.require_backend = lambda files: None
original = module.lifecycle.save
def save(path, value):
    stage = 'owned' if value.get('status') == 'owned' else 'returned' if value.get('allocation_returned') else 'pending'
    if os.environ['MODE'] == 'persist-' + stage and Path(path).name == 'receipt':
        raise OSError('synthetic persistence failure')
    return original(path, value)
module.lifecycle.save = save
sys.exit(module.main())
'''


@unittest.skipUnless(sys.platform == "linux", "fake CLI lifecycle processes require Linux/WSL")
class Processes(unittest.TestCase):
    def fixture(self, root, mode):
        bins = root / ".local/bin"
        bins.mkdir(parents=True)
        fake = bins / "colab"
        fake.write_text(f"#!{sys.executable}\n" + FAKE, encoding="utf-8")
        fake.chmod(0o755)
        runner = root / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        env = {**os.environ, "HOME": str(root), "PATH": str(bins) + ":" + os.environ["PATH"],
               "TEST_ROOT": str(root), "MODE": mode, "TEST_MODULES": str(ROOT / "tests"),
               "DLLM_CLI_SECONDS": "1", "DLLM_CLI_GRACE": ".1"}
        for key in ("DLLM_LIFECYCLE_LOCK_FD", "DLLM_OPERATION_IDENTITY", "DLLM_CAMPAIGN_ID",
                    "DLLM_CAMPAIGN_RECEIPT", "MYRIAD_BENCH_TOKEN", "MYRIAD_BENCH_URL"):
            env.pop(key, None)
        command = [sys.executable, str(runner), str(ROOT / "tests"), "--execute", "--output", str(root / "report.json"),
                   "--receipt", str(root / "receipt"), "--seconds", "90", "--budget-units", "4", "--spent-units", "0",
                   "--read-seconds", "1", "--recover-seconds", "15", "--cleanup-seconds", "1.5",
                   "--usage-seconds", ".1", "--poll-seconds", ".01", "--retry-seconds", ".01", "--grace", ".1"]
        return env, command

    def test_complete_partial_false_success_and_recovery_cleanup(self):
        for mode in ("success", "partial", "false-success", "bad-hash", "lost-start-ack", "foreign-ack", "network-fail",
                     "timeout", "usage-unknown", "stop-retry", "persist-returned", "persist-owned"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0 if mode in ("success", "stop-retry") else 1, output)
                report = json.loads((root / "report.json").read_text())
                calls = (root / "calls").read_text().splitlines()
                self.assertEqual(calls.count("new"), 1, output)
                self.assertIn("stop", calls, output)
                self.assertEqual(calls[-1], "usage", output)
                self.assertTrue(report["closed"], output)
                self.assertTrue(report["absence_confirmed"], output)
                self.assertFalse((root / "active").exists(), output)
                if mode == "lost-start-ack": self.assertTrue(report["retrieved_complete"], output)
                if mode == "partial":
                    self.assertTrue((root / "report-artifacts/recovery.json").exists())

    def test_no_foreign_stop_or_retry_ambiguous_allocation(self):
        for mode in ("ambiguous-new", "foreign-preflight", "low-balance", "foreign-campaign", "foreign-start",
                     "replaced-session"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                env, command = self.fixture(root, mode)
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=25)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines()
                self.assertNotIn("stop", calls)
                self.assertLessEqual(calls.count("new"), 1)

    def test_unknown_final_account_retains_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, command = self.fixture(root, "final-unknown")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=25)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertTrue((root / "receipt").exists())
            report = json.loads((root / "report.json").read_text())
            self.assertTrue(report["absence_confirmed"])
            self.assertFalse(report["closed"])

    def test_term_still_recovers_and_closes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, command = self.fixture(root, "term")
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 8
                while not (root / "remote-started").exists() and time.monotonic() < deadline: time.sleep(.01)
                self.assertTrue((root / "remote-started").exists())
                os.kill(process.pid, signal.SIGTERM)
                output, _ = process.communicate(timeout=25)
                self.assertEqual(process.returncode, 1, output)
                report = json.loads((root / "report.json").read_text())
                self.assertTrue(report["closed"], output)
                self.assertTrue(report["absence_confirmed"], output)
                self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_plan_cli_never_calls_real_or_fake_cli_and_budget_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, command = self.fixture(root, "success")
            command.remove("--execute")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((root / "calls").exists())
            command.extend(("--spent-units", "19"))
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((root / "calls").exists())


if __name__ == "__main__":
    unittest.main()
