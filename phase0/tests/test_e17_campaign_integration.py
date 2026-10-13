"""Real backend/parser/grader interfaces on explicit synthetic files and launcher observations only."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from essaim import aider_python, code_pilot, ephemeral_vm, local_linux
from essaim.aider_agent import TEST
from essaim.agent import SUBMIT
from test_code_pilot import FakeClient, answer
from test_e17_campaign import campaign, e17_vm

READY = {"isolation_ready": "ephemeral-vm-v1", "landlock_abi": None, "seccomp": "on",
         "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}
MARKERS = {"hostname_file": True, "drive_binary": True}


def dataset(directory):
    original = json.loads((code_pilot.ROOT / "code_pilot/tasks.json").read_text(encoding="utf-8"))
    directory.mkdir()
    (directory / "LICENSE.python").write_text("Synthetic license fixture", encoding="utf-8")
    (directory / "README.md").write_text("Synthetic attribution fixture", encoding="utf-8")
    for task in campaign.TASKS:
        module, names = aider_python.APIS[task]
        base = directory / "python/exercises/practice" / task
        (base / ".meta").mkdir(parents=True)
        (base / ".docs").mkdir()
        content = "\n".join(f"def {name}(*args, **kwargs): return 'expected'" for name in names)
        (base / (module + ".py")).write_text(content, encoding="utf-8")
        (base / ".meta/example.py").write_text(content, encoding="utf-8")
        (base / ".docs/instructions.md").write_text("Synthetic API fixture only.", encoding="utf-8")
        tests = "import unittest\nfrom " + module + " import " + names[0] + "\nclass Synthetic(unittest.TestCase):\n"
        tests += "\n".join(f"    def test_{i}(self): self.assertEqual({names[0]}('synthetic'), 'expected')"
                           for i in range(e17_vm.SMOKE_TESTS[task]))
        (base / (module + "_test.py")).write_text(tests, encoding="utf-8")
        config = {"files": {"solution": [module + ".py"], "test": [module + "_test.py"],
                            "example": [".meta/example.py"]}}
        (base / ".meta/config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = {**original, "files": {p.relative_to(directory).as_posix(): {
        "size": p.stat().st_size, "sha256": campaign.digest(p.read_bytes())}
        for p in directory.rglob("*") if p.is_file()}}
    path = directory.parent / "synthetic-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class BackendIntegration(unittest.TestCase):
    def test_final_markers_are_files_and_old_directory_is_irrelevant(self):
        paths = []
        def is_file(path):
            paths.append(path.as_posix())
            return path.as_posix() in ("/var/colab/hostname", "/opt/google/drive/drive")
        with mock.patch.object(Path, "is_file", is_file), mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(ephemeral_vm.platform, "machine", return_value="x86_64"), \
             mock.patch.object(ephemeral_vm.platform, "release", return_value="6.6-synthetic"), \
             mock.patch.object(e17_vm.platform, "system", return_value="Linux"):
            image = e17_vm.colab()
        self.assertEqual(image["markers"], MARKERS)
        self.assertEqual(paths, ["/var/colab/hostname", "/opt/google/drive/drive"])

    def test_real_parser_environment_smoke_grader_and_r00_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, runtime, output = root / "data", root / "runtime", root / "output"
            manifest = dataset(cache)
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin/python").write_text("Synthetic runtime; never executed", encoding="utf-8")
            output.mkdir()
            observed = []
            def launcher(env, command, timeout_s):
                self.assertIsInstance(env, ephemeral_vm.EphemeralVMEnv)
                env.isolation = dict(READY)
                observed.append((command, timeout_s))
                if command == "printf E17-ISOLATION-OK": return 0, "E17-ISOLATION-OK"
                self.assertIn("python _call.py", command)
                worker = (env.work / "_call.py").read_text(encoding="utf-8")
                self.assertIn("sys.addaudithook(_hook)", worker)
                source = next(p for p in env.work.glob("*.py") if p.name != "_call.py")
                value = "expected" if "return 'expected'" in source.read_text(encoding="utf-8") else None
                (env.work / "_return.json").write_text(json.dumps({"value": value}), encoding="utf-8")
                return 0, "synthetic worker observation"
            with mock.patch.object(aider_python, "MANIFEST", manifest), \
                 mock.patch.object(sys, "platform", "linux"), mock.patch.object(sys, "prefix", str(runtime)), \
                 mock.patch.object(ephemeral_vm.platform, "machine", return_value="x86_64"), \
                 mock.patch.object(ephemeral_vm.platform, "release", return_value="6.6-synthetic"), \
                 mock.patch.object(e17_vm.platform, "system", return_value="Linux"), \
                 mock.patch.object(ephemeral_vm, "colab_markers", return_value=MARKERS), \
                 mock.patch.object(local_linux.LocalLinuxEnv, "run", launcher), \
                 mock.patch.object(e17_vm.urllib.request, "urlopen", side_effect=AssertionError("no network")), \
                 mock.patch.object(code_pilot, "DeadlineClient", side_effect=AssertionError("no real endpoint")):
                self.assertEqual(e17_vm.colab()["markers"], MARKERS)
                self.assertEqual(e17_vm.campaign_isolation(runtime), READY)
                args = code_pilot.parser().parse_args(["--sandbox-backend", "ephemeral-vm", "--smoke",
                    "--tasks-dir", str(cache), "--runtime", str(runtime), "--output", str(output)])
                with mock.patch.object(sys, "stdout", io.StringIO()):
                    self.assertEqual(code_pilot.execute(args), 0)
                smoke = json.loads((output / "smoke.json").read_text(encoding="utf-8"))
                e17_vm.validate_smoke(smoke)
                args.reference = None
                for task in campaign.TASKS:
                    client = FakeClient([answer(f"```bash\n{command}\n```", model=campaign.QWEN)
                                         for command in (TEST, f"echo {SUBMIT}")])
                    row = code_pilot.run_one(args, task, "single", [campaign.QWEN], client, output, "synthetic")
                    self.assertTrue(row["passed"], row)
                    self.assertIsNone(row["isolation"], "controller actions do not launch the episode shell")
                    e17_vm.validate_calls(row["verifier"])
                    events = [json.loads(line) for line in (output / f"single__{task}.jsonl").read_text().splitlines()]
                    e17_vm.validate_episode(row, events)
                    with self.assertRaises(ValueError):
                        e17_vm.validate_episode(row, [*events, {"kind": "step", "command": "true", "isolation": None}])
                    official = [e["note"]["official_test"] for e in events if "official_test" in e.get("note", {})]
                    self.assertEqual(len(official), 1)
                    e17_vm.validate_calls(official[0])
                self.assertTrue(observed)
                self.assertEqual(args.max_tokens, 4096)
                self.assertEqual(args.total_tokens, 49152)
            for changed in ({**READY, "landlock_abi": -1}, {**READY, "seccomp": "off"},
                            {**READY, "namespaces": ["user"]}):
                with self.assertRaises(ValueError): e17_vm.receipt(changed)


class AuditRegressions(unittest.TestCase):
    def test_worker_provenance_real_declarations_and_recovery(self):
        from essaim.code_provenance import declarations
        from test_e17_campaign import Validation, synthetic
        hardware = {"cpu": "synthetic", "host_memory_gib": 32, "execution": "synthetic VM",
                    "gpu": "synthetic GPU", "cpu_threads": "1"}
        bootstrap = {"artifact": {"version": "synthetic UV"}, "phase": "synthetic complete"}
        for comparison in (False, True):
            with self.subTest(comparison=comparison), tempfile.TemporaryDirectory() as temporary:
                root, request = Path(temporary), Validation().request(comparison)
                provenance = e17_vm.execution_provenance(request["plan"], hardware)
                self.assertEqual(set(provenance), {"models"})
                files, inventory = synthetic(request, provenance=provenance, bootstrap=bootstrap)
                for name, data in files.items(): (root / name).write_bytes(data)
                self.assertEqual(declarations(root / "provenance.json", sorted(request["plan"]["models"]),
                                              100, .25), provenance["models"])
                self.assertEqual(json.loads(files["runtime.json"])["uv_bootstrap"], bootstrap)
                self.assertEqual(json.loads(files["manifest.json"])["runtime"]["uv_bootstrap"], bootstrap)
                self.assertTrue(campaign.validate_retrieved(root, inventory, request))
                forbidden = {**provenance, "uv_bootstrap": bootstrap}
                files, inventory = synthetic(request, provenance=forbidden, bootstrap=bootstrap)
                for name, data in files.items(): (root / name).write_bytes(data)
                with self.assertRaisesRegex(ValueError, "exact documented nonsecret fields"):
                    declarations(root / "provenance.json", sorted(request["plan"]["models"]), 100, .25)
                with self.assertRaisesRegex(ValueError, "exact documented nonsecret fields"):
                    campaign.validate_retrieved(root, inventory, request)

    def test_exportable_metadata_are_canonical_and_bound_to_both_plans(self):
        files = campaign.source_bundle()
        for name in ("e17-dev-single", "e17-dev-compare"):
            selected = campaign.plan(name)
            self.assertEqual(campaign.digest(files[campaign.PROPOSAL]), selected["proposal_sha256"])
            self.assertEqual(campaign.digest(files[campaign.SELECTION]), selected["selection_sha256"])
            campaign.validate_plan(selected)

    def test_slow_install_preserves_worker_deadline_and_refuses_episode(self):
        from types import SimpleNamespace
        import e17_campaign_remote as remote
        import read_timeout
        clock = [100.0]
        deadlines = []
        pilot = SimpleNamespace(run_one=mock.Mock(side_effect=AssertionError("episode must not start")))
        client = SimpleNamespace(loading_seconds=0, request_seconds=0, close=mock.Mock())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            request = {"plan": campaign.plan("e17-dev-single"), "work_seconds": 1800, "nonce": "synthetic"}
            remote.save(root / "request.json", request)
            def run(command, **kwargs):
                deadlines.append(kwargs["deadline"])
                if str(root / "phase0/e17_vm.py") in command:
                    self.assertEqual(float(command[-1]), 1900)
                    self.assertEqual(kwargs["seconds"], 600)
                    with self.assertRaisesRegex(TimeoutError, "whole R00 episode"):
                        e17_vm.run_episodes(request["plan"], None, client, root / "artifacts",
                                            float(command[-1]), {}, {"attempts": []}, pilot)
                else:
                    self.assertEqual(kwargs["seconds"], 600)
                    clock[0] += 600
                return 0
            with mock.patch.object(remote.time, "monotonic", side_effect=lambda: clock[0]), \
                 mock.patch.object(read_timeout, "run", side_effect=run), mock.patch.object(remote.signal, "signal"):
                remote.worker(root)
            self.assertEqual(deadlines, [1900]*3)
            self.assertEqual(json.loads((root / "request.json").read_text()), request)
            pilot.run_one.assert_not_called()
            client.close.assert_called_once()

    def test_controller_uses_absolute_deadline_for_preparation(self):
        from test_e17_campaign import Validation, synthetic
        request = {**Validation().request(), "work_seconds": 3600}
        files, _ = synthetic(request)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            output.mkdir()
            (root / "phase0").mkdir()
            e17_vm.write(root / "request.json", request)
            bundle = campaign.source_bundle()
            for name in (campaign.PROPOSAL, campaign.SELECTION):
                (root / "phase0" / name).write_bytes(bundle[name])
            (output / "smoke.json").write_bytes(files["smoke.json"])
            with mock.patch.object(e17_vm, "colab", return_value=json.loads(files["runtime.json"])["colab_image"]), \
                 mock.patch.object(e17_vm, "campaign_isolation", return_value=READY), \
                 mock.patch.object(aider_python, "MANIFEST", aider_python.MANIFEST), \
                 mock.patch.object(code_pilot, "execute", return_value=0), \
                 mock.patch.object(e17_vm.time, "monotonic", return_value=3500), \
                 mock.patch.object(e17_vm, "fetch", side_effect=TimeoutError("synthetic download stop")) as fetch:
                with self.assertRaisesRegex(TimeoutError, "synthetic download stop"):
                    e17_vm.execute(root, 3700)
            self.assertEqual(fetch.call_args.args[-1], 3700)

    def test_real_zero_budget_verify_episode_and_recovery(self):
        import copy
        from test_e17_campaign import Validation, synthetic
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "data"
            manifest = dataset(cache)
            class NeverExecuted(ephemeral_vm.EphemeralVMEnv):
                def __init__(self, *args, **kwargs):
                    raise AssertionError("zero budget must not construct an environment")
            with mock.patch.object(aider_python, "MANIFEST", manifest):
                grade = aider_python.verify(cache, campaign.TASKS[0], {"affine_cipher.py": "synthetic"},
                                            root / "runtime", timeout_s=0, environment=NeverExecuted)
            self.assertEqual((grade["status"], grade["passed"]), ("graded", False))
            self.assertTrue(grade["calls"])
            e17_vm.validate_calls(grade)
            for change in ({"seconds": .001}, {"seconds": False}, {"complete": False},
                           {"sandbox_backend": "strict"}, {"python_audit_hook": False},
                           {"isolation": {}}, {"candidate_failure": "timeout"}, {"diagnostics": "launched"}):
                broken = copy.deepcopy(grade)
                broken["calls"][0].update(change)
                with self.assertRaises(ValueError): e17_vm.validate_calls(broken)
            for change in ({"passed": True}, {"status": "error"}):
                with self.assertRaises(ValueError): e17_vm.validate_calls({**grade, **change})
            request = Validation().request()
            files, inventory = synthetic(request)
            rows = json.loads(files["verdicts.json"])
            row = rows[0]
            row["verifier"] = grade
            events = [{"kind": "isolation", "receipt": row["isolation"]},
                      {"kind": "step", "command": TEST, "note": {"official_test": grade}},
                      {"kind": "end", **{k: v for k, v in row.items() if k != "transcript_sha256"}}]
            e17_vm.validate_episode(row, events)
            data = b"\n".join(campaign.canonical(e) for e in events) + b"\n"
            files[f"{row['mode']}__{row['task']}.jsonl"] = data
            row["transcript_sha256"] = campaign.digest(data)
            files["verdicts.json"] = campaign.canonical(rows)
            inventory["files"] = {n: {"bytes": len(d), "sha256": campaign.digest(d)} for n, d in files.items()}
            output = root / "recovered"
            output.mkdir()
            for name, data in files.items(): (output / name).write_bytes(data)
            self.assertTrue(campaign.validate_retrieved(output, inventory, request))

    def test_readme_pinned_and_real_copy_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "data"
            path = dataset(cache)
            original = json.loads(path.read_text())
            view = e17_vm.dev_manifest(original)
            self.assertEqual(view["files"]["README.md"], original["files"]["README.md"])
            path.write_text(json.dumps(view))
            with mock.patch.object(aider_python, "MANIFEST", path):
                work = root / "work"
                work.mkdir()
                aider_python.copy_task(cache, campaign.TASKS[0], work)
                self.assertIn("Synthetic attribution fixture", (work / "ATTRIBUTION.md").read_text())

    def test_sorted_json_real_episode_loop(self):
        import time
        from types import SimpleNamespace
        selected = json.loads(campaign.canonical(campaign.plan("e17-dev-compare")))
        self.assertEqual(selected["mode_order"], ["single", "vote", "cascade", "reference"])
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("no repository docs on VM")):
            campaign.validate_plan(selected)
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def run_one(args, task, mode, peers, client, output, checksum):
                calls.append((mode, task))
                (output / f"{mode}__{task}.jsonl").write_text('{}\n')
                return {"status": "graded", "transcript_sha256": "synthetic", "verifier": {}}
            client = SimpleNamespace(loading_seconds=0, request_seconds=0, close=mock.Mock())
            registry = {"attempts": []}
            with mock.patch.object(e17_vm, "validate_episode"), mock.patch.object(e17_vm, "validate_calls"):
                e17_vm.run_episodes(selected, None, client, output, time.monotonic()+2000, {}, registry,
                                    SimpleNamespace(run_one=run_one))
            expected = [(m, t) for m in selected["mode_order"] for t in campaign.TASKS]
            self.assertEqual(calls, expected)
            self.assertEqual([(a["mode"], a["task"]) for a in registry["attempts"]], expected)
            client.close.assert_called_once()
        selected["mode_order"].reverse()
        with self.assertRaises(ValueError): campaign.validate_plan(selected)

    def test_large_artifact_real_rpc_emit_and_exact_hash(self):
        import base64
        import os
        import safe_cli
        import e17_campaign_remote as remote
        from test_e17_campaign import supervisor
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"campaign_id": "a"*32, "nonce": "synthetic", "plan": campaign.plan("e17-dev-compare"),
                       "sources": {"source"+str(i): "b"*64 for i in range(200)}}
            (root / "artifacts").mkdir()
            data = os.urandom(9 << 20)
            (root / "artifacts/controller.log").write_bytes(data)
            remote.save(root / "request.json", request)
            inventory = {"schema": 1, "request": request, "complete": False,
                         "files": {"controller.log": {"bytes": len(data), "sha256": campaign.digest(data)}}}
            remote.save(root / "inventory.json", inventory)
            instance = object.__new__(supervisor.Campaign)
            instance.request, instance.owner = request, {"synthetic": True}
            instance.reconcile = lambda: instance.owner
            instance.monitor = lambda: None
            observed = []
            def fake_cli(args, monitor):
                scope = {"__name__": "synthetic_rpc"}
                prefix, invocation = Path(args[-1]).read_text().rsplit("\nemit_response(", 1)
                exec(prefix, scope)
                scope["root_for"] = lambda request: root
                raw = io.StringIO()
                with mock.patch.object(sys, "stdout", raw): exec("emit_response(" + invocation, scope)
                encoded = raw.getvalue().encode()
                self.assertTrue(all(len(line)+1 <= 65536 for line in encoded.splitlines()))
                clean = io.StringIO()
                safe_cli.emit(io.BytesIO(encoded), clean)
                self.assertNotIn("omitted", clean.getvalue())
                observed.append(len(encoded))
                return clean.getvalue()
            instance.cli = fake_cli
            self.assertEqual(instance.rpc("status")["inventory"], inventory)
            recovered = bytearray()
            while len(recovered) < len(data):
                value = instance.rpc("chunk", {"name": "controller.log", "offset": len(recovered)})
                recovered.extend(base64.b64decode(value["data"], validate=True))
            self.assertEqual(campaign.digest(recovered), campaign.digest(data))
            self.assertEqual(len(observed), 4)
            for value in ({"request": request, "operation": "start", "started": True},
                          {"request": request, "operation": "cancel", "cancel_requested": True}):
                clean = io.StringIO()
                safe_cli.emit(io.BytesIO(("\n".join(remote.response_lines(value))+"\n").encode()), clean)
                self.assertEqual(remote.decode_response(clean.getvalue()), value)
            lines = remote.response_lines({"request": request, "data": base64.b64encode(data[:4<<20]).decode()})
            for broken in (lines[:-1], lines+lines[-1:], list(reversed(lines))):
                with self.assertRaises(ValueError): remote.decode_response("\n".join(broken))
            max_rpcs = campaign.MAX_ARTIFACT_BYTES // campaign.CHUNK + len(campaign.artifacts(request["plan"])) + 2
            self.assertEqual(max_rpcs, 40)
            self.assertLessEqual(max_rpcs * 30 + 600, 1800)


if __name__ == "__main__":
    unittest.main()
