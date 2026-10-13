"""Synthetic benchmark dispatch and complete local controller/stdio paths, without official candidate smoke."""
import copy
import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import e17_local as local
from essaim import aider_agent, bigcode_pilot, bigcode_tasks, code_pilot, local_linux

RECEIPT = {"isolation_ready": local_linux.VERSION, "landlock_abi": 7, "seccomp": "on",
           "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}


def grade(count, passed=True, benchmark="bigcode"):
    value = {"status": "graded", "passed": passed, "tests": count, "tests_expected": count,
             "tests_complete": True, "failures": 0 if passed else 1, "errors": 0, "broken": [],
             "skipped": 0, "expected_failures": 0, "unexpected_successes": 0, "seconds": 0,
             "sandbox_backend": "strict", "sandbox": local_linux.VERSION, "isolation": copy.deepcopy(RECEIPT),
             "calls": [{"test": "test_" + str(i), "complete": True, "sandbox_backend": "strict",
                        "isolation": copy.deepcopy(RECEIPT)} for i in range(count)]}
    if benchmark == "bigcode":
        value.update(trust_scope=bigcode_tasks.TRUST_SCOPE, adversarial_attestation=False, scorable=True)
    return value


def smoke(benchmark):
    tasks = local.TASKS if benchmark == "aider" else local.BIGCODE_TASKS
    counts = (16, 8, 20) if benchmark == "aider" else (6, 5)
    rows = []
    for task, count in zip(tasks, counts):
        row = {"task": task, "reference": grade(count, benchmark=benchmark)}
        if benchmark == "aider":
            row["incorrect"] = grade(count, False, benchmark)
        else:
            row["negative"] = {name: grade(count, False) for name in ("wrong", "exit0", "load_exception")}
        rows.append(row)
    value = {"model_measurement": False, "sandbox_backend": "strict", "tasks": rows}
    if benchmark == "bigcode":
        value.update(kind="official_unittest_dev_smoke", trust_scope=bigcode_tasks.TRUST_SCOPE,
                     adversarial_attestation=False)
    return value


def args(root, benchmark="bigcode", action="run", mode="single"):
    return SimpleNamespace(benchmark=benchmark, action=action, mode=mode, output=root / "fresh",
        runtime=Path(sys.prefix), windows_python=Path("synthetic"), binary="synthetic", archive="synthetic",
        models_dir="synthetic", tasks_dir=root / "cache")


class DispatchTests(unittest.TestCase):
    def test_cli_default_aider_and_explicit_bigcode_preserve_old_actions(self):
        common = ["--runtime", "runtime", "--windows-python", "python", "--binary", "binary", "--archive", "archive",
                  "--models-dir", "models", "--tasks-dir", "cache", "--output", "fresh"]
        for action in ("preflight", "smoke", "run"):
            for options, benchmark in (([], "aider"), (["--benchmark", "bigcode"], "bigcode")):
                with mock.patch.object(sys, "argv", ["e17_local.py", action, *common, *options]), \
                        mock.patch.object(local, "execute", return_value=0) as execute:
                    self.assertEqual(local.main(), 0)
                    self.assertEqual(execute.call_args.args[0].benchmark, benchmark)
        with self.assertRaises(ValueError):
            local.benchmark_adapter("unopened")
        self.assertIs(local.benchmark_adapter("aider")[0], code_pilot)
        self.assertIs(local.benchmark_adapter("bigcode")[0], bigcode_pilot)

    def test_stage_bigcode_reads_exactly_two_pinned_rows_and_refuses_corruption_or_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, output = root / "cache", root / "fresh"
            cache.mkdir()
            output.mkdir()
            content = {task + ".json": json.dumps({"task": task}).encode() for task in local.BIGCODE_TASKS}
            for name, raw in content.items():
                (cache / name).write_bytes(raw)
            (cache / "BigCodeBench_208.json").write_bytes(b"UNOPENED SYNTHETIC POISON")
            view = {"dev": local.BIGCODE_TASKS, "files": {n: bigcode_tasks.digest(b) for n, b in content.items()}}
            with mock.patch.object(bigcode_tasks, "provenance", return_value=view), \
                    mock.patch.object(bigcode_tasks, "check_data") as check, \
                    mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
                data, observed = local.stage_dev(cache, output, "bigcode")
                self.assertEqual(observed, view)
                self.assertEqual(set(p.name for p in data.iterdir()), set(content))
                self.assertEqual([c.args[0].name for c in read.call_args_list], sorted(content))
                check.assert_called_once_with(data)
                (cache / "BigCodeBench_120.json").write_bytes(b"mutated")
                other = root / "other"
                other.mkdir()
                with self.assertRaises(ValueError):
                    local.stage_dev(cache, other, "bigcode")
                self.assertFalse((other / "data").exists())

    def test_smoke_requires_all_counts_controls_receipts_and_unattested_scope(self):
        value = smoke("bigcode")
        local.validate_smoke(value, "bigcode")
        mutations = [lambda v: v["tasks"].pop(), lambda v: v.update(adversarial_attestation=True),
            lambda v: v.update(trust_scope="attested"), lambda v: v["tasks"][0]["negative"].pop("exit0"),
            lambda v: v["tasks"][0]["reference"].update(tests=1),
            lambda v: v["tasks"][0]["reference"].update(skipped=1),
            lambda v: v["tasks"][0]["negative"]["wrong"].update(passed=True),
            lambda v: v["tasks"][1]["negative"]["exit0"].update(isolation=None),
            lambda v: v["tasks"][0]["reference"]["isolation"].update(landlock_abi=2),
            lambda v: v["tasks"][0]["reference"]["isolation"].update(namespaces=[]),
            lambda v: v["tasks"][0]["reference"].update(sandbox_backend="ephemeral-vm")]
        for mutate in mutations:
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.assertRaises(ValueError):
                local.validate_smoke(changed, "bigcode")
        local.validate_smoke(smoke("aider"))

    def test_invalid_runtime_or_data_blocks_before_windows_relay(self):
        for action in ("preflight", "smoke", "run"):
            for broken in ("runtime", "data"):
                with self.subTest(action=action, broken=broken), tempfile.TemporaryDirectory() as directory:
                    options = args(Path(directory), action=action)
                    with mock.patch.object(local.sys, "platform", "linux"), \
                            mock.patch.object(local.platform, "release", return_value="synthetic-microsoft"), \
                            mock.patch.object(bigcode_tasks, "check_runtime",
                                side_effect=ValueError("runtime") if broken == "runtime" else None), \
                            mock.patch.object(local, "stage_dev", side_effect=ValueError("data")) as stage, \
                            mock.patch.object(local, "PipeClient") as relay:
                        with self.assertRaises(ValueError):
                            local.execute(options)
                        relay.assert_not_called()
                        if broken == "runtime":
                            stage.assert_not_called()
                        registry = json.loads((options.output / "registry.json").read_text())
                        self.assertFalse(registry["complete"])
                        self.assertFalse(registry["ready_for_test"])
                        self.assertFalse(registry["adversarial_attestation"])
                        self.assertIsNone(json.loads((options.output / "failure.json").read_text())["score"])

    def test_invalid_final_source_keeps_bigcode_scope_without_claiming_a_verifier_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_args = bigcode_pilot.parser().parse_args([])
            initial = {"solution.py": "def task_func(x): return None\n"}
            env = mock.Mock(isolation=copy.deepcopy(RECEIPT))
            with mock.patch.object(bigcode_tasks, "copy_task", return_value=["solution.py"]), \
                    mock.patch.object(bigcode_tasks, "collect_sources",
                        side_effect=[initial, bigcode_tasks.SourceArtifactError("synthetic invalid source")]), \
                    mock.patch.object(bigcode_tasks, "instructions", return_value="synthetic"), \
                    mock.patch.object(bigcode_pilot, "LocalLinuxEnv", return_value=env), \
                    mock.patch.object(aider_agent, "run_episode",
                        return_value={"stopped": "invalid_source_artifact", "steps": []}):
                row = bigcode_pilot.run_one(run_args, local.BIGCODE_TASKS[0], "single", [local.QWEN],
                                           mock.Mock(), root, "synthetic")
            self.assertEqual((row["status"], row["passed"]), ("graded", False))
            self.assertFalse(row["verifier"]["adversarial_attestation"])
            self.assertIsNone(row["verifier"]["isolation"])
            local.validate_bigcode_episode(row, local.BIGCODE_TASKS[0], "single")
            row["passed"] = True
            with self.assertRaises(ValueError):
                local.validate_bigcode_episode(row, local.BIGCODE_TASKS[0], "single")

    def test_controller_selects_metadata_runtime_protocol_and_only_completes_after_closure(self):
        for benchmark in ("aider", "bigcode"):
            for mode in ("single", "vote"):
                for outcome in ("complete", "incomplete", "bad_close", "bad_exit", "bad_scope", "bad_smoke"):
                    if benchmark == "aider" and outcome == "bad_scope":
                        continue
                    with self.subTest(benchmark=benchmark, mode=mode, outcome=outcome), \
                            tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                        root = Path(directory)
                        options = args(root, benchmark, mode=mode)
                        pilot, selected, resource = local.benchmark_adapter(benchmark)
                        env = mock.Mock(isolation=copy.deepcopy(RECEIPT))
                        env.run.return_value = (0, "E17-STRICT-OK")
                        client = mock.Mock(loading_seconds=1, request_seconds=2)
                        client.receive.side_effect = [{"ready": {"synthetic": True}},
                                                      {"closed": outcome != "bad_close"}]
                        if outcome == "bad_exit":
                            client.close.side_effect = RuntimeError("nonzero worker exit")
                        view = {"synthetic": True}
                        def row(run_args, task, row_mode, models, transport, output, campaign_hash):
                            count = (6 if task == selected[0] else 5) if benchmark == "bigcode" else 16
                            result = {"task": task, "mode": row_mode, "status": "graded", "passed": False,
                                      "transcript_sha256": "0" * 64, "verifier": grade(count, False, benchmark)}
                            if benchmark == "bigcode":
                                result.update(trust_scope=bigcode_tasks.TRUST_SCOPE, adversarial_attestation=False)
                            if outcome == "incomplete":
                                result.update(status="error", passed=None)
                            if outcome == "bad_scope":
                                result["adversarial_attestation"] = True
                            return result
                        specimen = smoke(benchmark)
                        if outcome == "bad_smoke":
                            specimen["tasks"][0]["reference"]["passed"] = False
                        for target, name, kwargs in [
                            (local.sys, "platform", {"new": "linux"}),
                            (local.platform, "release", {"return_value": "synthetic-microsoft"}),
                            (local, "win_path", {"return_value": "synthetic"}),
                            (local, "PipeClient", {"return_value": client}),
                            (local, "stage_dev", {"return_value": (root, view)}),
                            (importlib.metadata, "version", {"return_value": "synthetic"}),
                            (local_linux, "LocalLinuxEnv", {"return_value": env}),
                            (bigcode_tasks if benchmark == "bigcode" else pilot, "smoke", {"return_value": specimen})]:
                            stack.enter_context(mock.patch.object(target, name, **kwargs))
                        if benchmark == "bigcode":
                            check = stack.enter_context(mock.patch.object(bigcode_tasks, "check_runtime"))
                        run = stack.enter_context(mock.patch.object(pilot, "run_one", side_effect=row))
                        if outcome == "complete":
                            self.assertEqual(local.execute(options), 0)
                        else:
                            with self.assertRaises((RuntimeError, ValueError)):
                                local.execute(options)
                        registry = json.loads((options.output / "registry.json").read_text())
                        runtime = json.loads((options.output / "runtime.json").read_text())
                        self.assertEqual(registry["benchmark"], benchmark)
                        self.assertEqual(registry["tasks"], selected)
                        self.assertEqual(registry["complete"], outcome == "complete")
                        self.assertFalse(registry["test_frozen"])
                        self.assertFalse(registry["ready_for_test"])
                        self.assertEqual(runtime["lock_sha256"], local.file_hash(local.ROOT / resource / "uv.lock"))
                        if benchmark == "bigcode":
                            check.assert_called_once_with()
                            self.assertEqual(runtime["trust_scope"], bigcode_tasks.TRUST_SCOPE)
                            self.assertFalse(runtime["adversarial_attestation"])
                        if outcome == "bad_smoke":
                            run.assert_not_called()
                            continue
                        manifest = json.loads((options.output / "manifest.json").read_text())
                        self.assertEqual(manifest["system"], aider_agent.SYSTEM)
                        self.assertEqual(manifest["limits"], local.LIMITS)
                        self.assertEqual(manifest["provenance_sha256"],
                                         local.file_hash(options.output / "provenance.json"))
                        self.assertEqual(set(json.loads((options.output / "provenance.json").read_text())), {"models"})
                        for call in run.call_args_list:
                            self.assertIs(call.args[4], client)
                            for name, value in local.LIMITS.items():
                                self.assertEqual(getattr(call.args[0], name), value)
                        self.assertFalse((options.output / "summary.json").exists())


@unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("E17_WINDOWS_TEST_PYTHON"),
                     "WSL/Windows stdio integration requires an existing Windows interpreter")
class StdioTests(unittest.TestCase):
    def test_both_real_controller_agent_and_windows_worker_paths_without_model(self):
        script = '''import sys
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
import e17_local as local
local.inspect_assets=lambda cfg: {'synthetic':True}
class Client:
    loading_seconds=request_seconds=0
    def __init__(self,*a,**k): pass
    def post(self,path,json,timeout):
        self.request_seconds+=.01
        last=json['messages'][-1]['content']
        command=('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' if 'Official test GREEN' in last
                 else 'MYRIAD_RUN_OFFICIAL_TESTS' if last.startswith('exit code:') else 'edit')
        value={'choices':[{'finish_reason':'stop','message':{'content':'```bash\\n'+command+'\\n```'}}],
               'usage':{'completion_tokens':1}}
        return SimpleNamespace(json=lambda:value)
    def stop(self): pass
    def close(self): pass
local.WindowsClient=Client
local.worker()
'''
        for benchmark in ("aider", "bigcode"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                options = args(root, benchmark, mode="vote")
                options.windows_python = Path(os.environ["E17_WINDOWS_TEST_PYTHON"])
                pilot, selected, _ = local.benchmark_adapter(benchmark)
                real_pipe = local.PipeClient
                def transport(command, config, log):
                    command = [str(options.windows_python), "-c", script, local.win_path(local.ROOT)]
                    return real_pipe(command, config, log)
                class Environment:
                    isolation = copy.deepcopy(RECEIPT)
                    def __init__(self, work, runtime): self.work = work
                    def run(self, command, timeout):
                        if command == "edit":
                            (self.work / "solution.py").write_text("def task_func(x): return 4\n")
                            return 0, ""
                        return 0, "E17-STRICT-OK"
                    def close(self): pass
                def copy_sources(cache, task, work):
                    (work / "solution.py").write_text("def task_func(x): return None\n")
                    return ["solution.py"]
                counts = dict(zip(selected, (16, 8, 20) if benchmark == "aider" else (6, 5)))
                def verify(cache, task, sources, runtime, timeout):
                    return grade(counts[task], "return 4" in sources["solution.py"], benchmark)
                stack.enter_context(mock.patch.object(local, "PipeClient", side_effect=transport))
                stack.enter_context(mock.patch.object(local, "stage_dev", return_value=(root, {"synthetic": True})))
                stack.enter_context(mock.patch.object(local_linux, "LocalLinuxEnv", Environment))
                stack.enter_context(mock.patch.object(pilot, "LocalLinuxEnv", Environment))
                stack.enter_context(mock.patch.object(bigcode_tasks if benchmark == "bigcode" else pilot,
                                                      "smoke", return_value=smoke(benchmark)))
                stack.enter_context(mock.patch.object(importlib.metadata, "version", return_value="synthetic"))
                if benchmark == "bigcode":
                    stack.enter_context(mock.patch.object(bigcode_tasks, "check_runtime"))
                    target, copy_name = bigcode_tasks, "copy_task"
                else:
                    target, copy_name = code_pilot, "copy_task"
                stack.enter_context(mock.patch.object(target, copy_name, side_effect=copy_sources))
                stack.enter_context(mock.patch.object(target, "instructions", return_value="synthetic fixture"))
                stack.enter_context(mock.patch.object(target, "verify", side_effect=verify))
                self.assertEqual(local.execute(options), 0)
                registry = json.loads((options.output / "registry.json").read_text())
                self.assertTrue(registry["complete"])
                self.assertEqual([r["task"] for r in registry["attempts"]], selected)
                for row in json.loads((options.output / "verdicts.json").read_text()):
                    self.assertTrue(row["submission_accepted"], row)
                    self.assertTrue(row["passed"])
                    self.assertEqual(row["requests"], 6)  # Two peers, unchanged three-step R00 loop.
                    self.assertEqual(row["charged_request_ceilings"], 6 * 4096)
                    if benchmark == "bigcode":
                        self.assertFalse(row["adversarial_attestation"])
