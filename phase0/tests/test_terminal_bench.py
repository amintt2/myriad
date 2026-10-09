"""Harbor contracts and pilot safeguards, without Docker or a served model."""
import asyncio
import json
import os
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from essaim import agent, terminal_bench as bench


class FakeHarbor:
    """A borrowed async environment: teardown before verification would lose the artifact."""
    def __init__(self):
        self.calls, self.files, self.stopped = [], {}, False
        self.executing, self.cancelled = asyncio.Event(), asyncio.Event()

    async def exec(self, command, timeout_sec):
        self.calls.append((command, timeout_sec))
        words = shlex.split(command)
        if words[:2] != ["timeout", "--signal=KILL"] or words[3:5] != ["bash", "-lc"]:
            raise AssertionError("command must run in a timed shell inside Harbor")
        cmd = words[5]
        if cmd == "sleep 999":
            self.executing.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        if cmd == "write artifact":
            self.files["answer"] = "42"
            return SimpleNamespace(return_code=0, stdout="saved\n", stderr=None)
        if cmd == "slow command":
            return SimpleNamespace(return_code=137, stdout="partial\n", stderr="Killed\n")
        return SimpleNamespace(return_code=2, stdout=None, stderr="unknown command\n")

    def verify(self):
        if self.stopped:
            raise AssertionError("premature environment deletion")
        return {"verifier_result": {"rewards": {"reward": int(self.files.get("answer") == "42")}}}


class HarborBridge(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("timeout"),
                         "Linux bash and GNU timeout required")
    async def test_timeout_actually_kills_the_inner_shell(self):
        class ShellEnvironment:
            async def exec(self, command, timeout_sec):
                process = await asyncio.create_subprocess_exec("bash", "-c", command,
                                                              stdout=asyncio.subprocess.PIPE,
                                                              stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
                return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())
        env = agent.HarborEnv(ShellEnvironment(), asyncio.get_running_loop())
        code, output = await asyncio.to_thread(env.run, "printf 'start\\n'; sleep 0.5; printf 'end\\n'", 0.1)
        env.close()
        self.assertIn(code, (124, 137))  # GNU and uutils report KILL differently.
        self.assertIn("start", output)
        self.assertNotIn("end", output)

    async def test_execution_keeps_artifacts_for_verifier_and_preserves_errors(self):
        harbor = FakeHarbor()
        env = agent.HarborEnv(harbor, asyncio.get_running_loop())
        replies = iter(["```bash\nunknown\n```", "```bash\nwrite artifact\n```",
                        f"```bash\necho {agent.SUBMIT}\n```"])
        episode = await asyncio.to_thread(agent.run_episode, lambda *x: next(replies), env, "repair", ["m"])
        env.close()
        self.assertEqual(episode["stopped"], "submitted")
        self.assertEqual(episode["steps"][0].exit_code, 2)
        self.assertEqual(episode["steps"][0].output, "unknown command\n")
        self.assertTrue(bench.verdict(harbor.verify())["passed"])
        self.assertFalse(harbor.stopped)
        with self.assertRaises(RuntimeError):
            await asyncio.to_thread(env.run, "write artifact", 1)

    async def test_command_timeout_is_observation_not_success(self):
        harbor = FakeHarbor()
        env = agent.HarborEnv(harbor, asyncio.get_running_loop())
        self.assertEqual(await asyncio.to_thread(env.run, "slow command", 0.25), (137, "partial\nKilled\n"))
        self.assertEqual(harbor.calls[0][1], 6)
        self.assertIn("0.250000s", harbor.calls[0][0])
        env.close()
        self.assertFalse(bench.verdict(harbor.verify())["passed"])

    async def test_close_cancels_active_exec_and_unblocks_worker(self):
        harbor = FakeHarbor()
        env = agent.HarborEnv(harbor, asyncio.get_running_loop())
        worker = asyncio.create_task(asyncio.to_thread(env.run, "sleep 999", 60))
        await asyncio.wait_for(harbor.executing.wait(), 2)
        env.close()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(worker, 2)
        await asyncio.wait_for(harbor.cancelled.wait(), 2)
        self.assertFalse(harbor.stopped)


class PilotSafeguards(unittest.TestCase):
    def test_provenance_detects_changed_missing_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "task.toml"
            path.write_bytes(b"pinned")
            manifest = {"files": {"task.toml": {"size": 6, "sha256": bench.digest(b"pinned")}}}
            bench.check_tasks(directory, manifest)
            path.write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "provenance"):
                bench.check_tasks(directory, manifest)
            path.unlink()
            with self.assertRaises(ValueError):
                bench.check_tasks(directory, manifest)
            path.write_bytes(b"pinned")
            (directory / "unexpected").touch()
            with self.assertRaises(ValueError):
                bench.check_tasks(directory, manifest)

    def test_download_refuses_untrusted_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"commit": "fixed", "files": {"LICENSE": {"size": 6, "sha256": bench.digest(b"pinned")}}}
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = b"edited"
            with patch.object(bench.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "download provenance"):
                    bench.prepare(Path(tmp), manifest)
            self.assertFalse((Path(tmp) / "LICENSE").exists())

    def test_official_reward_is_required_and_agent_incident_is_preserved(self):
        self.assertFalse(bench.verdict({"agent_result": {"submitted": True}})["passed"])
        self.assertEqual(bench.verdict({"verifier_result": {"rewards": {"reward": 0}}})["status"], "graded")
        self.assertTrue(bench.verdict({"verifier_result": {"rewards": {"reward": 1}}})["passed"])
        for kind in ("AgentTimeoutError", "NonZeroAgentExitCodeError"):
            for reward in (0, 1):
                with self.subTest(kind=kind, reward=reward):
                    result = {"verifier_result": {"rewards": {"reward": reward}},
                              "exception_info": {"exception_type": kind}}
                    grade = bench.verdict(result)
                    self.assertEqual(grade["status"], "graded")
                    self.assertEqual(grade["passed"], bool(reward))
                    self.assertEqual(grade["official_rewards"], {"reward": reward})
                    self.assertEqual(grade["exception_type"], kind)

    def test_verifier_and_infrastructure_errors_leave_score_incomplete(self):
        for kind in ("AgentTimeoutError", "VerifierTimeoutError", "RewardFileNotFoundError", "DockerError"):
            with self.subTest(kind=kind):
                grade = bench.verdict({"exception_info": {"exception_type": kind}})
                self.assertEqual(grade["status"], "error")
                self.assertIsNone(grade["official_rewards"])
                self.assertEqual(grade["exception_type"], kind)
                rows = [{"mode": "single", "task": "a", **grade}]
                report = bench.summarize(rows, ["single"], ["a"])["modes"]["single"]
                self.assertFalse(report["complete"])
                self.assertIsNone(report["pass_at_1"])
        for kind in ("VerifierTimeoutError", "DockerError"):
            grade = bench.verdict({"verifier_result": {"rewards": {"reward": 1}},
                                   "exception_info": {"exception_type": kind}})
            self.assertEqual(grade["status"], "error")
            self.assertEqual(grade["official_rewards"], {"reward": 1})

    def test_incomplete_or_infrastructure_failure_has_no_score(self):
        tasks = ["a", "b", "c"]
        rows = [{"mode": "single", "task": name, "passed": False, "status": "graded"} for name in tasks]
        self.assertEqual(bench.summarize(rows, ["single"], tasks)["modes"]["single"]["pass_at_1"], 0)
        rows[0]["status"] = "error"
        self.assertIsNone(bench.summarize(rows, ["single"], tasks)["modes"]["single"]["pass_at_1"])
        self.assertIsNone(bench.summarize(rows[1:], ["single"], tasks)["modes"]["single"]["pass_at_1"])

    def test_endpoint_requires_authentication_and_no_embedded_secret(self):
        bench.validate_endpoint("http://127.0.0.1:8400/v1", "test-token")
        bench.validate_endpoint("https://example.org/v1", "test-token")
        for url, token in (("http://example.org/v1", "x"), ("https://x:secret@example.org/v1", "x"),
                           ("https://example.org/v1?token=secret", "x"), ("http://localhost/v1", "")):
            with self.assertRaises(ValueError):
                bench.validate_endpoint(url, token)

    def test_deadline_prevents_command_after_slow_generation(self):
        env = SimpleNamespace(run=unittest.mock.Mock())
        with patch.object(agent.time, "monotonic", side_effect=[0, 0, 2]):
            result = agent.run_episode(lambda *x: "```bash\nwrite artifact\n```", env, "t", ["m"], timeout_s=1)
        self.assertEqual(result["stopped"], "timeout")
        env.run.assert_not_called()

    def test_generation_allowance_stops_without_executing_partial_vote(self):
        env = SimpleNamespace(run=unittest.mock.Mock())
        def chat(model, messages):
            if model == "b":
                raise agent.GenerationLimit()
            return "```bash\nwrite artifact\n```"
        result = agent.run_episode(chat, env, "t", ["a", "b"], strategy="vote")
        self.assertEqual(result["stopped"], "generation_limit")
        env.run.assert_not_called()


try:
    from harbor.environments.base import ExecResult
    from harbor.models.task.config import TaskOS
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from harbor.verifier.verifier import Verifier
except ImportError:
    Verifier = None


@unittest.skipIf(Verifier is None, "dedicated Harbor environment required")
class OfficialHarbor(unittest.IsolatedAsyncioTestCase):
    async def test_pilot_builds_nine_fresh_trials_with_identical_limits(self):
        from harbor.trial.trial import Trial
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as tmp:
            args = bench.parser().parse_args(["--single", "a", "--compare", "--vote", "a", "b",
                                               "--reference", "large", "--max-steps", "7", "--timeout", "19",
                                               "--cmd-timeout", "2", "--total-tokens", "24", "--seed", "3"])
            configs = []
            async def create(config):
                configs.append(config)
                result = SimpleNamespace(model_dump=lambda **kwargs: {"verifier_result": {"rewards": {"reward": 0}}})
                return SimpleNamespace(run=AsyncMock(return_value=result))
            with patch.object(Trial, "create", side_effect=create):
                complete = await bench.run_trials(args, {"tasks": ["x", "y", "z"]}, Path(tmp), "1" * 32)
            self.assertTrue(complete)
            self.assertEqual(len({config.trial_name for config in configs}), 9)
            self.assertTrue(all(config.environment.delete for config in configs))
            self.assertEqual([config.agent.override_timeout_sec for config in configs], [19] * 9)
            for config in configs:
                keys = ("max_steps", "cmd_timeout_s", "total_tokens", "seed")
                self.assertEqual({key: config.agent.kwargs[key] for key in keys},
                                 {"max_steps": 7, "cmd_timeout_s": 2, "total_tokens": 24, "seed": 3})
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(set(summary["paired"]), {"single", "reference"})
            self.assertTrue(all(row["pass_at_1"] == 0 for row in summary["modes"].values()))

    async def test_invocations_isolate_docker_projects_and_record_run_ids(self):
        from harbor.environments.docker.docker import _sanitize_docker_compose_project_name
        from harbor.trial.single_step import SingleStepTrial
        from harbor.trial.trial import Trial
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as tmp:
            configs = []
            async def create(config):
                configs.append(config)
                result = SimpleNamespace(model_dump=lambda **kwargs: {"verifier_result": {"rewards": {"reward": 1}},
                    "exception_info": {"exception_type": "AgentTimeoutError"}})
                return SimpleNamespace(run=AsyncMock(return_value=result))
            outputs = [Path(tmp) / "one", Path(tmp) / "two"]
            with patch.object(bench, "check_tasks"), patch.object(bench, "check_harbor"), \
                    patch("harbor.models.task.task.Task"), patch.object(bench.subprocess, "run"), \
                    patch.dict(bench.os.environ, {"MYRIAD_BENCH_TOKEN": "test-token"}), \
                    patch.object(Trial, "create", side_effect=create), patch("builtins.print"):
                for output in outputs:
                    code = await asyncio.to_thread(bench.main, ["--single", "a", "--output", str(output)])
                    self.assertEqual(code, 0)
            manifests = [json.loads((output / "manifest.json").read_text()) for output in outputs]
            ids = [manifest["run_id"] for manifest in manifests]
            self.assertNotEqual(ids[0], ids[1])
            self.assertTrue(all(len(run_id) == 32 and set(run_id) <= set("0123456789abcdef") for run_id in ids))
            names, projects = [], []
            for output, run_id in zip(outputs, ids):
                rows = json.loads((output / "verdicts.json").read_text())
                self.assertTrue(all(row["passed"] and row["exception_type"] == "AgentTimeoutError" for row in rows))
                self.assertTrue(all(row["trial_name"].startswith(run_id + "__") for row in rows))
                names.extend(row["trial_name"] for row in rows)
            for config in configs:
                projects.append(_sanitize_docker_compose_project_name(config.trial_name + "__env"))
                trial = object.__new__(SingleStepTrial)
                trial.config = config
                verifier_session = trial._separate_verifier_session_id("trial")
                projects.append(_sanitize_docker_compose_project_name(verifier_session))
            self.assertEqual(len(set(names)), 6)
            self.assertEqual(len(set(projects)), 12)

    async def test_trial_cleanup_on_setup_failure_uses_official_lifecycle(self):
        from harbor.trial.trial import Trial
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "environment").mkdir()
            (task_dir / "instruction.md").write_text("repair")
            (task_dir / "task.toml").write_text('version = "1.0"\n')
            (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
            (task_dir / "tests" / "test.sh").write_text("exit 1\n")
            args = bench.parser().parse_args(["--single", "a"])
            config = bench.trial_config(args, task_dir, "single", ["a"], root / "trials", "1" * 32)
            env = SimpleNamespace(capabilities=SimpleNamespace(mounted=False), stop=AsyncMock())
            with patch("harbor.trial.trial.EnvironmentFactory.create_environment_from_config", return_value=env):
                trial = await Trial.create(config)
            with patch.object(trial, "_prepare", AsyncMock(side_effect=RuntimeError("setup failed"))), \
                    patch.object(trial, "_recover_outputs", AsyncMock()):
                result = await trial.run()
            env.stop.assert_awaited_once_with(delete=True)
            self.assertEqual(result.exception_info.exception_type, "RuntimeError")
            self.assertIsNone(result.verifier_result)
            self.assertTrue(trial.paths.result_path.exists())
            self.assertEqual(bench.verdict(result.model_dump(mode="json"))["status"], "error")

    async def test_agent_uses_authenticated_transport_and_shared_vote_allowance(self):
        import httpx
        from essaim.harbor_agent import MyriadAgent

        with tempfile.TemporaryDirectory() as tmp:
            requests = []
            def serve(request):
                requests.append(json.loads(request.content))
                self.assertEqual(request.headers["authorization"], "Bearer test-token")
                self.assertEqual(request.url.path, "/v1/chat/completions")
                return httpx.Response(200, json={"choices": [{"message": {"content": "```bash\nwrite artifact\n```"}}],
                                                  "usage": {"completion_tokens": 4}})
            client = httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="http://localhost/v1/",
                                      headers={"Authorization": "Bearer test-token"})
            instance = MyriadAgent(logs_dir=Path(tmp), families=["a", "b"], strategy="vote", max_steps=10,
                                   timeout_s=30, cmd_timeout_s=1, max_tokens=8, total_tokens=12, seed=7)
            with patch.dict(bench.os.environ, {"MYRIAD_BENCH_TOKEN": "test-token"}), \
                    patch("essaim.harbor_agent.httpx.AsyncClient", return_value=client):
                await instance.run("repair", FakeHarbor(), None)
            self.assertEqual([request["max_tokens"] for request in requests], [8, 4])
            self.assertEqual([request["seed"] for request in requests], [7, 7])
            log_text = (Path(tmp) / "episode.jsonl").read_text()
            log = [json.loads(line) for line in log_text.splitlines()]
            self.assertEqual(log[-1]["stopped"], "generation_limit")
            self.assertEqual(log[-1]["remaining_allowance"], 0)
            self.assertEqual(log[2]["note"]["chosen_votes"], 2)
            self.assertNotIn("test-token", log_text)

    async def test_agent_timeout_cancels_exec_and_flushes_log_before_teardown(self):
        import httpx
        from essaim.harbor_agent import MyriadAgent

        with tempfile.TemporaryDirectory() as tmp:
            client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
                "choices": [{"message": {"content": "```bash\nsleep 999\n```"}}]})),
                base_url="http://localhost/v1/")
            instance = MyriadAgent(logs_dir=Path(tmp), families=["a"], strategy="single", max_steps=10,
                                   timeout_s=30, cmd_timeout_s=1, max_tokens=8, total_tokens=12, seed=0)
            harbor = FakeHarbor()
            with patch.dict(bench.os.environ, {"MYRIAD_BENCH_TOKEN": "test-token"}), \
                    patch("essaim.harbor_agent.httpx.AsyncClient", return_value=client):
                worker = asyncio.create_task(instance.run("repair", harbor, None))
                await asyncio.wait_for(harbor.executing.wait(), 2)
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(worker, 2)
            await asyncio.wait_for(harbor.cancelled.wait(), 2)
            log = [json.loads(line) for line in (Path(tmp) / "episode.jsonl").read_text().splitlines()]
            self.assertEqual(log[-1]["stopped"], "timeout_or_cancelled")
            self.assertFalse(harbor.stopped)

    async def test_official_verifier_upload_execute_download_and_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "environment").mkdir()
            (task_dir / "instruction.md").write_text("produce an artifact")
            (task_dir / "task.toml").write_text('version = "1.0"\n')
            (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
            (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
            paths = TrialPaths(root / "trial")
            paths.verifier_dir.mkdir(parents=True)
            calls = []
            class VerifierEnv:
                os, capabilities = TaskOS.LINUX, SimpleNamespace(mounted=False)
                async def upload_dir(self, source_dir, target_dir):
                    calls.append(("upload", source_dir, target_dir))
                async def exec(self, command, **kwargs):
                    calls.append(("exec", command))
                    return ExecResult(return_code=0)
                async def download_dir(self, source_dir, target_dir):
                    calls.append(("download", source_dir, target_dir))
                    (target_dir / "reward.txt").write_text("1\n")
            verifier = Verifier(Task(task_dir), paths, VerifierEnv())
            result = await verifier.verify()
            self.assertEqual(result.rewards, {"reward": 1})
            self.assertEqual([call[0] for call in calls], ["upload", "exec", "exec", "download"])
            self.assertIn("/tests/test.sh", calls[2][1])
            self.assertTrue(bench.verdict({"verifier_result": result.model_dump()})["passed"])


if __name__ == "__main__":
    unittest.main()
