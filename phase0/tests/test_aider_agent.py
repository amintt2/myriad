"""E17 regressions with simulated chat, shell and official verifier; no inference or benchmark execution."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from essaim.agent import SUBMIT, run_episode as historical_episode
from essaim.aider_agent import Reply, TEST, complete_green, run_episode
from essaim.code_pilot import BudgetChat, parser, run_one
from essaim.local_linux import IsolationError


def block(command):
    return f"```bash\n{command}\n```"


def grade(passed=True, **extra):
    return {"status": "graded", "passed": passed, "tests": 1, "tests_expected": 1, "tests_complete": True,
            "failures": 0 if passed else 1, "errors": 0, "broken": [], "seconds": 0,
            "calls": [{"complete": True, "test": "official.test_value", "function": "encode"}],
            "feedback": [] if passed else [{"test": "official.test_value", "traceback": "wrong != expected"}], **extra}


class Env:
    def __init__(self, sources, code=0):
        self.sources, self.code, self.commands = sources, code, []
        self.isolation = {"simulated": True}

    def run(self, command, timeout_s):
        self.commands.append(command)
        if command.startswith("edit "):
            self.sources["solution.py"] = command[5:]
        return self.code, "shell exit 0 is not an official test"

    def close(self):
        pass


class Client:
    def __init__(self, commands, reason="stop"):
        self.commands, self.reason, self.calls = iter(commands), reason, []

    def post(self, path, json, timeout):
        self.calls.append(json)
        choice = {"message": {"content": block(next(self.commands))}, "finish_reason": self.reason}
        return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {
            "choices": [choice], "usage": {"completion_tokens": 2}})


class LoopTests(unittest.TestCase):
    def episode(self, commands, sources=None, verifier=None, code=0, **options):
        sources = sources if sources is not None else {"solution.py": "skeleton"}
        env = Env(sources, code)
        replies, observations, tested = iter(commands), [], []
        def chat(model, messages):
            observations.append(messages[-1]["content"])
            reply = next(replies)
            return reply if isinstance(reply, Reply) else block(reply)
        def test(snapshot, remaining):
            tested.append(dict(snapshot))
            self.assertGreater(remaining, 0)
            return verifier(snapshot) if verifier else grade(snapshot["solution.py"] == "correct")
        result = run_episode(chat, env, "task", ["small"], max_steps=len(commands), timeout_s=10,
                             snapshot=lambda: dict(sources), test_sources=test, **options)
        return result, env, observations, tested

    def test_immediate_submission_is_refused_and_can_recover(self):
        result, env, observations, tested = self.episode([
            f"echo {SUBMIT}", "edit correct", TEST, f"echo {SUBMIT}"])
        self.assertEqual(result["stopped"], "submitted")
        self.assertTrue(result["steps"][0].note["submission_refused"])
        self.assertIn(TEST, observations[1])
        self.assertEqual(env.commands, ["edit correct"])
        self.assertEqual(tested, [{"solution.py": "correct"}])

    def test_skeleton_exit_zero_cannot_authorize_submission(self):
        result, env, _, tested = self.episode(["python solution.py", f"echo {SUBMIT}"])
        self.assertEqual(result["stopped"], "max_steps")
        self.assertTrue(result["steps"][-1].note["submission_refused"])
        self.assertEqual(tested, [])
        self.assertEqual(env.commands, ["python solution.py"])

    def test_failure_correction_test_success_and_feedback(self):
        result, env, observations, tested = self.episode([TEST, "edit correct", TEST, f"echo {SUBMIT}"])
        self.assertEqual(result["stopped"], "submitted")
        self.assertIn("wrong != expected", observations[1])
        self.assertIn("GREEN", observations[3])
        self.assertEqual(tested, [{"solution.py": "skeleton"}, {"solution.py": "correct"}])
        self.assertEqual(env.commands, ["edit correct"])
        self.assertIn("solution.py", result["steps"][0].note["source_hashes"])

    def test_edit_after_green_requires_another_test_even_if_restored(self):
        result, _, _, tested = self.episode([TEST, "edit wrong", "edit correct", f"echo {SUBMIT}"],
                                           sources={"solution.py": "correct"})
        self.assertEqual(result["stopped"], "max_steps")
        self.assertTrue(result["steps"][-1].note["submission_refused"])
        self.assertEqual(len(tested), 1)
        result, _, _, _ = self.episode([TEST, "edit wrong", TEST, f"echo {SUBMIT}"],
                                       sources={"solution.py": "correct"}, verifier=lambda s: grade())
        self.assertEqual(result["stopped"], "submitted")
        self.assertEqual(result["tested_sources"], {"solution.py": "wrong"})

    def test_snapshot_mutation_during_test_does_not_turn_green(self):
        sources = {"solution.py": "correct"}
        def verifier(snapshot):
            self.assertEqual(snapshot, {"solution.py": "correct"})
            sources["solution.py"] = "changed"
            return grade()
        result, _, _, tested = self.episode([TEST, f"echo {SUBMIT}"], sources=sources, verifier=verifier)
        self.assertEqual(result["stopped"], "max_steps")
        self.assertEqual(tested, [{"solution.py": "correct"}])

    def test_zero_incomplete_or_missing_calls_are_never_green(self):
        for changes in ({"tests": 0, "tests_expected": 0}, {"tests_complete": False}, {"calls": []},
                        {"errors": 1}, {"broken": ["IsolationError"]}, {"status": "error"},
                        {"calls": [{"complete": False}]},
                        {"calls": [{"complete": True, "candidate_failure": "timeout"}]}):
            with self.subTest(changes=changes):
                self.assertFalse(complete_green(grade(**changes)))

    def test_infrastructure_and_timeout_are_not_green(self):
        with self.assertRaises(IsolationError):
            self.episode([TEST], verifier=lambda s: grade(status="error", passed=None))
        with self.assertRaises(IsolationError):
            self.episode([TEST], verifier=lambda s: (_ for _ in ()).throw(IsolationError("unavailable")))
        result, _, _, _ = self.episode(["edit correct", TEST, f"echo {SUBMIT}"], code=124)
        self.assertEqual(result["stopped"], "timeout")
        result, _, _, _ = self.episode(["edit correct", TEST, f"echo {SUBMIT}"], code=125)
        self.assertEqual(result["stopped"], "output_limit")
        result, _, _, _ = self.episode([TEST, f"echo {SUBMIT}"], verifier=lambda s: grade(
            passed=False, errors=1, calls=[{"complete": True, "candidate_failure": "timeout"}]))
        self.assertEqual(result["stopped"], "max_steps")

    def test_late_green_test_is_timeout(self):
        clock = [0.0]
        def test(sources, remaining):
            clock[0] = 11.0
            return grade()
        with patch("essaim.aider_agent.time.monotonic", side_effect=lambda: clock[0]):
            result = run_episode(lambda *a: block(TEST), Env({}), "task", ["small"], timeout_s=10,
                                 snapshot=lambda: {"solution.py": "correct"}, test_sources=test)
        self.assertEqual(result["stopped"], "timeout")

    def test_truncated_complete_block_is_not_executed(self):
        for reason in ("length", None, "content_filter"):
            result, env, observations, _ = self.episode([
                Reply(block("edit wrong"), reason), "edit correct", TEST, f"echo {SUBMIT}"])
            self.assertTrue(result["steps"][0].note["format_error"])
            self.assertIn("truncated", observations[1])
            self.assertEqual(env.commands, ["edit correct"])
            self.assertEqual(result["stopped"], "submitted")

    def test_truncated_peer_triggers_cascade_and_reference_is_checked(self):
        asks = []
        def chat(model, messages):
            asks.append(model)
            return Reply(block("edit wrong"), "length" if model in ("b", "ref") else "stop")
        env = Env({})
        result = run_episode(chat, env, "task", ["a", "b"], strategy="cascade", reference="ref", max_steps=1,
                             snapshot=lambda: {}, test_sources=lambda *a: grade())
        self.assertEqual(asks, ["a", "b", "ref"])
        self.assertEqual(env.commands, [])
        self.assertEqual(result["stopped"], "max_steps")

    def test_shared_budget_reserves_configurable_ceiling_and_last_remainder(self):
        client = Client(["edit correct", TEST])
        rows = []
        chat = BudgetChat(client, 4096, 5000, 7, time.monotonic() + 10, rows.append, strict=True)
        sources = {"solution.py": "skeleton"}
        result = run_episode(chat, Env(sources), "task", ["small"], snapshot=lambda: dict(sources),
                             test_sources=lambda *a: grade(), timeout_s=10)
        self.assertEqual(result["stopped"], "generation_limit")
        self.assertEqual([c["max_tokens"] for c in client.calls], [4096, 904])
        self.assertEqual(chat.accounting()["charged_request_ceilings"], 5000)
        self.assertEqual(chat.accounting()["completion_tokens"], 4)

    def test_budgeted_truncated_reply_records_usage_but_cannot_submit(self):
        client = Client([f"echo {SUBMIT}"], reason="length")
        chat = BudgetChat(client, 4096, 4096, 0, time.monotonic() + 10, lambda r: None, strict=True)
        result = run_episode(chat, Env({}), "task", ["small"], snapshot=lambda: {}, test_sources=lambda *a: grade())
        self.assertEqual(result["stopped"], "generation_limit")
        self.assertEqual(chat.requests[0]["finish_reason"], "length")
        self.assertEqual(chat.accounting()["completion_tokens"], 2)

    def test_historical_loop_and_backend_defaults_are_preserved(self):
        result = historical_episode(lambda *a: block(f"echo {SUBMIT}"), Env({}), "task", ["small"])
        self.assertEqual(result["stopped"], "submitted")
        self.assertEqual(parser().parse_args([]).max_tokens, 4096)
        self.assertEqual(parser().parse_args([]).total_tokens, 49152)
        self.assertEqual(parser(backend=object()).parse_args([]).max_tokens, 1024)
        self.assertEqual(parser().parse_args(["--max-tokens", "8192"]).max_tokens, 8192)


class ControllerTests(unittest.TestCase):
    def run_candidate(self, commands, verifier, reason="stop", **limits):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(tasks_dir=root, runtime=root, max_tokens=4096, total_tokens=49152, seed=0,
                                   max_steps=len(commands), timeout=10, cmd_timeout=1, verify_timeout=5,
                                   reference=None, **limits)
            sources = {"solution.py": "skeleton"}
            tested = []
            def verify(cache, task, snapshot, runtime, timeout):
                tested.append(dict(snapshot))
                self.assertLessEqual(timeout, args.verify_timeout)
                return verifier(snapshot)
            client = Client(commands, reason)
            with patch("essaim.code_pilot.copy_task", return_value=["solution.py"]), \
                 patch("essaim.code_pilot.collect_sources", side_effect=lambda *a: dict(sources)), \
                 patch("essaim.code_pilot.instructions", return_value="simulated task"), \
                 patch("essaim.code_pilot.LocalLinuxEnv", return_value=Env(sources)), \
                 patch("essaim.code_pilot.verify", side_effect=verify):
                row = run_one(args, "affine-cipher", "single", ["small"], client, root, "campaign")
            log = [json.loads(line) for line in (root / "single__affine-cipher.jsonl").read_text().splitlines()]
            return row, log, tested

    def test_controller_requires_submission_even_if_final_verifier_passes(self):
        for commands in ([f"echo {SUBMIT}"], ["edit correct"], ["edit correct", TEST]):
            row, _, _ = self.run_candidate(commands, lambda s: grade())
            self.assertEqual(row["status"], "graded")
            self.assertIs(row["passed"], False)
            self.assertIs(row["verifier"]["passed"], True)

    def test_controller_logs_snapshot_feedback_and_rechecks_final_sources(self):
        row, log, tested = self.run_candidate([TEST, "edit correct", TEST, f"echo {SUBMIT}"],
                                              lambda s: grade(s["solution.py"] == "correct"))
        self.assertTrue(row["passed"], row)
        self.assertTrue(row["submission_accepted"])
        self.assertEqual(tested, [{"solution.py": "skeleton"}, {"solution.py": "correct"},
                                  {"solution.py": "correct"}])
        steps = [r for r in log if r["kind"] == "step"]
        self.assertIn("wrong != expected", steps[0]["output"])
        self.assertEqual(steps[2]["note"]["source_hashes"], row["source_hashes"])

    def test_controller_verifier_error_is_infrastructure(self):
        row, log, tested = self.run_candidate([TEST], lambda s: grade(status="error", passed=None))
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["passed"])
        self.assertEqual(len(tested), 1)
        self.assertIn("official_test", [r for r in log if r["kind"] == "step"][0]["note"])

    def test_controller_refuses_truncated_submission(self):
        row, log, _ = self.run_candidate([f"echo {SUBMIT}"], lambda s: grade(), reason="length")
        self.assertIs(row["passed"], False)
        self.assertEqual(row["stopped"], "max_steps")
        self.assertTrue([r for r in log if r["kind"] == "step"][0]["note"]["format_error"])

    def test_controller_rejects_failed_or_incomplete_final_recheck(self):
        for final in (grade(False), grade(tests_complete=False), grade(seconds=6)):
            verdicts = iter([grade(), final])
            row, _, tested = self.run_candidate([TEST, f"echo {SUBMIT}"], lambda s: next(verdicts))
            self.assertEqual(row["stopped"], "submitted")
            self.assertIs(row["passed"], False)
            self.assertEqual(len(tested), 2)

    def test_controller_rejects_green_after_verification_budget(self):
        clock = [0.0]
        def verifier(snapshot):
            clock[0] = 6.0
            return grade()
        with patch("essaim.code_pilot.time.monotonic", side_effect=lambda: clock[0]):
            row, log, _ = self.run_candidate([TEST, f"echo {SUBMIT}"], verifier)
        self.assertEqual(row["stopped"], "max_steps")
        self.assertIs(row["passed"], False)
        self.assertIn("exceeded its time budget", [r for r in log if r["kind"] == "step"][0]["output"])

    def test_other_backend_still_uses_historical_submission_and_plain_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(tasks_dir=root, runtime=root, max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=1, timeout=10, cmd_timeout=1, verify_timeout=5, reference=None)
            backend = SimpleNamespace(copy_task=lambda *a: ["solution.py"],
                                      collect_sources=lambda *a: {"solution.py": "base"},
                                      instructions=lambda *a: "historical task",
                                      verify=lambda *a: {"status": "graded", "passed": True, "seconds": 0})
            with patch("essaim.code_pilot.LocalLinuxEnv", return_value=Env({})):
                row = run_one(args, "historical", "single", ["small"], Client([f"echo {SUBMIT}"], reason=None),
                              root, "campaign", backend)
            self.assertEqual(row["stopped"], "submitted")
            self.assertIs(row["passed"], True)


if __name__ == "__main__":
    unittest.main()
