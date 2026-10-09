"""Offline protocol and harness tests; real Linux isolation is enabled explicitly by the parent."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from essaim.agent import GenerationDeadline, GenerationLimit, SUBMIT, run_episode
from essaim.aider_python import collect_sources, copy_task, provenance, verify
from essaim.code_pilot import BudgetChat, campaign_lock, hash_json, model_modes, report, run_one
from essaim.code_provenance import declarations
from essaim.local_linux import IsolationError, LocalLinuxEnv, landlock, seccomp
from essaim.terminal_bench import check_tasks, validate_endpoint


class FakeEnv:
    def __init__(self):
        self.commands = []
        self.isolation = {"test_fixture": True}

    def run(self, command, timeout_s):
        self.commands.append(command)
        return 0, "visible test output"

    def close(self):
        pass


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = iter(answers), []

    def post(self, path, json, timeout):
        self.calls.append(json)
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: answer, headers={"server": "fake"})


def answer(text, usage=7, **extra):
    data = {"model": "fake-model", "choices": [{"message": {"content": text}}], **extra}
    if usage is not None:
        data["usage"] = {"completion_tokens": usage}
    return data


class ProtocolTests(unittest.TestCase):
    def test_cascade_uses_reference_on_disagreement(self):
        calls = []
        def chat(model, messages):
            calls.append(model)
            return f"```bash\n{f'echo {SUBMIT}' if model == 'ref' else f'echo {model}'}\n```"
        result = run_episode(chat, FakeEnv(), "task", ["a", "b"], "cascade", reference="ref")
        self.assertEqual(calls, ["a", "b", "ref"])
        self.assertEqual(result["stopped"], "submitted")
        self.assertTrue(result["steps"][0].note["cascade"])
        self.assertFalse(result["steps"][0].note["peer_vote"]["unanimous"])

    def test_cascade_uses_reference_on_format_error(self):
        calls = []
        def chat(model, messages):
            calls.append(model)
            return "bad format" if model == "b" else f"```bash\necho {SUBMIT}\n```"
        result = run_episode(chat, FakeEnv(), "task", ["a", "b"], "cascade", reference="ref")
        self.assertEqual(calls, ["a", "b", "ref"])
        self.assertEqual(result["stopped"], "submitted")

    def test_cascade_consensus_does_not_call_reference(self):
        calls = []
        def chat(model, messages):
            calls.append(model)
            return f"```bash\necho {SUBMIT}\n```"
        run_episode(chat, FakeEnv(), "task", ["a", "b"], "cascade", reference="ref")
        self.assertEqual(calls, ["a", "b"])

    def test_bad_reference_never_executes_peer_command(self):
        env = FakeEnv()
        def chat(model, messages):
            return "bad" if model == "ref" else f"```bash\necho {model}\n```"
        result = run_episode(chat, env, "task", ["a", "b"], "cascade", max_steps=1, reference="ref")
        self.assertEqual(env.commands, [])
        self.assertTrue(result["steps"][0].note["format_error"])

    def test_reference_budget_is_shared(self):
        client = FakeClient([answer("```bash\necho a\n```"), answer("```bash\necho b\n```")])
        chat = BudgetChat(client, 10, 20, 3, time.monotonic() + 60, lambda row: None)
        result = run_episode(chat, FakeEnv(), "task", ["a", "b"], "cascade", reference="ref")
        self.assertEqual(result["stopped"], "generation_limit")
        self.assertEqual(chat.accounting()["completion_tokens"], 14)
        self.assertEqual(chat.accounting()["charged_request_ceilings"], 20)

    def test_usage_unknown_is_not_a_ceiling(self):
        client = FakeClient([answer("ok", None), answer("ok", 3)])
        rows = []
        chat = BudgetChat(client, 10, 15, 4, time.monotonic() + 60, rows.append)
        chat("a", [])
        chat("b", [])
        accounting = chat.accounting()
        self.assertIsNone(accounting["completion_tokens"])
        self.assertEqual(accounting["known_completion_tokens"], 3)
        self.assertEqual(accounting["charged_request_ceilings"], 15)
        self.assertEqual([r["ceiling"] for r in rows], [10, 5])
        self.assertEqual(client.calls[0]["seed"], 4)
        self.assertEqual(client.calls[0]["myriad"]["k"], 1)

    def test_myriad_aggregate_not_selected_usage(self):
        client = FakeClient([answer("ok", 3, myriad={"total_completion_tokens": 9,
                                                    "peers_asked": 2, "peers_answered": 2})])
        chat = BudgetChat(client, 20, 20, 0, time.monotonic() + 60, lambda row: None)
        chat("a", [])
        self.assertEqual(chat.accounting()["completion_tokens"], 9)

    def test_replacement_usage_is_incomplete(self):
        client = FakeClient([answer("ok", 3, myriad={"total_completion_tokens": 9, "replacements": 1})])
        chat = BudgetChat(client, 20, 20, 0, time.monotonic() + 60, lambda row: None)
        chat("a", [])
        self.assertIsNone(chat.accounting()["completion_tokens"])
        self.assertEqual(chat.accounting()["known_completion_tokens"], 9)

    def test_failed_request_is_charged_and_logged(self):
        rows = []
        chat = BudgetChat(FakeClient([TimeoutError()]), 10, 10, 0, time.monotonic() + 60, rows.append)
        with self.assertRaises(TimeoutError):
            chat("a", [])
        with self.assertRaises(GenerationLimit):
            chat("a", [])
        self.assertEqual(rows[0]["exception_type"], "TimeoutError")
        self.assertEqual(chat.accounting()["charged_request_ceilings"], 10)

    def test_deadline_during_generation_is_an_episode_stop(self):
        rows = []
        chat = BudgetChat(FakeClient([TimeoutError()]), 10, 10, 0, 2.0, rows.append)
        with patch("essaim.code_pilot.time.monotonic", side_effect=[1.0, 1.0, 2.0, 2.0]):
            with self.assertRaises(GenerationDeadline):
                chat("a", [])
        self.assertEqual(rows[0]["stop_reason"], "episode_deadline")
        self.assertIsNone(chat.accounting()["completion_tokens"])

    def test_finish_reason_and_reasoning_are_preserved(self):
        data = answer("```bash\necho done\n```", 12)
        data["choices"][0]["finish_reason"] = "length"
        data["choices"][0]["message"]["reasoning_content"] = "returned thought"
        rows = []
        chat = BudgetChat(FakeClient([data]), 20, 20, 0, time.monotonic() + 60, rows.append)
        chat("a", [])
        self.assertEqual(rows[0]["finish_reason"], "length")
        self.assertEqual(rows[0]["message"]["reasoning_content"], "returned thought")
        self.assertEqual(chat.accounting()["completion_tokens"], 12)

    def test_reference_mode_is_part_of_the_predeclared_grid(self):
        args = SimpleNamespace(single="a", vote=["a", "b"], reference="ref")
        self.assertEqual(model_modes(args), {"single": ["a"], "vote": ["a", "b"],
                                            "cascade": ["a", "b"], "reference": ["ref"]})
        args.vote = None
        self.assertEqual(model_modes(args), {"single": ["a"], "reference": ["ref"]})

    def test_reported_usage_over_ceiling_stops(self):
        chat = BudgetChat(FakeClient([answer("ok", 11)]), 10, 10, 0, time.monotonic() + 60, lambda row: None)
        with self.assertRaises(ValueError):
            chat("a", [])
        self.assertEqual(chat.accounting()["charged_request_ceilings"], 10)

    def test_request_wall_timeout_closes_client(self):
        import asyncio
        from essaim.code_pilot import DeadlineClient
        closed = []
        class Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                closed.append(True)
            async def post(self, *args, **kwargs):
                await asyncio.sleep(10)
        with patch("httpx.AsyncClient", Client), self.assertRaises(TimeoutError):
            DeadlineClient("http://127.0.0.1:8400/v1", "test").post("chat/completions", {}, 0.01)
        self.assertEqual(closed, [True])

    def test_nonloopback_http_rejected(self):
        with self.assertRaises(ValueError):
            validate_endpoint("http://192.0.2.1/v1", "test")
        with self.assertRaises(ValueError):
            validate_endpoint("https://example.com/v1", "")
        validate_endpoint("http://127.0.0.1:8400/v1", "test")


def fixture(cache: Path):
    base = cache / "python/exercises/practice/affine-cipher"
    (base / ".meta").mkdir(parents=True)
    (base / ".docs").mkdir()
    config = {"files": {"solution": ["affine_cipher.py"], "test": ["affine_cipher_test.py"],
                        "example": [".meta/example.py"]}}
    (base / ".meta/config.json").write_text(json.dumps(config))
    (base / ".meta/example.py").write_text("SECRET REFERENCE")
    (base / ".docs/instructions.md").write_text("Implement encode and decode.")
    (base / "affine_cipher.py").write_text("def encode(*args): pass")
    (base / "affine_cipher_test.py").write_text(
        "import unittest\nfrom affine_cipher import encode\nclass Official(unittest.TestCase):\n"
        "    def test_value(self): self.assertEqual(encode('input', 5, 7), 'expected')\n")
    (cache / "README.md").write_text("Exercism attribution")
    (cache / "LICENSE.python").write_text("MIT")
    return base


class HarnessTests(unittest.TestCase):
    def test_campaign_lock_refuses_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with campaign_lock(root):
                with self.assertRaises(FileExistsError):
                    with campaign_lock(root):
                        self.fail("concurrent campaign entered")
            self.assertFalse((root / ".campaign.lock").exists())

    def test_kernel_protection_unavailable_is_error(self):
        with patch("essaim.local_linux.ctypes.CDLL") as library:
            library.return_value.syscall.return_value = -1
            with self.assertRaises(IsolationError):
                landlock(Path("/tmp"), Path("/tmp"))
        with patch("essaim.local_linux.platform.machine", return_value="x86_64"):
            with patch("essaim.local_linux.ctypes.CDLL") as library:
                library.return_value.prctl.return_value = -1
                with self.assertRaises(IsolationError):
                    seccomp()

    def test_pinned_selection(self):
        manifest = provenance()
        self.assertEqual(len(manifest["tasks"]), 34)
        self.assertEqual(manifest["pilot"], ["affine-cipher", "beer-song", "book-store"])
        self.assertEqual(len(manifest["files"]), 282)

    def test_copy_excludes_reference_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            work = root / "work"
            work.mkdir()
            allowed = copy_task(root / "cache", "affine-cipher", work)
            self.assertEqual(allowed, ["affine_cipher.py"])
            self.assertFalse((work / ".meta").exists())
            self.assertNotIn("SECRET REFERENCE", "".join(p.read_text() for p in work.rglob("*") if p.is_file()))
            self.assertTrue((work / "LICENSE").exists())

    def test_tampered_visible_test_not_submitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            work = root / "work"
            work.mkdir()
            allowed = copy_task(root / "cache", "affine-cipher", work)
            (work / "affine_cipher_test.py").write_text("forged tests")
            sources = collect_sources(work, allowed)
            self.assertEqual(set(sources), {"affine_cipher.py"})
            self.assertNotIn("forged tests", str(sources))

    def test_episode_changes_and_original_verification_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            fixture(cache)
            args = SimpleNamespace(tasks_dir=cache, runtime=root, max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=3, timeout=10, cmd_timeout=1, verify_timeout=5, reference=None)
            class AgentEnv(FakeEnv):
                def __init__(self, work, runtime):
                    super().__init__()
                    self.work = work
                def run(self, command, timeout_s):
                    (self.work / "affine_cipher.py").write_text("edited source")
                    (self.work / "affine_cipher_test.py").write_text("forged tests")
                    return 0, "visible output"
            def grader(cache_arg, task, sources, runtime, timeout):
                self.assertEqual(sources, {"affine_cipher.py": "edited source"})
                self.assertIn("assertEqual", (cache_arg / "python/exercises/practice" / task /
                                             "affine_cipher_test.py").read_text())
                return {"status": "graded", "passed": False, "seconds": 0}
            client = FakeClient([answer("```bash\nedit\n```", 2), answer(f"```bash\necho {SUBMIT}\n```", 3)])
            with patch("essaim.code_pilot.LocalLinuxEnv", AgentEnv), patch("essaim.code_pilot.verify", grader):
                row = run_one(args, "affine-cipher", "single", ["a"], client, root, "campaign")
            self.assertEqual(row["status"], "graded")
            self.assertIs(row["passed"], False)
            self.assertEqual(row["completion_tokens"], 5)
            self.assertEqual(row["charged_request_ceilings"], 20)
            log = [json.loads(line) for line in (root / "single__affine-cipher.jsonl").read_text().splitlines()]
            self.assertIn("changes", [r["kind"] for r in log])
            self.assertIn("isolation", [r["kind"] for r in log])

    def test_episode_verifier_error_is_not_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            args = SimpleNamespace(tasks_dir=root / "cache", runtime=root, max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=3, timeout=10, cmd_timeout=1, verify_timeout=5, reference=None)
            client = FakeClient([answer(f"```bash\necho {SUBMIT}\n```", 3)])
            with patch("essaim.code_pilot.LocalLinuxEnv", return_value=FakeEnv()):
                with patch("essaim.code_pilot.verify", side_effect=IsolationError("broken harness")):
                    row = run_one(args, "affine-cipher", "single", ["a"], client, root, "campaign")
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["passed"])

    def test_generation_deadline_verifies_existing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            args = SimpleNamespace(tasks_dir=root / "cache", runtime=root, max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=3, timeout=10, cmd_timeout=1, verify_timeout=5, reference=None)
            grade = {"status": "graded", "passed": False, "seconds": 0}
            with patch("essaim.code_pilot.LocalLinuxEnv", return_value=FakeEnv()):
                with patch("essaim.code_pilot.BudgetChat.__call__", side_effect=GenerationDeadline()):
                    with patch("essaim.code_pilot.verify", return_value=grade) as verifier:
                        row = run_one(args, "affine-cipher", "single", ["a"], FakeClient([]), root, "campaign")
            self.assertEqual(row["stopped"], "timeout")
            self.assertIs(row["passed"], False)
            self.assertEqual(verifier.call_count, 1)

    def test_reference_episode_uses_single_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            args = SimpleNamespace(tasks_dir=root / "cache", runtime=root, max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=3, timeout=10, cmd_timeout=1, verify_timeout=5, reference="ref")
            client = FakeClient([answer(f"```bash\necho {SUBMIT}\n```", 3)])
            with patch("essaim.code_pilot.LocalLinuxEnv", return_value=FakeEnv()):
                grade = {"status": "graded", "passed": False, "seconds": 0}
                with patch("essaim.code_pilot.verify", return_value=grade):
                    row = run_one(args, "affine-cipher", "reference", ["ref"], client, root, "campaign")
            self.assertEqual(row["status"], "graded")
            self.assertEqual([r["model"] for r in client.calls], ["ref"])

    def test_endpoint_failure_before_deadline_is_not_a_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            args = SimpleNamespace(tasks_dir=root / "cache", runtime=root, max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=3, timeout=10, cmd_timeout=1, verify_timeout=5, reference=None)
            with patch("essaim.code_pilot.LocalLinuxEnv", return_value=FakeEnv()):
                with patch("essaim.code_pilot.verify") as verifier:
                    row = run_one(args, "affine-cipher", "single", ["a"], FakeClient([TimeoutError()]), root, "campaign")
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["passed"])
            self.assertIsNone(row["completion_tokens"])
            self.assertEqual(verifier.call_count, 0)

    def test_verifier_uses_original_tests_outside_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            seen = []
            class Env:
                def __init__(self, work, runtime):
                    seen.append([p.name for p in work.iterdir()])
                    self.work = work
                def run(self, command, timeout_s):
                    self_outer.assertNotIn("expected", command)
                    (self.work / "_return.json").write_text('{"value":"expected"}')
                    return 0, "candidate diagnostics"
                def close(self):
                    pass
            self_outer = self
            with patch("essaim.aider_python.LocalLinuxEnv", Env):
                grade = verify(root / "cache", "affine-cipher", {"affine_cipher.py": "candidate"}, root)
            self.assertTrue(grade["passed"])
            self.assertNotIn("affine_cipher_test.py", seen[0])

    def test_invalid_candidate_output_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            with patch("essaim.aider_python.LocalLinuxEnv") as env:
                def create(work, runtime):
                    (work / "_return.json").write_text("invalid JSON")
                    return env.return_value
                env.side_effect = create
                env.return_value.run.return_value = (0, "")
                grade = verify(root / "cache", "affine-cipher", {"affine_cipher.py": "candidate"}, root)
            self.assertEqual(grade["status"], "graded")
            self.assertIs(grade["passed"], False)
            self.assertEqual(grade["calls"][0]["candidate_failure"], "invalid_json")

    def test_verifier_setup_error_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            with patch("essaim.aider_python.LocalLinuxEnv", side_effect=IsolationError("no kernel protection")):
                grade = verify(root / "cache", "affine-cipher", {"affine_cipher.py": "candidate"}, root)
            self.assertEqual(grade["status"], "error")
            self.assertIsNone(grade["passed"])

    def test_official_harness_error_without_candidate_call_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = fixture(root / "cache")
            test = base / "affine_cipher_test.py"
            test.write_text(test.read_text().replace("self.assertEqual(encode('input', 5, 7), 'expected')",
                                                    "raise RuntimeError('broken official harness')"))
            grade = verify(root / "cache", "affine-cipher", {"affine_cipher.py": "candidate"}, root)
            self.assertEqual(grade["status"], "error")
            self.assertIsNone(grade["passed"])

    def test_completed_wrong_value_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "cache")
            with patch("essaim.aider_python.LocalLinuxEnv") as env:
                def create(work, runtime):
                    (work / "_return.json").write_text('{"value":"wrong"}')
                    return env.return_value
                env.side_effect = create
                env.return_value.run.return_value = (0, "diagnostic output")
                grade = verify(root / "cache", "affine-cipher", {"affine_cipher.py": "candidate"}, root)
            self.assertEqual(grade["status"], "graded")
            self.assertIs(grade["passed"], False)

    def test_invalid_initial_source_is_infrastructure_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = fixture(root / "cache")
            (base / "affine_cipher.py").write_bytes(b"\xff")
            args = SimpleNamespace(tasks_dir=root / "cache", runtime=root, max_tokens=10, total_tokens=30,
                                   seed=0, timeout=10)
            client = FakeClient([])
            row = run_one(args, "affine-cipher", "single", ["a"], client, root, "campaign")
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["passed"])
            self.assertEqual(row["exception_type"], "SourceArtifactError")
            self.assertEqual(client.calls, [])

    def test_cache_hash_and_extra_file_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test").write_bytes(b"official")
            manifest = {"files": {"test": {"sha256": hashlib.sha256(b"official").hexdigest(), "size": 8}}}
            check_tasks(root, manifest)
            (root / "extra").write_bytes(b"tamper")
            with self.assertRaises(ValueError):
                check_tasks(root, manifest)
            (root / "extra").unlink()
            (root / "test").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                check_tasks(root, manifest)

    def test_report_refuses_missing_or_invalid_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = {"models": {"single": ["a"]}}
            with self.assertRaises(ValueError):
                report([], campaign, Path(tmp))
            rows = [{"mode": "single", "task": task, "status": "error", "passed": None}
                    for task in provenance()["pilot"]]
            with self.assertRaises(ValueError):
                report(rows, campaign, Path(tmp))
            self.assertFalse((Path(tmp) / "report.md").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "plot integration needs the pinned pilot runtime")
    def test_report_complete_outputs_and_unknown_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            campaign = {"models": {"single": ["a"]}, "watts": 100, "eur_kwh": 0.25}
            rows = []
            for i, task in enumerate(provenance()["pilot"]):
                transcript = output / f"single__{task}.jsonl"
                transcript.write_bytes(b"fixture\n")
                rows.append({"mode": "single", "task": task, "status": "graded", "passed": i == 0,
                             "campaign_hash": hash_json(campaign), "seconds": 36, "completion_tokens": None,
                             "charged_request_ceilings": 100,
                             "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()})
            summary = report(rows, campaign, output)
            metric = summary["modes"]["single"]
            self.assertAlmostEqual(metric["pass_at_1"], 1 / 3)
            self.assertAlmostEqual(metric["estimated_wh_per_task"], 1)
            self.assertAlmostEqual(metric["estimated_eur_per_task"], 0.00025)
            self.assertIsNone(metric["completion_tokens"])
            self.assertEqual(len(list(output.glob("*.svg"))), 2)
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            self.assertEqual(len(list(output.glob("*.pdf"))), 2)
            self.assertIn("inconnu", (output / "report.md").read_text(encoding="utf-8"))
            rows[0]["campaign_hash"] = "wrong"
            with self.assertRaises(ValueError):
                report(rows, campaign, output)

    def test_report_requires_reference_results_when_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            campaign = {"models": {"single": ["a"], "vote": ["a", "b"], "cascade": ["a", "b"],
                                   "reference": ["ref"]}, "watts": 100, "eur_kwh": 0.25}
            rows = []
            for mode in campaign["models"]:
                for task in provenance()["pilot"]:
                    transcript = output / f"{mode}__{task}.jsonl"
                    transcript.write_bytes(b"fixture\n")
                    rows.append({"mode": mode, "task": task, "status": "graded", "passed": False,
                                 "campaign_hash": hash_json(campaign), "seconds": 1, "completion_tokens": 2,
                                 "charged_request_ceilings": 10,
                                 "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()})
            with self.assertRaises(ValueError):
                report(rows[:9], campaign, output)
            with patch("essaim.code_pilot.figures"):
                summary = report(rows, campaign, output)
            self.assertEqual(summary["modes"]["reference"]["passed"], 0)
            self.assertEqual(summary["modes"]["reference"]["completion_tokens"], 6)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "plot integration needs the pinned pilot runtime")
    def test_report_figures_for_zero_and_all_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            campaign = {"models": {"single": ["a"], "reference": ["ref"]}, "watts": 100, "eur_kwh": 0.25}
            rows = []
            for mode in campaign["models"]:
                for task in provenance()["pilot"]:
                    transcript = output / f"{mode}__{task}.jsonl"
                    transcript.write_bytes(b"fixture\n")
                    rows.append({"mode": mode, "task": task, "status": "graded", "passed": mode == "reference",
                                 "campaign_hash": hash_json(campaign), "seconds": 1, "completion_tokens": 2,
                                 "charged_request_ceilings": 10,
                                 "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()})
            summary = report(rows, campaign, output)
            self.assertEqual(summary["modes"]["single"]["passed"], 0)
            self.assertEqual(summary["modes"]["reference"]["passed"], 3)
            for metric in ("seconds_per_task", "estimated_eur_per_task"):
                for ext in ("svg", "png", "pdf"):
                    path = output / f"accuracy_{metric}.{ext}"
                    self.assertGreater(path.stat().st_size, 1000)


def model_declaration():
    return {"declared_at": "2026-10-10T00:00:00Z", "endpoint_kind": "direct loopback test fixture",
            "model": "fixture/model", "alias": "a", "licence": "MIT", "repository": "fixture/model",
            "revision": "a" * 40, "file": "model.gguf", "quantization": "Q8_0", "bytes": 1, "sha256": "b" * 64,
            "server": {"version": "fixture", "commit": "c" * 9, "artifact": "server.tar.gz", "sha256": "d" * 64,
                       "context": 32768, "parallel": 1, "gpu_layers": 0, "reasoning_budget": 0, "jinja": True},
            "hardware": {"cpu": "fixture CPU", "host_memory_gib": 32, "execution": "Linux x86_64",
                         "gpu": "none", "cpu_threads": "default"},
            "energy": {"watts": 100, "eur_kwh": 0.25, "measured": False, "scope": "test fixture assumption"}}


class ProvenanceTests(unittest.TestCase):
    def test_bom_and_complete_structured_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(json.dumps(model_declaration()), encoding="utf-8-sig")
            loaded = declarations(path, ["a"], 100, 0.25)
            self.assertEqual(loaded["a"]["server"]["context"], 32768)
            self.assertEqual(loaded["a"]["server"]["reasoning_budget"], 0)

    def test_each_reference_and_peer_requires_a_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(json.dumps(model_declaration()))
            with self.assertRaises(ValueError):
                declarations(path, ["a", "ref"], 100, 0.25)
            path.write_text(json.dumps({"models": {"a": model_declaration(), "ref": model_declaration()}}))
            self.assertEqual(set(declarations(path, ["a", "ref"], 100, 0.25)), {"a", "ref"})

    def test_credentials_and_energy_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            data = model_declaration()
            data["server"]["api_key"] = "fixture"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                declarations(path, ["a"], 100, 0.25)
            path.write_text(json.dumps(model_declaration()))
            with self.assertRaises(ValueError):
                declarations(path, ["a"], 200, 0.25)


@unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("CODE_PILOT_LINUX_TESTS") == "1",
                     "explicit real Linux isolation tests")
class LinuxIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="myriad-isolation-test-")
        self.root = Path(self.temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.runtime = Path(os.environ.get("CODE_PILOT_RUNTIME", str(Path.home() / ".local/share/myriad-code-pilot/venv")))
        self.env = LocalLinuxEnv(self.work, self.runtime)

    def tearDown(self):
        self.env.close()
        self.temporary.cleanup()

    def test_edit_and_pytest(self):
        code, output = self.env.run("printf 'def add(a,b): return a+b\\n' > calc.py\n"
                                   "printf 'from calc import add\\ndef test_add(): assert add(2,3)==5\\n' > calc_test.py\n"
                                   "python -m pytest -q calc_test.py", 10)
        self.assertEqual(code, 0, output)
        self.assertIn("1 passed", output)

    def test_read_leaks_denied(self):
        forbidden = self.root / "secret"
        forbidden.write_text("PRIVATE")
        repo = Path(__file__).resolve().parents[2] / "CLAUDE.md"
        for path in (forbidden, repo, Path.home() / ".ssh", Path.home() / ".cache"):
            code, output = self.env.run("cat " + shlex.quote(str(path)), 5)
            self.assertNotEqual(code, 0)
            self.assertNotIn("PRIVATE", output)

    def test_external_write_and_metadata_denied(self):
        outside = self.root / "outside"
        outside.write_text("original")
        for command in (f"echo leaked > {outside}", f"chmod 000 {outside}", f"touch {outside}"):
            code, _ = self.env.run(command, 5)
            self.assertNotEqual(code, 0)
        self.assertEqual(outside.read_text(), "original")

    def test_raw_timestamp_syscalls_cannot_mutate_external_sentinel(self):
        outside = self.root / "outside"
        outside.write_text("original")
        before = outside.stat()
        # Include legacy futimesat (audit P1), plus the other pathname timestamp variants.
        script = ("import ctypes,json;libc=ctypes.CDLL(None,use_errno=True);"
                  f"path={str(outside)!r}.encode();times=(ctypes.c_long*4)(100,0,100,0);results=[]\n"
                  "for number,args in [(132,(path,times)),(235,(path,times)),"
                  "(261,(ctypes.c_long(-100),path,times)),(280,(ctypes.c_long(-100),path,times,ctypes.c_long(0)))]:\n"
                  " ret=libc.syscall(ctypes.c_long(number),*args);results.append([number,ret,ctypes.get_errno()])\n"
                  "print(json.dumps(results))")
        code, output = self.env.run("python -c " + shlex.quote(script), 5)
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output), [[number, -1, 1] for number in (132, 235, 261, 280)])
        after = outside.stat()
        self.assertEqual((after.st_atime_ns, after.st_mtime_ns), (before.st_atime_ns, before.st_mtime_ns))

    def test_readable_runtime_metadata_cannot_be_mutated(self):
        # Temporary sentinel only; no runtime library is modified. All three holes reproduced before the fix.
        with tempfile.TemporaryDirectory(prefix="myriad-metadata-", dir=self.runtime) as temporary:
            outside = Path(temporary) / "sentinel"
            outside.write_text("original")
            os.setxattr(outside, "user.myriad_sentinel", b"original")
            script = ("import ctypes,os,json,struct;libc=ctypes.CDLL(None,use_errno=True);"
                      f"path={str(outside)!r}.encode();value=ctypes.create_string_buffer(b'changed');"
                      "args=ctypes.create_string_buffer(struct.pack('=QII',ctypes.addressof(value),7,0));results=[]\n"
                      "for number,params in [(463,(ctypes.c_long(-100),path,ctypes.c_long(0),"
                      "b'user.myriad_sentinel',args,ctypes.c_long(16))),"
                      "(466,(ctypes.c_long(-100),path,ctypes.c_long(0),b'user.myriad_sentinel'))]:\n"
                      " ret=libc.syscall(ctypes.c_long(number),*params);results.append([number,ret,ctypes.get_errno()])\n"
                      "fd=os.open(path,os.O_RDONLY);flags=ctypes.c_int(0x40);"
                      "ret=libc.ioctl(fd,ctypes.c_ulong(0x40086602),ctypes.byref(flags));"
                      "results.append([16,ret,ctypes.get_errno()]);os.close(fd);print(json.dumps(results))")
            code, output = self.env.run("python -c " + shlex.quote(script), 5)
            self.assertEqual(code, 0, output)
            self.assertEqual(json.loads(output), [[number, -1, 1] for number in (463, 466, 16)])
            self.assertEqual(os.getxattr(outside, "user.myriad_sentinel"), b"original")
            import fcntl
            import struct
            with outside.open("rb") as sentinel:
                flags = struct.unpack("i", fcntl.ioctl(sentinel, 0x80086601, struct.pack("i", 0)))[0]
            self.assertEqual(flags & 0x40, 0)

    def test_file_setattr_cannot_mutate_external_inode_flags(self):
        import fcntl
        import struct
        outside = self.root / "outside"
        outside.write_text("original")
        def attributes():
            with outside.open("rb") as sentinel:
                return struct.unpack("i", fcntl.ioctl(sentinel, 0x80086601, struct.pack("i", 0)))[0]
        before = attributes()
        script = ("import ctypes,struct,json;libc=ctypes.CDLL(None,use_errno=True);"
                  "buf=ctypes.create_string_buffer(struct.pack('=QIIII',128,0,0,0,0));"
                  f"ret=libc.syscall(ctypes.c_long(469),ctypes.c_long(-100),{str(outside)!r}.encode(),buf,"
                  "ctypes.c_long(24),ctypes.c_long(0));print(json.dumps([ret,ctypes.get_errno()]))")
        code, output = self.env.run("python -c " + shlex.quote(script), 5)
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output), [-1, 1])
        self.assertEqual(attributes(), before)

    def test_symlink_escape_denied(self):
        outside = self.root / "outside"
        outside.write_text("original")
        (self.work / "link").symlink_to(outside)
        code, output = self.env.run("cat link", 5)
        self.assertNotEqual(code, 0)
        code, _ = self.env.run("echo escaped > link", 5)
        self.assertNotEqual(code, 0)
        self.assertEqual(outside.read_text(), "original")
        with self.assertRaises(ValueError):
            collect_sources(self.work, ["link"])

    def test_network_and_unix_socket_denied(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen()
        try:
            port = server.getsockname()[1]
            for family in ("socket.AF_INET", "socket.AF_UNIX"):
                command = f"python -c 'import socket; socket.socket({family})'"
                code, output = self.env.run(command, 5)
                self.assertNotEqual(code, 0)
                self.assertIn("Operation not permitted", output)
            code, _ = self.env.run(f"bash -c 'echo escape > /dev/tcp/127.0.0.1/{port}'", 5)
            self.assertNotEqual(code, 0)
        finally:
            server.close()

    def test_limits_clean_environment_and_namespace_escape(self):
        script = ("import json,os,resource;print(json.dumps({'memory':resource.getrlimit(resource.RLIMIT_AS),"
                  "'processes':resource.getrlimit(resource.RLIMIT_NPROC),"
                  "'leaked':'CODE_PILOT_SENTINEL' in os.environ,'home':os.environ['HOME']}))")
        with patch.dict(os.environ, {"CODE_PILOT_SENTINEL": "outside"}):
            code, output = self.env.run("python -c " + shlex.quote(script), 5)
        self.assertEqual(code, 0, output)
        data = json.loads(output)
        self.assertEqual(data["memory"], [512 << 20, 512 << 20])
        self.assertEqual(data["processes"], [32, 32])
        self.assertFalse(data["leaked"])
        self.assertEqual(data["home"], str(self.work))
        code, _ = self.env.run("unshare --user --map-root-user true", 5)
        self.assertNotEqual(code, 0)

    def test_timeout_kills_setsid_descendant(self):
        tag = "myriad_descendant_" + self.root.name
        script = "import os,time;os.setsid();time.sleep(2);open('escaped','w').write('alive');time.sleep(30)"
        command = f"python -c {shlex.quote(script)} {tag} & wait"
        found = []
        def scan():
            time.sleep(0.25)
            for proc in Path("/proc").iterdir():
                if proc.name.isdigit():
                    try:
                        if tag.encode() in (proc / "cmdline").read_bytes():
                            found.append(proc)
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        pass
        thread = threading.Thread(target=scan)
        thread.start()
        code, output = self.env.run(command, 0.8)
        thread.join()
        self.assertEqual(code, 124, output)
        self.assertTrue(found, "the escaped-session process must actually have started")
        time.sleep(0.2)
        for proc in found:
            if proc.exists():
                self.assertEqual((proc / "stat").read_text().split()[2], "Z", "live descendant after timeout")
        self.assertFalse((self.work / "escaped").exists())

    def test_fail_closed_with_unsupported_runtime(self):
        # Real launcher rejects an unsupported runtime before executing the model command.
        other = self.root / "runtime"
        (other / "bin").mkdir(parents=True)
        (other / "bin/python").symlink_to("/bin/echo")
        env = LocalLinuxEnv(self.work, other)
        with self.assertRaises(IsolationError):
            env.run("echo unprotected > escape", 5)
        self.assertFalse((self.work / "escape").exists())

    def test_verifier_rejects_exit_and_monkeypatch(self):
        cache = self.root / "cache"
        fixture(cache)
        for source in ("import os; os._exit(0)",
                       "import unittest; unittest.TestCase.assertEqual=lambda *a: None\n"
                       "def encode(*args): return 'wrong'"):
            grade = verify(cache, "affine-cipher", {"affine_cipher.py": source}, self.runtime)
            self.assertIsNot(grade["passed"], True)
        self.assertEqual(verify(cache, "affine-cipher", {"affine_cipher.py": "import os; os._exit(0)"},
                                self.runtime)["status"], "graded")

    def test_candidate_timeout_and_syntax_are_failures(self):
        cache = self.root / "cache"
        base = fixture(cache)
        test = base / "affine_cipher_test.py"
        second = "\n    def test_second(self): self.assertEqual(encode('input', 5, 7), 'expected')\n"
        test.write_text(test.read_text() + second)
        cases = ["while True: pass", "def encode(:"]
        for source in cases:
            grade = verify(cache, "affine-cipher", {"affine_cipher.py": source}, self.runtime, timeout_s=0.5)
            self.assertEqual(grade["status"], "graded", grade)
            self.assertIs(grade["passed"], False)
            self.assertEqual(grade["tests"], 1)
            self.assertEqual(grade["tests_expected"], 2)
            self.assertFalse(grade["tests_complete"])

    def test_candidate_prints_are_diagnostics_and_not_a_score(self):
        cache = self.root / "cache"
        fixture(cache)
        for returned, passed in (("expected", True), ("wrong", False)):
            source = ("import sys\nprint('module diagnostic')\ndef encode(*args):\n"
                      " print('function diagnostic')\n print('stderr diagnostic', file=sys.stderr)\n"
                      f" return {returned!r}\n")
            grade = verify(cache, "affine-cipher", {"affine_cipher.py": source}, self.runtime)
            self.assertEqual(grade["status"], "graded", grade)
            self.assertIs(grade["passed"], passed)
            self.assertIn("function diagnostic", grade["calls"][0]["diagnostics"])
            self.assertIn("stderr diagnostic", grade["calls"][0]["diagnostics"])
        source = "import os; print('{\"passed\":true}', flush=True); os._exit(0)"
        grade = verify(cache, "affine-cipher", {"affine_cipher.py": source}, self.runtime)
        self.assertEqual(grade["status"], "graded", grade)
        self.assertIs(grade["passed"], False)
        self.assertEqual(grade["calls"][0]["candidate_failure"], "invalid_result_artifact")

    def test_invalid_submitted_sources_are_graded_failures(self):
        cache = self.root / "cache"
        fixture(cache)
        args = SimpleNamespace(tasks_dir=cache, runtime=self.runtime, max_tokens=10, total_tokens=30, seed=0,
                               max_steps=3, timeout=10, cmd_timeout=5, verify_timeout=5, reference=None)
        commands = ["rm affine_cipher.py", "printf '\\377' > affine_cipher.py",
                    "rm affine_cipher.py; ln -s /etc/passwd affine_cipher.py",
                    "rm affine_cipher.py; mkdir affine_cipher.py"]
        for command in commands:
            client = FakeClient([answer(f"```bash\n{command}\n```", 2), answer(f"```bash\necho {SUBMIT}\n```", 3)])
            with patch("essaim.code_pilot.verify") as verifier:
                row = run_one(args, "affine-cipher", "single", ["a"], client, self.root, "campaign")
            self.assertEqual(row["status"], "graded", row)
            self.assertIs(row["passed"], False)
            self.assertEqual(row["verifier"]["candidate_failure"], "invalid_source_artifact")
            self.assertEqual(row["verifier"]["tests"], 0)
            self.assertFalse(row["verifier"]["tests_complete"])
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
