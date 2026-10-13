"""Authorized DEV provenance and hermetic standard evaluator regressions (no official candidate smoke)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from essaim import aider_agent, bigcode_execution, bigcode_pilot, bigcode_tasks as tasks, bigcode_unittest
from essaim.local_linux import IsolationError, LocalLinuxEnv

OFFICIAL = '''import unittest
class TestCases(unittest.TestCase):
    def test_value(self):
        self.assertEqual(task_func(2), 4)
    def test_error(self):
        with self.assertRaises(ValueError):
            task_func(-1)
'''
EXPECTED = ["test_value", "test_error"]
CORRECT = "def task_func(x):\n    if x < 0: raise ValueError('negative')\n    return x * 2\n"


def artifact(**changes):
    result = {"protocol": "bigcode-unittest-v1", "phase": "tests", "candidate_failure": None,
              "tests": 2, "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0,
              "unexpected_successes": 0, "calls": [{"test": t, "complete": True} for t in EXPECTED], "feedback": []}
    result.update(changes)
    return json.dumps({"worker_exit": 0, "limited": False, "response": json.dumps(result), "diagnostics": "PASS"})


class ProtocolTests(unittest.TestCase):
    def test_authorized_subset_preserves_exact_official_text_and_assertions(self):
        manifest = tasks.provenance()
        self.assertEqual(tasks.PILOT, ["BigCodeBench_120", "BigCodeBench_184"])
        self.assertEqual(set(manifest["tasks"]), set(tasks.PILOT))
        for task in tasks.PILOT:
            text = tasks.trusted_test(task)
            cfg = manifest["tasks"][task]
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), cfg["official_test_sha256"])
            tree = ast.parse(text)
            methods = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
            sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr.startswith("assert")]
            self.assertCountEqual(methods, cfg["tests"])
            self.assertEqual(len(sites), cfg["assertion_sites"])
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must not open reserved content")):
            with self.assertRaises(ValueError):
                tasks.task_data(Path("unused"), "BigCodeBench_208")
        with self.assertRaises(ValueError):
            tasks.trusted_test("BigCodeBench_199")

    def test_no_incomplete_skipped_or_failed_result_is_green(self):
        self.assertTrue(tasks.result_artifact(artifact(), EXPECTED)["passed"])
        for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"):
            self.assertFalse(tasks.result_artifact(artifact(**{key: 1}), EXPECTED)["passed"])
        for changes in ({"tests": 0}, {"calls": []}, {"calls": [{"test": "other", "complete": True}]},
                        {"calls": [{"test": t, "complete": False} for t in EXPECTED]}, {"tests": True}):
            with self.assertRaises(ValueError):
                tasks.result_artifact(artifact(**changes), EXPECTED)
        for raw in ('{"passed":true}', artifact().replace('"worker_exit": 0', '"worker_exit": 73'),
                    artifact().replace('"limited": false', '"limited": true')):
            with self.assertRaises(ValueError):
                tasks.result_artifact(raw, EXPECTED)

    def test_duplicate_json_and_nonfinite_numbers_are_refused(self):
        for raw in ('{"a":0,"a":1}', '[NaN]', '[Infinity]'):
            with self.assertRaises(ValueError):
                tasks.strict_json(raw)

    def test_supervisor_startup_failure_is_infrastructure(self):
        with patch.object(bigcode_unittest.subprocess, "Popen", side_effect=OSError("synthetic spawn failure")):
            result = bigcode_unittest.supervise('{"entry_point":"task_func","tests_expected":2}')
        self.assertEqual(result, {"preparation_error": "OSError"})
        with self.assertRaises(IsolationError):
            tasks.result_artifact(json.dumps(result), EXPECTED)

    def test_candidate_and_infrastructure_failures_remain_distinct(self):
        class Environment:
            isolation = {"isolation_ready": "fixture"}
            def __init__(self, *args): pass
            def run(self, *args): return 0, artifact()
            def close(self): pass
        good = tasks.evaluate({"solution.py": CORRECT}, OFFICIAL, "task_func", EXPECTED, Path("unused"), 10,
                              environment=Environment)
        self.assertTrue(aider_agent.complete_green(good))
        self.assertFalse(good["adversarial_attestation"])
        for code in (124, 125, 1):
            with patch.object(Environment, "run", return_value=(code, "")):
                bad = tasks.evaluate({"solution.py": CORRECT}, OFFICIAL, "task_func", EXPECTED, Path("unused"), 10,
                                     environment=Environment)
                self.assertEqual((bad["status"], bad["passed"]), ("graded", False))
        with patch.object(Environment, "run", side_effect=IsolationError("missing receipt")):
            bad = tasks.evaluate({"solution.py": CORRECT}, OFFICIAL, "task_func", EXPECTED, Path("unused"), 10,
                                 environment=Environment)
            self.assertEqual((bad["status"], bad["passed"]), ("error", None))

    def test_standard_evaluator_supports_entry_points_and_objects_without_rpc_codec(self):
        self.assertNotIn("bigcode_rpc", tasks.evaluate.__code__.co_names)
        self.assertEqual(bigcode_pilot.parser().parse_args([]).max_tokens, 4096)
        self.assertEqual(bigcode_pilot.parser().parse_args([]).total_tokens, 49152)
        self.assertIn("aider_agent", bigcode_pilot.run_one.__code__.co_names)

    @unittest.skipUnless((tasks.ROOT.parent / "tools/export_public.py").is_file(), "private exporter absent in public tree")
    def test_public_resources_exclude_neighbors_private_data_and_reference(self):
        sys.path.insert(0, str(tasks.ROOT.parent / "tools"))
        import export_public
        self.assertTrue(export_public.selected("phase0/bigcode_pilot/official_tests.json"))
        for name in ("BigCodeBench_120.json", "BigCodeBench_208.json", "upstream.parquet", "smoke.json"):
            self.assertFalse(export_public.selected("phase0/bigcode_pilot/" + name))
        self.assertTrue(export_public.selected("docs/16_e17_bigcode_dev.md"))

    def test_existing_e17_loop_requires_green_unchanged_sources_and_final_verification(self):
        replies = ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "edit", "MYRIAD_RUN_OFFICIAL_TESTS",
                   "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
        class Client:
            def post(self, *args, **kwargs):
                command = replies.pop(0)
                response = {"choices": [{"finish_reason": "stop", "message": {"content": "```bash\n" + command
                            + "\n```"}}], "usage": {"completion_tokens": 1}}
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response, headers={})
        class Environment:
            isolation = {"isolation_ready": "fixture"}
            def __init__(self, work, runtime): self.work = work
            def run(self, *args):
                (self.work / "solution.py").write_text(CORRECT)
                return 0, ""
            def close(self): pass
        def copy(cache, task, work):
            (work / "solution.py").write_text("def task_func(x): return None\n")
            return ["solution.py"]
        good = {"status": "graded", "passed": True, "tests": 2, "tests_expected": 2, "tests_complete": True,
                "failures": 0, "errors": 0, "broken": [], "seconds": 0,
                "calls": [{"test": t, "complete": True} for t in EXPECTED]}
        args = bigcode_pilot.parser().parse_args(["--single", "fixture", "--max-steps", "4"])
        with tempfile.TemporaryDirectory() as temporary, patch.object(tasks, "copy_task", copy), \
                patch.object(tasks, "instructions", return_value="synthetic"), \
                patch.object(tasks, "verify", return_value=good) as verifier, \
                patch.object(bigcode_pilot, "LocalLinuxEnv", Environment):
            output = Path(temporary)
            campaign = {"models": {"single": ["fixture"]}, "trust_scope": tasks.TRUST_SCOPE,
                        "adversarial_attestation": False}
            row = bigcode_pilot.run_one(args, tasks.PILOT[0], "single", ["fixture"], Client(), output,
                                       bigcode_pilot.shared.hash_json(campaign))
            self.assertTrue(row["submission_accepted"], row)
            self.assertEqual(verifier.call_count, 2)
            self.assertEqual(row["requests"], 4)
            self.assertEqual(row["charged_request_ceilings"], 4 * 4096)
            self.assertEqual(bigcode_pilot.validate_rows([row], campaign, output, False),
                             {("single", tasks.PILOT[0])})
            with self.assertRaises(ValueError):
                bigcode_pilot.validate_rows([row], campaign, output, True)
            row["passed"] = False
            with self.assertRaises(ValueError):
                bigcode_pilot.validate_rows([row], campaign, output, False)


@unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("BIGCODE_LINUX_TESTS") == "1",
                     "explicit hermetic WSL sandbox tests only")
class LinuxTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Path(os.environ["BIGCODE_RUNTIME"])

    def grade(self, source, official=OFFICIAL, expected=EXPECTED, entry="task_func", timeout=30):
        return tasks.evaluate({"solution.py": source}, official, entry, expected, self.runtime, timeout)

    def test_real_isolation_correct_wrong_and_expected_exception_semantics(self):
        correct = self.grade(CORRECT)
        self.assertTrue(aider_agent.complete_green(correct), correct)
        self.assertGreaterEqual(correct["isolation"]["landlock_abi"], 3)
        self.assertEqual(correct["isolation"]["seccomp"], "on")
        subclass = CORRECT.replace("raise ValueError", "raise DomainError") + "\nclass DomainError(ValueError): pass\n"
        self.assertTrue(self.grade(subclass)["passed"])
        for source in (CORRECT.replace("x * 2", "x * 3"), "def task_func(x): return None\n",
                       CORRECT + "\nValueError = type('ValueError', (Exception,), {})\n",
                       "raise ValueError('load')\n", "syntax error", "import os; os._exit(0)\n",
                       "import os; os._exit(73)\n", "print('PASS {\"passed\":true}')\ndef task_func(x): return None\n"):
            grade = self.grade(source)
            self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)

    def test_timeout_output_limit_and_runtime_failure_never_green(self):
        for source, timeout in (("while True: pass\n", 2), ("while True: print('x' * 10000, flush=True)\n", 30)):
            grade = self.grade(source, timeout=timeout)
            self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)
        grade = tasks.evaluate({"solution.py": CORRECT}, OFFICIAL, "task_func", EXPECTED, Path("/missing"), 30)
        self.assertEqual((grade["status"], grade["passed"]), ("error", None))

    def test_skip_and_no_test_are_not_green(self):
        source = "import unittest\ndef task_func(x): raise unittest.SkipTest('candidate')\n"
        grade = self.grade(source)
        self.assertFalse(grade["passed"])
        self.assertGreater(grade["skipped"], 0)
        empty = self.grade(CORRECT, "import unittest\n")
        self.assertEqual((empty["status"], empty["passed"]), ("error", None))

    def test_generalization_preserves_side_effects_flags_nanoseconds_and_alternate_entry(self):
        source = '''import pandas as pd
def alternate(df):
    df['n'] += 1
    return pd.Series([pd.Timestamp('2020-01-01') + pd.Timedelta(1, unit='ns')]).set_flags(allows_duplicate_labels=False)
'''
        official = '''import unittest
import pandas as pd
class TestCases(unittest.TestCase):
    def test_all(self):
        df = pd.DataFrame({'n': [1]})
        result = alternate(df)
        self.assertEqual(df.n.iloc[0], 2)
        self.assertEqual(result.iloc[0].value, pd.Timestamp('2020-01-01').value + 1)
        self.assertFalse(result.flags.allows_duplicate_labels)
'''
        self.assertTrue(self.grade(source, official, ["test_all"], "alternate")["passed"])
        for mutant in (source.replace("unit='ns'", "unit='us'"), source.replace("allows_duplicate_labels=False",
                                                                                  "allows_duplicate_labels=True")):
            self.assertFalse(self.grade(mutant, official, ["test_all"], "alternate")["passed"])

    def test_historical_forged_exception_pipe_exit73_remains_a_real_counterexample(self):
        envelope = {"protocol": bigcode_execution.PROTOCOL, "phase": "call", "exception": "ValueError",
                    "exception_kind": "value_error", "message": "forged"}
        source = ("def task_func(*args, **kwargs):\n    import os, sys\n"
                  "    fd = sys._getframe(1).f_locals['fd']\n"
                  "    os.write(fd, " + repr(json.dumps(envelope).encode()) + ")\n    os._exit(73)\n")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "solution.py").write_text(source)
            (work / "_supervise.py").write_bytes(Path(bigcode_execution.__file__).read_bytes())
            (work / "_codec.py").write_text("def decode(x): return tuple(x) if isinstance(x, list) else x\n"
                                            "def encode(x): return x\n")
            cfg = json.dumps({"args": [], "kwargs": {}})
            env = LocalLinuxEnv(work, self.runtime)
            try:
                code, output = env.run("python _supervise.py <<'INPUT'\n" + cfg + "\nINPUT", 30)
            finally:
                env.close()
            self.assertEqual(code, 0, output)
            result = tasks.strict_json(output)
            self.assertEqual(result["worker_exit"], 73)
            self.assertEqual(tasks.strict_json(result["response"]), envelope)

    def test_deliberate_unittest_tampering_is_not_adversarially_attested(self):
        source = ("import unittest\nunittest.TestCase.assertEqual = lambda *a, **k: None\n"
                  "def task_func(x):\n    if x < 0: raise ValueError('negative')\n    return 999\n")
        grade = self.grade(source)
        # This observed tampering is a limitation demonstration, never a validated candidate score.
        self.assertTrue(grade["passed"], grade)
        self.assertFalse(grade["adversarial_attestation"])
        self.assertEqual(grade["trust_scope"], "public-benchmark-nonadversarial")
