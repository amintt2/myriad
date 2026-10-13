"""Synthetic module fixtures through the public, non-adversarial BigCode unittest worker."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from essaim import bigcode_tasks, bigcode_unittest
from essaim.aider_agent import complete_green
from essaim.local_linux import LocalLinuxEnv

SOURCE = "def task_func(x): return x * 2\n"
NORMAL = '''import unittest
class Cases(unittest.TestCase):
    def test_value(self):
        self.assertEqual(task_func(2), 4)
'''
SETUP_REQUIRED = NORMAL.replace("class Cases", """ready = False
def setUpModule():
    global ready
    ready = True
class Cases""").replace("        self.assertEqual", "        self.assertTrue(ready)\n        self.assertEqual")
SETUP_ERROR = NORMAL.replace("class Cases", """def setUpModule():
    raise RuntimeError('synthetic setup error')
class Cases""")
TEARDOWN_ERROR = SETUP_ERROR.replace("setUpModule", "tearDownModule").replace("setup error", "teardown error")


class WorkerDone(BaseException):
    pass


def synthetic_worker(work, raw):
    """Intercept transport only for trusted synthetic code; this does not test an isolation boundary."""
    response, diagnostics = bytearray(), io.StringIO()
    old_work = Path.cwd()
    def write(fd, chunk):
        response.extend(chunk)
        return len(chunk)
    with patch.dict(sys.modules), patch.object(os, "write", side_effect=write), \
            patch.object(os, "_exit", side_effect=WorkerDone), contextlib.redirect_stdout(diagnostics):
        for name in ("solution", "official"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(work))
        try:
            os.chdir(work)
            try:
                bigcode_unittest.worker(raw, 999)
            except WorkerDone:
                pass
            else:
                raise AssertionError("worker must finish through its completion exit")
        finally:
            os.chdir(old_work)
            sys.path.pop(0)
    return json.dumps({"worker_exit": 0, "limited": False, "response": response.decode(),
                       "diagnostics": diagnostics.getvalue()})


class FixtureCases:
    def grade(self, official):
        captured = {}
        if self.strict:
            runtime = Path(os.environ.get("BIGCODE_RUNTIME",
                                         str(Path.home() / ".local/share/myriad-bigcode-pilot/venv")))
            class Environment(LocalLinuxEnv):
                def run(self, *args):
                    code, output = super().run(*args)
                    captured.update(code=code, output=output)
                    return code, output
        else:
            runtime = Path("unused-synthetic-runtime")
            class Environment:
                isolation = {"synthetic_worker_only": True}
                def __init__(self, work, runtime): self.work = work
                def run(self, command, timeout):
                    output = synthetic_worker(self.work, command.splitlines()[1])
                    captured.update(code=0, output=output)
                    return 0, output
                def close(self): pass
        grade = bigcode_tasks.evaluate({"solution.py": SOURCE}, official, "task_func", ["test_value"], runtime, 30,
                                       environment=Environment)
        outer = json.loads(captured["output"])
        worker = json.loads(outer["response"])
        evidence = os.environ.get("BIGCODE_MODULE_PROOFS")
        if evidence:
            directory = Path(evidence) / self.id()
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "solution.py").write_bytes(SOURCE.encode())
            (directory / "_official.py").write_bytes(official.encode())
            (directory / "result.json").write_text(json.dumps({"strict": self.strict, "worker": worker,
                "grade": grade, "outer": outer}, indent=2), encoding="utf-8")
        self.assertEqual(captured["code"], 0)
        self.assertEqual(outer["worker_exit"], 0)
        self.assertFalse(outer["limited"])
        self.assertEqual(grade["trust_scope"], bigcode_tasks.TRUST_SCOPE)
        self.assertFalse(grade["adversarial_attestation"])
        self.assertEqual(grade["tests_expected"], 1)
        if self.strict:
            self.assertGreaterEqual(grade["isolation"]["landlock_abi"], 3)
            self.assertEqual(grade["isolation"]["seccomp"], "on")
            self.assertEqual(set(grade["isolation"]["namespaces"]), {"user", "mount", "pid", "net", "ipc", "uts"})
        return grade, worker

    def test_setup_required_for_success(self):
        grade, worker = self.grade(SETUP_REQUIRED)
        self.assertTrue(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"], worker["failures"]), (1, 0, 0))

    def test_setup_failure_preserves_partial_worker_counts_and_prevents_green(self):
        grade, worker = self.grade(SETUP_ERROR)
        self.assertFalse(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"], worker["failures"], worker["calls"]), (0, 1, 0, []))
        self.assertEqual((grade["status"], grade["passed"], grade["tests_complete"]), ("graded", False, False))
        self.assertEqual(grade["candidate_failure"], "incomplete official tests")
        self.assertIn("setUpModule", worker["feedback"][0]["test"])

    def test_teardown_failure_prevents_green_after_completed_method(self):
        grade, worker = self.grade(TEARDOWN_ERROR)
        self.assertFalse(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"], worker["failures"]), (1, 1, 0))
        self.assertEqual(worker["calls"], [{"test": "test_value", "complete": True}])
        self.assertEqual((grade["status"], grade["passed"], grade["tests_complete"]), ("graded", False, True))
        self.assertEqual((grade["tests"], grade["errors"]), (1, 1))
        self.assertIn("tearDownModule", grade["feedback"][0]["test"])

    def test_normal_module_keeps_exact_counts_after_import(self):
        grade, worker = self.grade(NORMAL)
        self.assertTrue(complete_green(grade), grade)
        counts = ("tests", "errors", "failures", "skipped", "expected_failures", "unexpected_successes")
        self.assertEqual(tuple(worker[key] for key in counts), (1, 0, 0, 0, 0, 0))
        self.assertEqual(tuple(grade[key] for key in counts), (1, 0, 0, 0, 0, 0))
        self.assertEqual(worker["calls"], [{"test": "test_value", "complete": True}])


class SyntheticWorkerTests(FixtureCases, unittest.TestCase):
    strict = False


@unittest.skipUnless(sys.platform.startswith("linux"), "strict Linux worker; Windows synthetic fixtures only")
class StrictWorkerTests(FixtureCases, unittest.TestCase):
    strict = True


if __name__ == "__main__":
    unittest.main()
