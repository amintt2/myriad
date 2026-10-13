"""Synthetic unittest module fixtures through the production Aider worker and result validator."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from essaim import aider_python, aider_unittest, bigcode_unittest
from essaim.aider_agent import complete_green
from essaim.local_linux import LocalLinuxEnv
from tests.test_aider_generic import FUNCTION, fixture

SETUP_REQUIRED = '''import unittest
from operations import twice
ready = False
def setUpModule():
    global ready
    ready = True
class Cases(unittest.TestCase):
    def test_value(self):
        self.assertTrue(ready)
        self.assertEqual(twice(2), 4)
'''
SETUP_ERROR = '''import unittest
from operations import twice
def setUpModule():
    raise RuntimeError('synthetic setup error')
class Cases(unittest.TestCase):
    def test_value(self):
        self.assertEqual(twice(2), 4)
'''
TEARDOWN_ERROR = SETUP_ERROR.replace("def setUpModule():", "def tearDownModule():").replace(
    "synthetic setup error", "synthetic teardown error")
FIRST = '''import unittest
import operations
state = 'unset'
def setUpModule():
    global state
    state = 'first-ready'
    operations.events.append('first-up')
def tearDownModule():
    assert state == 'first-ready'
    operations.events.append('first-down')
class Cases(unittest.TestCase):
    def test_value(self):
        self.assertEqual(state, 'first-ready')
        self.assertEqual(operations.events, ['first-up'])
        self.assertEqual(operations.twice(2), 4)
'''
SECOND = '''import unittest
import operations
state = 'unset'
def setUpModule():
    global state
    assert operations.events == ['first-up', 'first-down']
    state = 'second-ready'
    operations.events.append('second-up')
def tearDownModule():
    assert state == 'second-ready'
    assert operations.events == ['first-up', 'first-down', 'second-up']
    operations.events.append('second-down')
class Cases(unittest.TestCase):
    def test_value(self):
        self.assertEqual(state, 'second-ready')
        self.assertEqual(operations.events, ['first-up', 'first-down', 'second-up'])
        self.assertEqual(operations.twice(3), 6)
'''


class WorkerDone(BaseException):
    pass


def synthetic_worker(work, raw):
    """Run only these trusted synthetic sources in-process; this is not a security-boundary test."""
    response, diagnostics = bytearray(), io.StringIO()
    old_work = Path.cwd()
    def write(fd, chunk):
        response.extend(chunk)
        return len(chunk)
    with patch.dict(sys.modules, {"_myriad_transport": bigcode_unittest}), \
            patch.object(os, "write", side_effect=write), patch.object(os, "_exit", side_effect=WorkerDone), \
            contextlib.redirect_stdout(diagnostics):
        for name in ("operations", "official_0", "official_1"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(work))
        try:
            os.chdir(work)
            try:
                aider_unittest.worker(raw, 999)
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
    def grade(self, tests):
        captured = {}
        if self.strict:
            runtime = Path(os.environ.get("CODE_PILOT_RUNTIME",
                                         str(Path.home() / ".local/share/myriad-code-pilot/venv")))
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
        sources = {"operations.py": FUNCTION + "\nevents = []\n"}
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            base = fixture(cache, sources, tests)
            originals = {name: (base / name).read_bytes() for name in tests}
            with patch.object(aider_python, "provenance", return_value={"tasks": ["synthetic-operations"]}):
                grade = aider_python.verify(cache, "synthetic-operations", sources, runtime, 30,
                                            environment=Environment)
            self.assertEqual(originals, {name: (base / name).read_bytes() for name in tests})
        outer = json.loads(captured["output"])
        worker = json.loads(outer["response"])
        evidence = os.environ.get("AIDER_MODULE_PROOFS")
        if evidence:
            directory = Path(evidence) / self.id()
            directory.mkdir(parents=True, exist_ok=False)
            for name, text in {**sources, **tests}.items():
                (directory / name).write_bytes(text.encode())
            with (directory / "result.json").open("x", encoding="utf-8") as out:
                json.dump({"strict": self.strict, "worker_exit": outer["worker_exit"], "worker": worker,
                           "grade": grade, "diagnostics": outer["diagnostics"]}, out, indent=2)
        self.assertEqual(captured["code"], 0)
        self.assertEqual(outer["worker_exit"], 0)
        if self.strict:
            self.assertGreaterEqual(grade["isolation"]["landlock_abi"], 3)
            self.assertEqual(grade["isolation"]["seccomp"], "on")
            self.assertEqual(set(grade["isolation"]["namespaces"]), {"user", "mount", "pid", "net", "ipc", "uts"})
        return grade, worker

    def test_setup_required_for_success(self):
        grade, worker = self.grade({"public_test.py": SETUP_REQUIRED})
        self.assertTrue(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"], worker["failures"]), (1, 0, 0))
        self.assertEqual(grade["tests_expected"], 1)

    def test_setup_failure_prevents_green_and_preserves_worker_counts(self):
        grade, worker = self.grade({"public_test.py": SETUP_ERROR})
        self.assertFalse(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"], worker["calls"]), (0, 1, []))
        self.assertEqual((grade["status"], grade["passed"], grade["tests_expected"]), ("graded", False, 1))
        self.assertFalse(grade["tests_complete"])
        self.assertIn("setUpModule (official_0)", worker["feedback"][0]["test"])

    def test_teardown_failure_prevents_green_after_completed_method(self):
        grade, worker = self.grade({"public_test.py": TEARDOWN_ERROR})
        self.assertFalse(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"]), (1, 1))
        self.assertEqual((grade["status"], grade["passed"], grade["tests"], grade["errors"]), ("graded", False, 1, 1))
        self.assertTrue(grade["tests_complete"])
        self.assertIn("tearDownModule (official_0)", grade["feedback"][0]["test"])

    def test_multiple_modules_apply_their_own_fixtures_in_order(self):
        grade, worker = self.grade({"first_test.py": FIRST, "second_test.py": SECOND})
        self.assertTrue(complete_green(grade), grade)
        self.assertEqual((worker["tests"], worker["errors"], worker["failures"]), (2, 0, 0))
        self.assertEqual([record["test"] for record in worker["calls"]],
                         ["official_0.Cases.test_value", "official_1.Cases.test_value"])
        self.assertEqual(grade["tests_expected"], 2)


class SyntheticWorkerTests(FixtureCases, unittest.TestCase):
    strict = False


@unittest.skipUnless(sys.platform.startswith("linux"), "strict Linux worker; Windows synthetic fixtures only")
class StrictWorkerTests(FixtureCases, unittest.TestCase):
    strict = True


if __name__ == "__main__":
    unittest.main()
