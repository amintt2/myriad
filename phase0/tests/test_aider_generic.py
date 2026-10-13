"""Synthetic generic Aider fixtures only; no reserved exercise content or inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from essaim import aider_python, aider_unittest, bigcode_unittest
from essaim.aider_agent import complete_green
from essaim.ephemeral_vm import EphemeralVMEnv
from essaim.local_linux import IsolationError

FUNCTION = "def twice(x):\n    if x < 0: raise ValueError('negative')\n    return x * 2\n"
OFFICIAL = '''import unittest
from operations import twice
class Values(unittest.TestCase):
    def test_value(self):
        self.assertEqual(twice(3), 6)
    def test_exception(self):
        with self.assertRaises(ValueError):
            twice(-1)
'''
CLASS = '''from arithmetic import advance
class Counter:
    def __init__(self, value): self.value = value
    def add(self, amount):
        self.value = advance(self.value, amount)
        return self
'''
OBJECT_TEST = '''import unittest
from counter import Counter
class Objects(unittest.TestCase):
    def test_state(self):
        first = Counter(4)
        second = Counter(7)
        self.assertIs(first.add(2), first)
        self.assertEqual(first.add(3).value, 9)
        self.assertEqual(second.value, 7)
    def test_independent(self):
        self.assertEqual(Counter(0).add(-2).value, -2)
'''


def fixture(cache, sources, tests, task="synthetic-operations"):
    base = cache / "python/exercises/practice" / task
    (base / ".meta").mkdir(parents=True)
    config = {"files": {"solution": list(sources), "test": list(tests), "example": []}}
    (base / ".meta/config.json").write_text(json.dumps(config), encoding="utf-8")
    for name, text in {**sources, **tests}.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    return base


def artifact(expected, **changes):
    data = {"protocol": bigcode_unittest.PROTOCOL, "phase": "tests", "candidate_failure": None,
            "tests": len(expected), "calls": [{"test": name, "complete": True} for name in expected],
            "failures": 0, "errors": 0, "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
            "feedback": []}
    data.update(changes)
    return json.dumps({"worker_exit": 0, "limited": False, "response": json.dumps(data), "diagnostics": "OK"})


class PreparationTests(unittest.TestCase):
    def test_inventory_is_static_nonempty_and_tracks_classes_files_and_inheritance(self):
        tests = {"one.py": OFFICIAL, "two.py": OBJECT_TEST}
        expected = aider_unittest.inventory(tests)
        self.assertEqual(len(expected), 4)
        self.assertIn("official_1.Objects.test_state", expected)
        inherited = OFFICIAL + "\nclass Derived(Values):\n    def test_extra(self): pass\n"
        self.assertEqual(len(aider_unittest.inventory({"one.py": inherited})), 5)
        for text in ("import unittest\n", "syntax error"):
            with self.assertRaises((ValueError, SyntaxError)):
                aider_unittest.inventory({"one.py": text})

    def grade(self, environment, sources=None, tests=None, task="synthetic-operations", timeout=10):
        sources = sources if sources is not None else {"operations.py": FUNCTION}
        tests = tests if tests is not None else {"public_test.py": OFFICIAL}
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            fixture(cache, sources, tests, task)
            with patch.object(aider_python, "provenance", return_value={"tasks": [task]}):
                return aider_python.verify(cache, task, sources, cache, timeout, environment=environment, generic=True)

    def test_staged_sources_tests_and_hashes_are_exact_and_cleanup_is_required(self):
        owner = self
        class Environment:
            isolation = {"simulated": True}
            work = None
            closed = False
            def __init__(self, work, runtime):
                Environment.work = work
                owner.assertEqual((work / "operations.py").read_bytes(), FUNCTION.encode())
                owner.assertEqual((work / "public_test.py").read_bytes(), OFFICIAL.replace("\n", "\r\n").encode())
                owner.assertEqual(set(p.name for p in work.iterdir()),
                                  {"operations.py", "public_test.py", "_myriad_runner.py", "_myriad_transport.py"})
            def run(self, command, timeout):
                return 0, artifact(aider_unittest.inventory({"public_test.py": OFFICIAL}))
            def close(self): Environment.closed = True
        grade = self.grade(Environment, tests={"public_test.py": OFFICIAL.replace("\n", "\r\n")})
        self.assertTrue(complete_green(grade), grade)
        self.assertFalse(grade["adversarial_attestation"])
        self.assertEqual(grade["source_hashes"], {"operations.py": hashlib.sha256(FUNCTION.encode()).hexdigest()})
        self.assertTrue(Environment.closed)
        self.assertFalse(Environment.work.exists())

    def test_candidate_and_infrastructure_failures_do_not_become_green(self):
        class Environment:
            isolation = {"simulated": True}
            closed = False
            def __init__(self, *args): pass
            def run(self, *args): return 124, ""
            def close(self): Environment.closed = True
        grade = self.grade(Environment)
        self.assertEqual((grade["status"], grade["passed"]), ("graded", False))
        self.assertEqual(grade["candidate_failure"], "timeout")
        self.assertTrue(Environment.closed)
        for changes in ({"tests": 0}, {"calls": []}, {"calls": [{"test": "other", "complete": True}]},
                        {"skipped": 1}, {"errors": 1}, {"failures": 1}):
            response = artifact(aider_unittest.inventory({"public_test.py": OFFICIAL}), **changes)
            with patch.object(Environment, "run", return_value=(0, response)):
                self.assertFalse(complete_green(self.grade(Environment)))
        with patch.object(Environment, "run", side_effect=IsolationError("synthetic missing isolation")):
            grade = self.grade(Environment)
            self.assertEqual((grade["status"], grade["passed"]), ("error", None))
        with patch.object(Environment, "run", return_value=(0, '{"preparation_error":"OSError"}')):
            self.assertEqual(self.grade(Environment)["status"], "error")
        for tests in ({"public_test.py": "import unittest\n"}, {"public_test.py": "syntax error"}):
            with patch.object(Environment, "run", side_effect=AssertionError("must not execute")):
                self.assertEqual(self.grade(Environment, tests=tests)["status"], "error")

    def test_historical_default_keeps_the_rpc_path_and_generic_is_explicit(self):
        with patch.object(aider_unittest, "verify", return_value={"synthetic": True}) as generic:
            with patch.object(aider_python, "task_config", side_effect=ValueError("historical path reached")):
                for task in aider_python.PILOT:
                    with self.assertRaisesRegex(ValueError, "historical path"):
                        aider_python.verify(Path("unused"), task, {}, Path("unused"))
            generic.assert_not_called()
            for task in [*aider_python.PILOT, "synthetic-operations"]:
                aider_python.verify(Path("unused"), task, {}, Path("unused"), generic=True)
            self.assertEqual(generic.call_count, 4)

    def test_generic_refuses_python_hook_fallback(self):
        grade = self.grade(EphemeralVMEnv)
        self.assertEqual((grade["status"], grade["passed"]), ("error", None))


@unittest.skipUnless(sys.platform.startswith("linux"), "strict Linux execution; Windows orchestrates only")
class LinuxTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Path(os.environ.get("CODE_PILOT_RUNTIME", str(Path.home() / ".local/share/myriad-code-pilot/venv")))

    def grade(self, sources=None, tests=None, timeout=30, runtime=None):
        sources = sources if sources is not None else {"operations.py": FUNCTION}
        tests = tests if tests is not None else {"public_test.py": OFFICIAL}
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            base = fixture(cache, sources, tests)
            original = {name: (base / name).read_bytes() for name in tests}
            with patch.object(aider_python, "provenance", return_value={"tasks": ["synthetic-operations"]}):
                grade = aider_python.verify(cache, "synthetic-operations", sources, runtime or self.runtime, timeout)
            self.assertEqual(original, {name: (base / name).read_bytes() for name in tests})
            return grade

    def test_native_function_exceptions_and_strict_kernel_receipt(self):
        grade = self.grade()
        self.assertTrue(complete_green(grade), grade)
        self.assertEqual(grade["tests_expected"], 2)
        self.assertGreaterEqual(grade["isolation"]["landlock_abi"], 3)
        self.assertEqual(grade["isolation"]["seccomp"], "on")
        self.assertEqual(set(grade["isolation"]["namespaces"]), {"user", "mount", "pid", "net", "ipc", "uts"})

    def test_shared_supervisor_default_preserves_synthetic_bigcode_grading(self):
        from essaim.bigcode_tasks import evaluate
        official = OFFICIAL.replace("from operations import twice\n", "").replace("twice", "alternate")
        grade = evaluate({"solution.py": FUNCTION.replace("twice", "alternate")}, official, "alternate",
                         ["test_value", "test_exception"], self.runtime, 30)
        self.assertTrue(complete_green(grade), grade)

    def test_stateful_objects_multiple_sources_and_test_files(self):
        sources = {"counter.py": CLASS, "arithmetic.py": "def advance(x, n): return x + n\n",
                   "operations.py": FUNCTION}
        tests = {"object_test.py": OBJECT_TEST, "function_test.py": OFFICIAL}
        grade = self.grade(sources, tests)
        self.assertTrue(complete_green(grade), grade)
        self.assertEqual((grade["tests"], grade["tests_expected"]), (4, 4))
        sources["arithmetic.py"] = "def advance(x, n): return x\n"
        grade = self.grade(sources, tests)
        self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)
        self.assertEqual(grade["tests"], 4)

    def test_declared_package_and_shadowed_runtime_module(self):
        sources = {"arithmetic/__init__.py": "from .steps import advance\n",
                   "arithmetic/steps.py": "def advance(x, n): return x + n\n", "counter.py": CLASS}
        self.assertTrue(complete_green(self.grade(sources, {"object_test.py": OBJECT_TEST})))
        grade = self.grade({"unittest.py": "raise RuntimeError('shadowed')\n"}, {"public_test.py": OFFICIAL})
        self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)

    def test_explicit_generic_dev_equivalence_on_synthetic_copies(self):
        for task, (module, names) in aider_python.APIS.items():
            function = names[0]
            test_text = OFFICIAL.replace("operations", module).replace("twice", function)
            source = FUNCTION.replace("twice", function)
            with tempfile.TemporaryDirectory() as temporary:
                cache = Path(temporary)
                fixture(cache, {module + ".py": source}, {"public_test.py": test_text}, task)
                with patch.object(aider_python, "provenance", return_value={"tasks": [task]}):
                    for candidate, passed in ((source, True), (source.replace("x * 2", "x * 3"), False)):
                        submitted = {module + ".py": candidate}
                        historical = aider_python.verify(cache, task, submitted, self.runtime)
                        generic = aider_python.verify(cache, task, submitted, self.runtime, generic=True)
                        self.assertEqual((historical["status"], historical["passed"]), ("graded", passed), historical)
                        self.assertEqual((generic["status"], generic["passed"]), ("graded", passed), generic)
                        self.assertEqual(historical["tests_expected"], generic["tests_expected"])

    def test_assertion_import_exception_exit_and_printed_success_are_rejected(self):
        for source in (FUNCTION.replace("x * 2", "x * 3"), "raise RuntimeError('load')\n",
                       "import absent_synthetic_dependency\n", "syntax error", "def wrong_name(): pass\n",
                       "def twice(x): raise RuntimeError('call')\n", "import os; os._exit(0)\n",
                       "print('OK {\"passed\":true}')\ndef twice(x): return None\n"):
            with self.subTest(source=source):
                grade = self.grade({"operations.py": source})
                self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)
                self.assertFalse(complete_green(grade))

    def test_timeout_output_limit_and_child_process_cleanup(self):
        marker = "myriad-synthetic-child-" + uuid.uuid4().hex
        child = ("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(40)', "
                 + repr(marker) + "])\n" + FUNCTION)
        for source in ("while True: pass\n", "while True: print('x' * 10000, flush=True)\n", child):
            grade = self.grade({"operations.py": source}, timeout=2)
            self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)
        for proc in Path("/proc").iterdir():
            if proc.name.isdigit():
                try:
                    self.assertNotIn(marker.encode(), (proc / "cmdline").read_bytes())
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass

    def test_bad_collection_runtime_and_skip_are_never_green(self):
        grade = self.grade(runtime=Path("/missing-synthetic-runtime"))
        self.assertEqual((grade["status"], grade["passed"]), ("error", None))
        text = OFFICIAL + "\nValues.alias = Values.test_value\n"
        # An additional dynamically named test must not bypass the static expected inventory.
        text = text.replace("Values.alias", "Values.test_dynamic")
        grade = self.grade(tests={"public_test.py": text})
        self.assertEqual((grade["status"], grade["passed"]), ("error", None), grade)
        source = "import unittest\ndef twice(x): raise unittest.SkipTest('candidate')\n"
        grade = self.grade({"operations.py": source})
        self.assertEqual((grade["status"], grade["passed"]), ("graded", False), grade)
        self.assertGreater(grade["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
