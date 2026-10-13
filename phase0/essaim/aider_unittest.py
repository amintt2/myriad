"""Generic public Aider unittest evaluation, without adversarial Python attestation."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

TRUST_SCOPE = "public-benchmark-nonadversarial"
RESERVED = {"_myriad_runner.py", "_myriad_transport.py"}


def inventory(tests: dict[str, str]) -> list[str]:
    """Count ordinary static unittest classes without importing candidate code in the controller."""
    expected = []
    for index, (relative, text) in enumerate(tests.items()):
        tree = ast.parse(text, filename=relative)
        modules, bases, classes = {"unittest"}, set(), {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.update(n.asname or n.name for n in node.names if n.name == "unittest")
            elif isinstance(node, ast.ImportFrom) and node.module == "unittest":
                bases.update(n.asname or n.name for n in node.names if n.name == "TestCase")
            elif isinstance(node, ast.ClassDef):
                inherited, recognized = set(), False
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in classes:
                        inherited.update(classes[base.id])
                        recognized = True
                    elif (isinstance(base, ast.Name) and base.id in bases
                          or isinstance(base, ast.Attribute) and base.attr == "TestCase"
                          and isinstance(base.value, ast.Name) and base.value.id in modules):
                        recognized = True
                if recognized:
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if member.name.startswith("test"):
                                inherited.add(member.name)
                    classes[node.name] = inherited
        for name, methods in classes.items():
            expected.extend(f"official_{index}.{name}.{method}" for method in sorted(methods))
    if not expected or len(expected) > 10000 or len(set(expected)) != len(expected):
        raise ValueError("nonempty bounded static unittest collection required")
    return expected


def worker(raw: str, fd: int) -> None:
    """Keep native imports, objects and state inside one isolated unittest process."""
    from _myriad_transport import PROTOCOL
    cfg = json.loads(raw)
    records, diagnostics = [], sys.stdout

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            records.append({"test": test.id(), "complete": False})
            super().startTest(test)

        def stopTest(self, test):
            records[-1]["complete"] = True
            super().stopTest(test)

    out = {"protocol": PROTOCOL, "phase": "load", "candidate_failure": None}
    try:
        try:
            for name, relative in cfg["modules"].items():
                module = importlib.import_module(name)
                if Path(getattr(module, "__file__", "")).resolve() != Path(relative).resolve():
                    raise ImportError("declared source module was shadowed")
        except BaseException as error:
            out["candidate_failure"] = type(error).__name__
        else:
            out["phase"] = "collection"
            try:
                suite = unittest.TestSuite()
                for index, relative in enumerate(cfg["tests"]):
                    module = types.ModuleType(f"official_{index}")
                    module.__file__ = str(Path(relative).resolve())
                    sys.modules[module.__name__] = module
                    text = Path(relative).read_bytes()
                    exec(compile(text, relative, "exec"), module.__dict__)
                    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
                def ids(tests):
                    for test in tests:
                        if isinstance(test, unittest.TestSuite):
                            yield from ids(test)
                        else:
                            yield test.id()
                if sorted(ids(suite)) != sorted(cfg["expected"]):
                    raise ValueError("official test collection mismatch")
                out["phase"] = "tests"
                result = unittest.TextTestRunner(stream=diagnostics, failfast=False, resultclass=Result).run(suite)
                out.update(tests=result.testsRun, calls=records, failures=len(result.failures),
                           errors=len(result.errors), skipped=len(result.skipped),
                           expected_failures=len(result.expectedFailures),
                           unexpected_successes=len(result.unexpectedSuccesses),
                           feedback=[{"test": t.id(), "traceback": detail[-4000:]}
                                     for t, detail in result.failures + result.errors])
            except BaseException as error:
                # Missing exported names from a declared solution are candidate failures at collection.
                candidate_import = isinstance(error, ImportError) and getattr(error, "name", None) in cfg["modules"]
                key = "candidate_failure" if out["phase"] == "tests" or candidate_import else "collection_error"
                out[key] = type(error).__name__
    finally:
        diagnostics.flush()
    payload = json.dumps(out, allow_nan=False).encode()
    while payload:
        payload = payload[os.write(fd, payload):]
    os._exit(0)


def verify(cache: Path, task: str, sources: dict[str, str], runtime: Path, timeout_s: float,
           *, environment=None) -> dict:
    from essaim import bigcode_tasks, bigcode_unittest
    from essaim.aider_python import SourceArtifactError, collect_sources, task_config
    from essaim.ephemeral_vm import EphemeralVMEnv
    from essaim.local_linux import LocalLinuxEnv
    from essaim.terminal_bench import task_path
    environment = environment or LocalLinuxEnv
    started, env = time.monotonic(), None
    grade = {"status": "error", "passed": None, "tests": 0, "tests_expected": 0, "tests_complete": False,
             "fail_fast": False, "failures": 0, "errors": 0, "broken": [], "calls": [], "feedback": [],
             "trust_scope": TRUST_SCOPE, "adversarial_attestation": False, "sandbox_backend": "strict",
             "sandbox": LocalLinuxEnv.version, "isolation": None, "source_hashes": {}, "test_hashes": {}}

    def failed(reason):
        grade.update(status="graded", passed=False, candidate_failure=reason)
        grade["calls"] = [{"test": "worker", "complete": True, "candidate_failure": reason}]

    try:
        if (not math.isfinite(timeout_s) or timeout_s < 0
                or isinstance(environment, type) and issubclass(environment, EphemeralVMEnv)):
            raise ValueError("generic verification requires strict isolation and a finite budget")
        base, files = task_config(cache, task)
        solutions, test_files = files["solution"], files["test"]
        declared = [*solutions, *test_files]
        if (not solutions or not test_files or len(set(declared)) != len(declared)
                or any(name in RESERVED or not name.endswith(".py") for name in declared)):
            raise ValueError("invalid declared Python source/test files")
        collect_sources(base, test_files)
        test_bytes = {name: task_path(base, name).read_bytes() for name in test_files}
        tests = {name: raw.decode("utf-8") for name, raw in test_bytes.items()}
        expected = inventory(tests)
        grade["tests_expected"] = len(expected)
        grade["test_hashes"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in test_bytes.items()}
        if (set(sources) != set(solutions) or any(type(text) is not str or len(text.encode()) > 1 << 20
                                                for text in sources.values())):
            failed("invalid_source_artifact")
            return grade
        grade["source_hashes"] = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
        modules = {}
        for name in solutions:
            path = Path(name)
            parts = path.with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts or any(not part.isidentifier() for part in parts):
                raise ValueError("declared solutions must be importable Python modules")
            module = ".".join(parts)
            if module in modules:
                raise ValueError("duplicate declared module")
            modules[module] = name
        with tempfile.TemporaryDirectory(prefix="myriad-aider-unittest-") as temporary:
            work = Path(temporary)
            for name, raw in {**{name: text.encode() for name, text in sources.items()}, **test_bytes}.items():
                path = task_path(work, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            # Read back the exact copied source snapshot before executing any candidate.
            collect_sources(work, solutions)
            if (any(task_path(work, name).read_bytes() != text.encode() for name, text in sources.items())
                    or any(task_path(work, name).read_bytes() != raw for name, raw in test_bytes.items())):
                raise SourceArtifactError("staged snapshot mismatch")
            (work / "_myriad_runner.py").write_bytes(Path(__file__).read_bytes())
            (work / "_myriad_transport.py").write_bytes(Path(bigcode_unittest.__file__).read_bytes())
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                failed("verification_budget")
                return grade
            cfg = json.dumps({"modules": {name: modules[name] for name in sorted(modules, key=lambda n: n.count("."))},
                              "tests": test_files, "expected": expected})
            env = environment(work, runtime)
            code, output = env.run("python _myriad_runner.py <<'MYRIAD_INPUT'\n" + cfg + "\nMYRIAD_INPUT", remaining)
            grade["isolation"] = env.isolation
            if code:
                failed("timeout" if code == 124 else "output_limit" if code == 125 else "nonzero_exit")
            else:
                try:
                    data = bigcode_tasks.result_artifact(output, expected)
                except ValueError as error:
                    failed(str(error)[:200])
                else:
                    grade.update(data, status="graded", tests_complete=True)
            if time.monotonic() - started >= timeout_s:
                failed("verification_budget")
    except Exception as error:
        grade.update(status="error", passed=None, broken=[type(error).__name__])
    finally:
        if env is not None:
            env.close()
        grade["seconds"] = time.monotonic() - started
    return grade


if __name__ == "__main__":
    from _myriad_transport import CAP, supervise
    incoming = sys.stdin.read(CAP + 1)
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(incoming, int(sys.argv[2]))
    else:
        print(json.dumps(supervise(incoming, worker_path=__file__), allow_nan=False))
