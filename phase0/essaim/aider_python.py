"""Public pinned Aider Python tasks, with a separate trusted verifier for the three-task pilot."""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import math
import sys
import tarfile
import tempfile
import time
import types
import unittest
import urllib.request
from pathlib import Path

from essaim.local_linux import IsolationError, LocalLinuxEnv
from essaim.terminal_bench import check_tasks, task_path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "code_pilot/tasks.json"
PILOT = ["affine-cipher", "beer-song", "book-store"]
APIS = {"affine-cipher": ("affine_cipher", ["encode", "decode"]), "beer-song": ("beer_song", ["recite"]),
        "book-store": ("book_store", ["total"])}


class CandidateFailure(RuntimeError):
    """Candidate execution/transport failed after the kernel boundary was established."""


class SourceArtifactError(ValueError):
    """A required source artifact is missing, unsafe, oversized or not UTF-8."""


def provenance() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["pilot"] != PILOT or len(manifest["tasks"]) != 34 or sorted(manifest["tasks"])[:3] != PILOT:
        raise ValueError("pilot selection differs from the predeclared lexical selection")
    return manifest


def prepare(directory: Path, manifest: dict) -> None:
    """No archive paths are extracted; only manifest-listed regular Python files are copied."""
    with urllib.request.urlopen(manifest["archive_url"], timeout=60) as response:
        archive = response.read(32 << 20)
    if hashlib.sha256(archive).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            relative = "/".join(member.name.split("/")[1:])
            if relative not in manifest["files"]:
                continue
            if not member.isfile():
                raise ValueError("nonregular archive entry")
            data = tar.extractfile(member).read()
            info = manifest["files"][relative]
            if len(data) != info["size"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError("archive member hash mismatch")
            path = task_path(directory, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    with urllib.request.urlopen(manifest["license_url"], timeout=30) as response:
        license_data = response.read(65536)
    info = manifest["files"]["LICENSE.python"]
    if len(license_data) != info["size"] or hashlib.sha256(license_data).hexdigest() != info["sha256"]:
        raise ValueError("license hash mismatch")
    task_path(directory, "LICENSE.python").write_bytes(license_data)
    check_tasks(directory, manifest)


def task_config(cache: Path, task: str) -> tuple[Path, dict]:
    if task not in provenance()["tasks"]:
        raise ValueError("unknown task")
    base = cache / "python/exercises/practice" / task
    cfg = json.loads((base / ".meta/config.json").read_text(encoding="utf-8"))
    return base, cfg["files"]


def copy_task(cache: Path, task: str, work: Path, sources: dict[str, str] | None = None) -> list[str]:
    base, files = task_config(cache, task)
    allowed = files["solution"]
    for relative in [*allowed, *files["test"]]:
        dst = task_path(work, relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((base / relative).read_bytes())
    docs = base / ".docs"
    for src in sorted(docs.glob("instructions*.md")):
        dst = work / src.name
        dst.write_bytes(src.read_bytes())
    for name in ("README.md", "LICENSE.python"):
        (work / ("ATTRIBUTION.md" if name == "README.md" else "LICENSE")).write_bytes((cache / name).read_bytes())
    if sources is not None:
        if set(sources) != set(allowed):
            raise ValueError("only all declared source files may be reapplied")
        for relative, content in sources.items():
            task_path(work, relative).write_text(content, encoding="utf-8")
    return allowed


def collect_sources(work: Path, allowed: list[str]) -> dict[str, str]:
    result = {}
    for relative in allowed:
        try:
            path = task_path(work, relative)
            if not path.is_file() or path.stat().st_size > 1 << 20 or path.stat().st_nlink != 1:
                raise ValueError("source must be a bounded, regular, nonlinked file")
            result[relative] = path.read_text(encoding="utf-8")
        except (ValueError, FileNotFoundError, IsADirectoryError, PermissionError) as error:
            raise SourceArtifactError("invalid required source artifact") from error
    return result


CALL_WORKER = '''import importlib, json, sys
write = open
serialize = json.dumps
cfg = json.loads(sys.stdin.read())
try:
    fn = getattr(importlib.import_module(cfg["module"]), cfg["function"])
    value = fn(*cfg["args"], **cfg["kwargs"])
    if type(value) not in (str, int, float, list, type(None)):
        raise TypeError("unsupported pilot return type")
    out = {"value": value}
except BaseException as error:
    out = {"exception": type(error).__name__, "args": list(error.args)}
with write("_return.json", "x", encoding="utf-8") as result:
    result.write(serialize(out, allow_nan=False))
'''


def verify(cache: Path, task: str, sources: dict[str, str], runtime: Path, timeout_s: float = 120) -> dict:
    """Run original tests in this trusted process, with RPC stand-ins for the candidate functions.

    Only inputs reach candidate subprocesses. Expected values, unittest and the result object stay here.
    This adapter supports the pilot's pure function API only; it deliberately refuses other exercises.
    """
    started = time.monotonic()
    if task not in APIS:
        raise ValueError("verification adapter covers only the preselected pilot")
    module_name, names = APIS[task]
    base, files = task_config(cache, task)
    broken = []
    proxy = types.ModuleType(module_name)
    calls = []
    current_test = {"id": None}

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            current_test["id"] = test.id()
            super().startTest(test)

    def failed(info, reason):
        info.update(complete=True, candidate_failure=reason)
        raise CandidateFailure(reason)

    def call(function, args, kwargs):
        call_info = {"function": function, "test": current_test["id"], "complete": False,
                     "seconds": 0.0, "isolation": None}
        calls.append(call_info)
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            failed(call_info, "verification_budget")
        with tempfile.TemporaryDirectory(prefix="myriad-verify-") as temporary:
            work = Path(temporary)
            # Fresh source-only copy: the candidate cannot read tests, answers or any prior invocation.
            if set(sources) != set(files["solution"]):
                raise ValueError("invalid source selection")
            for relative, content in sources.items():
                path = task_path(work, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            (work / "_call.py").write_text(CALL_WORKER, encoding="utf-8")
            cfg = json.dumps({"module": module_name, "function": function, "args": args, "kwargs": kwargs})
            request_start = time.monotonic()
            env = None
            try:
                env = LocalLinuxEnv(work, runtime)
                code, output = env.run("python _call.py <<'MYRIAD_INPUT'\n" + cfg + "\nMYRIAD_INPUT", min(10, remaining))
                call_info["diagnostics"] = output
                if code != 0:
                    failed(call_info, "timeout" if code == 124 else "output_limit" if code == 125 else "nonzero_exit")
                try:
                    returned = collect_sources(work, ["_return.json"])["_return.json"]
                except SourceArtifactError:
                    failed(call_info, "invalid_result_artifact")
                try:
                    data = json.loads(returned)
                except ValueError:
                    failed(call_info, "invalid_json")
                if not isinstance(data, dict) or set(data) not in ({"value"}, {"exception", "args"}):
                    failed(call_info, "invalid_response")
                if "exception" in data and (not isinstance(data["exception"], str) or not isinstance(data["args"], list)):
                    failed(call_info, "invalid_exception")
                if "value" in data and type(data["value"]) not in (str, int, float, list, type(None)):
                    failed(call_info, "invalid_value")
                if type(data.get("value")) is float and not math.isfinite(data["value"]):
                    failed(call_info, "nonfinite_value")
            except CandidateFailure:
                raise
            except Exception as error:
                broken.append(type(error).__name__)
                raise IsolationError("candidate response incomplete") from error
            finally:
                if env is not None:
                    env.close()
                call_info.update(seconds=time.monotonic() - request_start, isolation=getattr(env, "isolation", None))
            call_info["complete"] = True
            if "exception" in data:
                call_info["program_exception"] = data["exception"]
                kind = getattr(builtins, data["exception"], None)
                if not isinstance(kind, type) or not issubclass(kind, Exception) or not isinstance(data["args"], list):
                    raise RuntimeError("candidate exception")
                raise kind(*data["args"])
            return data["value"]

    for name in names:
        setattr(proxy, name, lambda *args, _name=name, **kwargs: call(_name, args, kwargs))
    old = sys.modules.get(module_name)
    sys.modules[module_name] = proxy
    try:
        suite = unittest.TestSuite()
        for relative in files["test"]:
            module = types.ModuleType("official_" + Path(relative).stem)
            exec(compile((base / relative).read_bytes(), relative, "exec"), module.__dict__)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        expected = suite.countTestCases()
        result = unittest.TextTestRunner(stream=io.StringIO(), failfast=True, resultclass=Result).run(suite)
        infra_errors = any(not any(c["test"] == test.id() and ("candidate_failure" in c or "program_exception" in c)
                                  for c in calls) for test, _ in result.errors)
        valid = expected > 0 and result.testsRun > 0 and not broken and not result.skipped and not infra_errors
        complete = result.testsRun == expected
        status = "graded" if valid and all(c["complete"] for c in calls) else "error"
        return {"status": status, "passed": result.wasSuccessful() if status == "graded" else None,
                "tests": result.testsRun, "tests_expected": expected, "tests_complete": complete, "fail_fast": True,
                "failures": len(result.failures), "errors": len(result.errors),
                "calls": calls, "seconds": time.monotonic() - started, "broken": broken}
    finally:
        if old is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old


def smoke(cache: Path, runtime: Path) -> dict:
    check_tasks(cache, provenance())
    rows = []
    for task in PILOT:
        base, files = task_config(cache, task)
        if len(files["solution"]) != 1 or len(files["example"]) != 1:
            raise ValueError("smoke expects one official reference file")
        sources = {files["solution"][0]: (base / files["example"][0]).read_text(encoding="utf-8")}
        positive = verify(cache, task, sources, runtime)
        module, names = APIS[task]
        incorrect = "\n".join(f"def {name}(*args, **kwargs): return None" for name in names)
        negative = verify(cache, task, {files["solution"][0]: incorrect}, runtime)
        if positive["status"] != "graded" or positive["passed"] is not True:
            raise IsolationError(f"official reference failed for {task}: {positive}")
        if negative["status"] != "graded" or negative["passed"] is not False:
            raise IsolationError(f"incorrect solution was not rejected for {task}")
        rows.append({"task": task, "reference": positive, "incorrect": negative})
    return {"kind": "deterministic_harness_smoke", "tasks": rows, "model_measurement": False}
