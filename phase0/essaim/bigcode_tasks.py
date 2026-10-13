"""Pinned E17 DEV data and standard isolated official unittest evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import tempfile
import time
from pathlib import Path

from essaim import bigcode_unittest
from essaim.aider_python import SourceArtifactError, collect_sources
from essaim.local_linux import IsolationError, LocalLinuxEnv, VERSION as SANDBOX

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "bigcode_pilot"
PILOT = ["BigCodeBench_120", "BigCodeBench_184"]
VERSION = "bigcode-dev-unittest-v1"
TRUST_SCOPE = "public-benchmark-nonadversarial"

def digest(raw: bytes) -> dict:
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def provenance() -> dict:
    data = strict_json((RESOURCE / "tasks.json").read_text(encoding="utf-8"))
    if (data["version"] != VERSION or data["dev"] != PILOT or set(data["tasks"]) != set(PILOT)
            or set(data["files"]) != {t + ".json" for t in PILOT}
            or data["trust_scope"] != TRUST_SCOPE or data["adversarial_attestation"] is not False):
        raise ValueError("DEV selection/protocol mismatch")
    for name, info in data["resources"].items():
        if digest((RESOURCE / name).read_bytes()) != info:
            raise ValueError("trusted resource provenance mismatch: " + name)
    return data


def task_data(cache: Path, task: str) -> dict:
    if task not in PILOT:
        raise ValueError("task outside authorized DEV")
    manifest = provenance()
    raw = (cache / (task + ".json")).read_bytes()
    if digest(raw) != manifest["files"][task + ".json"]:
        raise ValueError("task data provenance mismatch")
    return strict_json(raw.decode("utf-8"))


def check_data(cache: Path) -> None:
    for task in PILOT:
        task_data(cache, task)


def prepare(directory: Path, source: Path) -> None:
    """Copy exactly the authorized DEV rows from an existing cache, without network or parquet access."""
    check_data(source)
    directory.mkdir(parents=True, exist_ok=False)
    for task in PILOT:
        (directory / (task + ".json")).write_bytes((source / (task + ".json")).read_bytes())


def trusted_test(task: str) -> str:
    if task not in PILOT:
        raise ValueError("task outside authorized DEV")
    provenance()
    return strict_json((RESOURCE / "official_tests.json").read_text(encoding="utf-8"))[task]


def instructions(cache: Path, task: str) -> str:
    row = task_data(cache, task)
    return (row["instruct_prompt"] + "\n\nWrite the complete self-contained implementation to solution.py. "
            "Only solution.py is submitted. Official tests and reference are hidden. "
            "The controller evaluates a fresh source snapshot with the original official unittest assertions. "
            "Use MYRIAD_RUN_OFFICIAL_TESTS to test before submission.")


def check_runtime() -> None:
    import importlib.metadata
    data = json.loads((RESOURCE / "runtime.json").read_text(encoding="utf-8"))
    if platform.python_version() != data["python"] or digest(Path(sys.executable).read_bytes()) != data["binary"]:
        raise IsolationError("dedicated pinned interpreter required")
    for name, version in data["packages"].items():
        if importlib.metadata.version(name) != version:
            raise IsolationError("runtime package version mismatch: " + name)
    if digest((RESOURCE / "uv.lock").read_bytes()) != data["lock"]:
        raise IsolationError("runtime lock mismatch")
    for name, entry in data["licenses"].items():
        if {p: digest(raw) for p, raw in license_files(name).items()} != entry["files"]:
            raise IsolationError("runtime licence provenance mismatch: " + name)
    for entry in data["bundled_components"]:
        distribution = importlib.metadata.distribution(entry["package"])
        paths = [distribution.locate_file(p) for p in distribution.files or [] if p.name == entry["file"]]
        if (len(paths) != 1 or digest(paths[0].read_bytes()) != {k: entry[k] for k in ("size", "sha256")}):
            raise IsolationError("runtime bundled component provenance mismatch")


def license_files(name: str) -> dict[str, bytes]:
    import importlib.metadata
    distribution = importlib.metadata.distribution(name)
    result = {}
    for path in distribution.files or []:
        relative = str(path)
        if ".dist-info/" in relative and ("license" in relative.lower() or "copying" in relative.lower()):
            source = distribution.locate_file(path)
            if source.is_file():
                result[name + "/" + relative.split(".dist-info/", 1)[1]] = source.read_bytes()
    return result


def copy_task(cache: Path, task: str, work: Path, sources: dict[str, str] | None = None) -> list[str]:
    row = task_data(cache, task)
    if sources is not None and set(sources) != {"solution.py"}:
        raise SourceArtifactError("only solution.py may be submitted")
    content = row["code_prompt"] + "    raise NotImplementedError\n" if sources is None else sources["solution.py"]
    if type(content) is not str or len(content.encode("utf-8")) > 1 << 20:
        raise SourceArtifactError("invalid source size/type")
    (work / "solution.py").write_text(content, encoding="utf-8")
    return ["solution.py"]


def strict_json(raw: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def result_artifact(raw: str, expected: list[str]) -> dict:
    outer = strict_json(raw)
    if (type(outer) is dict and set(outer) == {"preparation_error"}
            and type(outer["preparation_error"]) is str):
        raise IsolationError("trusted supervisor preparation failed: " + outer["preparation_error"])
    if (type(outer) is not dict or set(outer) != {"worker_exit", "limited", "response", "diagnostics"}
            or type(outer["worker_exit"]) is not int or type(outer["limited"]) is not bool
            or type(outer["response"]) is not str or type(outer["diagnostics"]) is not str):
        raise ValueError("invalid supervisor artifact")
    if outer["limited"] or outer["worker_exit"] != 0:
        raise ValueError("output_limit" if outer["limited"] else "premature_worker_exit")
    data = strict_json(outer["response"])
    base = {"protocol", "phase", "candidate_failure"}
    if (type(data) is not dict or data.get("protocol") != bigcode_unittest.PROTOCOL
            or data.get("phase") not in ("load", "collection", "tests")):
        raise ValueError("invalid worker protocol")
    if set(data) == base and type(data["candidate_failure"]) is str:
        raise ValueError("candidate_" + data["candidate_failure"])
    if set(data) == base | {"collection_error"} and type(data["collection_error"]) is str:
        raise IsolationError("trusted test collection failed: " + data["collection_error"])
    counts = ("tests", "failures", "errors", "skipped", "expected_failures", "unexpected_successes")
    if (set(data) != base | set(counts) | {"calls", "feedback"} or data["phase"] != "tests"
            or data["candidate_failure"] is not None
            or any(type(data[k]) is not int or not 0 <= data[k] <= 10000 for k in counts)
            or type(data["calls"]) is not list or type(data["feedback"]) is not list):
        raise ValueError("invalid test result schema")
    if any(type(c) is not dict or set(c) != {"test", "complete"} or type(c["test"]) is not str
           or type(c["complete"]) is not bool for c in data["calls"]):
        raise ValueError("invalid test completion records")
    if (data["tests"] != len(expected) or len(data["calls"]) != len(expected)
            or sorted(c["test"] for c in data["calls"]) != sorted(expected)
            or any(not c["complete"] for c in data["calls"])):
        raise ValueError("incomplete official tests")
    if any(type(f) is not dict or set(f) != {"test", "traceback"} or type(f["test"]) is not str
           or type(f["traceback"]) is not str or len(f["traceback"]) > 4000 for f in data["feedback"]):
        raise ValueError("invalid test feedback")
    data["passed"] = all(data[k] == 0 for k in counts if k != "tests")
    return data


def evaluate(sources: dict[str, str], official: str, entry_point: str, expected: list[str], runtime: Path,
             timeout_s: float, *, environment=LocalLinuxEnv) -> dict:
    """Use this seam only for hermetic synthetic fixtures or provenance-checked authorized DEV."""
    started = time.monotonic()
    grade = {"status": "error", "passed": None, "tests": 0, "tests_expected": len(expected),
             "tests_complete": False, "failures": 0, "errors": 0, "broken": [], "calls": [], "feedback": [],
             "trust_scope": TRUST_SCOPE, "adversarial_attestation": False, "sandbox_backend": "strict",
             "sandbox": SANDBOX, "isolation": None, "scorable": False}
    env = None

    def failed(reason):
        grade.update(status="graded", passed=False, scorable=True, candidate_failure=reason)
        grade["calls"] = [{"test": "worker", "complete": True, "candidate_failure": reason}]

    try:
        if not math.isfinite(timeout_s) or timeout_s < 0 or not expected or len(set(expected)) != len(expected):
            raise ValueError("invalid verification budget/collection")
        if timeout_s == 0:
            failed("verification_budget")
            return grade
        if (set(sources) != {"solution.py"} or type(sources["solution.py"]) is not str
                or len(sources["solution.py"].encode()) > 1 << 20):
            failed("invalid_source_artifact")
            return grade
        with tempfile.TemporaryDirectory(prefix="myriad-bigcode-unittest-") as temporary:
            work = Path(temporary)
            (work / "solution.py").write_text(sources["solution.py"], encoding="utf-8")
            (work / "_official.py").write_text(official, encoding="utf-8")
            (work / "_runner.py").write_bytes(Path(bigcode_unittest.__file__).read_bytes())
            env = environment(work, runtime)
            cfg = json.dumps({"entry_point": entry_point, "tests_expected": len(expected)})
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                failed("verification_budget")
                return grade
            code, output = env.run("python _runner.py <<'MYRIAD_INPUT'\n" + cfg + "\nMYRIAD_INPUT", remaining)
            grade["isolation"] = env.isolation
            if code:
                failed("timeout" if code == 124 else "output_limit" if code == 125 else "nonzero_exit")
            else:
                try:
                    data = result_artifact(output, expected)
                except ValueError as error:
                    failed(str(error)[:200])
                else:
                    grade.update(data, status="graded", scorable=True, tests_complete=True)
            if time.monotonic() - started >= timeout_s:
                failed("verification_budget")
    except Exception as error:
        grade.update(status="error", passed=None, scorable=False, broken=[type(error).__name__])
    finally:
        if env is not None:
            env.close()
        grade["seconds"] = time.monotonic() - started
    return grade


def verify(cache: Path, task: str, sources: dict[str, str], runtime: Path, timeout_s: float = 120) -> dict:
    row = task_data(cache, task)
    cfg = provenance()["tasks"][task]
    result = evaluate(sources, trusted_test(task), row["entry_point"], cfg["tests"], runtime, timeout_s)
    result["assertion_sites"] = cfg["assertion_sites"]
    return result


def smoke(cache: Path, runtime: Path) -> dict:
    check_runtime()
    rows = []
    for task in PILOT:
        row = task_data(cache, task)
        reference = {"solution.py": row["code_prompt"] + row["canonical_solution"]}
        positive = verify(cache, task, reference, runtime)
        negatives = {name: verify(cache, task, {"solution.py": source}, runtime) for name, source in {
            "wrong": "def task_func(*a, **k): return None\n",
            "exit0": "import os; os._exit(0)\n",
            "load_exception": "raise ValueError('load failure')\n"}.items()}
        if positive["status"] != "graded" or positive["passed"] is not True or not positive["tests_complete"]:
            raise IsolationError("official reference failed without replacement: " + task)
        if any(g["status"] != "graded" or g["passed"] is not False for g in negatives.values()):
            raise IsolationError("negative control failed: " + task)
        rows.append({"task": task, "reference": positive, "negative": negatives})
    return {"kind": "official_unittest_dev_smoke", "model_measurement": False,
            "trust_scope": TRUST_SCOPE, "adversarial_attestation": False, "tasks": rows}
