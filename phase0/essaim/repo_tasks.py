"""Pinned real SymPy repositories and string-only RPC grading; assertions never enter the sandbox."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from essaim.aider_python import SourceArtifactError, collect_sources as collect_files
from essaim.local_linux import IsolationError, LocalLinuxEnv
from essaim.terminal_bench import check_tasks, task_path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ["sympy__sympy-11400", "sympy__sympy-11897", "sympy__sympy-12171"]
RESOURCE = ROOT / "repo_pilot"
VERSION = "repo-pilot-v1"
VISIBLE = {PILOT[0]: ("test_ccode.py", 30), PILOT[1]: ("test_latex.py", 1),
           PILOT[2]: ("test_mathematica.py", 9)}


def check_runtime() -> None:
    runtime = provenance()["runtime"]
    if platform.python_version() != runtime["python"] or importlib.metadata.version("mpmath") != runtime["mpmath"]:
        raise IsolationError("use the pinned compatible Python and mpmath runtime")
    binary = Path(sys.executable).read_bytes()
    if hashlib.sha256(binary).hexdigest() != runtime["python_binary"]["sha256"]:
        raise IsolationError("compatible interpreter provenance mismatch")


def provenance() -> dict:
    data = json.loads((RESOURCE / "tasks.json").read_text(encoding="utf-8"))
    selection = data["selection"]
    chosen = sorted(task for task, row in selection["candidates"].items() if row["state"] == "selected")
    if selection["pilot"] != PILOT or chosen[:selection["limit"]] != PILOT or selection["population"] != 77:
        raise ValueError("selection differs from the predeclared printer pilot")
    for relative, info in data["resources"].items():
        raw = task_path(RESOURCE, relative).read_bytes()
        if len(raw) != info["size"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError("trusted fixture/licence provenance mismatch")
    return data


def download(info: dict) -> bytes:
    with urllib.request.urlopen(info["url"], timeout=60) as response:
        raw = response.read((32 << 20) + 1)
    if len(raw) != info["size"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
        raise ValueError("download provenance mismatch")
    return raw


def prepare(directory: Path, manifest: dict) -> None:
    """Copy only explicitly pinned regular entries; no tar extraction and no Git history."""
    for task in PILOT:
        cfg = manifest["tasks"][task]
        with tarfile.open(fileobj=io.BytesIO(download(cfg["archive"])), mode="r:gz") as archive:
            seen = set()
            for entry in archive:
                if entry.isdir():
                    continue
                relative = "/".join(entry.name.split("/")[1:])
                name = task + "/base/" + relative
                if relative not in cfg["base_files"] or not entry.isfile() or name in seen:
                    raise ValueError("undeclared or nonregular archive entry")
                seen.add(name)
                raw = archive.extractfile(entry).read()
                info = manifest["files"][name]
                if len(raw) != info["size"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                    raise ValueError("archive member provenance mismatch")
                path = task_path(directory, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        for kind, relative in (("reference", cfg["reference_file"]), ("final_tests", cfg["official_test_file"])):
            name = task + "/" + kind + "/" + relative
            raw = download({"url": cfg[kind + "_url"], **manifest["files"][name]})
            path = task_path(directory, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    check_tasks(directory, manifest)


def copy_task(cache: Path, task: str, work: Path, sources: dict[str, str] | None = None) -> list[str]:
    cfg = provenance()["tasks"][task]
    if sources is None:
        for relative in cfg["base_files"]:
            dst = task_path(work, relative)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(task_path(cache, task + "/base/" + relative), dst)
        (work / "_submission.json").write_text(json.dumps({"sources": cfg["sources"]}), encoding="utf-8")
    else:
        if set(sources) != set(cfg["sources"]):
            raise ValueError("all and only declared repository sources must be submitted")
        if sum(len(text.encode("utf-8")) for text in sources.values()) > 16 << 20:
            raise SourceArtifactError("aggregate source limit exceeded")
        for relative, text in sources.items():
            dst = task_path(work, relative)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
    return cfg["sources"]


def collect_sources(work: Path, allowed: list[str]) -> dict[str, str]:
    sources = collect_files(work, allowed)
    if sum(len(text.encode("utf-8")) for text in sources.values()) > 16 << 20:
        raise SourceArtifactError("aggregate source limit exceeded")
    return sources


def instructions(cache: Path, task: str) -> str:
    cfg = provenance()["tasks"][task]
    return (cfg["instruction"] + "\nHistorical repair: " + cfg["pr"] + ". Instance: " + task + ". "
            "Your working directory is the real repository at the pinned pre-repair commit. "
            "Only existing non-test Python modules listed in _submission.json are submitted. "
            "Do not alter tests. Diagnostic command: " + visible_command(task) + ". "
            "These initial tests predate the repair; a correct LaTeX fix conflicts with the old Piecewise "
            "expectation. Their output and exit code are not the final score. "
            "Final string assertions run in a separate controller; no Git history or reference patch is available.")


def visible_command(task: str) -> str:
    filename, _ = VISIBLE[task]
    command = "python bin/test --no-subprocess --no-colors --seed 0 sympy/printing/tests/" + filename
    return command + (" -k test_latex_Piecewise" if task == PILOT[1] else "")


def visible_tests(cache: Path, task: str, runtime: Path, reference: bool) -> dict:
    """Check the native runner on pinned full copies only, never grade an agent submission this way."""
    started = time.monotonic()
    cfg = provenance()["tasks"][task]
    command = visible_command(task)
    with tempfile.TemporaryDirectory(prefix="myriad-repo-visible-") as temporary:
        work = Path(temporary)
        copy_task(cache, task, work)
        if reference:
            name = cfg["reference_file"]
            task_path(work, name).write_bytes(task_path(cache, task + "/reference/" + name).read_bytes())
        env = LocalLinuxEnv(work, runtime)
        try:
            code, output = env.run(command, 30)
        finally:
            env.close()
    # Only the unchanged historical LaTeX assertion conflicts with the corrected source.
    conflict = reference and task == PILOT[1]
    expected = "0 passed, 1 failed" if conflict else str(VISIBLE[task][1]) + " passed"
    if code != int(conflict) or not re.search(r"tests finished: " + expected + r", in [\d.]+ seconds", output):
        raise IsolationError("native visible-test runner mismatch: " + task + ": " + output)
    return {"command": command, "exit_code": code, "diagnostics": output, "isolation": env.isolation,
            "historical_assertion_conflict": conflict, "final_score": False, "seconds": time.monotonic() - started}


WORKER = '''import json, sys
write = open
serialize = json.dumps
cfg = json.loads(sys.stdin.read())
try:
    from sympy import *
    from sympy.printing import ccode, latex
    from sympy.printing.mathematica import mathematica_code as mcode
    x, y, z = symbols('x y z')
    exec(cfg['code'])
    if type(value) is not str or len(value.encode('utf-8')) > 65536:
        raise TypeError('printer must return a bounded string')
    out = {'value': value}
except BaseException as error:
    out = {'exception': type(error).__name__}
with write('_return.json', 'x', encoding='utf-8') as result:
    result.write(serialize(out, allow_nan=False))
'''


def cases(task: str) -> list[dict]:
    provenance()
    return json.loads((RESOURCE / "final_cases.json").read_text(encoding="utf-8"))[task]


def verify(cache: Path, task: str, sources: dict[str, str], runtime: Path, timeout_s: float = 120) -> dict:
    started = time.monotonic()
    fixtures = cases(task)
    calls = []
    broken = []
    for index, case in enumerate(fixtures):
        remaining = timeout_s - (time.monotonic() - started)
        call = {"test": case["test"], "case": index, "complete": False, "isolation": None}
        calls.append(call)
        if remaining <= 0:
            call.update(complete=True, candidate_failure="verification_budget")
            break
        env = None
        request_start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="myriad-repo-verify-") as temporary:
                work = Path(temporary)
                copy_task(cache, task, work, sources)
                (work / "_call.py").write_text(WORKER, encoding="utf-8")
                env = LocalLinuxEnv(work, runtime)
                # Fixed trusted expression construction only; expected strings never reach this child.
                cfg = json.dumps({"code": case["code"]})
                code, output = env.run("python _call.py <<'MYRIAD_INPUT'\n" + cfg + "\nMYRIAD_INPUT",
                                       min(10, remaining))
                call["diagnostics"] = output
                if code != 0:
                    call["candidate_failure"] = "timeout" if code == 124 else "nonzero_exit"
                else:
                    try:
                        raw = collect_sources(work, ["_return.json"])["_return.json"]
                        if len(raw.encode("utf-8")) > 128 << 10:
                            raise ValueError("response exceeds limit")
                        data = json.loads(raw)
                        if (type(data) is not dict or set(data) not in ({"value"}, {"exception"})
                                or type(next(iter(data.values()))) is not str):
                            raise ValueError("invalid string-only response")
                        if len(next(iter(data.values())).encode("utf-8")) > 65536:
                            raise ValueError("returned string exceeds limit")
                    except (SourceArtifactError, ValueError, RecursionError):
                        call["candidate_failure"] = "invalid_result_artifact"
                    else:
                        call["returned"] = data
                        if data.get("value") != case["expected"]:
                            call["candidate_failure"] = "program_exception" if "exception" in data else "wrong_string"
                call["complete"] = True
        except Exception as error:
            broken.append(type(error).__name__)
            break
        finally:
            if env is not None:
                env.close()
            call.update(seconds=time.monotonic() - request_start, isolation=getattr(env, "isolation", None))
        if "candidate_failure" in call:
            break
    status = "error" if broken or not calls or not all(c["complete"] for c in calls) else "graded"
    passed = not any("candidate_failure" in c for c in calls) if status == "graded" else None
    return {"status": status, "passed": passed,
            "tests": len(calls), "tests_expected": len(fixtures), "tests_complete": len(calls) == len(fixtures),
            "test_functions": sorted({c["test"] for c in fixtures}), "unit": "string_assertion_case",
            "fail_fast": True, "calls": calls, "broken": broken, "seconds": time.monotonic() - started}


def smoke(cache: Path, runtime: Path) -> dict:
    check_runtime()
    manifest = provenance()
    check_tasks(cache, manifest)
    rows = []
    for task in PILOT:
        cfg = manifest["tasks"][task]
        base = collect_sources(cache / task / "base", cfg["sources"])
        reference = dict(base)
        name = cfg["reference_file"]
        reference[name] = task_path(cache, task + "/reference/" + name).read_text(encoding="utf-8")
        negative = verify(cache, task, base, runtime)
        positive = verify(cache, task, reference, runtime)
        if negative["status"] != "graded" or negative["passed"] is not False:
            raise IsolationError("base source must fail: " + task)
        if positive["status"] != "graded" or positive["passed"] is not True or not positive["tests_complete"]:
            raise IsolationError("reference must pass every case: " + task + ": " + str(positive))
        diagnostics = {"base": visible_tests(cache, task, runtime, False),
                       "reference": visible_tests(cache, task, runtime, True)}
        rows.append({"task": task, "base": negative, "reference": positive, "visible_tests": diagnostics})
    return {"kind": "deterministic_real_repository_smoke", "model_measurement": False, "tasks": rows}
