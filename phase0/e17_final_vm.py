"""Frozen TEST VM controller: the existing scientific loop, sequential models and one consumed grid."""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from e17_campaign import ASSETS, LIMITS, canonical, digest
from e17_final_freeze import consume, slug, validate_plan, verify_vm_sources, package_versions
from e17_vm import (SequentialClient, campaign_isolation, colab, cuda_devices, cuda_environment,
                    execution_provenance, fetch, receipt, validate_image)


def save(path, value):
    from colab.e17_campaign_remote import save as atomic_save
    atomic_save(path, value)


def validate_grade(grade):
    if (type(grade) is not dict or grade.get("sandbox_backend") != "ephemeral-vm"
            or grade.get("sandbox") != "ephemeral-vm-v1" or grade.get("status") != "graded"
            or type(grade.get("passed")) is not bool or grade.get("trust_scope") != "public-benchmark-nonadversarial"
            or grade.get("adversarial_attestation") is not False):
        raise ValueError("infrastructure/backend result is not a candidate verdict")
    if grade.get("candidate_failure") in ("verification_budget", "invalid_source_artifact") and grade["passed"] is False:
        if grade.get("isolation") is None:
            return
    if grade.get("candidate_failure") == "incomplete official tests":
        raise ValueError("public test collection/completion unsupported; no final score")
    if grade.get("python_audit_hook") is not True:
        raise ValueError("public unittest Python audit hook missing")
    receipt(grade.get("isolation"))
    calls = grade.get("calls")
    if (not calls or any(type(c) is not dict or c.get("complete") is not True for c in calls)
            or any(type(grade.get(k)) is not int or grade[k] < 0 for k in ("tests", "tests_expected"))):
        raise ValueError("public unittest counters/completion missing")
    if any(grade.get(k, 0) != 0 for k in ("skipped", "expected_failures", "unexpected_successes")):
        raise ValueError("unsupported public test completion; final campaign remains incomplete")
    if grade["passed"] and (grade.get("tests_complete") is not True or grade["tests"] != grade["tests_expected"]
                            or len(calls) != grade["tests_expected"]
                            or any(grade.get(k) != 0 for k in
                                   ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"))):
        raise ValueError("stdout/partial public tests cannot establish green")


def validate_episode(row, events, cell, manifest):
    from essaim import aider_agent, code_pilot
    from essaim.agent import is_submit
    expected = {"task": slug(cell), "mode": cell["mode"], "reserved_task": cell["task"],
                "benchmark": cell["benchmark"], "cell_index": cell["index"],
                "campaign_hash": digest(canonical(manifest))}
    if (any(row.get(k) != v for k, v in expected.items()) or row.get("status") != "graded"
            or type(row.get("passed")) is not bool or row.get("sandbox_backend") != "ephemeral-vm"
            or row.get("sandbox") != "ephemeral-vm-v1" or not events or events[-1].get("kind") != "end"):
        raise ValueError("foreign/incomplete final episode")
    # run_one records its own row before the controller attaches reserved metadata.
    end = {k: v for k, v in row.items() if k not in
           ("transcript_sha256", "reserved_task", "benchmark", "cell_index")}
    if [e for e in events if e.get("kind") == "end"] != [{"kind": "end", **end}]:
        raise ValueError("transcript/verdict mismatch")
    starts = [e for e in events if e.get("kind") == "start"]
    if len(starts) != 1 or starts[0].get("system") != manifest["system"]:
        raise ValueError("frozen scientific prompt mismatch")
    validate_grade(row.get("verifier"))
    for event in events:
        if event.get("kind") == "step":
            command = event.get("command")
            if command is not None and command != aider_agent.TEST and not is_submit(command):
                receipt(event.get("isolation"))
        official = event.get("note", {}).get("official_test")
        if official is not None:
            validate_grade(official)
    requests = [e for e in events if e.get("kind") == "request"]
    from types import SimpleNamespace
    for request in requests:
        parameters = request.get("parameters", {})
        allowed_models = cell["models"] + ([cell["reference"]] if cell["reference"] else [])
        if (type(request.get("ceiling")) is not int or not 0 < request["ceiling"] <= LIMITS["max_tokens"]
                or request.get("model") not in allowed_models or parameters.get("model") != request["model"]
                or type(parameters.get("seed")) is not int or parameters.get("seed") != 0
                or type(parameters.get("temperature")) not in (int, float) or parameters.get("temperature") != 0
                or parameters.get("max_tokens") != request["ceiling"]
                or canonical(parameters.get("myriad")) != canonical(
                    {"k": 1, "early_stop": False, "format_instruction": False})
                or request.get("served_model", request["model"]) != request["model"]
                or not math.isfinite(request["seconds"]) or request["seconds"] < 0
                or any(v is not None and (type(v) is not int or not 0 <= v <= request["ceiling"])
                       for v in (request.get("completion_tokens"), request.get("known_completion_tokens")))):
            raise ValueError("request accounting differs from frozen ceilings")
    accounting = code_pilot.BudgetChat.accounting(SimpleNamespace(requests=requests))
    if (any(row.get(k) != v for k, v in accounting.items())
            or accounting["charged_request_ceilings"] > LIMITS["total_tokens"]
            or not math.isfinite(row["seconds"]) or row["seconds"] < 0):
        raise ValueError("episode/accounting mismatch")
    if row["passed"] and (row.get("submission_accepted") is not True or row.get("stopped") != "submitted"
                          or row["verifier"]["passed"] is not True):
        raise ValueError("green requires complete final tests and unchanged authorized submission")


def run_cells(plan, args, client, output, deadline, manifest, adapter_factory, *, run_one=None):
    from essaim.code_pilot import run_one as existing_run_one
    run_one = existing_run_one if run_one is None else run_one
    validate_plan(plan)
    entries = [{**cell, "state": "unstarted"} for cell in plan["cells"]]
    registry = {"freeze_sha256": plan["freeze_sha256"], "request": manifest["request"],
                "preparation_started": True, "attempts": entries, "complete": False}
    rows = []
    try:
        consume(output / "registry.json", registry)
        for entry, cell in zip(entries, plan["cells"]):
            if time.monotonic() + LIMITS["timeout"] + LIMITS["verify_timeout"] > deadline:
                raise TimeoutError("remaining VM envelope cannot reserve a whole frozen episode")
            # Adapter creation/preparation failures consume the cell, before content-dependent dispatch.
            entry["state"] = "started"
            save(output / "registry.json", registry)
            try:
                adapter, cache = adapter_factory(cell)
                args.tasks_dir = cache
                loading, request_seconds = client.loading_seconds, client.request_seconds
                row = run_one(args, slug(cell), cell["mode"], cell["models"], client, output,
                              digest(canonical(manifest)), final_adapter=adapter)
                row.update(reserved_task=cell["task"], benchmark=cell["benchmark"], cell_index=cell["index"])
                rows.append(row)
                save(output / "verdicts.json", rows)
                entry.update(state=row["status"], transcript_sha256=row["transcript_sha256"],
                             loading_seconds=client.loading_seconds - loading,
                             request_seconds=client.request_seconds - request_seconds)
                save(output / "registry.json", registry)
                raw = (output / f"{cell['mode']}__{slug(cell)}.jsonl").read_bytes()
                if digest(raw) != row["transcript_sha256"]:
                    raise ValueError("episode transcript hash mismatch")
                validate_episode(row, [json.loads(line) for line in raw.splitlines()], cell, manifest)
            except BaseException:
                entry["state"] = "error"
                save(output / "registry.json", registry)
                raise
        registry["complete"] = True
        save(output / "registry.json", registry)
        return rows
    finally:
        client.close()


def prepare_models(plan, root, deadline):
    directory = root / "server"
    directory.mkdir()
    hashes = {}
    for asset in ASSETS:
        archive = root / asset["file"]
        fetch("https://github.com/ggml-org/llama.cpp/releases/download/b11505/" + asset["file"],
              archive, asset, min(deadline, time.monotonic() + 600))
        with tarfile.open(archive) as source:
            if sum(m.size for m in source.getmembers()) > 2 << 30:
                raise ValueError("server extraction exceeds pinned bound")
            source.extractall(directory, filter="data")
            for member in source.getmembers():
                if member.isfile():
                    hashes[member.name] = digest((directory / member.name).read_bytes())
    binaries = [p for p in directory.rglob("llama-server") if p.is_file()]
    if len(binaries) != 1:
        raise ValueError("ambiguous server binary")
    binary = binaries[0]
    binary.chmod(0o755)
    environment = cuda_environment(binary, directory)
    devices = cuda_devices(binary, environment)
    result = subprocess.run([str(binary), "--version"], env=environment, timeout=30, capture_output=True,
                            text=True, check=True)
    version = result.stdout + result.stderr
    if "ff5888f99" not in version or "11505" not in version:
        raise ValueError("frozen server build mismatch")
    observed = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    if len(observed) != 1 or "A100" not in observed[0] or int(observed[0].split(",")[-1]) < 39000:
        raise ValueError("A100 capacity differs from frozen declaration")
    for model in plan["models"].values():
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['file']}"
        fetch(url, root / "models" / model["file"], model, min(deadline, time.monotonic() + 1200))
    return binary, environment, {"server_files": hashes, "server_version": version, "gpu_observed": observed,
                                 "cuda_devices": devices, "library_path": environment["LD_LIBRARY_PATH"]}


def generic_smoke(root, runtime, environment):
    from essaim.aider_unittest import verify
    # A historical DEV identifier is only a catalogue seam; every byte below is a fixed synthetic fixture.
    task = "affine-cipher"
    cache = root / "synthetic-aider-smoke"
    base = cache / "python/exercises/practice" / task
    (base / ".meta").mkdir(parents=True, exist_ok=False)
    config = {"files": {"solution": ["counter.py"], "test": ["public_test.py"]}}
    (base / ".meta/config.json").write_bytes(canonical(config))
    source = "class Counter:\n def __init__(self, value): self.value=value\n def add(self, x): self.value+=x; return self\n"
    tests = ("import unittest\nfrom counter import Counter\nSTATE=False\n"
             "def setUpModule():\n global STATE\n STATE=True\n"
             "class Public(unittest.TestCase):\n"
             " def test_module(self): self.assertTrue(STATE)\n"
             " def test_objects(self):\n  obj=Counter(2)\n  self.assertIs(obj.add(3),obj)\n"
             "  self.assertEqual(obj.value,5)\n")
    (base / "public_test.py").write_text(tests, encoding="utf-8")
    positive = verify(cache, task, {"counter.py": source}, runtime, 120, environment=environment)
    negative = verify(cache, task, {"counter.py": source.replace("+=x", "-=x")}, runtime, 120, environment=environment)
    return positive, negative


def execute(root, deadline):
    image = colab()  # Before any reserved content or download; no environment-variable fallback.
    request = json.loads((root / "request.json").read_text())
    source, output = root / "phase0", root / "artifacts"
    plan = verify_vm_sources(request, source)
    if not math.isfinite(deadline) or not 0 < deadline - time.monotonic() <= request["work_seconds"]:
        raise TimeoutError("expired final worker deadline")
    from essaim import bigcode_tasks, code_pilot, aider_agent
    from essaim.e17_final_tasks import AiderAdapter, BigCodeAdapter, prepare_reserved
    runtime = source / "bigcode_pilot/.venv"
    if (digest(aider_agent.SYSTEM.encode()) != plan["parent_decision"]["system_sha256"]
            or json.loads((source / "bigcode_pilot/runtime.json").read_text()) != plan["runtime_pin"]):
        raise ValueError("scientific prompt/runtime pins differ from approved freeze")
    if Path(sys.prefix).resolve() != runtime.resolve():
        raise ValueError("frozen dedicated VM runtime required")
    bigcode_tasks.check_runtime()  # Binary, packages, lock, licences and compiled components, before TEST.
    licence_hashes = {name: {relative: bigcode_tasks.digest(raw) for relative, raw in
                            bigcode_tasks.license_files(name).items()} for name in plan["runtime_pin"]["licenses"]}
    from essaim.e17_final_tasks import fetch_bytes
    notice = plan["runtime_pin"]["python_licence"]
    python_notice = fetch_bytes(notice["url"], notice["sha256"], 65536, deadline)
    with (root / "python-LICENSE").open("xb") as out:
        out.write(python_notice)
    isolation = campaign_isolation(runtime)
    consume(output / "preparation-started.json", {"freeze_sha256": plan["freeze_sha256"], "request": request})
    # Fixed synthetic smoke, no reserved instruction, test or oracle used by this preflight.
    from essaim.ephemeral_vm import EphemeralVMEnv
    synthetic = {"solution.py": "def task_func(): return 2\n"}
    tests = ("import unittest\nSTATE=False\ndef setUpModule():\n global STATE\n STATE=True\n"
             "class Smoke(unittest.TestCase):\n def test_fixed(self):\n"
             "  self.assertTrue(STATE)\n  self.assertEqual(task_func(), 2)\n")
    smoke = {"kind": "prefixed-synthetic-public-unittest", "freeze_sha256": plan["freeze_sha256"],
             "oracle_test_used": False, "positive": bigcode_tasks.evaluate(synthetic, tests, "task_func",
                 ["test_fixed"], runtime, 120, environment=EphemeralVMEnv),
             "negative": bigcode_tasks.evaluate({"solution.py": "def task_func(): return 3\n"}, tests,
                 "task_func", ["test_fixed"], runtime, 120, environment=EphemeralVMEnv)}
    smoke["aider_positive"], smoke["aider_negative"] = generic_smoke(root, runtime, EphemeralVMEnv)
    for key, passed in (("positive", True), ("negative", False), ("aider_positive", True), ("aider_negative", False)):
        validate_grade(smoke[key])
        if smoke[key]["passed"] is not passed:
            raise ValueError("prefixed synthetic VM smoke failed")
    save(output / "smoke.json", smoke)
    binary, environment, server = prepare_models(plan, root, deadline)
    dataset = prepare_reserved(plan, source, root / "data", deadline)
    save(output / "dataset-manifest.json", dataset)
    hardware = {"cpu": platform.processor() or platform.machine(), "host_memory_gib":
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3,
                "execution": "Colab VM, centralized simulation", "gpu": "A100", "cpu_threads": str(os.cpu_count())}
    provenance = execution_provenance(plan, hardware)
    for entry in provenance["models"].values():
        entry["declared_at"] = "parent-approved final freeze " + plan["freeze_sha256"]
    save(output / "provenance.json", provenance)
    import importlib.metadata
    components = []
    for entry in plan["runtime_pin"]["bundled_components"]:
        distribution = importlib.metadata.distribution(entry["package"])
        paths = [distribution.locate_file(p) for p in distribution.files or [] if p.name == entry["file"]]
        if len(paths) != 1:
            raise ValueError("ambiguous pinned runtime component")
        components.append({**entry, **bigcode_tasks.digest(paths[0].read_bytes())})
    runtime_info = {"python": platform.python_version(), "executable_sha256": digest(Path(sys.executable).read_bytes()),
                    "executable_bytes": Path(sys.executable).stat().st_size, "bundled_components": components,
                    "sources": request["sources"], "sandbox_backend": "ephemeral-vm", "colab_image": image,
                    "lock_sha256": digest((source / "bigcode_pilot/uv.lock").read_bytes()),
                    "packages": package_versions((d.metadata["Name"], d.version)
                                                 for d in importlib.metadata.distributions()),
                    "license_hashes": licence_hashes, "python_license": bigcode_tasks.digest(python_notice),
                    "recipe_verified": True, **server}
    save(output / "runtime.json", runtime_info)
    manifest = {"request": request, "freeze_sha256": plan["freeze_sha256"], "system": aider_agent.SYSTEM,
                "runtime": runtime_info, "execution_provenance": provenance, "dataset": dataset,
                "sandbox_backend": "ephemeral-vm", "isolation": isolation, "colab_image": image,
                "limits": LIMITS, "models": plan["models"], "cells": plan["cells"]}
    save(output / "manifest.json", manifest)
    args = code_pilot.parser().parse_args(["--sandbox-backend", "ephemeral-vm", "--runtime", str(runtime)])
    for name, value in LIMITS.items():
        setattr(args, name, value)
    args.reference = next(c["reference"] for c in plan["cells"] if c["mode"] == "cascade")
    def adapter_factory(cell):
        verify_vm_sources(request, source)
        cache = root / "data" / cell["benchmark"]
        adapter = (AiderAdapter(cell["task"], cache, dataset["aider"]) if cell["benchmark"] == "aider"
                   else BigCodeAdapter(cache, cell["task"], dataset["bigcode"]["records.json"]))
        return adapter, cache
    client = SequentialClient(binary, plan["models"], root / "models", output, deadline,
                              lambda url, key: code_pilot.DeadlineClient(url, "final-loopback"), environment)
    run_cells(plan, args, client, output, deadline, manifest, adapter_factory)
    verify_vm_sources(request, source)
    bigcode_tasks.check_runtime()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    execute(Path(sys.argv[1]), float(sys.argv[2]))
