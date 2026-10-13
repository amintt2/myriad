"""Offline exact scientific freeze and fail-closed final recovery; reserved content belongs only to the VM.

The distinct final VM/supervisor reuse the existing loop and lifecycle. Historical DEV R00 stays independent.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from e17_campaign import (ASSETS, LIMITS, MODELS, QWEN, GRANITE, REFERENCE, canonical, digest,
                          MAX_ARTIFACT_BYTES, CHUNK)

ROOT = Path(__file__).resolve().parent.parent
NAME = "e17-final-frozen"
MODES = {"single": [QWEN], "vote": [QWEN, GRANITE], "cascade": [QWEN, GRANITE], "reference": [REFERENCE]}
TASKS = [("aider", "bottle-song"), ("aider", "bowling"),
         ("bigcode", "BigCodeBench/208"), ("bigcode", "BigCodeBench/211")]
SELECTION = "docs/e17_selection_pre_mesure.json"
PROPOSAL = "docs/e17_proposition_p01.md"
ANALYSIS = "docs/17_e17_final_freeze.md"
MAX_BUNDLE = 8 << 20


def package_versions(entries):
    """Compare distribution identities exactly after standard name normalization, without collisions."""
    packages = {}
    for name, version in entries:
        if (type(name) is not str or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name)
                or type(version) is not str or not version):
            raise ValueError("invalid runtime distribution metadata")
        identity = re.sub(r"[-_.]+", "-", name).lower()
        if identity in packages:
            raise ValueError("ambiguous normalized runtime distribution identity")
        packages[identity] = version
    return packages


BLOCKERS = ["parent-scientific-decisions-and-exact-freeze-approval-required",
            "runtime-and-reserved-support-must-be-verified-on-VM-after-freeze"]
# No resource tasks.json, official assertions, caches, data, results, models or archived scientific traces.
RESOURCE_FILES = ["phase0/code_pilot/pyproject.toml", "phase0/code_pilot/uv.lock",
                  "phase0/bigcode_pilot/pyproject.toml", "phase0/bigcode_pilot/uv.lock",
                  "phase0/bigcode_pilot/runtime.json"]
FINAL_METADATA = ["phase0/code_pilot/tasks.json", "phase0/bigcode_pilot/tasks.json"]
REQUIRED_OPERATIONAL_FILES = {
    "phase0/analyze_e17_final.py",
    "phase0/e17_final_freeze.py", "phase0/e17_final_vm.py", "phase0/e17_campaign.py", "phase0/e17_vm.py",
    *["phase0/essaim/" + n + ".py" for n in ("aider_python", "aider_unittest", "aider_agent", "agent",
       "code_pilot", "bigcode_tasks", "bigcode_unittest", "e17_final_tasks", "ephemeral_vm", "local_linux",
       "sandbox", "terminal_bench", "code_provenance")],
    *["phase0/colab/" + n + ".py" for n in ("e17_final_remote", "e17_final_supervise", "e17_campaign",
       "e17_campaign_remote", "e17_probe", "e17_probe_remote", "session_json", "read_timeout", "budget", "safe_cli")],
    "phase0/colab/colab_phase0.sh"}


def read_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def code_bundle(root=ROOT):
    """Bounded explicit code whitelist, also fingerprinting future adapter .py files without importing them."""
    names = [SELECTION, PROPOSAL, ANALYSIS, *RESOURCE_FILES]
    for directory in ("phase0", "phase0/essaim", "phase0/colab", "phase0/tests"):
        names.extend(p.relative_to(root).as_posix() for p in (root / directory).glob("*.py"))
    names.extend(p.relative_to(root).as_posix() for p in (root / "phase0/colab").glob("*.sh"))
    files, total = {}, 0
    for name in sorted(set(names)):
        path = root / name
        # Refuse links/junctions at every level before reading; no traversal outside this worktree.
        for part in (path, *path.parents):
            if part == root.parent:
                break
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                raise ValueError("linked source refused")
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("missing or foreign source: " + name)
        size = path.stat().st_size
        if size > MAX_BUNDLE or total + size > MAX_BUNDLE:
            raise ValueError("code-only bundle exceeds 8 MiB")
        raw = path.read_bytes()
        total += len(raw)
        if total > MAX_BUNDLE:
            raise ValueError("code-only bundle exceeds 8 MiB")
        files[name] = raw
    return files


def cells():
    return [{"index": i, "mode": mode, "benchmark": benchmark, "task": task, "models": peers,
             "reference": REFERENCE if mode == "cascade" else None}
            for i, (mode, benchmark, task, peers) in enumerate(
                (m, b, t, MODES[m]) for m in MODES for b, t in TASKS)]


def slug(cell):
    return cell["benchmark"] + "--" + cell["task"].split("/")[-1]


def final_sources(root=ROOT):
    files = code_bundle(root)
    for name in FINAL_METADATA:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("foreign final metadata")
        files[name] = path.read_bytes()
    for path in (root / "docs").glob("e17_proposition_p0*.md"):
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    if sum(map(len, files.values())) > MAX_BUNDLE:
        raise ValueError("final code/metadata exceeds 8 MiB")
    return files


def source_hashes(files):
    return {n: {"bytes": len(v), "sha256": digest(v)} for n, v in files.items()}


def valid_hash(value, length):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def decision_validated(decision, sources):
    required = {"schema", "actor", "approved", "code_commit", "scientific_revision", "selected", "proposals",
                "sources", "adapter_fix_commit", "audit_sha256", "dev_smoke_sha256", "synthetic_smoke_sha256",
                "bigcode_record_encoding", "system_sha256"}
    if (set(decision) != required or type(decision["schema"]) is not int or decision["schema"] != 1
            or decision["actor"] != "parent" or decision["approved"] is not True
            or canonical(decision["sources"]) != canonical(sources)
            or not valid_hash(decision["code_commit"], 40)
            or not valid_hash(decision["adapter_fix_commit"], 40)
            or decision["adapter_fix_commit"] == "7fbe584fb8cebc1fa0c7612f5097f73aa9914258"
            or decision["bigcode_record_encoding"] != "json-sort-ascii-default-v1"
            or type(decision["scientific_revision"]) is not str
            or not re.fullmatch(r"R\d{2}", decision["scientific_revision"])
            or any(not valid_hash(decision[k], 64) for k in
                   ("audit_sha256", "dev_smoke_sha256", "synthetic_smoke_sha256", "system_sha256"))):
        raise ValueError("explicit complete parent scientific decision, fix/audit/smoke provenance required")
    proposals = decision["proposals"]
    if (type(proposals) is not list or not 1 <= len(proposals) <= 3
            or [p.get("id") for p in proposals] != ["P01", "P02", "P03"][:len(proposals)]):
        raise ValueError("at most three consecutive closed proposals, including P01")
    for proposal in proposals:
        name = "docs/e17_proposition_" + proposal["id"].lower() + ".md"
        if (set(proposal) != {"id", "decision", "sha256"}
                or proposal["decision"] not in ("accepted", "rejected", "invalidated")
                or sources.get(name, {}).get("sha256") != proposal["sha256"]):
            raise ValueError("proposal source or closed decision mismatch")
    if not any(p["id"] == decision["selected"] and p["decision"] == "accepted" for p in proposals):
        raise ValueError("parent-selected proposal must be explicitly accepted")
    return decision


def fixed_fields(policy):
    return {"cells": cells(), "mode_order": list(MODES), "models": MODELS, "limits": LIMITS, "gpu": "A100",
            "temperature": 0, "sandbox_backend": "ephemeral-vm", "model_loading": "sequential-one-gpu-slot",
            "server": {"version": "b11505", "commit": "ff5888f99", "assets": ASSETS, "context": 32768,
                       "parallel": 1, "gpu_layers": 999, "reasoning_budget": 0, "jinja": True},
            "budget": policy, "simulation": "centralized-on-one-vm",
            "trust_scope": "public-benchmark-nonadversarial", "adversarial_attestation": False,
            "attempt_policy": "started-is-consumed-no-resume-no-replay-no-replacement"}


def frozen_plan(files, policy, decision):
    if policy["recover_seconds"] < 2850:
        raise ValueError("final multipart recovery requires at least 2850 seconds reserved")
    metadata = reservation(files[SELECTION])
    hashes = source_hashes(files)
    decision_validated(decision, hashes)
    plan = {"schema": 1, "name": NAME, "partition": "TEST", "state": "frozen", "parent_decision": decision,
            "sources": hashes, "selection": metadata, "analysis_sha256": digest(files[ANALYSIS]),
            "selection_sha256": digest(files[SELECTION]), "proposal_sha256": digest(files[PROPOSAL]),
            "runtime_project": "bigcode_pilot", "runtime_recipe": "uv-frozen-3.12.15-linux-x86_64-gnu",
            "runtime_pin": read_json(files["phase0/bigcode_pilot/runtime.json"]),
            "runtime_verified": False, **fixed_fields(policy)}
    plan["freeze_sha256"] = digest(canonical(plan))
    return validate_plan(plan)


def validate_plan(plan):
    if type(plan) is not dict or plan.get("name") != NAME or plan.get("state") != "frozen":
        raise ValueError("missing or foreign scientific freeze")
    body = {k: v for k, v in plan.items() if k != "freeze_sha256"}
    if digest(canonical(body)) != plan.get("freeze_sha256"):
        raise ValueError("altered freeze")
    required = {"schema", "name", "partition", "state", "parent_decision", "sources", "selection",
                "analysis_sha256", "selection_sha256", "proposal_sha256", "runtime_project", "runtime_recipe",
                "runtime_verified", "runtime_pin", "freeze_sha256", *fixed_fields(plan["budget"])}
    if (set(plan) != required or type(plan["schema"]) is not int or plan["schema"] != 1 or plan["partition"] != "TEST"
            or plan["runtime_project"] != "bigcode_pilot" or plan["runtime_verified"] is not False
            or plan["runtime_recipe"] != "uv-frozen-3.12.15-linux-x86_64-gnu"):
        raise ValueError("foreign final protocol")
    policy = plan["budget"]
    if policy["recover_seconds"] < 2850:
        raise ValueError("insufficient reserved final recovery window")
    expected = budget(policy["declared_prior_units"], policy["policy"]["budget"], policy["policy"]["rate"],
                      policy["seconds"], policy["recover_seconds"], policy["cleanup_seconds"])
    if canonical(policy) != canonical(expected) or any(canonical(plan[k]) != canonical(v)
                                                      for k, v in fixed_fields(expected).items()):
        raise ValueError("final modes/models/pins/limits/budget differ from protocol")
    reservation(canonical(plan["selection"]))
    hashes = plan["sources"]
    if (type(hashes) is not dict or not (set([SELECTION, PROPOSAL, ANALYSIS, *FINAL_METADATA]) |
                                      REQUIRED_OPERATIONAL_FILES) <= set(hashes)
            or len(hashes) > 1000):
        raise ValueError("complete bounded final source inventory required")
    total = 0
    for name, info in hashes.items():
        allowed = (name in [SELECTION, PROPOSAL, ANALYSIS, *RESOURCE_FILES, *FINAL_METADATA]
                   or re.fullmatch(r"docs/e17_proposition_p0[123]\.md", name)
                   or re.fullmatch(r"phase0/(?:essaim/|tests/|colab/)?[\w.-]+\.(?:py|sh)", name))
        if (not allowed or type(info) is not dict or set(info) != {"bytes", "sha256"}
                or type(info["bytes"]) is not int or not 0 <= info["bytes"] <= MAX_BUNDLE
                or not valid_hash(info["sha256"], 64)):
            raise ValueError("foreign/unsafe source inventory")
        total += info["bytes"]
    if total > MAX_BUNDLE:
        raise ValueError("final source inventory exceeds 8 MiB")
    decision_validated(plan["parent_decision"], hashes)
    for field, name in (("selection_sha256", SELECTION), ("proposal_sha256", PROPOSAL), ("analysis_sha256", ANALYSIS)):
        if plan[field] != hashes[name]["sha256"]:
            raise ValueError("scientific metadata hash mismatch")
    return plan


def freeze(directory, output, decision, root=ROOT):
    prepared = validate(directory, root)
    files = final_sources(root)
    plan = frozen_plan(files, prepared["budget"], decision)
    output.mkdir(parents=True, exist_ok=False)
    raw = archive(files)
    with (output / "code.tar").open("xb") as dest:
        dest.write(raw)
    with (output / "freeze.json").open("xb") as dest:
        dest.write(canonical(plan))
    return plan


def validate_freeze(directory, root=ROOT):
    plan = validate_plan(read_json(bounded_read(directory / "freeze.json", MAX_BUNDLE)))
    files = final_sources(root)
    if plan != frozen_plan(files, plan["budget"], plan["parent_decision"]):
        raise ValueError("current sources differ from exact parent scientific freeze")
    if bounded_read(directory / "code.tar", MAX_BUNDLE + (2 << 20)) != archive(files):
        raise ValueError("missing or altered final code archive")
    return plan, files


def bounded_read(path, maximum):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("linked/foreign artifact refused")
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("missing or unbounded artifact")
    with path.open("rb") as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("artifact changed beyond its bound")
    return raw


def vm_name(name):
    return name.removeprefix("phase0/") if name.startswith("phase0/") else "final_metadata/" + name[5:]


def vm_bundle(files):
    return {vm_name(n): v for n, v in files.items()}


def verify_vm_sources(request, source):
    plan = validate_plan(request["plan"])
    expected = {vm_name(n): info["sha256"] for n, info in plan["sources"].items()}
    if request["sources"] != expected:
        raise ValueError("foreign uploaded source declaration")
    for name, info in plan["sources"].items():
        path = source / vm_name(name)
        if path.is_symlink() or path.stat().st_size != info["bytes"] or digest(path.read_bytes()) != info["sha256"]:
            raise ValueError("VM source hash/size mismatch")
    return plan


def consume(path, identity):
    """Exclusive durable marker; interruption or any later failure never removes this record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(canonical(identity))
        out.flush()
        os.fsync(out.fileno())
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def artifacts(plan):
    validate_plan(plan)
    return {"manifest.json", "registry.json", "smoke.json", "provenance.json", "verdicts.json", "runtime.json",
            "dataset-manifest.json", "vm-status.json", "controller.log", "server-events.jsonl",
            "preparation-started.json"} | {
                f"{c['mode']}__{slug(c)}.jsonl" for c in plan["cells"]}


def validate_inventory(value, request):
    validate_plan(request["plan"])
    if request.get("sources") != {vm_name(n): info["sha256"] for n, info in request["plan"]["sources"].items()}:
        raise ValueError("foreign runtime source declaration")
    if (type(value) is not dict or set(value) != {"schema", "request", "complete", "files"}
            or type(value["schema"]) is not int or value["schema"] != 1 or value["request"] != request
            or type(value["complete"]) is not bool or type(value["files"]) is not dict):
        raise ValueError("missing/foreign final inventory")
    total = 0
    for name, info in value["files"].items():
        if (name not in artifacts(request["plan"]) or type(info) is not dict or set(info) != {"bytes", "sha256"}
                or type(info["bytes"]) is not int or not 0 <= info["bytes"] <= MAX_ARTIFACT_BYTES
                or not valid_hash(info["sha256"], 64)):
            raise ValueError("unbounded/unsafe final artifact")
        total += info["bytes"]
    if total > MAX_ARTIFACT_BYTES:
        raise ValueError("final artifacts exceed 64 MiB")
    return value


def validate_retrieved(directory, inventory, request):
    validate_inventory(inventory, request)
    plan = request["plan"]
    for name, info in inventory["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != info["bytes"] or digest(path.read_bytes()) != info["sha256"]:
            raise ValueError("recovered final artifact hash/size mismatch")
    if not inventory["complete"] or set(inventory["files"]) != artifacts(plan):
        raise ValueError("interrupted/partial final campaign is incomplete, without score or replay")
    read = lambda name: read_json((directory / name).read_bytes())
    manifest, registry, status = read("manifest.json"), read("registry.json"), read("vm-status.json")
    if (status != {"request": request, "state": "complete", "error": None} or manifest.get("request") != request
            or manifest.get("freeze_sha256") != plan["freeze_sha256"] or registry.get("request") != request
            or registry.get("freeze_sha256") != plan["freeze_sha256"] or registry.get("complete") is not True
            or registry.get("preparation_started") is not True or manifest.get("cells") != plan["cells"]
            or manifest.get("limits") != LIMITS or manifest.get("models") != MODELS
            or digest(manifest.get("system", "").encode()) != plan["parent_decision"]["system_sha256"]
            or read("preparation-started.json") != {"freeze_sha256": plan["freeze_sha256"], "request": request}):
        raise ValueError("foreign freeze/provenance or incomplete consumed registry")
    from e17_vm import receipt, validate_image, validate_cuda
    from e17_final_vm import validate_episode, validate_grade
    receipt(manifest.get("isolation"))
    validate_image(manifest.get("colab_image"))
    runtime = read("runtime.json")
    pin = plan["runtime_pin"]
    if (runtime != manifest.get("runtime") or runtime.get("sources") != request["sources"]
            or runtime.get("recipe_verified") is not True or runtime.get("sandbox_backend") != "ephemeral-vm"
            or runtime.get("colab_image") != manifest.get("colab_image") or runtime.get("python") != pin["python"]
            or runtime.get("executable_sha256") != pin["binary"]["sha256"]
            or runtime.get("executable_bytes") != pin["binary"]["size"]
            or runtime.get("bundled_components") != pin.get("bundled_components", [])
            or runtime.get("lock_sha256") != pin["lock"]["sha256"]
            or type(runtime.get("packages")) is not dict or type(pin["packages"]) is not dict
            or package_versions(runtime["packages"].items()) != package_versions(pin["packages"].items())
            or runtime.get("license_hashes") != {n: entry["files"] for n, entry in pin.get("licenses", {}).items()}
            or runtime.get("python_license") != {k: pin["python_licence"][k] for k in ("size", "sha256")}):
        raise ValueError("runtime binary/package/lock/backend provenance mismatch")
    validate_cuda(runtime.get("cuda_devices"), runtime.get("library_path"))
    version, devices, server_files = runtime.get("server_version"), runtime.get("gpu_observed"), runtime.get("server_files")
    if (type(version) is not str or "11505" not in version or "ff5888f99" not in version
            or type(devices) is not list or len(devices) != 1 or type(devices[0]) is not str
            or "A100" not in devices[0] or int(devices[0].split(",")[-1]) < 39000
            or type(server_files) is not dict or not server_files or len(server_files) > 2000
            or sum(Path(n).name == "llama-server" for n in server_files) != 1
            or any(not valid_hash(value, 64) for value in server_files.values())):
        raise ValueError("recovered server build/assets/GPU capacity differ from freeze")
    from essaim.code_provenance import declarations
    models = declarations(directory / "provenance.json", sorted(MODELS), 100, .25)
    for name, fixed in MODELS.items():
        server = {k: v for k, v in plan["server"].items() if k != "assets"} | {
            "artifact": ASSETS[0]["file"], "sha256": ASSETS[0]["sha256"]}
        if any(models[name].get(k) != v for k, v in fixed.items()) or models[name]["server"] != server:
            raise ValueError("recovered model/server pins differ from freeze")
    if manifest.get("execution_provenance") != read("provenance.json"):
        raise ValueError("model manifest/provenance mismatch")
    dataset, smoke = read("dataset-manifest.json"), read("smoke.json")
    if (dataset != manifest.get("dataset") or dataset.get("freeze_sha256") != plan["freeze_sha256"]
            or dataset.get("selection_sha256") != plan["selection_sha256"] or dataset.get("oracle_used") is not False
            or smoke.get("kind") != "prefixed-synthetic-public-unittest"
            or smoke.get("freeze_sha256") != plan["freeze_sha256"] or smoke.get("oracle_test_used") is not False):
        raise ValueError("dataset/predeclared synthetic smoke provenance mismatch")
    for key, expected in (("positive", True), ("negative", False), ("aider_positive", True), ("aider_negative", False)):
        validate_grade(smoke[key])
        if smoke[key]["passed"] is not expected:
            raise ValueError("fixed synthetic smoke failed")
    rows, attempts = read("verdicts.json"), registry.get("attempts", [])
    if len(rows) != 16 or len(attempts) != 16:
        raise ValueError("exact sixteen consumed episodes required")
    for row, attempt, cell in zip(rows, attempts, plan["cells"]):
        if (any(attempt.get(k) != v for k, v in cell.items()) or attempt.get("state") != "graded"
                or attempt.get("transcript_sha256") != row.get("transcript_sha256")):
            raise ValueError("episode order/identity/replay mismatch")
        if any(type(attempt.get(k)) not in (int, float) or not math.isfinite(attempt[k]) or attempt[k] < 0
               or attempt[k] > row["seconds"] for k in ("loading_seconds", "request_seconds")):
            raise ValueError("invalid model loading/request accounting")
        raw = (directory / f"{cell['mode']}__{slug(cell)}.jsonl").read_bytes()
        if digest(raw) != row["transcript_sha256"]:
            raise ValueError("final transcript hash mismatch")
        validate_episode(row, [read_json(line) for line in raw.splitlines()], cell, manifest)
    return True


def reservation(raw):
    value = read_json(raw)
    if (type(value.get("schema")) is not int or value.get("schema") != 1 or value.get("content_opened") is not False
            or value.get("holdout_executed") is not False
            or value.get("status") != "metadata-only-reservation-before-new-scores"
            or value["aider"]["test"] != [t for b, t in TASKS if b == "aider"]
            or value["bigcode"]["test"] != [t for b, t in TASKS if b == "bigcode"]
            or any(value["bigcode"]["test_metadata"][t]["state"] != "not_examined"
                   for b, t in TASKS if b == "bigcode")
            or set(value["aider"]["dev"]) & set(value["aider"]["test"])
            or {t.replace("_", "/") for t in value["bigcode"]["dev"]} & set(value["bigcode"]["test"])):
        raise ValueError("reservation differs from the existing metadata-only selection")
    return value


def budget(spent, allowance, rate, seconds, recovery, cleanup):
    values = (spent, allowance, rate, seconds, recovery, cleanup)
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError("finite budget/window values required")
    charged_prior = max(1.15, spent)
    if (spent < 0 or min(allowance, rate, seconds, recovery, cleanup) <= 0
            or allowance + charged_prior > 20 or recovery >= seconds or seconds > 11500
            or recovery < 1800 or cleanup < 1800):
        raise ValueError("E17 total <=20 including prior reserve, bounded work and closure reserves required")
    # Same audited admission formula as Probe; no account read and no admission claim here.
    policy = {"budget": allowance, "minimum": 16.15, "prior_reserve": 1.15,
              "hours": seconds / 3600, "cleanup": cleanup, "rate": rate, "margin": 161}
    sys.path.insert(0, str(Path(__file__).resolve().parent / "colab"))
    from budget import preflight
    preflight({"schema": 1, "balance_units": allowance + policy["minimum"],
               "rate_units_hour": 0, "assignments": 0}, policy)
    return {"e17_total_cap": 20, "declared_prior_units": spent, "charged_prior_units": charged_prior,
            "prior_reserve_until_reconciliation": 1.15, "policy": policy, "seconds": seconds,
            "recover_seconds": recovery, "cleanup_seconds": cleanup, "usage_wait_seconds": 600,
            "read_seconds": 120, "usage_seconds": 30, "grace": 5, "retry_seconds": 1,
            "poll_seconds": 15, "account_verified": False, "final_budget_guaranteed": False}


def preparation(files, policy):
    reservation(files[SELECTION])
    return {"schema": 1, "name": NAME, "partition": "TEST", "state": "blocked-preparation",
            "frozen": False, "ready_for_test": False, "allocated": False, "score": None,
            "parent_revision": None, "blockers": BLOCKERS,
            "selection_sha256": digest(files[SELECTION]), "proposal_sha256": digest(files[PROPOSAL]),
            "analysis_sha256": digest(files[ANALYSIS]), "cells": cells(), "mode_order": list(MODES),
            "models": MODELS, "limits": LIMITS, "gpu": "A100", "temperature": 0,
            "sandbox_backend": "ephemeral-vm", "simulation": "centralized-on-one-vm",
            "model_loading": "sequential-one-gpu-slot",
            "server": {"version": "b11505", "commit": "ff5888f99", "assets": ASSETS,
                       "context": 32768, "parallel": 1, "gpu_layers": 999, "reasoning_budget": 0, "jinja": True},
            "budget": policy, "attempt_policy": "started-is-consumed-no-resume-no-replay-no-replacement",
            "trust_scope": "public-benchmark-nonadversarial", "adversarial_attestation": False,
            "sources": {n: {"bytes": len(v), "sha256": digest(v)} for n, v in files.items()}}


def archive(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as bundle:
        for name, raw in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(raw), 0o644
            bundle.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def prepare(output, policy, root=ROOT):
    files = code_bundle(root)
    selected = preparation(files, policy)
    raw = archive(files)
    selected["bundle"] = {"file": "code.tar", "bytes": len(raw), "sha256": digest(raw)}
    # An interrupted preparation is retained and cannot be overwritten or silently adopted.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "code.tar").open("xb") as dest:
        dest.write(raw)
    with (output / "preparation.json").open("xb") as dest:
        dest.write(canonical(selected))
    return selected


def validate(directory, root=ROOT):
    path = directory / "preparation.json"
    bundle = directory / "code.tar"
    if directory.is_symlink() or path.is_symlink() or bundle.is_symlink():
        raise ValueError("linked preparation refused")
    if path.stat().st_size > MAX_BUNDLE or bundle.stat().st_size > MAX_BUNDLE + (2 << 20):
        raise ValueError("unbounded preparation")
    selected = read_json(path.read_bytes())
    policy = selected["budget"]
    expected_policy = budget(policy["declared_prior_units"], policy["policy"]["budget"], policy["policy"]["rate"],
                             policy["seconds"], policy["recover_seconds"], policy["cleanup_seconds"])
    files = code_bundle(root)
    raw = archive(files)
    expected = preparation(files, expected_policy)
    expected["bundle"] = {"file": "code.tar", "bytes": len(raw), "sha256": digest(raw)}
    if selected != expected or bundle.read_bytes() != raw:
        raise ValueError("altered, missing, stale or foreign preparation")
    return selected


def require_final_freeze(directory, root=ROOT):
    if not (directory / "freeze.json").exists():
        raise ValueError("final freeze unavailable: explicit parent decisions and audited adapter fix required")
    return validate_freeze(directory, root)


def git_revision(root=ROOT):
    command = git_command(root)
    if subprocess.check_output([*command, "status", "--porcelain"]).strip():
        raise ValueError("scientific freeze requires a clean committed source tree")
    return subprocess.check_output([*command, "rev-parse", "HEAD"]).decode().strip()


def git_command(root=ROOT):
    metadata = root / ".git"
    if sys.platform.startswith("linux") and metadata.is_file():
        text = bounded_read(metadata, 4096).decode().strip()
        match = re.fullmatch(r"gitdir: ([A-Za-z]):/(.+)", text)
        if match:
            directory = Path("/mnt") / match[1].lower() / match[2]
            if not directory.is_dir():
                raise ValueError("Windows worktree Git metadata is unavailable under WSL")
            return ["git", "--git-dir=" + str(directory), "--work-tree=" + str(root)]
    return ["git", "-C", str(root)]


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Offline blocked preparation; no TEST content or Colab CLI")
    prep.add_argument("--output", type=Path, required=True)
    for name in ("spent-units", "budget-units", "max-rate-units-hour", "seconds"):
        prep.add_argument("--" + name, type=float, required=True)
    prep.add_argument("--recover-seconds", type=float, default=1800)
    prep.add_argument("--cleanup-seconds", type=float, default=1800)
    commands.add_parser("validate").add_argument("--preparation", type=Path, required=True)
    seal = commands.add_parser("freeze")
    seal.add_argument("--preparation", type=Path, required=True)
    seal.add_argument("--decision", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--parent-approved", action="store_true")
    verify = commands.add_parser("validate-freeze")
    verify.add_argument("--freeze", type=Path, required=True)
    template = commands.add_parser("decision-template")
    template.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.command == "decision-template":
        files = final_sources()
        from essaim.aider_agent import SYSTEM
        proposal_names = [n for n in sorted(files) if re.fullmatch(r"docs/e17_proposition_p0[123]\.md", n)]
        decision = {"schema": 1, "actor": "parent", "approved": False, "code_commit": None,
                    "adapter_fix_commit": None, "scientific_revision": None, "selected": None,
                    "proposals": [{"id": Path(n).stem[-3:].upper(), "decision": "pending", "sha256": digest(files[n])}
                                  for n in proposal_names], "sources": source_hashes(files),
                    "audit_sha256": None, "dev_smoke_sha256": None, "synthetic_smoke_sha256": None,
                    "system_sha256": digest(SYSTEM.encode()), "bigcode_record_encoding": "json-sort-ascii-default-v1"}
        consume(args.output, decision)
        print(json.dumps({"state": "unapproved-decision-template", "allocated": False}))
        return 0
    elif args.command == "prepare":
        policy = budget(args.spent_units, args.budget_units, args.max_rate_units_hour,
                        args.seconds, args.recover_seconds, args.cleanup_seconds)
        selected = prepare(args.output, policy)
    elif args.command == "validate":
        selected = validate(args.preparation)
    elif args.command == "freeze":
        if not args.parent_approved:
            raise ValueError("explicit parent approval is required before scientific freeze")
        decision = read_json(args.decision.read_bytes())
        revision = git_revision()
        if revision != decision["code_commit"]:
            raise ValueError("parent decision refers to a different code revision")
        result = subprocess.run([*git_command(), "merge-base", "--is-ancestor",
                                 decision["adapter_fix_commit"], revision], capture_output=True)
        if result.returncode:
            raise ValueError("exact audited adapter P2 fix is not integrated in this revision")
        selected = freeze(args.preparation, args.output, decision)
        print(json.dumps({"state": "frozen", "freeze_sha256": selected["freeze_sha256"],
                          "runtime_verified": False, "allocated": False}, sort_keys=True))
        return 0
    else:
        selected, _ = require_final_freeze(args.freeze)
        if git_revision() != selected["parent_decision"]["code_commit"]:
            raise ValueError("source revision changed after scientific freeze")
        print(json.dumps({"state": "frozen", "freeze_sha256": selected["freeze_sha256"], "allocated": False}))
        return 0
    print(json.dumps({"state": selected["state"], "ready_for_test": False, "allocated": False,
                      "cells": len(selected["cells"]), "blockers": selected["blockers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("final preparation refused: " + str(error))
        raise SystemExit(65)
