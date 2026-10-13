"""Fixed DEV R00 plans and fail-closed retrieval validation; no allocation or inference on import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ["affine-cipher", "beer-song", "book-store"]
QWEN, GRANITE, REFERENCE = "Qwen/Qwen3.5-2B", "ibm-granite/granite-4.2-3b", "Qwen/Qwen3.8-27B"
LIMITS = {"max_tokens": 4096, "total_tokens": 49152, "max_steps": 20, "timeout": 900,
          "cmd_timeout": 30, "verify_timeout": 120, "seed": 0}
R00_AGENT_SHA256 = "e072db88202563447da3f1ab14ea126edaf147febc05a4ad5d7b267e80246fa2"
MAX_ARTIFACT_BYTES = 64 << 20
CHUNK = 4 << 20
SELECTION = "e17-selection-pre-mesure.json"
PROPOSAL = "e17-proposition-p01.md"
MODELS = {
    QWEN: {"repository": "unsloth/Qwen3.5-2B-GGUF", "revision": "f6d5376be1edb4d416d56da11e5397a961aca8ae",
           "file": "Qwen3.5-2B-Q8_0.gguf", "bytes": 2012012800,
           "sha256": "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1"},
    GRANITE: {"repository": "ibm-granite/granite-4.2-3b-GGUF", "revision": "c40945d71cd90f249a56985e8155551a9188dc30",
              "file": "granite-4.2-3b-Q8_0.gguf", "bytes": 3892651552,
              "sha256": "fbe986738041418e26de9e123ba740cb654931f85bf572a71bd01f9e6b85e53d"},
    # Parent metadata from the GGUF repository, fixed before any DEV score.
    REFERENCE: {"repository": "unsloth/Qwen3.8-27B-GGUF", "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                "file": "Qwen3.8-27B-Q8_0.gguf", "bytes": 29047086048,
                "sha256": "a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348"}}
ASSETS = [
    {"file": "llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz", "bytes": 172008023,
     "sha256": "df35bc577d6c48e19062b49bbdb7697498051e3caa710f9073a0e0094d774b58"},
    {"file": "cudart-llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz", "bytes": 594377392,
     "sha256": "4761f7566bb7d6c14df80e9ee6082911d2278f0da8becf5a187812089ef6c8d5"}]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()


def plan(name, reference_revision=None, *, proposal_sha256=None, selection_sha256=None):
    if name not in ("e17-dev-single", "e17-dev-compare"):
        raise ValueError("only fixed DEV plans are supported")
    modes = {"single": [QWEN]}
    if name == "e17-dev-compare":
        modes.update(vote=[QWEN, GRANITE], cascade=[QWEN, GRANITE], reference=[REFERENCE])
    models = {m: dict(MODELS[m]) for peers in modes.values() for m in peers}
    if REFERENCE in models:
        if reference_revision not in (None, MODELS[REFERENCE]["revision"]):
            raise ValueError("comparison refuses a substituted GGUF repository revision")
    return {"schema": 1, "name": name, "partition": "DEV", "revision": "R00", "harness_commit": "067cf41",
            "proposal_sha256": proposal_sha256 or digest((ROOT.parent / "docs/e17_proposition_p01.md").read_bytes()),
            "selection_sha256": selection_sha256 or digest((ROOT.parent / "docs/e17_selection_pre_mesure.json").read_bytes()),
            "mode_order": list(modes), "proposal": "P01", "proposal_decision": "no-harness-change", "tasks": TASKS, "modes": modes,
            "dev_attempt": "initial-single-L4" if name == "e17-dev-single" else "comparison-A100",
            "models": models, "gpu": "L4" if name == "e17-dev-single" else "A100", "limits": LIMITS,
            "temperature": 0, "sandbox_backend": "ephemeral-vm", "simulation": "centralized-on-one-vm",
            "server": {"version": "b11505", "commit": "ff5888f99", "assets": ASSETS,
                       "context": 32768, "parallel": 1, "gpu_layers": 999, "reasoning_budget": 0,
                       "jinja": True}, "model_loading": "sequential-one-gpu-slot"}


def source_bundle():
    # Explicit directories, never data/results/app or a recursive repository export.
    paths = [*sorted((ROOT / "essaim").glob("*.py")), ROOT / "e17_campaign.py", ROOT / "e17_vm.py",
             ROOT / "colab/e17_campaign_remote.py", ROOT / "colab/read_timeout.py",
             ROOT / "code_pilot/pyproject.toml", ROOT / "code_pilot/uv.lock", ROOT / "code_pilot/tasks.json"]
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    if digest(files["essaim/aider_agent.py"].decode("utf-8").replace("\r\n", "\n").encode()) != R00_AGENT_SHA256:
        raise ValueError("R00 scientific harness changed; register a new proposal before measurement")
    files[SELECTION] = (ROOT.parent / "docs/e17_selection_pre_mesure.json").read_bytes()
    files[PROPOSAL] = (ROOT.parent / "docs/e17_proposition_p01.md").read_bytes()
    if sum(map(len, files.values())) > 8 << 20:
        raise ValueError("code-only transfer exceeds 8 MiB")
    return files


def validate_plan(selected):
    import re
    for key in ("proposal_sha256", "selection_sha256"):
        if not isinstance(selected.get(key), str) or not re.fullmatch("[0-9a-f]{64}", selected[key]):
            raise ValueError("pre-score metadata hash missing")
    fixed = plan(selected.get("name"), proposal_sha256=selected["proposal_sha256"],
                 selection_sha256=selected["selection_sha256"])
    if selected != fixed:
        raise ValueError("plan differs from fixed pre-score declaration")
    return selected


def episodes(selected):
    validate_plan(selected)
    return [(mode, task, selected["modes"][mode]) for mode in selected["mode_order"] for task in TASKS]


def artifacts(plan):
    return {"manifest.json", "registry.json", "smoke.json", "provenance.json", "verdicts.json",
            "runtime.json", "dataset-manifest.json", "vm-status.json", "controller.log", "server-events.jsonl"} | {
                f"{mode}__{task}.jsonl" for mode in plan["modes"] for task in TASKS}


def validate_inventory(value, request):
    if (not isinstance(value, dict) or value.get("request") != request or type(value.get("schema")) is not int
            or value["schema"] != 1
            or type(value.get("complete")) is not bool or not isinstance(value.get("files"), dict)):
        raise ValueError("missing or foreign artifact inventory")
    import re
    total = 0
    for name, info in value["files"].items():
        if (name not in artifacts(request["plan"]) or not isinstance(info, dict) or set(info) != {"bytes", "sha256"}
                or type(info["bytes"]) is not int or not 0 <= info["bytes"] <= MAX_ARTIFACT_BYTES
                or not isinstance(info["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", info["sha256"])):
            raise ValueError("unsafe or unbounded artifact inventory")
        total += info["bytes"]
    if total > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact retrieval exceeds 64 MiB")
    return value


def validate_retrieved(directory, inventory, request):
    if request["plan"] != plan(request["plan"].get("name")):
        raise ValueError("retrieved plan differs from local pre-score pins")
    validate_plan(request["plan"])
    validate_inventory(inventory, request)
    for name, info in inventory["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != info["bytes"] or digest(path.read_bytes()) != info["sha256"]:
            raise ValueError("recovered artifact hash/size mismatch")
    if not inventory["complete"] or set(inventory["files"]) != artifacts(request["plan"]):
        raise ValueError("incomplete recovery is not scientific success")
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    manifest, rows, status = read("manifest.json"), read("verdicts.json"), read("vm-status.json")
    if manifest.get("request") != request or status != {"request": request, "state": "complete", "error": None}:
        raise ValueError("foreign manifest or failed VM")
    from e17_vm import (validate_smoke, receipt, dev_manifest, validate_image, validate_calls,
                       validate_episode, validate_cuda)
    validate_smoke(read("smoke.json"))
    runtime, registry, provenance = read("runtime.json"), read("registry.json"), read("provenance.json")
    receipt(manifest.get("isolation"))
    validate_image(manifest.get("colab_image"))
    validate_cuda(runtime.get("cuda_devices"), runtime.get("library_path"))
    if (manifest.get("runtime") != runtime or runtime.get("sources") != request["sources"]
            or runtime.get("sandbox_backend") != "ephemeral-vm" or registry.get("request") != request
            or registry.get("revision") != "R00" or registry.get("proposal") != "P01"
            or registry.get("decision") != "no-harness-change" or registry.get("scientific_gain_claimed") is not False
            or registry.get("selection_sha256") != request["plan"]["selection_sha256"]
            or manifest.get("mode_order") != request["plan"]["mode_order"]
            or manifest.get("selection_sha256") != request["plan"]["selection_sha256"]
            or runtime.get("sources", {}).get(SELECTION) != request["plan"]["selection_sha256"]
            or runtime.get("sources", {}).get(PROPOSAL) != request["plan"]["proposal_sha256"]
            or registry.get("dev_attempt") != request["plan"]["dev_attempt"]
            or runtime.get("colab_image") != manifest.get("colab_image")
            or manifest.get("sandbox_backend") != "ephemeral-vm" or manifest.get("sandbox") != "ephemeral-vm-v1"
            or manifest.get("execution_provenance") != provenance or manifest.get("limits") != LIMITS
            or manifest.get("models") != request["plan"]["modes"]):
        raise ValueError("runtime, registry or declaration mismatch")
    from essaim.code_provenance import declarations
    models = declarations(directory / "provenance.json", sorted(request["plan"]["models"]), 100, .25)
    for name, fixed in request["plan"]["models"].items():
        if any(models[name].get(k) != v for k, v in fixed.items()):
            raise ValueError("model declaration differs from pre-score plan")
        fixed_server = {k: v for k, v in request["plan"]["server"].items() if k != "assets"} | {
            "artifact": ASSETS[0]["file"], "sha256": ASSETS[0]["sha256"]}
        if models[name]["server"] != fixed_server:
            raise ValueError("server declaration differs from pre-score plan")
    original = json.loads((ROOT / "code_pilot/tasks.json").read_text(encoding="utf-8"))
    dataset = read("dataset-manifest.json")
    if dataset != dev_manifest(original) or manifest.get("dataset_hash") != digest(canonical(dataset)):
        raise ValueError("DEV data view differs from original pins")
    expected = {(m, t) for m in request["plan"]["modes"] for t in TASKS}
    seen = set()
    for row in rows:
        key = (row.get("mode"), row.get("task"))
        if (key not in expected or key in seen or row.get("status") != "graded"
                or type(row.get("passed")) is not bool or row.get("campaign_hash") != digest(canonical(manifest))):
            raise ValueError("incomplete/foreign/duplicate verdict coverage")
        transcript = (directory / f"{key[0]}__{key[1]}.jsonl").read_bytes()
        events = [json.loads(line) for line in transcript.splitlines()]
        if (digest(transcript) != row.get("transcript_sha256") or not events or events[-1].get("kind") != "end"
                or any(events[-1].get(k) != v for k, v in row.items() if k != "transcript_sha256")):
            raise ValueError("transcript/end/verdict mismatch")
        validate_episode(row, events)
        grade = row.get("verifier", {})
        if (grade.get("status") != "graded" or type(grade.get("passed")) is not bool
                or row["passed"] and (row.get("submission_accepted") is not True or row.get("stopped") != "submitted"
                or grade["passed"] is not True or grade.get("tests_complete") is not True
                or grade.get("tests") != grade.get("tests_expected") or not grade.get("calls"))):
            raise ValueError("false success or missing final verifier")
        validate_calls(grade)
        seen.add(key)
    if [(r.get("mode"), r.get("task")) for r in rows] != [(m, t) for m, t, _ in episodes(request["plan"])]:
        raise ValueError("verdict execution order mismatch")
    if seen != expected:
        raise ValueError("missing episodes")
    attempts = registry.get("attempts", [])
    if ([(r.get("mode"), r.get("task")) for r in attempts] != [(m, t) for m, t, _ in episodes(request["plan"])]
            or len(attempts) != len(expected) or {(r.get("mode"), r.get("task")) for r in attempts} != expected
            or any(r.get("state") != "graded" for r in attempts)):
        raise ValueError("registry coverage incomplete")
    return True
