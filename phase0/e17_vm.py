"""VM-only DEV controller using R00 run_one and a sequential loopback llama-server client."""
from __future__ import annotations

import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from e17_campaign import ASSETS, LIMITS, TASKS, canonical, digest

SMOKE_TESTS = {"affine-cipher": 16, "beer-song": 8, "book-store": 20}


def write(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def colab():
    from essaim.ephemeral_vm import require_colab
    markers = require_colab()
    image = {"markers": markers, "kernel": platform.release(), "machine": platform.machine(),
             "system": platform.system()}
    validate_image(image)
    return image


def fetch(url, path, info, deadline):
    part = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    sha, size = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=min(30, max(.001, deadline - time.monotonic()))) as response:
            with part.open("wb") as out:
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("VM download deadline")
                    block = response.read(1 << 20)
                    if not block:
                        break
                    size += len(block)
                    if size > info["bytes"]:
                        raise ValueError("oversized download")
                    sha.update(block)
                    out.write(block)
        if time.monotonic() >= deadline or size != info["bytes"] or sha.hexdigest() != info["sha256"]:
            raise ValueError("download size/hash/deadline mismatch")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def receipt(value):
    expected = {"isolation_ready": "ephemeral-vm-v1", "landlock_abi": None, "seccomp": "on",
                "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}
    if value != expected:
        raise ValueError("exact ephemeral-vm-v1 isolation receipt required")


def validate_image(image):
    if (not isinstance(image, dict) or image.get("markers") != {"hostname_file": True, "drive_binary": True}
            or image.get("system") != "Linux" or image.get("machine") != "x86_64"
            or not isinstance(image.get("kernel"), str) or not image["kernel"]
            or "microsoft" in image["kernel"].lower()):
        raise ValueError("Colab image observations missing or incompatible")
    if any(value is not True for value in image["markers"].values()):
        raise ValueError("Colab markers must be observed booleans")


def validate_calls(grade):
    if grade.get("candidate_failure") == "invalid_source_artifact" and grade.get("passed") is False:
        return
    if grade.get("sandbox_backend") != "ephemeral-vm":
        raise ValueError("verifier backend missing")
    for call in grade.get("calls", []):
        if (call.get("complete") is not True or call.get("sandbox_backend") != "ephemeral-vm"
                or call.get("python_audit_hook") is not True):
            raise ValueError("verifier call backend/audit hook incomplete")
        if (call.get("candidate_failure") == "verification_budget" and grade.get("status") == "graded"
                and grade.get("passed") is False and "isolation" in call and call["isolation"] is None
                and type(call.get("seconds")) in (int, float) and call["seconds"] == 0
                and "diagnostics" not in call and "program_exception" not in call):
            continue
        receipt(call.get("isolation"))


def validate_episode(row, events):
    from essaim.aider_agent import TEST
    from essaim.agent import is_submit
    isolation = [e.get("receipt") for e in events if e.get("kind") == "isolation"]
    if len(isolation) != 1 or "isolation" not in row or isolation[0] != row["isolation"]:
        raise ValueError("episode isolation observation missing or inconsistent")
    if isolation[0] is not None:
        receipt(isolation[0])
    for event in events:
        command = event.get("command")
        if event.get("kind") == "step":
            if command is not None and command != TEST and not is_submit(command):
                receipt(event.get("isolation"))
                receipt(isolation[0])
            elif event.get("isolation") is not None:
                receipt(event["isolation"])
        official = event.get("note", {}).get("official_test")
        if official is not None:
            validate_calls(official)


def campaign_isolation(runtime):
    from essaim.ephemeral_vm import select_backend
    environment = select_backend("ephemeral-vm")
    with tempfile.TemporaryDirectory(prefix="myriad-e17-preflight-") as directory:
        env = environment(Path(directory), runtime)
        try:
            code, output = env.run("printf E17-ISOLATION-OK", 5)
            receipt(env.isolation)
            if code != 0 or output != "E17-ISOLATION-OK":
                raise ValueError("campaign isolation preflight failed")
            return env.isolation
        finally:
            env.close()


def validate_smoke(smoke):
    if (smoke.get("model_measurement") is not False or smoke.get("kind") != "deterministic_harness_smoke"
            or smoke.get("sandbox_backend") != "ephemeral-vm" or smoke.get("sandbox") != "ephemeral-vm-v1"
            or [row.get("task") for row in smoke.get("tasks", [])] != TASKS):
        raise ValueError("official DEV smoke coverage missing")
    for row in smoke["tasks"]:
        for key, passed in (("reference", True), ("incorrect", False)):
            grade = row[key]
            if grade.get("status") != "graded" or grade.get("passed") is not passed or not grade.get("calls"):
                raise ValueError("official smoke refused")
            if (grade.get("tests_expected") != SMOKE_TESTS[row["task"]] or passed and
                    (grade.get("tests_complete") is not True or grade.get("tests") != grade.get("tests_expected"))):
                raise ValueError("partial green smoke refused")
            validate_calls(grade)


def cuda_environment(binary, directory):
    libraries = sorted({str(binary.parent)} | {str(p.parent) for p in directory.rglob("lib*.so*")})
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join([*libraries, "/usr/lib64-nvidia",
                                               environment.get("LD_LIBRARY_PATH", "")])
    return environment


def cuda_devices(binary, environment):
    result = subprocess.run([str(binary), "--list-devices"], env=environment, capture_output=True,
                            text=True, timeout=30, check=True)
    devices = result.stdout + result.stderr
    validate_cuda(devices, environment["LD_LIBRARY_PATH"])
    return devices


def validate_cuda(devices, library_path):
    if (not isinstance(devices, str) or not any(line.strip().startswith("CUDA0:") for line in devices.splitlines())
            or not isinstance(library_path, str) or "/usr/lib64-nvidia" not in library_path.split(":")):
        raise ValueError("CUDA0 is unavailable; CPU fallback refused")


def dev_manifest(manifest):
    # Keep original catalogue metadata for the adapter's fixed historical selection check.
    # Only DEV source/test/example bytes and their license are staged and checked by the adapter.
    view = {**manifest, "files": {name: info for name, info in manifest["files"].items()
            if name in ("LICENSE.python", "README.md") or any(name.startswith("python/exercises/practice/" + task + "/")
                                              for task in TASKS)}}
    if not {"LICENSE.python", "README.md"} <= set(view["files"]) or not all(any(
                                                           n.startswith("python/exercises/practice/" + task + "/")
                                                           for n in view["files"]) for task in TASKS):
        raise ValueError("DEV subset or license missing")
    return view


class SequentialClient:
    def __init__(self, binary, models, directory, output, deadline, client_factory, environment=None):
        self.binary, self.models, self.directory = binary, models, directory
        self.output, self.deadline, self.client_factory = output, deadline, client_factory
        self.process = self.log = self.client = self.model = None
        self.environment = dict(os.environ) if environment is None else environment
        self.loading_seconds = self.request_seconds = 0.0
        self.events = (output / "server-events.jsonl").open("w", encoding="utf-8")

    def event(self, value):
        self.events.write(json.dumps(value, sort_keys=True) + "\n")
        self.events.flush()

    def stop(self):
        if self.process is not None:
            # Only the process group created by this object; never pgrep/pkill or PID discovery.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.event({"event": "stopped", "model": self.model, "returncode": self.process.returncode})
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.client = self.model = None

    def post(self, path, json, timeout):
        model = json["model"]
        if model not in self.models or path != "chat/completions":
            raise ValueError("undeclared model or endpoint")
        deadline = min(self.deadline, time.monotonic() + timeout)
        if self.model != model:
            loading_started = time.monotonic()
            self.stop()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            args = [str(self.binary), "-m", str(self.directory / self.models[model]["file"]),
                    "--alias", model, "--host", "127.0.0.1", "--port", str(port), "--jinja",
                    "-ngl", "999", "--device", "CUDA0", "-c", "32768", "-np", "1",
                    "--reasoning-budget", "0", "--no-webui"]
            self.log = (self.output.parent / "llama-current.log").open("wb")
            self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True, env=self.environment)
            self.model = model
            self.event({"event": "started", "model": model, "parameters": args[3:],
                        "binary_sha256": digest(self.binary.read_bytes())})
            while True:
                if time.monotonic() >= deadline or self.process.poll() is not None:
                    seconds = time.monotonic() - loading_started
                    self.loading_seconds += seconds
                    self.event({"event": "load-failed", "model": model, "seconds": seconds})
                    raise TimeoutError("sequential server startup failed")
                import httpx
                try:
                    with httpx.Client(trust_env=False) as client:
                        response = client.get(f"http://127.0.0.1:{port}/health",
                                              timeout=min(1, deadline - time.monotonic()))
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(min(.2, max(0, deadline - time.monotonic())))
            self.client = self.client_factory(f"http://127.0.0.1:{port}/v1", "")
            seconds = time.monotonic() - loading_started
            self.loading_seconds += seconds
            self.event({"event": "loaded", "model": model, "seconds": seconds})
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model loading consumed the episode budget")
        started = time.monotonic()
        error = None
        try:
            response = self.client.post(path, json=json, timeout=remaining)
            response.raise_for_status()
            if response.json().get("model") != model or time.monotonic() >= deadline:
                raise ValueError("foreign served model or late response")
            return response
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            seconds = time.monotonic() - started
            self.request_seconds += seconds
            self.event({"event": "request", "model": model, "seconds": seconds, "error": error})

    def close(self):
        try:
            self.stop()
        finally:
            self.events.close()


def run_episodes(plan, args, client, output, deadline, manifest, registry, pilot):
    from e17_campaign import episodes
    rows = []
    try:
        for mode, task, peers in episodes(plan):
            if time.monotonic() + 1020 > deadline:
                raise TimeoutError("remaining VM envelope cannot reserve a whole R00 episode")
            registry["attempts"].append({"mode": mode, "task": task, "state": "started"})
            write(output / "registry.json", registry)
            loading_before, requests_before = client.loading_seconds, client.request_seconds
            row = pilot.run_one(args, task, mode, peers, client, output, digest(canonical(manifest)))
            rows.append(row)
            write(output / "verdicts.json", rows)
            registry["attempts"][-1].update(state=row["status"], transcript_sha256=row["transcript_sha256"],
                loading_seconds=client.loading_seconds - loading_before,
                request_seconds=client.request_seconds - requests_before)
            write(output / "registry.json", registry)
            if row["status"] != "graded":
                raise RuntimeError("infrastructure episode is not a scientific verdict")
            events = [json.loads(line) for line in (output / f"{mode}__{task}.jsonl").read_text().splitlines()]
            validate_episode(row, events)
            validate_calls(row["verifier"])
    finally:
        client.close()


def execution_provenance(plan, hardware):
    return {"models": {name: {**model, "model": name, "alias": name, "licence": "apache-2.0",
        "declared_at": "pre-score R00 plan", "endpoint_kind": "direct-loopback-sequential",
        "quantization": "Q8_0", "server": {k: v for k, v in plan["server"].items() if k != "assets"} |
        {"artifact": ASSETS[0]["file"], "sha256": ASSETS[0]["sha256"]}, "hardware": hardware,
        "energy": {"watts": 100, "eur_kwh": .25, "measured": False, "scope": "assumed VM power only"}}
        for name, model in plan["models"].items()}}


def execute(root, deadline):
    image = colab()
    from essaim import code_pilot as pilot
    request = json.loads((root / "request.json").read_text())
    from e17_campaign import validate_plan, episodes, SELECTION, PROPOSAL
    plan, output = validate_plan(request["plan"]), root / "artifacts"
    if digest((root / "phase0" / SELECTION).read_bytes()) != plan["selection_sha256"]:
        raise ValueError("pre-measure selection hash mismatch")
    if digest((root / "phase0" / PROPOSAL).read_bytes()) != plan["proposal_sha256"]:
        raise ValueError("pre-measure proposal hash mismatch")
    # A missing backend integration is a preparation failure, never a fallback to strict or a patch of R00.
    if "--sandbox-backend" not in pilot.parser()._option_string_actions:
        raise RuntimeError("backend dependency: code_pilot --sandbox-backend is not integrated")
    from essaim import aider_python
    view = dev_manifest(pilot.provenance())
    write(output / "dataset-manifest.json", view)
    # A dedicated VM process selects an immutable metadata view; the R00 adapter and assertions are unchanged.
    aider_python.MANIFEST = output / "dataset-manifest.json"
    import math
    if not math.isfinite(deadline) or not 0 < deadline - time.monotonic() <= request["work_seconds"]:
        raise TimeoutError("invalid or expired worker deadline")
    runtime = root / "phase0/code_pilot/.venv"
    isolation = campaign_isolation(runtime)
    common = ["--runtime", str(runtime), "--tasks-dir", str(root / "data"),
              "--sandbox-backend", "ephemeral-vm"]
    for name, value in LIMITS.items():
        common.extend(("--" + name.replace("_", "-"), str(value)))
    prepare_args = pilot.parser().parse_args([*common, "--prepare"])
    if pilot.execute(prepare_args):
        raise RuntimeError("dataset preparation failed")
    smoke_args = pilot.parser().parse_args([*common, "--smoke", "--output", str(output)])
    if pilot.execute(smoke_args):
        raise RuntimeError("official VM smoke failed")
    smoke = json.loads((output / "smoke.json").read_text())
    validate_smoke(smoke)
    server_dir = root / "server"
    server_dir.mkdir()
    extracted = {}
    for asset in ASSETS:
        archive = root / asset["file"]
        fetch("https://github.com/ggml-org/llama.cpp/releases/download/b11505/" + asset["file"],
              archive, asset, min(deadline, time.monotonic() + 600))
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if sum(m.size for m in members) > 2 << 30:
                raise ValueError("oversized server archive")
            tar.extractall(server_dir, filter="data")
            for member in members:
                if member.isfile():
                    extracted[member.name] = digest((server_dir / member.name).read_bytes())
    binaries = [p for p in server_dir.rglob("llama-server") if p.is_file()]
    if len(binaries) != 1:
        raise ValueError("missing/ambiguous server binary")
    binary = binaries[0]
    binary.chmod(0o755)
    environment = cuda_environment(binary, server_dir)
    devices = cuda_devices(binary, environment)
    version = subprocess.run([str(binary), "--version"], env=environment, capture_output=True,
                             text=True, timeout=30, check=True)
    if "ff5888f99" not in version.stdout + version.stderr or "11505" not in version.stdout + version.stderr:
        raise ValueError("server build mismatch")
    model_dir = root / "models"
    for model in plan["models"].values():
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['file']}"
        fetch(url, model_dir / model["file"], model, min(deadline, time.monotonic() + 1200))
    hardware = {"cpu": platform.processor() or platform.machine(),
                "host_memory_gib": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3,
                "execution": "Colab VM, centralized simulation", "gpu": plan["gpu"],
                "cpu_threads": str(os.cpu_count())}
    observed_gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    if len(observed_gpu) != 1 or plan["gpu"] not in observed_gpu[0]:
        raise ValueError("unexpected GPU allocation")
    minimum_mib = 22000 if plan["gpu"] == "L4" else 39000
    if int(observed_gpu[0].split(",")[-1].strip()) < minimum_mib:
        raise ValueError("insufficient predeclared GPU capacity")
    bootstrap = json.loads((output / "runtime.json").read_text())["uv_bootstrap"]
    provenance = execution_provenance(plan, hardware)
    write(output / "provenance.json", provenance)
    from essaim.code_provenance import declarations
    declarations(output / "provenance.json", sorted(plan["models"]), 100, .25)
    import importlib.metadata
    runtime_info = {"python": sys.version, "executable_sha256": digest(Path(sys.executable).resolve().read_bytes()),
                    "uv_version": bootstrap["artifact"]["version"], "uv_bootstrap": bootstrap,
                    "lock_sha256": digest((root / "phase0/code_pilot/uv.lock").read_bytes()),
                    "uv_sha256": digest((root / "installer/bin/uv").read_bytes()),
                    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                    "sources": {n: digest((root / "phase0" / n).read_bytes()) for n in request["sources"]},
                    "server_files": extracted, "server_version": version.stdout + version.stderr,
                    "gpu_observed": observed_gpu, "sandbox_backend": "ephemeral-vm", "colab_image": image,
                    "cuda_devices": devices, "library_path": environment["LD_LIBRARY_PATH"]}
    write(output / "runtime.json", runtime_info)
    manifest = {"request": request, "version": pilot.VERSION_PILOT, "system": pilot.aider_agent.SYSTEM,
                "mode_order": plan["mode_order"], "selection_sha256": plan["selection_sha256"],
                "instructions": {t: pilot.instructions(root / "data", t) for t in TASKS},
                "dataset_hash": pilot.hash_json(pilot.provenance()), "runtime": runtime_info,
                "execution_provenance": provenance, "limits": LIMITS, "models": plan["modes"],
                "sandbox_backend": "ephemeral-vm", "sandbox": "ephemeral-vm-v1", "isolation": isolation,
                "colab_image": image}
    write(output / "manifest.json", manifest)
    registry = {"revision": "R00", "proposal": "P01", "decision": "no-harness-change",
                "request": request, "attempts": [], "scientific_gain_claimed": False,
                "selection_sha256": plan["selection_sha256"], "dev_attempt": plan["dev_attempt"], "single_comparison_replaces_initial": False}
    write(output / "registry.json", registry)
    args = pilot.parser().parse_args(common)
    args.reference = next(iter(plan["modes"].get("reference", [])), None)
    client = SequentialClient(binary, plan["models"], model_dir, output, deadline, pilot.DeadlineClient, environment)
    run_episodes(plan, args, client, output, deadline, manifest, registry, pilot)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    execute(Path(sys.argv[1]), float(sys.argv[2]))
