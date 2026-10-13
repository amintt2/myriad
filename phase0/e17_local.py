"""Offline DEV R00: trusted WSL controller, Windows Vulkan server over an owned stdio worker."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import platform
import queue
import selectors
import secrets
import socket
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

from e17_campaign import GRANITE, LIMITS, MODELS, QWEN, R00_AGENT_SHA256, TASKS, canonical, digest
from e17_vm import SequentialClient, dev_manifest

ROOT = Path(__file__).resolve().parent
ASSET = {"file": "llama-b11505-bin-win-vulkan-x64.zip", "bytes": 33442611,
         "sha256": "ace19f118b9e32382ec9b9b123aa19e8fdf54981fb82e82977b5298efaf01d49"}
WIRE_LIMIT = 2 << 20
BIGCODE_TASKS = ["BigCodeBench_120", "BigCodeBench_184"]


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            sha.update(block)
    return sha.hexdigest()


def save(path, value):
    path.write_bytes(canonical(value))


def win_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()


def owned_job():
    """Place this worker and all future server children in a kill-on-close Windows job."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION on the required Windows x64 runtime.
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("Windows x64 worker required")
    limits = ctypes.create_string_buffer(144)
    ctypes.c_uint32.from_buffer(limits, 16).value = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not handle or not kernel.SetInformationJobObject(handle, 9, limits, len(limits))
            or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess())):
        raise ctypes.WinError(ctypes.get_last_error())
    return handle  # Intentionally retained until worker exit; never inherited by children.


def inspect_assets(config):
    import importlib.metadata
    binary, archive = Path(config["binary"]), Path(config["archive"])
    if archive.stat().st_size != ASSET["bytes"] or file_hash(archive) != ASSET["sha256"]:
        raise ValueError("Windows Vulkan archive differs from the pinned release")
    files = {}
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir():
                continue
            parts = Path(entry.filename).parts
            if Path(entry.filename).is_absolute() or ".." in parts or entry.file_size > 128 << 20:
                raise ValueError("unsafe release member")
            target = binary.parent / entry.filename
            expected = digest(source.read(entry))
            if target.is_symlink() or not target.is_file() or file_hash(target) != expected:
                raise ValueError("installed server file differs from release: " + entry.filename)
            files[entry.filename] = expected
    if binary.name != "llama-server.exe" or binary.name not in files:
        raise ValueError("expected release server binary missing")
    for name in config["models"]:
        model = MODELS[name]
        path = Path(config["models_dir"]) / model["file"]
        if path.is_symlink() or path.stat().st_size != model["bytes"] or file_hash(path) != model["sha256"]:
            raise ValueError("local GGUF differs from catalogue")
    observations = {}
    for option in ("--version", "--list-devices"):
        result = subprocess.run([str(binary), option], capture_output=True, text=True, timeout=30, check=True)
        observations[option] = result.stdout + result.stderr
    if "11505" not in observations["--version"] or "ff5888f99" not in observations["--version"]:
        raise ValueError("server build mismatch")
    devices = observations["--list-devices"]
    if not any(line.strip().startswith("Vulkan0: AMD Radeon RX 6650 XT (8176 MiB,")
               for line in devices.splitlines()):
        raise ValueError("expected RX 6650 XT Vulkan0 device unavailable; CPU fallback refused")
    return {"server_files": files, "observations": observations, "models": config["models"],
            "windows_python": sys.version, "windows_python_sha256": file_hash(Path(sys.executable)),
            "packages": {"httpx": importlib.metadata.version("httpx")},
            "binary_sha256": file_hash(binary), "archive": ASSET}


class WindowsClient(SequentialClient):
    """Reuse sequential accounting, event log and bounded HTTP requests; replace only server lifecycle."""
    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.event({"event": "stopped", "model": self.model, "returncode": self.process.returncode})
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.client = self.model = None

    def post(self, path, json, timeout):
        model = json["model"]
        if model not in self.models or path != "chat/completions" or timeout <= 0:
            raise ValueError("undeclared model/endpoint or expired request")
        deadline = min(self.deadline, time.monotonic() + timeout)
        if self.model != model:
            started = time.monotonic()
            self.stop()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            args = [str(self.binary), "-m", str(self.directory / self.models[model]["file"]),
                    "--alias", model, "--host", "127.0.0.1", "--port", str(port), "--jinja",
                    "-ngl", "999", "--device", "Vulkan0", "-c", "32768", "-np", "1",
                    "--reasoning-budget", "0", "--no-webui"]
            # Each load has a distinct closed log; a previous model's evidence is never truncated.
            index = len(list(self.output.glob("server-*.log")))
            self.log = (self.output / f"server-{index:04d}.log").open("xb")
            self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            self.model = model
            self.event({"event": "started", "model": model, "parameters": args[3:]})
            import httpx
            try:
                while True:
                    if self.process.poll() is not None:
                        raise RuntimeError("Windows sequential server exited during startup")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Windows sequential server loading exhausted the request budget")
                    try:
                        with httpx.Client(trust_env=False) as client:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("Windows health client creation exhausted the request budget")
                            response = client.get(f"http://127.0.0.1:{port}/health",
                                                  timeout=min(1, remaining))
                        if response.status_code == 200:
                            break
                    except httpx.ConnectError as error:
                        cause = error
                        seen = set()
                        # Prefer explicit causes; from None can leave the system error only in context.
                        for _ in range(16):
                            if id(cause) in seen or isinstance(cause, OSError):
                                break
                            seen.add(id(cause))
                            next_cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
                            if next_cause is None:
                                break
                            cause = next_cause
                        else:
                            raise  # More than 16 exception nodes is not a confirmed startup refusal.
                        # A not-yet-listening owned server is expected during startup; other failures are not.
                        if not isinstance(cause, OSError) or cause.errno not in (errno.ECONNREFUSED, 10061):
                            raise
                    except httpx.TimeoutException:
                        pass
                    time.sleep(min(.2, max(0, deadline - time.monotonic())))
                self.client = self.client_factory(f"http://127.0.0.1:{port}/v1", "")
            finally:
                seconds = time.monotonic() - started
                self.loading_seconds += seconds
                self.event({"event": "load", "model": model, "seconds": seconds})
        return super().post(path, json, deadline - time.monotonic())


class WorkerInput:
    """Poll the Windows pipe without acquiring Python's buffered stdin lock during shutdown."""
    def __init__(self):
        import ctypes
        import msvcrt
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.fd = sys.stdin.fileno()
        self.handle = msvcrt.get_osfhandle(self.fd)
        self.pending, self.stopping = queue.Queue(), threading.Event()
        self.thread = threading.Thread(target=self.read)
        self.thread.start()

    def read(self):
        import ctypes
        data = bytearray()
        while not self.stopping.is_set():
            available = ctypes.c_ulong()
            if not self.kernel.PeekNamedPipe(self.handle, None, 0, None, ctypes.byref(available), None):
                if self.stopping.is_set():
                    return
                # Abnormal controller disappearance must kill the owned job even during HTTP generation.
                os._exit(143)
            if not available.value:
                self.stopping.wait(.02)
                continue
            block = os.read(self.fd, min(available.value, 65536))
            if not block:
                os._exit(143)
            data.extend(block)
            while b"\n" in data:
                line, _, remaining = data.partition(b"\n")
                data = bytearray(remaining)
                self.pending.put(bytes(line) + b"\n")
            if len(data) > WIRE_LIMIT:
                self.pending.put(b"x" * (WIRE_LIMIT + 1))
                return

    def get(self, timeout):
        return self.pending.get(timeout=timeout)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Windows stdin reader exit unconfirmed")


def loopback_client(url, key):
    from essaim.code_pilot import DeadlineClient
    if key:
        raise ValueError("local worker accepts no credential")
    # The reused client always emits Bearer; a public placeholder avoids an illegal trailing-space header.
    return DeadlineClient(url, "local-loopback")


def worker():
    if os.name != "nt":
        raise RuntimeError("worker requires Windows")
    job = owned_job()
    pending = WorkerInput()
    client = None
    try:
        line = pending.get(timeout=30)
        if len(line) > WIRE_LIMIT:
            raise ValueError("worker configuration too large")
        config = json.loads(line)
        session = config["session"]
        if not isinstance(session, str) or len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
            raise ValueError("invalid stdio session identity")
        deadline = time.monotonic() + 7200
        if config["models"] not in ([QWEN], [QWEN, GRANITE]):
            raise ValueError("only fixed DEV peers supported")
        observed = inspect_assets(config)
        if not config["execute"]:
            pending.close()
            print(json.dumps({"session": session, "id": 0, "ready": observed}), flush=True)
            return
        print(json.dumps({"session": session, "id": 0, "ready": observed}), flush=True)
        client = WindowsClient(Path(config["binary"]), {m: MODELS[m] for m in config["models"]},
                               Path(config["models_dir"]), Path(config["output"]),
                               deadline, loopback_client)
        previous = 0
        while True:
            line = pending.get(timeout=min(1020, max(.001, deadline - time.monotonic())))
            if len(line) > WIRE_LIMIT:
                raise ValueError("worker request too large")
            request = json.loads(line)
            if (request.get("session") != session or type(request.get("id")) is not int
                    or request["id"] != previous + 1):
                raise ValueError("foreign, duplicated or reordered stdio request")
            previous = request["id"]
            identity = {"session": session, "id": previous}
            if request == {**identity, "close": True}:
                pending.close()
                client.close()
                client = None
                print(json.dumps({**identity, "closed": True}), flush=True)
                return
            timeout = request.get("timeout")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= LIMITS["timeout"]:
                raise ValueError("invalid R00 request budget")
            if deadline <= time.monotonic():
                raise TimeoutError("Windows worker global deadline")
            import httpx
            try:
                response = client.post("chat/completions", request["json"],
                                       min(timeout, deadline - time.monotonic()))
                result = {"outcome": "response", "response": response.json()}
            except (TimeoutError, httpx.TimeoutException):
                # Cancel any still-running inference before acknowledging the deadline or accepting a new request.
                client.stop()
                result = {"outcome": "generation_timeout"}
            except Exception as error:
                client.stop()
                result = {"outcome": "infrastructure", "exception_type": type(error).__name__}
            result.update(identity, loading_seconds=client.loading_seconds, request_seconds=client.request_seconds)
            encoded = json.dumps(result)
            if len(encoded.encode()) >= WIRE_LIMIT:
                raise ValueError("worker response too large")
            print(encoded, flush=True)
    finally:
        pending.close()
        if client is not None:
            client.close()
        # Job handle remains live until process exit, including exceptional exits.
        assert job


class PipeClient:
    def __init__(self, command, config, log):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        self.loading_seconds = self.request_seconds = 0.0
        self.session, self.sequence = secrets.token_hex(16), 0
        self.pending, self.buffer = 0, bytearray()
        self.deadline, self.broken, self.normal_exit = time.monotonic() + 7200, False, False
        self.execute = config["execute"]
        self.write({**config, "session": self.session})

    def write(self, value):
        data = canonical(value) + b"\n"
        if len(data) > WIRE_LIMIT:
            raise ValueError("stdio request too large")
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def send(self, value):
        if self.pending is not None or self.broken or "session" in value or "id" in value:
            raise RuntimeError("stdio session is not synchronized")
        self.sequence += 1
        self.pending = self.sequence
        self.write({**value, "session": self.session, "id": self.pending})

    def receive(self, timeout):
        # Pipes under WSL are selectable; no TCP or Windows host address is needed.
        if self.pending is None or self.broken:
            raise RuntimeError("no valid outstanding stdio request")
        deadline = min(self.deadline, time.monotonic() + timeout)
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Windows stdio worker deadline")
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError("Windows stdio worker exited")
                self.buffer.extend(block)
                if len(self.buffer) > WIRE_LIMIT:
                    self.broken = True
                    raise ValueError("stdio response too large")
                if b"\n" in block:
                    if not self.buffer.endswith(b"\n") or self.buffer.count(b"\n") != 1:
                        self.broken = True
                        raise ValueError("unexpected extra stdio response")
                    try:
                        result = json.loads(self.buffer)
                        if not isinstance(result, dict):
                            raise ValueError("stdio response must be an object")
                    except ValueError:
                        self.broken = True
                        raise
                    self.buffer.clear()
                    if (result.get("session") != self.session or type(result.get("id")) is not int
                            or result["id"] != self.pending):
                        self.broken = True
                        raise ValueError("foreign, stale or reordered stdio response")
                    self.pending = None
                    self.normal_exit = result.get("closed") is True or "ready" in result and not self.execute
                    return result

    def accounting(self, result):
        for name in ("loading_seconds", "request_seconds"):
            value = result.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < getattr(self, name):
                self.broken = True
                raise ValueError("invalid sequential server accounting")
            setattr(self, name, value)

    def synchronize(self, timeout=15):
        if self.broken:
            raise RuntimeError("broken stdio session")
        if self.pending is not None:
            result = self.receive(min(15, timeout))
            self.accounting(result)  # A late response is drained only; it never supplies the next task's answer.
            if result.get("outcome") not in ("response", "generation_timeout"):
                self.broken = True
                raise RuntimeError("late stdio response reports infrastructure failure: " +
                                   str(result.get("exception_type")))

    def post(self, path, json, timeout):
        if path != "chat/completions" or json.get("model") not in (QWEN, GRANITE):
            raise ValueError("undeclared request")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= LIMITS["timeout"]:
            raise ValueError("invalid R00 request budget")
        deadline = min(self.deadline, time.monotonic() + timeout)
        self.synchronize(max(0, deadline - time.monotonic()))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("stdio synchronization consumed the episode budget")
        self.send({"json": json, "timeout": remaining})
        result = self.receive(max(0, deadline - time.monotonic()))
        self.accounting(result)
        if result.get("outcome") == "generation_timeout":
            from essaim.agent import GenerationDeadline
            raise GenerationDeadline("Windows generation exhausted its R00 request budget")
        if result.get("outcome") != "response":
            raise RuntimeError("Windows worker infrastructure failure: " + str(result.get("exception_type")))
        return SimpleNamespace(json=lambda: result["response"], raise_for_status=lambda: None, headers={})

    def close(self):
        # EOF reaches the worker's reader independently of an in-flight model call.
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
            if self.normal_exit and self.process.returncode != 0:
                raise RuntimeError("Windows worker failed after its normal shutdown acknowledgment")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Windows worker exit unconfirmed; no success may be reported")
        finally:
            self.process.stdout.close()


def benchmark_adapter(benchmark):
    if benchmark == "aider":
        from essaim import code_pilot
        return code_pilot, TASKS, "code_pilot"
    if benchmark == "bigcode":
        from essaim import bigcode_pilot
        if bigcode_pilot.tasks.PILOT != BIGCODE_TASKS:
            raise ValueError("BigCodeBench is limited to the two authorized DEV tasks")
        return bigcode_pilot, BIGCODE_TASKS, "bigcode_pilot"
    raise ValueError("unsupported local DEV benchmark")


def strict_receipt(receipt):
    from essaim.local_linux import VERSION
    return (isinstance(receipt, dict) and receipt.get("isolation_ready") == VERSION
            and receipt.get("seccomp") == "on" and type(receipt.get("landlock_abi")) is int
            and receipt["landlock_abi"] >= 3
            and receipt.get("namespaces") == ["user", "mount", "pid", "net", "ipc", "uts"])


def validate_bigcode_smoke(value):
    from essaim import aider_agent, bigcode_tasks
    from essaim.local_linux import VERSION
    if (value.get("kind") != "official_unittest_dev_smoke" or value.get("model_measurement") is not False
            or value.get("trust_scope") != bigcode_tasks.TRUST_SCOPE
            or value.get("adversarial_attestation") is not False
            or [r.get("task") for r in value.get("tasks", [])] != BIGCODE_TASKS):
        raise ValueError("BigCodeBench public nonadversarial DEV smoke coverage missing")
    for row, count in zip(value["tasks"], (6, 5)):
        if set(row.get("negative", {})) != {"wrong", "exit0", "load_exception"}:
            raise ValueError("BigCodeBench negative controls missing")
        for grade, passed in [(row["reference"], True), *[(g, False) for g in row["negative"].values()]]:
            if (grade.get("status") != "graded" or grade.get("passed") is not passed
                    or grade.get("scorable") is not True or grade.get("tests_expected") != count
                    or grade.get("trust_scope") != bigcode_tasks.TRUST_SCOPE
                    or grade.get("adversarial_attestation") is not False
                    or grade.get("sandbox_backend") != "strict" or grade.get("sandbox") != VERSION
                    or not strict_receipt(grade.get("isolation")) or not grade.get("calls")
                    or passed and (not aider_agent.complete_green(grade)
                                   or any(grade.get(k) != 0 for k in
                                          ("skipped", "expected_failures", "unexpected_successes")))):
                raise ValueError("incomplete, false green or nonstrict BigCodeBench smoke")


def validate_bigcode_episode(row, task, mode):
    from essaim import aider_agent, bigcode_tasks
    grade = row.get("verifier", {})
    if (row.get("task") != task or row.get("mode") != mode
            or row.get("trust_scope") != bigcode_tasks.TRUST_SCOPE
            or row.get("adversarial_attestation") is not False):
        raise ValueError("BigCodeBench episode identity/trust scope mismatch")
    if row.get("status") != "graded":
        return  # The common controller records this incomplete attempt and refuses campaign completion.
    if (type(row.get("passed")) is not bool or grade.get("status") != "graded"
            or grade.get("trust_scope") != bigcode_tasks.TRUST_SCOPE
            or grade.get("adversarial_attestation") is not False or grade.get("sandbox_backend") != "strict"):
        raise ValueError("BigCodeBench final grade scope mismatch")
    if grade.get("candidate_failure") not in ("invalid_source_artifact", "verification_budget"):
        if not strict_receipt(grade.get("isolation")):
            raise ValueError("BigCodeBench final verifier isolation receipt missing")
    if row["passed"] and (row.get("submission_accepted") is not True or not aider_agent.complete_green(grade)
                          or grade.get("tests_expected") != (6 if task == BIGCODE_TASKS[0] else 5)):
        raise ValueError("BigCodeBench success requires complete official verification")


def validate_smoke(value, benchmark="aider"):
    if benchmark == "bigcode":
        validate_bigcode_smoke(value)
        return
    benchmark_adapter(benchmark)
    from e17_vm import SMOKE_TESTS
    if (value.get("sandbox_backend") != "strict" or value.get("model_measurement") is not False
            or [row.get("task") for row in value.get("tasks", [])] != TASKS):
        raise ValueError("strict official DEV smoke coverage missing")
    for row in value["tasks"]:
        for key, passed in (("reference", True), ("incorrect", False)):
            grade = row[key]
            if (grade.get("status") != "graded" or grade.get("passed") is not passed
                    or not grade.get("calls")
                    or grade.get("tests_expected") != SMOKE_TESTS[row["task"]]
                    or passed and (grade.get("tests_complete") is not True
                                   or grade.get("tests") != SMOKE_TESTS[row["task"]])):
                raise ValueError("incomplete or false green official smoke")
            for call in grade.get("calls", []):
                receipt = call.get("isolation", {})
                if (call.get("sandbox_backend") != "strict" or receipt.get("seccomp") != "on"
                        or not isinstance(receipt.get("landlock_abi"), int) or receipt["landlock_abi"] < 3):
                    raise ValueError("strict verifier isolation receipt missing")


def stage_dev(cache, output, benchmark="aider"):
    if benchmark == "bigcode":
        from essaim import bigcode_tasks
        from essaim.terminal_bench import task_path
        benchmark_adapter(benchmark)
        view = bigcode_tasks.provenance()
        expected = {t + ".json" for t in BIGCODE_TASKS}
        if set(view["files"]) != expected or view["dev"] != BIGCODE_TASKS:
            raise ValueError("BigCodeBench staging must contain exactly DEV 120/184")
        rows = {}
        for name in sorted(expected):
            raw = task_path(cache, name).read_bytes()
            if bigcode_tasks.digest(raw) != view["files"][name]:
                raise ValueError("existing BigCodeBench DEV asset differs from original pin")
            rows[name] = raw
        save(output / "dataset-manifest.json", view)
        data = output / "data"
        data.mkdir()
        for name, raw in rows.items():
            (data / name).write_bytes(raw)
        bigcode_tasks.check_data(data)
        return data, view
    benchmark_adapter(benchmark)
    from essaim import aider_python
    from essaim.terminal_bench import check_tasks, task_path
    view = dev_manifest(aider_python.provenance())
    save(output / "dataset-manifest.json", view)
    data = output / "data"
    for name, info in view["files"].items():
        source = task_path(cache, name)
        if source.stat().st_size != info["size"] or file_hash(source) != info["sha256"]:
            raise ValueError("existing DEV asset differs from original pin")
        target = task_path(data, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    aider_python.MANIFEST = output / "dataset-manifest.json"
    check_tasks(data, view)
    return data, view


def execute(args):
    if not sys.platform.startswith("linux") or "microsoft" not in platform.release().lower():
        raise ValueError("trusted controller requires WSL strict Landlock")
    if Path(sys.prefix).resolve() != args.runtime.resolve():
        raise ValueError("use the existing dedicated pilot runtime interpreter")
    agent_source = (ROOT / "essaim/aider_agent.py").read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    if digest(agent_source) != R00_AGENT_SHA256:
        raise ValueError("R00 changed; register a revision before any measurement")
    benchmark = getattr(args, "benchmark", "aider")
    pilot, tasks, resource = benchmark_adapter(benchmark)
    args.output.mkdir()  # Exclusive fresh directory, including preflight and smoke.
    from essaim import code_pilot
    from essaim.local_linux import LocalLinuxEnv
    import importlib.metadata
    import tempfile
    modes = {"single": [QWEN]} if args.mode == "single" else {"vote": [QWEN, GRANITE]}
    models = [QWEN] if args.mode == "single" else [QWEN, GRANITE]
    registry = {"partition": "DEV", "revision": "R00", "proposal": "P01", "attempts": [],
                "simulation": "local-sequential", "distributed": False, "scientific_gain_claimed": False,
                "mode": args.mode, "unknown_modes": [m for m in ("single", "vote", "cascade", "reference")
                                                       if m != args.mode], "complete": False}
    context = {"benchmark": benchmark, "partition": "DEV", "tasks": tasks,
               "test_frozen": False, "ready_for_test": False}
    if benchmark == "bigcode":
        context.update(adapter_version=pilot.tasks.VERSION, trust_scope=pilot.tasks.TRUST_SCOPE,
                       adversarial_attestation=False)
    registry.update(context)
    registry["metadata_hashes"] = {name: file_hash(ROOT.parent / "docs" / name) for name in
                                   ("e17_proposition_p01.md", "e17_selection_pre_mesure.json")}
    save(args.output / "registry.json", registry)
    client = None
    with (args.output / "worker.log").open("xb") as log:
        try:
            cache, view = None, None
            if benchmark == "bigcode":
                # Check runtime pins and stage only authorized DEV before even starting the Windows relay.
                pilot.tasks.check_runtime()
                cache, view = stage_dev(args.tasks_dir, args.output, benchmark)
            with tempfile.TemporaryDirectory(prefix="myriad-local-preflight-") as temporary:
                env = LocalLinuxEnv(Path(temporary), args.runtime.resolve())
                try:
                    code, text = env.run("printf E17-STRICT-OK", 30)
                    if code != 0 or text != "E17-STRICT-OK":
                        raise ValueError("strict WSL sandbox preflight failed")
                    isolation = env.isolation
                finally:
                    env.close()
            config = {"binary": args.binary, "archive": args.archive, "models_dir": args.models_dir,
                      "models": models, "output": win_path(args.output), "execute": args.action == "run"}
            client = PipeClient([str(args.windows_python), win_path(Path(__file__)), "--worker"], config, log)
            observed = client.receive(180)["ready"]
            sources = ["code_pilot", "aider_agent", "aider_python", "agent", "local_linux", "code_provenance",
                       "terminal_bench", "ephemeral_vm", "sandbox"]
            packages = ("httpx", "pytest", "matplotlib")
            if benchmark == "bigcode":
                sources += ["bigcode_pilot", "bigcode_tasks", "bigcode_unittest"]
                packages = json.loads((ROOT / resource / "runtime.json").read_text())["packages"]
            runtime = {**context, "windows": observed, "python": sys.version,
                       "python_sha256": file_hash(Path(sys.executable)),
                       "lock_sha256": file_hash(ROOT / resource / "uv.lock"), "sandbox_backend": "strict",
                       "isolation": isolation, "kernel": platform.release(),
                       "packages": {p: importlib.metadata.version(p) for p in packages},
                       "sources": {p.name: file_hash(p) for p in [Path(__file__), ROOT / "e17_vm.py",
                           ROOT / "e17_campaign.py", *[ROOT / "essaim" / (n + ".py") for n in
                           sources]]}}
            save(args.output / "runtime.json", runtime)
            if args.action == "preflight":
                return 0
            if benchmark == "aider":
                cache, view = stage_dev(args.tasks_dir, args.output)
            smoke = (pilot.tasks.smoke if benchmark == "bigcode" else pilot.smoke)(cache, args.runtime)
            save(args.output / "smoke.json", smoke)
            validate_smoke(smoke, benchmark)
            if args.action == "smoke":
                return 0
            hardware = {"cpu": "Intel Core i5-10400F", "host_memory_gib": 32,
                        "execution": "Windows Vulkan server, trusted WSL controller; sequential local simulation",
                        "gpu": "AMD Radeon RX 6650 XT 8 GiB", "cpu_threads": "runtime default"}
            provenance = {"models": {m: {**MODELS[m], "model": m, "alias": m, "licence": "apache-2.0",
                "declared_at": "local DEV R00 before score", "endpoint_kind": "direct-loopback-stdio-sequential",
                "quantization": "Q8_0", "hardware": hardware,
                "server": {"version": "b11505", "commit": "ff5888f99", "artifact": ASSET["file"],
                           "sha256": ASSET["sha256"], "context": 32768, "parallel": 1, "gpu_layers": 999,
                           "reasoning_budget": 0, "jinja": True},
                "energy": {"watts": 100, "eur_kwh": .25, "measured": False, "scope": "assumed whole local PC"}}
                for m in models}}
            save(args.output / "provenance.json", provenance)
            from essaim.code_provenance import declarations
            declarations(args.output / "provenance.json", sorted(models), 100, .25)
            common = ["--runtime", str(args.runtime), "--tasks-dir", str(cache), "--sandbox-backend", "strict"]
            for key, value in LIMITS.items():
                common.extend(("--" + key.replace("_", "-"), str(value)))
            run_args = pilot.parser().parse_args(common)
            manifest = {**context, "runtime": runtime, "execution_provenance": provenance, "limits": LIMITS,
                        "models": modes, "tasks": tasks, "dataset_hash": code_pilot.hash_json(view),
                        "provenance_sha256": file_hash(args.output / "provenance.json"),
                        "system": pilot.aider_agent.SYSTEM, "sandbox_backend": "strict",
                        "isolation": isolation, "simulation": "local-sequential"}
            save(args.output / "manifest.json", manifest)
            rows = []
            for task in tasks:
                registry["attempts"].append({"task": task, "mode": args.mode, "state": "started"})
                save(args.output / "registry.json", registry)
                loading_before, requests_before = client.loading_seconds, client.request_seconds
                row = pilot.run_one(run_args, task, args.mode, models, client, args.output,
                                    code_pilot.hash_json(manifest))
                rows.append(row)
                save(args.output / "verdicts.json", rows)
                if benchmark == "bigcode":
                    validate_bigcode_episode(row, task, args.mode)
                client.synchronize()  # Seal this episode's transport before recording times or starting another task.
                registry["attempts"][-1].update(state=row["status"], transcript_sha256=row["transcript_sha256"],
                                               loading_seconds=client.loading_seconds - loading_before,
                                               request_seconds=client.request_seconds - requests_before)
                save(args.output / "registry.json", registry)
                if row["status"] != "graded":
                    raise RuntimeError("incomplete DEV episode; no score")
            client.send({"close": True})
            if client.receive(15).get("closed") is not True:
                raise RuntimeError("owned server closure unconfirmed")
            client.close()
            client = None
            registry["complete"] = True
            save(args.output / "registry.json", registry)
            # No article score or generic report: these are separate DEV attempts with unknown other modes.
            return 0
        except BaseException as error:
            save(args.output / "failure.json", {"complete": False, "error": type(error).__name__,
                                                "score": None})
            raise
        finally:
            if client is not None:
                client.close()


def main():
    if sys.argv[1:] == ["--worker"]:
        worker()
        return 0
    cli = argparse.ArgumentParser(description="E17 DEV local strict, sources et modèles existants seulement")
    cli.add_argument("action", choices=("preflight", "smoke", "run"))
    cli.add_argument("--benchmark", choices=("aider", "bigcode"), default="aider")
    cli.add_argument("--mode", choices=("single", "vote"), default="single")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--runtime", type=Path, required=True)
    cli.add_argument("--windows-python", type=Path, required=True)
    cli.add_argument("--binary", required=True)
    cli.add_argument("--archive", required=True)
    cli.add_argument("--models-dir", required=True)
    cli.add_argument("--tasks-dir", type=Path, required=True)
    return execute(cli.parse_args())


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
