"""VM-only bootstrap and bounded artifact RPC; identity is controller ownership, not candidate attestation."""
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(".part")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def root_for(request):
    import re
    if not re.fullmatch("[0-9a-f]{32}", request["campaign_id"]):
        raise ValueError("invalid campaign identity")
    return Path("/content/myriad-e17-" + request["campaign_id"])


def ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def rpc(request, operation, payload=None):
    root = root_for(request)
    if operation == "start":
        if any(Path("/content").glob("myriad-e17-*")):
            return {"request": request, "operation": operation, "started": False, "foreign_campaign": True}
        root.mkdir()  # Never adopt a directory or a process left by another attempt.
        save(root / "request.json", request)
        source = root / "phase0"
        source.mkdir()
        import zlib
        raw = zlib.decompress(base64.b64decode(payload))
        if len(raw) > 16 << 20:
            raise ValueError("source transfer too large")
        files = json.loads(raw)
        if set(files) != set(request["sources"]):
            raise ValueError("source coverage mismatch")
        for name, content in files.items():
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts or "\\" in name:
                raise ValueError("unsafe source path")
            data = base64.b64decode(content, validate=True)
            if hashlib.sha256(data).hexdigest() != request["sources"][name]:
                raise ValueError("uploaded source hash mismatch")
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        output = root / "artifacts"
        output.mkdir()
        with (root / "bootstrap.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, str(source / "colab/e17_campaign_remote.py"),
                                        "--worker", str(root)], stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        owner = {"pid": process.pid, "ticks": ticks(process.pid), "request": request}
        save(root / "worker.json", owner)
        return {"request": request, "operation": operation, "started": True}
    if json.loads((root / "request.json").read_text()) != request:
        raise PermissionError("foreign remote directory")
    if operation == "cancel":
        owner = json.loads((root / "worker.json").read_text())
        if owner["request"] != request:
            raise PermissionError("foreign worker")
        if not (root / "inventory.json").exists():
            try:
                if ticks(owner["pid"]) != owner["ticks"] or os.getpgid(owner["pid"]) != owner["pid"]:
                    raise PermissionError("worker PID reused")
                os.kill(owner["pid"], signal.SIGTERM)
            except FileNotFoundError:
                pass
        return {"request": request, "operation": operation, "cancel_requested": True}
    if operation == "status":
        inventory = root / "inventory.json"
        return {"request": request, "operation": operation, "inventory":
                json.loads(inventory.read_text()) if inventory.exists() else None}
    if operation == "chunk":
        inventory = json.loads((root / "inventory.json").read_text())
        name, offset = payload["name"], payload["offset"]
        if name not in inventory["files"] or type(offset) is not int or offset < 0:
            raise ValueError("unknown artifact/offset")
        path = root / "artifacts" / name
        if path.is_symlink():
            raise ValueError("symlink artifact")
        with path.open("rb") as src:
            src.seek(offset)
            data = src.read(4 << 20)
        return {"request": request, "operation": operation, "name": name, "offset": offset,
                "data": base64.b64encode(data).decode()}
    raise ValueError("unsupported operation")


# Every ACK, inventory and data response uses the same bounded wire format.
# Base64 prevents artifact text from being mistaken for CLI credentials by the sanitizer.
WIRE_BYTES = 8 << 20
FRAME_CHARS = 60000


def response_lines(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    if len(raw) > WIRE_BYTES:
        raise ValueError("RPC envelope too large")
    encoded = base64.b64encode(raw).decode("ascii")
    count = (len(encoded) + FRAME_CHARS - 1) // FRAME_CHARS
    checksum = hashlib.sha256(raw).hexdigest()
    return [f"E17_FRAME {i} {count} {checksum} " + encoded[i * FRAME_CHARS:(i + 1) * FRAME_CHARS]
            for i in range(count)]


def emit_response(value):
    for line in response_lines(value):
        print(line, flush=True)


def decode_response(text):
    lines = text.splitlines()
    if not lines or len(lines) > (WIRE_BYTES * 4 // 3 // FRAME_CHARS + 2):
        raise ValueError("missing/unbounded RPC frames")
    pieces, checksum = [], None
    for i, line in enumerate(lines):
        fields = line.split(" ", 4)
        if (len(fields) != 5 or fields[0] != "E17_FRAME" or fields[1] != str(i)
                or fields[2] != str(len(lines)) or len(fields[4]) > FRAME_CHARS
                or len(fields[3]) != 64 or checksum is not None and fields[3] != checksum):
            raise ValueError("invalid/duplicate/foreign RPC frame")
        checksum = fields[3]
        pieces.append(fields[4])
    raw = base64.b64decode("".join(pieces), validate=True)
    if len(raw) > WIRE_BYTES or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("RPC envelope hash/size mismatch")
    return json.loads(raw)


def worker(root):
    sys.path.insert(0, str(root / "phase0"))
    sys.path.insert(0, str(root / "phase0/colab"))
    from read_timeout import run
    from e17_campaign import artifacts, digest, MAX_ARTIFACT_BYTES
    request = json.loads((root / "request.json").read_text())
    output = root / "artifacts"
    error = None
    deadline = time.monotonic() + request["work_seconds"]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        install = root / "installer"
        commands = [[sys.executable, "-m", "venv", str(install)],
                    [str(install / "bin/python"), "-m", "pip", "install", "uv==0.11.21"],
                    [str(install / "bin/uv"), "sync", "--frozen", "--python", "3.12.15",
                     "--project", str(root / "phase0/code_pilot")],
                    [str(root / "phase0/code_pilot/.venv/bin/python"), str(root / "phase0/e17_vm.py"), str(root),
                     str(deadline)]]
        with (output / "controller.log").open("wb") as log:
            for command in commands:
                available = deadline - time.monotonic()
                if available <= 0 or run(command, seconds=min(600, available) if command != commands[-1] else available,
                                         grace=5, deadline=deadline, stdout=log, stderr=subprocess.STDOUT):
                    raise RuntimeError("bounded VM preparation/runtime failed")
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        state = {"request": request, "state": "complete" if error is None else "incomplete", "error": error}
        save(output / "vm-status.json", state)
        files, total = {}, 0
        for name in sorted(artifacts(request["plan"])):
            path = output / name
            if path.is_file() and not path.is_symlink():
                size = path.stat().st_size
                total += size
                if total > MAX_ARTIFACT_BYTES:
                    error = "ArtifactLimit"
                    break
                files[name] = {"bytes": size, "sha256": digest(path.read_bytes())}
        save(root / "inventory.json", {"schema": 1, "request": request, "complete": error is None, "files": files})


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    worker(Path(sys.argv[2]))
