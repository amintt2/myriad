"""Resume content-addressed chunks, verify remote hashes and require a fresh reconstruction receipt."""
import contextlib
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

from reconstruct import ACK, BLOCK, CHUNK, MAX_PARTS
from safe_cli import run

HERE = Path(__file__).resolve().parent
TRANSIENT = re.compile(r"name resolution|connection|timed? ?out|timeout|HTTP (?:408|429|5[0-9]{2})", re.I)


class CLIError(Exception):
    def __init__(self, code, transient):
        self.code, self.transient = code, transient


def cli(args, output=None):
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as out, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run(args)
        err.seek(0)
        diagnostic = err.read()
        sys.stderr.write(diagnostic)  # Already sanitized by safe_cli.
        transient = code == 124 or bool(TRANSIENT.search(diagnostic))
        out.seek(0)
        if code or output is None:
            while block := out.read(65536):
                sys.stdout.write(block)
                transient |= bool(TRANSIENT.search(block))
            sys.stdout.flush()
        else:
            while block := out.read(65536):
                output.write(block.encode("utf-8"))
        if code:
            raise CLIError(code, transient)


def pause(attempt):
    time.sleep(min(8, float(os.environ.get("DLLM_RETRY_SECONDS", "1")) * 2 ** attempt))


def remote_result(script, marker):
    with tempfile.TemporaryFile() as receipt:
        cli(["exec", "-s", "phase0", "--timeout", "180", "-f", str(script)], receipt)
        receipt.seek(0)
        values, error = [], False
        for line in receipt:
            error |= line.startswith(b"DLLM_TRANSFER_ERROR ")
            if line.startswith(marker.encode()):
                values.append(json.loads(line[len(marker):]))
        if error or len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("missing, duplicate or invalid remote acknowledgement")
        return values[0]


def verified_upload(path, remote, directory):
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size < BLOCK else None
    if digest is None:
        hashed = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(BLOCK): hashed.update(block)
        digest = hashed.hexdigest()
    script = directory / "verify_part.py"
    for attempt in range(4):
        config = {"name": Path(remote).name, "size": path.stat().st_size, "sha256": digest,
                  "challenge": uuid.uuid4().hex}
        script.write_text("CONFIG = " + repr(config) + "\n" + (HERE / "reconstruct.py").read_text()
                          .split('if __name__ == "__main__":')[0] + '\n'
                          + "print('DLLM_PART_ACK ' + json.dumps({**CONFIG, 'matches': "
                          + "matches(Path('/content') / CONFIG['name'], CONFIG['size'], CONFIG['sha256'])}))\n",
                          encoding="utf-8")
        try:
            result = remote_result(script, "DLLM_PART_ACK ")
            if (type(result.get("matches")) is bool and type(result.get("size")) is int
                    and result == {**config, "matches": True}):
                return
            if (type(result.get("matches")) is not bool or type(result.get("size")) is not int
                    or result != {**config, "matches": False}):
                raise ValueError("invalid part verification")
            if attempt == 3:
                break  # Final reconciliation with a fresh challenge, never a fourth upload.
            try:
                cli(["upload", "-s", "phase0", str(path), remote])
            except CLIError as exc:
                if not exc.transient:
                    raise
                # Probe again before any retry: upload may have succeeded despite a lost response.
                last = exc
                if attempt < 3:
                    pause(attempt)
                continue
            result = remote_result(script, "DLLM_PART_ACK ")
            if (type(result.get("matches")) is bool and type(result.get("size")) is int
                    and result == {**config, "matches": True}):
                return
            if (type(result.get("matches")) is not bool or type(result.get("size")) is not int
                    or result != {**config, "matches": False}):
                raise ValueError("invalid part verification")
            last = ValueError("remote part corrupted")
        except CLIError as exc:
            if not exc.transient:
                raise
            last = exc
        if attempt < 3:
            pause(attempt)
    raise last


def transfer(archive):
    size = archive.stat().st_size
    if not 0 < size <= CHUNK * MAX_PARTS:
        raise ValueError("archive exceeds transfer bounds")
    whole = hashlib.sha256()
    with archive.open("rb") as source:
        while block := source.read(BLOCK): whole.update(block)
    attempt = whole.hexdigest()[:32]
    prefix = f"dllm-transfer-{attempt}"
    with tempfile.TemporaryDirectory(prefix="myriad-transfer-", dir=archive.parent) as directory:
        directory = Path(directory)
        parts = []
        with archive.open("rb") as source:
            for i in range((size + CHUNK - 1) // CHUNK):
                name = f"{prefix}.part-{i:06d}"
                path = directory / name
                hashed, copied = hashlib.sha256(), 0
                with path.open("wb") as output:
                    while copied < min(CHUNK, size - i * CHUNK):
                        block = source.read(min(BLOCK, min(CHUNK, size - i * CHUNK) - copied))
                        if not block:
                            raise ValueError("archive changed during transfer")
                        output.write(block)
                        hashed.update(block)
                        copied += len(block)
                parts.append({"index": i, "name": name, "size": copied, "sha256": hashed.hexdigest()})
                verified_upload(path, f"/content/{name}", directory)
                path.unlink()
            if source.read(1):
                raise ValueError("archive changed during transfer")
        manifest = {"schema": 1, "attempt": attempt, "size": size, "sha256": whole.hexdigest(), "parts": parts}
        raw = json.dumps(manifest, sort_keys=True).encode()
        path = directory / f"{prefix}.json"
        path.write_bytes(raw)
        verified_upload(path, f"/content/{path.name}", directory)
        config = {k: manifest[k] for k in ("attempt", "size", "sha256")}
        config["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        script = directory / "reconstruct.py"
        for retry in range(3):
            fresh = {**config, "challenge": uuid.uuid4().hex}
            script.write_text("CONFIG = " + repr(fresh) + "\n" + (HERE / "reconstruct.py").read_text(),
                              encoding="utf-8")
            try:
                result = remote_result(script, ACK)
                if (type(result.get("schema")) is not int or type(result.get("size")) is not int
                        or result != {"schema": 1, "status": "ok", **fresh}):
                    raise ValueError("stale or invalid reconstruction acknowledgement")
                break
            except CLIError as exc:
                if not exc.transient or retry == 2:
                    raise
                pause(retry)
        print(f"archive transfer verified: {size} bytes, SHA-256 {whole.hexdigest()}, {len(parts)} parts", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        transfer(Path(sys.argv[1]))
    except CLIError as exc:
        sys.exit(exc.code)
    except (ValueError, OSError) as exc:
        print(f"archive transfer refused: {type(exc).__name__}", file=sys.stderr)
        sys.exit(65)
