"""Upload bounded files through the official CLI and require a fresh reconstruction receipt."""
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from reconstruct import ACK, BLOCK, CHUNK, MAX_PARTS

HERE = Path(__file__).resolve().parent


def cli(args, output=None):
    code = subprocess.call([sys.executable, str(HERE / "safe_cli.py"), *args], stdout=output)
    if code:
        if output is not None:
            output.seek(0)
            while block := output.read(65536):
                sys.stdout.buffer.write(block)  # Already sanitized by safe_cli.
            sys.stdout.flush()
        raise SystemExit(code if code >= 0 else 128 - code)


def transfer(archive):
    size = archive.stat().st_size
    if not 0 < size <= CHUNK * MAX_PARTS:
        raise ValueError("archive exceeds transfer bounds")
    attempt = uuid.uuid4().hex
    prefix = f"dllm-transfer-{attempt}"
    with tempfile.TemporaryDirectory(prefix="myriad-transfer-", dir=archive.parent) as directory:
        directory = Path(directory)
        whole, parts = hashlib.sha256(), []
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
                        whole.update(block)
                        copied += len(block)
                parts.append({"index": i, "name": name, "size": copied, "sha256": hashed.hexdigest()})
                cli(["upload", "-s", "phase0", str(path), f"/content/{name}"])
                path.unlink()
            if source.read(1):
                raise ValueError("archive changed during transfer")
        manifest = {"schema": 1, "attempt": attempt, "size": size, "sha256": whole.hexdigest(), "parts": parts}
        raw = json.dumps(manifest, sort_keys=True).encode()
        path = directory / f"{prefix}.json"
        path.write_bytes(raw)
        config = {k: manifest[k] for k in ("attempt", "size", "sha256")}
        config["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        cli(["upload", "-s", "phase0", str(path), f"/content/{path.name}"])
        script = directory / "reconstruct.py"
        script.write_text("CONFIG = " + repr(config) + "\n" + (HERE / "reconstruct.py").read_text(), encoding="utf-8")
        with tempfile.TemporaryFile() as receipt:
            cli(["exec", "-s", "phase0", "--timeout", "180", "-f", str(script)], receipt)
            receipt.seek(0)
            acknowledgement = None
            remote_error = False
            for line in receipt:
                if line.startswith(b"DLLM_TRANSFER_ERROR "):
                    remote_error = True
                if line.startswith(ACK.encode()):
                    if acknowledgement is not None:
                        raise ValueError("duplicate reconstruction acknowledgement")
                    acknowledgement = json.loads(line[len(ACK):])
            if (remote_error or not isinstance(acknowledgement, dict)
                    or type(acknowledgement.get("schema")) is not int
                    or acknowledgement != {"schema": 1, "status": "ok", **config}):
                receipt.seek(0)
                while block := receipt.read(65536):
                    sys.stdout.buffer.write(block)
                sys.stdout.flush()
                raise ValueError("missing, stale or invalid reconstruction acknowledgement")
        print(f"archive transfer verified: {size} bytes, SHA-256 {whole.hexdigest()}, {len(parts)} parts", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        transfer(Path(sys.argv[1]))
    except (ValueError, OSError) as exc:
        print(f"archive transfer refused: {type(exc).__name__}", file=sys.stderr)
        sys.exit(65)
