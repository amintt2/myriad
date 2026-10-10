"""Reconstruct one bounded, pinned transfer without touching model or result directories."""
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

CHUNK = 8 * 1024 * 1024
MAX_PARTS = 8192
BLOCK = 1024 * 1024
ACK = "DLLM_TRANSFER_ACK "


def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("transfer path is not a regular file")


def reconstruct(config, root=Path("/content")):
    attempt = config["attempt"]
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise ValueError("invalid attempt")
    prefix = f"dllm-transfer-{attempt}"
    manifest_path = root / f"{prefix}.json"
    regular(manifest_path)
    if manifest_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("manifest too large")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "attempt", "size", "sha256", "parts"}
            or type(manifest["schema"]) is not int or manifest["schema"] != 1):
        raise ValueError("invalid manifest schema")
    if any(manifest[k] != config[k] for k in ("attempt", "size", "sha256")):
        raise ValueError("archive identity mismatch")
    size, digest, parts = manifest["size"], manifest["sha256"], manifest["parts"]
    if type(size) is not int or not 0 < size <= CHUNK * MAX_PARTS:
        raise ValueError("invalid archive size")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid archive hash")
    if not isinstance(parts, list) or len(parts) != (size + CHUNK - 1) // CHUNK:
        raise ValueError("invalid part count")
    paths = []
    for i, part in enumerate(parts):
        name = f"{prefix}.part-{i:06d}"
        if not isinstance(part, dict) or set(part) != {"index", "name", "size", "sha256"}:
            raise ValueError("invalid part schema")
        if type(part["index"]) is not int or part["index"] != i or part["name"] != name:
            raise ValueError("invalid part index or path")
        expected_size = min(CHUNK, size - i * CHUNK)
        if type(part["size"]) is not int or part["size"] != expected_size:
            raise ValueError("invalid part size")
        if not isinstance(part["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"]):
            raise ValueError("invalid part hash")
        path = root / name
        regular(path)
        if path.stat().st_size != expected_size:
            raise ValueError("part size mismatch")
        paths.append(path)
    temporary = None
    try:
        whole = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=root, prefix=f"{prefix}.", delete=False) as output:
            temporary = Path(output.name)
            for part, path in zip(parts, paths):
                hashed, copied = hashlib.sha256(), 0
                with path.open("rb") as source:
                    while block := source.read(BLOCK):
                        copied += len(block)
                        if copied > part["size"]:
                            raise ValueError("part grew during reconstruction")
                        hashed.update(block)
                        whole.update(block)
                        output.write(block)
                if copied != part["size"] or hashed.hexdigest() != part["sha256"]:
                    raise ValueError("part hash mismatch")
            if output.tell() != size or whole.hexdigest() != digest:
                raise ValueError("archive hash mismatch")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, root / "dllm.tgz")
        temporary = None
        # Only this attempt's enumerated files, after full validation and atomic publication.
        for path in [*paths, manifest_path]:
            path.unlink()
        return {"schema": 1, "status": "ok", **config}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    # CONFIG is prepended locally to this versioned script, never read from remote state.
    try:
        result = reconstruct(CONFIG)
    except Exception as exc:
        print("DLLM_TRANSFER_ERROR " + type(exc).__name__, flush=True)
        raise SystemExit(1)
    print(ACK + json.dumps(result, sort_keys=True), flush=True)
