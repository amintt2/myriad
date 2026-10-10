"""Append-only wall observations, bound to a result manifest and published without regression."""
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


def atomic(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = f.name
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def evidence(path, raw):
    atomic(str(path) + ".rejected-" + hashlib.sha256(raw).hexdigest(), raw)


def header(result, meta):
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"kind": "timing_header", "schema": 1, "result": Path(result).name, "protocol_sha256": digest}


def parse(raw, expected=None):
    # A complete final object without LF is recoverable; a fragment anywhere is not.
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows:
        raise ValueError("empty timing trace")
    if expected is not None and rows[0] != expected:
        raise ValueError("timing trace does not match result/protocol")
    observations = rows[1:] if rows[0].get("kind") == "timing_header" else rows
    for row in observations:
        wall = row.get("wall_s") if isinstance(row, dict) else None
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            raise ValueError("invalid wall observation")
    return rows


def coverage(rows, result):
    observed = {r.get("id") for r in rows if r.get("kind") == "call"}
    ids = {json.loads(line)["id"] for line in Path(result).read_bytes().splitlines()}
    if not ids <= observed:
        raise ValueError("result observations missing; measurement total unknown")


def initialize(out):
    if hasattr(out, "_timing_path"):
        return
    path = Path(str(out.path) + ".timing.jsonl")
    meta_path = Path(str(out.path) + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    expected = header(out.path, meta) if meta and meta.get("measurements") == "e12-wall-v2" else None
    if path.exists():
        raw = path.read_bytes()
        try:
            rows = parse(raw, expected)
            if expected and Path(out.path).exists():
                coverage(rows, out.path)
        except (ValueError, UnicodeError, AttributeError) as e:
            evidence(path, raw)
            raise ValueError("timing incomplete/corrupt; token totals unknown; resume refused before inference") from e
        if not raw.endswith(b"\n"):
            atomic(path, raw + b"\n")
    else:
        if Path(out.path).exists() and Path(out.path).stat().st_size:
            raise ValueError("results exist without timing; total unknown; resume refused before inference")
        if expected is not None:
            atomic(path, (json.dumps(expected) + "\n").encode())
        else:
            atomic(path, b"")  # Synthetic callers without a protocol manifest.
    out._timing_path = path


def append(out, row):
    initialize(out)
    with out._timing_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def publish(source, destination, meta_path):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    meta = json.loads(Path(meta_path).read_text())
    result = str(destination).removesuffix(".timing.jsonl")
    expected = header(result, meta) if meta.get("measurements") == "e12-wall-v2" else None
    try:
        rows = parse(raw, expected)
        if destination.exists():
            old = parse(destination.read_bytes(), expected)
            if rows[:len(old)] != old:
                raise ValueError("timing recovery would lose or replace existing observations")
    except (ValueError, UnicodeError, AttributeError) as e:
        evidence(destination, raw)
        raise ValueError("timing recovery refused; complete totals not confirmed") from e
    atomic(destination, raw if raw.endswith(b"\n") else raw + b"\n")


if __name__ == "__main__":
    publish(*sys.argv[1:])
