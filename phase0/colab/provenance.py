"""Fingerprint the staged campaign sources, excluding data, checkpoints and generated results."""
import hashlib
import json
import sys
from pathlib import Path


def write(stage):
    root = Path(stage) / "phase0"
    sources = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and rel.parts[0] not in ("data", "results", ".venv", "e11_validation"):
            sources[rel.as_posix()] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "size": path.stat().st_size}
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "e12_campaign_sources.json").write_text(
        json.dumps({"schema": 1, "plan": "aa-1",
                    "benches": ["gpqa", "scicode"] if (root / "data" / "gpqa_diamond.csv").exists() else ["scicode"],
                    "sources": sources}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write(sys.argv[1])
