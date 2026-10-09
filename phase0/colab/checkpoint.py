"""Publish a downloaded cloud results file over the local copy, safely.

    python3 checkpoint.py <downloaded.jsonl> <local.jsonl>

Checks that the downloaded file has its manifest, only complete JSON lines (an incomplete
last line, being written on the VM, is dropped) and no duplicate keys; refuses to replace a
local copy that has more rows or another manifest; then publishes the pair atomically and
keeps the previous version as <local>.prev. Standard library only (runs in WSL's python3).
"""
import json
import os
import shutil
import sys


def key(r):
    """Row identity for every result kind: QCM (bench, id, run), generation (mode, id), E10 parallel
    sections: expansions (id, point) and judge verdicts (id, vs, order), E11 code: solutions (id, sample) and
    executions (id, prog)."""
    return (r.get("bench"), r.get("mode"), r.get("id"), r.get("run", 0), r.get("point"), r.get("vs"), r.get("order"),
            r.get("sample"), r.get("prog") if "sample" not in r else None)


def rows(path):
    with open(path, "rb") as f:
        raw = f.read()
    out, lines = [], raw.split(b"\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1:
                continue  # last line still being written
            raise SystemExit(f"{path}: ligne {i + 1} illisible")
    keys = [key(r) for r in out]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{path}: doublons")
    return out


def main():
    new, local = sys.argv[1], sys.argv[2]
    if not os.path.exists(new + ".meta.json"):
        raise SystemExit(f"{new}: manifeste absent")
    new_rows = rows(new)
    new_meta = json.load(open(new + ".meta.json", encoding="utf-8"))
    if os.path.exists(local):
        old_meta = json.load(open(local + ".meta.json", encoding="utf-8")) if os.path.exists(local + ".meta.json") else None
        if old_meta is not None and old_meta != new_meta:
            raise SystemExit(f"{os.path.basename(local)}: manifeste différent de la copie locale, rien remplacé")
        newer = {key(r): r for r in new_rows}
        lost = [key(r) for r in rows(local) if newer.get(key(r)) != r]
        if lost:  # every row already kept locally must be in the new copy, unchanged
            raise SystemExit(f"{os.path.basename(local)}: {len(lost)} lignes locales absentes ou différentes "
                             f"dans la nouvelle copie (ex. {lost[0]}), rien remplacé")
        shutil.copy2(local, local + ".prev")
    tmp = local + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows)
    shutil.copy2(new + ".meta.json", local + ".meta.json.tmp")
    os.replace(local + ".meta.json.tmp", local + ".meta.json")
    os.replace(tmp, local)


if __name__ == "__main__":
    main()
