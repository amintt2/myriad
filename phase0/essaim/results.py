"""Results files (JSONL) that can be resumed safely.

- A manifest (<file>.meta.json) records the configuration; resuming with a different
  configuration, or adopting a non-empty file that has no manifest, is refused.
- A lock file (owned by this process) prevents two processes from writing the same results.
- Only an incomplete final line is ever repaired, by an atomic rewrite.
- Duplicate keys are an error, both when resuming and when analysing.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: str | Path) -> dict:
    """Full SHA-256 of a weights file, cached next to it (keyed by size and mtime)."""
    p = Path(path)
    st = p.stat()
    cache = p.with_name(p.name + ".sha256.json")
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("size") == st.st_size and c.get("mtime") == st.st_mtime:
            return {"name": p.name, "size": st.st_size, "sha256": c["sha256"]}
    digest = sha256_file(p)
    _atomic_write(cache, json.dumps({"size": st.st_size, "mtime": st.st_mtime, "sha256": digest}))
    return {"name": p.name, "size": st.st_size, "sha256": digest}


def _atomic_write(path: Path, text: str):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _split_valid(raw: bytes) -> tuple[list[dict], bool]:
    """Parse complete JSON lines; returns (rows, had_incomplete_tail)."""
    rows, lines = [], raw.split(b"\n")
    tail = lines[-1]
    for i, line in enumerate(lines[:-1]):
        rows.append(json.loads(line.decode("utf-8")))  # a corrupt line in the middle is a real error
    if tail.strip():
        try:
            rows.append(json.loads(tail.decode("utf-8")))
            return rows, False
        except (json.JSONDecodeError, UnicodeDecodeError):
            return rows, True
    return rows, False


def read_rows(path: Path, key: tuple[str, ...] | None = None) -> list[dict]:
    """Rows of a results file (an incomplete final line is ignored). With `key`, duplicates raise."""
    if not path.exists():
        return []
    rows, cut = _split_valid(path.read_bytes())
    if cut:
        print(f"[results] dernière ligne incomplète ignorée dans {path.name}")
    if key:
        seen = set()
        for r in rows:
            k = tuple(r.get(x, 0) for x in key)
            if k in seen:
                raise ValueError(f"doublon {k} dans {path.name}")
            seen.add(k)
    return rows


def read_manifest(path: Path) -> dict | None:
    meta = path.with_name(path.name + ".meta.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None


class ResultsFile:
    def __init__(self, path: Path, manifest: dict, key: tuple[str, ...]):
        self.path, self.key, self.f, self._owns_lock = path, key, None, False
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_name(path.name + ".lock")
        self.token = f"{os.getpid()}:{os.urandom(8).hex()}"
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, self.token.encode())
            os.close(fd)
            self._owns_lock = True
        except FileExistsError:
            raise SystemExit(f"{path.name} est déjà en cours d'écriture (verrou {self.lock.name}). "
                             f"Si aucun processus ne l'utilise, supprime le verrou.")
        atexit.register(self.release)
        try:
            old = read_manifest(path)
            nonempty = path.exists() and path.stat().st_size > 0
            if nonempty and old is None:
                raise SystemExit(f"{path.name} existe sans manifeste : provenance inconnue, reprise refusée.")
            if old is not None and nonempty:
                diff = {k: (old.get(k), manifest.get(k)) for k in set(old) | set(manifest) if old.get(k) != manifest.get(k)}
                if diff:
                    raise SystemExit(f"Configuration différente de celle de {path.name} : {diff}. "
                                     f"Change de nom de fichier (--suffix) au lieu de mélanger deux expériences.")
            _atomic_write(path.with_name(path.name + ".meta.json"), json.dumps(manifest, indent=2, ensure_ascii=False))
            if path.exists():
                raw = path.read_bytes()
                rows, cut = _split_valid(raw)
                if cut or (raw and not raw.endswith(b"\n")):
                    # Repair atomically: drop an incomplete last line, or end a complete one with LF,
                    # so that the next append never glues two JSON objects together.
                    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                    print(f"[results] fin de fichier normalisée dans {path.name}" + (" (ligne incomplète retirée)" if cut else ""))
                read_rows(path, key)  # raises on duplicates
            self.done = {tuple(r.get(k, 0) for k in key) for r in read_rows(path)}
            self.f = open(path, "a", encoding="utf-8")
        except BaseException:
            self.release()
            raise

    def write(self, row: dict):
        k = tuple(row.get(x, 0) for x in self.key)
        if k in self.done:
            raise RuntimeError(f"doublon {k} dans {self.path.name}")
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
        self.done.add(k)

    def release(self):
        """Idempotent; only removes the lock if this object still owns it."""
        if self.f is not None and not self.f.closed:
            self.f.close()
        if self._owns_lock:
            try:
                if self.lock.read_text() == self.token:
                    self.lock.unlink()
            except FileNotFoundError:
                pass
            self._owns_lock = False
        atexit.unregister(self.release)
