"""Resumable, verified downloads (models, llama.cpp binaries).

A download writes to `<dest>.part`. If that file exists, the transfer resumes with an HTTP Range
request; a server that ignores Range (200 instead of 206) restarts it from zero. The SHA-256 is
computed while writing (the existing part is hashed first), the size and digest are checked at the
end, and only then is the file renamed to `dest`. A digest mismatch deletes the part file."""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

CHUNK = 1 << 20
USER_AGENT = "Myriad downloader"


class DownloadError(RuntimeError):
    pass


@dataclass
class Progress:
    label: str
    total: int | None = None
    done: int = 0
    resumed_from: int = 0
    state: str = "en attente"  # en attente, en cours, vérification, terminé, erreur, annulé
    error: str | None = None
    speed_bps: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        eta = None
        if self.total and self.speed_bps > 0 and self.state == "en cours":
            eta = round((self.total - self.done) / self.speed_bps, 1)
        return {"label": self.label, "total": self.total, "done": self.done, "state": self.state,
                "error": self.error, "speed_bps": round(self.speed_bps), "eta_s": eta,
                "resumed_from": self.resumed_from}


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _hash_prefix(path: Path) -> tuple["hashlib._Hash", int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h, n


async def download(url: str, dest: Path, sha256: str | None = None, size: int | None = None,
                   client: httpx.AsyncClient | None = None, progress: Progress | None = None,
                   cancel: asyncio.Event | None = None, on_progress: Callable[[Progress], None] | None = None,
                   ) -> Path:
    """Download `url` to `dest` (resuming `dest.part`), check size and SHA-256, return `dest`.
    An existing `dest` with the right digest (or size, when no digest is given) is kept."""
    dest = Path(dest)
    prog = progress or Progress(label=dest.name)
    prog.total = size
    sha256 = sha256.lower() if sha256 else None
    if dest.exists():
        ok = (await asyncio.to_thread(sha256_file, dest) == sha256) if sha256 else (size is None or dest.stat().st_size == size)
        if ok:
            prog.done = prog.total = dest.stat().st_size
            prog.state = "terminé"
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0, read=60.0),
                                         headers={"User-Agent": USER_AGENT})
    try:
        for attempt in range(2):  # the second attempt restarts from zero after a bad Range answer
            h, have = await asyncio.to_thread(_hash_prefix, part) if part.exists() else (hashlib.sha256(), 0)
            if size is not None and have > size:
                part.unlink()
                h, have = hashlib.sha256(), 0
            prog.resumed_from = prog.done = have
            prog.state = "en cours"
            headers = {"Range": f"bytes={have}-"} if have else {}
            if size is not None and have == size:
                break  # everything is already there: only the check remains
            async with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 416 and have:  # nothing left to send: check what we have
                    break
                if r.status_code not in (200, 206):
                    raise DownloadError(f"HTTP {r.status_code} pour {url}")
                if have and r.status_code == 200:  # Range ignored: start again
                    h, have = hashlib.sha256(), 0
                    prog.resumed_from = prog.done = 0
                if r.status_code == 206 and not r.headers.get("content-range", "").startswith(f"bytes {have}-"):
                    part.unlink(missing_ok=True)
                    if attempt == 0:
                        continue
                    raise DownloadError("réponse partielle incohérente")
                length = r.headers.get("content-length")
                if prog.total is None and length and length.isdigit():
                    prog.total = have + int(length)
                mode = "ab" if have else "wb"
                t_last, b_last = time.monotonic(), prog.done
                with open(part, mode) as f:
                    async for chunk in r.aiter_bytes(CHUNK):
                        if cancel is not None and cancel.is_set():
                            prog.state = "annulé"
                            raise asyncio.CancelledError()
                        f.write(chunk)
                        h.update(chunk)
                        prog.done += len(chunk)
                        if size is not None and prog.done > size:
                            raise DownloadError("fichier plus gros que prévu")
                        now = time.monotonic()
                        if now - t_last >= 0.5:
                            inst = (prog.done - b_last) / (now - t_last)
                            prog.speed_bps = inst if prog.speed_bps == 0 else 0.7 * prog.speed_bps + 0.3 * inst
                            t_last, b_last = now, prog.done
                            if on_progress:
                                on_progress(prog)
            break
        prog.state = "vérification"
        if on_progress:
            on_progress(prog)
        got = part.stat().st_size if part.exists() else 0
        if size is not None and got != size:
            raise DownloadError(f"taille inattendue : {got} octets au lieu de {size}")
        if sha256:
            digest = h.hexdigest() if prog.done == got else await asyncio.to_thread(sha256_file, part)
            if digest != sha256:
                part.unlink(missing_ok=True)
                raise DownloadError("empreinte SHA-256 incorrecte : fichier supprimé, relancez le téléchargement")
        os.replace(part, dest)
        prog.done = prog.total = dest.stat().st_size
        prog.state = "terminé"
        if on_progress:
            on_progress(prog)
        return dest
    except asyncio.CancelledError:
        prog.state = "annulé"
        raise
    except httpx.HTTPError as e:
        prog.state, prog.error = "erreur", f"réseau : {e}"
        raise DownloadError(prog.error) from e
    except (DownloadError, OSError) as e:
        prog.state, prog.error = "erreur", str(e)
        raise
    finally:
        if own:
            await client.aclose()
