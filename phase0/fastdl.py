"""Fast download of one file from a Hugging Face repo: N parallel HTTP range requests,
then a SHA-256 check against the hash Hugging Face publishes for that file.

    uv run python fastdl.py ibm-granite/granite-3.3-2b-instruct-GGUF granite-3.3-2b-instruct-Q8_0.gguf ../models/
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi

CHUNK = 32 * 1024 * 1024


async def fetch(url: str, dst: Path, size: int, conns: int):
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    queue: asyncio.Queue = asyncio.Queue()
    for r in ranges:
        queue.put_nowait(r)
    done = [0]
    t0 = time.time()
    with open(dst, "r+b") as f:
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            async def worker():
                while not queue.empty():
                    a, b = queue.get_nowait()
                    for attempt in range(5):
                        try:
                            r = await client.get(url, headers={"Range": f"bytes={a}-{b}"})
                            r.raise_for_status()
                            if len(r.content) != b - a + 1:
                                raise IOError("short read")
                            f.seek(a)
                            f.write(r.content)
                            done[0] += len(r.content)
                            break
                        except Exception:
                            if attempt == 4:
                                raise
                            await asyncio.sleep(2)
                    el = time.time() - t0
                    print(f"\r{done[0] / 1e9:.2f}/{size / 1e9:.2f} Go  {done[0] * 8 / el / 1e6:.0f} Mbit/s", end="", flush=True)
            await asyncio.gather(*[worker() for _ in range(conns)])
    print()


def main():
    repo, filename, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    conns = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    info = next(s for s in HfApi().model_info(repo, files_metadata=True).siblings if s.rfilename == filename)
    size, sha = info.size, info.lfs.sha256 if info.lfs else None
    if not sha:
        raise SystemExit(f"pas d'empreinte SHA-256 publiée pour {filename} : téléchargement refusé")
    dst = outdir / filename
    outdir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == size and sha256(dst) == sha:
            print("déjà là, SHA-256 vérifié :", dst)
            return
        print("fichier présent mais différent : nouveau téléchargement")
    part = dst.with_name(dst.name + ".part")  # the final name only ever holds a verified file
    with open(part, "wb") as f:
        f.truncate(size)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    asyncio.run(fetch(url, part, size, conns))
    if sha256(part) != sha:
        part.unlink()
        raise SystemExit(f"SHA-256 différent pour {filename} : fichier supprimé")
    os.replace(part, dst)
    print("SHA-256 vérifié :", dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    main()
