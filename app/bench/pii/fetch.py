"""Download the pinned candidates and datasets of the PII benchmark, checking the SHA-256 of LFS files.

    uv run --group pii-bench python -m bench.pii.fetch                 # everything except bunker-laya
    uv run --group pii-bench python -m bench.pii.fetch bunker-laya     # 1.7 GB, only to measure it

openpii's validation file is 1 GB and sorted by language: OPENPII_CHUNKS evenly spaced byte ranges of
OPENPII_CHUNK bytes are fetched with HTTP Range requests on the pinned commit (the content at a commit
is immutable; the partial file cannot be checked against the whole-file digest, so its own SHA-256 is
printed instead). The first and last partial lines of each range are dropped."""
from __future__ import annotations

import hashlib
import sys
import urllib.request

from huggingface_hub import hf_hub_download

from .registry import DATASETS, DATASET_PATH, DETECTOR_PATH, MODELS

OPENPII_CHUNKS = 30
OPENPII_CHUNK = 1536 * 1024
DATASET_SHA = {
    "nemotron": "1a4b0512ecb5370f0992d29d0f9c07351e6de13f0d7ea33bb18cecb984780247",
    "gretel": "014e1057978f030fce4f4cad7c93b9e3377fd0ded713d2bf4c7de00e3e3c8c72",
    "oasst": "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",
    "humaneval": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_model(name: str) -> None:
    m = MODELS[name]
    files = [m["onnx"], m["tokenizer"], m["config"], *m.get("extra", [])]
    dest = DETECTOR_PATH(name)
    for f in dict.fromkeys(files):
        p = hf_hub_download(m["repo"], f, revision=m["rev"], local_dir=dest)
        want = m.get("lfs", {}).get(f)
        got = sha256(p)
        if want and got != want:
            raise SystemExit(f"{name}: SHA-256 mismatch for {f}: {got} != {want}")
        print(f"{name}: {f} ok ({got[:12]}{', verified' if want else ''})")


def fetch_dataset(name: str) -> None:
    d = DATASETS[name]
    dest = DATASET_PATH(name)
    if name == "openpii":
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "validation.sample.jsonl"
        if not out.exists():
            url = f"https://huggingface.co/datasets/{d['repo']}/resolve/{d['rev']}/{d['file']}"
            parts = []
            for i in range(OPENPII_CHUNKS):
                start = (d["size"] - OPENPII_CHUNK) * i // (OPENPII_CHUNKS - 1)
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + OPENPII_CHUNK - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                if len(data) > OPENPII_CHUNK:
                    raise SystemExit("openpii: the server ignored the Range request")
                if start:
                    data = data[data.find(b"\n") + 1:]  # drop the partial first line
                parts.append(data[: data.rfind(b"\n") + 1])  # and the partial last one
            out.write_bytes(b"".join(parts))
        print(f"{name}: {out} ({out.stat().st_size} bytes, sha256 {sha256(out)[:16]})")
        return
    p = hf_hub_download(d["repo"], d["file"], revision=d["rev"], repo_type="dataset", local_dir=dest)
    got = sha256(p)
    if got != DATASET_SHA[name]:
        raise SystemExit(f"{name}: SHA-256 mismatch: {got}")
    print(f"{name}: {p} ok")


def main(argv: list[str]) -> None:
    names = argv or [n for n in MODELS if n != "bunker-laya"] + list(DATASETS)
    for n in names:
        (fetch_model if n in MODELS else fetch_dataset)(n)


if __name__ == "__main__":
    main(sys.argv[1:])
