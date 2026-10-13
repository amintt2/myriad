"""E16: single-stream speed of every file (quantized and Q8_0) with llama-bench, alone on the GPU.

For each file: prompt processing (pp512) and token generation (tg128), 3 repetitions, every layer on the GPU. The
launcher runs this job in the last stage, when no other job uses the GPU, so the numbers are not time-sliced with a
neighbour (the generation throughput of the E4/E11 jobs is: it is only comparable between jobs run together).
tg128 is the number that matters for a peer answering one request (decoding reads every weight once per token);
the consumer GPU estimate (analyze_e16.py) scales it by file size, it is not measured on this machine.

    LLAMA_SERVER=/content/llama/bin/llama-server python speed_quant.py
Writes results/solo_e16speed_colab_bench_all.jsonl (+ manifest), resumable, key (label).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import colab_jobs
import e16_quant
from essaim.results import ResultsFile, file_identity

RESULTS = Path(__file__).resolve().parent / "results"
PATH = RESULTS / f"solo_{e16_quant.SPEED_STEM}_colab_bench_all.jsonl"


def entries() -> list[tuple[str, str, Path]]:
    """(label, model, gguf path): the quantized files, then the Q8_0 file of the SPEED_Q8 models (E4's own files)."""
    out = [(e16_quant.label(q.model, q.tag), q.model, colab_jobs.MODELS / q.file) for q in e16_quant.QUANT]
    out += [(e16_quant.label(m, "Q8_0"), m, colab_jobs.MODELS / spec[1]) for m in e16_quant.SPEED_Q8
            for spec in [colab_jobs.SOLO[m]]]
    return out


def bench_binary() -> str:
    server = Path(os.environ["LLAMA_SERVER"])
    return str(server.with_name("llama-bench" + server.suffix))


def parse(raw: str) -> dict:
    """Mean and standard deviation (tokens/s) of the pp and tg tests in llama-bench's JSON output."""
    out = {}
    for r in json.loads(raw):
        if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            out["tg"] = {"n": r["n_gen"], "tps": r["avg_ts"], "sd": r.get("stddev_ts")}
        elif r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            out["pp"] = {"n": r["n_prompt"], "tps": r["avg_ts"], "sd": r.get("stddev_ts")}
        out["gpu"] = r.get("gpu_info")
        out["build"] = r.get("build_commit")
    if "tg" not in out or "pp" not in out:
        raise RuntimeError(f"sortie llama-bench inattendue : {raw[:300]}")
    return out


def gpu_name() -> str:
    """The GPU model, recorded in the manifest: a checkpoint restored on another GPU must not be completed there."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    if not name:
        raise SystemExit("nvidia-smi ne répond pas : pas de GPU")
    return name


def main():
    exe = bench_binary()
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    manifest = {"tool": "llama-bench", "engine": " | ".join(l.strip() for l in (version.stdout + version.stderr).splitlines()
                                                          if "version" in l.lower() or "built with" in l.lower()),
                "tests": {"pp": 512, "tg": 128, "repetitions": 3, "ngl": 999}, "alone_on_gpu": True, "gpu": gpu_name()}
    out = ResultsFile(PATH, manifest, key=("label",))
    try:
        for label, model, gguf in entries():
            if (label,) in out.done:
                continue
            if not gguf.exists():
                raise SystemExit(f"{gguf.name} absent : prepare_quant n'a pas téléchargé ce fichier ?")
            r = subprocess.run([exe, "-m", str(gguf), "-p", "512", "-n", "128", "-r", "3", "-ngl", "999", "-o", "json"],
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise SystemExit(f"llama-bench a échoué pour {gguf.name} : {r.stderr[-400:]}")
            row = {"label": label, "model": model, "file": gguf.name, "weights": file_identity(gguf), **parse(r.stdout)}
            out.write(row)
            print(f"{label}: tg128 {row['tg']['tps']:.1f} tok/s, pp512 {row['pp']['tps']:.0f} tok/s", flush=True)
    finally:
        out.release()
    print("done", flush=True)


if __name__ == "__main__":
    main()
