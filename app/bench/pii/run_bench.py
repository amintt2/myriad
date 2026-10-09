"""Run each detector over the benchmark documents, one fresh process per detector (clean RSS).

    uv run --group pii-bench python -m bench.pii.run_bench                  # every detector
    uv run --group pii-bench python -m bench.pii.run_bench nym-small-edge regex    # some of them

Raw outputs go to <cache>/runs/<detector>.jsonl.gz (+ .npz for embeddings); they are large and not
committed. Memory: peak working set (Windows) or max RSS (Linux/macOS) of the child process, measured
by the OS, plus the RSS right after the imports (runtime alone) and right after loading the model."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time

from .registry import CACHE, MODELS

RUNS = CACHE / "runs"
ALL = ["regex", "nym-small-edge", "nym-small-int8", "bert-small-pii", "ettin-32m", "minilm-nemotron",
       "gliner-pii-small", "e5-small-probe", "mminilm-probe", "gte-small-probe", "embeddinggemma2-probe", "bunker-laya"]


def _mem() -> dict:
    import psutil
    mi = psutil.Process().memory_info()
    out = {"rss_mb": mi.rss / 2**20}
    if hasattr(mi, "peak_wset"):
        out["peak_mb"] = mi.peak_wset / 2**20
    else:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["peak_mb"] = r / 2**20 if sys.platform == "darwin" else r / 1024
    return out


def child(name: str, threads: int | None) -> None:
    import numpy as np
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
    from myriad import privacy_rules  # noqa: F401

    from .data import load
    from .detectors import make
    m0 = _mem()
    docs = load()
    if os.environ.get("PII_BENCH_SUBSET") == "guarantee":  # slow detectors: skip gretel (not calibrated on)
        docs = [d for d in docs if d["guarantee"]]
    m_data = _mem()
    t0 = time.perf_counter()
    det = make(name, threads)
    load_s = time.perf_counter() - t0
    m1 = _mem()
    det.run("warm-up : Jean Dupont, 06 12 34 56 78")
    RUNS.mkdir(parents=True, exist_ok=True)
    vecs, owners = [], []
    with gzip.open(RUNS / f"{name}.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            t = time.perf_counter()
            r = det.run(d["text"])
            ms = (time.perf_counter() - t) * 1000
            rec = {"id": d["id"], "ms": round(ms, 3)}
            if "tokens" in r:
                rec["tokens"] = [[a, z, round(s, 4), ty] for a, z, s, ty in r["tokens"]]
            if "doc" in r:
                rec["doc"] = round(r["doc"], 5)
            if "chunks" in r:
                rec["chunks"] = [[a, z] for a, z, _ in r["chunks"]]
                for _, _, v in r["chunks"]:
                    vecs.append(v)
                    owners.append(i)
            f.write(json.dumps(rec) + "\n")
    if vecs:
        np.savez_compressed(RUNS / f"{name}.npz", vecs=np.stack(vecs), owners=np.array(owners))
    m2 = _mem()
    meta = {"detector": name, "threads": threads, "n_docs": len(docs), "load_s": round(load_s, 3),
            "rss_runtime_mb": round(m_data["rss_mb"], 1), "rss_imports_mb": round(m0["rss_mb"], 1),
            "rss_loaded_mb": round(m1["rss_mb"], 1), "rss_end_mb": round(m2["rss_mb"], 1),
            "peak_mb": round(m2["peak_mb"], 1),
            "model_mb": round(m1["rss_mb"] - m_data["rss_mb"], 1)}
    (RUNS / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))


def mem_only(name: str, threads: int | None) -> None:
    """RSS of the runtime + model alone (no dataset in memory): the footprint inside Myriad."""
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
    import numpy  # noqa: F401
    from myriad import privacy_rules  # noqa: F401

    from .detectors import make
    from .handmade import build
    m0 = _mem()
    det = make(name, threads)
    m1 = _mem()
    lat = []
    for d in build():
        t = time.perf_counter()
        det.run(d["text"])
        lat.append((time.perf_counter() - t) * 1000)
    long = "Bonjour, " + " ".join(["Jean Dupont habite au 12 rue des Lilas à Paris et son numéro est le 06 12 34 56 78."] * 40)
    det.run(long)
    m2 = _mem()
    lat.sort()
    out = {"detector": name, "threads": threads, "rss_runtime_mb": round(m0["rss_mb"], 1),
           "rss_loaded_mb": round(m1["rss_mb"], 1), "rss_after_mb": round(m2["rss_mb"], 1),
           "peak_mb": round(m2["peak_mb"], 1), "short_p50_ms": round(lat[len(lat) // 2], 2),
           "short_p95_ms": round(lat[int(len(lat) * 0.95)], 2)}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.mem{threads or ''}.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out))


def main(argv: list[str]) -> None:
    if argv and argv[0] in ("--child", "--mem"):
        threads = int(argv[2]) if len(argv) > 2 and argv[2] != "0" else None
        (child if argv[0] == "--child" else mem_only)(argv[1], threads)
        return
    mem = "--mem-all" in argv
    threads = next((a.split("=")[1] for a in argv if a.startswith("--threads=")), "0")
    names = [a for a in argv if not a.startswith("--")] or ALL
    for n in names:
        assert n == "regex" or n in MODELS, n
        print(f"--- {n}", flush=True)
        args = [sys.executable, "-m", "bench.pii.run_bench", "--mem" if mem else "--child", n, threads]
        subprocess.run(args, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


if __name__ == "__main__":
    main(sys.argv[1:])
