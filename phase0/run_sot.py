"""E10, parallel sections: can several peers write ONE long answer together, at the same time?

One peer writes a numbered skeleton (3 to 8 points); point i is expanded by peer i mod P of a fixed list
of P families, all points at once; the answer is the points and their expansions in order. Each model
runs alone on the GPU (one llama-server, --jinja, OpenAI chat endpoint, thinking off, greedy), so this
script is called once per model with the stages that model takes part in:

  baseline  the model answers every prompt alone (max 1024 tokens)
  outline   the model writes the skeleton of every prompt (the designated outline model only)
  expand    the model expands every point assigned to it (needs the complete outline file)

    uv run python run_sot.py --model Qwen/Qwen3.5-4B --gguf ../models/Qwen3.5-4B-Q8_0.gguf --stages outline
    uv run python run_sot.py --model Qwen/Qwen3.5-4B --gguf ../models/Qwen3.5-4B-Q8_0.gguf --stages baseline expand \
        --tag sot1 --outline-model Qwen/Qwen3.5-4B --peers Qwen/Qwen3.5-4B google/gemma-4-E4B-it ...

Writes results/sot_{base,outline,expand_<tag>}_<model><suffix>_<split>.jsonl (+ manifests), resumable.
Each row records tokens, prompt tokens, finish reason and wall ms; wall ms is measured under batched
serving (many requests share the GPU), so the speed analysis uses token counts, not these times.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from essaim import data, sot
from essaim.common import resolve_revision
from essaim.results import ResultsFile, file_identity
from run_solo import ChatServer

STAGES = ("baseline", "outline", "expand")
CTX_PER_SLOT = 3072


def run_calls(srv: ChatServer, out: ResultsFile, calls: list[tuple[dict, str, int]], parallel: int) -> int:
    """calls: (row identity and context, user message, max_tokens); skips the ones already written."""
    todo = [c for c in calls if tuple(c[0].get(k, 0) for k in out.key) not in out.done]
    ex = ThreadPoolExecutor(max_workers=parallel)
    k = 0
    try:
        futs = {ex.submit(sot.chat, srv.http, [{"role": "user", "content": u}], mt): row for row, u, mt in todo}
        for fut in as_completed(futs):
            g = fut.result()
            g.pop("logprobs")
            row = {**futs[fut], **g}
            if "points" in row:  # outline: parse the skeleton
                row.update(sot.parse_skeleton(g["text"]))
            out.write(row)
            k += 1
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--stages", nargs="+", required=True, choices=STAGES)
    ap.add_argument("--splits", nargs="+", default=list(data.MT_SPLITS), choices=list(data.MT_SPLITS))
    ap.add_argument("--n", type=int, default=None, help="first n prompts of each split (default: all)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--tag", help="name of the parallel configuration (outline model + peers), for expand")
    ap.add_argument("--outline-model")
    ap.add_argument("--peers", nargs="+", help="expansion peers, in round-robin order (point i -> peer i mod P)")
    a = ap.parse_args()
    stages = [s for s in STAGES if s in a.stages]  # outline before expand, whatever the command line order
    if "expand" in stages:
        if not (a.tag and a.outline_model and a.peers):
            ap.error("expand demande --tag, --outline-model et --peers")
        if a.model not in a.peers:
            ap.error(f"{a.model} n'est pas dans --peers")
        if "_" in a.tag:
            ap.error("--tag sans « _ » (il fait partie du nom des fichiers)")

    rev = resolve_revision(a.model)
    srv = ChatServer(a.gguf, a.parallel, CTX_PER_SLOT)
    try:
        weights = file_identity(a.gguf)
        for split in a.splits:
            items = data.mt_bench(a.n, split=split)
            ids = [it["id"] for it in items]
            common = {"model": a.model, "revision": rev, "backend": "llama.cpp (--jinja, OpenAI chat)",
                      "gguf": Path(a.gguf).name, "weights": weights, "engine": srv.version,
                      "data": data.dataset_identity("mtbench"), "split": split, "n": len(items),
                      "prompt": sot.PROMPT_VERSION, "temperature": 0.0, "seed": sot.SEED, "thinking": False,
                      "ctx_per_slot": CTX_PER_SLOT}
            for stage in stages:
                t0 = time.perf_counter()
                if stage == "baseline":
                    out = ResultsFile(sot.base_path(a.model, a.suffix, split),
                                      {**common, "stage": stage, "max_tokens": sot.BASE_MAX_TOKENS}, key=("id",))
                    calls = [({"id": it["id"], "category": it["category"], "model": a.model},
                              sot.baseline_prompt(it["question"]), sot.BASE_MAX_TOKENS) for it in items]
                elif stage == "outline":
                    out = ResultsFile(sot.outline_path(a.model, a.suffix, split),
                                      {**common, "stage": stage, "max_tokens": sot.OUTLINE_MAX_TOKENS,
                                       "points": [sot.MIN_POINTS, sot.MAX_POINTS], "max_point_words": sot.MAX_POINT_WORDS},
                                      key=("id",))
                    calls = [({"id": it["id"], "category": it["category"], "model": a.model, "points": None},
                              sot.outline_prompt(it["question"]), sot.OUTLINE_MAX_TOKENS) for it in items]
                else:
                    outlines, sha = sot.outline_inputs(a.outline_model, a.suffix, split, ids)
                    out = ResultsFile(sot.expand_path(a.tag, a.model, a.suffix, split),
                                      {**common, "stage": stage, "max_tokens": sot.EXPAND_MAX_TOKENS, "tag": a.tag,
                                       "outline_model": a.outline_model, "peers": a.peers, "outline_sha256": sha,
                                       "assignment": "round-robin: point i -> peers[i mod P]"},
                                      key=("id", "point"))
                    calls = []
                    for it in items:
                        pts = outlines[it["id"]]["points"]
                        for i, peer in enumerate(sot.assign(len(pts), a.peers)):
                            if peer == a.model:
                                calls.append(({"id": it["id"], "point": i, "n_points": len(pts), "point_text": pts[i],
                                               "category": it["category"], "model": a.model},
                                              sot.expand_prompt(it["question"], pts, i), sot.EXPAND_MAX_TOKENS))
                try:
                    k = run_calls(srv, out, calls, a.parallel)
                finally:
                    out.release()
                print(f"{stage} {split}: {k} nouveaux appels ({len(calls)} au total) en "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        srv.close()
    print("done", a.model, flush=True)


if __name__ == "__main__":
    main()
