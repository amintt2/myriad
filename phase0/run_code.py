"""E11 generation: every model writes code for every problem, alone, in batches on one GPU.

For each problem: one greedy solution (sample 0) and --samples solutions drawn at temperature 0.8 with fixed
seeds (samples 1..n). The model is served by llama-server (--jinja, OpenAI chat endpoint, thinking off), as in
E4 (run_solo.py). The code is extracted from the answer here (essaim/code.py) and again by the analysis, which
always re-extracts from the raw text. Nothing is executed here: see exec_code.py and essaim/sandbox.py.

    uv run python run_code.py --model Qwen/Qwen3.5-4B --gguf ../models/Qwen3.5-4B-Q8_0.gguf --suffix _colab
Writes results/code_<model><suffix>_<bench>_<split>.jsonl (+ manifest), resumable, key (id, sample).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from essaim import code, data, sot
from essaim.common import resolve_revision
from essaim.results import ResultsFile, file_identity
from run_solo import ChatServer

CTX_PER_SLOT = 3072  # same slots as E4, hence the same GPU memory reservation (colab_jobs.SOLO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--benches", nargs="+", default=list(code.BENCHES), choices=list(code.BENCHES))
    ap.add_argument("--splits", nargs="+", default=list(data.SPLITS), choices=list(data.SPLITS))
    ap.add_argument("--n", type=int, default=None, help="first n problems of each split (default: all)")
    ap.add_argument("--samples", type=int, default=4, help="sampled solutions per problem, besides the greedy one")
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()

    rev = resolve_revision(a.model)
    srv = ChatServer(a.gguf, a.parallel, CTX_PER_SLOT)
    try:
        weights = file_identity(a.gguf)
        for bench in a.benches:
            for split in a.splits:
                items = data.CODE_LOADERS[bench](a.n or data.PART[bench], split=split)
                manifest = {"model": a.model, "revision": rev, "backend": "llama.cpp (--jinja, OpenAI chat)",
                            "gguf": Path(a.gguf).name, "weights": weights, "engine": srv.version,
                            "data": data.dataset_identity(bench), "bench": bench, "split": split, "n": len(items),
                            "prompt": code.PROMPT_VERSION, "max_tokens": code.MAX_TOKENS, "thinking": False,
                            "ctx_per_slot": CTX_PER_SLOT, "samples": a.samples,
                            "greedy": {"temperature": 0.0, "seed": code.GREEDY_SEED},
                            "sampling": {**code.SAMPLING, "seeds": [code.sample_seed(s) for s in range(1, a.samples + 1)]}}
                out = ResultsFile(code.gen_path(a.model, a.suffix, bench, split), manifest, key=("id", "sample"))
                calls = [(it, s) for it in items for s in range(a.samples + 1) if (it["id"], s) not in out.done]
                t0, k = time.perf_counter(), 0
                ex = ThreadPoolExecutor(max_workers=a.parallel)
                try:
                    futs = {}
                    for it, s in calls:
                        extra = {} if s == 0 else {**code.SAMPLING, "seed": code.sample_seed(s)}
                        futs[ex.submit(sot.chat, srv.http, [{"role": "user", "content": code.prompt(bench, it)}],
                                       code.MAX_TOKENS, **extra)] = (it, s)
                    for fut in as_completed(futs):
                        (it, s), g = futs[fut], fut.result()
                        entry = code.entry_point(bench, it)
                        prog, status = code.extract_code(g["text"], entry)
                        out.write({"id": it["id"], "sample": s, "model": a.model, "bench": bench,
                                   "seed": code.sample_seed(s), "temperature": 0.0 if s == 0 else code.SAMPLING["temperature"],
                                   "text": g["text"], "code": prog, "extract": status, "prog": code.prog_id(prog),
                                   "finish": g["finish"], "n_tokens": g["n_tokens"], "prompt_tokens": g["prompt_tokens"],
                                   "ms": g["ms"], "reasoning": g["reasoning"]})
                        k += 1
                except BaseException:
                    ex.shutdown(wait=False, cancel_futures=True)
                    out.release()
                    raise
                ex.shutdown()
                out.release()
                print(f"{bench} {split}: {k} nouvelles solutions en {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        srv.close()
    print("done", a.model, flush=True)


if __name__ == "__main__":
    main()
